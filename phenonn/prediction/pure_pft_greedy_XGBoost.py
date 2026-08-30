#!/usr/bin/env python3
"""
pure_pft_greedy_XGBoost.py
==========================

Greedy, per-PFT XGBoost training on the *pure-PFT* pools produced by
`phenonn.data_creation.greedy_pure_pft` (the `selected_pixels_PFT{n}.nc` files, ORCHIDEE
numbering 1..15).

Idea — linear-mixture unmixing across PFTs
------------------------------------------
Observed LAI at a pixel is modelled as a cover-weighted sum of pure-PFT LAI
curves, each a function of the WEATHER only (the 15 PFT-fraction columns are
DROPPED from the feature vector — inside one pure pool they are near-constant
and would make a model extrapolate wildly when reused on another PFT's pixels):

    LAI_obs(pixel) ≈ Σ_p  frac_p(pixel) · L_p(weather)

Given an ORDER of PFTs [p1, p2, …], train one booster m_k = L_{p_k} at a time.
When training m_k on p_k's pool, remove the contributions of the PFTs already
fitted, then divide by p_k's own fraction so the booster learns the PURE curve:

    target_k = ( LAI_obs − Σ_{j<k} frac_{p_j} · m_j(X) ) / frac_{p_k}

The division by frac_{p_k} is what keeps the scheme consistent step-to-step
(each m predicts a pure L_p, not a fraction-weighted contribution). In a pure
pool frac_{p_k} ≈ 1 so it barely rescales; rows whose frac_{p_k} < --frac_floor
are dropped (dividing by a tiny fraction is unstable).

Reuses phenonn.training.xgb_train for everything tabular: the PRE-COMPUTED xgb features
(build_xgb_features), the LAI/PFT/CO2 join (build_split) and the site split.
Training ONLY — it saves one model per PFT plus a meta describing the order and
the residual formula. (No anomaly mode here: the residual scheme already removes
the previously-explained signal.)

Output
------
    {output_dir}/{experiment}/
        xgb_model_PFT{n}.json   one booster per PFT, in the given order
        meta.json               order, feature schema, per-PFT stats, args

Usage
-----
    python -m phenonn.prediction.pure_pft_greedy_XGBoost \\
        --greedy_dir       /data/sbarbu/PhenoNN/data/.../greedy_PFT_10000_0.1 \\
        --xgb_features_dir /data/sbarbu/PhenoNN/data/xgb_features_10% \\
        --target_dir       /data/sbarbu/PhenoNN/data/LAI_pixelset \\
        --pft_dir          /data/sbarbu/PhenoNN/data/PFT_pixelset \\
        --co2_path         /data/co2_1700_2023_TRENDYv2024.txt \\
        --train_years 1992-2009 --val_years 2010-2018 \\
        --output_dir runs --experiment greedy_pft_xgb
        # --pft_order 4,10,6   (optional; else read greedy_dir/extraction_order.txt)
"""

import argparse
import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except ImportError as e:  # pragma: no cover
    raise ImportError("xgboost is required — `pip install xgboost`. " f"Error: {e}")

from phenonn.utils.config import PFT_COLS, add_co2_features, add_cyclic_features
from phenonn.training.xgb_train import (
    build_split,
    load_co2_lut,
    load_selected_pixels,
    _parse_years,
    _predict_chunks,
    _split_pool,
)


PFT_FILE = "selected_pixels_PFT{n}.nc"
ORDER_FILE = "extraction_order.txt"


# ── PFT order resolution ─────────────────────────────────────────────────────


def _read_order(greedy_dir: str) -> List[int]:
    """Parse the ORCHIDEE PFT numbers, in extraction order, from
    greedy_pure_pft's extraction_order.txt (lines like '  1  PFT4  ...')."""
    path = os.path.join(greedy_dir, ORDER_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No --pft_order given and no {ORDER_FILE} in {greedy_dir}."
        )
    order: List[int] = []
    with open(path) as f:
        for line in f:
            m = re.search(r"\bPFT\s*(\d+)", line)
            if m:
                order.append(int(m.group(1)))
    if not order:
        raise ValueError(f"Could not parse any PFT number from {path}.")
    return order


def _feature_cols(base_cols: List[str], co2_on: bool, cyclic_on: bool) -> List[str]:
    """Weather features only — PFT fraction columns are intentionally excluded
    (see module docstring)."""
    cols = list(base_cols)
    if co2_on:
        cols.append("co2")
    if cyclic_on:
        cols += ["doy_sin", "doy_cos"]
    return cols


# ── Residual target ──────────────────────────────────────────────────────────


def _residual_target(
    df: pd.DataFrame,
    feature_cols: List[str],
    pft_num: int,
    prev_models: List[Tuple[int, "xgb.Booster", Optional[int]]],
    frac_floor: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """target = (LAI − Σ_{prev} frac_j · m_j(X)) / frac_{pft_num}.
    Returns (target, keep_mask). keep_mask drops NaN-LAI and low-frac rows."""
    lai = df["lai"].to_numpy(np.float32)
    frac_k = np.nan_to_num(df[PFT_COLS[pft_num - 1]].to_numpy(np.float32), nan=0.0)

    contrib = np.zeros(len(df), dtype=np.float32)
    if prev_models:
        X = df[feature_cols].to_numpy(np.float32)
        for j_num, booster_j, best_it_j in prev_models:
            pred_j = _predict_chunks(booster_j, X, feature_cols, best_it_j)
            frac_j = np.nan_to_num(
                df[PFT_COLS[j_num - 1]].to_numpy(np.float32), nan=0.0
            )
            contrib += frac_j * pred_j

    keep = np.isfinite(lai) & (frac_k > frac_floor)
    target = np.full(len(df), np.nan, dtype=np.float32)
    target[keep] = (lai[keep] - contrib[keep]) / frac_k[keep]
    return target, keep


# ── XGBoost training ─────────────────────────────────────────────────────────


def _make_params(args) -> dict:
    params = {
        "objective": args.objective,
        # rmse LAST → drives early stopping / best_iteration (XGBoost uses the
        # last eval_metric); both are reported at best_iteration.
        "eval_metric": ["mae", "rmse"],
        "max_depth": args.max_depth,
        "eta": args.learning_rate,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "min_child_weight": args.min_child_weight,
        "lambda": args.reg_lambda,
        "alpha": args.reg_alpha,
        "tree_method": "hist",
        "verbosity": 1,
    }
    if args.objective == "reg:pseudohubererror":
        params["huber_slope"] = args.huber_slope
    if args.use_gpu:
        params["device"] = "cuda"
    if args.n_threads:
        params["nthread"] = args.n_threads
    return params


def _train_booster(X_tr, y_tr, X_va, y_va, feature_cols, params, args):
    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_cols)
    dval = xgb.DMatrix(X_va, label=y_va, feature_names=feature_cols)
    evals_result: Dict[str, Dict[str, List[float]]] = {}
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=args.num_rounds,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=args.early_stopping,
        evals_result=evals_result,
        verbose_eval=50,
    )
    return booster, evals_result


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="Greedy per-PFT XGBoost (pure-PFT pools, residual unmixing)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--greedy_dir",
        required=True,
        help="Dir with selected_pixels_PFT{n}.nc (+ extraction_order.txt).",
    )
    p.add_argument(
        "--pft_order",
        default="",
        help="Comma ORCHIDEE PFT numbers, e.g. '4,10,6'. "
        "Empty → read greedy_dir/extraction_order.txt.",
    )
    p.add_argument("--xgb_features_dir", required=True)
    p.add_argument("--target_dir", required=True, help="Pixelset LAI_dekadal_{Y}.nc.")
    p.add_argument("--pft_dir", required=True, help="Pixelset PFTmap_{Y}.nc.")
    p.add_argument("--co2_path", default="", help="CO2 LUT (Annee_YYYY=VALUE).")

    p.add_argument("--train_years", required=True)
    p.add_argument("--val_years", required=True)
    p.add_argument(
        "--frac_floor",
        type=float,
        default=0.05,
        help="Skip rows whose current-PFT fraction is below this "
        "(dividing by a tiny fraction is unstable).",
    )

    # Per-PFT site split (early stopping) — mirrors phenonn.training.xgb_train
    p.add_argument("--n_val_sites", type=int, default=500)
    p.add_argument(
        "--val_fraction_of_grid",
        type=float,
        default=0.1,
        help="Val fraction of each PFT pool. 100 → OVERLAP "
        "(train ∪ val same sites; years must be disjoint).",
    )
    p.add_argument("--seed", type=int, default=42)

    # XGBoost hyperparams (same defaults as xgb_train)
    p.add_argument("--num_rounds", type=int, default=2000)
    p.add_argument("--early_stopping", type=int, default=50)
    p.add_argument("--max_depth", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=0.05)
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--colsample_bytree", type=float, default=0.8)
    p.add_argument("--min_child_weight", type=float, default=4.0)
    p.add_argument("--reg_lambda", type=float, default=1.0)
    p.add_argument("--reg_alpha", type=float, default=0.0)
    p.add_argument(
        "--objective",
        default="reg:squarederror",
        choices=["reg:squarederror", "reg:absoluteerror", "reg:pseudohubererror"],
    )
    p.add_argument("--huber_slope", type=float, default=1.0)
    p.add_argument("--use_gpu", action="store_true")
    p.add_argument("--n_threads", type=int, default=0)

    p.add_argument("--output_dir", default="runs")
    p.add_argument("--experiment", default="greedy_pft_xgb")
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    np.random.seed(args.seed)
    exp_dir = os.path.join(args.output_dir, args.experiment)
    os.makedirs(exp_dir, exist_ok=True)

    order = (
        [int(x) for x in args.pft_order.split(",") if x.strip()]
        if args.pft_order
        else _read_order(args.greedy_dir)
    )
    print(f"Experiment dir : {exp_dir}")
    print(f"PFT order      : {order}")

    train_years = _parse_years(args.train_years)
    val_years = _parse_years(args.val_years)
    overlap = args.val_fraction_of_grid >= 100
    if overlap and set(train_years) & set(val_years):
        raise ValueError("OVERLAP mode needs disjoint train/val years.")

    co2_lut = None
    if args.co2_path and add_co2_features:
        if not os.path.exists(args.co2_path):
            raise FileNotFoundError(args.co2_path)
        co2_lut = load_co2_lut(args.co2_path)
    co2_on = co2_lut is not None and add_co2_features
    cyclic_on = bool(add_cyclic_features)

    prev_models: List[Tuple[int, "xgb.Booster", Optional[int]]] = []
    base_cols_ref: Optional[List[str]] = None
    meta_pft: List[dict] = []

    for step, pft_num in enumerate(order, start=1):
        sel_path = os.path.join(args.greedy_dir, PFT_FILE.format(n=pft_num))
        if not os.path.exists(sel_path):
            print(
                f"\n[step {step}] PFT{pft_num} — missing {os.path.basename(sel_path)}, skipped"
            )
            continue
        pool = load_selected_pixels(sel_path)
        train_sites, val_sites = _split_pool(args, pool)
        print(
            f"\n[step {step}] PFT{pft_num}  pool={len(pool):,} "
            f"(train={len(train_sites):,}, val={len(val_sites):,})"
        )

        # ── Tables (PFT fraction cols kept in df for weighting, not in X) ──
        train_df, base_cols = build_split(
            args.xgb_features_dir,
            args.target_dir,
            args.pft_dir,
            train_years,
            train_sites,
            co2_lut,
            None,
            cyclic_on,
            f"PFT{pft_num}-train",
        )
        val_df, _ = build_split(
            args.xgb_features_dir,
            args.target_dir,
            args.pft_dir,
            val_years,
            val_sites,
            co2_lut,
            None,
            cyclic_on,
            f"PFT{pft_num}-val",
        )
        if base_cols_ref is None:
            base_cols_ref = base_cols
        elif base_cols != base_cols_ref:
            raise RuntimeError(
                "Feature columns differ between PFT steps — "
                "same xgb_features_dir must be used throughout."
            )
        feature_cols = _feature_cols(base_cols, co2_on, cyclic_on)

        # ── Residual target (remove already-fitted PFT contributions) ──
        y_tr_full, keep_tr = _residual_target(
            train_df, feature_cols, pft_num, prev_models, args.frac_floor
        )
        y_va_full, keep_va = _residual_target(
            val_df, feature_cols, pft_num, prev_models, args.frac_floor
        )

        X_tr = train_df.loc[keep_tr, feature_cols].to_numpy(np.float32)
        y_tr = y_tr_full[keep_tr]
        X_va = val_df.loc[keep_va, feature_cols].to_numpy(np.float32)
        y_va = y_va_full[keep_va]
        print(
            f"  usable rows  : train {len(y_tr):,}/{len(train_df):,}  "
            f"val {len(y_va):,}/{len(val_df):,}  "
            f"(NaN-LAI / frac<{args.frac_floor} dropped)"
        )
        if len(y_tr) == 0 or len(y_va) == 0:
            print(f"  → PFT{pft_num} skipped (empty train or val after masking).")
            continue

        # ── Train ──
        params = _make_params(args)
        t0 = time.time()
        booster, evals_result = _train_booster(
            X_tr, y_tr, X_va, y_va, feature_cols, params, args
        )
        best_it = int(booster.best_iteration)
        best_val_rmse = float(evals_result["val"]["rmse"][best_it])
        best_val_mae = float(evals_result["val"]["mae"][best_it])
        print(
            f"  trained      : {time.time() - t0:.1f}s, best_it={best_it}, "
            f"best_val_rmse={best_val_rmse:.5f}, best_val_mae={best_val_mae:.5f}"
        )

        model_path = os.path.join(exp_dir, f"xgb_model_PFT{pft_num}.json")
        booster.save_model(model_path)
        prev_models.append((pft_num, booster, best_it))
        meta_pft.append(
            {
                "step": step,
                "pft_orchidee": pft_num,
                "model_file": os.path.basename(model_path),
                "n_train_rows": int(len(y_tr)),
                "n_val_rows": int(len(y_va)),
                "n_train_sites": len(train_sites),
                "n_val_sites": len(val_sites),
                "best_iteration": best_it,
                "best_val_rmse": best_val_rmse,
                "best_val_mae": best_val_mae,
            }
        )

    if not meta_pft:
        raise RuntimeError("No PFT model was trained — check --greedy_dir / order.")

    meta = {
        "model_kind": "xgboost_greedy_pure_pft",
        "residual": "target_k = (LAI - sum_{j<k} frac_j * m_j(X)) / frac_k",
        "pft_columns_in_features": False,
        "order": [m["pft_orchidee"] for m in meta_pft],
        "feature_names": _feature_cols(base_cols_ref, co2_on, cyclic_on),
        "co2_enabled": co2_on,
        "cyclic_enabled": cyclic_on,
        "frac_floor": args.frac_floor,
        "train_years": train_years,
        "val_years": val_years,
        "per_pft": meta_pft,
        "args": vars(args),
    }
    if co2_lut is not None:
        meta["co2_lut"] = {str(k): v for k, v in co2_lut.items()}
    with open(os.path.join(exp_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nTrained {len(meta_pft)} PFT model(s) → {exp_dir}")
    print("Order / best val RMSE:")
    for m in meta_pft:
        print(
            f"  step {m['step']:>2}  PFT{m['pft_orchidee']:<2}  "
            f"val_rmse={m['best_val_rmse']:.5f}  ({m['n_train_rows']:,} rows)"
        )


if __name__ == "__main__":
    main()
