#!/usr/bin/env python3
"""
pure_pft_greedy_predict.py
==========================

Inference for the per-PFT boosters trained by
`phenonn.prediction.pure_pft_greedy_XGBoost`. Each booster m_p predicts the PURE-PFT LAI
curve L_p(weather); the LAI of a (mixed) pixel is recomposed as the
cover-weighted sum of the trained PFTs — the exact inverse of the training
residual (which divided by frac_k):

    LAI_pred(pixel) = Σ_{p trained} frac_p(pixel) · m_p(X)

So this makes ONE prediction per PFT and AGGREGATES them by the pixel's PFT
fractions. The weighted contribution of each PFT is also written out
(contrib_PFT{n}), and they sum exactly to lai_pred.

Mirrors `phenonn.prediction.xgb_predict`: same site/year/output/plot options, same CSV
columns + pooled/per-site metrics + plots. No anomaly mode (the greedy residual
scheme has none). The cover fractions are NOT renormalised — the raw weighted
sum is used, consistent with training; pixels dominated by an untrained PFT are
therefore under-predicted (that PFT's contribution is simply missing).

Usage
-----
    python -m prediction.pure_pft_greedy_predict \\
        --checkpoint runs/greedy_pft_xgb \\
        --selected_pixels /data/sbarbu/PhenoNN/data/selected_pixels_10%.nc \\
        --predict_years 2019 \\
        --output_csv runs/greedy_pft_xgb/predictions.csv
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb

from phenonn.utils.config import PFT_COLS, PFT_NAMES, add_cyclic_features
from phenonn.training.xgb_train import (
    build_split,
    _parse_years,
    _predict_chunks,
    load_co2_lut,
    load_selected_pixels,
)
from phenonn.prediction.xgb_predict import _load_pft_fracs
from phenonn.utils.diagnostics import (
    plot_pred_vs_obs,
    plot_gcc_curves,
    plot_gcc_curves_all,
)


PFT_FILE = "selected_pixels_PFT{n}.nc"


# ── CLI (mirrors phenonn.prediction.xgb_predict) ─────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="Greedy per-PFT XGBoost inference")
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Experiment dir with meta.json + xgb_model_PFT{n}.json, "
        "or the meta.json itself.",
    )
    p.add_argument("--xgb_features_dir", default="")
    p.add_argument("--target_dir", default="")
    p.add_argument("--pft_dir", default="")
    p.add_argument(
        "--selected_pixels",
        default="",
        help="Predict on this selected_pixels*.nc (recommended: the "
        "full mixed pool). Empty → union of the trained PFT pools.",
    )
    p.add_argument(
        "--sites",
        default="",
        help="Comma-separated explicit site IDs — overrides the pool.",
    )
    p.add_argument(
        "--n_predict_sites",
        type=int,
        default=0,
        help="If > 0, random subsample of this many sites.",
    )
    p.add_argument(
        "--predict_years",
        default="",
        help="'2015-2018', '2015,2016' or 'all'. Empty → val_years "
        "from the checkpoint.",
    )
    p.add_argument("--output_csv", default="predictions.csv")
    p.add_argument("--batch_size", type=int, default=2_000_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scatter_years", action="store_true")
    p.add_argument("--n_curves", type=int, default=50)
    p.add_argument("--pft_min_frac", type=float, default=0.05)
    return p.parse_args()


def parse_year_spec(spec: str):
    if not spec or spec.lower() == "all":
        return None
    if "-" in spec and "," not in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(y) for y in spec.split(",")]


def _resolve_meta(path: str):
    """Return (meta_dir, meta) from an experiment dir or a meta.json path."""
    meta_path = os.path.join(path, "meta.json") if os.path.isdir(path) else path
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.json not found: {meta_path}")
    with open(meta_path) as f:
        meta = json.load(f)
    if meta.get("model_kind") != "xgboost_greedy_pure_pft":
        raise ValueError(
            f"{meta_path} is not a greedy per-PFT checkpoint "
            f"(model_kind={meta.get('model_kind')!r})."
        )
    return os.path.dirname(os.path.abspath(meta_path)), meta


def _greedy_pool_union(greedy_dir: str, pft_nums) -> list:
    """Union of the site pools of the trained PFTs (fallback prediction set)."""
    seen, out = set(), []
    for n in pft_nums:
        path = os.path.join(greedy_dir, PFT_FILE.format(n=n))
        if not os.path.exists(path):
            continue
        for s in load_selected_pixels(path):
            if s not in seen:
                seen.add(s)
                out.append(s)
    return out


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    meta_dir, meta = _resolve_meta(args.checkpoint)
    margs = meta.get("args", {})

    co2_on = bool(meta.get("co2_enabled", False))
    cyclic_on = bool(meta.get("cyclic_enabled", add_cyclic_features))
    feature_cols = list(meta["feature_names"])
    per_pft = meta["per_pft"]
    order = [e["pft_orchidee"] for e in per_pft]
    print(f"Checkpoint : {meta_dir}")
    print(f"PFT models : {order}  ({len(per_pft)} boosters)")
    print(
        f"Features   : {len(feature_cols)} (PFT cols excluded), "
        f"co2={co2_on}, cyclic={cyclic_on}"
    )

    xgb_dir = args.xgb_features_dir or margs.get("xgb_features_dir", "")
    target_dir = args.target_dir or margs.get("target_dir", "")
    pft_dir = args.pft_dir or margs.get("pft_dir", "")
    if not xgb_dir or not target_dir or not pft_dir:
        raise ValueError(
            "Provide --xgb_features_dir, --target_dir, --pft_dir "
            "(or use a checkpoint that stored them)."
        )

    co2_lut = None
    if co2_on:
        if meta.get("co2_lut"):
            co2_lut = {int(k): float(v) for k, v in meta["co2_lut"].items()}
        elif margs.get("co2_path"):
            co2_lut = load_co2_lut(margs["co2_path"])

    # ── Sites ──
    if args.sites:
        site_ids = [s.strip() for s in args.sites.split(",") if s.strip()]
    elif args.selected_pixels:
        site_ids = load_selected_pixels(args.selected_pixels)
    else:
        site_ids = _greedy_pool_union(margs.get("greedy_dir", ""), order)
        print(
            "  (no --selected_pixels → predicting on the union of trained " "PFT pools)"
        )
    if not site_ids:
        raise RuntimeError("Empty site list.")
    if 0 < args.n_predict_sites < len(site_ids):
        rng = np.random.RandomState(args.seed)
        site_ids = rng.choice(
            site_ids, size=args.n_predict_sites, replace=False
        ).tolist()
    print(f"Sites      : {len(site_ids):,}")

    years = parse_year_spec(args.predict_years)
    if years is None:
        years = _parse_years(str(margs.get("val_years", "") or ""))
        if not years:
            raise ValueError("Could not infer --predict_years.")
    print(f"Years      : {years}")

    # ── Feature table (PFT frac columns kept for weighting; raw LAI target) ──
    print("Building feature table …")
    df, _ = build_split(
        xgb_dir,
        target_dir,
        pft_dir,
        years,
        site_ids,
        co2_lut,
        None,
        cyclic_on,
        "predict",
    )
    X = df[feature_cols].to_numpy(np.float32)

    # ── One prediction per PFT, aggregate by cover fraction ──
    print("Running per-PFT inference …")
    lai_pred = np.zeros(len(df), dtype=np.float64)
    contrib = {}
    for e in per_pft:
        p = e["pft_orchidee"]
        model_path = os.path.join(meta_dir, e["model_file"])
        booster = xgb.Booster()
        booster.load_model(model_path)
        pred_p = _predict_chunks(
            booster, X, feature_cols, e.get("best_iteration"), chunk=args.batch_size
        )
        frac_p = np.nan_to_num(df[PFT_COLS[p - 1]].to_numpy(np.float32), nan=0.0)
        c_p = frac_p * pred_p
        contrib[p] = c_p
        lai_pred += c_p
        print(f"  PFT{p:<2}  mean|frac·pred|={np.mean(np.abs(c_p)):.4f}")

    obs = df["lai"].to_numpy(np.float64)
    out = pd.DataFrame(
        {
            "site_id": df["site_id"].to_numpy(),
            "year": df["year"].to_numpy(),
            "month": df["month"].to_numpy(),
            "day": df["day"].to_numpy(),
            "doy": df["doy"].to_numpy(),
            "lai_pred": lai_pred,
            "lai_obs": obs,
            "lai_pred_norm": lai_pred,
            "lai_obs_norm": obs,
        }
    )
    out["error"] = out["lai_pred"] - out["lai_obs"]
    for p in order:  # per-PFT contributions
        out[f"contrib_PFT{p}"] = contrib[p]

    # ── Metrics (same conventions as phenonn.prediction.xgb_predict) ──
    valid = out.dropna(subset=["lai_obs"]).copy()
    print(f"\nObservations  : {len(valid):,} valid / {len(out):,} total")
    summary_lines: list = []
    if not valid.empty:
        o = valid["lai_obs"].to_numpy(float)
        pr = valid["lai_pred"].to_numpy(float)
        err = pr - o
        overall_rmse = float(np.sqrt(np.mean(err**2)))
        ss_res = float(np.sum(err**2))
        ss_tot = float(np.sum((o - o.mean()) ** 2))
        r2_global = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        oc = valid["lai_obs"] - valid.groupby("site_id")["lai_obs"].transform("mean")
        pc = valid["lai_pred"] - valid.groupby("site_id")["lai_pred"].transform("mean")
        ss_res_c = float(np.sum((oc - pc) ** 2))
        ss_tot_c = float(np.sum(oc**2))
        r2_centered = 1.0 - ss_res_c / ss_tot_c if ss_tot_c > 0 else float("nan")
        if o.std() > 1e-9 and pr.std() > 1e-9:
            r2_pearson = float(np.corrcoef(o, pr)[0, 1]) ** 2
        else:
            r2_pearson = float("nan")

        summary_lines += [
            "── Pooled metrics (aggregated Σ frac·m_p) ──",
            f"  Global    R²  (NSE)         : {r2_global:+.4f}",
            f"  Centered  R²  (NSE on dyn.) : {r2_centered:+.4f}",
            f"  Pearson²  R²  (correlation) : {r2_pearson:+.4f}",
            f"  Overall RMSE                : {overall_rmse:.4f}",
        ]
        site_r2s = []
        for _, g in valid.groupby("site_id"):
            oo = g["lai_obs"].values
            pp = g["lai_pred"].values
            sstot = float(np.sum((oo - oo.mean()) ** 2))
            if len(oo) < 5 or sstot <= 0:
                continue
            site_r2s.append(1.0 - float(np.sum((pp - oo) ** 2)) / sstot)
        if site_r2s:
            arr = np.array(site_r2s)
            n_pos = int(np.sum(arr > 0))
            summary_lines += [
                "",
                f"── Per-site R² (NSE) distribution ({len(arr)} sites) ──",
                f"  Median         : {np.median(arr):+.4f}",
                f"  Mean           : {np.mean(arr):+.4f}",
                f"  5th  percentile: {np.percentile(arr, 5):+.4f}",
                f"  95th percentile: {np.percentile(arr, 95):+.4f}",
                f"  Sites with R²>0: {n_pos:,} / {len(arr):,} "
                f"({100.0 * n_pos / len(arr):.1f}%)",
            ]
        print("\n" + "\n".join(summary_lines))

    # ── Save CSV + metrics ──
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f"\nPredictions → {args.output_csv}")
    base = os.path.splitext(args.output_csv)[0]
    if summary_lines:
        with open(base + "_metrics.txt", "w") as f:
            f.write("\n".join(summary_lines) + "\n")
        print(f"Metrics → {base}_metrics.txt")

    # ── Plots (same as phenonn.prediction.xgb_predict) ──
    if not valid.empty:
        try:
            plot_pred_vs_obs(
                valid["lai_pred"].values,
                valid["lai_obs"].values,
                filename=base + "_pred_vs_obs.png",
                title=f"Greedy per-PFT XGBoost — Pred vs obs LAI ({len(valid):,} pts)",
            )
            if valid["site_id"].nunique() >= 3:
                plot_gcc_curves(
                    valid,
                    filename=base + "_lai_curves.png",
                    site_col="site_id",
                    year_col="year",
                    doy_col="doy",
                )
            sites_all = valid["site_id"].unique()
            if args.n_curves and 0 < args.n_curves < len(sites_all):
                rng_c = np.random.RandomState(args.seed)
                keep = set(rng_c.choice(sites_all, size=args.n_curves, replace=False))
                curve_df = valid[valid["site_id"].isin(keep)]
            else:
                curve_df = valid
            pft_by_site = _load_pft_fracs(
                pft_dir, years[0], curve_df["site_id"].unique()
            )
            plot_gcc_curves_all(
                curve_df,
                filename=base + "_lai_curves_all.png",
                site_col="site_id",
                year_col="year",
                doy_col="doy",
                pft_by_site=pft_by_site,
                pft_names=PFT_NAMES,
                pft_min_frac=args.pft_min_frac,
            )
        except Exception as ex:  # noqa: BLE001
            print(f"[warn] plotting skipped: {ex}")

    if args.scatter_years and not valid.empty:
        year_dir = os.path.join(os.path.dirname(base) or ".", "scatter_year")
        os.makedirs(year_dir, exist_ok=True)
        for yr in sorted(int(y) for y in valid["year"].unique()):
            sub = valid[valid["year"] == yr]
            if len(sub) < 2:
                continue
            ss_res = float(np.sum((sub["lai_obs"] - sub["lai_pred"]) ** 2))
            ss_tot = float(np.sum((sub["lai_obs"] - sub["lai_obs"].mean()) ** 2))
            r2_y = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            rmse_y = float(np.sqrt(np.mean((sub["lai_pred"] - sub["lai_obs"]) ** 2)))
            try:
                plot_pred_vs_obs(
                    sub["lai_pred"].values,
                    sub["lai_obs"].values,
                    filename=os.path.join(year_dir, f"year_{yr}_pred_vs_obs.png"),
                    title=f"Year {yr} — R²={r2_y:+.4f} RMSE={rmse_y:.4f} n={len(sub):,}",
                )
            except Exception as ex:  # noqa: BLE001
                print(f"[warn] scatter {yr} skipped: {ex}")

    print("Done.")


if __name__ == "__main__":
    main()
