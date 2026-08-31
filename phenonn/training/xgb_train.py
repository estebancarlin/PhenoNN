#!/usr/bin/env python3
"""
xgb_train.py
============

LAI prediction — XGBoost gradient-boosting baseline on the PhenoNN *pixelset*
pipeline. A single global regressor predicts LAI at every (site, year, obs day)
from a flat tabular feature vector:

  • current-day dynamic features + backward rolling stats (7/30/90 d)
        → PRE-COMPUTED once by `phenonn.data_creation.build_xgb_features`
          (read here from xgb_features_{Y}.nc — NOT recomputed per run)
  • PFT fractions (15)          — joined from PFTmap_{Y}.nc  (pixelset)
  • CO2 yearly scalar           — joined from the CO2 LUT     (optional)
  • cyclic doy_sin/doy_cos      — added here iff add_cyclic_features (config)

Target LAI comes from LAI_dekadal_{Y}.nc (pixelset, 36 dekads aligned 1:1 with
the 36 obs rows). Anomaly mode (LAI − climatology(site, obs)) is supported.

Differences vs the NN pipeline
------------------------------
- No PyTorch / batching / normalization (XGBoost is scale-invariant).
- NaN features go straight into XGBoost (it learns a default split direction);
  only NaN-*target* rows are dropped.
- One sample = one observation day (not a 720-day window).

This script is torch-free (small pixelset readers are inlined) so it can run in
an xgboost-only environment. Install xgboost with `pip install xgboost`.

Usage
-----
    python -m prediction.xgb_train \\
        --xgb_features_dir /data/sbarbu/PhenoNN/data/xgb_features_10% \\
        --target_dir       /data/sbarbu/PhenoNN/data/LAI_pixelset \\
        --pft_dir          /data/sbarbu/PhenoNN/data/PFT_pixelset \\
        --selected_pixels  /data/sbarbu/PhenoNN/data/selected_pixels_10%.nc \\
        --co2_path         /data/co2_1700_2023_TRENDYv2024.txt \\
        --train_years 1992-2009 --val_years 2010-2018 \\
        --num_rounds 2000 --max_depth 8 --learning_rate 0.05 \\
        --output_dir runs --experiment xgb_baseline
"""

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

try:
    import xgboost as xgb
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "xgboost is required. Install with `pip install xgboost`. " f"Error: {e}"
    )

from phenonn.utils.config import (
    N_DEKAD_YEAR,
    PFT_COLS,
    PFT_FNAME,
    TARGETS_FNAME,
    add_co2_features,
    add_cyclic_features,
)
from phenonn.data_creation.build_xgb_features import XGB_FNAME


_PRIVATE_WANDB_CONFIG = {
    "xgb_features_dir",
    "target_dir",
    "pft_dir",
    "selected_pixels",
    "co2_path",
    "clim_target_dir",
    "output_dir",
    "load_tables",
}


def initialize_wandb(args, exp_dir):
    """Start optional W&B tracking without publishing local file paths."""
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError(
            'W&B monitoring requires `python -m pip install -e ".[tracking]"`.'
        ) from error

    config = {
        key: value
        for key, value in vars(args).items()
        if key not in _PRIVATE_WANDB_CONFIG
    }
    tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.experiment,
        group=args.wandb_group or None,
        tags=tags or None,
        mode=args.wandb_mode,
        dir=exp_dir,
        config=config,
    )
    run.define_metric("round")
    for namespace in ("train/*", "validation/*"):
        run.define_metric(namespace, step_metric="round")
    return run


# ── Torch-free pixelset readers (inlined, mirror phenonn.data.lai_dataset) ─────────


def load_co2_lut(path: str) -> Dict[int, float]:
    if path.lower().endswith(".nc"):
        with xr.open_dataset(path) as ds:
            if "year" not in ds or "co2" not in ds:
                raise ValueError(
                    f"{path!r} must contain 'year' and 'co2' variables."
                )
            years = np.asarray(ds["year"].values).astype(int)
            values = np.asarray(ds["co2"].values, dtype=float)
        lut = {int(year): float(value) for year, value in zip(years, values)}
        if not lut or not np.all(np.isfinite(values)):
            raise ValueError(f"No finite annual CO2 values found in {path!r}.")
        return lut

    lut: Dict[int, float] = {}
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if not key.strip().lower().startswith("annee_"):
                continue
            try:
                lut[int(key.split("_", 1)[1])] = float(value.strip())
            except ValueError:
                continue
    if not lut:
        raise ValueError(f"No `Annee_YYYY=VALUE` entries parsed from {path!r}.")
    return lut


def load_selected_pixels(path: str) -> List[str]:
    with xr.open_dataset(path) as ds:
        return [str(s) for s in np.asarray(ds["site_id"].values)]


def _read_site_vectors(
    da: xr.DataArray, site_ids: List[str], value_dim: str
) -> np.ndarray:
    """(n_site, size(value_dim)) float32; unknown sites → NaN rows."""
    idx_of = {str(s): i for i, s in enumerate(da["site_id"].values)}
    rows = np.array([idx_of.get(str(s), -1) for s in site_ids], dtype=np.int64)
    out = np.full((len(site_ids), da.sizes[value_dim]), np.nan, dtype=np.float32)
    ok = rows >= 0
    if ok.any():
        out[ok] = (
            da.isel(site=rows[ok])
            .transpose("site", value_dim)
            .values.astype(np.float32)
        )
    return out


def _read_pft(pft_dir: str, year: int, sites: List[str]) -> Optional[np.ndarray]:
    path = os.path.join(pft_dir, PFT_FNAME.format(year=year))
    if not os.path.exists(path):
        return None
    with xr.open_dataset(path, engine="netcdf4", decode_times=False) as ds:
        da = ds["pft_frac"] if "pft_frac" in ds.data_vars else ds[list(ds.data_vars)[0]]
        return _read_site_vectors(da, sites, "pft")  # (n_site, 15)


def _read_lai(target_dir: str, year: int, sites: List[str]) -> Optional[np.ndarray]:
    path = os.path.join(target_dir, TARGETS_FNAME.format(year=year))
    if not os.path.exists(path):
        return None
    with xr.open_dataset(path, engine="netcdf4") as ds:
        return _read_site_vectors(ds["LAI"], sites, "dekad")  # (n_site, 36)


def _climatology(
    target_dir: str, clim_years: List[int], sites: List[str]
) -> Dict[str, np.ndarray]:
    """Per-(site, obs) mean LAI over clim_years. {site: (36,) float32}, NaN
    where a site/obs is never observed."""
    ssum = np.zeros((len(sites), N_DEKAD_YEAR), dtype=np.float64)
    scnt = np.zeros((len(sites), N_DEKAD_YEAR), dtype=np.int64)
    for y in sorted(set(clim_years)):
        lai = _read_lai(target_dir, y, sites)
        if lai is None:
            continue
        fin = np.isfinite(lai)
        ssum += np.where(fin, lai, 0.0)
        scnt += fin
    mean = np.where(scnt > 0, ssum / np.maximum(scnt, 1), np.nan).astype(np.float32)
    return {s: mean[i] for i, s in enumerate(sites)}


# ── Per-year table assembly (reads the PRE-COMPUTED xgb features) ────────────


def _build_year(
    xgb_dir: str,
    target_dir: str,
    pft_dir: str,
    year: int,
    wanted_set: set,
    co2_lut: Optional[Dict[int, float]],
    clim: Optional[Dict[str, np.ndarray]],
    cyclic_on: bool,
) -> Optional[Tuple[pd.DataFrame, List[str]]]:
    path = os.path.join(xgb_dir, XGB_FNAME.format(year=year))
    if not os.path.exists(path):
        print(f"  [warn] missing {os.path.basename(path)} — year skipped")
        return None

    with xr.open_dataset(path, engine="netcdf4") as ds:
        cols = json.loads(ds.attrs["xgb_feature_columns"])
        sids_all = np.asarray(ds["site_id"].values).astype(str)
        keep = (
            np.array([s in wanted_set for s in sids_all])
            if wanted_set
            else np.ones(len(sids_all), dtype=bool)
        )
        sel = np.where(keep)[0]
        if sel.size == 0:
            return None
        sids = sids_all[sel]
        n_obs = ds.sizes["obs"]
        n_sel = sel.size
        feat = {c: ds[c].values[:, sel].reshape(-1) for c in cols}  # obs-major
        month = np.repeat(ds["month"].values.astype(np.int32), n_sel)
        day = np.repeat(ds["day"].values.astype(np.int32), n_sel)
        doy = np.repeat(ds["doy"].values.astype(np.int32), n_sel)

    site_col = np.tile(sids, n_obs)  # site fastest

    # ── Target LAI (36 dekads ↔ 36 obs rows) ──
    lai = _read_lai(target_dir, year, list(sids))  # (n_sel, 36)
    if lai is None:
        print(f"  [warn] no LAI for {year} — year skipped")
        return None
    lai_flat = lai.T.reshape(-1).astype(np.float32)  # obs-major
    if clim is not None:
        clim_arr = np.stack(
            [clim.get(s, np.full(N_DEKAD_YEAR, np.nan, np.float32)) for s in sids]
        )
        lai_flat = lai_flat - clim_arr.T.reshape(-1)

    df = pd.DataFrame(feat)

    # ── PFT fractions (constant per site/year, broadcast over the 36 obs) ──
    pft = _read_pft(pft_dir, year, list(sids))  # (n_sel, 15)
    if pft is None:
        print(f"  [warn] no PFT for {year} — year skipped")
        return None
    for k, name in enumerate(PFT_COLS):
        df[name] = np.tile(pft[:, k], n_obs)

    # ── CO2 scalar (per year) ──
    if co2_lut is not None and add_co2_features:
        df["co2"] = float(co2_lut.get(int(year), np.nan))

    # ── Cyclic day-of-year ──
    if cyclic_on:
        df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25).astype(np.float32)
        df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25).astype(np.float32)

    df["site_id"] = site_col
    df["year"] = int(year)
    df["month"] = month
    df["day"] = day
    df["doy"] = doy
    df["lai"] = lai_flat
    return df, cols


def build_split(
    xgb_dir: str,
    target_dir: str,
    pft_dir: str,
    years: List[int],
    wanted: List[str],
    co2_lut,
    clim,
    cyclic_on: bool,
    label: str,
) -> Tuple[pd.DataFrame, List[str]]:
    wanted_set = set(wanted)
    parts: List[pd.DataFrame] = []
    cols_ref: Optional[List[str]] = None
    t0 = time.time()
    for y in years:
        r = _build_year(
            xgb_dir, target_dir, pft_dir, y, wanted_set, co2_lut, clim, cyclic_on
        )
        if r is None:
            continue
        df, cols = r
        cols_ref = cols_ref or cols
        parts.append(df)
    if not parts:
        raise RuntimeError(f"No {label} data assembled — check dirs/years.")
    out = pd.concat(parts, ignore_index=True)
    print(
        f"  [{label}] {len(out):,} obs rows over {len(parts)} year(s) "
        f"({time.time() - t0:.1f}s)"
    )
    return out, cols_ref


def _feature_columns(base_cols: List[str], co2_on: bool, cyclic_on: bool) -> List[str]:
    """Canonical column order used for train AND predict."""
    cols = list(base_cols) + list(PFT_COLS)
    if co2_on:
        cols.append("co2")
    if cyclic_on:
        cols += ["doy_sin", "doy_cos"]
    return cols


# ── CPU chunked inference (shared with phenonn.prediction.xgb_predict) ───────────────


def _predict_chunks(booster, X, feature_cols, best_it, chunk: int = 2_000_000):
    """Predict in row chunks (small DMatrix each time) so a full predict of tens
    of millions of rows never allocates one giant buffer — a single GPU predict
    of the whole train set can exhaust the card. Uses the early-stopping best
    iteration when available."""
    rng = (0, int(best_it) + 1) if best_it is not None else None
    out = np.empty(len(X), dtype=np.float32)
    for s in range(0, len(X), chunk):
        d = xgb.DMatrix(X[s : s + chunk], feature_names=feature_cols)
        out[s : s + chunk] = (
            booster.predict(d, iteration_range=rng)
            if rng is not None
            else booster.predict(d)
        )
    return out


# ── CLI ──────────────────────────────────────────────────────────────────────


def _parse_years(s: str) -> List[int]:
    if not s:
        return []
    if "-" in s and "," not in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def parse_args():
    p = argparse.ArgumentParser(
        description="LAI — XGBoost baseline (pixelset)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--xgb_features_dir",
        required=True,
        help="Folder of xgb_features_{Y}.nc (phenonn.data_creation.build_xgb_features).",
    )
    p.add_argument("--target_dir", required=True, help="Pixelset LAI_dekadal_{Y}.nc.")
    p.add_argument("--pft_dir", required=True, help="Pixelset PFTmap_{Y}.nc.")
    p.add_argument(
        "--selected_pixels", required=True, help="selected_pixels.nc — the site pool."
    )
    p.add_argument("--co2_path", default="", help="CO2_annual.nc or text LUT.")

    p.add_argument("--train_years", required=True, help="e.g. '1992-2009'.")
    p.add_argument("--val_years", required=True, help="e.g. '2010-2018'.")

    # Site split (mirrors phenonn.training.train_full_ram)
    p.add_argument(
        "--n_train_sites",
        type=int,
        default=0,
        help="0 = every train-pool site; else random subset.",
    )
    p.add_argument("--n_val_sites", type=int, default=500)
    p.add_argument(
        "--val_fraction_of_grid",
        type=float,
        default=0.1,
        help="Fraction of the pool reserved for val. 100 → OVERLAP "
        "(train ∪ val same sites; years must be disjoint).",
    )
    p.add_argument("--seed", type=int, default=42)

    # Anomaly mode
    p.add_argument(
        "--anomaly_mode",
        action="store_true",
        help="Train on LAI − climatology(site, obs).",
    )
    p.add_argument("--clim_years", default="1992-2009")
    p.add_argument("--clim_target_dir", default="")

    # XGBoost hyperparams
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
    p.add_argument("--experiment", default="xgb_baseline")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", default="phenonn-lai")
    p.add_argument("--wandb_entity", default="")
    p.add_argument("--wandb_group", default="")
    p.add_argument("--wandb_tags", default="")
    p.add_argument(
        "--wandb_mode", choices=("online", "offline", "disabled"), default="online"
    )
    p.add_argument(
        "--save_tables",
        action="store_true",
        help="Also dump the train/val feature tables as parquet "
        "(+ tables_meta.json context) so they can be reloaded.",
    )
    p.add_argument(
        "--load_tables",
        default=None,
        help="Reuse train/val tables from a previous --save_tables "
        "run (dir with train_table.parquet, val_table.parquet, "
        "tables_meta.json). Skips the NetCDF build; site/year "
        "selection args are then ignored. The anomaly/clim/CO2 "
        "context is read back from tables_meta.json.",
    )
    return p.parse_args()


def _split_pool(args, all_sites: List[str]) -> Tuple[List[str], List[str]]:
    rng = np.random.RandomState(args.seed)
    if args.val_fraction_of_grid >= 100:
        return list(all_sites), list(all_sites)
    shuffled = rng.permutation(all_sites).tolist()
    n_val_pool = max(args.n_val_sites, int(args.val_fraction_of_grid * len(shuffled)))
    return shuffled[n_val_pool:], shuffled[:n_val_pool]


# ── Main ─────────────────────────────────────────────────────────────────────


def _build_tables(args):
    """Assemble the train/val feature tables from the pixelset NetCDFs (the
    default path). Returns (train_df, val_df, feature_cols, info)."""
    all_sites = load_selected_pixels(args.selected_pixels)
    train_pool, val_pool = _split_pool(args, all_sites)
    overlap = args.val_fraction_of_grid >= 100
    print(
        f"Site pool      : {len(all_sites):,} sites "
        f"({'OVERLAP' if overlap else 'DISJOINT'})"
    )

    rng = np.random.RandomState(args.seed)
    if args.n_train_sites and args.n_train_sites < len(train_pool):
        train_sites = rng.choice(
            train_pool, size=args.n_train_sites, replace=False
        ).tolist()
    else:
        train_sites = list(train_pool)
    if overlap:
        # Temporal validation on the SAME pixels as train (split only by the
        # disjoint years). Subsample --n_val_sites of the train sites to cap
        # RAM; --n_val_sites=0 → use every train site.
        if args.n_val_sites and args.n_val_sites < len(train_sites):
            val_sites = rng.choice(
                train_sites, size=args.n_val_sites, replace=False
            ).tolist()
        else:
            val_sites = list(train_sites)
    else:
        val_sites = rng.choice(
            val_pool,
            size=min(args.n_val_sites, len(val_pool)),
            replace=False,
        ).tolist()
    print(f"Sites used     : train={len(train_sites):,}, val={len(val_sites):,}")

    train_years = _parse_years(args.train_years)
    val_years = _parse_years(args.val_years)
    if overlap and set(train_years) & set(val_years):
        raise ValueError(
            "OVERLAP mode needs disjoint train/val years "
            "(same sites otherwise leak across the split)."
        )
    print(f"Years          : train={train_years}, val={val_years}")

    co2_lut = None
    if args.co2_path and add_co2_features:
        if not os.path.exists(args.co2_path):
            raise FileNotFoundError(args.co2_path)
        co2_lut = load_co2_lut(args.co2_path)
        print(f"CO2 LUT        : {len(co2_lut):,} years")
    co2_on = co2_lut is not None and add_co2_features
    cyclic_on = bool(add_cyclic_features)

    clim = None
    if args.anomaly_mode:
        clim_years = _parse_years(args.clim_years)
        clim_dir = args.clim_target_dir or args.target_dir
        wanted = list(set(train_sites) | set(val_sites))
        print(f"Climatology    : {len(clim_years)} years × {len(wanted):,} sites …")
        clim = _climatology(clim_dir, clim_years, wanted)

    print("\nBuilding train table …")
    train_df, base_cols = build_split(
        args.xgb_features_dir,
        args.target_dir,
        args.pft_dir,
        train_years,
        train_sites,
        co2_lut,
        clim,
        cyclic_on,
        "train",
    )
    print("Building val table …")
    val_df, _ = build_split(
        args.xgb_features_dir,
        args.target_dir,
        args.pft_dir,
        val_years,
        val_sites,
        co2_lut,
        clim,
        cyclic_on,
        "val",
    )

    feature_cols = _feature_columns(base_cols, co2_on, cyclic_on)
    info = {
        "train_site_ids": train_sites,
        "val_site_ids": val_sites,
        "train_years": train_years,
        "val_years": val_years,
        "co2_on": co2_on,
        "co2_lut": co2_lut,
        "tables_source": None,
    }
    return train_df, val_df, feature_cols, info


def _load_tables(path: str) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Reload the train/val tables + their context sidecar written by
    --save_tables, skipping the NetCDF build entirely."""
    tr_path = os.path.join(path, "train_table.parquet")
    va_path = os.path.join(path, "val_table.parquet")
    tm_path = os.path.join(path, "tables_meta.json")
    for p in (tr_path, va_path, tm_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} missing — was the source run trained with --save_tables?"
            )
    train_df = pd.read_parquet(tr_path)
    val_df = pd.read_parquet(va_path)
    with open(tm_path) as f:
        tm = json.load(f)
    feature_cols = tm["feature_cols"]
    missing = [
        c for c in feature_cols if c not in train_df.columns or c not in val_df.columns
    ]
    if missing:
        raise ValueError(f"Loaded tables missing feature columns: {missing}")
    print(
        f"Loaded tables  : train {len(train_df):,} rows, "
        f"val {len(val_df):,} rows, {len(feature_cols)} features, "
        f"anomaly={bool(tm.get('anomaly_mode', False))}  (from {path})"
    )
    return train_df, val_df, tm


def main():
    args = parse_args()
    np.random.seed(args.seed)
    exp_dir = os.path.join(args.output_dir, args.experiment)
    os.makedirs(exp_dir, exist_ok=True)
    print(f"Experiment dir : {exp_dir}")
    wandb_run = initialize_wandb(args, exp_dir)
    if wandb_run is not None:
        print(f"W&B run        : {wandb_run.url or args.wandb_mode}")

    if args.load_tables:
        train_df, val_df, tm = _load_tables(args.load_tables)
        feature_cols = tm["feature_cols"]
        # Adopt the source run's target semantics so the new checkpoint's meta
        # (and any later prediction) stays correct — esp. anomaly / clim / CO2.
        args.anomaly_mode = bool(tm.get("anomaly_mode", args.anomaly_mode))
        args.clim_years = tm.get("clim_years", args.clim_years)
        args.clim_target_dir = tm.get("clim_target_dir", args.clim_target_dir)
        args.target_dir = args.target_dir or tm.get("target_dir", "")
        co2_lut = (
            {int(k): float(v) for k, v in tm["co2_lut"].items()}
            if tm.get("co2_lut")
            else None
        )
        info = {
            "train_site_ids": tm.get("train_site_ids", []),
            "val_site_ids": tm.get("val_site_ids", []),
            "train_years": tm.get("train_years", []),
            "val_years": tm.get("val_years", []),
            "co2_on": bool(tm.get("co2_on", "co2" in feature_cols)),
            "co2_lut": co2_lut,
            "tables_source": args.load_tables,
        }
    else:
        train_df, val_df, feature_cols, info = _build_tables(args)

    n_tr0, n_va0 = len(train_df), len(val_df)
    train_df = train_df.dropna(subset=["lai"]).reset_index(drop=True)
    val_df = val_df.dropna(subset=["lai"]).reset_index(drop=True)
    print(
        f"\nUsable rows    : train {len(train_df):,}/{n_tr0:,}  "
        f"val {len(val_df):,}/{n_va0:,}  (NaN-target dropped)"
    )

    if train_df.empty or val_df.empty:
        raise RuntimeError("Empty train or val set after NaN-target drop.")

    if args.save_tables and not args.load_tables:
        train_df.to_parquet(os.path.join(exp_dir, "train_table.parquet"))
        val_df.to_parquet(os.path.join(exp_dir, "val_table.parquet"))
        tables_meta = {
            "feature_cols": feature_cols,
            "anomaly_mode": bool(args.anomaly_mode),
            "clim_years": args.clim_years,
            "clim_target_dir": args.clim_target_dir,
            "target_dir": args.target_dir,
            "co2_on": info["co2_on"],
            "co2_lut": (
                {str(k): v for k, v in info["co2_lut"].items()}
                if info["co2_lut"] is not None
                else None
            ),
            "train_site_ids": info["train_site_ids"],
            "val_site_ids": info["val_site_ids"],
            "train_years": info["train_years"],
            "val_years": info["val_years"],
        }
        with open(os.path.join(exp_dir, "tables_meta.json"), "w") as f:
            json.dump(tables_meta, f, indent=2)

    X_tr = train_df[feature_cols].to_numpy(np.float32)
    y_tr = train_df["lai"].to_numpy(np.float32)
    X_va = val_df[feature_cols].to_numpy(np.float32)
    y_va = val_df["lai"].to_numpy(np.float32)

    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_cols)
    dval = xgb.DMatrix(X_va, label=y_va, feature_names=feature_cols)

    params = {
        "objective": args.objective,
        # rmse LAST → it drives early stopping / best_iteration (XGBoost uses the
        # last eval_metric). Both metrics are reported at best_iteration below.
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

    print(f"\nXGBoost params : {params}")
    print(f"Features       : {len(feature_cols)} columns")
    t0 = time.time()
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
    best_it = int(booster.best_iteration)
    best_val_rmse = float(evals_result["val"]["rmse"][best_it])
    best_val_mae = float(evals_result["val"]["mae"][best_it])
    print(
        f"Trained        : {time.time() - t0:.1f}s, best_iteration={best_it}, "
        f"best_val_rmse={best_val_rmse:.5f}, best_val_mae={best_val_mae:.5f}"
    )
    if wandb_run is not None:
        for round_idx, val_rmse in enumerate(evals_result["val"]["rmse"]):
            wandb_run.log(
                {
                    "round": round_idx,
                    "train/mae": evals_result["train"]["mae"][round_idx],
                    "train/rmse": evals_result["train"]["rmse"][round_idx],
                    "validation/mae": evals_result["val"]["mae"][round_idx],
                    "validation/rmse": val_rmse,
                }
            )

    # ── Save model + metadata + feature importance. Training ONLY — run
    #    `python -m phenonn.prediction.xgb_predict` for predictions / metrics / plots. ──
    model_path = os.path.join(exp_dir, "xgb_model.json")
    booster.save_model(model_path)
    meta = {
        "model_kind": "xgboost_pixelset",
        "args": vars(args),
        "feature_names": feature_cols,
        "train_site_ids": info["train_site_ids"],
        "val_site_ids": info["val_site_ids"],
        "train_years": info["train_years"],
        "val_years": info["val_years"],
        "anomaly_mode": bool(args.anomaly_mode),
        "co2_enabled": info["co2_on"],
        "tables_source": info["tables_source"],
        "best_iteration": best_it,
        "best_val_rmse": best_val_rmse,
        "best_val_mae": best_val_mae,
        "evals_history": evals_result,
    }
    if info["co2_lut"] is not None:
        meta["co2_lut"] = {str(k): v for k, v in info["co2_lut"].items()}
    with open(os.path.join(exp_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    if wandb_run is not None:
        wandb_run.summary["best_iteration"] = best_it
        wandb_run.summary["best_validation_rmse"] = best_val_rmse
        wandb_run.summary["best_validation_mae"] = best_val_mae
        wandb_run.finish()
    print(f"Model → {model_path}")

    try:
        imp = sorted(
            booster.get_score(importance_type="gain").items(), key=lambda kv: -kv[1]
        )
        with open(os.path.join(exp_dir, "feature_importance.json"), "w") as f:
            json.dump(imp, f, indent=2)
        print("\nTop 15 features by gain:")
        for name, val in imp[:15]:
            print(f"  {name:24s}  {val:12.2f}")
    except Exception as e:  # noqa: BLE001
        print(f"  could not extract feature importance: {e}")

    print(
        f"\nTraining done → {exp_dir}\n"
        f"Evaluate:  python -m phenonn.prediction.xgb_predict --checkpoint "
        f"{model_path} --predict_sites val"
    )


if __name__ == "__main__":
    main()
