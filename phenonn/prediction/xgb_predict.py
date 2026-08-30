#!/usr/bin/env python3
"""
xgb_predict.py
==============

Inference for an XGBoost checkpoint trained by `phenonn.training.xgb_train`. This is the
XGBoost twin of `prediction.predict` (the NN inference script): it takes the
SAME site / year / output / plotting options, and writes a CSV with the same
columns so the existing diagnostics apply unchanged:

    site_id, year, month, day, doy, lai_pred, lai_obs,
    lai_pred_norm, lai_obs_norm, error

The feature table is rebuilt with `phenonn.training.xgb_train.build_split`, so the columns
are byte-for-byte the ones the model was trained on. Inference runs on CPU in
row chunks (a full GPU predict of tens of millions of rows can OOM the card).

Anomaly mode: the model predicts LAI − climatology; predictions are converted
back to physical LAI by adding the (site, obs) climatology (recomputed from the
checkpoint's clim_years), exactly like `prediction.predict`.

Usage
-----
    python -m phenonn.prediction.xgb_predict \\
        --checkpoint runs_xgb/xgb_anomaly_baseline/xgb_model.json \\
        --xgb_features_dir /data/sbarbu/PhenoNN/data/xgb_features_10% \\
        --target_dir /data/sbarbu/PhenoNN/data/pixelset_10%/LAI_pixelset \\
        --pft_dir    /data/sbarbu/PhenoNN/data/pixelset_10%/PFT_pixelset \\
        --predict_sites val \\
        --output_csv runs_xgb/xgb_anomaly_baseline/predictions.csv
"""

import argparse
import datetime
import json
import os

import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb

from phenonn.utils.config import PFT_FNAME, PFT_NAMES, add_cyclic_features
from phenonn.training.xgb_train import (
    build_split,
    _climatology,
    _parse_years,
    _predict_chunks,
    load_co2_lut,
    load_selected_pixels,
)
from phenonn.utils.diagnostics import (
    plot_pred_vs_obs,
    plot_gcc_curves,
    plot_gcc_curves_all,
)


_OBS_DATES = [
    (m, d, datetime.date(2001, m, d).timetuple().tm_yday)
    for m in range(1, 13)
    for d in (5, 15, 25)
]
_DAY_IDX = {5: 0, 15: 1, 25: 2}


# ── CLI (mirrors prediction.predict) ─────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="PhenoNN XGBoost inference")
    p.add_argument(
        "--checkpoint",
        required=True,
        help="xgb_model.json, its meta.json, or the experiment dir.",
    )
    p.add_argument(
        "--xgb_features_dir",
        default="",
        help="Folder of xgb_features_{Y}.nc (precomputed features). "
        "Empty → the dir stored in the checkpoint args.",
    )
    p.add_argument("--target_dir", default="")
    p.add_argument("--pft_dir", default="")
    p.add_argument(
        "--predict_sites",
        default="val",
        choices=["val", "train", "all", "grid", "test"],
    )
    p.add_argument(
        "--selected_pixels",
        default="",
        help="Predict only on this selected_pixels*.nc — overrides "
        "--sites and --predict_sites.",
    )
    p.add_argument(
        "--sites",
        default="",
        help="Comma-separated explicit site IDs — overrides " "--predict_sites.",
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
    p.add_argument("--row_min", type=int, default=-1)
    p.add_argument("--row_max", type=int, default=-1)
    p.add_argument("--col_min", type=int, default=-1)
    p.add_argument("--col_max", type=int, default=-1)
    p.add_argument(
        "--load_table",
        default="",
        help="Reuse a parquet table dumped by xgb_train --save_tables "
        "(e.g. runs/xxx/val_table.parquet) instead of rebuilding "
        "from the NetCDFs. Site/year args are then ignored. For an "
        "anomaly checkpoint the reused 'lai' is already LAI−clim; "
        "pass --target_dir so climatology can be re-added.",
    )
    p.add_argument("--output_csv", default="predictions.csv")
    p.add_argument(
        "--batch_size",
        type=int,
        default=2_000_000,
        help="Rows per predict chunk. On GPU, lower it if the card OOMs.",
    )
    p.add_argument(
        "--use_gpu",
        action="store_true",
        help="Run inference on the GPU (device=cuda). Rows are still "
        "predicted in --batch_size chunks to bound VRAM.",
    )
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


def _resolve_checkpoint(path: str):
    """Return (model_path, meta) from a model json / meta json / dir."""
    if os.path.isdir(path):
        model_path = os.path.join(path, "xgb_model.json")
        meta_path = os.path.join(path, "meta.json")
    elif os.path.basename(path) == "meta.json":
        meta_path = path
        model_path = os.path.join(os.path.dirname(path), "xgb_model.json")
    else:
        model_path = path
        meta_path = os.path.join(os.path.dirname(path), "meta.json")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"xgb model not found: {model_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.json not found next to the model: {meta_path}")
    with open(meta_path) as f:
        meta = json.load(f)
    return model_path, meta


def _grid_ids(rr, cc):
    return [
        f"pix_{r:04d}_{c:05d}"
        for r in range(rr[0], rr[1] + 1)
        for c in range(cc[0], cc[1] + 1)
    ]


def _resolve_sites(args, meta, margs) -> list:
    if args.selected_pixels:
        return load_selected_pixels(args.selected_pixels)
    if args.sites:
        return [s.strip() for s in args.sites.split(",") if s.strip()]
    mode = args.predict_sites
    train_pool = list(meta.get("train_site_ids", []))
    val_pool = list(meta.get("val_site_ids", []))
    if mode == "val":
        return val_pool
    if mode == "train":
        return train_pool
    if mode == "all":
        return sorted(set(train_pool) | set(val_pool))
    rmin = args.row_min if args.row_min >= 0 else int(margs.get("row_min", 0))
    rmax = args.row_max if args.row_max >= 0 else int(margs.get("row_max", 0))
    cmin = args.col_min if args.col_min >= 0 else int(margs.get("col_min", 0))
    cmax = args.col_max if args.col_max >= 0 else int(margs.get("col_max", 0))
    grid = _grid_ids((rmin, rmax), (cmin, cmax))
    if mode == "grid":
        return grid
    if mode == "test":
        if not val_pool:
            raise RuntimeError("'test' mode needs val_site_ids in the checkpoint.")
        return sorted(set(grid) - set(val_pool))
    raise ValueError(mode)


def _load_pft_fracs(pft_dir: str, year: int, site_ids) -> dict:
    """{site_id: (N_PFT,) fraction vector} from PFTmap_{year}.nc (pixelset)."""
    path = os.path.join(pft_dir, PFT_FNAME.format(year=year))
    if not os.path.exists(path):
        return {}
    ds = xr.open_dataset(path, engine="netcdf4", decode_times=False)
    da = ds["pft_frac"] if "pft_frac" in ds.data_vars else ds[list(ds.data_vars)[0]]
    all_sites = np.asarray(ds["site_id"].values).astype(str)
    arr = da.transpose("pft", "site").values.astype(np.float32)
    ds.close()
    idx_of = {s: i for i, s in enumerate(all_sites)}
    return {s: arr[:, idx_of[s]] for s in map(str, site_ids) if s in idx_of}


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    model_path, meta = _resolve_checkpoint(args.checkpoint)
    margs = meta.get("args", {})

    is_anomaly = bool(meta.get("anomaly_mode", False))
    co2_on = bool(meta.get("co2_enabled", False))
    cyclic_on = bool(add_cyclic_features)
    feature_cols = list(meta["feature_names"])
    best_it = meta.get("best_iteration", None)

    print(f"Checkpoint    : {model_path}")
    print(
        f"Best iteration: {best_it}  (val_rmse={meta.get('best_val_rmse', float('nan')):.5f})"
    )
    print(
        f"Flags         : anomaly={is_anomaly}, co2={co2_on}, "
        f"cyclic={cyclic_on}, {len(feature_cols)} features"
    )

    xgb_dir = args.xgb_features_dir or margs.get("xgb_features_dir", "")
    target_dir = args.target_dir or margs.get("target_dir", "")
    pft_dir = args.pft_dir or margs.get("pft_dir", "")

    if args.load_table:
        if is_anomaly and not (target_dir or margs.get("clim_target_dir")):
            raise ValueError(
                "Anomaly + --load_table needs the LAI dir to re-add climatology: "
                "pass --target_dir (or use a checkpoint that stored one)."
            )
        df = pd.read_parquet(args.load_table)
        need = list(feature_cols) + ["site_id", "year", "month", "day", "doy", "lai"]
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise ValueError(f"{args.load_table} missing columns: {missing}")
        years = sorted(int(y) for y in pd.unique(df["year"]))
        print(
            f"Loaded table  : {len(df):,} rows, years={years}, "
            f"anomaly={is_anomaly}  (from {args.load_table})"
        )
    else:
        if not xgb_dir or not target_dir or not pft_dir:
            raise ValueError(
                "Provide --xgb_features_dir, --target_dir, --pft_dir "
                "(or use a checkpoint that stored them)."
            )

        # ── CO2 LUT: reuse the one stored in the checkpoint, else reload ──
        co2_lut = None
        if co2_on:
            if meta.get("co2_lut"):
                co2_lut = {int(k): float(v) for k, v in meta["co2_lut"].items()}
            elif margs.get("co2_path"):
                co2_lut = load_co2_lut(margs["co2_path"])

        # ── Sites & years ──
        site_ids = _resolve_sites(args, meta, margs)
        if not site_ids:
            raise RuntimeError("Empty site list.")
        if 0 < args.n_predict_sites < len(site_ids):
            rng = np.random.RandomState(args.seed)
            site_ids = rng.choice(
                site_ids, size=args.n_predict_sites, replace=False
            ).tolist()
        print(f"Sites         : {len(site_ids):,}")

        years = parse_year_spec(args.predict_years)
        if years is None:
            years = _parse_years(str(margs.get("val_years", "") or ""))
            if not years:
                raise ValueError("Could not infer --predict_years.")
        print(f"Years         : {years}")

        # ── Rebuild the feature table (same columns as training; raw LAI target) ──
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

    # ── Load model + predict (chunked; CPU by default, GPU with --use_gpu) ──
    booster = xgb.Booster()
    booster.load_model(model_path)
    if args.use_gpu:
        booster.set_param({"device": "cuda"})
    X = df[feature_cols].to_numpy(np.float32)
    print(f"Running inference … (device={'cuda' if args.use_gpu else 'cpu'})")
    pred = _predict_chunks(booster, X, feature_cols, best_it, chunk=args.batch_size)

    site = df["site_id"].to_numpy()
    month = df["month"].to_numpy()
    day = df["day"].to_numpy()
    doy = df["doy"].to_numpy()
    year = df["year"].to_numpy()
    obs_raw = df["lai"].to_numpy(np.float64)

    if is_anomaly:
        # Convert anomaly prediction back to physical LAI: + climatology(site,obs)
        clim_years = _parse_years(str(margs.get("clim_years", "") or ""))
        clim_dir = margs.get("clim_target_dir") or target_dir
        clim = _climatology(clim_dir, clim_years, sorted(set(map(str, site))))
        obs_idx = (month - 1) * 3 + np.array([_DAY_IDX.get(int(d), 0) for d in day])
        codes, uniq = pd.factorize(site)
        cmat = np.stack(
            [clim.get(str(s), np.full(36, np.nan, np.float32)) for s in uniq]
        )
        clim_val = cmat[codes, obs_idx]
        pred_phys = pred + clim_val
        pred_norm = pred
        if args.load_table:
            # A reused table already stores the anomaly (LAI−clim) in 'lai'.
            obs_norm = obs_raw
            obs_phys = obs_raw + clim_val
        else:
            # build_split(clim=None) → 'lai' is raw obs; anomaly is obs−clim.
            obs_norm = obs_raw - clim_val
            obs_phys = obs_raw
    else:
        pred_phys = pred
        obs_phys = obs_raw
        pred_norm = pred
        obs_norm = obs_raw

    out = pd.DataFrame(
        {
            "site_id": site,
            "year": year,
            "month": month,
            "day": day,
            "doy": doy,
            "lai_pred": pred_phys,
            "lai_obs": obs_phys,
            "lai_pred_norm": pred_norm,
            "lai_obs_norm": obs_norm,
        }
    )
    out["error"] = out["lai_pred"] - out["lai_obs"]

    # ── Metrics (same conventions as prediction.predict) ──
    valid = out.dropna(subset=["lai_obs"]).copy()
    print(f"\nObservations  : {len(valid):,} valid / {len(out):,} total")
    summary_lines: list = []
    if not valid.empty:
        obs = valid["lai_obs"].to_numpy(float)
        pr = valid["lai_pred"].to_numpy(float)
        err = pr - obs
        overall_rmse = float(np.sqrt(np.mean(err**2)))
        ss_res = float(np.sum(err**2))
        ss_tot = float(np.sum((obs - obs.mean()) ** 2))
        r2_global = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        oc = valid["lai_obs"] - valid.groupby("site_id")["lai_obs"].transform("mean")
        pc = valid["lai_pred"] - valid.groupby("site_id")["lai_pred"].transform("mean")
        ss_res_c = float(np.sum((oc - pc) ** 2))
        ss_tot_c = float(np.sum(oc**2))
        r2_centered = 1.0 - ss_res_c / ss_tot_c if ss_tot_c > 0 else float("nan")
        if obs.std() > 1e-9 and pr.std() > 1e-9:
            r2_pearson = float(np.corrcoef(obs, pr)[0, 1]) ** 2
        else:
            r2_pearson = float("nan")

        summary_lines.append("── Pooled metrics ──")
        summary_lines.append(f"  Global    R²  (NSE)         : {r2_global:+.4f}")
        summary_lines.append(f"  Centered  R²  (NSE on dyn.) : {r2_centered:+.4f}")
        summary_lines.append(f"  Pearson²  R²  (correlation) : {r2_pearson:+.4f}")
        summary_lines.append(f"  Overall RMSE                : {overall_rmse:.4f}")

        site_r2s = []
        for _, g in valid.groupby("site_id"):
            o = g["lai_obs"].values
            pp = g["lai_pred"].values
            sstot = float(np.sum((o - o.mean()) ** 2))
            if len(o) < 5 or sstot <= 0:
                continue
            site_r2s.append(1.0 - float(np.sum((pp - o) ** 2)) / sstot)
        if site_r2s:
            arr = np.array(site_r2s)
            n_pos = int(np.sum(arr > 0))
            summary_lines.append("")
            summary_lines.append(
                f"── Per-site R² (NSE) distribution ({len(arr)} sites) ──"
            )
            summary_lines.append(f"  Median         : {np.median(arr):+.4f}")
            summary_lines.append(f"  Mean           : {np.mean(arr):+.4f}")
            summary_lines.append(f"  5th  percentile: {np.percentile(arr, 5):+.4f}")
            summary_lines.append(f"  95th percentile: {np.percentile(arr, 95):+.4f}")
            summary_lines.append(
                f"  Sites with R²>0: {n_pos:,} / {len(arr):,} "
                f"({100.0 * n_pos / len(arr):.1f}%)"
            )
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

    # ── Plots (same as prediction.predict) ──
    if not valid.empty:
        try:
            plot_pred_vs_obs(
                valid["lai_pred"].values,
                valid["lai_obs"].values,
                filename=base + "_pred_vs_obs.png",
                title=f"XGBoost — Predicted vs observed LAI ({len(valid):,} pts)",
            )
            if is_anomaly:
                plot_pred_vs_obs(
                    valid["lai_pred_norm"].values,
                    valid["lai_obs_norm"].values,
                    filename=base + "_pred_vs_obs_anomaly.png",
                    title=f"ΔLAI anomaly ({len(valid):,} pts)",
                    xlabel="Observed ΔLAI",
                    ylabel="Predicted ΔLAI",
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
        except Exception as e:  # noqa: BLE001
            print(f"[warn] plotting skipped: {e}")

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
            except Exception as e:  # noqa: BLE001
                print(f"[warn] scatter {yr} skipped: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
