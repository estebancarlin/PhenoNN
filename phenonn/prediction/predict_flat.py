# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Stefan Barbu, Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""
LAI Prediction — Flat CSV Inference

Loads a checkpoint produced by ``phenonn train-flat`` and predicts LAI at the 36
observation days (5th, 15th, 25th of each month) for every (site_id, year)
pair in the chosen site set.

Output
------
predictions_flat.csv : str
    One row per (site, year, observation day)
pred_vs_obs.png : str
    Scatter plot on real LAI values
pred_vs_obs_norm.png : str
    Scatter plot on normalized values
lai_curves.png : str
    Annual curves for 3 representative sites
lai_curves_all.png : str
    Annual curves for all sites (grid)

Usage
-----
Basic prediction:
    python -m phenonn.prediction.predict_flat \\
        --checkpoint runs/exp_flat/checkpoints/best_model.pth \\
        --features_csv data/features.csv \\
        --target_csv data/targets.csv

Predict on all sites for specific years:
    python -m phenonn.prediction.predict_flat \\
        --checkpoint runs/exp_flat/checkpoints/best_model.pth \\
        --predict_sites all \\
        --predict_years 2002,2003

Examples
--------
Run prediction on validation sites:
    >>> python -m phenonn.prediction.predict_flat \\
    ...     --checkpoint runs/exp_flat/checkpoints/best_model.pth \\
    ...     --features_csv data/features.csv \\
    ...     --target_csv data/targets.csv

Run prediction on all sites for years 2020-2022:
    >>> python -m phenonn.prediction.predict_flat \\
    ...     --checkpoint runs/exp_flat/checkpoints/best_model.pth \\
    ...     --features_csv data/features.csv \\
    ...     --target_csv data/targets.csv \\
    ...     --predict_sites all \\
    ...     --predict_years 2020,2021,2022

Notes
-----
The input features CSV must contain daily data with columns:
    site_id, date, year, month, day, pft1_frac..pft15_frac,
    tmin, tmax, daylength, prcp, srad, vpd, swe

The target CSV must contain LAI observations for days 5, 15, 25 of each month:
    site_id, date, year, month, day, LAI
"""

import os
import argparse
import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from phenonn.data.dataset_flat import LAIDataset, get_site_ids
from phenonn.training.train_flat import build_model
from phenonn.utils.diagnostics import (
    plot_pred_vs_obs,
    plot_gcc_curves,
    plot_gcc_curves_all,
)
from phenonn.utils.utils import EasyDict

# Observation (month, day, doy) for each of the 36 annual positions
_OBS_DATES = [
    (month, day, datetime.date(2001, month, day).timetuple().tm_yday)
    for month in range(1, 13)
    for day in [5, 15, 25]
]


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="LAI flat-CSV inference")
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Path to best_model.pth from phenonn train-flat",
    )
    p.add_argument(
        "--features_csv",
        default="",
        help="Features CSV (defaults to path stored in checkpoint)",
    )
    p.add_argument(
        "--target_csv",
        default="",
        help="Targets CSV (defaults to path stored in checkpoint)",
    )
    p.add_argument(
        "--predict_sites",
        default="val",
        choices=["val", "train", "all"],
        help="Which sites to predict on: 'val' (held-out), "
        "'train', or 'all'. Default: 'val'.",
    )
    p.add_argument(
        "--predict_years",
        default="all",
        help="Comma-separated years (e.g. '2001,2002') or 'all'",
    )
    p.add_argument(
        "--output_csv", default="predictions_flat.csv", help="Path for the output CSV"
    )
    p.add_argument("--batch_size", type=int, default=64)
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────


def run_prediction_flat():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load checkpoint ──
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = EasyDict(ckpt["args"])
    norm_stats = ckpt["norm_stats"]
    train_ids = ckpt.get("train_site_ids", [])
    val_ids = ckpt.get("val_site_ids", [])

    print(f"Checkpoint : {args.checkpoint}")
    print(f"Epoch      : {ckpt['epoch']}")
    print(f"val_loss   : {ckpt.get('val_loss', float('nan')):.6f}")
    print(f"val_R²     : {ckpt.get('val_r2', float('nan')):.4f}")
    print(f"val_RMSE   : {ckpt.get('val_rmse', float('nan')):.5f}")

    # CSV paths: CLI overrides take precedence over checkpoint paths
    features_csv = args.features_csv or train_args.get("features_csv", "")
    target_csv = args.target_csv or train_args.get("target_csv", "")
    if not features_csv or not target_csv:
        raise ValueError(
            "Could not determine CSV paths. Provide --features_csv and --target_csv."
        )

    # ── Rebuild model ──
    model = build_model(train_args).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Model      : {train_args.get('type')} + Every10DaysWrapper  "
        f"({n_params:,} parameters)"
    )

    # ── Resolve site IDs ──
    if args.predict_sites == "all":
        site_ids = get_site_ids(features_csv)
        print(f"Predicting on ALL {len(site_ids)} sites")
    elif args.predict_sites == "val":
        site_ids = val_ids if val_ids else get_site_ids(features_csv)
        label = "VALIDATION" if val_ids else "ALL (no val split found)"
        print(f"Predicting on {len(site_ids)} {label} sites")
    else:  # train
        site_ids = train_ids if train_ids else get_site_ids(features_csv)
        label = "TRAINING" if train_ids else "ALL (no train split found)"
        print(f"Predicting on {len(site_ids)} {label} sites")

    # ── Resolve years ──
    predict_years = None
    if args.predict_years.lower() != "all":
        predict_years = [int(y) for y in args.predict_years.split(",")]
    year_label = predict_years if predict_years else "all available"

    # ── Build dataset ──
    dataset = LAIDataset(
        features_csv,
        target_csv,
        norm_stats,
        site_ids=site_ids,
        seq_length=train_args.get("seq_length", 720),
        years=predict_years,
        normalize_target=True,
    )
    print(f"Samples    : {len(dataset)}  (years: {year_label})")

    if len(dataset) == 0:
        raise RuntimeError(
            "No prediction samples. Check that the requested sites and years "
            "are present in the CSVs and that seq_length <= available history."
        )

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # ── Inference ──
    lai_mean = norm_stats["LAI"]["mean"]
    lai_std = norm_stats["LAI"]["std"]
    rows = []

    print("\nRunning inference...")
    with torch.no_grad():
        for i_batch, (features, targets) in enumerate(loader):
            preds = model(features.to(device)).cpu()  # (B, 1, 36)
            batch_start = i_batch * args.batch_size

            for j in range(preds.size(0)):
                idx = batch_start + j
                meta = dataset.get_site_info(idx)
                pred_norm = preds[j, 0, :].numpy()  # (36,)
                tgt_norm = targets[j, 0, :].numpy()  # (36,)

                pred_real = pred_norm * lai_std + lai_mean
                tgt_real = tgt_norm * lai_std + lai_mean

                for k, (month, day, doy) in enumerate(_OBS_DATES):
                    rows.append(
                        {
                            "site_id": meta["site_id"],
                            "year": meta["year"],
                            "month": month,
                            "day": day,
                            "doy": doy,
                            "lai_pred": float(pred_real[k]),
                            "lai_obs": float(tgt_real[k]),
                            "lai_pred_norm": float(pred_norm[k]),
                            "lai_obs_norm": float(tgt_norm[k]),
                        }
                    )

    df = pd.DataFrame(rows)
    df["error"] = df["lai_pred"] - df["lai_obs"]

    # ── Per-site, per-year metrics ──
    print("\n── Per-site, per-year metrics ──")
    for (site, year), g in df.groupby(["site_id", "year"]):
        rmse = float(np.sqrt(np.mean(g["error"] ** 2)))
        ss_res = float(np.sum(g["error"] ** 2))
        ss_tot = float(np.sum((g["lai_obs"] - g["lai_obs"].mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        print(f"  {site:30s}  {year}  RMSE={rmse:.4f}  R²={r2:.4f}  n={len(g)}")

    # Per-site R² (pooled across all years)
    site_r2s = []
    for site, g in df.groupby("site_id"):
        ss_res = float(np.sum((g["lai_obs"] - g["lai_pred"]) ** 2))
        ss_tot = float(np.sum((g["lai_obs"] - g["lai_obs"].mean()) ** 2))
        if ss_tot > 0:
            site_r2s.append(1.0 - ss_res / ss_tot)
    if site_r2s:
        print(f"\nMedian per-site R²  : {np.median(site_r2s):.4f}")
        print(f"Mean   per-site R²  : {np.mean(site_r2s):.4f}")

    overall_rmse = float(np.sqrt(np.mean(df["error"] ** 2)))
    print(f"Overall RMSE        : {overall_rmse:.4f}")
    print(
        f"Total predictions   : {len(df):,}  "
        f"({df['site_id'].nunique()} sites × {df['year'].nunique()} years × 36 obs)"
    )

    # ── Save CSV ──
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"\nPredictions saved to {args.output_csv}")

    base = os.path.splitext(args.output_csv)[0]
    n_pts = len(df)
    n_sites = df["site_id"].nunique()

    # ── Pred vs obs scatter (real values) ──
    plot_pred_vs_obs(
        df["lai_pred"].values,
        df["lai_obs"].values,
        filename=base + "_pred_vs_obs.png",
        title=f"Predicted vs observed LAI — {n_pts:,} points",
    )

    # ── Pred vs obs scatter (normalized values) ──
    plot_pred_vs_obs(
        df["lai_pred_norm"].values,
        df["lai_obs_norm"].values,
        filename=base + "_pred_vs_obs_norm.png",
        title=f"Predicted vs observed LAI (normalized) — {n_pts:,} points",
        xlabel="Observed LAI (normalized)",
        ylabel="Predicted LAI (normalized)",
    )

    # ── Annual curves — 3 representative sites (low/medium/high R²) ──
    if n_sites >= 3:
        selected = plot_gcc_curves(
            df,
            filename=base + "_lai_curves.png",
            site_col="site_id",
            year_col="year",
            doy_col="doy",
        )
        print(f"Curve plot sites: {selected}")
    else:
        print("Fewer than 3 sites — skipping 3-site curve plot")

    # ── Annual curves — all sites grid ──
    if n_sites >= 1:
        plot_gcc_curves_all(
            df,
            filename=base + "_lai_curves_all.png",
            site_col="site_id",
            year_col="year",
            doy_col="doy",
        )

    print("\nDone.")


if __name__ == "__main__":
    run_prediction_flat()
