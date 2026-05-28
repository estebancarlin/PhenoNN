#!/usr/bin/env python3
"""
LAI Prediction — Year-by-Year Inference

Loads a trained model checkpoint and predicts the full annual LAI curve.
Supports both site-split and year-split models.

Usage
-----
# Predict on validation sites, all available years:
python -m phenocam.predict \
    --checkpoint ./runs/exp01/checkpoints/best_model.pth \
    --data_dir ./data/DB/

# Predict on all sites, specific years:
python -m phenocam.predict \
    --checkpoint ./runs/exp01/checkpoints/best_model.pth \
    --data_dir ./data/DB/ \
    --predict_sites all \
    --predict_years 2022,2023

# Predict on training sites only:
python -m phenocam.predict \
    --checkpoint ./runs/exp01/checkpoints/best_model.pth \
    --data_dir ./data/DB/ \
    --predict_sites train \
    --predict_years 2022
"""

import os
import argparse
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from phenocam.dataset import (
    PhenoCamDataset,
    extract_pft_and_site,
    load_norm_stats,
)
from util.model_loader import load_model
from util.diagnostics import plot_pred_vs_obs, plot_gcc_curves, plot_gcc_curves_all
from util.utils import EasyDict


def parse_args():
    p = argparse.ArgumentParser(description="PhenoCam year-by-year inference")
    p.add_argument("--checkpoint", type=str, required=True,
                    help="Path to best_model.pth")
    p.add_argument("--data_dir", type=str, required=True,
                    help="Directory with site CSVs")
    p.add_argument("--predict_years", type=str, default="all",
                    help="Comma-separated years (e.g. '2022,2023') or 'all'")
    p.add_argument("--predict_sites", type=str, default="val",
                    choices=["val", "train", "all"],
                    help="Which sites to predict on: 'val' (validation sites from "
                         "site-split), 'train', or 'all'. Default 'val'.")
    p.add_argument("--output_csv", type=str, default="predictions.csv",
                    help="Where to save predictions")
    p.add_argument("--batch_size", type=int, default=128)
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load checkpoint ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = EasyDict(ckpt["args"])
    norm_stats = ckpt["norm_stats"]
    pft_list = ckpt["pft_list"]
    split_mode = train_args.get("split_mode", "year")

    # Recover file lists from checkpoint (saved by main.py for site splits)
    saved_train_files = ckpt.get("train_files", None)
    saved_val_files = ckpt.get("val_files", None)

    lai_norms = ckpt.get("lai_norms", None)
    if lai_norms:
        print(f"Per-site LAI normalization: {len(lai_norms)} sites")
    else:
        print("No per-site LAI normalization")

    # ── Rebuild model ──
    # At inference we always want single-day output, so force n_target_days=1
    train_args.n_target_days = 1
    model = load_model(train_args).to(device)

    # Load weights — handle mismatch if trained with LastNDaysWrapper
    state_dict = ckpt["model_state_dict"]
    # Both SingleDayWrapper and LastNDaysWrapper store weights under base_model.*
    # so state_dict is compatible
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Model loaded from {args.checkpoint}  (epoch {ckpt['epoch']}, "
          f"val_R²={ckpt.get('val_r2', '?'):.4f})")
    print(f"Training split_mode: {split_mode}")

    # ── Resolve which site files to use ──
    all_site_files = sorted(glob.glob(os.path.join(args.data_dir, "*.csv")))
    if not all_site_files:
        raise FileNotFoundError(f"No CSV files in {args.data_dir}")

    if args.predict_sites == "all":
        site_files = all_site_files
        print(f"Predicting on ALL {len(site_files)} sites")

    elif args.predict_sites == "val":
        if saved_val_files is not None:
            # Remap to current data_dir (paths may differ between train/predict)
            available = set(os.path.basename(f) for f in all_site_files)
            site_files = [os.path.join(args.data_dir, os.path.basename(f))
                          for f in saved_val_files
                          if os.path.basename(f) in available]
            print(f"Predicting on {len(site_files)} VALIDATION sites (from checkpoint)")
        elif split_mode == "year":
            site_files = all_site_files
            print(f"Year-split model: all {len(site_files)} sites used (no site holdout)")
        else:
            print("WARNING: checkpoint has no saved val_files. Using all sites.")
            site_files = all_site_files

    elif args.predict_sites == "train":
        if saved_train_files is not None:
            available = set(os.path.basename(f) for f in all_site_files)
            site_files = [os.path.join(args.data_dir, os.path.basename(f))
                          for f in saved_train_files
                          if os.path.basename(f) in available]
            print(f"Predicting on {len(site_files)} TRAINING sites (from checkpoint)")
        elif split_mode == "year":
            site_files = all_site_files
            print(f"Year-split model: all {len(site_files)} sites used (no site holdout)")
        else:
            print("WARNING: checkpoint has no saved train_files. Using all sites.")
            site_files = all_site_files

    if not site_files:
        raise RuntimeError("No site files matched. Check --data_dir and --predict_sites.")

    # ── Resolve which years to predict ──
    predict_years = None  # None = all available years
    if args.predict_years.lower() != "all":
        predict_years = [int(y) for y in args.predict_years.split(",")]

    # ── Build dataset ──
    feature_mode = train_args.get("feature_mode", "all")
    full_year = train_args.get("full_year", False)
    if feature_mode != "all":
        print(f"Feature mode: {feature_mode}")
    if full_year:
        print("Full-year prediction mode (365 days per sample)")
    dataset = PhenoCamDataset(
        site_files, norm_stats,
        seq_length=train_args.seq_length,
        pft_list=pft_list,
        years=predict_years,
        stride=1,
        lai_norms=lai_norms,
        feature_mode=feature_mode,
        full_year=full_year,
    )
    year_str = predict_years if predict_years else "all"
    print(f"Prediction samples: {len(dataset)} "
          f"({len(site_files)} sites × years {year_str})")

    if len(dataset) == 0:
        raise RuntimeError("No prediction samples. Check year availability and seq_length.")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=True)

    # ── Predict ──
    all_preds = []
    all_targets = []
    all_meta = []
    all_preds_norm = []
    all_targets_norm = []

    with torch.no_grad():
        for i_batch, (features, targets) in enumerate(loader):
            preds = model(features.to(device)).cpu()
            batch_start = i_batch * args.batch_size

            for j in range(preds.size(0)):
                idx = batch_start + j
                meta = dataset.get_site_info(idx)
                lai_min = meta.get("lai_min")
                lai_max = meta.get("lai_max")
                site_idx, day_idx = dataset.samples[idx]

                if full_year:
                    # preds: (B, 1, 365), targets: (B, 1, 365)
                    pred_seq = preds[j, 0, :].numpy()    # (365,)
                    tgt_seq = targets[j, 0, :].numpy()   # (365,)
                    pred_length = len(pred_seq)

                    for k in range(pred_length):
                        d = day_idx - pred_length + 1 + k
                        site_data = dataset.site_data[site_idx]
                        yr = int(site_data["years"][d])

                        row_meta = {
                            "site": meta["site"],
                            "pft": meta["pft"],
                            "year": yr,
                            "day_index": d,
                            "lai_min": lai_min,
                            "lai_max": lai_max,
                        }
                        all_meta.append(row_meta)
                        all_preds_norm.append(float(pred_seq[k]))
                        all_targets_norm.append(float(tgt_seq[k]))

                        if lai_min is not None and lai_max is not None:
                            denom = lai_max - lai_min
                            all_preds.append(float(pred_seq[k]) * denom + lai_min)
                            all_targets.append(float(tgt_seq[k]) * denom + lai_min)
                        else:
                            all_preds.append(float(pred_seq[k]))
                            all_targets.append(float(tgt_seq[k]))
                else:
                    # Standard: preds (B, 1), targets (B, 1)
                    pred_norm = preds[j, 0].item()
                    tgt_norm = targets[j, 0].item()

                    all_meta.append(meta)
                    all_preds_norm.append(pred_norm)
                    all_targets_norm.append(tgt_norm)

                    if lai_min is not None and lai_max is not None:
                        denom = lai_max - lai_min
                        all_preds.append(pred_norm * denom + lai_min)
                        all_targets.append(tgt_norm * denom + lai_min)
                    else:
                        all_preds.append(pred_norm)
                        all_targets.append(tgt_norm)



    

    # ── Assemble results ──
    df = pd.DataFrame(all_meta)
    df["lai_pred"] = all_preds
    df["lai_obs"] = all_targets
    df["error"] = df["lai_pred"] - df["lai_obs"]
    df["lai_pred_norm"] = all_preds_norm
    df["lai_obs_norm"] = all_targets_norm


    # Per-site, per-year stats
    print("\n── Per-site, per-year metrics ──")
    for (site, year), g in df.groupby(["site", "year"]):
        rmse = np.sqrt(np.mean(g["error"] ** 2))
        ss_res = np.sum(g["error"] ** 2)
        ss_tot = np.sum((g["lai_obs"] - g["lai_obs"].mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        print(f"  {site:25s}  {year}  RMSE={rmse:.5f}  R²={r2:.4f}  n={len(g)}")

    site_r2s = []
    for site, g in df.groupby("site"):  
        ss_res = np.sum((g.lai_obs - g.lai_pred) ** 2)
        ss_tot = np.sum((g.lai_obs - g.lai_obs.mean()) ** 2)
        if ss_tot > 0:
            site_r2s.append(1 - ss_res / ss_tot)
    print(f"Median per-site R²: {np.median(site_r2s):.3f}")

    # Overall RMSE
    errors = np.asarray(all_preds) - np.asarray(all_targets)
    rmse_all = float(np.sqrt(np.mean(errors ** 2)))
    print(f"\nOverall RMSE: {rmse_all:.5f}")

    # Save CSV
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Predictions saved to {args.output_csv}")

    # Pred-vs-obs scatter
    plot_filename = os.path.splitext(args.output_csv)[0] + "_pred_vs_obs.png"
    plot_pred_vs_obs(
        df["lai_pred"].values, df["lai_obs"].values,
        filename=plot_filename,
        title=f"Predicted vs observed — {len(df):,} points",
    )

    # LAI annual curves for low / medium / high R² sites
    if df["site"].nunique() >= 3:
        curves_filename = os.path.splitext(args.output_csv)[0] + "_gcc_curves.png"
        selected = plot_gcc_curves(
            df,
            filename=curves_filename,
            site_col="site",
            year_col="year",
            doy_col="day_index",
        )
        print(f"Selected sites for curve plot: {selected}")

    # LAI curves for ALL sites (grid sorted by R²)
    if df["site"].nunique() >= 1:
        all_curves_filename = os.path.splitext(args.output_csv)[0] + "_gcc_curves_all.png"
        plot_gcc_curves_all(
            df,
            filename=all_curves_filename,
            site_col="site",
            year_col="year",
            doy_col="day_index",
        )

    #Normalized pred-vs-obs scatter
    if lai_norms:
        plot_filename = os.path.splitext(args.output_csv)[0] + "_pred_vs_obs_norm.png"
        plot_pred_vs_obs(
            df["lai_pred_norm"].values, df["lai_obs_norm"].values,
            filename=plot_filename,
            title=f"Predicted vs observed (normalized) — {len(df):,} points",
        )

    # LAI curves for ALL sites, normalized (if norms available)
    if df["site"].nunique() >= 1:
        if lai_norms:
            all_curves_norm_filename = os.path.splitext(args.output_csv)[0] + "_gcc_curves_all_norm.png"
            plot_gcc_curves_all(
                df,
                filename=all_curves_norm_filename,
                site_col="site",
                year_col="year",
                doy_col="day_index",
                pred_col="lai_pred_norm",
                obs_col="lai_obs_norm",
            )


if __name__ == "__main__":
    main()