# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Stefan Barbu, Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Train LAI models from per-site ``{PFT}_{site}.csv`` files.

Use ``phenonn train --help`` for options. Maintained examples and known legacy
model limitations are documented in ``doc/source/training.rst``.
"""

import os
import argparse
import glob
import time
import json
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from phenonn.utils.logger import Logger
from phenonn.utils.utils import FileUtils
from phenonn.utils.evaluater import get_loss_function

from phenonn.data.dataset import (
    PhenoCamDataset,
    compute_norm_stats,
    load_norm_stats,
    split_sites_by_fraction,
    extract_pft_and_site,
    load_lai_norms,
    DYNAMIC_FEATURES,
    CYCLIC_FEATURES,
    STATIC_FEATURES,
)
from phenonn.utils.model_loader import load_model
from phenonn.utils.diagnostics import (
    plot_loss_histories,
    plot_metric_histories,
    make_history_dicts,
)

# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="LAI Prediction")

    # Data
    p.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing site CSVs ({PFT}_{site}.csv)",
    )
    p.add_argument(
        "--stats_path",
        type=str,
        default="",
        help="Path to precomputed norm_stats.json (computed if empty)",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="./runs",
        help="Root output directory for logs, checkpoints, etc.",
    )
    p.add_argument(
        "--experiment",
        type=str,
        default="exp01",
        help="Experiment name (creates sub-folder)",
    )
    p.add_argument(
        "--use_site_features",
        type=str,
        default="all",
        help="Which static site features to use: 'all', 'none', or comma-separated list (e.g. 'lat,lon')",
    )
    p.add_argument(
        "--use_derived_features",
        type=str,
        default="all",
        help="Which derived features to use: 'all', 'none', or comma-separated list (e.g. 'gdd_0,cdd')",
    )

    # Normalization
    p.add_argument(
        "--gcc_norms_csv",
        type=str,
        default="",
        help="Path to per-site LAI min/max CSV for inter-site "
        "normalization (e.g. gcc_rcc_mins_site_veg.csv)",
    )
    p.add_argument(
        "--residual_csv",
        type=str,
        default="",
        help="Path to predictions.csv from a first model run. "
        "If provided, the target becomes obs-pred (residual learning).",
    )

    # Split strategy
    p.add_argument(
        "--split_mode",
        type=str,
        default="site",
        choices=["site", "year"],
        help="'site': leave-site-out. 'year': all sites, split by year.",
    )
    p.add_argument(
        "--val_fraction",
        type=float,
        default=0.2,
        help="Fraction of sites for validation (split_mode=site)",
    )
    p.add_argument(
        "--train_years",
        type=str,
        default="",
        help="Comma-separated training years (split_mode=year)",
    )
    p.add_argument(
        "--val_years",
        type=str,
        default="",
        help="Comma-separated validation years (split_mode=year)",
    )

    # Model
    p.add_argument(
        "--type",
        type=str,
        default="lstm",
        choices=[
            "lstm",
            "gru",
            "fcn",
            "fullyconnected",
            "transformer",
            "linear",
            "linear_perday",
            "bitransformer",
            "1year_bitransformer",
            "1year_lstm",
        ],
    )
    p.add_argument("--hidden_size", type=int, default=32)
    p.add_argument("--feed_forward_trans", type=int, default=4)
    p.add_argument("--feed_forward_encoder", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--seq_length", type=int, default=365)
    p.add_argument("--embed_size", type=int, default=64)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--forward_expansion", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--dropout_trans", type=float, default=0.0)

    # Training
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--learning_rate", type=float, default=2e-3)
    p.add_argument(
        "--loss_type",
        type=str,
        default="nmae",
        choices=[
            "mse",
            "huber",
            "mae",
            "nmae",
            "nmse",
            "wmse",
            "logcosh",
            "smoothl1",
            "gradient",
        ],
    )
    p.add_argument(
        "--huber_beta",
        type=float,
        default=1.0,
        help="Delta parameter for Huber loss (only used if --loss_type huber)",
    )
    p.add_argument(
        "--gradient_loss_weight",
        type=float,
        default=0.5,
        help="Weight λ for temporal gradient term (only if --loss_type gradient)",
    )
    p.add_argument(
        "--gradient_base_loss",
        type=str,
        default="mse",
        choices=["mse", "huber", "mae"],
        help="Base loss for gradient-aware loss",
    )
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (epochs without improvement)",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=7,
        help="Stride between consecutive training windows (default 7 "
        "to reduce correlation between samples)",
    )
    p.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping (0 = no clipping)",
    )
    p.add_argument(
        "--sites_per_epoch",
        type=int,
        default=0,
        help="Number of training sites sampled per epoch (0 = all sites). "
        "Each epoch draws a different random subset for regularisation.",
    )
    p.add_argument(
        "--random_stride",
        type=int,
        default=0,
        help="Number of random samples per site-year per epoch (0 = use fixed stride). "
        "Each epoch draws a different random subset of days.",
    )
    p.add_argument(
        "--feature_mode",
        type=str,
        default="all",
        choices=["all", "site_only", "meteo_only"],
        help="Feature selection: 'all' (default), 'site_only' (cyclic+static+PFT "
        "for climatology model), 'meteo_only' (dynamic+cyclic for anomaly model)",
    )
    p.add_argument(
        "--full_year",
        action="store_true",
        help="Predict a full year (365 days) from a 730-day input window. "
        "Use with --type 1year_bitransformer and 1year_lstm --seq_length 730.",
    )

    return p.parse_args()


def parse_year_list(s: str):
    """Parse '2018,2019,2020' or '2018-2020' into [2018, 2019, 2020]."""
    if not s:
        return []
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


# ── Training / Validation loops ──────────────────────────────────────────────


def train_one_epoch(model, loader, criterion, optimizer, device, max_grad_norm=1.0):
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_grad = 0.0
    n_samples = 0
    has_components = hasattr(criterion, "last_recon")

    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        preds = model(features)
        loss = criterion(preds, targets)
        loss.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        bs = features.size(0)
        total_loss += loss.item() * bs
        if has_components:
            total_recon += criterion.last_recon * bs
            total_grad += criterion.last_grad * bs
        n_samples += bs

    avg_loss = total_loss / max(n_samples, 1)
    avg_recon = total_recon / max(n_samples, 1)
    avg_grad = total_grad / max(n_samples, 1)
    return avg_loss, avg_recon, avg_grad


@torch.no_grad()
def validate(
    model, loader, criterion, device, dataset=None, n_target_days=1, full_year=False
):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    n_samples = 0

    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)

        preds = model(features)
        loss = criterion(preds, targets)

        total_loss += loss.item() * features.size(0)
        n_samples += features.size(0)

        if full_year:
            # preds/targets: (B, 1, 365) — flatten all days for pooled metrics
            all_preds.append(preds.reshape(-1, 1).cpu())
            all_targets.append(targets.reshape(-1, 1).cpu())
        elif n_target_days > 1:
            all_preds.append(preds[:, :, -1:].cpu())
            all_targets.append(targets[:, :, -1:].cpu())
        else:
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    avg_loss = total_loss / max(n_samples, 1)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    # RMSE (always on last-day scalar)
    rmse = torch.sqrt(torch.mean((all_preds - all_targets) ** 2)).item()

    # Pooled R²
    ss_res = torch.sum((all_targets - all_preds) ** 2)
    ss_tot = torch.sum((all_targets - all_targets.mean()) ** 2)
    r2 = (1 - ss_res / ss_tot).item() if ss_tot > 0 else 0.0

    # Per-site median R² (much more informative for site-split)
    r2_per_site = np.nan
    if not full_year:
        if (
            dataset is not None
            and hasattr(dataset, "samples")
            and hasattr(dataset, "site_data")
        ):
            site_preds = {}
            site_targets = {}
            for i in range(len(all_preds)):
                site_idx, _ = dataset.samples[i]
                site_name = dataset.site_data[site_idx]["site"]
                site_preds.setdefault(site_name, []).append(all_preds[i, 0].item())
                site_targets.setdefault(site_name, []).append(all_targets[i, 0].item())

            site_r2s = []
            for site_name in site_preds:
                p = np.array(site_preds[site_name])
                t = np.array(site_targets[site_name])
                ss_r = np.sum((t - p) ** 2)
                ss_t = np.sum((t - t.mean()) ** 2)
                if ss_t > 1e-8:
                    site_r2s.append(1 - ss_r / ss_t)
            if site_r2s:
                r2_per_site = float(np.median(site_r2s))

    return avg_loss, rmse, r2, r2_per_site


# ── Main ─────────────────────────────────────────────────────────────────────


def run_training():
    args = parse_args()

    # ── Reproducibility ──
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── Directories ──
    exp_dir = os.path.join(args.output_dir, args.experiment)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    log_dir = os.path.join(exp_dir, "logs")
    for d in [ckpt_dir, log_dir]:
        FileUtils.makedir(d)

    logger = Logger(
        console_output=True,
        file_output=True,
        log_file=os.path.join(log_dir, "train.log"),
    )
    logger.show_header("LAI Prediction")

    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Discover site files ──
    site_files = sorted(glob.glob(os.path.join(args.data_dir, "*.csv")))
    if not site_files:
        raise FileNotFoundError(f"No CSV files found in {args.data_dir}")
    logger.info(f"Found {len(site_files)} site files")

    # Discover PFTs
    pft_list = sorted(set(extract_pft_and_site(f)[0] for f in site_files))
    logger.info(f"PFT codes: {pft_list}")

    # ── Per-site LAI normalization ──
    lai_norms = None
    if args.gcc_norms_csv:
        logger.start_task("Loading per-site LAI norms", args.gcc_norms_csv)
        lai_norms = load_lai_norms(args.gcc_norms_csv, site_files)
        n_matched = sum(
            1
            for f in site_files
            if os.path.splitext(os.path.basename(f))[0] in lai_norms
        )
        logger.info(f"Matched {n_matched}/{len(site_files)} sites with LAI norms")
        logger.success("Per-site LAI normalization enabled (target in [0, 1])")
    else:
        logger.info("No --gcc_norms_csv provided; target = raw LAI")

    # ── Split sites/years FIRST (before norm stats) ──
    logger.start_task("Building datasets", f"split_mode={args.split_mode}")

    if args.split_mode == "site":
        train_files, val_files = split_sites_by_fraction(
            site_files, val_fraction=args.val_fraction, seed=args.seed
        )
        logger.info(f"Train sites: {len(train_files)}, Val sites: {len(val_files)}")
        norm_files = train_files  # norm stats from TRAIN sites only
    else:  # year split
        train_files = site_files
        val_files = site_files
        norm_files = site_files  # all sites visible in year split

    # ── Normalization stats (computed on training data only) ──
    stats_path = args.stats_path or os.path.join(exp_dir, "norm_stats.json")
    if args.stats_path and os.path.exists(stats_path):
        logger.info(f"Loading existing norm stats from {stats_path}")
        norm_stats = load_norm_stats(stats_path)
    else:
        logger.start_task(
            "Computing normalization statistics",
            f"From {len(norm_files)} training files",
        )
        norm_stats = compute_norm_stats(norm_files, save_path=stats_path)
        logger.success(f"Stats saved to {stats_path}")

    for feat, s in norm_stats.items():
        logger.info(f"  {feat:15s}  mean={s['mean']:.4f}  std={s['std']:.4f}")

    # ── Build datasets ──
    n_target_days = 2 if args.loss_type == "gradient" else 1

    # Feature count depends on mode
    n_features = len(CYCLIC_FEATURES)  # always included
    if args.feature_mode in ("all", "meteo_only"):
        n_features += len(DYNAMIC_FEATURES)
    if args.feature_mode in ("all", "site_only"):
        n_features += len(STATIC_FEATURES) + len(pft_list)
    args.feature_channel = n_features
    args.output_channel = 1
    logger.info(f"Feature mode: {args.feature_mode} → {n_features} channels")

    residual_csv = args.residual_csv if args.residual_csv else None
    if residual_csv:
        logger.start_task(
            "Residual learning mode", f"Target = obs - pred from {residual_csv}"
        )

    # Helper to build a training dataset from a list of files
    def make_train_ds(files):
        return PhenoCamDataset(
            files,
            norm_stats,
            seq_length=args.seq_length,
            pft_list=pft_list,
            stride=args.stride,
            lai_norms=lai_norms,
            n_target_days=n_target_days,
            residual_csv=residual_csv,
            random_stride=args.random_stride,
            feature_mode=args.feature_mode,
            full_year=args.full_year,
        )

    if args.split_mode == "site":
        val_ds = PhenoCamDataset(
            val_files,
            norm_stats,
            seq_length=args.seq_length,
            pft_list=pft_list,
            stride=1,
            lai_norms=lai_norms,
            n_target_days=n_target_days,
            residual_csv=residual_csv,
            feature_mode=args.feature_mode,
            full_year=args.full_year,
        )
        all_train_files = train_files
    else:  # year split
        train_years = parse_year_list(args.train_years)
        val_years = parse_year_list(args.val_years)
        if not train_years or not val_years:
            raise ValueError("--train_years and --val_years required for year split")
        logger.info(f"Train years: {train_years}, Val years: {val_years}")

        def make_train_ds(files):
            return PhenoCamDataset(
                files,
                norm_stats,
                seq_length=args.seq_length,
                pft_list=pft_list,
                years=train_years,
                stride=args.stride,
                lai_norms=lai_norms,
                n_target_days=n_target_days,
                residual_csv=residual_csv,
                random_stride=args.random_stride,
                feature_mode=args.feature_mode,
                full_year=args.full_year,
            )

        val_ds = PhenoCamDataset(
            site_files,
            norm_stats,
            seq_length=args.seq_length,
            pft_list=pft_list,
            years=val_years,
            stride=1,
            lai_norms=lai_norms,
            n_target_days=n_target_days,
            residual_csv=residual_csv,
            feature_mode=args.feature_mode,
            full_year=args.full_year,
        )
        all_train_files = site_files

    # Site subsampling config
    use_site_sampling = args.sites_per_epoch > 0 and args.sites_per_epoch < len(
        all_train_files
    )
    rng = np.random.RandomState(args.seed)

    if use_site_sampling:
        logger.info(
            f"Site subsampling: {args.sites_per_epoch}/{len(all_train_files)} "
            f"training sites per epoch"
        )
        # Build initial dataset with full files for sample count logging
        train_ds = make_train_ds(all_train_files)
        logger.info(f"Full train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    else:
        train_ds = make_train_ds(all_train_files)
        logger.info(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    if n_target_days > 1:
        logger.info(
            f"Gradient loss mode: predicting last {n_target_days} days per window"
        )

    if args.random_stride > 0:
        logger.info(
            f"Random stride: {args.random_stride} samples per site-year per epoch "
            f"(from {train_ds._all_valid_count} total valid samples)"
        )

    if residual_csv:
        logger.info(
            f"Residual mode: {len(train_ds._residual_lookup)} entries loaded "
            f"from predictions CSV"
        )
        logger.info(
            f"Samples with valid residuals — train: {len(train_ds)}, val: {len(val_ds)}"
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # ── Model ──
    logger.start_task("Initializing model", f"Type: {args.type}")
    args.n_target_days = n_target_days
    model = load_model(args).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {n_params:,}")

    # ── Optimizer & loss ──
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    argu_loss = argparse.Namespace(
        beta_delta=args.huber_beta,
        gradient_loss_weight=getattr(args, "gradient_loss_weight", 0.5),
        gradient_base_loss=getattr(args, "gradient_base_loss", "mse"),
    )
    criterion = get_loss_function(args.loss_type, argu_loss)
    logger.info(f"Loss function: {args.loss_type}")

    # ── Training loop ──
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0

    # Track histories for diagnostic plots
    train_hist, valid_hist = make_history_dicts()

    logger.start_task("Training", f"{args.num_epochs} epochs, patience={args.patience}")

    for epoch in range(1, args.num_epochs + 1):
        t0 = time.time()

        # ── Rebuild train loader with random site subset ──
        if use_site_sampling:
            epoch_files = list(
                rng.choice(all_train_files, size=args.sites_per_epoch, replace=False)
            )
            train_ds = make_train_ds(epoch_files)
            train_loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=True,
            )
        elif args.random_stride > 0:
            # Resample random days each epoch, rebuild loader
            train_ds.resample()
            train_loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=True,
            )
        else:
            # Fixed loader, build once on first epoch
            if epoch == 1:
                train_loader = DataLoader(
                    train_ds,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    drop_last=True,
                )

        train_loss, loss_recon, loss_grad = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            max_grad_norm=args.max_grad_norm,
        )
        val_loss, val_rmse, val_r2, val_r2_persite = validate(
            model,
            val_loader,
            criterion,
            device,
            dataset=val_ds,
            n_target_days=n_target_days,
            full_year=args.full_year,
        )
        scheduler.step(val_loss)

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        r2ps_str = f"{val_r2_persite:.4f}" if not np.isnan(val_r2_persite) else "n/a"

        # Base log line
        log_parts = [
            f"Epoch {epoch:3d}/{args.num_epochs}",
            f"train_loss={train_loss:.6f}",
        ]
        # Add loss components if using gradient loss
        if args.loss_type == "gradient":
            log_parts.append(f"L_recon={loss_recon:.6f}")
            log_parts.append(f"L_grad={loss_grad:.6f}")

        log_parts.extend(
            [
                f"val_loss={val_loss:.6f}",
                f"val_RMSE={val_rmse:.5f}",
                f"val_R²={val_r2:.4f}",
                f"val_R²_persite={r2ps_str}",
                f"lr={lr:.2e}",
                f"({elapsed:.1f}s)",
            ]
        )
        logger.info("  ".join(log_parts))

        # Record metrics for diagnostic plots
        train_hist["loss"].append(train_loss)
        train_hist["rmse"].append(np.nan)  # not computed on train set
        train_hist["r2"].append(np.nan)
        valid_hist["loss"].append(val_loss)
        valid_hist["rmse"].append(val_rmse)
        valid_hist["r2"].append(val_r2)

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            ckpt_path = os.path.join(ckpt_dir, "best_model.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_rmse": val_rmse,
                    "val_r2": val_r2,
                    "val_r2_persite": val_r2_persite,
                    "args": vars(args),
                    "norm_stats": norm_stats,
                    "pft_list": pft_list,
                    "train_files": train_files,
                    "val_files": val_files,
                    "lai_norms": lai_norms,
                },
                ckpt_path,
            )
            logger.success(f"  ✓ New best model saved (val_loss={val_loss:.6f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                logger.warning(
                    f"Early stopping at epoch {epoch} "
                    f"(no improvement for {args.patience} epochs)"
                )
                break

    logger.success(
        f"Training complete. Best epoch: {best_epoch}, "
        f"best val_loss: {best_val_loss:.6f}"
    )

    # ── Diagnostic plots ──
    if len(train_hist["loss"]) > 0:
        plot_loss_histories(
            train_hist["loss"],
            valid_hist["loss"],
            filename=os.path.join(exp_dir, "loss_history.png"),
            logger=logger,
        )
        plot_metric_histories(
            train_hist,
            valid_hist,
            filename=os.path.join(exp_dir, "metric_history.png"),
            logger=logger,
        )

    # ── Save final config ──
    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    logger.info(f"Config saved to {config_path}")


if __name__ == "__main__":
    run_training()
