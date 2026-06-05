# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Stefan Barbu, Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""
Training pipeline for LAI prediction from flat CSV datasets.

This script trains sequence-based neural networks to predict Leaf Area Index
(LAI) from meteorological, ecological, and land-cover features stored in two
flat CSV files containing daily predictors and sparse LAI observations.

The workflow is intended for datasets that fit comfortably in memory and
provides a complete training pipeline including dataset construction,
normalization, model initialization, validation, checkpointing, and
diagnostic visualization.

Problem Formulation
-------------------
Each sample corresponds to a unique ``(site_id, year)`` pair.

Input:
    Tensor of shape::

        (n_features, sequence_length)

    containing a rolling history of environmental forcing variables,
    typically spanning approximately two years.

Output:
    Tensor of shape::

        (1, 36)

    containing 36 LAI observations for the target year, corresponding to
    days 5, 15, and 25 of each month.

The model predicts the complete annual LAI trajectory from the historical
environmental context.

Supported Model Architectures
-----------------------------
LSTM
    Multi-layer recurrent neural network using Long Short-Term Memory units.

GRU
    Multi-layer recurrent neural network using Gated Recurrent Units.

Transformer
    Self-attention encoder model for sequence modeling.

BiTransformer
    Transformer architecture incorporating environmental forcing and
    plant functional type information.

All models are wrapped by ``Every10DaysWrapper`` to produce LAI predictions
at the observation frequency used by the target dataset.

Dataset Splitting
-----------------
Two validation strategies are supported.

Site Split
    Training and validation use disjoint sets of sites.

    This evaluates spatial generalization and answers the question:

        "Can the model predict LAI at previously unseen locations?"

Year Split
    All sites are retained, but training and validation use different years.

    This evaluates temporal generalization and answers the question:

        "Can the model predict vegetation dynamics in unseen years?"

Normalization
-------------
Feature normalization statistics are computed from the training set and
stored in a JSON file for reproducibility.

If a normalization file already exists, it can be reused to avoid
recomputation.

The normalization pipeline:

1. Computes feature means and standard deviations.
2. Applies z-score normalization to input features.
3. Normalizes target values for training stability.
4. Stores statistics alongside model checkpoints.

Training Features
-----------------
* Mini-batch training with PyTorch DataLoaders.
* Configurable model architectures.
* Multiple loss functions (MSE, MAE, Huber, SmoothL1, normalized losses,
  and gradient-aware losses).
* Gradient clipping.
* Adaptive learning-rate scheduling.
* Early stopping.
* Best-model checkpointing.
* Training and validation diagnostics.
* Reproducible experiment logging.

Evaluation Metrics
------------------
Validation performance is reported using:

Loss
    Training objective selected by ``--loss_type``.

RMSE
    Root Mean Squared Error computed across all predicted LAI values.

R²
    Coefficient of determination computed across all validation samples.

Data Requirements
-----------------
Feature CSV
    Daily meteorological, ecological, and land-cover predictors indexed
    by site and date.

Target CSV
    Sparse LAI observations containing three measurements per month
    (days 5, 15, and 25).

The dataset builder constructs fixed-length historical windows ending in
the target year and aligns them with the corresponding annual LAI sequence.

Examples
--------
Train an LSTM model:

>>> python -m phenonn.run_train_flat \
...     --features_csv data/features.csv \
...     --target_csv data/targets.csv \
...     --type lstm \
...     --hidden_size 128 \
...     --num_layers 2 \
...     --num_epochs 50 \
...     --batch_size 32

Train a transformer model:

>>> python -m phenonn.run_train_flat \
...     --features_csv data/features.csv \
...     --target_csv data/targets.csv \
...     --type transformer \
...     --embed_size 128 \
...     --num_layers 4

Notes
-----
This training script is intended for moderate-sized datasets that can be
indexed efficiently from flat CSV files. For very large archives distributed
across yearly files, the streaming workflow implemented in
``main_big.py`` provides a more scalable alternative.
"""

import os
import argparse
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from phenonn.utils.logger import Logger
from phenonn.utils.utils import FileUtils
from phenonn.utils.evaluater import get_loss_function

from phenonn.data.dataset_flat import (
    LAIDataset,
    compute_norm_stats,
    load_norm_stats,
    get_site_ids,
    split_sites_by_fraction,
    ALL_FEATURES,
    PFT_COLS,
)
from phenonn.models.rnn import RNN_LSTM, RNN_GRU
from phenonn.models.transformer import EncoderTorch
from phenonn.models.transformerbis import BiTransformer
from phenonn.utils.wrappers import Every10DaysWrapper, permuteWrapper
from phenonn.utils.diagnostics import (
    plot_loss_histories,
    plot_metric_histories,
    make_history_dicts,
)


FEATURE_CHANNELS = len(ALL_FEATURES)  # 31


# ── CLI ───────────────────────────────────────────────────────────────────────


def _base_parser(description: str) -> argparse.ArgumentParser:
    """Return a parser pre-loaded with all shared arguments."""
    p = argparse.ArgumentParser(description=description)

    # Data
    p.add_argument(
        "--features_csv", required=True, help="Path to flat daily features CSV"
    )
    p.add_argument(
        "--target_csv",
        required=True,
        help="Path to sparse LAI targets CSV (days 5/15/25)",
    )
    p.add_argument(
        "--stats_path",
        default="",
        help="Path to precomputed norm_stats.json (computed if empty)",
    )
    p.add_argument("--output_dir", default="./runs", help="Root output directory")
    p.add_argument(
        "--experiment", default="exp_flat", help="Experiment name (creates sub-folder)"
    )

    # Split
    p.add_argument(
        "--split_mode",
        default="site",
        choices=["site", "year"],
        help="'site': leave-site-out. 'year': all sites, split by year.",
    )
    p.add_argument(
        "--val_fraction",
        type=float,
        default=0.2,
        help="Fraction of sites held out for validation (split_mode=site)",
    )
    p.add_argument(
        "--train_years",
        default="",
        help="Comma-separated or range (e.g. 2000-2002) training years",
    )
    p.add_argument(
        "--val_years", default="", help="Comma-separated or range validation years"
    )
    p.add_argument("--seed", type=int, default=42)

    # Model
    p.add_argument(
        "--type",
        default="lstm",
        choices=["lstm", "gru", "transformer", "bitransformer", "aelstm"],
        help="Base model architecture",
    )
    p.add_argument(
        "--n_attn_blocks",
        type=int,
        default=2,
        help="Number of self-attention blocks after the LSTM in aelstm",
    )
    p.add_argument(
        "--dropout_att",
        type=float,
        default=0.0,
        help="Dropout inside aelstm self-attention blocks "
        "(separate from --dropout which controls the LSTM)",
    )
    p.add_argument(
        "--hidden_size",
        type=int,
        default=64,
        help="LSTM/GRU hidden size or Transformer d_model",
    )
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument(
        "--seq_length",
        type=int,
        default=720,
        help="Feature window length in days (default 720 ≈ 2 years)",
    )
    p.add_argument(
        "--embed_size", type=int, default=64, help="Transformer embedding size"
    )
    p.add_argument(
        "--nhead", type=int, default=4, help="Transformer number of attention heads"
    )
    p.add_argument("--forward_expansion", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument(
        "--dropout_trans",
        type=float,
        default=0.0,
        help="Dropout in BiTransformer's Transformer block",
    )
    p.add_argument(
        "--feed_forward_trans",
        type=int,
        default=4,
        help="Feedforward multiplier in BiTransformer's Transformer block "
        "(dim_feedforward = feed_forward_trans * hidden_size)",
    )
    p.add_argument(
        "--feed_forward_encoder",
        type=int,
        default=4,
        help="Feedforward multiplier in BiTransformer's Encoder blocks",
    )

    # Training
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument(
        "--loss_type",
        default="mse",
        choices=["mse", "mae", "huber", "nmae", "nmse", "smoothl1", "gradient"],
        help="Loss function",
    )
    p.add_argument("--huber_beta", type=float, default=1.0)
    p.add_argument(
        "--gradient_loss_weight",
        type=float,
        default=0.5,
        help="λ for temporal gradient penalty (loss_type=gradient)",
    )
    p.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Gradient clipping (0 = disabled)",
    )
    p.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (in full epochs)",
    )
    p.add_argument("--num_workers", type=int, default=4)

    return p


def parse_args():
    return _base_parser("LAI Prediction — Flat CSV").parse_args()


def parse_args_bis():
    p = _base_parser("LAI Prediction — Flat CSV + Sharding")

    # Sharding
    p.add_argument(
        "--n_shards",
        type=int,
        default=5,
        help="Number of shards to split training sites into. "
        "Each full epoch iterates through all shards in sequence. "
        "Rule of thumb: set so one shard fits comfortably in RAM.",
    )
    p.add_argument(
        "--norm_sample_size",
        type=int,
        default=1000,
        help="Number of sites used to compute normalization stats. "
        "A random sample avoids loading the full dataset at once. "
        "Default 1000 is sufficient for stable mean/std estimates.",
    )
    p.add_argument(
        "--n_val_sites",
        type=int,
        default=500,
        help="Max number of validation sites kept in RAM throughout training. "
        "0 = use all val sites (may be large). Default: 500.",
    )

    return p.parse_args()


def parse_year_list(s: str):
    """Parse '2000,2001,2002' or '2000-2002' into [2000, 2001, 2002]."""
    if not s:
        return []
    if "-" in s and "," not in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


# ── Model factory ─────────────────────────────────────────────────────────────


def build_model(args) -> nn.Module:
    """Instantiate base model and wrap with Every10DaysWrapper."""
    t = args.type.lower()
    if t == "lstm":
        base = RNN_LSTM(
            feature_channel=FEATURE_CHANNELS,
            output_channel=1,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
        )
    elif t == "gru":
        base = RNN_GRU(
            feature_channel=FEATURE_CHANNELS,
            output_channel=1,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
        )
    elif t == "transformer":
        base = EncoderTorch(
            feature_channel=FEATURE_CHANNELS,
            output_channel=1,
            embed_size=args.embed_size,
            num_layers=args.num_layers,
            heads=args.nhead,
            forward_expansion=args.forward_expansion,
            seq_length=args.seq_length,
            dropout=args.dropout,
            causal=False,
        )
    elif t == "bitransformer":
        # BiTransformer expects (B, L, C); permuteWrapper handles the (B, C, L) ↔ (B, L, C) swap.
        # n_pft = 15: the last 15 features are always the PFT fraction columns.
        base = permuteWrapper(
            BiTransformer(
                input_dim=FEATURE_CHANNELS,
                d_model=args.hidden_size,
                feed_forward_trans=args.feed_forward_trans,
                feed_forward_encoder=args.feed_forward_encoder,
                output_dim=1,
                nr_blocks=args.num_layers,
                dropout_trans=args.dropout_trans,
                dropout_encoder=args.dropout,
                n_pft=len(PFT_COLS),  # 15
            )
        )
    else:
        raise ValueError(f"Unsupported model type: {args.type}")
    return Every10DaysWrapper(base)


# ── Training / validation ─────────────────────────────────────────────────────


def train_one_epoch(model, loader, criterion, optimizer, device, max_grad_norm):
    model.train()
    total_loss, n = 0.0, 0
    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        preds = model(features)  # (B, 1, 36)
        loss = criterion(preds, targets)
        loss.backward()
        if max_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += loss.item() * features.size(0)
        n += features.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, n = 0.0, 0
    all_preds, all_targets = [], []

    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)
        preds = model(features)  # (B, 1, 36)
        total_loss += criterion(preds, targets).item() * features.size(0)
        n += features.size(0)
        # Flatten all 36 obs per sample for pooled metrics
        all_preds.append(preds.reshape(-1).cpu())
        all_targets.append(targets.reshape(-1).cpu())

    avg_loss = total_loss / max(n, 1)
    p = torch.cat(all_preds)
    t = torch.cat(all_targets)
    rmse = torch.sqrt(torch.mean((p - t) ** 2)).item()
    ss_res = torch.sum((t - p) ** 2)
    ss_tot = torch.sum((t - t.mean()) ** 2)
    r2 = (1.0 - ss_res / (ss_tot + 1e-12)).item()
    return avg_loss, rmse, r2


# ── Main ──────────────────────────────────────────────────────────────────────


def run_training_flat():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Directories
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
    logger.show_header("LAI Prediction — Flat CSV")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(
        f"Feature channels: {FEATURE_CHANNELS}  |  seq_length: {args.seq_length}"
    )

    # ── Site IDs ──
    all_site_ids = get_site_ids(args.features_csv)
    logger.info(f"Found {len(all_site_ids)} unique sites")

    # ── Split ──
    train_years = val_years = None
    if args.split_mode == "site":
        train_ids, val_ids = split_sites_by_fraction(
            all_site_ids, val_fraction=args.val_fraction, seed=args.seed
        )
        norm_ids = train_ids
        logger.info(f"Site split — train: {len(train_ids)}, val: {len(val_ids)}")
    else:
        train_years = parse_year_list(args.train_years)
        val_years = parse_year_list(args.val_years)
        if not train_years or not val_years:
            raise ValueError(
                "--train_years and --val_years are required for year split"
            )
        train_ids = val_ids = all_site_ids
        norm_ids = all_site_ids
        logger.info(f"Year split — train years: {train_years}, val years: {val_years}")

    # ── Normalization stats ──
    stats_path = args.stats_path or os.path.join(exp_dir, "norm_stats.json")
    if args.stats_path and os.path.exists(stats_path):
        logger.info(f"Loading norm stats from {stats_path}")
        norm_stats = load_norm_stats(stats_path)
    else:
        logger.start_task(
            "Computing normalization stats", f"from {len(norm_ids)} training sites"
        )
        norm_stats = compute_norm_stats(
            args.features_csv,
            args.target_csv,
            norm_ids,
            save_path=stats_path,
        )
        logger.success(f"Norm stats saved to {stats_path}")

    for feat, s in norm_stats.items():
        logger.info(f"  {feat:20s}  mean={s['mean']:+.4f}  std={s['std']:.4f}")

    # ── Datasets ──
    train_ds = LAIDataset(
        args.features_csv,
        args.target_csv,
        norm_stats,
        site_ids=train_ids,
        seq_length=args.seq_length,
        years=train_years,
        normalize_target=True,
    )
    val_ds = LAIDataset(
        args.features_csv,
        args.target_csv,
        norm_stats,
        site_ids=val_ids,
        seq_length=args.seq_length,
        years=val_years,
        normalize_target=True,
    )
    logger.info(f"Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}")

    if len(train_ds) == 0:
        raise RuntimeError(
            "Training dataset is empty. Check your CSV paths, site IDs, "
            "and that seq_length <= available history."
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # ── Model ──
    logger.start_task("Initializing model", f"{args.type} + Every10DaysWrapper")
    model = build_model(args).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {n_params:,}")

    # ── Optimizer / scheduler / loss ──
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    loss_args = argparse.Namespace(
        beta_delta=args.huber_beta,
        gradient_loss_weight=args.gradient_loss_weight,
        gradient_base_loss="mse",
    )
    criterion = get_loss_function(args.loss_type, loss_args)
    logger.info(f"Loss: {args.loss_type}")

    # ── Training loop ──
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    train_hist, valid_hist = make_history_dicts()

    logger.start_task("Training", f"{args.num_epochs} epochs, patience={args.patience}")

    for epoch in range(1, args.num_epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, args.max_grad_norm
        )
        val_loss, val_rmse, val_r2 = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        logger.info(
            f"Epoch {epoch:3d}/{args.num_epochs}  "
            f"train={train_loss:.6f}  "
            f"val={val_loss:.6f}  "
            f"RMSE={val_rmse:.5f}  "
            f"R²={val_r2:.4f}  "
            f"lr={lr:.2e}  ({elapsed:.1f}s)"
        )

        train_hist["loss"].append(train_loss)
        train_hist["rmse"].append(float("nan"))
        train_hist["r2"].append(float("nan"))
        valid_hist["loss"].append(val_loss)
        valid_hist["rmse"].append(val_rmse)
        valid_hist["r2"].append(val_r2)

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
                    "args": vars(args),
                    "norm_stats": norm_stats,
                    "train_site_ids": train_ids,
                    "val_site_ids": val_ids,
                },
                ckpt_path,
            )
            logger.success(f"  ✓ Best model saved (val_loss={val_loss:.6f})")
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

    # ── Save config ──
    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    logger.info(f"Config saved to {config_path}")


if __name__ == "__main__":
    run_training_flat()
