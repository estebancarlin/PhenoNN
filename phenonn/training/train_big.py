# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Stefan Barbu, Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""
Training pipeline for large-scale LAI prediction from per-year CSV archives.

This script implements a scalable training workflow for Leaf Area Index (LAI)
models on datasets that are too large to fit entirely in memory. Rather than
loading all available sites and years, each training epoch operates on a
randomly sampled subset of sites and years, allowing efficient training on
multi-decade continental or global datasets.

The workflow is designed around yearly feature and target files that are
streamed from disk on demand through ``BigLAIDataset``. Only the files
required for the current epoch are accessed, minimizing memory usage and
startup time.

Training Strategy
-----------------
At the beginning of each epoch:

1. A random subset of sites is sampled from a predefined spatial grid.
2. A random subset of training years is selected.
3. A new ``BigLAIDataset`` is instantiated for the sampled sites and years.
4. The model is trained for one epoch on the resulting subset.

This procedure provides stochastic coverage of very large datasets while
keeping memory requirements approximately constant.

Validation Strategy
-------------------
A fixed validation dataset is constructed once at startup from a disjoint
pool of sites and remains unchanged throughout training.

This design ensures:

* Consistent model evaluation across epochs.
* No overlap between training and validation sites.
* Stable early-stopping and model-selection criteria.

Missing Data Handling
---------------------
Target observations may contain missing values originating from unavailable
satellite products or incomplete source archives.

Rather than discarding entire samples, the training and validation loops use
a NaN-safe mean squared error loss that:

* ignores missing target positions,
* computes loss only over valid observations,
* preserves partially observed samples,
* avoids substantial data loss from sparse missing values.

If an entire target sequence is missing, the sample contributes zero loss and
does not affect parameter updates.

Normalization
-------------
No feature or target normalization is applied.

This choice eliminates the expensive full-dataset scan required to compute
global statistics and enables immediate training on very large corpora.

Models therefore operate directly on raw physical units.

Future workflows may optionally introduce normalization using statistics
computed from representative subsets of sites.

Supported Functionality
-----------------------
* Random site and year sampling per epoch.
* Streaming dataset loading from yearly CSV files.
* Fixed validation split with disjoint site pools.
* NaN-safe loss computation.
* Gradient clipping.
* Learning-rate scheduling.
* Early stopping.
* Automatic checkpointing of the best model.
* Training diagnostics and metric visualization.
* Experiment configuration export.

Data Layout
-----------
Feature files:

    features_dir/feature_{year}.csv

Target files:

    target_dir/target_{year}.csv

Each yearly file contains observations for all available sites within that
year. Dataset construction filters rows to the selected site identifiers.

Input and Output Shapes
-----------------------
Input features:

    (batch_size, n_features, sequence_length)

Targets:

    (batch_size, 1, 36)

Predictions:

    (batch_size, 1, 36)

where the 36 target values correspond to dekadal or periodic LAI
observations within the prediction year.

Examples
--------
Train an LSTM model on a large spatial grid:

>>> phenonn train-big \
...     --features_dir data/data_features \
...     --target_dir data/data_targets \
...     --row_min 900 --row_max 1099 \
...     --col_min 1500 --col_max 1699 \
...     --train_years 1993-2010 \
...     --val_years 2011-2018 \
...     --n_sites_per_epoch 1000 \
...     --n_years_per_epoch 5 \
...     --n_val_sites 2000 \
...     --type lstm \
...     --hidden_size 128 \
...     --num_layers 2 \
...     --num_epochs 30

Notes
-----
This script is intended for large-scale LAI training experiments where the
full dataset may contain millions of site-year combinations distributed
across multiple decades. Memory usage is primarily determined by the
number of sites and years sampled for the current epoch rather than the
total size of the archive.
"""

import os
import json
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from phenonn.utils.logger import Logger
from phenonn.utils.utils import FileUtils
from phenonn.utils.evaluater import get_loss_function
from phenonn.utils.diagnostics import (
    plot_loss_histories,
    plot_metric_histories,
    make_history_dicts,
)

from phenonn.data.dataset_big import (
    BigLAIDataset,
    generate_site_ids_from_range,
    ALL_FEATURES,
)
from phenonn.training.train_flat import (
    _base_parser,
    parse_year_list,
    build_model,
)


FEATURE_CHANNELS = len(ALL_FEATURES)  # 31


# ── NaN-safe MSE & training loops ────────────────────────────────────────────
#
# Some (site, year) targets coming from BigLAIDataset contain NaN for the
# dekads whose source H5 file was missing (e.g. 1995, 1997, 2010, 2015, 2017
# in the user's THEIA AVHRR archive). Rather than drop the whole 36-value
# sample for one missing day (~30% data loss), we feed it through unchanged
# and mask out the NaN positions in the loss computation.


def nan_safe_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    MSE that ignores positions where `target` is NaN.

    Reduction is the mean over the *valid* positions, so the loss magnitude
    stays comparable across samples regardless of how many days were
    missing.
    """
    mask = torch.isfinite(target)
    n_valid = mask.sum()
    if n_valid.item() == 0:
        # All NaN — return a finite zero loss so backward() is a no-op.
        return torch.zeros(
            (), device=pred.device, dtype=pred.dtype, requires_grad=pred.requires_grad
        )
    target_clean = torch.where(mask, target, torch.zeros_like(target))
    diff = pred - target_clean
    return (diff.pow(2) * mask.to(pred.dtype)).sum() / n_valid


def train_one_epoch(model, loader, criterion, optimizer, device, max_grad_norm):
    """
    Drop-in replacement for train_flat.train_one_epoch using NaN-safe MSE.
    `criterion` is accepted but ignored — kept in the signature so that the
    rest of train_big.py doesn't need to change.
    """
    model.train()
    total_weighted_loss, total_valid = 0.0, 0
    for features, targets in loader:
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad()
        preds = model(features)  # (B, 1, 36)
        loss = nan_safe_mse(preds, targets)
        if loss.requires_grad and torch.isfinite(loss).item():
            loss.backward()
            if max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        n_valid = int(torch.isfinite(targets).sum().item())
        if n_valid > 0:
            total_weighted_loss += loss.item() * n_valid
            total_valid += n_valid
    return total_weighted_loss / max(total_valid, 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    """
    Drop-in replacement for train_flat.validate using NaN-safe MSE for the
    loss and masking before computing RMSE / R².
    """
    model.eval()
    total_weighted_loss, total_valid = 0.0, 0
    all_preds, all_targets = [], []

    for features, targets in loader:
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        preds = model(features)
        loss = nan_safe_mse(preds, targets)

        n_valid = int(torch.isfinite(targets).sum().item())
        if n_valid > 0:
            total_weighted_loss += loss.item() * n_valid
            total_valid += n_valid

        all_preds.append(preds.reshape(-1).cpu())
        all_targets.append(targets.reshape(-1).cpu())

    avg_loss = total_weighted_loss / max(total_valid, 1)

    p = torch.cat(all_preds)
    t = torch.cat(all_targets)
    mask = torch.isfinite(t)
    p = p[mask]
    t = t[mask]

    if p.numel() == 0:
        return avg_loss, float("nan"), float("nan")

    rmse = torch.sqrt(torch.mean((p - t) ** 2)).item()
    ss_res = torch.sum((t - p) ** 2)
    ss_tot = torch.sum((t - t.mean()) ** 2)
    r2 = (1.0 - ss_res / (ss_tot + 1e-12)).item()
    return avg_loss, rmse, r2


# ── CLI ───────────────────────────────────────────────────────────────────────


def _remove_args(parser, dests):
    """
    Properly remove arguments from an argparse parser by `dest` name.

    argparse stores actions in three places:
      - parser._actions               (the master list)
      - parser._option_string_actions ({option_string: action})
      - each action_group._group_actions
    All three need to be updated for a re-`add_argument` of the same
    option string to succeed.
    """
    to_remove = {dest for dest in dests}
    for action in list(parser._actions):
        if action.dest in to_remove:
            parser._actions.remove(action)
            for opt in action.option_strings:
                parser._option_string_actions.pop(opt, None)
            for group in parser._action_groups:
                if action in group._group_actions:
                    group._group_actions.remove(action)


def parse_args():
    """
    Reuse train_flat's _base_parser for model / loss / training args, but
    *replace* the data-source args with the per-year-folder + grid-range form.
    """
    base = _base_parser("LAI training — big per-year corpus")

    # Drop train_flat's CSV-path / split args (incompatible with the per-year
    # folder layout) so we can re-define them below with the right semantics.
    _remove_args(
        base,
        [
            "features_csv",
            "target_csv",
            "stats_path",
            "split_mode",
            "val_fraction",
            "train_years",
            "val_years",
        ],
    )

    # ── Data layout ──
    base.add_argument(
        "--features_dir",
        required=True,
        help="Folder containing feature_{year}.csv files",
    )
    base.add_argument(
        "--target_dir", required=True, help="Folder containing target_{year}.csv files"
    )

    # ── Site grid range ──
    base.add_argument("--row_min", type=int, required=True)
    base.add_argument("--row_max", type=int, required=True)
    base.add_argument("--col_min", type=int, required=True)
    base.add_argument("--col_max", type=int, required=True)

    # ── Year ranges (train / val) ──
    base.add_argument(
        "--train_years", required=True, help="e.g. '2000-2015' or '2000,2001,2002'"
    )
    base.add_argument(
        "--val_years",
        required=True,
        help="e.g. '2016-2018' (disjoint from train_years)",
    )

    # ── Per-epoch sampling ──
    base.add_argument(
        "--n_sites_per_epoch",
        type=int,
        default=500,
        help="Random sites loaded per training epoch",
    )
    base.add_argument(
        "--n_years_per_epoch",
        type=int,
        default=3,
        help="Random years sampled per training epoch",
    )
    base.add_argument(
        "--n_val_sites",
        type=int,
        default=200,
        help="Sites used in the (fixed) validation dataset",
    )
    base.add_argument(
        "--val_fraction_of_grid",
        type=float,
        default=0.1,
        help="Fraction of site grid reserved for validation pool",
    )

    return base.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────


def run_training_big():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── Output dirs / logger ──
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
    logger.show_header("LAI training — Big per-year corpus")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(
        f"Feature channels: {FEATURE_CHANNELS}  |  seq_length: {args.seq_length}  "
        f"|  normalize: OFF (raw scales)"
    )

    # ── Build candidate site grid ──
    all_site_ids = generate_site_ids_from_range(
        (args.row_min, args.row_max),
        (args.col_min, args.col_max),
    )
    logger.info(
        f"Site grid       : rows [{args.row_min}, {args.row_max}]  "
        f"cols [{args.col_min}, {args.col_max}]  → "
        f"{len(all_site_ids):,} candidate IDs"
    )

    # ── Train/val site split (random, disjoint pools) ──
    rng_split = np.random.RandomState(args.seed)
    shuffled = rng_split.permutation(all_site_ids).tolist()
    n_val_pool = max(args.n_val_sites, int(args.val_fraction_of_grid * len(shuffled)))
    val_pool = shuffled[:n_val_pool]
    train_pool = shuffled[n_val_pool:]
    logger.info(f"Site pools      : train={len(train_pool):,}  val={len(val_pool):,}")

    # ── Year ranges ──
    train_years = parse_year_list(args.train_years)
    val_years = parse_year_list(args.val_years)
    logger.info(f"Train years     : {train_years}")
    logger.info(f"Val years       : {val_years}")

    # ── Validation dataset (fixed for the run) ──
    n_val_used = min(args.n_val_sites, len(val_pool))
    val_sites = rng_split.choice(val_pool, size=n_val_used, replace=False).tolist()

    logger.start_task(
        "Building validation dataset",
        f"{n_val_used} sites × {len(val_years)} years",
    )
    val_ds = BigLAIDataset(
        features_dir=args.features_dir,
        target_dir=args.target_dir,
        years=val_years,
        site_ids=val_sites,
        seq_length=args.seq_length,
        normalize=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    logger.info(f"Validation samples: {len(val_ds):,}")
    if len(val_ds) == 0:
        raise RuntimeError(
            "Validation dataset is empty. Check that val_years files exist "
            "in {target_dir} and that {val_sites} are present in them."
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
    logger.info(
        f"Per-epoch sampling: {args.n_sites_per_epoch} sites × "
        f"{args.n_years_per_epoch} years"
    )

    # ── Training loop ──
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    train_hist, valid_hist = make_history_dicts()

    logger.start_task(
        "Training",
        f"{args.num_epochs} epochs, patience={args.patience}",
    )

    for epoch in range(1, args.num_epochs + 1):
        t0 = time.time()

        # Fresh random sample of sites and years for this epoch
        epoch_rng = np.random.RandomState(args.seed + epoch)
        n_sites = min(args.n_sites_per_epoch, len(train_pool))
        n_years = min(args.n_years_per_epoch, len(train_years))
        sampled_sites = epoch_rng.choice(
            train_pool, size=n_sites, replace=False
        ).tolist()
        sampled_years = sorted(
            epoch_rng.choice(train_years, size=n_years, replace=False).tolist()
        )

        logger.info(
            f"\nEpoch {epoch:3d}/{args.num_epochs}  "
            f"sites: {n_sites}  years: {sampled_years}"
        )

        train_ds = BigLAIDataset(
            features_dir=args.features_dir,
            target_dir=args.target_dir,
            years=sampled_years,
            site_ids=sampled_sites,
            seq_length=args.seq_length,
            normalize=False,
        )
        if len(train_ds) == 0:
            logger.warning("  Empty training dataset for this epoch — skipping")
            continue

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        logger.info(f"  Train samples: {len(train_ds):,}")

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            args.max_grad_norm,
        )
        val_loss, val_rmse, val_r2 = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        logger.info(
            f"  train={train_loss:.6f}  val={val_loss:.6f}  "
            f"RMSE={val_rmse:.5f}  R²={val_r2:.4f}  "
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
                    "norm_stats": None,  # no normalization
                    "train_site_ids": train_pool,
                    "val_site_ids": val_sites,
                    "model_kind": "big_no_norm",
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

        # Explicit release before next epoch
        del train_ds, train_loader

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
    run_training_big()
