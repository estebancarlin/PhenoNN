# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Stefan Barbu, Kazem Ardaneh

"""Train unchanged PhenoNN sequence models on selected-site NetCDF files."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from netCDF4 import Dataset
from torch.utils.data import DataLoader

from phenonn.data.dataset_netcdf import GlobalLAIDataset
from phenonn.models.fcn import FCN
from phenonn.models.rnn import RNN_GRU, RNN_LSTM
from phenonn.models.transformer import EncoderTorch
from phenonn.models.transformerbis import BiTransformer
from phenonn.utils.wrappers import Every10DaysWrapper, permuteWrapper


GLOBAL_MODEL_TYPES = ("lstm", "gru", "transformer", "bitransformer", "fcn")
PRIVATE_WANDB_CONFIG = {
    "era_dir",
    "norm_stats",
    "output_dir",
    "selection",
    "target_dir",
}


def parse_years(value):
    if "-" in value and "," not in value:
        start, end = value.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(year) for year in value.split(",")]


def read_site_pool(selection_path, split, primary_only=False):
    with Dataset(selection_path) as selection:
        site_ids = np.asarray(selection.variables["site_id"][:]).astype(str)
        splits = selection.variables["split"][:]
        primary = selection.variables["is_primary"][:].astype(bool)
        chunks = selection.variables["arco_chunk_id"][:]
    keep = splits == split
    if primary_only:
        keep &= primary
    return site_ids[keep], chunks[keep]


def sample_sites_by_chunk(site_ids, chunks, n_chunks, max_sites, rng):
    unique_chunks = np.unique(chunks)
    count = min(n_chunks, unique_chunks.size)
    selected_chunks = rng.choice(unique_chunks, size=count, replace=False)
    indices = np.flatnonzero(np.isin(chunks, selected_chunks))
    if max_sites and indices.size > max_sites:
        indices = rng.choice(indices, size=max_sites, replace=False)
    return site_ids[np.sort(indices)].tolist()


def build_model(
    model_type,
    feature_channels,
    hidden_size,
    num_layers,
    seq_length=720,
    embed_size=64,
    nhead=4,
    forward_expansion=4,
    dropout=0.0,
    dropout_trans=0.0,
    feed_forward_trans=4,
    feed_forward_encoder=4,
):
    if model_type in ("lstm", "gru"):
        model_class = RNN_LSTM if model_type == "lstm" else RNN_GRU
        base = model_class(
            feature_channel=feature_channels,
            output_channel=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )
    elif model_type == "transformer":
        base = EncoderTorch(
            feature_channel=feature_channels,
            output_channel=1,
            embed_size=embed_size,
            num_layers=num_layers,
            heads=nhead,
            forward_expansion=forward_expansion,
            seq_length=seq_length,
            dropout=dropout,
        )
    elif model_type == "bitransformer":
        base = permuteWrapper(
            BiTransformer(
                input_dim=feature_channels,
                d_model=hidden_size,
                feed_forward_trans=feed_forward_trans,
                feed_forward_encoder=feed_forward_encoder,
                output_dim=1,
                nr_blocks=num_layers,
                dropout_trans=dropout_trans,
                dropout_encoder=dropout,
                n_pft=15,
            )
        )
    elif model_type == "fcn":
        base = FCN(
            feature_channel=feature_channels,
            output_channel=1,
            num_layers=num_layers,
            hidden_size=hidden_size,
            seq_length=seq_length,
            dim_expand=0,
        )
    else:
        raise ValueError(f"Unsupported global model type: {model_type}")
    return Every10DaysWrapper(base)


def model_arguments(configuration, feature_channels):
    """Return architecture arguments, including defaults for older checkpoints."""
    return {
        "model_type": configuration["type"],
        "feature_channels": feature_channels,
        "hidden_size": configuration["hidden_size"],
        "num_layers": configuration["num_layers"],
        "seq_length": configuration.get("seq_length", 720),
        "embed_size": configuration.get("embed_size", 64),
        "nhead": configuration.get("nhead", 4),
        "forward_expansion": configuration.get("forward_expansion", 4),
        "dropout": configuration.get("dropout", 0.0),
        "dropout_trans": configuration.get("dropout_trans", 0.0),
        "feed_forward_trans": configuration.get("feed_forward_trans", 4),
        "feed_forward_encoder": configuration.get("feed_forward_encoder", 4),
    }


def nan_safe_mse(prediction, target):
    valid = torch.isfinite(target)
    if not torch.any(valid):
        return prediction.sum() * 0.0
    clean_target = torch.where(valid, target, torch.zeros_like(target))
    squared_error = torch.square(prediction - clean_target)
    return squared_error[valid].mean()


def initialize_wandb(args, configuration, output_dir, model):
    """Start optional W&B tracking without exposing local data paths."""
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError(
            'W&B monitoring requires `python -m pip install -e ".[tracking]"`.'
        ) from error

    tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
    public_configuration = {
        key: value
        for key, value in configuration.items()
        if key not in PRIVATE_WANDB_CONFIG
    }
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.experiment,
        group=args.wandb_group or None,
        tags=tags or None,
        mode=args.wandb_mode,
        dir=str(output_dir),
        config=public_configuration,
    )
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    run.config.update({"model_parameters": parameter_count}, allow_val_change=True)
    run.define_metric("epoch")
    for namespace in ("train/*", "validation/*", "optimizer/*", "runtime/*"):
        run.define_metric(namespace, step_metric="epoch")
    return run


def wandb_epoch_metrics(record, learning_rate, epoch_seconds):
    """Flatten one history record into stable W&B metric names."""
    return {
        "epoch": record["epoch"],
        "train/mse": record["train"]["mse"],
        "train/rmse": record["train"]["rmse"],
        "train/n_valid": record["train"]["n_valid"],
        "train/sites": record["train_sites"],
        "train/years": len(record["train_years"]),
        "validation/mse": record["validation"]["mse"],
        "validation/rmse": record["validation"]["rmse"],
        "validation/r2": record["validation"].get("r2", float("nan")),
        "validation/n_valid": record["validation"]["n_valid"],
        "optimizer/learning_rate": learning_rate,
        "runtime/epoch_seconds": epoch_seconds,
    }


def run_epoch(model, loader, device, optimizer=None, max_grad_norm=1.0):
    training = optimizer is not None
    model.train(training)
    total_squared_error = 0.0
    total_valid = 0
    predictions = []
    targets = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for features, target in loader:
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad()
            prediction = model(features)
            loss = nan_safe_mse(prediction, target)
            if training:
                loss.backward()
                if max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            valid = torch.isfinite(target)
            total_squared_error += float(
                torch.square(prediction[valid] - target[valid]).sum().item()
            )
            total_valid += int(valid.sum().item())
            if not training:
                predictions.append(prediction[valid].detach().cpu())
                targets.append(target[valid].detach().cpu())
    mse = total_squared_error / max(total_valid, 1)
    metrics = {"mse": mse, "rmse": mse**0.5, "n_valid": total_valid}
    if not training and targets:
        prediction = torch.cat(predictions)
        target = torch.cat(targets)
        denominator = torch.sum(torch.square(target - target.mean()))
        metrics["r2"] = float(
            1.0 - torch.sum(torch.square(target - prediction)) / (denominator + 1e-12)
        )
    return metrics


def dataset_arguments(args):
    return {
        "era_dir": args.era_dir,
        "target_dir": args.target_dir,
        "selection_path": args.selection,
        "seq_length": args.seq_length,
        "normalize": args.normalize,
        "norm_stats_path": args.norm_stats if args.normalize else None,
    }


def run_training_global(args=None):
    if args is None:
        args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_years = parse_years(args.train_years)
    validation_years = parse_years(args.validation_years)
    train_ids, train_chunks = read_site_pool(args.selection, split=0)
    validation_ids, _ = read_site_pool(args.selection, split=1, primary_only=True)
    if args.max_validation_sites and validation_ids.size > args.max_validation_sites:
        validation_ids = np.sort(
            rng.choice(
                validation_ids,
                size=args.max_validation_sites,
                replace=False,
            )
        )

    common = dataset_arguments(args)
    validation = GlobalLAIDataset(
        **common,
        years=validation_years,
        site_ids=validation_ids,
        split="validation",
        include_auxiliary=False,
    )
    validation_loader = DataLoader(
        validation,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = build_model(**model_arguments(vars(args), validation.feature_channels)).to(
        device
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    output_dir = Path(args.output_dir) / args.experiment
    output_dir.mkdir(parents=True, exist_ok=True)
    configuration = vars(args).copy()
    configuration.update(
        {
            "device": str(device),
            "feature_names": validation.feature_names,
            "feature_channels": validation.feature_channels,
            "raw_units_default": not args.normalize,
        }
    )
    (output_dir / "config.json").write_text(
        json.dumps(configuration, indent=2) + "\n", encoding="utf-8"
    )
    wandb_run = initialize_wandb(args, configuration, output_dir, model)
    if wandb_run is not None:
        print(f"W&B run: {wandb_run.url or args.wandb_mode}", flush=True)

    history = []
    best_validation = float("inf")
    best_epoch = 0
    for epoch in range(1, args.num_epochs + 1):
        epoch_started = time.perf_counter()
        selected_sites = sample_sites_by_chunk(
            train_ids,
            train_chunks,
            args.train_chunks_per_epoch,
            args.train_sites_per_epoch,
            rng,
        )
        count_year = min(args.train_years_per_epoch, len(train_years))
        selected_years = sorted(
            rng.choice(train_years, size=count_year, replace=False).tolist()
        )
        training = GlobalLAIDataset(
            **common,
            years=selected_years,
            site_ids=selected_sites,
            split="train",
            include_auxiliary=True,
        )
        training_loader = DataLoader(
            training,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        train_metrics = run_epoch(
            model,
            training_loader,
            device,
            optimizer=optimizer,
            max_grad_norm=args.max_grad_norm,
        )
        validation_metrics = run_epoch(model, validation_loader, device)
        record = {
            "epoch": epoch,
            "train_sites": len(selected_sites),
            "train_years": selected_years,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if wandb_run is not None:
            wandb_run.log(
                wandb_epoch_metrics(
                    record,
                    optimizer.param_groups[0]["lr"],
                    time.perf_counter() - epoch_started,
                )
            )
        if validation_metrics["mse"] < best_validation:
            best_validation = validation_metrics["mse"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "validation": validation_metrics,
                    "config": configuration,
                },
                output_dir / "best_model.pth",
            )
        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )
    if wandb_run is not None:
        wandb_run.summary["best_epoch"] = best_epoch
        wandb_run.summary["best_validation_mse"] = best_validation
        wandb_run.summary["best_validation_rmse"] = best_validation**0.5
        wandb_run.finish()
    return model, history


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--era-dir", required=True)
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--output-dir", default="./runs")
    parser.add_argument("--experiment", default="global_lai")
    parser.add_argument("--train-years", default="1993-2014")
    parser.add_argument("--validation-years", default="2015-2016")
    parser.add_argument("--type", choices=GLOBAL_MODEL_TYPES, default="lstm")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--seq-length", type=int, default=720)
    parser.add_argument("--embed-size", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--forward-expansion", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--dropout-trans", type=float, default=0.0)
    parser.add_argument("--feed-forward-trans", type=int, default=4)
    parser.add_argument("--feed-forward-encoder", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--train-chunks-per-epoch", type=int, default=32)
    parser.add_argument("--train-sites-per-epoch", type=int, default=1000)
    parser.add_argument("--train-years-per-epoch", type=int, default=5)
    parser.add_argument("--max-validation-sites", type=int, default=500)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--norm-stats", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="phenonn-global-lai")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-group", default="")
    parser.add_argument("--wandb-tags", default="")
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="online"
    )
    parsed = parser.parse_args()
    if parsed.normalize and not parsed.norm_stats:
        parser.error("--normalize requires --norm-stats")
    return parsed


if __name__ == "__main__":
    run_training_global()
