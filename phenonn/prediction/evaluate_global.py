# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Stefan Barbu, Kazem Ardaneh

"""Evaluate a global NetCDF checkpoint on an explicit primary-site split."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from phenonn.data.dataset_netcdf import GlobalLAIDataset, SPLIT_CODES
from phenonn.training.train_global import (
    build_model,
    model_arguments,
    parse_years,
    read_site_pool,
    run_epoch,
)


def evaluate_global(args=None):
    if args is None:
        args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    configuration = checkpoint["config"]
    split_code = SPLIT_CODES[args.split]
    site_ids, _ = read_site_pool(
        configuration["selection"], split=split_code, primary_only=True
    )
    if args.max_sites and site_ids.size > args.max_sites:
        rng = np.random.default_rng(args.seed)
        site_ids = np.sort(rng.choice(site_ids, size=args.max_sites, replace=False))
    years = parse_years(args.years)
    dataset = GlobalLAIDataset(
        era_dir=configuration["era_dir"],
        target_dir=configuration["target_dir"],
        selection_path=configuration["selection"],
        years=years,
        site_ids=site_ids,
        split=args.split,
        include_auxiliary=False,
        seq_length=configuration["seq_length"],
        normalize=configuration["normalize"],
        norm_stats_path=(
            configuration["norm_stats"] if configuration["normalize"] else None
        ),
    )
    model = build_model(**model_arguments(configuration, dataset.feature_channels))
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    metrics = run_epoch(model, loader, device)
    result = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "split": args.split,
        "primary_only": True,
        "years": years,
        "n_sites": int(site_ids.size),
        "n_samples": len(dataset),
        "normalized": bool(configuration["normalize"]),
        "metrics": metrics,
    }
    output = (
        Path(args.output)
        if args.output
        else checkpoint_path.parent
        / f"evaluation_{args.split}_{years[0]}_{years[-1]}.json"
    )
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=tuple(SPLIT_CODES), required=True)
    parser.add_argument("--years", required=True)
    parser.add_argument("--max-sites", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate_global()
