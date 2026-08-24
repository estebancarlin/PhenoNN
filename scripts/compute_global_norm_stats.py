#!/usr/bin/env python3
"""Compute optional global-feature normalization statistics."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from phenonn.data.dataset_netcdf import (
    GLOBAL_ALL_FEATURES,
    LOG_TRANSFORM_FEATURES,
    METEO_FEATURES,
    PFT_FEATURES,
    _non_leap_indices,
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def empty_accumulator():
    return {"count": 0, "sum": 0.0, "sum_squares": 0.0}


def update(accumulator, values, log_transform=False):
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    values = values[finite]
    if log_transform:
        values = np.log1p(np.clip(values, 0.0, None))
    accumulator["count"] += int(values.size)
    accumulator["sum"] += float(values.sum())
    accumulator["sum_squares"] += float(np.square(values).sum())


def finalize(accumulator):
    count = accumulator["count"]
    if count == 0:
        raise RuntimeError("Cannot finalize empty normalization accumulator")
    mean = accumulator["sum"] / count
    variance = max(accumulator["sum_squares"] / count - mean * mean, 0.0)
    return {"mean": mean, "std": max(variance**0.5, 1e-8), "count": count}


def parse_years(value):
    if "-" in value and "," not in value:
        start, end = value.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(year) for year in value.split(",")]


def compute(args):
    target_years = parse_years(args.train_years)
    feature_years = list(range(min(target_years) - 1, max(target_years) + 1))
    with Dataset(args.selection) as selection:
        split = selection.variables["split"][:]
        primary = selection.variables["is_primary"][:].astype(bool)
        site_indices = np.flatnonzero((split == 0) & primary).astype(np.int32)
    if args.max_sites and site_indices.size > args.max_sites:
        rng = np.random.default_rng(args.seed)
        site_indices = np.sort(
            rng.choice(site_indices, size=args.max_sites, replace=False)
        ).astype(np.int32)
    if not site_indices.size:
        raise RuntimeError("No spatial-training primary sites were found")

    accumulators = {feature: empty_accumulator() for feature in GLOBAL_ALL_FEATURES}
    for year in feature_years:
        path = args.era_dir / f"ERA5_daily_pixelset_{year}.nc"
        indices = _non_leap_indices(year)
        print(f"ERA {year}", flush=True)
        with Dataset(path) as source:
            n_source_site = len(source.dimensions["site"])
            for source_start in range(0, n_source_site, args.site_batch):
                source_stop = min(n_source_site, source_start + args.site_batch)
                left = int(np.searchsorted(site_indices, source_start, side="left"))
                right = int(np.searchsorted(site_indices, source_stop, side="left"))
                if left == right:
                    continue
                local_sites = site_indices[left:right] - source_start
                for feature in METEO_FEATURES:
                    slab = np.ma.filled(
                        source.variables[feature][indices, source_start:source_stop],
                        np.nan,
                    )
                    values = slab[:, local_sites]
                    update(
                        accumulators[feature],
                        values,
                        log_transform=feature in LOG_TRANSFORM_FEATURES,
                    )

    with Dataset(args.target_dir / "CO2_annual.nc") as source:
        co2 = {
            int(year): float(value)
            for year, value in zip(
                source.variables["year"][:], source.variables["co2"][:]
            )
        }
    for year in target_years:
        update(
            accumulators["co2"],
            np.full(site_indices.size, co2[year], dtype=np.float64),
        )
        path = args.target_dir / "pft" / f"PFTmap_{year}.nc"
        print(f"PFT {year}", flush=True)
        with Dataset(path) as source:
            fractions = source.variables["pft_frac"][:, :]
            for pft_index, feature in enumerate(PFT_FEATURES):
                values = fractions[pft_index, site_indices]
                update(accumulators[feature], values)

    statistics = {
        feature: finalize(accumulators[feature]) for feature in GLOBAL_ALL_FEATURES
    }
    result = {
        "normalized": True,
        "baseline_default_normalized": False,
        "feature_order": GLOBAL_ALL_FEATURES,
        "log_transform_features": sorted(
            set(GLOBAL_ALL_FEATURES) & LOG_TRANSFORM_FEATURES
        ),
        "selection": str(args.selection.resolve()),
        "selection_sha256": sha256(args.selection),
        "site_filter": "split=0 and is_primary=1",
        "n_sites": int(site_indices.size),
        "target_years": target_years,
        "feature_years": feature_years,
        "max_sites": args.max_sites,
        "seed": args.seed,
        "statistics": statistics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(f"Complete: {args.output}", flush=True)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--era-dir", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-years", default="1993-2014")
    parser.add_argument("--site-batch", type=int, default=2048)
    parser.add_argument("--max-sites", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    compute(parse_args())


if __name__ == "__main__":
    main()
