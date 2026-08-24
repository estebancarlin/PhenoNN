#!/usr/bin/env python3
"""
create_stats_file.py
====================

Scan the preprocessed NetCDF cubes for a year range and emit one
`norm_stats.json` that contains `{feat: {"mean", "std"}}` for every channel
the training pipeline expects.

Coverage
--------
  - DYNAMIC features        (METEO + DERIVED in phenon.training.config)
  - PFT_COLS                pft1_frac..pft15_frac (averaged across years)
  - LAI                     the target z-scoring stats
  - co2                     when --co2_path is provided

Heavy-tailed dynamic features are log1p-transformed BEFORE the
mean/std are computed, matching what the LAIDataset does at training
time.

Output format is byte-identical to the LaiNN convention so existing
diagnostic scripts continue to work.

Usage
-----
    python -m phenon.dataset_creation.create_stats_file \\
        --features_dir /data/sbarbu/era5_features \\
        --target_dir   /data/sbarbu/targets \\
        --pft_dir      /data/sbarbu/pft \\
        --co2_path     /data/sbarbu/co2_1700_2023_TRENDYv2024.txt \\
        --year_start 1992 --year_end 2010 \\
        --output       /data/sbarbu/norm_stats.json
"""

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import xarray as xr

from phenon.training.config import (
    CO2_FEATURES, DYNAMIC_FEATURES, FEATURES_FNAME, LOG_TRANSFORM_FEATURES,
    N_PFT, PFT_COLS, PFT_FNAME, TARGETS_FNAME,
)


# ── Torch-free local helpers ────────────────────────────────────────────────
# Copied verbatim from phenon.data.lai_dataset so this preprocessing script stays
# importable in the torch-less venv_ERA5 (lai_dataset imports torch at module
# load). Same convention as compute_valid_pixels.py / build_pixelset_targets.py,
# which keep their own copy of _load_pft_vector_array.


def load_co2_lut(path: str) -> Dict[int, float]:
    """Parse the LaiNN-style CO2 file: one `Annee_YYYY=VALUE` per line."""
    lut: Dict[int, float] = {}
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key.lower().startswith("annee_"):
                continue
            try:
                year = int(key.split("_", 1)[1])
                lut[year] = float(value.strip())
            except ValueError:
                continue
    if not lut:
        raise ValueError(f"No `Annee_YYYY=VALUE` entries parsed from {path!r}.")
    return lut


def _load_pft_vector_array(ds: xr.Dataset) -> xr.DataArray:
    """
    Return a single DataArray with a `pft` dim from a PFT file, supporting:

      1. ORCHIDEE PFTmap `maxvegetfrac(time_counter=1, veget=15, lat, lon)`.
      2. 15 separate variables `pft1_frac..pft15_frac`.
      3. One var with a `pft` dim (pixelset `pft_frac(pft, site)`, or grid).
    """
    rename = {}
    for src, dst in [("lat", "latitude"), ("lon", "longitude"),
                     ("veget", "pft")]:
        if src in ds.coords or src in ds.dims:
            rename[src] = dst
    if rename:
        ds = ds.rename(rename)

    if "maxvegetfrac" in ds.data_vars:
        da = ds["maxvegetfrac"]
        if "time_counter" in da.dims:
            da = da.isel(time_counter=0, drop=True)
        return da

    pft_named = [f"pft{k}_frac" for k in range(1, N_PFT + 1)]
    if all(v in ds.data_vars for v in pft_named):
        return xr.concat(
            [ds[v].expand_dims(pft=[k - 1])
             for k, v in enumerate(pft_named, start=1)],
            dim="pft",
        )

    for cand in ("pft_fraction", "PFT", "pft", "pft_frac", "fraction"):
        if cand in ds.data_vars and "pft" in ds[cand].dims:
            return ds[cand]

    raise ValueError(
        f"Could not detect a PFT layout in {list(ds.data_vars)}. "
        f"Expected one of: 'maxvegetfrac(time_counter, veget, lat, lon)' "
        f"(ORCHIDEE), 15 'pftK_frac' vars, or a 3-D var with a 'pft' dim."
    )


# ── Running stats accumulator ───────────────────────────────────────────────


class RunningStats:
    """Two-pass safe online merge of sum, sum-of-squares and count."""

    __slots__ = ("n", "sum_x", "sum_x2")

    def __init__(self) -> None:
        self.n     = 0
        self.sum_x = 0.0
        self.sum_x2 = 0.0

    def update_from_array(self, arr: np.ndarray) -> None:
        finite = np.isfinite(arr)
        if not finite.any():
            return
        x = arr[finite].astype(np.float64)
        self.n     += int(x.size)
        self.sum_x += float(x.sum())
        self.sum_x2 += float(np.square(x).sum())

    def merge(self, other: "RunningStats") -> None:
        self.n     += other.n
        self.sum_x += other.sum_x
        self.sum_x2 += other.sum_x2

    def mean_std(self) -> Dict[str, float]:
        if self.n == 0:
            return {"mean": 0.0, "std": 1.0}
        mean = self.sum_x / self.n
        var  = max(self.sum_x2 / self.n - mean * mean, 0.0)
        std  = math.sqrt(var)
        return {"mean": mean, "std": max(std, 1e-8)}


# ── Feature stats year by year ──────────────────────────────────────────────


def _scan_features_year(
    path: Path, accs: Dict[str, RunningStats],
) -> None:
    ds = xr.open_dataset(path)
    for feat in DYNAMIC_FEATURES:
        if feat not in ds:
            continue
        arr = ds[feat].values.astype(np.float32)
        if feat in LOG_TRANSFORM_FEATURES:
            arr = np.log1p(np.clip(arr, 0.0, None))
        accs[feat].update_from_array(arr)
    ds.close()


def _scan_targets_year(path: Path, lai_acc: RunningStats) -> None:
    ds = xr.open_dataset(path)
    lai_acc.update_from_array(ds["LAI"].values.astype(np.float32))
    ds.close()


def _scan_pft_year(path: Path, accs: Dict[str, RunningStats]) -> None:
    # Pixelset PFT files (pft_frac(pft, site)) are read like
    # lai_dataset._open_pft_array: engine="netcdf4", decode_times=False so the
    # whole process stays on one libhdf5 (features/targets also use netcdf4).
    # Mixing h5netcdf here loads a second bundled libhdf5 → "NetCDF: HDF error".
    ds = xr.open_dataset(path, engine="netcdf4", decode_times=False)
    pft = _load_pft_vector_array(ds)
    arr = pft.values.astype(np.float32)          # (15, site) for pixelset
    for k in range(N_PFT):
        accs[PFT_COLS[k]].update_from_array(arr[k])
    ds.close()


def _co2_stats(co2_lut: Dict[int, float], years: List[int]) -> Dict[str, float]:
    vals = np.array([co2_lut[y] for y in years if y in co2_lut],
                    dtype=np.float64)
    if vals.size == 0:
        return {"mean": 0.0, "std": 1.0}
    mean = float(vals.mean())
    std  = float(vals.std(ddof=0))
    return {"mean": mean, "std": max(std, 1e-8)}


# ── CLI ─────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--features_dir", required=True)
    p.add_argument("--target_dir",   required=True)
    p.add_argument("--pft_dir",      required=True)
    p.add_argument("--co2_path", default="",
                   help="Optional CO2 LUT (Annee_YYYY=VALUE).")
    p.add_argument("--year_start", type=int, required=True)
    p.add_argument("--year_end",   type=int, required=True)
    p.add_argument("--output", required=True,
                   help="Path of the norm_stats.json to write.")
    return p.parse_args()


def main():
    args = parse_args()
    features_dir = Path(args.features_dir).resolve()
    target_dir   = Path(args.target_dir).resolve()
    pft_dir      = Path(args.pft_dir).resolve()
    years = list(range(args.year_start, args.year_end + 1))

    print(f"Features dir : {features_dir}")
    print(f"Target dir   : {target_dir}")
    print(f"PFT dir      : {pft_dir}")
    print(f"Years        : {args.year_start} → {args.year_end}")
    print(f"Output       : {args.output}")

    # Running stats accumulators
    dyn_accs = {feat: RunningStats() for feat in DYNAMIC_FEATURES}
    pft_accs = {feat: RunningStats() for feat in PFT_COLS}
    lai_acc  = RunningStats()

    n_ok_feat = n_ok_tgt = n_ok_pft = 0
    t0 = time.time()
    for y in years:
        feat_path = features_dir / FEATURES_FNAME.format(year=y)
        tgt_path  = target_dir   / TARGETS_FNAME.format(year=y)
        pft_path  = pft_dir      / PFT_FNAME.format(year=y)

        if feat_path.exists():
            _scan_features_year(feat_path, dyn_accs)
            n_ok_feat += 1
        else:
            print(f"  [warn] missing features {feat_path.name}")

        if tgt_path.exists():
            _scan_targets_year(tgt_path, lai_acc)
            n_ok_tgt += 1
        else:
            print(f"  [warn] missing targets {tgt_path.name}")

        if pft_path.exists():
            _scan_pft_year(pft_path, pft_accs)
            n_ok_pft += 1
        else:
            print(f"  [warn] missing PFT {pft_path.name}")

        if y % 5 == 0:
            print(f"  scanned through {y}  ({time.time() - t0:.1f}s)")

    print(f"\nScanned: features={n_ok_feat}/{len(years)}, "
          f"targets={n_ok_tgt}/{len(years)}, "
          f"pft={n_ok_pft}/{len(years)}  ({time.time() - t0:.1f}s)")

    # ── Assemble stats dict ──
    stats: Dict[str, Dict[str, float]] = {}
    for feat in DYNAMIC_FEATURES:
        stats[feat] = dyn_accs[feat].mean_std()
    for feat in PFT_COLS:
        stats[feat] = pft_accs[feat].mean_std()
    stats["LAI"] = lai_acc.mean_std()

    if args.co2_path:
        if not os.path.exists(args.co2_path):
            raise FileNotFoundError(args.co2_path)
        lut = load_co2_lut(args.co2_path)
        if CO2_FEATURES:
            stats["co2"] = _co2_stats(lut, years)

    # Pretty print one line per feature
    print("\nResulting stats:")
    for feat, s in stats.items():
        print(f"  {feat:20s}  mean={s['mean']:+.4f}  std={s['std']:.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".",
                exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n✓ saved to {args.output}")


if __name__ == "__main__":
    main()
