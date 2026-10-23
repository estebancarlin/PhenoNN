#!/usr/bin/env python3
"""
inspect_era5_pixelset.py
========================

Quick health-check of the ERA5 pixelset files feeding pft_diagnostics.py.

Reports, for the year range (optionally restricted to a selected_pixels subset):

  * the `time` dimension length of each year — this exposes the 365-vs-366
    mismatch. `pft_diagnostics.py` aligns both onto a fixed 366-day grid, so
    mixed lengths are expected but worth checking.
  * per year : number of cells, total values, NaN count and NaN %.
  * per feature (pooled over all years) : total values, NaN count and NaN %.

Read-only, torch-free, scipy-free.

Usage
-----
    python analysis/diagnostics/inspect_era5_pixelset.py \\
        --era5_dir "$DATA/era5_pixelset" \\
        --year_start 2000 --year_end 2001 \\
        --selected_pixels "$DATA/selected_pixels005_1.nc" \\
        --parent_map "$DATA/selected_pixels01_1.nc" \\
        [--features Tmin,tp_sum] \\
        [--output_csv runs/era5_missing.csv]
"""

import argparse
import os
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr

from phenonn.data.lai_dataset import load_parent01_map
from phenonn.utils.config import METEO_BASE, FEATURES_FNAME


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--era5_dir", required=True, help="Folder of ERA5_daily_pixelset_{Y}.nc."
    )
    p.add_argument("--year_start", type=int, required=True)
    p.add_argument("--year_end", type=int, required=True)
    p.add_argument(
        "--selected_pixels",
        default="",
        help="Optional selected_pixels*.nc to restrict the count to "
        "that site subset (default: all sites in each file).",
    )
    p.add_argument(
        "--parent_map",
        default="",
        help="selected_pixels01_1.nc mapping 0.05° site_ids to their 0.1° "
        "ERA5 parent cells. Required when --selected_pixels holds 0.05° "
        "(pix_...) ids but the ERA5 cubes are 0.1° (E... ids), as in "
        "scripts/smoke_train.sh.",
    )
    p.add_argument(
        "--features",
        default="",
        help="Comma-separated feature names (default: METEO_BASE + "
        "any SMI present).",
    )
    p.add_argument(
        "--output_csv",
        default="",
        help="Optional path to write the per-(year, feature) table.",
    )
    return p.parse_args()


def _wanted_sites(path: str) -> Optional[np.ndarray]:
    if not path:
        return None
    with xr.open_dataset(path) as sel:
        return np.asarray(sel["site_id"].values).astype(str)


def main():
    args = parse_args()
    era5_dir = Path(args.era5_dir).resolve()
    wanted = _wanted_sites(args.selected_pixels)
    # ERA5 cubes may live on the deduplicated 0.1° grid while the pool holds
    # 0.05° ids: translate through the parent map before matching (same join
    # as training, see scripts/smoke_train.sh).
    feat_wanted = wanted
    if args.parent_map:
        if wanted is None:
            print("[warn] --parent_map given without --selected_pixels; ignoring.")
        else:
            pm = load_parent01_map(args.parent_map)
            feat_wanted = np.array(sorted({pm[s] for s in wanted if s in pm}))
            n_drop = len(wanted) - sum(1 for s in wanted if s in pm)
            print(
                f"Parent map  : {args.parent_map} ({len(pm):,} links; "
                f"{feat_wanted.size:,} ERA5 cells for {len(wanted):,} sites"
                + (f", {n_drop} unmatched" if n_drop else "")
                + ")"
            )

    if args.features:
        feats = [s.strip() for s in args.features.split(",") if s.strip()]
    else:
        feats = list(METEO_BASE)  # extended with SMI below if present

    years = range(args.year_start, args.year_end + 1)

    time_lengths = Counter()
    per_year = []  # (year, n_time, n_sel, total, nan)
    rows = []  # (year, feature, total, nan) for the CSV
    feat_total = Counter()
    feat_nan = Counter()
    seen_extra = set()

    for year in years:
        fpath = era5_dir / FEATURES_FNAME.format(year=year)
        if not fpath.exists():
            print(f"  ✗ {year} missing {fpath.name}")
            continue
        with xr.open_dataset(fpath, engine="netcdf4", decode_times=False) as ds:
            n_time = int(ds.sizes["time"])
            file_ids = np.asarray(ds["site_id"].values).astype(str)
            if feat_wanted is None:
                mask = np.ones(file_ids.shape, dtype=bool)
            else:
                mask = np.isin(file_ids, feat_wanted)
            n_sel = int(mask.sum())

            # auto-add SMI (or any requested extra) if present and not listed
            file_feats = [f for f in feats if f in ds]
            if not args.features:
                for extra in ("SMI",):
                    if extra in ds and extra not in file_feats:
                        file_feats.append(extra)
                        seen_extra.add(extra)

            y_total = 0
            y_nan = 0
            for f in file_feats:
                arr = ds[f].transpose("time", "site").values[:, mask].astype(np.float32)
                tot = arr.size
                nan = int((~np.isfinite(arr)).sum())
                rows.append((year, f, tot, nan))
                feat_total[f] += tot
                feat_nan[f] += nan
                y_total += tot
                y_nan += nan

        time_lengths[n_time] += 1
        per_year.append((year, n_time, n_sel, y_total, y_nan))
        pct = 100.0 * y_nan / y_total if y_total else 0.0
        print(
            f"  ✓ {year}: time={n_time}  cells={n_sel:,}  "
            f"NaN={y_nan:,}/{y_total:,} ({pct:.3f}%)"
        )

    if seen_extra:
        feats = feats + sorted(seen_extra)

    # ── Summary ──
    print("\n=== time-dimension lengths ===")
    for length, n in sorted(time_lengths.items()):
        print(f"  {length} days : {n} year(s)")
    if len(time_lengths) > 1:
        modal = time_lengths.most_common(1)[0][0]
        odd = [y for (y, nt, *_) in per_year if nt != modal]
        print(
            f"  → mixed lengths! modal={modal}; pft_diagnostics aligns "
            f"365/366-day years on a fixed 366-day grid. "
            f"Off-length years: {odd}"
        )

    print("\n=== NaN per feature (all years pooled) ===")
    for f in feats:
        tot = feat_total.get(f, 0)
        if not tot:
            print(f"  {f:10s} : absent")
            continue
        nan = feat_nan.get(f, 0)
        print(f"  {f:10s} : {nan:,}/{tot:,} ({100.0 * nan / tot:.3f}%)")

    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        import csv

        with open(args.output_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["year", "feature", "total", "nan", "nan_pct"])
            for year, f, tot, nan in rows:
                w.writerow(
                    [year, f, tot, nan, f"{100.0 * nan / tot:.4f}" if tot else ""]
                )
        print(f"\n  → per-(year, feature) table: {args.output_csv}")


if __name__ == "__main__":
    main()
