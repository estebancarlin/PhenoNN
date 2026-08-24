#!/usr/bin/env python3
"""
preprocess_features.py
======================

Take the daily ERA5-Land *pixelset* cubes produced by
build_daily_dataset_pixelset.py, attach derived ecological features
(GDD, CDD, NCD, Botta), and rechunk on the `site` dim for training-time
random access:

    (time = -1, site = site_chunk)

`add_derived_features` operates only along the `time` axis, so it is
dim-agnostic and works directly on the (time, site) pixelset.

Why this rechunking matters
---------------------------
The training Dataset pulls, for each (site, year) sample, a ~720-day window
at one site. With time=-1 (the full year in one chunk per site tile), a
window fits in 1-2 chunks → a single contiguous read instead of stitching
365 daily chunks.

Inputs
------
{input_dir}/ERA5_daily_pixelset_{YYYY}.nc   (one per year, dims (time, site))

Outputs
-------
{output_dir}/ERA5_features_{YYYY}.nc        (one per year: base + 7 derived vars)

Optional step: only needed when training with the derived pheno features
(config add_pheno_features=True). Otherwise training reads the pixelset files
directly and this step is skipped.

Usage
-----
    python -m phenon.dataset_creation.preprocess_features \\
        --input_dir  /data/sbarbu/PhenoNN/data/era5_10% \\
        --output_dir /data/sbarbu/PhenoNN/data/era5_features \\
        --year_start 1992 --year_end 2019
"""

import argparse
import time as time_mod
from pathlib import Path
from typing import List

import xarray as xr
from dask.diagnostics import ProgressBar

from phenon.dataset_creation.feature_engineering import add_derived_features
from phenon.dataset_creation.grid_utils import make_encoding


def _process_year(
    year: int,
    input_dir: Path,
    output_dir: Path,
    site_chunk: int,
    complevel: int,
) -> None:
    """
    Add derived ecological features (GDD/CDD/NCD/Botta) to the daily
    (time, site) pixelset cube produced by build_daily_dataset_pixelset.py
    and rewrite a training-friendly file ERA5_features_{Y}.nc.

    The new input has dim (time, site); add_derived_features operates
    along the `time` axis only so the function is dim-agnostic.
    """
    in_path = input_dir / f"ERA5_daily_pixelset_{year}.nc"
    if not in_path.exists():
        print(f"  ✗ {year} skipped — {in_path} missing")
        return

    t0 = time_mod.time()
    print(f"\n── {year} ─────────────────────────────")
    ds = xr.open_dataset(in_path, chunks={"time": -1})

    ds = add_derived_features(ds)
    print(f"  added derived features (lazy)  +{time_mod.time() - t0:5.1f}s")

    # Training-friendly chunking on the site dim.
    n_site = ds.sizes["site"]
    site_chunk = min(site_chunk, n_site)
    ds = ds.chunk({"time": -1, "site": site_chunk})

    encoding = make_encoding(
        ds,
        chunks={"time": ds.sizes["time"], "site": site_chunk},
        complevel=complevel,
    )

    out_path = output_dir / f"ERA5_features_{year}.nc"
    print(f"  writing {out_path.name} …")
    with ProgressBar(dt=5.0):
        ds.to_netcdf(out_path, engine="netcdf4", encoding=encoding)
    ds.close()
    print(f"  ✓ done in {time_mod.time() - t0:6.1f}s  →  {out_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input_dir",  required=True,
                   help="Folder with ERA5_daily_pixelset_{Y}.nc files")
    p.add_argument("--output_dir", required=True,
                   help="Where to write ERA5_features_{Y}.nc")
    p.add_argument("--year_start", type=int, default=1981)
    p.add_argument("--year_end",   type=int, default=2025)
    p.add_argument("--site_chunk", type=int, default=4096,
                   help="Site chunk size in the output (default 4096). "
                        "Match build_daily_dataset_pixelset.py for cache "
                        "alignment.")
    p.add_argument("--complevel",  type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    input_dir  = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input dir   : {input_dir}")
    print(f"Output dir  : {output_dir}")
    print(f"Years       : {args.year_start} → {args.year_end}")
    print(f"Site chunk  : {args.site_chunk}")
    print(f"Compression : zlib {args.complevel}")

    skipped: List[int] = []
    for year in range(args.year_start, args.year_end + 1):
        try:
            _process_year(
                year, input_dir, output_dir,
                site_chunk=args.site_chunk,
                complevel=args.complevel,
            )
        except FileNotFoundError as e:
            print(f"  ✗ {year} skipped — {e}")
            skipped.append(year)
        except Exception as e:                       # noqa: BLE001
            print(f"  ✗ {year} failed — {type(e).__name__}: {e}")
            skipped.append(year)

    print("\nDone.")
    if skipped:
        print(f"Skipped years: {skipped}")


if __name__ == "__main__":
    main()
