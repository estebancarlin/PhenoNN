#!/usr/bin/env python3
"""
preprocess_targets.py
=====================

Convert the per-dekad THEIA GEOV2 AVHRR LAI .h5 archive (one file per dekad,
3600 × 7200 uint8 on a 0.05° grid) into one NetCDF per year aligned on a
reference 0.1° grid.

THEIA convention
----------------
  - Grid       : 3600 × 7200 at 0.05°, top-left at (90°N, 180°W)
  - Nodata     : raw_value ∈ {255, 256} → NaN
  - Scaling    : LAI = raw_value / 30
  - File name  : THEIA_GEOV2_R01_AVHRR_LAI_{YYYYMMDD}.h5
  - Dekadal    : 36 dates / year = days {5, 15, 25} of every month

Reference-grid alignment
------------------------
We do not hard-code the target coords; instead we read them from any
0.1° NetCDF passed via `--ref_nc`. Two practical sources:

  - ORCHIDEE PFTmap_{Y}.nc  (1800 × 3600, coords named lat/lon).
    Recommended in pipelines where ERA5-Land daily files do not yet
    exist at the time targets are built. The coarsen output is exactly
    1800 × 3600, so the reindex is a coordinate snap.

  - An ERA5-Land daily NetCDF (1801 × 3600, latitude/longitude).
    Use only when an ERA5 file is already available.

Downsampling is 2×2 NaN-aware mean, then we reindex onto the reference
grid with method="nearest" tol=0.06° (auto-detects lat vs latitude in the
reference) so coordinate conventions match exactly.

Outputs
-------
{output_dir}/LAI_dekadal_{YYYY}.nc
    dims : (dekad=36, latitude, longitude)
    vars : LAI (float32, NaN where missing)
    chunks : (dekad=36, latitude=lat_chunk, longitude=lon_chunk)

Usage
-----
    # Aligned on the PFT grid (recommended — works before ERA5 daily exists)
    python -m phenon.dataset_creation.preprocess_targets \\
        --h5_dir     /bdd/THEIA/AVHRR/LAI \\
        --ref_nc     /data/sbarbu/PhenoNN/data/PFT_map0.1/PFTmap_1992.nc \\
        --output_dir /data/sbarbu/targets \\
        --year_start 1981 --year_end 2025
"""

import argparse
import glob
import os
import re
import time as time_mod
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar

from phenon.dataset_creation.grid_utils import (
    read_reference_grid, slice_to_bbox,
    coarsen_2x2_nanmean, reindex_to_reference, make_encoding,
)
from phenon.dataset_creation._ncwrite import to_netcdf_subprocess


# ── THEIA grid (immutable) ───────────────────────────────────────────────────

THEIA_RES   = 0.05
THEIA_NROWS = 3600
THEIA_NCOLS = 7200
THEIA_NODATA = {255, 256}
THEIA_SCALE = 30.0

DEKAD_DAYS = (5, 15, 25)
N_DEKAD_YEAR = 36


def _theia_lat_lon():
    """Pixel-center lat / lon vectors of the native THEIA GEOV2 0.05° grid."""
    lat = 90.0 - (np.arange(THEIA_NROWS) + 0.5) * THEIA_RES   # 89.975 → -89.975
    lon = -180.0 + (np.arange(THEIA_NCOLS) + 0.5) * THEIA_RES  # -179.975 → 179.975
    return lat, lon


# ── H5 discovery ─────────────────────────────────────────────────────────────


def scan_h5(h5_dir: Path) -> Dict[str, str]:
    """Return {YYYYMMDD: path} for every THEIA H5 found under `h5_dir`."""
    rx = re.compile(r"THEIA_GEOV2_R01_AVHRR_LAI_(\d{8})\.h5$")
    out: Dict[str, str] = {}
    for fp in sorted(glob.glob(str(h5_dir / "**" / "THEIA_GEOV2_R01_AVHRR_LAI_*.h5"),
                                recursive=True)):
        m = rx.search(os.path.basename(fp))
        if m:
            out[m.group(1)] = fp
    return out


def expected_dates(year: int) -> List[str]:
    """List of YYYYMMDD strings for the 36 dekads of `year`."""
    return [f"{year:04d}{m:02d}{d:02d}"
            for m in range(1, 13) for d in DEKAD_DAYS]


# ── Per-dekad load + decode ──────────────────────────────────────────────────


def load_decoded_dekad(
    h5_path: Optional[str],
    bbox_rows: Optional[tuple] = None,
    bbox_cols: Optional[tuple] = None,
) -> np.ndarray:
    """
    Read one H5 file, decode uint8 → float32 LAI (NaN where nodata),
    optionally cropped on the native grid.

    Returns an (n_rows, n_cols) float32 array (the bbox-crop shape, or the
    full 3600 × 7200 if bbox is None).
    """
    if h5_path is None or not os.path.exists(h5_path):
        # Missing file → array full of NaN, same shape as the crop
        nr = (bbox_rows[1] - bbox_rows[0] + 1) if bbox_rows else THEIA_NROWS
        nc = (bbox_cols[1] - bbox_cols[0] + 1) if bbox_cols else THEIA_NCOLS
        return np.full((nr, nc), np.nan, dtype=np.float32)

    with h5py.File(h5_path, "r") as h5:
        ds = h5["LAI"]
        if bbox_rows and bbox_cols:
            r0, r1 = bbox_rows
            c0, c1 = bbox_cols
            raw = ds[r0:r1 + 1, c0:c1 + 1].astype(np.int32)
        else:
            raw = ds[:].astype(np.int32)

    lai = raw.astype(np.float32) / THEIA_SCALE
    nodata_mask = np.isin(raw, list(THEIA_NODATA))
    lai[nodata_mask] = np.nan
    return lai


def _bbox_to_theia_indices(
    bbox: Optional[tuple],
) -> tuple:
    """
    Translate a (lat_min, lat_max, lon_min, lon_max) bbox to inclusive THEIA
    row / col ranges (0-indexed). Returns (None, None) if bbox is None.
    """
    if bbox is None:
        return None, None
    lat_min, lat_max, lon_min, lon_max = bbox

    def lat_to_row(lat): return int(np.floor((90.0 - lat) / THEIA_RES))
    def lon_to_col(lon):
        lon_norm = lon if lon <= 180 else lon - 360
        return int(np.floor((lon_norm + 180.0) / THEIA_RES))

    r0 = max(0, lat_to_row(lat_max))                  # north → small row
    r1 = min(THEIA_NROWS - 1, lat_to_row(lat_min))
    c0 = max(0, lon_to_col(lon_min))
    c1 = min(THEIA_NCOLS - 1, lon_to_col(lon_max))
    # Add a 1-cell margin so the 2×2 coarsen has full neighbourhoods at edges
    r0 = max(0, r0 - 1)
    r1 = min(THEIA_NROWS - 1, r1 + 1)
    c0 = max(0, c0 - 1)
    c1 = min(THEIA_NCOLS - 1, c1 + 1)
    return (r0, r1), (c0, c1)


# ── Per-year orchestration ───────────────────────────────────────────────────


def process_year(
    year: int,
    h5_dir: Path,
    output_dir: Path,
    ref_lat: np.ndarray, ref_lon: np.ndarray,
    h5_index: Dict[str, str],
    bbox: Optional[tuple],
    lat_chunk: int, lon_chunk: int,
    complevel: int,
    min_valid_per_cell: int = 2,
) -> None:
    print(f"\n── {year} ─────────────────────────────")
    t0 = time_mod.time()

    # Map full-grid bbox → THEIA row/col indices for fast slab reads
    bbox_rows, bbox_cols = _bbox_to_theia_indices(bbox)
    theia_lat, theia_lon = _theia_lat_lon()
    if bbox_rows and bbox_cols:
        theia_lat = theia_lat[bbox_rows[0]: bbox_rows[1] + 1]
        theia_lon = theia_lon[bbox_cols[0]: bbox_cols[1] + 1]

    # Read every dekad in chronological order
    n_missing = 0
    cubes = []
    for ymd in expected_dates(year):
        h5_path = h5_index.get(ymd)
        if h5_path is None:
            n_missing += 1
        cubes.append(load_decoded_dekad(h5_path, bbox_rows, bbox_cols))
    print(f"  loaded {N_DEKAD_YEAR} dekads "
          f"(missing: {n_missing})  +{time_mod.time() - t0:5.1f}s")

    cube = np.stack(cubes, axis=0)                        # (36, n_lat, n_lon)
    da = xr.DataArray(
        cube,
        dims=("dekad", "lat", "lon"),
        coords={
            "dekad": np.arange(N_DEKAD_YEAR, dtype=np.int16),
            "lat":   theia_lat,
            "lon":   theia_lon,
            "month": ("dekad", np.repeat(np.arange(1, 13), 3).astype(np.int8)),
            "day":   ("dekad", np.tile([5, 15, 25], 12).astype(np.int8)),
            "date":  ("dekad", [date(year, m, d).isoformat()
                                for m in range(1, 13) for d in DEKAD_DAYS]),
        },
        name="LAI",
    )

    # 2×2 mean → 0.1° intermediate grid (NaN-aware with min_valid threshold)
    da = coarsen_2x2_nanmean(
        da, lat_name="lat", lon_name="lon",
        min_valid=min_valid_per_cell,
    )
    print(f"  coarsened to 0.1°  "
          f"(min_valid={min_valid_per_cell}/4)  "
          f"+{time_mod.time() - t0:5.1f}s  "
          f"({da.sizes['lat']} × {da.sizes['lon']})")

    # Reindex onto the exact reference (latitude, longitude) coords
    da = reindex_to_reference(da, ref_lat, ref_lon, "lat", "lon")
    print(f"  reindexed to reference grid  +{time_mod.time() - t0:5.1f}s  "
          f"({da.sizes['latitude']} × {da.sizes['longitude']})")

    ds = da.astype("float32").to_dataset()
    ds["LAI"].attrs.update({
        "long_name": "Leaf Area Index (THEIA GEOV2 AVHRR, dekadal mean)",
        "units":     "m2 m-2",
        "source":    "THEIA GEOV2 AVHRR 0.05° downsampled 2×2 to ERA5-Land 0.1°",
    })

    encoding = make_encoding(
        ds,
        chunks={"dekad": N_DEKAD_YEAR,
                "latitude":  lat_chunk,
                "longitude": lon_chunk},
        complevel=complevel,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"LAI_dekadal_{year}.nc"
    print(f"  writing {out_path.name} …")
    # Write from a fresh, h5py-free subprocess: this process has already
    # imported h5py (THEIA reads) / h5netcdf (PFTmap ref grid), whose bundled
    # libhdf5 conflicts with netCDF4's and would crash a same-process write
    # with "NetCDF: HDF error". See phenon/data/_ncwrite.py.
    to_netcdf_subprocess(ds, out_path, encoding=encoding, engine="netcdf4")
    print(f"  ✓ done in {time_mod.time() - t0:6.1f}s  →  {out_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--h5_dir",     required=True,
                   help="Folder containing THEIA_GEOV2_R01_AVHRR_LAI_*.h5 "
                        "(scanned recursively).")
    p.add_argument("--ref_nc",     required=True,
                   help="Any 0.1° NetCDF whose lat/lon (or latitude/longitude) "
                        "coords define the target grid. Typical choices: "
                        "ORCHIDEE PFTmap_{Y}.nc (1800×3600) or an ERA5-Land "
                        "daily file (1801×3600).")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--year_start", type=int, default=1981)
    p.add_argument("--year_end",   type=int, default=2025)
    p.add_argument("--lat_min", type=float, default=None)
    p.add_argument("--lat_max", type=float, default=None)
    p.add_argument("--lon_min", type=float, default=None)
    p.add_argument("--lon_max", type=float, default=None)
    p.add_argument("--lat_chunk", type=int, default=32)
    p.add_argument("--lon_chunk", type=int, default=32)
    p.add_argument("--complevel", type=int, default=4)
    p.add_argument("--min_valid_per_cell", type=int, default=2,
                   choices=[1, 2, 3, 4],
                   help="Minimum number of finite AVHRR pixels (out of 4) "
                        "required to keep a downsampled ERA5-grid cell. "
                        "Default 2 (drop cells with only 1 valid pixel).")
    return p.parse_args()


def main():
    args = parse_args()
    h5_dir     = Path(args.h5_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    bbox = None
    bbox_args = (args.lat_min, args.lat_max, args.lon_min, args.lon_max)
    if any(v is not None for v in bbox_args):
        if not all(v is not None for v in bbox_args):
            raise ValueError("Pass ALL four lat/lon bounds or NONE.")
        bbox = bbox_args

    ref_lat, ref_lon = read_reference_grid(args.ref_nc)
    # Optional crop of the reference grid before writing
    if bbox is not None:
        ref_da = xr.DataArray(
            np.zeros((len(ref_lat), len(ref_lon)), dtype="float32"),
            dims=("latitude", "longitude"),
            coords={"latitude": ref_lat, "longitude": ref_lon},
        )
        ref_da = slice_to_bbox(ref_da, *bbox)
        ref_lat = ref_da["latitude"].values
        ref_lon = ref_da["longitude"].values

    print(f"H5 dir      : {h5_dir}")
    print(f"Ref NC      : {args.ref_nc}")
    print(f"Output dir  : {output_dir}")
    print(f"Years       : {args.year_start} → {args.year_end}")
    print(f"Bbox        : {bbox if bbox else 'OFF (full grid)'}")
    print(f"Ref grid    : {len(ref_lat)} × {len(ref_lon)}")
    print(f"Spatial chunk: {args.lat_chunk} × {args.lon_chunk}")

    print("Indexing H5 archive …")
    h5_index = scan_h5(h5_dir)
    print(f"  {len(h5_index):,} H5 files indexed")

    skipped: List[int] = []
    for year in range(args.year_start, args.year_end + 1):
        try:
            process_year(
                year, h5_dir, output_dir,
                ref_lat, ref_lon, h5_index, bbox,
                args.lat_chunk, args.lon_chunk, args.complevel,
                min_valid_per_cell=args.min_valid_per_cell,
            )
        except Exception as e:                       # noqa: BLE001
            print(f"  ✗ {year} failed — {type(e).__name__}: {e}")
            skipped.append(year)

    print("\nDone.")
    if skipped:
        print(f"Skipped years: {skipped}")


if __name__ == "__main__":
    main()
