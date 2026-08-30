#!/usr/bin/env python3
"""
preprocess_targets.py
=====================

Convert the per-dekad THEIA GEOV2 AVHRR LAI .h5 archive (one file per dekad,
3600 × 7200 uint8 on a 0.05° grid) into one NetCDF per year CO-REGISTERED with
the ERA5-Land 0.1° grid.

THEIA convention
----------------
  - Grid       : 3600 × 7200 at 0.05°, top-left at (90°N, 180°W)
  - Registration : CELL-registered → pixel centres on 0.025° odd multiples
                   (89.975, 89.925, … / -179.975, …). NO centre lands on a
                   0.1° multiple.
  - Nodata     : raw_value ∈ {255, 256} → NaN
  - Scaling    : LAI = raw_value / 30
  - File name  : THEIA_GEOV2_R01_AVHRR_LAI_{YYYYMMDD}.h5
  - Dekadal    : 36 dates / year = days {5, 15, 25} of every month

ERA5-Land co-registration (the whole point of this rewrite)
-----------------------------------------------------------
ERA5-Land is NODE-registered: its pixel centres sit on the 0.1° multiples
(…89.9, 89.8… / lon .0). A plain 2×2 mean of THEIA on THEIA's own boundaries
produces 0.1° centres at .x5 (89.95…) — half a pixel (0.05°) NW of every ERA5
node. That half-pixel is exactly the mismatch measured between the ERA5 and
LAI/PFT pixelsets.

Fix: drop ONE 0.05° row and column first, so each 2×2 block straddles an ERA5
node — mean(89.925, 89.875) = 89.90 (an ERA5 latitude), not 89.95. The output
therefore lands EXACTLY on the ERA5-Land 0.1° grid. The shift is computed on the
GLOBAL THEIA index so it stays correct with or without a bbox crop.

Longitude is kept in -180..180 (THEIA's own convention, 3600 pts, .x0). This is
the *same* physical grid as ERA5-Land's 0..360, up to the exact 180° index roll
that build_daily_dataset_pixelset.py already applies ((lon_idx + N_LON//2) %
N_LON) — that roll becomes exact once the half-pixel is gone, so nothing
downstream needs changing beyond regenerating selected_pixels.nc against this
new grid.

Target grid is the canonical ERA5-Land 0.1° grid built analytically from its
known indexing (grid_utils.era5_land_grid — no reference file needed).
Downsampling is 2×2 NaN-aware mean, then a nearest-snap (tol=0.06°) onto the
target coords.

Outputs
-------
{output_dir}/LAI_dekadal_{YYYY}.nc
    dims : (dekad=36, latitude=1801, longitude=3600)  [-180..180, .x0]
    vars : LAI, LAI_RMSE (float32, NaN where missing / poles / dateline edge)
           LAI_RMSE is the THEIA LAI-RMSE band (retrieval uncertainty), same
           decode/regrid as LAI — consumed by compute_valid_pixels to reject
           low-quality dekads.
    chunks : (dekad=36, latitude=lat_chunk, longitude=lon_chunk)

Usage
-----
    python -m phenonn.data_creation.preprocess_targets \\
        --h5_dir     /bdd/THEIA/AVHRR/LAI \\
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

from phenonn.data_creation.grid_utils import (
    slice_to_bbox,
    era5_land_grid,
    coarsen_2x2_nanmean,
    reindex_to_reference,
    make_encoding,
)
from phenonn.data_creation._ncwrite import to_netcdf_subprocess


# ── THEIA grid (immutable) ───────────────────────────────────────────────────

THEIA_RES = 0.05
THEIA_NROWS = 3600
THEIA_NCOLS = 7200
THEIA_NODATA = {255, 256}
THEIA_SCALE = 30.0

DEKAD_DAYS = (5, 15, 25)
N_DEKAD_YEAR = 36


def _theia_lat_lon():
    """Pixel-center lat / lon vectors of the native THEIA GEOV2 0.05° grid."""
    lat = 90.0 - (np.arange(THEIA_NROWS) + 0.5) * THEIA_RES  # 89.975 → -89.975
    lon = -180.0 + (np.arange(THEIA_NCOLS) + 0.5) * THEIA_RES  # -179.975 → 179.975
    return lat, lon


# ── H5 discovery ─────────────────────────────────────────────────────────────


def scan_h5(h5_dir: Path) -> Dict[str, str]:
    """Return {YYYYMMDD: path} for every THEIA H5 found under `h5_dir`."""
    rx = re.compile(r"THEIA_GEOV2_R01_AVHRR_LAI_(\d{8})\.h5$")
    out: Dict[str, str] = {}
    for fp in sorted(
        glob.glob(str(h5_dir / "**" / "THEIA_GEOV2_R01_AVHRR_LAI_*.h5"), recursive=True)
    ):
        m = rx.search(os.path.basename(fp))
        if m:
            out[m.group(1)] = fp
    return out


def expected_dates(year: int) -> List[str]:
    """List of YYYYMMDD strings for the 36 dekads of `year`."""
    return [f"{year:04d}{m:02d}{d:02d}" for m in range(1, 13) for d in DEKAD_DAYS]


# ── Per-dekad load + decode ──────────────────────────────────────────────────


def load_decoded_dekad(
    h5_path: Optional[str],
    bbox_rows: Optional[tuple] = None,
    bbox_cols: Optional[tuple] = None,
    bands: tuple = ("LAI",),
) -> Dict[str, np.ndarray]:
    """
    Read one H5 file, decode uint8 → float32 (value = DN / 30, NaN where
    nodata) for each requested band, optionally cropped on the native grid.

    Returns {band: (n_rows, n_cols) float32} (the bbox-crop shape, or the full
    3600 × 7200 if bbox is None). A missing file — or a band absent from the
    file — yields an all-NaN array, so LAI still processes even when the RMSE
    band is unavailable for a given dekad. Both LAI and LAI-RMSE share
    SCALING_FACTOR=30 and the same nodata codes.
    """
    nr = (bbox_rows[1] - bbox_rows[0] + 1) if bbox_rows else THEIA_NROWS
    nc = (bbox_cols[1] - bbox_cols[0] + 1) if bbox_cols else THEIA_NCOLS
    if h5_path is None or not os.path.exists(h5_path):
        return {b: np.full((nr, nc), np.nan, dtype=np.float32) for b in bands}

    out: Dict[str, np.ndarray] = {}
    with h5py.File(h5_path, "r") as h5:
        for b in bands:
            if b not in h5:
                out[b] = np.full((nr, nc), np.nan, dtype=np.float32)
                continue
            ds = h5[b]
            if bbox_rows and bbox_cols:
                r0, r1 = bbox_rows
                c0, c1 = bbox_cols
                raw = ds[r0 : r1 + 1, c0 : c1 + 1].astype(np.int32)
            else:
                raw = ds[:].astype(np.int32)
            val = raw.astype(np.float32) / THEIA_SCALE
            val[np.isin(raw, list(THEIA_NODATA))] = np.nan
            out[b] = val
    return out


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

    def lat_to_row(lat):
        return int(np.floor((90.0 - lat) / THEIA_RES))

    def lon_to_col(lon):
        lon_norm = lon if lon <= 180 else lon - 360
        return int(np.floor((lon_norm + 180.0) / THEIA_RES))

    r0 = max(0, lat_to_row(lat_max))  # north → small row
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
    ref_lat: np.ndarray,
    ref_lon: np.ndarray,
    h5_index: Dict[str, str],
    bbox: Optional[tuple],
    lat_chunk: int,
    lon_chunk: int,
    complevel: int,
    min_valid_per_cell: int = 2,
) -> None:
    print(f"\n── {year} ─────────────────────────────")
    t0 = time_mod.time()

    # Map full-grid bbox → THEIA row/col indices for fast slab reads
    bbox_rows, bbox_cols = _bbox_to_theia_indices(bbox)
    theia_lat, theia_lon = _theia_lat_lon()
    if bbox_rows and bbox_cols:
        theia_lat = theia_lat[bbox_rows[0] : bbox_rows[1] + 1]
        theia_lon = theia_lon[bbox_cols[0] : bbox_cols[1] + 1]

    # Read every dekad in chronological order: LAI + its per-pixel RMSE band.
    n_missing = 0
    decoded = []
    for ymd in expected_dates(year):
        h5_path = h5_index.get(ymd)
        if h5_path is None:
            n_missing += 1
        decoded.append(
            load_decoded_dekad(h5_path, bbox_rows, bbox_cols, bands=("LAI", "LAI-RMSE"))
        )
    print(
        f"  loaded {N_DEKAD_YEAR} dekads "
        f"(missing: {n_missing})  +{time_mod.time() - t0:5.1f}s"
    )

    coords = {
        "dekad": np.arange(N_DEKAD_YEAR, dtype=np.int16),
        "lat": theia_lat,
        "lon": theia_lon,
        "month": ("dekad", np.repeat(np.arange(1, 13), 3).astype(np.int8)),
        "day": ("dekad", np.tile([5, 15, 25], 12).astype(np.int8)),
        "date": (
            "dekad",
            [date(year, m, d).isoformat() for m in range(1, 13) for d in DEKAD_DAYS],
        ),
    }

    # ── Half-pixel co-registration with ERA5-Land ──
    # THEIA centres are at .x25/.x75; a plain 2×2 mean on THEIA's own boundaries
    # lands the 0.1° centres at .x5, half a pixel off ERA5-Land's .x0 nodes.
    # Drop one 0.05° row/col first so each 2×2 block straddles an ERA5 node:
    #   mean(89.925, 89.875) = 89.90 (an ERA5 latitude), not 89.95.
    # A pair must START on an ODD global THEIA index → the shift is derived from
    # the slab's GLOBAL start row/col so it is correct with or without a bbox.
    r0_global = bbox_rows[0] if bbox_rows else 0
    c0_global = bbox_cols[0] if bbox_cols else 0
    row_shift = 0 if (r0_global % 2 == 1) else 1
    col_shift = 0 if (c0_global % 2 == 1) else 1
    print(f"  ERA5 co-registration shift: drop {row_shift} row / {col_shift} col")

    def _regrid(band_key: str, name: str) -> xr.DataArray:
        """Stack the 36 dekads of one band, then shift → 2×2 nanmean → snap
        onto the ERA5 target grid (identical treatment for LAI and RMSE)."""
        cube = np.stack([d[band_key] for d in decoded], axis=0)  # (36, nlat, nlon)
        da = xr.DataArray(cube, dims=("dekad", "lat", "lon"), coords=coords, name=name)
        da = da.isel(lat=slice(row_shift, None), lon=slice(col_shift, None))
        da = coarsen_2x2_nanmean(
            da, lat_name="lat", lon_name="lon", min_valid=min_valid_per_cell
        )
        return reindex_to_reference(da, ref_lat, ref_lon, "lat", "lon")

    lai = _regrid("LAI", "LAI").astype("float32")
    rmse = _regrid("LAI-RMSE", "LAI_RMSE").astype("float32")
    print(
        f"  coarsened + reindexed LAI & RMSE  "
        f"(min_valid={min_valid_per_cell}/4)  +{time_mod.time() - t0:5.1f}s  "
        f"({lai.sizes['latitude']} × {lai.sizes['longitude']})"
    )

    ds = lai.to_dataset()
    ds["LAI_RMSE"] = rmse
    ds["LAI"].attrs.update(
        {
            "long_name": "Leaf Area Index (THEIA GEOV2 AVHRR, dekadal mean)",
            "units": "m2 m-2",
            "source": "THEIA GEOV2 AVHRR 0.05° downsampled 2×2 (ERA5-node "
            "co-registered) to ERA5-Land 0.1°",
        }
    )
    ds["LAI_RMSE"].attrs.update(
        {
            "long_name": "LAI dekadal retrieval RMSE (THEIA GEOV2 AVHRR LAI-RMSE, "
            "2×2 mean)",
            "units": "m2 m-2",
            "source": "THEIA GEOV2 AVHRR LAI-RMSE 0.05° (DN/30), ERA5-node "
            "co-registered to ERA5-Land 0.1°",
        }
    )

    encoding = make_encoding(
        ds,
        chunks={"dekad": N_DEKAD_YEAR, "latitude": lat_chunk, "longitude": lon_chunk},
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
    p.add_argument(
        "--h5_dir",
        required=True,
        help="Folder containing THEIA_GEOV2_R01_AVHRR_LAI_*.h5 "
        "(scanned recursively).",
    )
    p.add_argument("--output_dir", required=True)
    p.add_argument("--year_start", type=int, default=1981)
    p.add_argument("--year_end", type=int, default=2025)
    p.add_argument("--lat_min", type=float, default=None)
    p.add_argument("--lat_max", type=float, default=None)
    p.add_argument("--lon_min", type=float, default=None)
    p.add_argument("--lon_max", type=float, default=None)
    p.add_argument("--lat_chunk", type=int, default=32)
    p.add_argument("--lon_chunk", type=int, default=32)
    p.add_argument("--complevel", type=int, default=4)
    p.add_argument(
        "--min_valid_per_cell",
        type=int,
        default=2,
        choices=[1, 2, 3, 4],
        help="Minimum number of finite AVHRR pixels (out of 4) "
        "required to keep a downsampled ERA5-grid cell. "
        "Default 2 (drop cells with only 1 valid pixel).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    h5_dir = Path(args.h5_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    bbox = None
    bbox_args = (args.lat_min, args.lat_max, args.lon_min, args.lon_max)
    if any(v is not None for v in bbox_args):
        if not all(v is not None for v in bbox_args):
            raise ValueError("Pass ALL four lat/lon bounds or NONE.")
        bbox = bbox_args

    ref_lat, ref_lon = era5_land_grid()
    # Optional crop of the target grid before writing
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
    print("Target grid : ERA5-Land 0.1° (-180..180, .x0)")
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
                year,
                h5_dir,
                output_dir,
                ref_lat,
                ref_lon,
                h5_index,
                bbox,
                args.lat_chunk,
                args.lon_chunk,
                args.complevel,
                min_valid_per_cell=args.min_valid_per_cell,
            )
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ {year} failed — {type(e).__name__}: {e}")
            skipped.append(year)

    print("\nDone.")
    if skipped:
        print(f"Skipped years: {skipped}")


if __name__ == "__main__":
    main()
