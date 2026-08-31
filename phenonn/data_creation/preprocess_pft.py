#!/usr/bin/env python3
"""
preprocess_pft.py
=================

Regrid the native 0.05° ORCHIDEE/ESACCI PFT map onto the ERA5-Land 0.1° grid,
CO-REGISTERED exactly like preprocess_targets does for LAI — so LAI, PFT and
ERA5 all share the same `.x0` pixel centres (and one shared lat_idx/lon_idx
downstream).

Source (per year)
-----------------
    PFTmap_{YYYY}.nc : maxvegetfrac(time_counter=1, veget=15, lat=3600, lon=7200)
    - 0.05° CELL-registered grid (centres on 0.025 odd multiples), _FillValue
      1e20 over ocean, PFT1 = bare soil (veget index 0).
    - lat/lon are EXPLICIT coordinate vectors (unlike the THEIA LAI .h5), so we
      read them and derive the co-registration shift from the coordinate VALUES,
      not array parity → correct whatever the file's lat orientation or lon
      convention (0..360 is rolled to -180..180 to match the LAI grid).

Method (identical to preprocess_targets)
----------------------------------------
Drop ONE 0.05° row/col so each 2×2 block straddles an ERA5 node (mean of two
0.05° centres = a 0.1° multiple), then NaN-aware 2×2 area-mean of the 15
fractions, then snap onto the canonical ERA5-Land 0.1° grid
(grid_utils.era5_land_grid — built analytically, no reference file). Output kept
in -180..180 like the LAI target.

Output
------
    {output_dir}/PFTmap_{YYYY}.nc
        maxvegetfrac(veget=15, latitude=1801, longitude=3600)  [-180..180, .x0]
        NaN over ocean / poles / dateline edge.

Usage
-----
    python -m phenonn.data_creation.preprocess_pft \\
        --pft05_dir  /data/sbarbu/PhenoNN/data/PFTmap_1991_2019_0.05 \\
        --output_dir /data/sbarbu/PhenoNN/data/PFT_grid \\
        --year_start 1992 --year_end 2019
"""

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import xarray as xr

from phenonn.utils.config import N_PFT, PFT_FNAME
from phenonn.data_creation.grid_utils import (
    slice_to_bbox,
    era5_land_grid,
    coarsen_2x2_nanmean,
    reindex_to_reference,
    make_encoding,
)
from phenonn.data_creation._ncwrite import to_netcdf_subprocess


PFT_SRC_VAR = "maxvegetfrac"


def _coreg_shift(coord: np.ndarray, res_target: float = 0.1) -> int:
    """
    Number of leading cells (0 or 1) to drop so a 2×2 mean lands on the ERA5
    target nodes (`.x0`). Picks the parity whose first pair-mean is closest to a
    `res_target` multiple. Coordinate-value based → agnostic to lat orientation,
    lon convention and any crop.
    """

    def off(x: float) -> float:
        r = x / res_target
        return abs(r - round(r))

    m0 = 0.5 * (float(coord[0]) + float(coord[1]))
    m1 = 0.5 * (float(coord[1]) + float(coord[2]))
    return 0 if off(m0) <= off(m1) else 1


def _load_pft05(path: Path) -> xr.DataArray:
    """
    Return maxvegetfrac as (veget, lat, lon) with ocean = NaN, coords named
    lat/lon, longitude normalised to [-180, 180) (sorted) to match the LAI grid.
    Read with h5netcdf (the ORCHIDEE PFTmap only opens on the h5py stack) and
    decode_times=False (the 'years since' time_counter would otherwise choke).
    """
    ds = xr.open_dataset(path, engine="h5netcdf", decode_times=False)
    da = ds[PFT_SRC_VAR]
    if "time_counter" in da.dims:
        da = da.isel(time_counter=0, drop=True)
    ren = {}
    if "latitude" in da.dims:
        ren["latitude"] = "lat"
    if "longitude" in da.dims:
        ren["longitude"] = "lon"
    if ren:
        da = da.rename(ren)
    da = da.load()  # ~3 GB (15×3600×7200 f32) — fits 32 GB
    if float(da["lon"].max()) > 180.0:  # 0..360 → -180..180 to match LAI grid
        da = da.assign_coords(lon=(((da["lon"] + 180.0) % 360.0) - 180.0))
        da = da.sortby("lon")
    ds.close()
    return da


def _process_year(
    year: int,
    pft05_dir: Path,
    output_dir: Path,
    bbox: Optional[tuple],
    min_valid_per_cell: int,
    lat_chunk: int,
    lon_chunk: int,
    complevel: int,
) -> None:
    src = pft05_dir / PFT_FNAME.format(year=year)
    if not src.exists():
        print(f"  ✗ {year} skipped — missing {src.name}")
        return

    da = _load_pft05(src)  # (veget, lat, lon), -180..180

    # Target grid (canonical ERA5-Land 0.1°, -180..180), optionally cropped.
    ref_lat, ref_lon = era5_land_grid()
    if bbox is not None:
        da = slice_to_bbox(da, *bbox, lat_name="lat", lon_name="lon")
        ref = xr.DataArray(
            np.zeros((ref_lat.size, ref_lon.size), dtype="float32"),
            dims=("latitude", "longitude"),
            coords={"latitude": ref_lat, "longitude": ref_lon},
        )
        ref = slice_to_bbox(ref, *bbox)
        ref_lat = ref["latitude"].values
        ref_lon = ref["longitude"].values

    # One-cell co-registration shift, from the source coordinate values.
    row_shift = _coreg_shift(da["lat"].values)
    col_shift = _coreg_shift(da["lon"].values)
    da = da.isel(lat=slice(row_shift, None), lon=slice(col_shift, None))
    print(f"  {year}: co-registration shift drop {row_shift} row / {col_shift} col")

    # 2×2 area-mean of the fractions, then snap onto the ERA5 target grid.
    da = coarsen_2x2_nanmean(
        da, lat_name="lat", lon_name="lon", min_valid=min_valid_per_cell
    )
    da = reindex_to_reference(da, ref_lat, ref_lon, "lat", "lon")
    da.name = PFT_SRC_VAR

    ds = da.astype("float32").to_dataset()
    ds[PFT_SRC_VAR].attrs.update(
        {
            "long_name": "PFT fractions (ORCHIDEE/ESACCI, dekad-invariant)",
            "units": "-",
            "source": "0.05° ORCHIDEE PFT map downsampled 2×2 (ERA5-node "
            "co-registered) to ERA5-Land 0.1°",
        }
    )

    encoding = make_encoding(
        ds,
        chunks={"veget": N_PFT, "latitude": lat_chunk, "longitude": lon_chunk},
        complevel=complevel,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / PFT_FNAME.format(year=year)
    print(f"  writing {out_path.name} …")
    # h5netcdf (h5py) read above → isolate the netcdf4 write in a subprocess to
    # dodge the dual-libhdf5 conflict. See phenon/data/_ncwrite.py.
    to_netcdf_subprocess(ds, out_path, encoding=encoding, engine="netcdf4")
    print(
        f"  ✓ {out_path.name}  "
        f"({ds.sizes['latitude']} × {ds.sizes['longitude']}, "
        f"veget={ds.sizes['veget']})"
    )


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--pft05_dir",
        required=True,
        help="Folder with the 0.05° PFTmap_{YYYY}.nc files.",
    )
    p.add_argument(
        "--output_dir",
        required=True,
        help="Where to write the ERA5-grid PFTmap_{YYYY}.nc.",
    )
    p.add_argument("--year_start", type=int, default=1981)
    p.add_argument("--year_end", type=int, default=2025)
    p.add_argument("--lat_min", type=float, default=None)
    p.add_argument("--lat_max", type=float, default=None)
    p.add_argument("--lon_min", type=float, default=None)
    p.add_argument("--lon_max", type=float, default=None)
    p.add_argument(
        "--min_valid_per_cell",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="Min finite 0.05° pixels (out of 4) to keep a cell "
        "(default 1: fractions are static land cover).",
    )
    p.add_argument("--lat_chunk", type=int, default=32)
    p.add_argument("--lon_chunk", type=int, default=32)
    p.add_argument("--complevel", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    pft05_dir = Path(args.pft05_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    bbox = None
    bbox_args = (args.lat_min, args.lat_max, args.lon_min, args.lon_max)
    if any(v is not None for v in bbox_args):
        if not all(v is not None for v in bbox_args):
            raise ValueError("Pass ALL four lat/lon bounds or NONE.")
        bbox = bbox_args

    print(f"PFT 0.05° dir : {pft05_dir}")
    print(f"Output dir    : {output_dir}")
    print("Target grid   : ERA5-Land 0.1° (-180..180, .x0)")
    print(f"Years         : {args.year_start} → {args.year_end}")
    print(f"Bbox          : {bbox if bbox else 'OFF (full grid)'}")

    skipped: List[int] = []
    for year in range(args.year_start, args.year_end + 1):
        try:
            _process_year(
                year,
                pft05_dir,
                output_dir,
                bbox,
                args.min_valid_per_cell,
                args.lat_chunk,
                args.lon_chunk,
                args.complevel,
            )
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ {year} failed — {type(e).__name__}: {e}")
            skipped.append(year)

    print("\nDone.")
    if skipped:
        print(f"Skipped years: {skipped}")


if __name__ == "__main__":
    main()
