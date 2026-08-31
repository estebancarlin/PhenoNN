#!/usr/bin/env python3
"""
build_boxes_pixelset.py
=======================

Build a `selected_pixels.nc` from a set of coarse cells given by their centre
(lon, lat) — e.g. 2°×2° cells — by taking every 0.1° ERA5-Land pixel that falls
inside each box, then extract the per-year LAI and PFT *pixelset* cubes for
those pixels by reusing `build_pixelset_targets._process_year`.

A 2°×2° cell on the 0.1° grid = 20×20 = 400 pixels. Boxes use a HALF-OPEN
interval [c - s/2, c + s/2), so a pixel on a shared edge belongs to exactly one
cell (adjacent boxes never duplicate a pixel).

Grid alignment (important)
--------------------------
`lat_idx / lon_idx` are indices into the LAI/PFT GRID files that
`build_pixelset_targets` will index as `grid[:, lat_idx, lon_idx]`. The
reference lat/lon vectors are therefore read from one of those grid files
(default: {target_dir}/LAI_dekadal_{year_start}.nc) so the indices line up
exactly. Read with engine="netcdf4" to stay on the same libhdf5 as the
extraction (avoids the dual-libhdf5 conflict on the cluster venv).

Output
------
`selected_pixels.nc` — same schema as select_pixels.py:
    dims: site
    vars: lat_idx, lon_idx, latitude, longitude, site_id (pix_LLLL_OOOOO)
    plus: cell_lon, cell_lat  (the 2° centre each pixel belongs to → easy
          per-cell aggregation later; ignored by build_pixelset_targets)

Usage
-----
    python -m phenonn.data_creation.build_boxes_pixelset \\
        --output_selected /data/.../selected_pixels_boxes.nc \\
        --target_dir      /data/.../LAI_grid \\
        --pft_dir         /data/.../PFT_grid \\
        --out_target_dir  /data/.../LAI_pixelset \\
        --out_pft_dir     /data/.../PFT_pixelset \\
        --year_start 2000 --year_end 2005
    # defaults to the 12 cells at 1°E and 77°E; override with --boxes.
"""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from phenonn.data_creation.build_pixelset_targets import _process_year, _read_selected


# (lon_E, lat_N) centres of the requested 2°×2° cells.
DEFAULT_BOXES = [
    (1.0, 7.0),
    (1.0, 9.0),
    (1.0, 11.0),
    (1.0, 43.0),
    (1.0, 45.0),
    (77.0, 11.0),
    (77.0, 13.0),
    (77.0, 15.0),
    (77.0, 17.0),
    (77.0, 37.0),
    (77.0, 39.0),
    (77.0, 41.0),
]


def encode_site_id(lat_idx: int, lon_idx: int) -> str:
    """`pix_LLLL_OOOOO` (inlined to avoid pulling torch via phenonn.data.lai_dataset)."""
    return f"pix_{int(lat_idx):04d}_{int(lon_idx):05d}"


def _read_grid_latlon(path: str):
    """(lat, lon) float64 vectors from a grid file (netcdf4 engine)."""
    ds = xr.open_dataset(path, engine="netcdf4", decode_times=False)
    lat_name = (
        "latitude" if ("latitude" in ds.coords or "latitude" in ds.dims) else "lat"
    )
    lon_name = (
        "longitude" if ("longitude" in ds.coords or "longitude" in ds.dims) else "lon"
    )
    lat = ds[lat_name].values.astype(np.float64)
    lon = ds[lon_name].values.astype(np.float64)
    ds.close()
    return lat, lon


def _idx_in_box(
    coord: np.ndarray, center: float, half: float, eps: float = 1e-6
) -> np.ndarray:
    """Indices of grid coords in the half-open interval [center-half, center+half)."""
    lo, hi = center - half, center + half
    return np.where((coord >= lo - eps) & (coord < hi - eps))[0]


def parse_boxes(s: str):
    """'1,7;77,11' → [(1.0, 7.0), (77.0, 11.0)]  (lon, lat)."""
    out = []
    for tok in s.split(";"):
        tok = tok.strip()
        if not tok:
            continue
        lon, lat = tok.split(",")
        out.append((float(lon), float(lat)))
    return out


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--output_selected", required=True, help="Path to write the selected_pixels.nc."
    )
    p.add_argument(
        "--boxes",
        default="",
        help="Semicolon list of 'lon,lat' box centres (default: the "
        "12 cells at 1°E and 77°E).",
    )
    p.add_argument(
        "--cell_size",
        type=float,
        default=2.0,
        help="Box side length in degrees (default 2.0).",
    )
    p.add_argument(
        "--ref_grid",
        default="",
        help="Grid file to read the reference lat/lon from (default: "
        "{target_dir}/LAI_dekadal_{year_start}.nc).",
    )
    # ── Extraction (LAI + PFT). Provide these to also build the pixelset cubes ──
    p.add_argument("--target_dir", default="", help="Grid LAI_dekadal_*.nc dir.")
    p.add_argument("--pft_dir", default="", help="Grid PFTmap_*.nc dir.")
    p.add_argument("--out_target_dir", default="", help="Pixelset LAI out dir.")
    p.add_argument("--out_pft_dir", default="", help="Pixelset PFT out dir.")
    p.add_argument("--year_start", type=int, default=-1)
    p.add_argument("--year_end", type=int, default=-1)
    p.add_argument("--complevel", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    boxes = parse_boxes(args.boxes) if args.boxes else list(DEFAULT_BOXES)
    half = args.cell_size / 2.0

    do_extract = bool(
        args.target_dir and args.pft_dir and args.out_target_dir and args.out_pft_dir
    )
    if do_extract and (args.year_start < 0 or args.year_end < 0):
        raise ValueError("--year_start/--year_end are required when extracting.")

    # ── Reference grid (must match the LAI/PFT grid indexed later) ──
    ref_grid = args.ref_grid
    if not ref_grid:
        if not (args.target_dir and args.year_start >= 0):
            raise ValueError(
                "Provide --ref_grid, or --target_dir together with --year_start "
                "so the LAI grid of that year can be used as reference."
            )
        from phenonn.utils.config import TARGETS_FNAME

        ref_grid = str(
            Path(args.target_dir) / TARGETS_FNAME.format(year=args.year_start)
        )
    lat, lon = _read_grid_latlon(ref_grid)
    lon_360 = float(np.nanmax(lon)) > 180.0
    print(f"Reference grid : {ref_grid}")
    print(
        f"  lat {lat.min():.3f}..{lat.max():.3f} ({lat.size})  "
        f"lon {lon.min():.3f}..{lon.max():.3f} ({lon.size})  "
        f"{'[0,360]' if lon_360 else '[-180,180]'}"
    )

    # ── Select the 0.1° pixels inside every box ──
    lat_all, lon_all, clon_all, clat_all = [], [], [], []
    for clon, clat in boxes:
        cg = clon % 360 if lon_360 else clon
        li = _idx_in_box(lat, clat, half)
        oi = _idx_in_box(lon, cg, half)
        if li.size == 0 or oi.size == 0:
            print(
                f"  ! box lon={clon:g}E lat={clat:g}N -> EMPTY "
                f"(lat {li.size}, lon {oi.size}) — outside grid?"
            )
            continue
        LA, LO = np.meshgrid(li, oi, indexing="ij")
        lat_all.append(LA.ravel())
        lon_all.append(LO.ravel())
        clon_all.append(np.full(LA.size, clon, np.float32))
        clat_all.append(np.full(LA.size, clat, np.float32))
        print(
            f"  box lon={clon:g}E lat={clat:g}N -> "
            f"{li.size}×{oi.size} = {LA.size} px"
        )

    if not lat_all:
        raise RuntimeError("No pixel selected — check --boxes vs grid extent.")

    lat_idx = np.concatenate(lat_all).astype(np.int64)
    lon_idx = np.concatenate(lon_all).astype(np.int64)
    cell_lon = np.concatenate(clon_all)
    cell_lat = np.concatenate(clat_all)

    # Dedup (safety; half-open boxes shouldn't overlap) + sort lat-major.
    key = lat_idx * (int(lon.size) + 1) + lon_idx
    _, uniq = np.unique(key, return_index=True)
    order = uniq[np.argsort(key[uniq])]
    lat_idx, lon_idx = lat_idx[order], lon_idx[order]
    cell_lon, cell_lat = cell_lon[order], cell_lat[order]

    site_ids = np.array(
        [encode_site_id(a, o) for a, o in zip(lat_idx, lon_idx)], dtype=object
    )
    latitude = lat[lat_idx].astype(np.float32)
    longitude = lon[lon_idx].astype(np.float32)
    print(f"Total selected : {lat_idx.size:,} pixels across {len(boxes)} box(es)")

    out = xr.Dataset(
        {
            "lat_idx": (["site"], lat_idx.astype(np.int32)),
            "lon_idx": (["site"], lon_idx.astype(np.int32)),
            "latitude": (["site"], latitude),
            "longitude": (["site"], longitude),
            "site_id": (["site"], site_ids),
            "cell_lon": (["site"], cell_lon),
            "cell_lat": (["site"], cell_lat),
        },
        attrs={
            "source": "build_boxes_pixelset.py",
            "cell_size": float(args.cell_size),
            "n_boxes": len(boxes),
            "n_selected": int(lat_idx.size),
            "ref_grid": str(ref_grid),
            "boxes": "; ".join(f"{cl:g}E,{ca:g}N" for cl, ca in boxes),
        },
    )
    enc = {
        v: {"zlib": True, "complevel": 4}
        for v in ("lat_idx", "lon_idx", "latitude", "longitude", "cell_lon", "cell_lat")
    }
    outp = Path(args.output_selected).resolve()
    outp.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(outp, engine="netcdf4", encoding=enc)
    print(f"✓ selected_pixels → {outp}")

    # ── Extract LAI + PFT pixelset cubes (reuse the existing extractor) ──
    if not do_extract:
        print("No --target_dir/--pft_dir/out dirs → skipped LAI/PFT extraction.")
        return
    sel = _read_selected(outp)
    tdir, pdir = Path(args.target_dir), Path(args.pft_dir)
    otd, opd = Path(args.out_target_dir), Path(args.out_pft_dir)
    print(f"\nExtracting LAI/PFT for {args.year_start}→{args.year_end} …")
    skipped = []
    for year in range(args.year_start, args.year_end + 1):
        try:
            _process_year(year, sel, tdir, pdir, otd, opd, args.complevel)
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ {year} failed — {type(e).__name__}: {e}")
            skipped.append(year)
    print("\nDone." + (f"  Skipped: {skipped}" if skipped else ""))


if __name__ == "__main__":
    main()
