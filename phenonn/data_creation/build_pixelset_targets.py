#!/usr/bin/env python3
"""
build_pixelset_targets.py
=========================

Pre-extract the full-grid LAI and PFT files into *pixelset* files indexed by
the selected sites — the same flat (site)-indexed layout as
ERA5_daily_pixelset_{Y}.nc.

Why
---
`LAI_dekadal_{Y}.nc` (chunked 36×32×32, deflate 1) and `PFTmap_{Y}.nc`
(chunked 1×4×600×1200, deflate 4 + shuffle) are FULL 1800×3600 grids. Reading a
few hundred scattered sites from them with a vectorised
`isel(latitude=…, longitude=…)` forces HDF5 to decompress almost every chunk,
and the tiny 32×32 LAI chunks (6441 of them) overflow the 1 MB HDF5 chunk cache,
so the same chunks are decompressed many times (thrashing). Over NFS this is
minutes per year. Extracting once to a (dekad, site) / (pft, site) file turns
every later read into a few kB.

Output (same filenames as the grid files, written into the --out_* dirs):
    {out_target_dir}/LAI_dekadal_{Y}.nc   LAI(dekad, site)
    {out_pft_dir}/PFTmap_{Y}.nc           pft_frac(pft, site)
both carrying site_id / lat_idx / lon_idx / latitude / longitude coords so the
data loader can map site_id → row exactly like the ERA5 pixelset.

Point training at the new dirs:
    --target_dir {out_target_dir}  --pft_dir {out_pft_dir}
`phenonn.data.lai_dataset` auto-detects the pixelset layout (a `site` dim).

On the spirit `.venv_ERA5`, netcdf4 and h5py ship separate bundled libhdf5 that
corrupt each other in one process. Importing phenon pulls netcdf4's libhdf5
first, so BOTH grids are read with engine="netcdf4" (this node opens the
ORCHIDEE PFTmap fine that way) to keep the whole process on one libhdf5; the
write is still isolated in a fresh subprocess (see phenon/data/_ncwrite.py).

Usage
-----
    python -m phenonn.data_creation.build_pixelset_targets \\
        --selected_pixels /data/sbarbu/PhenoNN/data/selected_pixels.nc \\
        --target_dir      /data/sbarbu/PhenoNN/data/LAI/LAI2000_2005 \\
        --pft_dir         /data/sbarbu/PhenoNN/data/PFT_map0.1 \\
        --out_target_dir  /data/sbarbu/PhenoNN/data/LAI_pixelset \\
        --out_pft_dir     /data/sbarbu/PhenoNN/data/PFT_pixelset \\
        --year_start 1999 --year_end 2005
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import xarray as xr

from phenonn.utils.config import N_DEKAD_YEAR, N_PFT, PFT_FNAME, TARGETS_FNAME
from phenonn.data_creation._ncwrite import to_netcdf_subprocess


def _load_pft_vector_array(ds: xr.Dataset) -> xr.DataArray:
    """
    Return a single DataArray of dims (pft, latitude, longitude) from a PFT
    file. Inlined here (instead of importing from phenonn.data.lai_dataset) so this
    preprocessing step does not pull in torch. Mirrors the loader in
    phenonn.data.lai_dataset / phenonn.data_creation.compute_valid_pixels.
    """
    rename = {}
    for src, dst in [("lat", "latitude"), ("lon", "longitude"), ("veget", "pft")]:
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
            [ds[v].expand_dims(pft=[k - 1]) for k, v in enumerate(pft_named, start=1)],
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


def _read_selected(selected_path: Path):
    """Return (site_id, lat_idx, lon_idx, latitude, longitude) arrays."""
    ds = xr.open_dataset(selected_path)
    sel = {
        "site_id": np.asarray(ds["site_id"].values).astype(str),
        "lat_idx": ds["lat_idx"].values.astype(np.int64),
        "lon_idx": ds["lon_idx"].values.astype(np.int64),
        "latitude": np.asarray(ds["latitude"].values, dtype=np.float32),
        "longitude": np.asarray(ds["longitude"].values, dtype=np.float32),
    }
    ds.close()
    return sel


def _site_coords(sel) -> dict:
    return {
        "site_id": ("site", sel["site_id"]),
        "lat_idx": ("site", sel["lat_idx"]),
        "lon_idx": ("site", sel["lon_idx"]),
        "latitude": ("site", sel["latitude"]),
        "longitude": ("site", sel["longitude"]),
    }


def _process_year(
    year: int,
    sel,
    target_dir: Path,
    pft_dir: Path,
    out_target_dir: Path,
    out_pft_dir: Path,
    complevel: int,
) -> None:
    lat_idx, lon_idx = sel["lat_idx"], sel["lon_idx"]

    # ── LAI grid → pixelset ──
    tpath = target_dir / TARGETS_FNAME.format(year=year)
    if tpath.exists():
        # engine="netcdf4" for BOTH reads: importing phenon pulls netcdf4's
        # libhdf5 into this process first, so a later h5netcdf read trips the
        # dual-libhdf5 conflict ("H5DSget_num_scales <0"). On this node netcdf4
        # opens the ORCHIDEE PFTmap fine, so the whole process stays on netcdf4
        # (the documented fix). The write is still isolated in a subprocess.
        with xr.open_dataset(tpath, engine="netcdf4") as tgt:
            # One sequential decompression pass of the whole grid, then a numpy
            # fancy-index — no scattered isel, no chunk-cache thrashing.
            lai_full = tgt["LAI"].transpose("dekad", "latitude", "longitude").values
            vals = lai_full[:, lat_idx, lon_idx].astype(np.float32)  # (36, n_site)
            dekad = (
                tgt["dekad"].values
                if "dekad" in tgt.coords
                else np.arange(N_DEKAD_YEAR)
            )
            lai_attrs = dict(tgt["LAI"].attrs)
        out = xr.Dataset(
            {"LAI": (("dekad", "site"), vals)},
            coords={"dekad": dekad, **_site_coords(sel)},
        )
        out["LAI"].attrs.update(lai_attrs)
        enc = {
            "LAI": {
                "zlib": True,
                "complevel": complevel,
                "_FillValue": np.float32(np.nan),
            }
        }
        out_target_dir.mkdir(parents=True, exist_ok=True)
        to_netcdf_subprocess(
            out,
            out_target_dir / TARGETS_FNAME.format(year=year),
            encoding=enc,
            engine="netcdf4",
        )
        n_nan = int(np.isnan(vals).all(axis=0).sum())
        print(f"  ✓ LAI    {year}  ({vals.shape[1]:,} sites, {n_nan} all-NaN)")
    else:
        print(f"  ✗ LAI    {year} skipped — missing {tpath.name}")

    # ── PFT grid → pixelset ──
    ppath = pft_dir / PFT_FNAME.format(year=year)
    if ppath.exists():
        with xr.open_dataset(ppath, engine="netcdf4", decode_times=False) as pds:
            pda = _load_pft_vector_array(pds).transpose("pft", "latitude", "longitude")
            pft_full = pda.values
            pv = pft_full[:, lat_idx, lon_idx].astype(np.float32)  # (15, n_site)
        outp = xr.Dataset(
            {"pft_frac": (("pft", "site"), pv)},
            coords={"pft": np.arange(N_PFT, dtype=np.int32), **_site_coords(sel)},
        )
        enc = {
            "pft_frac": {
                "zlib": True,
                "complevel": complevel,
                "_FillValue": np.float32(np.nan),
            }
        }
        out_pft_dir.mkdir(parents=True, exist_ok=True)
        to_netcdf_subprocess(
            outp,
            out_pft_dir / PFT_FNAME.format(year=year),
            encoding=enc,
            engine="netcdf4",
        )
        print(f"  ✓ PFT    {year}  ({pv.shape[1]:,} sites)")
    else:
        print(f"  ✗ PFT    {year} skipped — missing {ppath.name}")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--selected_pixels",
        required=True,
        help="selected_pixels.nc (site_id, lat_idx, lon_idx, " "latitude, longitude).",
    )
    p.add_argument("--target_dir", required=True, help="Grid LAI_dekadal_*.nc in.")
    p.add_argument("--pft_dir", required=True, help="Grid PFTmap_*.nc in.")
    p.add_argument("--out_target_dir", required=True, help="Pixelset LAI out.")
    p.add_argument("--out_pft_dir", required=True, help="Pixelset PFT out.")
    p.add_argument("--year_start", type=int, required=True)
    p.add_argument("--year_end", type=int, required=True)
    p.add_argument("--complevel", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    selected_path = Path(args.selected_pixels).resolve()
    target_dir = Path(args.target_dir).resolve()
    pft_dir = Path(args.pft_dir).resolve()
    out_target = Path(args.out_target_dir).resolve()
    out_pft = Path(args.out_pft_dir).resolve()

    sel = _read_selected(selected_path)
    print(f"Selected pixels : {selected_path}  ({len(sel['site_id']):,} sites)")
    print(f"Grid LAI in     : {target_dir}")
    print(f"Grid PFT in     : {pft_dir}")
    print(f"Pixelset LAI out: {out_target}")
    print(f"Pixelset PFT out: {out_pft}")
    print(f"Years           : {args.year_start} → {args.year_end}")

    skipped: List[int] = []
    for year in range(args.year_start, args.year_end + 1):
        try:
            _process_year(
                year, sel, target_dir, pft_dir, out_target, out_pft, args.complevel
            )
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ {year} failed — {type(e).__name__}: {e}")
            skipped.append(year)

    print("\nDone.")
    if skipped:
        print(f"Skipped years: {skipped}")


if __name__ == "__main__":
    main()
