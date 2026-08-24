#!/usr/bin/env python3
"""
filter_pixels_by_pft.py
=======================

Take an existing `selected_pixels.nc` and write a *subset* keeping only the
sites dominated by a given PFT, e.g. `selected_pixels_PFT5.nc`. Training can
then run on that subset (`--selected_pixels selected_pixels_PFT5.nc`) while
REUSING the datasets already built for the full `selected_pixels.nc`
(LAI_pixelset / PFT_pixelset / ERA5 pixelset) — those are indexed by `site_id`,
and a subset of sites needs no re-extraction.

Criterion
---------
A site is kept when its PFT fraction for the requested PFT is ≥ `--min_frac`
(default 0.5, i.e. the cell is dominated by that PFT) in EVERY year of
`--year_start .. --year_end`. `--pft` is the ORCHIDEE PFT number 1..15 (mapped
to axis index pft-1 in the PFT pixelset). A site failing the threshold in any
year — or missing/NaN in any year — is excluded.

Output has exactly the same layout as `selected_pixels.nc` (a `site` dim with
lat_idx / lon_idx / latitude / longitude / site_id), so it is a drop-in for
`--selected_pixels`.

Usage
-----
    python -m phenon.dataset_creation.filter_pixels_by_pft \\
        --selected_pixels /data/sbarbu/PhenoNN/data/selected_pixels.nc \\
        --pft_dir         /data/sbarbu/PhenoNN/data/pixelset_10%/PFT_pixelset \\
        --pft 5 --year_start 2000 --year_end 2005 --min_frac 0.5
    # → /data/sbarbu/PhenoNN/data/selected_pixels_PFT5.nc
"""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from phenon.training.config import N_PFT, PFT_FNAME


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--selected_pixels", required=True,
                   help="Input selected_pixels.nc (full site pool).")
    p.add_argument("--pft_dir", required=True,
                   help="PFT pixelset dir (PFTmap_{year}.nc, pft_frac(pft, site)).")
    p.add_argument("--pft", type=int, required=True,
                   help="ORCHIDEE PFT number 1..15 to filter on.")
    p.add_argument("--year_start", type=int, required=True,
                   help="First year of the PFT pixelset checked.")
    p.add_argument("--year_end", type=int, required=True,
                   help="Last year (inclusive); the threshold must hold every year.")
    p.add_argument("--min_frac", type=float, default=0.5,
                   help="Keep sites with pft_frac >= this threshold (default 0.5).")
    p.add_argument("--output", default="",
                   help="Output path (default: <selected_pixels dir>/"
                        "selected_pixels_PFT{pft}.nc).")
    return p.parse_args()


def main():
    args = parse_args()
    if not (1 <= args.pft <= N_PFT):
        raise SystemExit(f"--pft must be in 1..{N_PFT} (ORCHIDEE numbering).")
    pft_axis = args.pft - 1   # ORCHIDEE 1..15 → 0-based axis in the pixelset

    sel_path = Path(args.selected_pixels).resolve()
    sel = xr.open_dataset(sel_path)
    site_ids = np.asarray(sel["site_id"].values).astype(str)
    pft_dir = Path(args.pft_dir).resolve()
    thr = np.float32(args.min_frac)

    # ── A site is kept only if pft_frac >= thr in EVERY year (AND over years).
    #    Missing/NaN in any year fails the site. ──
    keep = np.ones(site_ids.shape, dtype=bool)
    years = range(args.year_start, args.year_end + 1)
    for year in years:
        pft_path = pft_dir / PFT_FNAME.format(year=year)
        if not pft_path.exists():
            raise SystemExit(f"Missing PFT pixelset file: {pft_path}")
        with xr.open_dataset(pft_path, engine="netcdf4", decode_times=False) as pds:
            pf_sites = np.asarray(pds["site_id"].values).astype(str)
            frac_all = pds["pft_frac"].isel(pft=pft_axis).values.astype(np.float32)
        idx_of = {s: i for i, s in enumerate(pf_sites)}
        rows = np.array([idx_of.get(s, -1) for s in site_ids], dtype=np.int64)
        frac = np.full(site_ids.shape, np.nan, dtype=np.float32)
        ok = rows >= 0
        frac[ok] = frac_all[rows[ok]]
        keep &= np.isfinite(frac) & (frac >= thr)
        print(f"  {year}: {int(keep.sum()):,} site(s) still passing")

    keep_idx = np.where(keep)[0]

    out = xr.Dataset(
        {
            "lat_idx":   (["site"], sel["lat_idx"].values[keep_idx].astype(np.int32)),
            "lon_idx":   (["site"], sel["lon_idx"].values[keep_idx].astype(np.int32)),
            "latitude":  (["site"], sel["latitude"].values[keep_idx].astype(np.float32)),
            "longitude": (["site"], sel["longitude"].values[keep_idx].astype(np.float32)),
            "site_id":   (["site"], site_ids[keep_idx].astype(object)),
        },
        attrs={
            "source":                "filter_pixels_by_pft.py",
            "source_selected_pixels": sel_path.name,
            "filter_pft_orchidee":   int(args.pft),
            "filter_pft_axis":       int(pft_axis),
            "filter_years":          f"{args.year_start}-{args.year_end}",
            "filter_min_frac":       float(args.min_frac),
            "n_input":               int(site_ids.size),
            "n_selected":            int(keep_idx.size),
        },
    )
    sel.close()

    out_path = (Path(args.output).resolve() if args.output
                else sel_path.parent / f"selected_pixels_PFT{args.pft}.nc")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {
        "lat_idx":   {"zlib": True, "complevel": 4, "dtype": "int32"},
        "lon_idx":   {"zlib": True, "complevel": 4, "dtype": "int32"},
        "latitude":  {"zlib": True, "complevel": 4, "dtype": "float32"},
        "longitude": {"zlib": True, "complevel": 4, "dtype": "float32"},
    }
    out.to_netcdf(out_path, engine="netcdf4", encoding=encoding)

    print(f"PFT{args.pft} (axis {pft_axis}), years "
          f"{args.year_start}-{args.year_end}, min_frac={args.min_frac} (all years)")
    print(f"  {keep_idx.size:,} / {site_ids.size:,} sites kept → {out_path}")


if __name__ == "__main__":
    main()
