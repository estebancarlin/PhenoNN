#!/usr/bin/env python3
"""
plot_pixels_lai_pft.py
======================

Same figure as `aggregate_boxes.py --plot_dir`, but per INDIVIDUAL pixel
(no 2° aggregation): one PNG per site listed in a selected_pixels.nc, with one
panel per year (PFT stacked bar on top of that year's dekadal LAI curve).

Reuses `aggregate_boxes._generate_plots` verbatim — this script only reads the
per-site LAI/PFT vectors from the pixelset cubes and hands them over as one row
per (site, year, dekad/pft), using site_id in place of the cell id.

Inputs
------
  --selected_pixels  selected_pixels.nc  (site_id [, latitude, longitude])
  --lai_dir          pixelset LAI dir  (LAI_dekadal_{Y}.nc, LAI(dekad, site))
  --pft_dir          pixelset PFT dir  (PFTmap_{Y}.nc,      pft_frac(pft, site))

Usage
-----
    python -m plot_diagnositcs.plot_pixels_lai_pft \\
        --selected_pixels /data/.../selected_pixels_boxes13.nc \\
        --lai_dir /data/.../LAI_pixelset13 \\
        --pft_dir /data/.../PFT_pixelset13 \\
        --years   1992-2018 \\
        --out_dir /data/.../pixel_plots \\
        [--sites pix_0123_04567,...] [--max_sites 50]

Warning: one PNG per pixel — a big selected_pixels.nc means thousands of files.
Use --sites / --max_sites to restrict.
"""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from phenonn.utils.config import (
    N_DEKAD_YEAR,
    N_PFT,
    PFT_FNAME,
    PFT_NAMES,
    TARGETS_FNAME,
)
from phenonn.data_creation.aggregate_boxes import (
    DEKAD_DOY,
    _generate_plots,
    _open_pixelset,
    parse_years,
)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--selected_pixels",
        required=True,
        help="selected_pixels.nc listing the sites to plot.",
    )
    p.add_argument("--lai_dir", required=True, help="Pixelset LAI dir.")
    p.add_argument("--pft_dir", required=True, help="Pixelset PFT dir.")
    p.add_argument("--years", required=True, help="'1992-2018' or '2000,2003'.")
    p.add_argument("--out_dir", required=True, help="Folder for the per-pixel PNGs.")
    p.add_argument(
        "--sites", default="", help="Comma-separated site_id subset (default: all)."
    )
    p.add_argument(
        "--max_sites",
        type=int,
        default=0,
        help="Cap the number of pixels plotted (0 = no cap).",
    )
    p.add_argument(
        "--ncols",
        type=int,
        default=5,
        help="Number of year panels per row (default 5).",
    )
    p.add_argument(
        "--legend_min_frac",
        type=float,
        default=0.02,
        help="A PFT enters the legend if its fraction reaches this in "
        "any year of the pixel (default 0.02).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # ── Site list (+ lon/lat for the panel titles) ──
    sds = xr.open_dataset(args.selected_pixels)
    site_ids = np.asarray(sds["site_id"].values).astype(str)
    lat = (
        np.asarray(sds["latitude"].values, dtype=float)
        if "latitude" in sds
        else np.full(site_ids.shape, np.nan)
    )
    lon = (
        np.asarray(sds["longitude"].values, dtype=float)
        if "longitude" in sds
        else np.full(site_ids.shape, np.nan)
    )
    sds.close()

    wanted = list(site_ids)
    if args.sites:
        keep = {s.strip() for s in args.sites.split(",") if s.strip()}
        wanted = [s for s in wanted if s in keep]
    if args.max_sites > 0:
        wanted = wanted[: args.max_sites]
    wanted_set = set(wanted)
    lonlat = {
        s: (float(lon[i]), float(lat[i]))
        for i, s in enumerate(site_ids)
        if s in wanted_set
    }
    print(f"Pixels to plot : {len(wanted):,}  (from {len(site_ids):,} in file)")
    if len(wanted) > 500:
        print("  ! that is a lot of PNGs — consider --sites / --max_sites.")

    years = parse_years(args.years)

    # ── Read per-site vectors, one aggregate-style row per (site, year, k) ──
    lai_rows, pft_rows = [], []
    for y in years:
        lsids, lai = _open_pixelset(
            Path(args.lai_dir) / TARGETS_FNAME.format(year=y), "LAI", "dekad"
        )
        psids, pft = _open_pixelset(
            Path(args.pft_dir) / PFT_FNAME.format(year=y), "pft_frac", "pft"
        )
        if lai is not None:
            col = {s: i for i, s in enumerate(lsids)}
            for s in wanted:
                i = col.get(s)
                if i is None:
                    continue
                lo, la = lonlat[s]
                v = lai[:, i]  # (36,)
                for k in range(N_DEKAD_YEAR):
                    m = float(v[k])
                    lai_rows.append(
                        [s, lo, la, y, k + 1, DEKAD_DOY[k], m, int(np.isfinite(m)), 1]
                    )
        if pft is not None:
            col = {s: i for i, s in enumerate(psids)}
            for s in wanted:
                i = col.get(s)
                if i is None:
                    continue
                lo, la = lonlat[s]
                fr = pft[:, i]  # (15,)
                for k in range(N_PFT):
                    m = float(fr[k])
                    pft_rows.append(
                        [s, lo, la, y, k + 1, PFT_NAMES[k], m, int(np.isfinite(m)), 1]
                    )
        print(
            f"  {y}: read "
            f"(LAI {'ok' if lai is not None else '—'}, "
            f"PFT {'ok' if pft is not None else '—'})"
        )

    if not lai_rows:
        raise RuntimeError("No LAI data read — check --lai_dir / --years / sites.")

    print(f"\nPlotting → {args.out_dir}")
    _generate_plots(lai_rows, pft_rows, args.out_dir, args.ncols, args.legend_min_frac)
    print("Done.")


if __name__ == "__main__":
    main()
