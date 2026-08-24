#!/usr/bin/env python3
"""
greedy_pure_pft.py
==================

Greedy hierarchical extraction of *pure-PFT* training pools from a
`selected_pixels.nc`, for the "train one PFT at a time" experiment.

Method
------
Let `frac_p[cell, year]` be the fraction of PFT p in a cell. Starting from an
empty set of "claimed" PFTs, repeat:

  1. For every not-yet-claimed PFT p, a cell *qualifies* for p when, in EVERY
     year of [--year_start, --year_end]:
         frac_p            >  --min_abs            (absolute presence, e.g. 1%)
         frac_p / denom    >  purity threshold     (renormalised purity)
     where  denom = 1 − Σ_{c claimed} frac_c   (per cell, per year; cumulative).
  2. Pick the PFT with the most qualifying cells. If even the best has < --n
     cells, lower the purity threshold by --reduce_step and retry this step;
     the threshold resets to --purity for the next step. If the floor
     --min_purity is reached and still < --n, stop.
  3. Write `selected_pixels_PFT{ORCHIDEE_number}.nc` with the qualifying cells,
     mark the PFT claimed, and continue.

Cells are NOT removed from the pool, so a cell may appear in several PFT files
(gated by --min_abs). PFT numbering is ORCHIDEE 1..15 (axis index = number-1).

Each output file has the same layout as `selected_pixels.nc` (a `site` dim with
lat_idx / lon_idx / latitude / longitude / site_id), so it is a drop-in for
training's `--selected_pixels`, reusing the datasets already built for the full
pool (LAI/PFT/ERA5 pixelsets are keyed by site_id).

Usage
-----
    python -m phenon.dataset_creation.greedy_pure_pft --selected_pixels /data/sbarbu/PhenoNN/data/selected_pixels_100%.nc --pft_dir         /data/sbarbu/PhenoNN/data/pixelset_100%/PFT_pixelset --year_start 2000 --year_end 2005 --purity 1 --min_abs 0.1 --n 10000 --reduce_step 0.01 --out_dir /data/sbarbu/PhenoNN/data/pixelset_100%/greedy_PFT_10000_0.1
"""

import argparse
from pathlib import Path
from typing import Dict, List

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
    p.add_argument("--year_start", type=int, required=True)
    p.add_argument("--year_end",   type=int, required=True,
                   help="Conditions must hold in every year of the range.")
    p.add_argument("--purity", type=float, default=0.95,
                   help="Start purity threshold frac_p/denom (default 0.95).")
    p.add_argument("--min_abs", type=float, default=0.01,
                   help="Absolute frac_p floor for membership (default 0.01).")
    p.add_argument("--n", type=int, default=100,
                   help="Min cells to accept a PFT at the current threshold.")
    p.add_argument("--reduce_step", type=float, default=0.05,
                   help="Purity decrement when no PFT reaches --n (default 0.05).")
    p.add_argument("--min_purity", type=float, default=0.5,
                   help="Floor below which the threshold is not reduced.")
    p.add_argument("--max_pft", type=int, default=N_PFT,
                   help="Stop after extracting this many PFTs (default all).")
    p.add_argument("--out_dir", default="",
                   help="Output dir (default: selected_pixels' dir).")
    return p.parse_args()


def _aligned_frac(pft_path: Path, site_ids: np.ndarray) -> np.ndarray:
    """Return frac (N_PFT, n_site) aligned to site_ids; NaN for missing sites."""
    with xr.open_dataset(pft_path, engine="netcdf4", decode_times=False) as pds:
        pf_sites = np.asarray(pds["site_id"].values).astype(str)
        arr = pds["pft_frac"].transpose("pft", "site").values.astype(np.float32)
    idx_of = {s: i for i, s in enumerate(pf_sites)}
    rows = np.array([idx_of.get(s, -1) for s in site_ids], dtype=np.int64)
    out = np.full((arr.shape[0], site_ids.size), np.nan, dtype=np.float32)
    ok = rows >= 0
    out[:, ok] = arr[:, rows[ok]]
    return out


def _qualify(F: Dict[int, np.ndarray], denom: Dict[int, np.ndarray],
             p: int, thr: float, min_abs: float) -> np.ndarray:
    """Boolean mask (n_site): PFT p qualifies in EVERY year at threshold thr."""
    mask = None
    for y, Fy in F.items():
        fp = Fy[p]
        d  = denom[y]
        with np.errstate(invalid="ignore", divide="ignore"):
            purity = fp / d
        ok_y = (np.isfinite(fp) & (fp > min_abs)
                & np.isfinite(d) & (d > 0.0) & (purity > thr))
        mask = ok_y if mask is None else (mask & ok_y)
    return mask


def main():
    args = parse_args()
    sel_path = Path(args.selected_pixels).resolve()
    pft_dir  = Path(args.pft_dir).resolve()
    out_dir  = Path(args.out_dir).resolve() if args.out_dir else sel_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    sel = xr.open_dataset(sel_path)
    site_ids  = np.asarray(sel["site_id"].values).astype(str)
    lat_idx   = sel["lat_idx"].values.astype(np.int32)
    lon_idx   = sel["lon_idx"].values.astype(np.int32)
    latitude  = sel["latitude"].values.astype(np.float32)
    longitude = sel["longitude"].values.astype(np.float32)
    sel.close()
    n_site = site_ids.size

    years = list(range(args.year_start, args.year_end + 1))
    F: Dict[int, np.ndarray] = {}
    for y in years:
        ppath = pft_dir / PFT_FNAME.format(year=y)
        if not ppath.exists():
            raise SystemExit(f"Missing PFT pixelset file: {ppath}")
        F[y] = _aligned_frac(ppath, site_ids)
    print(f"{n_site:,} sites, years {args.year_start}-{args.year_end}, "
          f"purity={args.purity}, min_abs={args.min_abs}, n={args.n}")

    encoding = {
        "lat_idx":   {"zlib": True, "complevel": 4, "dtype": "int32"},
        "lon_idx":   {"zlib": True, "complevel": 4, "dtype": "int32"},
        "latitude":  {"zlib": True, "complevel": 4, "dtype": "float32"},
        "longitude": {"zlib": True, "complevel": 4, "dtype": "float32"},
    }

    claimed: List[int] = []
    remaining = list(range(N_PFT))           # 0-based axis indices
    summary = []

    while remaining and len(claimed) < args.max_pft:
        # denom per year (per cell): 1 − Σ claimed fractions (cumulative)
        denom: Dict[int, np.ndarray] = {}
        for y, Fy in F.items():
            d = np.ones(n_site, dtype=np.float32)
            for c in claimed:
                d = d - Fy[c]
            denom[y] = d

        thr = args.purity
        best_p = best_mask = None
        best_count = -1
        placed = False
        while thr >= args.min_purity - 1e-9:
            best_p, best_mask, best_count = None, None, -1
            for p in remaining:
                mask = _qualify(F, denom, p, thr, args.min_abs)
                count = int(mask.sum())
                if count > best_count:
                    best_count, best_p, best_mask = count, p, mask
            if best_count >= args.n:
                placed = True
                break
            thr = round(thr - args.reduce_step, 6)

        if not placed:
            print(f"Stop: best remaining PFT has only {best_count} cells "
                  f"(< n={args.n}) even at purity floor {args.min_purity}.")
            break

        keep_idx = np.where(best_mask)[0]
        pft_num = best_p + 1                 # ORCHIDEE numbering
        out = xr.Dataset(
            {
                "lat_idx":   (["site"], lat_idx[keep_idx]),
                "lon_idx":   (["site"], lon_idx[keep_idx]),
                "latitude":  (["site"], latitude[keep_idx]),
                "longitude": (["site"], longitude[keep_idx]),
                "site_id":   (["site"], site_ids[keep_idx].astype(object)),
            },
            attrs={
                "source":                  "greedy_pure_pft.py",
                "source_selected_pixels":  sel_path.name,
                "filter_pft_orchidee":     int(pft_num),
                "step_order":              len(claimed) + 1,
                "claimed_before":          ",".join(str(c + 1) for c in claimed),
                "purity_threshold_used":   float(thr),
                "min_abs_frac":            float(args.min_abs),
                "years":                   f"{args.year_start}-{args.year_end}",
                "n_selected":              int(keep_idx.size),
            },
        )
        out_path = out_dir / f"selected_pixels_PFT{pft_num}.nc"
        out.to_netcdf(out_path, engine="netcdf4", encoding=encoding)
        print(f"  step {len(claimed)+1}: PFT{pft_num}  "
              f"{keep_idx.size:,} cells (purity>{thr:g}) → {out_path.name}")
        summary.append((len(claimed) + 1, pft_num, thr, int(keep_idx.size)))

        claimed.append(best_p)
        remaining.remove(best_p)

    lines = ["Order extracted (step, PFT, purity_used, n_cells):"]
    for s in summary:
        lines.append(f"  {s[0]:>2}  PFT{s[1]:<2}  purity>{s[2]:g}  {s[3]:,}")
    text = "\n".join(lines)
    print("\n" + text)
    (out_dir / "extraction_order.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
