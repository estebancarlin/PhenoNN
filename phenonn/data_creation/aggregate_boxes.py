#!/usr/bin/env python3
"""
aggregate_boxes.py
==================

Aggregate the box pixelset (built by build_boxes_pixelset.py) to ONE value per
coarse cell: a spatial mean over the 0.1° pixels of each 2° box.

Produces, per (cell, year):
  * LAI   : mean over the cell's valid pixels → a 36-dekad series
  * PFT   : mean over the cell's pixels of each of the 15 fractions
            (the aggregated composition; sums to ~1 like the inputs)

Cell membership comes from the `cell_lon / cell_lat` coords written by
build_boxes_pixelset.py into selected_pixels.nc. The mean is NaN-safe (ocean /
desert all-NaN pixels are skipped and counted).

Inputs
------
  --selected_pixels  selected_pixels.nc  (must carry cell_lon / cell_lat)
  --lai_dir          pixelset LAI dir  (LAI_dekadal_{Y}.nc, LAI(dekad, site))
  --pft_dir          pixelset PFT dir  (PFTmap_{Y}.nc,      pft_frac(pft, site))

Output (tidy CSV, long format)
------------------------------
  {out_prefix}_lai.csv : cell_id, cell_lon, cell_lat, year, dekad, doy,
                         lai_mean, n_valid_px, n_px
  {out_prefix}_pft.csv : cell_id, cell_lon, cell_lat, year, pft_index,
                         pft_name, frac_mean, n_valid_px, n_px

Optional plots (--plot_dir)
---------------------------
One PNG per cell: a small-multiples grid with ONE panel per year, each showing
a 100 %-stacked bar of that year's PFT proportions on top of the year's dekadal
LAI curve. matplotlib is imported lazily (only when --plot_dir is given).

Usage
-----
    python -m phenonn.data_creation.aggregate_boxes \\
        --selected_pixels /data/.../selected_pixels_boxes.nc \\
        --lai_dir /data/.../LAI_pixelset_boxes \\
        --pft_dir /data/.../PFT_pixelset_boxes \\
        --years   2000-2005 \\
        --out_prefix /data/.../boxes_agg \\
        --plot_dir  /data/.../boxes_plots      # optional
"""

import argparse
import csv
import datetime
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import xarray as xr

from phenonn.utils.config import (
    DEKAD_DAYS,
    N_DEKAD_YEAR,
    N_PFT,
    PFT_FNAME,
    PFT_NAMES,
    TARGETS_FNAME,
)


# DOY of each of the 36 dekads (days 5/15/25 of every month, non-leap year).
DEKAD_DOY = [
    datetime.date(2001, m, d).timetuple().tm_yday
    for m in range(1, 13)
    for d in DEKAD_DAYS
]


def cell_id(lon: float, lat: float) -> str:
    ew = "E" if lon >= 0 else "W"
    ns = "N" if lat >= 0 else "S"
    return f"cell_{abs(lon):g}{ew}_{abs(lat):g}{ns}"


def parse_years(s: str):
    if "-" in s and "," not in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def _open_pixelset(path: Path, var: str, dim: str):
    """Return (site_id[str], values[(size(dim), n_site)] float64) or (None, None)."""
    if not path.exists():
        return None, None
    ds = xr.open_dataset(path, engine="netcdf4", decode_times=False)
    sids = np.asarray(ds["site_id"].values).astype(str)
    arr = ds[var].transpose(dim, "site").values.astype(np.float64)
    ds.close()
    return sids, arr


def _cell_mean(block: np.ndarray):
    """NaN-safe mean over axis=1 of a (K, n) block. Returns (mean[K], n_valid[K])."""
    finite = np.isfinite(block)
    nval = finite.sum(axis=1)
    s = np.where(finite, block, 0.0).sum(axis=1)
    mean = np.where(nval > 0, s / np.maximum(nval, 1), np.nan)
    return mean, nval


def _cols_by_cell(sids, cell_of, cells):
    """{cell_key: [column indices in this file]}."""
    d = {c: [] for c in cells}
    for i, s in enumerate(sids):
        c = cell_of.get(s)
        if c in d:
            d[c].append(i)
    return d


def _generate_plots(lai_rows, pft_rows, out_dir, ncols, legend_min_frac):
    """One PNG per cell: per-year panels, PFT stacked bar on top of the LAI
    curve. Works straight off the in-memory aggregation rows (no CSV re-read,
    no pandas). matplotlib is imported here so aggregation-only runs don't need it."""
    import math

    import matplotlib

    matplotlib.use("Agg")  # headless server
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    from matplotlib.patches import Patch

    def _color(i):  # stable colour per PFT (1..15)
        return plt.get_cmap("tab20")((i - 1) % 20)

    lai_c, pft_c = defaultdict(list), defaultdict(list)
    for r in lai_rows:
        lai_c[r[0]].append(r)  # r = [cid, lo, la, y, dekad, doy, mean, ...]
    for r in pft_rows:
        pft_c[r[0]].append(r)  # r = [cid, lo, la, y, pft_idx, name, mean, ...]

    os.makedirs(out_dir, exist_ok=True)
    for cid in sorted(lai_c):
        Lrows, Prows = lai_c[cid], pft_c.get(cid, [])
        lo, la = Lrows[0][1], Lrows[0][2]
        years = sorted({r[3] for r in Lrows})

        lai_year = defaultdict(list)  # year -> [(doy, mean)]
        finite = []
        for r in Lrows:
            lai_year[r[3]].append((r[5], r[6]))
            if np.isfinite(r[6]):
                finite.append(r[6])
        ymax = max(finite) * 1.08 if finite else 1.0

        pft_year = defaultdict(dict)  # year -> {pft_idx: frac}
        pmax = defaultdict(float)
        for r in Prows:
            pft_year[r[3]][r[4]] = r[6]
            if np.isfinite(r[6]):
                pmax[r[4]] = max(pmax[r[4]], r[6])
        present = [
            i for i in range(1, N_PFT + 1) if pmax.get(i, 0.0) >= legend_min_frac
        ]

        n = len(years)
        nc = max(1, min(ncols, n))
        nr = math.ceil(n / nc)
        fig = plt.figure(figsize=(nc * 3.1, nr * 2.5 + 1.4))
        outer = fig.add_gridspec(nr, nc, hspace=0.55, wspace=0.28)
        for j, y in enumerate(years):
            rr, cc = divmod(j, nc)
            inner = GridSpecFromSubplotSpec(
                2, 1, subplot_spec=outer[rr, cc], height_ratios=[1, 5], hspace=0.08
            )
            ax_bar = fig.add_subplot(inner[0])
            ax_lai = fig.add_subplot(inner[1])

            left = 0.0
            fr = pft_year.get(y, {})
            for i in range(1, N_PFT + 1):
                f = fr.get(i, 0.0)
                if not np.isfinite(f) or f <= 0:
                    continue
                ax_bar.barh(0, f, left=left, height=1.0, color=_color(i))
                left += f
            ax_bar.set_xlim(0, 1)
            ax_bar.set_ylim(-0.5, 0.5)
            ax_bar.set_xticks([])
            ax_bar.set_yticks([])
            ax_bar.set_title(str(int(y)), fontsize=10, pad=2)

            series = sorted(lai_year.get(y, []))
            xs = [d for d, _ in series]
            ys = [v for _, v in series]
            ax_lai.plot(xs, ys, marker=".", ms=3, lw=1.2, color="forestgreen")
            ax_lai.set_xlim(0, 366)
            ax_lai.set_ylim(0, ymax)
            ax_lai.tick_params(labelsize=7)
            ax_lai.set_xticks([1, 120, 240, 360])
            if cc == 0:
                ax_lai.set_ylabel("LAI", fontsize=8)
            if rr == nr - 1:
                ax_lai.set_xlabel("DOY", fontsize=8)

        for j in range(n, nr * nc):
            rr, cc = divmod(j, nc)
            fig.add_subplot(outer[rr, cc]).axis("off")

        handles = [Patch(color=_color(i), label=PFT_NAMES[i - 1]) for i in present]
        if handles:
            fig.legend(
                handles=handles,
                loc="lower center",
                ncol=min(len(handles), 5),
                fontsize=8,
                frameon=False,
                bbox_to_anchor=(0.5, 0.0),
            )
        fig.suptitle(
            f"{cid}   (centre {lo:g}°E, {la:g}°N)   LAI & PFT per year",
            fontsize=13,
            y=0.995,
        )
        path = os.path.join(out_dir, f"{cid}.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ plot {cid} → {path}")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--selected_pixels",
        required=True,
        help="selected_pixels.nc with cell_lon / cell_lat "
        "(from build_boxes_pixelset.py).",
    )
    p.add_argument("--lai_dir", required=True, help="Pixelset LAI dir.")
    p.add_argument("--pft_dir", required=True, help="Pixelset PFT dir.")
    p.add_argument("--years", required=True, help="'2000-2005' or '2000,2003'.")
    p.add_argument(
        "--out_prefix",
        required=True,
        help="Output CSV prefix ({prefix}_lai.csv, {prefix}_pft.csv).",
    )
    p.add_argument(
        "--plot_dir",
        default="",
        help="If set, also write one PNG per cell here "
        "(per-year LAI curve + PFT stacked bar).",
    )
    p.add_argument(
        "--ncols",
        type=int,
        default=5,
        help="Plots: number of year panels per row (default 5).",
    )
    p.add_argument(
        "--legend_min_frac",
        type=float,
        default=0.02,
        help="Plots: a PFT enters the legend if its fraction reaches "
        "this in any year of the cell (default 0.02).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # ── Cell membership from selected_pixels.nc ──
    sds = xr.open_dataset(args.selected_pixels)
    if "cell_lon" not in sds or "cell_lat" not in sds:
        raise ValueError(
            f"{args.selected_pixels} has no cell_lon/cell_lat — regenerate it "
            f"with build_boxes_pixelset.py."
        )
    site_ids = np.asarray(sds["site_id"].values).astype(str)
    clon = np.asarray(sds["cell_lon"].values, dtype=float)
    clat = np.asarray(sds["cell_lat"].values, dtype=float)
    sds.close()

    # Round keys to avoid float32 equality pitfalls.
    keys = [(round(float(lo), 3), round(float(la), 3)) for lo, la in zip(clon, clat)]
    cell_of = {s: k for s, k in zip(site_ids, keys)}
    cells = sorted(set(keys))  # (lon, lat), lon-major
    print(f"Cells         : {len(cells)}  ({sum(1 for _ in site_ids):,} pixels)")

    years = parse_years(args.years)

    lai_rows = []  # [cid, lon, lat, year, dekad, doy, mean, n_valid, n_px]
    pft_rows = []  # [cid, lon, lat, year, pft_idx, name, mean, n_valid, n_px]
    for y in years:
        lai_sids, lai = _open_pixelset(
            Path(args.lai_dir) / TARGETS_FNAME.format(year=y), "LAI", "dekad"
        )
        pft_sids, pft = _open_pixelset(
            Path(args.pft_dir) / PFT_FNAME.format(year=y), "pft_frac", "pft"
        )
        if lai is None and pft is None:
            print(f"  {y}: no LAI and no PFT file — skipped")
            continue

        lai_cols = _cols_by_cell(lai_sids, cell_of, cells) if lai is not None else {}
        pft_cols = _cols_by_cell(pft_sids, cell_of, cells) if pft is not None else {}

        for lo, la in cells:
            cid = cell_id(lo, la)
            if lai is not None:
                cols = lai_cols.get((lo, la), [])
                if cols:
                    mean, nval = _cell_mean(lai[:, cols])  # (36,), (36,)
                    for k in range(N_DEKAD_YEAR):
                        lai_rows.append(
                            [
                                cid,
                                lo,
                                la,
                                y,
                                k + 1,
                                DEKAD_DOY[k],
                                float(mean[k]),
                                int(nval[k]),
                                len(cols),
                            ]
                        )
            if pft is not None:
                cols = pft_cols.get((lo, la), [])
                if cols:
                    mean, nval = _cell_mean(pft[:, cols])  # (15,), (15,)
                    for k in range(N_PFT):
                        pft_rows.append(
                            [
                                cid,
                                lo,
                                la,
                                y,
                                k + 1,
                                PFT_NAMES[k],
                                float(mean[k]),
                                int(nval[k]),
                                len(cols),
                            ]
                        )
        print(
            f"  {y}: aggregated "
            f"(LAI {'ok' if lai is not None else '—'}, "
            f"PFT {'ok' if pft is not None else '—'})"
        )

    # ── Write CSVs ──
    def _fmt(v):
        return f"{v:.5f}" if isinstance(v, float) else v

    out_lai = f"{args.out_prefix}_lai.csv"
    out_pft = f"{args.out_prefix}_pft.csv"
    Path(out_lai).parent.mkdir(parents=True, exist_ok=True)
    with open(out_lai, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "cell_id",
                "cell_lon",
                "cell_lat",
                "year",
                "dekad",
                "doy",
                "lai_mean",
                "n_valid_px",
                "n_px",
            ]
        )
        w.writerows([_fmt(x) for x in r] for r in lai_rows)
    with open(out_pft, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "cell_id",
                "cell_lon",
                "cell_lat",
                "year",
                "pft_index",
                "pft_name",
                "frac_mean",
                "n_valid_px",
                "n_px",
            ]
        )
        w.writerows([_fmt(x) for x in r] for r in pft_rows)
    print(f"\n✓ LAI series → {out_lai}  ({len(lai_rows):,} rows)")
    print(f"✓ PFT agg    → {out_pft}  ({len(pft_rows):,} rows)")

    # ── Compact summary per cell (pooled over years) ──
    lai_pool = defaultdict(list)
    for cid, lo, la, y, k, doy, m, nv, npx in lai_rows:
        if np.isfinite(m):
            lai_pool[cid].append(m)
    pft_pool = defaultdict(lambda: defaultdict(list))
    for cid, lo, la, y, k, name, m, nv, npx in pft_rows:
        if np.isfinite(m):
            pft_pool[cid][name].append(m)

    print("\n── Summary (pooled over years) ──")
    for lo, la in cells:
        cid = cell_id(lo, la)
        lm = lai_pool.get(cid, [])
        lai_str = f"mean LAI={np.mean(lm):.3f}" if lm else "LAI all-NaN"
        tops = sorted(
            ((np.mean(v), n) for n, v in pft_pool.get(cid, {}).items()), reverse=True
        )[:4]
        pft_str = ", ".join(f"{n}={m:.2f}" for m, n in tops if m >= 0.01)
        print(f"  {cid:18s} {lai_str:18s} | {pft_str}")

    # ── Optional per-cell PNGs ──
    if args.plot_dir:
        print(f"\nPlotting per-cell figures → {args.plot_dir}")
        _generate_plots(
            lai_rows, pft_rows, args.plot_dir, args.ncols, args.legend_min_frac
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
