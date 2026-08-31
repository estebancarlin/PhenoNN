#!/usr/bin/env python3
"""
build_xgb_features.py
=====================

Pre-compute the tabular XGBoost features ONCE, so the training runs never redo
the expensive rolling-window engineering.

For every target year Y and every site in the ERA5 daily pixelset file, this
writes one row per LAI observation day (36 dekads: days 5/15/25 of each month)
containing:

  • the current-day dynamic features (DYNAMIC_FEATURES, e.g. Tmin..PET)
  • backward rolling-window statistics of the same features over the site's own
    daily history (windows default 7/30/90 days, anchored ON the obs day and
    reaching into the previous year Y-1):
        temperatures / radiation / VPD : mean over the window
        tp_sum (precip) / PET          : sum over the window

The PFT fractions, the yearly CO2 scalar and the (optional) cyclic day-of-year
features are NOT stored here — they carry no daily history and are joined
cheaply at train time by `prediction.xgb_train`. Only the history-dependent
block (the part that made the reference script slow per run) is materialised.

Output (flat pixelset layout, like ERA5_daily_pixelset_{Y}.nc):
    {out_dir}/xgb_features_{Y}.nc   dims (obs=36, site)
        one variable per feature column (current-day + rolling)
        coords: site_id, obs, month, day, doy
        attrs : xgb_feature_columns (JSON list), rolling_windows (JSON list)

This script is torch-free (like create_stats_file.py / build_pixelset_targets.py)
so it runs in the h5-only preprocessing venv. Reads use engine="netcdf4"; the
write is isolated in a fresh subprocess (phenon/data/_ncwrite.py) to dodge the
netCDF4 / h5py dual-libhdf5 conflict.

Usage
-----
    python -m phenonn.data_creation.build_xgb_features \\
        --features_dir /data/sbarbu/PhenoNN/data/era5_10% \\
        --year_start 1991 --year_end 2019 \\
        --windows 7,30,90 \\
        --output_dir /data/sbarbu/PhenoNN/data/xgb_features_10%

(Give a --year_start one year EARLIER than your first training year so the
Y-1 history file exists for the rolling windows.)
"""

import argparse
import datetime
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import xarray as xr

from phenonn.utils.config import DYNAMIC_FEATURES, FEATURES_FNAME
from phenonn.data_creation._ncwrite import to_netcdf_subprocess


XGB_FNAME = "xgb_features_{year}.nc"

# (variable → reduction) for the backward rolling windows. Only DYNAMIC_FEATURES
# that appear here are rolled; the rest (e.g. cumulative gdd_*/cdd/ncd/botta_*
# when add_pheno_features is on) keep only their current-day value, since they
# already encode the season-to-date trajectory. Adapted to the ERA5 names:
#   temperatures / radiation / VPD → mean ; precip / PET → sum.
ROLLING_OPS: Dict[str, str] = {
    "Tmin": "mean",
    "Tmax": "mean",
    "Tmean": "mean",
    "ssrd_sum": "mean",
    "strd_sum": "mean",
    "Rn_tot": "mean",
    "VPD_max": "mean",
    "VPD_mean": "mean",
    "tp_sum": "sum",
    "PET": "sum",
}

DEFAULT_WINDOWS = [7, 30, 90]


# ── Obs-day calendar (leap-safe) ─────────────────────────────────────────────


def _obs_month_day() -> List[tuple]:
    """The 36 (month, day) dekad anchors: days 5/15/25 of each month."""
    return [(m, d) for m in range(1, 13) for d in (5, 15, 25)]


def _obs_doys(year: int) -> np.ndarray:
    """Day-of-year (1-based) of each obs anchor FOR THIS year — accounts for
    leap years (unlike the NN's fixed non-leap 2001 positions)."""
    return np.array(
        [datetime.date(year, m, d).timetuple().tm_yday for m, d in _obs_month_day()],
        dtype=np.int64,
    )


# ── Daily pixelset reading ───────────────────────────────────────────────────


def _read_daily(path: Path, sites: List[str]) -> Optional[Dict[str, np.ndarray]]:
    """Return {var: (T, n_site) float32} for `sites`, NaN for sites absent from
    the file. Reads each variable whole (one sequential pass) then numpy-indexes
    — no scattered isel/chunk thrashing. None if the file is missing."""
    if not path.exists():
        return None
    with xr.open_dataset(path, engine="netcdf4") as ds:
        file_sites = np.asarray(ds["site_id"].values).astype(str)
        idx_of = {s: i for i, s in enumerate(file_sites)}
        rows = np.array([idx_of.get(s, -1) for s in sites], dtype=np.int64)
        ok = rows >= 0
        n_t = ds.sizes["time"]
        out: Dict[str, np.ndarray] = {}
        for var in DYNAMIC_FEATURES:
            full = ds[var].transpose("time", "site").values.astype(np.float32)
            arr = np.full((n_t, len(sites)), np.nan, dtype=np.float32)
            arr[:, ok] = full[:, rows[ok]]
            out[var] = arr
            del full
    return out


def _rolling(
    V: np.ndarray, obs_pos: np.ndarray, windows: List[int], op: str
) -> Dict[int, np.ndarray]:
    """NaN-aware backward window stat at each obs position, for every window.

    V: (T, n_site). Uses a leading-zero cumulative sum so a window [a..b]
    (inclusive) is a single subtraction; the finite-count cumsum makes mean/sum
    ignore NaN days. Windows near the series start are naturally truncated."""
    finite = np.isfinite(V)
    Vz = np.where(finite, V, 0.0).astype(np.float64)
    T, n = V.shape
    C = np.zeros((T + 1, n), dtype=np.float64)
    np.cumsum(Vz, axis=0, out=C[1:])
    Cc = np.zeros((T + 1, n), dtype=np.float64)
    np.cumsum(finite.astype(np.float64), axis=0, out=Cc[1:])
    b = obs_pos
    res: Dict[int, np.ndarray] = {}
    for w in windows:
        a = np.maximum(0, obs_pos - w + 1)
        wsum = C[b + 1] - C[a]
        wcnt = Cc[b + 1] - Cc[a]
        if op == "mean":
            val = np.where(wcnt > 0, wsum / np.maximum(wcnt, 1.0), np.nan)
        else:  # sum
            val = np.where(wcnt > 0, wsum, np.nan)
        res[w] = val.astype(np.float32)
    return res


# ── Per-year build ───────────────────────────────────────────────────────────


def _process_year(
    year: int,
    features_dir: Path,
    out_dir: Path,
    windows: List[int],
    sites: Optional[List[str]],
    complevel: int,
) -> None:
    y_path = features_dir / FEATURES_FNAME.format(year=year)
    if not y_path.exists():
        print(f"  ✗ {year} skipped — missing {y_path.name}")
        return

    # Site list = the sites present in the Y file (optionally restricted).
    with xr.open_dataset(y_path, engine="netcdf4") as ds:
        file_sites = list(np.asarray(ds["site_id"].values).astype(str))
    site_list = [s for s in file_sites if s in set(sites)] if sites else file_sites
    if not site_list:
        print(f"  ✗ {year} skipped — no requested site in {y_path.name}")
        return

    dyn_y = _read_daily(y_path, site_list)
    dyn_ym1 = _read_daily(
        features_dir / FEATURES_FNAME.format(year=year - 1), site_list
    )

    if dyn_ym1 is None:
        # No previous year: prepend a 365-day NaN block so obs positions and
        # windows stay well-defined (windows into it are just NaN-ignored).
        t_ym1 = 365
        print(f"  [warn] {year}: no {year - 1} file — early-year windows truncated")
    else:
        t_ym1 = dyn_ym1[DYNAMIC_FEATURES[0]].shape[0]

    n_site = len(site_list)
    v_full: Dict[str, np.ndarray] = {}
    for var in DYNAMIC_FEATURES:
        ym1 = (
            dyn_ym1[var]
            if dyn_ym1 is not None
            else np.full((t_ym1, n_site), np.nan, dtype=np.float32)
        )
        v_full[var] = np.concatenate([ym1, dyn_y[var]], axis=0)  # (T_total, n_site)

    obs_pos = t_ym1 + (_obs_doys(year) - 1)  # (36,)
    md = _obs_month_day()
    months = np.array([m for m, _ in md], dtype=np.int32)
    days = np.array([d for _, d in md], dtype=np.int32)
    doys = _obs_doys(year).astype(np.int32)

    data_vars: Dict[str, tuple] = {}
    cols_order: List[str] = []
    # Current-day features
    for var in DYNAMIC_FEATURES:
        cur = v_full[var][obs_pos, :].astype(np.float32)  # (36, n_site)
        data_vars[var] = (("obs", "site"), cur)
        cols_order.append(var)
    # Rolling features
    for var in DYNAMIC_FEATURES:
        op = ROLLING_OPS.get(var)
        if op is None:
            continue
        roll = _rolling(v_full[var], obs_pos, windows, op)
        for w in windows:
            name = f"{var}_{op}{w}"
            data_vars[name] = (("obs", "site"), roll[w])
            cols_order.append(name)

    out = xr.Dataset(
        data_vars,
        coords={
            "obs": np.arange(len(md), dtype=np.int32),
            "month": ("obs", months),
            "day": ("obs", days),
            "doy": ("obs", doys),
            "site_id": ("site", np.asarray(site_list)),
        },
    )
    out.attrs["xgb_feature_columns"] = json.dumps(cols_order)
    out.attrs["rolling_windows"] = json.dumps(list(windows))
    out.attrs["source_year"] = int(year)

    enc = {
        c: {"zlib": True, "complevel": complevel, "_FillValue": np.float32(np.nan)}
        for c in cols_order
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    to_netcdf_subprocess(
        out, out_dir / XGB_FNAME.format(year=year), encoding=enc, engine="netcdf4"
    )
    print(f"  ✓ {year}  ({n_site:,} sites × 36 obs, {len(cols_order)} cols)")


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--features_dir", required=True, help="Folder of ERA5_daily_pixelset_{Y}.nc."
    )
    p.add_argument("--year_start", type=int, required=True)
    p.add_argument("--year_end", type=int, required=True)
    p.add_argument(
        "--windows",
        default="7,30,90",
        help="Comma-separated backward window lengths in days.",
    )
    p.add_argument(
        "--selected_pixels",
        default="",
        help="Optional selected_pixels.nc to restrict the site set "
        "(default: every site in each daily file).",
    )
    p.add_argument(
        "--output_dir", required=True, help="Where to write xgb_features_{Y}.nc."
    )
    p.add_argument("--complevel", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    features_dir = Path(args.features_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    windows = [int(w) for w in args.windows.split(",") if w.strip()]
    if not windows:
        raise ValueError("--windows must list at least one integer.")

    sites = None
    if args.selected_pixels:
        with xr.open_dataset(args.selected_pixels) as ds:
            sites = [str(s) for s in np.asarray(ds["site_id"].values)]

    print(f"Features dir : {features_dir}")
    print(f"Output dir   : {out_dir}")
    print(f"Years        : {args.year_start} → {args.year_end}")
    print(f"Windows      : {windows}")
    print(f"Rolled vars  : {[v for v in DYNAMIC_FEATURES if v in ROLLING_OPS]}")
    print(
        f"Sites        : {'all in file' if sites is None else f'{len(sites):,} selected'}"
    )

    for year in range(args.year_start, args.year_end + 1):
        try:
            _process_year(year, features_dir, out_dir, windows, sites, args.complevel)
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ {year} failed — {type(e).__name__}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
