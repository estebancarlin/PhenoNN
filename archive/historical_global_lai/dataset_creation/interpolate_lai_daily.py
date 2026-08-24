#!/usr/bin/env python3
"""
interpolate_lai_daily.py
========================

Turn the 36 dekadal LAI observations of each year into a smooth DAILY LAI curve
(365 days) by fitting a **double-logistic** phenology model per site and year,
then evaluating it on days 1..365.

Why a double-logistic and not a plain interpolation
---------------------------------------------------
The dekads sit on days 5/15/25 of each month (irregular spacing, ~10-11 days
apart) and often carry NaN where the source LAI was cloudy/missing. A double
logistic (Beck 2006 / Zhang 2003) is the standard seasonal model:

    f(t) = base + amp * [ sigmoid(mS*(t - S)) - sigmoid(mA*(t - A)) ]

    base : winter / dormant LAI floor
    amp  : green-season amplitude
    S    : spring inflection (green-up)      A = S + dur  (dur > 0)
    A    : autumn inflection (senescence)
    mS,mA: green-up / senescence rates

It does NOT pass through the 36 points exactly — it *smooths* them and fills the
gaps with a physically-shaped curve. It assumes ONE growing season per calendar
year (a single green-up then senescence). Sites that violate that — double
cropping, evergreen with no clear cycle, or Southern-Hemisphere seasons that
straddle 1 January — will fit poorly or fail to converge; those are written as
NaN and counted, never silently replaced by another method.

Input  (pixelset LAI, from build_pixelset_targets.py):
    {target_dir}/LAI_dekadal_{Y}.nc     LAI(dekad=36, site)

Output (same flat pixelset layout, new day axis):
    {out_dir}/LAI_daily_{Y}.nc          LAI(day=365, site)
        coord day = DOY 1..365 ; site coords copied from the input file.

Torch-free (runs in the h5-only preprocessing venv); needs scipy. Reads use
engine="netcdf4"; the write is isolated in a subprocess (phenon/data/_ncwrite.py)
to dodge the netCDF4 / h5py dual-libhdf5 conflict.

Usage
-----
    python -m phenon.dataset_creation.interpolate_lai_daily \\
        --target_dir /data/sbarbu/PhenoNN/data/LAI_pixelset \\
        --out_dir    /data/sbarbu/PhenoNN/data/LAI_daily \\
        --year_start 1999 --year_end 2019
"""

import argparse
import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import xarray as xr

try:
    from scipy.optimize import curve_fit
except ImportError as e:                                    # noqa: BLE001
    raise SystemExit(
        "interpolate_lai_daily needs scipy — `pip install scipy` in the venv."
    ) from e

from phenon.training.config import N_DEKAD_YEAR, TARGETS_FNAME
from phenon.dataset_creation._ncwrite import to_netcdf_subprocess


DAILY_FNAME = "LAI_daily_{year}.nc"
N_DAYS = 365
MIN_VALID = 8          # need > 6 params; keep a margin for a stable fit
_SITE_COORDS = ("site_id", "lat_idx", "lon_idx", "latitude", "longitude")


# ── Dekad calendar (leap-safe) ───────────────────────────────────────────────


def _dekad_doys(year: int) -> np.ndarray:
    """DOY (1-based) of the 36 dekad anchors (days 5/15/25 of each month) for
    THIS year, accounting for leap years."""
    return np.array(
        [datetime.date(year, m, d).timetuple().tm_yday
         for m in range(1, 13) for d in (5, 15, 25)],
        dtype=np.float64,
    )


# ── Double-logistic model + per-site fit ─────────────────────────────────────


def _double_logistic(t, base, amp, mS, S, mA, dur):
    """Beck-style double logistic. A = S + dur (dur > 0) enforces a green-up
    then a senescence (never the reverse)."""
    A = S + dur
    up = 1.0 / (1.0 + np.exp(-mS * (t - S)))
    down = 1.0 / (1.0 + np.exp(-mA * (t - A)))
    return base + amp * (up - down)


def _fit_site(doy: np.ndarray, y: np.ndarray) -> Optional[np.ndarray]:
    """Fit the double logistic to one site's finite dekads. Returns the 6
    parameters, or None if too few points / no convergence."""
    finite = np.isfinite(y)
    if int(finite.sum()) < MIN_VALID:
        return None
    t = doy[finite]
    v = y[finite]

    vmin, vmax = float(v.min()), float(v.max())
    amp0 = max(vmax - vmin, 1e-3)
    # Crude S/A guess: first/last dekad above the half-amplitude level.
    half = vmin + 0.5 * amp0
    above = t[v >= half]
    if above.size:
        S0, A0 = float(above.min()), float(above.max())
    else:
        S0, A0 = 100.0, 260.0
    dur0 = A0 - S0 if A0 - S0 > 10.0 else 120.0

    p0 = [vmin, amp0, 0.1, S0, 0.1, dur0]
    lower = [0.0, 0.0, 1e-3, 1.0, 1e-3, 1.0]
    upper = [vmax, 3.0 * amp0, 1.0, 365.0, 1.0, 600.0]
    # Keep p0 strictly inside the box (curve_fit rejects boundary starts).
    p0 = [min(max(pi, lo + 1e-6), hi - 1e-6)
          for pi, lo, hi in zip(p0, lower, upper)]
    try:
        popt, _ = curve_fit(_double_logistic, t, v, p0=p0,
                            bounds=(lower, upper), maxfev=10000)
    except (RuntimeError, ValueError):
        return None
    return popt


# ── Per-year build ───────────────────────────────────────────────────────────


def _process_year(year: int, target_dir: Path, out_dir: Path,
                  complevel: int) -> None:
    tpath = target_dir / TARGETS_FNAME.format(year=year)
    if not tpath.exists():
        print(f"  ✗ {year} skipped — missing {tpath.name}")
        return

    with xr.open_dataset(tpath, engine="netcdf4") as ds:
        lai = ds["LAI"].transpose("dekad", "site").values.astype(np.float64)
        if lai.shape[0] != N_DEKAD_YEAR:
            print(f"  ✗ {year} skipped — {lai.shape[0]} dekads, expected 36")
            return
        site_coords = {c: (("site",), np.asarray(ds[c].values))
                       for c in _SITE_COORDS if c in ds.coords}
        lai_attrs = dict(ds["LAI"].attrs)

    doy = _dekad_doys(year)                       # (36,)
    days = np.arange(1, N_DAYS + 1, dtype=np.float64)
    n_site = lai.shape[1]
    out = np.full((N_DAYS, n_site), np.nan, dtype=np.float32)

    n_all_nan = 0
    n_failed = 0
    for j in range(n_site):
        col = lai[:, j]
        if not np.isfinite(col).any():
            n_all_nan += 1
            continue
        popt = _fit_site(doy, col)
        if popt is None:
            n_failed += 1
            continue
        curve = _double_logistic(days, *popt)
        out[:, j] = np.clip(curve, 0.0, None).astype(np.float32)

    ds_out = xr.Dataset(
        {"LAI": (("day", "site"), out)},
        coords={"day": np.arange(1, N_DAYS + 1, dtype=np.int32), **site_coords},
    )
    ds_out["LAI"].attrs.update(lai_attrs)
    ds_out["LAI"].attrs["interpolation"] = "double_logistic"
    ds_out.attrs["source_dekadal_file"] = tpath.name
    ds_out.attrs["n_sites"] = int(n_site)
    ds_out.attrs["n_sites_all_nan_input"] = int(n_all_nan)
    ds_out.attrs["n_sites_fit_failed"] = int(n_failed)

    enc = {"LAI": {"zlib": True, "complevel": complevel,
                   "_FillValue": np.float32(np.nan)}}
    out_dir.mkdir(parents=True, exist_ok=True)
    to_netcdf_subprocess(ds_out, out_dir / DAILY_FNAME.format(year=year),
                         encoding=enc, engine="netcdf4")
    n_ok = n_site - n_all_nan - n_failed
    print(f"  ✓ {year}  ({n_ok:,}/{n_site:,} sites fit, "
          f"{n_failed} no-converge, {n_all_nan} empty)")


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--target_dir", required=True,
                   help="Folder of pixelset LAI_dekadal_{Y}.nc.")
    p.add_argument("--out_dir", required=True,
                   help="Where to write LAI_daily_{Y}.nc.")
    p.add_argument("--year_start", type=int, required=True)
    p.add_argument("--year_end",   type=int, required=True)
    p.add_argument("--complevel",  type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    target_dir = Path(args.target_dir).resolve()
    out_dir    = Path(args.out_dir).resolve()

    print(f"Pixelset LAI in : {target_dir}")
    print(f"Daily LAI out   : {out_dir}")
    print(f"Years           : {args.year_start} → {args.year_end}")
    print(f"Model           : double logistic, {N_DAYS}-day output\n")

    for year in range(args.year_start, args.year_end + 1):
        try:
            _process_year(year, target_dir, out_dir, args.complevel)
        except Exception as e:                            # noqa: BLE001
            print(f"  ✗ {year} failed — {type(e).__name__}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
