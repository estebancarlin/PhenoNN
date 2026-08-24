#!/usr/bin/env python3
"""
build_daily_dataset_pixelset.py
================================

Same as build_daily_dataset_nosnow.py but produces a FLAT (time, site) cube
restricted to the pixels listed in `selected_pixels.nc`.

Pipeline position
-----------------
    compute_valid_pixels.py     →  valid_pixels_{Y}.nc
    select_pixels.py            →  selected_pixels.nc          ← input here
    THIS SCRIPT                 →  ERA5_daily_pixelset_{Y}.nc

Output schema
-------------
    dims:   time=365, site=N_selected
    vars:   Tmin, Tmax, Tmean, Tdew_mean, sp_mean,
            ssrd_sum, strd_sum, tp_sum,
            VPD_max, VPD_mean, Rn_mean, Rn_tot, PET
    coords: time, latitude(site), longitude(site), site_id(site),
            lat_idx(site), lon_idx(site)

Why this is faster than the global / bbox cube
----------------------------------------------
The on-disk NetCDF chunking of ERA5 is `(1, 1801, 3600)`, so reading a
sub-domain still loads the full global grid per hourly step. **But** with
vectorised pixel extraction (`isel(latitude=Xarr, longitude=Yarr)`) we
keep only the N_selected values per timestep in RAM. Downstream:
    - resample, VPD/Rn/PET work on tiny (time, site) arrays
    - compression + write is on ~N_selected × 365 × 13 values per year
      instead of N_lat × N_lon × 365 × 13.
The wall clock is mostly the disk read (unchanged) but RAM is ~100×
smaller and the write time becomes negligible.

Usage
-----
    python build_daily_dataset_pixelset.py \\
        --input_dir  /bdd/ERA5-Land/NETCDF/GLOBAL_01/hourly/FC_SF \\
        --output_dir /data/sbarbu/era5_pixelset \\
        --selected_pixels /data/sbarbu/selected_pixels.nc \\
        --year_start 1981 --year_end 2025
"""

import argparse
import gc
import time as time_mod
from pathlib import Path
from typing import Dict, List

import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar


HOURLY_VARS = ["t2m", "d2m", "sp", "ssrd", "strd", "tp"]


# ── Tetens saturation vapour pressure ───────────────────────────────────────


def _e_sat(T_celsius):
    return 0.6108 * np.exp(17.27 * T_celsius / (T_celsius + 237.3))


# ── Open + vectorised pixel extraction ──────────────────────────────────────


def _open_var_year(
    var: str, year: int, input_dir: Path, chunks: Dict[str, int],
    lat_idx_da: xr.DataArray, lon_idx_da: xr.DataArray,
) -> xr.DataArray:
    """
    Open the 12 monthly hourly files for one (var, year) pair and extract
    only the selected pixels via vectorised xarray indexing.

    `lat_idx_da` and `lon_idx_da` are DataArrays with dim 'site' so that
    the result has shape (time, site).
    """
    patterns = [f"{var}.{year}*.nc", f"{var}_{year}*.nc"]
    files: List[str] = []
    for pat in patterns:
        files.extend(str(p) for p in input_dir.rglob(pat))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(
            f"No files found for variable {var!r} year {year} under "
            f"{input_dir} (patterns: {patterns})"
        )

    ds = xr.open_mfdataset(files, combine="by_coords",
                           parallel=False, chunks=chunks)
    da = ds[var].sel(time=str(year))
    da = da.isel(latitude=lat_idx_da, longitude=lon_idx_da)
    return da


def _resample(da: xr.DataArray, how: str) -> xr.DataArray:
    return getattr(da.resample(time="1D"), how)()


def _resample_accum(da: xr.DataArray) -> xr.DataArray:
    """Daily total of an ERA5-Land accumulation (cumulative since 00 UTC).

    The hourly value holds the running total since the start of the day, and
    the full 24 h total is stamped at 00:00 of the NEXT day (the first hourly
    step is 01:00). Shift the time axis back 1 h so each 24 h total falls on
    its own calendar day, then take the last step = the full daily total.
    """
    shifted = da.assign_coords(time=da["time"] - np.timedelta64(1, "h"))
    # skipna=False: take the literal last hourly step of each day (the cumulative
    # endpoint). dask does not implement `last` with skipna=True; skipna=False is
    # also what we want here (ocean NaN stays NaN).
    return shifted.resample(time="1D").last(skipna=False)


def _build_daily_aggregations(
    year: int, input_dir: Path, chunks: Dict[str, int],
    lat_idx_da: xr.DataArray, lon_idx_da: xr.DataArray,
) -> xr.Dataset:
    daily = xr.Dataset()

    t2m = _open_var_year("t2m", year, input_dir, chunks, lat_idx_da, lon_idx_da)
    daily["Tmin"]  = _resample(t2m, "min")  - 273.15
    daily["Tmax"]  = _resample(t2m, "max")  - 273.15
    daily["Tmean"] = _resample(t2m, "mean") - 273.15

    d2m = _open_var_year("d2m", year, input_dir, chunks, lat_idx_da, lon_idx_da)
    daily["Tdew_mean"] = _resample(d2m, "mean") - 273.15

    sp = _open_var_year("sp", year, input_dir, chunks, lat_idx_da, lon_idx_da)
    daily["sp_mean"] = _resample(sp, "mean") / 100.0

    # ssrd / strd / tp are ERA5-Land accumulations reset at 00 UTC: the hourly
    # value already holds the running total since the start of the day, so the
    # daily total is the LAST step of the day (via _resample_accum, which also
    # shifts -1 h so the 24 h total lands on the right day), NOT the sum of the
    # 24 cumulative values (summing those over-counts ~5-12x). Units unchanged.
    ssrd = _open_var_year("ssrd", year, input_dir, chunks, lat_idx_da, lon_idx_da)
    daily["ssrd_sum"] = _resample_accum(ssrd) / 1e6

    strd = _open_var_year("strd", year, input_dir, chunks, lat_idx_da, lon_idx_da)
    daily["strd_sum"] = _resample_accum(strd) / 1e6

    tp = _open_var_year("tp", year, input_dir, chunks, lat_idx_da, lon_idx_da)
    daily["tp_sum"] = _resample_accum(tp) * 1000.0

    return daily


# ── Vectorised derived vars (identical to nosnow) ───────────────────────────


def _derive_vpd(daily: xr.Dataset) -> xr.Dataset:
    e_a = _e_sat(daily["Tdew_mean"])
    daily["VPD_max"]  = (_e_sat(daily["Tmax"])  - e_a).clip(min=0.0)
    daily["VPD_mean"] = (_e_sat(daily["Tmean"]) - e_a).clip(min=0.0)
    return daily


def _derive_radiation(daily: xr.Dataset) -> xr.Dataset:
    ssrd_avg = daily["ssrd_sum"] * 1e6 / 86400.0
    strd_avg = daily["strd_sum"] * 1e6 / 86400.0
    tmean_K  = daily["Tmean"] + 273.15
    rn_mean  = (1 - 0.23) * ssrd_avg + strd_avg - 0.97 * 5.67e-8 * tmean_K ** 4
    daily["Rn_mean"] = rn_mean
    daily["Rn_tot"]  = rn_mean * 0.0864
    return daily


def _derive_pet(daily: xr.Dataset) -> xr.Dataset:
    Tmean = daily["Tmean"]
    delta = 4098.0 * _e_sat(Tmean) / (Tmean + 237.3) ** 2
    P_kpa = daily["sp_mean"] / 10.0
    gamma = 0.000665 * P_kpa
    pet   = 1.26 * (delta / (delta + gamma)) * (daily["Rn_tot"] / 2.45)
    daily["PET"] = xr.where(daily["Rn_tot"] < 0, 0.0, pet)
    return daily


# ── Attributes / encoding ───────────────────────────────────────────────────


_ATTRS = {
    "Tmin":      ("Daily minimum 2 m air temperature",            "degC"),
    "Tmax":      ("Daily maximum 2 m air temperature",            "degC"),
    "Tmean":     ("Daily mean 2 m air temperature",               "degC"),
    "Tdew_mean": ("Daily mean 2 m dew-point temperature",         "degC"),
    "sp_mean":   ("Daily mean surface pressure",                  "hPa"),
    "ssrd_sum":  ("Daily downward short-wave radiation",          "MJ m-2"),
    "strd_sum":  ("Daily downward long-wave radiation",           "MJ m-2"),
    "tp_sum":    ("Daily total precipitation",                    "mm"),
    "VPD_max":   ("Daily max vapour pressure deficit (Tmax)",     "kPa"),
    "VPD_mean":  ("Daily mean vapour pressure deficit (Tmean)",   "kPa"),
    "Rn_mean":   ("Daily mean net radiation",                     "W m-2"),
    "Rn_tot":    ("Daily total net radiation",                    "MJ m-2 day-1"),
    "PET":       ("Priestley–Taylor potential evapotranspiration","mm day-1"),
}


def _set_attrs_and_encoding(
    daily: xr.Dataset, year: int, complevel: int = 4,
    site_chunk: int = 4096,
) -> dict:
    for name, (long_name, units) in _ATTRS.items():
        if name in daily:
            daily[name].attrs["long_name"] = long_name
            daily[name].attrs["units"]     = units

    daily["time"].encoding.update({
        "units":    f"days since {year}-01-01 00:00:00",
        "calendar": "proleptic_gregorian",
        "dtype":    "int32",
    })

    n_time = daily.sizes["time"]
    n_site = daily.sizes["site"]
    encoding = {}
    for name in daily.data_vars:
        encoding[name] = {
            "zlib":      True,
            "complevel": complevel,
            "dtype":     "float32",
            "_FillValue": np.float32(np.nan),
            "chunksizes": (n_time, min(site_chunk, n_site)),
        }
    return encoding


# ── Per-year orchestration ──────────────────────────────────────────────────


def process_year(
    year: int,
    input_dir: Path,
    output_dir: Path,
    lat_idx_da: xr.DataArray, lon_idx_da: xr.DataArray,
    selected_meta: xr.Dataset,
    time_chunk: int,
    lat_chunk: int,
    lon_chunk: int,
    complevel: int,
) -> Path:
    print(f"\n── Year {year} ─────────────────────────────")
    t0 = time_mod.time()

    chunks = {"time": time_chunk,
              "latitude":  lat_chunk,
              "longitude": lon_chunk}

    daily = _build_daily_aggregations(
        year, input_dir, chunks, lat_idx_da, lon_idx_da,
    )
    print(f"  built daily aggregations (lazy)  +{time_mod.time()-t0:5.1f}s")

    daily = _derive_vpd(daily)
    daily = _derive_radiation(daily)
    daily = _derive_pet(daily)
    print(f"  derived VPD / Rn / PET (lazy)    +{time_mod.time()-t0:5.1f}s")

    # Drop intermediate vars not used as features downstream:
    #   - Tdew_mean : only needed to compute VPD
    #   - sp_mean   : only needed to compute PET
    #   - Rn_mean   : only needed to compute Rn_tot
    daily = daily.drop_vars(
        [v for v in ("Tdew_mean", "sp_mean", "Rn_mean") if v in daily.data_vars]
    )

    # Only keep `site_id` as a coord on the 'site' dim.
    daily = daily.assign_coords({
        "site_id": ("site", selected_meta["site_id"].values),
    })

    encoding = _set_attrs_and_encoding(daily, year, complevel=complevel)
    daily.attrs["title"]  = (
        f"ERA5-Land daily aggregations + derived variables, "
        f"pixel subset — {year}"
    )
    daily.attrs["source"]      = "build_daily_dataset_pixelset.py"
    daily.attrs["n_selected"]  = int(daily.sizes["site"])

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"ERA5_daily_pixelset_{year}.nc"
    print(f"  writing {out_path.name} ...")
    with ProgressBar(dt=5.0):
        daily.to_netcdf(out_path, engine="netcdf4", encoding=encoding)

    daily.close()
    del daily
    gc.collect()

    print(f"  ✓ done in {time_mod.time()-t0:6.1f}s  →  {out_path}")
    return out_path


# ── CLI ─────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input_dir",  required=True,
                   help="Folder with per-variable monthly hourly NetCDFs.")
    p.add_argument("--output_dir", required=True,
                   help="Where to write ERA5_daily_pixelset_<year>.nc.")
    p.add_argument("--selected_pixels", required=True,
                   help="selected_pixels.nc produced by select_pixels.py.")
    p.add_argument("--year_start", type=int, default=1981)
    p.add_argument("--year_end",   type=int, default=2025)
    p.add_argument("--time_chunk", type=int, default=24)
    p.add_argument("--lat_chunk",  type=int, default=360)
    p.add_argument("--lon_chunk",  type=int, default=720)
    p.add_argument("--complevel",  type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    input_dir  = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    sel_path   = Path(args.selected_pixels).resolve()

    # ── Load selected pixels ──
    selected = xr.open_dataset(sel_path)
    n_site = selected.sizes["site"]
    print(f"Input dir   : {input_dir}")
    print(f"Output dir  : {output_dir}")
    print(f"Selected    : {sel_path.name}  ({n_site:,} pixels)")
    print(f"Years       : {args.year_start} → {args.year_end}")
    print(f"Time chunk  : {args.time_chunk} h")
    print(f"Hourly tile : {args.lat_chunk} × {args.lon_chunk}")
    print(f"Compression : zlib level {args.complevel}")

    # Wrap the int arrays as xarray DataArrays with dim 'site' so that
    # subsequent .isel(latitude=lat_da, longitude=lon_da) returns a
    # (time, site) array.
    #
    # Longitude convention fix: selected_pixels indices were computed on the
    # LAI/PFT grid whose longitude runs -180..180 (index 0 = -180 deg). ERA5-Land
    # longitude runs 0..360 (index 0 = 0 deg). Same 0.1 deg step / 3600 points,
    # but a 180 deg origin offset, so the raw lon index must be shifted by half
    # the grid before indexing ERA5 — otherwise every pixel is read 180 deg of
    # longitude away (wrong hemisphere, ocean -> NaN). Latitude shares the same
    # order on both grids and needs no shift.
    N_LON = 3600
    lon_idx_era5 = (selected["lon_idx"].values.astype(np.int64)
                    + N_LON // 2) % N_LON
    lat_idx_da = xr.DataArray(
        selected["lat_idx"].values.astype(np.int64), dims="site",
    )
    lon_idx_da = xr.DataArray(lon_idx_era5, dims="site")

    skipped: List[int] = []
    year_times: List[float] = []
    years_to_do = list(range(args.year_start, args.year_end + 1))
    n_total = len(years_to_do)

    for i, year in enumerate(years_to_do, start=1):
        t_year = time_mod.time()
        try:
            process_year(
                year, input_dir, output_dir,
                lat_idx_da, lon_idx_da, selected,
                time_chunk=args.time_chunk,
                lat_chunk=args.lat_chunk,
                lon_chunk=args.lon_chunk,
                complevel=args.complevel,
            )
            year_times.append(time_mod.time() - t_year)
        except FileNotFoundError as e:
            print(f"  ✗ {year} skipped — {e}")
            skipped.append(year)
        except Exception as e:                       # noqa: BLE001
            print(f"  ✗ {year} failed — {type(e).__name__}: {e}")
            skipped.append(year)

        n_done = i
        n_left = n_total - n_done
        if year_times and n_left > 0:
            avg = sum(year_times) / len(year_times)
            eta_s = avg * n_left
            mm, ss = divmod(int(eta_s), 60)
            hh, mm = divmod(mm, 60)
            print(f"  [progress] {n_done}/{n_total} years done  "
                  f"avg/year={avg:5.1f}s  ETA={hh:d}h{mm:02d}m{ss:02d}s")

    selected.close()
    print("\nDone.")
    if skipped:
        print(f"Skipped years: {skipped}")


if __name__ == "__main__":
    main()
