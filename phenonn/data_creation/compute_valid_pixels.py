#!/usr/bin/env python3
"""
compute_valid_pixels.py
=======================

Build the per-year mask of usable pixels. A pixel is kept when

    Σ_k PFT_k(year)   >  --pft_threshold      (= it is a land pixel)
    PFT1 (bare soil)  <  --max_bare_frac      (not mostly bare ground)
    n_valid_dekads    >= --min_valid_dekads   (default 26 = 26/36 ≈ 70 % obs)

where a dekad counts as valid when its LAI is finite (not THEIA nodata) AND its
RELATIVE RMSE  LAI_RMSE / max_year(LAI) < --max_rmse (default 0.3). Using the
ratio rather than the absolute RMSE avoids a set dominated by low-LAI pixels
(absolute RMSE grows with LAI). The LAI_RMSE band is produced by
preprocess_targets alongside LAI.

Output: {output_dir}/valid_pixels_{YYYY}.nc  with one int8 variable `valid`
on the (latitude, longitude) ERA5-Land 0.1° grid (same coords as the target
file).

The data loader (`phenonn.data.lai_dataset.LAIDataset`) uses this mask via
`list_valid_sites(valid_dir, years)` to draw its random sample pool — no
NaN pixels ever get fed to the model.

Usage
-----
    python -m phenonn.data_creation.compute_valid_pixels \\
        --target_dir /data/sbarbu/targets \\
        --pft_dir    /data/sbarbu/pft \\
        --output_dir /data/sbarbu/valid_pixels \\
        --year_start 1992 --year_end 2018 \\
        --pft_threshold 0.5 --max_bare_frac 0.5 \\
        --min_valid_dekads 26 --max_rmse 0.3
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import xarray as xr

from phenonn.utils.config import (
    N_DEKAD_YEAR,
    N_PFT,
    PFT_FNAME,
    TARGETS_FNAME,
    VALID_FNAME,
)
from phenonn.data_creation._ncwrite import to_netcdf_subprocess


def _load_pft_vector_array(ds: xr.Dataset) -> xr.DataArray:
    """
    Return a single DataArray of dims (pft, latitude, longitude) from a PFT
    file, supporting three layouts:

      1. ORCHIDEE PFTmap (native ESACCI / IPSL output):
         `maxvegetfrac(time_counter=1, veget=15, lat, lon)`
         → squeeze time_counter, rename veget → pft, lat/lon → latitude/longitude.

      2. 15 separate variables `pft1_frac..pft15_frac` on (lat/latitude, lon/longitude).

      3. One 3-D variable on (pft, lat/latitude, lon/longitude) — fallback
         for `pft_fraction`, `PFT`, `pft`, `pft_frac`, `fraction`.

    Inlined here (instead of importing from phenonn.data.lai_dataset) so this
    preprocessing step does not pull in torch.
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


def _process_year(
    year: int,
    target_dir: Path,
    pft_dir: Path,
    output_dir: Path,
    pft_threshold: float,
    max_bare_frac: float,
    min_valid_dekads: int,
    max_rmse: float,
    complevel: int,
) -> None:
    tgt_path = target_dir / TARGETS_FNAME.format(year=year)
    pft_path = pft_dir / PFT_FNAME.format(year=year)
    if not tgt_path.exists():
        print(f"  ✗ {year} skipped — missing {tgt_path.name}")
        return
    if not pft_path.exists():
        print(f"  ✗ {year} skipped — missing {pft_path.name}")
        return

    # Read EVERYTHING with the h5py family (h5netcdf) in this process. The
    # ORCHIDEE PFTmap can only be opened by h5netcdf, and on the cluster venv
    # the netcdf4 and h5py wheels bundle separate libhdf5 that corrupt each
    # other when both load in one process (here: netcdf4 read of the target
    # would then make the h5netcdf PFT read fail in H5DSget_num_scales).
    # The actual NetCDF write is already isolated in a fresh subprocess
    # (to_netcdf_subprocess), so the parent stays netcdf4-free. See
    # phenon/data/_ncwrite.py.
    tgt = xr.open_dataset(tgt_path, engine="h5netcdf")
    pft_ds = xr.open_dataset(pft_path, engine="h5netcdf", decode_times=False)
    pft = _load_pft_vector_array(pft_ds)

    if "LAI_RMSE" not in tgt:
        raise KeyError(
            f"{tgt_path.name} has no LAI_RMSE variable — rerun preprocess_targets "
            f"(it now writes the RMSE band required by the dekad criterion)."
        )

    # Land pixels (Σ PFT > threshold) and bare-soil fraction (PFT1). Handle the
    # case where the PFT file is on coords slightly different from the target's
    # grid (reindex both with nearest).
    pft_sum = pft.sum(dim="pft")
    bare = pft.isel(pft=0)  # PFT1 = bare soil (ORCHIDEE ordering)
    if pft_sum.sizes.get("latitude") != tgt.sizes.get("latitude") or pft_sum.sizes.get(
        "longitude"
    ) != tgt.sizes.get("longitude"):
        rk = dict(
            latitude=tgt["latitude"],
            longitude=tgt["longitude"],
            method="nearest",
            tolerance=0.06,
        )
        pft_sum = pft_sum.reindex(**rk)
        bare = bare.reindex(**rk)
    land_mask = pft_sum > pft_threshold
    bare_ok = bare < max_bare_frac

    # A dekad is valid when LAI is finite (not THEIA nodata) AND its retrieval
    # RMSE is small RELATIVE to the pixel's yearly peak LAI: LAI_RMSE / max(LAI)
    # < max_rmse. Using the ratio (not the absolute RMSE) avoids a set dominated
    # by low-LAI pixels — absolute RMSE grows with LAI, so a fixed cutoff would
    # reject high-LAI pixels and admit only sparse ones.
    max_lai = tgt["LAI"].max(dim="dekad")  # per-pixel yearly peak LAI
    rel_rmse = tgt["LAI_RMSE"] / max_lai  # inf/NaN where peak≈0 → dropped
    dekad_ok = tgt["LAI"].notnull() & (rel_rmse < max_rmse)
    valid_count = dekad_ok.sum(dim="dekad")
    enough_obs = valid_count >= min_valid_dekads

    valid = (land_mask & bare_ok & enough_obs).astype("int8").rename("valid")
    valid.attrs.update(
        {
            "long_name": "Usable pixel for the LAI training set",
            "description": (
                "True when Σ PFT > pft_threshold AND PFT1 "
                "(bare soil) < max_bare_frac AND at least "
                "min_valid_dekads dekads have finite LAI with "
                "LAI_RMSE / max_year(LAI) < max_rmse."
            ),
            "pft_threshold": float(pft_threshold),
            "max_bare_frac": float(max_bare_frac),
            "min_valid_dekads": int(min_valid_dekads),
            "max_rmse": float(max_rmse),
        }
    )

    out = valid.to_dataset()
    out["valid_count"] = valid_count.astype("int16")
    out["valid_count"].attrs["long_name"] = (
        "Number of dekads with finite LAI "
        "and LAI_RMSE/max(LAI) < max_rmse "
        "(0..36)"
    )
    out["pft_sum"] = pft_sum.astype("float32")
    out["pft_sum"].attrs["long_name"] = "Sum of PFT fractions per pixel"
    out["bare_frac"] = bare.astype("float32")
    out["bare_frac"].attrs["long_name"] = "PFT1 (bare soil) fraction per pixel"

    encoding = {
        "valid": {"zlib": True, "complevel": complevel, "dtype": "int8"},
        "valid_count": {"zlib": True, "complevel": complevel, "dtype": "int16"},
        "pft_sum": {
            "zlib": True,
            "complevel": complevel,
            "dtype": "float32",
            "_FillValue": np.float32(np.nan),
        },
        "bare_frac": {
            "zlib": True,
            "complevel": complevel,
            "dtype": "float32",
            "_FillValue": np.float32(np.nan),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / VALID_FNAME.format(year=year)
    # Reads PFT via h5netcdf (h5py) earlier in this process, so a same-process
    # netcdf4 write trips "NetCDF: HDF error" on the cluster venv. Write from a
    # fresh h5py-free subprocess. See phenon/data/_ncwrite.py.
    to_netcdf_subprocess(out, out_path, encoding=encoding, engine="netcdf4")
    n_valid = int(valid.sum())
    n_total = int(valid.size)
    print(
        f"  ✓ {out_path.name}  {n_valid:,} / {n_total:,} "
        f"({100 * n_valid / max(n_total, 1):.1f} %) valid pixels"
    )
    tgt.close()
    pft_ds.close()


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--target_dir", required=True)
    p.add_argument("--pft_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--year_start", type=int, default=1981)
    p.add_argument("--year_end", type=int, default=2025)
    p.add_argument(
        "--pft_threshold",
        type=float,
        default=0.5,
        help="Minimum Σ PFT to call the pixel 'land' (default 0.5).",
    )
    p.add_argument(
        "--max_bare_frac",
        type=float,
        default=0.5,
        help="Maximum PFT1 (bare soil) fraction to keep a pixel " "(default 0.5).",
    )
    p.add_argument(
        "--min_valid_dekads",
        type=int,
        default=26,
        help="Minimum valid dekads per year (default 26 ≈ 72 %).",
    )
    p.add_argument(
        "--max_rmse",
        type=float,
        default=0.3,
        help="Max RELATIVE RMSE (LAI_RMSE / yearly max LAI) for a "
        "dekad to count as valid (default 0.3).",
    )
    p.add_argument("--complevel", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    target_dir = Path(args.target_dir).resolve()
    pft_dir = Path(args.pft_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    print(f"Target dir : {target_dir}")
    print(f"PFT dir    : {pft_dir}")
    print(f"Output dir : {output_dir}")
    print(f"Years      : {args.year_start} → {args.year_end}")
    print(
        f"Criteria   : Σ PFT > {args.pft_threshold}  AND  "
        f"bare(PFT1) < {args.max_bare_frac}  AND  "
        f"n_valid ≥ {args.min_valid_dekads}/{N_DEKAD_YEAR}  "
        f"(dekad valid if LAI finite & RMSE/max(LAI) < {args.max_rmse})"
    )

    skipped: List[int] = []
    for year in range(args.year_start, args.year_end + 1):
        try:
            _process_year(
                year,
                target_dir,
                pft_dir,
                output_dir,
                args.pft_threshold,
                args.max_bare_frac,
                args.min_valid_dekads,
                args.max_rmse,
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
