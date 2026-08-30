#!/usr/bin/env python3
"""
add_soil_moisture_proxy.py
==========================

Add a Soil-Moisture-Index proxy column (``SMI``) to the ERA5 pixelset files.

For every day ``t`` the proxy is a causal, linearly-decaying weighted sum of the
last 30 days of precipitation (``tp_sum``):

    SMI_t = ( Σ_{i=0}^{30}  P_{t-i} · (30 - i) / 30 ) / 31

i.e. today's rain has weight 1, the rain 30 days ago has weight 0, and the whole
sum is divided by 31 (31 terms, i = 0 … 30).

Edge (start of year)
--------------------
Each yearly file is processed on its own; there is no previous-year rain to fill
the warm-up. The first ~30 days therefore use a TRUNCATED window (missing prior
days treated as 0) while keeping the fixed ``/31`` of the formula, so the proxy
ramps up over the first month. This is a plain causal convolution with
zero-padding.

Input  (ERA5 pixelset, from build_daily_dataset_pixelset.py):
    {era5_dir}/ERA5_daily_pixelset_{Y}.nc     tp_sum(time=365, site), …

Output (same file layout, one extra variable):
    {out_dir}/ERA5_daily_pixelset_{Y}.nc      … + SMI(time, site)
        --out_dir may equal --era5_dir to overwrite in place.

Torch-free (runs in the h5-only preprocessing venv). Reads use engine="netcdf4";
the write is isolated in a subprocess (phenon/data/_ncwrite.py) to dodge the
netCDF4 / h5py dual-libhdf5 conflict.

Usage
-----
    python -m phenonn.data_creation.add_soil_moisture_proxy \\
        --era5_dir /data/sbarbu/PhenoNN/data/ERA5_pixelset \\
        --out_dir  /data/sbarbu/PhenoNN/data/ERA5_pixelset_smi \\
        --year_start 1999 --year_end 2019
"""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from phenonn.utils.config import FEATURES_FNAME
from phenonn.data_creation._ncwrite import to_netcdf_subprocess


PRECIP_VAR = "tp_sum"
SMI_VAR = "SMI"
WINDOW = 30  # look-back length; 31 terms (i = 0 … WINDOW)


def _weights() -> np.ndarray:
    """Causal weights w[i] = (WINDOW - i) / WINDOW for i = 0 … WINDOW."""
    i = np.arange(WINDOW + 1, dtype=np.float64)
    return (WINDOW - i) / WINDOW


def _soil_moisture_index(precip: np.ndarray) -> np.ndarray:
    """SMI for a (time, site) precipitation array, zero-padded warm-up.

    out[t] = ( Σ_{i=0}^{WINDOW} w[i]·P[t-i] ) / (WINDOW + 1)
    """
    w = _weights()
    n_time = precip.shape[0]
    out = np.zeros_like(precip, dtype=np.float64)
    for i, wi in enumerate(w):
        # P[t-i] feeds out[t] for t >= i (earlier days are the missing warm-up).
        out[i:] += wi * precip[: n_time - i]
    out /= WINDOW + 1
    return out


# ── Per-year build ───────────────────────────────────────────────────────────


def _process_year(year: int, era5_dir: Path, out_dir: Path, complevel: int) -> None:
    fpath = era5_dir / FEATURES_FNAME.format(year=year)
    if not fpath.exists():
        print(f"  ✗ {year} skipped — missing {fpath.name}")
        return

    with xr.open_dataset(fpath, engine="netcdf4") as ds:
        ds = ds.load()

    if PRECIP_VAR not in ds:
        print(f"  ✗ {year} skipped — no '{PRECIP_VAR}' in {fpath.name}")
        return

    precip = ds[PRECIP_VAR].transpose("time", "site").values.astype(np.float64)
    smi = _soil_moisture_index(precip).astype(np.float32)

    ds[SMI_VAR] = (("time", "site"), smi)
    ds[SMI_VAR].attrs.update(
        {
            "long_name": "soil moisture proxy (30-day linearly-weighted precip)",
            "units": ds[PRECIP_VAR].attrs.get("units", ""),
            "formula": "SMI_t = sum_{i=0}^{30} tp_sum[t-i]*(30-i)/30 / 31",
            "window_days": WINDOW,
            "warmup": "truncated window (missing prior days = 0), fixed /31",
        }
    )

    # Fresh encoding for every data var (drop the source's filter hints to avoid
    # conflicts through the subprocess pickle) but PRESERVE the source chunking.
    # The ERA5 pixelset is deliberately chunked (full time, block of sites) so
    # RamLAIDataset's per-site read — all dates for a scattered subset of sites —
    # hits few chunks. Letting netCDF fall back to its default chunking splits
    # the time axis and inflates every chunk, blowing up the dataset build time.
    smi_chunks = ds[PRECIP_VAR].encoding.get("chunksizes")  # read before clearing
    encoding = {}
    for name, var in ds.data_vars.items():
        orig_chunks = var.encoding.get("chunksizes")
        var.encoding = {}
        enc = {"zlib": True, "complevel": complevel}
        if orig_chunks is not None:
            enc["chunksizes"] = tuple(orig_chunks)
        encoding[name] = enc
    if smi_chunks is not None:  # SMI is a fresh var
        encoding[SMI_VAR]["chunksizes"] = tuple(smi_chunks)
    encoding[SMI_VAR]["_FillValue"] = np.float32(np.nan)

    out_dir.mkdir(parents=True, exist_ok=True)
    to_netcdf_subprocess(
        ds,
        out_dir / FEATURES_FNAME.format(year=year),
        encoding=encoding,
        engine="netcdf4",
    )
    print(f"  ✓ {year}  (+{SMI_VAR}, {smi.shape[1]:,} sites)")


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--era5_dir", required=True, help="Folder of ERA5_daily_pixelset_{Y}.nc."
    )
    p.add_argument(
        "--out_dir",
        required=True,
        help="Where to write the augmented files "
        "(may equal --era5_dir to overwrite in place).",
    )
    p.add_argument("--year_start", type=int, required=True)
    p.add_argument("--year_end", type=int, required=True)
    p.add_argument("--complevel", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    era5_dir = Path(args.era5_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    print(f"ERA5 pixelset in : {era5_dir}")
    print(f"Augmented out    : {out_dir}")
    print(f"Years            : {args.year_start} → {args.year_end}")
    print(f"Proxy            : {WINDOW}-day linearly-weighted {PRECIP_VAR}\n")

    for year in range(args.year_start, args.year_end + 1):
        try:
            _process_year(year, era5_dir, out_dir, args.complevel)
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ {year} failed — {type(e).__name__}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
