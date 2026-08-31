"""
Derived ecological features for the daily ERA5-Land feature cube.

Reproduces the formulas of LaiNN/phenocam/feature_engineering.py but adapts
them to xarray Datasets indexed by (time, lat, lon) instead of per-site
pandas DataFrames. All accumulators reset on January 1st of each year so
they are byte-for-byte compatible with what main_big.py used to compute.

Variables added
---------------
  gdd_0, gdd_5, gdd_10  Growing Degree Days at 0/5/10 °C thresholds
  cdd                  Chilling Degree Days  (cold accumulation below 5 °C)
  ncd                  Number of Chilling Days (binary count, Tmean < 5 °C)
  botta_threshold      Botta et al. (2000) GDD threshold
                       G_thres = 964 · exp(-0.0058 · NCD) - 12.8
  botta_forcing        GDD_5 / max(G_thres, 1)  ∈ ~[0, > 1]
"""

import numpy as np
import xarray as xr


GDD_THRESHOLDS = [0, 5, 10]
CHILLING_THRESHOLD = 5

BOTTA_C1 = 964.0
BOTTA_C2 = -0.0058
BOTTA_C3 = -12.8


def _yearly_cumsum(da: xr.DataArray) -> xr.DataArray:
    """
    Cumulative sum along `time`, **reset on January 1st of each year**.

    Implemented via xarray.groupby("time.year").cumsum() — exact same semantic
    as `df.groupby("year").cumsum()` used in the per-site pandas version.
    """
    return da.groupby("time.year").cumsum(dim="time")


def add_derived_features(daily: xr.Dataset) -> xr.Dataset:
    """
    Add GDD/CDD/NCD/Botta variables to a daily ERA5-Land Dataset.

    Requires `Tmin`, `Tmax` (in °C). Computes the daily mean internally so
    upstream code doesn't have to expose `Tmean` separately, but reuses it
    if already present.
    """
    if "Tmean" in daily:
        Tmean = daily["Tmean"]
    else:
        Tmean = (daily["Tmin"] + daily["Tmax"]) / 2.0

    Tmean = Tmean.fillna(Tmean.mean(dim="time"))  # very light fill for cumsum stability

    out = daily

    # GDD at multiple thresholds
    for tb in GDD_THRESHOLDS:
        daily_contrib = xr.where(Tmean - tb > 0, Tmean - tb, 0.0)
        out[f"gdd_{tb}"] = _yearly_cumsum(daily_contrib).astype("float32")

    # CDD — chilling degree days
    daily_chill = xr.where(
        CHILLING_THRESHOLD - Tmean > 0, CHILLING_THRESHOLD - Tmean, 0.0
    )
    out["cdd"] = _yearly_cumsum(daily_chill).astype("float32")

    # NCD — count of chilling days
    is_chill = (Tmean < CHILLING_THRESHOLD).astype("float32")
    out["ncd"] = _yearly_cumsum(is_chill).astype("float32")

    # Botta threshold and forcing
    out["botta_threshold"] = (
        BOTTA_C1 * np.exp(BOTTA_C2 * out["ncd"]) + BOTTA_C3
    ).astype("float32")
    safe_thresh = xr.where(out["botta_threshold"] > 1.0, out["botta_threshold"], 1.0)
    out["botta_forcing"] = (out["gdd_5"] / safe_thresh).astype("float32")

    return out


DERIVED_FEATURES = [
    "gdd_0",
    "gdd_5",
    "gdd_10",
    "cdd",
    "ncd",
    "botta_threshold",
    "botta_forcing",
]
