# phenocam/feature_engineering.py
#
# Derived ecological features computed from raw meteorological data.
# Called inside load_site() AFTER sorting by (year, doy) and BEFORE normalization.

import numpy as np
import pandas as pd


# ── Thresholds ───────────────────────────────────────────────────────────────

GDD_THRESHOLDS = [0, 5, 10]   # °C — multiple thresholds let the model pick
CHILLING_THRESHOLD = 5         # °C — standard for temperate deciduous

# ── Botta et al. (2000) onset parameters ─────────────────────────────────────

BOTTA_C1 = 964.0       # (-)
BOTTA_C2 = -0.0058     # (-) negative: more chilling → lower threshold
BOTTA_C3 = -12.8       # (-)


# ── Public API ───────────────────────────────────────────────────────────────


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add GDD, CDD, and Botta onset features to a site dataframe.

    Must be called AFTER sorting by (year, doy) and BEFORE normalization.
    All features reset to zero on January 1st of each year.

    Parameters
    ----------
    df : pd.DataFrame
        Site data with columns: year, doy, tmin, tmax (at minimum).

    Returns
    -------
    pd.DataFrame
        Same dataframe with added columns:
        - gdd_0, gdd_5, gdd_10 : Growing Degree Days at 0/5/10°C thresholds
        - cdd : Chilling Degree Days (cold accumulation below 5°C)
        - ncd : Number of Chilling Days (count of days where Tmean < 5°C)
        - botta_threshold : GDD threshold from Botta et al. (2000)
            G_thres = 964 * exp(-0.0058 * NCD) - 12.8
        - botta_forcing : GDD_0 / G_thres (onset proximity indicator)
            ~0 in winter, approaches 1 at onset, >1 during growing season
    """
    df = df.copy()

    # Mean daily temperature (fill isolated NaNs for cumsum stability)
    t_mean = ((df["tmin"] + df["tmax"]) / 2).ffill().bfill()

    # ── GDD at multiple thresholds (annual reset on Jan 1) ──
    for t_base in GDD_THRESHOLDS:
        col = f"gdd_{t_base}"
        daily_contrib = np.maximum(t_mean - t_base, 0)
        df[col] = daily_contrib.groupby(df["year"]).cumsum()

    # ── CDD — Chilling degree days (annual reset on Jan 1) ──
    daily_chill = np.maximum(CHILLING_THRESHOLD - t_mean, 0)
    df["cdd"] = daily_chill.groupby(df["year"]).cumsum()

    # ── NCD — Number of chilling days (binary count, annual reset) ──
    is_chill_day = (t_mean < CHILLING_THRESHOLD).astype(float)
    df["ncd"] = is_chill_day.groupby(df["year"]).cumsum()

    # ── Botta GDD threshold: G_thres = C1 * exp(C2 * NCD) + C3 ──
    # Decreases as chilling accumulates: more cold → easier onset
    df["botta_threshold"] = BOTTA_C1 * np.exp(BOTTA_C2 * df["ncd"]) + BOTTA_C3

    # ── Botta forcing ratio: GDD_0 / G_thres ──
    # Onset occurs when this crosses 1.0
    # Clip threshold to avoid division by zero or negative values
    safe_threshold = np.maximum(df["botta_threshold"], 1.0)
    df["botta_forcing"] = df["gdd_5"] / safe_threshold

    return df


# ── Feature names (import these in dataset.py) ──────────────────────────────

#DERIVED_FEATURES = ["gdd_0", "gdd_5", "gdd_10", "cdd", "ncd","botta_threshold", "botta_forcing"]

#DERIVED_LOG_FEATURES = {"gdd_0", "gdd_5", "gdd_10", "cdd", "ncd","botta_threshold"}