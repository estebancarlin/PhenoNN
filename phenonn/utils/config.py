"""
Central constants for the PhenoNN training and inference pipeline.

This module is the single source of truth for:
  - the list and ORDERING of features expected at every layer (Dataset →
    model → wrappers → checkpoint)
  - the PFT count and labels (ORCHIDEE 15-PFT convention)
  - dekadal calendar constants (36 obs/year on days 5/15/25)

Editing the toggles `add_pheno_features`, `add_cyclic_features` and
`add_co2_features` updates the feature count globally. Models built before
toggling and resumed afterwards will fail with a clear input-channel
mismatch, which is the desired behaviour.
"""

from typing import List

# ── Feature toggles ──────────────────────────────────────────────────────────

add_pheno_features = False  # GDD/CDD/NCD/Botta — added by preprocess_features.py
add_cyclic_features = False  # doy_sin / doy_cos — computed at sample time
add_co2_features = True  # one CO2 value per year broadcast over the window


# ── Feature lists ────────────────────────────────────────────────────────────

# Variables stored in ERA5_daily_pixelset_{Y}.nc (10 features).
# Intermediate vars Tdew_mean / sp_mean / Rn_mean are dropped at write time
# in build_daily_dataset_pixelset.py — they exist only to feed VPD / PET / Rn_tot.
METEO_BASE: List[str] = [
    "Tmin",
    "Tmax",
    "Tmean",
    "ssrd_sum",
    "strd_sum",
    "tp_sum",
    "VPD_max",
    "VPD_mean",
    "Rn_tot",
    "PET",
]

DERIVED_FEATURES: List[str] = [
    "gdd_0",
    "gdd_5",
    "gdd_10",
    "cdd",
    "ncd",
    "botta_threshold",
    "botta_forcing",
]

DYNAMIC_FEATURES: List[str] = METEO_BASE + (
    DERIVED_FEATURES if add_pheno_features else []
)

CYCLIC_FEATURES: List[str] = ["doy_sin", "doy_cos"] if add_cyclic_features else []

CO2_FEATURES: List[str] = ["co2"] if add_co2_features else []

PFT_COLS: List[str] = [f"pft{k}_frac" for k in range(1, 16)]
N_PFT = 15

ALL_FEATURES: List[str] = DYNAMIC_FEATURES + CYCLIC_FEATURES + CO2_FEATURES + PFT_COLS
FEATURE_CHANNELS = len(ALL_FEATURES)

# Index where the PFT block starts in the canonical feature tensor.
PFT_START = len(DYNAMIC_FEATURES) + len(CYCLIC_FEATURES) + len(CO2_FEATURES)

# Heavy-tailed variables that benefit from log1p before z-scoring. Match the
# choices of LaiNN/dataset_target_feature.py adapted to the ERA5 names.
LOG_TRANSFORM_FEATURES = {
    "tp_sum",
    "ssrd_sum",
    "strd_sum",
    "VPD_max",
    "VPD_mean",
    "gdd_0",
    "gdd_5",
    "gdd_10",
    "cdd",
    "ncd",
}


# ── Dekadal calendar ─────────────────────────────────────────────────────────

DEKAD_DAYS = (5, 15, 25)
N_DEKAD_YEAR = 36
SEQ_LENGTH_DEFAULT = 720


# ── PFT labels (ORCHIDEE 15-PFT) ─────────────────────────────────────────────

PFT_NAMES: List[str] = [
    "bare soil",
    "trop BL evergreen",
    "trop BL raingreen",
    "temp NL evergreen",
    "temp BL evergreen",
    "temp BL summergreen",
    "bor NL evergreen",
    "bor BL summergreen",
    "bor NL summergreen",
    "temp C3 grass",
    "C4 grass",
    "C3 agriculture",
    "C4 agriculture",
    "trop C3 grass",
    "bor C3 grass",
]


# ── File-naming conventions used by the preprocessing pipeline ───────────────

FEATURES_FNAME = "ERA5_daily_pixelset_{year}.nc"
TARGETS_FNAME = "LAI_dekadal_{year}.nc"
PFT_FNAME = "PFTmap_{year}.nc"
VALID_FNAME = "valid_pixels_{year}.nc"
SELECTED_PIXELS_FNAME = "selected_pixels.nc"
