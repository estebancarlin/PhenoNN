# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Stefan Barbu, Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""
PhenoNN Data Module

Provides dataset classes and data processing utilities for phenology prediction:
- PhenoCamDataset: Per-site CSV format (original)
- LAIDataset: Flat CSV format (features + targets)
- Feature engineering: GDD, CDD, Botta onset features

"""

# Dataset classes
from .dataset import (
    PhenoCamDataset,
    load_site,
    extract_pft_and_site,
    compute_norm_stats as _compute_norm_stats_deprecated,  # Legacy, use normalization.py
    load_norm_stats as _load_norm_stats_deprecated,
    split_sites_by_fraction,
    load_lai_norms,
    # Feature constants
    DYNAMIC_FEATURES,
    CYCLIC_FEATURES,
    STATIC_FEATURES,
    LOG_TRANSFORM_FEATURES,
)

from .dataset_flat import (
    LAIDataset,
    get_site_ids,
    split_sites_by_fraction as split_sites_flat,
    ALL_FEATURES,
    PFT_COLS,
    N_OBS_PER_YEAR,
    TARGET_DAYS_OF_MONTH,
)

from .dataset_big import (
    BigLAIDataset,
    generate_site_ids_from_range,
    get_pixel_index,
)

from .dataset_netcdf import (
    GlobalLAIDataset,
    GLOBAL_ALL_FEATURES,
    METEO_FEATURES as GLOBAL_METEO_FEATURES,
    PFT_FEATURES as GLOBAL_PFT_FEATURES,
)


# Feature engineering
from .feature_engineering import (
    add_derived_features,
    # Threshold constants
    GDD_THRESHOLDS,
    CHILLING_THRESHOLD,
    BOTTA_C1,
    BOTTA_C2,
    BOTTA_C3,
)


# Re-export commonly used functions for convenience
__all__ = [
    # Dataset classes
    "PhenoCamDataset",
    "LAIDataset",
    # Data loading utilities
    "load_site",
    "extract_pft_and_site",
    "_compute_norm_stats_deprecated",
    "_load_norm_stats_deprecated",
    "get_site_ids",
    "load_lai_norms",
    # Dataset splitting
    "split_sites_by_fraction",
    "split_sites_flat",
    # Feature engineering
    "add_derived_features",
    "GDD_THRESHOLDS",
    "CHILLING_THRESHOLD",
    "BOTTA_C1",
    "BOTTA_C2",
    "BOTTA_C3",
    # Feature constants
    "DYNAMIC_FEATURES",
    "CYCLIC_FEATURES",
    "STATIC_FEATURES",
    "ALL_FEATURES",
    "PFT_COLS",
    "LOG_TRANSFORM_FEATURES",
    # Dataset constants
    "N_OBS_PER_YEAR",
    "TARGET_DAYS_OF_MONTH",
    "BigLAIDataset",
    "generate_site_ids_from_range",
    "get_pixel_index",
    "GlobalLAIDataset",
    "GLOBAL_ALL_FEATURES",
    "GLOBAL_METEO_FEATURES",
    "GLOBAL_PFT_FEATURES",
]
