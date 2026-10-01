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
    CYCLIC_FEATURES,
    # Feature constants
    DYNAMIC_FEATURES,
    LOG_TRANSFORM_FEATURES,
    STATIC_FEATURES,
    PhenoCamDataset,
    extract_pft_and_site,
    load_lai_norms,
    load_site,
    split_sites_by_fraction,
)
from .dataset import (
    compute_norm_stats as _compute_norm_stats_deprecated,  # Legacy, use normalization.py
)
from .dataset import (
    load_norm_stats as _load_norm_stats_deprecated,
)
from .dataset_big import (
    BigLAIDataset,
    generate_site_ids_from_range,
    get_pixel_index,
)
from .dataset_flat import (
    ALL_FEATURES,
    N_OBS_PER_YEAR,
    PFT_COLS,
    TARGET_DAYS_OF_MONTH,
    LAIDataset,
    get_site_ids,
)
from .dataset_flat import (
    split_sites_by_fraction as split_sites_flat,
)
from .dataset_netcdf import (
    GLOBAL_ALL_FEATURES,
    GlobalLAIDataset,
)
from .dataset_netcdf import (
    METEO_FEATURES as GLOBAL_METEO_FEATURES,
)
from .dataset_netcdf import (
    PFT_FEATURES as GLOBAL_PFT_FEATURES,
)

# Feature engineering
from .feature_engineering import (
    BOTTA_C1,
    BOTTA_C2,
    BOTTA_C3,
    CHILLING_THRESHOLD,
    # Threshold constants
    GDD_THRESHOLDS,
    add_derived_features,
)

# Re-export commonly used functions for convenience
__all__ = [
    "ALL_FEATURES",
    "BOTTA_C1",
    "BOTTA_C2",
    "BOTTA_C3",
    "CHILLING_THRESHOLD",
    "CYCLIC_FEATURES",
    # Feature constants
    "DYNAMIC_FEATURES",
    "GDD_THRESHOLDS",
    "GLOBAL_ALL_FEATURES",
    "GLOBAL_METEO_FEATURES",
    "GLOBAL_PFT_FEATURES",
    "LOG_TRANSFORM_FEATURES",
    # Dataset constants
    "N_OBS_PER_YEAR",
    "PFT_COLS",
    "STATIC_FEATURES",
    "TARGET_DAYS_OF_MONTH",
    "BigLAIDataset",
    "GlobalLAIDataset",
    "LAIDataset",
    # Dataset classes
    "PhenoCamDataset",
    "_compute_norm_stats_deprecated",
    "_load_norm_stats_deprecated",
    # Feature engineering
    "add_derived_features",
    "extract_pft_and_site",
    "generate_site_ids_from_range",
    "get_pixel_index",
    "get_site_ids",
    "load_lai_norms",
    # Data loading utilities
    "load_site",
    # Dataset splitting
    "split_sites_by_fraction",
    "split_sites_flat",
]
# The 0.1-degree pipeline lives in lai_dataset; retain the established CSV and
# global NetCDF exports above for existing experiments and checkpoints.
from .lai_dataset import RamLAIDataset as RamLAIDataset

__all__.append("RamLAIDataset")
