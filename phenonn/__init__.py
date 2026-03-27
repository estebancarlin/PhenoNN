#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PhenoNN - Deep learning for phenology prediction using climate data.

This package provides LSTM-based models to predict Green Chromatic Coordinate (GCC)
from climate data, including dynamic variables (temperature, precipitation, etc.)
and static variables (mean annual temperature, mean annual precipitation).

The models are trained on data from different plant functional types (PFTs):
- DB: Deciduous Broadleaf
- EN: Evergreen Needleleaf
- GR: Grassland

The package includes modules for:
- Data loading and preprocessing (dataloader_phenodata)
- LSTM model definition (lstm)
- Model training (lstm_train)
- Prediction (lstm_pred)
- Hyperparameter tuning (lstm_hp_tuning)

Examples:
    >>> from phenonn import LSTM, PhenoDataset
    >>> from phenonn import run_lstm_pred, run_lstm_train

    # Train a model
    >>> run_lstm_train(m='full', pft='GR', nr_features=8, target='gcc_lowess', batch_size=4)

    # Make predictions
    >>> run_lstm_pred(m='full', pft='GR', batch_size=4, input_path='./example')
"""

import sys
import logging
import torch
import os

# Import version information (dynamic version from pyproject.toml)
try:
    from .version import __version__, get_version, get_versions
except ImportError:
    # Fallback for development without version.py
    __version__ = "0.1.0.dev0"

    def get_version():
        return __version__

    def get_versions():
        return {"version": __version__}


__author__ = "Kazem Ardaneh"
__email__ = "kazem.arrdaneh@gmail.com"
__license__ = "BSD"
__copyright__ = f"Copyright (c) 2024 {__author__}"

# Import core classes and functions for easy access
from .dataloader_phenodata import (
    PhenoDataset,
    IndexDataset,
    Standardizer,
    Normalizer,
    leave_onesite_out_crossvalidation,
    identity,
)

from .lstm import LSTM

# Import main functions from modules
from .lstm_train import (
    train_model as train_lstm_model,
    valid_model,
    run_lstm_train,
)

from .lstm_pred import (
    model_predict,
    run_lstm_pred,
)

from .lstm_hp_tuning import (
    run_lstm_hp_tuning,
)

# Define what gets imported with "from phenonn import *"
__all__ = [
    # Core classes
    "LSTM",
    "PhenoDataset",
    "IndexDataset",
    "Standardizer",
    "Normalizer",
    # Functions
    "leave_onesite_out_crossvalidation",
    "identity",
    "train_lstm_model",
    "valid_model",
    "run_lstm_train",
    "run_lstm_pred",
    "run_lstm_hp_tuning",
    "model_predict",
    # Version info
    "__version__",
    "get_version",
    "get_versions",
    "__author__",
    "__email__",
    "__license__",
]

# Package metadata
__description__ = "LSTM-based phenology prediction from climate data"
__url__ = "https://github.com/kardaneh/PhenoNN"

# Set up logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)

# Set CUDA device if specified
if "CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not hasattr(torch, "_phenonn_device_printed"):
    torch._phenonn_device_printed = True
    print(f"PhenoNN using device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
# Optional: Print version info in interactive mode


def _print_version_info():
    """Print version information when imported in interactive mode."""
    if hasattr(sys, "ps1") and not hasattr(sys, "_phenonn_version_printed"):
        sys._phenonn_version_printed = True
        print(f"PhenoNN version {__version__}")
        print(f"Author: {__author__}")
        print(f"Repository: {__url__}")
        print("Type 'help(phenonn)' for more information.")


# Uncomment to show version info in interactive mode
_print_version_info()
