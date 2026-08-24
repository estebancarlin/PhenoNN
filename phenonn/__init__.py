# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Stefan Barbu, Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""
PhenoNN - Deep learning for phenology prediction using climate data.

This package provides LSTM, GRU, and Transformer models to predict
Green Chromatic Coordinate (GCC) or Leaf Area Index (LAI) from climate data.
"""

import logging
import torch
import os

# Version (will be updated by setuptools)
try:
    from .version import __version__, get_version, get_versions
except ImportError:
    __version__ = "0.1.0.dev0"

__author__ = "Kazem Ardaneh and Stefan Barbu"
__email__ = "krdaneh@ipsl.fr"
__license__ = "CC BY-NC-SA 4.0"

# Core exports
from .data.dataset import PhenoCamDataset, split_sites_by_fraction, extract_pft_and_site
from .data.dataset_flat import (
    LAIDataset,
    get_site_ids,
    split_sites_by_fraction as split_sites_flat,
)
from .data.feature_engineering import add_derived_features

# Model exports
from phenonn.models.rnn import RNN_LSTM, RNN_GRU
from phenonn.models.transformer import EncoderTorch
from phenonn.models.transformerbis import (
    CombinedModel,
    BiTransformer,
)

# Training exports
from .training.train import run_training
from .training.train_flat import run_training_flat
from .training.train_big import run_training_big

# Prediction exports
from .prediction.predict import run_prediction
from .prediction.predict_flat import run_prediction_flat

# Set up device
if "CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

__all__ = [
    # Data
    "PhenoCamDataset",
    "LAIDataset",
    "compute_norm_stats",
    "load_norm_stats",
    "add_derived_features",
    "split_sites_by_fraction",
    "split_sites_flat",
    "get_site_ids",
    "extract_pft_and_site",
    # Models
    "RNN_LSTM",
    "RNN_GRU",
    "EncoderTorch",
    "CombinedModel",
    "BiTransformer",
    # Training
    "run_training",
    "run_training_flat",
    "run_training_big",
    # Prediction
    "run_prediction",
    "run_prediction_flat",
    # Version
    "__version__",
    "get_version",
    "get_versions",
    "DEVICE",
]


# Package metadata
__description__ = "LSTM-based phenology prediction from climate data"
__url__ = "https://github.com/estebancarlin/PhenoNN"

# Set up logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)

if not hasattr(torch, "_phenonn_device_printed"):
    torch._phenonn_device_printed = True
    print(f"PhenoNN using device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
# Optional: Print version info in interactive mode


def _print_version_info():
    """Print version information when imported in interactive mode."""
    print(f"PhenoNN version {__version__}")
    print(f"Author: {__author__}")
    print(f"Repository: {__url__}")
    print("Type 'help(phenonn)' for more information.")


# Uncomment to show version info in interactive mode
_print_version_info()
