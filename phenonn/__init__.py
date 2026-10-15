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
# Data exports (kept surface only)
from .data import (
    RamLAIDataset,
)


# Model exports
from phenonn.models.rnn import RNN_LSTM, RNN_GRU
from phenonn.models.aelstm import AELSTM
from phenonn.models.bitransformer import BiTransformerV2
from phenonn.models.transformer import (
    EncoderTorch,
    CausalSparseTransformer,
    SinusoidalPositionalEncoding,
)

# Training exports (kept surface only)
from .training.train_full_ram import (
    main as train,
    load_shared,
    train_one_config,
)

# Prediction exports (kept surface only)
from .prediction.predict import main as predict

# Set up device
if "CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

__all__ = [
    # Data
    "LAIDataset",
    "RamLAIDataset",
    # Models
    "RNN_LSTM",
    "RNN_GRU",
    "AELSTM",
    "BiTransformerV2",
    "EncoderTorch",
    "CausalSparseTransformer",
    "SinusoidalPositionalEncoding",
    # Training
    "train",
    "load_shared",
    "train_one_config",
    # Prediction
    "predict",
    # Version
    "__version__",
    "get_version",
    "get_versions",
    "DEVICE",
]

# Package metadata
__description__ = "LSTM-based phenology prediction from climate data"
__url__ = "https://github.com/kardaneh/PhenoNN"

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
    print("Type 'phenonn --help' for more information.")


# Uncomment to show version info in interactive mode
_print_version_info()
