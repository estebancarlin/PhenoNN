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

import importlib
import logging

# Version (will be updated by setuptools)
try:
    from .version import __version__, get_version, get_versions
except ImportError:
    __version__ = "0.1.0.dev0"

__author__ = "Kazem Ardaneh and Stefan Barbu"
__email__ = "krdaneh@ipsl.fr"
__license__ = "CC BY-NC-SA 4.0"

__all__ = [
    # Data
    "LAIDataset",
    "RamLAIDataset",
    # Models
    "RNN_LSTM",
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

_LAZY = {
    "LAIDataset": (".data", "LAIDataset"),
    "RamLAIDataset": (".data", "RamLAIDataset"),
    "RNN_LSTM": (".models.rnn", "RNN_LSTM"),
    "RNN_GRU": (".models.rnn", "RNN_GRU"),
    "AELSTM": (".models.aelstm", "AELSTM"),
    "BiTransformerV2": (".models.bitransformer", "BiTransformerV2"),
    "EncoderTorch": (".models.transformer", "EncoderTorch"),
    "CausalSparseTransformer": (".models.transformer", "CausalSparseTransformer"),
    "SinusoidalPositionalEncoding": (
        ".models.transformer",
        "SinusoidalPositionalEncoding",
    ),
    "train": (".training.train_full_ram", "main"),
    "load_shared": (".training.train_full_ram", "load_shared"),
    "train_one_config": (".training.train_full_ram", "train_one_config"),
    "predict": (".prediction.predict", "main"),
}


def __getattr__(name):
    if name == "DEVICE":
        import os

        if "CUDA_VISIBLE_DEVICES" in os.environ:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        import torch

        value = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        globals()[name] = value
        return value
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        value = getattr(importlib.import_module(mod_name, __name__), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)


def _print_version_info():
    """Print version information when imported in interactive mode."""
    print(f"PhenoNN version {__version__}")
    print(f"Author: {__author__}")
    print(f"Repository: {__url__}")
    print("Type 'help(phenonn)' for more information.")


# Uncomment to show version info in interactive mode
_print_version_info()
