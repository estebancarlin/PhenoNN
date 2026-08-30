# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Stefan Barbu, Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""
PhenoNN - Deep learning for phenology (LAI) prediction from climate data.

Sub-packages
------------
  - data_creation : dataset building / preprocessing
  - data          : the RAM-resident LAI dataset
  - models        : model architectures
  - training      : train_full_ram (+ xgb_train baseline)
  - prediction    : inference (+ XGBoost / greedy baselines)
  - analysis      : post-training studies, baselines, diagnostics
  - utils         : config, loss, model_factory, wrappers, logger, diagnostics
"""

import logging
import os

import torch

try:
    from .version import __version__, get_version, get_versions
except ImportError:  # pragma: no cover
    __version__ = "0.1.0.dev0"

    def get_version():
        return __version__

    def get_versions():
        return {"version": __version__}


__author__ = "Kazem Ardaneh and Stefan Barbu"
__email__ = "kardaneh@ipsl.fr"
__license__ = "CC BY-NC-SA 4.0"
__url__ = "https://github.com/kardaneh/PhenoNN"

# Device selection (kept importable as phenonn.DEVICE for compatibility).
if "CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ["__version__", "get_version", "get_versions", "DEVICE"]
