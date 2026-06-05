# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Stefan Barbu, Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""
PhenoNN Training Module

Provides training functions for both per-site CSV and flat CSV formats.
"""

from .train import run_training
from .train_flat import run_training_flat
from .train_big import run_training_big


__all__ = [
    # CLI entry points
    "run_training",
    "run_training_flat",
    "run_training_big",
]
