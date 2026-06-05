# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Stefan Barbu, Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""
PhenoNN Prediction Module

Provides functions for running predictions with trained models:
- Single-year prediction (per-site CSV format)
- Flat CSV prediction (features/targets format)
"""

from .predict import run_prediction
from .predict_flat import run_prediction_flat

__all__ = [
    "run_prediction",
    "run_prediction_flat",
]
