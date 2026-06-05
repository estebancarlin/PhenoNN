# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Stefan Barbu, Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""
PhenoNN Utility Modules

Provides logging, diagnostics, model utilities, and evaluation functions.
"""

from .logger import Logger
from .diagnostics import (
    plot_loss_histories,
    plot_metric_histories,
    plot_pred_vs_obs,
    plot_gcc_curves,
    plot_gcc_curves_all,
    make_history_dicts,
)
from .evaluater import get_loss_function
from .model_utils import ModelUtils
from .model_loader import load_model
from .utils import FileUtils, EasyDict
from .wrappers import Every10DaysWrapper, permuteWrapper, LastNDaysWrapper

__all__ = [
    # Logger
    "Logger",
    # Diagnostics
    "plot_loss_histories",
    "plot_metric_histories",
    "plot_pred_vs_obs",
    "plot_gcc_curves",
    "plot_gcc_curves_all",
    "make_history_dicts",
    # Evaluation
    "get_loss_function",
    # Model utilities
    "ModelUtils",
    "load_model",
    # General utilities
    "FileUtils",
    "EasyDict",
    # Wrappers
    "Every10DaysWrapper",
    "permuteWrapper",
    "LastNDaysWrapper",
]
