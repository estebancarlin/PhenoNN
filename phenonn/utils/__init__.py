# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Stefan Barbu, Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""
PhenoNN utility package.

Public API
----------
Configuration
    ALL_FEATURES, DYNAMIC_FEATURES, CYCLIC_FEATURES, CO2_FEATURES,
    METEO_BASE, PFT_COLS, PFT_NAMES, PFT_START, FEATURE_CHANNELS,
    N_PFT, N_DEKAD_YEAR, DEKAD_DAYS, SEQ_LENGTH_DEFAULT,
    LOG_TRANSFORM_FEATURES, FEATURES_FNAME, TARGETS_FNAME, PFT_FNAME,
    VALID_FNAME, SELECTED_PIXELS_FNAME
Wrappers
    DailyWrapper, Every10DaysWrapper, PFTMixingWrapper, permuteWrapper
Model loader
    build_model, build_model_pft
Loss
    nan_safe_loss, nan_safe_mse, make_loss_fn
Diagnostics
    plot_loss_histories, plot_metric_histories, plot_pred_vs_obs,
    plot_gcc_curves, plot_gcc_curves_all, make_history_dicts
Logger
    Logger
General helpers
    FileUtils, EasyDict
"""

from .config import (
    ALL_FEATURES,
    DYNAMIC_FEATURES,
    CYCLIC_FEATURES,
    CO2_FEATURES,
    METEO_BASE,
    PFT_COLS,
    PFT_NAMES,
    PFT_START,
    FEATURE_CHANNELS,
    N_PFT,
    N_DEKAD_YEAR,
    DEKAD_DAYS,
    SEQ_LENGTH_DEFAULT,
    LOG_TRANSFORM_FEATURES,
    FEATURES_FNAME,
    TARGETS_FNAME,
    PFT_FNAME,
    VALID_FNAME,
    SELECTED_PIXELS_FNAME,
)
from .wrappers import (
    DailyWrapper,
    Every10DaysWrapper,
    PFTMixingWrapper,
    permuteWrapper,
)
from .model_loader import build_model, build_model_pft
from .evaluater import nan_safe_loss, nan_safe_mse, make_loss_fn
from .diagnostics import (
    plot_loss_histories,
    plot_metric_histories,
    plot_pred_vs_obs,
    plot_gcc_curves,
    plot_gcc_curves_all,
    make_history_dicts,
)
from .logger import Logger
from .utils import FileUtils, EasyDict

__all__ = [
    "ALL_FEATURES",
    "DYNAMIC_FEATURES",
    "CYCLIC_FEATURES",
    "CO2_FEATURES",
    "METEO_BASE",
    "PFT_COLS",
    "PFT_NAMES",
    "PFT_START",
    "FEATURE_CHANNELS",
    "N_PFT",
    "N_DEKAD_YEAR",
    "DEKAD_DAYS",
    "SEQ_LENGTH_DEFAULT",
    "LOG_TRANSFORM_FEATURES",
    "FEATURES_FNAME",
    "TARGETS_FNAME",
    "PFT_FNAME",
    "VALID_FNAME",
    "SELECTED_PIXELS_FNAME",
    "DailyWrapper",
    "Every10DaysWrapper",
    "PFTMixingWrapper",
    "permuteWrapper",
    "build_model",
    "build_model_pft",
    "nan_safe_loss",
    "nan_safe_mse",
    "make_loss_fn",
    "plot_loss_histories",
    "plot_metric_histories",
    "plot_pred_vs_obs",
    "plot_gcc_curves",
    "plot_gcc_curves_all",
    "make_history_dicts",
    "Logger",
    "FileUtils",
    "EasyDict",
]
