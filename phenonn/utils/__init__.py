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

import importlib

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

_CONFIG = {
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
}

_LAZY = {
    "DailyWrapper": (".wrappers", "DailyWrapper"),
    "Every10DaysWrapper": (".wrappers", "Every10DaysWrapper"),
    "PFTMixingWrapper": (".wrappers", "PFTMixingWrapper"),
    "permuteWrapper": (".wrappers", "permuteWrapper"),
    "build_model": (".model_loader", "build_model"),
    "build_model_pft": (".model_loader", "build_model_pft"),
    "nan_safe_loss": (".evaluater", "nan_safe_loss"),
    "nan_safe_mse": (".evaluater", "nan_safe_mse"),
    "make_loss_fn": (".evaluater", "make_loss_fn"),
    "plot_loss_histories": (".diagnostics", "plot_loss_histories"),
    "plot_metric_histories": (".diagnostics", "plot_metric_histories"),
    "plot_pred_vs_obs": (".diagnostics", "plot_pred_vs_obs"),
    "plot_gcc_curves": (".diagnostics", "plot_gcc_curves"),
    "plot_gcc_curves_all": (".diagnostics", "plot_gcc_curves_all"),
    "make_history_dicts": (".diagnostics", "make_history_dicts"),
    "Logger": (".logger", "Logger"),
    "FileUtils": (".utils", "FileUtils"),
    "EasyDict": (".utils", "EasyDict"),
}


def __getattr__(name):
    if name in _CONFIG:
        mod_name, attr = ".config", name
    elif name in _LAZY:
        mod_name, attr = _LAZY[name]
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(mod_name, __name__), attr)
    globals()[name] = value
    return value


def __dir__():
    return sorted(__all__)
