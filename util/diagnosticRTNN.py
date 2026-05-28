"""
Plotting utilities for RTnn model visualization.

This module provides functions for visualizing model predictions, training metrics,
and data statistics. It includes tools for creating line plots, hexbin plots,
histograms, and metric history plots using matplotlib.

The module supports:
- Visualization of radiative transfer model predictions vs targets
- Absorption rate plotting for different channels
- Training and validation metric histories
- Statistical distributions of input variables
- Various normalization scheme visualizations

Dependencies
------------
matplotlib : For plotting
mpltex : For line styles
scikit-learn : For R² score calculation
xarray : For NetCDF data handling
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import mpltex
import math
from matplotlib import rcParams as mpl
from sklearn.metrics import r2_score
import matplotlib.ticker as ticker
import random
import xarray as xr
import os
import collections

params = {
    "font.family": "DejaVu Sans",
    #    'figure.dpi': 300,
    #    'savefig.dpi': 300,
    "lines.linewidth": 2,
    "lines.dashed_pattern": [4, 2],
    "lines.dashdot_pattern": [6, 3, 2, 3],
    "lines.dotted_pattern": [2, 3],
    "mathtext.rm": "arial",
    "axes.labelsize": 15,
    "axes.titlesize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "legend.fontsize": 15,
    "legend.loc": "best",
    "legend.frameon": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
}
mpl.update(params)


def stats(file_list, logger, output_dir, norm_mapping=None, plots=False):
    """
    Compute statistics and generate histograms for variables in NetCDF files.

    Reads a collection of NetCDF files, computes descriptive statistics for each
    variable, and generates histogram plots saved to disk. In addition to raw
    statistics, transformed statistics using logarithmic (log1p) and square-root
    transformations are also computed.

    Parameters
    ----------
    file_list : list of str
        Paths to the NetCDF files to process.
    logger : logging.Logger
        Logger used to report progress and informational messages.
    output_dir : str
        Directory where histogram plots will be saved.
    norm_mapping : dict, optional
        Dictionary to update with computed statistics. If None, a new dictionary
        is created. Default is None.

    Returns
    -------
    dict
        Dictionary mapping variable names to their computed statistics. Each
        variable contains the following entries:

        Raw statistics:
            - vmin : float
            - vmax : float
            - vmean : float
            - vstd : float

        Robust statistics:
            - q1 : float
            - q3 : float
            - iqr : float
            - median : float

        Log-transformed statistics (log1p):
            - log_min : float
            - log_max : float
            - log_mean : float
            - log_std : float
            - log_q1 : float
            - log_q3 : float
            - log_iqr : float
            - log_median : float

        Square-root-transformed statistics:
            - sqrt_min : float
            - sqrt_max : float
            - sqrt_mean : float
            - sqrt_std : float
            - sqrt_q1 : float
            - sqrt_q3 : float
            - sqrt_iqr : float
            - sqrt_median : float

    Examples
    --------
    >>> norm_mapping = stats(
    ...     file_list=["data_1995.nc", "data_1996.nc"],
    ...     logger=logger,
    ...     output_dir="./stats"
    ... )
    >>> norm_mapping["coszang"]["vmean"]
    0.5
    """
    variable_data = collections.defaultdict(list)

    if norm_mapping is None:
        norm_mapping = {}

    logger.info("Starting statistics computation for normalization")
    for fpath in file_list:
        try:
            ds = xr.open_dataset(fpath)
            logger.info(f"Processing file: {fpath}")
            for var_name in ds.data_vars:
                data = ds[var_name].values
                variable_data[var_name].append(data.flatten())
            ds.close()
        except Exception as e:
            logger.warning(f"Skipping file {fpath} due to error: {e}")
            continue

    for var_name, arrays in variable_data.items():
        full_data = np.concatenate(arrays)
        if full_data.size == 0:
            logger.warning(f"{var_name} is empty after filtering, skipping.")
            continue

        vmin = float(np.min(full_data))
        vmax = float(np.max(full_data))
        vmean = float(np.mean(full_data))
        vstd = float(np.std(full_data))
        q1 = float(np.percentile(full_data, 25))
        q3 = float(np.percentile(full_data, 75))
        iqr = q3 - q1 if q3 != q1 else 1.0
        median = float(np.median(full_data))

        log_data = np.log1p(np.clip(full_data, a_min=0, a_max=None))
        log_min = float(log_data.min())
        log_max = float(log_data.max())
        log_mean = float(log_data.mean())
        log_std = float(log_data.std())
        log_q1 = float(np.percentile(log_data, 25))
        log_q3 = float(np.percentile(log_data, 75))
        log_iqr = log_q3 - log_q1 if log_q3 != log_q1 else 1.0
        log_median = float(np.median(log_data))

        sqrt_data = np.sqrt(np.clip(full_data, a_min=0, a_max=None))
        sqrt_min = float(sqrt_data.min())
        sqrt_max = float(sqrt_data.max())
        sqrt_mean = float(sqrt_data.mean())
        sqrt_std = float(sqrt_data.std())
        sqrt_q1 = float(np.percentile(sqrt_data, 25))
        sqrt_q3 = float(np.percentile(sqrt_data, 75))
        sqrt_iqr = sqrt_q3 - sqrt_q1 if sqrt_q3 != sqrt_q1 else 1.0
        sqrt_median = float(np.median(sqrt_data))

        norm_mapping[var_name] = {
            "vmin": vmin,
            "vmax": vmax,
            "vmean": vmean,
            "vstd": vstd,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "median": median,
            "log_min": log_min,
            "log_max": log_max,
            "log_mean": log_mean,
            "log_std": log_std,
            "log_q1": log_q1,
            "log_q3": log_q3,
            "log_iqr": log_iqr,
            "log_median": log_median,
            "sqrt_min": sqrt_min,
            "sqrt_max": sqrt_max,
            "sqrt_mean": sqrt_mean,
            "sqrt_std": sqrt_std,
            "sqrt_q1": sqrt_q1,
            "sqrt_q3": sqrt_q3,
            "sqrt_iqr": sqrt_iqr,
            "sqrt_median": sqrt_median,
        }

        if plots:
            norm_label = ""
            file_suffix = "_histogram.png"

            fig = plt.figure(figsize=(8, 5))
            ax = fig.add_subplot(111)
            ax.hist(full_data, bins=200)
            ax.set_yscale("log")
            ax.set_title(f"Histogram of {var_name}{norm_label}")
            ax.set_xlabel(var_name + norm_label)
            ax.set_ylabel("Log Count")
            ax.grid(True)
            out_path = os.path.join(output_dir, f"{var_name}{file_suffix}")
            plt.savefig(out_path, bbox_inches="tight")
            plt.close(fig)

    return norm_mapping


def plot_RTM(predicts, targets, filename):
    """
    Plot radiative transfer model predictions against targets.

    Creates a 2x2 grid of line plots showing predicted vs true values for
    four different flux channels. Plots random samples from the batch.

    Parameters
    ----------
    predicts : torch.Tensor
        Model predictions of shape (batch_size, 4, seq_length).
    targets : torch.Tensor
        Ground truth targets of shape (batch_size, 4, seq_length).
    filename : str
        Path where the plot will be saved.

    Notes
    -----
    - Plots up to 7 random samples from the batch
    - Uses mpltex for line styles
    - Y-axis range: 0-1
    - X-axis: level indices
    """
    predicts = predicts.cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()
    num_samples = predicts.shape[0]
    sample_indices = random.sample(range(num_samples), 7)

    fig = plt.figure(tight_layout=True, figsize=(12, 12))
    plt.subplots_adjust(
        hspace=0.2, wspace=0.2, left=0.1, right=0.9, top=0.9, bottom=0.1
    )
    gs = gridspec.GridSpec(2, 2)

    name_dict = {
        (0, 0): {"plotname": r"$\mathrm{Flux_1}$"},
        (0, 1): {"plotname": r"$\mathrm{Flux_2}$"},
        (1, 0): {"plotname": r"$\mathrm{Flux_3}$"},
        (1, 1): {"plotname": r"$\mathrm{Flux_4}$"},
    }

    index_map = [(0, 0), (0, 1), (1, 0), (1, 1)]
    legend_lines = []
    legend_labels = []

    for idx, (i, j) in enumerate(index_map):
        ax = fig.add_subplot(gs[i, j])
        linestyles = mpltex.linestyle_generator()
        for sample_index in sample_indices:
            (pred_line,) = ax.plot(
                predicts[sample_index, idx, :], label="Predict", **next(linestyles)
            )
            (true_line,) = ax.plot(
                targets[sample_index, idx, :], label="True", **next(linestyles)
            )
            if idx == 0:
                legend_lines.extend([pred_line, true_line])
                legend_labels.extend([pred_line.get_label(), true_line.get_label()])

        ax.xaxis.set_major_locator(ticker.MultipleLocator(3))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
        ax.set_ylabel(name_dict[(i, j)]["plotname"])
        ax.set_xlabel(r"$\mathrm{Level}$")
        ax.set_ylim(0, 1)
        ax.set_xlim(0, predicts.shape[-1] - 1)
        fig.legend(
            handles=legend_lines,
            labels=legend_labels,
            loc="center right",
            bbox_to_anchor=(1.2, 0.5),
            borderaxespad=0.5,
            frameon=False,
            ncol=1,
        )
    plt.savefig(filename, bbox_inches="tight")
    plt.close(fig)


def plot_HeatRate(abs12_predict, abs12_target, abs34_predict, abs34_target, filename):
    """
    Plot absorption rates for two channel groups.

    Creates a 2-panel figure showing absorption rates for channels 1-2 and 3-4.

    Parameters
    ----------
    abs12_predict : torch.Tensor
        Predicted absorption for channels 1-2 of shape (batch_size, 1, seq_length).
    abs12_target : torch.Tensor
        True absorption for channels 1-2.
    abs34_predict : torch.Tensor
        Predicted absorption for channels 3-4.
    abs34_target : torch.Tensor
        True absorption for channels 3-4.
    filename : str
        Path where the plot will be saved.

    Notes
    -----
    - Upper panel: channels 1-2
    - Lower panel: channels 3-4
    - Plots random samples from the batch
    """
    abs12_predict = abs12_predict.cpu().detach().numpy()
    abs12_target = abs12_target.cpu().detach().numpy()
    abs34_predict = abs34_predict.cpu().detach().numpy()
    abs34_target = abs34_target.cpu().detach().numpy()

    num_samples = abs12_predict.shape[0]
    sample_indices = random.sample(range(num_samples), 7)

    fig = plt.figure(tight_layout=False, figsize=(6, 12))
    plt.subplots_adjust(
        hspace=0.2, wspace=0.2, left=0.1, right=0.9, top=0.9, bottom=0.1
    )
    gs = gridspec.GridSpec(2, 1, figure=fig)

    legend_lines = []
    legend_labels = []

    ax1 = fig.add_subplot(gs[0, 0])
    linestyles = mpltex.linestyle_generator()

    for sample_index in sample_indices:
        (pred_line,) = ax1.plot(
            abs12_predict[sample_index, 0, :], label="Predict", **next(linestyles)
        )
        (true_line,) = ax1.plot(
            abs12_target[sample_index, 0, :], label="True", **next(linestyles)
        )

        legend_lines.extend([pred_line, true_line])
        legend_labels.extend([pred_line.get_label(), true_line.get_label()])

    ax1.set_ylabel(r"$\mathrm{Absorption_{12}}$")
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(3))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
    ax1.set_xlim(0, abs12_predict.shape[-1] - 1)
    ax1.set_ylim(0, 1)

    ax2 = fig.add_subplot(gs[1, 0])
    linestyles = mpltex.linestyle_generator()

    for sample_index in sample_indices:
        ax2.plot(abs34_predict[sample_index, 0, :], label="Predict", **next(linestyles))
        ax2.plot(abs34_target[sample_index, 0, :], label="True", **next(linestyles))

    ax2.set_ylabel(r"$\mathrm{Absorption_{34}}$")
    ax2.set_xlabel(r"$\mathrm{Level}$")
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(3))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
    ax2.set_xlim(0, abs34_predict.shape[-1] - 1)
    ax2.set_ylim(0, 1)

    fig.legend(
        handles=legend_lines,
        labels=legend_labels,
        loc="center right",
        bbox_to_anchor=(1.3, 0.5),
        borderaxespad=0.5,
        frameon=False,
        ncol=1,
    )

    plt.savefig(filename, bbox_inches="tight")
    plt.close(fig)


def plot_flux_and_abs_lines(
    predicts,
    targets,
    abs12_predict=None,
    abs12_target=None,
    abs34_predict=None,
    abs34_target=None,
    filename="output_lines.png",
    logger=None,
):
    """
    Create line plots for fluxes and absorption rates.

    Generates a multi-panel figure with line plots for four flux channels and
    optionally two absorption panels. Each panel shows predictions vs targets.

    Parameters
    ----------
    predicts : torch.Tensor
        Model predictions for fluxes of shape (batch_size, 4, seq_length).
    targets : torch.Tensor
        Ground truth fluxes.
    abs12_predict : torch.Tensor, optional
        Predicted absorption for channels 1-2.
    abs12_target : torch.Tensor, optional
        True absorption for channels 1-2.
    abs34_predict : torch.Tensor, optional
        Predicted absorption for channels 3-4.
    abs34_target : torch.Tensor, optional
        True absorption for channels 3-4.
    filename : str, optional
        Output filename. Default is "output_lines.png".
    logger : logging.Logger, optional
        Logger instance for logging messages. If None, no logging is performed.

    Notes
    -----
    Figure layout:
        - 2x2 grid for fluxes (upwelling/downwelling for two channels)
        - Optional 1x2 grid for absorption rates (if provided)
    """
    predicts = predicts.cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()
    include_abs12 = abs12_predict is not None and abs12_target is not None
    include_abs34 = abs34_predict is not None and abs34_target is not None
    include_abs = include_abs12 and include_abs34

    if include_abs:
        fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    plt.subplots_adjust(
        hspace=0.3, wspace=0.3, left=0.1, right=0.9, top=0.9, bottom=0.1
    )

    num_samples = predicts.shape[0]
    sample_indices = random.sample(range(num_samples), 7)

    index_map = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}

    name_dict = {
        (0, 0): {"plotname": r"$\mathrm{Flux_{1u}}$"},
        (0, 1): {"plotname": r"$\mathrm{Flux_{1d}}$"},
        (1, 0): {"plotname": r"$\mathrm{Flux_{2u}}$"},
        (1, 1): {"plotname": r"$\mathrm{Flux_{2d}}$"},
    }

    legend_lines = []
    legend_labels = []

    for (r, c), props in name_dict.items():
        flux_idx = index_map[(r, c)]
        ax = axes[r, c]
        linestyles = mpltex.linestyle_generator()

        for sample_index in sample_indices:
            (pred_line,) = ax.plot(
                predicts[sample_index, flux_idx, :], label="Predict", **next(linestyles)
            )
            (true_line,) = ax.plot(
                targets[sample_index, flux_idx, :], label="True", **next(linestyles)
            )
            if (r, c) == (0, 0):
                legend_lines.extend([pred_line, true_line])
                legend_labels.extend([pred_line.get_label(), true_line.get_label()])

        ax.set_xlabel(r"$\mathrm{Level}$")
        ax.set_ylabel(props["plotname"])
        ax.set_xlim(0, predicts.shape[-1] - 1)
        ax.set_ylim(0, 1)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(3))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))

    if include_abs:
        ax = axes[2, 0]
        linestyles = mpltex.linestyle_generator()
        for sample_index in sample_indices:
            (pred_line,) = ax.plot(
                abs12_predict[sample_index, 0, :].cpu().detach().numpy(),
                label="Predict)",
                **next(linestyles),
            )
            (true_line,) = ax.plot(
                abs12_target[sample_index, 0, :].cpu().detach().numpy(),
                label="True",
                **next(linestyles),
            )

        ax.set_ylabel(r"$\mathrm{Absorption_{1}}$")
        ax.set_xlabel(r"$\mathrm{Level}$")
        ax.set_xlim(0, abs12_predict.shape[-1] - 1)
        ax.set_ylim(0, 1)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(3))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))

        ax = axes[2, 1]
        linestyles = mpltex.linestyle_generator()
        for sample_index in sample_indices:
            (pred_line,) = ax.plot(
                abs34_predict[sample_index, 0, :].cpu().detach().numpy(),
                label="Predict",
                **next(linestyles),
            )
            (true_line,) = ax.plot(
                abs34_target[sample_index, 0, :].cpu().detach().numpy(),
                label="True",
                **next(linestyles),
            )

        ax.set_ylabel(r"$\mathrm{Absorption_{2}}$")
        ax.set_xlabel(r"$\mathrm{Level}$")
        ax.set_xlim(0, abs34_predict.shape[-1] - 1)
        ax.set_ylim(0, 1)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(3))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))

    fig.legend(
        handles=legend_lines,
        labels=legend_labels,
        loc="center right",
        bbox_to_anchor=(1.1, 0.5),
        borderaxespad=0.5,
        frameon=False,
        ncol=1,
    )

    plt.savefig(filename, bbox_inches="tight")
    plt.close(fig)
    if logger:
        logger.info(f"Saved line plot to {filename}")
    else:
        print(f"Saved line plot to {filename}")


def plot_flux_and_abs(
    predicts,
    targets,
    abs12_predict=None,
    abs12_target=None,
    abs34_predict=None,
    abs34_target=None,
    filename="output.png",
    logger=None,
):
    """
    Create hexbin plots for fluxes and absorption rates.

    Generates a multi-panel figure with hexbin density plots showing the
    relationship between predicted and true values. Useful for assessing
    prediction accuracy across the entire dataset.

    Parameters
    ----------
    predicts : torch.Tensor
        Model predictions for fluxes of shape (batch_size, 4, seq_length).
    targets : torch.Tensor
        Ground truth fluxes.
    abs12_predict : torch.Tensor, optional
        Predicted absorption for channels 1-2.
    abs12_target : torch.Tensor, optional
        True absorption for channels 1-2.
    abs34_predict : torch.Tensor, optional
        Predicted absorption for channels 3-4.
    abs34_target : torch.Tensor, optional
        True absorption for channels 3-4.
    filename : str, optional
        Output filename. Default is "output.png".
    logger : logging.Logger, optional
        Logger instance for logging messages. If None, no logging is performed.

    Notes
    -----
    - Hexbin plots use logarithmic color scale
    - Includes diagonal reference line (y=x)
    - Displays R² score in the top-left corner of each panel
    - Shared colorbar on the right
    """

    include_abs12 = abs12_predict is not None and abs12_target is not None
    include_abs34 = abs34_predict is not None and abs34_target is not None
    include_abs = include_abs12 and include_abs34

    if include_abs:
        fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    plt.subplots_adjust(
        hspace=0.3, wspace=0.3, left=0.1, right=0.9, top=0.9, bottom=0.1
    )

    index_map = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}

    name_dict = {
        (0, 0): {"name": "Flux1u", "plotname": r"$\mathrm{Flux_{1u}}$"},
        (0, 1): {"name": "Flux1d", "plotname": r"$\mathrm{Flux_{1d}}$"},
        (1, 0): {"name": "Flux2u", "plotname": r"$\mathrm{Flux_{2u}}$"},
        (1, 1): {"name": "Flux2d", "plotname": r"$\mathrm{Flux_{2d}}$"},
    }

    for (r, c), props in name_dict.items():
        flux_idx = index_map[(r, c)]
        y_pred = predicts[:, flux_idx, :].reshape(-1).detach().cpu().numpy()
        y_true = targets[:, flux_idx, :].reshape(-1).detach().cpu().numpy()
        ax = axes[r, c]

        hb = ax.hexbin(
            y_true, y_pred, gridsize=100, cmap="jet", bins="log", vmin=1, vmax=1e6
        )
        # ax.set_xlim(y_true.min(), y_true.max())
        # ax.set_ylim(y_true.min(), y_true.max())
        ax.xaxis.set_major_locator(
            ticker.MultipleLocator((y_true.max() - y_true.min()) / 4)
        )
        ax.yaxis.set_major_locator(
            ticker.MultipleLocator((y_true.max() - y_true.min()) / 4)
        )
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.plot(
            [y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()],
            "r:",
            linewidth=0.5,
        )
        r2 = r2_score(y_true, y_pred)
        ax.text(0.05, 0.9, f"$R^2$: {r2:.5f}", transform=ax.transAxes, fontsize=10)
        flux_name = props["plotname"]
        ax.set_title(rf"{flux_name}")
        ax.set_xlabel("Observed")
        ax.set_ylabel("Predicted")

    if include_abs:
        ax = axes[2, 0]
        y_pred = abs12_predict.reshape(-1).detach().cpu().numpy()
        y_true = abs12_target.reshape(-1).detach().cpu().numpy()
        hb = ax.hexbin(
            y_true, y_pred, gridsize=100, cmap="jet", bins="log", vmin=1, vmax=1e6
        )
        # ax.set_xlim(y_true.min(), y_true.max())
        # ax.set_ylim(y_true.min(), y_true.max())
        ax.xaxis.set_major_locator(
            ticker.MultipleLocator((y_true.max() - y_true.min()) / 4)
        )
        ax.yaxis.set_major_locator(
            ticker.MultipleLocator((y_true.max() - y_true.min()) / 4)
        )
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

        ax.plot(
            [y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()],
            "r:",
            linewidth=0.5,
        )
        r2 = r2_score(y_true, y_pred)
        ax.text(0.05, 0.9, f"$R^2$: {r2:.5f}", transform=ax.transAxes, fontsize=10)
        ax.set_title(r"$\mathrm{Abs_{1}}$")
        ax.set_xlabel("Observed")
        ax.set_ylabel("Predicted")

        ax = axes[2, 1]
        y_pred = abs34_predict.reshape(-1).detach().cpu().numpy()
        y_true = abs34_target.reshape(-1).detach().cpu().numpy()
        hb = ax.hexbin(
            y_true, y_pred, gridsize=100, cmap="jet", bins="log", vmin=1, vmax=1e6
        )
        ax.plot(
            [y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()],
            "r:",
            linewidth=0.5,
        )
        # ax.set_xlim(y_true.min(), y_true.max())
        # ax.set_ylim(y_true.min(), y_true.max())
        ax.xaxis.set_major_locator(
            ticker.MultipleLocator((y_true.max() - y_true.min()) / 4)
        )
        ax.yaxis.set_major_locator(
            ticker.MultipleLocator((y_true.max() - y_true.min()) / 4)
        )
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        r2 = r2_score(y_true, y_pred)
        ax.text(0.05, 0.9, f"$R^2$: {r2:.5f}", transform=ax.transAxes, fontsize=10)
        ax.set_title(r"$\mathrm{Abs_{2}}$")
        ax.set_xlabel("Observed")
        ax.set_ylabel("Predicted")

    # Shared colorbar on the right
    cbar_ax = fig.add_axes([0.92, 0.1, 0.015, 0.8])
    fig.colorbar(hb, cax=cbar_ax, label=r"$\mathrm{\log_{10}[Count]}$")
    plt.savefig(filename, bbox_inches="tight")
    plt.close(fig)
    if logger:
        logger.info(f"Saved hexbin plot to {filename}")
    else:
        print(f"Saved hexbin plot to {filename}")


def plot_metric_histories(
    train_history,
    valid_history,
    filename="training_validation_metrics.png",
    logger=None,
):
    """
    Plot training and validation metrics over epochs.

    Creates a multi-panel figure showing the evolution of various metrics
    (e.g., NMAE, NMSE, R2) over training epochs.

    Parameters
    ----------
    train_history : dict
        Dictionary with metric names as keys and lists of training values.
    valid_history : dict
        Dictionary with metric names as keys and lists of validation values.
    filename : str, optional
        Output filename. Default is "training_validation_metrics.png".
    logger : logging.Logger, optional
        Logger instance for logging messages. If None, no logging is performed.

    Notes
    -----
    - Metrics are plotted on a logarithmic scale
    - Each metric gets its own panel
    - Panels are arranged in a grid (3 columns)
    - Blue lines: training, Orange lines: validation
    """
    num_metrics = len(train_history)
    cols = 3
    rows = math.ceil(num_metrics / cols)

    fig = plt.figure(tight_layout=True, figsize=(5 * cols, 4 * rows))
    gs = gridspec.GridSpec(rows, cols)

    for idx, key in enumerate(train_history):
        row, col = divmod(idx, cols)
        ax = fig.add_subplot(gs[row, col])
        linestyles = mpltex.linestyle_generator(markers=[])

        ax.plot(train_history[key], label="train", **next(linestyles))
        ax.plot(valid_history[key], label="valid", **next(linestyles))
        ax.set_yscale("log")
        # ax.set_title(key.replace('_', ' ').upper())
        ax.set_xlabel("Epoch")
        ax.set_ylabel(key.replace("_", " ").upper())
        ax.legend()
        ax.grid(True)

    plt.savefig(filename, bbox_inches="tight")
    plt.close(fig)
    if logger:
        logger.info(f"Saved metric history plot to {filename}")
    else:
        print(f"Saved metric history plot to {filename}")


def plot_loss_histories(
    train_loss, valid_loss, filename="training_validation_loss.png", logger=None
):
    """
    Plot training and validation loss over epochs.

    Creates a single-panel figure showing the loss evolution during training.

    Parameters
    ----------
    train_loss : list or array
        Training loss values over epochs.
    valid_loss : list or array
        Validation loss values over epochs.
    filename : str, optional
        Output filename. Default is "training_validation_loss.png".
    logger : logging.Logger, optional
        Logger instance for logging messages. If None, no logging is performed.

    Notes
    -----
    - Uses logarithmic scale for y-axis
    - Blue line: training loss
    - Orange line: validation loss
    - Includes grid for better readability
    """
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    linestyles = mpltex.linestyle_generator(markers=[])
    ax.plot(train_loss, label="train", **next(linestyles))
    ax.plot(valid_loss, label="valid", **next(linestyles))
    ax.set_yscale("log")
    ax.set_title("LOSS")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss Value")
    ax.legend()
    ax.grid(True)
    plt.savefig(filename, bbox_inches="tight")
    plt.close(fig)
    if logger:
        logger.info(f"Saved loss history plot to {filename}")
    else:
        print(f"Saved loss history plot to {filename}")


def plot_spatial_temporal_density(
    sindex_tracker,
    tindex_tracker,
    mode="train",
    save_dir="./tests_plots",
    filename="density_scatter",
    figsize=(10, 10),
    logger=None,
):
    """
    Plot a density scatter plot of spatial index vs temporal index with marginal histograms.

    This function creates a 2D density scatter plot (hexbin) showing the
    distribution of spatial indices (processor ranks) across temporal indices,
    with:
    - Right plot: Histogram of temporal index distribution
    - Top plot: Histogram of spatial index distribution

    Parameters
    ----------
    sindex_tracker : list or array-like
        List of spatial indices (processor ranks) for each data sample.
    tindex_tracker : list or array-like
        List of temporal indices for each data sample.
    mode : str, optional
        Dataset mode identifier ("train", "validation", "test").
    save_dir : str, optional
        Directory path where the plot will be saved.
    filename : str, optional
        Base name for the output file.
    figsize : tuple, optional
        Figure size as (width, height) in inches.
    logger : logging.Logger, optional
        Logger instance for logging messages. If None, no logging is performed.

    Returns
    -------
    str
        Path to the saved plot file.
    """

    if not sindex_tracker or not tindex_tracker:
        print(f"No data to plot for {mode} mode")
        return None

    # Convert to numpy arrays
    sindex_tracker = np.array(sindex_tracker)
    tindex_tracker = np.array(tindex_tracker)

    # Get limits
    min_sindex = int(sindex_tracker.min())
    max_sindex = int(sindex_tracker.max())
    min_time = int(tindex_tracker.min())
    max_time = int(tindex_tracker.max())

    # Create figure with GridSpec for custom layout
    fig = plt.figure(figsize=figsize)

    # Define grid:
    # - Top histogram takes 20% height
    # - Main hexbin takes 80% height
    # - Right histogram takes 20% width
    # - Left main takes 80% width
    # Add space between panels
    gs = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[0.2, 0.8],
        width_ratios=[0.8, 0.2],
        hspace=0.2,
        wspace=0.2,
    )

    # Main hexbin plot (bottom-left)
    ax_main = fig.add_subplot(gs[1, 0])

    # Right histogram (bottom-right) - temporal index distribution
    ax_right = fig.add_subplot(gs[1, 1])

    # Top histogram (top-left) - spatial index distribution
    ax_top = fig.add_subplot(gs[0, 0])

    # Top-right corner is empty
    ax_empty = fig.add_subplot(gs[0, 1])
    ax_empty.axis("off")

    # Create density scatter plot (hexbin) in main axis
    hb = ax_main.hexbin(
        sindex_tracker,
        tindex_tracker,
        gridsize=100,
        extent=[min_sindex - 0.5, max_sindex + 0.5, min_time, max_time],
        cmap="jet",
        bins="log",
        mincnt=1,
        edgecolors="none",
    )

    ax_main.set_xlabel("Spatial Index (Processor Rank)")
    ax_main.set_ylabel("Temporal Index")
    ax_main.set_xlim(min_sindex - 0.5, max_sindex + 0.5)
    ax_main.set_ylim(min_time, max_time)
    ax_main.grid(True, alpha=0.3, linestyle="--")

    # Right plot: Histogram of temporal index distribution (horizontal bars)
    unique_tindices = np.arange(min_time, max_time + 1)
    temporal_counts = [np.sum(tindex_tracker == t) for t in unique_tindices]

    ax_right.barh(
        unique_tindices,
        temporal_counts,
        height=0.8,
        color="coral",
        alpha=0.7,
        edgecolor="black",
    )
    ax_right.set_xlabel("Frequency")
    ax_right.set_ylim(min_time, max_time)
    ax_right.grid(True, alpha=0.3, linestyle="--", axis="x")
    ax_right.tick_params(axis="both")

    # Top plot: Histogram of spatial index distribution (vertical bars)
    unique_sindices = np.arange(min_sindex, max_sindex + 1)
    spatial_counts = [np.sum(sindex_tracker == s) for s in unique_sindices]

    ax_top.bar(
        unique_sindices,
        spatial_counts,
        width=0.8,
        color="steelblue",
        alpha=0.7,
        edgecolor="black",
    )
    ax_top.set_ylabel("Frequency")
    ax_top.set_xlim(min_sindex - 0.5, max_sindex + 0.5)
    ax_top.grid(True, alpha=0.3, linestyle="--", axis="y")
    ax_top.tick_params(axis="both")

    # Add colorbar at the bottom, spanning the width of the main plot
    # Get the position of the main plot
    main_pos = ax_main.get_position()
    # Create colorbar axes below the main plot, with same width
    cbar_ax = fig.add_axes([main_pos.x0, main_pos.y0 - 0.1, main_pos.width, 0.02])
    cbar = fig.colorbar(hb, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(r"$\log_{10}[\mathrm{Count}]$")

    # Save figure
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{filename}_{mode}.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    if logger:
        logger.info(
            f"Saved density scatter plot with marginal histograms to: {save_path}"
        )
    else:
        print(f"Saved density scatter plot with marginal histograms to: {save_path}")
