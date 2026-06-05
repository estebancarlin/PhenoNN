# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Stefan Barbu, Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""
Evaluation utilities for RTnn model assessment.

This module provides comprehensive evaluation tools for radiative transfer
neural network models, including custom loss functions, metric computation,
and visualization helpers.

The module includes:
- Custom loss functions (NMSE, NMAE, combined MSE-MAE, LogCosh, Weighted MSE)
- Metric calculators for evaluation (MSE, MAE, MBE, R², NMSE, NMAE, MARE, GMRAE)
- Data normalization/de-normalization utilities
- Absorption rate calculations
- Main evaluation loop for LSM models

Dependencies
------------
torch : For tensor operations and loss functions
numpy : For numerical operations
plot_helper : For visualization utilities
"""

import torch
import torch.nn as nn
import numpy as np
import sys
from tqdm import tqdm
import os
from phenonn.utils.diagnosticRTNN import (
    plot_flux_and_abs_lines,
    plot_flux_and_abs,
)

sys.path.append("..")


class GradientAwareLoss(nn.Module):
    """
    MSE + temporal gradient penalty on the last N predicted days.

    Combines two terms:
        L = MSE(ŷ, y) + λ * MSE(Δŷ, Δy)

    where Δ denotes the discrete temporal difference between consecutive days:
        Δy = y[..., 1:] - y[..., :-1]

    Designed for n_target_days=2: the model predicts [GCC(t-1), GCC(t)]
    and the loss penalises both the values and the day-to-day change.

    Parameters
    ----------
    grad_weight : float
        Weight λ for the gradient term. 0 = pure MSE.
        Typical values: 0.1–1.0.
    base_loss : str
        Base reconstruction loss: 'mse' or 'huber'.
    huber_delta : float
        Delta for Huber loss (only used if base_loss='huber').

    Input shapes
    ------------
    pred : torch.Tensor
        Shape (batch, 1, N) — last N days predictions (typically N=2)
    target : torch.Tensor
        Shape (batch, 1, N) — last N days targets

    Examples
    --------
    >>> criterion = GradientAwareLoss(grad_weight=0.5)
    >>> pred = torch.randn(32, 1, 2)     # [ŷ(t-1), ŷ(t)]
    >>> target = torch.randn(32, 1, 2)   # [y(t-1), y(t)]
    >>> loss = criterion(pred, target)
    """

    def __init__(self, grad_weight=0.5, base_loss="mse", huber_delta=1.0):
        super().__init__()
        self.grad_weight = grad_weight
        if base_loss == "huber":
            self.base = nn.HuberLoss(delta=huber_delta)
        else:
            self.base = nn.MSELoss()

    def forward(self, pred, target):
        # Base reconstruction loss on all N target days
        loss_recon = self.base(pred, target)

        if self.grad_weight <= 0:
            return loss_recon

        # Temporal gradient: Δy = y(t) - y(t-1)
        # pred/target shape: (B, 1, N) with N >= 2
        grad_pred = pred[:, :, 1:] - pred[:, :, :-1]  # (B, 1, N-1)
        grad_target = target[:, :, 1:] - target[:, :, :-1]

        loss_grad = self.base(grad_pred, grad_target)

        return (1 - self.grad_weight) * loss_recon + self.grad_weight * loss_grad


class NMSELoss(nn.Module):
    """
    Normalized Mean Squared Error Loss.

    Computes MSE normalized by the mean square of the target values.
    Useful when the scale of the target variable varies.

    Parameters
    ----------
    eps : float, optional
        Small constant for numerical stability. Default is 1e-8.

    Examples
    --------
    >>> criterion = NMSELoss()
    >>> loss = criterion(predictions, targets)
    """

    def __init__(self, eps=1e-8):
        super(NMSELoss, self).__init__()
        self.eps = eps
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        mse = self.mse(pred, target)
        norm = torch.mean(target**2) + self.eps
        return mse / norm


class NMAELoss(nn.Module):
    """
    Normalized Mean Absolute Error Loss.

    Computes MAE normalized by the mean absolute value of the target.
    Provides a scale-invariant error metric.

    Parameters
    ----------
    eps : float, optional
        Small constant for numerical stability. Default is 1e-8.
    """

    def __init__(self, eps=1e-8):
        super(NMAELoss, self).__init__()
        self.eps = eps
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        mae = self.l1(pred, target)
        norm = torch.mean(torch.abs(target)) + self.eps
        return mae / norm


class MetricTracker:
    """
    A utility class for tracking and computing statistics of metric values.

    This class maintains a running average of metric values and provides
    methods to compute mean and root mean squared values.

    Attributes
    ----------
    value : float
        Cumulative weighted sum of metric values
    count : int
        Total number of samples processed

    Examples
    --------
    >>> tracker = MetricTracker()
    >>> tracker.update(10.0, 5)  # value=10.0, count=5 samples
    >>> tracker.update(20.0, 3)  # value=20.0, count=3 samples
    >>> print(tracker.getmean())  # (10*5 + 20*3) / (5+3) = 110/8 = 13.75
    13.75
    >>> print(tracker.getsqrtmean())  # sqrt(13.75)
    3.7080992435478315
    """

    def __init__(self):
        """
        Initialize MetricTracker with zero values.
        """
        self.reset()

    def reset(self):
        """
        Reset all tracked values to zero.

        Returns
        -------
        None
        """
        self.value = 0.0
        self.count = 0
        self.value_sq = 0.0

    def update(self, value, count):
        """
        Update the tracker with new metric values.

        Parameters
        ----------
        value : float
            The metric value to add
        count : int
            Number of samples this value represents (weight)

        Returns
        -------
        None
        """
        self.count += count
        self.value += value * count
        self.value_sq += (value**2) * count

    def getmean(self):
        """
        Calculate the mean of all tracked values.

        Returns
        -------
        float
            Weighted mean of all values: total_value / total_count

        Raises
        ------
        ZeroDivisionError
            If no values have been added (count == 0)
        """
        if self.count == 0:
            raise ZeroDivisionError("Cannot compute mean with zero samples")
        return self.value / self.count

    def getstd(self):
        """
        Calculate the standard deviation of all tracked values.

        Returns
        -------
        float
            Weighted standard deviation of all values:
            sqrt(E(x^2) - (E(x))^2)

        Raises
        ------
        ZeroDivisionError
            If no values have been added (count == 0)
        """
        if self.count == 0:
            raise ZeroDivisionError("Cannot compute std with zero samples")
        mean = self.getmean()
        variance = self.value_sq / self.count - mean**2
        return np.sqrt(max(variance, 0.0))  # numerical safety

    def getsqrtmean(self):
        """
        Calculate the square root of the mean of all tracked values.

        Returns
        -------
        float
            Square root of the weighted mean: sqrt(total_value / total_count)

        Raises
        ------
        ZeroDivisionError
            If no values have been added (count == 0)
        """
        return np.sqrt(self.getmean())


def get_loss_function(loss_type, args, logger=None):
    """
    Factory function to instantiate the requested loss function.

    Parameters
    ----------
    loss_type : str
        Type of loss function. Options:
        - 'mse': Mean Squared Error
        - 'mae': Mean Absolute Error
        - 'nmae': Normalized Mean Absolute Error
        - 'nmse': Normalized Mean Squared Error
        - 'wmse': Weighted Mean Squared Error
        - 'logcosh': Log-Cosh loss
        - 'smoothl1': Smooth L1 Loss (Huber-like)
        - 'huber': Huber Loss
    args : argparse.Namespace
        Arguments containing loss-specific parameters (e.g., beta_delta for Huber).

    Returns
    -------
    torch.nn.Module
        Initialized loss function.

    Raises
    ------
    ValueError
        If loss_type is not supported or required parameters are missing.

    Examples
    --------
    >>> args = argparse.Namespace(beta_delta=1.0)
    >>> criterion = get_loss_function('huber', args)
    """
    if loss_type == "mse":
        if logger:
            logger.info("Using Mean Squared Error (MSE) loss")
        return nn.MSELoss()
    elif loss_type == "rmse":
        if logger:
            logger.info("Using Root Mean Squared Error (RMSE) loss")
        return lambda pred, target: torch.sqrt(nn.MSELoss()(pred, target))
    elif loss_type == "mae":
        if logger:
            logger.info("Using Mean Absolute Error (MAE) loss")
        return nn.L1Loss()
    elif loss_type == "nmae":
        if logger:
            logger.info("Using Normalized Mean Absolute Error (NMAE) loss")
        return NMAELoss()
    elif loss_type == "nmse":
        if logger:
            logger.info("Using Normalized Mean Squared Error (NMSE) loss")
        return NMSELoss()
    elif loss_type in ["smoothl1", "huber"]:
        if not hasattr(args, "beta_delta"):
            raise ValueError(f"{loss_type.capitalize()}Loss requires --beta_delta")
        if logger:
            logger.info(
                f"Using {loss_type.capitalize()} loss with delta={args.beta_delta}"
            )
        return (
            nn.SmoothL1Loss(beta=args.beta_delta)
            if loss_type == "smoothl1"
            else nn.HuberLoss(delta=args.beta_delta)
        )
    elif loss_type == "gradient":
        grad_w = getattr(args, "gradient_loss_weight", 0.5)
        base = getattr(args, "gradient_base_loss", "mse")
        if logger:
            logger.info(f"Using GradientAwareLoss (λ={grad_w}, base={base})")
        return GradientAwareLoss(grad_weight=grad_w, base_loss=base)

    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")


def mse_all(pred, true):
    """
    Compute Mean Squared Error.

    Parameters
    ----------
    pred : torch.Tensor
        Predictions.
    true : torch.Tensor
        Ground truth.

    Returns
    -------
    tuple
        (num_elements, mse_value)
    """
    return pred.numel(), torch.mean((pred - true) ** 2)


def mbe_all(pred, true):
    """
    Compute Mean Bias Error.

    Parameters
    ----------
    pred : torch.Tensor
        Predictions.
    true : torch.Tensor
        Ground truth.

    Returns
    -------
    tuple
        (num_elements, mbe_value)
    """
    return pred.numel(), torch.mean(pred - true)


def mae_all(pred, true):
    """
    Compute Mean Absolute Error.

    Parameters
    ----------
    pred : torch.Tensor
        Predictions.
    true : torch.Tensor
        Ground truth.

    Returns
    -------
    tuple
        (num_elements, mae_value)
    """
    return pred.numel(), torch.mean(torch.abs(pred - true))


def r2_all(pred, true):
    """
    Calculate R2 (coefficient of determination) between predicted and true values.

    Computes the R2 metric and returns both the number of elements and
    the R2 value.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted values from the model
    true : torch.Tensor
        Ground truth values

    Returns
    -------
    tuple
        (num_elements, r2_value) where:
        - num_elements (int): Total number of elements in the tensors
        - r2_value (torch.Tensor): R2 score

    Notes
    -----
    R2 is calculated as:

        R2 = 1 - sum((true - pred)^2) / sum((true - mean(true))^2)

    This implementation is fully torch-based and works on CPU and GPU.
    """

    if pred.shape != true.shape:
        raise RuntimeError(f"Shape mismatch: pred {pred.shape} vs true {true.shape}")

    eps = 1e-12  # Small value to avoid division by zero when variance is zero
    num_elements = pred.numel()

    # Flatten
    pred_flat = pred.reshape(-1)
    true_flat = true.reshape(-1)

    # Residual sum of squares
    ss_res = torch.sum((true_flat - pred_flat) ** 2)

    # Total sum of squares
    true_mean = torch.mean(true_flat)
    ss_tot = torch.sum((true_flat - true_mean) ** 2)

    # R2 score
    r2_value = 1.0 - ss_res / (ss_tot + eps)

    return num_elements, r2_value


def nmae_all(pred, true):
    """
    Compute Normalized Mean Absolute Error.

    Parameters
    ----------
    pred : torch.Tensor
        Predictions.
    true : torch.Tensor
        Ground truth.

    Returns
    -------
    tuple
        (num_elements, nmae_value)
    """
    mae = torch.mean(torch.abs(pred - true))
    norm = torch.mean(torch.abs(true)) + 1e-8
    nmae = mae / norm
    return pred.numel(), nmae


def nmse_all(pred, true):
    """
    Compute Normalized Mean Squared Error.

    Parameters
    ----------
    pred : torch.Tensor
        Predictions.
    true : torch.Tensor
        Ground truth.

    Returns
    -------
    tuple
        (num_elements, nmse_value)
    """
    mse = torch.mean((pred - true) ** 2)
    norm = torch.mean(true**2) + 1e-8
    nmse = mse / norm
    return pred.numel(), nmse


def mare_all(pred, true):
    """
    Compute Mean Absolute Relative Error.

    Parameters
    ----------
    pred : torch.Tensor
        Predictions.
    true : torch.Tensor
        Ground truth.

    Returns
    -------
    tuple
        (num_elements, mare_value)
    """
    relative_error = torch.abs(pred - true) / (torch.abs(true) + 1e-8)
    mare = torch.mean(relative_error)
    return pred.numel(), mare


def gmrae_all(pred, true):
    """
    Compute Geometric Mean Relative Absolute Error.

    Parameters
    ----------
    pred : torch.Tensor
        Predictions.
    true : torch.Tensor
        Ground truth.

    Returns
    -------
    tuple
        (num_elements, gmrae_value)
    """
    eps = 1e-8
    relative_errors = torch.abs(pred - true) / (torch.abs(true) + eps)
    log_rel_errors = torch.log(relative_errors + eps)
    gmrae = torch.exp(torch.mean(log_rel_errors))
    return pred.numel(), gmrae


def unnorm_mpas(pred, targ, norm_mapping, normalization_type, idxmap):
    """
    Reverse normalization to recover original physical values.

    Applies inverse transformation based on the normalization method used
    during preprocessing.

    Parameters
    ----------
    pred : torch.Tensor
        Normalized predictions of shape (batch, channels, seq_length).
    targ : torch.Tensor
        Normalized targets.
    norm_mapping : dict
        Dictionary containing normalization statistics for each variable.
    normalization_type : dict
        Dictionary mapping variable names to normalization types.
    idxmap : dict
        Dictionary mapping channel indices to variable names.

    Returns
    -------
    tuple
        (unnormalized_predictions, unnormalized_targets)

    Supported normalization types:
        - minmax: x * (max - min) + min
        - standard: x * std + mean
        - robust: x * iqr + median
        - log1p_*: expm1(x * scale + offset)
        - sqrt_*: (x * scale + offset) ** 2

    Examples
    --------
    >>> idxmap = {0: 'collim_alb', 1: 'collim_tran'}
    >>> upred, utarg = unnorm_mpas(pred, targ, norm_mapping, norm_type, idxmap)
    """
    device = pred.device
    upred = torch.zeros_like(pred, device=device)
    utarg = torch.zeros_like(targ, device=device)

    for i, var_name in idxmap.items():
        norm_type = normalization_type.get(var_name, "log1p_minmax")
        norm = norm_mapping[var_name]

        if norm_type == "standard":
            mean = norm["vmean"]
            std = norm["vstd"]
            upred[:, i, :] = pred[:, i, :] * std + mean
            utarg[:, i, :] = targ[:, i, :] * std + mean

        elif norm_type == "minmax":
            vmin = norm["vmin"]
            vmax = norm["vmax"]
            upred[:, i, :] = pred[:, i, :] * (vmax - vmin) + vmin
            utarg[:, i, :] = targ[:, i, :] * (vmax - vmin) + vmin

        elif norm_type == "robust":
            median = norm["median"]
            iqr = norm["iqr"]
            upred[:, i, :] = pred[:, i, :] * iqr + median
            utarg[:, i, :] = targ[:, i, :] * iqr + median

        elif norm_type == "log1p_minmax":
            log_min = norm["log_min"]
            log_max = norm["log_max"]
            unnorm_pred = pred[:, i, :] * (log_max - log_min) + log_min
            unnorm_targ = targ[:, i, :] * (log_max - log_min) + log_min
            upred[:, i, :] = torch.expm1(unnorm_pred)
            utarg[:, i, :] = torch.expm1(unnorm_targ)

        elif norm_type == "log1p_standard":
            mean = norm["log_mean"]
            std = norm["log_std"]
            unnorm_pred = pred[:, i, :] * std + mean
            unnorm_targ = targ[:, i, :] * std + mean
            upred[:, i, :] = torch.expm1(unnorm_pred)
            utarg[:, i, :] = torch.expm1(unnorm_targ)

        elif norm_type == "log1p_robust":
            median = norm["log_median"]
            iqr = norm["log_iqr"]
            unnorm_pred = pred[:, i, :] * iqr + median
            unnorm_targ = targ[:, i, :] * iqr + median
            upred[:, i, :] = torch.expm1(unnorm_pred)
            utarg[:, i, :] = torch.expm1(unnorm_targ)

        elif norm_type == "sqrt_minmax":
            sqrt_min = norm["sqrt_min"]
            sqrt_max = norm["sqrt_max"]
            unnorm_pred = pred[:, i, :] * (sqrt_max - sqrt_min) + sqrt_min
            unnorm_targ = targ[:, i, :] * (sqrt_max - sqrt_min) + sqrt_min
            upred[:, i, :] = unnorm_pred**2
            utarg[:, i, :] = unnorm_targ**2

        elif norm_type == "sqrt_standard":
            mean = norm["sqrt_mean"]
            std = norm["sqrt_std"]
            unnorm_pred = pred[:, i, :] * std + mean
            unnorm_targ = targ[:, i, :] * std + mean
            upred[:, i, :] = unnorm_pred**2
            utarg[:, i, :] = unnorm_targ**2

        elif norm_type == "sqrt_robust":
            median = norm["sqrt_median"]
            iqr = norm["sqrt_iqr"]
            unnorm_pred = pred[:, i, :] * iqr + median
            unnorm_targ = targ[:, i, :] * iqr + median
            upred[:, i, :] = unnorm_pred**2
            utarg[:, i, :] = unnorm_targ**2
        else:
            raise ValueError(
                f"Unsupported normalization type '{norm_type}' for variable '{var_name}'"
            )

    return upred, utarg


def calc_abs(pred, targ, p=None):
    """
    Calculate absorption rates from flux predictions.

    Computes net absorption rates for two channel groups (1-2 and 3-4) using
    the difference between upwelling and downwelling fluxes.

    Parameters
    ----------
    pred : torch.Tensor
        Predictions of shape (batch, 4, seq_length) where channels 0-1 are
        for first group and 2-3 for second group.
    targ : torch.Tensor
        Targets of same shape as pred.
    p : torch.Tensor, optional
        Pressure levels for atmospheric heating rate calculation.
        If provided, computes heating rate using pressure gradients.

    Returns
    -------
    tuple
        (abs12_pred, abs12_targ, abs34_pred, abs34_targ) where each is a
        tensor of shape (batch, 1, seq_length-1).

    Notes
    -----
    - If p is None: returns d(net) where net = up - down
    - If p is provided: returns heating rate using d(net)/dp
    """
    abs12_pred = calc_hr(pred[:, 0:1, :], pred[:, 1:2, :], p)
    abs12_targ = calc_hr(targ[:, 0:1, :], targ[:, 1:2, :], p)
    abs34_pred = calc_hr(pred[:, 2:3, :], pred[:, 3:4, :], p)
    abs34_targ = calc_hr(targ[:, 2:3, :], targ[:, 3:4, :], p)

    return abs12_pred, abs12_targ, abs34_pred, abs34_targ


def calc_hr(up, down, p=None):
    """
    Calculate heating rate from upwelling and downwelling fluxes.

    Computes the net radiative heating rate by taking the vertical derivative
    of net flux (upwelling - downwelling). If pressure levels are provided,
    calculates the actual atmospheric heating rate using pressure gradients.

    Parameters
    ----------
    up : torch.Tensor
        Upwelling flux tensor of shape (batch, channels, seq_length).
    down : torch.Tensor
        Downwelling flux tensor of shape (batch, channels, seq_length).
    p : torch.Tensor, optional
        Pressure levels of shape (seq_length,) or (batch, seq_length).
        If provided, computes physical heating rate. If None, returns
        the derivative of net flux.

    Returns
    -------
    torch.Tensor
        If p is None:
            Returns the negative derivative of net flux with respect to
            level index: -d(net)/dz (or d(net)/d(level)) of shape
            (batch, channels, seq_length - 1)
        If p is provided:
            Returns the atmospheric heating rate in K/day using the formula:
            heating_rate = (g * 8.64e4) / (cp * 100) * d(net)/dp
            where g = 9.8066 m/s², cp = 1004 J/(kg·K) (calculated as 7*R/2 with R=287)

    Notes
    -----
    - The derivative is computed using finite differences: net[i+1] - net[i]
    - For pressure-based calculation, uses dp = p[i+1] - p[i]
    - The factor 8.64e4 converts from W/m² to K/day
    - The factor 100 converts pressure from hPa to Pa

    Examples
    --------
    >>> # Calculate net flux derivative
    >>> hr = calc_hr(up, down)
    >>> hr.shape
    torch.Size([32, 4, 9])  # for seq_length=10

    >>> # Calculate actual heating rate with pressure levels
    >>> pressure = torch.linspace(1000, 100, 10)  # hPa
    >>> heating_rate = calc_hr(up, down, p=pressure)
    >>> heating_rate.shape
    torch.Size([32, 4, 9])
    """
    net = up - down
    dnet = net - torch.roll(net, 1, 2)

    if p is not None:
        g = 9.8066
        r = 287.0
        cp = 7.0 * r / 2.0
        fac = g * 8.64e4 / (cp * 100)

        dp = p - torch.roll(p, 1, 2)
        return dnet[:, :, 1:] / dp[:, :, 1:] * fac
    else:
        return -dnet[:, :, 1:]


def run_validation(
    loader,
    model,
    norm_mapping,
    normalization_type,
    index_mapping,
    device,
    args,
    epoch,
    logger=None,
    base_dir="./results",
):
    """
    Evaluate model accuracy on LSM dataset.

    Performs comprehensive evaluation including:
    - Loss computation for main fluxes and absorption rates
    - Metric calculation (NMAE, NMSE, R²)
    - Optional plotting of predictions vs targets

    Parameters
    ----------
    loader : torch.utils.data.DataLoader
        Data loader for evaluation dataset.
    model : torch.nn.Module
        Trained model to evaluate.
    norm_mapping : dict
        Normalization statistics for variables.
    normalization_type : dict
        Normalization types per variable.
    index_mapping : dict
        Mapping from channel indices to variable names.
    device : torch.device
        Device to run evaluation on.
    args : argparse.Namespace
        Arguments containing loss type, beta, etc.
    epoch : int
        Current epoch number (for plotting).
    logger : logging.Logger, optional
        Logger for informational messages. If None, no logging is performed.

    Returns
    -------
    tuple
        (valid_loss, valid_metrics) where valid_metrics is a dictionary
        containing computed metrics for fluxes, abs12, and abs34.

    Examples
    --------
    >>> valid_loss, metrics = run_validation(
    ...     test_loader, model, norm_mapping, norm_type, idxmap, device, args, epoch
    ... )
    >>> print(f"Validation loss: {valid_loss:.4f}")
    >>> print(f"NMAE: {metrics['fluxes_NMAE']:.4f}")
    """
    model.eval()

    valid_loss_types = [
        "mse",
        "mae",
        "nmae",
        "nmse",
        "wmse",
        "logcosh",
        "smoothl1",
        "huber",
    ]
    loss_type = args.loss_type.lower()
    assert (
        loss_type in valid_loss_types
    ), f"Invalid loss_type (should be one of {valid_loss_types})"
    func = get_loss_function(loss_type, args)

    metric_names = ["NMAE", "NMSE", "R2"]
    metric_funcs = {"NMAE": nmae_all, "NMSE": nmse_all, "R2": r2_all}
    output_keys = ["fluxes", "abs12", "abs34"]
    valid_metrics = {
        f"{k}_{m}": MetricTracker() for k in output_keys for m in metric_names
    }

    valid_loss = MetricTracker()

    # Progress bar for validation
    loop = tqdm(
        enumerate(loader),
        total=len(loader),
        desc=f"Validation Epoch {epoch}",
        leave=False,
    )

    with torch.no_grad():
        for batch_idx, (feature, targets) in loop:
            feature_shape = feature.shape
            target_shape = targets.shape
            inner_batch_size = feature_shape[0] * feature_shape[1]
            feature = feature.reshape(
                inner_batch_size, feature_shape[2], feature_shape[3]
            ).to(device=device)
            targets = targets.reshape(
                inner_batch_size, target_shape[2], target_shape[3]
            ).to(device=device)

            predicts = model(feature)

            predicts_unnorm, targets_unnorm = unnorm_mpas(
                predicts, targets, norm_mapping, normalization_type, index_mapping
            )
            abs12_predict, abs12_target, abs34_predict, abs34_target = calc_abs(
                predicts_unnorm, targets_unnorm
            )

            output_dict = {
                "fluxes": (predicts, targets),
                "abs12": (abs12_predict, abs12_target),
                "abs34": (abs34_predict, abs34_target),
            }

            for key in output_keys:
                pred, tgt = output_dict[key]
                for metric in metric_names:
                    metric_key = f"{key}_{metric}"
                    if metric_key not in valid_metrics:
                        raise KeyError(
                            f"Metric key '{metric_key}' not found in valid_metrics"
                        )
                    count, value = metric_funcs[metric](pred, tgt)
                    valid_metrics[metric_key].update(value.item(), count)

            main_count, main_val = predicts.numel(), func(predicts, targets)
            abs12_count, abs12_val = (
                abs12_predict.numel(),
                func(abs12_predict, abs12_target),
            )
            abs34_count, abs34_val = (
                abs34_predict.numel(),
                func(abs34_predict, abs34_target),
            )

            weighted_loss = (1.0 - args.beta) * main_val * main_count + args.beta * (
                abs12_val * abs12_count + abs34_val * abs34_count
            )
            total_count = (1.0 - args.beta) * main_count + args.beta * (
                abs12_count + abs34_count
            )
            total_loss = weighted_loss / total_count

            valid_loss.update(total_loss.item(), 1)
            loop.set_postfix(loss=total_loss.item())

            if epoch == args.num_epochs - 1:
                if logger:
                    logger.info(f"Doing plot for batch {batch_idx} in final epoch")
                else:
                    print(f"Doing plot for batch {batch_idx} in final epoch")

                # plot_RTM(predicts_unnorm, targets_unnorm, os.path.join(base_dir, f"Flux{batch_idx}_{args.test_year}.png"))
                # plot_HeatRate(abs12_predict, abs12_target, abs34_predict, abs34_target, os.path.join(base_dir, f"Abs{batch_idx}_{args.test_year}.png"))
                plot_flux_and_abs_lines(
                    predicts_unnorm,
                    targets_unnorm,
                    abs12_predict=abs12_predict,
                    abs12_target=abs12_target,
                    abs34_predict=abs34_predict,
                    abs34_target=abs34_target,
                    filename=os.path.join(
                        base_dir, f"Lineplot_Flux_Abs{batch_idx}_{args.test_year}.png"
                    ),
                )
                plot_flux_and_abs(
                    predicts_unnorm,
                    targets_unnorm,
                    abs12_predict=abs12_predict,
                    abs12_target=abs12_target,
                    abs34_predict=abs34_predict,
                    abs34_target=abs34_target,
                    filename=os.path.join(
                        base_dir, f"flux_abs_hexbin_{batch_idx}_{args.test_year}.png"
                    ),
                )

    return valid_loss.getmean(), {
        k: (tracker.getsqrtmean() if k.lower().endswith("mse") else tracker.getmean())
        for k, tracker in valid_metrics.items()
    }
