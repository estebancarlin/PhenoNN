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
