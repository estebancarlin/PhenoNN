# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors:
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""
Linear regression baselines for LAI prediction.

This module provides simple linear models that serve as reference baselines
for evaluating more complex neural network architectures. Both models map
meteorological feature sequences to a single LAI prediction while maintaining
the same input/output interface as the project's single-day prediction models.

Implemented Models
------------------
LinearBaseline
    A full-window linear regression model that flattens the entire input
    sequence and applies a single fully connected layer. Each feature at
    each timestep receives an independent weight, allowing the model to
    learn temporally specific linear relationships.

    Mathematically:

        y = w^T x + b

    where ``x`` is the flattened ``(C × L)`` feature window.

    This model is equivalent to ordinary least squares regression over all
    feature–time combinations and provides a strong linear benchmark.

PerDayLinearBaseline
    A per-timestep linear model that ignores temporal history. The model
    applies a shared linear mapping from features to LAI and returns the
    prediction associated with the final timestep of the input sequence.

    This baseline evaluates how much predictive information is contained
    in the current day's meteorological conditions alone. Strong performance
    indicates that temporal context contributes little beyond instantaneous
    feature values.

Input and Output Conventions
----------------------------
Both models follow the project's single-day prediction interface:

Input:
    Tensor of shape ``(batch_size, feature_channels, sequence_length)``.

Output:
    Tensor of shape ``(batch_size, 1)``.

Notes
-----
* Neither model contains hidden layers, nonlinear activations, recurrence,
  attention mechanisms, or convolutional operations.
* The models are intentionally simple and interpretable, making them useful
  as lower-bound performance baselines.
* Performance gains achieved by more sophisticated architectures can be
  interpreted relative to these linear reference models.
"""

import torch
import torch.nn as nn


class LinearBaseline(nn.Module):
    """
    Full-window linear regression baseline.

    Flattens the entire (feature_channel × seq_length) input into one vector,
    applies a single Linear layer to produce a scalar prediction.

    This is mathematically equivalent to:
        gcc_pred = w @ flatten(input) + b

    where w has C×L weights. It can learn things like "temperature on day 300
    matters more than temperature on day 50" because each (channel, timestep)
    pair gets its own weight.

    Parameters
    ----------
    feature_channel : int
        Number of input feature channels.
    seq_length : int
        Window length in days (365).

    Examples
    --------
    >>> model = LinearBaseline(feature_channel=16, seq_length=365)
    >>> x = torch.randn(32, 16, 365)
    >>> y = model(x)
    >>> y.shape
    torch.Size([32, 1])
    """

    def __init__(self, feature_channel: int, seq_length: int = 365):
        super().__init__()
        self.feature_channel = feature_channel
        self.seq_length = seq_length
        self.linear = nn.Linear(feature_channel * seq_length, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, feature_channel, seq_length)

        Returns
        -------
        (batch, 1)
        """
        # Flatten: (B, C, L) → (B, C*L)
        x = x.reshape(x.size(0), -1)
        return self.linear(x)


class PerDayLinearBaseline(nn.Module):
    """
    Per-day linear regression baseline (no temporal context).

    Applies the same Linear(C, 1) to every timestep independently, then
    returns the prediction for the last day. This tests whether today's
    feature values alone (without any history) can predict today's LAI.

    If this baseline scores well, it means the temporal context from the
    365-day window isn't adding much — the model is mostly using the
    current day's meteorology.

    Parameters
    ----------
    feature_channel : int
        Number of input feature channels.

    Examples
    --------
    >>> model = PerDayLinearBaseline(feature_channel=16)
    >>> x = torch.randn(32, 16, 365)
    >>> y = model(x)
    >>> y.shape
    torch.Size([32, 1])
    """

    def __init__(self, feature_channel: int):
        super().__init__()
        self.linear = nn.Linear(feature_channel, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, feature_channel, seq_length)

        Returns
        -------
        (batch, 1)
        """
        # Permute to (B, L, C), take last day, apply linear
        x = x.permute(0, 2, 1)  # (B, L, C)
        last_day = x[:, -1, :]  # (B, C)
        return self.linear(last_day)
