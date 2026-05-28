# Model wrapper for single-day prediction.
#
# The RTnn models all output (batch, output_channel, seq_length).
# For GCC prediction we want (batch, 1) — the value on the last day
# of the input window. This wrapper extracts that last timestep
# without modifying any of the existing model code.

import datetime
import torch
import torch.nn as nn


def _build_obs_positions() -> list:
    """
    0-indexed positions within a 365-day (non-leap) year for days 5, 15, 25
    of every month — the 36 days on which LAI observations are available.
    """
    positions = []
    for month in range(1, 13):
        for day in [5, 15, 25]:
            doy = datetime.date(2001, month, day).timetuple().tm_yday
            positions.append(doy - 1)
    return positions


_OBS_POSITIONS = _build_obs_positions()  # 36 positions


class SingleDayWrapper(nn.Module):
    """
    Wraps any RTnn model to output a single scalar per sample.

    Takes the last timestep of the wrapped model's sequence output,
    yielding shape (batch, output_channel) instead of
    (batch, output_channel, seq_length).

    Parameters
    ----------
    base_model : nn.Module
        Any model with forward signature (batch, C_in, L) -> (batch, C_out, L).

    Examples
    --------
    >>> from rtnn.models.rnn import RNN_LSTM
    >>> base = RNN_LSTM(feature_channel=14, output_channel=1,
    ...                 hidden_size=128, num_layers=2)
    >>> model = SingleDayWrapper(base)
    >>> x = torch.randn(32, 14, 365)
    >>> y = model(x)
    >>> y.shape
    torch.Size([32, 1])
    """

    def __init__(self, base_model: nn.Module) -> None:
        super().__init__()
        self.base_model = base_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, feature_channels, seq_length).

        Returns
        -------
        torch.Tensor
            Output of shape (batch, output_channel).
        """
        # Full sequence output: (batch, output_channel, seq_length)
        out = self.base_model(x)
        # Take last timestep: (batch, output_channel)
        return out[:, :, -1]

class permuteWrapper(nn.Module):
    """
    Permutes input dimensions to match expected order for RTnn models.

    RTnn models expect input shape (batch, feature_channels, seq_length).
    If your data is in (batch, seq_length, feature_channels), this wrapper
    permutes the dimensions before passing to the base model.

    Parameters
    ----------
    base_model : nn.Module
        Any model with forward signature (batch, C_in, L) -> (batch, C_out, L).

    Examples
    --------
    >>> from rtnn.models.rnn import RNN_LSTM
    >>> base = RNN_LSTM(feature_channel=14, output_channel=1,
    ...                 hidden_size=128, num_layers=2)
    >>> model = permuteWrapper(base)
    >>> x = torch.randn(32, 365, 14)  # Note seq_length and feature_channels swapped
    >>> y = model(x)
    >>> y.shape
    torch.Size([32, 1])
    """
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        # (B, features, seq) → (B, seq, features)
        x = x.permute(0, 2, 1)
        out = self.base_model(x)
        # (B, seq, output) → (B, output, seq)
        return out.permute(0, 2, 1)


class LastNDaysWrapper(nn.Module):
    """
    Wraps any RTnn model to output the last N timesteps.
 
    Takes the last N timesteps of the wrapped model's sequence output,
    yielding shape (batch, output_channel, N) instead of
    (batch, output_channel, seq_length).
 
    Used for gradient-aware loss where we need GCC(t) and GCC(t-1)
    while keeping the full input window for context.
 
    Parameters
    ----------
    base_model : nn.Module
        Any model with forward signature (batch, C_in, L) -> (batch, C_out, L).
    n_days : int
        Number of trailing timesteps to keep (default: 2).
 
    Examples
    --------
    >>> base = RNN_LSTM(feature_channel=16, output_channel=1,
    ...                 hidden_size=128, num_layers=2)
    >>> model = LastNDaysWrapper(base, n_days=2)
    >>> x = torch.randn(32, 16, 365)
    >>> y = model(x)
    >>> y.shape
    torch.Size([32, 1, 2])
    """
 
    def __init__(self, base_model: nn.Module, n_days: int = 2) -> None:
        super().__init__()
        self.base_model = base_model
        self.n_days = n_days
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, feature_channels, seq_length)
 
        Returns
        -------
        (batch, output_channel, n_days)
        """
        out = self.base_model(x)          # (B, C_out, L)
        return out[:, :, -self.n_days:]    # (B, C_out, n_days)


class Every10DaysWrapper(nn.Module):
    """
    Wraps any RTnn model to output predictions at the 36 LAI observation days.

    Extracts the last 365 timesteps of the model output, then selects the
    36 positions corresponding to days 5, 15, 25 of each month in a
    non-leap year.

    Input  : (B, C_in, L)  with L ≥ 365
    Output : (B, C_out, 36)

    Parameters
    ----------
    base_model : nn.Module
        Any model with forward signature (B, C_in, L) -> (B, C_out, L).

    Examples
    --------
    >>> base = RNN_LSTM(feature_channel=31, output_channel=1,
    ...                 hidden_size=128, num_layers=2)
    >>> model = Every10DaysWrapper(base)
    >>> x = torch.randn(32, 31, 720)
    >>> y = model(x)
    >>> y.shape
    torch.Size([32, 1, 36])
    """

    def __init__(self, base_model: nn.Module) -> None:
        super().__init__()
        self.base_model = base_model
        pos = torch.tensor(_OBS_POSITIONS, dtype=torch.long)
        self.register_buffer("obs_positions", pos)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, feature_channels, seq_length)  — seq_length ≥ 365

        Returns
        -------
        (batch, output_channel, 36)
        """
        out = self.base_model(x)                       # (B, C_out, L)
        last_year = out[:, :, -365:]                   # (B, C_out, 365)
        return last_year[:, :, self.obs_positions]     # (B, C_out, 36)