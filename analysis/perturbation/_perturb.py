"""
Physical-space input perturbation for trained PhenoNN models.

A ``Perturbation`` installs a forward-pre-hook on a built model, exactly like
the feature-ablation study. It does not change ``phenonn/``. For each targeted
meteo channel the hook receives the already-normalized input tensor
``x`` with shape ``(B, C, L)`` and performs this sequence:

1. Convert normalized values to physical units. Invert the z-score and apply
   ``expm1`` to log1p features.
2. Apply the requested change in physical units: ``physical = physical * mul +
   add``.
3. Convert back to normalized units with log1p and z-score.

A ``+5 C`` or ``x0.5 precipitation`` change is therefore expressed in real
units, and only the requested channels move. VPD, PET, radiation, and the other
channels are held fixed. An optional day-of-year window restricts the change to
a season, for example summer for a heatwave or winter for frost. Positions
outside the window keep their original value.

The 720 or 730 day input window ends on DOY 365 of the target year, so position
``i`` maps to a DOY on a 365 day calendar with the last position equal to DOY
365. See :func:`doy_of_positions`.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from phenonn.utils.config import ALL_FEATURES, LOG_TRANSFORM_FEATURES

__all__ = [
    "SEASONS",
    "ChannelPerturbation",
    "Perturbation",
    "doy_of_positions",
    "season_mask",
]


def doy_of_positions(seq_length: int) -> np.ndarray:
    """Return the 1-based day of year for each input-window position.

    Parameters
    ----------
    seq_length : int
        Number of daily positions in the model input window.

    Returns
    -------
    numpy.ndarray
        Integer array of shape ``(seq_length,)``. The last position is DOY 365,
        and earlier positions wrap backwards on a 365 day calendar.
    """
    back = (seq_length - 1 - np.arange(seq_length)) % 365
    return 365 - back  # 1 (oldest wrap) .. 365 (last day)


def season_mask(
    seq_length: int,
    ranges: Sequence[tuple[int, int]] | None,
) -> np.ndarray:
    """Build a boolean mask for one or more inclusive day-of-year ranges.

    Parameters
    ----------
    seq_length : int
        Number of daily positions in the model input window.
    ranges : sequence of tuple of int, optional
        Inclusive ``(start, end)`` day-of-year ranges. A range with
        ``start > end`` wraps across the new year, for example
        ``(335, 59)`` for December to February. ``None`` or an empty sequence
        selects the whole window.

    Returns
    -------
    numpy.ndarray
        Boolean array of shape ``(seq_length,)`` that is true inside any of the
        requested ranges.
    """
    if not ranges:
        return np.ones(seq_length, dtype=bool)
    doy = doy_of_positions(seq_length)
    mask = np.zeros(seq_length, dtype=bool)
    for lo, hi in ranges:
        if lo <= hi:
            mask |= (doy >= lo) & (doy <= hi)
        else:  # Wrap around December to January.
            mask |= (doy >= lo) | (doy <= hi)
    return mask


# Named DOY windows (non-leap calendar) for the seasonal scenarios.
SEASONS = {
    "year": None,
    "summer": [(152, 243)],  # Jun 1 to Aug 31 (JJA)
    "winter": [(335, 59)],  # Dec 1 to Feb 28 (DJF, wraps)
    "spring": [(60, 151)],  # Mar 1 to May 31 (MAM)
    "autumn": [(244, 334)],  # Sep 1 to Nov 30 (SON)
    "growing": [(91, 304)],  # Apr 1 to Oct 31
}


class ChannelPerturbation:
    """One channel's change in physical units.

    The new value is ``physical * mul + add``.

    Parameters
    ----------
    feature : str
        Feature name that must appear in ``ALL_FEATURES``.
    add : float, optional
        Additive change in physical units. The default is ``0.0``.
    mul : float, optional
        Multiplicative change in physical units. The default is ``1.0``.
    """

    def __init__(self, feature: str, add: float = 0.0, mul: float = 1.0) -> None:
        if feature not in ALL_FEATURES:
            raise SystemExit(f"unknown feature {feature!r}; not in ALL_FEATURES")
        self.feature = feature
        self.ch = ALL_FEATURES.index(feature)
        self.add = float(add)
        self.mul = float(mul)
        self.is_log = feature in LOG_TRANSFORM_FEATURES


class Perturbation:
    """A set of channel changes and an optional seasonal window.

    The object can be installed as a forward-pre-hook on a built model. It
    stores one mask per input length and device on first use.

    Parameters
    ----------
    channels : sequence of ChannelPerturbation
        Channel changes to apply. At least one is required.
    norm_stats : mapping or None
        Per-feature normalization statistics. Use ``None`` when the model
        receives raw physical inputs.
    season_ranges : sequence of tuple of int, optional
        Inclusive day-of-year ranges for the change. ``None`` applies it to the
        whole window.
    """

    def __init__(self, channels, norm_stats, season_ranges=None) -> None:
        if not channels:
            raise SystemExit("Perturbation needs at least one channel.")
        self.channels = list(channels)
        self.norm_stats = norm_stats  # None means the inputs are raw.
        self.season_ranges = season_ranges
        self._mask_cache = {}

    def _mask(self, seq_length: int, device) -> torch.Tensor:
        """Return the seasonal mask for one input length and device."""
        if seq_length not in self._mask_cache:
            mask = season_mask(seq_length, self.season_ranges).astype(np.float32)
            self._mask_cache[seq_length] = torch.from_numpy(mask)
        return self._mask_cache[seq_length].to(device)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Return a perturbed copy of the normalized input ``x``.

        Parameters
        ----------
        x : torch.Tensor
            Normalized input tensor with shape ``(B, C, L)``.

        Returns
        -------
        torch.Tensor
            A new tensor with the same shape as ``x``. Channels not listed in
            ``channels`` are unchanged.
        """
        seq_length = x.shape[2]
        mask = self._mask(seq_length, x.device)  # (L,)
        keep = 1.0 - mask
        x = x.clone()
        for channel_perturbation in self.channels:
            z = x[:, channel_perturbation.ch, :]  # (B, L), normalized
            if self.norm_stats is not None:
                stats = self.norm_stats.get(channel_perturbation.feature)
                if stats is None:
                    raise SystemExit(
                        f"norm_stats has no entry for {channel_perturbation.feature!r}."
                    )
                mean = float(stats["mean"])
                std = max(float(stats["std"]), 1e-8)
                if channel_perturbation.is_log:
                    physical = torch.expm1(z * std + mean)
                else:
                    physical = z * std + mean
            else:  # Raw inputs: z is already physical.
                mean, std = 0.0, 1.0
                physical = z

            physical_new = (
                physical * channel_perturbation.mul + channel_perturbation.add
            )
            if self.norm_stats is not None and channel_perturbation.is_log:
                z_new = (torch.log1p(physical_new.clamp(min=0.0)) - mean) / std
            elif self.norm_stats is not None:
                z_new = (physical_new - mean) / std
            else:
                z_new = physical_new
            x[:, channel_perturbation.ch, :] = z * keep + z_new * mask
        return x

    def hook(self, _module, inputs):
        """Forward-pre-hook that applies the perturbation to the first input."""
        tensor = inputs[0]
        if not torch.is_tensor(tensor):
            return None
        return (self.apply(tensor),) + tuple(inputs[1:])

    def describe(self) -> str:
        """Return a short human-readable description of the perturbation."""
        season = "year" if not self.season_ranges else str(self.season_ranges)
        parts = []
        for channel_perturbation in self.channels:
            operations = []
            if channel_perturbation.mul != 1.0:
                operations.append(f"x{channel_perturbation.mul:g}")
            if channel_perturbation.add != 0.0:
                operations.append(f"{channel_perturbation.add:+g}")
            description = " ".join(operations) or "(no-op)"
            parts.append(f"{channel_perturbation.feature} {description}")
        return f"[{'; '.join(parts)}] over {season}"
