# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Model architectures for RTnn."""

from .rnn import RNN_LSTM, RNN_GRU
from .fcn import FCN
from .transformer import EncoderTorch
from .transformerbis import (
    SimpleLSTM,
    TransformerNTS,
    SimpleTransformer,
    CombinedModel,
    SimpleNetwork,
    BiTransformer,
)
from .linear_baseline import (
    LinearBaseline,
    PerDayLinearBaseline,
)

__all__ = [
    "RNN_LSTM",
    "RNN_GRU",
    "FCN",
    "EncoderTorch",
    "SimpleLSTM",
    "TransformerNTS",
    "SimpleTransformer",
    "CombinedModel",
    "SimpleNetwork",
    "BiTransformer",
    "LinearBaseline",
    "PerDayLinearBaseline",
]
