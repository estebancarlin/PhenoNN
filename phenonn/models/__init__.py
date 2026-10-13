# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Kept model architectures (Week 1 cleanup).

Legacy models (attn_lstm, dimchangemodule, fcn, linear_baseline,
transformerbis) live under ``legacy/models/`` and are deleted January 2027.
"""

import importlib

__all__ = [
    "RNN_LSTM",
    "RNN_GRU",
    "AELSTM",
    "BiTransformerV2",
    "EncoderTorch",
    "CausalSparseTransformer",
    "SinusoidalPositionalEncoding",
]

_LAZY = {
    "RNN_LSTM": (".rnn", "RNN_LSTM"),
    "RNN_GRU": (".rnn", "RNN_GRU"),
    "AELSTM": (".aelstm", "AELSTM"),
    "BiTransformerV2": (".bitransformer", "BiTransformerV2"),
    "EncoderTorch": (".transformer", "EncoderTorch"),
    "CausalSparseTransformer": (".transformer", "CausalSparseTransformer"),
    "SinusoidalPositionalEncoding": (
        ".transformer",
        "SinusoidalPositionalEncoding",
    ),
}


def __getattr__(name):
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        value = getattr(importlib.import_module(mod_name, __name__), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
