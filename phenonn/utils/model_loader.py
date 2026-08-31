# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Stefan Barbu, Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""
Phenonn Model Loader

Factory module for instantiating deep learning models.

This module provides a unified interface to build different neural
network architectures (RNNs, Transformers, FCNs, and linear baselines)
and automatically wraps them to ensure consistent output formatting
for training and evaluation pipelines.

Design
------
All models are constructed as base networks and then wrapped using
lightweight adapters that enforce consistent output shapes:

- SingleDayWrapper:
    Ensures output shape (batch, 1) for standard single-day prediction.

- LastNDaysWrapper:
    Returns predictions over multiple target days (used in gradient loss
    or full-sequence forecasting).

- permuteWrapper:
    Handles tensor dimension reordering for transformer-based models.

- Special cases:
    Some models (e.g. linear baselines) are returned without wrapping
    because they already produce correctly shaped outputs.

Supported models
-----------------
- RNN-based models:
    - LSTM (RNN_LSTM)
    - GRU (RNN_GRU)
    - 1-year LSTM variant (sequence-to-sequence style)

- Transformer models:
    - Encoder-only Transformer (EncoderTorch)
    - BiTransformer variants
    - Combined Transformer-RNN hybrid (transformerbis)

- Feed-forward models:
    - FCN (Fully Connected Network)

- Linear baselines:
    - LinearBaseline
    - PerDayLinearBaseline

Wrapper logic
-------------
The final wrapper depends on training configuration:

- args.n_target_days > 1:
    → LastNDaysWrapper is used (multi-day regression / gradient loss)

- otherwise:
    → SingleDayWrapper is used (standard single-step prediction)

Special cases:
- 1-year models automatically return LastNDaysWrapper(365)

Parameters
----------
args : argparse.Namespace or EasyDict
    Configuration object containing at least:

    type : str
        Model type identifier (e.g., 'lstm', 'transformer', 'fcn').
    feature_channel : int
        Number of input features (meteorology + cyclic + static + PFT).
    output_channel : int
        Number of output targets (typically 1 for LAI/GCC).
    seq_length : int
        Input sequence length (e.g., 365 days).

    Plus architecture-specific hyperparameters:
        hidden_size, num_layers, embed_size, nhead, dropout, etc.

Returns
-------
nn.Module
    Wrapped model ready for training. Output shape depends on wrapper:

    - (batch, 1) for SingleDayWrapper
    - (batch, n_target_days) for LastNDaysWrapper

Raises
------
ValueError
    If `args.type` does not match any supported architecture.

Notes
-----
- This module standardizes heterogeneous architectures under a single
  training interface.
- Wrapping ensures compatibility with loss functions and dataset outputs.
- Transformer-based models may internally permute tensor dimensions
  using `permuteWrapper`.

See Also
--------
phenonn.models.rnn.RNN_LSTM
phenonn.models.rnn.RNN_GRU
phenonn.models.transformer.EncoderTorch
phenonn.models.fcn.FCN
phenonn.models.linear_baseline.LinearBaseline
"""

from phenonn.models.rnn import RNN_LSTM, RNN_GRU
from phenonn.models.transformer import EncoderTorch
from phenonn.models.linear_baseline import LinearBaseline, PerDayLinearBaseline
from phenonn.models.fcn import FCN
from phenonn.models.transformerbis import CombinedModel, BiTransformer
from .wrappers import SingleDayWrapper, permuteWrapper, LastNDaysWrapper


def load_model(args):
    """
    Instantiate and wrap a model for single-day GCC prediction.

    Parameters
    ----------
    args : argparse.Namespace or EasyDict
        Must contain at minimum:

            type : str
                One of 'lstm', 'gru', 'transformer', 'fcn', 'fullyconnected'.
            feature_channel : int
                Number of input feature channels (meteo + cyclic + static + PFT).
            output_channel : int
                Number of output channels (1 for gcc_lowess).
            seq_length : int
                Window length (365).

        Plus architecture-specific parameters (see original model_loader).

    Returns
    -------
    SingleDayWrapper
        Model whose forward returns (batch, output_channel).
    """
    model_type = args.type.lower()

    # ── Linear baselines (no wrapper needed, already output (B, 1)) ──

    if model_type == "linear":
        return LinearBaseline(
            feature_channel=args.feature_channel,
            seq_length=args.seq_length,
        )

    if model_type == "linear_perday":
        return PerDayLinearBaseline(
            feature_channel=args.feature_channel,
        )

    if model_type in ["lstm", "gru"]:
        model_class = RNN_LSTM if model_type == "lstm" else RNN_GRU
        base_model = model_class(
            feature_channel=args.feature_channel,
            output_channel=args.output_channel,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
        )

    elif model_type == "1year_lstm":
        base_model = RNN_LSTM(
            feature_channel=args.feature_channel,
            output_channel=args.output_channel,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
        )
        return LastNDaysWrapper(base_model, n_days=365)

    elif model_type == "transformer":
        base_model = EncoderTorch(
            feature_channel=args.feature_channel,
            output_channel=args.output_channel,
            embed_size=args.embed_size,
            num_layers=args.num_layers,
            heads=args.nhead,
            forward_expansion=getattr(args, "forward_expansion", 4) or 4,
            seq_length=args.seq_length,
            dropout=args.dropout,
            causal=False,
        )

    elif model_type == "transformerbis":
        base_model = permuteWrapper(
            CombinedModel(
                input_dim=args.feature_channel,
                hidden_dim=args.hidden_size,
                hidden_dim_trans=args.hidden_size,
                output_dim=args.output_channel,
                d_model=32,
                nr_blocks=3,
                n_pft=10,
            )
        )

    elif model_type == "bitransformer":
        base_model = permuteWrapper(
            BiTransformer(
                input_dim=args.feature_channel,
                hidden_dim=args.hidden_size,
                hidden_dim_trans=args.hidden_size,
                output_dim=args.output_channel,
                d_model=32,
                nr_blocks=3,
                n_pft=3,
            )
        )

    elif model_type == "1year_bitransformer":
        base_model = permuteWrapper(
            BiTransformer(
                input_dim=args.feature_channel,
                d_model=args.hidden_size,
                feed_forward_trans=args.feed_forward_trans,
                feed_forward_encoder=args.feed_forward_encoder,
                output_dim=args.output_channel,
                nr_blocks=args.num_layers,
                dropout_trans=args.dropout_trans,
                dropout_encoder=args.dropout,
                n_pft=9,
            )
        )
        return LastNDaysWrapper(base_model, n_days=365)

    elif model_type in ["fcn", "fullyconnected"]:
        base_model = FCN(
            feature_channel=args.feature_channel,
            output_channel=args.output_channel,
            num_layers=args.num_layers,
            hidden_size=args.hidden_size,
            seq_length=args.seq_length,
            dim_expand=0,
        )

    else:
        raise ValueError(f"Model type '{args.type}' is not implemented.")

    # Choose wrapper based on training mode:
    #   n_target_days > 1 → LastNDaysWrapper (gradient loss, output last N days)
    #   otherwise         → SingleDayWrapper (standard, output last day only)
    n_target = getattr(args, "n_target_days", 1)
    if n_target > 1:
        return LastNDaysWrapper(base_model, n_days=n_target)

    return SingleDayWrapper(base_model)
