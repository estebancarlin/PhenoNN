# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""
Transformer-based encoder model for sequence modeling.

This module implements a Transformer encoder architecture using PyTorch's
native ``nn.TransformerEncoder`` components. It is designed for processing
structured sequence data, such as time series or vertical profiles, where
contextual relationships across positions are important.

The model projects input features into an embedding space, adds learnable
positional encodings, and processes the sequence through stacked self-attention
layers before projecting to the desired output channels.

Features
--------
- Learnable input projection to embedding space
- Learnable positional embeddings for sequence order awareness
- Multi-head self-attention via Transformer encoder layers
- Configurable depth, attention heads, and feedforward expansion
- Dropout for regularization
- Final 1D convolution for channel-wise output projection
- Support for attention masks and padding masks

Notes
-----
- Inputs are expected in the shape (batch_size, feature_channel, seq_length).
- Internally, inputs are permuted to (batch_size, seq_length, feature_channel)
  to match Transformer expectations.
- Positional embeddings are added to the projected input features.
- The ``mask`` argument is used for attention masking (e.g., causal masking).
- The ``src_key_padding_mask`` is used to ignore padded positions in sequences.
- The final output preserves the sequence length and maps embeddings to
  ``output_channel`` dimensions.

Dependencies
------------
- torch
- torch.nn
- typing

Examples
--------
Basic usage:

>>> model = EncoderTorch(
...     feature_channel=6,
...     output_channel=4,
...     embed_size=128,
...     num_layers=3,
...     heads=4,
...     forward_expansion=4,
...     seq_length=10,
...     dropout=0.1
... )
>>> x = torch.randn(32, 6, 10)
>>> y = model(x)
>>> y.shape
torch.Size([32, 4, 10])

Using attention masks:

>>> mask = torch.triu(torch.ones(10, 10), diagonal=1).bool()
>>> y = model(x, mask=mask)
"""

import torch
import torch.nn as nn
from typing import Optional


class EncoderTorch(nn.Module):
    def __init__(
        self,
        feature_channel: int,
        output_channel: int,
        embed_size: int,
        num_layers: int,
        heads: int,
        forward_expansion: int,
        seq_length: int,
        dropout: float,
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError(f"num_layers must be at least 1, got {num_layers}")

        self.embed_size = embed_size
        self.seq_length = seq_length

        # Input projection
        self.input_proj = nn.Linear(feature_channel, embed_size)

        # Positional embedding
        self.position_embedding = nn.Embedding(seq_length, embed_size)

        # PyTorch TransformerEncoderLayer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=heads,
            dim_feedforward=forward_expansion * embed_size,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )

        # Stack layers
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)

        # Final projection
        self.final = nn.Conv1d(embed_size, output_channel, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: (batch, feature_channel, seq_length)
        """

        # (batch, seq, feature)
        x = x.permute(0, 2, 1)
        N, seq_len, _ = x.shape

        # Positional encoding
        positions = (
            torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(N, seq_len)
        )
        pos_embed = self.position_embedding(positions)

        # Input projection + position
        x = self.input_proj(x)
        x = x + pos_embed
        x = self.dropout(x)

        # Transformer encoder
        x = self.encoder(x, mask=mask, src_key_padding_mask=src_key_padding_mask)

        # Back to (batch, channels, seq)
        x = x.permute(0, 2, 1)
        x = self.final(x)

        return x
