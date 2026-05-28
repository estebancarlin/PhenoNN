# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

import torch
import torch.nn as nn
from typing import Optional


class SelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism.

    This module implements scaled dot-product attention with multiple heads.

    Parameters
    ----------
    embed_size : int
        Size of the embedding dimension.
    heads : int
        Number of attention heads.

    Attributes
    ----------
    embed_size : int
        Embedding dimension.
    heads : int
        Number of attention heads.
    head_dim : int
        Dimension of each attention head (embed_size // heads).
    values : nn.Linear
        Linear layer for value projections.
    keys : nn.Linear
        Linear layer for key projections.
    queries : nn.Linear
        Linear layer for query projections.
    fc_out : nn.Linear
        Final output linear layer.

    Examples
    --------
    >>> attention = SelfAttention(embed_size=128, heads=4)
    >>> values = torch.randn(32, 10, 128)
    >>> keys = torch.randn(32, 10, 128)
    >>> query = torch.randn(32, 10, 128)
    >>> out = attention(values, keys, query, mask=None)
    >>> out.shape
    torch.Size([32, 10, 128])
    """

    def __init__(self, embed_size: int, heads: int) -> None:
        """
        Initialize the SelfAttention module.

        Parameters
        ----------
        embed_size : int
            Size of the embedding dimension.
        heads : int
            Number of attention heads.

        Raises
        ------
        AssertionError
            If embed_size is not divisible by heads.
        """
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=True)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=True)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=True)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(
        self,
        values: torch.Tensor,
        keys: torch.Tensor,
        query: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the self-attention mechanism.

        Parameters
        ----------
        values : torch.Tensor
            Value tensor of shape (batch_size, value_len, embed_size).
        keys : torch.Tensor
            Key tensor of shape (batch_size, key_len, embed_size).
        query : torch.Tensor
            Query tensor of shape (batch_size, query_len, embed_size).
        mask : torch.Tensor, optional
            Attention mask of shape broadcastable to (batch, heads, query_len, key_len).
            Values of 1 = allowed, 0 = blocked. Default is None (no masking).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, query_len, embed_size).

        Notes
        -----
        The attention mechanism follows the formula:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

        where d_k = head_dim (dimension of each attention head).
        """
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split embedding into heads
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)

        # Apply linear projections
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        # Compute attention scores: (batch, heads, query_len, key_len)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        # Apply mask if provided (1 = allowed, 0 = blocked)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Scale by sqrt(head_dim) — the per-head dimension, not the full embed_size
        # This is the standard scaled dot-product attention: softmax(QK^T / sqrt(d_k))
        attention = torch.softmax(energy / (self.head_dim ** 0.5), dim=3)

        # Apply attention to values
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        # Final projection
        out = self.fc_out(out)

        return out


class TransformerBlock(nn.Module):
    """
    Transformer block consisting of self-attention and feed-forward layers.

    This module applies multi-head self-attention followed by a feed-forward
    network, with layer normalization and residual connections.

    Parameters
    ----------
    embed_size : int
        Size of the embedding dimension.
    heads : int
        Number of attention heads.
    dropout : float
        Dropout rate.
    forward_expansion : int
        Expansion factor for the feed-forward network.

    Attributes
    ----------
    attention : SelfAttention
        Multi-head self-attention module.
    norm1 : nn.LayerNorm
        First layer normalization.
    norm2 : nn.LayerNorm
        Second layer normalization.
    feed_forward : nn.Sequential
        Feed-forward network.
    dropout : nn.Dropout
        Dropout layer.

    Examples
    --------
    >>> block = TransformerBlock(128, 4, dropout=0.1, forward_expansion=4)
    >>> x = torch.randn(32, 10, 128)
    >>> out = block(x, x, x, mask=None)
    >>> out.shape
    torch.Size([32, 10, 128])
    """

    def __init__(
        self, embed_size: int, heads: int, dropout: float, forward_expansion: int
    ) -> None:
        """
        Initialize the TransformerBlock.

        Parameters
        ----------
        embed_size : int
            Size of the embedding dimension.
        heads : int
            Number of attention heads.
        dropout : float
            Dropout rate.
        forward_expansion : int
            Expansion factor for the feed-forward network.
        """
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        value: torch.Tensor,
        key: torch.Tensor,
        query: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the transformer block.

        Parameters
        ----------
        value : torch.Tensor
            Value tensor of shape (batch_size, seq_len, embed_size).
        key : torch.Tensor
            Key tensor of shape (batch_size, seq_len, embed_size).
        query : torch.Tensor
            Query tensor of shape (batch_size, seq_len, embed_size).
        mask : torch.Tensor, optional
            Attention mask. Default is None.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, seq_len, embed_size).
        """
        # Self-attention with residual connection
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))

        # Feed-forward with residual connection
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))

        return out


class Encoder(nn.Module):
    """
    Transformer encoder for sequence-to-sequence processing.

    This module applies positional encoding and a stack of transformer blocks
    to transform input sequences. Supports optional causal masking for
    autoregressive / forecasting setups.

    Parameters
    ----------
    feature_channel : int
        Number of input features.
    output_channel : int
        Number of output channels.
    embed_size : int
        Size of the embedding dimension.
    num_layers : int
        Number of transformer blocks.
    heads : int
        Number of attention heads.
    forward_expansion : int
        Expansion factor for feed-forward networks.
    seq_length : int
        Length of the input sequence.
    dropout : float
        Dropout rate.
    causal : bool, optional
        If True, apply a lower-triangular causal mask so that each position
        can only attend to itself and earlier positions. Default is False.

    Attributes
    ----------
    embed_size : int
        Embedding dimension.
    seq_length : int
        Input sequence length.
    causal : bool
        Whether causal masking is active.
    first : nn.Linear
        Initial linear projection.
    first_act : nn.ReLU
        Activation function.
    position_embedding : nn.Embedding
        Positional embeddings.
    layers : nn.ModuleList
        Stack of transformer blocks.
    dropout : nn.Dropout
        Dropout layer.
    final : nn.Conv1d
        Final convolution to map to output channels.

    Examples
    --------
    >>> encoder = Encoder(
    ...     feature_channel=6,
    ...     output_channel=4,
    ...     embed_size=64,
    ...     num_layers=2,
    ...     heads=4,
    ...     forward_expansion=4,
    ...     seq_length=10,
    ...     dropout=0.1
    ... )
    >>> x = torch.randn(32, 6, 10)
    >>> y = encoder(x)
    >>> y.shape
    torch.Size([32, 4, 10])

    Causal mode:
    >>> encoder_causal = Encoder(
    ...     feature_channel=6, output_channel=4, embed_size=64,
    ...     num_layers=2, heads=4, forward_expansion=4,
    ...     seq_length=10, dropout=0.1, causal=True
    ... )
    >>> y = encoder_causal(x)
    >>> y.shape
    torch.Size([32, 4, 10])
    """

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
        causal: bool = False,
    ) -> None:
        """
        Initialize the Transformer encoder.

        Parameters
        ----------
        feature_channel : int
            Number of input features.
        output_channel : int
            Number of output channels.
        embed_size : int
            Size of the embedding dimension.
        num_layers : int
            Number of transformer blocks.
        heads : int
            Number of attention heads.
        forward_expansion : int
            Expansion factor for feed-forward networks.
        seq_length : int
            Length of the input sequence.
        dropout : float
            Dropout rate.
        causal : bool, optional
            If True, apply a triangular causal mask. Default is False.

        Raises
        ------
        ValueError
            If num_layers is less than 1.
        """
        super(Encoder, self).__init__()

        if num_layers < 1:
            raise ValueError(f"num_layers must be at least 1, got {num_layers}")

        self.embed_size = embed_size
        self.seq_length = seq_length
        self.causal = causal

        # Precompute the causal mask: shape (1, 1, seq_length, seq_length)
        # Broadcasts over (batch, heads). Convention: 1 = allowed, 0 = blocked.
        # Stored as a buffer so it moves to GPU with the model and is saved
        # in state_dict, but is not a learnable parameter.
        if causal:
            causal_mask = torch.tril(
                torch.ones(seq_length, seq_length, dtype=torch.uint8)
            )
            self.register_buffer(
                "causal_mask", causal_mask.view(1, 1, seq_length, seq_length)
            )
        else:
            self.causal_mask = None

        # Initial projection from features to embeddings
        self.first = nn.Linear(feature_channel, embed_size)
        self.first_act = nn.ReLU()

        # Positional embeddings
        self.position_embedding = nn.Embedding(seq_length, embed_size)

        # Stack of transformer blocks
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

        # Final projection to output channels
        self.final = nn.Conv1d(
            embed_size, output_channel, kernel_size=1, padding=0, bias=True
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the transformer encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, feature_channel, seq_length).
        mask : torch.Tensor, optional
            External attention mask. If provided, overrides the built-in causal
            mask. Shape must be broadcastable to (batch, heads, seq, seq).
            Convention: 1 = allowed, 0 = blocked. Default is None.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_channel, seq_length).

        Notes
        -----
        The forward pass:
        1. Permutes input to (batch, seq, features)
        2. Applies linear projection to embeddings
        3. Adds positional embeddings
        4. Passes through transformer blocks (with causal mask if enabled)
        5. Permutes back and applies final convolution
        """
        # Permute to (batch, seq, features) for transformer
        x = torch.permute(x, (0, 2, 1))
        N = x.shape[0]

        # Positional embeddings
        positions = (
            torch.arange(0, self.seq_length).expand(N, self.seq_length).to(x.device)
        )
        positions = self.position_embedding(positions)

        # Initial projection and add positional embeddings
        out = self.first_act(self.first(x))
        out = out + positions

        # Decide which mask to use:
        #   - caller-supplied mask takes priority (for custom masking)
        #   - otherwise use the built-in causal mask (if causal=True)
        #   - otherwise None (full attention)
        effective_mask = mask if mask is not None else self.causal_mask

        # Apply transformer blocks
        for layer in self.layers:
            out = layer(out, out, out, effective_mask)

        # Permute back and apply final convolution
        out = torch.permute(out, (0, 2, 1))
        out = self.final(out)

        return out


class EncoderTorch(nn.Module):
    """
    Transformer encoder using PyTorch's built-in TransformerEncoder.

    Same input/output interface as Encoder but uses nn.TransformerEncoderLayer
    internally. Supports causal masking via PyTorch's
    nn.Transformer.generate_square_subsequent_mask.

    Parameters
    ----------
    feature_channel : int
        Number of input features.
    output_channel : int
        Number of output channels.
    embed_size : int
        Size of the embedding dimension.
    num_layers : int
        Number of transformer blocks.
    heads : int
        Number of attention heads.
    forward_expansion : int
        Expansion factor for feed-forward networks.
    seq_length : int
        Length of the input sequence.
    dropout : float
        Dropout rate.
    causal : bool, optional
        If True, apply a causal mask. Default is False.
    """

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
        causal: bool = True,
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError(f"num_layers must be at least 1, got {num_layers}")

        self.embed_size = embed_size
        self.seq_length = seq_length
        self.causal = causal

        # Precompute causal mask for PyTorch's TransformerEncoder
        # PyTorch convention: -inf = blocked, 0 = allowed (additive mask)
        if causal:
            causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_length)
            self.register_buffer("causal_mask", causal_mask)
        else:
            self.causal_mask = None

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
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, feature_channel, seq_length).
        mask : torch.Tensor, optional
            External attention mask. If provided, overrides the built-in causal
            mask. Shape (seq_length, seq_length), additive (-inf = blocked).
        src_key_padding_mask : torch.Tensor, optional
            Padding mask of shape (batch, seq_length). True = padded/ignored.

        Returns
        -------
        torch.Tensor
            Output of shape (batch, output_channel, seq_length).
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

        # Decide which mask to use
        effective_mask = mask if mask is not None else self.causal_mask

        # Transformer encoder
        x = self.encoder(
            x, mask=effective_mask, src_key_padding_mask=src_key_padding_mask
        )

        # Back to (batch, channels, seq)
        x = x.permute(0, 2, 1)
        x = self.final(x)

        return x