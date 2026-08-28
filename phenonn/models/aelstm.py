# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Stefan Barbu, Kazem Ardaneh

"""Attention-Enhanced LSTM used by the laboratory global pipeline.

This implementation follows the original ``global_LAI`` model: an LSTM is
followed by causal transformer-style self-attention blocks and a 1D output
projection. The causal mask prevents future days from entering each output.
"""

import torch
import torch.nn as nn


class SelfAttentionBlock(nn.Module):
    """Pre-norm self-attention block with residual connections."""

    def __init__(self, d_model, nhead, ff_expansion=4, dropout=0.0):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by nhead ({nhead})"
            )
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.drop_attn = nn.Dropout(dropout)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_expansion * d_model),
            nn.GELU(),
            nn.Linear(ff_expansion * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask=None):
        h = self.norm_attn(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + self.drop_attn(attn_out)
        return x + self.ff(self.norm_ff(x))


class AELSTM(nn.Module):
    """Original laboratory Attention-Enhanced LSTM sequence model."""

    def __init__(
        self,
        feature_channel,
        output_channel,
        hidden_size,
        num_layers,
        n_attn_blocks=2,
        nhead=4,
        ff_expansion=4,
        dropout=0.0,
        dropout_att=0.0,
        seq_length=720,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=feature_channel,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.attn_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    d_model=hidden_size,
                    nhead=nhead,
                    ff_expansion=ff_expansion,
                    dropout=dropout_att,
                )
                for _ in range(n_attn_blocks)
            ]
        )
        self.norm_out = nn.LayerNorm(hidden_size)
        self.final = nn.Conv1d(hidden_size, output_channel, kernel_size=1, bias=True)
        self.register_buffer(
            "causal_mask",
            nn.Transformer.generate_square_subsequent_mask(seq_length),
            persistent=False,
        )

    def init_hidden(self, batch_size, device):
        hidden = torch.zeros(
            self.num_layers, batch_size, self.hidden_size, device=device
        )
        return hidden, hidden.clone()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        hidden = self.init_hidden(x.size(0), x.device)
        out, _ = self.lstm(x, hidden)
        causal_mask = self.causal_mask[: out.size(1), : out.size(1)]
        if causal_mask.dtype != out.dtype:
            causal_mask = causal_mask.to(out.dtype)
        for block in self.attn_blocks:
            out = block(out, attn_mask=causal_mask)
        out = self.norm_out(out).permute(0, 2, 1)
        return self.final(out)
