# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Christian Reimers
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""
Neural network architectures for phenological and vegetation dynamics modeling.

This module contains a collection of recurrent, transformer-based, and
feed-forward neural network models originally developed for vegetation
greenness prediction from meteorological and environmental forcing data.
The architectures range from simple baselines to multi-stage transformer
models that explicitly separate environmental stress estimation from
phenological state prediction.

The implementations are adapted from work by Christian Reimers and
collaborators at the Max Planck Institute for Biogeochemistry and are
used within the PhenoCam LAI modeling framework.

Implemented Models
------------------
SimpleLSTM
    Single-layer Long Short-Term Memory (LSTM) network followed by a
    linear projection. Produces sequence-wise predictions from temporal
    environmental forcing data.

TransformerNTS
    Minimal transformer architecture consisting of a single transformer
    encoder-decoder block. Serves as a lightweight transformer baseline.

SimpleTransformer
    Causal transformer encoder model using autoregressive masking.
    Learns temporal dependencies within environmental forcing sequences
    and predicts vegetation dynamics at each timestep.

CombinedModel
    Two-stage architecture that first estimates an intermediate
    environmental stress signal using a transformer module and then
    combines this signal with plant functional type (PFT) information
    before phenological prediction.

SimpleNetwork
    Feed-forward temporal model implemented as a sequence of 1×1
    convolutions. Equivalent to applying a multilayer perceptron
    independently at each timestep.

BiTransformer
    Improved bidirectional transformer architecture that combines
    environmental stress estimation with causal temporal encoding and
    configurable plant functional type inputs. This model serves as the
    primary transformer implementation for vegetation prediction tasks.

Supporting Components
---------------------
PositionalEncoding
    Sinusoidal positional encoding module following the formulation
    introduced in "Attention Is All You Need" (Vaswani et al., 2017).
    Provides temporal position information to transformer-based models.

Model Inputs
------------
Most models expect input tensors of shape:

    (batch_size, sequence_length, n_features)

where each timestep contains meteorological, environmental, and
optionally plant functional type (PFT) information.

Model Outputs
-------------
Models generally return predictions of shape:

    (batch_size, sequence_length, output_dim)

where ``output_dim`` typically represents one or more vegetation
state variables such as greenness, LAI, or intermediate stress
indicators.

Notes
-----
* Several architectures support a ``return_stress`` mode that exposes
  intermediate latent stress representations used internally by the model.
* Transformer-based models employ causal attention masks to prevent
  future information leakage during sequence prediction.
* Plant Functional Type (PFT) variables are assumed to occupy the final
  feature channels of the input tensor when required.
* Within the PhenoCam LAI workflow, wrapper modules may permute tensor
  dimensions to match the project's standard convention:

      (batch_size, feature_channels, sequence_length)

  while the original models operate on:

      (batch_size, sequence_length, feature_channels)

References
----------
Original vegetation prediction architectures developed at the
Max Planck Institute for Biogeochemistry.
"""

import copy
import torch


class SimpleLSTM(torch.nn.Module):
    def __init__(self, input_dim=26, hidden_dim=128, output_dim=2):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, return_stress=False):
        # LSTM layer
        lstm_out, (hn, cn) = self.lstm(x)
        # We only want the output from the last time step
        out = self.fc(lstm_out)
        if return_stress:
            return out, torch.zeros_like(out[:, :, 0:1])
        return out


class PositionalEncoding(torch.nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()

        self.dropout = torch.nn.Dropout(dropout_p)

        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)
        division_term = 10000 ** (torch.arange(0, dim_model, 2).float() / dim_model)

        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embeding: torch.tensor) -> torch.tensor:
        return self.dropout(
            token_embeding + self.pos_encoding[: token_embeding.size(0), :]
        )


class TransformerNTS(torch.nn.Module):
    def __init__(
        self,
        input_dim=26,
        hidden_dim_trans=1024,
        d_model=32,
    ):
        super().__init__()
        self.lin1 = torch.nn.Linear(input_dim, d_model)
        self.trans = torch.nn.Transformer(
            num_encoder_layers=1,
            num_decoder_layers=1,
            d_model=d_model,
            dim_feedforward=hidden_dim_trans,
        )
        self.lin2 = torch.nn.Linear(d_model, 2)

    def forward(self, x):
        x = self.lin1(x)
        x = self.trans(x, x)
        x = self.lin2(x)

        return x


class SimpleTransformer(torch.nn.Module):
    def __init__(
        self,
        input_dim=26,
        hidden_dim=1024,
        output_dim=2,
        d_model=32,
        nr_blocks=3,
    ):
        super().__init__()
        self.lin3 = torch.nn.Linear(input_dim, d_model)
        self.encoder = torch.nn.ModuleList(
            [
                torch.nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=4,
                    dim_feedforward=hidden_dim,
                    dropout=0,
                )
                for i in range(nr_blocks)
            ]
        )

        self.lin4 = torch.nn.Linear(d_model, 2)

        self.positional_encoder = PositionalEncoding(
            dim_model=d_model, dropout_p=0, max_len=730
        )

    def forward(self, x, return_stress=False):
        # pft = x[:, :, -10:]

        size = x.size(1)
        mask = torch.tril(torch.ones(size, size) == 1)  # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float("-inf"))  # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, 0.0)  # Convert ones to 0

        x = x.permute(1, 0, 2)
        x = self.lin3(x)
        for i in range(len(self.encoder)):
            x = self.encoder[i](x, mask)

        x = self.lin4(x)
        x = x.permute(1, 0, 2)

        if return_stress:
            return x, torch.zeros_like(x[:, :, 0])
        return x


class CombinedModel(torch.nn.Module):
    def __init__(
        self,
        input_dim=26,
        hidden_dim=1024,
        hidden_dim_trans=1024,
        output_dim=2,
        d_model=32,
        nr_blocks=3,
    ):
        super().__init__()
        self.lin1 = torch.nn.Linear(input_dim, d_model)
        self.trans = torch.nn.Transformer(
            num_encoder_layers=1,
            num_decoder_layers=1,
            d_model=d_model,
            dim_feedforward=hidden_dim_trans,
        )
        self.lin2 = torch.nn.Linear(d_model, 1)

        self.lin3 = torch.nn.Linear(11, d_model)
        self.encoder = torch.nn.ModuleList(
            [
                torch.nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=4,
                    dim_feedforward=hidden_dim,
                    dropout=0,
                )
                for i in range(nr_blocks)
            ]
        )

        self.lin4 = torch.nn.Linear(d_model, 2)

        self.positional_encoder = PositionalEncoding(
            dim_model=d_model, dropout_p=0, max_len=730
        )

    def forward(self, x, return_stress=False):
        pft = x[:, :, -10:]

        size = x.size(1)
        mask = torch.tril(torch.ones(size, size) == 1)  # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float("-inf"))  # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, 0.0)  # Convert ones to 0

        x = self.lin1(x)
        x = self.trans(x, x)
        x = self.lin2(x)
        if return_stress:
            y = copy.deepcopy(x)
        x = torch.concatenate([x, pft], axis=-1)
        x = x.permute(1, 0, 2)
        x = self.lin3(x)
        for i in range(len(self.encoder)):
            x = self.encoder[i](x, mask)

        x = self.lin4(x)
        x = x.permute(1, 0, 2)

        if return_stress:
            return x, y
        return x

    def forward_from_stress(self, x, pft):
        size = x.size(1)
        mask = torch.tril(torch.ones(size, size) == 1)  # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float("-inf"))  # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, 0.0)  # Convert ones to 0

        x = torch.concatenate([x, pft], axis=-1)
        x = x.permute(1, 0, 2)
        x = self.lin3(x)
        for i in range(len(self.encoder)):
            x = self.encoder[i](x, mask)

        x = self.lin4(x)
        return x.permute(1, 0, 2)


class SimpleNetwork(torch.nn.Module):
    def __init__(self, hidden_dim=1024):
        super().__init__()

        self.linear1 = torch.nn.Conv1d(
            in_channels=26,
            out_channels=hidden_dim,
            kernel_size=1,
        )
        self.linear2 = torch.nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=2,
            kernel_size=1,
        )
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = x.transpose(2, 1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = x.transpose(2, 1)

        return x


# ___Bi Transformer model________


class BiTransformer(torch.nn.Module):
    def __init__(
        self,
        input_dim=26,
        feed_forward_trans=4,
        feed_forward_encoder=4,
        output_dim=2,
        d_model=256,
        nr_blocks=3,
        dropout_trans=0.1,
        dropout_encoder=0.1,
        n_pft=1,  # <-- new: number of trailing PFT features
    ):
        super().__init__()
        self.n_pft = n_pft

        self.lin1 = torch.nn.Linear(input_dim, d_model)
        self.trans = torch.nn.Transformer(
            num_encoder_layers=1,
            num_decoder_layers=1,
            d_model=d_model,
            dim_feedforward=feed_forward_trans * d_model,
            dropout=dropout_trans,
        )
        self.lin2 = torch.nn.Linear(d_model, 1)

        self.lin3 = torch.nn.Linear(1 + n_pft, d_model)  # <-- was hardcoded 11
        self.encoder = torch.nn.ModuleList(
            [
                torch.nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=4,
                    dim_feedforward=feed_forward_encoder
                    * d_model,  # <-- was hidden_dim, now scaled with d_model
                    dropout=dropout_encoder,
                )
                for _ in range(nr_blocks)
            ]
        )
        self.lin4 = torch.nn.Linear(d_model, output_dim)  # <-- was hardcoded 2

        self.positional_encoder = PositionalEncoding(
            dim_model=d_model, dropout_p=0.1, max_len=730
        )

    def _causal_mask(self, size, device):
        mask = torch.triu(
            torch.full((size, size), float("-inf"), device=device), diagonal=1
        )
        return mask

    def forward(self, x, return_stress=False):
        pft = x[:, :, -self.n_pft :]  # <-- uses param
        mask = self._causal_mask(x.size(1), x.device)

        x = self.lin1(x)

        x = self.trans(x, x)
        x = self.lin2(x)
        if return_stress:
            y = x.clone()  # <-- clone, not deepcopy
        x = torch.cat([x, pft], dim=-1)
        x = x.permute(1, 0, 2)
        x = self.lin3(x)
        for layer in self.encoder:
            x = layer(x, mask)
        x = self.lin4(x)
        x = x.permute(1, 0, 2)

        if return_stress:
            return x, y
        return x
