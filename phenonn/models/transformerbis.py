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

CombinedModel
    Two-stage architecture that first estimates an intermediate
    environmental stress signal using a transformer module and then
    combines this signal with plant functional type (PFT) information
    before phenological prediction.

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

import torch


class PositionalEncoding(torch.nn.Module):
    """
    Sinusoidal positional encoding module.

    Adds fixed positional encodings to token embeddings as described in
    the Transformer architecture. The positional encodings are computed
    using sine and cosine functions of different frequencies and stored
    as a non-trainable buffer.

    Parameters
    ----------
    dim_model : int
        Dimensionality of the token embeddings.
    dropout_p : float
        Dropout probability applied after adding positional encodings.
    max_len : int
        Maximum sequence length for which positional encodings are
        precomputed.

    Attributes
    ----------
    dropout : torch.nn.Dropout
        Dropout layer applied to the sum of token embeddings and
        positional encodings.
    pos_encoding : torch.Tensor
        Tensor of shape (max_len, 1, dim_model) containing the
        precomputed positional encodings.

    Notes
    -----
    For position ``pos`` and embedding dimension ``i``:

    - Even dimensions:

      ``PE(pos, 2i) = sin(pos * scale_i)``

    - Odd dimensions:

      ``PE(pos, 2i + 1) = cos(pos * scale_i)``

    where

    ``scale_i = 10000^(2i / dim_model)``

    Examples
    --------
    >>> pe = PositionalEncoding(dim_model=512, dropout_p=0.1, max_len=5000)
    >>> x = torch.randn(20, 32, 512)
    >>> y = pe(x)
    >>> y.shape
    torch.Size([20, 32, 512])
    """

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
        """
        Add positional encodings to token embeddings.

        Parameters
        ----------
        token_embeding : torch.Tensor
            Input embeddings of shape (sequence_length, batch_size,
            dim_model).

        Returns
        -------
        torch.Tensor
            Embeddings with positional encodings added and dropout applied.
            Shape is identical to the input:
            (sequence_length, batch_size, dim_model).
        """
        return self.dropout(
            token_embeding + self.pos_encoding[: token_embeding.size(0), :]
        )


class CombinedModel(torch.nn.Module):
    """
    Hybrid Transformer-based model for sequence prediction with auxiliary features.

    This model combines:
    - A linear projection of input features into model space
    - A Transformer module for sequence-to-sequence processing
    - Additional concatenated features (PFT inputs)
    - A stack of Transformer encoder layers with causal masking
    - Final linear projection to output space

    Parameters
    ----------
    input_dim : int, default=26
        Number of input features per timestep.
    hidden_dim : int, default=1024
        Feedforward dimension inside Transformer encoder layers.
    hidden_dim_trans : int, default=1024
        Feedforward dimension inside the built-in Transformer module.
    output_dim : int, default=2
        Number of output features per timestep.
    d_model : int, default=32
        Internal embedding dimension used throughout the model.
    nr_blocks : int, default=3
        Number of stacked TransformerEncoderLayer blocks.

    Attributes
    ----------
    lin1 : torch.nn.Linear
        Projects input features into model dimension (d_model).
    trans : torch.nn.Transformer
        Transformer module applied to the projected input.
    lin2 : torch.nn.Linear
        Reduces Transformer output to a single feature.
    lin3 : torch.nn.Linear
        Projects concatenated features into model dimension.
    encoder : torch.nn.ModuleList
        Stack of TransformerEncoderLayer blocks.
    lin4 : torch.nn.Linear
        Final projection from model dimension to output space (2).
    positional_encoder : PositionalEncoding
        Sinusoidal positional encoding module.

    Notes
    -----
    - The model uses a causal (lower-triangular) attention mask to prevent
      attending to future timesteps.
    - Input tensors are expected in shape:
      (batch_size, sequence_length, input_dim)
    - Internally, tensors are permuted to:
      (sequence_length, batch_size, d_model)
      for Transformer processing.
    """

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
        """
        Forward pass of the model using full input sequence.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, input_dim).
        return_stress : bool, default=False
            If True, also returns intermediate representation after lin2.

        Returns
        -------
        torch.Tensor or tuple of torch.Tensor
            If return_stress=False:
                Output tensor of shape
                (batch_size, sequence_length, 2)

            If return_stress=True:
                Tuple (output, stress_tensor)
                where:
                - output: final predictions (B, T, 2)
                - stress_tensor: intermediate representation (B, T, 1)
        """
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
            y = x.clone()  # <-- clone, not deepcopy
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
        """
        Forward pass starting from intermediate stress representation.

        This bypasses the initial Transformer and lin1/lin2 layers,
        and instead continues processing from a reduced representation
        combined with auxiliary PFT features.

        Parameters
        ----------
        x : torch.Tensor
            Stress-like representation of shape (batch_size, sequence_length, 1).
        pft : torch.Tensor
            Auxiliary features of shape (batch_size, sequence_length, 10).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, sequence_length, 2).
        """
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


# ___Bi Transformer model________


class BiTransformer(torch.nn.Module):
    """
    Bidirectional Transformer-style sequence model with auxiliary features (PFT).

    This model combines:
    - A linear embedding layer for input features
    - A Transformer encoder-decoder block for global sequence mixing
    - A causal (autoregressive) TransformerEncoder stack
    - Concatenation of auxiliary PFT features
    - Final projection to output space

    The architecture is designed for sequence-to-sequence prediction where
    past context is enforced via a causal attention mask.

    Parameters
    ----------
    input_dim : int, default=26
        Number of input features per timestep.
    feed_forward_trans : int, default=4
        Multiplier for Transformer feedforward dimension.
    feed_forward_encoder : int, default=4
        Multiplier for encoder feedforward dimension.
    output_dim : int, default=2
        Number of output features per timestep.
    d_model : int, default=256
        Hidden representation size used throughout the model.
    nr_blocks : int, default=3
        Number of TransformerEncoderLayer blocks.
    dropout_trans : float, default=0.1
        Dropout rate used inside the Transformer module.
    dropout_encoder : float, default=0.1
        Dropout rate used in encoder layers.
    n_pft : int, default=1
        Number of trailing auxiliary features (PFT) appended to input.

    Attributes
    ----------
    lin1 : torch.nn.Linear
        Projects input features into d_model space.
    trans : torch.nn.Transformer
        Transformer encoder-decoder module for global sequence mixing.
    lin2 : torch.nn.Linear
        Reduces Transformer output to a single-channel representation.
    lin3 : torch.nn.Linear
        Projects concatenated [stress, PFT] features into d_model space.
    encoder : torch.nn.ModuleList
        Stack of TransformerEncoderLayer blocks with causal masking.
    lin4 : torch.nn.Linear
        Final projection to output_dim.
    positional_encoder : PositionalEncoding
        Sinusoidal positional encoding module (defined but not used in forward).

    Notes
    -----
    - Input tensors are expected in shape:
      (batch_size, sequence_length, input_dim)
    - The last `n_pft` features are treated as auxiliary inputs and split off.
    - A causal (upper-triangular) mask is applied to prevent future information
      leakage in the encoder stack.
    - Internally, tensors are permuted to:
      (sequence_length, batch_size, d_model)
      for TransformerEncoder processing.
    """

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
        """
        Forward pass of the BiTransformer model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, input_dim),
            where the last `n_pft` channels are auxiliary PFT features.
        return_stress : bool, default=False
            If True, also returns intermediate representation after lin2.

        Returns
        -------
        torch.Tensor or tuple of torch.Tensor
            If return_stress=False:
                Output tensor of shape (batch_size, sequence_length, output_dim)

            If return_stress=True:
                Tuple (output, stress_tensor)
                where:
                - output: final predictions (B, T, output_dim)
                - stress_tensor: intermediate scalar representation (B, T, 1)
        """
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
