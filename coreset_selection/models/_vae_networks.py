"""
VAE neural network architecture classes.

Contains the nn.Module subclasses for encoder/decoder used by the VAE.
Split from models/vae.py for modularity.

Includes:
- TabularVAE: legacy MSE-only VAE (backward compatible)
- MixedTypeVAE: product-of-likelihoods decoder for heterogeneous tabular data
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, NamedTuple, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Column specification for mixed-type VAE
# ---------------------------------------------------------------------------

@dataclass
class ColumnSpec:
    """Column type specification for MixedTypeVAE.

    Classifies each column index in the preprocessed feature matrix X_scaled
    into one of three likelihood families:

    - **continuous** (Gaussian NLL): standardized numeric/ordinal columns
    - **binary** (BCE): binary categoricals and missingness indicators (K ≤ 2)
    - **categorical** (CE): multi-class categoricals (K > 2)

    Attributes
    ----------
    continuous_idx : np.ndarray
        Column indices for Gaussian-NLL reconstruction.
    binary_idx : np.ndarray
        Column indices for BCE reconstruction (logits).
    categorical_specs : list of (col_idx, K_j) tuples
        For each multi-class categorical: its column index and cardinality.
    n_total : int
        Total number of input columns (= X_scaled.shape[1]).
    """

    continuous_idx: np.ndarray
    binary_idx: np.ndarray
    categorical_specs: List[Tuple[int, int]] = field(default_factory=list)
    n_total: int = 0

    @property
    def n_continuous(self) -> int:
        return len(self.continuous_idx)

    @property
    def n_binary(self) -> int:
        return len(self.binary_idx)

    @property
    def n_categorical(self) -> int:
        return len(self.categorical_specs)


# ---------------------------------------------------------------------------
# Decoder output container
# ---------------------------------------------------------------------------

class DecoderOutput(NamedTuple):
    """Structured output from MixedTypeVAE decoder."""

    continuous: torch.Tensor            # (B, n_continuous) — Gaussian means
    binary: torch.Tensor                # (B, n_binary) — logits
    categorical: List[torch.Tensor]     # list of (B, K_j) logit tensors


# ---------------------------------------------------------------------------
# Legacy VAE (MSE-only, backward compatible)
# ---------------------------------------------------------------------------

class TabularVAE(nn.Module):
    """
    Variational Autoencoder for tabular (numeric) data.

    Architecture:
    - Encoder: input -> hidden -> (mu, logvar)
    - Decoder: z -> hidden -> reconstruction

    Attributes
    ----------
    input_dim : int
        Input feature dimension
    latent_dim : int
        Latent space dimension
    hidden_dim : int
        Hidden layer dimension
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent mean and log-variance (\u03b2-VAE encoder)."""
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """Reparameterization trick: z = \u03bc + \u03c3 \u2299 \u03b5, \u03b5 ~ N(0,I)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent z to reconstructed input (\u03b2-VAE decoder)."""
        return self.decoder(z)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass: x \u2192 (x\u0302, \u03bc, log \u03c3\u00b2)."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


# ---------------------------------------------------------------------------
# Mixed-type VAE (product-of-likelihoods decoder)
# ---------------------------------------------------------------------------

class MixedTypeVAE(nn.Module):
    """Variational Autoencoder with type-specific decoder heads.

    The decoder factorizes over columns with per-type likelihoods:

    - **Gaussian** for continuous columns (σ² = 1 after standardization,
      so NLL reduces to MSE up to constants).
    - **Bernoulli** for binary columns (BCE with logits).
    - **Categorical** for multi-class columns (CE with softmax logits).

    The encoder processes the same flat float32 input as :class:`TabularVAE`.
    For multi-class categoricals (K > 2), an optional learned embedding layer
    replaces the single integer-code column in the encoder input.  In the
    current Brazil telecom dataset all categoricals are binary (K ≤ 2), so the
    embedding path is a no-op.

    Parameters
    ----------
    column_spec : ColumnSpec
        Column type mapping for the preprocessed feature matrix.
    latent_dim : int
        Latent space dimension.
    hidden_dim : int
        Hidden layer width.
    cat_embedding_dim : int
        Embedding dimension for each K > 2 categorical column.
    """

    def __init__(
        self,
        column_spec: ColumnSpec,
        latent_dim: int,
        hidden_dim: int = 128,
        cat_embedding_dim: int = 16,
    ):
        super().__init__()

        self.column_spec = column_spec
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # ---- Encoder input construction ----
        # Continuous + binary columns enter as-is (flat float32).
        # Multi-class categoricals get learned embeddings.
        n_flat = column_spec.n_continuous + column_spec.n_binary
        self.cat_embeddings = nn.ModuleList()
        total_embed_dim = 0
        for _col_idx, K_j in column_spec.categorical_specs:
            emb = nn.Embedding(K_j, cat_embedding_dim)
            self.cat_embeddings.append(emb)
            total_embed_dim += cat_embedding_dim

        encoder_input_dim = n_flat + total_embed_dim

        # ---- Encoder ----
        self.encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # ---- Decoder shared trunk ----
        self.decoder_trunk = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # ---- Decoder type-specific heads ----
        self.head_continuous = (
            nn.Linear(hidden_dim, column_spec.n_continuous)
            if column_spec.n_continuous > 0
            else None
        )
        self.head_binary = (
            nn.Linear(hidden_dim, column_spec.n_binary)
            if column_spec.n_binary > 0
            else None
        )
        self.head_categorical = nn.ModuleList()
        for _col_idx, K_j in column_spec.categorical_specs:
            self.head_categorical.append(nn.Linear(hidden_dim, K_j))

        # Pre-register index tensors as buffers (moved to device automatically)
        self.register_buffer(
            "_cont_idx",
            torch.from_numpy(column_spec.continuous_idx).long(),
        )
        self.register_buffer(
            "_bin_idx",
            torch.from_numpy(column_spec.binary_idx).long(),
        )

    # ---- Encoder ----

    def _build_encoder_input(self, x: torch.Tensor) -> torch.Tensor:
        """Assemble encoder input from flat feature vector *x*.

        Continuous and binary columns are gathered as-is.  Multi-class
        categoricals are replaced by their learned embeddings.
        """
        parts: List[torch.Tensor] = []

        # Continuous columns (standardised floats)
        if self._cont_idx.numel() > 0:
            parts.append(x[:, self._cont_idx])

        # Binary columns (0/1 floats, unscaled)
        if self._bin_idx.numel() > 0:
            parts.append(x[:, self._bin_idx])

        # Multi-class categorical embeddings
        for i, (col_idx, _K_j) in enumerate(self.column_spec.categorical_specs):
            ids = x[:, col_idx].long()
            parts.append(self.cat_embeddings[i](ids))

        return torch.cat(parts, dim=1)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode raw input *x* to latent (μ, log σ²)."""
        h = self._build_encoder_input(x)
        h = self.encoder(h)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor,
    ) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # ---- Decoder ----

    def decode(self, z: torch.Tensor) -> DecoderOutput:
        """Decode latent *z* into type-specific reconstructions."""
        h = self.decoder_trunk(z)

        cont = self.head_continuous(h) if self.head_continuous is not None else h.new_empty(h.size(0), 0)
        binary = self.head_binary(h) if self.head_binary is not None else h.new_empty(h.size(0), 0)
        cat_logits = [head(h) for head in self.head_categorical]

        return DecoderOutput(continuous=cont, binary=binary, categorical=cat_logits)

    # ---- Forward ----

    def forward(
        self, x: torch.Tensor,
    ) -> Tuple[DecoderOutput, torch.Tensor, torch.Tensor]:
        """Full forward pass: x → (DecoderOutput, μ, log σ²)."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        decoder_out = self.decode(z)
        return decoder_out, mu, logvar
