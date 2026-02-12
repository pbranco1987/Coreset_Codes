"""
VAE neural network architecture classes.

Contains the nn.Module subclasses for encoder/decoder used by the VAE.
Split from models/vae.py for modularity.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


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
        """Encode input to latent mean and log-variance (Sec V-A, \u03b2-VAE encoder)."""
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """Reparameterization trick: z = \u03bc + \u03c3 \u2299 \u03b5, \u03b5 ~ N(0,I) (Sec V-A)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent z to reconstructed input (Sec V-A, \u03b2-VAE decoder)."""
        return self.decoder(z)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass: x \u2192 (x\u0302, \u03bc, log \u03c3\u00b2) (Sec V-A)."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
