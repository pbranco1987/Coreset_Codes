"""
Models module for coreset selection.

This module provides:
- TabularVAE: Variational Autoencoder for tabular data
- VAETrainer: Training and embedding utilities
- PCA utilities for dimensionality reduction
"""

from .vae import TabularVAE, VAETrainer

from .pca import (
    fit_pca,
    pca_embed,
    explained_variance_ratio,
    cumulative_explained_variance,
    components_for_variance,
    IncrementalPCAWrapper,
)

__all__ = [
    # VAE
    "TabularVAE",
    "VAETrainer",
    # PCA
    "fit_pca",
    "pca_embed",
    "explained_variance_ratio",
    "cumulative_explained_variance",
    "components_for_variance",
    "IncrementalPCAWrapper",
]
