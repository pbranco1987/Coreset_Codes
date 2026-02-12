"""
R11 Post-hoc Diagnostics -- Objective computation functions.

Split from ``r7_diagnostics.py`` for maintainability.
Public API is re-exported via ``r7_diagnostics.py`` (facade).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


# =============================================================================
# Surrogate Sensitivity Analysis (Section VIII.K -- Table IV)
# =============================================================================


@dataclass
class SurrogateSensitivityConfig:
    """
    Configuration for surrogate sensitivity analysis.

    Per manuscript Section VIII.K:
    - RFF dimension sweep: m \u2208 {500, 1000, 2000, 4000}
    - Anchor count sweep: A \u2208 {50, 100, 200, 400}
    - Reference setting: m=2000, A=200 (default)

    Attributes
    ----------
    rff_dims : List[int]
        RFF dimensions to test.
    anchor_counts : List[int]
        Anchor counts to test.
    reference_rff : int
        Reference RFF dimension.
    reference_anchors : int
        Reference anchor count.
    n_samples : int
        Number of subsets to sample for correlation analysis.
    """
    rff_dims: List[int] = field(default_factory=lambda: [500, 1000, 2000, 4000])
    anchor_counts: List[int] = field(default_factory=lambda: [50, 100, 200, 400])
    reference_rff: int = 2000
    reference_anchors: int = 200
    n_samples: int = 100


def compute_rff_mmd(
    X: np.ndarray,
    idx: np.ndarray,
    rff_dim: int,
    bandwidth: float,
    seed: int = 0,
) -> float:
    """
    Compute MMD\u00b2 using Random Fourier Features.

    Parameters
    ----------
    X : np.ndarray
        Full dataset, shape (N, d).
    idx : np.ndarray
        Subset indices.
    rff_dim : int
        Number of random Fourier features.
    bandwidth : float
        RBF kernel bandwidth (\u03c3\u00b2).
    seed : int
        Random seed for RFF sampling.

    Returns
    -------
    float
        Estimated MMD\u00b2 value.
    """
    rng = np.random.default_rng(seed)
    d = X.shape[1]

    # Sample random frequencies \u03c9 ~ N(0, \u03c3\u207b\u00b2I)
    W = rng.normal(size=(d, rff_dim)) / np.sqrt(bandwidth)
    b = rng.uniform(0, 2 * np.pi, size=rff_dim)

    # Compute random features  \u03c6(x) = \u221a(2/m) cos(Wx + b)
    def rff(Z):
        return np.sqrt(2.0 / rff_dim) * np.cos(Z @ W + b)

    phi_full = rff(X)
    phi_sub = rff(X[idx])

    # Mean embeddings
    mu_full = phi_full.mean(axis=0)
    mu_sub = phi_sub.mean(axis=0)

    # MMD\u00b2 estimate = ||\u03bc_P \u2212 \u03bc_Q||\u00b2
    mmd = np.sum((mu_full - mu_sub) ** 2)
    return float(mmd)


def compute_anchored_sinkhorn(
    X: np.ndarray,
    idx: np.ndarray,
    n_anchors: int,
    eta: float = 0.05,
    max_iter: int = 100,
    seed: int = 0,
) -> float:
    """
    Compute Sinkhorn divergence with anchor approximation.

    Parameters
    ----------
    X : np.ndarray
        Full dataset, shape (N, d).
    idx : np.ndarray
        Subset indices.
    n_anchors : int
        Number of anchor points.
    eta : float
        Regularization scale (\u03b5 = \u03b7 \u00b7 median(d\u00b2)).
    max_iter : int
        Maximum Sinkhorn iterations.
    seed : int
        Random seed.

    Returns
    -------
    float
        Anchored Sinkhorn divergence estimate.
    """
    from ..objectives.sinkhorn import AnchorSinkhorn
    from ..config.dataclasses import SinkhornConfig

    cfg = SinkhornConfig(
        n_anchors=n_anchors,
        eta=eta,
        max_iter=max_iter,
        anchor_method="kmeans",
    )

    sink = AnchorSinkhorn.build(X, cfg, seed=seed)
    return sink.sinkhorn_divergence_subset(X, idx)
