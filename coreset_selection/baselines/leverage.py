"""
Ridge leverage score sampling baseline methods.

Contains:
- baseline_rls: Ridge leverage score sampling
- baseline_rls_quota: Ridge leverage scores with geographic quotas
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..geo import GeoInfo, GeographicConstraintProjector
from .utils import (
    rff_features,
    ridge_leverage_scores_from_features,
    weighted_sample_without_replacement,
    quota_sample,
)


def baseline_rls(
    X: np.ndarray,
    k: int,
    sigma_sq: float,
    rff_dim: int = 1000,
    ridge: float = 1e-6,
    seed: int = 0,
) -> np.ndarray:
    """
    Ridge leverage score (RLS) sampling.
    
    Samples points with probability proportional to their ridge leverage
    scores in the RFF feature space. This provides theoretical guarantees
    for kernel approximation.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix, shape (n_samples, n_features)
    k : int
        Number of points to select
    sigma_sq : float
        RBF kernel bandwidth
    rff_dim : int
        Number of random Fourier features
    ridge : float
        Ridge regularization parameter
    seed : int
        Random seed
        
    Returns
    -------
    np.ndarray
        Selected indices
    """
    # Compute RFF features
    Phi = rff_features(X, m=rff_dim, sigma_sq=sigma_sq, seed=seed)
    
    # Compute leverage scores
    scores = ridge_leverage_scores_from_features(Phi, ridge=ridge)
    
    # Sample proportionally to scores
    rng = np.random.default_rng(seed + 1)
    idx_pool = np.arange(len(X))
    
    return weighted_sample_without_replacement(idx_pool, scores, k, rng)


def baseline_rls_from_phi(
    Phi: np.ndarray,
    k: int,
    ridge: float = 1e-6,
    seed: int = 0,
) -> np.ndarray:
    """
    Ridge leverage score sampling from pre-computed features.
    
    Parameters
    ----------
    Phi : np.ndarray
        Feature matrix (e.g., RFF features)
    k : int
        Number of points to select
    ridge : float
        Ridge regularization parameter
    seed : int
        Random seed
        
    Returns
    -------
    np.ndarray
        Selected indices
    """
    # Compute leverage scores
    scores = ridge_leverage_scores_from_features(Phi, ridge=ridge)
    
    # Sample proportionally to scores
    rng = np.random.default_rng(seed)
    idx_pool = np.arange(len(Phi))
    
    return weighted_sample_without_replacement(idx_pool, scores, k, rng)


def baseline_rls_quota(
    X: np.ndarray,
    Phi: np.ndarray,
    geo: GeoInfo,
    k: int,
    alpha_geo: float,
    ridge: float = 1e-6,
    seed: int = 0,
    min_one_per_group: bool = True,
) -> np.ndarray:
    """
    Ridge leverage score sampling with geographic quotas.
    
    Samples within each group proportionally to leverage scores,
    respecting the KL-optimal quota allocation.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix, shape (n_samples, n_features)
    Phi : np.ndarray
        Pre-computed RFF features
    geo : GeoInfo
        Geographic group information
    k : int
        Total number of points to select
    alpha_geo : float
        Dirichlet smoothing parameter
    ridge : float
        Ridge regularization parameter
    seed : int
        Random seed
    min_one_per_group : bool
        Whether to ensure at least one per group
        
    Returns
    -------
    np.ndarray
        Selected indices
    """
    rng = np.random.default_rng(seed)
    
    # Compute leverage scores globally (captures relative importance)
    scores = ridge_leverage_scores_from_features(Phi, ridge=ridge)
    
    # Get quota allocation
    projector = GeographicConstraintProjector(
        geo=geo,
        alpha_geo=alpha_geo,
        min_one_per_group=min_one_per_group,
    )
    target_counts = projector.target_counts(k)
    
    # Sample within each group proportionally to leverage scores
    weights_by_group = [scores[idx_g] for idx_g in geo.group_to_indices]
    
    selected = quota_sample(
        indices_by_group=geo.group_to_indices,
        target_counts=target_counts,
        weights_by_group=weights_by_group,
        rng=rng,
    )
    
    return selected


def baseline_rls_local_quota(
    Phi: np.ndarray,
    geo: GeoInfo,
    k: int,
    alpha_geo: float,
    ridge: float = 1e-6,
    seed: int = 0,
    min_one_per_group: bool = True,
) -> np.ndarray:
    """
    RLS with local leverage scores per group.
    
    Computes leverage scores independently within each group,
    then samples proportionally within each group.
    
    Parameters
    ----------
    Phi : np.ndarray
        Pre-computed RFF features
    geo : GeoInfo
        Geographic group information
    k : int
        Total number of points to select
    alpha_geo : float
        Dirichlet smoothing parameter
    ridge : float
        Ridge regularization parameter
    seed : int
        Random seed
    min_one_per_group : bool
        Whether to ensure at least one per group
        
    Returns
    -------
    np.ndarray
        Selected indices
    """
    rng = np.random.default_rng(seed)
    
    # Get quota allocation
    projector = GeographicConstraintProjector(
        geo=geo,
        alpha_geo=alpha_geo,
        min_one_per_group=min_one_per_group,
    )
    target_counts = projector.target_counts(k)
    
    # Compute local leverage scores and sample
    selected = []
    for g in range(geo.G):
        count_g = int(target_counts[g])
        if count_g == 0:
            continue
            
        idx_g = geo.group_to_indices[g]
        Phi_g = Phi[idx_g]
        
        if count_g >= len(idx_g):
            selected.append(idx_g)
        else:
            # Local leverage scores
            scores_g = ridge_leverage_scores_from_features(Phi_g, ridge=ridge)
            
            # Sample within group
            local_sel = weighted_sample_without_replacement(
                np.arange(len(idx_g)), scores_g, count_g, rng
            )
            selected.append(idx_g[local_sel])
    
    return np.concatenate(selected) if selected else np.array([], dtype=int)


def compute_effective_dimension(
    Phi: np.ndarray,
    ridge: float = 1e-6,
) -> float:
    """
    Compute statistical effective dimension (sum of leverage scores).
    
    The effective dimension d_eff = tr(Φ(ΦᵀΦ + λI)⁻¹Φᵀ) = Σᵢ ℓᵢ
    
    Parameters
    ----------
    Phi : np.ndarray
        Feature matrix
    ridge : float
        Ridge parameter
        
    Returns
    -------
    float
        Effective dimension
    """
    scores = ridge_leverage_scores_from_features(Phi, ridge=ridge)
    return float(np.sum(scores))


def optimal_rls_sample_size(
    Phi: np.ndarray,
    ridge: float = 1e-6,
    epsilon: float = 0.1,
    delta: float = 0.1,
) -> int:
    """
    Compute theoretically optimal RLS sample size.
    
    For (1+ε)-approximation with probability 1-δ, need
    O(d_eff · log(d_eff/δ) / ε²) samples.
    
    Parameters
    ----------
    Phi : np.ndarray
        Feature matrix
    ridge : float
        Ridge parameter
    epsilon : float
        Approximation error tolerance
    delta : float
        Failure probability
        
    Returns
    -------
    int
        Recommended sample size
    """
    d_eff = compute_effective_dimension(Phi, ridge)
    
    # Simplified bound: O(d_eff * log(d_eff) / ε²)
    sample_size = int(np.ceil(
        d_eff * np.log(d_eff / delta + 1) / (epsilon ** 2)
    ))
    
    return max(1, min(sample_size, len(Phi)))
