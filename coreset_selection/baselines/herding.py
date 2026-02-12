"""
Kernel herding baseline methods.

Contains:
- kernel_herding_rff: Kernel herding using Random Fourier Features
- baseline_kernel_herding_quota: Kernel herding with geographic quotas
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..geo import GeoInfo, GeographicConstraintProjector
from .utils import rff_features


def kernel_herding_rff(
    Phi: np.ndarray,
    k: int,
    seed: int = 0,
) -> np.ndarray:
    """
    Kernel herding using pre-computed RFF features.
    
    Greedily selects points to minimize MMD to the full distribution.
    At each step, selects the point that most reduces the mean embedding
    discrepancy.
    
    Parameters
    ----------
    Phi : np.ndarray
        RFF feature matrix, shape (n_samples, n_features)
    k : int
        Number of points to select
    seed : int
        Random seed (for tie-breaking)
        
    Returns
    -------
    np.ndarray
        Selected indices in order of selection
    """
    Phi = np.asarray(Phi, dtype=np.float64)
    n, m = Phi.shape
    rng = np.random.default_rng(seed)
    
    if k >= n:
        return np.arange(n)
    
    # Mean embedding of full distribution
    mu_full = Phi.mean(axis=0)
    
    # Track selected points and running sum
    selected = []
    selected_set = set()
    running_sum = np.zeros(m)
    
    for t in range(k):
        # Current mean of selected (if any)
        if t == 0:
            # First point: maximize alignment with mean
            scores = Phi @ mu_full
        else:
            # Subsequent: maximize reduction in MMD
            # New mean would be (running_sum + Phi[i]) / (t + 1)
            # Want to minimize || (running_sum + Phi[i]) / (t + 1) - mu_full ||²
            # Equivalent to maximizing Phi[i] · (mu_full - running_sum / (t + 1))
            # Or: Phi[i] · ((t+1) * mu_full - running_sum) / (t + 1)
            target = (t + 1) * mu_full - running_sum
            scores = Phi @ target
        
        # Mask out already selected
        for idx in selected:
            scores[idx] = -np.inf
        
        # Select best (with random tie-breaking)
        max_score = scores.max()
        candidates = np.where(np.abs(scores - max_score) < 1e-10)[0]
        best_idx = rng.choice(candidates)
        
        selected.append(best_idx)
        selected_set.add(best_idx)
        running_sum += Phi[best_idx]
    
    return np.array(selected, dtype=int)


def baseline_kernel_herding(
    X: np.ndarray,
    k: int,
    sigma_sq: float,
    rff_dim: int = 1000,
    seed: int = 0,
) -> np.ndarray:
    """
    Kernel herding baseline using RBF kernel.
    
    Computes RFF features and applies kernel herding.
    
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
    seed : int
        Random seed
        
    Returns
    -------
    np.ndarray
        Selected indices
    """
    # Compute RFF features
    Phi = rff_features(X, m=rff_dim, sigma_sq=sigma_sq, seed=seed)
    
    # Apply herding
    return kernel_herding_rff(Phi, k, seed=seed + 1)


def baseline_kernel_herding_quota(
    X: np.ndarray,
    Phi: np.ndarray,
    geo: GeoInfo,
    k: int,
    alpha_geo: float,
    seed: int,
    min_one_per_group: bool = True,
) -> np.ndarray:
    """
    Kernel herding with geographic quota constraints.
    
    Applies kernel herding within each geographic group according
    to the KL-optimal quota allocation.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix, shape (n_samples, n_features)
    Phi : np.ndarray
        Pre-computed RFF features, shape (n_samples, rff_dim)
    geo : GeoInfo
        Geographic group information
    k : int
        Total number of points to select
    alpha_geo : float
        Dirichlet smoothing parameter
    seed : int
        Random seed
    min_one_per_group : bool
        Whether to ensure at least one per group
        
    Returns
    -------
    np.ndarray
        Selected indices
    """
    Phi = np.asarray(Phi, dtype=np.float64)
    
    # Get quota allocation
    projector = GeographicConstraintProjector(
        geo=geo,
        alpha_geo=alpha_geo,
        min_one_per_group=min_one_per_group,
    )
    target_counts = projector.target_counts(k)
    
    # Apply herding within each group
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
            # Herding within group
            local_sel = kernel_herding_rff(Phi_g, count_g, seed=seed + g)
            selected.append(idx_g[local_sel])
    
    return np.concatenate(selected) if selected else np.array([], dtype=int)


def baseline_herding_global_then_quota(
    Phi: np.ndarray,
    geo: GeoInfo,
    k: int,
    alpha_geo: float,
    seed: int,
    oversample_factor: float = 2.0,
    min_one_per_group: bool = True,
) -> np.ndarray:
    """
    Global kernel herding followed by quota adjustment.
    
    First runs kernel herding globally with oversampling, then
    adjusts to satisfy quota constraints.
    
    Parameters
    ----------
    Phi : np.ndarray
        RFF features, shape (n_samples, rff_dim)
    geo : GeoInfo
        Geographic group information
    k : int
        Target number of points
    alpha_geo : float
        Dirichlet smoothing parameter
    seed : int
        Random seed
    oversample_factor : float
        Factor to oversample before quota adjustment
    min_one_per_group : bool
        Whether to ensure at least one per group
        
    Returns
    -------
    np.ndarray
        Selected indices
    """
    from .utils import ensure_quota_feasible
    
    rng = np.random.default_rng(seed)
    
    # Get quota allocation
    projector = GeographicConstraintProjector(
        geo=geo,
        alpha_geo=alpha_geo,
        min_one_per_group=min_one_per_group,
    )
    target_counts = projector.target_counts(k)
    
    # Oversample with global herding
    k_oversample = min(int(k * oversample_factor), Phi.shape[0])
    oversampled = kernel_herding_rff(Phi, k_oversample, seed=seed)
    
    # Adjust to quota
    selected = ensure_quota_feasible(
        idx_selected=oversampled,
        group_ids=geo.group_ids,
        target_counts=target_counts,
        rng=rng,
    )
    
    return selected
