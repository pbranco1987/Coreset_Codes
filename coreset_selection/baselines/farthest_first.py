"""
Farthest-first traversal (k-center) baseline methods.

Contains:
- baseline_farthest_first: Greedy farthest-first point selection
- baseline_farthest_first_quota: Farthest-first with geographic quotas
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..geo import GeoInfo, GeographicConstraintProjector


def baseline_farthest_first(
    X: np.ndarray,
    k: int,
    seed: int,
    initial_idx: Optional[int] = None,
) -> np.ndarray:
    """
    Farthest-first traversal (greedy k-center).
    
    Greedily selects points that maximize minimum distance to the
    current selection. This approximates the k-center objective.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix, shape (n_samples, n_features)
    k : int
        Number of points to select
    seed : int
        Random seed (for initial point if not specified)
    initial_idx : Optional[int]
        Index of initial point. If None, chosen randomly.
        
    Returns
    -------
    np.ndarray
        Selected indices in order of selection
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    
    if k >= n:
        return np.arange(n)
    
    # Initialize with first point
    if initial_idx is None:
        initial_idx = rng.integers(n)
    
    selected = [initial_idx]
    selected_set = {initial_idx}
    
    # Track minimum distance to selected set for each point
    min_dists = np.full(n, np.inf)
    min_dists[initial_idx] = 0
    
    # Update distances from first point
    dists_to_first = np.linalg.norm(X - X[initial_idx], axis=1)
    min_dists = np.minimum(min_dists, dists_to_first)
    
    for _ in range(k - 1):
        # Mask out already selected
        min_dists_masked = min_dists.copy()
        for idx in selected:
            min_dists_masked[idx] = -np.inf
        
        # Select farthest point
        next_idx = np.argmax(min_dists_masked)
        selected.append(next_idx)
        selected_set.add(next_idx)
        
        # Update minimum distances
        dists_to_new = np.linalg.norm(X - X[next_idx], axis=1)
        min_dists = np.minimum(min_dists, dists_to_new)
    
    return np.array(selected, dtype=int)


def baseline_farthest_first_quota(
    X: np.ndarray,
    geo: GeoInfo,
    k: int,
    alpha_geo: float,
    seed: int,
    min_one_per_group: bool = True,
) -> np.ndarray:
    """
    Farthest-first traversal with geographic quota constraints.
    
    Applies farthest-first within each geographic group according
    to the KL-optimal quota allocation.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix, shape (n_samples, n_features)
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
    X = np.asarray(X, dtype=np.float64)
    
    # Get quota allocation
    projector = GeographicConstraintProjector(
        geo=geo,
        alpha_geo=alpha_geo,
        min_one_per_group=min_one_per_group,
    )
    target_counts = projector.target_counts(k)
    
    # Apply farthest-first within each group
    selected = []
    for g in range(geo.G):
        count_g = int(target_counts[g])
        if count_g == 0:
            continue
            
        idx_g = geo.group_to_indices[g]
        X_g = X[idx_g]
        
        if count_g >= len(idx_g):
            selected.append(idx_g)
        else:
            # Farthest-first within group
            local_sel = _farthest_first_local(X_g, count_g, seed=seed + g)
            selected.append(idx_g[local_sel])
    
    return np.concatenate(selected) if selected else np.array([], dtype=int)


def _farthest_first_local(
    X: np.ndarray,
    k: int,
    seed: int,
) -> np.ndarray:
    """
    Local farthest-first for a subset.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix for the subset
    k : int
        Number of points to select
    seed : int
        Random seed
        
    Returns
    -------
    np.ndarray
        Local indices of selected points
    """
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    
    if k >= n:
        return np.arange(n)
    
    # Start from centroid-closest point for better coverage
    centroid = X.mean(axis=0)
    dists_to_centroid = np.linalg.norm(X - centroid, axis=1)
    initial_idx = np.argmin(dists_to_centroid)
    
    selected = [initial_idx]
    min_dists = np.linalg.norm(X - X[initial_idx], axis=1)
    min_dists[initial_idx] = -np.inf
    
    for _ in range(k - 1):
        next_idx = np.argmax(min_dists)
        selected.append(next_idx)
        
        dists_to_new = np.linalg.norm(X - X[next_idx], axis=1)
        min_dists = np.minimum(min_dists, dists_to_new)
        min_dists[next_idx] = -np.inf
    
    return np.array(selected, dtype=int)


def baseline_farthest_first_global_then_quota(
    X: np.ndarray,
    geo: GeoInfo,
    k: int,
    alpha_geo: float,
    seed: int,
    oversample_factor: float = 2.0,
    min_one_per_group: bool = True,
) -> np.ndarray:
    """
    Global farthest-first followed by quota adjustment.
    
    First runs farthest-first globally with oversampling, then
    adjusts to satisfy quota constraints.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix, shape (n_samples, n_features)
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
    
    # Oversample with global farthest-first
    k_oversample = min(int(k * oversample_factor), X.shape[0])
    oversampled = baseline_farthest_first(X, k_oversample, seed=seed)
    
    # Adjust to quota
    selected = ensure_quota_feasible(
        idx_selected=oversampled,
        group_ids=geo.group_ids,
        target_counts=target_counts,
        rng=rng,
    )
    
    return selected


def kcenter_cost(X: np.ndarray, selected: np.ndarray) -> float:
    """
    Compute k-center cost (maximum distance to nearest center).
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    selected : np.ndarray
        Indices of selected centers
        
    Returns
    -------
    float
        Maximum distance from any point to its nearest center
    """
    X = np.asarray(X, dtype=np.float64)
    centers = X[selected]
    
    # Compute distance from each point to nearest center
    min_dists = np.full(X.shape[0], np.inf)
    for c in centers:
        dists = np.linalg.norm(X - c, axis=1)
        min_dists = np.minimum(min_dists, dists)
    
    return float(np.max(min_dists))
