"""
Shared utilities for baseline coreset selection methods.

Contains:
- weighted_sample_without_replacement for importance sampling
- rff_features for Random Fourier Feature computation
- ridge_leverage_scores_from_features for leverage score computation
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def weighted_sample_without_replacement(
    idx_pool: np.ndarray,
    weights: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample k indices without replacement according to weights.
    
    Uses the Gumbel-max trick for efficient weighted sampling.
    
    Parameters
    ----------
    idx_pool : np.ndarray
        Array of candidate indices
    weights : np.ndarray
        Sampling weights (need not be normalized)
    k : int
        Number of samples to draw
    rng : np.random.Generator
        Random number generator
        
    Returns
    -------
    np.ndarray
        Selected indices
        
    Raises
    ------
    ValueError
        If k < 0, k > len(idx_pool), or weights are invalid
    """
    idx_pool = np.asarray(idx_pool)
    weights = np.asarray(weights, dtype=np.float64)
    
    if len(idx_pool) != len(weights):
        raise ValueError("idx_pool and weights must have same length")
    
    # Validate k
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    if k == 0:
        return np.array([], dtype=idx_pool.dtype)
    if k > len(idx_pool):
        raise ValueError(f"k={k} exceeds pool size {len(idx_pool)}")
    
    # Validate weights
    if not np.all(np.isfinite(weights)):
        raise ValueError("weights must be finite")
    if np.sum(weights) <= 0:
        raise ValueError("weights must have positive sum")
    
    # Normalize weights
    weights = np.maximum(weights, 1e-30)
    weights = weights / weights.sum()
    
    # Gumbel-max trick: add Gumbel noise to log-weights, take top-k
    log_weights = np.log(weights)
    gumbel_noise = -np.log(-np.log(rng.uniform(size=len(weights)) + 1e-30) + 1e-30)
    perturbed = log_weights + gumbel_noise
    
    # Select top-k
    top_k_idx = np.argpartition(perturbed, -k)[-k:]
    
    return idx_pool[top_k_idx]


def rff_features(
    X: np.ndarray,
    m: int,
    sigma_sq: float,
    seed: int,
) -> np.ndarray:
    """
    Compute Random Fourier Features for RBF kernel approximation.
    
    Approximates k(x, y) ≈ φ(x)ᵀφ(y) where k is the RBF kernel
    with bandwidth σ².
    
    Parameters
    ----------
    X : np.ndarray
        Input data, shape (n_samples, n_features)
    m : int
        Number of random features (output dimension = m)
    sigma_sq : float
        RBF kernel bandwidth (variance)
    seed : int
        Random seed
        
    Returns
    -------
    np.ndarray
        RFF features, shape (n_samples, m)
        
    Raises
    ------
    ValueError
        If sigma_sq <= 0 or not finite, or if m <= 0
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    m = int(m)
    
    # Validate parameters
    sigma_sq = float(sigma_sq)
    if not np.isfinite(sigma_sq) or sigma_sq <= 0:
        raise ValueError(f"sigma_sq must be positive and finite, got {sigma_sq}")
    if m <= 0:
        raise ValueError(f"m (rff_dim) must be positive, got {m}")
    
    rng = np.random.default_rng(seed)
    
    # Sample frequencies from N(0, 1/(2σ²))
    # For RBF: k(x,y) = exp(-||x-y||²/(2σ²))
    # Frequency scale = 1/σ
    freq_scale = 1.0 / np.sqrt(sigma_sq)
    W = rng.standard_normal((d, m)) * freq_scale
    b = rng.uniform(0, 2 * np.pi, size=m)
    
    # Compute features per manuscript: sqrt(2/m) * cos(Xω + b)
    projection = X @ W + b
    Phi = np.sqrt(2.0 / m) * np.cos(projection)
    
    return Phi


def ridge_leverage_scores_from_features(
    Phi: np.ndarray,
    ridge: float = 1e-6,
) -> np.ndarray:
    """
    Compute ridge leverage scores from feature matrix.
    
    Ridge leverage score of point i is:
        ℓᵢ = φᵢᵀ(ΦᵀΦ + λI)⁻¹φᵢ
    
    Parameters
    ----------
    Phi : np.ndarray
        Feature matrix, shape (n_samples, n_features)
    ridge : float
        Regularization parameter
        
    Returns
    -------
    np.ndarray
        Leverage scores, shape (n_samples,)
    """
    Phi = np.asarray(Phi, dtype=np.float64)
    n, m = Phi.shape
    
    # Compute (ΦᵀΦ + λI)⁻¹
    gram = Phi.T @ Phi + ridge * np.eye(m)
    
    try:
        # Use Cholesky for numerical stability
        L = np.linalg.cholesky(gram)
        # Solve L @ Z = Φᵀ
        Z = np.linalg.solve(L, Phi.T)
        # Leverage scores are column norms squared
        scores = np.sum(Z ** 2, axis=0)
    except np.linalg.LinAlgError:
        # Fall back to pseudoinverse
        gram_inv = np.linalg.pinv(gram)
        scores = np.einsum('ij,jk,ik->i', Phi, gram_inv, Phi)
    
    return scores


def quota_sample(
    indices_by_group: list,
    target_counts: np.ndarray,
    weights_by_group: Optional[list] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Sample from groups according to target quota counts.
    
    Parameters
    ----------
    indices_by_group : list
        List of arrays, each containing indices for one group
    target_counts : np.ndarray
        Number of samples to draw from each group
    weights_by_group : Optional[list]
        Optional weights for each group (for weighted sampling)
    rng : Optional[np.random.Generator]
        Random number generator (default: np.random.default_rng())
        
    Returns
    -------
    np.ndarray
        Selected indices
    """
    if rng is None:
        rng = np.random.default_rng()
    
    target_counts = np.asarray(target_counts, dtype=int)
    
    selected = []
    for g, (idx_g, count) in enumerate(zip(indices_by_group, target_counts)):
        idx_g = np.asarray(idx_g)
        count = int(count)
        
        if count == 0:
            continue
        
        if count > len(idx_g):
            raise ValueError(f"Group {g}: need {count} samples but only {len(idx_g)} available")
        
        if weights_by_group is not None and weights_by_group[g] is not None:
            # Weighted sampling
            w = np.asarray(weights_by_group[g])
            chosen = weighted_sample_without_replacement(idx_g, w, count, rng)
        else:
            # Uniform sampling
            chosen = rng.choice(idx_g, size=count, replace=False)
        
        selected.append(chosen)
    
    return np.concatenate(selected) if selected else np.array([], dtype=int)


def ensure_quota_feasible(
    idx_selected: np.ndarray,
    group_ids: np.ndarray,
    target_counts: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Adjust a selection to match quota constraints.
    
    Parameters
    ----------
    idx_selected : np.ndarray
        Currently selected indices
    group_ids : np.ndarray
        Group assignment for all points
    target_counts : np.ndarray
        Target count for each group
    rng : Optional[np.random.Generator]
        Random number generator
        
    Returns
    -------
    np.ndarray
        Adjusted selection satisfying quotas
    """
    if rng is None:
        rng = np.random.default_rng()
    
    idx_selected = np.asarray(idx_selected)
    group_ids = np.asarray(group_ids)
    target_counts = np.asarray(target_counts, dtype=int)
    
    n_groups = len(target_counts)
    
    # Build indices by group for full dataset
    all_indices_by_group = [np.where(group_ids == g)[0] for g in range(n_groups)]
    
    # Current counts per group
    current_counts = np.zeros(n_groups, dtype=int)
    for idx in idx_selected:
        g = group_ids[idx]
        current_counts[g] += 1
    
    # Adjust each group
    new_selection = []
    for g in range(n_groups):
        target = target_counts[g]
        current = current_counts[g]
        
        # Get currently selected in this group
        mask = np.isin(idx_selected, all_indices_by_group[g])
        selected_in_g = idx_selected[mask]
        
        if current == target:
            # Exact match
            new_selection.append(selected_in_g)
        elif current > target:
            # Remove excess
            keep = rng.choice(selected_in_g, size=target, replace=False)
            new_selection.append(keep)
        else:
            # Add more
            unselected = np.setdiff1d(all_indices_by_group[g], selected_in_g)
            needed = target - current
            if len(unselected) < needed:
                raise ValueError(f"Group {g}: cannot add {needed} points, only {len(unselected)} available")
            add = rng.choice(unselected, size=needed, replace=False)
            new_selection.append(np.concatenate([selected_in_g, add]))
    
    return np.concatenate(new_selection) if new_selection else np.array([], dtype=int)
