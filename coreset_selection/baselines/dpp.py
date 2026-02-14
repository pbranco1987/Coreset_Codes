"""
Determinantal Point Process (DPP) baseline methods.

Contains:
- baseline_dpp: k-DPP sampling for diverse subsets
- baseline_dpp_quota: k-DPP with geographic quotas
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..geo import GeoInfo, GeographicConstraintProjector
from .utils import rff_features


def baseline_dpp(
    X: np.ndarray,
    k: int,
    sigma_sq: float,
    rff_dim: int = 500,
    seed: int = 0,
    max_iter: int = 1000,
) -> np.ndarray:
    """
    k-DPP sampling for diverse subset selection.
    
    Uses the greedy MAP approximation to k-DPP, which greedily
    selects points that maximize the determinant of the kernel
    submatrix.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix, shape (n_samples, n_features)
    k : int
        Number of points to select
    sigma_sq : float
        RBF kernel bandwidth
    rff_dim : int
        Number of random Fourier features for kernel approximation
    seed : int
        Random seed
    max_iter : int
        Maximum iterations for greedy selection
        
    Returns
    -------
    np.ndarray
        Selected indices
    """
    # Compute RFF features for kernel approximation
    Phi = rff_features(X, m=rff_dim, sigma_sq=sigma_sq, seed=seed)
    
    return greedy_kdpp_from_features(Phi, k, seed=seed + 1)


def greedy_kdpp_from_features(
    Phi: np.ndarray,
    k: int,
    seed: int = 0,
) -> np.ndarray:
    """
    Greedy k-DPP MAP approximation from feature matrix.
    
    The kernel is K[i,j] = Φ[i] · Φ[j]. Greedily maximizes
    log det(K_S) by selecting points that maximize marginal gain.
    
    Parameters
    ----------
    Phi : np.ndarray
        Feature matrix, shape (n_samples, n_features)
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
    n, d = Phi.shape
    rng = np.random.default_rng(seed)
    
    if k >= n:
        return np.arange(n)
    
    # Normalize features for numerical stability
    norms = np.linalg.norm(Phi, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    Phi_norm = Phi / norms
    
    selected = []
    selected_set = set()
    
    # Track orthogonal complement basis
    # Q stores orthonormal vectors spanning selected points
    Q = np.zeros((d, k), dtype=np.float64)
    
    for t in range(k):
        if t == 0:
            # First point: select by norm (quality)
            scores = np.sum(Phi ** 2, axis=1)
        else:
            # Subsequent: select by marginal gain in log-det
            # marginal gain ∝ ||φᵢ - proj_Q(φᵢ)||²
            # where Q spans already selected points
            Q_active = Q[:, :t]
            # Fast diagonal: ||proj_Q(φᵢ)||² = sum_j (φᵢ·q_j)² per row
            proj = Phi_norm @ Q_active          # (n, t) — O(n·t)
            proj_norms_sq = np.sum(proj ** 2, axis=1)  # (n,) — O(n·t)

            # Residual norm squared (diversity component)
            residual_sq = 1.0 - proj_norms_sq
            residual_sq = np.maximum(residual_sq, 0)
            
            # Quality component (original norm)
            quality = np.sum(Phi ** 2, axis=1)
            
            # Combined score: quality × diversity
            scores = quality * residual_sq
        
        # Mask out already selected
        if selected:
            scores[selected] = -np.inf
        
        # Select best
        max_score = np.max(scores)
        candidates = np.where(np.abs(scores - max_score) < 1e-10)[0]
        best_idx = rng.choice(candidates)
        
        selected.append(best_idx)
        selected_set.add(best_idx)
        
        # Update Q with new orthonormal vector
        if t < k - 1:
            new_vec = Phi_norm[best_idx].copy()
            if t > 0:
                Q_active = Q[:, :t]
                new_vec = new_vec - Q_active @ (Q_active.T @ new_vec)
            
            new_norm = np.linalg.norm(new_vec)
            if new_norm > 1e-10:
                Q[:, t] = new_vec / new_norm
    
    return np.array(selected, dtype=int)


def baseline_dpp_quota(
    X: np.ndarray,
    Phi: np.ndarray,
    geo: GeoInfo,
    k: int,
    alpha_geo: float,
    seed: int,
    min_one_per_group: bool = True,
) -> np.ndarray:
    """
    k-DPP with geographic quota constraints.
    
    Applies greedy k-DPP within each geographic group according
    to the KL-optimal quota allocation.
    
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
    
    # Apply k-DPP within each group
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
            # k-DPP within group
            local_sel = greedy_kdpp_from_features(Phi_g, count_g, seed=seed + g)
            selected.append(idx_g[local_sel])
    
    return np.concatenate(selected) if selected else np.array([], dtype=int)


def sample_exact_kdpp(
    L: np.ndarray,
    k: int,
    seed: int = 0,
    max_attempts: int = 100,
) -> np.ndarray:
    """
    Exact k-DPP sampling using eigendecomposition.
    
    WARNING: This is O(n³) and only suitable for small n.
    
    Parameters
    ----------
    L : np.ndarray
        L-ensemble kernel matrix, shape (n, n)
    k : int
        Size of subset to sample
    seed : int
        Random seed
    max_attempts : int
        Maximum sampling attempts
        
    Returns
    -------
    np.ndarray
        Selected indices
    """
    L = np.asarray(L, dtype=np.float64)
    n = L.shape[0]
    rng = np.random.default_rng(seed)
    
    if k >= n:
        return np.arange(n)
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    eigenvalues = np.maximum(eigenvalues, 0)  # Numerical stability
    
    # Sample k eigenvectors
    for attempt in range(max_attempts):
        # Sample each eigenvector independently with prob λᵢ/(1+λᵢ)
        probs = eigenvalues / (1 + eigenvalues)
        selected_eigvecs = rng.random(n) < probs
        
        if selected_eigvecs.sum() == k:
            break
        elif selected_eigvecs.sum() > k:
            # Remove excess
            selected_indices = np.where(selected_eigvecs)[0]
            keep = rng.choice(selected_indices, size=k, replace=False)
            selected_eigvecs = np.zeros(n, dtype=bool)
            selected_eigvecs[keep] = True
            break
        elif selected_eigvecs.sum() < k and attempt == max_attempts - 1:
            # Add more to reach k
            unselected = np.where(~selected_eigvecs)[0]
            add = rng.choice(unselected, size=k - selected_eigvecs.sum(), replace=False)
            selected_eigvecs[add] = True
    
    # Build V matrix from selected eigenvectors
    V = eigenvectors[:, selected_eigvecs]
    
    # Sample points one by one
    selected = []
    for _ in range(k):
        if V.shape[1] == 0:
            # Fallback: random selection
            remaining = np.setdiff1d(np.arange(n), selected)
            selected.append(rng.choice(remaining))
            continue
        
        # Probability proportional to row norm squared
        probs = np.sum(V ** 2, axis=1)
        probs[selected] = 0  # Exclude already selected
        probs = probs / (probs.sum() + 1e-30)
        
        idx = rng.choice(n, p=probs)
        selected.append(idx)
        
        # Update V by removing component in direction of selected point
        if len(selected) < k:
            v_idx = V[idx]
            v_norm = np.linalg.norm(v_idx)
            if v_norm > 1e-10:
                v_unit = v_idx / v_norm
                V = V - np.outer(V @ v_unit, v_unit)
    
    return np.array(selected, dtype=int)
