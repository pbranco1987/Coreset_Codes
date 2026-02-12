"""
Additional evaluation metrics for coreset quality.

Contains:
- coverage_stats: Mean/max distance coverage metrics
- diversity_metrics: Diversity measures for selected points
- representation_error: Feature-space representation error
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def coverage_stats(
    X: np.ndarray,
    S_idx: np.ndarray,
    eval_idx: np.ndarray = None,
) -> Tuple[float, float]:
    """
    Compute coverage statistics for a coreset.
    
    Coverage measures how well the coreset "covers" the data space.
    Returns the mean and maximum distance from any point to its
    nearest coreset point.
    
    Parameters
    ----------
    X : np.ndarray
        Full feature matrix, shape (N, d)
    S_idx : np.ndarray
        Indices of coreset points
    eval_idx : np.ndarray, optional
        Indices to evaluate coverage on. If None, uses all points.
        
    Returns
    -------
    Tuple[float, float]
        (mean_distance, max_distance) to nearest coreset point
    """
    X = np.asarray(X, dtype=np.float64)
    S_idx = np.asarray(S_idx, dtype=int)
    
    if len(S_idx) == 0:
        return np.inf, np.inf
    
    if eval_idx is None:
        eval_idx = np.arange(len(X))
    
    X_eval = X[eval_idx]
    X_coreset = X[S_idx]
    
    # Compute min distance from each eval point to coreset
    min_dists = np.full(len(X_eval), np.inf)
    
    # Chunked computation for memory efficiency
    chunk_size = 1000
    for i in range(0, len(X_coreset), chunk_size):
        X_chunk = X_coreset[i:i + chunk_size]
        # Pairwise distances: (n_eval, chunk_size)
        dists = np.sqrt(
            np.sum((X_eval[:, None, :] - X_chunk[None, :, :]) ** 2, axis=2)
        )
        min_dists = np.minimum(min_dists, dists.min(axis=1))
    
    return float(np.mean(min_dists)), float(np.max(min_dists))


def k_center_cost(
    X: np.ndarray,
    S_idx: np.ndarray,
) -> float:
    """
    Compute k-center objective (minimax distance).
    
    This is the maximum distance from any point to its nearest
    center (coreset point).
    
    Parameters
    ----------
    X : np.ndarray
        Full feature matrix
    S_idx : np.ndarray
        Indices of center points
        
    Returns
    -------
    float
        k-center cost (max min-distance)
    """
    _, max_dist = coverage_stats(X, S_idx)
    return max_dist


def k_median_cost(
    X: np.ndarray,
    S_idx: np.ndarray,
    weights: np.ndarray = None,
) -> float:
    """
    Compute k-median objective (sum of distances to nearest center).
    
    Parameters
    ----------
    X : np.ndarray
        Full feature matrix
    S_idx : np.ndarray
        Indices of center points
    weights : np.ndarray, optional
        Point weights. If None, uniform weights.
        
    Returns
    -------
    float
        k-median cost
    """
    X = np.asarray(X, dtype=np.float64)
    S_idx = np.asarray(S_idx, dtype=int)
    
    if len(S_idx) == 0:
        return np.inf
    
    N = len(X)
    if weights is None:
        weights = np.ones(N) / N
    
    X_coreset = X[S_idx]
    
    # Compute min distance from each point to coreset
    min_dists = np.full(N, np.inf)
    
    chunk_size = 1000
    for i in range(0, len(X_coreset), chunk_size):
        X_chunk = X_coreset[i:i + chunk_size]
        dists = np.sqrt(
            np.sum((X[:, None, :] - X_chunk[None, :, :]) ** 2, axis=2)
        )
        min_dists = np.minimum(min_dists, dists.min(axis=1))
    
    return float(np.sum(weights * min_dists))


def diversity_score(
    X: np.ndarray,
    S_idx: np.ndarray,
) -> float:
    """
    Compute diversity score (sum of pairwise distances in coreset).
    
    Higher is more diverse.
    
    Parameters
    ----------
    X : np.ndarray
        Full feature matrix
    S_idx : np.ndarray
        Indices of coreset points
        
    Returns
    -------
    float
        Diversity score
    """
    X = np.asarray(X, dtype=np.float64)
    S_idx = np.asarray(S_idx, dtype=int)
    
    if len(S_idx) < 2:
        return 0.0
    
    X_coreset = X[S_idx]
    
    # Sum of pairwise distances
    total = 0.0
    for i in range(len(X_coreset)):
        for j in range(i + 1, len(X_coreset)):
            total += np.linalg.norm(X_coreset[i] - X_coreset[j])
    
    return float(total)


def min_pairwise_distance(
    X: np.ndarray,
    S_idx: np.ndarray,
) -> float:
    """
    Compute minimum pairwise distance in coreset.
    
    Useful for detecting clustering/redundancy.
    
    Parameters
    ----------
    X : np.ndarray
        Full feature matrix
    S_idx : np.ndarray
        Indices of coreset points
        
    Returns
    -------
    float
        Minimum pairwise distance
    """
    X = np.asarray(X, dtype=np.float64)
    S_idx = np.asarray(S_idx, dtype=int)
    
    if len(S_idx) < 2:
        return np.inf
    
    X_coreset = X[S_idx]
    
    min_dist = np.inf
    for i in range(len(X_coreset)):
        for j in range(i + 1, len(X_coreset)):
            d = np.linalg.norm(X_coreset[i] - X_coreset[j])
            min_dist = min(min_dist, d)
    
    return float(min_dist)


def representation_error(
    X: np.ndarray,
    S_idx: np.ndarray,
    method: str = "mean",
) -> float:
    """
    Compute representation error in feature space.
    
    Measures how well the coreset represents the full dataset
    statistics.
    
    Parameters
    ----------
    X : np.ndarray
        Full feature matrix
    S_idx : np.ndarray
        Indices of coreset points
    method : str
        Error type: "mean" (mean difference), "cov" (covariance difference)
        
    Returns
    -------
    float
        Representation error
    """
    X = np.asarray(X, dtype=np.float64)
    S_idx = np.asarray(S_idx, dtype=int)
    
    if len(S_idx) == 0:
        return np.inf
    
    X_coreset = X[S_idx]
    
    if method == "mean":
        # L2 error in mean
        mean_full = X.mean(axis=0)
        mean_coreset = X_coreset.mean(axis=0)
        return float(np.linalg.norm(mean_full - mean_coreset))
    
    elif method == "cov":
        # Frobenius norm of covariance difference
        cov_full = np.cov(X.T)
        cov_coreset = np.cov(X_coreset.T)
        
        # Handle 1D case
        if cov_full.ndim == 0:
            return float(abs(cov_full - cov_coreset))
        
        return float(np.linalg.norm(cov_full - cov_coreset, 'fro'))
    
    else:
        raise ValueError(f"Unknown method: {method}")


def all_metrics(
    X: np.ndarray,
    S_idx: np.ndarray,
    eval_idx: np.ndarray = None,
) -> Dict[str, float]:
    """
    Compute all coverage and diversity metrics.
    
    Parameters
    ----------
    X : np.ndarray
        Full feature matrix
    S_idx : np.ndarray
        Indices of coreset points
    eval_idx : np.ndarray, optional
        Indices for coverage evaluation
        
    Returns
    -------
    Dict[str, float]
        Dictionary of all metrics
    """
    mean_cov, max_cov = coverage_stats(X, S_idx, eval_idx)
    
    return {
        "coverage_mean": mean_cov,
        "coverage_max": max_cov,
        "k_center_cost": max_cov,
        "k_median_cost": k_median_cost(X, S_idx),
        "diversity": diversity_score(X, S_idx),
        "min_pairwise_dist": min_pairwise_distance(X, S_idx),
        "mean_repr_error": representation_error(X, S_idx, "mean"),
        "cov_repr_error": representation_error(X, S_idx, "cov"),
    }
