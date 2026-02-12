"""
K-means based baseline methods.

Contains:
- baseline_kmeans_reps: K-means cluster representative selection
- baseline_kmeans_reps_quota: K-means with geographic quotas
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

from ..geo import GeoInfo, GeographicConstraintProjector
from .utils import quota_sample


def baseline_kmeans_reps(
    X: np.ndarray,
    k: int,
    seed: int,
    use_minibatch: bool = True,
    max_iter: int = 300,
) -> np.ndarray:
    """
    Select k cluster representatives using k-means.
    
    Runs k-means clustering and selects the point closest to each
    cluster centroid.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix, shape (n_samples, n_features)
    k : int
        Number of points to select (= number of clusters)
    seed : int
        Random seed
    use_minibatch : bool
        Whether to use MiniBatchKMeans for large datasets
    max_iter : int
        Maximum iterations for k-means
        
    Returns
    -------
    np.ndarray
        Selected indices
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    
    if k >= n:
        return np.arange(n)
    
    # Choose k-means variant
    if use_minibatch and n > 10000:
        kmeans = MiniBatchKMeans(
            n_clusters=k,
            random_state=seed,
            max_iter=max_iter,
            batch_size=min(1024, n),
            n_init=3,
        )
    else:
        kmeans = KMeans(
            n_clusters=k,
            random_state=seed,
            max_iter=max_iter,
            n_init=10,
        )
    
    # Fit k-means
    labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_
    
    # Find closest point to each centroid
    selected = np.zeros(k, dtype=int)
    for c in range(k):
        cluster_mask = labels == c
        cluster_indices = np.where(cluster_mask)[0]
        
        if len(cluster_indices) == 0:
            # Empty cluster: pick random point
            selected[c] = np.random.default_rng(seed + c).integers(n)
        else:
            # Find closest to centroid
            cluster_points = X[cluster_indices]
            dists = np.linalg.norm(cluster_points - centroids[c], axis=1)
            closest_in_cluster = np.argmin(dists)
            selected[c] = cluster_indices[closest_in_cluster]
    
    selected = np.unique(selected)
    # Ensure exact cardinality (empty clusters can introduce duplicates)
    if selected.size < k:
        rng = np.random.default_rng(seed + 12345)
        pool = np.setdiff1d(np.arange(n, dtype=int), selected, assume_unique=False)
        if pool.size > 0:
            extra = rng.choice(pool, size=min(pool.size, k - selected.size), replace=False)
            selected = np.concatenate([selected, extra])
    # If still larger (shouldn't happen), trim deterministically
    if selected.size > k:
        selected = selected[:k]
    return selected


def baseline_kmeans_reps_quota(
    X: np.ndarray,
    geo: GeoInfo,
    k: int,
    alpha_geo: float,
    seed: int,
    min_one_per_group: bool = True,
    use_minibatch: bool = True,
) -> np.ndarray:
    """
    K-means representatives with geographic quota constraints.
    
    Runs k-means within each geographic group and selects representatives
    according to the KL-optimal quota allocation.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix, shape (n_samples, n_features)
    geo : GeoInfo
        Geographic group information
    k : int
        Total number of points to select
    alpha_geo : float
        Dirichlet smoothing parameter for KL computation
    seed : int
        Random seed
    min_one_per_group : bool
        Whether to ensure at least one sample per group
    use_minibatch : bool
        Whether to use MiniBatchKMeans for large groups
        
    Returns
    -------
    np.ndarray
        Selected indices
    """
    X = np.asarray(X, dtype=np.float64)
    rng = np.random.default_rng(seed)
    
    # Get quota allocation
    projector = GeographicConstraintProjector(
        geo=geo,
        alpha_geo=alpha_geo,
        min_one_per_group=min_one_per_group,
    )
    target_counts = projector.target_counts(k)
    
    # Select representatives from each group
    selected = []
    for g in range(geo.G):
        count_g = int(target_counts[g])
        if count_g == 0:
            continue
            
        idx_g = geo.group_to_indices[g]
        X_g = X[idx_g]
        
        if count_g >= len(idx_g):
            # Select all from group
            selected.append(idx_g)
        elif count_g == 1:
            # Single point: pick closest to group centroid
            centroid = X_g.mean(axis=0)
            dists = np.linalg.norm(X_g - centroid, axis=1)
            selected.append(np.array([idx_g[np.argmin(dists)]]))
        else:
            # K-means within group
            local_reps = _kmeans_reps_local(
                X_g, count_g, seed=seed + g, use_minibatch=use_minibatch
            )
            selected.append(idx_g[local_reps])
    
    return np.concatenate(selected) if selected else np.array([], dtype=int)


def _kmeans_reps_local(
    X: np.ndarray,
    k: int,
    seed: int,
    use_minibatch: bool = True,
) -> np.ndarray:
    """
    Local k-means representative selection for a subset.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix for the subset
    k : int
        Number of representatives
    seed : int
        Random seed
    use_minibatch : bool
        Whether to use MiniBatchKMeans
        
    Returns
    -------
    np.ndarray
        Local indices (within X) of selected points
    """
    n = X.shape[0]
    
    if k >= n:
        return np.arange(n)
    
    # Choose k-means variant
    if use_minibatch and n > 1000:
        kmeans = MiniBatchKMeans(
            n_clusters=k,
            random_state=seed,
            max_iter=100,
            batch_size=min(256, n),
            n_init=3,
        )
    else:
        kmeans = KMeans(
            n_clusters=k,
            random_state=seed,
            max_iter=100,
            n_init=3,
        )
    
    labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_
    
    selected = []
    for c in range(k):
        cluster_mask = labels == c
        cluster_indices = np.where(cluster_mask)[0]
        
        if len(cluster_indices) == 0:
            continue
            
        cluster_points = X[cluster_indices]
        dists = np.linalg.norm(cluster_points - centroids[c], axis=1)
        closest = cluster_indices[np.argmin(dists)]
        selected.append(closest)
    
    return np.array(selected, dtype=int)


def baseline_kmeans_plusplus(
    X: np.ndarray,
    k: int,
    seed: int,
) -> np.ndarray:
    """
    K-means++ initialization as a coreset selection method.
    
    Uses the k-means++ seeding algorithm which selects points
    with probability proportional to squared distance from
    existing selections.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix, shape (n_samples, n_features)
    k : int
        Number of points to select
    seed : int
        Random seed
        
    Returns
    -------
    np.ndarray
        Selected indices
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    
    if k >= n:
        return np.arange(n)
    
    selected = []
    
    # First point: uniform random
    first = rng.integers(n)
    selected.append(first)
    
    # Subsequent points: proportional to D²
    min_dists_sq = np.full(n, np.inf)
    
    for _ in range(k - 1):
        # Update min distances
        last_point = X[selected[-1]]
        dists_sq = np.sum((X - last_point) ** 2, axis=1)
        min_dists_sq = np.minimum(min_dists_sq, dists_sq)
        
        # Zero out already selected
        min_dists_sq[selected] = 0
        
        # Sample proportional to D²
        probs = min_dists_sq / (min_dists_sq.sum() + 1e-30)
        next_idx = rng.choice(n, p=probs)
        selected.append(next_idx)
    
    return np.array(selected, dtype=int)
