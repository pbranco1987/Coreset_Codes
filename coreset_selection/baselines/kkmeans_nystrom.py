"""
Kernel K-Means Sampling for Nystrom Approximation baseline.

Implements the coreset selection baseline derived from:

    Y. Zhang, J. Liao, et al.
    "Kernel K-Means Sampling for Nystrom Approximation."
    IEEE Trans. Neural Networks and Learning Systems, 2018.
    https://ieeexplore.ieee.org/document/8267102

Key insight (Theorem 1 of the paper): the Frobenius-norm error upper bound
of the Nystrom approximation equals the kernel k-means objective plus a
data-dependent constant.  Therefore, kernel k-means *centroids* (or their
nearest data-point pre-images) are provably optimal landmark points for
Nystrom low-rank approximation.

Pipeline
--------
1. Run kernel k-means with c clusters on the data (or a subsample) to obtain
   c cluster centroids in feature space.
2. Select the medoid of each cluster (the data point closest to the centroid
   in kernel space) as a Nystrom landmark.
3. Form the rank-restricted Nystrom approximation with these c landmarks:
   C = K[:, landmarks],  W = K[landmarks, landmarks],
   R = C . U_{W,ell} . Lambda_{W,ell}^{-1/2}.
4. Truncated SVD of R to rank s yields low-dimensional features B in R^{n x s}.
5. Run linear k-means on B to produce the final k clusters.
6. Return the medoid of each final cluster as the selected coreset indices.

Contains:
- baseline_kkmeans_nystrom      : unconstrained (exact-k) variant
- baseline_kkmeans_nystrom_quota: stratified variant with KL-optimal quotas

References
----------
[1] Y. Zhang, J. Liao, et al. "Kernel K-Means Sampling for Nystrom
    Approximation." IEEE TNNLS, 2018.
[2] S. Wang, A. Gittens, M. W. Mahoney. "Scalable Kernel K-Means Clustering
    with Nystrom Approximation: Relative-Error Bounds." JMLR 20, 2019.
[3] D. Oglic, T. Gartner. "Nystrom Method with Kernel K-means++ Samples as
    Landmarks." ICML, 2017 (conceptual basis).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

from ..geo import GeoInfo, GeographicConstraintProjector

# Import internal helpers and re-export for backward compatibility
from ._kkm_helpers import (
    _rbf_kernel_matrix,
    _kernel_kmeans_lloyd,
    _kkmeans_nystrom_features,
    _select_medoids_from_features,
)


# ===================================================================
# Public API: unconstrained (exact-k) baseline
# ===================================================================

def baseline_kkmeans_nystrom(
    X: np.ndarray,
    k: int,
    seed: int,
    sigma_sq: float,
    c: Optional[int] = None,
    s: Optional[int] = None,
    subsample: Optional[int] = None,
    max_iter: int = 300,
) -> np.ndarray:
    """KKM-Sampling Nystrom coreset selection (unconstrained).

    Runs the full KKM-Sampling Nystrom pipeline of [1]:
      1. Kernel k-means on (subsampled) data -> c landmark medoids
      2. Nystrom features via those landmarks, truncated to rank s
      3. Linear k-means on Nystrom features -> k clusters
      4. Medoid of each cluster -> selected coreset indices

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Feature matrix.
    k : int
        Number of points to select (= number of final clusters).
    seed : int
        Random seed.
    sigma_sq : float
        RBF kernel bandwidth (variance), consistent with the project's
        median-heuristic convention sigma^2 = median(d^2)/2.
    c : int, optional
        Number of Nystrom landmarks / preliminary clusters.
        Default: min(max(10*k, 100), n//2).
    s : int, optional
        Target Nystrom feature dimension.  Default: k.
    subsample : int, optional
        If set, subsample this many points for the preliminary kernel
        k-means (scalability).  Default: min(n, 2000).
    max_iter : int
        Max iterations for the final linear k-means.

    Returns
    -------
    np.ndarray
        Selected indices, shape (k_actual,).
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    rng = np.random.default_rng(seed)

    if k >= n:
        return np.arange(n)

    # Default hyper-parameters
    if c is None:
        c = min(max(10 * k, 100), n // 2)
    c = max(c, k + 1)          # need at least k+1 landmarks
    c = min(c, n)
    if s is None:
        s = max(k, 2)
    if subsample is None:
        subsample = min(n, 2000)

    # Step 1-2: Nystrom features via KKM-sampled landmarks
    B = _kkmeans_nystrom_features(
        X, c=c, s=s, sigma_sq=sigma_sq,
        max_iter=min(max_iter, 30),
        subsample=subsample,
        rng=rng,
    )

    # Step 3: Linear k-means on Nystrom features
    if n > 10000:
        kmeans = MiniBatchKMeans(
            n_clusters=k, random_state=rng.integers(0, 2**31),
            max_iter=max_iter, batch_size=min(1024, n), n_init=3,
        )
    else:
        kmeans = KMeans(
            n_clusters=k, random_state=rng.integers(0, 2**31),
            max_iter=max_iter, n_init=5,
        )
    labels = kmeans.fit_predict(B)

    # Step 4: Select medoid per cluster
    selected = _select_medoids_from_features(B, labels, k, rng)

    # Ensure exact cardinality
    selected = np.unique(selected)
    if selected.size < k:
        pool = np.setdiff1d(np.arange(n, dtype=int), selected)
        if pool.size > 0:
            extra = rng.choice(pool, size=min(pool.size, k - selected.size), replace=False)
            selected = np.concatenate([selected, extra])
    if selected.size > k:
        selected = selected[:k]

    return selected


# ===================================================================
# Public API: quota-constrained (stratified) baseline
# ===================================================================

def baseline_kkmeans_nystrom_quota(
    X: np.ndarray,
    geo: GeoInfo,
    k: int,
    alpha_geo: float,
    seed: int,
    sigma_sq: float,
    min_one_per_group: bool = True,
    c_per_group: Optional[int] = None,
    s: Optional[int] = None,
    max_iter: int = 300,
) -> np.ndarray:
    """KKM-Sampling Nystrom coreset selection with geographic quotas.

    Applies the KKM-Sampling Nystrom pipeline within each geographic group
    according to the KL-optimal quota allocation c*(k), mirroring the
    stratified variant pattern used by the other baselines.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Feature matrix.
    geo : GeoInfo
        Geographic group information.
    k : int
        Total number of points to select.
    alpha_geo : float
        Dirichlet smoothing parameter for KL quota computation.
    seed : int
        Random seed.
    sigma_sq : float
        RBF kernel bandwidth (variance).
    min_one_per_group : bool
        Whether to enforce >= 1 sample per group.
    c_per_group : int, optional
        Number of Nystrom landmarks per group.  Default: auto-scaled.
    s : int, optional
        Target Nystrom feature dimension.  Default: derived from count_g.
    max_iter : int
        Max iterations for linear k-means.

    Returns
    -------
    np.ndarray
        Selected indices satisfying the quota allocation.
    """
    X = np.asarray(X, dtype=np.float64)
    rng = np.random.default_rng(seed)

    # Get KL-optimal quota allocation
    projector = GeographicConstraintProjector(
        geo=geo,
        alpha_geo=alpha_geo,
        min_one_per_group=min_one_per_group,
    )
    target_counts = projector.target_counts(k)

    selected = []
    for g in range(geo.G):
        count_g = int(target_counts[g])
        if count_g == 0:
            continue

        idx_g = geo.group_to_indices[g]
        X_g = X[idx_g]
        n_g = len(idx_g)

        if count_g >= n_g:
            # Select all from this group
            selected.append(idx_g)
            continue

        if count_g == 1:
            # Single point: pick medoid of the whole group in input space
            centroid = X_g.mean(axis=0)
            dists = np.linalg.norm(X_g - centroid, axis=1)
            selected.append(np.array([idx_g[np.argmin(dists)]]))
            continue

        # Determine local hyper-parameters
        c_local = c_per_group if c_per_group is not None else min(
            max(10 * count_g, 50), n_g // 2
        )
        c_local = max(c_local, count_g + 1)
        c_local = min(c_local, n_g)
        s_local = s if s is not None else max(count_g, 2)
        sub_local = min(n_g, 2000)

        # Run the KKM-Nystrom pipeline locally
        local_sel = _kkmeans_nystrom_local(
            X_g, k=count_g, c=c_local, s=s_local,
            sigma_sq=sigma_sq, max_iter=max_iter,
            subsample=sub_local, rng=np.random.default_rng(seed + g),
        )
        selected.append(idx_g[local_sel])

    return np.concatenate(selected) if selected else np.array([], dtype=int)


def _kkmeans_nystrom_local(
    X: np.ndarray,
    k: int,
    c: int,
    s: int,
    sigma_sq: float,
    max_iter: int = 300,
    subsample: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Run the KKM-Nystrom pipeline on a local subset, return local indices."""
    if rng is None:
        rng = np.random.default_rng(0)
    n = X.shape[0]

    if k >= n:
        return np.arange(n)

    B = _kkmeans_nystrom_features(
        X, c=c, s=s, sigma_sq=sigma_sq,
        max_iter=min(max_iter, 30),
        subsample=subsample, rng=rng,
    )

    if n > 1000:
        kmeans = MiniBatchKMeans(
            n_clusters=k, random_state=rng.integers(0, 2**31),
            max_iter=min(max_iter, 100), batch_size=min(256, n), n_init=3,
        )
    else:
        kmeans = KMeans(
            n_clusters=k, random_state=rng.integers(0, 2**31),
            max_iter=min(max_iter, 100), n_init=3,
        )
    labels = kmeans.fit_predict(B)

    sel = _select_medoids_from_features(B, labels, k, rng)
    sel = np.unique(sel)

    # Pad if needed
    if sel.size < k:
        pool = np.setdiff1d(np.arange(n, dtype=int), sel)
        if pool.size > 0:
            extra = rng.choice(pool, size=min(pool.size, k - sel.size), replace=False)
            sel = np.concatenate([sel, extra])
    if sel.size > k:
        sel = sel[:k]

    return sel
