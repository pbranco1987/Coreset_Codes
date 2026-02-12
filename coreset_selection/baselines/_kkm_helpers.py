"""Internal helpers for the KKM-Sampling Nystrom baseline.

Extracted from kkmeans_nystrom.py. Contains the RBF kernel matrix helper,
kernel k-means Lloyd's algorithm, Nystrom feature computation, and medoid
selection from feature-space clusters.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.linalg import svd, eigh


# ---------------------------------------------------------------------------
# Kernel helpers (RBF, consistent with the project's sigma^2 convention)
# ---------------------------------------------------------------------------

def _rbf_kernel_matrix(X: np.ndarray, Y: np.ndarray, sigma_sq: float) -> np.ndarray:
    """Gaussian RBF kernel: K(x,y) = exp(-||x-y||^2 / (2*sigma^2)).

    Parameters
    ----------
    X : (n, d)
    Y : (m, d)
    sigma_sq : float
        Kernel bandwidth (variance).  Matches the project convention where
        sigma^2 = median(||x_i - x_j||^2) / 2.

    Returns
    -------
    K : (n, m)
    """
    X_sq = np.sum(X ** 2, axis=1, keepdims=True)
    Y_sq = np.sum(Y ** 2, axis=1, keepdims=True)
    D2 = X_sq + Y_sq.T - 2.0 * (X @ Y.T)
    D2 = np.maximum(D2, 0.0)
    return np.exp(-D2 / (2.0 * sigma_sq))


# ---------------------------------------------------------------------------
# Core: Kernel k-means in feature space (Lloyd's algorithm via kernel trick)
# ---------------------------------------------------------------------------

def _kernel_kmeans_lloyd(
    K: np.ndarray,
    n_clusters: int,
    max_iter: int = 30,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Lloyd's algorithm in kernel space.

    At each iteration, distances to centroids are computed via the kernel trick:
      d(x_j, mu_i) = K_{jj} - (2/|J_i|) sum_{l in J_i} K_{jl}
                     + (1/|J_i|^2) sum_{l,m in J_i} K_{lm}

    Parameters
    ----------
    K : (n, n)  Kernel matrix (symmetric PSD).
    n_clusters : int
    max_iter : int
    rng : Generator

    Returns
    -------
    labels : (n,) cluster assignments
    """
    if rng is None:
        rng = np.random.default_rng(0)

    n = K.shape[0]
    diag_K = np.diag(K).copy()

    # Random initialisation ensuring every cluster has >= 1 member
    labels = rng.integers(0, n_clusters, size=n)
    for c in range(n_clusters):
        if not np.any(labels == c):
            labels[rng.integers(0, n)] = c

    for _ in range(max_iter):
        dists = np.full((n, n_clusters), np.inf)
        for c in range(n_clusters):
            idx_c = np.where(labels == c)[0]
            nc = len(idx_c)
            if nc == 0:
                continue
            mean_k = K[:, idx_c].sum(axis=1) / nc
            intra = K[np.ix_(idx_c, idx_c)].sum() / (nc * nc)
            dists[:, c] = diag_K - 2.0 * mean_k + intra

        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        # Repair empty clusters
        for c in range(n_clusters):
            if not np.any(labels == c):
                labels[rng.integers(0, n)] = c

    return labels


# ---------------------------------------------------------------------------
# Core: KKM-Sampling Nystrom pipeline -> returns low-dim features B
# ---------------------------------------------------------------------------

def _kkmeans_nystrom_features(
    X: np.ndarray,
    c: int,
    s: int,
    sigma_sq: float,
    max_iter: int = 30,
    subsample: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Compute Nystrom features using KKM-sampled landmarks (Sec III of [1]).

    Steps
    -----
    (a) Run kernel k-means with *c* clusters on the data (or a subsample)
        to find optimal landmarks.
    (b) Select medoids (nearest data points to centroids) as landmarks.
    (c) Form rank-restricted Nystrom approximation  R = C U_{W,ell} Lambda^{-1/2}.
    (d) Truncated SVD of R to rank *s* -> features B = U_s Sigma_s.

    Parameters
    ----------
    X : (n, d)
    c : int   - number of Nystrom landmarks (= kernel k-means clusters)
    s : int   - target feature dimension after SVD truncation
    sigma_sq : float  - RBF bandwidth (variance)
    max_iter : int
    subsample : int or None - subsample size for the preliminary kernel
                k-means (for scalability)
    rng : Generator

    Returns
    -------
    B : (n, s_actual) low-dimensional Nystrom features
    """
    if rng is None:
        rng = np.random.default_rng(0)

    n = X.shape[0]

    # ---- (a) Preliminary kernel k-means to select c landmarks ----
    if subsample is not None and subsample < n:
        sub_idx = rng.choice(n, size=subsample, replace=False)
        X_sub = X[sub_idx]
    else:
        sub_idx = np.arange(n)
        X_sub = X

    K_sub = _rbf_kernel_matrix(X_sub, X_sub, sigma_sq)
    labels_pre = _kernel_kmeans_lloyd(K_sub, n_clusters=c, max_iter=max_iter, rng=rng)

    # ---- (b) Select medoids as landmarks ----
    landmark_local = []
    for ci in range(c):
        idx_ci = np.where(labels_pre == ci)[0]
        if len(idx_ci) == 0:
            landmark_local.append(rng.integers(0, len(sub_idx)))
            continue
        K_clust = K_sub[np.ix_(idx_ci, idx_ci)]
        nc = len(idx_ci)
        mean_col = K_clust.sum(axis=1) / nc
        intra_mean = K_clust.sum() / (nc * nc)
        d_to_centroid = np.diag(K_clust) - 2.0 * mean_col + intra_mean
        best_local = np.argmin(d_to_centroid)
        landmark_local.append(idx_ci[best_local])

    landmark_global = sub_idx[np.array(landmark_local)]
    landmark_global = np.unique(landmark_global)
    c_actual = len(landmark_global)

    # ---- (c) Nystrom approximation ----
    X_land = X[landmark_global]
    C = _rbf_kernel_matrix(X, X_land, sigma_sq)        # (n, c_actual)
    W = _rbf_kernel_matrix(X_land, X_land, sigma_sq)   # (c_actual, c_actual)

    # Regularised pseudo-inverse via truncated eigendecomposition
    ell = max(s + 1, int(np.ceil(c_actual / 2)))
    ell = min(ell, c_actual)

    eigvals, eigvecs = eigh(W)
    # eigh returns ascending order - take top ell
    idx_top = np.argsort(eigvals)[::-1][:ell]
    eigvals_top = eigvals[idx_top]
    eigvecs_top = eigvecs[:, idx_top]
    # Discard numerically zero eigenvalues
    pos_mask = eigvals_top > 1e-10
    eigvals_top = eigvals_top[pos_mask]
    eigvecs_top = eigvecs_top[:, pos_mask]

    # R = C . U_{W,ell} . Lambda^{-1/2}   in R^{n x ell}
    R = C @ eigvecs_top @ np.diag(eigvals_top ** (-0.5))

    # ---- (d) Truncated SVD of R -> features B ----
    s_actual = min(s, R.shape[1])
    U_R, Sig_R, _ = svd(R, full_matrices=False)
    B = U_R[:, :s_actual] * Sig_R[:s_actual]

    return B


# ---------------------------------------------------------------------------
# Helper: select medoids from cluster labels + features
# ---------------------------------------------------------------------------

def _select_medoids_from_features(
    B: np.ndarray,
    labels: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pick the point closest to the centroid in feature space per cluster.

    Returns
    -------
    selected : (k_actual,) global indices of medoids (may be < k if
               clusters are empty; caller handles padding).
    """
    centroids_dict = {}
    for c in range(k):
        idx_c = np.where(labels == c)[0]
        if len(idx_c) > 0:
            centroids_dict[c] = B[idx_c].mean(axis=0)

    selected = []
    for c in range(k):
        idx_c = np.where(labels == c)[0]
        if len(idx_c) == 0:
            continue
        dists = np.linalg.norm(B[idx_c] - centroids_dict[c], axis=1)
        best = idx_c[np.argmin(dists)]
        selected.append(best)

    return np.array(selected, dtype=int)
