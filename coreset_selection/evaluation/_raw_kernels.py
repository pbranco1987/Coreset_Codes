"""
Standalone kernel / math utility functions for raw-space evaluation.

Extracted from ``evaluation.raw_space`` for modularity.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _rbf_kernel(X1: np.ndarray, X2: np.ndarray, sigma_sq: float) -> np.ndarray:
    """Compute RBF kernel matrix exp(-||x-x'||^2/(2*sigma^2))."""
    X1 = np.asarray(X1, dtype=np.float64)
    X2 = np.asarray(X2, dtype=np.float64)
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    n1 = np.sum(X1 * X1, axis=1, keepdims=True)
    n2 = np.sum(X2 * X2, axis=1, keepdims=True).T
    D2 = n1 + n2 - 2.0 * (X1 @ X2.T)
    D2 = np.maximum(D2, 0.0)
    return np.exp(-D2 / (2.0 * float(sigma_sq)))


def _median_sq_dist(X: np.ndarray, seed: int, max_pairs: int = 20000) -> float:
    """
    Median heuristic for squared distance.

    Samples up to max_pairs random pairs and returns median ||x_i - x_j||^2.
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if n <= 1:
        return 1.0

    rng = np.random.default_rng(seed)

    # If dataset is small, compute all pairwise distances
    # n(n-1)/2 <= max_pairs -> n ~ 200 for max_pairs=20000, so usually we sample.
    n_pairs = min(max_pairs, n * (n - 1) // 2)
    if n_pairs <= 0:
        return 1.0

    i = rng.integers(0, n, size=n_pairs, endpoint=False)
    j = rng.integers(0, n, size=n_pairs, endpoint=False)
    # Avoid i == j by resampling a small number (cheap)
    mask_eq = i == j
    if np.any(mask_eq):
        j[mask_eq] = (j[mask_eq] + 1) % n

    diffs = X[i] - X[j]
    d2 = np.sum(diffs * diffs, axis=1)
    med = float(np.median(d2))
    return med if np.isfinite(med) and med > 0 else 1.0


def _center_gram(K: np.ndarray) -> np.ndarray:
    """Center a Gram matrix with H = I - 11^T/n: Kc = HKH."""
    K = np.asarray(K, dtype=np.float64)
    row_mean = K.mean(axis=1, keepdims=True)
    col_mean = K.mean(axis=0, keepdims=True)
    total_mean = float(K.mean())
    return K - row_mean - col_mean + total_mean


def _safe_cholesky_solve(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Solve A X = B for SPD A.

    Returns (X, used_cholesky). If Cholesky fails, falls back to eigendecomposition.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    try:
        L = np.linalg.cholesky(A)
        # Solve L Y = B
        Y = np.linalg.solve(L, B)
        # Solve L^T X = Y
        X = np.linalg.solve(L.T, Y)
        return X, True
    except np.linalg.LinAlgError:
        # Robust fallback: symmetric eigendecomposition
        w, V = np.linalg.eigh(A)
        w = np.maximum(w, 1e-12)
        Ainv = (V / w) @ V.T
        return Ainv @ B, False
