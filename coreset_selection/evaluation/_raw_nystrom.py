"""
Nystrom-specific utility functions for raw-space evaluation.

Extracted from ``evaluation.raw_space`` for modularity.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ._raw_kernels import _rbf_kernel, _safe_cholesky_solve


def _nystrom_components(
    X_E: np.ndarray,
    X_S: np.ndarray,
    sigma_sq: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute Nystrom blocks C and W and the stabilization ridge lambda_nys.

    Returns (C, W, lambda_nys).
    """
    C = _rbf_kernel(X_E, X_S, sigma_sq)        # (|E|, k)
    W = _rbf_kernel(X_S, X_S, sigma_sq)        # (k, k)
    k = max(1, X_S.shape[0])
    lambda_nys = 1e-6 * float(np.trace(W)) / float(k)
    return C, W, float(lambda_nys)


def _nystrom_approx_gram(C: np.ndarray, W: np.ndarray, lambda_nys: float) -> np.ndarray:
    """
    Compute K_hat = C (W + lambda I)^{-1} C^T.
    """
    k = W.shape[0]
    W_reg = W + float(lambda_nys) * np.eye(k)
    # Solve W_reg^{-1} C^T
    A, _ = _safe_cholesky_solve(W_reg, C.T)   # (k, |E|)
    K_hat = C @ A                              # (|E|, |E|)
    return K_hat


def _nystrom_features(
    C: np.ndarray,
    W: np.ndarray,
    lambda_nys: float,
) -> np.ndarray:
    """
    Compute Nystrom feature matrix Phi = C (W + lambda I)^{-T/2} so that Phi Phi^T = K_hat.
    """
    k = W.shape[0]
    W_reg = W + float(lambda_nys) * np.eye(k)

    try:
        L = np.linalg.cholesky(W_reg)  # W_reg = L L^T
        # Phi^T = L^{-1} C^T  => Phi = (L^{-1} C^T)^T
        Phi_T = np.linalg.solve(L, C.T)  # (k, |E|)
        Phi = Phi_T.T                   # (|E|, k)
        return Phi
    except np.linalg.LinAlgError:
        # Fallback via eigendecomposition: W_reg^{-1/2} = V diag(w^{-1/2}) V^T
        w, V = np.linalg.eigh(W_reg)
        w = np.maximum(w, 1e-12)
        Winv_sqrt = (V * (1.0 / np.sqrt(w))) @ V.T
        Phi = C @ Winv_sqrt
        return Phi


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _kfold_indices(n: int, n_splits: int, rng: np.random.Generator):
    """Simple shuffled K-fold split yielding (train_idx, val_idx) pairs."""
    idx = np.arange(n, dtype=int)
    rng.shuffle(idx)
    folds = np.array_split(idx, n_splits)
    for i in range(n_splits):
        val = folds[i]
        train = np.concatenate([folds[j] for j in range(n_splits) if j != i])
        yield train, val


def _select_lambda_ridge(
    Phi: np.ndarray,
    y: np.ndarray,
    lambdas: np.ndarray,
    n_folds: int,
    seed: int,
) -> float:
    """
    Select ridge lambda by K-fold CV on Nystrom features.

    Important: folds are fixed across the lambda grid (standard CV practice),
    and the same grid is reused across methods and targets.
    """
    Phi = np.asarray(Phi, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    n = y.shape[0]
    if n == 0:
        return float(lambdas[0])

    # Fixed folds
    rng = np.random.default_rng(seed)
    idx = np.arange(n, dtype=int)
    rng.shuffle(idx)
    folds = np.array_split(idx, int(n_folds))
    splits = []
    for i in range(len(folds)):
        va = folds[i]
        tr = np.concatenate([folds[j] for j in range(len(folds)) if j != i])
        splits.append((tr, va))

    best_lam = float(lambdas[0])
    best_score = float("inf")

    for lam in lambdas:
        fold_rmses = []
        for tr, va in splits:
            Phi_tr = Phi[tr]
            Phi_va = Phi[va]
            y_tr = y[tr]
            y_va = y[va]

            A = Phi_tr.T @ Phi_tr + float(lam) * np.eye(Phi_tr.shape[1])
            b = Phi_tr.T @ y_tr
            w, _ = _safe_cholesky_solve(A, b)
            y_hat = Phi_va @ w
            fold_rmses.append(_rmse(y_va, y_hat))

        score = float(np.mean(fold_rmses)) if fold_rmses else float("inf")
        if score < best_score:
            best_score = score
            best_lam = float(lam)

    return best_lam
