"""
Nystrom log-determinant diversity objective.

Contains:
- NystromLogDet: Log-determinant of the Nystrom kernel sub-matrix for diversity
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist


@dataclass
class NystromLogDet:
    """Encourages landmark diversity via log-determinant of the Nystrom kernel sub-matrix.

    Minimising ``-log|K_{S,S} + lambda*I|`` pushes the selected subset toward
    well-conditioned, diverse landmark configurations in kernel space.

    Cost per evaluation: O(k^2 d + k^3) -- comparable to MMD / Sinkhorn.

    Attributes
    ----------
    X : np.ndarray
        Full dataset, shape (N, d)
    sigma_sq : float
        RBF bandwidth sigma^2
    reg : float
        Tikhonov regularisation lambda (default 1e-6)
    """
    X: np.ndarray          # (N, d) full dataset
    sigma_sq: float        # RBF bandwidth sigma^2
    reg: float             # Tikhonov regularisation lambda (default 1e-6)

    # -- factory -----------------------------------------------------------
    @staticmethod
    def build(
        X: np.ndarray,
        sigma_sq: float | None = None,
        reg: float = 1e-6,
    ) -> "NystromLogDet":
        """Build a NystromLogDet instance.

        Parameters
        ----------
        X : np.ndarray
            Full dataset, shape (N, d)
        sigma_sq : float or None
            RBF bandwidth. If None, use the median heuristic.
        reg : float
            Tikhonov regularisation parameter (default 1e-6)

        Returns
        -------
        NystromLogDet
            Initialized estimator
        """
        if sigma_sq is None:
            # Median heuristic on a subsample for efficiency
            rng = np.random.RandomState(42)
            n = min(2000, X.shape[0])
            idx = rng.choice(X.shape[0], n, replace=False)
            dists = cdist(X[idx], X[idx], 'sqeuclidean')
            sigma_sq = float(np.median(dists[np.triu_indices(n, k=1)]))
            if sigma_sq < 1e-12:
                sigma_sq = 1.0
        return NystromLogDet(X=X, sigma_sq=sigma_sq, reg=reg)

    # -- evaluation --------------------------------------------------------
    def logdet_subset(self, idx: np.ndarray) -> float:
        """Return -log|K_{S,S} + reg*I|.

        Parameters
        ----------
        idx : (k,) integer array -- selected indices.

        Returns
        -------
        neg_logdet : float
            Negative log-determinant (lower = more diverse).
        """
        X_S = self.X[idx]                              # (k, d)
        # RBF kernel matrix  K_{ij} = exp(-||x_i - x_j||^2 / (2 sigma^2))
        sq_dists = cdist(X_S, X_S, 'sqeuclidean')     # (k, k)
        K = np.exp(-sq_dists / (2.0 * self.sigma_sq))  # (k, k)
        K += self.reg * np.eye(len(idx))               # regularise

        # Cholesky -> log-det = 2 * sum(log(diag(L)))
        try:
            L = np.linalg.cholesky(K)
            logdet = 2.0 * np.sum(np.log(np.diag(L)))
        except np.linalg.LinAlgError:
            # Fallback: eigenvalues
            eigvals = np.linalg.eigvalsh(K)
            eigvals = np.maximum(eigvals, 1e-30)
            logdet = float(np.sum(np.log(eigvals)))

        return -logdet  # minimise -> maximise diversity
