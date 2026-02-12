"""
Mathematical utility functions.
"""

import numpy as np


def median_sq_dist(X: np.ndarray, sample_size: int = 2048, seed: int = 0) -> float:
    """
    Median pairwise squared distance on a random subsample (for bandwidth scaling).
    
    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n, d)
    sample_size : int
        Number of samples to use for estimation
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    float
        Median squared distance, clipped to be at least 1e-6
    """
    rng = np.random.default_rng(seed)
    n = int(X.shape[0])
    if n <= 1:
        return 1.0
    m = min(n, int(sample_size))
    idx = rng.choice(n, size=m, replace=False)
    Y = X[idx].astype(np.float64, copy=False)
    G = Y @ Y.T
    sq = np.maximum(np.diag(G)[:, None] + np.diag(G)[None, :] - 2.0 * G, 0.0)
    triu = sq[np.triu_indices(m, k=1)]
    med = float(np.median(triu)) if triu.size else 1.0
    return max(med, 1e-6)
