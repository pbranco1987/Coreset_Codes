"""
Sinkhorn divergence via anchor approximation.

Contains:
- AnchorSinkhorn: Anchor-based Sinkhorn divergence estimator
- sinkhorn2_safe: Numerically stable Sinkhorn solver
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..utils.math import median_sq_dist
from ..utils.debug_timing import timer


@dataclass
class AnchorSinkhorn:
    """
    Anchor-based Sinkhorn divergence estimator.
    
    Uses a fixed set of anchor points to approximate Sinkhorn divergence
    between subsets and the full dataset efficiently.
    
    S(P, Q) = OT_ε(P, Q) - 0.5 * OT_ε(P, P) - 0.5 * OT_ε(Q, Q)
    
    Attributes
    ----------
    anchors : np.ndarray
        Anchor points, shape (n_anchors, d)
    weights : np.ndarray
        Anchor weights ã (k-means cluster fractions), shape (n_anchors,)
    cost_anchor_full : np.ndarray
        Optional cached cost matrix from anchors to full dataset, shape (n_anchors, N)
    cost_aa : np.ndarray
        Cost matrix among anchors, shape (n_anchors, n_anchors)
    ot_pp : float
        Pre-computed OT_ε( P̃_A , P̃_A ) in original cost units
    reg : float
        Entropic regularization parameter
    max_iter : int
        Maximum Sinkhorn iterations
    stop_thr : float
        Convergence threshold
    cost_scale : float
        Cost scaling factor
    """
    anchors: np.ndarray
    weights: np.ndarray
    cost_anchor_full: np.ndarray
    cost_aa: np.ndarray
    ot_pp: float
    reg: float
    max_iter: int
    stop_thr: float
    cost_scale: float

    @staticmethod
    def build(
        X: np.ndarray,
        cfg,  # SinkhornConfig
        seed: int = 0,
    ) -> "AnchorSinkhorn":
        """
        Build an AnchorSinkhorn estimator.
        
        Per manuscript Section 5.6:
        - Anchors computed by k-means with k-means++ seeding
        - ε = η · median(||r_i - r_j||²) with η from config
        - 100 Sinkhorn iterations (log-stabilized)
        
        Parameters
        ----------
        X : np.ndarray
            Full dataset, shape (N, d)
        cfg : SinkhornConfig
            Sinkhorn configuration
        seed : int
            Random seed
            
        Returns
        -------
        AnchorSinkhorn
            Initialized estimator
        """
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        
        n_anchors = cfg.n_anchors
        eta = cfg.eta  # η = 0.05 per manuscript
        max_iter = cfg.max_iter
        stop_thr = cfg.stop_thr
        
        # Select anchors (k-means with k-means++ seeding per manuscript)
        labels = None
        with timer.section("select_anchors", method=cfg.anchor_method, n_anchors=n_anchors):
            if cfg.anchor_method == "kmeans":
                if n_anchors >= n:
                    anchors = X.copy()
                    labels = np.arange(n, dtype=int)
                else:
                    # Lazy import: only import sklearn when actually needed
                    from sklearn.cluster import MiniBatchKMeans

                    km = MiniBatchKMeans(
                        n_clusters=int(n_anchors),
                        random_state=int(seed),
                        batch_size=min(1024, n),
                        n_init=3,
                        init="k-means++",
                    )
                    km.fit(X)
                    anchors = km.cluster_centers_
                    labels = km.labels_
            elif cfg.anchor_method == "random":
                anchors = _select_anchors_random(X, n_anchors, seed)
            elif cfg.anchor_method == "farthest":
                anchors = _select_anchors_farthest(X, n_anchors, seed)
            else:
                anchors = _select_anchors_kmeans(X, n_anchors, seed)
        
        timer.checkpoint("Anchors selected", n_anchors_actual=anchors.shape[0])
        
        # Compute median squared distance for ε (manuscript Eq. from Section 5.6)
        # ε = η · median(||r_i - r_j||²)
        with timer.section("compute_cost_scale"):
            if cfg.cost_scale > 0:
                median_sq = cfg.cost_scale
            else:
                median_sq = median_sq_dist(X, sample_size=2048, seed=seed)
        
        # Entropic regularization: ε = η · median(||r_i - r_j||²)
        reg = eta * median_sq
        
        # Cost scaling: we divide costs by median_sq for numerical stability
        # and then use reg_normalized = eta (since ε/median_sq = η)
        cost_scale = median_sq
        
        # Compute cost matrices (squared Euclidean / scale)
        with timer.section("compute_cost_matrices"):
            cost_aa = _pairwise_sq_distances(anchors, anchors) / cost_scale
            cost_anchor_full = _pairwise_sq_distances(anchors, X) / cost_scale
        
        timer.checkpoint("Cost matrices computed", 
                         cost_aa_shape=cost_aa.shape, 
                         cost_anchor_full_shape=cost_anchor_full.shape)
        
        # Weights: ã_t = fraction of points assigned to anchor t (manuscript Section 5.6)
        if cfg.anchor_method == "kmeans" and labels is not None:
            counts = np.bincount(np.asarray(labels, dtype=int), minlength=anchors.shape[0]).astype(np.float64)
            counts = np.maximum(counts, 0.0)
            if counts.sum() <= 0:
                weights_anchors = np.ones(anchors.shape[0], dtype=np.float64) / anchors.shape[0]
            else:
                weights_anchors = counts / counts.sum()
        else:
            weights_anchors = np.ones(anchors.shape[0], dtype=np.float64) / anchors.shape[0]

        # Pre-compute OT(P̃_A, P̃_A) once (manuscript anchored surrogate)
        ot_pp_scaled = sinkhorn2_logstab(
            weights_anchors, weights_anchors, cost_aa,
            reg=eta, numItermax=max_iter, stopThr=stop_thr
        )
        ot_pp = float(ot_pp_scaled * cost_scale)
        
        return AnchorSinkhorn(
            anchors=anchors.astype(np.float32),
            weights=weights_anchors,
            cost_anchor_full=cost_anchor_full.astype(np.float32),
            cost_aa=cost_aa.astype(np.float32),
            ot_pp=ot_pp,
            reg=eta,  # normalized regularization
            max_iter=max_iter,
            stop_thr=stop_thr,
            cost_scale=cost_scale,
        )

    def sinkhorn_divergence_subset(
        self,
        X: np.ndarray,
        idx: np.ndarray,
    ) -> float:
        """
        Compute Sinkhorn divergence between subset and full dataset.
        
        S(Q, P) = OT_ε(Q, P) - 0.5 * OT_ε(Q, Q) - 0.5 * OT_ε(P, P)
        
        Uses anchor approximation for efficiency.
        
        Parameters
        ----------
        X : np.ndarray
            Full dataset (needed for subset extraction)
        idx : np.ndarray
            Indices of the subset
            
        Returns
        -------
        float
            Sinkhorn divergence estimate
        """
        idx = np.asarray(idx, dtype=int)
        
        if idx.size == 0:
            return 1e18
        
        X_sub = X[idx]
        k = len(idx)
        
        # Cost matrices for subset.
        # Use cached anchor-to-full costs for O(Ak) extraction.
        if getattr(self, "cost_anchor_full", None) is not None and self.cost_anchor_full.shape[1] >= idx.max() + 1:
            cost_aq = np.asarray(self.cost_anchor_full[:, idx], dtype=np.float64)
        else:
            cost_aq = _pairwise_sq_distances(self.anchors, X_sub) / self.cost_scale
        cost_qq = _pairwise_sq_distances(X_sub, X_sub) / self.cost_scale
        
        # Uniform weights
        weights_q = np.ones(k) / k
        
        # OT(P̃_A, Q_s)
        ot_pq_scaled = sinkhorn2_logstab(
            self.weights, weights_q, cost_aq,
            reg=self.reg, numItermax=self.max_iter, stopThr=self.stop_thr
        )
        ot_pq = float(ot_pq_scaled * self.cost_scale)
        
        # OT(Q, Q)
        ot_qq_scaled = sinkhorn2_logstab(
            weights_q, weights_q, cost_qq,
            reg=self.reg, numItermax=self.max_iter, stopThr=self.stop_thr
        )
        ot_qq = float(ot_qq_scaled * self.cost_scale)
        
        # Sinkhorn divergence
        # Debiased Sinkhorn divergence in original cost units
        sd = ot_pq - 0.5 * ot_qq - 0.5 * self.ot_pp
        
        return float(max(0.0, sd))


def sinkhorn2_safe(
    a: np.ndarray,
    b: np.ndarray,
    M: np.ndarray,
    *,
    reg: float,
    numItermax: int = 1000,
    stopThr: float = 1e-6,
    warn: bool = False,
) -> float:
    """
    Compute Sinkhorn distance (squared) with numerical stability.
    
    Solves the entropy-regularized optimal transport problem:
    OT_ε(a, b) = min_π <π, M> + ε * KL(π || a ⊗ b)
    
    Parameters
    ----------
    a : np.ndarray
        Source distribution, shape (n,)
    b : np.ndarray
        Target distribution, shape (m,)
    M : np.ndarray
        Cost matrix, shape (n, m)
    reg : float
        Entropic regularization parameter
    numItermax : int
        Maximum iterations
    stopThr : float
        Convergence threshold
    warn : bool
        Whether to warn on non-convergence
        
    Returns
    -------
    float
        Sinkhorn distance (transport cost)
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    
    n, m = M.shape
    
    # Ensure valid distributions
    a = np.maximum(a, 1e-30)
    b = np.maximum(b, 1e-30)
    a = a / a.sum()
    b = b / b.sum()
    
    # Gibbs kernel
    K = np.exp(-M / reg)
    K = np.maximum(K, 1e-30)
    
    # Initialize dual variables
    u = np.ones(n)
    v = np.ones(m)
    
    # Sinkhorn iterations
    for i in range(numItermax):
        u_prev = u.copy()
        
        # Update v
        Ktu = K.T @ u
        v = b / np.maximum(Ktu, 1e-30)
        
        # Update u
        Kv = K @ v
        u = a / np.maximum(Kv, 1e-30)
        
        # Check convergence
        err = np.max(np.abs(u - u_prev))
        if err < stopThr:
            break
    
    # Compute transport cost
    # P = diag(u) @ K @ diag(v)
    # cost = <P, M> = sum(P * M)
    P = u[:, None] * K * v[None, :]
    cost = np.sum(P * M)
    
    return float(cost)


def sinkhorn2_logstab(
    a: np.ndarray,
    b: np.ndarray,
    M: np.ndarray,
    *,
    reg: float,
    numItermax: int = 100,
    stopThr: float = 1e-6,
) -> float:
    """Log-stabilized Sinkhorn distance (transport cost).

    This implementation updates scaling factors in the log domain:
        log_u = log_a - logsumexp(logK + log_v)
        log_v = log_b - logsumexp(logK^T + log_u)
    where logK = -M/reg.

    Returns the transport cost <P, M> in the same units as M.
    """
    from scipy.special import logsumexp

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)

    a = np.maximum(a, 1e-30)
    b = np.maximum(b, 1e-30)
    a = a / a.sum()
    b = b / b.sum()

    log_a = np.log(a)
    log_b = np.log(b)

    # logK = -M/reg; keep finite
    logK = -M / float(reg)

    n, m = M.shape
    log_u = np.zeros(n, dtype=np.float64)
    log_v = np.zeros(m, dtype=np.float64)

    for _ in range(int(numItermax)):
        log_u_prev = log_u.copy()
        log_u = log_a - logsumexp(logK + log_v[None, :], axis=1)
        log_v = log_b - logsumexp(logK.T + log_u[None, :], axis=1)

        err = float(np.max(np.abs(log_u - log_u_prev)))
        if err < stopThr:
            break

    # Compute transport plan in log domain, then cost
    logP = log_u[:, None] + logK + log_v[None, :]
    # Safe exponentiation; values are typically negative
    P = np.exp(np.clip(logP, -80.0, 80.0))
    cost = float(np.sum(P * M))
    return cost


def _assign_kmeans_labels(X: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Assign each x in X to the nearest anchor (centroid) by squared distance."""
    X = np.asarray(X, dtype=np.float64)
    A = np.asarray(anchors, dtype=np.float64)
    # Compute argmin_j ||x - a_j||^2 efficiently
    x2 = np.sum(X * X, axis=1, keepdims=True)
    a2 = np.sum(A * A, axis=1)[None, :]
    d2 = x2 + a2 - 2.0 * (X @ A.T)
    return np.argmin(d2, axis=1)


def _pairwise_sq_distances(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Compute pairwise squared Euclidean distances."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    
    X_sqnorm = np.sum(X ** 2, axis=1, keepdims=True)
    Y_sqnorm = np.sum(Y ** 2, axis=1)
    
    sq_dist = X_sqnorm + Y_sqnorm - 2 * X @ Y.T
    sq_dist = np.maximum(sq_dist, 0.0)
    
    return sq_dist


def _select_anchors_kmeans(
    X: np.ndarray,
    n_anchors: int,
    seed: int,
) -> np.ndarray:
    """Select anchors using k-means centroids."""
    from sklearn.cluster import MiniBatchKMeans
    
    n = len(X)
    if n_anchors >= n:
        return X.copy()
    
    kmeans = MiniBatchKMeans(
        n_clusters=n_anchors,
        random_state=seed,
        batch_size=min(1024, n),
        n_init=3,
    )
    kmeans.fit(X)
    
    return kmeans.cluster_centers_


def _select_anchors_random(
    X: np.ndarray,
    n_anchors: int,
    seed: int,
) -> np.ndarray:
    """Select anchors uniformly at random."""
    rng = np.random.default_rng(seed)
    n = len(X)
    
    if n_anchors >= n:
        return X.copy()
    
    idx = rng.choice(n, size=n_anchors, replace=False)
    return X[idx].copy()


def _select_anchors_farthest(
    X: np.ndarray,
    n_anchors: int,
    seed: int,
) -> np.ndarray:
    """Select anchors using farthest-first traversal."""
    rng = np.random.default_rng(seed)
    n = len(X)
    
    if n_anchors >= n:
        return X.copy()
    
    # Start from random point
    selected = [rng.integers(n)]
    min_dists = np.full(n, np.inf)
    
    for _ in range(n_anchors - 1):
        # Update min distances
        last_point = X[selected[-1]]
        dists = np.sum((X - last_point) ** 2, axis=1)
        min_dists = np.minimum(min_dists, dists)
        min_dists[selected] = -np.inf
        
        # Select farthest
        next_idx = np.argmax(min_dists)
        selected.append(next_idx)
    
    return X[selected].copy()
