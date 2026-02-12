"""
Unified objective computation for coreset selection.

Contains:
- SpaceObjectiveComputer: Computes all objectives (SKL, MMD², Sinkhorn) efficiently
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .mmd import RFFMMD
from .sinkhorn import AnchorSinkhorn
from .skl import symmetric_kl_diag_gaussians
from ..utils.debug_timing import timer


@dataclass
class SpaceObjectiveComputer:
    """
    Unified objective computer for a representation space.
    
    Computes all three objectives (SKL, MMD², Sinkhorn) for a given subset.
    Pre-computes statistics for the full dataset to enable efficient
    subset evaluation.
    
    Attributes
    ----------
    X : np.ndarray
        Representation matrix (N, d)
    logvars : Optional[np.ndarray]
        Log-variance from VAE (N, d), or None for non-VAE spaces
    rff : RFFMMD
        RFF-based MMD estimator
    sink : AnchorSinkhorn
        Anchor-based Sinkhorn estimator
    mu_full : np.ndarray
        Mean of X over full dataset
    var_full : np.ndarray
        Variance of X over full dataset (including point variance for VAE)
    """
    X: np.ndarray
    logvars: Optional[np.ndarray]
    rff: RFFMMD
    sink: AnchorSinkhorn
    mu_full: np.ndarray
    var_full: np.ndarray

    @staticmethod
    def build(
        X: np.ndarray,
        logvars: Optional[np.ndarray],
        rff: RFFMMD,
        sink: AnchorSinkhorn,
    ) -> "SpaceObjectiveComputer":
        """
        Build a SpaceObjectiveComputer.
        
        Parameters
        ----------
        X : np.ndarray
            Representation matrix (N, d)
        logvars : Optional[np.ndarray]
            Log-variance from VAE encoder (N, d), or None
        rff : RFFMMD
            Pre-built RFF-based MMD estimator
        sink : AnchorSinkhorn
            Pre-built anchor-based Sinkhorn estimator
            
        Returns
        -------
        SpaceObjectiveComputer
            Initialized computer
        """
        X = np.asarray(X, dtype=np.float32)
        mu_full = X.mean(axis=0).astype(np.float64)

        if logvars is not None:
            lv = np.asarray(logvars, dtype=np.float32)
            # Total variance = E[Var(X|z)] + Var(E[X|z])
            # For VAE: Var(X|z) = exp(logvar), E[X|z] = X
            var_point = np.exp(lv.astype(np.float64))
            var_full = var_point.mean(axis=0) + X.astype(np.float64).var(axis=0, ddof=0)
        else:
            var_full = X.astype(np.float64).var(axis=0, ddof=0)
        
        var_full = np.maximum(var_full, 1e-12)

        return SpaceObjectiveComputer(
            X=X,
            logvars=logvars,
            rff=rff,
            sink=sink,
            mu_full=mu_full,
            var_full=var_full,
        )

    def compute_all(self, idx: np.ndarray) -> Dict[str, float]:
        """
        Compute all objectives for a subset.
        
        Parameters
        ----------
        idx : np.ndarray
            Indices of the subset
            
        Returns
        -------
        Dict[str, float]
            Dictionary with keys "skl", "mmd2", "sinkhorn"
        """
        idx = np.asarray(idx, dtype=int)
        
        if idx.size == 0:
            return {"skl": 1e18, "mmd2": 1e18, "sinkhorn": 1e18}

        # SKL
        skl = self.compute_skl(idx)

        # MMD² (RFF)
        mmd2 = self.compute_mmd2(idx)

        # Sinkhorn divergence
        sinkhorn = self.compute_sinkhorn(idx)

        return {"skl": skl, "mmd2": mmd2, "sinkhorn": sinkhorn}

    def compute_skl(self, idx: np.ndarray) -> float:
        """
        Compute symmetric KL divergence for a subset.
        
        Parameters
        ----------
        idx : np.ndarray
            Indices of the subset
            
        Returns
        -------
        float
            SKL divergence
        """
        idx = np.asarray(idx, dtype=int)
        
        if idx.size == 0:
            return 1e18
        
        # Subset statistics
        mu_sub = self.X[idx].mean(axis=0).astype(np.float64)
        
        if self.logvars is not None:
            var_point = np.exp(self.logvars[idx].astype(np.float64))
            var_sub = var_point.mean(axis=0) + self.X[idx].astype(np.float64).var(axis=0, ddof=0)
        else:
            var_sub = self.X[idx].astype(np.float64).var(axis=0, ddof=0)
        
        var_sub = np.maximum(var_sub, 1e-9)
        
        skl = symmetric_kl_diag_gaussians(self.mu_full, self.var_full, mu_sub, var_sub)
        
        return float(skl) if np.isfinite(skl) else 1e18

    def compute_mmd2(self, idx: np.ndarray) -> float:
        """
        Compute MMD² for a subset using RFF approximation.
        
        Parameters
        ----------
        idx : np.ndarray
            Indices of the subset
            
        Returns
        -------
        float
            MMD² estimate
        """
        idx = np.asarray(idx, dtype=int)
        
        if idx.size == 0:
            return 1e18
        
        mmd2 = self.rff.mmd2_subset(idx)
        
        return float(mmd2) if np.isfinite(mmd2) else 1e18

    def compute_sinkhorn(self, idx: np.ndarray) -> float:
        """
        Compute Sinkhorn divergence for a subset.
        
        Parameters
        ----------
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
        
        try:
            sd = self.sink.sinkhorn_divergence_subset(self.X, idx)
        except Exception:
            sd = 1e18
        
        return float(sd) if np.isfinite(sd) else 1e18

    def compute_single(self, idx: np.ndarray, objective: str) -> float:
        """
        Compute a single objective for a subset.
        
        Parameters
        ----------
        idx : np.ndarray
            Indices of the subset
        objective : str
            Objective name ("skl", "mmd", "mmd2", or "sinkhorn")
            
        Returns
        -------
        float
            Objective value
        """
        objective = objective.lower()
        
        if objective == "skl":
            return self.compute_skl(idx)
        elif objective in ("mmd", "mmd2"):
            return self.compute_mmd2(idx)
        elif objective == "sinkhorn":
            return self.compute_sinkhorn(idx)
        else:
            raise ValueError(f"Unknown objective: {objective}")


def build_space_objective_computer(
    X: np.ndarray,
    logvars: Optional[np.ndarray],
    mmd_cfg,  # MMDConfig
    sinkhorn_cfg,  # SinkhornConfig
    seed: int = 0,
) -> SpaceObjectiveComputer:
    """
    Convenience function to build a SpaceObjectiveComputer from configs.
    
    Parameters
    ----------
    X : np.ndarray
        Representation matrix
    logvars : Optional[np.ndarray]
        VAE log-variances (or None)
    mmd_cfg : MMDConfig
        MMD configuration
    sinkhorn_cfg : SinkhornConfig
        Sinkhorn configuration
    seed : int
        Random seed
        
    Returns
    -------
    SpaceObjectiveComputer
        Initialized computer
    """
    from ..utils.math import median_sq_dist
    
    with timer.section("build_space_objective_computer", X_shape=X.shape, has_logvars=logvars is not None):
        # Compute bandwidth using median heuristic
        # Median heuristic for RBF k(x,y)=exp(-||x-y||^2/(2σ^2)) uses σ^2 = median(||x-y||^2)/2.
        with timer.section("compute_median_sq_dist"):
            sigma_sq = median_sq_dist(X, sample_size=2048, seed=seed) / 2.0
            sigma_sq = sigma_sq * mmd_cfg.bandwidth_mult
        timer.checkpoint("Bandwidth computed", sigma_sq=sigma_sq)
        
        # Build RFF-MMD estimator
        with timer.section("build_RFFMMD", rff_dim=mmd_cfg.rff_dim):
            rff = RFFMMD.build(
                X=X,
                rff_dim=mmd_cfg.rff_dim,
                sigma_sq=sigma_sq,
                seed=seed,
            )
        
        # Build anchor Sinkhorn estimator
        with timer.section("build_AnchorSinkhorn", n_anchors=sinkhorn_cfg.n_anchors):
            sink = AnchorSinkhorn.build(
                X=X,
                cfg=sinkhorn_cfg,
                seed=seed + 1,
            )
        
        # Build unified computer
        with timer.section("build_SpaceObjectiveComputer"):
            computer = SpaceObjectiveComputer.build(
                X=X,
                logvars=logvars,
                rff=rff,
                sink=sink,
            )
        
        timer.checkpoint("Objective computer ready")
        return computer
