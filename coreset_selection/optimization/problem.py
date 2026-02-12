"""
Multi-objective optimization problem definition for coreset selection.

Contains:
- CoresetMOOProblem: pymoo Problem subclass for NSGA-II optimization
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from pymoo.core.problem import Problem

from ..geo.info import GeoInfo
from ..geo.kl import kl_pi_hat_from_counts
from ..objectives.mmd import RFFMMD
from ..objectives.sinkhorn import AnchorSinkhorn
from ..objectives.skl import symmetric_kl_diag_gaussians


class CoresetMOOProblem(Problem):
    """
    Multi-objective coreset selection problem for pymoo.
    
    Objectives (configurable, default bi-objective):
    - MMD²: Maximum Mean Discrepancy (RFF approximation)
    - Sinkhorn: Sinkhorn divergence (anchor approximation)
    - SKL: Symmetric KL divergence (ablation only, R7)
    - Geo-KL: Geographic KL divergence (optional, not used by default)
    
    Decision Variables:
    - Binary mask indicating selected points
    
    Attributes
    ----------
    k : int
        Target coreset size
    X_repr : np.ndarray
        Representation matrix (N, d)
    geo : GeoInfo
        Geographic group information
    geo_cfg : GeoConfig
        Geographic configuration
    rff : RFFMMD
        RFF-based MMD estimator
    sink : AnchorSinkhorn
        Anchor-based Sinkhorn estimator
    logvars : Optional[np.ndarray]
        VAE log-variances (N, d)
    objectives : Tuple[str, ...]
        Objective names to optimize
    include_geo : bool
        Whether to include geographic KL as objective
    name : str
        Problem name (for logging)
    """
    
    def __init__(
        self,
        *,
        k: int,
        X_repr: np.ndarray,
        geo: GeoInfo,
        geo_cfg,  # GeoConfig
        rff: RFFMMD,
        sink: AnchorSinkhorn,
        logvars: Optional[np.ndarray] = None,
        objectives: Tuple[str, ...] = ("mmd", "sinkhorn"),
        include_geo: bool = False,
        name: str = "space",
    ):
        """
        Initialize the multi-objective problem.
        
        Parameters
        ----------
        k : int
            Target coreset size
        X_repr : np.ndarray
            Representation matrix (N, d)
        geo : GeoInfo
            Geographic group information
        geo_cfg : GeoConfig
            Geographic configuration
        rff : RFFMMD
            Pre-built RFF-MMD estimator
        sink : AnchorSinkhorn
            Pre-built anchor Sinkhorn estimator
        logvars : Optional[np.ndarray]
            VAE log-variances for SKL computation
        objectives : Tuple[str, ...]
            Which objectives to include ("skl", "mmd", "sinkhorn")
        include_geo : bool
            Whether to add geo_kl as an objective
        name : str
            Problem name for identification
        """
        self.k = k
        self.X_repr = np.asarray(X_repr, dtype=np.float32)
        self.geo = geo
        self.geo_cfg = geo_cfg
        self.rff = rff
        self.sink = sink
        self.logvars = logvars
        self.objectives = tuple(obj.lower() for obj in objectives)
        self.include_geo = include_geo
        self.name = name
        
        # Compute number of objectives
        n_obj = len(self.objectives)
        if include_geo:
            n_obj += 1
        
        # Pre-compute full dataset statistics for SKL
        # Per manuscript Section 5.5.3, variance clamping to [exp(-10), exp(2)]
        VAR_MIN = np.exp(-10)
        VAR_MAX = np.exp(2)
        
        self._mu_full = self.X_repr.mean(axis=0).astype(np.float64)
        if self.logvars is not None:
            # Clamp log-variances before exp
            logvars_clamped = np.clip(self.logvars.astype(np.float64), -10, 2)
            var_point = np.exp(logvars_clamped)
            # v_N = (1/N) Σ Σ_i + Var({μ_i}_{i=1}^N)
            self._var_full = var_point.mean(axis=0) + self.X_repr.astype(np.float64).var(axis=0, ddof=0)
        else:
            self._var_full = self.X_repr.astype(np.float64).var(axis=0, ddof=0)
        # Clamp to [exp(-10), exp(2)]
        self._var_full = np.clip(self._var_full, VAR_MIN, VAR_MAX)
        
        # Initialize pymoo Problem
        n_var = len(X_repr)
        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            n_ieq_constr=0,
            xl=0,
            xu=1,
            vtype=bool,
        )

    def _evaluate(self, Xmask, out, *args, **kwargs):
        """
        Evaluate objective functions for a population of solutions.
        
        Parameters
        ----------
        Xmask : np.ndarray
            Population of binary masks, shape (pop_size, n_var)
        out : dict
            Output dictionary for objective values
        """
        pop_size = Xmask.shape[0]
        n_obj = self.n_obj
        
        F = np.zeros((pop_size, n_obj), dtype=np.float64)
        
        for i in range(pop_size):
            mask = Xmask[i].astype(bool)
            idx = np.where(mask)[0]
            
            obj_idx = 0
            
            # Compute requested objectives
            for obj_name in self.objectives:
                if obj_name == "skl":
                    F[i, obj_idx] = self._compute_skl(idx)
                elif obj_name in ("mmd", "mmd2"):
                    F[i, obj_idx] = self._compute_mmd2(idx)
                elif obj_name == "sinkhorn":
                    F[i, obj_idx] = self._compute_sinkhorn(idx)
                else:
                    raise ValueError(f"Unknown objective: {obj_name}")
                obj_idx += 1
            
            # Geographic KL (if included)
            if self.include_geo:
                F[i, obj_idx] = self._compute_geo_kl(idx)
        
        out["F"] = F

    def _compute_skl(self, idx: np.ndarray) -> float:
        """
        Compute symmetric KL divergence for subset.
        
        Per manuscript Section 5.5.3:
        - Variance clamping to [exp(-10), exp(2)] for numerical stability
        - Uses moment-matched diagonal Gaussian summaries
        """
        if len(idx) == 0:
            return 1e18
        
        # Variance clamping bounds per manuscript
        VAR_MIN = np.exp(-10)
        VAR_MAX = np.exp(2)
        
        X_sub = self.X_repr[idx]
        mu_sub = X_sub.mean(axis=0).astype(np.float64)
        
        if self.logvars is not None:
            # VAE mode: include posterior variances
            # Clamp log-variances before exp for numerical stability
            logvars_clamped = np.clip(self.logvars[idx].astype(np.float64), -10, 2)
            var_point = np.exp(logvars_clamped)
            # v_S = (1/k) Σ Σ_i + Var({μ_i}_{i∈S})
            var_sub = var_point.mean(axis=0) + X_sub.astype(np.float64).var(axis=0, ddof=0)
        else:
            var_sub = X_sub.astype(np.float64).var(axis=0, ddof=0)
        
        # Clamp variances to [exp(-10), exp(2)] per manuscript
        var_sub = np.clip(var_sub, VAR_MIN, VAR_MAX)
        
        # Also ensure full population variance uses same clamping
        var_full_clamped = np.clip(self._var_full, VAR_MIN, VAR_MAX)
        
        skl = symmetric_kl_diag_gaussians(self._mu_full, var_full_clamped, mu_sub, var_sub)
        
        return float(skl) if np.isfinite(skl) else 1e18

    def _compute_mmd2(self, idx: np.ndarray) -> float:
        """Compute MMD² using RFF approximation."""
        if len(idx) == 0:
            return 1e18
        
        mmd2 = self.rff.mmd2_subset(idx)
        return float(mmd2) if np.isfinite(mmd2) else 1e18

    def _compute_sinkhorn(self, idx: np.ndarray) -> float:
        """Compute Sinkhorn divergence using anchor approximation."""
        if len(idx) == 0:
            return 1e18
        
        try:
            sd = self.sink.sinkhorn_divergence_subset(self.X_repr, idx)
        except Exception:
            sd = 1e18
        
        return float(sd) if np.isfinite(sd) else 1e18

    def _compute_geo_kl(self, idx: np.ndarray) -> float:
        """Compute geographic KL divergence."""
        if len(idx) == 0:
            return 1e18
        
        # Count selections per group
        counts = np.zeros(self.geo.G, dtype=int)
        for i in idx:
            g = self.geo.group_ids[i]
            counts[g] += 1
        
        k_actual = len(idx)
        
        kl = kl_pi_hat_from_counts(
            self.geo.pi, counts, k_actual, self.geo_cfg.alpha_geo
        )
        
        return float(kl) if np.isfinite(kl) else 1e18

    def get_objective_names(self) -> Tuple[str, ...]:
        """Get names of all objectives in order."""
        names = list(self.objectives)
        if self.include_geo:
            names.append("geo_kl")
        return tuple(names)

    def evaluate_single(self, mask: np.ndarray) -> Dict[str, float]:
        """
        Evaluate a single solution and return named objectives.
        
        Parameters
        ----------
        mask : np.ndarray
            Binary selection mask
            
        Returns
        -------
        Dict[str, float]
            Mapping from objective name to value
        """
        idx = np.where(mask)[0]
        
        result = {}
        
        for obj_name in self.objectives:
            if obj_name == "skl":
                result["skl"] = self._compute_skl(idx)
            elif obj_name in ("mmd", "mmd2"):
                result["mmd"] = self._compute_mmd2(idx)
            elif obj_name == "sinkhorn":
                result["sinkhorn"] = self._compute_sinkhorn(idx)
        
        if self.include_geo:
            result["geo_kl"] = self._compute_geo_kl(idx)
        
        return result
