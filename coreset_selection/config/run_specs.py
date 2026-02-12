"""coreset_selection.config.run_specs

Run specifications (R0–R14) aligned with the manuscript "Constrained Nyström
Landmark Selection for Scalable Telecom Analytics".

The manuscript defines a small *run matrix* that varies:
  - representation space used for optimization (raw / PCA / VAE mean)
  - proportionality constraints (population-share, count quota, joint, none)
  - objective set (MMD+Sinkhorn by default; SKL only as an ablation)
  - whether the run is a k-sweep, dimension-sweep, and/or multi-seed

This module provides:
  - RunSpec: lightweight run definition
  - get_run_specs(): registry for R0–R14
  - apply_run_spec(): convert a RunSpec into an ExperimentConfig

Notes
-----
* We keep the historical flag `use_quota_constraints` for backward
  compatibility, but the canonical switch is `GeoConfig.constraint_mode`.
* The manuscript's "R0" quota path is used by quota-mode baselines and quota
  constraint runs; it can be executed independently.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Sequence, Tuple

from .dataclasses import ExperimentConfig


# Manuscript cardinality grid 𝒦
K_GRID: Tuple[int, ...] = (50, 100, 200, 300, 400, 500)

# Representation dimension grid for VAE/PCA sweep (R13/R14)
D_GRID: Tuple[int, ...] = (4, 8, 16, 32, 64, 128)


@dataclass(frozen=True)
class RunSpec:
    """Single experiment/run specification."""

    run_id: str
    description: str

    # Optimization space and objectives
    space: str = "raw"  # raw|pca|vae
    objectives: Tuple[str, ...] = ("mmd", "sinkhorn")

    # Constraints
    constraint_mode: str = "population_share"  # none|population_share|municipality_share_quota|joint
    enforce_exact_k: bool = True

    # Run control
    k: int = 300
    sweep_k: Optional[Tuple[int, ...]] = None
    sweep_dim: Optional[Tuple[int, ...]] = None  # Dimension sweep grid (e.g. D_GRID for R13/R14)
    n_reps: int = 1

    # Per-k replicate overrides.  When a k-sweep is active, this dict
    # provides k-specific replicate counts that override ``n_reps``.
    # Keys are k values; values are the number of replicates for that k.
    # Any k not present in the dict falls back to ``n_reps``.
    n_reps_by_k: Optional[Dict[int, int]] = None

    # Feature flags
    baselines_enabled: bool = False
    eval_enabled: bool = True

    # Cache requirements
    requires_vae: bool = False
    requires_pca: bool = False
    cache_build_mode: str = "lazy"  # lazy|skip

    # Optional dependency (kept for R6 legacy post-hoc; not enforced by tooling)
    depends_on_runs: Tuple[str, ...] = ()

    def get_n_reps_for_k(self, k: int) -> int:
        """Return the number of replicates for a given k value.

        If ``n_reps_by_k`` contains an entry for *k*, that value is
        returned; otherwise the default ``n_reps`` is used.
        """
        if self.n_reps_by_k is not None and k in self.n_reps_by_k:
            return self.n_reps_by_k[k]
        return self.n_reps

    @property
    def max_n_reps(self) -> int:
        """Maximum replicate count across all k values (for cache pre-build)."""
        if self.n_reps_by_k:
            return max(self.n_reps, *self.n_reps_by_k.values())
        return self.n_reps


def get_run_specs() -> Dict[str, RunSpec]:
    """Return the manuscript-aligned run registry."""
    return {
        # R0: quota path and KL_min(k) over the manuscript grid
        "R0": RunSpec(
            run_id="R0",
            description="Quota path c*(k) and KL_min(k) on 𝒦 (count-based, w≡1)",
            space="raw",
            objectives=(),
            constraint_mode="municipality_share_quota",
            enforce_exact_k=True,
            k=300,
            sweep_k=K_GRID,
            n_reps=1,
            baselines_enabled=False,
            eval_enabled=False,
            requires_vae=False,
            requires_pca=False,
            cache_build_mode="skip",
        ),

        # R1: PRIMARY configuration (multi-seed at k=300, single-seed elsewhere)
        "R1": RunSpec(
            run_id="R1",
            description="Primary: raw space, population-share constraint, MMD+Sinkhorn, k-sweep",
            space="raw",
            objectives=("mmd", "sinkhorn"),
            constraint_mode="population_share",
            enforce_exact_k=True,
            k=300,
            sweep_k=K_GRID,
            n_reps=1,
            n_reps_by_k={300: 5},
            baselines_enabled=False,
            eval_enabled=True,
            requires_vae=False,
            requires_pca=False,
        ),

        # R2/R3: single-objective ablations under primary constraint
        "R2": RunSpec(
            run_id="R2",
            description="Ablation: MMD-only under population-share constraint (k=300)",
            space="raw",
            objectives=("mmd",),
            constraint_mode="population_share",
            k=300,
            sweep_k=None,
            n_reps=1,
        ),
        "R3": RunSpec(
            run_id="R3",
            description="Ablation: Sinkhorn-only under population-share constraint (k=300)",
            space="raw",
            objectives=("sinkhorn",),
            constraint_mode="population_share",
            k=300,
            sweep_k=None,
            n_reps=1,
        ),

        # R4: constraint swap (municipality-share quota)
        "R4": RunSpec(
            run_id="R4",
            description="Constraint swap: municipality-share quota (w≡1) with MMD+Sinkhorn (k=300)",
            space="raw",
            objectives=("mmd", "sinkhorn"),
            constraint_mode="municipality_share_quota",
            k=300,
            sweep_k=None,
            n_reps=1,
        ),

        # R5: joint constraints (single-seed; 5-seed reserved for population_share + k=300)
        "R5": RunSpec(
            run_id="R5",
            description="Joint constraints: population-share + municipality-share quota, MMD+Sinkhorn (k=300)",
            space="raw",
            objectives=("mmd", "sinkhorn"),
            constraint_mode="joint",
            k=300,
            sweep_k=None,
            n_reps=1,
        ),

        # R6: no proportionality constraints (exact-k only)
        "R6": RunSpec(
            run_id="R6",
            description="Constraint ablation: exact-k only (no proportionality), MMD+Sinkhorn (k=300)",
            space="raw",
            objectives=("mmd", "sinkhorn"),
            constraint_mode="none",
            k=300,
            sweep_k=None,
            n_reps=1,
        ),

        # R7: SKL ablation (VAE mean space; tri-objective)
        "R7": RunSpec(
            run_id="R7",
            description="SKL ablation: VAE mean space, population-share, MMD+Sinkhorn+SKL (k=300)",
            space="vae",
            objectives=("mmd", "sinkhorn", "skl"),
            constraint_mode="population_share",
            k=300,
            sweep_k=None,
            n_reps=1,
            requires_vae=True,
        ),

        # R8/R9: representation transfer
        "R8": RunSpec(
            run_id="R8",
            description="Representation transfer: PCA space, population-share, MMD+Sinkhorn (k=300)",
            space="pca",
            objectives=("mmd", "sinkhorn"),
            constraint_mode="population_share",
            k=300,
            sweep_k=None,
            n_reps=1,
            requires_pca=True,
        ),
        "R9": RunSpec(
            run_id="R9",
            description="Representation transfer: VAE mean space, population-share, MMD+Sinkhorn (k=300)",
            space="vae",
            objectives=("mmd", "sinkhorn"),
            constraint_mode="population_share",
            k=300,
            sweep_k=None,
            n_reps=1,
            requires_vae=True,
        ),

        # R10: baseline suite (k=300)
        "R10": RunSpec(
            run_id="R10",
            description="Baseline suite at k=300 (with diagnostics + constraints as configured)",
            space="raw",
            objectives=(),
            constraint_mode="population_share",
            k=300,
            sweep_k=None,
            n_reps=1,
            baselines_enabled=True,
            eval_enabled=True,
        ),

        # R11: diagnostics (proxy stability, alignment). Implemented as eval-only hook.
        "R11": RunSpec(
            run_id="R11",
            description="Diagnostics: proxy stability + objective/metric alignment (k=300)",
            space="raw",
            objectives=(),
            constraint_mode="population_share",
            k=300,
            sweep_k=None,
            n_reps=1,
            baselines_enabled=False,
            eval_enabled=True,
        ),

        # R12: effort sweep (still uses solver, but run_scenario controls parameters externally)
        "R12": RunSpec(
            run_id="R12",
            description="Effort sweep: vary NSGA-II effort knobs under population-share (k=300)",
            space="raw",
            objectives=("mmd", "sinkhorn"),
            constraint_mode="population_share",
            k=300,
            sweep_k=None,
            n_reps=1,
        ),

        # R13: VAE latent dimension sweep
        "R13": RunSpec(
            run_id="R13",
            description="VAE latent dim sweep: D in {4,8,16,32,64,128}, MMD+Sinkhorn, pop-share constraint, k=300",
            space="vae",
            objectives=("mmd", "sinkhorn"),             # Bi-objective Pareto front
            constraint_mode="population_share",          # Population-per-state proportionality constraint
            enforce_exact_k=True,
            k=300,
            sweep_k=None,
            sweep_dim=D_GRID,
            n_reps=1,
            requires_vae=True,
            requires_pca=False,
        ),

        # R14: PCA dimension sweep
        "R14": RunSpec(
            run_id="R14",
            description="PCA dim sweep: D in {4,8,16,32,64,128}, MMD+Sinkhorn, pop-share constraint, k=300",
            space="pca",
            objectives=("mmd", "sinkhorn"),             # Bi-objective Pareto front
            constraint_mode="population_share",          # Population-per-state proportionality constraint
            enforce_exact_k=True,
            k=300,
            sweep_k=None,
            sweep_dim=D_GRID,
            n_reps=1,
            requires_vae=False,
            requires_pca=True,
        ),
    }


def apply_run_spec(
    base_cfg: ExperimentConfig,
    spec: RunSpec,
    rep_id: int,
    *,
    dim_override: Optional[int] = None,
) -> ExperimentConfig:
    """Apply a RunSpec onto a base ExperimentConfig.

    Parameters
    ----------
    dim_override : int, optional
        When running a dimension sweep (R13/R14), override the VAE latent_dim
        or PCA n_components with this value.  ``None`` (default) keeps the
        base config dimensions unchanged.
    """
    # Cache path
    cache_path = os.path.join(base_cfg.files.cache_dir, f"rep{rep_id:02d}", "assets.npz")

    # Geo / constraints
    geo_cfg = replace(
        base_cfg.geo,
        constraint_mode=str(spec.constraint_mode),
        # Keep the historical flag coherent
        use_quota_constraints=(str(spec.constraint_mode) in {"municipality_share_quota", "joint"}),
    )

    # Solver
    solver_cfg = replace(
        base_cfg.solver,
        k=int(spec.k),
        objectives=tuple(spec.objectives),
        enabled=(len(spec.objectives) > 0),
        enforce_exact_k=bool(spec.enforce_exact_k),
    )

    # Evaluation and baselines
    eval_cfg = replace(base_cfg.eval, enabled=bool(spec.eval_enabled))
    baselines_cfg = replace(base_cfg.baselines, enabled=bool(spec.baselines_enabled))

    # Space
    cfg = replace(
        base_cfg,
        run_id=str(spec.run_id),
        rep_id=int(rep_id),
        space=str(spec.space),
        files=replace(base_cfg.files, cache_path=cache_path),
        geo=geo_cfg,
        solver=solver_cfg,
        eval=eval_cfg,
        baselines=baselines_cfg,
    )

    # Dimension override for R13/R14 dimension sweeps
    if dim_override is not None:
        if spec.space == "vae":
            cfg = replace(cfg, vae=replace(cfg.vae, latent_dim=int(dim_override)))
        elif spec.space == "pca":
            cfg = replace(cfg, pca=replace(cfg.pca, n_components=int(dim_override)))

    return cfg
