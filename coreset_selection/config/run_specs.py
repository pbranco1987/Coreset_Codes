"""coreset_selection.config.run_specs

Run specifications (R0–R15) aligned with the manuscript "Constrained Nyström
Landmark Selection for Scalable Telecom Analytics".

The manuscript defines a small *run matrix* that varies:
  - representation space used for optimization (VAE default / raw / PCA ablation)
  - proportionality constraints (population-share, count quota, joint, none)
  - objective set (MMD+Sinkhorn by default; SKL only as an ablation)
  - whether the run is a k-sweep, dimension-sweep, and/or multi-seed

**Optimization vs evaluation spaces:**
  - The *optimization space* (``space`` field) determines which representation
    is used for NSGA-II (VAE latent embeddings by default).
  - *All evaluations* are always conducted in the standardised raw attribute
    space, regardless of the optimization space.  This ensures fair and
    comparable metrics across all experiment configurations.
  - Raw-space (R8) and PCA-space (R9) configurations exist as representation-
    transfer ablation experiments to compare against the VAE default.

This module provides:
  - RunSpec: lightweight run definition
  - get_run_specs(): registry for R0–R15, G1–G48, and v2 aliases
  - apply_run_spec(): convert a RunSpec into an ExperimentConfig

Notes
-----
* K is user-defined (via ``-k`` or the ``K_GRID`` sweep); the user must
  always specify a value — there is no hardcoded default.
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
K_GRID: Tuple[int, ...] = (30, 50, 100, 200, 300, 400, 500)

# Representation dimension grid for VAE/PCA sweep (R13/R14)
D_GRID: Tuple[int, ...] = (8, 16, 32, 64)


@dataclass(frozen=True)
class RunSpec:
    """Single experiment/run specification."""

    run_id: str
    description: str

    # Optimization space and objectives
    space: str = "vae"  # vae|raw|pca  (VAE default; raw/PCA are ablation-only)
    objectives: Tuple[str, ...] = ("mmd", "sinkhorn")

    # Constraints
    constraint_mode: str = "population_share"  # none|population_share|municipality_share_quota|joint
    enforce_exact_k: bool = True

    # Run control
    k: int = None  # Must be provided by the user via -k
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

    # Per-run objective parameter overrides (raw-space R8 needs larger RFF/anchors)
    mmd_rff_dim: Optional[int] = None        # Override MMDConfig.rff_dim
    sinkhorn_n_anchors: Optional[int] = None  # Override SinkhornConfig.n_anchors

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
    specs = {
        # R0: quota path and KL_min(k) over the manuscript grid
        "R0": RunSpec(
            run_id="R0",
            description="Quota path c*(k) and KL_min(k) on 𝒦 (count-based, w≡1)",
            space="raw",
            objectives=(),
            constraint_mode="municipality_share_quota",
            enforce_exact_k=True,
            sweep_k=K_GRID,
            n_reps=1,
            baselines_enabled=False,
            eval_enabled=False,
            requires_vae=False,
            requires_pca=False,
            cache_build_mode="skip",
        ),

        # R1: PRIMARY configuration — VAE space, multi-seed (5 replicates at user k)
        "R1": RunSpec(
            run_id="R1",
            description="Primary: VAE space, population-share constraint, MMD+Sinkhorn, multi-seed",
            space="vae",
            objectives=("mmd", "sinkhorn"),
            constraint_mode="population_share",
            enforce_exact_k=True,
            sweep_k=K_GRID,
            n_reps=5,
            baselines_enabled=False,
            eval_enabled=True,
            requires_vae=True,
            requires_pca=False,
        ),

        # R2/R3: single-objective ablations under primary constraint (VAE space)
        "R2": RunSpec(
            run_id="R2",
            description="Ablation: MMD-only, VAE space, population-share constraint",
            space="vae",
            objectives=("mmd",),
            constraint_mode="population_share",
            sweep_k=None,
            n_reps=1,
            requires_vae=True,
        ),
        "R3": RunSpec(
            run_id="R3",
            description="Ablation: Sinkhorn-only, VAE space, population-share constraint",
            space="vae",
            objectives=("sinkhorn",),
            constraint_mode="population_share",
            sweep_k=None,
            n_reps=1,
            requires_vae=True,
        ),

        # R4: constraint swap — municipality-share quota (VAE space)
        "R4": RunSpec(
            run_id="R4",
            description="Constraint swap: municipality-share quota (w≡1), VAE space, MMD+Sinkhorn",
            space="vae",
            objectives=("mmd", "sinkhorn"),
            constraint_mode="municipality_share_quota",
            sweep_k=None,
            n_reps=1,
            requires_vae=True,
        ),

        # R5: joint constraints (VAE space)
        "R5": RunSpec(
            run_id="R5",
            description="Joint constraints: population-share + municipality-share quota, VAE space, MMD+Sinkhorn",
            space="vae",
            objectives=("mmd", "sinkhorn"),
            constraint_mode="joint",
            sweep_k=None,
            n_reps=1,
            requires_vae=True,
        ),

        # R6: no proportionality constraints — exact-k only (VAE space)
        "R6": RunSpec(
            run_id="R6",
            description="Constraint ablation: exact-k only (no proportionality), VAE space, MMD+Sinkhorn",
            space="vae",
            objectives=("mmd", "sinkhorn"),
            constraint_mode="none",
            sweep_k=None,
            n_reps=1,
            requires_vae=True,
        ),

        # R7: SKL ablation (VAE mean space; tri-objective)
        "R7": RunSpec(
            run_id="R7",
            description="SKL ablation: VAE mean space, population-share, MMD+Sinkhorn+SKL",
            space="vae",
            objectives=("mmd", "sinkhorn", "skl"),
            constraint_mode="population_share",
            sweep_k=None,
            n_reps=1,
            requires_vae=True,
        ),

        # R8/R9: representation transfer ablations (raw and PCA k-sweeps)
        # These compare optimization in alternative spaces against the VAE
        # default (R1).  All evaluation is still in standardised raw space.
        "R8": RunSpec(
            run_id="R8",
            description="Repr transfer ablation: raw space, population-share, MMD+Sinkhorn, k-sweep",
            space="raw",
            objectives=("mmd", "sinkhorn"),
            constraint_mode="population_share",
            enforce_exact_k=True,
            sweep_k=K_GRID,
            n_reps=1,
            requires_vae=False,
            requires_pca=False,
            mmd_rff_dim=20_000,
            sinkhorn_n_anchors=400,
        ),
        "R9": RunSpec(
            run_id="R9",
            description="Repr transfer ablation: PCA space, population-share, MMD+Sinkhorn, k-sweep",
            space="pca",
            objectives=("mmd", "sinkhorn"),
            constraint_mode="population_share",
            enforce_exact_k=True,
            sweep_k=K_GRID,
            n_reps=1,
            requires_vae=False,
            requires_pca=True,
        ),

        # R10: baseline suite (VAE space) — 5 replicates matching R1's seeds
        "R10": RunSpec(
            run_id="R10",
            description="Baseline suite: VAE space (with diagnostics + constraints as configured)",
            space="vae",
            objectives=(),
            constraint_mode="population_share",
            sweep_k=None,
            n_reps=5,
            baselines_enabled=True,
            eval_enabled=True,
            requires_vae=True,
        ),

        # R11: diagnostics — proxy stability, alignment (VAE space)
        "R11": RunSpec(
            run_id="R11",
            description="Diagnostics: proxy stability + objective/metric alignment, VAE space",
            space="vae",
            objectives=(),
            constraint_mode="population_share",
            sweep_k=None,
            n_reps=1,
            baselines_enabled=False,
            eval_enabled=True,
            requires_vae=True,
        ),

        # R12: effort sweep — vary NSGA-II effort knobs (VAE space)
        "R12": RunSpec(
            run_id="R12",
            description="Effort sweep: vary NSGA-II effort knobs, VAE space, population-share",
            space="vae",
            objectives=("mmd", "sinkhorn"),
            constraint_mode="population_share",
            sweep_k=None,
            n_reps=1,
            requires_vae=True,
        ),

        # R13: VAE latent dimension sweep
        "R13": RunSpec(
            run_id="R13",
            description="VAE latent dim sweep: D in {4,8,16,32,64,128}, MMD+Sinkhorn, pop-share constraint",
            space="vae",
            objectives=("mmd", "sinkhorn"),             # Bi-objective Pareto front
            constraint_mode="population_share",          # Population-per-state proportionality constraint
            enforce_exact_k=True,
            sweep_k=None,
            sweep_dim=D_GRID,
            n_reps=1,
            requires_vae=True,
            requires_pca=False,
        ),

        # R14: PCA dimension sweep
        "R14": RunSpec(
            run_id="R14",
            description="PCA dim sweep: D in {4,8,16,32,64,128}, MMD+Sinkhorn, pop-share constraint",
            space="pca",
            objectives=("mmd", "sinkhorn"),             # Bi-objective Pareto front
            constraint_mode="population_share",          # Population-per-state proportionality constraint
            enforce_exact_k=True,
            sweep_k=None,
            sweep_dim=D_GRID,
            n_reps=1,
            requires_vae=False,
            requires_pca=True,
        ),

        # R15: Tri-objective with Nystrom log-det diversity
        "R15": RunSpec(
            run_id="R15",
            description="Tri-objective: MMD+Sinkhorn+NystromLogDet, VAE, pop-share",
            space="vae",
            objectives=("mmd", "sinkhorn", "nystrom_logdet"),
            constraint_mode="population_share",
            enforce_exact_k=True,
            sweep_k=(100,),
            n_reps=1,
            requires_vae=True,
        ),

        # ================================================================
        # G-series: Geographic constraint sweep (v2 experiments)
        #
        # Goal: Fair apples-to-apples comparison of NSGA-II vs baselines
        # across all 4 constraint modes, using hard deterministic quotas
        # for population-share so NSGA-II and baselines are directly
        # comparable.
        #
        # All G-series experiments are bi-objective (MMD+Sinkhorn),
        # raw space (no representation transfer confound), k=100,
        # 5 replicates for statistical power.
        # ================================================================

        # --- G1–G4: NSGA-II with each constraint mode (raw, bi-obj, 5 reps) ---

        # G1: NSGA-II, raw, population-share HARD quota (matches baselines)
        "G1": RunSpec(
            run_id="G1",
            description="Geo sweep: NSGA-II, raw, population-share hard quota, bi-obj, 5 reps",
            space="raw",
            objectives=("mmd", "sinkhorn"),
            constraint_mode="population_share_quota",
            enforce_exact_k=True,
            sweep_k=(100,),
            n_reps=5,
            requires_vae=False,
            requires_pca=False,
            mmd_rff_dim=20_000,
            sinkhorn_n_anchors=400,
        ),

        # G2: NSGA-II, raw, municipality-share quota
        "G2": RunSpec(
            run_id="G2",
            description="Geo sweep: NSGA-II, raw, municipality-share quota, bi-obj, 5 reps",
            space="raw",
            objectives=("mmd", "sinkhorn"),
            constraint_mode="municipality_share_quota",
            enforce_exact_k=True,
            sweep_k=(100,),
            n_reps=5,
            requires_vae=False,
            requires_pca=False,
            mmd_rff_dim=20_000,
            sinkhorn_n_anchors=400,
        ),

        # G3: NSGA-II, raw, joint (population-share + municipality-share)
        "G3": RunSpec(
            run_id="G3",
            description="Geo sweep: NSGA-II, raw, joint constraints, bi-obj, 5 reps",
            space="raw",
            objectives=("mmd", "sinkhorn"),
            constraint_mode="joint",
            enforce_exact_k=True,
            sweep_k=(100,),
            n_reps=5,
            requires_vae=False,
            requires_pca=False,
            mmd_rff_dim=20_000,
            sinkhorn_n_anchors=400,
        ),

        # G4: NSGA-II, raw, unconstrained (exact-k only)
        "G4": RunSpec(
            run_id="G4",
            description="Geo sweep: NSGA-II, raw, no geo constraint, bi-obj, 5 reps",
            space="raw",
            objectives=("mmd", "sinkhorn"),
            constraint_mode="none",
            enforce_exact_k=True,
            sweep_k=(100,),
            n_reps=5,
            requires_vae=False,
            requires_pca=False,
            mmd_rff_dim=20_000,
            sinkhorn_n_anchors=400,
        ),

        # --- G5–G8: Baselines with each constraint mode (raw, 5 reps) ---
        # These use the baseline runner (run_pop_baselines) with --regime,
        # not the NSGA-II solver. The RunSpecs serve as documentation and
        # for cache/eval coordination.

        # G5: Baselines, raw, pop-share hard quota
        "G5": RunSpec(
            run_id="G5",
            description="Geo sweep: baselines, raw, population-share hard quota, 5 reps",
            space="raw",
            objectives=(),
            constraint_mode="population_share_quota",
            enforce_exact_k=True,
            sweep_k=(100,),
            n_reps=5,
            baselines_enabled=True,
            eval_enabled=True,
            requires_vae=False,
            requires_pca=False,
        ),

        # G6: Baselines, raw, municipality-share quota
        "G6": RunSpec(
            run_id="G6",
            description="Geo sweep: baselines, raw, municipality-share quota, 5 reps",
            space="raw",
            objectives=(),
            constraint_mode="municipality_share_quota",
            enforce_exact_k=True,
            sweep_k=(100,),
            n_reps=5,
            baselines_enabled=True,
            eval_enabled=True,
            requires_vae=False,
            requires_pca=False,
        ),

        # G7: Baselines, raw, joint constraints
        "G7": RunSpec(
            run_id="G7",
            description="Geo sweep: baselines, raw, joint constraints, 5 reps",
            space="raw",
            objectives=(),
            constraint_mode="joint",
            enforce_exact_k=True,
            sweep_k=(100,),
            n_reps=5,
            baselines_enabled=True,
            eval_enabled=True,
            requires_vae=False,
            requires_pca=False,
        ),

        # G8: Baselines, raw, no geo constraint (exact-k only)
        "G8": RunSpec(
            run_id="G8",
            description="Geo sweep: baselines, raw, no geo constraint, 5 reps",
            space="raw",
            objectives=(),
            constraint_mode="none",
            enforce_exact_k=True,
            sweep_k=(100,),
            n_reps=5,
            baselines_enabled=True,
            eval_enabled=True,
            requires_vae=False,
            requires_pca=False,
        ),

        # --- G9–G12: NSGA-II with each constraint mode (VAE, bi-obj, 5 reps) ---

        # G9: NSGA-II, VAE, population-share HARD quota (matches baselines)
        "G9": RunSpec(
            run_id="G9",
            description="Geo sweep: NSGA-II, VAE, population-share hard quota, bi-obj, 5 reps",
            space="vae",
            objectives=("mmd", "sinkhorn"),
            constraint_mode="population_share_quota",
            enforce_exact_k=True,
            sweep_k=(100,),
            n_reps=5,
            requires_vae=True,
            requires_pca=False,
        ),

        # G10: NSGA-II, VAE, municipality-share quota
        "G10": RunSpec(
            run_id="G10",
            description="Geo sweep: NSGA-II, VAE, municipality-share quota, bi-obj, 5 reps",
            space="vae",
            objectives=("mmd", "sinkhorn"),
            constraint_mode="municipality_share_quota",
            enforce_exact_k=True,
            sweep_k=(100,),
            n_reps=5,
            requires_vae=True,
            requires_pca=False,
        ),

        # G11: NSGA-II, VAE, joint (population-share + municipality-share)
        "G11": RunSpec(
            run_id="G11",
            description="Geo sweep: NSGA-II, VAE, joint constraints, bi-obj, 5 reps",
            space="vae",
            objectives=("mmd", "sinkhorn"),
            constraint_mode="joint",
            enforce_exact_k=True,
            sweep_k=(100,),
            n_reps=5,
            requires_vae=True,
            requires_pca=False,
        ),

        # G12: NSGA-II, VAE, unconstrained (exact-k only)
        "G12": RunSpec(
            run_id="G12",
            description="Geo sweep: NSGA-II, VAE, no geo constraint, bi-obj, 5 reps",
            space="vae",
            objectives=("mmd", "sinkhorn"),
            constraint_mode="none",
            enforce_exact_k=True,
            sweep_k=(100,),
            n_reps=5,
            requires_vae=True,
            requires_pca=False,
        ),

        # --- G13–G16: Baselines with each constraint mode (VAE, 5 reps) ---

        # G13: Baselines, VAE, pop-share hard quota
        "G13": RunSpec(
            run_id="G13",
            description="Geo sweep: baselines, VAE, population-share hard quota, 5 reps",
            space="vae",
            objectives=(),
            constraint_mode="population_share_quota",
            enforce_exact_k=True,
            sweep_k=(100,),
            n_reps=5,
            baselines_enabled=True,
            eval_enabled=True,
            requires_vae=True,
            requires_pca=False,
        ),

        # G14: Baselines, VAE, municipality-share quota
        "G14": RunSpec(
            run_id="G14",
            description="Geo sweep: baselines, VAE, municipality-share quota, 5 reps",
            space="vae",
            objectives=(),
            constraint_mode="municipality_share_quota",
            enforce_exact_k=True,
            sweep_k=(100,),
            n_reps=5,
            baselines_enabled=True,
            eval_enabled=True,
            requires_vae=True,
            requires_pca=False,
        ),

        # G15: Baselines, VAE, joint constraints
        "G15": RunSpec(
            run_id="G15",
            description="Geo sweep: baselines, VAE, joint constraints, 5 reps",
            space="vae",
            objectives=(),
            constraint_mode="joint",
            enforce_exact_k=True,
            sweep_k=(100,),
            n_reps=5,
            baselines_enabled=True,
            eval_enabled=True,
            requires_vae=True,
            requires_pca=False,
        ),

        # G16: Baselines, VAE, no geo constraint (exact-k only)
        "G16": RunSpec(
            run_id="G16",
            description="Geo sweep: baselines, VAE, no geo constraint, 5 reps",
            space="vae",
            objectives=(),
            constraint_mode="none",
            enforce_exact_k=True,
            sweep_k=(100,),
            n_reps=5,
            baselines_enabled=True,
            eval_enabled=True,
            requires_vae=True,
            requires_pca=False,
        ),

        # ================================================================
        # G17–G22: SOFT constraint variants (NSGA-II only)
        #
        # Baselines always use hard deterministic quotas, so soft
        # variants are NSGA-II internal comparisons showing the effect
        # of soft vs hard constraint enforcement.
        # ================================================================

        # G17: NSGA-II, raw, population-share SOFT (KL tolerance tau=0.02)
        "G17": RunSpec(
            run_id="G17",
            description="Soft vs hard: NSGA-II, raw, pop-share SOFT, bi-obj, 5 reps",
            space="raw",
            objectives=("mmd", "sinkhorn"),
            constraint_mode="population_share",
            enforce_exact_k=True,
            sweep_k=(100,),
            n_reps=5,
            requires_vae=False,
            requires_pca=False,
            mmd_rff_dim=20_000,
            sinkhorn_n_anchors=400,
        ),

        # G18: NSGA-II, raw, municipality-share SOFT (KL tolerance tau=0.02)
        "G18": RunSpec(
            run_id="G18",
            description="Soft vs hard: NSGA-II, raw, muni-share SOFT, bi-obj, 5 reps",
            space="raw",
            objectives=("mmd", "sinkhorn"),
            constraint_mode="municipality_share",
            enforce_exact_k=True,
            sweep_k=(100,),
            n_reps=5,
            requires_vae=False,
            requires_pca=False,
            mmd_rff_dim=20_000,
            sinkhorn_n_anchors=400,
        ),

        # G19: NSGA-II, VAE, population-share SOFT (KL tolerance tau=0.02)
        "G19": RunSpec(
            run_id="G19",
            description="Soft vs hard: NSGA-II, VAE, pop-share SOFT, bi-obj, 5 reps",
            space="vae",
            objectives=("mmd", "sinkhorn"),
            constraint_mode="population_share",
            enforce_exact_k=True,
            sweep_k=(100,),
            n_reps=5,
            requires_vae=True,
            requires_pca=False,
        ),

        # G20: NSGA-II, VAE, municipality-share SOFT (KL tolerance tau=0.02)
        "G20": RunSpec(
            run_id="G20",
            description="Soft vs hard: NSGA-II, VAE, muni-share SOFT, bi-obj, 5 reps",
            space="vae",
            objectives=("mmd", "sinkhorn"),
            constraint_mode="municipality_share",
            enforce_exact_k=True,
            sweep_k=(100,),
            n_reps=5,
            requires_vae=True,
            requires_pca=False,
        ),

        # G21: NSGA-II, PCA, population-share SOFT
        "G21": RunSpec(
            run_id="G21",
            description="Soft vs hard: NSGA-II, PCA, pop-share SOFT, bi-obj, 5 reps",
            space="pca",
            objectives=("mmd", "sinkhorn"),
            constraint_mode="population_share",
            enforce_exact_k=True,
            sweep_k=(100,),
            n_reps=5,
            requires_vae=False,
            requires_pca=True,
        ),

        # G22: NSGA-II, PCA, municipality-share SOFT
        "G22": RunSpec(
            run_id="G22",
            description="Soft vs hard: NSGA-II, PCA, muni-share SOFT, bi-obj, 5 reps",
            space="pca",
            objectives=("mmd", "sinkhorn"),
            constraint_mode="municipality_share",
            enforce_exact_k=True,
            sweep_k=(100,),
            n_reps=5,
            requires_vae=False,
            requires_pca=True,
        ),

        # ================================================================
        # G23–G30: PCA space geographic sweep (HARD constraints)
        #
        # Completes the 3-space coverage: raw (G1-G8), VAE (G9-G16),
        # PCA (G23-G30).  Same constraint modes and mechanisms.
        # ================================================================

        # --- G23–G26: NSGA-II, PCA, HARD constraint modes ---

        # G23: NSGA-II, PCA, population-share HARD quota
        "G23": RunSpec(
            run_id="G23",
            description="Geo sweep: NSGA-II, PCA, pop-share hard quota, bi-obj, 5 reps",
            space="pca",
            objectives=("mmd", "sinkhorn"),
            constraint_mode="population_share_quota",
            enforce_exact_k=True,
            sweep_k=(100,),
            n_reps=5,
            requires_vae=False,
            requires_pca=True,
        ),

        # G24: NSGA-II, PCA, municipality-share quota
        "G24": RunSpec(
            run_id="G24",
            description="Geo sweep: NSGA-II, PCA, muni-share quota, bi-obj, 5 reps",
            space="pca",
            objectives=("mmd", "sinkhorn"),
            constraint_mode="municipality_share_quota",
            enforce_exact_k=True,
            sweep_k=(100,),
            n_reps=5,
            requires_vae=False,
            requires_pca=True,
        ),

        # G25: NSGA-II, PCA, joint constraint
        "G25": RunSpec(
            run_id="G25",
            description="Geo sweep: NSGA-II, PCA, joint constraints, bi-obj, 5 reps",
            space="pca",
            objectives=("mmd", "sinkhorn"),
            constraint_mode="joint",
            enforce_exact_k=True,
            sweep_k=(100,),
            n_reps=5,
            requires_vae=False,
            requires_pca=True,
        ),

        # G26: NSGA-II, PCA, unconstrained
        "G26": RunSpec(
            run_id="G26",
            description="Geo sweep: NSGA-II, PCA, no geo constraint, bi-obj, 5 reps",
            space="pca",
            objectives=("mmd", "sinkhorn"),
            constraint_mode="none",
            enforce_exact_k=True,
            sweep_k=(100,),
            n_reps=5,
            requires_vae=False,
            requires_pca=True,
        ),

        # --- G27–G30: Baselines, PCA, HARD constraint modes ---

        # G27: Baselines, PCA, pop-share hard quota
        "G27": RunSpec(
            run_id="G27",
            description="Geo sweep: baselines, PCA, pop-share hard quota, 5 reps",
            space="pca",
            objectives=(),
            constraint_mode="population_share_quota",
            enforce_exact_k=True,
            sweep_k=(100,),
            n_reps=5,
            baselines_enabled=True,
            eval_enabled=True,
            requires_vae=False,
            requires_pca=True,
        ),

        # G28: Baselines, PCA, municipality-share quota
        "G28": RunSpec(
            run_id="G28",
            description="Geo sweep: baselines, PCA, muni-share quota, 5 reps",
            space="pca",
            objectives=(),
            constraint_mode="municipality_share_quota",
            enforce_exact_k=True,
            sweep_k=(100,),
            n_reps=5,
            baselines_enabled=True,
            eval_enabled=True,
            requires_vae=False,
            requires_pca=True,
        ),

        # G29: Baselines, PCA, joint quota
        "G29": RunSpec(
            run_id="G29",
            description="Geo sweep: baselines, PCA, joint constraints, 5 reps",
            space="pca",
            objectives=(),
            constraint_mode="joint",
            enforce_exact_k=True,
            sweep_k=(100,),
            n_reps=5,
            baselines_enabled=True,
            eval_enabled=True,
            requires_vae=False,
            requires_pca=True,
        ),

        # G30: Baselines, PCA, no geo constraint
        "G30": RunSpec(
            run_id="G30",
            description="Geo sweep: baselines, PCA, no geo constraint, 5 reps",
            space="pca",
            objectives=(),
            constraint_mode="none",
            enforce_exact_k=True,
            sweep_k=(100,),
            n_reps=5,
            baselines_enabled=True,
            eval_enabled=True,
            requires_vae=False,
            requires_pca=True,
        ),

        # ================================================================
        # G31–G48: SOFT-constraint baselines and mixed-joint experiments
        #
        # G31–G36:  Baselines under SOFT single constraints (pop-share
        #           and muni-share) across raw and PCA spaces.
        #           VAE equivalents (G33/G34) are served by R10b/R10d
        #           and are NOT added here to avoid duplication.
        #
        # G37–G48:  Mixed-joint constraint experiments where one axis
        #           uses soft KL-repair and the other uses hard quotas.
        #           Covers NSGA-II and baselines in raw, VAE, and PCA.
        # ================================================================

        # --- G31–G32: Baselines, RAW, SOFT single constraints ---

        # G31: Baselines, raw, pop-share SOFT
        "G31": RunSpec(
            run_id="G31",
            description="Baselines, raw, pop-share SOFT, 5 reps",
            space="raw",
            objectives=(),
            constraint_mode="population_share",
            enforce_exact_k=True,
            sweep_k=(100,),
            n_reps=5,
            baselines_enabled=True,
            eval_enabled=True,
            requires_vae=False,
            requires_pca=False,
        ),

        # G32: Baselines, raw, muni-share SOFT
        "G32": RunSpec(
            run_id="G32",
            description="Baselines, raw, muni-share SOFT, 5 reps",
            space="raw",
            objectives=(),
            constraint_mode="municipality_share",
            enforce_exact_k=True,
            sweep_k=(100,),
            n_reps=5,
            baselines_enabled=True,
            eval_enabled=True,
            requires_vae=False,
            requires_pca=False,
        ),

        # G33: SKIP — dup of R10 (B_v_ps: baselines, VAE, pop-share SOFT)
        # G34: SKIP — dup of R10 (B_v_ms: baselines, VAE, muni-share SOFT)

        # --- G35–G36: Baselines, PCA, SOFT single constraints ---

        # G35: Baselines, PCA, pop-share SOFT
        "G35": RunSpec(
            run_id="G35",
            description="Baselines, PCA, pop-share SOFT, 5 reps",
            space="pca",
            objectives=(),
            constraint_mode="population_share",
            enforce_exact_k=True,
            sweep_k=(100,),
            n_reps=5,
            baselines_enabled=True,
            eval_enabled=True,
            requires_vae=False,
            requires_pca=True,
        ),

        # G36: Baselines, PCA, muni-share SOFT
        "G36": RunSpec(
            run_id="G36",
            description="Baselines, PCA, muni-share SOFT, 5 reps",
            space="pca",
            objectives=(),
            constraint_mode="municipality_share",
            enforce_exact_k=True,
            sweep_k=(100,),
            n_reps=5,
            baselines_enabled=True,
            eval_enabled=True,
            requires_vae=False,
            requires_pca=True,
        ),

        # --- G37–G40: Mixed-joint, RAW space ---

        # G37: NSGA-II, raw, joint SOFT-pop + HARD-muni
        "G37": RunSpec(
            run_id="G37",
            description="NSGA-II, raw, joint SOFT-pop + HARD-muni, bi-obj, 5 reps",
            space="raw",
            objectives=("mmd", "sinkhorn"),
            constraint_mode="joint_soft_hard",
            enforce_exact_k=True,
            sweep_k=(100,),
            n_reps=5,
            requires_vae=False,
            requires_pca=False,
            mmd_rff_dim=20_000,
            sinkhorn_n_anchors=400,
        ),

        # G38: NSGA-II, raw, joint HARD-pop + SOFT-muni
        "G38": RunSpec(
            run_id="G38",
            description="NSGA-II, raw, joint HARD-pop + SOFT-muni, bi-obj, 5 reps",
            space="raw",
            objectives=("mmd", "sinkhorn"),
            constraint_mode="joint_hard_soft",
            enforce_exact_k=True,
            sweep_k=(100,),
            n_reps=5,
            requires_vae=False,
            requires_pca=False,
            mmd_rff_dim=20_000,
            sinkhorn_n_anchors=400,
        ),

        # G39: Baselines, raw, joint SOFT-pop + HARD-muni
        "G39": RunSpec(
            run_id="G39",
            description="Baselines, raw, joint SOFT-pop + HARD-muni, 5 reps",
            space="raw",
            objectives=(),
            constraint_mode="joint_soft_hard",
            enforce_exact_k=True,
            sweep_k=(100,),
            n_reps=5,
            baselines_enabled=True,
            eval_enabled=True,
            requires_vae=False,
            requires_pca=False,
        ),

        # G40: Baselines, raw, joint HARD-pop + SOFT-muni
        "G40": RunSpec(
            run_id="G40",
            description="Baselines, raw, joint HARD-pop + SOFT-muni, 5 reps",
            space="raw",
            objectives=(),
            constraint_mode="joint_hard_soft",
            enforce_exact_k=True,
            sweep_k=(100,),
            n_reps=5,
            baselines_enabled=True,
            eval_enabled=True,
            requires_vae=False,
            requires_pca=False,
        ),

        # --- G41–G44: Mixed-joint, VAE space ---

        # G41: NSGA-II, VAE, joint SOFT-pop + HARD-muni
        "G41": RunSpec(
            run_id="G41",
            description="NSGA-II, VAE, joint SOFT-pop + HARD-muni, bi-obj, 5 reps",
            space="vae",
            objectives=("mmd", "sinkhorn"),
            constraint_mode="joint_soft_hard",
            enforce_exact_k=True,
            sweep_k=(100,),
            n_reps=5,
            requires_vae=True,
            requires_pca=False,
        ),

        # G42: NSGA-II, VAE, joint HARD-pop + SOFT-muni
        "G42": RunSpec(
            run_id="G42",
            description="NSGA-II, VAE, joint HARD-pop + SOFT-muni, bi-obj, 5 reps",
            space="vae",
            objectives=("mmd", "sinkhorn"),
            constraint_mode="joint_hard_soft",
            enforce_exact_k=True,
            sweep_k=(100,),
            n_reps=5,
            requires_vae=True,
            requires_pca=False,
        ),

        # G43: Baselines, VAE, joint SOFT-pop + HARD-muni
        "G43": RunSpec(
            run_id="G43",
            description="Baselines, VAE, joint SOFT-pop + HARD-muni, 5 reps",
            space="vae",
            objectives=(),
            constraint_mode="joint_soft_hard",
            enforce_exact_k=True,
            sweep_k=(100,),
            n_reps=5,
            baselines_enabled=True,
            eval_enabled=True,
            requires_vae=True,
            requires_pca=False,
        ),

        # G44: Baselines, VAE, joint HARD-pop + SOFT-muni
        "G44": RunSpec(
            run_id="G44",
            description="Baselines, VAE, joint HARD-pop + SOFT-muni, 5 reps",
            space="vae",
            objectives=(),
            constraint_mode="joint_hard_soft",
            enforce_exact_k=True,
            sweep_k=(100,),
            n_reps=5,
            baselines_enabled=True,
            eval_enabled=True,
            requires_vae=True,
            requires_pca=False,
        ),

        # --- G45–G48: Mixed-joint, PCA space ---

        # G45: NSGA-II, PCA, joint SOFT-pop + HARD-muni
        "G45": RunSpec(
            run_id="G45",
            description="NSGA-II, PCA, joint SOFT-pop + HARD-muni, bi-obj, 5 reps",
            space="pca",
            objectives=("mmd", "sinkhorn"),
            constraint_mode="joint_soft_hard",
            enforce_exact_k=True,
            sweep_k=(100,),
            n_reps=5,
            requires_vae=False,
            requires_pca=True,
        ),

        # G46: NSGA-II, PCA, joint HARD-pop + SOFT-muni
        "G46": RunSpec(
            run_id="G46",
            description="NSGA-II, PCA, joint HARD-pop + SOFT-muni, bi-obj, 5 reps",
            space="pca",
            objectives=("mmd", "sinkhorn"),
            constraint_mode="joint_hard_soft",
            enforce_exact_k=True,
            sweep_k=(100,),
            n_reps=5,
            requires_vae=False,
            requires_pca=True,
        ),

        # G47: Baselines, PCA, joint SOFT-pop + HARD-muni
        "G47": RunSpec(
            run_id="G47",
            description="Baselines, PCA, joint SOFT-pop + HARD-muni, 5 reps",
            space="pca",
            objectives=(),
            constraint_mode="joint_soft_hard",
            enforce_exact_k=True,
            sweep_k=(100,),
            n_reps=5,
            baselines_enabled=True,
            eval_enabled=True,
            requires_vae=False,
            requires_pca=True,
        ),

        # G48: Baselines, PCA, joint HARD-pop + SOFT-muni
        "G48": RunSpec(
            run_id="G48",
            description="Baselines, PCA, joint HARD-pop + SOFT-muni, 5 reps",
            space="pca",
            objectives=(),
            constraint_mode="joint_hard_soft",
            enforce_exact_k=True,
            sweep_k=(100,),
            n_reps=5,
            baselines_enabled=True,
            eval_enabled=True,
            requires_vae=False,
            requires_pca=True,
        ),
    }



    # ==================================================================
    # V2 naming convention: create aliases for every v2 experiment name
    # ==================================================================
    #
    # The v2 naming convention uses families:
    #   P     — Prerequisites (quota path computation)
    #   K_*   — Cardinality sweeps (k=30..500)
    #   A_*   — Ablations (objective / extreme constraint)
    #   N_*_* — NSGA-II point experiments (space x constraint, k=100)
    #   B_*_* — Baseline point experiments (space x constraint, k=100)
    #   T_*   — Tuning & sensitivity (effort, dim sweeps)
    #   D     — Diagnostics (proxy stability)
    #
    # Each v2 alias maps to an existing RunSpec (R-series or G-series)
    # with the run_id updated to the v2 name.
    #
    # Some v2 grid cells do NOT have a dedicated G-series entry because
    # they are identical to a K-sweep or A-ablation config at k=100.
    # These are listed in the "deduplication map":
    #   N_v_ps  -> K_vae at k=100  (= R1)
    #   N_v_0   -> A_none          (= R6)
    #   N_v_hh  -> A_jhh           (= R5)
    #   N_r_ps  -> K_raw at k=100  (= R8)
    #   N_p_ps  -> K_pca at k=100  (= R9)
    #
    # For these dedup cells, we create the v2 alias pointing to the
    # source RunSpec with overridden run_id AND sweep_k=(100,) so that
    # when used directly (not as part of a k-sweep), only k=100 is run.

    # -- Straight 1:1 aliases (source RunSpec exists, just rename) ------

    _V2_DIRECT_MAP = {
        # P family
        "P":        "R0",

        # K family (keep full k-sweep)
        "K_vae":    "R1",
        "K_raw":    "R8",
        "K_pca":    "R9",

        # A family — single-rep ablations
        "A_mmd":    "R2",
        "A_sink":   "R3",
        "A_jhh":    "R5",   # joint HARD+HARD, VAE
        "A_none":   "R6",   # unconstrained, VAE
        "A_tri":    "R15",  # tri-objective LogDet

        # N family — NSGA-II grid (VAE space)
        "N_v_mh":   "R4",   # formerly R4: NSGA-II, VAE, muni-HARD
        "N_v_ph":   "G9",   # NSGA-II, VAE, pop-HARD
        "N_v_ms":   "G20",  # NSGA-II, VAE, muni-SOFT
        "N_v_sh":   "G41",  # NSGA-II, VAE, joint SOFT-pop + HARD-muni
        "N_v_hs":   "G42",  # NSGA-II, VAE, joint HARD-pop + SOFT-muni

        # N family — NSGA-II grid (RAW space)
        "N_r_ph":   "G1",
        "N_r_mh":   "G2",
        "N_r_hh":   "G3",
        "N_r_0":    "G4",
        "N_r_ms":   "G18",
        "N_r_sh":   "G37",
        "N_r_hs":   "G38",

        # N family — NSGA-II grid (PCA space)
        "N_p_ph":   "G23",
        "N_p_mh":   "G24",
        "N_p_hh":   "G25",
        "N_p_0":    "G26",
        "N_p_ms":   "G22",
        "N_p_sh":   "G45",
        "N_p_hs":   "G46",

        # B family — Baselines (VAE space)
        "B_v_ph":   "G13",  # baselines, VAE, pop-HARD
        "B_v_mh":   "G14",  # baselines, VAE, muni-HARD
        "B_v_hh":   "G15",  # baselines, VAE, joint HH
        "B_v_sh":   "G43",  # baselines, VAE, joint S+H
        "B_v_hs":   "G44",  # baselines, VAE, joint H+S

        # B family — Baselines (RAW space)
        "B_r_ph":   "G5",
        "B_r_mh":   "G6",
        "B_r_hh":   "G7",
        "B_r_0":    "G8",
        "B_r_ps":   "G31",
        "B_r_ms":   "G32",
        "B_r_sh":   "G39",
        "B_r_hs":   "G40",

        # B family — Baselines (PCA space)
        "B_p_ph":   "G27",
        "B_p_mh":   "G28",
        "B_p_hh":   "G29",
        "B_p_0":    "G30",
        "B_p_ps":   "G35",
        "B_p_ms":   "G36",
        "B_p_sh":   "G47",
        "B_p_hs":   "G48",

        # T family
        "T_eff":    "R12",
        "T_vdim":   "R13",
        "T_pdim":   "R14",

        # D family
        "D":        "R11",
    }

    for _v2_name, _old_name in _V2_DIRECT_MAP.items():
        if _old_name in specs:
            specs[_v2_name] = replace(specs[_old_name], run_id=_v2_name)

    # -- Per-space diagnostics: D_vae, D_raw, D_pca ----
    if "R11" in specs:
        specs["D_vae"] = replace(specs["R11"], run_id="D_vae", space="vae",
                                 requires_vae=True, requires_pca=False)
        specs["D_raw"] = replace(specs["R11"], run_id="D_raw", space="raw",
                                 requires_vae=False, requires_pca=False)
        specs["D_pca"] = replace(specs["R11"], run_id="D_pca", space="pca",
                                 requires_vae=False, requires_pca=True)

    # -- Dedup aliases: v2 grid cells served by K-sweep or A-ablation --
    # These need sweep_k=(100,) override so they only run k=100 when
    # invoked directly, unlike the K-sweep parent which runs all K_GRID.

    _V2_DEDUP_MAP = {
        # v2_name: (source_run_id, description_suffix)
        "N_v_ps":  ("R1",  "NSGA-II, VAE, pop-share SOFT (= K_vae@k=100)"),
        "N_r_ps":  ("R8",  "NSGA-II, RAW, pop-share SOFT (= K_raw@k=100)"),
        "N_p_ps":  ("R9",  "NSGA-II, PCA, pop-share SOFT (= K_pca@k=100)"),
        "N_v_0":   ("R6",  "NSGA-II, VAE, unconstrained  (= A_none)"),
        "N_v_hh":  ("R5",  "NSGA-II, VAE, joint HH       (= A_jhh)"),
    }

    for _v2_name, (_src, _desc) in _V2_DEDUP_MAP.items():
        if _src in specs:
            specs[_v2_name] = replace(
                specs[_src],
                run_id=_v2_name,
                description=_desc,
                sweep_k=(100,),
                n_reps=5,
            )

    # -- New dedicated RunSpecs for B_v_0, B_v_ps, B_v_ms ---------------
    # In the original codebase, R10 was a single entry for "baselines, VAE,
    # pop-share SOFT".  The v2 design splits baselines into per-constraint
    # entries.  B_v_ph/B_v_mh/B_v_hh are served by G13/G14/G15 above.
    # B_v_0, B_v_ps, B_v_ms need fresh RunSpecs:

    # B_v_0: Baselines, VAE, unconstrained
    specs["B_v_0"] = RunSpec(
        run_id="B_v_0",
        description="Baselines, VAE, unconstrained, 5 reps",
        space="vae",
        objectives=(),
        constraint_mode="none",
        enforce_exact_k=True,
        sweep_k=(100,),
        n_reps=5,
        baselines_enabled=True,
        eval_enabled=True,
        requires_vae=True,
        requires_pca=False,
    )

    # B_v_ps: Baselines, VAE, pop-share SOFT
    specs["B_v_ps"] = RunSpec(
        run_id="B_v_ps",
        description="Baselines, VAE, pop-share SOFT, 5 reps",
        space="vae",
        objectives=(),
        constraint_mode="population_share",
        enforce_exact_k=True,
        sweep_k=(100,),
        n_reps=5,
        baselines_enabled=True,
        eval_enabled=True,
        requires_vae=True,
        requires_pca=False,
    )

    # B_v_ms: Baselines, VAE, muni-share SOFT
    specs["B_v_ms"] = RunSpec(
        run_id="B_v_ms",
        description="Baselines, VAE, muni-share SOFT, 5 reps",
        space="vae",
        objectives=(),
        constraint_mode="municipality_share",
        enforce_exact_k=True,
        sweep_k=(100,),
        n_reps=5,
        baselines_enabled=True,
        eval_enabled=True,
        requires_vae=True,
        requires_pca=False,
    )

    return specs

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
    # Cache path — use a dimension-specific directory when dim_override is set
    # to prevent T_vdim / T_pdim jobs from corrupting the default d=32 cache
    # that all other experiments rely on.
    if dim_override is not None:
        cache_dir = f"{base_cfg.files.cache_dir}_d{dim_override}"
    else:
        cache_dir = base_cfg.files.cache_dir
    cache_path = os.path.join(cache_dir, f"rep{rep_id:02d}", "assets.npz")

    # Geo / constraints
    geo_cfg = replace(
        base_cfg.geo,
        constraint_mode=str(spec.constraint_mode),
        # Keep the historical flag coherent
        use_quota_constraints=(str(spec.constraint_mode) in {"municipality_share_quota", "joint", "population_share_quota", "joint_soft_hard", "joint_hard_soft"}),
    )

    # Solver
    solver_k = int(spec.k) if spec.k is not None else base_cfg.solver.k
    solver_cfg = replace(
        base_cfg.solver,
        k=solver_k,
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

    # Per-run objective parameter overrides (e.g. R8 raw-space needs larger RFF/anchors)
    if spec.mmd_rff_dim is not None:
        cfg = replace(cfg, mmd=replace(cfg.mmd, rff_dim=int(spec.mmd_rff_dim)))
    if spec.sinkhorn_n_anchors is not None:
        cfg = replace(cfg, sinkhorn=replace(cfg.sinkhorn, n_anchors=int(spec.sinkhorn_n_anchors)))

    # Auto-enable Nystrom log-det when listed as an objective
    if "nystrom_logdet" in spec.objectives:
        cfg = replace(cfg, nystrom_logdet=replace(cfg.nystrom_logdet, enabled=True))

    # Dimension override for R13/R14 dimension sweeps
    if dim_override is not None:
        if spec.space == "vae":
            cfg = replace(cfg, vae=replace(cfg.vae, latent_dim=int(dim_override)))
        elif spec.space == "pca":
            cfg = replace(cfg, pca=replace(cfg.pca, n_components=int(dim_override)))

    return cfg
