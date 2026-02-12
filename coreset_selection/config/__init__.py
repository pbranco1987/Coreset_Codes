"""
Configuration module for coreset selection experiments.

This module provides:
- Configuration dataclasses for all experiment components
- Run specifications for manuscript experiments (R0-R12)
- Manuscript constants for reproducibility
- Utilities for applying and combining configurations
"""

from .dataclasses import (
    FilesConfig,
    PreprocessingConfig,
    VAEConfig,
    GeoConfig,
    SinkhornConfig,
    MMDConfig,
    EffortSweepGrid,
    SolverConfig,
    EvalConfig,
    BaselinesConfig,
    AblationsConfig,
    SweepConfig,
    ManuscriptFiguresConfig,
    PCAConfig,
    ExperimentConfig,
    ReplicateAssets,
    ResultsBundle,
)

from .run_specs import RunSpec, get_run_specs, apply_run_spec, K_GRID

from .constants import (
    # Dataset constants
    N_MUNICIPALITIES,
    G_STATES,
    D_FEATURES,
    # Cardinality grid
    K_PRIMARY,
    # NSGA-II parameters
    NSGA2_POP_SIZE,
    NSGA2_N_GENERATIONS,
    NSGA2_CROSSOVER_PROB,
    NSGA2_MUTATION_PROB,
    # Objective parameters
    RFF_DIM_DEFAULT,
    RFF_DIM_SENSITIVITY,
    SINKHORN_N_ANCHORS,
    SINKHORN_ETA,
    SINKHORN_MAX_ITER,
    SINKHORN_ANCHOR_SENSITIVITY,
    SKL_VAR_CLAMP_MIN,
    SKL_VAR_CLAMP_MAX,
    # Geographic parameters
    ALPHA_GEO,
    # VAE parameters
    VAE_LATENT_DIM,
    VAE_HIDDEN_DIM,
    VAE_EPOCHS,
    VAE_EARLY_STOPPING_PATIENCE,
    VAE_BATCH_SIZE,
    VAE_LR,
    VAE_KL_WEIGHT,
    # Evaluation parameters
    EVAL_SIZE,
    EVAL_TRAIN_FRAC,
    KPCA_COMPONENTS,
    NYSTROM_LAMBDA,
    # Experimental design
    N_REPLICATES_PRIMARY,
    N_REPLICATES_DEFAULT,
    RUN_IDS,
    RUN_OBJECTIVES,
    BASELINE_METHODS,
    # Figure/table tracking
    N_FIGURES,
    N_TABLES,
    FIGURE_FILES,
    TABLE_LABELS,
    # Manuscript-aligned artifact lists
    MANUSCRIPT_FIGURE_FILES,
    COMPLEMENTARY_FIGURE_FILES,
    LEGACY_FIGURE_FILES,
    MANUSCRIPT_TABLE_LABELS,
    MANUSCRIPT_TABLE_FILES,
    COMPLEMENTARY_TABLE_FILES,
    # Multi-target coverage
    COVERAGE_TARGETS,
    COVERAGE_TARGET_NAMES,
    # Numerical stability
    EPS_NORM,
    EPS_LOG,
)

__all__ = [
    # Configuration dataclasses
    "FilesConfig",
    "PreprocessingConfig",
    "VAEConfig",
    "GeoConfig",
    "SinkhornConfig",
    "MMDConfig",
    "EffortSweepGrid",
    "SolverConfig",
    "EvalConfig",
    "BaselinesConfig",
    "AblationsConfig",
    "SweepConfig",
    "ManuscriptFiguresConfig",
    "PCAConfig",
    "ExperimentConfig",
    "ReplicateAssets",
    "ResultsBundle",
    # Run specifications
    "RunSpec",
    "get_run_specs",
    "apply_run_spec",
    "K_GRID",
    # Manuscript constants
    "N_MUNICIPALITIES",
    "G_STATES",
    "D_FEATURES",
    "K_PRIMARY",
    "NSGA2_POP_SIZE",
    "NSGA2_N_GENERATIONS",
    "NSGA2_CROSSOVER_PROB",
    "NSGA2_MUTATION_PROB",
    "RFF_DIM_DEFAULT",
    "RFF_DIM_SENSITIVITY",
    "SINKHORN_N_ANCHORS",
    "SINKHORN_ETA",
    "SINKHORN_MAX_ITER",
    "SINKHORN_ANCHOR_SENSITIVITY",
    "SKL_VAR_CLAMP_MIN",
    "SKL_VAR_CLAMP_MAX",
    "ALPHA_GEO",
    "VAE_LATENT_DIM",
    "VAE_HIDDEN_DIM",
    "VAE_EPOCHS",
    "VAE_EARLY_STOPPING_PATIENCE",
    "VAE_BATCH_SIZE",
    "VAE_LR",
    "VAE_KL_WEIGHT",
    "EVAL_SIZE",
    "EVAL_TRAIN_FRAC",
    "KPCA_COMPONENTS",
    "NYSTROM_LAMBDA",
    "N_REPLICATES_PRIMARY",
    "N_REPLICATES_DEFAULT",
    "RUN_IDS",
    "RUN_OBJECTIVES",
    "BASELINE_METHODS",
    "N_FIGURES",
    "N_TABLES",
    "FIGURE_FILES",
    "TABLE_LABELS",
    "MANUSCRIPT_FIGURE_FILES",
    "COMPLEMENTARY_FIGURE_FILES",
    "LEGACY_FIGURE_FILES",
    "MANUSCRIPT_TABLE_LABELS",
    "MANUSCRIPT_TABLE_FILES",
    "COMPLEMENTARY_TABLE_FILES",
    "COVERAGE_TARGETS",
    "COVERAGE_TARGET_NAMES",
    "EPS_NORM",
    "EPS_LOG",
]
