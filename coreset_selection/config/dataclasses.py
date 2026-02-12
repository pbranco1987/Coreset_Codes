"""
Configuration dataclasses for coreset selection experiments.

Re-export facade — all dataclasses are defined in sub-modules and
re-exported here so that existing ``from .config.dataclasses import …``
statements continue to work unchanged.

Sub-modules
-----------
_dc_components : Component-level configs (Files, Preprocessing, VAE, Geo,
                 Sinkhorn, MMD, PCA)
_dc_experiment : Experiment-level configs (EffortSweepGrid, Solver, Eval,
                 Baselines, Ablations, Sweep, ManuscriptFigures, Experiment)
_dc_results    : Result containers (ReplicateAssets, ResultsBundle)
"""

from ._dc_components import (
    FilesConfig,
    PreprocessingConfig,
    VAEConfig,
    GeoConfig,
    SinkhornConfig,
    MMDConfig,
    PCAConfig,
)

from ._dc_experiment import (
    EffortSweepGrid,
    SolverConfig,
    EvalConfig,
    BaselinesConfig,
    AblationsConfig,
    SweepConfig,
    ManuscriptFiguresConfig,
    ExperimentConfig,
)

from ._dc_results import (
    ReplicateAssets,
    ResultsBundle,
)

__all__ = [
    # Component-level configs
    "FilesConfig",
    "PreprocessingConfig",
    "VAEConfig",
    "GeoConfig",
    "SinkhornConfig",
    "MMDConfig",
    "PCAConfig",
    # Experiment-level configs
    "EffortSweepGrid",
    "SolverConfig",
    "EvalConfig",
    "BaselinesConfig",
    "AblationsConfig",
    "SweepConfig",
    "ManuscriptFiguresConfig",
    "ExperimentConfig",
    # Result containers
    "ReplicateAssets",
    "ResultsBundle",
]
