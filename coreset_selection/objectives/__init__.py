"""
Objective functions for coreset selection.

This module provides distributional divergence measures:
- SKL: Symmetric KL divergence for diagonal Gaussians
- MMD²: Maximum Mean Discrepancy via Random Fourier Features
- Sinkhorn: Sinkhorn divergence via anchor approximation
- NystromLogDet: Log-determinant diversity via Nystrom kernel sub-matrix

And a unified interface:
- SpaceObjectiveComputer: Computes all objectives efficiently
"""

from .skl import (
    symmetric_kl_diag_gaussians,
    kl_diag_gaussians,
    jeffreys_divergence_diag_gaussians,
)

from .mmd import (
    RFFMMD,
    compute_rff_features,
    mmd2_exact,
)

from .nystrom_logdet import NystromLogDet

from .sinkhorn import (
    AnchorSinkhorn,
    sinkhorn2_safe,
)

from .computer import (
    SpaceObjectiveComputer,
    build_space_objective_computer,
)

__all__ = [
    # SKL
    "symmetric_kl_diag_gaussians",
    "kl_diag_gaussians",
    "jeffreys_divergence_diag_gaussians",
    # MMD
    "RFFMMD",
    "compute_rff_features",
    "mmd2_exact",
    # Nystrom log-det
    "NystromLogDet",
    # Sinkhorn
    "AnchorSinkhorn",
    "sinkhorn2_safe",
    # Unified computer
    "SpaceObjectiveComputer",
    "build_space_objective_computer",
]
