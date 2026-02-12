"""Constraint utilities.

This package implements the *weighted proportionality* constraints described
in the manuscript, including population-share proportionality and the joint
case with municipality-share quotas.

Phase 5 additions:
- ``calibration`` module for τ tolerance estimation and sensitivity sweeps.
"""

from .proportionality import (
    ProportionalityConstraint,
    ProportionalityConstraintSet,
    build_population_share_constraint,
    build_municipality_share_constraint,
)

from .calibration import (
    estimate_feasible_tau_range,
    tau_sensitivity_sweep,
    tau_sweep_to_csv_rows,
    TauSweepResult,
)

__all__ = [
    "ProportionalityConstraint",
    "ProportionalityConstraintSet",
    "build_population_share_constraint",
    "build_municipality_share_constraint",
    # Phase 5: τ calibration
    "estimate_feasible_tau_range",
    "tau_sensitivity_sweep",
    "tau_sweep_to_csv_rows",
    "TauSweepResult",
]
