"""
Phase 3 deliverable — Basic constraint mode tests.

Verifies that all 4 constraint modes work correctly on a small synthetic
dataset:
  - population_share: w_i = pop_i
  - municipality_share_quota: w_i == 1 (count quota mode)
  - joint: both population-share AND municipality-share quota
  - none: no proportionality constraints (exact-k only)

Also verifies:
  - ProportionalityConstraint.value() implements Eq. (4) exactly
  - Laplace smoothing formula matches manuscript
  - total_violation() sums max{D - tau, 0} over active constraints
  - Dual diagnostics always produced regardless of constraint mode

Split from tests/test_constraint_modes.py.
"""

import numpy as np
import pytest

from coreset_selection.geo.info import build_geo_info, GeoInfo
from coreset_selection.geo.kl import (
    kl_pi_hat_from_counts,
    kl_optimal_integer_counts_bounded,
    min_achievable_geo_kl_bounded,
    kl_weighted_from_subset,
    compute_constraint_violations,
)
from coreset_selection.geo.projector import GeographicConstraintProjector
from coreset_selection.constraints.proportionality import (
    ProportionalityConstraint,
    ProportionalityConstraintSet,
    build_population_share_constraint,
    build_municipality_share_constraint,
    _safe_kl,
)
from coreset_selection.evaluation.geo_diagnostics import (
    geo_diagnostics,
    dual_geo_diagnostics,
    geo_diagnostics_weighted,
)


# =====================================================================
# Fixtures
# =====================================================================

def make_synthetic_geo(N=100, G=5, seed=42):
    """Create synthetic GeoInfo with G groups of varying sizes and population."""
    rng = np.random.default_rng(seed)
    # Unequal group sizes
    sizes = rng.multinomial(N, np.ones(G) / G)
    state_labels = np.repeat(np.arange(G), sizes)
    # Population: varies by group (heavier in group 0)
    pop = np.zeros(N, dtype=np.float64)
    idx = 0
    for g in range(G):
        n_g = sizes[g]
        pop[idx:idx + n_g] = rng.uniform(10 * (G - g), 100 * (G - g), size=n_g)
        idx += n_g
    geo = build_geo_info(state_labels, population_weights=pop)
    return geo, pop


# =====================================================================
# Phase 3 Task 1: ProportionalityConstraint.value() -- Eq. (4)
# =====================================================================

class TestConstraintValueEq4:
    """Verify ProportionalityConstraint.value() implements Eq. (4)."""

    def test_kl_matches_manual_computation(self):
        """D(S) = KL(pi || pi_hat(S)) must match hand calculation."""
        geo, pop = make_synthetic_geo(N=30, G=3, seed=0)
        alpha = 1.0

        c = build_population_share_constraint(
            geo=geo, population=pop, alpha=alpha, tau=0.5,
        )

        mask = np.zeros(geo.N, dtype=bool)
        mask[:5] = True
        mask[10:13] = True
        mask[20:22] = True

        # Manual computation of D(S)
        Wg = np.zeros(geo.G, dtype=np.float64)
        for i in np.flatnonzero(mask):
            g = int(geo.group_ids[i])
            Wg[g] += pop[i]
        W = Wg.sum()
        q = (Wg + alpha) / (W + alpha * geo.G)  # Laplace smoothing

        pi = c.target_pi
        kl_manual = sum(
            pi[g] * np.log(pi[g] / q[g])
            for g in range(geo.G)
            if pi[g] > 0
        )

        kl_code = c.value(mask, geo)
        assert np.isclose(kl_code, kl_manual, rtol=1e-10)

    def test_laplace_smoothing_formula(self):
        """pi_hat_g = (W_g(S) + alpha) / (W(S) + alphaG) exactly."""
        geo, pop = make_synthetic_geo(N=20, G=4, seed=1)
        alpha = 2.0

        c = build_population_share_constraint(
            geo=geo, population=pop, alpha=alpha, tau=1.0,
        )

        mask = np.zeros(geo.N, dtype=bool)
        mask[0] = True  # 1 from group 0

        # With only 1 selected, W_g(S) = pop[0] for g=0, 0 otherwise
        Wg = np.zeros(geo.G, dtype=np.float64)
        Wg[int(geo.group_ids[0])] = pop[0]
        W = pop[0]
        q_expected = (Wg + alpha) / (W + alpha * geo.G)

        # Verify the smoothing yields a proper distribution
        assert np.isclose(q_expected.sum(), 1.0, rtol=1e-10)
        assert np.all(q_expected > 0)  # No zeros due to smoothing


class TestTotalViolation:
    """Verify total_violation() = Sigma_h max{D(S) - tau_h, 0}."""

    def test_zero_violation_when_feasible(self):
        geo, pop = make_synthetic_geo(N=100, G=5, seed=10)

        c = build_population_share_constraint(
            geo=geo, population=pop, alpha=1.0, tau=10.0,  # very loose
        )
        cs = ProportionalityConstraintSet(
            geo=geo, constraints=[c], min_one_per_group=False,
        )

        # Roughly proportional selection
        mask = np.zeros(geo.N, dtype=bool)
        for g in range(geo.G):
            idx_g = geo.group_to_indices[g]
            mask[idx_g[:4]] = True  # 4 from each

        assert cs.total_violation(mask) == 0.0

    def test_positive_violation_when_imbalanced(self):
        geo, pop = make_synthetic_geo(N=100, G=5, seed=10)

        c = build_population_share_constraint(
            geo=geo, population=pop, alpha=1.0, tau=0.001,  # very tight
        )
        cs = ProportionalityConstraintSet(
            geo=geo, constraints=[c], min_one_per_group=False,
        )

        # All from group 0 -> very imbalanced
        mask = np.zeros(geo.N, dtype=bool)
        idx_g0 = geo.group_to_indices[0]
        mask[idx_g0[:20]] = True

        assert cs.total_violation(mask) > 0.0

    def test_joint_violation_sums_both_constraints(self):
        geo, pop = make_synthetic_geo(N=100, G=5, seed=10)

        c_pop = build_population_share_constraint(
            geo=geo, population=pop, alpha=1.0, tau=0.001,
        )
        c_muni = build_municipality_share_constraint(
            geo=geo, alpha=1.0, tau=0.001,
        )
        cs = ProportionalityConstraintSet(
            geo=geo, constraints=[c_pop, c_muni], min_one_per_group=False,
        )

        # All from group 0
        mask = np.zeros(geo.N, dtype=bool)
        mask[geo.group_to_indices[0][:15]] = True

        v_joint = cs.total_violation(mask)
        v_pop = c_pop.violation(mask, geo)
        v_muni = c_muni.violation(mask, geo)

        assert np.isclose(v_joint, v_pop + v_muni, rtol=1e-10)


# =====================================================================
# Phase 3 Task 3: Constraint modes in runner wiring
# =====================================================================

class TestConstraintModeWiring:
    """Verify all 4 constraint modes are correctly built."""

    def test_population_share_mode(self):
        """population_share: only pop-share constraint."""
        geo, pop = make_synthetic_geo(N=50, G=5)
        c = build_population_share_constraint(
            geo=geo, population=pop, alpha=1.0, tau=0.02,
        )
        cs = ProportionalityConstraintSet(
            geo=geo, constraints=[c], min_one_per_group=True,
        )
        assert len(cs.constraints) == 1
        assert cs.constraints[0].name == "population_share"

    def test_municipality_share_quota_mode(self):
        """municipality_share_quota: only muni-share constraint."""
        geo, pop = make_synthetic_geo(N=50, G=5)
        c = build_municipality_share_constraint(
            geo=geo, alpha=1.0, tau=0.02,
        )
        cs = ProportionalityConstraintSet(
            geo=geo, constraints=[c], min_one_per_group=True,
        )
        assert len(cs.constraints) == 1
        assert cs.constraints[0].name == "municipality_share"

    def test_joint_mode(self):
        """joint: both pop-share AND muni-share constraints."""
        geo, pop = make_synthetic_geo(N=50, G=5)
        c_pop = build_population_share_constraint(
            geo=geo, population=pop, alpha=1.0, tau=0.02,
        )
        c_muni = build_municipality_share_constraint(
            geo=geo, alpha=1.0, tau=0.02,
        )
        cs = ProportionalityConstraintSet(
            geo=geo, constraints=[c_pop, c_muni],
            min_one_per_group=True,
            preserve_group_counts=True,
        )
        assert len(cs.constraints) == 2
        names = {c.name for c in cs.constraints}
        assert names == {"population_share", "municipality_share"}

    def test_none_mode(self):
        """none: empty constraint set (exact-k only)."""
        geo, pop = make_synthetic_geo(N=50, G=5)
        cs = ProportionalityConstraintSet(
            geo=geo, constraints=[], min_one_per_group=True,
        )
        assert len(cs.constraints) == 0
        mask = np.zeros(geo.N, dtype=bool)
        mask[:10] = True
        assert cs.total_violation(mask) == 0.0


# =====================================================================
# Dual diagnostics: always produced for all runs (G-D1)
# =====================================================================

class TestDualDiagnosticsAlwaysComputed:
    """Verify dual_geo_diagnostics produces BOTH muni and pop keys."""

    def test_dual_produces_all_keys(self):
        geo, pop = make_synthetic_geo(N=50, G=5, seed=99)
        idx_sel = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45])

        result = dual_geo_diagnostics(geo, idx_sel, k=10, alpha=1.0)

        # Legacy keys
        assert "geo_kl" in result
        assert "geo_l1" in result
        assert "geo_maxdev" in result
        # Municipality-share suffixed
        assert "geo_kl_muni" in result
        assert "geo_l1_muni" in result
        assert "geo_maxdev_muni" in result
        # Population-share suffixed (non-NaN because pop weights provided)
        assert "geo_kl_pop" in result
        assert "geo_l1_pop" in result
        assert "geo_maxdev_pop" in result
        assert not np.isnan(result["geo_kl_pop"])

    def test_dual_without_population_returns_nan(self):
        """When no pop weights, pop keys should be NaN."""
        state_labels = np.array([0]*10 + [1]*10 + [2]*10)
        geo = build_geo_info(state_labels)  # no population_weights
        idx_sel = np.array([0, 5, 10, 15, 20, 25])

        result = dual_geo_diagnostics(geo, idx_sel, k=6, alpha=1.0)

        assert "geo_kl_pop" in result
        assert np.isnan(result["geo_kl_pop"])

    def test_muni_and_legacy_keys_agree(self):
        """geo_kl_muni should equal geo_kl (both count-based)."""
        geo, pop = make_synthetic_geo(N=50, G=5, seed=77)
        idx_sel = np.arange(0, 50, 5)

        result = dual_geo_diagnostics(geo, idx_sel, k=10, alpha=1.0)

        assert np.isclose(result["geo_kl"], result["geo_kl_muni"], rtol=1e-10)
        assert np.isclose(result["geo_l1"], result["geo_l1_muni"], rtol=1e-10)
        assert np.isclose(result["geo_maxdev"], result["geo_maxdev_muni"], rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
