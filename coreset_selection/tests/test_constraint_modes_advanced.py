"""
Phase 3 deliverable — Advanced / combined constraint tests.

Verifies:
  - Swap-repair donor/recipient selection logic (Algorithm 2)
  - KL floor formula and lazy-heap correctness (Algorithm 1)
  - Constraint-domination ordering (NSGA-II internal)

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
# Phase 3 Task 2: Swap-repair donor/recipient logic (Algorithm 2)
# =====================================================================

class TestSwapRepairLogic:
    """Verify Algorithm 2 donor/recipient selection."""

    def test_donor_is_overrepresented(self):
        """Donor = group with smallest s_g = pi_g / pi_hat_g(S)."""
        geo, pop = make_synthetic_geo(N=60, G=3, seed=20)

        c = build_municipality_share_constraint(
            geo=geo, alpha=1.0, tau=0.001,
        )
        cs = ProportionalityConstraintSet(
            geo=geo, constraints=[c], min_one_per_group=True, max_iters=1,
        )

        # Overload group 0
        mask = np.zeros(geo.N, dtype=bool)
        mask[geo.group_to_indices[0][:15]] = True
        mask[geo.group_to_indices[1][0]] = True
        mask[geo.group_to_indices[2][0]] = True

        rng = np.random.default_rng(42)
        repaired = cs.repair(mask, rng)

        # After 1 iteration, group 0 should have lost one
        c0_before = mask[geo.group_to_indices[0]].sum()
        c0_after = repaired[geo.group_to_indices[0]].sum()
        assert c0_after <= c0_before

    def test_repair_early_termination_when_feasible(self):
        """Repair should stop when V(S) = 0."""
        geo, pop = make_synthetic_geo(N=60, G=3, seed=20)

        c = build_municipality_share_constraint(
            geo=geo, alpha=1.0, tau=10.0,  # very loose
        )
        cs = ProportionalityConstraintSet(
            geo=geo, constraints=[c], min_one_per_group=False, max_iters=200,
        )

        # Already feasible mask
        mask = np.zeros(geo.N, dtype=bool)
        for g in range(geo.G):
            mask[geo.group_to_indices[g][:4]] = True

        rng = np.random.default_rng(42)
        repaired = cs.repair(mask, rng)

        # Should be unchanged (feasible -> early termination)
        np.testing.assert_array_equal(mask, repaired)

    def test_repair_preserves_cardinality(self):
        """Swap repair should not change |S|."""
        geo, pop = make_synthetic_geo(N=60, G=3, seed=20)

        c = build_municipality_share_constraint(
            geo=geo, alpha=1.0, tau=0.001,
        )
        cs = ProportionalityConstraintSet(
            geo=geo, constraints=[c], min_one_per_group=True, max_iters=50,
        )

        mask = np.zeros(geo.N, dtype=bool)
        mask[geo.group_to_indices[0][:10]] = True
        mask[geo.group_to_indices[1][0]] = True
        mask[geo.group_to_indices[2][0]] = True
        k_before = mask.sum()

        rng = np.random.default_rng(42)
        repaired = cs.repair(mask, rng)

        assert repaired.sum() == k_before


# =====================================================================
# Phase 3 Task 4: KL floor formula (Algorithm 1)
# =====================================================================

class TestKLFloorFormula:
    """Verify KL_min(k) formula and lazy-heap correctness."""

    def test_kl_floor_decreases_with_k(self):
        """KL_min(k) should decrease (or stay flat) as k increases."""
        pi = np.array([0.4, 0.3, 0.2, 0.1])
        group_sizes = np.array([100, 80, 60, 40])

        kl_prev = float("inf")
        for k in [10, 20, 50, 100, 150]:
            kl_min, _ = min_achievable_geo_kl_bounded(
                pi, group_sizes, k, alpha_geo=1.0, min_one_per_group=True,
            )
            assert kl_min <= kl_prev + 1e-12
            kl_prev = kl_min

    def test_kl_floor_matches_closed_form(self):
        """KL_min(k) = C_pi + log(k + alphaG) - Sigma pi_g log(c*_g + alpha)."""
        pi = np.array([0.5, 0.3, 0.2])
        group_sizes = np.array([50, 30, 20])
        k = 30
        alpha = 1.0
        G = len(pi)

        kl_min, counts = min_achievable_geo_kl_bounded(
            pi, group_sizes, k, alpha_geo=alpha, min_one_per_group=True,
        )

        # C_pi = Sigma pi_g log pi_g (negative entropy)
        C_pi = sum(pi[g] * np.log(pi[g]) for g in range(G) if pi[g] > 0)
        expected = (
            C_pi
            + np.log(k + alpha * G)
            - sum(pi[g] * np.log(counts[g] + alpha) for g in range(G) if pi[g] > 0)
        )

        assert np.isclose(kl_min, expected, rtol=1e-10)

    def test_lower_bound_ell_g_equals_1(self):
        """When k >= G, ell_g = 1 for all g with pi_g > 0."""
        pi = np.array([0.3, 0.3, 0.2, 0.2])
        group_sizes = np.array([20, 20, 15, 15])
        k = 20  # >= G = 4

        _, counts = min_achievable_geo_kl_bounded(
            pi, group_sizes, k, alpha_geo=1.0, min_one_per_group=True,
        )

        # All groups with pi_g > 0 must have c*_g >= 1
        for g in range(len(pi)):
            if pi[g] > 0:
                assert counts[g] >= 1, f"Group {g} has count {counts[g]} < 1"

    def test_greedy_beats_proportional_allocation(self):
        """KL-optimal counts should achieve <= KL of proportional allocation."""
        pi = np.array([0.5, 0.3, 0.15, 0.05])
        group_sizes = np.array([50, 30, 15, 5])
        k = 20
        alpha = 1.0

        kl_opt, counts_opt = min_achievable_geo_kl_bounded(
            pi, group_sizes, k, alpha_geo=alpha, min_one_per_group=True,
        )

        # Proportional (largest remainder)
        from coreset_selection.geo.kl import proportional_allocation
        counts_prop = proportional_allocation(pi, k, group_sizes)
        # Ensure lower bounds
        counts_prop = np.maximum(counts_prop, 1)
        # Adjust sum
        while counts_prop.sum() > k:
            g = np.argmax(counts_prop - 1)
            counts_prop[g] -= 1
        while counts_prop.sum() < k:
            g = np.argmin(counts_prop)
            if counts_prop[g] < group_sizes[g]:
                counts_prop[g] += 1

        kl_prop = kl_pi_hat_from_counts(pi, counts_prop, k, alpha)

        assert kl_opt <= kl_prop + 1e-10


# =====================================================================
# Constraint-domination ordering (Phase 3 Task 3 verification)
# =====================================================================

class TestConstraintDominatedSort:
    """Verify constraint-domination sorting from nsga2_internal."""

    def test_feasible_dominates_infeasible(self):
        from coreset_selection.optimization.nsga2_internal import _constraint_dominated_sort

        # 4 solutions: 2 feasible, 2 infeasible
        F = np.array([
            [1.0, 2.0],   # sol 0: feasible, dominated
            [0.5, 0.5],   # sol 1: feasible, non-dominated
            [0.1, 0.1],   # sol 2: infeasible (best objectives but violated)
            [0.3, 0.3],   # sol 3: infeasible
        ])
        CV = np.array([0.0, 0.0, 0.5, 0.3])

        fronts, rank = _constraint_dominated_sort(F, CV)

        # Feasible solutions should have lower rank than infeasible
        assert rank[0] < rank[2]
        assert rank[0] < rank[3]
        assert rank[1] < rank[2]
        assert rank[1] < rank[3]

    def test_infeasible_ranked_by_violation(self):
        from coreset_selection.optimization.nsga2_internal import _constraint_dominated_sort

        F = np.array([
            [0.1, 0.1],   # infeasible, CV=0.5
            [0.2, 0.2],   # infeasible, CV=0.1 (less violated)
        ])
        CV = np.array([0.5, 0.1])

        fronts, rank = _constraint_dominated_sort(F, CV)

        # Less violated should have lower rank
        assert rank[1] < rank[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
