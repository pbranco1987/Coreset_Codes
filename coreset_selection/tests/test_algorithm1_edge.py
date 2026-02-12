"""
Tests for Algorithm 1: KL-Guided Quota Construction — edge cases and stress tests.

Verifies:
- Lazy-heap optimality via brute-force exhaustive enumeration
- Weighted KL interface (manuscript Eq. 3-4)
- Constraint violation feasibility checks

Split from tests/test_algorithm1.py.
"""

import numpy as np
import pytest

from coreset_selection.geo.kl import (
    kl_pi_hat_from_counts,
    kl_optimal_integer_counts_bounded,
    min_achievable_geo_kl_bounded,
    proportional_allocation,
)


# =====================================================================
# Phase 6: Lazy-heap optimality and incremental quota path
# =====================================================================

class TestLazyHeapOptimality:
    """Verify the lazy-heap allocator matches brute-force on small cases."""

    def test_brute_force_g3(self):
        """Brute-force exhaustive check for G=3 (Milestone 2.5 acceptance criterion).

        Enumerate *all* feasible integer partitions of k into G=3 groups
        and verify the greedy allocation achieves the global minimum KL.
        """
        from itertools import product as iproduct

        pi = np.array([0.5, 0.3, 0.2])
        group_sizes = np.array([8, 6, 4])
        alpha = 1.0

        for k in range(3, 12):
            kl_min, counts_greedy = min_achievable_geo_kl_bounded(
                pi=pi, group_sizes=group_sizes, k=k,
                alpha_geo=alpha, min_one_per_group=True,
            )

            # Brute-force: enumerate all valid (c0, c1, c2) with sum = k
            best_kl = np.inf
            best_counts = None
            for c0 in range(1, min(group_sizes[0], k) + 1):
                for c1 in range(1, min(group_sizes[1], k - c0) + 1):
                    c2 = k - c0 - c1
                    if c2 < 1 or c2 > group_sizes[2]:
                        continue
                    c = np.array([c0, c1, c2])
                    kl = kl_pi_hat_from_counts(pi, c, k, alpha)
                    if kl < best_kl:
                        best_kl = kl
                        best_counts = c.copy()

            assert best_counts is not None, f"No feasible allocation at k={k}"
            assert np.isclose(kl_min, best_kl, rtol=1e-10), (
                f"k={k}: greedy KL={kl_min:.8f} != brute-force KL={best_kl:.8f}"
            )

    def test_brute_force_g3_no_min_one(self):
        """Brute-force with min_one_per_group=False."""
        from itertools import product as iproduct

        pi = np.array([0.6, 0.3, 0.1])
        group_sizes = np.array([10, 10, 10])
        alpha = 1.0

        for k in [2, 5, 8]:
            kl_min, counts_greedy = min_achievable_geo_kl_bounded(
                pi=pi, group_sizes=group_sizes, k=k,
                alpha_geo=alpha, min_one_per_group=False,
            )

            best_kl = np.inf
            for c0 in range(0, min(group_sizes[0], k) + 1):
                for c1 in range(0, min(group_sizes[1], k - c0) + 1):
                    c2 = k - c0 - c1
                    if c2 < 0 or c2 > group_sizes[2]:
                        continue
                    c = np.array([c0, c1, c2])
                    kl = kl_pi_hat_from_counts(pi, c, k, alpha)
                    if kl < best_kl:
                        best_kl = kl

            assert np.isclose(kl_min, best_kl, rtol=1e-10), (
                f"k={k}: greedy KL={kl_min:.8f} != brute-force KL={best_kl:.8f}"
            )

    def test_heap_matches_naive_random(self):
        """Random instances: heap-based result should match naive scan."""
        # We can't compare to old naive code directly, but we can verify
        # optimality properties.
        rng = np.random.default_rng(123)
        for _ in range(30):
            G = rng.integers(3, 15)
            pi = rng.dirichlet(np.ones(G))
            group_sizes = rng.integers(5, 50, size=G)
            k = rng.integers(G, min(int(group_sizes.sum()) - 1, G * 10))
            alpha = 1.0

            kl_min, counts = min_achievable_geo_kl_bounded(
                pi=pi, group_sizes=group_sizes, k=k,
                alpha_geo=alpha, min_one_per_group=True,
            )

            # Basic invariants
            assert counts.sum() == k
            assert np.all(counts >= 1)
            assert np.all(counts <= group_sizes)

            # No single swap should improve KL
            for _ in range(50):
                i, j = rng.choice(G, size=2, replace=False)
                c2 = counts.copy()
                if c2[i] > 1 and c2[j] < group_sizes[j]:
                    c2[i] -= 1
                    c2[j] += 1
                    kl2 = kl_pi_hat_from_counts(pi, c2, k, alpha)
                    assert kl2 >= kl_min - 1e-10


# =====================================================================
# Phase 2: Weighted KL interface (manuscript Eq. 3-4)
# =====================================================================

class TestWeightedKL:
    """Tests for the weighted proportionality KL interface."""

    def test_kl_weighted_matches_count_based(self):
        """kl_weighted_from_subset with w==1 should match kl_pi_hat_from_counts."""
        from coreset_selection.geo.info import build_geo_info
        from coreset_selection.geo.kl import kl_weighted_from_subset

        state_labels = np.array([0]*10 + [1]*8 + [2]*2)
        geo = build_geo_info(state_labels)
        N = len(state_labels)

        mask = np.zeros(N, dtype=bool)
        mask[:3] = True   # 3 from group 0
        mask[10:12] = True  # 2 from group 1
        mask[18] = True    # 1 from group 2

        weights = np.ones(N)
        alpha = 1.0

        kl_w = kl_weighted_from_subset(
            target_pi=geo.pi, weights=weights, group_ids=geo.group_ids,
            mask=mask, alpha=alpha,
        )

        # Count-based
        counts = np.array([3, 2, 1])
        k = 6
        kl_c = kl_pi_hat_from_counts(
            pi=geo.pi, counts=counts, k=k, alpha=alpha,
        )

        assert np.isclose(kl_w, kl_c, rtol=1e-10), f"weighted={kl_w}, count={kl_c}"

    def test_kl_weighted_population_share(self):
        """kl_weighted with population weights should differ from count-based."""
        from coreset_selection.geo.info import build_geo_info
        from coreset_selection.geo.kl import kl_weighted_from_subset

        state_labels = np.array([0]*5 + [1]*5)
        pop = np.array([100.0]*5 + [10.0]*5)  # group 0 much higher population
        geo = build_geo_info(state_labels, population_weights=pop)

        mask = np.zeros(10, dtype=bool)
        mask[0] = True   # 1 from group 0 (high pop)
        mask[5] = True   # 1 from group 1 (low pop)

        kl_pop = kl_weighted_from_subset(
            target_pi=geo.pi_pop, weights=pop, group_ids=geo.group_ids,
            mask=mask, alpha=1.0,
        )
        kl_count = kl_weighted_from_subset(
            target_pi=geo.pi, weights=np.ones(10), group_ids=geo.group_ids,
            mask=mask, alpha=1.0,
        )

        # They should be different because the target distributions differ
        assert kl_pop != kl_count

    def test_constraint_violations_feasible(self):
        """A well-proportioned subset should have zero violation."""
        from coreset_selection.geo.info import build_geo_info
        from coreset_selection.constraints.proportionality import (
            build_population_share_constraint,
            ProportionalityConstraintSet,
        )

        N = 100
        state_labels = np.repeat(np.arange(5), 20)
        pop = np.ones(N)
        geo = build_geo_info(state_labels, population_weights=pop)

        # Select exactly 4 from each group (perfectly proportional)
        mask = np.zeros(N, dtype=bool)
        for g in range(5):
            mask[g*20 : g*20 + 4] = True

        c = build_population_share_constraint(
            geo=geo, population=pop, alpha=1.0, tau=0.5,
        )
        cs = ProportionalityConstraintSet(
            geo=geo, constraints=[c], min_one_per_group=True,
        )

        assert cs.is_feasible(mask)
        assert cs.total_violation(mask) == 0.0

    def test_constraint_violations_infeasible(self):
        """A poorly proportioned subset should have positive violation."""
        from coreset_selection.geo.info import build_geo_info
        from coreset_selection.constraints.proportionality import (
            build_population_share_constraint,
            ProportionalityConstraintSet,
        )

        N = 100
        state_labels = np.repeat(np.arange(5), 20)
        pop = np.ones(N)
        geo = build_geo_info(state_labels, population_weights=pop)

        # Select all from group 0, none from others
        mask = np.zeros(N, dtype=bool)
        mask[:20] = True

        c = build_population_share_constraint(
            geo=geo, population=pop, alpha=1.0, tau=0.01,
        )
        cs = ProportionalityConstraintSet(
            geo=geo, constraints=[c], min_one_per_group=True,
        )

        assert cs.total_violation(mask) > 0.0

    def test_joint_constraint_set(self):
        """Joint (population + municipality) constraint set works."""
        from coreset_selection.geo.info import build_geo_info
        from coreset_selection.constraints.proportionality import (
            build_population_share_constraint,
            build_municipality_share_constraint,
            ProportionalityConstraintSet,
        )

        N = 60
        state_labels = np.repeat(np.arange(3), 20)
        pop = np.concatenate([100*np.ones(20), 50*np.ones(20), 10*np.ones(20)])
        geo = build_geo_info(state_labels, population_weights=pop)

        c_pop = build_population_share_constraint(
            geo=geo, population=pop, alpha=1.0, tau=0.5,
        )
        c_muni = build_municipality_share_constraint(
            geo=geo, alpha=1.0, tau=0.5,
        )
        cs = ProportionalityConstraintSet(
            geo=geo, constraints=[c_pop, c_muni],
            min_one_per_group=True,
            preserve_group_counts=True,
        )

        # Select 3 from each group (proportional in count-share)
        mask = np.zeros(N, dtype=bool)
        for g in range(3):
            mask[g*20 : g*20 + 3] = True

        vals = cs.values(mask)
        assert "population_share" in vals
        assert "municipality_share" in vals
        assert cs.total_violation(mask) >= 0.0
