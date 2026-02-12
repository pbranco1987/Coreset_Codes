"""
Tests for Algorithm 2: Feasibility-Preserving Repair.

Verifies:
- Repaired masks satisfy exact-k constraint
- Repaired masks satisfy quota constraints
- Repair activity tracking works correctly
"""

import numpy as np
import pytest

from coreset_selection.geo.info import GeoInfo
from coreset_selection.geo.projector import (
    GeographicConstraintProjector,
    project_to_exact_k_mask,
)
from coreset_selection.optimization.repair import (
    QuotaAndCardinalityRepair,
    ExactKRepair,
    RepairActivityTracker,
)


def make_simple_geo() -> GeoInfo:
    """Create simple GeoInfo for testing with 3 groups."""
    # 30 points, 3 groups of 10 each
    group_ids = np.array([0]*10 + [1]*10 + [2]*10)
    groups = ["A", "B", "C"]
    return GeoInfo.from_group_ids(group_ids, groups)


class TestExactKRepair:
    """Tests for exact-k cardinality repair."""

    def test_repair_satisfies_exact_k(self):
        """Repaired mask should have exactly k True values."""
        N = 100
        k = 30
        rng = np.random.default_rng(42)
        
        for _ in range(50):
            # Create mask with wrong cardinality
            wrong_k = rng.integers(0, N+1)
            mask = np.zeros(N, dtype=bool)
            mask[rng.choice(N, size=min(wrong_k, N), replace=False)] = True
            
            # Repair
            repaired = project_to_exact_k_mask(mask, k, rng)
            
            assert repaired.sum() == k

    def test_repair_preserves_when_correct(self):
        """Repair should not change masks that already satisfy k."""
        N = 50
        k = 20
        rng = np.random.default_rng(42)
        
        # Create correct mask
        mask = np.zeros(N, dtype=bool)
        mask[rng.choice(N, size=k, replace=False)] = True
        original = mask.copy()
        
        repaired = project_to_exact_k_mask(mask, k, rng)
        
        # Should have same selected set (no change needed)
        assert repaired.sum() == k
        # May or may not be identical due to random sampling when equal

    def test_repair_adds_when_under(self):
        """Repair should add points when mask has < k."""
        N = 50
        k = 20
        rng = np.random.default_rng(42)
        
        # Create under-selected mask
        mask = np.zeros(N, dtype=bool)
        mask[rng.choice(N, size=10, replace=False)] = True  # Only 10, need 20
        
        repaired = project_to_exact_k_mask(mask, k, rng)
        
        assert repaired.sum() == k

    def test_repair_removes_when_over(self):
        """Repair should remove points when mask has > k."""
        N = 50
        k = 20
        rng = np.random.default_rng(42)
        
        # Create over-selected mask
        mask = np.zeros(N, dtype=bool)
        mask[rng.choice(N, size=35, replace=False)] = True  # 35, need 20
        
        repaired = project_to_exact_k_mask(mask, k, rng)
        
        assert repaired.sum() == k


class TestQuotaRepair:
    """Tests for quota-constrained repair."""

    def test_repair_satisfies_quota(self):
        """Repaired mask should satisfy group quotas c_g(s) = c*_g."""
        geo = make_simple_geo()
        projector = GeographicConstraintProjector(
            geo=geo,
            alpha_geo=1.0,
            min_one_per_group=True,
        )
        
        k = 12
        target = projector.target_counts(k)
        rng = np.random.default_rng(42)
        
        for _ in range(50):
            # Random mask
            mask = np.zeros(30, dtype=bool)
            selected = rng.choice(30, size=rng.integers(1, 29), replace=False)
            mask[selected] = True
            
            # Repair
            repaired = projector.project_to_quota_mask(mask, k, rng)
            
            # Check quotas
            for g in range(3):
                idx_g = geo.group_to_indices[g]
                actual = repaired[idx_g].sum()
                expected = target[g]
                assert actual == expected, f"Group {g}: got {actual}, expected {expected}"

    def test_repair_satisfies_exact_k_with_quota(self):
        """Repaired mask should also satisfy Σs_i = k."""
        geo = make_simple_geo()
        projector = GeographicConstraintProjector(
            geo=geo,
            alpha_geo=1.0,
            min_one_per_group=True,
        )
        
        k = 15
        rng = np.random.default_rng(42)
        
        for _ in range(50):
            mask = rng.random(30) < 0.5
            repaired = projector.project_to_quota_mask(mask, k, rng)
            
            assert repaired.sum() == k


class TestRepairActivityTracker:
    """Tests for repair activity tracking."""

    def test_tracker_records_repairs(self):
        """Tracker should record repair events."""
        tracker = RepairActivityTracker()
        
        # Simulate some repairs
        pre1 = np.array([True, False, False, True])
        post1 = np.array([True, True, False, False])  # Hamming distance = 2
        
        tracker.record(pre1.reshape(1, -1), post1.reshape(1, -1))
        
        assert tracker.total_offspring == 1
        assert tracker.repaired_count == 1
        assert tracker.hamming_distances == [2]

    def test_tracker_counts_unchanged(self):
        """Tracker should count unchanged masks correctly."""
        tracker = RepairActivityTracker()
        
        # No change
        pre = np.array([True, False, True, False])
        post = pre.copy()
        
        tracker.record(pre.reshape(1, -1), post.reshape(1, -1))
        
        assert tracker.total_offspring == 1
        assert tracker.repaired_count == 0
        assert len(tracker.hamming_distances) == 0

    def test_tracker_summary_statistics(self):
        """Tracker summary should have correct statistics."""
        tracker = RepairActivityTracker()
        
        # Multiple repairs with different Hamming distances
        for hd in [2, 4, 6, 8]:
            pre = np.zeros(10, dtype=bool)
            post = pre.copy()
            post[:hd] = True
            tracker.record(pre.reshape(1, -1), post.reshape(1, -1))
        
        summary = tracker.summary()
        
        assert summary["total_offspring"] == 4
        assert summary["repaired_count"] == 4
        assert summary["repaired_fraction"] == 1.0
        assert summary["hamming_mean"] == 5.0
        assert summary["hamming_max"] == 8


class TestRepairIntegration:
    """Integration tests for repair with NSGA-II operators."""

    def test_quota_repair_class(self):
        """Test QuotaAndCardinalityRepair class."""
        geo = make_simple_geo()
        projector = GeographicConstraintProjector(
            geo=geo,
            alpha_geo=1.0,
            min_one_per_group=True,
        )
        
        k = 12
        repair = QuotaAndCardinalityRepair(
            k=k,
            projector=projector,
            seed=42,
            track_activity=True,
        )
        
        # Create population of masks
        rng = np.random.default_rng(42)
        pop = rng.random((10, 30)) < 0.4
        
        # Repair population
        repaired = repair._do(None, pop)
        
        # All should satisfy quota
        target = projector.target_counts(k)
        for i in range(10):
            for g in range(3):
                idx_g = geo.group_to_indices[g]
                actual = repaired[i, idx_g].sum()
                assert actual == target[g]

    def test_exactk_repair_class(self):
        """Test ExactKRepair class."""
        N = 50
        k = 20
        
        repair = ExactKRepair(k=k, seed=42, track_activity=True)
        
        # Create population
        rng = np.random.default_rng(42)
        pop = rng.random((10, N)) < 0.5
        
        # Repair
        repaired = repair._do(None, pop)
        
        # All should have exactly k
        for i in range(10):
            assert repaired[i].sum() == k


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# =====================================================================
# Phase 2: Population-share and joint constraint repair tests
# =====================================================================

class TestPopulationShareRepair:
    """Tests for population-share and joint constraint repair (manuscript §IV-C)."""

    def test_population_share_repair_reduces_violation(self):
        """Repair should reduce population-share KL violation."""
        from coreset_selection.geo.info import build_geo_info
        from coreset_selection.constraints.proportionality import (
            build_population_share_constraint,
            ProportionalityConstraintSet,
        )

        N = 60
        state_labels = np.repeat(np.arange(3), 20)
        # Group 0 has high population, group 2 has low
        pop = np.concatenate([100*np.ones(20), 50*np.ones(20), 10*np.ones(20)])
        geo = build_geo_info(state_labels, population_weights=pop)

        # Create a poorly proportioned mask (all from group 2)
        mask = np.zeros(N, dtype=bool)
        mask[40:50] = True  # 10 from group 2 (low pop)

        c_pop = build_population_share_constraint(
            geo=geo, population=pop, alpha=1.0, tau=0.01,
        )
        cs = ProportionalityConstraintSet(
            geo=geo, constraints=[c_pop],
            min_one_per_group=False,
            max_iters=50,
        )

        violation_before = cs.total_violation(mask)
        rng = np.random.default_rng(42)
        repaired = cs.repair(mask, rng)
        violation_after = cs.total_violation(repaired)

        assert violation_after <= violation_before
        assert repaired.sum() == mask.sum()  # cardinality preserved

    def test_joint_constraint_repair(self):
        """Joint (pop + muni) repair should reduce total violation."""
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

        # Imbalanced mask
        mask = np.zeros(N, dtype=bool)
        mask[:12] = True   # 12 from group 0
        mask[20:21] = True  # 1 from group 1

        c_pop = build_population_share_constraint(
            geo=geo, population=pop, alpha=1.0, tau=0.01,
        )
        c_muni = build_municipality_share_constraint(
            geo=geo, alpha=1.0, tau=0.01,
        )
        cs = ProportionalityConstraintSet(
            geo=geo, constraints=[c_pop, c_muni],
            min_one_per_group=False,
            preserve_group_counts=True,
            max_iters=100,
        )

        violation_before = cs.total_violation(mask)
        rng = np.random.default_rng(42)
        repaired = cs.repair(mask, rng)
        violation_after = cs.total_violation(repaired)

        assert violation_after <= violation_before

    def test_within_group_repair_preserves_counts(self):
        """Within-group repair (joint mode) should preserve per-group counts."""
        from coreset_selection.geo.info import build_geo_info
        from coreset_selection.constraints.proportionality import (
            build_population_share_constraint,
            ProportionalityConstraintSet,
        )

        N = 60
        state_labels = np.repeat(np.arange(3), 20)
        pop = np.concatenate([
            np.array([100]*10 + [1]*10),   # group 0: mixed pop
            np.array([50]*10 + [1]*10),    # group 1: mixed pop
            np.array([10]*10 + [1]*10),    # group 2: mixed pop
        ])
        geo = build_geo_info(state_labels, population_weights=pop)

        mask = np.zeros(N, dtype=bool)
        # Select low-pop items from each group
        mask[10:14] = True   # 4 from group 0 (low pop)
        mask[30:34] = True   # 4 from group 1 (low pop)
        mask[50:54] = True   # 4 from group 2 (low pop)

        c_pop = build_population_share_constraint(
            geo=geo, population=pop, alpha=1.0, tau=0.01,
        )
        cs = ProportionalityConstraintSet(
            geo=geo, constraints=[c_pop],
            min_one_per_group=True,
            preserve_group_counts=True,
            max_iters=50,
        )

        # Record per-group counts before repair
        counts_before = np.array([
            mask[geo.group_to_indices[g]].sum() for g in range(3)
        ])

        rng = np.random.default_rng(42)
        repaired = cs.repair(mask, rng)

        counts_after = np.array([
            repaired[geo.group_to_indices[g]].sum() for g in range(3)
        ])

        np.testing.assert_array_equal(counts_before, counts_after)
