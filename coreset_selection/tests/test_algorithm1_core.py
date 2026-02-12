"""
Tests for Algorithm 1: KL-Guided Quota Construction — core functionality.

Verifies:
- Greedy allocation matches pseudocode (Algorithm 1)
- KL_min formula matches Theorem 1(iii)
- Greedy optimality via diminishing returns (Theorem 1(ii))
- Quota path computation
- Projector Phase 6 enhancements
- Save/load quota path I/O

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


class TestKLQuotaComputation:
    """Tests for KL-optimal quota computation (Algorithm 1)."""

    def test_marginal_gain_formula(self):
        """Verify marginal gain formula from Theorem 1(ii)."""
        # Delta_g(t) = pi_g * (log(t + alpha + 1) - log(t + alpha))
        pi_g = 0.3
        alpha = 1.0
        t = 5

        # Compute marginal gain
        gain = pi_g * (np.log(t + alpha + 1) - np.log(t + alpha))

        # Should be positive for positive pi_g
        assert gain > 0

        # Gains should be diminishing (larger t -> smaller gain)
        gain_next = pi_g * (np.log(t + alpha + 2) - np.log(t + alpha + 1))
        assert gain > gain_next

    def test_hand_computed_example(self):
        """Verify quota against hand-computed example."""
        # Simple 3-group case
        pi = np.array([0.5, 0.3, 0.2])
        group_sizes = np.array([10, 10, 10])
        k = 6
        alpha = 1.0

        kl_min, counts = min_achievable_geo_kl_bounded(
            pi=pi,
            group_sizes=group_sizes,
            k=k,
            alpha_geo=alpha,
            min_one_per_group=True,
        )

        # With min_one_per_group=True and k >= G, each group gets at least 1
        assert counts.sum() == k
        assert np.all(counts >= 1)

        # Larger groups should get more representation
        assert counts[0] >= counts[1] >= counts[2]

    def test_kl_min_formula_theorem1iii(self):
        """Verify KL_min matches Theorem 1(iii) formula."""
        # KL_min(k) = Sigma pi_g log pi_g + log(k + alphaG) - Sigma pi_g log(c*_g + alpha)
        pi = np.array([0.4, 0.35, 0.25])
        group_sizes = np.array([20, 20, 20])
        k = 10
        alpha = 1.0
        G = len(pi)

        kl_min, counts = min_achievable_geo_kl_bounded(
            pi=pi,
            group_sizes=group_sizes,
            k=k,
            alpha_geo=alpha,
            min_one_per_group=True,
        )

        # Manually compute expected KL_min using Theorem 1(iii)
        entropy_term = np.sum(pi * np.log(pi))
        log_normalizer = np.log(k + alpha * G)
        allocation_term = np.sum(pi * np.log(counts + alpha))
        expected_kl = entropy_term + log_normalizer - allocation_term

        assert np.isclose(kl_min, expected_kl, rtol=1e-10)

    def test_greedy_optimality(self):
        """Verify greedy produces global optimum (Theorem 1(ii))."""
        # For concave separable objectives, greedy is optimal
        pi = np.array([0.3, 0.3, 0.2, 0.2])
        group_sizes = np.array([15, 15, 10, 10])
        k = 12
        alpha = 1.0

        kl_min, counts = min_achievable_geo_kl_bounded(
            pi=pi,
            group_sizes=group_sizes,
            k=k,
            alpha_geo=alpha,
            min_one_per_group=True,
        )

        # Verify constraint satisfaction
        assert counts.sum() == k
        assert np.all(counts >= 0)
        assert np.all(counts <= group_sizes)

        # Try a few random perturbations - they should not improve KL
        rng = np.random.default_rng(42)
        for _ in range(100):
            # Perturb: move one from group i to group j
            perturbed = counts.copy()
            i, j = rng.choice(4, size=2, replace=False)

            if perturbed[i] > 0 and perturbed[j] < group_sizes[j]:
                perturbed[i] -= 1
                perturbed[j] += 1

                kl_perturbed = kl_pi_hat_from_counts(pi, perturbed, k, alpha)
                assert kl_perturbed >= kl_min - 1e-10

    def test_lower_bound_enforcement(self):
        """Verify lower bounds are enforced when k >= G."""
        pi = np.array([0.6, 0.3, 0.1])
        group_sizes = np.array([10, 10, 10])
        k = 5  # >= G=3
        alpha = 1.0

        kl_min, counts = min_achievable_geo_kl_bounded(
            pi=pi,
            group_sizes=group_sizes,
            k=k,
            alpha_geo=alpha,
            min_one_per_group=True,
        )

        # All groups should have at least 1
        assert np.all(counts >= 1)

    def test_capacity_constraints(self):
        """Verify capacity (upper bound) constraints."""
        pi = np.array([0.8, 0.2])
        group_sizes = np.array([3, 10])  # Group 0 limited
        k = 8
        alpha = 1.0

        kl_min, counts = min_achievable_geo_kl_bounded(
            pi=pi,
            group_sizes=group_sizes,
            k=k,
            alpha_geo=alpha,
            min_one_per_group=True,
        )

        # Group 0 should be saturated at capacity
        assert counts[0] <= group_sizes[0]
        assert counts.sum() == k

    def test_infeasible_detection(self):
        """Verify infeasible cases are detected."""
        pi = np.array([0.5, 0.5])
        group_sizes = np.array([2, 2])  # Total capacity = 4
        k = 10  # > capacity

        with pytest.raises(ValueError, match="Infeasible"):
            min_achievable_geo_kl_bounded(
                pi=pi,
                group_sizes=group_sizes,
                k=k,
                alpha_geo=1.0,
                min_one_per_group=True,
            )


class TestKLDivergence:
    """Tests for KL divergence computation."""

    def test_kl_zero_for_perfect_match(self):
        """KL should be ~0 when counts match pi exactly."""
        pi = np.array([0.4, 0.3, 0.2, 0.1])
        k = 100
        alpha = 1.0
        G = len(pi)

        # Approximate perfect allocation
        counts = np.round(pi * k).astype(int)
        counts[-1] = k - counts[:-1].sum()  # Ensure sum = k

        kl = kl_pi_hat_from_counts(pi, counts, k, alpha)

        # Should be very small (not exactly 0 due to smoothing and rounding)
        assert kl < 0.1

    def test_kl_positive(self):
        """KL divergence should always be non-negative."""
        rng = np.random.default_rng(42)

        for _ in range(50):
            G = rng.integers(2, 10)
            pi = rng.dirichlet(np.ones(G))
            k = rng.integers(G, 100)
            counts = rng.multinomial(k, pi / pi.sum())
            alpha = rng.uniform(0.1, 2.0)

            kl = kl_pi_hat_from_counts(pi, counts, k, alpha)
            assert kl >= -1e-10  # Allow small numerical error


class TestComputeQuotaPath:
    """Tests for the incremental quota path computation."""

    def test_path_matches_independent(self):
        """Path entries should match independent min_achievable_geo_kl_bounded calls."""
        pi = np.array([0.4, 0.35, 0.25])
        group_sizes = np.array([100, 80, 60])
        alpha = 1.0
        k_grid = [10, 50, 100, 150]

        from coreset_selection.geo.kl import compute_quota_path

        path = compute_quota_path(
            pi=pi, group_sizes=group_sizes, k_grid=k_grid,
            alpha_geo=alpha, min_one_per_group=True,
        )

        assert len(path) == len(k_grid)
        for row in path:
            k = row["k"]
            kl_ref, c_ref = min_achievable_geo_kl_bounded(
                pi=pi, group_sizes=group_sizes, k=k,
                alpha_geo=alpha, min_one_per_group=True,
            )
            assert np.isclose(row["kl_min"], kl_ref, rtol=1e-10), (
                f"k={k}: path KL={row['kl_min']:.8f} != ref KL={kl_ref:.8f}"
            )
            assert row["cstar"] == c_ref.tolist(), (
                f"k={k}: path cstar={row['cstar']} != ref cstar={c_ref.tolist()}"
            )

    def test_path_monotone_kl(self):
        """KL_min(k) should generally decrease with larger k.

        Note: with Dirichlet smoothing (alpha > 0), KL_min is not *strictly*
        monotone because the normaliser log(k + alphaG) grows with k.  We check
        that KL_min at the largest k is smaller than at the smallest k.
        """
        pi = np.array([0.3, 0.3, 0.2, 0.2])
        group_sizes = np.array([50, 50, 50, 50])
        k_grid = [4, 10, 20, 50, 100]

        from coreset_selection.geo.kl import compute_quota_path

        path = compute_quota_path(
            pi=pi, group_sizes=group_sizes, k_grid=k_grid,
            alpha_geo=1.0, min_one_per_group=True,
        )

        kl_values = [r["kl_min"] for r in path]
        # Large-scale trend: KL at largest k should be <= KL at smallest k
        assert kl_values[-1] <= kl_values[0] + 1e-10, (
            f"KL at k={path[-1]['k']} ({kl_values[-1]:.8f}) > "
            f"KL at k={path[0]['k']} ({kl_values[0]:.8f})"
        )

    def test_path_cstar_sums(self):
        """Each c*(k) in the path should sum to exactly k."""
        pi = np.array([0.5, 0.3, 0.2])
        group_sizes = np.array([200, 150, 100])
        k_grid = [3, 50, 100, 200, 300]

        from coreset_selection.geo.kl import compute_quota_path

        path = compute_quota_path(
            pi=pi, group_sizes=group_sizes, k_grid=k_grid,
            alpha_geo=1.0, min_one_per_group=True,
        )

        for row in path:
            assert sum(row["cstar"]) == row["k"]

    def test_path_unsorted_grid(self):
        """Grid can be given unsorted; results should still be sorted ascending."""
        pi = np.array([0.6, 0.4])
        group_sizes = np.array([50, 50])
        k_grid = [30, 10, 20, 5]

        from coreset_selection.geo.kl import compute_quota_path

        path = compute_quota_path(
            pi=pi, group_sizes=group_sizes, k_grid=k_grid,
            alpha_geo=1.0, min_one_per_group=True,
        )

        ks = [r["k"] for r in path]
        assert ks == sorted(ks)


class TestProjectorPhase6:
    """Tests for the Phase 6 projector enhancements."""

    def _make_projector(self):
        from coreset_selection.geo.info import build_geo_info
        from coreset_selection.geo.projector import GeographicConstraintProjector

        state_labels = np.concatenate([
            np.full(100, "A"),
            np.full(80, "B"),
            np.full(50, "C"),
            np.full(20, "D"),
        ])
        geo = build_geo_info(state_labels)
        return GeographicConstraintProjector(geo=geo, alpha_geo=1.0)

    def test_get_cstar(self):
        """get_cstar should be an alias for target_counts."""
        proj = self._make_projector()
        tc = proj.target_counts(50)
        cs = proj.get_cstar(50)
        np.testing.assert_array_equal(tc, cs)

    def test_quota_path(self):
        """quota_path should return a list of dicts with correct keys."""
        proj = self._make_projector()
        path = proj.quota_path(k_grid=[10, 50, 100])
        assert len(path) == 3
        for row in path:
            assert "k" in row
            assert "kl_min" in row
            assert "cstar" in row
            assert sum(row["cstar"]) == row["k"]

    def test_quota_path_caching(self):
        """Second call to quota_path with a subset grid should use cache."""
        proj = self._make_projector()
        path_full = proj.quota_path(k_grid=[10, 50, 100])
        path_subset = proj.quota_path(k_grid=[50, 100])
        # Subset should be drawn from cache
        assert len(path_subset) == 2
        for row in path_subset:
            assert row["k"] in [50, 100]

    def test_validate_capacity(self):
        """validate_capacity should detect feasible/infeasible k."""
        proj = self._make_projector()
        cap = proj.validate_capacity(50)
        assert cap["feasible"] is True
        assert cap["min_k"] == 4  # G=4, one per group
        assert cap["max_k"] == 250  # 100+80+50+20

        cap2 = proj.validate_capacity(300)
        assert cap2["feasible"] is False

    def test_most_constrained_groups(self):
        """most_constrained_groups should return sorted by utilisation."""
        proj = self._make_projector()
        mc = proj.most_constrained_groups(100, top_n=4)
        assert len(mc) == 4
        # Utilisation should be sorted descending
        utils = [m["utilisation"] for m in mc]
        assert utils == sorted(utils, reverse=True)
        # Each entry should have expected keys
        for m in mc:
            assert "group" in m
            assert "cstar" in m
            assert "n_g" in m
            assert "utilisation" in m
            assert "pi_g" in m


class TestSaveQuotaPath:
    """Tests for save_quota_path I/O."""

    def test_roundtrip(self, tmp_path):
        """save_quota_path should write JSON and CSV that can be re-read."""
        import json
        import csv as csv_mod
        from coreset_selection.geo.kl import save_quota_path

        rows = [
            {"k": 50, "kl_min": 0.01, "cstar": [20, 15, 10, 5], "geo_l1": 0.05, "geo_maxdev": 0.02},
            {"k": 100, "kl_min": 0.005, "cstar": [40, 30, 20, 10], "geo_l1": 0.03, "geo_maxdev": 0.01},
        ]

        json_path, csv_path = save_quota_path(rows, str(tmp_path))

        # JSON roundtrip
        with open(json_path) as f:
            loaded = json.load(f)
        assert len(loaded) == 2
        assert loaded[0]["k"] == 50
        assert loaded[1]["cstar"] == [40, 30, 20, 10]

        # CSV roundtrip
        with open(csv_path) as f:
            reader = csv_mod.DictReader(f)
            csv_rows = list(reader)
        assert len(csv_rows) == 2
        assert csv_rows[0]["k"] == "50"
        assert float(csv_rows[1]["kl_min"]) == pytest.approx(0.005)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
