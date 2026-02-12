"""
Evaluation Protocol Tests — Geographic evaluation.

Verifies geographic diagnostics and state-conditioned KPI stability:

7. Geographic diagnostics:  KL, L1, maxdev (municipality-share + population-share)
8. State-conditioned KPI stability: drift and Kendall tau

Split from tests/test_evaluation_protocol.py.
"""

from __future__ import annotations

import numpy as np
import pytest


# ===================================================================
# Test: Geographic diagnostics
# ===================================================================

class TestGeoDiagnosticsProtocol:
    """Verify geographic KL, L1, maxdev computations."""

    def test_kl_nonneg(self):
        """KL divergence should be nonneg."""
        from coreset_selection.geo.info import GeoInfo
        from coreset_selection.evaluation.geo_diagnostics import geo_diagnostics

        geo = GeoInfo.from_group_ids(np.array([0, 0, 0, 1, 1]))
        S_idx = np.array([0, 3])
        result = geo_diagnostics(geo, S_idx, k=2)
        assert result["geo_kl"] >= 0.0

    def test_perfect_proportionality(self):
        """When subset exactly matches population proportions, KL ~ 0."""
        from coreset_selection.geo.info import GeoInfo
        from coreset_selection.evaluation.geo_diagnostics import geo_diagnostics

        # 4 points per group, select 2 per group
        group_ids = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        geo = GeoInfo.from_group_ids(group_ids)
        S_idx = np.array([0, 1, 4, 5])  # 2 from each group
        result = geo_diagnostics(geo, S_idx, k=4, alpha=1.0)

        # L1 should be exactly 0 (perfect proportionality)
        assert result["geo_l1"] < 1e-10
        assert result["geo_maxdev"] < 1e-10
        # KL should be near zero (not exactly due to Laplace smoothing mismatch)
        assert result["geo_kl"] < 0.1

    def test_l1_formula(self):
        """Hand-verify L1 = sum(|pi_g - pi_hat_g|)."""
        from coreset_selection.geo.info import GeoInfo
        from coreset_selection.evaluation.geo_diagnostics import geo_diagnostics

        # 6 points: [0,0,0,0, 1,1] => pi = [4/6, 2/6] = [2/3, 1/3]
        # Select [0, 4] => pi_hat = [1/2, 1/2]
        # L1 = |2/3 - 1/2| + |1/3 - 1/2| = 1/6 + 1/6 = 1/3
        group_ids = np.array([0, 0, 0, 0, 1, 1])
        geo = GeoInfo.from_group_ids(group_ids)
        S_idx = np.array([0, 4])
        result = geo_diagnostics(geo, S_idx, k=2)

        expected_l1 = abs(4 / 6 - 1 / 2) + abs(2 / 6 - 1 / 2)
        np.testing.assert_allclose(result["geo_l1"], expected_l1, atol=1e-10)

    def test_maxdev_formula(self):
        """Hand-verify maxdev = max_g |pi_g - pi_hat_g|."""
        from coreset_selection.geo.info import GeoInfo
        from coreset_selection.evaluation.geo_diagnostics import geo_diagnostics

        group_ids = np.array([0, 0, 0, 0, 1, 1])
        geo = GeoInfo.from_group_ids(group_ids)
        S_idx = np.array([0, 4])
        result = geo_diagnostics(geo, S_idx, k=2)

        expected_maxdev = max(abs(4 / 6 - 1 / 2), abs(2 / 6 - 1 / 2))
        np.testing.assert_allclose(result["geo_maxdev"], expected_maxdev, atol=1e-10)


# ===================================================================
# Test: KPI stability
# ===================================================================

class TestKPIStabilityProtocol:
    """Verify state-conditioned KPI stability metrics."""

    def test_drift_formula(self):
        r"""Hand-verify KPI drift.

        For target t, state g:
          full_mean_g = mean(y[i, t] for i in state g)
          sub_mean_g  = mean(y[i, t] for i in S and state g)
          drift_g = |full_mean_g - sub_mean_g|
          max_drift = max_g drift_g
          avg_drift = mean_g drift_g
        """
        from coreset_selection.evaluation.kpi_stability import state_kpi_stability

        # 6 points, 2 groups, 1 target
        y = np.array([[1.0], [2.0], [3.0], [10.0], [20.0], [30.0]])
        state_labels = np.array([0, 0, 0, 1, 1, 1])
        S_idx = np.array([0, 3])  # select one from each group

        result = state_kpi_stability(
            y=y, state_labels=state_labels, S_idx=S_idx,
        )

        # Group 0: full_mean = (1+2+3)/3 = 2.0, sub_mean = 1.0/1 = 1.0, drift = 1.0
        # Group 1: full_mean = (10+20+30)/3 = 20.0, sub_mean = 10.0/1 = 10.0, drift = 10.0
        expected_max = 10.0
        expected_avg = (1.0 + 10.0) / 2.0

        # Keys depend on target naming: 1 col -> suffix ""
        assert "max_kpi_drift" in result
        np.testing.assert_allclose(result["max_kpi_drift"], expected_max, atol=1e-10)
        assert "avg_kpi_drift" in result
        np.testing.assert_allclose(result["avg_kpi_drift"], expected_avg, atol=1e-10)

    def test_kendall_tau_perfect(self):
        """Perfect state-mean preservation should yield tau = 1."""
        from coreset_selection.evaluation.kpi_stability import state_kpi_stability

        # 4 groups, monotonically increasing means.
        # Select the median element from each group to preserve ordering.
        y = np.array([
            [1.0], [2.0], [3.0],    # group 0, mean=2
            [4.0], [5.0], [6.0],    # group 1, mean=5
            [7.0], [8.0], [9.0],    # group 2, mean=8
            [10.0], [11.0], [12.0], # group 3, mean=11
        ])
        state_labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
        # Select medians: 2.0, 5.0, 8.0, 11.0 -> ranking preserved
        S_idx = np.array([1, 4, 7, 10])

        result = state_kpi_stability(
            y=y, state_labels=state_labels, S_idx=S_idx,
        )
        assert "kendall_tau" in result
        assert result["kendall_tau"] > 0.9
