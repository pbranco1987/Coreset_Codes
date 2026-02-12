"""Tests for incremental evaluation modules.

Validates:
- downstream_metrics: tail errors, per-state metrics, aggregates
- method_comparison: effect isolation, rank tables, dominance
- enhanced_evaluator: combined metrics pipeline
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.downstream_metrics import (
    tail_absolute_errors,
    per_state_downstream_metrics,
    aggregate_group_metrics,
    full_downstream_evaluation,
    multitarget_downstream_evaluation,
)
from evaluation.method_comparison import (
    effect_isolation_table,
    rank_table,
    stability_comparison,
    comprehensive_comparison,
    pairwise_dominance_matrix,
)


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def regression_data():
    rng = np.random.default_rng(42)
    n = 500
    y_true = rng.standard_normal(n)
    y_pred = y_true + rng.standard_normal(n) * 0.3
    states = rng.choice(5, size=n)
    return y_true, y_pred, states


@pytest.fixture
def multitarget_data():
    rng = np.random.default_rng(42)
    n = 500
    T = 2
    y_true = rng.standard_normal((n, T))
    y_pred = y_true + rng.standard_normal((n, T)) * 0.3
    states = rng.choice(5, size=n)
    return y_true, y_pred, states


@pytest.fixture
def method_results():
    """Simulated per-method metric dicts."""
    return {
        "KKN": {
            "overall_rmse_4G": 0.35, "macro_rmse_4G": 0.38,
            "worst_group_rmse_4G": 0.55, "abs_err_p90_4G": 0.40,
            "overall_r2_4G": 0.82, "macro_r2_4G": 0.78,
            "nystrom_error": 0.15,
        },
        "SKKN": {
            "overall_rmse_4G": 0.32, "macro_rmse_4G": 0.33,
            "worst_group_rmse_4G": 0.42, "abs_err_p90_4G": 0.36,
            "overall_r2_4G": 0.85, "macro_r2_4G": 0.83,
            "nystrom_error": 0.17,  # slightly worse Frobenius
        },
        "Pareto": {
            "overall_rmse_4G": 0.30, "macro_rmse_4G": 0.31,
            "worst_group_rmse_4G": 0.38, "abs_err_p90_4G": 0.33,
            "overall_r2_4G": 0.87, "macro_r2_4G": 0.86,
            "nystrom_error": 0.16,
        },
    }


# =====================================================================
# Test downstream_metrics
# =====================================================================

class TestTailErrors:
    def test_quantiles_ordering(self, regression_data):
        y_true, y_pred, _ = regression_data
        result = tail_absolute_errors(y_true, y_pred)
        assert result["abs_err_p90"] <= result["abs_err_p95"]
        assert result["abs_err_p95"] <= result["abs_err_p99"]
        assert result["abs_err_p99"] <= result["abs_err_max"]

    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        result = tail_absolute_errors(y, y)
        assert result["abs_err_p95"] == pytest.approx(0.0)


class TestPerStateMetrics:
    def test_all_states_present(self, regression_data):
        y_true, y_pred, states = regression_data
        result = per_state_downstream_metrics(y_true, y_pred, states)
        assert len(result) == len(np.unique(states))

    def test_per_state_has_all_keys(self, regression_data):
        y_true, y_pred, states = regression_data
        result = per_state_downstream_metrics(y_true, y_pred, states)
        for state_metrics in result.values():
            assert "rmse" in state_metrics
            assert "mae" in state_metrics
            assert "r2" in state_metrics
            assert "n" in state_metrics

    def test_positive_rmse(self, regression_data):
        y_true, y_pred, states = regression_data
        result = per_state_downstream_metrics(y_true, y_pred, states)
        for state_metrics in result.values():
            assert state_metrics["rmse"] >= 0
            assert state_metrics["mae"] >= 0


class TestAggregateGroupMetrics:
    def test_worst_geq_macro(self, regression_data):
        y_true, y_pred, states = regression_data
        per_state = per_state_downstream_metrics(y_true, y_pred, states)
        agg = aggregate_group_metrics(per_state)
        assert agg["worst_group_rmse"] >= agg["macro_rmse"]
        assert agg["best_group_rmse"] <= agg["macro_rmse"]

    def test_n_groups(self, regression_data):
        y_true, y_pred, states = regression_data
        per_state = per_state_downstream_metrics(y_true, y_pred, states)
        agg = aggregate_group_metrics(per_state)
        assert agg["n_groups_evaluated"] == len(np.unique(states))


class TestFullDownstreamEvaluation:
    def test_with_states(self, regression_data):
        y_true, y_pred, states = regression_data
        result = full_downstream_evaluation(y_true, y_pred, states, "_4G")
        assert "overall_rmse_4G" in result
        assert "macro_rmse_4G" in result
        assert "worst_group_rmse_4G" in result
        assert "abs_err_p90_4G" in result
        assert "abs_err_p95_4G" in result

    def test_without_states(self, regression_data):
        y_true, y_pred, _ = regression_data
        result = full_downstream_evaluation(y_true, y_pred, target_suffix="_4G")
        assert "overall_rmse_4G" in result
        assert "macro_rmse_4G" not in result  # no states → no macro


class TestMultitarget:
    def test_both_targets(self, multitarget_data):
        y_true, y_pred, states = multitarget_data
        result = multitarget_downstream_evaluation(
            y_true, y_pred, states, ["_4G", "_5G"],
        )
        assert "overall_rmse_4G" in result
        assert "overall_rmse_5G" in result
        assert "mean_macro_rmse" in result


# =====================================================================
# Test method_comparison
# =====================================================================

class TestEffectIsolation:
    def test_decomposition_adds_up(self, method_results):
        metrics = ["overall_rmse_4G", "macro_rmse_4G"]
        iso = effect_isolation_table(method_results, metrics)
        for m, row in iso.items():
            dt = row["delta_total"]
            dc = row["delta_constraint"]
            do = row["delta_objective"]
            assert dt == pytest.approx(dc + do, abs=1e-10)

    def test_positive_improvement(self, method_results):
        """NSGA-II should be better than KKN on RMSE metrics."""
        metrics = ["overall_rmse_4G"]
        iso = effect_isolation_table(method_results, metrics)
        assert iso["overall_rmse_4G"]["delta_total"] > 0  # KKN > Pareto


class TestRankTable:
    def test_ranks_are_valid(self, method_results):
        metrics = ["overall_rmse_4G", "macro_rmse_4G"]
        ranks = rank_table(method_results, metrics, lower_is_better=True)
        for meth, r in ranks.items():
            assert 1 <= r["rank_overall_rmse_4G"] <= len(method_results)
            assert 1 <= r["rank_macro_rmse_4G"] <= len(method_results)

    def test_avg_rank_computed(self, method_results):
        metrics = ["overall_rmse_4G"]
        ranks = rank_table(method_results, metrics)
        for meth, r in ranks.items():
            assert "avg_rank" in r
            assert np.isfinite(r["avg_rank"])


class TestPairwiseDominance:
    def test_self_never_dominates(self, method_results):
        metrics = ["overall_rmse_4G", "macro_rmse_4G"]
        dom = pairwise_dominance_matrix(method_results, metrics)
        for meth in method_results:
            assert dom[meth][meth] is False


class TestStabilityComparison:
    def test_mean_and_std(self):
        reps = {
            "A": [{"rmse": 0.3}, {"rmse": 0.35}, {"rmse": 0.32}],
            "B": [{"rmse": 0.4}, {"rmse": 0.38}],
        }
        result = stability_comparison(reps, ["rmse"])
        assert "mean_rmse" in result["A"]
        assert "std_rmse" in result["A"]
        assert result["A"]["n_reps_rmse"] == 3
        assert result["B"]["n_reps_rmse"] == 2


class TestComprehensiveComparison:
    def test_returns_required_keys(self, method_results):
        report = comprehensive_comparison(method_results)
        assert "summary_avg_rank" in report
        assert "Pareto" in report["summary_avg_rank"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
