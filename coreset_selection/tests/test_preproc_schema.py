"""
Preprocessing tests — schema validation, column detection, type inference,
constraint API consistency, and tau calibration.

Split from tests/test_preprocessing.py.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Phase 4.3: Target leakage prevention
# ---------------------------------------------------------------------------

class TestTargetLeakagePrevention:
    """Verify target columns are correctly detected and removable."""

    def test_detect_primary_targets(self):
        from coreset_selection.data.target_columns import detect_target_columns

        names = ["feat_1", "feat_2", "cov_area_4G", "cov_area_5G", "population"]
        detected = detect_target_columns(names)

        assert "cov_area_4G" in detected
        assert "cov_area_5G" in detected
        assert "feat_1" not in detected
        assert "population" not in detected

    def test_detect_internal_aliases(self):
        from coreset_selection.data.target_columns import detect_target_columns

        names = ["y_4G", "y_5G", "target", "some_feature"]
        detected = detect_target_columns(names)

        assert "y_4G" in detected
        assert "y_5G" in detected
        assert "target" in detected
        assert "some_feature" not in detected

    def test_detect_multitarget_columns(self):
        from coreset_selection.data.target_columns import detect_target_columns

        names = ["cov_hh_4G", "cov_res_all", "cov_area_4G_5G", "normal_feat"]
        detected = detect_target_columns(names)

        assert "cov_hh_4G" in detected
        assert "cov_res_all" in detected
        assert "cov_area_4G_5G" in detected
        assert "normal_feat" not in detected

    def test_remove_target_columns(self):
        from coreset_selection.data.target_columns import remove_target_columns

        X = np.random.default_rng(0).normal(size=(50, 5))
        names = ["feat_a", "cov_area_4G", "feat_b", "y_5G", "feat_c"]

        X_clean, kept, removed = remove_target_columns(X, names)

        assert X_clean.shape == (50, 3)
        assert kept == ["feat_a", "feat_b", "feat_c"]
        assert set(removed) == {"cov_area_4G", "y_5G"}

    def test_validate_no_leakage_raises(self):
        from coreset_selection.data.target_columns import validate_no_leakage

        with pytest.raises(ValueError, match="Target leakage"):
            validate_no_leakage(["feat_1", "cov_area_4G", "feat_2"])

    def test_validate_no_leakage_passes(self):
        from coreset_selection.data.target_columns import validate_no_leakage

        # Should not raise
        validate_no_leakage(["feat_1", "feat_2", "population", "latitude"])

    def test_regression_feature_equals_target_is_dropped(self):
        """Regression test: a synthetic column that IS the target must be removed."""
        from coreset_selection.data.target_columns import remove_target_columns

        rng = np.random.default_rng(42)
        N, d = 100, 6
        X = rng.normal(size=(N, d))

        # Column 2 is secretly the target (named cov_area_4G)
        target = rng.uniform(0, 1, size=N)
        X[:, 2] = target

        names = ["f0", "f1", "cov_area_4G", "f3", "f4", "f5"]

        X_clean, kept, removed = remove_target_columns(X, names)

        assert "cov_area_4G" in removed
        assert X_clean.shape[1] == 5
        # The target column should not appear in the clean matrix
        # (verify by checking no column is identical to `target`)
        for j in range(X_clean.shape[1]):
            assert not np.allclose(X_clean[:, j], target), (
                f"Column {j} ({kept[j]}) in cleaned matrix is identical to the target!"
            )


# ---------------------------------------------------------------------------
# Phase 5.1 / 5.2: Constraint API and dual diagnostics
# ---------------------------------------------------------------------------

class TestConstraintAPIConsistency:
    """Verify canonical constraint API (Phase 5.1) and dual diagnostics (5.2)."""

    @staticmethod
    def _make_geo_with_pop(N=150, G=5, seed=10):
        from coreset_selection.geo.info import build_geo_info
        rng = np.random.default_rng(seed)
        labels = np.repeat(np.arange(G), N // G)
        # Pad if N not divisible by G
        if len(labels) < N:
            labels = np.concatenate([labels, rng.integers(0, G, size=N - len(labels))])
        pop = rng.uniform(100, 10000, size=N)
        return build_geo_info(labels, population_weights=pop), pop

    def test_constraint_returns_all_required_fields(self):
        """ProportionalityConstraintSet.values() must return named KL values."""
        from coreset_selection.constraints.proportionality import (
            build_population_share_constraint,
            build_municipality_share_constraint,
            ProportionalityConstraintSet,
        )

        geo, pop = self._make_geo_with_pop()

        c_pop = build_population_share_constraint(geo=geo, population=pop, alpha=1.0, tau=0.02)
        c_muni = build_municipality_share_constraint(geo=geo, alpha=1.0, tau=0.02)
        cs = ProportionalityConstraintSet(
            geo=geo, constraints=[c_pop, c_muni], min_one_per_group=True,
        )

        mask = np.zeros(geo.N, dtype=bool)
        rng = np.random.default_rng(0)
        mask[rng.choice(geo.N, size=30, replace=False)] = True

        vals = cs.values(mask)
        assert "population_share" in vals
        assert "municipality_share" in vals
        assert isinstance(vals["population_share"], float)

    def test_constraint_evaluation_returns_finite(self):
        """KL values should be finite for any non-empty subset."""
        from coreset_selection.constraints.proportionality import (
            build_population_share_constraint,
        )

        geo, pop = self._make_geo_with_pop()
        c = build_population_share_constraint(geo=geo, population=pop, alpha=1.0, tau=0.02)

        mask = np.zeros(geo.N, dtype=bool)
        mask[:10] = True

        val = c.value(mask, geo)
        assert np.isfinite(val)

    def test_constraint_smoothed_kl_never_inf_for_nonempty(self):
        """Laplace smoothing should prevent infinite KL even when some groups empty."""
        from coreset_selection.constraints.proportionality import (
            build_municipality_share_constraint,
        )

        geo, _ = self._make_geo_with_pop()
        c = build_municipality_share_constraint(geo=geo, alpha=1.0, tau=0.5)

        # Select only from group 0 -> groups 1..G-1 have zero count
        mask = np.zeros(geo.N, dtype=bool)
        g0_idx = geo.group_to_indices[0]
        mask[g0_idx[:5]] = True

        val = c.value(mask, geo)
        assert np.isfinite(val), "Laplace smoothing should prevent KL = +inf"

    def test_dual_diagnostics_always_both_weight_regimes(self):
        """dual_geo_diagnostics must return both _muni and _pop keys."""
        from coreset_selection.evaluation.geo_diagnostics import dual_geo_diagnostics

        geo, _ = self._make_geo_with_pop()
        rng = np.random.default_rng(55)
        idx_sel = rng.choice(geo.N, size=30, replace=False)

        result = dual_geo_diagnostics(geo, idx_sel, k=30, alpha=1.0)

        required_keys = [
            "geo_kl", "geo_l1", "geo_maxdev",
            "geo_kl_muni", "geo_l1_muni", "geo_maxdev_muni",
            "geo_kl_pop", "geo_l1_pop", "geo_maxdev_pop",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

        # All values should be finite (pop weights were provided)
        for key in required_keys:
            assert np.isfinite(result[key]), f"{key} is not finite: {result[key]}"

    def test_multi_constraint_feasibility_is_conjunction(self):
        """Multi-constraint feasibility: Eq. (11) says feasible iff ALL tau_h satisfied."""
        from coreset_selection.constraints.proportionality import (
            build_population_share_constraint,
            build_municipality_share_constraint,
            ProportionalityConstraintSet,
        )

        geo, pop = self._make_geo_with_pop()

        # Very tight tau: almost certainly infeasible for random selection
        c_pop = build_population_share_constraint(geo=geo, population=pop, alpha=1.0, tau=1e-6)
        c_muni = build_municipality_share_constraint(geo=geo, alpha=1.0, tau=10.0)  # loose

        cs = ProportionalityConstraintSet(
            geo=geo, constraints=[c_pop, c_muni], min_one_per_group=False,
        )

        # Random mask
        mask = np.zeros(geo.N, dtype=bool)
        mask[:20] = True

        # Population constraint likely violated, municipality likely OK
        v_pop = c_pop.violation(mask, geo)
        v_muni = c_muni.violation(mask, geo)

        if v_pop > 0 and v_muni == 0:
            # Should be infeasible because conjunction requires BOTH
            assert not cs.is_feasible(mask)


# ---------------------------------------------------------------------------
# Phase 5.3: tau calibration and sensitivity sweep
# ---------------------------------------------------------------------------

class TestTauCalibration:
    """Verify tau calibration helper and sensitivity sweep."""

    @staticmethod
    def _make_geo(N=300, G=5, seed=10):
        from coreset_selection.geo.info import build_geo_info
        rng = np.random.default_rng(seed)
        labels = np.repeat(np.arange(G), N // G)
        if len(labels) < N:
            labels = np.concatenate([labels, rng.integers(0, G, size=N - len(labels))])
        pop = rng.uniform(100, 10000, size=N)
        return build_geo_info(labels, population_weights=pop), pop

    def test_estimate_feasible_tau_range_keys(self):
        """Result should contain expected summary statistics."""
        from coreset_selection.constraints.calibration import estimate_feasible_tau_range

        geo, pop = self._make_geo()
        result = estimate_feasible_tau_range(geo, k=50, weight_type="muni", n_samples=50, seed=0)

        for key in ["tau_min_achievable", "tau_q10", "tau_median", "tau_mean", "tau_std"]:
            assert key in result, f"Missing key: {key}"
            assert np.isfinite(result[key]), f"{key} is not finite"

    def test_kl_floor_below_random_median(self):
        """KL_min(k) should be <= the median KL of random subsets."""
        from coreset_selection.constraints.calibration import estimate_feasible_tau_range

        geo, _ = self._make_geo()
        result = estimate_feasible_tau_range(geo, k=50, weight_type="muni", n_samples=100, seed=42)

        assert result["tau_min_achievable"] <= result["tau_median"] + 1e-10

    def test_tau_sweep_feasibility_monotone(self):
        """Loosening tau should never decrease the feasible fraction."""
        from coreset_selection.constraints.calibration import tau_sensitivity_sweep

        geo, _ = self._make_geo()
        results = tau_sensitivity_sweep(
            geo, k=50,
            tau_values=[0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
            weight_type="muni",
            n_samples=100,
            seed=0,
        )

        fracs = [r.feasible_frac for r in results]
        for i in range(1, len(fracs)):
            assert fracs[i] >= fracs[i - 1] - 1e-10, (
                f"Feasible fraction decreased from tau={results[i-1].tau} to tau={results[i].tau}"
            )

    def test_tau_sweep_achievability_flag(self):
        """is_achievable should be True iff tau >= kl_min_achievable."""
        from coreset_selection.constraints.calibration import tau_sensitivity_sweep

        geo, _ = self._make_geo()
        results = tau_sensitivity_sweep(
            geo, k=50,
            tau_values=[1e-8, 0.001, 0.01, 0.1, 1.0],
            weight_type="muni",
            n_samples=50,
            seed=0,
        )

        for r in results:
            if r.tau >= r.kl_min_achievable:
                assert r.is_achievable
            else:
                assert not r.is_achievable

    def test_tau_sweep_csv_export(self):
        """tau_sweep_to_csv_rows should produce valid dicts."""
        from coreset_selection.constraints.calibration import (
            tau_sensitivity_sweep,
            tau_sweep_to_csv_rows,
        )

        geo, _ = self._make_geo()
        results = tau_sensitivity_sweep(
            geo, k=50, tau_values=[0.01, 0.05], n_samples=20, seed=0,
        )
        rows = tau_sweep_to_csv_rows(results)

        assert len(rows) == 2
        assert "tau" in rows[0]
        assert "feasible_frac" in rows[0]


# ---------------------------------------------------------------------------
# Phase 2: Target type detection tests
# ---------------------------------------------------------------------------

class TestTargetTypeDetection:
    """Verify auto-detection of regression vs classification targets."""

    def test_continuous_float_is_regression(self):
        from coreset_selection.evaluation.classification_metrics import infer_target_type
        y = np.random.default_rng(0).normal(size=100)
        assert infer_target_type(y) == "regression"

    def test_integer_low_cardinality_is_classification(self):
        from coreset_selection.evaluation.classification_metrics import infer_target_type
        y = np.random.default_rng(0).integers(0, 5, size=100)
        assert infer_target_type(y) == "classification"

    def test_integer_high_cardinality_is_regression(self):
        from coreset_selection.evaluation.classification_metrics import infer_target_type
        y = np.arange(1000)
        assert infer_target_type(y) == "regression"

    def test_float_integers_low_card_is_classification(self):
        """Float-stored integers with low cardinality -> classification."""
        from coreset_selection.evaluation.classification_metrics import infer_target_type
        y = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0] * 20)
        assert infer_target_type(y) == "classification"

    def test_auto_dispatch_regression(self):
        from coreset_selection.evaluation.downstream_metrics import evaluate_target_auto
        y_true = np.random.default_rng(0).normal(size=50)
        y_pred = y_true + np.random.default_rng(1).normal(0, 0.1, size=50)
        result = evaluate_target_auto(y_true, y_pred, target_type="regression")
        assert "overall_rmse" in result
        assert "accuracy" not in result

    def test_auto_dispatch_classification(self):
        from coreset_selection.evaluation.downstream_metrics import evaluate_target_auto
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        y_pred = np.array([0, 1, 1, 0, 2, 2, 0, 1, 2, 1])
        result = evaluate_target_auto(y_true, y_pred, target_type="classification")
        assert "accuracy" in result
        assert "cohens_kappa" in result
        assert "overall_rmse" not in result


# ---------------------------------------------------------------------------
# Phase 2: Feature schema mixed-type inference tests
# ---------------------------------------------------------------------------

class TestFeatureSchemaPhase2:
    """Verify feature schema handles mixed-type DataFrames correctly."""

    def test_infer_mixed_types(self):
        import pandas as pd
        from coreset_selection.data.feature_schema import infer_schema, FeatureType

        df = pd.DataFrame({
            "continuous": np.random.default_rng(0).normal(size=100),
            "int_cat": np.random.default_rng(0).integers(0, 5, size=100),
            "str_cat": np.random.default_rng(0).choice(["a", "b", "c"], size=100),
            "high_card_int": np.arange(100),
            "bool_col": np.random.default_rng(0).choice([True, False], size=100),
        })

        schema = infer_schema(df)
        assert schema.column_types["continuous"] == FeatureType.NUMERIC
        assert schema.column_types["int_cat"] == FeatureType.CATEGORICAL
        assert schema.column_types["str_cat"] == FeatureType.CATEGORICAL
        assert schema.column_types["high_card_int"] == FeatureType.NUMERIC
        assert schema.column_types["bool_col"] == FeatureType.CATEGORICAL

    def test_explicit_ordinal_override(self):
        import pandas as pd
        from coreset_selection.data.feature_schema import infer_schema, FeatureType

        df = pd.DataFrame({
            "rating": np.random.default_rng(0).integers(1, 6, size=100),
        })

        schema = infer_schema(df, ordinal_columns=["rating"])
        assert schema.column_types["rating"] == FeatureType.ORDINAL

    def test_feature_columns_includes_all_types(self):
        import pandas as pd
        from coreset_selection.data.feature_schema import infer_schema

        df = pd.DataFrame({
            "num": np.random.default_rng(0).normal(size=50),
            "cat": np.random.default_rng(0).choice(["x", "y"], size=50),
        })

        schema = infer_schema(df, ordinal_columns=[], categorical_columns=["cat"])
        feats = schema.feature_columns
        assert "num" in feats
        assert "cat" in feats
