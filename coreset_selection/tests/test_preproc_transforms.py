"""
Preprocessing tests — feature transformations, scaling, encoding, and metrics.

Split from tests/test_preprocessing.py.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Phase 4.1: Preprocessing contract
# ---------------------------------------------------------------------------

class TestPreprocessingContract:
    """Verify the preprocessing pipeline matches manuscript Section VII."""

    @staticmethod
    def _make_synthetic(N=200, d=10, seed=0):
        """Create a small synthetic dataset with controlled missingness."""
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(N, d)).astype(np.float64)

        # Inject some NaN in columns 0, 3
        X[rng.choice(N, 20, replace=False), 0] = np.nan
        X[rng.choice(N, 15, replace=False), 3] = np.nan

        # Make column 5 heavy-tailed, non-negative (exponential)
        X[:, 5] = rng.exponential(scale=100.0, size=N)
        X[:, 5] = np.abs(X[:, 5])  # ensure non-negative

        feature_names = [f"feat_{j}" for j in range(d)]
        train_idx = np.arange(0, int(0.8 * N))
        val_idx = np.arange(int(0.8 * N), N)
        return X, feature_names, train_idx, val_idx

    def test_missingness_indicators_added(self):
        """Missingness indicators should only be added for columns with NaN."""
        from coreset_selection.data.cache import _preprocess_fit_transform

        X, names, train_idx, _ = self._make_synthetic()
        X_out, meta = _preprocess_fit_transform(
            X_unscaled=X, train_idx=train_idx, feature_names=names,
        )

        missing_names = meta["missing_feature_names"]
        # Columns 0 and 3 had NaNs
        assert "feat_0__missing" in missing_names
        assert "feat_3__missing" in missing_names
        # Column 1 had no NaN
        assert "feat_1__missing" not in missing_names
        # Output should have more columns than input
        assert X_out.shape[1] > X.shape[1]

    def test_imputation_uses_train_only(self):
        """Imputed values should come from train-split medians, not full data."""
        from coreset_selection.data.cache import _impute_by_train_median

        rng = np.random.default_rng(42)
        N, d = 100, 3
        X = rng.normal(size=(N, d)).astype(np.float64)
        train_idx = np.arange(50)

        # Set all of val column 0 to NaN, and put a distinct value in train
        X[:50, 0] = 10.0  # train has constant 10
        X[50:, 0] = np.nan  # val is missing

        X_imp = _impute_by_train_median(X, train_idx)

        # All val entries should be imputed with train median = 10.0
        assert np.allclose(X_imp[50:, 0], 10.0)

    def test_standardization_train_stats(self):
        """After standardization, train split should have mean~0, std~1."""
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(7)
        N = 200
        X = rng.normal(loc=50, scale=10, size=(N, 5)).astype(np.float64)
        train_idx = np.arange(160)

        scaler = StandardScaler()
        scaler.fit(X[train_idx])
        X_sc = scaler.transform(X)

        train_mean = X_sc[train_idx].mean(axis=0)
        train_std = X_sc[train_idx].std(axis=0)

        np.testing.assert_allclose(train_mean, 0.0, atol=1e-10)
        np.testing.assert_allclose(train_std, 1.0, atol=1e-10)

    def test_log1p_only_on_heavytail_nonneg(self):
        """log1p should only be applied to heavy-tailed non-negative columns."""
        from coreset_selection.data.cache import _detect_log1p_cols

        rng = np.random.default_rng(3)
        N = 500
        X = np.zeros((N, 4), dtype=np.float64)
        train_idx = np.arange(400)

        # Col 0: standard normal (has negatives) -> should NOT be transformed
        X[:, 0] = rng.normal(size=N)
        # Col 1: binary (0/1) -> should NOT be transformed
        X[:, 1] = rng.choice([0, 1], size=N).astype(float)
        # Col 2: truly heavy-tailed non-negative (lognormal, very skewed)
        # q99/q50 ratio will be >> 25 for this distribution
        X[:, 2] = rng.lognormal(mean=1.0, sigma=3.0, size=N)
        # Col 3: uniform [0, 1] -- not heavy-tailed
        X[:, 3] = rng.uniform(0, 1, size=N)

        log1p_cols = _detect_log1p_cols(X, train_idx)
        assert 0 not in log1p_cols, "Normal col with negatives should not be log-transformed"
        assert 1 not in log1p_cols, "Binary col should not be log-transformed"
        assert 2 in log1p_cols, "Heavy-tailed lognormal should be log-transformed"


# ---------------------------------------------------------------------------
# Phase 2: Type-aware preprocessing tests
# ---------------------------------------------------------------------------

class TestTypeAwarePreprocessing:
    """Verify Phase 2 type-aware preprocessing pipeline."""

    @staticmethod
    def _make_mixed_data(N=200, seed=0):
        """Create a synthetic dataset with numeric, ordinal, and categorical cols."""
        rng = np.random.default_rng(seed)
        d = 5
        X = np.zeros((N, d), dtype=np.float64)

        # Col 0: numeric (normal)
        X[:, 0] = rng.normal(50.0, 10.0, size=N)
        # Col 1: ordinal (integers 1..5)
        X[:, 1] = rng.integers(1, 6, size=N).astype(float)
        # Col 2: categorical (integer codes 0..3)
        X[:, 2] = rng.integers(0, 4, size=N).astype(float)
        # Col 3: heavy-tailed numeric (lognormal)
        X[:, 3] = rng.lognormal(mean=1.0, sigma=3.0, size=N)
        # Col 4: categorical with NaN
        X[:, 4] = rng.integers(0, 3, size=N).astype(float)
        X[rng.choice(N, 20, replace=False), 4] = np.nan

        feature_names = ["num_feat", "ord_feat", "cat_feat", "heavy_num", "cat_missing"]
        feature_types = ["numeric", "ordinal", "categorical", "numeric", "categorical"]
        train_idx = np.arange(0, int(0.8 * N))
        return X, feature_names, feature_types, train_idx

    def test_categorical_no_log1p(self):
        """Categorical columns should never be log-transformed."""
        from coreset_selection.data.cache import _preprocess_fit_transform

        X, names, types, train_idx = self._make_mixed_data()
        X_out, meta = _preprocess_fit_transform(
            X_unscaled=X, train_idx=train_idx,
            feature_names=names, feature_types=types,
        )

        # log1p should not include categorical columns
        for log_name in meta["log1p_feature_names"]:
            idx = names.index(log_name) if log_name in names else -1
            if idx >= 0:
                assert types[idx] != "categorical", (
                    f"Categorical column {log_name} was log-transformed"
                )

    def test_ordinal_no_log1p_default(self):
        """Ordinal columns should not be log-transformed by default."""
        from coreset_selection.data.cache import _preprocess_fit_transform

        X, names, types, train_idx = self._make_mixed_data()
        X_out, meta = _preprocess_fit_transform(
            X_unscaled=X, train_idx=train_idx,
            feature_names=names, feature_types=types,
            log1p_ordinals=False,
        )

        for log_name in meta["log1p_feature_names"]:
            idx = names.index(log_name) if log_name in names else -1
            if idx >= 0:
                assert types[idx] != "ordinal", (
                    f"Ordinal column {log_name} was log-transformed"
                )

    def test_categorical_mode_imputation(self):
        """Categorical NaN should be imputed with mode, not median."""
        from coreset_selection.data.cache import _impute_typeaware

        rng = np.random.default_rng(42)
        N = 100
        X = np.zeros((N, 2), dtype=np.float64)
        # Col 0: categorical with mode = 2 (appears 50 times in train)
        X[:50, 0] = 2.0  # 50 times
        X[50:70, 0] = 1.0  # 20 times
        X[70:80, 0] = 0.0  # 10 times
        X[80:, 0] = np.nan  # to be imputed

        train_idx = np.arange(80)
        feature_types = ["categorical", "numeric"]

        X_imp = _impute_typeaware(X, train_idx, feature_types)
        # NaN entries should be imputed with mode = 2.0
        assert np.allclose(X_imp[80:, 0], 2.0), (
            f"Expected mode=2.0, got {X_imp[80:, 0]}"
        )

    def test_no_missingness_indicator_for_categorical(self):
        """Categorical columns should not get missingness indicators."""
        from coreset_selection.data.cache import _preprocess_fit_transform

        X, names, types, train_idx = self._make_mixed_data()
        X_out, meta = _preprocess_fit_transform(
            X_unscaled=X, train_idx=train_idx,
            feature_names=names, feature_types=types,
        )

        # cat_missing (col 4) has NaN but should not produce a __missing indicator
        assert "cat_missing__missing" not in meta["missing_feature_names"], (
            "Categorical column should not produce a missingness indicator"
        )

    def test_feature_types_propagated(self):
        """Feature types metadata should be preserved through preprocessing."""
        from coreset_selection.data.cache import _preprocess_fit_transform

        X, names, types, train_idx = self._make_mixed_data()
        X_out, meta = _preprocess_fit_transform(
            X_unscaled=X, train_idx=train_idx,
            feature_names=names, feature_types=types,
        )

        ft = meta["feature_types"]
        assert ft[0] == "numeric"
        assert ft[1] == "ordinal"
        assert ft[2] == "categorical"


# ---------------------------------------------------------------------------
# Phase 2: Classification metrics tests
# ---------------------------------------------------------------------------

class TestClassificationMetrics:
    """Verify Phase 2 classification metric implementations."""

    def test_accuracy_perfect(self):
        from coreset_selection.evaluation.classification_metrics import accuracy
        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 2, 0, 1])
        assert accuracy(y_true, y_pred) == 1.0

    def test_accuracy_half(self):
        from coreset_selection.evaluation.classification_metrics import accuracy
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 0, 0, 0])
        assert accuracy(y_true, y_pred) == 0.5

    def test_cohens_kappa_perfect(self):
        from coreset_selection.evaluation.classification_metrics import cohens_kappa
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        assert abs(cohens_kappa(y_true, y_pred) - 1.0) < 1e-10

    def test_cohens_kappa_random(self):
        """Random predictions should have kappa near 0."""
        from coreset_selection.evaluation.classification_metrics import cohens_kappa
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 3, size=10000)
        y_pred = rng.integers(0, 3, size=10000)
        k = cohens_kappa(y_true, y_pred)
        assert abs(k) < 0.1, f"Expected kappa ~ 0 for random, got {k}"

    def test_macro_f1_binary(self):
        from coreset_selection.evaluation.classification_metrics import macro_f1
        y_true = np.array([1, 1, 0, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 0, 1, 1])
        f1 = macro_f1(y_true, y_pred)
        assert 0.0 < f1 < 1.0

    def test_full_classification_eval_has_keys(self):
        from coreset_selection.evaluation.classification_metrics import (
            full_classification_evaluation,
        )
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        y_pred = np.array([0, 1, 1, 0, 2, 2, 0, 1, 2, 1])
        states = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])

        result = full_classification_evaluation(y_true, y_pred, states, target_suffix="_test")
        assert "accuracy_test" in result
        assert "cohens_kappa_test" in result
        assert "macro_f1_test" in result
        assert "macro_accuracy_test" in result
        assert "worst_group_accuracy_test" in result

    def test_confusion_matrix_correct(self):
        from coreset_selection.evaluation.classification_metrics import confusion_matrix_dict
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])
        cm = confusion_matrix_dict(y_true, y_pred)
        assert cm["classes"] == [0, 1]
        # [[1, 1], [1, 1]]
        assert cm["matrix"][0][0] == 1  # TN
        assert cm["matrix"][0][1] == 1  # FP
        assert cm["matrix"][1][0] == 1  # FN
        assert cm["matrix"][1][1] == 1  # TP


# ---------------------------------------------------------------------------
# Phase 3: Type-aware preprocessing metadata and reproducibility tests
# ---------------------------------------------------------------------------

class TestPhase3PreprocessingMetadata:
    """Verify Phase 3 enhancements: extended feature_types_out, imputation
    statistics, per-type column lists, and scale-mask correctness."""

    @staticmethod
    def _make_mixed_data(N=200, seed=0):
        """Create a synthetic dataset with numeric, ordinal, and categorical cols."""
        rng = np.random.default_rng(seed)
        d = 6
        X = np.zeros((N, d), dtype=np.float64)

        # Col 0: numeric (normal) -- some NaN
        X[:, 0] = rng.normal(50.0, 10.0, size=N)
        X[rng.choice(N, 15, replace=False), 0] = np.nan

        # Col 1: ordinal (integers 1..5) -- some NaN
        X[:, 1] = rng.integers(1, 6, size=N).astype(float)
        X[rng.choice(N, 10, replace=False), 1] = np.nan

        # Col 2: categorical (integer codes 0..3) -- some NaN
        X[:, 2] = rng.integers(0, 4, size=N).astype(float)
        X[rng.choice(N, 20, replace=False), 2] = np.nan

        # Col 3: numeric heavy-tailed (lognormal, no NaN)
        X[:, 3] = rng.lognormal(mean=1.0, sigma=3.0, size=N)

        # Col 4: ordinal no NaN
        X[:, 4] = rng.integers(0, 10, size=N).astype(float)

        # Col 5: categorical no NaN
        X[:, 5] = rng.integers(0, 2, size=N).astype(float)

        feature_names = [
            "num_miss", "ord_miss", "cat_miss",
            "heavy_num", "ord_clean", "cat_clean",
        ]
        feature_types = [
            "numeric", "ordinal", "categorical",
            "numeric", "ordinal", "categorical",
        ]
        train_idx = np.arange(0, int(0.8 * N))
        return X, feature_names, feature_types, train_idx

    def test_feature_types_out_length_matches_output(self):
        """feature_types_out should have one entry per output column."""
        from coreset_selection.data.cache import _preprocess_fit_transform

        X, names, types, train_idx = self._make_mixed_data()
        X_out, meta = _preprocess_fit_transform(
            X_unscaled=X, train_idx=train_idx,
            feature_names=names, feature_types=types,
        )
        fto = meta["feature_types_out"]
        assert len(fto) == X_out.shape[1], (
            f"feature_types_out length {len(fto)} != output columns {X_out.shape[1]}"
        )

    def test_missingness_indicators_typed_as_numeric(self):
        """Appended missingness indicators should be typed as 'numeric'."""
        from coreset_selection.data.cache import _preprocess_fit_transform

        X, names, types, train_idx = self._make_mixed_data()
        X_out, meta = _preprocess_fit_transform(
            X_unscaled=X, train_idx=train_idx,
            feature_names=names, feature_types=types,
        )
        fto = meta["feature_types_out"]
        n_orig = len(names)
        # All entries beyond the original columns are missingness indicators
        for i in range(n_orig, len(fto)):
            assert fto[i] == "numeric", (
                f"Missingness indicator at position {i} has type '{fto[i]}', expected 'numeric'"
            )

    def test_per_type_column_lists_present(self):
        """Metadata should include categorical_columns, ordinal_columns, numeric_columns."""
        from coreset_selection.data.cache import _preprocess_fit_transform

        X, names, types, train_idx = self._make_mixed_data()
        _, meta = _preprocess_fit_transform(
            X_unscaled=X, train_idx=train_idx,
            feature_names=names, feature_types=types,
        )
        assert "categorical_columns" in meta
        assert "ordinal_columns" in meta
        assert "numeric_columns" in meta

        assert set(meta["categorical_columns"]) == {"cat_miss", "cat_clean"}
        assert set(meta["ordinal_columns"]) == {"ord_miss", "ord_clean"}
        assert set(meta["numeric_columns"]) == {"num_miss", "heavy_num"}

    def test_impute_values_present_and_correct_types(self):
        """impute_values should map column names to fill values with correct strategies."""
        from coreset_selection.data.cache import _preprocess_fit_transform

        X, names, types, train_idx = self._make_mixed_data()
        _, meta = _preprocess_fit_transform(
            X_unscaled=X, train_idx=train_idx,
            feature_names=names, feature_types=types,
        )
        iv = meta["impute_values"]
        # Columns with NaN: num_miss (0), ord_miss (1), cat_miss (2)
        assert "num_miss" in iv, "Numeric column with NaN should have impute value"
        assert "ord_miss" in iv, "Ordinal column with NaN should have impute value"
        assert "cat_miss" in iv, "Categorical column with NaN should have impute value"
        # Columns without NaN should NOT appear
        assert "heavy_num" not in iv
        assert "ord_clean" not in iv
        assert "cat_clean" not in iv

    def test_categorical_impute_is_mode(self):
        """Categorical imputation should use mode, not median."""
        from coreset_selection.data.cache import _impute_typeaware

        rng = np.random.default_rng(99)
        N = 120
        X = np.zeros((N, 1), dtype=np.float64)
        # Mode = 3 (appears 60 times in train)
        X[:60, 0] = 3.0
        X[60:80, 0] = 1.0
        X[80:100, 0] = 0.0
        X[100:, 0] = np.nan  # 20 NaN entries

        train_idx = np.arange(100)
        X_imp, stats = _impute_typeaware(
            X, train_idx, ["categorical"],
            categorical_strategy="mode",
            return_stats=True,
        )
        assert 0 in stats, "Column 0 should have impute stat"
        assert stats[0] == 3.0, f"Expected mode=3.0, got {stats[0]}"
        assert np.all(X_imp[100:, 0] == 3.0)

    def test_ordinal_impute_is_rounded_median(self):
        """Ordinal imputation with 'median' strategy should use rounded median."""
        from coreset_selection.data.cache import _impute_typeaware

        N = 100
        X = np.zeros((N, 1), dtype=np.float64)
        # Train values: mostly 3 and 4, median should be ~3.5 -> rounds to 4
        X[:40, 0] = 3.0
        X[40:80, 0] = 4.0
        X[80:, 0] = np.nan

        train_idx = np.arange(80)
        X_imp, stats = _impute_typeaware(
            X, train_idx, ["ordinal"],
            ordinal_strategy="median",
            return_stats=True,
        )
        # Median of [3]*40 + [4]*40 = 3.5, rounded = 4.0
        expected = 4.0
        assert stats[0] == expected, f"Expected rounded median={expected}, got {stats[0]}"
        assert np.all(X_imp[80:, 0] == expected)

    def test_numeric_impute_is_median(self):
        """Numeric imputation should use train median (not rounded)."""
        from coreset_selection.data.cache import _impute_typeaware

        N = 100
        X = np.zeros((N, 1), dtype=np.float64)
        X[:80, 0] = np.arange(80, dtype=np.float64)  # median ~39.5
        X[80:, 0] = np.nan

        train_idx = np.arange(80)
        X_imp, stats = _impute_typeaware(
            X, train_idx, ["numeric"],
            return_stats=True,
        )
        expected = np.median(np.arange(80, dtype=np.float64))
        assert abs(stats[0] - expected) < 1e-10, (
            f"Expected median={expected}, got {stats[0]}"
        )

    def test_no_categorical_missingness_indicator_in_extended_types(self):
        """Categorical columns with NaN should NOT produce __missing indicators
        in the output, and feature_types_out should reflect this."""
        from coreset_selection.data.cache import _preprocess_fit_transform

        X, names, types, train_idx = self._make_mixed_data()
        _, meta = _preprocess_fit_transform(
            X_unscaled=X, train_idx=train_idx,
            feature_names=names, feature_types=types,
        )
        # cat_miss has NaN but should not get a __missing indicator
        assert "cat_miss__missing" not in meta["feature_names"]
        assert "cat_miss__missing" not in meta["missing_feature_names"]

        # Numeric and ordinal with NaN SHOULD get indicators
        assert "num_miss__missing" in meta["missing_feature_names"]
        assert "ord_miss__missing" in meta["missing_feature_names"]

    def test_scale_mask_respects_categorical_no_scale(self):
        """When scale_categoricals=False, categorical columns should be unscaled."""
        from coreset_selection.data.cache import _preprocess_fit_transform
        from sklearn.preprocessing import StandardScaler

        X, names, types, train_idx = self._make_mixed_data()
        X_raw, meta = _preprocess_fit_transform(
            X_unscaled=X, train_idx=train_idx,
            feature_names=names, feature_types=types,
        )
        extended_types = meta["feature_types_out"]

        # Build scale mask (mirroring build_replicate_cache logic)
        n_total = X_raw.shape[1]
        scale_mask = np.ones(n_total, dtype=bool)
        for j in range(n_total):
            ft = extended_types[j] if j < len(extended_types) else "numeric"
            if ft == "categorical":
                scale_mask[j] = False

        # Categorical columns should be False in mask
        cat_indices = [i for i, ft in enumerate(extended_types) if ft == "categorical"]
        for ci in cat_indices:
            assert not scale_mask[ci], (
                f"Categorical column at index {ci} should NOT be scaled"
            )

        # Verify scaling leaves categorical columns unchanged
        scaler = StandardScaler()
        scaleable_idx = np.flatnonzero(scale_mask)
        scaler.fit(X_raw[np.ix_(train_idx, scaleable_idx)])
        X_scaled = X_raw.copy()
        X_scaled[:, scaleable_idx] = scaler.transform(X_raw[:, scaleable_idx])

        for ci in cat_indices:
            np.testing.assert_array_equal(
                X_scaled[:, ci], X_raw[:, ci],
                err_msg=f"Categorical column {ci} was modified by scaling",
            )

    def test_impute_typeaware_return_stats_flag(self):
        """When return_stats=False (default), only array is returned."""
        from coreset_selection.data.cache import _impute_typeaware

        N = 50
        X = np.ones((N, 2), dtype=np.float64)
        X[0, 0] = np.nan
        train_idx = np.arange(40)

        result = _impute_typeaware(X, train_idx, ["numeric", "numeric"])
        # Should be just an array, not a tuple
        assert isinstance(result, np.ndarray)

        result2 = _impute_typeaware(
            X, train_idx, ["numeric", "numeric"], return_stats=True,
        )
        assert isinstance(result2, tuple)
        assert len(result2) == 2


# ---------------------------------------------------------------------------
# Phase 6: Categorical encoding stability + unknown handling tests
# ---------------------------------------------------------------------------

class TestCategoricalEncodingStability:
    """Phase 6: Verify categorical integer encoding is stable and handles
    unknown / missing values correctly."""

    def test_encoding_is_sorted_deterministic(self):
        """Encoding should sort unique values by str() and assign 0..K-1."""
        import pandas as pd

        df = pd.DataFrame({
            "cat": ["banana", "apple", "cherry", "apple", "banana"],
        })
        # Expected sorted order: apple(0), banana(1), cherry(2)
        raw_vals = df["cat"]
        non_null = raw_vals.dropna().unique()
        sorted_cats = sorted(non_null, key=lambda v: str(v))
        cat_map = {v: i for i, v in enumerate(sorted_cats)}

        assert cat_map == {"apple": 0, "banana": 1, "cherry": 2}

        codes = raw_vals.map(cat_map).fillna(-1).astype(np.float64)
        expected = np.array([1.0, 0.0, 2.0, 0.0, 1.0])
        np.testing.assert_array_equal(codes.values, expected)

    def test_encoding_handles_nan_as_minus_one(self):
        """NaN / missing categories should be encoded as -1."""
        import pandas as pd

        df = pd.DataFrame({
            "cat": ["x", np.nan, "y", None, "x"],
        })
        non_null = df["cat"].dropna().unique()
        sorted_cats = sorted(non_null, key=lambda v: str(v))
        cat_map = {v: i for i, v in enumerate(sorted_cats)}

        codes = df["cat"].map(cat_map).fillna(-1).astype(np.float64)
        expected = np.array([0.0, -1.0, 1.0, -1.0, 0.0])
        np.testing.assert_array_equal(codes.values, expected)

    def test_encoding_handles_unseen_category(self):
        """Values not in the map should become -1 (unknown code)."""
        import pandas as pd

        cat_map = {"a": 0, "b": 1}
        series = pd.Series(["a", "b", "c", "a"])
        codes = series.map(cat_map).fillna(-1).astype(np.float64)
        expected = np.array([0.0, 1.0, -1.0, 0.0])
        np.testing.assert_array_equal(codes.values, expected)

    def test_encoding_numeric_categories_sorted_by_string(self):
        """Numeric category values should still be sorted by str() for stability."""
        import pandas as pd

        df = pd.DataFrame({
            "cat": [10, 2, 1, 10, 2],
        })
        non_null = df["cat"].dropna().unique()
        sorted_cats = sorted(non_null, key=lambda v: str(v))
        # str() order: "1"(->0), "10"(->1), "2"(->2) -- lexicographic!
        cat_map = {v: i for i, v in enumerate(sorted_cats)}
        assert cat_map == {1: 0, 10: 1, 2: 2}


# ---------------------------------------------------------------------------
# Phase 6: Manuscript mode gating
# ---------------------------------------------------------------------------

class TestManuscriptModeGating:
    """Verify that strict_manuscript_mode flag is available and defaults to False."""

    def test_default_is_not_strict(self):
        from coreset_selection.config.dataclasses import PreprocessingConfig
        cfg = PreprocessingConfig()
        assert cfg.strict_manuscript_mode is False

    def test_strict_mode_can_be_enabled(self):
        from coreset_selection.config.dataclasses import PreprocessingConfig
        cfg = PreprocessingConfig(strict_manuscript_mode=True)
        assert cfg.strict_manuscript_mode is True
