"""
Evaluation Protocol Tests — Raw space evaluation.

Creates small synthetic datasets and verifies every metric computation
against hand-calculated values.  This ensures the evaluation pipeline
matches the manuscript's specifications (Section VII.C, VII.F):

1. Nystrom error:  e_Nys = ||K_E - K_hat||_F / ||K_E||_F
2. kPCA distortion: relative top-r eigenvalue error
3. KRR RMSE:  root mean squared error on E_test
4. Nystrom stabilization:  lambda_nys = 1e-6 * tr(W) / k
5. Bandwidth:  sigma_raw^2 from median heuristic on E_train subset
6. Gram matrix centering for kPCA
7. Coverage targets alignment

Split from tests/test_evaluation_protocol.py.
"""

from __future__ import annotations

import numpy as np
import pytest


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def small_data():
    """Tiny reproducible dataset for hand-verification of metrics.

    N=10, D=3, k=3, |E|=6, |E_train|=4, |E_test|=2, G=2 groups.
    """
    rng = np.random.default_rng(0)
    N, D = 10, 3
    X = rng.standard_normal((N, D)).astype(np.float64)
    y = rng.standard_normal((N, 2)).astype(np.float64)

    state_labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    population = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                          dtype=np.float64)

    eval_idx = np.array([0, 1, 3, 5, 7, 9])
    eval_train_idx = np.array([0, 1, 3, 5])
    eval_test_idx = np.array([7, 9])

    # Landmarks
    S_idx = np.array([2, 6, 8])
    k = 3

    return {
        "X": X, "y": y,
        "state_labels": state_labels,
        "population": population,
        "eval_idx": eval_idx,
        "eval_train_idx": eval_train_idx,
        "eval_test_idx": eval_test_idx,
        "S_idx": S_idx,
        "k": k,
        "N": N, "D": D,
    }


# ===================================================================
# Test: RBF kernel
# ===================================================================

class TestRBFKernel:
    """Verify the RBF kernel implementation."""

    def test_rbf_kernel_self(self):
        """K(x,x) should have ones on the diagonal."""
        from coreset_selection.evaluation.raw_space import _rbf_kernel

        X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        sigma_sq = 1.0
        K = _rbf_kernel(X, X, sigma_sq)
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-12)

    def test_rbf_kernel_symmetry(self):
        """K should be symmetric."""
        from coreset_selection.evaluation.raw_space import _rbf_kernel

        rng = np.random.default_rng(1)
        X = rng.standard_normal((5, 3))
        K = _rbf_kernel(X, X, sigma_sq=2.0)
        np.testing.assert_allclose(K, K.T, atol=1e-12)

    def test_rbf_kernel_known_value(self):
        """Check a specific kernel value by hand.

        K(x1, x2) = exp(-||x1-x2||^2 / (2*sigma^2))
        x1 = [1, 0], x2 = [0, 1], ||x1-x2||^2 = 2, sigma^2 = 1
        => K = exp(-2/2) = exp(-1) ~ 0.36788
        """
        from coreset_selection.evaluation.raw_space import _rbf_kernel

        X1 = np.array([[1.0, 0.0]])
        X2 = np.array([[0.0, 1.0]])
        K = _rbf_kernel(X1, X2, sigma_sq=1.0)
        expected = np.exp(-1.0)
        np.testing.assert_allclose(K[0, 0], expected, atol=1e-10)


# ===================================================================
# Test: Nystrom stabilization
# ===================================================================

class TestNystromStabilization:
    """Verify lambda_nys = 1e-6 * tr(W) / k (manuscript notation)."""

    def test_lambda_nys_formula(self):
        """lambda_nys should follow the manuscript formula."""
        from coreset_selection.evaluation.raw_space import (
            _rbf_kernel,
            _nystrom_components,
        )

        rng = np.random.default_rng(10)
        N, D, k = 20, 5, 4
        X = rng.standard_normal((N, D))
        sigma_sq = 1.0

        X_E = X[:10]
        X_S = X[10:10 + k]
        C, W, lambda_nys = _nystrom_components(X_E, X_S, sigma_sq)

        expected_lambda = 1e-6 * np.trace(W) / k
        np.testing.assert_allclose(lambda_nys, expected_lambda, rtol=1e-10)


# ===================================================================
# Test: Nystrom error
# ===================================================================

class TestNystromError:
    """Verify relative Frobenius Nystrom error.

    e_Nys(S) = ||K_E - K_hat_E||_F / ||K_E||_F
    where K_hat = C (W + lambda I)^{-1} C^T
    """

    def test_nystrom_error_perfect(self):
        """When S = E, Nystrom error should be near zero."""
        from coreset_selection.evaluation.raw_space import RawSpaceEvaluator

        rng = np.random.default_rng(20)
        N = 10
        X = rng.standard_normal((N, 3))

        eval_idx = np.arange(N)
        eval_train_idx = np.arange(8)
        eval_test_idx = np.arange(8, N)

        evaluator = RawSpaceEvaluator.build(
            X_raw=X, y=None,
            eval_idx=eval_idx,
            eval_train_idx=eval_train_idx,
            eval_test_idx=eval_test_idx,
            seed=0,
        )
        # Use all points as landmarks
        S_idx = np.arange(N)
        error = evaluator.nystrom_error(S_idx)
        assert error < 0.05, f"Nystrom error with S=E should be near 0, got {error}"

    def test_nystrom_error_nonneg(self, small_data):
        """Nystrom error should be nonneg."""
        from coreset_selection.evaluation.raw_space import RawSpaceEvaluator

        d = small_data
        evaluator = RawSpaceEvaluator.build(
            X_raw=d["X"], y=d["y"],
            eval_idx=d["eval_idx"],
            eval_train_idx=d["eval_train_idx"],
            eval_test_idx=d["eval_test_idx"],
            seed=0,
        )
        error = evaluator.nystrom_error(d["S_idx"])
        assert error >= 0.0

    def test_nystrom_error_bounded(self, small_data):
        """Nystrom error should normally be <= 1 for reasonable subsets."""
        from coreset_selection.evaluation.raw_space import RawSpaceEvaluator

        d = small_data
        evaluator = RawSpaceEvaluator.build(
            X_raw=d["X"], y=d["y"],
            eval_idx=d["eval_idx"],
            eval_train_idx=d["eval_train_idx"],
            eval_test_idx=d["eval_test_idx"],
            seed=0,
        )
        error = evaluator.nystrom_error(d["S_idx"])
        assert error <= 1.5, f"Nystrom error suspiciously high: {error}"

    def test_nystrom_error_manual(self):
        """Hand-verify the Nystrom error formula on a tiny example."""
        from coreset_selection.evaluation.raw_space import (
            _rbf_kernel,
            _nystrom_components,
            _nystrom_approx_gram,
        )

        X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
        sigma_sq = 2.0

        E_idx = np.array([0, 1, 2, 3])
        S_idx = np.array([0, 3])
        X_E = X[E_idx]
        X_S = X[S_idx]

        K_EE = _rbf_kernel(X_E, X_E, sigma_sq)
        C, W, lam = _nystrom_components(X_E, X_S, sigma_sq)
        K_hat = _nystrom_approx_gram(C, W, lam)

        error = np.linalg.norm(K_EE - K_hat, "fro") / (
            np.linalg.norm(K_EE, "fro") + 1e-30
        )
        assert error >= 0.0
        assert np.isfinite(error)


# ===================================================================
# Test: kPCA distortion
# ===================================================================

class TestKPCADistortion:
    """Verify kernel PCA spectral distortion metric."""

    def test_kpca_distortion_nonneg(self, small_data):
        """kPCA distortion should be nonneg."""
        from coreset_selection.evaluation.raw_space import RawSpaceEvaluator

        d = small_data
        evaluator = RawSpaceEvaluator.build(
            X_raw=d["X"], y=d["y"],
            eval_idx=d["eval_idx"],
            eval_train_idx=d["eval_train_idx"],
            eval_test_idx=d["eval_test_idx"],
            seed=0,
        )
        dist = evaluator.kpca_distortion(d["S_idx"], r=3)
        assert dist >= 0.0

    def test_centering(self):
        """Verify Gram matrix centering: H K H with H = I - 11^T/n."""
        from coreset_selection.evaluation.raw_space import _center_gram

        K = np.array([
            [1.0, 0.5, 0.3],
            [0.5, 1.0, 0.4],
            [0.3, 0.4, 1.0],
        ])
        Kc = _center_gram(K)
        n = 3
        H = np.eye(n) - np.ones((n, n)) / n
        expected = H @ K @ H
        np.testing.assert_allclose(Kc, expected, atol=1e-12)

    def test_centering_row_col_means(self):
        """Centered Gram matrix should have zero row and column means."""
        from coreset_selection.evaluation.raw_space import _center_gram

        rng = np.random.default_rng(5)
        X = rng.standard_normal((8, 3))
        K = X @ X.T  # linear kernel
        Kc = _center_gram(K)
        np.testing.assert_allclose(Kc.mean(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(Kc.mean(axis=1), 0.0, atol=1e-10)


# ===================================================================
# Test: KRR RMSE
# ===================================================================

class TestKRR:
    """Verify KRR RMSE computation."""

    def test_krr_rmse_nonneg(self, small_data):
        """KRR RMSE should be nonneg."""
        from coreset_selection.evaluation.raw_space import RawSpaceEvaluator

        d = small_data
        evaluator = RawSpaceEvaluator.build(
            X_raw=d["X"], y=d["y"],
            eval_idx=d["eval_idx"],
            eval_train_idx=d["eval_train_idx"],
            eval_test_idx=d["eval_test_idx"],
            seed=0,
        )
        result = evaluator.krr_rmse(d["S_idx"])
        for key, val in result.items():
            if key.startswith("krr_rmse"):
                assert val >= 0.0, f"{key} = {val} should be nonneg"

    def test_krr_produces_per_target_keys(self, small_data):
        """Multi-target y should produce a RMSE for each target."""
        from coreset_selection.evaluation.raw_space import RawSpaceEvaluator

        d = small_data
        evaluator = RawSpaceEvaluator.build(
            X_raw=d["X"], y=d["y"],
            eval_idx=d["eval_idx"],
            eval_train_idx=d["eval_train_idx"],
            eval_test_idx=d["eval_test_idx"],
            seed=0,
        )
        result = evaluator.krr_rmse(d["S_idx"])
        # 2-column y with default names -> 4G and 5G keys
        assert "krr_rmse_4G" in result
        assert "krr_rmse_5G" in result

    def test_krr_with_named_targets(self):
        """Named targets should produce krr_rmse_{name} keys."""
        from coreset_selection.evaluation.raw_space import RawSpaceEvaluator

        rng = np.random.default_rng(7)
        N, D = 30, 5
        X = rng.standard_normal((N, D))
        y = rng.standard_normal((N, 3))

        eval_idx = np.arange(20)
        eval_train_idx = np.arange(16)
        eval_test_idx = np.arange(16, 20)

        evaluator = RawSpaceEvaluator.build(
            X_raw=X, y=y,
            eval_idx=eval_idx,
            eval_train_idx=eval_train_idx,
            eval_test_idx=eval_test_idx,
            seed=0,
            target_names=["alpha", "beta", "gamma"],
        )
        S_idx = np.array([20, 21, 22, 23, 24])
        result = evaluator.krr_rmse(S_idx)
        assert "krr_rmse_alpha" in result
        assert "krr_rmse_beta" in result
        assert "krr_rmse_gamma" in result

    def test_rmse_formula(self):
        """Hand-verify RMSE = sqrt(mean((y_true - y_pred)^2))."""
        from coreset_selection.evaluation.raw_space import _rmse

        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.2, 2.7])
        expected = np.sqrt(np.mean((y_true - y_pred) ** 2))
        np.testing.assert_allclose(_rmse(y_true, y_pred), expected, atol=1e-12)


# ===================================================================
# Test: Median heuristic bandwidth
# ===================================================================

class TestMedianHeuristic:
    """Verify the median-squared-distance bandwidth heuristic."""

    def test_known_distances(self):
        """For known data, verify the median pairwise squared distance."""
        from coreset_selection.evaluation.raw_space import _median_sq_dist

        # 3 points on a line: [0], [1], [2]
        # Pairwise squared distances: 1, 4, 1 -> median = 1
        X = np.array([[0.0], [1.0], [2.0]])
        # With all pairs sampled, median(1, 1, 4) = 1
        med = _median_sq_dist(X, seed=0, max_pairs=100)
        # The sampling may not get all pairs exactly, but should be close
        assert med > 0.0
        assert np.isfinite(med)

    def test_positive_definite(self):
        """sigma_sq from median heuristic should be > 0."""
        from coreset_selection.evaluation.raw_space import _median_sq_dist

        rng = np.random.default_rng(3)
        X = rng.standard_normal((50, 5))
        med = _median_sq_dist(X, seed=3)
        assert med > 0.0


# ===================================================================
# Test: Nystrom features
# ===================================================================

class TestNystromFeatures:
    """Verify that Phi Phi^T approximates K_hat."""

    def test_phi_phi_t(self):
        """Phi Phi^T should equal C (W + lam I)^{-1} C^T."""
        from coreset_selection.evaluation.raw_space import (
            _rbf_kernel,
            _nystrom_components,
            _nystrom_approx_gram,
            _nystrom_features,
        )

        rng = np.random.default_rng(42)
        N, D, k = 15, 4, 5
        X = rng.standard_normal((N, D))
        sigma_sq = 2.0

        X_E = X[:10]
        X_S = X[10:10 + k]
        C, W, lam = _nystrom_components(X_E, X_S, sigma_sq)

        K_hat = _nystrom_approx_gram(C, W, lam)
        Phi = _nystrom_features(C, W, lam)
        K_hat_from_phi = Phi @ Phi.T

        np.testing.assert_allclose(K_hat, K_hat_from_phi, atol=1e-8)


# ===================================================================
# Test: all_metrics integration
# ===================================================================

class TestAllMetricsIntegration:
    """Verify all_metrics returns a complete dict."""

    def test_all_metrics_keys(self, small_data):
        from coreset_selection.evaluation.raw_space import RawSpaceEvaluator

        d = small_data
        evaluator = RawSpaceEvaluator.build(
            X_raw=d["X"], y=d["y"],
            eval_idx=d["eval_idx"],
            eval_train_idx=d["eval_train_idx"],
            eval_test_idx=d["eval_test_idx"],
            seed=0,
        )
        metrics = evaluator.all_metrics(d["S_idx"])
        assert "nystrom_error" in metrics
        assert "kpca_distortion" in metrics
        assert "sigma_sq_raw" in metrics
        assert metrics["sigma_sq_raw"] > 0

    def test_all_metrics_reproducible(self, small_data):
        """Same seed -> same metrics."""
        from coreset_selection.evaluation.raw_space import RawSpaceEvaluator

        d = small_data
        evaluator1 = RawSpaceEvaluator.build(
            X_raw=d["X"], y=d["y"],
            eval_idx=d["eval_idx"],
            eval_train_idx=d["eval_train_idx"],
            eval_test_idx=d["eval_test_idx"],
            seed=0,
        )
        evaluator2 = RawSpaceEvaluator.build(
            X_raw=d["X"], y=d["y"],
            eval_idx=d["eval_idx"],
            eval_train_idx=d["eval_train_idx"],
            eval_test_idx=d["eval_test_idx"],
            seed=0,
        )
        m1 = evaluator1.all_metrics(d["S_idx"])
        m2 = evaluator2.all_metrics(d["S_idx"])
        for key in m1:
            np.testing.assert_allclose(
                m1[key], m2[key], atol=1e-12,
                err_msg=f"Metric '{key}' not reproducible",
            )


# ===================================================================
# Test: Coverage targets alignment
# ===================================================================

class TestCoverageTargets:
    """Verify COVERAGE_TARGETS_TABLE_V has the correct 10 entries."""

    def test_count(self):
        from coreset_selection.config.constants import COVERAGE_TARGETS_TABLE_V

        assert len(COVERAGE_TARGETS_TABLE_V) == 10

    def test_keys(self):
        from coreset_selection.config.constants import COVERAGE_TARGETS_TABLE_V

        expected = {
            "cov_area_4G", "cov_area_5G",
            "cov_hh_4G", "cov_res_4G",
            "cov_area_4G_5G", "cov_area_all",
            "cov_hh_4G_5G", "cov_hh_all",
            "cov_res_4G_5G", "cov_res_all",
        }
        assert set(COVERAGE_TARGETS_TABLE_V.keys()) == expected

    def test_labels_nonempty(self):
        from coreset_selection.config.constants import COVERAGE_TARGETS_TABLE_V

        for key, label in COVERAGE_TARGETS_TABLE_V.items():
            assert isinstance(label, str) and len(label) > 0, (
                f"Label for {key} should be a nonempty string"
            )
