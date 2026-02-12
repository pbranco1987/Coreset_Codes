"""
Tests for evaluation metrics: Nyström, kPCA, KRR.

Verifies:
- Nyström approximation quality
- kPCA distortion bounds
- KRR RMSE computation
"""

import numpy as np
import pytest


class TestNystromApproximation:
    """Tests for Nyström Gram-matrix approximation."""

    def test_nystrom_approximation_quality(self):
        """Large k should give small Nyström error."""
        from coreset_selection.evaluation.raw_space import RawSpaceEvaluator
        
        rng = np.random.default_rng(42)
        N = 200
        d = 10
        X = rng.normal(size=(N, d))
        
        # Create evaluation split
        eval_size = 100
        eval_idx = rng.choice(N, size=eval_size, replace=False)
        train_size = int(0.8 * eval_size)
        train_idx = eval_idx[:train_size]
        test_idx = eval_idx[train_size:]
        
        evaluator = RawSpaceEvaluator.build(
            X_raw=X,
            y=None,
            eval_idx=eval_idx,
            eval_train_idx=train_idx,
            eval_test_idx=test_idx,
            seed=42,
        )
        
        # Test with increasing k
        errors = []
        for k in [10, 20, 50]:
            S_idx = rng.choice(N, size=k, replace=False)
            e_nys = evaluator.nystrom_error(S_idx)
            errors.append(e_nys)
        
        # Error should generally decrease with larger k
        # (not strictly monotonic due to random selection)
        assert errors[0] > errors[-1] * 0.5 or errors[-1] < 0.5

    def test_nystrom_error_bounds(self):
        """Nyström error should be in [0, some_reasonable_bound]."""
        from coreset_selection.evaluation.raw_space import RawSpaceEvaluator
        
        rng = np.random.default_rng(42)
        N = 100
        d = 5
        X = rng.normal(size=(N, d))
        
        eval_idx = np.arange(50)
        train_idx = np.arange(40)
        test_idx = np.arange(40, 50)
        
        evaluator = RawSpaceEvaluator.build(
            X_raw=X,
            y=None,
            eval_idx=eval_idx,
            eval_train_idx=train_idx,
            eval_test_idx=test_idx,
            seed=42,
        )
        
        for _ in range(20):
            k = rng.integers(5, 30)
            S_idx = rng.choice(N, size=k, replace=False)
            e_nys = evaluator.nystrom_error(S_idx)
            
            assert 0 <= e_nys <= 2.0  # Reasonable upper bound


class TestKPCADistortion:
    """Tests for kernel PCA spectral distortion."""

    def test_kpca_distortion_bounds(self):
        """kPCA distortion should be bounded."""
        from coreset_selection.evaluation.raw_space import RawSpaceEvaluator
        
        rng = np.random.default_rng(42)
        N = 100
        d = 5
        X = rng.normal(size=(N, d))
        
        eval_idx = np.arange(50)
        train_idx = np.arange(40)
        test_idx = np.arange(40, 50)
        
        evaluator = RawSpaceEvaluator.build(
            X_raw=X,
            y=None,
            eval_idx=eval_idx,
            eval_train_idx=train_idx,
            eval_test_idx=test_idx,
            seed=42,
        )
        
        for _ in range(20):
            k = rng.integers(5, 30)
            S_idx = rng.choice(N, size=k, replace=False)
            e_kpca = evaluator.kpca_distortion(S_idx, r=10)
            
            # Distortion should be non-negative
            assert e_kpca >= 0

    def test_kpca_improves_with_larger_k(self):
        """kPCA distortion should generally improve with larger k."""
        from coreset_selection.evaluation.raw_space import RawSpaceEvaluator
        
        rng = np.random.default_rng(42)
        N = 200
        d = 10
        X = rng.normal(size=(N, d))
        
        eval_idx = rng.choice(N, size=100, replace=False)
        train_idx = eval_idx[:80]
        test_idx = eval_idx[80:]
        
        evaluator = RawSpaceEvaluator.build(
            X_raw=X,
            y=None,
            eval_idx=eval_idx,
            eval_train_idx=train_idx,
            eval_test_idx=test_idx,
            seed=42,
        )
        
        # Sample with different sizes and check trend
        small_errors = []
        large_errors = []
        
        for _ in range(10):
            S_small = rng.choice(N, size=10, replace=False)
            S_large = rng.choice(N, size=50, replace=False)
            
            small_errors.append(evaluator.kpca_distortion(S_small, r=10))
            large_errors.append(evaluator.kpca_distortion(S_large, r=10))
        
        # Larger k should have lower error on average
        assert np.mean(large_errors) < np.mean(small_errors)


class TestKRR:
    """Tests for Kernel Ridge Regression evaluation."""

    def test_krr_rmse_computation(self):
        """KRR RMSE should be computed correctly."""
        from coreset_selection.evaluation.raw_space import RawSpaceEvaluator
        
        rng = np.random.default_rng(42)
        N = 200
        d = 10
        X = rng.normal(size=(N, d))
        
        # Generate synthetic targets
        true_w = rng.normal(size=d)
        y = X @ true_w + 0.1 * rng.normal(size=N)
        
        eval_idx = rng.choice(N, size=100, replace=False)
        train_idx = eval_idx[:80]
        test_idx = eval_idx[80:]
        
        evaluator = RawSpaceEvaluator.build(
            X_raw=X,
            y=y.reshape(-1, 1),
            eval_idx=eval_idx,
            eval_train_idx=train_idx,
            eval_test_idx=test_idx,
            seed=42,
        )
        
        # Compute KRR RMSE
        S_idx = rng.choice(N, size=30, replace=False)
        result = evaluator.krr_rmse(S_idx)
        
        assert "krr_rmse" in result
        assert result["krr_rmse"] >= 0

    def test_krr_lambda_selection(self):
        """KRR should select reasonable lambda via CV."""
        from coreset_selection.evaluation.raw_space import RawSpaceEvaluator
        
        rng = np.random.default_rng(42)
        N = 200
        d = 10
        X = rng.normal(size=(N, d))
        y = rng.normal(size=N)  # Random targets
        
        eval_idx = rng.choice(N, size=100, replace=False)
        train_idx = eval_idx[:80]
        test_idx = eval_idx[80:]
        
        evaluator = RawSpaceEvaluator.build(
            X_raw=X,
            y=y.reshape(-1, 1),
            eval_idx=eval_idx,
            eval_train_idx=train_idx,
            eval_test_idx=test_idx,
            seed=42,
        )
        
        S_idx = rng.choice(N, size=30, replace=False)
        result = evaluator.krr_rmse(S_idx)
        
        assert "krr_lambda" in result
        # Lambda should be in the grid range
        assert 1e-7 <= result["krr_lambda"] <= 1e7


class TestAllMetrics:
    """Tests for all_metrics aggregation."""

    def test_all_metrics_returns_complete(self):
        """all_metrics should return all enabled metrics."""
        from coreset_selection.evaluation.raw_space import RawSpaceEvaluator
        
        rng = np.random.default_rng(42)
        N = 100
        d = 5
        X = rng.normal(size=(N, d))
        y = rng.normal(size=(N, 2))  # Two targets
        
        eval_idx = np.arange(50)
        train_idx = np.arange(40)
        test_idx = np.arange(40, 50)
        
        evaluator = RawSpaceEvaluator.build(
            X_raw=X,
            y=y,
            eval_idx=eval_idx,
            eval_train_idx=train_idx,
            eval_test_idx=test_idx,
            seed=42,
        )
        
        S_idx = rng.choice(N, size=20, replace=False)
        metrics = evaluator.all_metrics(S_idx)
        
        # Should have all expected keys
        assert "nystrom_error" in metrics
        assert "kpca_distortion" in metrics
        assert "krr_rmse_4G" in metrics or "krr_rmse" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
