"""
Tests for objective functions: SKL, MMD, Sinkhorn.

Verifies:
- SKL symmetry and correctness
- MMD properties (zero for same set)
- Sinkhorn debiasing (SD(P,P) = 0)
- Variance clamping bounds
- RFF approximation quality
"""

import numpy as np
import pytest

from coreset_selection.objectives.skl import (
    symmetric_kl_diag_gaussians,
    kl_diag_gaussians,
    clamp_variance,
    compute_moment_matched_gaussian,
    VAE_VARIANCE_CLAMP_MIN,
    VAE_VARIANCE_CLAMP_MAX,
)
from coreset_selection.objectives.mmd import (
    RFFMMD,
    compute_rff_features,
    mmd_from_rff,
    mmd2_exact,
)


class TestSKL:
    """Tests for Symmetric KL divergence."""

    def test_skl_symmetric(self):
        """SKL(A, B) should equal SKL(B, A)."""
        rng = np.random.default_rng(42)
        
        for _ in range(50):
            d = rng.integers(5, 20)
            mu1 = rng.normal(size=d)
            var1 = np.abs(rng.normal(size=d)) + 0.1
            mu2 = rng.normal(size=d)
            var2 = np.abs(rng.normal(size=d)) + 0.1
            
            skl_12 = symmetric_kl_diag_gaussians(mu1, var1, mu2, var2)
            skl_21 = symmetric_kl_diag_gaussians(mu2, var2, mu1, var1)
            
            assert np.isclose(skl_12, skl_21, rtol=1e-10)

    def test_skl_zero_for_same(self):
        """SKL(A, A) should be 0."""
        rng = np.random.default_rng(42)
        
        for _ in range(20):
            d = rng.integers(5, 20)
            mu = rng.normal(size=d)
            var = np.abs(rng.normal(size=d)) + 0.1
            
            skl = symmetric_kl_diag_gaussians(mu, var, mu, var)
            
            assert np.isclose(skl, 0.0, atol=1e-10)

    def test_skl_non_negative(self):
        """SKL should always be non-negative."""
        rng = np.random.default_rng(42)
        
        for _ in range(100):
            d = rng.integers(5, 20)
            mu1 = rng.normal(size=d)
            var1 = np.abs(rng.normal(size=d)) + 0.1
            mu2 = rng.normal(size=d)
            var2 = np.abs(rng.normal(size=d)) + 0.1
            
            skl = symmetric_kl_diag_gaussians(mu1, var1, mu2, var2)
            
            assert skl >= -1e-10

    def test_variance_clamping_bounds(self):
        """Variance clamping should respect manuscript bounds."""
        # Check bounds
        assert np.isclose(VAE_VARIANCE_CLAMP_MIN, np.exp(-10))
        assert np.isclose(VAE_VARIANCE_CLAMP_MAX, np.exp(2))
        
        # Check clamping function
        var = np.array([1e-10, 0.1, 1.0, 10.0, 100.0])
        clamped = clamp_variance(var)
        
        assert np.all(clamped >= VAE_VARIANCE_CLAMP_MIN)
        assert np.all(clamped <= VAE_VARIANCE_CLAMP_MAX)

    def test_moment_matched_gaussian(self):
        """Moment matching should preserve E[z] and Var[z]."""
        rng = np.random.default_rng(42)
        
        n = 100
        d = 10
        mu_all = rng.normal(size=(n, d))
        var_all = np.abs(rng.normal(size=(n, d))) + 0.1
        
        m, v = compute_moment_matched_gaussian(mu_all, var_all)
        
        # Mean should be mean of means
        expected_m = np.mean(mu_all, axis=0)
        assert np.allclose(m, expected_m, rtol=1e-10)
        
        # Variance should be mean of variances + variance of means
        var_all_clamped = clamp_variance(var_all)
        expected_v = np.mean(var_all_clamped, axis=0) + np.var(mu_all, axis=0, ddof=0)
        assert np.allclose(v, expected_v, rtol=1e-10)


class TestMMD:
    """Tests for Maximum Mean Discrepancy."""

    def test_mmd_zero_for_same_set(self):
        """MMD(P, P) should be approximately 0."""
        rng = np.random.default_rng(42)
        n, d = 100, 10
        X = rng.normal(size=(n, d))
        sigma_sq = 1.0
        
        mmd_estimator = RFFMMD.build(X, rff_dim=2000, sigma_sq=sigma_sq, seed=42)
        
        # Full set vs full set
        idx_all = np.arange(n)
        mmd2 = mmd_estimator.mmd2_subset(idx_all)
        
        # Should be close to 0 (some variance from RFF)
        assert mmd2 < 0.01

    def test_mmd_non_negative(self):
        """MMD² should always be non-negative."""
        rng = np.random.default_rng(42)
        n, d = 100, 10
        X = rng.normal(size=(n, d))
        sigma_sq = 1.0
        
        mmd_estimator = RFFMMD.build(X, rff_dim=2000, sigma_sq=sigma_sq, seed=42)
        
        for _ in range(50):
            k = rng.integers(10, 50)
            idx = rng.choice(n, size=k, replace=False)
            mmd2 = mmd_estimator.mmd2_subset(idx)
            
            assert mmd2 >= -1e-10

    def test_mmd_from_rff_function(self):
        """Test standalone mmd_from_rff function."""
        rng = np.random.default_rng(42)
        n, d = 100, 10
        X = rng.normal(size=(n, d))
        sigma_sq = 1.0
        rff_dim = 2000
        
        Phi = compute_rff_features(X, rff_dim, sigma_sq, seed=42)
        
        # Compare with RFFMMD class
        mmd_estimator = RFFMMD.build(X, rff_dim=rff_dim, sigma_sq=sigma_sq, seed=42)
        
        for _ in range(10):
            k = rng.integers(10, 50)
            idx = rng.choice(n, size=k, replace=False)
            
            mmd_fn = mmd_from_rff(Phi, idx)
            mmd_class = mmd_estimator.mmd2_subset(idx)
            
            # Should give same result (same RFF)
            # Note: may differ slightly due to different mean computation
            assert np.isclose(mmd_fn, mmd_class, rtol=0.1)

    def test_rff_approximation_quality(self):
        """RFF-MMD should approximate exact MMD."""
        rng = np.random.default_rng(42)
        n, d = 50, 5  # Small for exact computation
        X = rng.normal(size=(n, d))
        sigma_sq = 1.0
        
        # Split into two sets
        idx1 = np.arange(n // 2)
        idx2 = np.arange(n // 2, n)
        
        # Exact MMD
        mmd_exact = mmd2_exact(X[idx1], X[idx2], sigma_sq)
        
        # RFF MMD with many features should be close
        mmd_estimator = RFFMMD.build(X, rff_dim=4000, sigma_sq=sigma_sq, seed=42)
        mmd_rff = mmd_estimator.mmd2_between_subsets(idx1, idx2)
        
        # Should be in same ballpark (RFF is an approximation)
        assert np.abs(mmd_exact - mmd_rff) < 0.5 * max(mmd_exact, mmd_rff) + 0.01


class TestSinkhorn:
    """Tests for Sinkhorn divergence."""

    def test_sinkhorn_debiased_zero_for_same(self):
        """Debiased Sinkhorn SD(P, P) should be 0.
        
        NOTE: This test uses n_anchors = n (exact anchors) to verify the
        core Sinkhorn divergence identity SD(P, P) = 0. When n_anchors < n,
        the implementation uses an anchored surrogate that computes
        SD(P̃, Q) where P̃ is the anchor distribution, so SD(full, full)
        would not be ~0.
        """
        from coreset_selection.objectives.sinkhorn import AnchorSinkhorn
        from coreset_selection.config.dataclasses import SinkhornConfig
        
        rng = np.random.default_rng(42)
        # Use smaller n for faster test execution
        n, d = 50, 10
        X = rng.normal(size=(n, d))
        
        # Use all points as anchors to test the true SD(P,P)=0 property
        # Use random anchor method to avoid sklearn import overhead
        cfg = SinkhornConfig(n_anchors=n, eta=0.05, max_iter=50, anchor_method="random")
        sink = AnchorSinkhorn.build(X, cfg, seed=42)
        
        # Full set vs full set
        idx_all = np.arange(n)
        sd = sink.sinkhorn_divergence_subset(X, idx_all)
        
        # Should be close to 0 when using exact anchors
        assert sd < 1e-3

    def test_sinkhorn_anchored_surrogate_relative(self):
        """Test that anchored surrogate behaves sensibly (relative property).
        
        When n_anchors < n, SD(full, full) should be <= SD(random_subset, full)
        since the full set is the best approximation to the full distribution.
        """
        from coreset_selection.objectives.sinkhorn import AnchorSinkhorn
        from coreset_selection.config.dataclasses import SinkhornConfig
        
        rng = np.random.default_rng(42)
        n, d = 100, 10
        X = rng.normal(size=(n, d))
        
        # Use fewer anchors than data points (anchored surrogate)
        cfg = SinkhornConfig(n_anchors=50, eta=0.05, max_iter=100, anchor_method="random")
        sink = AnchorSinkhorn.build(X, cfg, seed=42)
        
        # Full set should have lower (or equal) divergence than random subset
        idx_full = np.arange(n)
        sd_full = sink.sinkhorn_divergence_subset(X, idx_full)
        
        idx_rand = rng.choice(n, size=20, replace=False)
        sd_rand = sink.sinkhorn_divergence_subset(X, idx_rand)
        
        assert sd_full <= sd_rand + 1e-6  # Allow small numerical tolerance
        assert sd_full >= 0  # Should be non-negative

    def test_sinkhorn_non_negative(self):
        """Sinkhorn divergence should be non-negative."""
        from coreset_selection.objectives.sinkhorn import AnchorSinkhorn
        from coreset_selection.config.dataclasses import SinkhornConfig
        
        rng = np.random.default_rng(42)
        n, d = 100, 10
        X = rng.normal(size=(n, d))
        
        cfg = SinkhornConfig(n_anchors=50, eta=0.05, max_iter=100)
        sink = AnchorSinkhorn.build(X, cfg, seed=42)
        
        for _ in range(20):
            k = rng.integers(10, 50)
            idx = rng.choice(n, size=k, replace=False)
            sd = sink.sinkhorn_divergence_subset(X, idx)
            
            assert sd >= -1e-10

    def test_anchor_weights_sum_to_one(self):
        """Anchor weights should sum to 1."""
        from coreset_selection.objectives.sinkhorn import AnchorSinkhorn
        from coreset_selection.config.dataclasses import SinkhornConfig
        
        rng = np.random.default_rng(42)
        n, d = 100, 10
        X = rng.normal(size=(n, d))
        
        cfg = SinkhornConfig(n_anchors=50, eta=0.05, max_iter=100, anchor_method="kmeans")
        sink = AnchorSinkhorn.build(X, cfg, seed=42)
        
        assert np.isclose(sink.weights.sum(), 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
