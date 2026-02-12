"""
Symmetric KL divergence for diagonal Gaussian distributions.

Implements the SKL objective from manuscript Section 4.4.1:
- f_SKL(s) = SKL(G_N, G_S) = KL(G_N || G_S) + KL(G_S || G_N)

Where G_N and G_S are moment-matched diagonal Gaussian summaries of the
population and subset VAE posteriors respectively.

Contains:
- symmetric_kl_diag_gaussians: Compute SKL between two diagonal Gaussians
- clamp_variance: Apply manuscript-compliant variance clamping
- VAE_VARIANCE_CLAMP_MIN/MAX: Clamping bounds from manuscript §5.8.3
"""

from __future__ import annotations

import numpy as np


# Variance clamping bounds per manuscript Section 5.8.3 (lines 1586-1590)
# "We clamp encoder variances element-wise to Σ_ij ∈ [exp(-10), exp(2)]"
VAE_VARIANCE_CLAMP_MIN = np.exp(-10)  # ≈ 4.54e-5
VAE_VARIANCE_CLAMP_MAX = np.exp(2)     # ≈ 7.389


def clamp_variance(var: np.ndarray) -> np.ndarray:
    """
    Apply manuscript-compliant variance clamping for SKL stability.
    
    Per manuscript Section 5.8.3:
    "We clamp encoder variances element-wise to Σ_ij ∈ [exp(-10), exp(2)]
    before constructing G_N and G_S and computing f_SKL."
    
    Parameters
    ----------
    var : np.ndarray
        Variance array to clamp
        
    Returns
    -------
    np.ndarray
        Clamped variance array
    """
    return np.clip(var, VAE_VARIANCE_CLAMP_MIN, VAE_VARIANCE_CLAMP_MAX)


def symmetric_kl_diag_gaussians(
    mu_p: np.ndarray,
    var_p: np.ndarray,
    mu_q: np.ndarray,
    var_q: np.ndarray,
    apply_clamping: bool = True,
) -> float:
    """
    Compute symmetric KL divergence between two diagonal Gaussians.
    
    Implements manuscript Equation following line 636:
    SKL(·) = 0.5 * Σ_j [ v_{1j}/v_{2j} + v_{2j}/v_{1j} - 2 + (Δm_j)²(1/v_{1j} + 1/v_{2j}) ]
    
    This is equivalent to KL(P||Q) + KL(Q||P).
    
    Parameters
    ----------
    mu_p : np.ndarray
        Mean of distribution P (G_N), shape (d,)
    var_p : np.ndarray
        Variance of distribution P (G_N), shape (d,)
    mu_q : np.ndarray
        Mean of distribution Q (G_S), shape (d,)
    var_q : np.ndarray
        Variance of distribution Q (G_S), shape (d,)
    apply_clamping : bool
        Whether to apply manuscript variance clamping (default True)
        
    Returns
    -------
    float
        Symmetric KL divergence
    """
    mu_p = np.asarray(mu_p, dtype=np.float64)
    var_p = np.asarray(var_p, dtype=np.float64)
    mu_q = np.asarray(mu_q, dtype=np.float64)
    var_q = np.asarray(var_q, dtype=np.float64)
    
    # Apply manuscript-compliant variance clamping (Section 5.8.3)
    if apply_clamping:
        var_p = clamp_variance(var_p)
        var_q = clamp_variance(var_q)
    else:
        # Fallback: ensure positive variances
        var_p = np.maximum(var_p, 1e-12)
        var_q = np.maximum(var_q, 1e-12)
    
    d = len(mu_p)
    
    # Mean difference squared
    diff_sq = (mu_p - mu_q) ** 2
    
    # Direct formula from manuscript (line 636-641):
    # SKL = 0.5 * Σ_j [ v_{pj}/v_{qj} + v_{qj}/v_{pj} - 2 + (Δm_j)²(1/v_{pj} + 1/v_{qj}) ]
    skl = 0.5 * np.sum(
        var_p / var_q + var_q / var_p - 2.0 +
        diff_sq * (1.0 / var_p + 1.0 / var_q)
    )
    
    return float(skl)


def kl_diag_gaussians(
    mu_p: np.ndarray,
    var_p: np.ndarray,
    mu_q: np.ndarray,
    var_q: np.ndarray,
    apply_clamping: bool = True,
) -> float:
    """
    Compute KL divergence KL(P || Q) between two diagonal Gaussians.
    
    Implements manuscript Equation (lines 626-634):
    KL(N(m1,diag(v1)) || N(m2,diag(v2))) = 
        0.5 * Σ_j [ log(v_{2j}/v_{1j}) + v_{1j}/v_{2j} + (Δm_j)²/v_{2j} - 1 ]
    
    Parameters
    ----------
    mu_p : np.ndarray
        Mean of distribution P, shape (d,)
    var_p : np.ndarray
        Variance of distribution P, shape (d,)
    mu_q : np.ndarray
        Mean of distribution Q, shape (d,)
    var_q : np.ndarray
        Variance of distribution Q, shape (d,)
    apply_clamping : bool
        Whether to apply manuscript variance clamping (default True)
        
    Returns
    -------
    float
        KL divergence KL(P || Q)
    """
    mu_p = np.asarray(mu_p, dtype=np.float64)
    var_p = np.asarray(var_p, dtype=np.float64)
    mu_q = np.asarray(mu_q, dtype=np.float64)
    var_q = np.asarray(var_q, dtype=np.float64)
    
    # Apply manuscript-compliant variance clamping
    if apply_clamping:
        var_p = clamp_variance(var_p)
        var_q = clamp_variance(var_q)
    else:
        var_p = np.maximum(var_p, 1e-12)
        var_q = np.maximum(var_q, 1e-12)
    
    diff_sq = (mu_p - mu_q) ** 2
    
    kl = 0.5 * np.sum(
        np.log(var_q / var_p) + var_p / var_q + diff_sq / var_q - 1.0
    )
    
    return float(kl)


def jeffreys_divergence_diag_gaussians(
    mu_p: np.ndarray,
    var_p: np.ndarray,
    mu_q: np.ndarray,
    var_q: np.ndarray,
    apply_clamping: bool = True,
) -> float:
    """
    Compute Jeffreys divergence (symmetrized KL) between diagonal Gaussians.
    
    J(P, Q) = KL(P || Q) + KL(Q || P)
    
    Note: This equals symmetric_kl_diag_gaussians() since our SKL definition
    is already the sum of both KL terms.
    
    Parameters
    ----------
    mu_p, var_p : np.ndarray
        Parameters of distribution P
    mu_q, var_q : np.ndarray
        Parameters of distribution Q
    apply_clamping : bool
        Whether to apply manuscript variance clamping (default True)
        
    Returns
    -------
    float
        Jeffreys divergence
    """
    return symmetric_kl_diag_gaussians(mu_p, var_p, mu_q, var_q, apply_clamping)


def compute_moment_matched_gaussian(
    mu_all: np.ndarray,
    var_all: np.ndarray,
) -> tuple:
    """
    Compute moment-matched diagonal Gaussian from VAE posteriors.
    
    Implements manuscript Equations (lines 598-608):
    - m = (1/n) Σ_i μ_i
    - v = (1/n) Σ_i Σ_i + Var({μ_i})
    
    This matches E[z] and diag(Cov(z)) under the mixture and is the
    forward-KL information projection onto diagonal Gaussians.
    
    Parameters
    ----------
    mu_all : np.ndarray
        Posterior means, shape (n, d_z)
    var_all : np.ndarray
        Posterior variances, shape (n, d_z)
        
    Returns
    -------
    tuple
        (mean, variance) of moment-matched Gaussian, each shape (d_z,)
    """
    mu_all = np.asarray(mu_all, dtype=np.float64)
    var_all = np.asarray(var_all, dtype=np.float64)
    
    # Apply clamping to input variances
    var_all = clamp_variance(var_all)
    
    # Mean of means
    m = np.mean(mu_all, axis=0)
    
    # Mean of variances + variance of means
    mean_var = np.mean(var_all, axis=0)
    var_mu = np.var(mu_all, axis=0, ddof=0)  # Population variance
    v = mean_var + var_mu
    
    return m, v
