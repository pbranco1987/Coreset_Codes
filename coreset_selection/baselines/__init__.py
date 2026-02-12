"""
Baseline coreset selection methods.

This module provides various baseline methods for comparison with
the multi-objective Pareto optimization approach.

Methods included:
- Uniform random sampling
- K-means cluster representatives
- Kernel herding (MMD minimization)
- Farthest-first traversal (k-center)
- Ridge leverage score sampling
- Determinantal point processes (k-DPP)

Each method has both unconstrained and quota-constrained variants.
"""

from .uniform import (
    baseline_uniform,
    baseline_uniform_quota,
    baseline_uniform_stratified,
    baseline_uniform_population_weighted,
)

from .kmeans import (
    baseline_kmeans_reps,
    baseline_kmeans_reps_quota,
    baseline_kmeans_plusplus,
)

from .herding import (
    kernel_herding_rff,
    baseline_kernel_herding,
    baseline_kernel_herding_quota,
    baseline_herding_global_then_quota,
)

from .farthest_first import (
    baseline_farthest_first,
    baseline_farthest_first_quota,
    baseline_farthest_first_global_then_quota,
    kcenter_cost,
)

from .leverage import (
    baseline_rls,
    baseline_rls_from_phi,
    baseline_rls_quota,
    baseline_rls_local_quota,
    compute_effective_dimension,
    optimal_rls_sample_size,
)

from .dpp import (
    baseline_dpp,
    greedy_kdpp_from_features,
    baseline_dpp_quota,
    sample_exact_kdpp,
)

from .kernel_thinning import (
    baseline_kernel_thinning,
    baseline_kernel_thinning_quota,
)

from .kkmeans_nystrom import (
    baseline_kkmeans_nystrom,
    baseline_kkmeans_nystrom_quota,
)

from .utils import (
    weighted_sample_without_replacement,
    rff_features,
    ridge_leverage_scores_from_features,
    quota_sample,
    ensure_quota_feasible,
)

from .variant_generator import (
    BaselineVariantGenerator,
    BaselineResult,
    METHOD_REGISTRY,
    VARIANT_PAIRS,
)

__all__ = [
    # Uniform
    "baseline_uniform",
    "baseline_uniform_quota",
    "baseline_uniform_stratified",
    "baseline_uniform_population_weighted",
    # K-means
    "baseline_kmeans_reps",
    "baseline_kmeans_reps_quota",
    "baseline_kmeans_plusplus",
    # Herding
    "kernel_herding_rff",
    "baseline_kernel_herding",
    "baseline_kernel_herding_quota",
    "baseline_herding_global_then_quota",
    # Farthest-first
    "baseline_farthest_first",
    "baseline_farthest_first_quota",
    "baseline_farthest_first_global_then_quota",
    "kcenter_cost",
    # Leverage
    "baseline_rls",
    "baseline_rls_from_phi",
    "baseline_rls_quota",
    "baseline_rls_local_quota",
    "compute_effective_dimension",
    "optimal_rls_sample_size",
    # DPP
    "baseline_dpp",
    "greedy_kdpp_from_features",
    "baseline_dpp_quota",
    "sample_exact_kdpp",
    # Kernel thinning
    "baseline_kernel_thinning",
    "baseline_kernel_thinning_quota",
    # KKM-Sampling Nyström
    "baseline_kkmeans_nystrom",
    "baseline_kkmeans_nystrom_quota",
    # Utils
    "weighted_sample_without_replacement",
    "rff_features",
    "ridge_leverage_scores_from_features",
    "quota_sample",
    "ensure_quota_feasible",
    # G7: Structured variant generator
    "BaselineVariantGenerator",
    "BaselineResult",
    "METHOD_REGISTRY",
    "VARIANT_PAIRS",
]


# Convenience dict mapping method names to functions
BASELINE_METHODS = {
    "uniform": baseline_uniform,
    "uniform_quota": baseline_uniform_quota,
    "uniform_stratified": baseline_uniform_stratified,
    "kmeans": baseline_kmeans_reps,
    "kmeans_quota": baseline_kmeans_reps_quota,
    "kmeans_plusplus": baseline_kmeans_plusplus,
    "herding": baseline_kernel_herding,
    "herding_quota": baseline_kernel_herding_quota,
    "farthest_first": baseline_farthest_first,
    "farthest_first_quota": baseline_farthest_first_quota,
    "rls": baseline_rls,
    "rls_quota": baseline_rls_quota,
    "dpp": baseline_dpp,
    "dpp_quota": baseline_dpp_quota,
    "kernel_thinning": baseline_kernel_thinning,
    "kernel_thinning_quota": baseline_kernel_thinning_quota,
    "kkmeans_nystrom": baseline_kkmeans_nystrom,
    "kkmeans_nystrom_quota": baseline_kkmeans_nystrom_quota,
}


def get_baseline_method(name: str):
    """
    Get baseline method by name.
    
    Parameters
    ----------
    name : str
        Method name (e.g., "uniform", "kmeans_quota")
        
    Returns
    -------
    Callable
        The baseline function
        
    Raises
    ------
    KeyError
        If method name not found
    """
    if name not in BASELINE_METHODS:
        raise KeyError(
            f"Unknown baseline method: {name}. "
            f"Available: {list(BASELINE_METHODS.keys())}"
        )
    return BASELINE_METHODS[name]
