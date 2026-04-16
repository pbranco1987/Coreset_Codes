# API Reference — Baselines (`coreset_selection.baselines`)

The baselines package implements **8 competing coreset selection methods** used for championship comparisons. Each method has (a) an unconstrained variant that ignores geographic structure and (b) a quota-constrained variant that respects `c*(k)`. The manuscript further distinguishes SOFT-constraint variants (post-hoc swap-repair) implemented in `scripts/launchers/run_baselines.py`.

Public symbols are re-exported from `coreset_selection/baselines/__init__.py`.

## Baseline Protocol

Every baseline function in this package follows the same high-level signature shape:

```python
def baseline_<method>(
    X: np.ndarray,              # feature matrix, (N, d)
    k: int,                     # coreset size
    *,
    seed: int = 0,
    **method_specific_kwargs,
) -> np.ndarray                 # integer indices, shape (k,)
```

Quota-constrained variants additionally take `geo: GeoInfo` and (optionally) a pre-computed `target_counts: np.ndarray`.

The `BASELINE_METHODS` registry (dict) at the module level maps canonical method names to callable references for programmatic dispatch.

## Section Map

1. [Registry & Dispatch](#registry--dispatch)
2. [Uniform Sampling](#uniform-sampling)
3. [k-Means Representatives](#k-means-representatives)
4. [Kernel Herding](#kernel-herding)
5. [Farthest-First (k-Center)](#farthest-first-k-center)
6. [Ridge Leverage Scores](#ridge-leverage-scores)
7. [Determinantal Point Processes (k-DPP)](#determinantal-point-processes-k-dpp)
8. [Kernel Thinning](#kernel-thinning)
9. [Kernel k-Means with Nyström](#kernel-k-means-with-nyström)
10. [Utility Functions](#utility-functions)
11. [Variant Generator](#variant-generator)

---

## Registry & Dispatch

### `coreset_selection.baselines.BASELINE_METHODS`

**Kind:** constant (`Dict[str, Callable]`)
**Source:** `coreset_selection/baselines/__init__.py:143`

**Summary:** Map from canonical method name to baseline function.

**Example keys:** `"uniform"`, `"kmeans"`, `"herding"`, `"farthest_first"`, `"rls"`, `"dpp"`, `"kernel_thinning"`, `"kkmeans_nystrom"`, and their `"_quota"` variants.

### `coreset_selection.baselines.get_baseline_method`

**Kind:** function
**Source:** `coreset_selection/baselines/__init__.py:165`

**Summary:** Dispatch helper — look up a baseline function by name.

**Signature:** `def get_baseline_method(name: str) -> Callable`

**Raises:** `KeyError` if `name` is not registered.

**Example:**
```python
from coreset_selection.baselines import get_baseline_method
fn = get_baseline_method("kernel_thinning_quota")
S = fn(X, k=100, geo=geo, seed=4200)
```

---

## Uniform Sampling

### `coreset_selection.baselines.baseline_uniform`

**Kind:** function — Simple random subset of size `k`.

### `coreset_selection.baselines.baseline_uniform_quota`

**Kind:** function — Uniform sampling within each group, respecting `c*(k)`.

### `coreset_selection.baselines.baseline_uniform_stratified`

**Kind:** function — Stratified uniform sampling with proportional allocation (Hamilton method).

### `coreset_selection.baselines.baseline_uniform_population_weighted`

**Kind:** function — Weighted uniform sampling with population weights.

---

## k-Means Representatives

### `coreset_selection.baselines.baseline_kmeans_reps`

**Kind:** function — Run k-means on `X` and return the index of the nearest real point to each centroid.

### `coreset_selection.baselines.baseline_kmeans_reps_quota`

**Kind:** function — Per-group k-means with `c*(k)` clusters in each group.

### `coreset_selection.baselines.baseline_kmeans_plusplus`

**Kind:** function — k-means++ initialisation only (no Lloyd iterations), returning the chosen centres.

---

## Kernel Herding

### `coreset_selection.baselines.kernel_herding_rff`

**Kind:** function — Low-level kernel herding on a pre-computed RFF feature matrix.

**Description:** Implements the greedy MMD-minimising selection: point `t+1` maximises `Φ_i · ((t+1) · μ_full - Σ_{j∈S_t} Φ_j)`. Returns indices in selection order.

### `coreset_selection.baselines.baseline_kernel_herding`

**Kind:** function — High-level wrapper that computes RFF internally.

### `coreset_selection.baselines.baseline_kernel_herding_quota`

**Kind:** function — Per-group kernel herding respecting `c*(k)`.

### `coreset_selection.baselines.baseline_herding_global_then_quota`

**Kind:** function — Run global herding, then re-balance to satisfy the quota.

---

## Farthest-First (k-Center)

### `coreset_selection.baselines.baseline_farthest_first`

**Kind:** function — Gonzalez's farthest-first traversal (2-approximation to the k-center problem).

### `coreset_selection.baselines.baseline_farthest_first_quota`

**Kind:** function — Per-group farthest-first.

### `coreset_selection.baselines.baseline_farthest_first_global_then_quota`

**Kind:** function — Global farthest-first followed by quota rebalancing.

### `coreset_selection.baselines.kcenter_cost`

**Kind:** function — Compute the realised k-center objective of a given coreset (for reporting).

---

## Ridge Leverage Scores

### `coreset_selection.baselines.baseline_rls`

**Kind:** function — Sample proportional to ridge leverage scores computed on the full kernel matrix.

### `coreset_selection.baselines.baseline_rls_from_phi`

**Kind:** function — Compute RLS from a precomputed feature matrix `Φ` (the usual approach with RFFs).

### `coreset_selection.baselines.baseline_rls_quota`

**Kind:** function — RLS sampling per group to satisfy the quota.

### `coreset_selection.baselines.baseline_rls_local_quota`

**Kind:** function — RLS sampling where leverage is recomputed locally within each group.

### `coreset_selection.baselines.compute_effective_dimension`

**Kind:** function — Estimate the effective dimension `d_eff(λ)` used by RLS theory.

### `coreset_selection.baselines.optimal_rls_sample_size`

**Kind:** function — Theoretical sample size `k` required for a given approximation guarantee.

---

## Determinantal Point Processes (k-DPP)

### `coreset_selection.baselines.baseline_dpp`

**Kind:** function — Sample a size-`k` subset from a k-DPP defined by a kernel matrix.

### `coreset_selection.baselines.greedy_kdpp_from_features`

**Kind:** function — Greedy determinantal selection using RFF features (avoids forming the full kernel).

### `coreset_selection.baselines.baseline_dpp_quota`

**Kind:** function — Per-group k-DPP sampling.

### `coreset_selection.baselines.sample_exact_kdpp`

**Kind:** function — Exact k-DPP sampling via eigendecomposition (for small N).

---

## Kernel Thinning

### `coreset_selection.baselines.baseline_kernel_thinning`

**Kind:** function — Ramdas et al.'s compress/thin algorithm for MMD minimisation.

### `coreset_selection.baselines.baseline_kernel_thinning_quota`

**Kind:** function — Per-group kernel thinning.

---

## Kernel k-Means with Nyström

### `coreset_selection.baselines.baseline_kkmeans_nystrom`

**Kind:** function — Kernel k-means using Nyström approximation for scalability.

### `coreset_selection.baselines.baseline_kkmeans_nystrom_quota`

**Kind:** function — Per-group kernel k-means.

---

## Utility Functions

### `coreset_selection.baselines.weighted_sample_without_replacement`

**Kind:** function — Inverse-CDF sampling without replacement from an arbitrary weight vector.

### `coreset_selection.baselines.rff_features`

**Kind:** function — Compute random Fourier features (re-exported from `coreset_selection.objectives.mmd` but convenient here).

### `coreset_selection.baselines.ridge_leverage_scores_from_features`

**Kind:** function — Ridge leverage scores `τ_i(λ) = φ_i^⊤ (Σ + λI)^{-1} φ_i`.

### `coreset_selection.baselines.quota_sample`

**Kind:** function — Generic "apply a picker per group" driver.

### `coreset_selection.baselines.ensure_quota_feasible`

**Kind:** function — Sanity check that a proposed quota vector respects group capacities.

---

## Variant Generator

### `coreset_selection.baselines.BaselineVariantGenerator`

**Kind:** class — Structured generator emitting combinations of (method, quota_mode, space) for batch runs.

### `coreset_selection.baselines.BaselineResult`

**Kind:** dataclass — Return container for a single baseline run (indices, method name, wall-clock, etc.).

### `coreset_selection.baselines.METHOD_REGISTRY`

**Kind:** constant — Alternative dispatch dict (includes `BaselineVariantGenerator`-specific method entries).

### `coreset_selection.baselines.VARIANT_PAIRS`, `POP_QUOTA_PAIRS`, `JOINT_QUOTA_PAIRS`

**Kind:** constant — Named tuples / lists of method-variant pairs used by the experiment matrix.

---

## See Also

- [scripts/launchers/run_baselines.py](./scripts.md#run_baselines) — adds post-hoc soft-KL repair to any baseline output.
- [evaluation](./evaluation.md) — every baseline output is fed through `RawSpaceEvaluator` for fair comparison.
- Manuscript Table VI (baseline taxonomy), Section VI-D (championship methodology).
