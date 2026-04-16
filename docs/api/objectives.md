# API Reference — Objectives (`coreset_selection.objectives`)

The objectives package defines the **distributional divergences** that NSGA-II minimises. Every NSGA-II individual is scored by some subset of `{SKL, MMD², Sinkhorn, NystromLogDet}`. The unified `SpaceObjectiveComputer` is the canonical interface; individual classes are useful for ablations.

Public symbols are re-exported from `coreset_selection/objectives/__init__.py`.

## Section Map

1. [Unified Computer](#unified-computer)
2. [SKL (Symmetric KL)](#skl-symmetric-kl)
3. [MMD² (Maximum Mean Discrepancy)](#mmd²-maximum-mean-discrepancy)
4. [Sinkhorn Divergence](#sinkhorn-divergence)
5. [Nyström Log-Det](#nyström-log-det)

---

## Unified Computer

### `coreset_selection.objectives.SpaceObjectiveComputer`

**Kind:** class
**Source:** `coreset_selection/objectives/computer.py`

**Summary:** Single object that, given a representation `X` of shape `(N, d)`, can evaluate any requested objective on an arbitrary boolean mask.

**Description:** The central performance optimisation of the pipeline. Internally caches RFF features, anchor points, kernel sub-matrices, and Gaussian statistics that are invariant under mask changes. A single `SpaceObjectiveComputer` is instantiated once per run and reused across thousands of NSGA-II individuals — each individual evaluation is then `O(k)` or `O(k²)` rather than `O(N)`. It accepts `MMDConfig` and `SinkhornConfig` to tune RFF dimensionality and Sinkhorn regularisation, respectively.

**Key methods:**
- `compute_skl(mask) -> float` — symmetric-KL objective.
- `compute_mmd2(mask) -> float` — MMD² objective.
- `compute_sinkhorn(mask) -> float` — Sinkhorn divergence objective.
- `compute_nystrom_logdet(mask) -> float` — log-det diversity objective.
- `compute_all(mask, objectives: Tuple[str,...]) -> np.ndarray` — batch evaluation on requested objectives.

**See also:** `build_space_objective_computer`, individual objective classes below.

---

### `coreset_selection.objectives.build_space_objective_computer`

**Kind:** function
**Source:** `coreset_selection/objectives/computer.py`

**Summary:** Factory function that constructs a fully-initialised `SpaceObjectiveComputer` given a representation and config objects.

**Signature:**
```python
def build_space_objective_computer(
    X: np.ndarray,
    logvars: Optional[np.ndarray] = None,
    mmd_cfg: Optional[MMDConfig] = None,
    sinkhorn_cfg: Optional[SinkhornConfig] = None,
    seed: int = 0,
) -> SpaceObjectiveComputer
```

**Parameters:**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `X` | `np.ndarray, shape (N, d)` | required | Representation (VAE latent, PCA projection, or raw features). |
| `logvars` | `np.ndarray or None` | `None` | VAE log-variances per point; required for SKL mode. |
| `mmd_cfg` | `MMDConfig or None` | `MMDConfig()` | RFF dimensionality, bandwidth strategy. |
| `sinkhorn_cfg` | `SinkhornConfig or None` | `SinkhornConfig()` | Regularisation, anchor count, iteration limit. |
| `seed` | `int` | `0` | Seeds RFF draws and anchor sampling. |

**Returns:** ready-to-use `SpaceObjectiveComputer`.

**Example:**
```python
from coreset_selection.objectives import build_space_objective_computer

computer = build_space_objective_computer(X=Z_vae, logvars=Z_logvar, seed=4200)
f = computer.compute_all(mask, ("mmd", "sinkhorn"))
print(f)  # array([0.0042, 12.37])
```

---

## SKL (Symmetric KL)

### `coreset_selection.objectives.symmetric_kl_diag_gaussians`

**Kind:** function
**Source:** `coreset_selection/objectives/skl.py`

**Summary:** Symmetric KL between two diagonal Gaussians given means and log-variances.

**Description:** Closed-form expression `SKL(p, q) = 0.5 * (KL(p‖q) + KL(q‖p))` for diagonal covariances. Used by `SpaceObjectiveComputer.compute_skl` to compare the empirical moments of the coreset to those of the full set under the VAE posterior.

---

### `coreset_selection.objectives.kl_diag_gaussians`

**Kind:** function
**Source:** `coreset_selection/objectives/skl.py`

**Summary:** Forward KL between two diagonal Gaussians.

---

### `coreset_selection.objectives.jeffreys_divergence_diag_gaussians`

**Kind:** function
**Source:** `coreset_selection/objectives/skl.py`

**Summary:** Jeffreys divergence (symmetrised, unscaled) between two diagonal Gaussians.

---

## MMD² (Maximum Mean Discrepancy)

### `coreset_selection.objectives.RFFMMD`

**Kind:** class
**Source:** `coreset_selection/objectives/mmd.py`

**Summary:** Random-Fourier-Feature approximation of MMD² between an empirical target and a coreset.

**Description:** Instantiates a bank of `m` random Fourier features (Gaussian kernel, bandwidth from median heuristic or supplied), pre-computes the feature matrix `Φ` and the full-set mean `μ_full = Φ.mean(axis=0)`. Each evaluation is then `||μ_full - μ_S||² = ||μ_full - Φ[S].mean(axis=0)||²`, which is `O(k·m)` per individual.

**Key methods:**
- `evaluate(mask) -> float` — MMD² value.
- `.Phi` — cached feature matrix.
- `.mu_full` — cached full-set mean.

**See also:** `compute_rff_features`, `mmd2_exact`.

---

### `coreset_selection.objectives.compute_rff_features`

**Kind:** function
**Source:** `coreset_selection/objectives/mmd.py`

**Summary:** Compute an `(N, m)` RFF feature matrix from raw features `X`.

---

### `coreset_selection.objectives.mmd2_exact`

**Kind:** function
**Source:** `coreset_selection/objectives/mmd.py`

**Summary:** Exact (non-RFF) MMD² — `O(N²)`. Used for validation and tests; not in the hot path.

---

## Sinkhorn Divergence

### `coreset_selection.objectives.AnchorSinkhorn`

**Kind:** class
**Source:** `coreset_selection/objectives/sinkhorn.py`

**Summary:** Anchor-based Sinkhorn divergence between a coreset and a shared anchor set.

**Description:** Computes Sinkhorn divergence via a fixed anchor bank (default 200 points). The anchors are drawn once at `build()` time so that per-evaluation cost depends only on `k` and the anchor count, not on `N`. Uses a safe log-domain Sinkhorn iteration to avoid numerical overflow (manuscript Appendix).

**Key methods:**
- `build(X, cfg, seed)` — class method returning an initialised instance.
- `evaluate(mask) -> float` — Sinkhorn divergence value.

---

### `coreset_selection.objectives.sinkhorn2_safe`

**Kind:** function
**Source:** `coreset_selection/objectives/sinkhorn.py`

**Summary:** Numerically-safe log-domain Sinkhorn algorithm.

---

## Nyström Log-Det

### `coreset_selection.objectives.NystromLogDet`

**Kind:** class
**Source:** `coreset_selection/objectives/nystrom_logdet.py`

**Summary:** Log-determinant diversity measure via Nyström kernel sub-matrix.

**Description:** Computes `log det(K_SS + λI)` — a diversity signal that rewards spread-out coresets. Used as the third objective in the tri-objective ablation (R15) and as a candidate for the A4 experiment once bi-objective is shown suboptimal.

---

## See Also

- [optimization](./optimization.md) — consumes `SpaceObjectiveComputer` inside `nsga2_optimize`.
- [utils](./utils.md) — `median_sq_dist` used as default bandwidth for MMD.
- Manuscript Section III (objectives), Appendix B (Sinkhorn numerics).
