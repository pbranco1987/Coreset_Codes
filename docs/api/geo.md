# API Reference — Geographic Layer (`coreset_selection.geo`)

The geo layer provides the **spatial grouping structure** (27 Brazilian states here, but generalisable) and all utilities required to express and project geographic proportionality constraints. Every NSGA-II run, every baseline, and every constraint evaluation goes through objects defined in this layer.

Public symbols are re-exported from `coreset_selection/geo/__init__.py`.

## Section Map

1. [Geographic Info](#geographic-info)
2. [KL Utilities](#kl-utilities)
3. [Constraint Projection](#constraint-projection)
4. [Shapefile Utilities (optional)](#shapefile-utilities-optional)

---

## Geographic Info

### `coreset_selection.geo.GeoInfo`

**Kind:** dataclass
**Source:** `coreset_selection/geo/info.py:17`

**Summary:** Holds all geographic-group information (membership, sizes, population weights, target distributions) for both municipality-share and population-share constraint modes.

**Description:** `GeoInfo` is the central object carrying group structure through the pipeline. Every constraint builder, every projector, and every diagnostic expects a `GeoInfo` instance. It supports two weight modes simultaneously (manuscript Section IV-B): **municipality-share** (`w_i ≡ 1`, target `π_g = n_g/N`) and **population-share** (`w_i = pop_i`, target `π_g^(pop)`). The fields `pi` and `pi_pop` are pre-computed so downstream code never recomputes target distributions per generation.

**Attributes:**
| Name | Type | Description |
|------|------|-------------|
| `groups` | `List[str]` | Unique group names (e.g., 27 Brazilian state codes, ordered). |
| `group_ids` | `np.ndarray, shape (N,)` | Group index (0..G-1) assigned to each point. |
| `group_to_indices` | `List[np.ndarray]` | Inverse map: for each group, the indices of its members. |
| `pi` | `np.ndarray, shape (G,)` | Municipality-share target: `π_g = n_g / N`. |
| `group_sizes` | `np.ndarray, shape (G,)` | Capacity `n_g` per group. |
| `population_weights` | `np.ndarray or None, shape (N,)` | Per-municipality weights `w_i = pop_i` (if available). |
| `pi_pop` | `np.ndarray or None, shape (G,)` | Population-share target: `Σ_{i∈I_g} pop_i / Σ pop`. |

**Properties:** `.G` (number of groups), `.N` (total points).

**Key methods:**
- `get_group_name(group_idx) -> str` — group name lookup.
- `get_group_idx(group_name) -> int` — inverse lookup.
- `get_target_distribution(weight_type="muni") -> np.ndarray` — returns `pi` or `pi_pop` based on mode.

**See also:** `build_geo_info`, `GeographicConstraintProjector`.

---

### `coreset_selection.geo.build_geo_info`

**Kind:** function
**Source:** `coreset_selection/geo/info.py` (around line 170)

**Summary:** Construct a `GeoInfo` from a label vector and optional population weights.

**Description:** Deduplicates state labels to get the group list, builds the group-index vector, and computes all target distributions. When `population_weights` is provided, the returned object supports both municipality-share and population-share modes. This is the standard entry point called once per replicate at cache time (and again by every launcher to avoid serialising complex objects in the cache).

**Signature:**
```python
def build_geo_info(
    state_labels: np.ndarray,
    population_weights: Optional[np.ndarray] = None,
) -> GeoInfo
```

**Parameters:**
| Name | Type | Description |
|------|------|-------------|
| `state_labels` | `np.ndarray, shape (N,)` | Per-point group label (string or int). |
| `population_weights` | `np.ndarray or None, shape (N,)` | Optional per-point weights; when given, enables population-share mode. |

**Returns:** `GeoInfo` populated for both modes when weights are supplied.

**Example:**
```python
from coreset_selection.data import load_replicate_cache
from coreset_selection.geo import build_geo_info

assets = load_replicate_cache("cache/rep00/assets.npz")
geo = build_geo_info(assets.state_labels, population_weights=assets.population)
print(geo.G, geo.N, geo.pi[:3], geo.pi_pop[:3])
```

**See also:** `GeoInfo`, `merge_small_groups`.

---

### `coreset_selection.geo.merge_small_groups`

**Kind:** function
**Source:** `coreset_selection/geo/info.py`

**Summary:** Merge groups below a size threshold into a combined "other" group.

**Description:** Useful when experimenting with datasets containing very small geographic groups where feasibility would otherwise force 1 representative per group to dominate the coreset. Not used by the primary manuscript pipeline.

---

## KL Utilities

Functions implementing the KL divergence between target and achieved share distributions, plus Algorithm 2 (KL-optimal integer quotas) from the manuscript.

### `coreset_selection.geo.kl_pi_hat_from_counts`

**Kind:** function
**Source:** `coreset_selection/geo/kl.py`

**Summary:** KL divergence from target π to achieved distribution derived from count vector `c`.

**Description:** Core scalar used by every proportionality constraint. Implements `D_α(π, π̂(c)) = Σ π_g log(π_g / π̂_g^(α)(c))` with Laplace smoothing `α` (default `α=1` per manuscript). Accepts either a count vector (for Theorem 1 / Algorithm 2) or a weighted sum (population-share mode).

---

### `coreset_selection.geo.kl_weighted_from_subset`

**Kind:** function
**Source:** `coreset_selection/geo/kl.py`

**Summary:** Population-share KL divergence from a subset mask.

**Description:** Computes `D^(w)(S)` directly from a binary selection mask and per-point weights. This is the version called inside NSGA-II's constraint evaluation per offspring.

---

### `coreset_selection.geo.compute_constraint_violations`

**Kind:** function
**Source:** `coreset_selection/geo/kl.py`

**Summary:** Evaluate multiple simultaneous constraints on a single mask.

---

### `coreset_selection.geo.kl_optimal_integer_counts_bounded`

**Kind:** function
**Source:** `coreset_selection/geo/kl.py`

**Summary:** **Algorithm 2 (manuscript): KL-optimal integer quota vector subject to lower/upper bounds.**

**Description:** Given `k`, `π`, `α`, and per-group bounds `(ℓ_g, u_g)`, computes the unique integer vector `c*(k) ∈ ℤ^G` that minimises `D_α(π, π̂(c))` subject to `Σ c_g = k` and `ℓ_g ≤ c_g ≤ u_g`. Uses the telescoping-marginal-gain greedy proof from the manuscript appendix to run in `O(k log G)`. This is the basis of every HARD-quota constraint in the pipeline.

---

### `coreset_selection.geo.min_achievable_geo_kl_bounded`

**Kind:** function
**Source:** `coreset_selection/geo/kl.py`

**Summary:** Minimum achievable `D_α(π, π̂)` for a given `k` (the `KL_min(k)` floor).

**Description:** Companion to `kl_optimal_integer_counts_bounded`: returns the KL value attained at the optimum. Used by `adaptive_tau.py` as the starting point of the probe phase and by the manuscript's Figure 1.

---

### `coreset_selection.geo.proportional_allocation`

**Kind:** function
**Source:** `coreset_selection/geo/kl.py`

**Summary:** Largest-remainder (Hamilton) proportional allocation as a reference baseline for quota schemes.

---

### `coreset_selection.geo.compute_quota_path`

**Kind:** function
**Source:** `coreset_selection/geo/kl.py`

**Summary:** Compute `(c*(k), KL_min(k))` for every `k` in a grid.

**Description:** Produces the "quota path" — a table of optimal quotas and KL floors across cardinalities. Saved once per dataset and reused by every downstream run.

---

### `coreset_selection.geo.save_quota_path`

**Kind:** function
**Source:** `coreset_selection/geo/kl.py`

**Summary:** Persist a quota-path table to disk as CSV/JSON.

---

## Constraint Projection

### `coreset_selection.geo.GeographicConstraintProjector`

**Kind:** class
**Source:** `coreset_selection/geo/projector.py`

**Summary:** Project an arbitrary boolean mask to the nearest feasible mask under geographic (count) constraints.

**Description:** The primary workhorse for hard-quota constraints. Given a `GeoInfo`, a weight mode, and the `α` smoothing constant, builds internal lookups so that subsequent `.project_to_quota_mask(mask, k, rng=...)` calls run in `O(k)` per call. Also exposes `.target_counts(k)` which wraps `kl_optimal_integer_counts_bounded` into a memoised form. Called once during `_repair_mask` every NSGA-II generation.

**Key methods:**
- `target_counts(k: int) -> np.ndarray` — memoised `c*(k)`.
- `project_to_quota_mask(mask, k, rng) -> np.ndarray` — exact-k projection onto quota.
- `project_to_exact_k_mask(mask, k, rng) -> np.ndarray` — cardinality-only projection (no quota).

**See also:** `project_to_exact_k_mask`, `build_feasible_quota_mask`.

---

### `coreset_selection.geo.project_to_exact_k_mask`

**Kind:** function
**Source:** `coreset_selection/geo/projector.py`

**Summary:** Cardinality-only projection: enforce `|S| = k`.

---

### `coreset_selection.geo.build_feasible_quota_mask`

**Kind:** function
**Source:** `coreset_selection/geo/projector.py`

**Summary:** Construct a random feasible mask satisfying `c*(k)` exactly.

**Description:** Used to seed the NSGA-II initial population with feasible solutions.

---

### `coreset_selection.geo.compute_quota_violation`

**Kind:** function
**Source:** `coreset_selection/geo/projector.py`

**Summary:** Return `|c_g(S) - c*(k)_g|` summed over groups — amount by which a mask fails the quota.

---

## Shapefile Utilities (optional)

These are only imported when `geopandas` is installed. Absent otherwise; the package still loads.

### `coreset_selection.geo.load_brazil_municipalities`

**Kind:** function — loads a GeoDataFrame of Brazilian municipalities for choropleth plotting.

### `coreset_selection.geo.get_brazil_border`

**Kind:** function — returns the outer Brazilian polygon for map overlays.

### `coreset_selection.geo.plot_choropleth_panel`

**Kind:** function — produces a state-level choropleth panel (used in supplementary figures).

---

## See Also

- [data](./data.md) — produces the `state_labels` and `population` arrays consumed here.
- [constraints](./constraints.md) — uses `GeoInfo` as the backbone for soft-KL constraints.
- [optimization](./optimization.md) — every NSGA-II run instantiates a `GeographicConstraintProjector`.
- Manuscript Theorem 1 and Algorithm 2 (see `docs/METHODOLOGY.md`).
