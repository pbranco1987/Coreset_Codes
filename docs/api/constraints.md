# API Reference — Constraints (`coreset_selection.constraints`)

The constraints package implements the **weighted proportionality constraints** from manuscript Section IV-B. A constraint here is a *soft* KL-divergence penalty with tolerance `τ`: a mask `S` is feasible when `D^(w)(S) ≤ τ`. HARD-quota constraints live in `geo/` (via `GeographicConstraintProjector`); soft KL constraints live here.

Public symbols are re-exported from `coreset_selection/constraints/__init__.py`.

## Section Map

1. [Constraint Classes](#constraint-classes)
2. [Constraint Builders](#constraint-builders)
3. [Tau Calibration (Phase 5)](#tau-calibration-phase-5)

---

## Constraint Classes

### `coreset_selection.constraints.ProportionalityConstraint`

**Kind:** dataclass
**Source:** `coreset_selection/constraints/proportionality.py:40`

**Summary:** Single soft KL constraint with a given weight vector and tolerance `τ`.

**Description:** Encapsulates all data required to evaluate `D^(w)(S) ≤ τ` for a fixed weight vector `w` (either population-share or municipality-share). Each instance stores the pre-computed target distribution `π_target`, the per-point weight vector, the smoothing constant `α`, and the tolerance `τ`. Instances are created via `build_population_share_constraint` or `build_municipality_share_constraint` — direct construction is rarely needed. The `value(mask, geo)` method returns the current KL; `.violation(mask, geo)` returns `max(KL - τ, 0)`.

**Attributes (typical):**
| Name | Type | Description |
|------|------|-------------|
| `name` | `str` | Human-readable label (e.g., `"pop_share"`, `"muni_share"`). |
| `weights` | `np.ndarray, shape (N,)` | Per-point weights `w_i`. |
| `pi_target` | `np.ndarray, shape (G,)` | Target distribution. |
| `alpha` | `float` | Laplace smoothing (default `1.0`). |
| `tau` | `float` | Tolerance. |

**Key methods:**
- `value(mask, geo) -> float` — current KL value `D^(w)(S)`.
- `violation(mask, geo) -> float` — `max(D^(w)(S) - τ, 0)`.
- `is_feasible(mask, geo) -> bool` — `D^(w)(S) ≤ τ`.

**See also:** `ProportionalityConstraintSet`, `build_population_share_constraint`.

---

### `coreset_selection.constraints.ProportionalityConstraintSet`

**Kind:** dataclass
**Source:** `coreset_selection/constraints/proportionality.py:168`

**Summary:** Container for multiple simultaneous soft constraints plus a swap-based repair operator.

**Description:** When more than one weight mode is active simultaneously (e.g., joint soft pop-share + soft muni-share), their violations are summed and repair runs against the aggregated violation. The set also holds meta-options: `min_one_per_group` (enforce at least one representative per state — a minimum floor), `preserve_group_counts` (lock in current counts and only swap within groups), and `max_iters` for the repair loop. The `.repair(mask, rng)` method is the heart of every NSGA-II generation's constraint handling.

**Attributes (typical):**
| Name | Type | Description |
|------|------|-------------|
| `geo` | `GeoInfo` | Geographic structure. |
| `constraints` | `List[ProportionalityConstraint]` | Individual soft constraints. |
| `min_one_per_group` | `bool` | Enforce ≥ 1 representative per group. |
| `preserve_group_counts` | `bool` | Swap only within groups. |
| `max_iters` | `int` | Maximum repair iterations. |

**Key methods:**
- `total_violation(mask) -> float` — sum over member constraints.
- `is_feasible(mask) -> bool` — all constraints satisfied.
- `repair(mask, rng) -> np.ndarray` — heuristic swap-based repair (Algorithm 1 in the manuscript appendix). Greedily swaps pairs to reduce the single worst violation; terminates when feasible or no improvement is possible.

**See also:** `ProportionalityConstraint`, `GeographicConstraintProjector`, `_repair_mask` (in optimization).

---

## Constraint Builders

### `coreset_selection.constraints.build_population_share_constraint`

**Kind:** function
**Source:** `coreset_selection/constraints/proportionality.py:92`

**Summary:** Convenience constructor for a population-share soft constraint.

**Description:** Returns a `ProportionalityConstraint` with `w_i = population[i]` and target `π_g^(pop) = Σ_{i∈I_g} pop_i / Σ pop`. This is the primary manuscript constraint used in `popsoft` mode by `adaptive_tau.py`.

**Signature:**
```python
def build_population_share_constraint(
    geo: GeoInfo,
    population: np.ndarray,
    alpha: float = 1.0,
    tau: float = 0.02,
    name: str = "pop_share",
) -> ProportionalityConstraint
```

**Parameters:**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `geo` | `GeoInfo` | required | Geographic group structure. |
| `population` | `np.ndarray, shape (N,)` | required | Per-point population weights. |
| `alpha` | `float` | `1.0` | Laplace smoothing (fixed across manuscript). |
| `tau` | `float` | `0.02` | Feasibility tolerance (overridden by adaptive-tau calibration). |
| `name` | `str` | `"pop_share"` | Constraint label for logging. |

**Returns:** a `ProportionalityConstraint` instance ready to be wrapped in a `ProportionalityConstraintSet`.

**Example:**
```python
from coreset_selection.geo import build_geo_info
from coreset_selection.constraints import (
    build_population_share_constraint, ProportionalityConstraintSet,
)

geo = build_geo_info(state_labels, population_weights=pop)
c_pop = build_population_share_constraint(geo, pop, tau=0.02)
cs = ProportionalityConstraintSet(geo=geo, constraints=[c_pop])
```

**See also:** `build_municipality_share_constraint`, `estimate_feasible_tau_range`.

---

### `coreset_selection.constraints.build_municipality_share_constraint`

**Kind:** function
**Source:** `coreset_selection/constraints/proportionality.py:132`

**Summary:** Convenience constructor for a municipality-share soft constraint (uniform weights).

**Description:** Returns a `ProportionalityConstraint` with `w_i ≡ 1` and target `π_g = n_g / N`. Used by `adaptive_tau.py --constraint-mode munisoft` and by joint constraint modes where muni-share is the soft component.

**Signature:**
```python
def build_municipality_share_constraint(
    geo: GeoInfo,
    alpha: float = 1.0,
    tau: float = 0.02,
    name: str = "muni_share",
) -> ProportionalityConstraint
```

---

## Tau Calibration (Phase 5)

Utilities supporting the adaptive-tau protocol: probing the feasible range and running τ sensitivity sweeps.

### `coreset_selection.constraints.estimate_feasible_tau_range`

**Kind:** function
**Source:** `coreset_selection/constraints/calibration.py`

**Summary:** Estimate a lower bound for a feasible τ (based on greedy/analytical bounds) and a conservative upper bound.

**Description:** Used by `adaptive_tau.py` to seed the probe phase with an informed starting point (`greedy KL floor`, Corollary 2) instead of blindly searching.

---

### `coreset_selection.constraints.tau_sensitivity_sweep`

**Kind:** function
**Source:** `coreset_selection/constraints/calibration.py`

**Summary:** Evaluate Pareto-front quality across multiple τ values to visualise the feasibility/quality trade-off.

**Description:** Runs NSGA-II (or reads existing fronts) at several τ values and returns the resulting `TauSweepResult` summarising feasibility fraction, front cardinality, and hypervolume as a function of τ.

---

### `coreset_selection.constraints.tau_sweep_to_csv_rows`

**Kind:** function
**Source:** `coreset_selection/constraints/calibration.py`

**Summary:** Flatten a `TauSweepResult` into CSV rows for table export.

---

### `coreset_selection.constraints.TauSweepResult`

**Kind:** dataclass
**Source:** `coreset_selection/constraints/calibration.py`

**Summary:** Container for τ sweep outputs (per-τ feasibility fraction, front size, objective values).

---

## See Also

- [geo](./geo.md) — provides `GeoInfo` and HARD-quota projection.
- [optimization](./optimization.md) — `_repair_mask` wraps `ProportionalityConstraintSet.repair`.
- [scripts/launchers/adaptive_tau.py](./scripts.md#adaptive_tau) — the adaptive-tau state machine using these constraints.
- Manuscript Section IV-B (weighted proportionality), Algorithm 1 (repair).
