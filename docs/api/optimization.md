# API Reference — Optimization (`coreset_selection.optimization`)

The optimization package contains the **NSGA-II algorithm and supporting utilities** for multi-objective coreset search under geographic constraints. It also exposes Pareto-representative selection utilities (knee, best-per-objective, Chebyshev).

Public symbols are re-exported from `coreset_selection/optimization/__init__.py`.

## Section Map

1. [NSGA-II Entry Points](#nsga-ii-entry-points)
2. [Pareto Selection](#pareto-selection)
3. [Legacy Pymoo Operators (optional)](#legacy-pymoo-operators-optional)

---

## NSGA-II Entry Points

### `coreset_selection.optimization.nsga2_optimize`

**Kind:** function
**Source:** `coreset_selection/optimization/nsga2_internal.py`

**Summary:** Run a full NSGA-II optimisation and return the final Pareto front (constraint-dominated).

**Description:** Self-contained NSGA-II for 2–3 objectives with exact-`k` cardinality and proportionality constraints. Uses uniform binary crossover, a quota-preserving swap mutation, per-offspring constraint repair (via `ProportionalityConstraintSet.repair`), and constraint-dominated ranking (Deb's constrained-domination). Initialisation: seeds the population with `_greedy_kl_init` (Corollary 2 greedy floor mask) so that at least one feasible solution is present from generation 0. Returns a data structure containing the final population, the Pareto front, per-generation logs, and timing.

**Signature (representative):**
```python
def nsga2_optimize(
    computer: SpaceObjectiveComputer,
    projector: GeographicConstraintProjector,
    constraint_set: Optional[ProportionalityConstraintSet],
    k: int,
    objectives: Tuple[str, ...] = ("mmd", "sinkhorn"),
    pop_size: int = 300,
    n_generations: int = 1500,
    crossover_prob: float = 0.9,
    mutation_prob: float = 0.1,
    rng: Optional[np.random.Generator] = None,
    adaptive_tau_check_every: Optional[int] = None,
    adaptive_tau_factor: float = 1.5,
    adaptive_tau_max: float = 1.0,
    verbose: bool = False,
) -> ParetoFrontData
```

**Parameters:**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `computer` | `SpaceObjectiveComputer` | required | Pre-built objective evaluator. |
| `projector` | `GeographicConstraintProjector` | required | Hard-quota projector for feasibility. |
| `constraint_set` | `ProportionalityConstraintSet or None` | required | Soft-KL constraints (pass `None` for unconstrained). |
| `k` | `int` | required | Exact coreset cardinality. |
| `objectives` | `Tuple[str, ...]` | `("mmd", "sinkhorn")` | Active objectives. |
| `pop_size` | `int` | `300` | NSGA-II population size. |
| `n_generations` | `int` | `1500` | Number of generations. |
| `crossover_prob` | `float` | `0.9` | Uniform crossover probability. |
| `mutation_prob` | `float` | `0.1` | Per-offspring mutation probability. |
| `rng` | `np.random.Generator or None` | `None` | RNG (constructed from seed if `None`). |
| `adaptive_tau_check_every` | `int or None` | `None` | If set, relax `τ` by `adaptive_tau_factor` every this many gens when stagnant. |
| `verbose` | `bool` | `False` | Print per-generation progress. |

**Returns:** `ParetoFrontData` with `.X` (boolean masks, shape `(n_front, N)`), `.F` (objective values), and bookkeeping.

**See also:** `scripts/launchers/adaptive_tau.py` — calls `nsga2_optimize` inside the 3-phase tau calibration state machine; `fast_non_dominated_sort`.

---

### `coreset_selection.optimization.fast_non_dominated_sort`

**Kind:** function
**Source:** `coreset_selection/optimization/nsga2_internal.py`

**Summary:** Deb et al.'s O(M·N²) non-dominated sort.

**Description:** Given an objective matrix `F` of shape `(N, M)`, returns a list of fronts where front `0` is the set of non-dominated indices, front `1` dominates only front-0 members, and so on. Used inside NSGA-II and at post-processing time to extract the final front from the terminal population.

**Signature:**
```python
def fast_non_dominated_sort(F: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]
```

**Returns:** `(fronts, rank)` where `fronts[i]` is an integer array of indices at rank `i` and `rank[j]` is the front index for individual `j`.

---

## Pareto Selection

### `coreset_selection.optimization.crowding_distance`

**Kind:** function
**Source:** `coreset_selection/optimization/selection.py`

**Summary:** NSGA-II crowding distance for diversity preservation.

**Description:** For a given front, computes per-individual crowding distance (sum of normalised objective-space gaps to neighbours). Used by tournament selection and to select a diverse subset of a front when the desired count is smaller than the front size.

---

### `coreset_selection.optimization.feasible_filter`

**Kind:** function
**Source:** `coreset_selection/optimization/selection.py`

**Summary:** Mask Pareto-front members by feasibility.

---

### `coreset_selection.optimization.select_knee`

**Kind:** function
**Source:** `coreset_selection/optimization/selection.py`

**Summary:** Pick the Pareto-front point with minimum Euclidean distance to the utopia point `(0, 0)` in normalised objective space.

**Description:** Canonical single-compromise-point selection. Normalises each objective independently via min-max, computes the distance to the all-zero normalised ideal point, and returns the index of the closest front member. This is the "naive knee" (selection strategy A1 in the manuscript).

**Signature:**
```python
def select_knee(F: np.ndarray) -> int
```

---

### `coreset_selection.optimization.select_pareto_representatives`

**Kind:** function
**Source:** `coreset_selection/optimization/selection.py`

**Summary:** Return a dictionary of named representative indices from a Pareto front.

**Description:** For each objective `obj_i`, returns the argmin index as `best-{obj_i}`. Also returns the knee index and (optionally) the Chebyshev point. This is the function `adaptive_tau.py` calls right before saving `representatives/*.npz`.

**Signature:**
```python
def select_pareto_representatives(
    F: np.ndarray,
    objectives: Tuple[str, ...] = ("mmd", "sinkhorn"),
) -> Dict[str, int]
```

**Returns:** mapping like `{"knee": 17, "best-mmd": 5, "best-sinkhorn": 42}`.

**Example:**
```python
from coreset_selection.optimization import select_pareto_representatives

reps = select_pareto_representatives(F_pareto, objectives=("mmd","sinkhorn"))
print(reps)  # {"knee": 17, "best-mmd": 5, "best-sinkhorn": 42}
```

---

## Legacy Pymoo Operators (optional)

These are only imported when `pymoo` is installed. They are used by the original legacy experiment runner; the new `adaptive_tau.py` does **not** depend on pymoo.

### `coreset_selection.optimization.CoresetMOOProblem`

**Kind:** class (pymoo problem wrapper).

### `coreset_selection.optimization.QuotaBinarySampling`, `ExactKSampling`

**Kind:** class (pymoo sampling strategies).

### `coreset_selection.optimization.QuotaAndCardinalityRepair`, `ExactKRepair`, `LeastHarmQuotaRepair`, `LeastHarmExactKRepair`, `RepairActivityTracker`

**Kind:** class (pymoo repair operators).

### `coreset_selection.optimization.UniformBinaryCrossover`, `QuotaSwapMutation`

**Kind:** class (pymoo variation operators).

### `coreset_selection.optimization._HAS_PYMOO`

**Kind:** constant (`bool`) — `True` when pymoo is importable.

---

## See Also

- [objectives](./objectives.md) — `SpaceObjectiveComputer` feeds `nsga2_optimize`.
- [constraints](./constraints.md) and [geo](./geo.md) — constraint objects consumed by `nsga2_optimize`.
- [scripts](./scripts.md#adaptive_tau) — `adaptive_tau.py` is the main caller of `nsga2_optimize`.
- Manuscript Section IV (NSGA-II), Algorithm 1 (repair operator).
