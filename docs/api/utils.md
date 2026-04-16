# API Reference — Utilities (`coreset_selection.utils`)

Cross-cutting helpers used throughout the package: math utilities, I/O helpers, deterministic seeding, and plotting style constants for manuscript-consistent figures.

Public symbols are re-exported from `coreset_selection/utils/__init__.py`.

## Section Map

1. [Math](#math)
2. [I/O](#io)
3. [Seeding](#seeding)
4. [Plotting](#plotting)

---

## Math

### `coreset_selection.utils.median_sq_dist`

**Kind:** function
**Source:** `coreset_selection/utils/math.py`

**Summary:** Median squared pairwise distance — the standard "median heuristic" RBF bandwidth.

**Description:** For a feature matrix `X`, subsamples up to `n_sample` rows (default 2000), computes pairwise squared distances, and returns their median. This is the default `sigma_sq` for RBF kernels in `RawSpaceEvaluator.build`, `RFFMMD`, and `AnchorSinkhorn`.

**Signature:**
```python
def median_sq_dist(X: np.ndarray, n_sample: int = 2000, seed: int = 0) -> float
```

**Returns:** scalar median squared distance.

**Example:**
```python
from coreset_selection.utils import median_sq_dist
sigma_sq = median_sq_dist(X_scaled, n_sample=2000, seed=4200)
```

---

## I/O

### `coreset_selection.utils.ensure_dir`

**Kind:** function
**Source:** `coreset_selection/utils/io.py`

**Summary:** Create a directory (and parents) if it does not exist. Idempotent.

**Signature:** `def ensure_dir(path: str | Path) -> Path`

**Returns:** `Path` object pointing at the (now-existing) directory.

---

## Seeding

### `coreset_selection.utils.set_global_seed`

**Kind:** function
**Source:** `coreset_selection/utils/seed.py`

**Summary:** Seed `random`, `numpy`, and (if available) `torch` with the same integer.

**Description:** Use at the start of any top-level script that relies on deterministic behaviour. Downstream code should prefer local `np.random.Generator` instances (obtained via `get_rng`) over global RNG state.

### `coreset_selection.utils.stable_hash_int`

**Kind:** function
**Source:** `coreset_selection/utils/seed.py`

**Summary:** Deterministic, cross-platform hash of an arbitrary object to an integer (for seeding derived RNGs).

### `coreset_selection.utils.get_rng`

**Kind:** function
**Source:** `coreset_selection/utils/seed.py`

**Summary:** Return an `np.random.Generator` seeded from an integer or an object.

### `coreset_selection.utils.seed_sequence`

**Kind:** function
**Source:** `coreset_selection/utils/seed.py`

**Summary:** Build an `np.random.SeedSequence` with a stable high-entropy spawn pattern.

---

## Plotting

### `coreset_selection.utils.set_manuscript_style`

**Kind:** function
**Source:** `coreset_selection/utils/plotting.py`

**Summary:** Configure matplotlib rcParams to match the manuscript's figure style (fonts, sizes, grid style).

**Description:** Call once at the top of any figure script. Affects font family (Times), font sizes, line widths, and grid defaults.

### `coreset_selection.utils.get_method_style`

**Kind:** function — return a `(color, marker, linestyle)` tuple for a given method name.

### `coreset_selection.utils.get_objective_color`

**Kind:** function — deterministic color for an objective name.

### `coreset_selection.utils.objective_label`

**Kind:** function — LaTeX label for an objective (e.g., `"mmd"` → `"$f_{\\mathrm{MMD}}$"`).

### `coreset_selection.utils.method_label`

**Kind:** function — display name for a method (e.g., `"herding_quota"` → `"Kernel Herding (quota)"`).

### `coreset_selection.utils.figure_size`

**Kind:** function — return `(width, height)` inches for a manuscript-standard panel.

### `coreset_selection.utils.add_panel_label`

**Kind:** function — place a subplot label (e.g., `(a)`) at a canonical position.

### `coreset_selection.utils.truncate_colormap`

**Kind:** function — crop a colormap to a sub-range (for heatmaps with narrow value ranges).

### Palette Constants

- `PALETTE_METHODS` — dict mapping method canonical name → color.
- `PALETTE_OBJECTIVES` — dict mapping objective name → color.
- `MARKERS_METHODS` — dict mapping method canonical name → matplotlib marker.

---

## See Also

- [evaluation](./evaluation.md) — `RawSpaceEvaluator.build` uses `median_sq_dist` as bandwidth default.
- [objectives](./objectives.md) — MMD and Sinkhorn use the same bandwidth heuristic.
