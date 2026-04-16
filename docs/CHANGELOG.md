# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project follows semantic versioning for the public API defined in [docs/api/](./api/index.md).

## [Unreleased]

### Documentation
- New flagship `docs/api/` reference (10 pages) covering every public symbol in `coreset_selection` and every CLI entry point.
- New onboarding docs: `GETTING_STARTED.md`, `INSTALL.md`, `TROUBLESHOOTING.md`, `CHANGELOG.md`, `docs/README.md` index.
- Root-level `CONTRIBUTING.md` describing development setup, PR workflow, and step-by-step templates for adding baselines/objectives/constraints.
- `docs/METHODOLOGY.md` extended with an *Implementation Mapping* section pointing every manuscript equation to its Python file.

### Removed
- `scripts/build_manuscript.py` — manuscript-writing (LaTeX table/figure generation) code is out of scope for the public repository. The file has been moved to `Coreset_Codes_scratch/scripts_removed/` for personal reference. Committed `manuscript/generated/tables/*.tex` and `manuscript/generated/figures/*.pdf` remain as last-compile artifacts.

### Fixed
- Hardcoded absolute path in `scripts/analysis/championship.py` replaced with a configurable `--experiments-dir` CLI argument (defaults to `$CORESET_EXPERIMENTS_DIR` or `./experiments_v2`).
- Undocumented internal helpers in `scripts/bootstrap_reeval.py` (`_resolve_base_cache`, `_median_sq_dist`) now carry full docstrings.

### Examples
- New `examples/` folder with 5 Jupyter notebook tutorials: loading caches, running a minimal experiment, visualising the Pareto front, interpreting metrics, adding a new baseline.

---

## [v2.0] — 2026 (Current)

This is the **onboarding-ready reorganisation**. The major upstream code changes were:

### Added
- **Adaptive τ calibration** (`scripts/launchers/adaptive_tau.py`): 3-phase protocol (probe → bisect → production) that automatically finds the smallest `τ` at which the NSGA-II population maintains ~50 % feasibility. Replaces the previous hand-tuned `τ = 0.02` for all `(k, space, constraint)` combinations.
- **Construction/evaluation separation**: NSGA-II now saves only coreset indices (`coreset.npz`) and a separate `evaluate_coresets.py` script computes all metrics. This guarantees a single code path for every metric and decouples re-evaluation from re-optimisation.
- **Target-leakage guards** (`coreset_selection/data/target_columns.py`): 35+ regex rules detect known downstream targets; `validate_no_leakage` hard-fails if any target column survives into the feature matrix before VAE/PCA training.
- **Standalone evaluation script** (`scripts/analysis/evaluate_coresets.py`): single authoritative metric computation with batch mode, skip-if-exists idempotency, and pattern filtering.

### Fixed
- **S ∩ E overlap fix** (`coreset_selection/evaluation/raw_space.py`): when a coreset index is also in the evaluation set, it is excluded from `K_EE`, `Phi`, and downstream features before metric computation. Prevents landmarks from evaluating themselves, which previously inflated Nyström error scores and KRR accuracy.
- **StandardScaler fit scope** in `scripts/bootstrap_reeval.py`: now fit on the training partition only (not the full dataset), preventing subtle train/test leakage.
- **Friedman test block structure** in `scripts/analysis/championship.py`: each (metric, rep) pair is one block, giving `N = n_metrics × n_reps` blocks for the Friedman χ² statistic. Preserves within-replicate variability.
- **`restart.sh` argument consistency**: removed a brittle 2,366-line file that hardcoded 256 specific jobs; dispatcher now auto-discovers jobs from `experiments_v2/` metadata.

### Changed
- **Scripts reorganisation**: 51 files → 15 files. Removed 8 archived, superseded scripts; 6 legacy job manifests; 6 one-off deploy scripts; 3 triply-redundant extraction scripts; 2 duplicate championship scripts. The remaining 15 are organised into `analysis/`, `bootstrap/`, `deploy/`, `infra/`, `launchers/`, plus two root scripts (`bootstrap_reeval.py`, `build_manuscript.py` — the latter removed in the "Unreleased" section above).
- **Repository reorganisation**: `manuscript/` is now a dedicated subdirectory; `docs/` holds all markdown documentation; `data/` contains only raw CSVs; `EXPERIMENTS-*` at root for experiment outputs. Root directory dropped from 191 items to 8.

### Removed
- Manuscript-writing code (final removal; see Unreleased).
- 60+ scratch and debug scripts relocated to `Coreset_Codes_scratch/` (outside the repository).
- Old experiment outputs (27 directories, multiple GB).

---

## [v1.x] (Historical)

Pre-reorganisation codebase. See Git history for details.

Highlights:
- Initial NSGA-II implementation with hand-tuned `τ`.
- Legacy pymoo-based experiment runner (`ExperimentRunner`).
- Original championship analysis with pre-averaged Friedman blocks (corrected in v2.0).

---

## API Stability Policy

Symbols documented in [docs/api/](./api/index.md) constitute the public API. We follow these rules:

1. **Backwards-incompatible changes** require a major version bump (v2 → v3).
2. **New features** and non-breaking fixes bump the minor version (v2.0 → v2.1).
3. **Deprecated symbols** remain in the API for at least one minor version with a `DeprecationWarning`, then are removed in the next major version.
4. **Private symbols** (underscore-prefixed) may change without notice, *except* those that are explicitly re-exported in a subpackage `__init__.py` (e.g., `_br_to_float`) — these are treated as public.
