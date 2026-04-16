# Examples

Hands-on Jupyter notebook tutorials for the `coreset_selection` pipeline. Each notebook is self-contained and targets **< 5 minutes** runtime on a laptop (VAE training dominates in notebook 02; everything else is sub-second to seconds).

Read them in order the first time you work with this repository.

## Prerequisites

- `pip install -e ".[dev]"` (includes `jupyter` and `matplotlib`).
- A built replicate cache: `python scripts/launchers/build_caches.py --reps 0` (required for notebooks 01, 02, 03, 04).

## Notebooks

| # | Notebook | What you'll learn |
|---|----------|-------------------|
| 01 | [01_load_and_inspect_cache.ipynb](./01_load_and_inspect_cache.ipynb) | How `assets.npz` is structured; how to confirm target-leakage protection worked. |
| 02 | [02_run_minimal_experiment.ipynb](./02_run_minimal_experiment.ipynb) | Smallest possible end-to-end run: build → evaluate → inspect metrics. |
| 03 | [03_visualize_pareto_front.ipynb](./03_visualize_pareto_front.ipynb) | How the Pareto front looks and where the knee/best-MMD/best-SH selections land. |
| 04 | [04_interpret_metrics.ipynb](./04_interpret_metrics.ipynb) | How to read `metrics.csv`: what every column means and how to compare methods. |
| 05 | [05_add_new_baseline.ipynb](./05_add_new_baseline.ipynb) | Template for contributing a new baseline selection method. |

## Common pitfalls

- **"No module named `coreset_selection`"** — run notebooks from the repo root or set `PYTHONPATH`. The first cell of each notebook does this automatically if needed.
- **"`assets.npz` not found"** — you haven't built a cache. Run `python scripts/launchers/build_caches.py --reps 0` first.
- **Notebook 02 takes too long** — VAE training in the cache build dominates; that's the 5-minute step. Subsequent notebooks reuse the same cache and are fast.

## Cross-references

- For the theory, see [docs/METHODOLOGY.md](../docs/METHODOLOGY.md).
- For the full pipeline, see [scripts/PIPELINE.md](../scripts/PIPELINE.md).
- For function reference, see [docs/api/](../docs/api/index.md).
