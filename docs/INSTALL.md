# Installation Guide

Detailed setup instructions for local development and reproduction.

## Requirements

- **Python 3.10 or newer** (3.11+ recommended).
- **Operating system:** Linux (tested on Ubuntu 22.04), macOS, or Windows (WSL2 strongly recommended on Windows).
- **Disk space:** ~2 GB for code + dependencies, 5–10 GB for replicate caches, 20–50 GB for full experiment outputs.
- **RAM:** 16 GB minimum, 32 GB+ comfortable. Larger `k` values (400–500) benefit from more RAM.
- **GPU:** Optional. VAE training is CPU-feasible (~5–10 minutes per replicate) but 3–5× faster on GPU.

## Step-by-Step Install

### 1. Clone the repository

```bash
git clone <repo-url> Coreset_Codes
cd Coreset_Codes
```

If the raw data is stored via Git LFS:

```bash
git lfs install
git lfs pull
```

### 2. Create a virtual environment

Using `venv` (standard library):

```bash
python -m venv venv

# Activate (Linux/macOS):
source venv/bin/activate

# Activate (Windows PowerShell):
venv\Scripts\Activate.ps1

# Activate (Windows Git Bash):
source venv/Scripts/activate
```

Using `conda` (alternative):

```bash
conda create -n coreset python=3.11
conda activate coreset
```

### 3. Install the package

Editable install (recommended for development):

```bash
pip install -e .
```

Or, if a `pyproject.toml` specifies optional extras:

```bash
pip install -e ".[dev]"      # includes test/doc dependencies
pip install -e ".[cuda]"     # includes GPU PyTorch
```

### 4. Verify

```bash
python -c "import coreset_selection; print(coreset_selection.__version__)"
# Expected output: 0.1.0
```

## Core Dependencies

| Package | Version | Why |
|---------|---------|-----|
| `numpy` | ≥ 1.23 | Every numerical array. |
| `scipy` | ≥ 1.10 | Friedman test, Nemenyi CD, Sinkhorn numerics. |
| `scikit-learn` | ≥ 1.2 | PCA, KRR, baseline models (KNN/RF/GBT/LR). |
| `pandas` | ≥ 1.5 | CSV I/O, result merging. |
| `torch` | ≥ 2.0 | VAE training. |
| `matplotlib` | ≥ 3.7 | Figures (in examples/). |

## Optional Dependencies

| Package | Extra | Purpose |
|---------|-------|---------|
| `pymoo` | — | Legacy pymoo-based NSGA-II operators (new pipeline does not need it). |
| `geopandas` | — | Choropleth visualisations in `coreset_selection.geo.shapefile`. |
| `jupyter`, `ipython` | `dev` | Running notebooks in `examples/`. |
| `pytest` | `dev` | Running the test suite in `coreset_selection/tests/`. |

## GPU Setup (Optional)

For GPU-accelerated VAE training:

1. Verify your CUDA version: `nvidia-smi`.
2. Install PyTorch with matching CUDA:

```bash
# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

3. Test:

```bash
python -c "import torch; print(torch.cuda.is_available())"
# Expected: True
```

4. Use `--device cuda` in `build_caches.py`:

```bash
python scripts/launchers/build_caches.py --device cuda
```

## Platform-Specific Notes

### Windows

- **WSL2 is strongly recommended.** The pipeline runs natively on Windows but path handling (especially with `pathlib` and `.gitignore` interactions) has fewer surprises in WSL2.
- If you must use native Windows:
  - Use Git Bash or PowerShell 7+, not `cmd.exe`.
  - Paths with spaces need quoting: `python "scripts/launchers/adaptive_tau.py"`.
  - Git LFS sometimes requires manual `git lfs pull` after clone.

### macOS (Apple Silicon)

- PyTorch has native Apple Silicon support via MPS. Use `--device mps` for VAE training (≈ 2× faster than CPU on M1/M2/M3).
- Install scientific stack via `pip` (not `conda-forge`) for best Apple Silicon wheel availability.

### Linux

- No known issues. Tested on Ubuntu 22.04 and 24.04.

## Installing the Raw Data

The raw Brazilian telecom CSVs are stored separately (they exceed typical repository size limits). If the `data/` directory is empty after cloning:

1. **Check for Git LFS:** `git lfs ls-files` — if files appear, run `git lfs pull`.
2. **Download separately:** see the data-access section in the repo README (if applicable) for a download URL or external mirror.
3. **Use synthetic data for testing:** the `coreset_selection.data.load_synthetic_data` function produces a small synthetic dataset sufficient for unit tests and tutorials.

Required files in `data/`:

| File | Size | Description |
|------|------|-------------|
| `smp_main.csv` | ~500 MB | Main feature matrix (5,569 municipalities × 1,700 columns). |
| `metadata.csv` | ~100 KB | Column metadata (types, descriptions). |
| `city_populations.csv` | ~300 KB | Population per municipality. |
| `df_indicators_flat_by_municipality.csv` | ~50 MB | Extra regression/classification targets. |
| `qf_mean_by_year.csv` | ~1 MB | QoS survey data (optional). |

## Smoke Test

After installation, run the quickest end-to-end test:

```bash
# Build one cache, run tiny NSGA-II, evaluate.
python scripts/launchers/build_caches.py --reps 0
python scripts/launchers/adaptive_tau.py \
    --k 30 --space vae --constraint-mode popsoft --rep 0 \
    --cache-dir experiments/adaptive_tau_k100_ps_vae/replicate_cache \
    --output-dir /tmp/runs_out_smoketest
python scripts/analysis/evaluate_coresets.py \
    --experiment-dir /tmp/runs_out_smoketest/nsga2-vae-popsoft-k30-rep0 \
    --cache-path experiments/adaptive_tau_k100_ps_vae/replicate_cache/rep00/assets.npz
```

Expected: a `metrics.csv` containing ~30 rows (one per Pareto-front member), with finite values in every metric column.

If any step fails, see [TROUBLESHOOTING.md](./TROUBLESHOOTING.md).

## Uninstalling

```bash
pip uninstall coreset_selection
rm -rf venv/
```

## Running the Test Suite

```bash
pip install -e ".[dev]"
pytest coreset_selection/tests/
```

Expected: all tests pass on Python 3.10+. Failures on older Python or missing optional dependencies (e.g., `pymoo`) are common; see the test output for specifics.
