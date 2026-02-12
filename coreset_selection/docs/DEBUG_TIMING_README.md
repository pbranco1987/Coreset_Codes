# Debug Timing for Coreset Selection Bottleneck Analysis

This document explains how to use the debug timing instrumentation added to identify performance bottlenecks in the coreset selection pipeline.

## Quick Start

To enable detailed timing logs, set the `CORESET_DEBUG` environment variable:

```bash
# Enable debug timing
export CORESET_DEBUG=1

# Run your experiment
python -m coreset_selection.run_scenario R1 --k-values 100 --rep-ids 0

# Or run inline:
CORESET_DEBUG=1 python -m coreset_selection.run_scenario R1 --k-values 100 --rep-ids 0
```

## What Gets Timed

The instrumentation covers all major stages **before** and **during** NSGA-II:

### 1. Cache Building (`build_replicate_cache`)
- `load_data_manager` - Loading raw data from disk
- `extract_raw_data` - Extracting features and labels  
- `build_geo_info_for_cache` - Building geographic information
- `stratified_split` - Creating train/val splits
- `preprocessing` - Data preprocessing and standardization
- `create_eval_split` - Creating evaluation splits
- `VAE_training` - Training the VAE model (often the biggest bottleneck!)
- `VAE_embedding` - Computing VAE embeddings for all data
- `PCA_fitting` - Fitting PCA and transforming data
- `save_cache_to_disk` - Saving cache file

### 2. Cache Loading (`load_replicate_cache`)
- Time to load the .npz cache file

### 3. Objective Computer Building (`build_space_objective_computer`)
- `compute_median_sq_dist` - Computing median squared distance for bandwidth
- `build_RFFMMD` - Building Random Fourier Feature MMD estimator
- `build_AnchorSinkhorn` - Building Sinkhorn divergence estimator
  - `select_anchors` - K-means clustering for anchor selection
  - `compute_cost_scale` - Computing cost scaling
  - `compute_cost_matrices` - Computing pairwise distance matrices

### 4. Geographic Projector Building
- `build_geo_info` - Building geographic information
- `build_projector` - Building constraint projector

### 5. NSGA-II Optimization
- Overall timing for the optimization loop

## Sample Output

```
[12:34:56] [DEBUG-TIMING] ▶ START [   0.00s] ExperimentRunner.run ({'run_id': 'R1', 'rep_id': 0})
[12:34:56] [DEBUG-TIMING]   ▶ START [   0.01s] ensure_replicate_cache
[12:34:56] [DEBUG-TIMING]     ▶ START [   0.02s] build_replicate_cache ({'rep_id': 0})
[12:34:56] [DEBUG-TIMING]       ▶ START [   0.03s] load_data_manager
[12:34:58] [DEBUG-TIMING]       ◀ END   [   2.15s] load_data_manager (took 2.123s)
[12:34:58] [DEBUG-TIMING]       ● CHECK [   2.16s] Data extracted ({'N': 5570, 'n_features': 621})
...
[12:35:45] [DEBUG-TIMING]       ▶ START [  49.12s] VAE_training ({'epochs': 100, 'latent_dim': 32})
[12:38:22] [DEBUG-TIMING]       ◀ END   [ 206.45s] VAE_training (took 157.330s)  <-- BOTTLENECK!
...
[12:39:00] [DEBUG-TIMING] ● CHECK [240.15s] Ready to start NSGA-II ({'k': 100, 'pop_size': 100, ...})
[12:39:00] [DEBUG-TIMING]   ▶ START [240.16s] NSGA-II optimization ({'space': 'vae', 'k': 100})
```

## Log Symbols

- `▶ START` - Beginning of a timed section
- `◀ END` - End of a timed section (includes duration)
- `● CHECK` - Checkpoint marker with metadata

## Saving Logs to File

To save logs to a file in addition to stderr:

```bash
export CORESET_DEBUG=1
export CORESET_DEBUG_LOG_DIR=./debug_logs

python -m coreset_selection.run_scenario R1 --k-values 100 --rep-ids 0
```

Logs will be saved to `./debug_logs/timing_YYYYMMDD_HHMMSS.log`.

## Timing Summary

At the end of execution, a summary table is printed showing all timed sections and their durations:

```
================================================================================
TIMING SUMMARY
================================================================================
ExperimentRunner.run: 312.456s
  ensure_replicate_cache: 210.234s
    build_replicate_cache: 208.567s
      load_data_manager: 2.123s
      VAE_training: 157.330s
      ...
  load_replicate_cache: 1.234s
  build_objective_computers: 15.678s
    build_RFFMMD: 3.456s
    build_AnchorSinkhorn: 12.222s
  NSGA-II optimization: 85.432s
================================================================================
TOTAL ELAPSED: 312.456s
================================================================================
```

## Typical Bottlenecks

Based on the codebase, these are likely bottleneck areas:

1. **VAE Training** (`VAE_training`) - Often the largest time sink, especially on CPU
   - Solution: Use GPU (`--device cuda`), reduce epochs, or use cached representations

2. **K-means for Anchors** (`select_anchors`) - Can be slow for large datasets
   - Solution: Reduce `n_anchors` in SinkhornConfig

3. **Cost Matrix Computation** (`compute_cost_matrices`) - O(N × n_anchors) memory/time
   - Solution: Reduce dataset size or anchor count

4. **Cache Building** - First run for each replicate is slow
   - Solution: Pre-build caches, reuse existing caches

5. **Data Loading** - Can be slow for large datasets
   - Solution: Use faster storage, convert to efficient formats

## Programmatic Usage

You can also enable debug mode programmatically:

```python
from coreset_selection.utils.debug_timing import enable_debug, timer

# Enable debugging
enable_debug()

# Use the timer directly
with timer.section("my_computation"):
    # ... your code ...
    timer.checkpoint("intermediate step", some_value=42)
```

## Disabling Debug Mode

Simply don't set the environment variable, or set it to `0`:

```bash
export CORESET_DEBUG=0
# or
unset CORESET_DEBUG
```

The timing code has minimal overhead when disabled.
