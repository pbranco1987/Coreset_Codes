# Troubleshooting

Common errors and their fixes. If you hit something not listed here, please open an issue.

## Installation

### `ModuleNotFoundError: No module named 'coreset_selection'`

**Cause:** the package is not installed, or the wrong Python interpreter is active.

**Fix:**
```bash
# Verify active interpreter
which python         # or: where python (Windows)

# Install in editable mode
pip install -e .

# Sanity check
python -c "import coreset_selection; print(coreset_selection.__version__)"
```

If you see the package path and `0.1.0` printed, the install is fine. If the interpreter path is unexpected, activate your virtualenv first.

---

### `torch not found` during `build_caches.py`

**Cause:** `torch` is an optional runtime dependency.

**Fix:** install the appropriate PyTorch wheel (see [INSTALL.md § GPU Setup](./INSTALL.md#gpu-setup-optional)) or the CPU wheel:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## Data Loading

### `KeyError` or `ValueError: column 'X' not found`

**Cause:** the raw CSV schema changed, or a column expected by the loader is absent.

**Fix:**
1. Check the column name with `head -n 1 data/smp_main.csv` (Linux/macOS) or Excel.
2. If a column was renamed, update the corresponding entry in `coreset_selection/data/preprocessing.py` (the `resolve_column` tolerates case/whitespace but not full renames).
3. If a column is genuinely missing, you may be working with a dataset version different from the one the pipeline was designed for.

---

### Brazilian decimal format errors

**Symptom:** `ValueError: could not convert string to float: '1.234,56'`.

**Cause:** the Brazilian locale uses `,` as decimal separator and `.` as thousands separator, but some pandas readers default to US format.

**Fix:** the pipeline handles this via `coreset_selection.data._br_to_float` — if you see this error, a column is being read through a code path that bypasses the helper. Check that your new loader wraps Brazilian-format columns in `_br_to_float` before numeric coercion.

---

### Missing `data/*.csv` files after clone

**Cause:** Git LFS did not download the data.

**Fix:**
```bash
git lfs install
git lfs pull
```

If you do not have LFS access, see [INSTALL.md § Installing the Raw Data](./INSTALL.md#installing-the-raw-data).

---

## Cache Building

### VAE training won't converge

**Symptom:** `build_caches.py` prints unusually high reconstruction loss or fails to early-stop within 1,500 epochs.

**Likely causes:**
1. **Data not properly standardised.** Check `X_scaled`: its columns should have mean ≈ 0 and std ≈ 1. If not, a preprocessing step failed.
2. **Latent dim too small for the feature diversity.** Try `latent_dim=64` via config override (`cfg.models.vae.latent_dim = 64`).
3. **Learning rate instability.** Reduce `cfg.models.vae.lr` from 1e-3 to 5e-4.
4. **Target columns leaked in.** Should not happen because `validate_no_leakage` fires first, but check that it actually ran (should print a validation message).

**Fix in order:** inspect data stats → try larger latent dim → lower lr.

---

### `ValueError: Leakage detected: the following target columns are still present`

**Cause:** a known evaluation target column leaked into the feature matrix.

**Fix:** this is a **safety feature, not a bug**. The listed columns are added to the `TARGET_COLUMN_PATTERNS` registry but then somehow survived feature selection. Inspect `coreset_selection/data/target_columns.py` and `coreset_selection/data/cache.py:remove_target_columns` — one of the two is not catching the offending column. Either add a more specific regex to `TARGET_COLUMN_PATTERNS` or include the column name in `cfg.preprocessing.target_columns`.

---

### `FileNotFoundError: assets.npz`

**Cause:** cache was never built, or the cache directory does not exist.

**Fix:**
```bash
# Rebuild the cache
python scripts/launchers/build_caches.py --reps <rep_id>

# Verify
ls experiments/adaptive_tau_k100_ps_vae/replicate_cache/rep<rep_id>/
# Expected: assets.npz
```

---

## NSGA-II / adaptive_tau

### `[PARETO-WARNING] No feasible solutions in returned front`

**Cause:** adaptive-tau failed to find a `τ` large enough to accommodate any feasible solution. This is usually a sign that `τ` is being held tighter than `KL_min(k)` (the analytical floor) or that the constraint is malformed.

**Fix:**
1. Inspect `adaptive-tau-log.json` — the final `tau_hi` should be strictly larger than `KL_min(k)`. If not, the probe phase never succeeded.
2. Increase `--k` — very small `k` may be infeasible under a given constraint (e.g., 5 coreset points with a 27-group pop-share constraint cannot hit all groups).
3. Relax `min_one_per_group` if it is enabled and `k < G`.

---

### Out-of-memory at large `k`

**Cause:** `RawSpaceEvaluator` caches `K_EE` (the full-E Gram matrix), which is `|E|² × 8 bytes`. With `|E| = 2000` that's 32 MB — fine — but some metric code materialises additional `|E| × k` matrices.

**Fix:**
1. Reduce RFF feature count (default 2000) in `MMDConfig(rff_dim=1000)`.
2. Reduce Sinkhorn anchor count from 200 to 100 via `SinkhornConfig(n_anchors=100)`.
3. If evaluating (not optimising), you may also reduce the Nyström approximation rank.

---

### Tau converges suspiciously fast or very slow

**Cause:** the probe phase is bracketing `τ` too wide (fast convergence masks a local trap) or too narrow (many bisect steps).

**Fix:**
1. Check `adaptive-tau-log.json` for the `state` transitions. Normal runs show: probe for ~30–90 gens, bisect for ~200–400 gens, production for `COMMITTED_GENS`.
2. If probe immediately succeeds at the greedy floor, your `τ` floor might be too loose. Inspect `greedy_kl_floor` in `manifest.json`.
3. If bisect oscillates, relax `BISECT_TOLERANCE` (default 5 %) via code override.

---

## Evaluation

### `S ∩ E` warnings during evaluation

**Not a bug.** The evaluator logs when it excludes overlapping indices between the coreset and the evaluation set. This is the documented S ∩ E overlap fix. You can quantify the overlap via `n_excluded` in the `_NystromCache` object.

---

### `metrics.csv` contains NaN values

**Causes:**
1. **Target column is all-NaN for the eval_test split.** Check that targets are not 100 % missing for the evaluation points.
2. **KRR regulariser too small.** `λ_nys = 1e-6 × tr(W)/k` — with degenerate `W`, the regularised system may be singular. Try bumping `λ_nys` to `1e-4 × tr(W)/k`.
3. **Division by zero in kPCA distortion.** If the full-eval kernel spectrum is numerically rank-deficient, the distortion metric produces NaN. Skip kPCA distortion for those cases.

---

### Cache mismatch error in `bootstrap_reeval.py`

**Symptom:** `RuntimeError: config hash does not match cache hash`.

**Cause:** you rebuilt the cache with different settings (latent dim, eval size, etc.) than the cache the bootstrap expects.

**Fix:**
- Either rebuild the experiment (cache + optimisation) in lock-step, or
- Use the original cache if you still have it, or
- Pass `--force-cache-hash` (if available) to skip the check (at your own risk).

---

### Championship table numbers don't match the manuscript

**Cause:** most likely you are running a pre-S∩E-fix branch or an older cache.

**Fix:**
1. Verify your code contains the S ∩ E fix: `grep -n 'n_excluded' coreset_selection/evaluation/raw_space.py` — should return at least one hit.
2. Verify the bootstrap/eval code is the fixed version (post-March 2026).
3. Re-run `evaluate_coresets.py --force` to regenerate `metrics.csv` from saved coresets.

---

## Scripts and Environment

### `python: can't open file 'scripts/launchers/adaptive_tau.py': [Errno 2] No such file or directory`

**Cause:** you are not running from the repository root.

**Fix:**
```bash
cd /path/to/Coreset_Codes     # always run from repo root
python scripts/launchers/adaptive_tau.py ...
```

All scripts resolve project paths relative to `Path(__file__).resolve().parents[2]`, so the working directory usually doesn't matter — but the command-line argument `scripts/...` is interpreted relative to CWD.

---

### Permission denied on shell scripts

**Symptom:** `bash: ./scripts/deploy/labgele.sh: Permission denied`.

**Fix:**
```bash
chmod +x scripts/deploy/labgele.sh
```

Or run via `bash`:
```bash
bash scripts/deploy/labgele.sh 300
```

---

### tmux session already exists

**Symptom:** `duplicate session: coreset1`.

**Fix:**
```bash
tmux kill-session -t coreset1
# Or list and kill selectively:
tmux list-sessions
tmux kill-session -t <name>
```

---

## Manuscript Generation (Removed)

**Note:** `scripts/build_manuscript.py` was intentionally removed in the onboarding-ready reorganisation. The repository no longer contains LaTeX-writing code; the `manuscript/generated/*.tex` and `.pdf` files are committed artifacts from the last paper compile. To regenerate them, you would need to reproduce the championship analysis and manually write LaTeX tables — this is out of scope for the public repository.

---

## Still stuck?

1. Check [CHANGELOG.md](./CHANGELOG.md) for known regressions.
2. Search existing issues.
3. Open a new issue with: (a) OS & Python version, (b) the command you ran, (c) the full traceback, (d) `git rev-parse HEAD`.
