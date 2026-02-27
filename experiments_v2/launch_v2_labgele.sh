#!/usr/bin/env bash
# ============================================================================
# LABGELE — v2 Full Experiment Launch (tmux, 205 jobs)
#
# Creates a tmux session "v2_labgele" with one window per job.
# Navigate windows: Ctrl+B,N (next) / Ctrl+B,P (prev) / Ctrl+B,W (list)
#
# Jobs breakdown:
#   Block 1: K_vae   — 7 k-values x 5 reps = 35 windows
#   Block 2: K_raw   — 7 k-values x 5 reps = 35 windows
#   Block 3: K_pca   — 7 k-values x 5 reps = 35 windows
#   Block 4: N_v     — 5 constraints x 5 reps = 25 windows
#   Block 5: N_r     — 7 constraints x 5 reps = 35 windows
#   Block 6: N_p     — 7 constraints x 5 reps = 35 windows
#   Block 7: T_eff   — 5 reps (each sweeps 6 effort levels internally) = 5 windows
#                                                           Total: 205 windows
#
# Seeds: base seed 2026, reps 0-4 -> seeds 2026-2030
# Cache: replicate_cache_seed2026
# Output: experiments_v2
#
# Usage:
#   ssh -p 2222 jupyter-pbranco@161.24.23.23
#   cd ~/Coreset_Codes
#   bash experiments_v2/launch_v2_labgele.sh
# ============================================================================
set -euo pipefail

SESSION="v2_labgele"
SEED=2026
CACHE_DIR="replicate_cache_seed2026"
OUTPUT_DIR="experiments_v2"
K_GRID=(30 50 100 200 300 400 500)
REPS=(0 1 2 3 4)

# Thread control (each job is single-threaded for maximum parallelism)
ENV_VARS="export CORESET_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
MKL_THREADING_LAYER=GNU OMP_MAX_ACTIVE_LEVELS=1 PYTHONIOENCODING=utf-8 \
TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1"

# Kill existing session if any
tmux kill-session -t "$SESSION" 2>/dev/null || true

echo "============================================================"
echo "  LABGELE — v2 Full Experiment Launch (205 jobs)"
echo "  Session: $SESSION"
echo "  Seed: $SEED   Cache: $CACHE_DIR   Output: $OUTPUT_DIR"
echo "============================================================"

WIN=0
FIRST=1

# ────────────────────────────────────────────────────────────────
# Helper: create a tmux window running a command
# Usage: launch_window "window_name" "run_id" "extra_args"
# ────────────────────────────────────────────────────────────────
launch_window() {
  local wname="$1"
  local run_id="$2"
  local extra="$3"
  local cmd="cd ~/Coreset_Codes && source venv/bin/activate && $ENV_VARS && python3 -m coreset_selection scenario --run-id $run_id $extra --seed $SEED --cache-dir $CACHE_DIR --output-dir $OUTPUT_DIR; echo 'DONE'; read"

  if [ "$FIRST" -eq 1 ]; then
    tmux new-session -d -s "$SESSION" -n "$wname" "$cmd"
    FIRST=0
  else
    tmux new-window -t "$SESSION" -n "$wname" "$cmd"
  fi
  WIN=$((WIN + 1))
}

# ================================================================
# BLOCK 1: K_vae — Primary cardinality sweep, VAE space (35 jobs)
# ================================================================
echo "[Block 1] K_vae: 7 k-values x 5 reps = 35 jobs..."
for K in "${K_GRID[@]}"; do
  for R in "${REPS[@]}"; do
    launch_window "K_vae_k${K}_r${R}" "K_vae" "-k $K --rep-ids $R"
  done
done

# ================================================================
# BLOCK 2: K_raw — Primary cardinality sweep, RAW space (35 jobs)
# ================================================================
echo "[Block 2] K_raw: 7 k-values x 5 reps = 35 jobs..."
for K in "${K_GRID[@]}"; do
  for R in "${REPS[@]}"; do
    launch_window "K_raw_k${K}_r${R}" "K_raw" "-k $K --rep-ids $R"
  done
done

# ================================================================
# BLOCK 3: K_pca — Primary cardinality sweep, PCA space (35 jobs)
# ================================================================
echo "[Block 3] K_pca: 7 k-values x 5 reps = 35 jobs..."
for K in "${K_GRID[@]}"; do
  for R in "${REPS[@]}"; do
    launch_window "K_pca_k${K}_r${R}" "K_pca" "-k $K --rep-ids $R"
  done
done

# ================================================================
# BLOCK 4: N_v — NSGA-II, VAE grid, 5 constraint modes (25 jobs)
#   Constraints: ph, ms, mh, sh, hs
# ================================================================
echo "[Block 4] N_v: 5 constraints x 5 reps = 25 jobs..."
for C in ph ms mh sh hs; do
  for R in "${REPS[@]}"; do
    launch_window "N_v_${C}_r${R}" "N_v_${C}" "-k 100 --rep-ids $R"
  done
done

# ================================================================
# BLOCK 5: N_r — NSGA-II, RAW grid, 7 constraint modes (35 jobs)
#   Constraints: 0, ph, mh, ms, hh, sh, hs
# ================================================================
echo "[Block 5] N_r: 7 constraints x 5 reps = 35 jobs..."
for C in 0 ph mh ms hh sh hs; do
  for R in "${REPS[@]}"; do
    launch_window "N_r_${C}_r${R}" "N_r_${C}" "-k 100 --rep-ids $R"
  done
done

# ================================================================
# BLOCK 6: N_p — NSGA-II, PCA grid, 7 constraint modes (35 jobs)
#   Constraints: 0, ph, mh, ms, hh, sh, hs
# ================================================================
echo "[Block 6] N_p: 7 constraints x 5 reps = 35 jobs..."
for C in 0 ph mh ms hh sh hs; do
  for R in "${REPS[@]}"; do
    launch_window "N_p_${C}_r${R}" "N_p_${C}" "-k 100 --rep-ids $R"
  done
done

# ================================================================
# BLOCK 7: T_eff — Effort sweep (5 jobs, each sweeps 6 levels)
# ================================================================
echo "[Block 7] T_eff: 5 reps = 5 jobs..."
for R in "${REPS[@]}"; do
  launch_window "T_eff_r${R}" "T_eff" "-k 100 --rep-ids $R"
done

# ================================================================
echo ""
echo "============================================================"
echo "  LAUNCHED $WIN jobs in tmux session '$SESSION'"
echo ""
echo "  Attach:    tmux attach -t $SESSION"
echo "  Navigate:  Ctrl+B,N (next) / Ctrl+B,P (prev)"
echo "  List:      Ctrl+B,W (window picker)"
echo "============================================================"

echo "All 205 windows launched."
