#!/usr/bin/env bash
# ============================================================================
# LESSONIA — v2 Full Experiment Launch (tmux, 185 jobs)
#
# Creates a tmux session "v2_lessonia" with one window per job.
# Each window runs a single `python3 -m coreset_selection scenario` call.
#
# Job breakdown:
#   P  (Prerequisites/quota):  7 k-values × 1 rep              =   7 windows
#   A  (Ablations):            5 ablations × 5 reps             =  25 windows
#   B_v (Baselines, VAE):      8 constraints × 5 reps           =  40 windows
#   B_r (Baselines, RAW):      8 constraints × 5 reps           =  40 windows
#   B_p (Baselines, PCA):      8 constraints × 5 reps           =  40 windows
#   T_vdim (VAE dim sweep):    3 dims × 5 reps                  =  15 windows
#   T_pdim (PCA dim sweep):    3 dims × 5 reps                  =  15 windows
#   D  (Diagnostics):          3 spaces                          =   3 windows
#                                                         Total = 185 windows
#
# Seeds: base 2026, reps 0-4 -> effective seeds 2026-2030
# K_GRID = (30, 50, 100, 200, 300, 400, 500)
# D_GRID for dim sweep = {8, 16, 64} (d=32 is the control via K_vae/K_pca)
#
# Navigate windows: Ctrl+B,N (next) / Ctrl+B,P (prev) / Ctrl+B,W (list)
#
# Usage:
#   ssh -p 2222 pbranco@161.24.29.21
#   cd ~/Coreset_Codes
#   bash experiments_v2/launch_v2_lessonia.sh
# ============================================================================
set -euo pipefail

SESSION="v2_lessonia"
SEED=2026
CACHE_DIR="replicate_cache_seed2026"
OUTPUT_DIR="experiments_v2"

# Kill existing session if any
tmux kill-session -t "$SESSION" 2>/dev/null || true

echo "============================================================"
echo "  LESSONIA — v2 Full Experiment Launch (185 jobs)"
echo "  Session: $SESSION"
echo "  Seed: $SEED   Cache: $CACHE_DIR   Output: $OUTPUT_DIR"
echo "============================================================"

WIN=0

# ────────────────────────────────────────────────────────────────
# Helper: build the command string for a given window
# ────────────────────────────────────────────────────────────────
make_cmd() {
  local run_id="$1"
  local k="$2"
  local rep="$3"
  local extra="${4:-}"
  echo "cd ~/Coreset_Codes && source venv/bin/activate && python3 -m coreset_selection scenario --run-id ${run_id} -k ${k} --seed ${SEED} --rep-ids ${rep} --cache-dir ${CACHE_DIR} --output-dir ${OUTPUT_DIR} ${extra}; echo 'DONE'; read"
}

# ────────────────────────────────────────────────────────────────
# Helper: launch a tmux window (first window creates the session)
# ────────────────────────────────────────────────────────────────
launch() {
  local wname="$1"
  local cmd="$2"
  if [ "$WIN" -eq 0 ]; then
    tmux new-session -d -s "$SESSION" -n "$wname" "$cmd"
  else
    tmux new-window -t "$SESSION" -n "$wname" "$cmd"
  fi
  WIN=$((WIN + 1))
}

# ================================================================
# BLOCK 1: P (Prerequisites/quota) — 7 k-values, rep 0
# ================================================================
echo "[Block 1] P: Prerequisites/quota (7 windows)..."
for K in 30 50 100 200 300 400 500; do
  WNAME="P_k${K}"
  launch "$WNAME" "$(make_cmd P $K 0)"
done

# ================================================================
# BLOCK 2: A (Ablations) — 5 ablation types × 5 reps = 25 windows
# ================================================================
echo "[Block 2] A: Ablations (25 windows)..."
for ABLATION in mmd sink tri none jhh; do
  for REP in 0 1 2 3 4; do
    WNAME="A_${ABLATION}_r${REP}"
    launch "$WNAME" "$(make_cmd A_${ABLATION} 100 $REP)"
  done
done

# ================================================================
# BLOCK 3: B_v (Baselines, VAE) — 8 constraints × 5 reps = 40 windows
# ================================================================
echo "[Block 3] B_v: Baselines VAE (40 windows)..."
for CONSTRAINT in 0 ps ph ms mh hh sh hs; do
  for REP in 0 1 2 3 4; do
    WNAME="B_v_${CONSTRAINT}_r${REP}"
    launch "$WNAME" "$(make_cmd B_v_${CONSTRAINT} 100 $REP)"
  done
done

# ================================================================
# BLOCK 4: B_r (Baselines, RAW) — 8 constraints × 5 reps = 40 windows
# ================================================================
echo "[Block 4] B_r: Baselines RAW (40 windows)..."
for CONSTRAINT in 0 ps ph ms mh hh sh hs; do
  for REP in 0 1 2 3 4; do
    WNAME="B_r_${CONSTRAINT}_r${REP}"
    launch "$WNAME" "$(make_cmd B_r_${CONSTRAINT} 100 $REP)"
  done
done

# ================================================================
# BLOCK 5: B_p (Baselines, PCA) — 8 constraints × 5 reps = 40 windows
# ================================================================
echo "[Block 5] B_p: Baselines PCA (40 windows)..."
for CONSTRAINT in 0 ps ph ms mh hh sh hs; do
  for REP in 0 1 2 3 4; do
    WNAME="B_p_${CONSTRAINT}_r${REP}"
    launch "$WNAME" "$(make_cmd B_p_${CONSTRAINT} 100 $REP)"
  done
done

# ================================================================
# BLOCK 6: T_vdim (VAE dim sweep) — 3 dims × 5 reps = 15 windows
#   d=8, d=16, d=64 (d=32 is the control served by K_vae@k=100)
# ================================================================
echo "[Block 6] T_vdim: VAE dim sweep, 3 dims x 5 reps = 15 windows..."
for D in 8 16 64; do
  for REP in 0 1 2 3 4; do
    WNAME="T_vdim_d${D}_r${REP}"
    launch "$WNAME" "$(make_cmd T_vdim 100 $REP "--dim-override $D")"
  done
done

# ================================================================
# BLOCK 7: T_pdim (PCA dim sweep) — 3 dims × 5 reps = 15 windows
#   d=8, d=16, d=64 (d=32 is the control served by K_pca@k=100)
# ================================================================
echo "[Block 7] T_pdim: PCA dim sweep, 3 dims x 5 reps = 15 windows..."
for D in 8 16 64; do
  for REP in 0 1 2 3 4; do
    WNAME="T_pdim_d${D}_r${REP}"
    launch "$WNAME" "$(make_cmd T_pdim 100 $REP "--dim-override $D")"
  done
done

# ================================================================
# BLOCK 8: D (Diagnostics) — 3 spaces = 3 windows
# ================================================================
echo "[Block 8] D: Diagnostics, 3 spaces = 3 windows..."
for SPACE in vae raw pca; do
  WNAME="D_${SPACE}"
  launch "$WNAME" "$(make_cmd D_${SPACE} 100 0)"
done

# ================================================================
echo ""
echo "============================================================"
echo "  LESSONIA — v2 Experiment Launch"
echo "  Session: $SESSION"
echo "  Launched: $WIN windows"
echo ""
echo "  Attach:    tmux attach -t $SESSION"
echo "  Navigate:  Ctrl+B,N (next) / Ctrl+B,P (prev)"
echo "  List:      Ctrl+B,W (window picker)"
echo "============================================================"
echo "All 185 windows launched."
