#!/usr/bin/env bash
# ============================================================================
# LOCAL MACHINE — Joint-Constrained Baselines
#
# Runs 8 baseline methods under joint (pop + muni) quota constraints:
#   7 k values (30, 50, 100, 200, 300, 400, 500) x 1 rep (rep00 only)
#   = 7 parallel jobs
#
# Each job runs in all 3 spaces (raw, vae, pca) and evaluates in raw space.
# Results match R10 pipeline format.
#
# For appendix: R5 (joint NSGA-II, 1 rep) vs joint baselines (1 rep).
# Only rep00 (seed 123) is needed since R5 has a single replicate.
#
# Usage (Git Bash):
#   cd /c/Users/pbranco/Coreset_Codes
#   bash scripts/R10_joint_baselines.sh
# ============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT_DIR="$(pwd)"

SEED=123
CACHE_DIR="replicate_cache"
OUTPUT_DIR="runs_out_pop_baselines"
LOGDIR="logs/joint_baselines"

mkdir -p "$OUTPUT_DIR" "$LOGDIR"

# Thread control
export CORESET_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MKL_THREADING_LAYER=GNU
export OMP_MAX_ACTIVE_LEVELS=1
export PYTHONIOENCODING=utf-8
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1

NJOBS=7
echo "============================================================"
echo "  Joint-Constrained Baselines (seed=$SEED)"
echo "  7 k x 1 rep = $NJOBS parallel jobs"
echo "  Regime: joint_quota (J-prefix methods)"
echo "  Spaces: raw, vae, pca"
echo "  Output: $OUTPUT_DIR"
echo "============================================================"
echo ""

PIDS=()
JOB=0

for K in 30 50 100 200 300 400 500; do
    JOB=$((JOB + 1))
    REP=0
    LOG="$LOGDIR/joint_quota_k${K}_rep${REP}.log"
    echo "[$JOB/$NJOBS] Launching joint_quota k=$K rep=$REP ..."
    python -m coreset_selection.scripts.run_pop_baselines \
        --k "$K" --rep-id "$REP" --regime joint_quota \
        --spaces "raw,vae,pca" \
        --cache-dir "$CACHE_DIR" --output-dir "$OUTPUT_DIR" \
        --seed "$SEED" \
        > "$LOG" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "All $NJOBS jobs launched. PIDs: ${PIDS[*]}"
echo "Logs: $LOGDIR/"
echo ""
echo "Monitor with:  tail -f $LOGDIR/joint_quota_k100_rep0.log"
echo "Check status:  ps aux | grep run_pop_baselines | grep -v grep | wc -l"
echo ""

# Wait for all jobs
FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "============================================================"
if [ "$FAILED" -eq 0 ]; then
    echo "  ALL $NJOBS JOINT-QUOTA JOBS COMPLETED SUCCESSFULLY"
else
    echo "  $FAILED / $NJOBS JOBS FAILED (check logs)"
fi
echo "============================================================"
