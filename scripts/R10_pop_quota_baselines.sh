#!/usr/bin/env bash
# ============================================================================
# LOCAL MACHINE — Population-Share Quota Baselines
#
# Runs 8 baseline methods under pop-quota constraints across:
#   7 k values (30, 50, 100, 200, 300, 400, 500) x 5 reps (rep00-rep04)
#   = 35 parallel jobs
#
# Each job runs in all 3 spaces (raw, vae, pca) and evaluates in raw space.
# Results match the R10 pipeline format for downstream rank aggregation.
#
# For main paper Section VIII.E: R1 vs pop-quota baselines (5 reps, paired).
# For appendix: rep00 used for R8/R9 cross-space comparisons.
#
# Usage (Git Bash):
#   cd /c/Users/pbranco/Coreset_Codes
#   bash scripts/R10_pop_quota_baselines.sh
# ============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT_DIR="$(pwd)"

SEED=123
CACHE_DIR="replicate_cache"
OUTPUT_DIR="runs_out_pop_baselines"
LOGDIR="logs/pop_baselines"

mkdir -p "$OUTPUT_DIR" "$LOGDIR"

# Thread control — 35 parallel jobs, limit threads per process
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

NJOBS=35
echo "============================================================"
echo "  Pop-Quota Baselines (seed=$SEED)"
echo "  7 k x 5 reps = $NJOBS parallel jobs"
echo "  Regime: pop_quota (P-prefix methods)"
echo "  Spaces: raw, vae, pca"
echo "  Output: $OUTPUT_DIR"
echo "============================================================"
echo ""

PIDS=()
JOB=0

for K in 30 50 100 200 300 400 500; do
    for REP in 0 1 2 3 4; do
        JOB=$((JOB + 1))
        LOG="$LOGDIR/pop_quota_k${K}_rep${REP}.log"
        echo "[$JOB/$NJOBS] Launching pop_quota k=$K rep=$REP ..."
        python -m coreset_selection.scripts.run_pop_baselines \
            --k "$K" --rep-id "$REP" --regime pop_quota \
            --spaces "raw,vae,pca" \
            --cache-dir "$CACHE_DIR" --output-dir "$OUTPUT_DIR" \
            --seed "$SEED" \
            > "$LOG" 2>&1 &
        PIDS+=($!)
    done
done

echo ""
echo "All $NJOBS jobs launched. PIDs: ${PIDS[*]}"
echo "Logs: $LOGDIR/"
echo ""
echo "Monitor with:  tail -f $LOGDIR/pop_quota_k100_rep0.log"
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
    echo "  ALL $NJOBS POP-QUOTA JOBS COMPLETED SUCCESSFULLY"
else
    echo "  $FAILED / $NJOBS JOBS FAILED (check logs)"
fi
echo "============================================================"
