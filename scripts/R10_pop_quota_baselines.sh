#!/usr/bin/env bash
# ============================================================================
# LOCAL MACHINE — Pop-Quota + Joint Baseline Experiments (42 jobs)
#
# Experiment structure:
#   Pop-quota:   7 k-values x 5 reps = 35 jobs  (Jobs 1-35)
#   Joint-quota: 7 k-values x 1 rep  =  7 jobs  (Jobs 36-42)
#   Total: 42 parallel jobs
#
# Each job runs 8 baseline methods across 3 spaces (raw, vae, pca)
# and evaluates in raw space.  Results match R10 pipeline format.
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

# Thread control — 42 parallel jobs, limit threads per process
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

TOTAL_JOBS=42

echo "============================================================"
echo "  Pop-Quota + Joint Baseline Experiments (seed=$SEED)"
echo ""
echo "  Pop-quota:   7k x 5 reps = 35 jobs  (Jobs 1-35)"
echo "  Joint-quota: 7k x 1 rep  =  7 jobs  (Jobs 36-42)"
echo "  Total: $TOTAL_JOBS parallel jobs"
echo ""
echo "  Regime: pop_quota (P-prefix) + joint_quota (J-prefix)"
echo "  Spaces: raw, vae, pca"
echo "  Output: $OUTPUT_DIR"
echo "============================================================"
echo ""

PIDS=()
JOB=0

# ── Pop-quota: 7 k-values x 5 reps = 35 jobs (Jobs 1-35) ──
echo "--- Launching pop-quota jobs (35 jobs) ---"
for K in 30 50 100 200 300 400 500; do
    for REP in 0 1 2 3 4; do
        JOB=$((JOB + 1))
        LOG="$LOGDIR/pop_quota_k${K}_rep${REP}.log"
        echo "  [Job $JOB/$TOTAL_JOBS] pop_quota k=$K rep=$REP -> $LOG"
        python -m coreset_selection.scripts.run_pop_baselines \
            --k "$K" --rep-id "$REP" --regime pop_quota \
            --spaces "raw,vae,pca" \
            --cache-dir "$CACHE_DIR" --output-dir "$OUTPUT_DIR" \
            --seed "$SEED" \
            --job-num "$JOB" --total-jobs "$TOTAL_JOBS" \
            > "$LOG" 2>&1 &
        PIDS+=($!)
    done
done

# ── Joint-quota: 7 k-values x 1 rep = 7 jobs (Jobs 36-42) ──
echo ""
echo "--- Launching joint-quota jobs (7 jobs) ---"
for K in 30 50 100 200 300 400 500; do
    JOB=$((JOB + 1))
    LOG="$LOGDIR/joint_quota_k${K}_rep0.log"
    echo "  [Job $JOB/$TOTAL_JOBS] joint_quota k=$K rep=0 -> $LOG"
    python -m coreset_selection.scripts.run_pop_baselines \
        --k "$K" --rep-id 0 --regime joint_quota \
        --spaces "raw,vae,pca" \
        --cache-dir "$CACHE_DIR" --output-dir "$OUTPUT_DIR" \
        --seed "$SEED" \
        --job-num "$JOB" --total-jobs "$TOTAL_JOBS" \
        > "$LOG" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "All $TOTAL_JOBS jobs launched. PIDs: ${PIDS[*]}"
echo "Logs: $LOGDIR/"
echo ""
echo "Monitor:"
echo "  tail -f $LOGDIR/pop_quota_k100_rep0.log       # watch a pop-quota job"
echo "  tail -f $LOGDIR/joint_quota_k100_rep0.log      # watch a joint job"
echo "  ps aux | grep run_pop_baselines | grep -v grep | wc -l   # count active"
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
    echo "  ALL $TOTAL_JOBS JOBS COMPLETED SUCCESSFULLY"
else
    echo "  $FAILED / $TOTAL_JOBS JOBS FAILED (check logs)"
fi
echo "============================================================"
