#!/bin/bash
# Run a single bootstrap job. Called by xargs.
# Usage: run_one_bootstrap.sh RUN_ID REP_ID

# Limit numpy/BLAS threads to avoid contention with parallel jobs
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

cd ~/Coreset_Codes
source ~/Coreset_Codes/venv/bin/activate

RUN_ID="$1"
REP_ID="$2"
LOG="bootstrap_results/log_${RUN_ID}_rep$(printf %02d $REP_ID).txt"
FINAL="bootstrap_results/bootstrap_raw_${RUN_ID}_rep$(printf %02d $REP_ID).csv"

# Skip if already done
if [ -f "$FINAL" ]; then
    echo "[SKIP] $RUN_ID rep$(printf %02d $REP_ID) — already complete"
    exit 0
fi

echo "[START] $RUN_ID rep$(printf %02d $REP_ID) at $(date)"

python3 scripts/bootstrap_reeval.py \
    --run-id "$RUN_ID" \
    --rep-id "$REP_ID" \
    --n-bootstrap 50 \
    --output-dir bootstrap_results/ \
    > "$LOG" 2>&1

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "[DONE] $RUN_ID rep$(printf %02d $REP_ID) at $(date)"
else
    echo "[FAIL] $RUN_ID rep$(printf %02d $REP_ID) exit=$EXIT_CODE at $(date)"
fi
