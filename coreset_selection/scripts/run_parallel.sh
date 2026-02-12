#!/bin/bash
# Run multiple experiments in parallel with proper thread control
#
# Usage:
#   ./run_parallel.sh r1 r2 r3 r4     # Run r1-r4 in parallel
#   ./run_parallel.sh -j2 r1 r2       # Force 2 threads each
#   ./run_parallel.sh all             # Run r0-r9 in parallel
#
# Automatically calculates threads per job based on CPU cores.

set -e

# Detect CPU cores
NCORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)

# Parse arguments
THREADS=""
JOBS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -j*)
            if [[ ${#1} -gt 2 ]]; then
                THREADS="${1:2}"
            else
                THREADS="$2"
                shift
            fi
            ;;
        --threads=*)
            THREADS="${1#*=}"
            ;;
        all)
            JOBS+=(r0 r1 r2 r3 r4 r5 r6 r7 r8 r9)
            ;;
        r[0-9]|r10)
            JOBS+=("$1")
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
    shift
done

if [[ ${#JOBS[@]} -eq 0 ]]; then
    echo "Usage: $0 [-jN] r1 r2 r3 ... | all"
    echo ""
    echo "Examples:"
    echo "  $0 r1 r2 r3 r4       # Run 4 experiments in parallel"
    echo "  $0 -j4 r1 r2         # Force 4 threads each"
    echo "  $0 all               # Run all r0-r9"
    echo ""
    echo "Detected $NCORES CPU cores"
    exit 0
fi

NJOBS=${#JOBS[@]}

# Calculate threads per job if not specified
if [[ -z "$THREADS" ]]; then
    THREADS=$((NCORES / NJOBS))
    [[ $THREADS -lt 1 ]] && THREADS=1
fi

echo "=========================================="
echo "Parallel Experiment Runner"
echo "=========================================="
echo "CPU cores:    $NCORES"
echo "Jobs:         $NJOBS (${JOBS[*]})"
echo "Threads/job:  $THREADS"
echo "=========================================="
echo ""

# Launch all jobs in background
PIDS=()
for job in "${JOBS[@]}"; do
    echo "[$(date +%H:%M:%S)] Starting $job with $THREADS threads..."
    python -m coreset_selection -j"$THREADS" "$job" &
    PIDS+=($!)
done

echo ""
echo "All jobs launched. Waiting for completion..."
echo ""

# Wait for all jobs and collect exit codes
FAILED=()
for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    job=${JOBS[$i]}
    if wait "$pid"; then
        echo "[$(date +%H:%M:%S)] ✓ $job completed"
    else
        echo "[$(date +%H:%M:%S)] ✗ $job FAILED"
        FAILED+=("$job")
    fi
done

echo ""
echo "=========================================="
if [[ ${#FAILED[@]} -eq 0 ]]; then
    echo "All ${#JOBS[@]} jobs completed successfully!"
else
    echo "FAILED: ${FAILED[*]}"
    exit 1
fi
