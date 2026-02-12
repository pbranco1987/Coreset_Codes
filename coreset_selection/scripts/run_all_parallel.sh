#!/bin/bash
# ==============================================================================
# run_all_parallel.sh
# ==============================================================================
# Run all experiment scenarios (R0-R12) in parallel.
#
# This script provides multiple parallelization options:
# 1. GNU Parallel (recommended if available)
# 2. Background jobs with simple wait
# 3. Sequential execution (fallback)
#
# Usage:
#   ./run_all_parallel.sh                    # Auto-detect best method
#   ./run_all_parallel.sh --method=parallel  # Force GNU Parallel
#   ./run_all_parallel.sh --method=bg        # Force background jobs
#   ./run_all_parallel.sh --method=seq       # Force sequential
#   ./run_all_parallel.sh --scenarios=R1,R2,R3  # Run specific scenarios
#   ./run_all_parallel.sh --jobs=4           # Limit parallel jobs
# ==============================================================================

set -e

# Default configuration
DATA_DIR="${DATA_DIR:-data}"
OUTPUT_DIR="${OUTPUT_DIR:-runs_out}"
CACHE_DIR="${CACHE_DIR:-replicate_cache}"
SEED="${SEED:-123}"
DEVICE="${DEVICE:-cpu}"
PYTHON="${PYTHON:-python}"
N_REPLICATES="${N_REPLICATES:-}"
METHOD=""
JOBS=0
SCENARIOS=""
LOG_DIR="logs"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --method=*)
            METHOD="${1#*=}"
            shift
            ;;
        --jobs=*)
            JOBS="${1#*=}"
            shift
            ;;
        --scenarios=*)
            SCENARIOS="${1#*=}"
            shift
            ;;
        --n-replicates=*)
            N_REPLICATES="${1#*=}"
            shift
            ;;
        --data-dir=*)
            DATA_DIR="${1#*=}"
            shift
            ;;
        --output-dir=*)
            OUTPUT_DIR="${1#*=}"
            shift
            ;;
        --cache-dir=*)
            CACHE_DIR="${1#*=}"
            shift
            ;;
        --seed=*)
            SEED="${1#*=}"
            shift
            ;;
        --device=*)
            DEVICE="${1#*=}"
            shift
            ;;
        --python=*)
            PYTHON="${1#*=}"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --method=METHOD       Parallelization method: parallel, bg, seq (auto-detect)"
            echo "  --jobs=N              Number of parallel jobs (0=auto)"
            echo "  --scenarios=S         Comma-separated scenarios to run (default: all R0-R12)"
            echo "  --n-replicates=N      Number of replicates per scenario (default: from RunSpec)"
            echo "  --data-dir=DIR        Data directory (default: data)"
            echo "  --output-dir=DIR      Output directory (default: runs_out)"
            echo "  --cache-dir=DIR       Cache directory (default: replicate_cache)"
            echo "  --seed=N              Random seed (default: 123)"
            echo "  --device=DEV          Compute device: cpu or cuda (default: cpu)"
            echo "  --python=PATH         Python executable (default: python)"
            echo "  --help                Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set default scenarios if not specified
if [ -z "$SCENARIOS" ]; then
    SCENARIOS="R0,R1,R2,R3,R4,R5,R6,R7,R8,R9,R10,R11,R12"
fi

# Convert comma-separated to array
IFS=',' read -ra SCENARIO_ARRAY <<< "$SCENARIOS"

# Auto-detect parallelization method
if [ -z "$METHOD" ]; then
    if command -v parallel &> /dev/null; then
        METHOD="parallel"
    else
        METHOD="bg"
    fi
fi

# Auto-detect number of jobs
if [ "$JOBS" -eq 0 ]; then
    if [ -f /proc/cpuinfo ]; then
        JOBS=$(grep -c ^processor /proc/cpuinfo)
    elif command -v sysctl &> /dev/null; then
        JOBS=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
    else
        JOBS=4
    fi
fi

# Create log directory
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "CORESET SELECTION PARALLEL RUNNER"
echo "============================================================"
echo "Scenarios: ${SCENARIO_ARRAY[*]}"
echo "Method: $METHOD"
echo "Jobs: $JOBS"
echo "Data dir: $DATA_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Cache dir: $CACHE_DIR"
echo "Device: $DEVICE"
echo "============================================================"
echo ""

# Function to run a single scenario
run_scenario() {
    local scenario=$1
    local log_file="$LOG_DIR/${scenario}_$(date +%Y%m%d_%H%M%S).log"
    
    echo "[$(date +%H:%M:%S)] Starting $scenario..."
    
    # Build command with optional n-replicates
    local cmd="$PYTHON -m coreset_selection.run_scenario $scenario"
    cmd="$cmd --data-dir $DATA_DIR"
    cmd="$cmd --output-dir $OUTPUT_DIR"
    cmd="$cmd --cache-dir $CACHE_DIR"
    cmd="$cmd --seed $SEED"
    cmd="$cmd --device $DEVICE"
    
    if [ -n "$N_REPLICATES" ]; then
        cmd="$cmd --n-replicates $N_REPLICATES"
    fi
    
    eval "$cmd" > "$log_file" 2>&1
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] $scenario completed successfully"
    else
        echo "[$(date +%H:%M:%S)] $scenario FAILED (exit code: $exit_code)"
        echo "  Log: $log_file"
    fi
    
    return $exit_code
}

export -f run_scenario
export DATA_DIR OUTPUT_DIR CACHE_DIR SEED DEVICE PYTHON LOG_DIR N_REPLICATES

START_TIME=$(date +%s)

case $METHOD in
    parallel)
        echo "Using GNU Parallel with $JOBS workers..."
        echo ""
        
        printf '%s\n' "${SCENARIO_ARRAY[@]}" | \
            parallel --jobs "$JOBS" --halt soon,fail=1 --progress \
            "run_scenario {}"
        ;;
    
    bg)
        echo "Using background jobs with $JOBS workers..."
        echo ""
        
        # Track running jobs
        declare -a PIDS
        declare -A PID_SCENARIO
        RUNNING=0
        FAILED=0
        
        for scenario in "${SCENARIO_ARRAY[@]}"; do
            # Wait if we've reached max jobs
            while [ $RUNNING -ge $JOBS ]; do
                for pid in "${PIDS[@]}"; do
                    if ! kill -0 "$pid" 2>/dev/null; then
                        wait "$pid" || FAILED=$((FAILED + 1))
                        RUNNING=$((RUNNING - 1))
                        PIDS=("${PIDS[@]/$pid}")
                    fi
                done
                sleep 1
            done
            
            # Start new job
            run_scenario "$scenario" &
            PID=$!
            PIDS+=("$PID")
            PID_SCENARIO[$PID]=$scenario
            RUNNING=$((RUNNING + 1))
        done
        
        # Wait for remaining jobs
        echo ""
        echo "Waiting for remaining jobs to complete..."
        for pid in "${PIDS[@]}"; do
            if [ -n "$pid" ]; then
                wait "$pid" || FAILED=$((FAILED + 1))
            fi
        done
        
        if [ $FAILED -gt 0 ]; then
            echo ""
            echo "WARNING: $FAILED scenario(s) failed"
            exit 1
        fi
        ;;
    
    seq)
        echo "Running sequentially..."
        echo ""
        
        FAILED=0
        for scenario in "${SCENARIO_ARRAY[@]}"; do
            run_scenario "$scenario" || FAILED=$((FAILED + 1))
        done
        
        if [ $FAILED -gt 0 ]; then
            echo ""
            echo "WARNING: $FAILED scenario(s) failed"
            exit 1
        fi
        ;;
    
    *)
        echo "Unknown method: $METHOD"
        exit 1
        ;;
esac

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "============================================================"
echo "COMPLETED"
echo "============================================================"
echo "Total time: ${ELAPSED}s"
echo "Logs: $LOG_DIR/"
echo "============================================================"
