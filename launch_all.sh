#!/usr/bin/env bash
# ===========================================================================
# launch_all.sh  —  Run experiment scenarios (R0–R12) in parallel
#
# Shared cluster — uses only the cores YOU request, not all 192.
#
# TWO MODES:
#
#   1) INDIVIDUAL (default): Prints the commands for you to paste into
#      separate terminal tabs. You see live output in each tab.
#
#   2) BATCH (--batch): Runs all scenarios in background with log files.
#
# Usage:
#   # Print commands for 13 tabs, using 64 cores total
#   bash launch_all.sh
#
#   # Print commands using 96 cores total
#   bash launch_all.sh -c 96
#
#   # Print commands for only R1 and R10
#   bash launch_all.sh --scenarios R1,R10
#
#   # Batch mode: all in background, logs to logs/<scenario>.log
#   bash launch_all.sh --batch
#
#   # Batch + tmux: one pane per scenario (live output everywhere)
#   bash launch_all.sh --tmux
# ===========================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
DATA_DIR="$PROJECT_DIR/data"
OUTPUT_DIR="$PROJECT_DIR/runs_out"
CACHE_DIR="$PROJECT_DIR/replicate_cache"
LOG_DIR="$PROJECT_DIR/logs"
SEED=123

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
TOTAL_CORES=64         # conservative default — won't hog the cluster
MODE="individual"      # print per-tab commands
SKIP_CACHE=false
RESUME=false
SCENARIO_LIST=""

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--cores)       TOTAL_CORES="$2"; shift 2 ;;
        --cores=*)        TOTAL_CORES="${1#*=}"; shift ;;
        --seed)           SEED="$2"; shift 2 ;;
        --seed=*)         SEED="${1#*=}"; shift ;;
        --output-dir)     OUTPUT_DIR="$2"; shift 2 ;;
        --output-dir=*)   OUTPUT_DIR="${1#*=}"; shift ;;
        --cache-dir)      CACHE_DIR="$2"; shift 2 ;;
        --cache-dir=*)    CACHE_DIR="${1#*=}"; shift ;;
        --scenarios)      SCENARIO_LIST="$2"; shift 2 ;;
        --scenarios=*)    SCENARIO_LIST="${1#*=}"; shift ;;
        --batch)          MODE="batch"; shift ;;
        --tabs)           MODE="tabs"; shift ;;
        --tmux)           MODE="tmux"; shift ;;
        --skip-cache)     SKIP_CACHE=true; shift ;;
        --resume)         RESUME=true; shift ;;
        --help|-h)
            cat <<'HELPEOF'
Usage: bash launch_all.sh [OPTIONS]

Options:
  -c, --cores N        Total cores to use (default: 64).
                       Each scenario gets N / num_scenarios threads.
                       Be considerate of other cluster users.

  --seed N             Random seed (default: 123).
  --output-dir DIR     Output directory (default: $PROJECT_DIR/runs_out).
  --cache-dir DIR      Cache directory (default: $PROJECT_DIR/replicate_cache).

  --scenarios LIST     Comma-separated scenario IDs (default: all R0–R12).
                       Example: --scenarios R1,R2,R3,R10

  (default)            Print one command per scenario for manual terminal tabs.
  --batch              Run all in background (logs to logs/<scenario>.log).
  --tabs               Open gnome-terminal tabs automatically.
  --tmux               Create a tmux session with one pane per scenario.

  --skip-cache         Skip Phase 0 cache pre-build (if already built).

  --resume             Resume mode: skip completed (run, k, rep) combos,
                       re-run incomplete ones, create missing.

Examples:
  bash launch_all.sh                             # 64 cores, print tab commands
  bash launch_all.sh -c 96                       # 96 cores, print tab commands
  bash launch_all.sh -c 80 --scenarios R1,R10    # 80 cores, only R1 and R10
  bash launch_all.sh -c 96 --batch               # 96 cores, all in background
  bash launch_all.sh -c 96 --tmux                # 96 cores, tmux session
HELPEOF
            exit 0
            ;;
        *) echo "Unknown option: $1 (use --help)" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Resolve scenario list
# ---------------------------------------------------------------------------
if [ -n "$SCENARIO_LIST" ]; then
    IFS=',' read -ra SCENARIOS <<< "$SCENARIO_LIST"
else
    SCENARIOS=(R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14)
fi
N_SCENARIOS=${#SCENARIOS[@]}

# ---------------------------------------------------------------------------
# Thread budget
# ---------------------------------------------------------------------------
THREADS_PER_SCENARIO=$((TOTAL_CORES / N_SCENARIOS))
if [ "$THREADS_PER_SCENARIO" -lt 2 ]; then THREADS_PER_SCENARIO=2; fi
if [ "$THREADS_PER_SCENARIO" -gt 24 ]; then THREADS_PER_SCENARIO=24; fi

ACTUAL_CORES_USED=$((THREADS_PER_SCENARIO * N_SCENARIOS))
MACHINE_CORES=$(nproc 2>/dev/null || echo "?")

echo "================================================================"
echo "  CORESET SELECTION — PARALLEL LAUNCHER"
echo "================================================================"
echo "  Machine cores:         $MACHINE_CORES"
echo "  Cores requested:       $TOTAL_CORES"
echo "  Scenarios:             $N_SCENARIOS (${SCENARIOS[*]})"
echo "  Threads per scenario:  $THREADS_PER_SCENARIO"
echo "  Actual cores used:     $ACTUAL_CORES_USED / $MACHINE_CORES"
echo "  Mode:                  $MODE"
echo "  Seed:                  $SEED"
echo "  Output dir:            $OUTPUT_DIR"
echo "  Cache dir:             $CACHE_DIR"
echo "  Resume:                $RESUME"
echo "================================================================"
echo ""

# Build resume flag string
RESUME_FLAG=""
if [ "$RESUME" = true ]; then RESUME_FLAG="--resume"; fi

if [ "$ACTUAL_CORES_USED" -gt "$((MACHINE_CORES * 3 / 4))" ] 2>/dev/null; then
    echo "  WARNING: Using $ACTUAL_CORES_USED of $MACHINE_CORES cores (>75%)."
    echo "  Consider reducing with -c if other users share this cluster."
    echo ""
fi

mkdir -p "$LOG_DIR" "$OUTPUT_DIR" "$CACHE_DIR"

# ---------------------------------------------------------------------------
# Phase 0: Pre-build replicate caches (sequential, one-time cost)
# ---------------------------------------------------------------------------
if [ "$SKIP_CACHE" = false ]; then
    CACHE_THREADS=$TOTAL_CORES
    if [ "$CACHE_THREADS" -gt 48 ]; then CACHE_THREADS=48; fi

    echo "[Phase 0] Pre-building replicate caches (VAE + PCA)..."
    echo "[Phase 0] Using $CACHE_THREADS threads for cache build."
    echo "[Phase 0] Replicates needed: 0, 1, 2, 3, 4"
    echo ""

    cd "$PROJECT_DIR"
    OMP_NUM_THREADS=$CACHE_THREADS \
    MKL_NUM_THREADS=$CACHE_THREADS \
    OPENBLAS_NUM_THREADS=$CACHE_THREADS \
    NUMEXPR_MAX_THREADS=$CACHE_THREADS \
    python -c "
import sys, os
sys.path.insert(0, '.')
from coreset_selection.cli import build_base_config
from coreset_selection.data.cache import prebuild_full_cache

base_cfg = build_base_config(
    output_dir='$OUTPUT_DIR',
    cache_dir='$CACHE_DIR',
    data_dir='$DATA_DIR',
    seed=$SEED,
)
for rep in range(5):
    print(f'[Phase 0] Building cache for replicate {rep}...')
    prebuild_full_cache(base_cfg, rep, seed=$SEED)
    print(f'[Phase 0] Replicate {rep} done.')
print('[Phase 0] All caches built.')
"
    echo ""
    echo "[Phase 0] Cache pre-build complete."
    echo ""
fi

# ---------------------------------------------------------------------------
# Phase 1: Launch scenarios
# ---------------------------------------------------------------------------

# Environment block that each terminal/process needs
ENV_BLOCK="export CORESET_NUM_THREADS=$THREADS_PER_SCENARIO
export OMP_NUM_THREADS=$THREADS_PER_SCENARIO
export MKL_NUM_THREADS=$THREADS_PER_SCENARIO
export OPENBLAS_NUM_THREADS=$THREADS_PER_SCENARIO
export NUMEXPR_MAX_THREADS=$THREADS_PER_SCENARIO
export VECLIB_MAXIMUM_THREADS=$THREADS_PER_SCENARIO
export MKL_THREADING_LAYER=GNU
export OMP_MAX_ACTIVE_LEVELS=1
export KMP_BLOCKTIME=200"

case "$MODE" in

# =======================================================================
# INDIVIDUAL MODE (default) — print one command per tab
# =======================================================================
individual)
    echo "================================================================"
    echo "  STEP 1: In EACH new terminal tab, paste this env block FIRST:"
    echo "================================================================"
    echo ""
    echo "$ENV_BLOCK"
    echo "cd $PROJECT_DIR"
    echo ""
    echo "================================================================"
    echo "  STEP 2: Then paste the command for that tab's scenario:"
    echo "================================================================"
    echo ""

    for RUN_ID in "${SCENARIOS[@]}"; do
        echo "# --- $RUN_ID ---"
        echo "python -m coreset_selection.run_scenario $RUN_ID --data-dir $DATA_DIR --output-dir $OUTPUT_DIR --cache-dir $CACHE_DIR --seed $SEED --parallel-experiments $N_SCENARIOS $RESUME_FLAG"
        echo ""
    done

    echo "================================================================"
    echo "  Or, copy-paste one FULL block per tab (env + command together):"
    echo "================================================================"
    echo ""

    for RUN_ID in "${SCENARIOS[@]}"; do
        echo "# ===================== $RUN_ID ====================="
        echo "$ENV_BLOCK"
        echo "cd $PROJECT_DIR"
        echo "python -m coreset_selection.run_scenario $RUN_ID --data-dir $DATA_DIR --output-dir $OUTPUT_DIR --cache-dir $CACHE_DIR --seed $SEED --parallel-experiments $N_SCENARIOS $RESUME_FLAG"
        echo ""
    done

    echo "================================================================"
    echo "  $N_SCENARIOS scenarios × $THREADS_PER_SCENARIO threads each"
    echo "  = $ACTUAL_CORES_USED cores used out of $MACHINE_CORES"
    echo "================================================================"
    ;;

# =======================================================================
# BATCH MODE — all in background with log files
# =======================================================================
batch)
    export CORESET_NUM_THREADS=$THREADS_PER_SCENARIO
    export OMP_NUM_THREADS=$THREADS_PER_SCENARIO
    export MKL_NUM_THREADS=$THREADS_PER_SCENARIO
    export OPENBLAS_NUM_THREADS=$THREADS_PER_SCENARIO
    export NUMEXPR_MAX_THREADS=$THREADS_PER_SCENARIO
    export VECLIB_MAXIMUM_THREADS=$THREADS_PER_SCENARIO
    export MKL_THREADING_LAYER=GNU
    export OMP_MAX_ACTIVE_LEVELS=1
    export KMP_BLOCKTIME=200

    cd "$PROJECT_DIR"
    PIDS=()
    for RUN_ID in "${SCENARIOS[@]}"; do
        LOG_FILE="$LOG_DIR/${RUN_ID}.log"
        echo "  Starting $RUN_ID → $LOG_FILE"

        python -m coreset_selection.run_scenario "$RUN_ID" \
            --data-dir "$DATA_DIR" \
            --output-dir "$OUTPUT_DIR" \
            --cache-dir "$CACHE_DIR" \
            --seed $SEED \
            --parallel-experiments $N_SCENARIOS \
            --fail-fast $RESUME_FLAG \
            > "$LOG_FILE" 2>&1 &

        PIDS+=($!)
    done

    echo ""
    echo "  All $N_SCENARIOS scenarios launched."
    echo "  Monitor:  tail -f $LOG_DIR/*.log"
    echo ""

    echo "Waiting for all scenarios to complete..."
    FAILED=0
    for i in "${!SCENARIOS[@]}"; do
        RUN_ID="${SCENARIOS[$i]}"
        PID="${PIDS[$i]}"
        if wait "$PID"; then
            echo "  [OK]   $RUN_ID"
        else
            echo "  [FAIL] $RUN_ID (exit code $?)"
            FAILED=$((FAILED + 1))
        fi
    done

    echo ""
    if [ "$FAILED" -eq 0 ]; then
        echo "ALL $N_SCENARIOS SCENARIOS COMPLETED SUCCESSFULLY"
    else
        echo "$FAILED / $N_SCENARIOS SCENARIOS FAILED — check $LOG_DIR/"
    fi
    ;;

# =======================================================================
# GNOME-TERMINAL TABS MODE
# =======================================================================
tabs)
    TAB_ARGS=()
    for RUN_ID in "${SCENARIOS[@]}"; do
        CMD="export CORESET_NUM_THREADS=$THREADS_PER_SCENARIO; \
export OMP_NUM_THREADS=$THREADS_PER_SCENARIO; \
export MKL_NUM_THREADS=$THREADS_PER_SCENARIO; \
export OPENBLAS_NUM_THREADS=$THREADS_PER_SCENARIO; \
export NUMEXPR_MAX_THREADS=$THREADS_PER_SCENARIO; \
export MKL_THREADING_LAYER=GNU; \
export OMP_MAX_ACTIVE_LEVELS=1; \
export KMP_BLOCKTIME=200; \
cd '$PROJECT_DIR' && \
python -m coreset_selection.run_scenario $RUN_ID \
    --data-dir '$DATA_DIR' \
    --output-dir '$OUTPUT_DIR' \
    --cache-dir '$CACHE_DIR' \
    --seed $SEED \
    --parallel-experiments $N_SCENARIOS $RESUME_FLAG; \
echo ''; echo '=== $RUN_ID FINISHED (exit \$?) ==='; read -p 'Press Enter to close.'"

        TAB_ARGS+=(--tab --title="$RUN_ID" -- bash -c "$CMD")
    done

    echo "Opening gnome-terminal with $N_SCENARIOS tabs..."
    gnome-terminal "${TAB_ARGS[@]}"
    echo "Done. Each tab runs with $THREADS_PER_SCENARIO threads ($ACTUAL_CORES_USED cores total)."
    ;;

# =======================================================================
# TMUX MODE — one pane per scenario, live output everywhere
# =======================================================================
tmux)
    SESSION="coreset_seed${SEED}"
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    tmux new-session -d -s "$SESSION" -x 200 -y 50

    ENV_INLINE="export CORESET_NUM_THREADS=$THREADS_PER_SCENARIO; \
export OMP_NUM_THREADS=$THREADS_PER_SCENARIO; \
export MKL_NUM_THREADS=$THREADS_PER_SCENARIO; \
export OPENBLAS_NUM_THREADS=$THREADS_PER_SCENARIO; \
export NUMEXPR_MAX_THREADS=$THREADS_PER_SCENARIO; \
export MKL_THREADING_LAYER=GNU; \
export OMP_MAX_ACTIVE_LEVELS=1; \
export KMP_BLOCKTIME=200; \
cd '$PROJECT_DIR'"

    for i in "${!SCENARIOS[@]}"; do
        RUN_ID="${SCENARIOS[$i]}"

        if [ "$i" -eq 0 ]; then
            tmux send-keys -t "$SESSION" "$ENV_INLINE" Enter
        else
            tmux split-window -t "$SESSION" -h
            tmux send-keys -t "$SESSION" "$ENV_INLINE" Enter
            tmux select-layout -t "$SESSION" tiled
        fi

        tmux send-keys -t "$SESSION" \
            "python -m coreset_selection.run_scenario $RUN_ID \
--data-dir '$DATA_DIR' --output-dir '$OUTPUT_DIR' --cache-dir '$CACHE_DIR' \
--seed $SEED --parallel-experiments $N_SCENARIOS $RESUME_FLAG" Enter
    done

    tmux select-layout -t "$SESSION" tiled
    echo "tmux session '$SESSION' created with $N_SCENARIOS panes."
    echo "  Cores used: $ACTUAL_CORES_USED / $MACHINE_CORES"
    echo "  Attach: tmux attach -t $SESSION"
    echo "  Detach: Ctrl-b d"
    tmux attach -t "$SESSION"
    ;;

esac
