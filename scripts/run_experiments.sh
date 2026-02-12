#!/usr/bin/env bash
# ============================================================================
# run_experiments.sh — Launch all experiments inside a tmux session
#
# Usage:
#   bash scripts/run_experiments.sh --seed 123 --server-name labgele
#   bash scripts/run_experiments.sh --seed 456 --server-name lessonia
#
# What it does:
#   1. Auto-detects CPU cores and configures thread limits
#   2. Creates a tmux session named "coreset_<server>"
#   3. Runs ALL scenarios (R0-R14) via the parallel runner
#   4. Logs all output to logs/<server>_<timestamp>.log
#   5. Writes a DONE marker when finished (for monitoring)
#
# The tmux session survives SSH disconnects. Reconnect with:
#   tmux attach -t coreset_<server>
# ============================================================================
set -euo pipefail

# ---- Parse arguments ----
SEED=123
SERVER_NAME="$(hostname)"
N_WORKERS=""  # auto-detect

while [[ $# -gt 0 ]]; do
    case $1 in
        --seed)       SEED="$2";        shift 2 ;;
        --server-name) SERVER_NAME="$2"; shift 2 ;;
        --n-workers)  N_WORKERS="$2";   shift 2 ;;
        *)            echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ---- Setup ----
PROJECT_DIR="$HOME/Coreset_Codes"
VENV_DIR="$PROJECT_DIR/venv"
NCORES=$(nproc 2>/dev/null || echo 4)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/${SERVER_NAME}_${TIMESTAMP}.log"
OUTPUT_DIR="$PROJECT_DIR/runs_out_${SERVER_NAME}"
CACHE_DIR="$PROJECT_DIR/replicate_cache"
TMUX_SESSION="coreset_${SERVER_NAME}"
DONE_MARKER="$PROJECT_DIR/DONE_${SERVER_NAME}"

# Auto-detect workers: run all 15 scenarios in parallel, limited by cores.
# Each scenario gets at least 4 threads. So max workers = cores / 4.
if [ -z "$N_WORKERS" ]; then
    N_WORKERS=$(( NCORES / 4 ))
    if [ "$N_WORKERS" -lt 1 ]; then N_WORKERS=1; fi
    if [ "$N_WORKERS" -gt 15 ]; then N_WORKERS=15; fi
fi

mkdir -p "$LOG_DIR"
rm -f "$DONE_MARKER"

echo "============================================"
echo "  Experiment Launcher: $SERVER_NAME"
echo "============================================"
echo "  Seed:       $SEED"
echo "  Workers:    $N_WORKERS"
echo "  CPU cores:  $NCORES"
echo "  Output:     $OUTPUT_DIR"
echo "  Log:        $LOG_FILE"
echo "  tmux:       $TMUX_SESSION"
echo "============================================"
echo ""

# ---- Build the command ----
CMD="source $VENV_DIR/bin/activate && \
cd $PROJECT_DIR/coreset_selection && \
python -m coreset_selection.parallel_runner \
    --scenarios R0,R1,R2,R3,R4,R5,R6,R7,R8,R9,R10,R11,R12,R13,R14 \
    --n-workers $N_WORKERS \
    --data-dir $PROJECT_DIR/data \
    --output-dir $OUTPUT_DIR \
    --cache-dir $CACHE_DIR \
    --seed $SEED \
    --device cpu \
    2>&1 | tee $LOG_FILE; \
echo \"\$(date): Experiments completed with exit code \$?\" >> $LOG_FILE; \
touch $DONE_MARKER; \
echo ''; \
echo '=========================================='; \
echo '  ALL EXPERIMENTS FINISHED'; \
echo '  DONE marker: $DONE_MARKER'; \
echo '=========================================='"

# ---- Launch in tmux ----
if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "WARNING: tmux session '$TMUX_SESSION' already exists."
    echo "To view it:   tmux attach -t $TMUX_SESSION"
    echo "To kill it:   tmux kill-session -t $TMUX_SESSION"
    echo ""
    read -p "Kill existing session and start fresh? [y/N] " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        tmux kill-session -t "$TMUX_SESSION"
    else
        echo "Aborted."
        exit 1
    fi
fi

echo "Starting tmux session '$TMUX_SESSION'..."
tmux new-session -d -s "$TMUX_SESSION" "$CMD"

echo ""
echo "Experiments are running in the background."
echo ""
echo "Useful commands:"
echo "  tmux attach -t $TMUX_SESSION     # Watch live output"
echo "  cat $LOG_FILE                     # Read log file"
echo "  ls -l $DONE_MARKER               # Check if done"
echo "  tail -f $LOG_FILE                 # Follow log in real-time"
echo ""
echo "You can safely disconnect SSH. The experiments will keep running."
