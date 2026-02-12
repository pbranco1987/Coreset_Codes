#!/bin/bash
# Launch experiments in tmux with visual monitoring
#
# Usage:
#   ./tmux_experiments.sh r1 r2 r3 r4    # Run specific experiments
#   ./tmux_experiments.sh all            # Run r0-r9
#   ./tmux_experiments.sh -t 16 r1 r2    # Force 16 threads each
#
# This creates a tmux session with one pane per experiment,
# so you can see all output in real-time.

set -e

SESSION="experiments"
THREADS=""
JOBS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--threads)
            THREADS="$2"
            shift 2
            ;;
        all)
            JOBS+=(r0 r1 r2 r3 r4 r5 r6 r7 r8 r9)
            shift
            ;;
        r[0-9]|r10)
            JOBS+=("$1")
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [[ ${#JOBS[@]} -eq 0 ]]; then
    echo "Usage: $0 [-t THREADS] r1 r2 r3 ... | all"
    echo ""
    echo "Examples:"
    echo "  $0 r1 r2 r3 r4       # Run 4 experiments with monitoring"
    echo "  $0 -t 16 r1 r2       # Force 16 threads each"
    echo "  $0 all               # Run all r0-r9"
    echo ""
    echo "Controls inside tmux:"
    echo "  Ctrl+b then arrow keys  - Switch between panes"
    echo "  Ctrl+b then z           - Zoom current pane (toggle)"
    echo "  Ctrl+b then d           - Detach (experiments keep running)"
    echo "  tmux attach -t $SESSION - Reattach later"
    exit 0
fi

NJOBS=${#JOBS[@]}
NCORES=$(nproc 2>/dev/null || echo 16)

# Calculate threads if not specified
if [[ -z "$THREADS" ]]; then
    THREADS=$((NCORES / NJOBS))
    [[ $THREADS -lt 4 ]] && THREADS=4
    [[ $THREADS -gt 32 ]] && THREADS=32
fi

echo "Starting $NJOBS experiments in tmux session '$SESSION'"
echo "Threads per job: $THREADS"
echo ""

# Kill existing session if present
tmux kill-session -t "$SESSION" 2>/dev/null || true

# Create new session with first experiment
FIRST="${JOBS[0]}"
CMD="OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS python -m coreset_selection $FIRST; echo '=== $FIRST DONE (press Enter) ==='; read"

tmux new-session -d -s "$SESSION" -n "exp" "$CMD"

# Add remaining experiments in split panes
for ((i=1; i<${#JOBS[@]}; i++)); do
    JOB="${JOBS[$i]}"
    CMD="OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS python -m coreset_selection $JOB; echo '=== $JOB DONE (press Enter) ==='; read"
    
    # Alternate between horizontal and vertical splits for better layout
    if (( i % 2 == 1 )); then
        tmux split-window -h -t "$SESSION" "$CMD"
    else
        tmux split-window -v -t "$SESSION" "$CMD"
    fi
    
    # Rebalance panes
    tmux select-layout -t "$SESSION" tiled
done

# Final layout adjustment
tmux select-layout -t "$SESSION" tiled

echo "Attaching to tmux session..."
echo ""
echo "Quick controls:"
echo "  Ctrl+b z    - Zoom/unzoom current pane"
echo "  Ctrl+b ↑↓←→ - Navigate panes"  
echo "  Ctrl+b d    - Detach (keeps running)"
echo ""

# Attach to session
tmux attach -t "$SESSION"
