#!/bin/bash
# ============================================================================
# run_parallel_tmux.sh — Launch N concurrent tmux sessions with auto-threading
#
# Each session gets 15 windows (R0-R14), each running one experiment.
# Thread count is auto-calculated: total_cores / N_sessions.
# Each session uses a different seed for independent replicates.
#
# Usage:
#   bash scripts/run_parallel_tmux.sh 1              # 1 session (seed=123)
#   bash scripts/run_parallel_tmux.sh 2              # 2 sessions (seeds 123, 456)
#   bash scripts/run_parallel_tmux.sh 3              # 3 sessions (seeds 123, 456, 789)
#   bash scripts/run_parallel_tmux.sh 2 200 300      # 2 sessions with custom seeds
#
# Watching experiments in real time:
#   tmux attach -t coreset1          # attach to session 1
#   Ctrl+B, n                        # next window (next experiment)
#   Ctrl+B, p                        # previous window
#   Ctrl+B, w                        # list all 15 windows, pick one
#   Ctrl+B, s                        # switch between sessions
#   Ctrl+B, d                        # detach (keeps running)
#
# List all sessions:
#   tmux ls
# ============================================================================
set -euo pipefail

N_SESSIONS="${1:-1}"
shift || true

# Default seeds: 123, 456, 789, 1012, 1345, ...
DEFAULT_SEEDS=(123 456 789 1012 1345 1678 1901 2234 2567 2890)

# Use custom seeds if provided, otherwise use defaults
SEEDS=()
if [ $# -gt 0 ]; then
    SEEDS=("$@")
else
    for i in $(seq 0 $(( N_SESSIONS - 1 ))); do
        SEEDS+=("${DEFAULT_SEEDS[$i]}")
    done
fi

TOTAL_CORES=$(nproc)
THREADS_PER_SESSION=$(( TOTAL_CORES / N_SESSIONS ))
[ "$THREADS_PER_SESSION" -lt 2 ] && THREADS_PER_SESSION=2

echo "============================================"
echo "  Cores: $TOTAL_CORES"
echo "  Sessions: $N_SESSIONS"
echo "  Threads/session: $THREADS_PER_SESSION"
echo "  Seeds: ${SEEDS[*]}"
echo "============================================"
echo ""

# Thread limits are set via environment variables (before python starts)
TENV="OMP_NUM_THREADS=$THREADS_PER_SESSION MKL_NUM_THREADS=$THREADS_PER_SESSION OPENBLAS_NUM_THREADS=$THREADS_PER_SESSION"

SWEEPS="r0 r1 r8 r9"
SINGLES="r2 r3 r4 r5 r6 r7 r10 r11 r12 r13 r14"

for S in $(seq 1 "$N_SESSIONS"); do
    IDX=$(( S - 1 ))
    SEED="${SEEDS[$IDX]}"
    SESSION="coreset${S}"
    OUTDIR="runs_out_seed${SEED}"

    # Kill existing session if present
    tmux kill-session -t "$SESSION" 2>/dev/null || true

    tmux new-session -d -s "$SESSION"

    for r in $SWEEPS; do
        tmux new-window -t "$SESSION" -n "$r"
        tmux send-keys -t "$SESSION:$r" \
            "cd ~/Coreset_Codes && source venv/bin/activate && $TENV python3 -m coreset_selection $r -k 50,100,200,300,400,500 --seed $SEED --output-dir $OUTDIR" Enter
    done

    for r in $SINGLES; do
        tmux new-window -t "$SESSION" -n "$r"
        tmux send-keys -t "$SESSION:$r" \
            "cd ~/Coreset_Codes && source venv/bin/activate && $TENV python3 -m coreset_selection $r -k 100 --seed $SEED --output-dir $OUTDIR" Enter
    done

    echo "  Session '$SESSION' launched (seed=$SEED, output=$OUTDIR, threads=$THREADS_PER_SESSION)"
done

echo ""
echo "============================================"
echo "  All sessions launched!"
echo ""
echo "  Attach:   tmux attach -t coreset1"
echo "  List:     tmux ls"
echo ""
echo "  Inside tmux:"
echo "    Ctrl+B, n   next window"
echo "    Ctrl+B, p   previous window"
echo "    Ctrl+B, w   list all windows"
echo "    Ctrl+B, s   switch session"
echo "    Ctrl+B, d   detach"
echo "============================================"
