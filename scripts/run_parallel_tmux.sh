#!/bin/bash
# ============================================================================
# run_parallel_tmux.sh — Launch N concurrent tmux sessions with auto-threading
#
# Each session gets 15 windows (R0-R14), each running one experiment.
# Thread count is auto-calculated: total_cores / N_sessions.
#
# Usage:
#   bash scripts/run_parallel_tmux.sh 1          # 1 session, all cores
#   bash scripts/run_parallel_tmux.sh 2          # 2 sessions, half cores each
#   bash scripts/run_parallel_tmux.sh 3          # 3 sessions, third each
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
TOTAL_CORES=$(nproc)
THREADS_PER_SESSION=$(( TOTAL_CORES / N_SESSIONS ))
[ "$THREADS_PER_SESSION" -lt 2 ] && THREADS_PER_SESSION=2

echo "============================================"
echo "  Cores: $TOTAL_CORES"
echo "  Sessions: $N_SESSIONS"
echo "  Threads/session: $THREADS_PER_SESSION"
echo "============================================"
echo ""

SWEEPS="r0 r1 r8 r9"
SINGLES="r2 r3 r4 r5 r6 r7 r10 r11 r12 r13 r14"

for S in $(seq 1 "$N_SESSIONS"); do
    SESSION="coreset${S}"

    # Kill existing session if present
    tmux kill-session -t "$SESSION" 2>/dev/null || true

    tmux new-session -d -s "$SESSION"

    for r in $SWEEPS; do
        tmux new-window -t "$SESSION" -n "$r"
        tmux send-keys -t "$SESSION:$r" \
            "cd ~/Coreset_Codes && source venv/bin/activate && python3 -m coreset_selection $r -j $THREADS_PER_SESSION -k 50,100,200,300,400,500" Enter
    done

    for r in $SINGLES; do
        tmux new-window -t "$SESSION" -n "$r"
        tmux send-keys -t "$SESSION:$r" \
            "cd ~/Coreset_Codes && source venv/bin/activate && python3 -m coreset_selection $r -j $THREADS_PER_SESSION -k 100" Enter
    done

    echo "  Session '$SESSION' launched (15 windows, -j $THREADS_PER_SESSION)"
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
