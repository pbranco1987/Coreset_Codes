#!/bin/bash
# ============================================================================
# run_parallel_tmux.sh — Launch N concurrent tmux sessions with auto-threading
#
# Each session gets 15 windows (R0-R14), each running one experiment.
# Thread count is auto-calculated: total_cores / N_sessions.
# Each session uses a different seed for independent replicates.
# Machine ID is auto-detected from hostname — no two servers share seeds.
#
# PHASE 1: Builds replicate caches for ALL seeds sequentially (VAE training).
#           This runs ONCE using all cores, before any experiments start.
# PHASE 2: Launches all tmux sessions in parallel with thread-limited experiments.
#
# Usage:
#   bash scripts/run_parallel_tmux.sh <N_SESSIONS> [OPTIONS]
#
#   Machine is auto-detected from hostname. Seed assignment:
#     labgele  (machine 0): seeds 123, 456, ...
#     lessonia (machine 1): seeds 789, 1012, ...
#     (other)  (machine 2): seeds 1345, 1678, ...
#
# Per-session k values:
#   --k1 <val>       k for single-k experiments in session 1 (default: 100)
#   --k2 <val>       k for single-k experiments in session 2 (default: 100)
#   --range1 <vals>  comma-separated k range for sweep experiments in session 1
#                    (default: 50,100,200,300,400,500)
#   --range2 <vals>  comma-separated k range for sweep experiments in session 2
#                    (default: 50,100,200,300,400,500)
#
#   Sweep experiments (r0, r1, r8, r9) use --rangeN.
#   All other experiments (r2-r7, r10-r14) use --kN.
#
# Examples:
#   bash scripts/run_parallel_tmux.sh 2                              # defaults for both
#   bash scripts/run_parallel_tmux.sh 2 --k1 100 --k2 300            # different single k
#   bash scripts/run_parallel_tmux.sh 2 --k1 100 --range1 50,100,200 --k2 300 --range2 200,300,400
#   bash scripts/run_parallel_tmux.sh 2 42 99                        # custom seeds (no k flags)
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

# ---- Parse --kN and --rangeN options, collect remaining args as seeds ----
declare -A SESSION_K        # SESSION_K[1]=100, SESSION_K[2]=300
declare -A SESSION_RANGE    # SESSION_RANGE[1]="50,100,200"
EXTRA_ARGS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --range[0-9]*)
            # Must check --range before --k to avoid prefix conflict
            _snum="${1#--range}"
            if [ -z "$_snum" ] || ! [[ "$_snum" =~ ^[0-9]+$ ]]; then
                echo "ERROR: Invalid option $1 (expected --range1, --range2, ...)"
                exit 1
            fi
            shift
            SESSION_RANGE[$_snum]="$1"
            ;;
        --k[0-9]*)
            _snum="${1#--k}"
            if [ -z "$_snum" ] || ! [[ "$_snum" =~ ^[0-9]+$ ]]; then
                echo "ERROR: Invalid option $1 (expected --k1, --k2, ...)"
                exit 1
            fi
            shift
            SESSION_K[$_snum]="$1"
            ;;
        *)
            EXTRA_ARGS+=("$1")
            ;;
    esac
    shift
done

# Default k values
DEFAULT_SINGLE_K=100
DEFAULT_RANGE="50,100,200,300,400,500"

# ---- Auto-detect machine ID from hostname ----
HOSTNAME_LOWER="$(hostname | tr '[:upper:]' '[:lower:]')"
case "$HOSTNAME_LOWER" in
    *labgele*)   MACHINE_ID=0 ;;
    *lessonia*)  MACHINE_ID=1 ;;
    *srv03*|*server3*) MACHINE_ID=2 ;;
    *srv04*|*server4*) MACHINE_ID=3 ;;
    *srv05*|*server5*) MACHINE_ID=4 ;;
    *srv06*|*server6*) MACHINE_ID=5 ;;
    *srv07*|*server7*) MACHINE_ID=6 ;;
    *srv08*|*server8*) MACHINE_ID=7 ;;
    *srv09*|*server9*) MACHINE_ID=8 ;;
    *srv10*|*server10*) MACHINE_ID=9 ;;
    *)
        # Fallback: hash the hostname to get a stable machine ID
        MACHINE_ID=$(( $(echo "$HOSTNAME_LOWER" | cksum | cut -d' ' -f1) % 10 ))
        echo "WARNING: Unknown hostname '$HOSTNAME_LOWER', using machine ID $MACHINE_ID (from hash)"
        ;;
esac

# 20 pre-defined seeds — enough for 10 machines × 2 sessions each
ALL_SEEDS=(123 456 789 1012 1345 1678 1901 2234 2567 2890
           3123 3456 3789 4012 4345 4678 4901 5234 5567 5890)

# Use custom seeds if extra args provided, otherwise slice by machine ID
SEEDS=()
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    SEEDS=("${EXTRA_ARGS[@]}")
else
    OFFSET=$(( MACHINE_ID * N_SESSIONS ))
    for i in $(seq 0 $(( N_SESSIONS - 1 ))); do
        IDX=$(( OFFSET + i ))
        if [ "$IDX" -ge "${#ALL_SEEDS[@]}" ]; then
            echo "ERROR: Machine $MACHINE_ID with $N_SESSIONS sessions exceeds available seeds (max ${#ALL_SEEDS[@]})"
            exit 1
        fi
        SEEDS+=("${ALL_SEEDS[$IDX]}")
    done
fi

TOTAL_CORES=$(nproc)
THREADS_PER_SESSION=$(( TOTAL_CORES / N_SESSIONS ))
[ "$THREADS_PER_SESSION" -lt 2 ] && THREADS_PER_SESSION=2

echo "============================================"
echo "  Hostname: $(hostname)"
echo "  Machine ID: $MACHINE_ID"
echo "  Cores: $TOTAL_CORES"
echo "  Sessions: $N_SESSIONS"
echo "  Threads/session: $THREADS_PER_SESSION"
echo "  Seeds: ${SEEDS[*]}"
for S in $(seq 1 "$N_SESSIONS"); do
    _sk="${SESSION_K[$S]:-$DEFAULT_SINGLE_K}"
    _rng="${SESSION_RANGE[$S]:-$DEFAULT_RANGE}"
    echo "  Session $S: k=$_sk  range=$_rng"
done
echo "============================================"
echo ""

# ===========================================================================
# PHASE 1: Pre-build replicate caches for each seed (uses ALL cores)
# ===========================================================================
echo "PHASE 1: Building replicate caches (VAE + PCA)..."
echo "  This uses all $TOTAL_CORES cores and must finish before experiments start."
echo ""

cd ~/Coreset_Codes
source venv/bin/activate

for SEED in "${SEEDS[@]}"; do
    echo "  Building caches for seed=$SEED ..."
    python3 -m coreset_selection prep \
        --seed "$SEED" \
        --cache-dir "replicate_cache_seed${SEED}" \
        --data-dir data \
        --n-replicates 5 \
        --device cpu
    echo "  Seed $SEED: cache ready."
done

echo ""
echo "PHASE 1 complete. All caches built."
echo ""

# ===========================================================================
# PHASE 2: Launch tmux sessions (thread-limited)
# ===========================================================================
echo "PHASE 2: Launching experiment sessions..."
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
    CACHEDIR="replicate_cache_seed${SEED}"

    # Per-session k values (fall back to defaults)
    SINGLE_K="${SESSION_K[$S]:-$DEFAULT_SINGLE_K}"
    RANGE_K="${SESSION_RANGE[$S]:-$DEFAULT_RANGE}"

    # Kill existing session if present
    tmux kill-session -t "$SESSION" 2>/dev/null || true

    tmux new-session -d -s "$SESSION"

    for r in $SWEEPS; do
        tmux new-window -t "$SESSION" -n "$r"
        tmux send-keys -t "$SESSION:$r" \
            "cd ~/Coreset_Codes && source venv/bin/activate && $TENV python3 -m coreset_selection $r -k $RANGE_K --seed $SEED --output-dir $OUTDIR --cache-dir $CACHEDIR" Enter
    done

    for r in $SINGLES; do
        tmux new-window -t "$SESSION" -n "$r"
        tmux send-keys -t "$SESSION:$r" \
            "cd ~/Coreset_Codes && source venv/bin/activate && $TENV python3 -m coreset_selection $r -k $SINGLE_K --seed $SEED --output-dir $OUTDIR --cache-dir $CACHEDIR" Enter
    done

    echo "  Session '$SESSION' launched (seed=$SEED, k=$SINGLE_K, range=$RANGE_K, output=$OUTDIR, cache=$CACHEDIR, threads=$THREADS_PER_SESSION)"
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
