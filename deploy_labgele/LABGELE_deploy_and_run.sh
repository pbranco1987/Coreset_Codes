#!/usr/bin/env bash
# ============================================================================
# LABGELE — Deploy and Run Pop-Quota/Joint Baseline Experiments
#
# USAGE (from JupyterHub terminal):
#   1. Upload pop_baselines_deploy.tar.gz to ~/Coreset_Codes/deploy_labgele/
#   2. Run:
#      cd ~/Coreset_Codes
#      bash deploy_labgele/LABGELE_deploy_and_run.sh
#
# This script:
#   Step 1: Extracts updated code files from the tarball
#   Step 2: Verifies the cache exists
#   Step 3: Runs a quick smoke test (k=30, rep0, pop_quota, vae only)
#   Step 4: Launches all 42 jobs in tmux
# ============================================================================
set -euo pipefail

cd ~/Coreset_Codes
PROJECT_DIR="$(pwd)"
TARBALL="deploy_labgele/pop_baselines_deploy.tar.gz"

echo "============================================================"
echo "  LABGELE — Deploy Pop-Quota + Joint Baselines"
echo "============================================================"
echo ""

# ── Step 1: Extract code files ──
echo "[Step 1] Extracting updated code files ..."
if [ ! -f "$TARBALL" ]; then
    echo "  [ERROR] $TARBALL not found!"
    echo "  Upload it from local machine first."
    exit 1
fi

# Backup originals
for f in coreset_selection/geo/projector.py \
         coreset_selection/baselines/_vg_helpers.py \
         coreset_selection/baselines/variant_generator.py \
         coreset_selection/baselines/__init__.py; do
    if [ -f "$f" ] && [ ! -f "${f}.bak_pre_pop" ]; then
        cp "$f" "${f}.bak_pre_pop"
        echo "  Backed up: $f"
    fi
done

tar xzf "$TARBALL"
echo "  [OK] Code files extracted"
echo ""

# ── Step 2: Verify cache ──
echo "[Step 2] Verifying cache ..."
CACHE_DIR="replicate_cache"
MISSING=0
for REP in 0 1 2 3 4; do
    CACHE_FILE="$CACHE_DIR/rep$(printf '%02d' $REP)/assets.npz"
    if [ -f "$CACHE_FILE" ]; then
        echo "  [OK]   $CACHE_FILE"
    else
        echo "  [MISS] $CACHE_FILE"
        MISSING=$((MISSING + 1))
    fi
done

if [ "$MISSING" -gt 0 ]; then
    echo ""
    echo "  WARNING: $MISSING cache files missing."
    echo "  Pop-quota needs rep00-rep04. Joint needs rep00 only."
    echo "  Some jobs may fail. Continue anyway? (Ctrl+C to abort)"
    sleep 5
fi
echo ""

# ── Step 3: Quick smoke test ──
echo "[Step 3] Running smoke test (k=30, rep0, pop_quota, vae only) ..."
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1

python -m coreset_selection.scripts.run_pop_baselines \
    --k 30 --rep-id 0 --regime pop_quota \
    --spaces vae \
    --cache-dir "$CACHE_DIR" \
    --output-dir runs_out_pop_baselines_test \
    --seed 123 2>&1 | tail -20

if [ $? -ne 0 ]; then
    echo "  [ERROR] Smoke test failed! Check the error above."
    exit 1
fi
echo "  [OK] Smoke test passed"
echo ""

# ── Step 4: Launch all jobs in tmux ──
echo "[Step 4] Launching all 42 jobs in tmux ..."

SEED=123
OUTPUT_DIR="runs_out_pop_baselines"
LOGDIR="logs/pop_baselines"
SESSION="pop_baselines"

mkdir -p "$OUTPUT_DIR" "$LOGDIR"

# Kill existing session if any
tmux kill-session -t "$SESSION" 2>/dev/null || true

# Environment
export CORESET_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONIOENCODING=utf-8

# Create tmux session with monitor window
tmux new-session -d -s "$SESSION" -n "monitor"
tmux send-keys -t "$SESSION:monitor" \
    "watch -n 30 'echo \"=== Pop-Quota + Joint Baselines ===\"; ps aux | grep run_pop_baselines | grep -v grep | wc -l; echo \"active jobs\"; echo \"\"; ls -lt logs/pop_baselines/ 2>/dev/null | head -10'" C-m

# Pop-quota: one tmux window per k value (5 reps as background processes)
for K in 30 50 100 200 300 400 500; do
    WNAME="pop_k${K}"
    tmux new-window -t "$SESSION" -n "$WNAME"

    CMD="cd $PROJECT_DIR && export OMP_NUM_THREADS=1 && export MKL_NUM_THREADS=1 && "
    for REP in 0 1 2 3 4; do
        CMD+="python -m coreset_selection.scripts.run_pop_baselines "
        CMD+="--k $K --rep-id $REP --regime pop_quota "
        CMD+="--spaces raw,vae,pca "
        CMD+="--cache-dir $CACHE_DIR --output-dir $OUTPUT_DIR --seed $SEED "
        CMD+="> $LOGDIR/pop_quota_k${K}_rep${REP}.log 2>&1 & "
    done
    CMD+="echo 'k=$K: 5 pop-quota reps launched (PIDs: '\$(jobs -p)')'; wait; echo '=== k=$K: ALL POP-QUOTA DONE ==='"

    tmux send-keys -t "$SESSION:$WNAME" "$CMD" C-m
done

# Joint: single tmux window with all 7 k values
tmux new-window -t "$SESSION" -n "joint"
JCMD="cd $PROJECT_DIR && export OMP_NUM_THREADS=1 && export MKL_NUM_THREADS=1 && "
for K in 30 50 100 200 300 400 500; do
    JCMD+="python -m coreset_selection.scripts.run_pop_baselines "
    JCMD+="--k $K --rep-id 0 --regime joint_quota "
    JCMD+="--spaces raw,vae,pca "
    JCMD+="--cache-dir $CACHE_DIR --output-dir $OUTPUT_DIR --seed $SEED "
    JCMD+="> $LOGDIR/joint_quota_k${K}_rep0.log 2>&1 & "
done
JCMD+="echo '7 joint jobs launched'; wait; echo '=== ALL JOINT DONE ==='"
tmux send-keys -t "$SESSION:joint" "$JCMD" C-m

echo ""
echo "============================================================"
echo "  ALL 42 JOBS LAUNCHED IN TMUX SESSION: $SESSION"
echo "============================================================"
echo ""
echo "  tmux attach -t $SESSION              # attach"
echo "  tmux list-windows -t $SESSION        # list windows"
echo "  tail -f $LOGDIR/pop_quota_k100_rep0.log   # watch a job"
echo "  ps aux | grep run_pop_baselines | grep -v grep | wc -l  # count"
echo ""
echo "  Results will be in: $OUTPUT_DIR/"
echo ""
