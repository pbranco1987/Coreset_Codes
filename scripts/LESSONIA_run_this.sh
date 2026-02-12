#!/usr/bin/env bash
# ============================================================================
# LESSONIA — Copy-paste this ENTIRE block into the Lessonia X2Go terminal.
# It does everything: clone, install, and launch experiments.
# ============================================================================
set -euo pipefail

echo "================================================"
echo "  LESSONIA: Full Setup + Experiment Launch"
echo "  Seed: 456"
echo "================================================"

PROJECT_DIR="$HOME/Coreset_Codes"
VENV_DIR="$PROJECT_DIR/venv"

# ---- Step 1: Clone repo ----
if [ -d "$PROJECT_DIR/.git" ]; then
    echo "[1/4] Repo exists, pulling latest..."
    cd "$PROJECT_DIR" && git pull && git lfs pull
else
    echo "[1/4] Cloning repository..."
    # Install git-lfs if needed
    command -v git-lfs >/dev/null 2>&1 || {
        echo "  Installing git-lfs..."
        sudo apt-get install -y git-lfs 2>/dev/null || conda install -y -c conda-forge git-lfs 2>/dev/null || true
    }
    git lfs install
    git clone https://github.com/pbranco1987/Coreset_Codes.git "$PROJECT_DIR"
    cd "$PROJECT_DIR"
fi

# ---- Step 2: Python environment ----
echo "[2/4] Setting up Python environment..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel --quiet

# ---- Step 3: Install package ----
echo "[3/4] Installing coreset_selection..."
cd "$PROJECT_DIR/coreset_selection"
pip install -e . --quiet
cd "$PROJECT_DIR"

# Quick test
python -c "import coreset_selection; print('  Import test: OK')"

# ---- Step 4: Launch experiments in tmux ----
echo "[4/4] Launching experiments (seed=456)..."

NCORES=$(nproc 2>/dev/null || echo 4)
N_WORKERS=$(( NCORES / 4 ))
[ "$N_WORKERS" -lt 1 ] && N_WORKERS=1
[ "$N_WORKERS" -gt 15 ] && N_WORKERS=15

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p "$PROJECT_DIR/logs"
LOG_FILE="$PROJECT_DIR/logs/lessonia_${TIMESTAMP}.log"
OUTPUT_DIR="$PROJECT_DIR/runs_out_lessonia"
TMUX_SESSION="coreset_lessonia"

# Kill existing session if any
tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true

CMD="source $VENV_DIR/bin/activate && \
cd $PROJECT_DIR/coreset_selection && \
python -m coreset_selection.parallel_runner \
    --scenarios R0,R1,R2,R3,R4,R5,R6,R7,R8,R9,R10,R11,R12,R13,R14 \
    --n-workers $N_WORKERS \
    --data-dir $PROJECT_DIR/data \
    --output-dir $OUTPUT_DIR \
    --cache-dir $PROJECT_DIR/replicate_cache \
    --seed 456 \
    --device cpu \
    2>&1 | tee $LOG_FILE; \
touch $PROJECT_DIR/DONE_lessonia; \
echo ''; \
echo '========================================'; \
echo '  LESSONIA: ALL EXPERIMENTS FINISHED'; \
echo '========================================';"

tmux new-session -d -s "$TMUX_SESSION" "$CMD"

echo ""
echo "================================================"
echo "  LESSONIA: Experiments launched in tmux!"
echo "  CPU cores: $NCORES | Workers: $N_WORKERS"
echo "  Log: $LOG_FILE"
echo "================================================"
echo ""
echo "  tmux attach -t $TMUX_SESSION   # Watch live"
echo "  ls -l $PROJECT_DIR/DONE_lessonia       # Check if done"
echo "  tail -f $LOG_FILE              # Follow log"
echo ""
echo "  You can safely close this terminal."
echo "================================================"
