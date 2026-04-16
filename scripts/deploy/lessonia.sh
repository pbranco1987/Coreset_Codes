#!/usr/bin/env bash
# ============================================================================
# LESSONIA — Copy-paste this ENTIRE block into the Lessonia X2Go terminal.
# It does everything: clone, install, and launch experiments.
#
# Usage:  bash scripts/LESSONIA_run_this.sh [K]
#         Default K: 300
#
# Examples:
#   bash scripts/LESSONIA_run_this.sh 300   # Run all scenarios at K=300
#   bash scripts/LESSONIA_run_this.sh 200   # Run all scenarios at K=200
# ============================================================================
set -euo pipefail

K_VALUE="${1:-300}"

echo "================================================"
echo "  LESSONIA: Full Setup + Experiment Launch"
echo "  Seed: 456 | K: $K_VALUE"
echo "================================================"

PROJECT_DIR="$HOME/Coreset_Codes"
VENV_DIR="$PROJECT_DIR/venv"

# ---- Step 1: Clone repo ----
if [ -d "$PROJECT_DIR/.git" ]; then
    echo "[1/4] Repo exists, pulling latest..."
    cd "$PROJECT_DIR" && git pull && git lfs pull
else
    echo "[1/4] Cloning repository..."
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

python -c "import coreset_selection; print('  Import test: OK')"

# ---- Step 4: Launch experiments ----
echo "[4/4] Launching experiments (seed=456, K=$K_VALUE)..."

NCORES=$(nproc 2>/dev/null || echo 4)
N_WORKERS=$(( NCORES / 4 ))
[ "$N_WORKERS" -lt 1 ] && N_WORKERS=1
[ "$N_WORKERS" -gt 15 ] && N_WORKERS=15

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p "$PROJECT_DIR/logs"
LOG_FILE="$PROJECT_DIR/logs/lessonia_k${K_VALUE}_${TIMESTAMP}.log"
OUTPUT_DIR="$PROJECT_DIR/runs_out_lessonia"

# Write a launcher script that nohup will execute
LAUNCHER="$PROJECT_DIR/logs/_launcher_lessonia.sh"
cat > "$LAUNCHER" << 'INNEREOF'
#!/usr/bin/env bash
source "$HOME/Coreset_Codes/venv/bin/activate"
cd "$HOME/Coreset_Codes/coreset_selection"
python -m coreset_selection.parallel_runner \
    --scenarios R0,R1,R2,R3,R4,R5,R6,R7,R8,R9,R10,R11,R12,R13,R14 \
    --n-workers NWORKERS_PLACEHOLDER \
    --data-dir "$HOME/Coreset_Codes/data" \
    --output-dir "$HOME/Coreset_Codes/runs_out_lessonia" \
    --cache-dir "$HOME/Coreset_Codes/replicate_cache" \
    --seed 456 \
    --device cpu \
    --k KVALUE_PLACEHOLDER
EXIT_CODE=$?
touch "$HOME/Coreset_Codes/DONE_lessonia_kKVALUE_PLACEHOLDER"
echo ""
echo "========================================"
echo "  LESSONIA: ALL EXPERIMENTS FINISHED"
echo "  Exit code: $EXIT_CODE"
echo "========================================"
INNEREOF

# Substitute actual worker count and K value
sed -i "s/NWORKERS_PLACEHOLDER/$N_WORKERS/g" "$LAUNCHER"
sed -i "s/KVALUE_PLACEHOLDER/$K_VALUE/g" "$LAUNCHER"
chmod +x "$LAUNCHER"

# Try screen > tmux > nohup (in order of preference)
SESSION_NAME="coreset_lessonia_k${K_VALUE}"
DONE_MARKER="$PROJECT_DIR/DONE_lessonia_k${K_VALUE}"

if command -v screen >/dev/null 2>&1; then
    screen -dmS "$SESSION_NAME" bash -c "$LAUNCHER 2>&1 | tee $LOG_FILE"
    echo ""
    echo "================================================"
    echo "  LESSONIA: Experiments launched in screen!"
    echo "  CPU cores: $NCORES | Workers: $N_WORKERS | K: $K_VALUE"
    echo "  Log: $LOG_FILE"
    echo "================================================"
    echo ""
    echo "  screen -r $SESSION_NAME        # Watch live"
    echo "  ls -l $DONE_MARKER             # Check if done"
    echo "  tail -f $LOG_FILE              # Follow log"
elif command -v tmux >/dev/null 2>&1; then
    tmux new-session -d -s "$SESSION_NAME" "$LAUNCHER 2>&1 | tee $LOG_FILE"
    echo ""
    echo "================================================"
    echo "  LESSONIA: Experiments launched in tmux!"
    echo "  CPU cores: $NCORES | Workers: $N_WORKERS | K: $K_VALUE"
    echo "  Log: $LOG_FILE"
    echo "================================================"
    echo ""
    echo "  tmux attach -t $SESSION_NAME   # Watch live"
    echo "  ls -l $DONE_MARKER             # Check if done"
    echo "  tail -f $LOG_FILE              # Follow log"
else
    nohup bash "$LAUNCHER" > "$LOG_FILE" 2>&1 &
    BGPID=$!
    echo ""
    echo "================================================"
    echo "  LESSONIA: Experiments launched with nohup!"
    echo "  CPU cores: $NCORES | Workers: $N_WORKERS | K: $K_VALUE"
    echo "  PID: $BGPID"
    echo "  Log: $LOG_FILE"
    echo "================================================"
    echo ""
    echo "  tail -f $LOG_FILE              # Follow log"
    echo "  ls -l $DONE_MARKER             # Check if done"
    echo "  kill $BGPID                    # Stop if needed"
fi

echo ""
echo "  You can safely close this terminal."
echo "================================================"
