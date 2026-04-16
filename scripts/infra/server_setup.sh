#!/usr/bin/env bash
# ============================================================================
# server_setup.sh — One-time setup on a remote server
#
# Usage:
#   bash scripts/server_setup.sh
#
# What it does:
#   1. Clones the repository from GitHub (with LFS)
#   2. Creates a Python virtual environment
#   3. Installs the package and dependencies
#   4. Validates that data files are present
#   5. Reports CPU core count for thread tuning
# ============================================================================
set -euo pipefail

REPO_URL="https://github.com/pbranco1987/Coreset_Codes.git"
PROJECT_DIR="$HOME/Coreset_Codes"
VENV_DIR="$PROJECT_DIR/venv"

echo "============================================"
echo "  Server Setup: $(hostname)"
echo "============================================"
echo ""

# ---- Step 1: Clone repository ----
if [ -d "$PROJECT_DIR" ]; then
    echo "[1/5] Repository already exists at $PROJECT_DIR"
    echo "      Pulling latest changes..."
    cd "$PROJECT_DIR"
    git pull
    git lfs pull
else
    echo "[1/5] Cloning repository..."
    # Install git-lfs if not already available
    if ! command -v git-lfs &> /dev/null; then
        echo "      Installing git-lfs..."
        # Try apt first (Debian/Ubuntu), then conda
        if command -v apt-get &> /dev/null; then
            sudo apt-get install -y git-lfs 2>/dev/null || echo "      apt install failed, trying conda..."
        fi
        if ! command -v git-lfs &> /dev/null && command -v conda &> /dev/null; then
            conda install -y -c conda-forge git-lfs
        fi
    fi
    git lfs install
    git clone "$REPO_URL" "$PROJECT_DIR"
    cd "$PROJECT_DIR"
fi

echo ""

# ---- Step 2: Create virtual environment ----
echo "[2/5] Setting up Python virtual environment..."
if [ -d "$VENV_DIR" ]; then
    echo "      Virtual environment already exists."
else
    python3 -m venv "$VENV_DIR"
    echo "      Created venv at $VENV_DIR"
fi

# Activate
source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel --quiet

echo ""

# ---- Step 3: Install package ----
echo "[3/5] Installing coreset_selection package..."
cd "$PROJECT_DIR/coreset_selection"
pip install -e . --quiet
echo "      Package installed successfully."

echo ""

# ---- Step 4: Validate data files ----
echo "[4/5] Validating data files..."
DATA_DIR="$PROJECT_DIR/data"
MISSING=0

for f in smp_main.csv metadata.csv city_populations.csv; do
    if [ -f "$DATA_DIR/$f" ]; then
        SIZE=$(du -h "$DATA_DIR/$f" | cut -f1)
        echo "      OK: $f ($SIZE)"
    else
        echo "      MISSING: $f"
        MISSING=1
    fi
done

# Check LFS files are actual content (not just pointers)
for f in smp_main.csv df_indicators_flat_by_municipality.csv smp_smaller.csv; do
    if [ -f "$DATA_DIR/$f" ]; then
        FIRSTLINE=$(head -c 20 "$DATA_DIR/$f")
        if echo "$FIRSTLINE" | grep -q "version https://git-lfs"; then
            echo "      WARNING: $f is an LFS pointer (not downloaded). Running git lfs pull..."
            cd "$PROJECT_DIR" && git lfs pull
            break
        fi
    fi
done

if [ "$MISSING" -eq 1 ]; then
    echo ""
    echo "  WARNING: Some data files are missing!"
    echo "  Make sure git-lfs is installed and run: git lfs pull"
fi

echo ""

# ---- Step 5: Report system info ----
NCORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "unknown")
echo "[5/5] System info:"
echo "      Hostname:    $(hostname)"
echo "      CPU cores:   $NCORES"
echo "      Python:      $(python3 --version)"
echo "      Project dir: $PROJECT_DIR"
echo ""

# Quick import test
python3 -c "import coreset_selection; print('      Import test: OK')"

echo ""
echo "============================================"
echo "  Setup complete on $(hostname)"
echo "  CPU cores: $NCORES"
echo ""
echo "  Next step: run experiments with:"
echo "    bash scripts/run_experiments.sh --seed 123 --server-name $(hostname)"
echo "============================================"
