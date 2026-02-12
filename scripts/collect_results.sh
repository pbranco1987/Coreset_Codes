#!/usr/bin/env bash
# ============================================================================
# collect_results.sh — Pull experiment results from both servers
#
# Usage (from your local machine):
#   bash scripts/collect_results.sh
#
# What it does:
#   1. Checks if both servers have finished (DONE markers)
#   2. Downloads runs_out directories from both servers via scp
#   3. Places them in results/labgele/ and results/lessonia/
# ============================================================================
set -euo pipefail

# ---- Server configuration ----
LABGELE_HOST="pbranco@161.24.23.23"
LABGELE_PORT=2222
LESSONIA_HOST="pbranco@161.24.29.21"
LESSONIA_PORT=2222

REMOTE_PROJECT="Coreset_Codes"
LOCAL_RESULTS="results"

mkdir -p "$LOCAL_RESULTS"

echo "============================================"
echo "  Results Collection"
echo "============================================"
echo ""

# ---- Function to check if server is done ----
check_done() {
    local host=$1
    local port=$2
    local name=$3
    echo -n "  Checking $name... "
    if ssh -p "$port" "$host" "test -f ~/$REMOTE_PROJECT/DONE_${name}" 2>/dev/null; then
        echo "DONE"
        return 0
    else
        echo "NOT DONE (or unreachable)"
        return 1
    fi
}

# ---- Check status ----
echo "Checking experiment status:"
LABGELE_DONE=0
LESSONIA_DONE=0

check_done "$LABGELE_HOST" "$LABGELE_PORT" "labgele" && LABGELE_DONE=1
check_done "$LESSONIA_HOST" "$LESSONIA_PORT" "lessonia" && LESSONIA_DONE=1

echo ""

if [ "$LABGELE_DONE" -eq 0 ] && [ "$LESSONIA_DONE" -eq 0 ]; then
    echo "Neither server has finished yet."
    read -p "Continue downloading partial results? [y/N] " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

# ---- Download from LABGELE ----
echo ""
echo "Downloading from LABGELE (161.24.23.23)..."
mkdir -p "$LOCAL_RESULTS/labgele"
scp -r -P "$LABGELE_PORT" \
    "$LABGELE_HOST:~/$REMOTE_PROJECT/runs_out_labgele/" \
    "$LOCAL_RESULTS/labgele/runs_out/" \
    2>/dev/null && echo "  LABGELE: done" || echo "  LABGELE: failed (may not have results yet)"

# Also grab logs
scp -r -P "$LABGELE_PORT" \
    "$LABGELE_HOST:~/$REMOTE_PROJECT/logs/" \
    "$LOCAL_RESULTS/labgele/logs/" \
    2>/dev/null || true

# ---- Download from Lessonia ----
echo ""
echo "Downloading from Lessonia (161.24.29.21)..."
mkdir -p "$LOCAL_RESULTS/lessonia"
scp -r -P "$LESSONIA_PORT" \
    "$LESSONIA_HOST:~/$REMOTE_PROJECT/runs_out_lessonia/" \
    "$LOCAL_RESULTS/lessonia/runs_out/" \
    2>/dev/null && echo "  Lessonia: done" || echo "  Lessonia: failed (may not have results yet)"

# Also grab logs
scp -r -P "$LESSONIA_PORT" \
    "$LESSONIA_HOST:~/$REMOTE_PROJECT/logs/" \
    "$LOCAL_RESULTS/lessonia/logs/" \
    2>/dev/null || true

# ---- Summary ----
echo ""
echo "============================================"
echo "  Collection Complete"
echo "============================================"
echo "  Results saved to: $LOCAL_RESULTS/"
echo ""
echo "  Next step: merge results with:"
echo "    python scripts/merge_results.py"
echo "============================================"
