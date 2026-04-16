#!/usr/bin/env bash
# ============================================================================
# collect_and_merge.sh — Pull experiment results from remote servers and merge
#
# Combines two previously separate steps into one pipeline:
#   1. SCP results from LABGELE and/or Lessonia to local results/ directory
#   2. Merge all server CSVs into a single combined output
#
# Usage (from project root):
#   bash scripts/infra/collect_and_merge.sh              # both servers
#   bash scripts/infra/collect_and_merge.sh --labgele     # LABGELE only
#   bash scripts/infra/collect_and_merge.sh --lessonia    # Lessonia only
#
# Output:
#   results/labgele/runs_out/   — raw results from LABGELE
#   results/lessonia/runs_out/  — raw results from Lessonia
#   results/combined/           — merged CSVs tagged by server
# ============================================================================
set -euo pipefail

# ── Server configuration ──
LABGELE_HOST="pbranco@161.24.23.23"
LABGELE_PORT=2222
LESSONIA_HOST="pbranco@161.24.29.21"
LESSONIA_PORT=2222
REMOTE_PROJECT="Coreset_Codes"
LOCAL_RESULTS="results"

# ── Parse arguments ──
DO_LABGELE=1
DO_LESSONIA=1

for arg in "$@"; do
    case "$arg" in
        --labgele)   DO_LESSONIA=0 ;;
        --lessonia)  DO_LABGELE=0 ;;
        --help|-h)
            echo "Usage: bash scripts/infra/collect_and_merge.sh [--labgele|--lessonia]"
            exit 0 ;;
    esac
done

mkdir -p "$LOCAL_RESULTS"

echo "============================================"
echo "  Step 1: Collect results from servers"
echo "============================================"

# ── Helper: check if server is done ──
check_done() {
    local host=$1 port=$2 name=$3
    echo -n "  Checking $name... "
    if ssh -p "$port" "$host" "test -f ~/$REMOTE_PROJECT/DONE_${name}" 2>/dev/null; then
        echo "DONE"
        return 0
    else
        echo "NOT DONE (or unreachable)"
        return 1
    fi
}

# ── Download from LABGELE ──
if [ "$DO_LABGELE" -eq 1 ]; then
    echo ""
    check_done "$LABGELE_HOST" "$LABGELE_PORT" "labgele" || true
    echo "  Downloading from LABGELE..."
    mkdir -p "$LOCAL_RESULTS/labgele"
    scp -r -P "$LABGELE_PORT" \
        "$LABGELE_HOST:~/$REMOTE_PROJECT/runs_out_labgele/" \
        "$LOCAL_RESULTS/labgele/runs_out/" \
        2>/dev/null && echo "  LABGELE: done" || echo "  LABGELE: failed"
    # Also grab logs
    scp -r -P "$LABGELE_PORT" \
        "$LABGELE_HOST:~/$REMOTE_PROJECT/logs/" \
        "$LOCAL_RESULTS/labgele/logs/" 2>/dev/null || true
fi

# ── Download from Lessonia ──
if [ "$DO_LESSONIA" -eq 1 ]; then
    echo ""
    check_done "$LESSONIA_HOST" "$LESSONIA_PORT" "lessonia" || true
    echo "  Downloading from Lessonia..."
    mkdir -p "$LOCAL_RESULTS/lessonia"
    scp -r -P "$LESSONIA_PORT" \
        "$LESSONIA_HOST:~/$REMOTE_PROJECT/runs_out_lessonia/" \
        "$LOCAL_RESULTS/lessonia/runs_out/" \
        2>/dev/null && echo "  Lessonia: done" || echo "  Lessonia: failed"
    scp -r -P "$LESSONIA_PORT" \
        "$LESSONIA_HOST:~/$REMOTE_PROJECT/logs/" \
        "$LOCAL_RESULTS/lessonia/logs/" 2>/dev/null || true
fi

echo ""
echo "============================================"
echo "  Step 2: Merge results"
echo "============================================"

# ── Merge CSVs using inline Python ──
# Finds all all_results.csv files, tags each row with its server of origin,
# and writes a combined CSV.

python3 - "$LOCAL_RESULTS" <<'PYTHON_EOF'
"""Merge experiment results from multiple servers into a combined CSV."""
import csv
import os
import sys
from pathlib import Path

results_dir = Path(sys.argv[1])
combined_dir = results_dir / "combined"
combined_dir.mkdir(parents=True, exist_ok=True)
output_path = combined_dir / "all_results_combined.csv"

all_rows = []
fieldnames_set = set()

# Discover and merge all_results.csv files
for server_dir in sorted(results_dir.iterdir()):
    if not server_dir.is_dir() or server_dir.name == "combined":
        continue
    server_name = server_dir.name  # e.g., "labgele", "lessonia"

    for csv_path in sorted(server_dir.rglob("all_results.csv")):
        rel = csv_path.relative_to(server_dir)
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["_server"] = server_name
                row["_source_csv"] = str(rel)
                all_rows.append(row)
                fieldnames_set.update(row.keys())

if not all_rows:
    print("  No all_results.csv files found. Nothing to merge.")
    sys.exit(0)

# Write combined CSV with server tagging
fieldnames = ["_server", "_source_csv"] + sorted(
    k for k in fieldnames_set if not k.startswith("_")
)
with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(all_rows)

print(f"  Merged {len(all_rows)} rows from {len(set(r['_server'] for r in all_rows))} servers")
print(f"  Output: {output_path}")
PYTHON_EOF

echo ""
echo "============================================"
echo "  Done"
echo "============================================"
