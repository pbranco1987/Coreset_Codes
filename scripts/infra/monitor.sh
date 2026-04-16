#!/usr/bin/env bash
# ============================================================================
# Experiment Monitor — Organized tmux session with one window per scenario
#
# Usage:  bash scripts/monitor.sh [OUTPUT_DIR] [K]
#         Default OUTPUT_DIR: ~/Coreset_Codes/runs_out_labgele
#         When K is given, monitors OUTPUT_DIR/k<K>/ and uses session monitor_k<K>
#
# Examples:
#   bash scripts/monitor.sh ~/Coreset_Codes/runs_out_labgele 300
#   bash scripts/monitor.sh ~/Coreset_Codes/runs_out_labgele      # no K subfolder
#
# Navigation:
#   Ctrl+B, n        → next scenario window
#   Ctrl+B, p        → previous scenario window
#   Ctrl+B, <number> → jump to window by number (0=dashboard, 1=R0, ...)
#   Ctrl+B, w        → list all windows (pick one)
#   Ctrl+B, d        → detach (experiments keep running)
#
# Reattach:  tmux attach -t monitor_k300   (or whatever K you used)
# ============================================================================
set -euo pipefail

OUTPUT_DIR="${1:-$HOME/Coreset_Codes/runs_out_labgele}"
K_VALUE="${2:-}"
MAIN_LOG_DIR="$HOME/Coreset_Codes/logs"

# When K is specified, scope into the k-subfolder
if [ -n "$K_VALUE" ]; then
    OUTPUT_DIR="$OUTPUT_DIR/k${K_VALUE}"
    SESSION="monitor_k${K_VALUE}"
else
    SESSION="monitor"
fi

SCENARIO_LOG_DIR="$OUTPUT_DIR/logs"

# Kill existing monitor session if any
tmux kill-session -t "$SESSION" 2>/dev/null || true

SCENARIOS=(R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14)

# ---- Window 0: Dashboard ----
tmux new-session -d -s "$SESSION" -n "dashboard" "watch -n 10 '\
echo \"=== EXPERIMENT DASHBOARD ===\"; \
echo \"\"; \
printf \"%-6s | %-12s | %s\n\" \"Config\" \"Status\" \"Last output\"; \
printf \"%-6s-+-%-12s-+-%s\n\" \"------\" \"------------\" \"-----------------------------\"; \
for s in ${SCENARIOS[*]}; do \
  LOG=\"$SCENARIO_LOG_DIR/\$s.log\"; \
  if [ -f \"\$LOG\" ] && grep -q \"completed\" \"\$LOG\" 2>/dev/null; then \
    ST=\"✓ DONE\"; \
  elif [ -f \"\$LOG\" ]; then \
    ST=\"✗ FAILED\"; \
  elif [ -d \"$OUTPUT_DIR/\$s\" ]; then \
    ST=\"⏳ RUNNING\"; \
  else \
    ST=\"⏸ PENDING\"; \
  fi; \
  if [ -f \"\$LOG\" ]; then \
    LAST=\$(grep -v \"^$\" \"\$LOG\" | tail -1 | cut -c1-45); \
  else \
    LAST=\"-\"; \
  fi; \
  printf \"%-6s | %-12s | %s\n\" \"\$s\" \"\$ST\" \"\$LAST\"; \
done; \
echo \"\"; \
echo \"Updated: \$(date +%H:%M:%S)\"; \
echo \"\"; \
echo \"Ctrl+B,n = next window | Ctrl+B,w = list all | Ctrl+B,d = detach\"'"

# ---- Windows 1-15: One per scenario ----
for s in "${SCENARIOS[@]}"; do
    tmux new-window -t "$SESSION" -n "$s" "bash -c '\
echo \"=== Scenario $s ===\"; \
echo \"Log: $SCENARIO_LOG_DIR/$s.log\"; \
echo \"Output: $OUTPUT_DIR/$s/\"; \
echo \"\"; \
echo \"Waiting for log file...\"; \
while [ ! -f \"$SCENARIO_LOG_DIR/$s.log\" ]; do sleep 3; done; \
echo \"Log found! Showing output:\"; \
echo \"\"; \
cat \"$SCENARIO_LOG_DIR/$s.log\"; \
echo \"\"; \
echo \"=== $s FINISHED ===\"; \
echo \"Press any key to exit.\"; \
read -n 1'"
done

# ---- Window 16: Main runner log (live) ----
tmux new-window -t "$SESSION" -n "main" "bash -c '\
echo \"=== Main Runner Log ===\"; \
LATEST=\$(ls -t $MAIN_LOG_DIR/labgele_*.log 2>/dev/null | head -1); \
if [ -n \"\$LATEST\" ]; then \
  tail -f \"\$LATEST\"; \
else \
  echo \"Waiting for main log...\"; \
  while ! ls $MAIN_LOG_DIR/labgele_*.log 1>/dev/null 2>&1; do sleep 3; done; \
  LATEST=\$(ls -t $MAIN_LOG_DIR/labgele_*.log | head -1); \
  tail -f \"\$LATEST\"; \
fi'"

# Start on dashboard
tmux select-window -t "$SESSION:0"

echo ""
echo "================================================"
echo "  Monitor session ready!"
echo "================================================"
echo ""
echo "  Navigation (inside tmux):"
echo "    Ctrl+B, n     → next scenario"
echo "    Ctrl+B, p     → previous scenario"
echo "    Ctrl+B, 0     → dashboard"
echo "    Ctrl+B, w     → list all windows"
echo "    Ctrl+B, d     → detach"
echo ""
echo "  Reattach later:  tmux attach -t $SESSION"
echo "================================================"
echo ""

# Auto-attach
tmux attach -t "$SESSION"
