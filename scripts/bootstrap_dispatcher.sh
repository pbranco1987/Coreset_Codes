#!/bin/bash
# Bootstrap dispatcher: manages tmux windows with controlled parallelism
# Cleans up finished windows and backfills from queue.
#
# Safety guarantees:
# - Each job's python process exits 0 (verified) or 1 (verification failed).
# - cleanup_finished() checks output files: jobs missing CSV+metadata are re-queued.
# - A job is only considered "done" when both the final CSV AND metadata sidecar
#   exist — not just the CSV.
# - Failed jobs are re-queued up to MAX_RETRIES times before being permanently
#   logged to FAILED_LOG.
#
# Usage: bash scripts/bootstrap_dispatcher.sh [MAX_PARALLEL] [JOBS_FILE]
#   MAX_PARALLEL  — max tmux windows at a time (default 64)
#   JOBS_FILE     — optional pre-built job list (one "run_id rep_id" per line).
#                   If omitted, jobs are discovered automatically.

MAX_PARALLEL=${1:-64}
JOBS_FILE_ARG="${2:-}"
MAX_RETRIES=3
SESSION=$(tmux display-message -p "#{session_name}")

echo "=== Bootstrap Dispatcher ==="
echo "Session: $SESSION"
echo "Max parallel: $MAX_PARALLEL"
echo "Max retries per job: $MAX_RETRIES"
echo ""

cd ~/Coreset_Codes
source ~/Coreset_Codes/venv/bin/activate

RESULTS_DIR="$HOME/Coreset_Codes/bootstrap_results"
FAILED_LOG="$RESULTS_DIR/_failed_jobs.log"
EXPERIMENTS_DIR="$HOME/Coreset_Codes/experiments_v2"
mkdir -p "$RESULTS_DIR"

# ── Build job list ──
if [ -n "$JOBS_FILE_ARG" ] && [ -f "$JOBS_FILE_ARG" ]; then
    echo "Using pre-built jobs file: $JOBS_FILE_ARG"
    JOBS_SOURCE="$JOBS_FILE_ARG"
else
    echo "Auto-discovering jobs from experiments_v2/ ..."
    JOBS_SOURCE=$(mktemp)
    python3 -c "
import os, json, sys

exp_dir = os.path.expanduser('~/Coreset_Codes/experiments_v2')
results_dir = os.path.expanduser('~/Coreset_Codes/bootstrap_results')
count = 0

for run_id in sorted(os.listdir(exp_dir)):
    run_path = os.path.join(exp_dir, run_id)
    if not os.path.isdir(run_path):
        continue
    if run_id.startswith('B_') or run_id in ('logs', '_corrupted_backup', 'bootstrap_baselines'):
        continue
    for rep in range(5):
        rep_dir = os.path.join(run_path, 'rep%02d' % rep)
        config_path = os.path.join(rep_dir, 'config.json')
        if not os.path.isfile(config_path):
            continue
        with open(config_path) as f:
            cfg = json.load(f)
        k = cfg['solver']['k']
        if k >= 400:
            continue
        space = cfg['space']
        pareto_path = os.path.join(rep_dir, 'results', '%s_pareto.npz' % space)
        if not os.path.isfile(pareto_path):
            continue
        # Check BOTH final CSV AND metadata sidecar exist
        final_csv = os.path.join(results_dir, 'bootstrap_raw_%s_rep%02d.csv' % (run_id, rep))
        meta_json = os.path.join(results_dir, 'bootstrap_meta_%s_rep%02d.json' % (run_id, rep))
        if os.path.isfile(final_csv) and os.path.isfile(meta_json):
            continue
        print('%s %d' % (run_id, rep))
        count += 1

print('# Total: %d' % count, file=sys.stderr)
" > "$JOBS_SOURCE" 2>&1
fi

declare -a ALL_JOBS
while IFS= read -r line; do
    [[ "$line" == \#* ]] && continue
    [[ -z "$line" ]] && continue
    ALL_JOBS+=("$line")
done < "$JOBS_SOURCE"

# Clean up temp file only if we created it
[ -z "$JOBS_FILE_ARG" ] && rm -f "$JOBS_SOURCE"

TOTAL=${#ALL_JOBS[@]}
echo "Total incomplete jobs: $TOTAL"

if [ "$TOTAL" -eq 0 ]; then
    echo "All jobs complete!"
    exit 0
fi

# ── Track active windows by PID ──
declare -A ACTIVE_PIDS     # WNAME -> python PID
declare -A ACTIVE_JOBS     # WNAME -> "RUN_ID REP_ID"
declare -A RETRY_COUNTS    # "RUN_ID REP_ID" -> retry count
declare -a REQUEUE         # Jobs to re-try

# On restart: detect already-running J_ windows from a previous dispatcher
for wname in $(tmux list-windows -t "$SESSION" -F "#{window_name}" 2>/dev/null | grep "^J_"); do
    pane_pid=$(tmux list-panes -t "$SESSION:$wname" -F "#{pane_pid}" 2>/dev/null | head -1)
    if [ -n "$pane_pid" ]; then
        child_pid=$(ps --ppid "$pane_pid" -o pid= --no-headers 2>/dev/null | head -1 | tr -d ' ')
        if [ -n "$child_pid" ]; then
            ACTIVE_PIDS[$wname]=$child_pid
        else
            # Finished window from previous run — kill it
            tmux kill-window -t "$SESSION:$wname" 2>/dev/null
        fi
    fi
done

EXISTING=${#ACTIVE_PIDS[@]}
if [ "$EXISTING" -gt 0 ]; then
    echo "Adopted $EXISTING running windows from previous dispatcher"
fi

# ── Functions ──

is_job_complete() {
    # A job is only complete when BOTH final CSV and metadata sidecar exist
    # AND the metadata records "complete": true (not a partial snapshot).
    local RUN_ID=$1
    local REP_ID=$2
    local REP_FMT=$(printf "%02d" "$REP_ID")
    local CSV="$RESULTS_DIR/bootstrap_raw_${RUN_ID}_rep${REP_FMT}.csv"
    local META="$RESULTS_DIR/bootstrap_meta_${RUN_ID}_rep${REP_FMT}.json"
    [ -f "$CSV" ] && [ -f "$META" ] && \
        python3 -c "import json,sys; m=json.load(open('${META}')); sys.exit(0 if m.get('complete',False) else 1)" 2>/dev/null
}

cleanup_finished() {
    # Check each tracked window; remove finished ones.
    # For finished jobs: verify output files exist. If not, re-queue.
    local cleaned=0
    local requeued=0
    for wname in "${!ACTIVE_PIDS[@]}"; do
        local pid=${ACTIVE_PIDS[$wname]}
        if ! kill -0 "$pid" 2>/dev/null; then
            # Process gone — job finished. Check output files.
            local job_key="${ACTIVE_JOBS[$wname]}"
            local run_id=$(echo "$job_key" | awk '{print $1}')
            local rep_id=$(echo "$job_key" | awk '{print $2}')

            tmux kill-window -t "$SESSION:$wname" 2>/dev/null
            unset "ACTIVE_PIDS[$wname]"
            unset "ACTIVE_JOBS[$wname]"
            cleaned=$((cleaned+1))

            if [ -n "$run_id" ] && [ -n "$rep_id" ]; then
                if is_job_complete "$run_id" "$rep_id"; then
                    local rep_fmt=$(printf "%02d" "$rep_id")
                    echo "[VERIFIED] $run_id rep${rep_fmt} — output CSV + metadata OK"
                else
                    # Output missing or incomplete — re-queue
                    local retries=${RETRY_COUNTS["$job_key"]:-0}
                    retries=$((retries+1))
                    RETRY_COUNTS["$job_key"]=$retries

                    local rep_fmt=$(printf "%02d" "$rep_id")
                    if [ "$retries" -le "$MAX_RETRIES" ]; then
                        echo "[REQUEUE] $run_id rep${rep_fmt} — output verification FAILED (attempt $retries/$MAX_RETRIES)"
                        REQUEUE+=("$job_key")
                        requeued=$((requeued+1))
                    else
                        echo "[FAILED] $run_id rep${rep_fmt} — exceeded $MAX_RETRIES retries, logging to $FAILED_LOG"
                        echo "$(date '+%Y-%m-%d %H:%M:%S') FAILED $run_id rep${rep_fmt} after $MAX_RETRIES retries" >> "$FAILED_LOG"
                    fi
                fi
            fi
        fi
    done
    if [ "$cleaned" -gt 0 ]; then
        echo "[CLEANUP] Removed $cleaned finished windows ($requeued re-queued)"
    fi
}

get_n_bootstrap() {
    # B=50 draws for all k values (k=30,50,100,200,300).
    # k>=400 are excluded at job-list build time.
    echo 50
}

launch_job() {
    local RUN_ID=$1
    local REP_ID=$2
    local WNAME="J_${RUN_ID}_r${REP_ID}"
    WNAME="${WNAME:0:30}"

    # Skip if already active
    if [ -n "${ACTIVE_PIDS[$WNAME]+x}" ]; then
        return 1
    fi

    local REP_FMT=$(printf "%02d" "$REP_ID")
    local N_BOOT
    N_BOOT=$(get_n_bootstrap "$RUN_ID" "$REP_ID")
    local CMD="cd ~/Coreset_Codes && source ~/Coreset_Codes/venv/bin/activate && export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 && python3 bootstrap_reeval.py --run-id ${RUN_ID} --rep-id ${REP_ID} --n-bootstrap ${N_BOOT} --n-reg 5 --n-cls 5 --output-dir bootstrap_results/ 2>&1 | tee bootstrap_results/log_${RUN_ID}_rep${REP_FMT}.txt"

    tmux new-window -t "$SESSION" -n "$WNAME"
    tmux send-keys -t "$SESSION:$WNAME" "$CMD" Enter

    # Wait briefly for the python process to start, then capture its PID
    sleep 1
    local pane_pid=$(tmux list-panes -t "$SESSION:$WNAME" -F "#{pane_pid}" 2>/dev/null | head -1)
    if [ -n "$pane_pid" ]; then
        # Find the python child
        local child_pid=$(ps --ppid "$pane_pid" -o pid= --no-headers 2>/dev/null | head -1 | tr -d ' ')
        if [ -n "$child_pid" ]; then
            ACTIVE_PIDS[$WNAME]=$child_pid
        else
            # Process might not have started yet; store pane_pid as fallback
            ACTIVE_PIDS[$WNAME]=$pane_pid
        fi
    fi
    ACTIVE_JOBS[$WNAME]="$RUN_ID $REP_ID"
    return 0
}

echo "Starting dispatcher loop at $(date)"
echo "================================================================"

JOB_IDX=0
LAUNCHED=0

while true; do
    # Clean up finished windows first (checks output, re-queues failures)
    cleanup_finished

    RUNNING=${#ACTIVE_PIDS[@]}

    # Pull re-queued jobs into the launch candidates
    # (processed before advancing JOB_IDX so they get priority)
    while [ "$RUNNING" -lt "$MAX_PARALLEL" ] && [ "${#REQUEUE[@]}" -gt 0 ]; do
        JOB="${REQUEUE[0]}"
        REQUEUE=("${REQUEUE[@]:1}")  # pop front
        RUN_ID=$(echo "$JOB" | awk '{print $1}')
        REP_ID=$(echo "$JOB" | awk '{print $2}')
        REP_FMT=$(printf "%02d" "$REP_ID")

        if is_job_complete "$RUN_ID" "$REP_ID"; then
            echo "[SKIP-REQUEUE] $RUN_ID rep${REP_FMT} — already complete"
            continue
        fi

        if launch_job "$RUN_ID" "$REP_ID"; then
            LAUNCHED=$((LAUNCHED+1))
            RUNNING=$((RUNNING+1))
            echo "[RELAUNCH] $RUN_ID rep${REP_FMT} (running=$RUNNING)"
        fi
    done

    # Launch new jobs if we have capacity
    while [ "$RUNNING" -lt "$MAX_PARALLEL" ] && [ "$JOB_IDX" -lt "$TOTAL" ]; do
        JOB="${ALL_JOBS[$JOB_IDX]}"
        RUN_ID=$(echo "$JOB" | awk '{print $1}')
        REP_ID=$(echo "$JOB" | awk '{print $2}')
        REP_FMT=$(printf "%02d" "$REP_ID")
        JOB_IDX=$((JOB_IDX+1))

        if is_job_complete "$RUN_ID" "$REP_ID"; then
            echo "[SKIP] $RUN_ID rep${REP_FMT} — already complete"
            continue
        fi

        if launch_job "$RUN_ID" "$REP_ID"; then
            LAUNCHED=$((LAUNCHED+1))
            RUNNING=$((RUNNING+1))
            echo "[LAUNCH] $RUN_ID rep${REP_FMT} (running=$RUNNING, launched=$LAUNCHED/$TOTAL)"
        fi
    done

    DONE_NOW=$(ls "$RESULTS_DIR"/bootstrap_raw_*.csv 2>/dev/null | wc -l)
    META_NOW=$(ls "$RESULTS_DIR"/bootstrap_meta_*.json 2>/dev/null | wc -l)
    CKPT_NOW=$(ls "$RESULTS_DIR"/.ckpt_*.json 2>/dev/null | wc -l)

    if [ "$JOB_IDX" -ge "$TOTAL" ] && [ "$RUNNING" -eq 0 ] && [ "${#REQUEUE[@]}" -eq 0 ]; then
        echo ""
        echo "================================================================"
        echo "=== ALL JOBS COMPLETE ==="
        echo "Final CSVs: $DONE_NOW  Metadata: $META_NOW"
        if [ -f "$FAILED_LOG" ]; then
            NFAILED=$(wc -l < "$FAILED_LOG")
            echo "Permanently failed: $NFAILED (see $FAILED_LOG)"
        fi
        echo "Finished at: $(date)"
        break
    fi

    echo "[STATUS $(date '+%H:%M:%S')] Running=$RUNNING  Queued=$((TOTAL-JOB_IDX))  Requeue=${#REQUEUE[@]}  Done=$DONE_NOW  Meta=$META_NOW  Ckpt=$CKPT_NOW"
    sleep 30
done
