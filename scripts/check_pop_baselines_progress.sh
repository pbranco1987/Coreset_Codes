#!/usr/bin/env bash
# ============================================================================
# Progress monitor for pop-quota + joint baseline experiments
#
# Pipeline per job (8 methods x 3 spaces = 24 method-space combos):
#
#   For each method-space combo, the driver evaluates a 6-stage pipeline:
#     [a] Selection   — pick k indices (U/KM/KH/FF/RLS/DPP/KT/KKN)
#     [b] Geo diag    — geo_kl_muni, geo_kl_pop, coverage  (silent)
#     [c] Nystrom+KRR — nystrom_error, kpca_distortion,
#                        krr_rmse per target, worst_group_rmse  (silent)
#     [d] KPI stab    — state-level KPI stability  (silent)
#     [e] Multi-model — 3 reg models x 12 targets (36 fits)
#                      + 4 cls models x 15 targets (60 fits)
#                      = 96 model fits total (runs in parallel w/ 4 workers)
#                      prints: "parallel: 12 reg + 15 cls targets, 4 workers...
#                               done (Xs)"                       <-- TRACKED
#     [f] QoS         — OLS + Ridge + ElasticNet w/ fixed effects  (silent)
#
#   Each "done (" line = 1 combo fully evaluated (all 6 stages complete).
#
# Job totals:
#   Pop-quota: 7 k x 5 reps = 35 jobs  ->  35 x 24 =  840 combos
#   Joint:     7 k x 1 rep  =  7 jobs  ->   7 x 24 =  168 combos
#   Grand total:              42 jobs  ->            1008 combos
#                                       -> 1008 x 96 = 96,768 model fits
#                                          + nystrom, KRR, geo, QoS per combo
#
# Usage:
#   bash scripts/check_pop_baselines_progress.sh [LOGDIR]
#   (default LOGDIR = logs/pop_baselines)
#
# Auto-refresh mode:
#   watch -n 60 bash scripts/check_pop_baselines_progress.sh
# ============================================================================
set -euo pipefail

LOGDIR="${1:-logs/pop_baselines}"

if [ ! -d "$LOGDIR" ]; then
    echo "[ERROR] Log directory not found: $LOGDIR"
    exit 1
fi

# ── Constants ──
METHODS_PER_SPACE=8        # U, KM, KH, FF, RLS, DPP, KT, KKN
SPACES=3                   # raw, vae, pca
COMBOS_PER_JOB=$((METHODS_PER_SPACE * SPACES))   # 24
MODEL_FITS_PER_COMBO=96    # 3 reg-models x 12 targets + 4 cls-models x 15 targets

POP_K_VALUES=(30 50 100 200 300 400 500)
POP_REPS=(0 1 2 3 4)
JOINT_REPS=(0)

N_K=${#POP_K_VALUES[@]}             # 7
N_POP_REPS=${#POP_REPS[@]}          # 5
N_JOINT_REPS=${#JOINT_REPS[@]}      # 1

TOTAL_POP_JOBS=$((N_K * N_POP_REPS))      # 35
TOTAL_JOINT_JOBS=$((N_K * N_JOINT_REPS))   # 7
TOTAL_JOBS=$((TOTAL_POP_JOBS + TOTAL_JOINT_JOBS))  # 42

TOTAL_POP_COMBOS=$((TOTAL_POP_JOBS * COMBOS_PER_JOB))      # 840
TOTAL_JOINT_COMBOS=$((TOTAL_JOINT_JOBS * COMBOS_PER_JOB))   # 168
TOTAL_COMBOS=$((TOTAL_POP_COMBOS + TOTAL_JOINT_COMBOS))     # 1008

TOTAL_MODEL_FITS=$((TOTAL_COMBOS * MODEL_FITS_PER_COMBO))   # 96768

# ── Counters ──
pop_logs=0; joint_logs=0
pop_combos_done=0; joint_combos_done=0
pop_finished=0; joint_finished=0
pop_errors=0; joint_errors=0

declare -a job_lines=()
total_eval_secs=0  # sum of eval times from "done (Xs)" lines

for f in "$LOGDIR"/*.log; do
    [ -e "$f" ] || continue
    fname=$(basename "$f" .log)

    # Count completed combos: "done (" from multi_model_evaluator
    ndone=$(grep -c 'done ([0-9]' "$f" 2>/dev/null || echo 0)

    # Sum up evaluation times from "done (123.4s)" lines for ETA
    while IFS= read -r line; do
        secs=$(echo "$line" | grep -oP 'done \(\K[0-9.]+' 2>/dev/null || true)
        if [ -n "$secs" ]; then
            # Integer arithmetic only; truncate decimal
            int_secs=${secs%%.*}
            total_eval_secs=$((total_eval_secs + int_secs))
        fi
    done < <(grep 'done ([0-9]' "$f" 2>/dev/null || true)

    # Job finished?
    finished=0
    if grep -q "DONE  regime=" "$f" 2>/dev/null; then
        finished=1
    fi

    # Errors? (exclude [warn] and KT- lines which are non-fatal)
    has_error=0
    if grep -qi "Traceback\|Error\|FAILED" "$f" 2>/dev/null; then
        fatal_line=$(grep -i "Traceback\|Error\|FAILED" "$f" 2>/dev/null \
                     | grep -v "\[warn\]" | grep -v "KT-" | head -1)
        [ -n "$fatal_line" ] && has_error=1
    fi

    # Last activity indicator
    last_line=$(tail -1 "$f" 2>/dev/null || echo "")
    activity=""
    if [[ "$last_line" == *"parallel:"* ]]; then
        activity="multi-model (96 fits)"
    elif [[ "$last_line" == *"KT-SPLIT"* ]]; then
        pct=$(echo "$last_line" | grep -oP '\(\K[0-9]+(?=%)' || echo "?")
        activity="KT-split ${pct}%"
    elif [[ "$last_line" == *"KT-REFINE init"* ]]; then
        pct=$(echo "$last_line" | grep -oP '\(\K[0-9]+(?=%)' || echo "?")
        activity="KT-refine ${pct}%"
    elif [[ "$last_line" == *"KT-SWAP"* ]]; then
        pct=$(echo "$last_line" | grep -oP '\(\K[0-9]+(?=%)' || echo "?")
        activity="KT-swap ${pct}%"
    elif [[ "$last_line" == *"KT-REFINE init done"* ]]; then
        activity="KT-swapping"
    elif [[ "$last_line" == *"done ("* ]]; then
        activity="between combos"
    elif [[ "$last_line" == *"DONE"* ]]; then
        activity="finished"
    elif [[ "$last_line" == *"Saved"* ]]; then
        activity="saving results"
    elif [[ "$last_line" == *"Loading"* ]]; then
        activity="loading cache"
    elif [[ "$last_line" == *"Building"* ]]; then
        activity="building"
    elif [[ "$last_line" == *"Running"* ]]; then
        activity="starting evals"
    fi

    # Categorize pop vs joint
    if [[ "$fname" == pop_quota_* ]]; then
        pop_logs=$((pop_logs + 1))
        pop_combos_done=$((pop_combos_done + ndone))
        pop_finished=$((pop_finished + finished))
        pop_errors=$((pop_errors + has_error))
    elif [[ "$fname" == joint_quota_* ]]; then
        joint_logs=$((joint_logs + 1))
        joint_combos_done=$((joint_combos_done + ndone))
        joint_finished=$((joint_finished + finished))
        joint_errors=$((joint_errors + has_error))
    fi

    # Per-job progress
    pct_job=0
    [ "$COMBOS_PER_JOB" -gt 0 ] && pct_job=$((ndone * 100 / COMBOS_PER_JOB))

    status="running"
    if [ "$finished" -eq 1 ]; then
        status="DONE"
    elif [ "$has_error" -eq 1 ]; then
        status="ERROR"
    elif [ "$ndone" -eq 0 ]; then
        if echo "$last_line" | grep -q "KT-" 2>/dev/null; then
            status="selecting"
        else
            status="starting"
        fi
    fi

    job_lines+=("$(printf "%-32s  %2d/%d  %3d%%  %-10s  %s" \
        "$fname" "$ndone" "$COMBOS_PER_JOB" "$pct_job" "$status" "$activity")")
done

# ── Totals ──
actual_logs=$((pop_logs + joint_logs))
total_combos_done=$((pop_combos_done + joint_combos_done))
total_finished=$((pop_finished + joint_finished))
total_errors=$((pop_errors + joint_errors))
total_model_fits_done=$((total_combos_done * MODEL_FITS_PER_COMBO))

overall_pct=0; pop_pct=0; joint_pct=0
[ "$TOTAL_COMBOS" -gt 0 ] && overall_pct=$((total_combos_done * 100 / TOTAL_COMBOS))
[ "$TOTAL_POP_COMBOS" -gt 0 ] && pop_pct=$((pop_combos_done * 100 / TOTAL_POP_COMBOS))
[ "$TOTAL_JOINT_COMBOS" -gt 0 ] && joint_pct=$((joint_combos_done * 100 / TOTAL_JOINT_COMBOS))

# ── Active processes ──
active_procs=$(ps aux 2>/dev/null | grep "[p]ython.*coreset_selection" | wc -l || echo "?")

# ── ETA estimate (two methods: wall-clock and eval-time) ──
oldest_log=$(ls -tr "$LOGDIR"/*.log 2>/dev/null | head -1)
eta_str="N/A"
elapsed_str=""
if [ -n "$oldest_log" ] && [ "$total_combos_done" -gt 0 ]; then
    start_epoch=$(stat -c %Y "$oldest_log" 2>/dev/null || stat -f %m "$oldest_log" 2>/dev/null || echo "")
    now_epoch=$(date +%s)
    if [ -n "$start_epoch" ]; then
        elapsed=$((now_epoch - start_epoch))
        elapsed_h=$((elapsed / 3600))
        elapsed_m=$(( (elapsed % 3600) / 60 ))
        elapsed_str="${elapsed_h}h${elapsed_m}m"

        if [ "$elapsed" -gt 0 ] && [ "$total_combos_done" -lt "$TOTAL_COMBOS" ]; then
            remaining=$((TOTAL_COMBOS - total_combos_done))
            secs_per_combo=$((elapsed / total_combos_done))
            eta_secs=$((remaining * secs_per_combo))
            eta_h=$((eta_secs / 3600))
            eta_m=$(( (eta_secs % 3600) / 60 ))
            eta_str="~${eta_h}h${eta_m}m remaining"
        elif [ "$total_combos_done" -ge "$TOTAL_COMBOS" ]; then
            eta_str="all complete!"
        fi
    fi
fi

# Average eval time per combo (from actual timings)
avg_eval_str="N/A"
if [ "$total_combos_done" -gt 0 ]; then
    avg_secs=$((total_eval_secs / total_combos_done))
    avg_eval_str="${avg_secs}s"
fi

# ════════════════════════════════════════════════════════════════════
# Print report
# ════════════════════════════════════════════════════════════════════
echo ""
echo "=================================================================="
echo "     Pop-Quota + Joint Baseline -- Progress Report"
echo "=================================================================="
echo ""

# ── Progress bar ──
bar_width=50
filled=$((overall_pct * bar_width / 100))
empty=$((bar_width - filled))
bar=""
for ((i=0; i<filled; i++)); do bar+="#"; done
for ((i=0; i<empty; i++)); do bar+="."; done
printf "  Overall: [%s] %3d%%\n" "$bar" "$overall_pct"
echo ""

# ── Summary table ──
printf "  %-12s  %6s / %-6s  %5s  %6s / %-5s  %s\n" \
    "Regime" "Combos" "Total" "Pct" "Jobs" "Total" "Status"
echo "  ------------ -------- ------  -----  ------- -----  ------"
printf "  %-12s  %6d / %-6d  %4d%%  %5d / %-5d" \
    "pop-quota" "$pop_combos_done" "$TOTAL_POP_COMBOS" "$pop_pct" \
    "$pop_finished" "$TOTAL_POP_JOBS"
[ "$pop_errors" -gt 0 ] && printf "  %d errors" "$pop_errors" || printf "  OK"
echo ""
printf "  %-12s  %6d / %-6d  %4d%%  %5d / %-5d" \
    "joint" "$joint_combos_done" "$TOTAL_JOINT_COMBOS" "$joint_pct" \
    "$joint_finished" "$TOTAL_JOINT_JOBS"
[ "$joint_errors" -gt 0 ] && printf "  %d errors" "$joint_errors" || printf "  OK"
echo ""
echo "  ------------ -------- ------  -----  ------- -----  ------"
printf "  %-12s  %6d / %-6d  %4d%%  %5d / %-5d\n" \
    "TOTAL" "$total_combos_done" "$TOTAL_COMBOS" "$overall_pct" \
    "$total_finished" "$TOTAL_JOBS"
echo ""

# ── Model fits summary ──
printf "  Model fits completed: %d / %d (multi-model stage only)\n" \
    "$total_model_fits_done" "$TOTAL_MODEL_FITS"
printf "  Avg multi-model time per combo: %s\n" "$avg_eval_str"
echo ""

# ── Timing ──
printf "  Log files: %d / %d  |  Active python procs: %s\n" \
    "$actual_logs" "$TOTAL_JOBS" "$active_procs"
[ -n "$elapsed_str" ] && printf "  Elapsed: %s  |  ETA: %s\n" "$elapsed_str" "$eta_str"
echo ""

# ── Per-k breakdown ──
echo "-- Per-k breakdown -------------------------------------------------"
printf "  %-6s  %-30s  %-22s\n" "k" "pop-quota (5 reps)" "joint (1 rep)"
echo "  -----  ------------------------------ ----------------------"

for K in "${POP_K_VALUES[@]}"; do
    # Pop-quota
    pk_done=0; pk_fin=0
    for REP in "${POP_REPS[@]}"; do
        logf="$LOGDIR/pop_quota_k${K}_rep${REP}.log"
        if [ -f "$logf" ]; then
            n=$(grep -c 'done ([0-9]' "$logf" 2>/dev/null || echo 0)
            pk_done=$((pk_done + n))
            grep -q "DONE  regime=" "$logf" 2>/dev/null && pk_fin=$((pk_fin + 1))
        fi
    done
    pk_total=$((N_POP_REPS * COMBOS_PER_JOB))
    pk_pct=$((pk_done * 100 / pk_total))

    # Joint
    jk_done=0; jk_fin=0
    logf="$LOGDIR/joint_quota_k${K}_rep0.log"
    if [ -f "$logf" ]; then
        jk_done=$(grep -c 'done ([0-9]' "$logf" 2>/dev/null || echo 0)
        grep -q "DONE  regime=" "$logf" 2>/dev/null && jk_fin=1
    fi
    jk_pct=$((jk_done * 100 / COMBOS_PER_JOB))

    pk_stat=""; [ "$pk_fin" -eq "$N_POP_REPS" ] && pk_stat=" ALL DONE"
    jk_stat=""; [ "$jk_fin" -eq 1 ] && jk_stat=" DONE"

    printf "  k=%-4d %3d/%-3d (%3d%%) %d/5 done%-9s %2d/%-2d (%3d%%) %d/1 done%s\n" \
        "$K" \
        "$pk_done" "$pk_total" "$pk_pct" "$pk_fin" "$pk_stat" \
        "$jk_done" "$COMBOS_PER_JOB" "$jk_pct" "$jk_fin" "$jk_stat"
done
echo ""

# ── Per-job details ──
echo "-- Per-job status --------------------------------------------------"
printf "  %-32s  %6s  %4s  %-10s  %s\n" "Job" "Combos" "Pct" "Status" "Activity"
echo "  --------------------------------  ------  ----  ----------  --------"
# Sort by combo count descending, then alphabetically
IFS=$'\n'
sorted_lines=($(printf '%s\n' "${job_lines[@]}" | sort -t'/' -k1 -rn 2>/dev/null \
    || printf '%s\n' "${job_lines[@]}"))
unset IFS

for line in "${sorted_lines[@]}"; do
    echo "  $line"
done
echo ""

# ── Errors ──
if [ "$total_errors" -gt 0 ]; then
    echo "-- ERRORS ----------------------------------------------------------"
    for f in "$LOGDIR"/*.log; do
        [ -e "$f" ] || continue
        fatal=$(grep -i "Traceback\|Error\|FAILED" "$f" 2>/dev/null \
            | grep -v "\[warn\]" | grep -v "KT-" | head -3)
        if [ -n "$fatal" ]; then
            echo "  $(basename $f):"
            echo "$fatal" | sed 's/^/    /'
            echo ""
        fi
    done
fi

echo "-- Last updated: $(date '+%Y-%m-%d %H:%M:%S') -------------------------"
echo ""
echo "  Each combo = 1 of 8 methods in 1 of 3 spaces, evaluated through:"
echo "    [a] Selection  (U/KM/KH/FF/RLS/DPP/KT/KKN)"
echo "    [b] Geo diagnostics  (geo_kl_muni, geo_kl_pop, coverage)"
echo "    [c] Nystrom + KRR + state-stability  (worst_group_rmse)"
echo "    [d] KPI stability  (state-level consistency)"
echo "    [e] Multi-model downstream  (3 reg x 12 tgt + 4 cls x 15 tgt = 96 fits)"
echo "    [f] QoS downstream  (OLS, Ridge, ElasticNet + fixed effects)"
echo "  Tracked via 'done' from stage [e]. Grand total: 96,768 model fits."
echo ""
