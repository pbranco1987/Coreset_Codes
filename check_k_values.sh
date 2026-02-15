#!/bin/bash
# Check solver.k from config.json for all completed experiments (both seeds)
# This reads the ORIGINAL k values used before --resume was applied.

echo "================================================================"
echo "  K VALUES FROM COMPLETED EXPERIMENTS (pre-resume)"
echo "================================================================"

for SEED_DIR in runs_out_seed123 runs_out_seed456; do
    echo ""
    echo "============================================================"
    echo "  $SEED_DIR"
    echo "============================================================"

    if [ ! -d "$SEED_DIR" ]; then
        echo "  (directory not found)"
        continue
    fi

    # Find all config.json files under repNN directories
    for config in $(find "$SEED_DIR" -path "*/rep*/config.json" 2>/dev/null | sort); do
        # Extract run name and rep from path
        # e.g. runs_out_seed123/R2/rep00/config.json -> R2/rep00
        rel="${config#$SEED_DIR/}"
        run_rep="${rel%/config.json}"

        # Check if this rep actually completed (has wall_clock.json or all_results.csv)
        rep_dir="$(dirname "$config")"
        results_dir="$rep_dir/results"
        has_wc="N"; has_ar="N"
        [ -f "$results_dir/wall_clock.json" ] && has_wc="Y"
        [ -f "$results_dir/all_results.csv" ] && has_ar="Y"
        [ -f "$results_dir/quota_path.json" ] && has_ar="Y"  # R0 uses quota_path.json

        if [ "$has_wc" = "Y" ] || [ "$has_ar" = "Y" ]; then
            status="COMPLETE"
        else
            status="PARTIAL "
        fi

        # Extract solver.k from config.json
        k=$(python3 -c "import json; print(json.load(open('$config'))['solver']['k'])" 2>/dev/null)

        echo "  [$status] $run_rep: k=$k"
    done
done

echo ""
echo "================================================================"
echo "  Done."
echo "================================================================"
