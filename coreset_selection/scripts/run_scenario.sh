#!/bin/bash
# ==============================================================================
# Individual scenario runner scripts
# ==============================================================================
# This file contains template scripts for running each scenario independently.
# These can be used directly or adapted for job schedulers like SLURM, SGE, PBS.
#
# To use: Copy the relevant section to a separate file and execute it.
# ==============================================================================

# Default configuration (can be overridden by environment variables)
DATA_DIR="${DATA_DIR:-data}"
OUTPUT_DIR="${OUTPUT_DIR:-runs_out}"
CACHE_DIR="${CACHE_DIR:-replicate_cache}"
SEED="${SEED:-123}"
DEVICE="${DEVICE:-cpu}"
PYTHON="${PYTHON:-python}"

# ==============================================================================
# R0: Quota computation
# ==============================================================================
run_r0() {
    echo "Running R0: Quota computation..."
    $PYTHON -m coreset_selection.run_scenario R0 \
        --data-dir "$DATA_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --cache-dir "$CACHE_DIR" \
        --seed "$SEED" \
        --device "$DEVICE"
}

# ==============================================================================
# R1: NSGA-II main (tri-objective with quota, VAE space)
# ==============================================================================
run_r1() {
    echo "Running R1: NSGA-II main..."
    $PYTHON -m coreset_selection.run_scenario R1 \
        --data-dir "$DATA_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --cache-dir "$CACHE_DIR" \
        --seed "$SEED" \
        --device "$DEVICE"
}

# ==============================================================================
# R2: Geography ablation (exact-k only)
# ==============================================================================
run_r2() {
    echo "Running R2: Geography ablation..."
    $PYTHON -m coreset_selection.run_scenario R2 \
        --data-dir "$DATA_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --cache-dir "$CACHE_DIR" \
        --seed "$SEED" \
        --device "$DEVICE"
}

# ==============================================================================
# R3: Objective ablation (bi-objective: MMD, SD)
# ==============================================================================
run_r3() {
    echo "Running R3: Objective ablation (no SKL)..."
    $PYTHON -m coreset_selection.run_scenario R3 \
        --data-dir "$DATA_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --cache-dir "$CACHE_DIR" \
        --seed "$SEED" \
        --device "$DEVICE"
}

# ==============================================================================
# R4: Objective ablation (bi-objective: SKL, SD)
# ==============================================================================
run_r4() {
    echo "Running R4: Objective ablation (no MMD)..."
    $PYTHON -m coreset_selection.run_scenario R4 \
        --data-dir "$DATA_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --cache-dir "$CACHE_DIR" \
        --seed "$SEED" \
        --device "$DEVICE"
}

# ==============================================================================
# R5: Objective ablation (bi-objective: SKL, MMD)
# ==============================================================================
run_r5() {
    echo "Running R5: Objective ablation (no Sinkhorn)..."
    $PYTHON -m coreset_selection.run_scenario R5 \
        --data-dir "$DATA_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --cache-dir "$CACHE_DIR" \
        --seed "$SEED" \
        --device "$DEVICE"
}

# ==============================================================================
# R6: Baselines
# ==============================================================================
run_r6() {
    echo "Running R6: Baselines..."
    $PYTHON -m coreset_selection.run_scenario R6 \
        --data-dir "$DATA_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --cache-dir "$CACHE_DIR" \
        --seed "$SEED" \
        --device "$DEVICE"
}

# ==============================================================================
# R7: Post-hoc diagnostics
# NOTE: R7 depends on R1 outputs. Run R1 first!
# ==============================================================================
run_r7() {
    echo "Running R7: Post-hoc diagnostics..."
    echo "NOTE: R7 requires R1 to have completed first."
    $PYTHON -m coreset_selection.run_scenario R7 \
        --data-dir "$DATA_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --cache-dir "$CACHE_DIR" \
        --seed "$SEED" \
        --device "$DEVICE" \
        --source-run R1 \
        --source-space vae \
        --source-k 300
}

# ==============================================================================
# R8: Representation transfer (PCA space)
# ==============================================================================
run_r8() {
    echo "Running R8: Representation transfer (PCA)..."
    $PYTHON -m coreset_selection.run_scenario R8 \
        --data-dir "$DATA_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --cache-dir "$CACHE_DIR" \
        --seed "$SEED" \
        --device "$DEVICE"
}

# ==============================================================================
# R9: Representation transfer (raw space)
# ==============================================================================
run_r9() {
    echo "Running R9: Representation transfer (raw)..."
    $PYTHON -m coreset_selection.run_scenario R9 \
        --data-dir "$DATA_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --cache-dir "$CACHE_DIR" \
        --seed "$SEED" \
        --device "$DEVICE"
}

# ==============================================================================
# R10: Constraint ablation - no constraints
# ==============================================================================
run_r10() {
    echo "Running R10: Constraint ablation (no quota + no exact-k)..."
    $PYTHON -m coreset_selection.run_scenario R10 \
        --data-dir "$DATA_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --cache-dir "$CACHE_DIR" \
        --seed "$SEED" \
        --device "$DEVICE"
}

# ==============================================================================
# R11: Constraint ablation - quota only
# ==============================================================================
run_r11() {
    echo "Running R11: Constraint ablation (quota only)..."
    $PYTHON -m coreset_selection.run_scenario R11 \
        --data-dir "$DATA_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --cache-dir "$CACHE_DIR" \
        --seed "$SEED" \
        --device "$DEVICE"
}

# ==============================================================================
# Main: Run specified scenario or show help
# ==============================================================================
case "${1:-help}" in
    R0|r0) run_r0 ;;
    R1|r1) run_r1 ;;
    R2|r2) run_r2 ;;
    R3|r3) run_r3 ;;
    R4|r4) run_r4 ;;
    R5|r5) run_r5 ;;
    R6|r6) run_r6 ;;
    R7|r7) run_r7 ;;
    R8|r8) run_r8 ;;
    R9|r9) run_r9 ;;
    R10|r10) run_r10 ;;
    R11|r11) run_r11 ;;
    all)
        run_r0
        run_r1
        run_r2
        run_r3
        run_r4
        run_r5
        run_r6
        run_r7
        run_r8
        run_r9
        run_r10
        run_r11
        ;;
    *)
        echo "Usage: $0 <SCENARIO>"
        echo ""
        echo "Available scenarios:"
        echo "  R0  - Quota computation"
        echo "  R1  - NSGA-II main (3-objective with quota, VAE space)"
        echo "  R2  - Geography ablation (exact-k only)"
        echo "  R3  - Objective ablation (no SKL)"
        echo "  R4  - Objective ablation (no MMD)"
        echo "  R5  - Objective ablation (no Sinkhorn)"
        echo "  R6  - Baselines"
        echo "  R7  - Post-hoc diagnostics (requires R1)"
        echo "  R8  - Representation transfer (PCA)"
        echo "  R9  - Representation transfer (raw)"
        echo "  R10 - Constraint ablation: no constraints"
        echo "  R11 - Constraint ablation: quota only"
        echo "  all - Run all scenarios sequentially"
        echo ""
        echo "Environment variables:"
        echo "  DATA_DIR    - Data directory (default: data)"
        echo "  OUTPUT_DIR  - Output directory (default: runs_out)"
        echo "  CACHE_DIR   - Cache directory (default: replicate_cache)"
        echo "  SEED        - Random seed (default: 123)"
        echo "  DEVICE      - Compute device: cpu or cuda (default: cpu)"
        echo "  PYTHON      - Python executable (default: python)"
        ;;
esac
