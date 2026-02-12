#!/bin/bash
#SBATCH --job-name=coreset_scenarios
#SBATCH --partition=standard
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-9
#SBATCH --output=logs/scenario_%A_%a.out
#SBATCH --error=logs/scenario_%A_%a.err

# ==============================================================================
# SLURM Array Job Script for Coreset Selection Experiments
# ==============================================================================
# This script runs all R0-R12 scenarios as a SLURM array job.
# Each array task runs one scenario independently.
#
# Submit with: sbatch slurm_array_job.sh
#
# Note: R6 depends on R1 outputs. If running all scenarios, either:
# 1. Submit R6 separately after R1 completes, or
# 2. Accept that R6 might fail if R1 hasn't finished
# ==============================================================================

# Ensure log directory exists
mkdir -p logs

# Load any required modules (customize for your cluster)
# module load python/3.10
# module load cuda/11.8  # if using GPU

# Activate virtual environment (customize path)
# source /path/to/venv/bin/activate

# Configuration (customize for your setup)
DATA_DIR="${DATA_DIR:-/path/to/data}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/runs_out}"
CACHE_DIR="${CACHE_DIR:-/path/to/replicate_cache}"
SEED="${SEED:-123}"
DEVICE="${DEVICE:-cpu}"
PYTHON="${PYTHON:-python}"

# Scenario array (indexed by SLURM_ARRAY_TASK_ID)
SCENARIOS=(R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12)

# Get scenario for this array task
RUN_ID=${SCENARIOS[$SLURM_ARRAY_TASK_ID]}

echo "============================================================"
echo "SLURM Job: $SLURM_JOB_ID"
echo "Array Task: $SLURM_ARRAY_TASK_ID"
echo "Scenario: $RUN_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "============================================================"

# Run the scenario
$PYTHON -m coreset_selection.run_scenario "$RUN_ID" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --cache-dir "$CACHE_DIR" \
    --seed "$SEED" \
    --device "$DEVICE"

EXIT_CODE=$?

echo "============================================================"
echo "Completed: $RUN_ID"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "============================================================"

exit $EXIT_CODE
