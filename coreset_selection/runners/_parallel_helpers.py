"""Helper functions for parallel scenario execution.

Extracted from parallel_runner.py to reduce module size.
Contains dependency resolution, command building, and script generation helpers.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional


def get_scenario_dependencies() -> Dict[str, List[str]]:
    """
    Return the dependency graph for scenarios.

    Most scenarios are independent, but R7 depends on R1 outputs.
    """
    return {
        "R0": [],
        "R1": [],
        "R2": [],
        "R3": [],
        "R4": [],
        "R5": [],
        "R6": [],
        "R7": ["R1"],  # R7 uses fixed subsets from R1
        "R8": [],
        "R9": [],
        "R10": [],
        "R11": [],
        "R12": [],
        "R13": [],
        "R14": [],
    }


def topological_sort_scenarios(scenarios: List[str]) -> List[List[str]]:
    """
    Sort scenarios into execution waves based on dependencies.

    Returns a list of lists, where each inner list contains scenarios
    that can be run in parallel (no dependencies on each other).
    """
    deps = get_scenario_dependencies()

    # Filter to only requested scenarios
    deps = {s: [d for d in deps.get(s, []) if d in scenarios] for s in scenarios}

    waves = []
    remaining = set(scenarios)

    while remaining:
        # Find scenarios with no remaining dependencies
        ready = {s for s in remaining if not any(d in remaining for d in deps.get(s, []))}

        if not ready:
            # Circular dependency or all deps outside remaining - force progress
            ready = {min(remaining)}

        waves.append(sorted(ready))
        remaining -= ready

    return waves


def build_scenario_command(
    run_id: str,
    *,
    data_dir: str = "data",
    output_dir: str = "runs_out",
    cache_dir: str = "replicate_cache",
    seed: int = 123,
    device: str = "cpu",
    k_values: Optional[List[int]] = None,
    rep_ids: Optional[List[int]] = None,
    n_replicates: Optional[int] = None,
    fail_fast: bool = False,
    parallel_experiments: Optional[int] = None,
    python_executable: str = "python",
) -> List[str]:
    """Build command-line arguments for running a single scenario."""
    cmd = [
        python_executable,
        "-m", "coreset_selection.run_scenario",
        run_id,
        "--data-dir", data_dir,
        "--output-dir", output_dir,
        "--cache-dir", cache_dir,
        "--seed", str(seed),
        "--device", device,
    ]

    if k_values:
        cmd.extend(["--k-values", ",".join(map(str, k_values))])

    if rep_ids:
        cmd.extend(["--rep-ids", ",".join(map(str, rep_ids))])
    elif n_replicates is not None:
        cmd.extend(["--n-replicates", str(n_replicates)])

    if fail_fast:
        cmd.append("--fail-fast")

    # Thread control: forward the number of concurrent experiments so each
    # subprocess can self-limit BLAS/OpenMP threads (see run_scenario.py).
    if parallel_experiments is not None:
        cmd.extend(["--parallel-experiments", str(int(parallel_experiments))])

    return cmd


def generate_shell_commands(
    scenarios: List[str],
    *,
    data_dir: str = "data",
    output_dir: str = "runs_out",
    cache_dir: str = "replicate_cache",
    seed: int = 123,
    device: str = "cpu",
    python_executable: str = "python",
) -> str:
    """
    Generate shell commands for running scenarios independently.

    This is useful for HPC environments where you want to submit each
    scenario as a separate job.
    """
    lines = [
        "#!/bin/bash",
        "# Auto-generated parallel execution commands",
        f"# Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "#",
        "# These commands can be run independently and in parallel.",
        "# For HPC clusters, submit each command as a separate job.",
        "#",
        "",
    ]

    waves = topological_sort_scenarios(scenarios)

    for wave_idx, wave in enumerate(waves):
        lines.append(f"# === Wave {wave_idx + 1}: {', '.join(wave)} ===")
        lines.append(f"# (These {len(wave)} scenarios can run in parallel)")
        lines.append("")

        for run_id in wave:
            cmd = build_scenario_command(
                run_id,
                data_dir=data_dir,
                output_dir=output_dir,
                cache_dir=cache_dir,
                seed=seed,
                device=device,
                python_executable=python_executable,
            )
            lines.append(" ".join(cmd))

        lines.append("")

    return "\n".join(lines)


def generate_slurm_script(
    scenarios: List[str],
    *,
    data_dir: str = "data",
    output_dir: str = "runs_out",
    cache_dir: str = "replicate_cache",
    seed: int = 123,
    device: str = "cpu",
    partition: str = "standard",
    time_limit: str = "24:00:00",
    memory: str = "32G",
    python_executable: str = "python",
) -> str:
    """Generate a SLURM array job script for parallel execution."""
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=coreset_scenarios",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --mem={memory}",
        f"#SBATCH --array=0-{len(scenarios)-1}",
        "#SBATCH --output=logs/scenario_%A_%a.out",
        "#SBATCH --error=logs/scenario_%A_%a.err",
        "",
        "# Auto-generated SLURM array job script",
        f"# Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "# Ensure log directory exists",
        "mkdir -p logs",
        "",
        "# Scenario array",
        f"SCENARIOS=({' '.join(scenarios)})",
        "",
        "# Get scenario for this array task",
        "RUN_ID=${SCENARIOS[$SLURM_ARRAY_TASK_ID]}",
        "",
        "echo \"Running scenario: $RUN_ID\"",
        "",
        f"{python_executable} -m coreset_selection.run_scenario $RUN_ID \\",
        f"    --data-dir {data_dir} \\",
        f"    --output-dir {output_dir} \\",
        f"    --cache-dir {cache_dir} \\",
        f"    --seed {seed} \\",
        f"    --device {device}",
        "",
        "echo \"Completed scenario: $RUN_ID with exit code: $?\"",
    ]

    return "\n".join(lines)
