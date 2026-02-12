"""
NSGA-II logging, statistics tracking, and formatting utilities.

Extracted from ``nsga2_internal.py`` to reduce module size.
All names are re-exported from ``nsga2_internal`` for backward compatibility.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import numpy as np


# =============================================================================
# VERBOSE LOGGING UTILITIES
# =============================================================================

@dataclass
class NSGA2Stats:
    """
    Statistics tracker for NSGA-II optimization.

    Collects comprehensive metrics across generations for analysis
    and verbose reporting.

    Attributes
    ----------
    generation : int
        Current generation number
    n_fronts : int
        Number of Pareto fronts in current population
    front0_size : int
        Size of the first (non-dominated) front
    obj_min : np.ndarray
        Minimum value for each objective across population
    obj_max : np.ndarray
        Maximum value for each objective across population
    obj_mean : np.ndarray
        Mean value for each objective across population
    obj_std : np.ndarray
        Standard deviation for each objective across population
    front0_obj_min : np.ndarray
        Minimum objective values on Pareto front
    front0_obj_max : np.ndarray
        Maximum objective values on Pareto front
    front0_spread : np.ndarray
        Spread (max - min) of objectives on Pareto front
    crowding_mean : float
        Mean crowding distance in population
    crowding_std : float
        Std dev of crowding distance
    n_repairs : int
        Number of repairs performed this generation
    repair_rate : float
        Fraction of offspring requiring repair
    mean_repair_magnitude : float
        Average number of bits changed per repair
    population_diversity : float
        Hamming distance based diversity metric
    gen_time_sec : float
        Wall-clock time for this generation
    cumulative_time_sec : float
        Total elapsed time since start
    hypervolume : Optional[float]
        Hypervolume indicator (computed if requested)
    """
    generation: int = 0
    n_fronts: int = 0
    front0_size: int = 0

    # Objective statistics (population-wide)
    obj_min: np.ndarray = field(default_factory=lambda: np.array([]))
    obj_max: np.ndarray = field(default_factory=lambda: np.array([]))
    obj_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    obj_std: np.ndarray = field(default_factory=lambda: np.array([]))

    # Front 0 statistics
    front0_obj_min: np.ndarray = field(default_factory=lambda: np.array([]))
    front0_obj_max: np.ndarray = field(default_factory=lambda: np.array([]))
    front0_spread: np.ndarray = field(default_factory=lambda: np.array([]))

    # Crowding distance statistics
    crowding_mean: float = 0.0
    crowding_std: float = 0.0

    # Repair statistics
    n_repairs: int = 0
    repair_rate: float = 0.0
    mean_repair_magnitude: float = 0.0

    # Diversity metrics
    population_diversity: float = 0.0

    # Timing
    gen_time_sec: float = 0.0
    cumulative_time_sec: float = 0.0

    # Advanced metrics
    hypervolume: Optional[float] = None


def _compute_population_diversity(pop_X: np.ndarray) -> float:
    """
    Compute population diversity based on pairwise Hamming distances.

    Returns the mean pairwise Hamming distance normalized by genome length,
    giving a value in [0, 1] where 1 indicates maximum diversity.

    Parameters
    ----------
    pop_X : np.ndarray
        Population of binary masks, shape (pop_size, N)

    Returns
    -------
    float
        Normalized mean pairwise Hamming distance
    """
    pop_size, N = pop_X.shape
    if pop_size < 2:
        return 0.0

    # Sample pairs for efficiency (full pairwise is O(n²))
    n_samples = min(100, pop_size * (pop_size - 1) // 2)

    total_dist = 0.0
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility

    for _ in range(n_samples):
        i, j = rng.choice(pop_size, size=2, replace=False)
        hamming = np.sum(pop_X[i] != pop_X[j])
        total_dist += hamming / N

    return total_dist / n_samples


def _compute_hypervolume_2d(F: np.ndarray, ref_point: np.ndarray) -> float:
    """
    Compute 2D hypervolume indicator.

    For 2 objectives, uses the simple sweep algorithm.

    Parameters
    ----------
    F : np.ndarray
        Objective values for Pareto front, shape (n_points, 2)
    ref_point : np.ndarray
        Reference point for hypervolume computation

    Returns
    -------
    float
        Hypervolume indicator value
    """
    if F.shape[1] != 2:
        return np.nan  # Only implemented for 2D

    # Filter points dominated by reference
    valid = np.all(F < ref_point, axis=1)
    F = F[valid]

    if F.shape[0] == 0:
        return 0.0

    # Sort by first objective
    order = np.argsort(F[:, 0])
    F = F[order]

    hv = 0.0
    prev_y = ref_point[1]

    for i in range(F.shape[0]):
        hv += (ref_point[0] - F[i, 0]) * (prev_y - F[i, 1])
        prev_y = F[i, 1]

    return hv


def _format_objective_stats(
    obj_names: Sequence[str],
    obj_min: np.ndarray,
    obj_max: np.ndarray,
    obj_mean: np.ndarray,
    obj_std: np.ndarray,
) -> str:
    """Format objective statistics as a readable string."""
    lines = []
    for i, name in enumerate(obj_names):
        lines.append(
            f"    {name:12s}: min={obj_min[i]:10.6f}  max={obj_max[i]:10.6f}  "
            f"mean={obj_mean[i]:10.6f}  std={obj_std[i]:10.6f}"
        )
    return "\n".join(lines)


def _format_progress_bar(current: int, total: int, width: int = 40) -> str:
    """Create a simple ASCII progress bar."""
    pct = current / max(1, total)
    filled = int(width * pct)
    bar = "\u2588" * filled + "\u2591" * (width - filled)
    return f"[{bar}] {current:4d}/{total:4d} ({100*pct:5.1f}%)"


class NSGA2VerboseLogger:
    """
    Verbose logging handler for NSGA-II optimization.

    Manages output formatting and statistics collection based on
    verbosity level.

    Parameters
    ----------
    verbose : Union[bool, int]
        Verbosity level:
        - 0/False: Silent
        - 1/True: Basic progress every 10 generations
        - 2: Detailed per-generation statistics
        - 3: Full diagnostics with diversity and timing
    objectives : Sequence[str]
        Names of objectives being optimized
    n_gen : int
        Total number of generations
    pop_size : int
        Population size
    k : int
        Target coreset size
    use_quota : bool
        Whether geographic quota constraints are enabled

    enforce_exact_k : bool
        Whether to enforce the exact-k cardinality constraint.
    """

    def __init__(
        self,
        verbose: Union[bool, int],
        objectives: Sequence[str],
        n_gen: int,
        pop_size: int,
        k: int,
        use_quota: bool,
        enforce_exact_k: bool,
    ):
        self.verbose = int(verbose) if isinstance(verbose, bool) else verbose
        self.objectives = list(objectives)
        self.n_gen = n_gen
        self.pop_size = pop_size
        self.k = k
        self.use_quota = use_quota
        self.enforce_exact_k = enforce_exact_k

        self.start_time = time.time()
        self.gen_start_time = self.start_time
        self.history: List[NSGA2Stats] = []

        # Reference point for hypervolume (set after first evaluation)
        self.hv_ref_point: Optional[np.ndarray] = None

    def log_header(self) -> None:
        """Print optimization header with configuration summary."""
        if self.verbose < 1:
            return

        header = f"""
\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557
\u2551                         NSGA-II OPTIMIZATION                                  \u2551
\u2560\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2563
\u2551  Configuration:                                                               \u2551
\u2551    \u2022 Objectives:      {', '.join(self.objectives):54s} \u2551
\u2551    \u2022 Population size: {self.pop_size:<54d} \u2551
\u2551    \u2022 Generations:     {self.n_gen:<54d} \u2551
\u2551    \u2022 Coreset size k:  {self.k:<54d} \u2551
\u2551    \u2022 Quota mode:      {str(self.use_quota):<54s} \u2551
\u2551    \u2022 Exact-k repair:  {str(self.enforce_exact_k):<54s} \u2551
\u2560\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2563
\u2551  Starting optimization...                                                     \u2551
\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255d
"""
        print(header)
        sys.stdout.flush()

    def log_initialization(self, pop_F: np.ndarray) -> None:
        """Log initial population statistics."""
        if self.verbose < 2:
            return

        # Set hypervolume reference point (nadir + margin)
        self.hv_ref_point = np.max(pop_F, axis=0) * 1.1

        obj_min = np.min(pop_F, axis=0)
        obj_max = np.max(pop_F, axis=0)
        obj_mean = np.mean(pop_F, axis=0)
        obj_std = np.std(pop_F, axis=0)

        print("\n\u250c\u2500 INITIAL POPULATION \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510")
        print("\u2502  Objective statistics after random initialization:                            \u2502")
        print("\u2502" + _format_objective_stats(self.objectives, obj_min, obj_max, obj_mean, obj_std).replace('\n', '\n\u2502'))
        print("\u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518\n")
        sys.stdout.flush()

    def start_generation(self) -> None:
        """Mark the start of a new generation for timing."""
        self.gen_start_time = time.time()

    def log_generation(
        self,
        gen: int,
        pop_F: np.ndarray,
        pop_X: np.ndarray,
        fronts: List[np.ndarray],
        crowd: np.ndarray,
        gen_repairs_needed: List[bool],
        gen_repair_magnitudes: List[int],
    ) -> NSGA2Stats:
        """
        Log generation statistics and return stats object.

        Parameters
        ----------
        gen : int
            Current generation (0-indexed)
        pop_F : np.ndarray
            Objective values for population
        pop_X : np.ndarray
            Binary masks for population
        fronts : List[np.ndarray]
            Pareto fronts from non-dominated sorting
        crowd : np.ndarray
            Crowding distances
        gen_repairs_needed : List[bool]
            Whether repair was needed for each offspring
        gen_repair_magnitudes : List[int]
            Number of bits changed in each repair

        Returns
        -------
        NSGA2Stats
            Statistics for this generation
        """
        gen_time = time.time() - self.gen_start_time
        cumulative_time = time.time() - self.start_time

        # Compute statistics
        obj_min = np.min(pop_F, axis=0)
        obj_max = np.max(pop_F, axis=0)
        obj_mean = np.mean(pop_F, axis=0)
        obj_std = np.std(pop_F, axis=0)

        f0 = fronts[0]
        front0_F = pop_F[f0]
        front0_obj_min = np.min(front0_F, axis=0)
        front0_obj_max = np.max(front0_F, axis=0)
        front0_spread = front0_obj_max - front0_obj_min

        crowding_mean = float(np.mean(crowd))
        crowding_std = float(np.std(crowd))

        n_repairs = sum(gen_repairs_needed)
        repair_rate = n_repairs / max(1, len(gen_repairs_needed))
        mean_repair_mag = np.mean(gen_repair_magnitudes) if gen_repair_magnitudes else 0.0

        # Compute diversity (expensive, only for verbose >= 3)
        diversity = 0.0
        if self.verbose >= 3:
            diversity = _compute_population_diversity(pop_X)

        # Compute hypervolume for 2D/3D cases
        hv = None
        if self.verbose >= 3 and self.hv_ref_point is not None:
            if len(self.objectives) == 2:
                hv = _compute_hypervolume_2d(front0_F, self.hv_ref_point)

        stats = NSGA2Stats(
            generation=gen,
            n_fronts=len(fronts),
            front0_size=f0.size,
            obj_min=obj_min,
            obj_max=obj_max,
            obj_mean=obj_mean,
            obj_std=obj_std,
            front0_obj_min=front0_obj_min,
            front0_obj_max=front0_obj_max,
            front0_spread=front0_spread,
            crowding_mean=crowding_mean,
            crowding_std=crowding_std,
            n_repairs=n_repairs,
            repair_rate=repair_rate,
            mean_repair_magnitude=mean_repair_mag,
            population_diversity=diversity,
            gen_time_sec=gen_time,
            cumulative_time_sec=cumulative_time,
            hypervolume=hv,
        )
        self.history.append(stats)

        # Log based on verbosity
        self._print_generation_log(gen, stats)

        return stats

    def _print_generation_log(self, gen: int, stats: NSGA2Stats) -> None:
        """Print generation log based on verbosity level."""
        if self.verbose < 1:
            return

        # Determine if we should print this generation
        is_milestone = (gen % 10 == 0) or (gen == self.n_gen - 1) or (gen < 5)

        if self.verbose == 1 and not is_milestone:
            return

        # Build output based on verbosity
        if self.verbose == 1:
            # Basic: one-line progress
            progress = _format_progress_bar(gen + 1, self.n_gen, width=30)
            print(
                f"\r[NSGA-II] {progress} \u2502 Front\u2080: {stats.front0_size:3d} \u2502 "
                f"Time: {stats.cumulative_time_sec:6.1f}s",
                end="" if gen < self.n_gen - 1 else "\n"
            )
            sys.stdout.flush()

        elif self.verbose == 2:
            # Detailed: per-generation with objective stats
            if is_milestone or gen % 50 == 0:
                print(f"\n\u250c\u2500 Generation {gen+1:4d}/{self.n_gen} \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510")
                print(f"\u2502  Pareto fronts: {stats.n_fronts:3d}    Front\u2080 size: {stats.front0_size:3d}    Time: {stats.gen_time_sec:.2f}s (total: {stats.cumulative_time_sec:.1f}s)")
                print("\u2502")
                print("\u2502  Population objective statistics:")
                for i, name in enumerate(self.objectives):
                    print(f"\u2502    {name:10s}: [{stats.obj_min[i]:8.5f}, {stats.obj_max[i]:8.5f}]  \u03bc={stats.obj_mean[i]:8.5f}  \u03c3={stats.obj_std[i]:8.5f}")
                print("\u2502")
                print("\u2502  Pareto front spread:")
                for i, name in enumerate(self.objectives):
                    print(f"\u2502    {name:10s}: [{stats.front0_obj_min[i]:8.5f}, {stats.front0_obj_max[i]:8.5f}]  \u0394={stats.front0_spread[i]:8.5f}")
                print(f"\u2502")
                print(f"\u2502  Repairs: {stats.n_repairs:3d} ({100*stats.repair_rate:5.1f}%)  Avg magnitude: {stats.mean_repair_magnitude:.1f} bits")
                print(f"\u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518")
                sys.stdout.flush()
            else:
                # Compact line for non-milestone generations
                obj_str = "  ".join(f"{name}={stats.obj_mean[i]:.5f}" for i, name in enumerate(self.objectives))
                print(f"  Gen {gen+1:4d}: Front\u2080={stats.front0_size:3d}  {obj_str}")
                sys.stdout.flush()

        elif self.verbose >= 3:
            # Full diagnostics
            print(f"\n{'\u2550'*80}")
            print(f"  GENERATION {gen+1:4d} / {self.n_gen}")
            print(f"{'\u2550'*80}")
            print(f"\n  TIMING:")
            print(f"    \u2022 Generation time:   {stats.gen_time_sec:8.3f} s")
            print(f"    \u2022 Cumulative time:   {stats.cumulative_time_sec:8.1f} s")
            print(f"    \u2022 Est. remaining:    {(self.n_gen - gen - 1) * stats.gen_time_sec:8.1f} s")

            print(f"\n  POPULATION STRUCTURE:")
            print(f"    \u2022 Number of fronts:  {stats.n_fronts:8d}")
            print(f"    \u2022 Front\u2080 size:       {stats.front0_size:8d} ({100*stats.front0_size/self.pop_size:.1f}% of pop)")
            print(f"    \u2022 Mean crowding:     {stats.crowding_mean:8.4f} \u00b1 {stats.crowding_std:.4f}")
            print(f"    \u2022 Pop. diversity:    {stats.population_diversity:8.4f} (Hamming)")

            print(f"\n  OBJECTIVE VALUES (Population):")
            print(f"    {'Objective':<12s} {'Min':>12s} {'Max':>12s} {'Mean':>12s} {'Std':>12s}")
            print(f"    {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
            for i, name in enumerate(self.objectives):
                print(f"    {name:<12s} {stats.obj_min[i]:12.6f} {stats.obj_max[i]:12.6f} {stats.obj_mean[i]:12.6f} {stats.obj_std[i]:12.6f}")

            print(f"\n  PARETO FRONT (Front\u2080):")
            print(f"    {'Objective':<12s} {'Min':>12s} {'Max':>12s} {'Spread':>12s}")
            print(f"    {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
            for i, name in enumerate(self.objectives):
                print(f"    {name:<12s} {stats.front0_obj_min[i]:12.6f} {stats.front0_obj_max[i]:12.6f} {stats.front0_spread[i]:12.6f}")

            if stats.hypervolume is not None:
                print(f"\n    Hypervolume indicator: {stats.hypervolume:.6f}")

            print(f"\n  REPAIR STATISTICS:")
            print(f"    \u2022 Repairs needed:    {stats.n_repairs:8d} / {self.pop_size} ({100*stats.repair_rate:.1f}%)")
            print(f"    \u2022 Avg. repair size:  {stats.mean_repair_magnitude:8.1f} bits changed")

            sys.stdout.flush()

    def log_completion(self, X_pareto: np.ndarray, F_pareto: np.ndarray) -> None:
        """Log optimization completion with final summary."""
        if self.verbose < 1:
            return

        total_time = time.time() - self.start_time

        # Compute final Pareto front statistics
        obj_min = np.min(F_pareto, axis=0)
        obj_max = np.max(F_pareto, axis=0)
        obj_mean = np.mean(F_pareto, axis=0)
        spread = obj_max - obj_min

        # Find extreme points
        extremes = {}
        for i, name in enumerate(self.objectives):
            min_idx = np.argmin(F_pareto[:, i])
            extremes[f"min_{name}"] = (min_idx, F_pareto[min_idx])

        summary = f"""
\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557
\u2551                      NSGA-II OPTIMIZATION COMPLETE                           \u2551
\u2560\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2563
\u2551  Results Summary:                                                             \u2551
\u2551    \u2022 Total generations:     {self.n_gen:<49d} \u2551
\u2551    \u2022 Final Pareto set size: {X_pareto.shape[0]:<49d} \u2551
\u2551    \u2022 Total runtime:         {total_time:>8.2f} seconds{' '*36} \u2551
\u2551    \u2022 Avg time/generation:   {total_time/self.n_gen:>8.4f} seconds{' '*36} \u2551
\u2560\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2563
\u2551  Final Pareto Front Objective Ranges:                                         \u2551"""

        for i, name in enumerate(self.objectives):
            line = f"\u2551    {name:12s}: [{obj_min[i]:10.6f}, {obj_max[i]:10.6f}]  spread={spread[i]:10.6f}"
            summary += f"\n{line:<79s} \u2551"

        summary += f"""
\u2560\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2563
\u2551  Extreme Points (minimizers):                                                 \u2551"""

        for i, name in enumerate(self.objectives):
            idx, vals = extremes[f"min_{name}"]
            vals_str = ", ".join(f"{v:.5f}" for v in vals)
            line = f"\u2551    Best {name:8s}: idx={idx:3d}  F=[{vals_str}]"
            summary += f"\n{line:<79s} \u2551"

        summary += """
\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255d
"""
        print(summary)
        sys.stdout.flush()

    def get_history_summary(self) -> Dict[str, np.ndarray]:
        """
        Get optimization history as arrays for analysis.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing:
            - 'generations': Generation numbers
            - 'front0_sizes': Pareto front sizes
            - 'obj_means': Mean objective values per generation
            - 'diversity': Population diversity per generation
            - 'repair_rates': Repair rates per generation
            - 'gen_times': Time per generation
        """
        if not self.history:
            return {}

        return {
            'generations': np.array([s.generation for s in self.history]),
            'front0_sizes': np.array([s.front0_size for s in self.history]),
            'obj_means': np.array([s.obj_mean for s in self.history]),
            'obj_mins': np.array([s.obj_min for s in self.history]),
            'diversity': np.array([s.population_diversity for s in self.history]),
            'repair_rates': np.array([s.repair_rate for s in self.history]),
            'gen_times': np.array([s.gen_time_sec for s in self.history]),
            'cumulative_times': np.array([s.cumulative_time_sec for s in self.history]),
        }
