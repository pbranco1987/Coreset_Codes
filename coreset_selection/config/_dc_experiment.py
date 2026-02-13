"""
Experiment-level configuration dataclasses.

Contains configuration structures for experiment orchestration:
- Effort sweep grid
- NSGA-II solver parameters
- Evaluation metrics
- Baseline methods
- Ablation studies
- Parameter sweeps
- Manuscript figure generation
- Top-level experiment configuration
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from ._dc_components import (
    FilesConfig,
    PreprocessingConfig,
    VAEConfig,
    GeoConfig,
    SinkhornConfig,
    MMDConfig,
    PCAConfig,
)


@dataclass
class EffortSweepGrid:
    """Structured parameter grid for the R12 effort sweep (Sec VII.G, VIII.L).

    Each ``(pop_size, n_gen)`` pair defines one effort level.  The default
    grid uses a Cartesian product matching the manuscript specification.

    Attributes
    ----------
    pop_sizes : Tuple[int, ...]
        Population sizes to test.
    n_gens : Tuple[int, ...]
        Generation counts to test (crossed with pop_sizes when
        ``paired=False``; zipped element-wise when ``paired=True``).
    paired : bool
        If True, zip pop_sizes and n_gens pair-wise.  If False (default),
        iterate over the full Cartesian product.
    """
    pop_sizes: Tuple[int, ...] = (20, 50, 100, 150, 200, 300, 400)
    n_gens: Tuple[int, ...] = (100, 300, 500, 700, 1000, 1500, 2000)
    paired: bool = True

    def grid(self) -> List[Tuple[int, int]]:
        """Return the list of ``(P, T)`` effort-level pairs."""
        if self.paired:
            return list(zip(self.pop_sizes, self.n_gens))
        # Cartesian product
        return [(p, t) for p in self.pop_sizes for t in self.n_gens]


@dataclass
class SolverConfig:
    """
    Multi-objective solver configuration.

    Uses NSGA-II for all experiment configurations (2-3 objectives).

    Attributes
    ----------
    k : int
        Target coreset size
    pop_size : int
        Population size
    n_gen : int
        Number of generations
    crossover_prob : float
        Crossover probability
    mutation_prob : float
        Mutation probability
    objectives : Tuple[str, ...]
        Objective function names ("skl", "mmd", "sinkhorn")
    algorithm : str
        Optimization algorithm (always "nsga2")
    enabled : bool
        Whether to run optimization
    effort_grid : EffortSweepGrid
        Structured parameter grid for R12 effort sweep.  Overrides the
        legacy ``CORESET_R12_*`` environment variables when present.
    verbose : int
        Verbosity level for optimization output:
        - 0: Silent mode (no output)
        - 1: Basic progress (every 10 generations, with progress bar)
        - 2: Detailed per-generation statistics (objective min/max/mean/std,
             Pareto front spread, repair statistics)
        - 3: Full diagnostics (includes population diversity metrics,
             hypervolume indicator, timing estimates)

        Can also accept bool for backward compatibility:
        - False -> 0 (silent)
        - True -> 1 (basic progress)

    enforce_exact_k : bool
        If True, enforce exact cardinality (sum(mask)=k) via repair.
        If False, do not repair to a fixed cardinality (subset size may drift).
        This is mainly intended for constraint-handling ablations.
    """
    k: int = None  # Must be provided by the user via --k
    pop_size: int = 200
    n_gen: int = 1000
    crossover_prob: float = 0.9
    mutation_prob: float = 0.2
    # Default objective pair for the primary configuration
    objectives: Tuple[str, ...] = ("mmd", "sinkhorn")
    algorithm: str = "nsga2"
    enabled: bool = True
    effort_grid: EffortSweepGrid = field(default_factory=EffortSweepGrid)
    verbose: int = 0
    enforce_exact_k: bool = True

    def get_algorithm(self) -> str:
        """
        Get the optimization algorithm.

        Returns
        -------
        str
            Always "nsga2" (NSGA-III removed since all configs have <=3 objectives).
        """
        return "nsga2"


@dataclass
class EvalConfig:
    """
    Evaluation configuration.

    Per manuscript Section 5.8 and Section 5.9:
    - The raw-space evaluation index set E has fixed size |E|=2000 sampled
      stratified by state.
    - The evaluation train/test split (E_train, E_test) uses an 80/20 split
      within each state.

    Attributes
    ----------
    enabled : bool
        Whether to run evaluation
    eval_size : int
        Size of raw-space evaluation index set |E|
    eval_train_frac : float
        Fraction of E used for training in KRR evaluation (default 0.8)
    nystrom_enabled : bool
        Compute Nystrom approximation error
    kpca_enabled : bool
        Compute kernel PCA distortion
    krr_enabled : bool
        Compute kernel ridge regression error
    n_kpca_components : int
        Number of kPCA components to compare
    multi_model_enabled : bool
        Run extended downstream evaluation (KNN, RF, LR, GBT) on
        regression and classification targets using Nystrom features
    qos_enabled : bool
        Run QoS (Quality of Service / IQS) downstream evaluation using
        OLS, Ridge, Elastic Net, PLS, and Constrained OLS models.
        Trains each model on the coreset and evaluates on the test set.
    qos_models : list of str
        Which QoS model families to run.  Any subset of
        ``["ols", "ridge", "elastic_net", "pls", "constrained",
        "heuristic"]``.  Default: all.
    qos_run_fixed_effects : bool
        Whether to also run fixed-effects (entity-demeaned) variants
        of the QoS models.
    """
    enabled: bool = True
    eval_size: int = 2000
    eval_train_frac: float = 0.8
    nystrom_enabled: bool = True
    kpca_enabled: bool = True
    krr_enabled: bool = True
    n_kpca_components: int = 20
    multi_model_enabled: bool = True
    qos_enabled: bool = True
    qos_models: List[str] = field(default_factory=lambda: [
        "ols", "ridge", "elastic_net", "pls", "constrained", "heuristic",
    ])
    qos_run_fixed_effects: bool = True


@dataclass
class BaselinesConfig:
    """
    Baseline methods configuration.

    Attributes
    ----------
    enabled : bool
        Whether to run baseline methods
    methods : List[str]
        List of baseline method names to run
    """
    enabled: bool = True
    methods: List[str] = field(default_factory=lambda: [
        "uniform",
        "kmeans",
        "herding",
        "farthest_first",
        "rls",
        "dpp",
        "kernel_thinning",
    ])


@dataclass
class AblationsConfig:
    """
    Ablation study configuration.

    Attributes
    ----------
    geo_ablation : bool
        Run geographic constraint ablation
    objective_ablation : bool
        Run objective function ablation
    representation_ablation : bool
        Run representation space ablation
    """
    geo_ablation: bool = False
    objective_ablation: bool = False
    representation_ablation: bool = False


@dataclass
class SweepConfig:
    """
    Parameter sweep configuration.

    Per manuscript, the cardinality grid K = {50, 100, 200, 300, 400, 500}.

    Attributes
    ----------
    k_values : List[int]
        Coreset sizes to sweep over (matches manuscript kGrid)
    n_replicates : int
        Number of replicates per configuration (5 per manuscript)
    """
    k_values: List[int] = field(default_factory=lambda: [50, 100, 200, 300, 400, 500])
    n_replicates: int = 5


@dataclass
class ManuscriptFiguresConfig:
    """
    Manuscript figure generation configuration.

    Attributes
    ----------
    generate_pareto_plots : bool
        Generate Pareto front visualizations
    generate_cardinality_curves : bool
        Generate metric vs. k curves
    generate_geo_plots : bool
        Generate geographic distribution plots
    figure_format : List[str]
        Output formats for figures
    dpi : int
        Resolution for raster formats
    """
    generate_pareto_plots: bool = True
    generate_cardinality_curves: bool = True
    generate_geo_plots: bool = True
    figure_format: List[str] = field(default_factory=lambda: ["png", "pdf"])
    dpi: int = 300


@dataclass
class ExperimentConfig:
    """
    Complete experiment configuration.

    Aggregates all sub-configurations for a full experiment run.

    Attributes
    ----------
    run_id : str
        Unique run identifier (e.g., "R1", "R2_ablation")
    rep_id : int
        Replicate identifier
    seed : int
        Base random seed
    device : str
        Compute device ('cpu' or 'cuda')
    files : FilesConfig
        File path configuration
    preprocessing : PreprocessingConfig
        Feature typing / preprocessing configuration (Phase 1)
    vae : VAEConfig
        VAE model configuration
    geo : GeoConfig
        Geographic constraint configuration
    sinkhorn : SinkhornConfig
        Sinkhorn divergence configuration
    mmd : MMDConfig
        MMD configuration
    solver : SolverConfig
        NSGA-II solver configuration
    eval : EvalConfig
        Evaluation configuration
    baselines : BaselinesConfig
        Baseline methods configuration
    ablations : AblationsConfig
        Ablation study configuration
    sweep : SweepConfig
        Parameter sweep configuration
    figures : ManuscriptFiguresConfig
        Figure generation configuration
    pca : PCAConfig
        PCA configuration
    preprocessing : PreprocessingConfig
        Feature typing / preprocessing configuration (Phase 1)
    """
    run_id: str = "R0"
    rep_id: int = 0
    seed: int = 123
    device: str = "cpu"
    # Default optimization space (VAE latent; raw/PCA are ablation-only)
    space: str = "vae"
    files: FilesConfig = field(default_factory=FilesConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    vae: VAEConfig = field(default_factory=VAEConfig)
    geo: GeoConfig = field(default_factory=GeoConfig)
    sinkhorn: SinkhornConfig = field(default_factory=SinkhornConfig)
    mmd: MMDConfig = field(default_factory=MMDConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    baselines: BaselinesConfig = field(default_factory=BaselinesConfig)
    ablations: AblationsConfig = field(default_factory=AblationsConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    figures: ManuscriptFiguresConfig = field(default_factory=ManuscriptFiguresConfig)
    pca: PCAConfig = field(default_factory=PCAConfig)
