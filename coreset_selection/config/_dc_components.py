"""
Component-level configuration dataclasses.

Contains configuration structures for individual experiment components:
- File paths and I/O
- Preprocessing / feature typing
- VAE model parameters
- Geographic constraints
- Sinkhorn divergence
- MMD computation
- PCA baseline representation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class FilesConfig:
    """
    File and directory path configuration.

    Supports two data modes:
    1. Brazil telecom mode (use_brazil_telecom=True): Uses smp_main.csv
    2. Legacy mode: Uses separate cobertura/atendidos/setores files

    Attributes
    ----------
    data_dir : str
        Root directory for input data files
    output_dir : str
        Root directory for experiment outputs
    cache_dir : str
        Directory for replicate caches
    cache_path : str
        Path to specific replicate cache file
    use_brazil_telecom : bool
        If True, use Brazil telecom data format
    main_data_file : str
        Main features file (Brazil telecom mode)
    metadata_file : str
        Metadata file with coordinates (Brazil telecom mode)
    population_file : str
        Population file (Brazil telecom mode)
    population_csv : str
        Path to population CSV file (legacy mode)
    cobertura_zip : str
        Path to cobertura data ZIP (legacy mode)
    atendidos_zip : str
        Path to atendidos data ZIP (legacy mode)
    setores_parquet : str
        Path to setores parquet file (legacy mode)
    """
    data_dir: str = "data"
    output_dir: str = "runs_out"
    cache_dir: str = "replicate_cache"
    cache_path: str = ""
    # Brazil telecom mode settings
    use_brazil_telecom: bool = True
    main_data_file: str = "smp_main.csv"
    metadata_file: str = "metadata.csv"
    population_file: str = "city_populations.csv"
    # Legacy mode settings
    population_csv: str = ""
    cobertura_zip: str = ""
    atendidos_zip: str = ""
    setores_parquet: str = ""

    def __post_init__(self):
        """Set default file paths if not specified."""
        import os
        if not self.population_csv:
            self.population_csv = os.path.join(self.data_dir, "population_muni.csv")
        if not self.cobertura_zip:
            self.cobertura_zip = os.path.join(self.data_dir, "cobertura.zip")
        if not self.atendidos_zip:
            self.atendidos_zip = os.path.join(self.data_dir, "atendidos.zip")
        if not self.setores_parquet:
            self.setores_parquet = os.path.join(self.data_dir, "setores.parquet")


@dataclass
class PreprocessingConfig:
    """
    Feature typing and preprocessing configuration (Phase 1).

    Controls how columns in the input CSV are classified (numeric, ordinal,
    categorical, ignore, target) and how each type is preprocessed.

    Explicit column lists take highest priority; heuristic inference fills in
    the rest.

    Attributes
    ----------
    categorical_columns : List[str]
        Columns to force-classify as categorical (integer-encoded, no one-hot).
    ordinal_columns : List[str]
        Columns to force-classify as ordinal (integer-valued, ordered).
    ignore_columns : List[str]
        Columns to exclude from the feature matrix entirely.
    target_columns : List[str]
        Columns to treat as prediction targets (excluded from features).
        When non-empty, these override the default 4G/5G target detection
        so you can swap targets without editing regexes.
    treat_low_cardinality_int_as_categorical : bool
        If True, integer columns with <= ``low_cardinality_threshold`` unique
        values are auto-classified as categorical.
    low_cardinality_threshold : int
        Unique-value cutoff for the low-cardinality heuristic.
    high_cardinality_drop_threshold : Optional[int]
        If set, *inferred* categorical columns with more unique values than
        this are dropped to ``ignore`` (safety against accidental huge
        encodings). Explicit overrides are never dropped.
    treat_bool_as_categorical : bool
        If True, boolean columns become categorical; otherwise numeric.
    auto_detect_targets : bool
        If True, also apply the regex-based target detection from
        ``data/target_columns.py`` (in addition to explicit targets).
    strict_manuscript_mode : bool
        If True, enforce manuscript constants (D=621, etc.).  If False
        (default), allow any number of features.
    """
    categorical_columns: List[str] = field(default_factory=list)
    ordinal_columns: List[str] = field(default_factory=list)
    ignore_columns: List[str] = field(default_factory=list)
    target_columns: List[str] = field(default_factory=list)
    treat_low_cardinality_int_as_categorical: bool = True
    low_cardinality_threshold: int = 25
    high_cardinality_drop_threshold: Optional[int] = None
    treat_bool_as_categorical: bool = True
    auto_detect_targets: bool = True
    strict_manuscript_mode: bool = False
    # Phase 2: type-aware preprocessing knobs
    scale_ordinals: bool = True
    scale_categoricals: bool = False
    log1p_categoricals: bool = False
    log1p_ordinals: bool = False
    categorical_impute_strategy: str = "mode"  # "mode" or "missing_code"
    ordinal_impute_strategy: str = "median"     # "median" or "mode"
    # Phase 2: target type detection
    target_type: str = "auto"  # "auto", "regression", "classification"
    classification_cardinality_threshold: int = 50


@dataclass
class VAEConfig:
    """
    Variational Autoencoder configuration.

    Attributes
    ----------
    latent_dim : int
        Dimension of latent space
    hidden_dim : int
        Dimension of hidden layers
    epochs : int
        Number of training epochs
    batch_size : int
        Training batch size
    lr : float
        Learning rate
    kl_weight : float
        Weight for KL divergence term (beta-VAE)
    early_stopping_patience : int
        Patience for early stopping (0 = disabled)
    validation_frac : float
        Fraction of data for validation

    log_every : int
        Print training metrics every N epochs
    val_every : int
        Evaluate validation loss every N epochs when early stopping is disabled
    embed_batch_size : int
        Batch size used for embedding extraction (0 = use full batch)
    torch_threads : int
        Intra-op CPU threads for PyTorch (4 = fixed per experiment)
    torch_interop_threads : int
        Inter-op CPU threads for PyTorch (4 = fixed per experiment)
    """
    # Per manuscript Table 1
    latent_dim: int = 32
    hidden_dim: int = 128
    epochs: int = 750  # VAE training epochs (with early stopping + LR scheduler)
    batch_size: int = 256
    lr: float = 1e-3
    kl_weight: float = 0.1
    early_stopping_patience: int = 80  # Early stopping patience (epochs)
    validation_frac: float = 0.1

    # Performance / logging controls
    log_every: int = 10
    val_every: int = 5
    embed_batch_size: int = 0

    # Threading: fixed at 4 threads per experiment.
    torch_threads: int = 4
    torch_interop_threads: int = 4


@dataclass
class GeoConfig:
    """
    Geographic constraint configuration.

    Attributes
    ----------
    alpha_geo : float
        Dirichlet smoothing parameter for KL computation
    min_one_per_group : bool
        Ensure at least one sample per geographic group
    use_quota_constraints : bool
        Use quota-based constraints (vs. unconstrained)
    include_geo_objective : bool
        Include geographic KL as an optimization objective
    group_column : str
        Column name for geographic grouping
    """
    # Smoothing parameter alpha in the smoothed proportionality diagnostics
    alpha_geo: float = 1.0
    # Lower bounds l_g (implemented as "at least one per state" in the paper)
    min_one_per_group: bool = True

    # ------------------------------------------------------------------
    # Proportionality constraint modes (manuscript Section IV-B)
    # ------------------------------------------------------------------
    # - population_share: weights w_i = pop_i (PRIMARY constraint)
    # - municipality_share_quota: count quota c*(k) (w_i = 1, quota mode)
    # - joint: both population-share and municipality-share quota
    # - none: no proportionality constraints (exact-k only)
    constraint_mode: str = "population_share"

    # Backward-compat flag: when True, count-quota mode is enabled.
    # Internally this is mapped to constraint_mode.
    use_quota_constraints: bool = False

    # Tolerances tau_h for inequality KL constraints.
    # The paper treats tau as user-controlled; defaults are conservative.
    tau_population: float = 0.02
    tau_municipality: float = 0.02

    include_geo_objective: bool = False
    group_column: str = "UF"


@dataclass
class SinkhornConfig:
    """
    Sinkhorn divergence configuration.

    Per manuscript Section 5.6:
    - eps = eta * median(||r_i - r_j||^2) with eta = 0.05
    - A = 200 anchors computed via k-means with k-means++ seeding
    - 100 Sinkhorn iterations

    Attributes
    ----------
    n_anchors : int
        Number of anchor points for approximation (A = 200)
    eta : float
        Scaling factor for entropic regularization (eta = 0.05)
    max_iter : int
        Maximum Sinkhorn iterations (100 per manuscript)
    stop_thr : float
        Convergence threshold
    cost_scale : float
        Cost matrix scaling factor (0 = auto from median heuristic)
    anchor_method : str
        Method for selecting anchors ('kmeans', 'random', 'farthest')
    """
    n_anchors: int = 200
    eta: float = 0.05
    max_iter: int = 100
    stop_thr: float = 1e-6
    cost_scale: float = 0.0
    anchor_method: str = "kmeans"


@dataclass
class MMDConfig:
    """
    Maximum Mean Discrepancy configuration.

    Attributes
    ----------
    rff_dim : int
        Number of Random Fourier Features
    bandwidth_mult : float
        Multiplier for median heuristic bandwidth
    """
    rff_dim: int = 2000
    bandwidth_mult: float = 1.0


@dataclass
class PCAConfig:
    """
    PCA configuration for baseline representation.

    Attributes
    ----------
    n_components : int
        Number of PCA components
    whiten : bool
        Whether to whiten the components
    """
    n_components: int = 32
    whiten: bool = False
