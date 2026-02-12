"""
Manuscript Constants.

Central repository for all numerical constants specified in the manuscript
"Constrained Nyström Landmark Selection for Scalable Telecom Analytics".

These constants ensure reproducibility and compliance with the experimental
specifications in the paper (Table 1 hyperparameters and Table 2 run matrix).
"""

from __future__ import annotations

import numpy as np


# =============================================================================
# Dataset Constants (Section 5.1)
# =============================================================================

# Expected dataset size after preprocessing
N_MUNICIPALITIES = 5569  # N = 5569 municipalities
G_STATES = 27           # G = 27 states (including Federal District)
# Per the manuscript (Section 5.4 and Section 5.7), the Brazil telecom dataset
# used in experiments has **D = 621** numeric covariates. This count includes
# population and geographic coordinates.
D_FEATURES = 621        # D = 621 numeric covariates

# =============================================================================
# Cardinality Grid (Table 4)
# =============================================================================

K_GRID = [50, 100, 200, 300, 400, 500]
K_PRIMARY = 300  # Primary k value used for most analyses

# =============================================================================
# Dimension Grid for VAE/PCA Representation Sweep (R13/R14)
# =============================================================================

D_GRID = (4, 8, 16, 32, 64, 128)

# =============================================================================
# NSGA-II Parameters (Section 5.3, Algorithm 3)
# =============================================================================

NSGA2_POP_SIZE = 200        # P = 200 (population size)
NSGA2_N_GENERATIONS = 1000  # T = 1000 (number of generations)
NSGA2_CROSSOVER_PROB = 0.9  # Crossover probability
NSGA2_MUTATION_PROB = 0.2   # Mutation probability p_m = 0.2 (manuscript Table 1)

# =============================================================================
# Objective Function Parameters (Section 5.4-5.6)
# =============================================================================

# MMD / RFF Parameters
RFF_DIM_DEFAULT = 2000  # m = 2000 random Fourier features
RFF_DIM_SENSITIVITY = [500, 1000, 2000, 4000]  # m values for sensitivity analysis

# Sinkhorn Parameters (Section 5.6)
SINKHORN_N_ANCHORS = 200     # A = 200 anchor points
SINKHORN_ETA = 0.05          # η = 0.05 (epsilon scale factor)
SINKHORN_MAX_ITER = 100      # 100 log-stabilized iterations
SINKHORN_ANCHOR_SENSITIVITY = [50, 100, 200, 400]  # A values for sensitivity

# SKL Variance Clamping (Section 5.8.3)
SKL_VAR_CLAMP_MIN = np.exp(-10)  # ≈ 4.54e-5
SKL_VAR_CLAMP_MAX = np.exp(2)    # ≈ 7.389

# =============================================================================
# Geographic Constraint Parameters (Section 5.1)
# =============================================================================

ALPHA_GEO = 1.0  # α_geo = 1.0 (Dirichlet smoothing parameter)

# =============================================================================
# VAE Parameters (Section 5.8.1)
# =============================================================================

VAE_LATENT_DIM = 32       # d_z = 32 latent dimensions (manuscript Table 1)
VAE_HIDDEN_DIM = 128      # Hidden layer dimension
VAE_EPOCHS = 1500         # Training epochs E = 1500 (with early stopping)
VAE_EARLY_STOPPING_PATIENCE = 200  # Early stopping patience (epochs)
VAE_BATCH_SIZE = 256      # Batch size = 256 (manuscript Table 1)
VAE_LR = 1e-3             # Learning rate
VAE_KL_WEIGHT = 0.1       # β-VAE KL weight

# =============================================================================
# PCA Parameters (Section 5.8.2)
# =============================================================================

PCA_DIM = 32              # d_z = 32 (latent dim matches VAE for fair comparison)

# =============================================================================
# Multi-Target Coverage (Table V - KRR Multi-task)
# =============================================================================
# The manuscript evaluates KRR predictive fidelity on 10 coverage
# indicators (Table V).  The first two (area-coverage 4G/5G) are the
# primary targets; the next two are per-technology; the remaining six
# are derived combined/averaged targets constructed after loading.
#
# Canonical target names matching manuscript Table V exactly:
COVERAGE_TARGETS_TABLE_V = {
    "cov_area_4G":       "Area (4G)",
    "cov_area_5G":       "Area (5G)",
    "cov_hh_4G":         "Households (4G)",
    "cov_res_4G":        "Residents (4G)",
    "cov_area_4G_5G":    "Area (4G + 5G)",
    "cov_area_all":      "Area (All)",
    "cov_hh_4G_5G":      "Households (4G + 5G)",
    "cov_hh_all":         "Households (All)",
    "cov_res_4G_5G":     "Residents (4G + 5G)",
    "cov_res_all":        "Residents (All)",
}

# Backward-compatible alias
COVERAGE_TARGETS = COVERAGE_TARGETS_TABLE_V

# Ordered list for consistent iteration / table column order (Table V row order)
COVERAGE_TARGET_NAMES = list(COVERAGE_TARGETS_TABLE_V.keys())

# Legacy name mapping: old pipeline keys → new Table V keys
# (used by _discover_coverage_targets and _build_multitarget_y for bridging)
_LEGACY_TARGET_KEY_MAP = {
    "cov_households_4G": "cov_hh_4G",
    "cov_households_5G": "cov_hh_5G",  # intermediate; not in Table V directly
    "cov_residents_4G":  "cov_res_4G",
    "cov_residents_5G":  "cov_res_5G",  # intermediate; not in Table V directly
}

# =============================================================================
# Evaluation Parameters (Section 5.9)
# =============================================================================

EVAL_SIZE = 2000          # |E| = 2000 evaluation indices
EVAL_TRAIN_FRAC = 0.8     # 80/20 train/test split within E
KPCA_COMPONENTS = 20      # r = 20 kPCA components
NYSTROM_LAMBDA = 1e-6     # λ_nys scaling factor (multiplied by tr(W)/k)

# =============================================================================
# Experimental Design (Table 2)
# =============================================================================

N_REPLICATES_PRIMARY = 5  # 5 replicates for R1 at k=300 only (population_share + MMD+Sinkhorn)
N_REPLICATES_DEFAULT = 1  # 1 replicate for all other configurations

# Run specifications (manuscript Table 2)
RUN_IDS = ["R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11", "R12", "R13", "R14"]

# =============================================================================
# Effort Sweep Grid (R12, Section VII.G / VIII.L)
# =============================================================================
# Explicit effort grid for the NSGA-II effort sweep experiment.
# Each entry is a coupled (pop_size, n_gen) pair.
EFFORT_GRID = [
    {"pop_size": 20,  "n_gen": 100},
    {"pop_size": 50,  "n_gen": 300},
    {"pop_size": 100, "n_gen": 500},
    {"pop_size": 150, "n_gen": 700},
    {"pop_size": 200, "n_gen": 1000},
    {"pop_size": 300, "n_gen": 1500},
    {"pop_size": 400, "n_gen": 2000},
]

# Objectives by run — per manuscript Table 2:
#   Default bi-objective: (MMD, Sinkhorn divergence)
#   SKL is an ablation objective only (R7)
RUN_OBJECTIVES = {
    "R0": [],                          # Quota computation only
    "R1": ["mmd", "sinkhorn"],         # PRIMARY: bi-objective
    "R2": ["mmd"],                     # MMD-only ablation
    "R3": ["sinkhorn"],                # Sinkhorn-only ablation
    "R4": ["mmd", "sinkhorn"],         # Municipality-share constraint swap
    "R5": ["mmd", "sinkhorn"],         # Joint constraints
    "R6": ["mmd", "sinkhorn"],         # No proportionality constraints (exact-k only)
    "R7": ["mmd", "sinkhorn", "skl"],  # SKL ablation (tri-objective, VAE mean space)
    "R8": ["mmd", "sinkhorn"],         # PCA space representation transfer
    "R9": ["mmd", "sinkhorn"],         # VAE mean space representation transfer
    "R10": [],                         # Baseline suite
    "R11": [],                         # Diagnostics (proxy stability, alignment)
    "R12": ["mmd", "sinkhorn"],        # Effort sweep
    "R13": ["mmd", "sinkhorn"],        # VAE latent dimension sweep
    "R14": ["mmd", "sinkhorn"],        # PCA dimension sweep
}

# =============================================================================
# Baseline Methods (Section 5.7)
# =============================================================================

BASELINE_METHODS = [
    "uniform",        # Random uniform sampling
    "kmeans",         # k-means representatives (closest/medoid)
    "herding",        # Kernel herding in RFF space
    "farthest_first", # Farthest-first traversal
    "kernel_thinning",# Kernel thinning (Dwivedi et al.)
    "rls",            # Approximate ridge leverage-score Nyström sampling
    "dpp",            # Approximate k-DPP sampling
]

# =============================================================================
# Figure and Table Counts (for validation)
# =============================================================================

N_FIGURES = 18  # Total figures in manuscript
N_TABLES = 12   # Total tables in manuscript

# ---- Manuscript-referenced figures (Sec VIII, Figs 1-4) ----
# These filenames must match the \includegraphics{} paths in main.tex exactly.
MANUSCRIPT_FIGURE_FILES = [
    "geo_ablation_tradeoff_scatter.pdf",       # Fig 1: R6 composition drift vs Nyström error
    "distortion_cardinality_R1.pdf",           # Fig 2: R1 budget-fidelity 2×2 panel
    "regional_validity_k300.pdf",              # Fig 3: R1/R5 state KPI stability
    "objective_metric_alignment_heatmap.pdf",  # Fig 4: R11 Spearman heatmap
]

# ---- Complementary / extra figures (produced by manuscript_artifacts.py) ----
# Phase 10a (Figs N1–N6) + Phase 10b (Figs N7–N12): Narrative-strengthening figures
COMPLEMENTARY_FIGURE_FILES = [
    # Phase 10a: Figs N1–N6
    "kl_floor_vs_k.pdf",                      # Fig N1: KL_min(k) planning curve + τ thresholds (R0)
    "pareto_front_mmd_sd_k300.pdf",            # Fig N2: Pareto front + knee-point + R2/R3 overlay (R1)
    "objective_ablation_bars_k300.pdf",        # Fig N3: Grouped bar ablation (R1/R2/R3)
    "constraint_comparison_bars_k300.pdf",     # Fig N4: 2×2 constraint regime panel (R1/R4/R5/R6)
    "effort_quality_tradeoff.pdf",             # Fig N5: Effort sweep + diminishing returns (R12)
    "baseline_comparison_grouped.pdf",         # Fig N6: Multi-metric baseline comparison (R10)
    # Phase 10b: Figs N7–N12
    "multi_seed_stability_boxplot.pdf",        # Fig N7: R1/R5 seed robustness boxplots
    "state_kpi_heatmap.pdf",                   # Fig N8: Per-state KPI drift heatmap + small-state marks
    "composition_shift_sankey.pdf",            # Fig N9: π_g vs π̂_g(S) side-by-side (R6 vs R1)
    "pareto_front_evolution.pdf",              # Fig N10: Front overlay by generation checkpoint
    "nystrom_error_distribution.pdf",          # Fig N11: e_Nys histogram/KDE across Pareto solutions
    "krr_worst_state_rmse_vs_k.pdf",           # Fig N12: Worst-state RMSE vs k (equity analysis)
    # Additional complementary analyses
    "representation_transfer_bars.pdf",        # R8/R9 vs R1 representation transfer
    "time_vs_k_combined.pdf",                  # Time complexity vs k (extra)
    "skl_ablation_comparison.pdf",             # R7 SKL ablation: bi-obj vs tri-obj
    "cumulative_pareto_improvement.pdf",       # Pareto front evolution proxy (R12)
    "constraint_tightness_vs_fidelity.pdf",    # Constraint tightness vs operator fidelity
]

# ---- Legacy generator figures (produced by generator.py / manuscript_generator.py) ----
LEGACY_FIGURE_FILES = [
    "klmin_vs_k.pdf",                          # Fig. 5
    "pareto3d_k300_R1.pdf",                    # Fig. 6
    "pareto2d_skl_mmd_k300_R1.pdf",            # Fig. 7a
    "pareto2d_skl_sd_k300_R1.pdf",             # Fig. 7b
    "pareto2d_mmd_sd_k300_R1.pdf",             # Fig. 7c
    "pareto2d_biobjective_k300.pdf",           # Fig. 8
    "baseline_comparison_k300.pdf",            # Fig. 10
    "objective_ablation_fronts_k300.pdf",      # Fig. 12
    "representation_transfer_k300.pdf",        # Fig. 13
    "surrogate_rankcorr.pdf",                  # Fig. 14
    "surrogate_scatter_reference_vs_alt.pdf",  # Fig. 15
    "repair_magnitude_histograms.pdf",         # Fig. 16
    "objective_metric_scatter_examples.pdf",   # Fig. 18
]

# Combined list (for backward compatibility)
FIGURE_FILES = list(dict.fromkeys(
    MANUSCRIPT_FIGURE_FILES + COMPLEMENTARY_FIGURE_FILES + LEGACY_FIGURE_FILES
))

# ---- Manuscript-referenced table labels (\label{tab:...} in main.tex) ----
MANUSCRIPT_TABLE_LABELS = [
    "tab:exp-settings",         # Table I: Hyperparameters
    "tab:run-matrix",           # Table II: Run matrix
    "tab:r1-by-k",              # Table III: R1 metric envelope vs k
    "tab:proxy-stability",      # Table IV: Proxy stability diagnostics
    "tab:krr-multitask-k300",   # Table V: Multi-target KRR RMSE
]

# ---- Manuscript-referenced table filenames ----
MANUSCRIPT_TABLE_FILES = [
    "exp_settings.tex",         # Table I  (G6: auto-generated from constants)
    "run_matrix.tex",           # Table II (G6: auto-generated from run_specs)
    "r1_by_k.tex",              # Table III
    "proxy_stability.tex",      # Table IV
    "krr_multitask_k300.tex",   # Table V
]

# ---- Complementary table filenames ----
COMPLEMENTARY_TABLE_FILES = [
    "baseline_summary_k300.csv",           # R10 baseline results
    "baseline_variants_summary.csv",       # G7: structured baseline variant summary
    "baseline_paired_comparison.csv",      # G7: paired exactk-vs-quota comparison
    "constraint_diagnostics_k300.csv",     # Cross-config constraint diagnostics
    "objective_ablation_k300.tex",         # R2/R3 vs R1 metrics
    "representation_transfer_k300.tex",    # R8/R9 vs R1 metrics
    "effort_sweep_k300.tex",              # R12 effort vs quality
    "effort_grid_config.csv",             # G10: R12 parameter grid specification
    "effort_sweep_summary.csv",           # G10: R12 structured knee-representative summary
    "time_complexity_summary.tex",         # Time complexity per k
    "skl_ablation_summary.tex",           # R7 SKL increment
    "multi_seed_statistics.tex",           # R1/R5 mean ± std across seeds
    # Phase 11: Narrative-strengthening tables (N1–N7)
    "constraint_diagnostics_cross_config.tex",        # N1: Cross-config proportionality
    "constraint_diagnostics_cross_config.csv",        # N1: companion CSV
    "objective_ablation_summary.tex",                 # N2: R1 vs R2 vs R3 ablation
    "objective_ablation_summary.csv",                 # N2: companion CSV
    "representation_transfer_summary.tex",            # N3: R1 vs R8 vs R9 transfer
    "representation_transfer_summary.csv",            # N3: companion CSV
    "skl_ablation_summary.csv",                       # N4: companion CSV
    "multi_seed_statistics.csv",                      # N5: companion CSV
    "worst_state_rmse_by_k.tex",                      # N6: Worst-state RMSE equity
    "worst_state_rmse_by_k.csv",                      # N6: companion CSV
    "baseline_paired_unconstrained_vs_quota.tex",     # N7: Baseline paired comparison
    "baseline_paired_unconstrained_vs_quota.csv",     # N7: companion CSV
]

# Legacy table labels (backward compatibility)
TABLE_LABELS = [
    "tab:run-matrix",
    "tab:klmin-summary",
    "tab:front-stats",
    "tab:cardinality-metrics",
    "tab:baseline-quota",
    "tab:baseline-unconstrained",
    "tab:objective-ablations",
    "tab:repr-transfer",
    "tab:surrogate-sensitivity",
    "tab:repair-activity",
    "tab:crossspace-objectives",
    "tab:obj-metric-align",
]

# =============================================================================
# Threading Constants
# =============================================================================

N_THREADS = 4  # Fixed thread count per experiment (intra-op and environment)

# =============================================================================
# Numerical Stability Constants
# =============================================================================

EPS_NORM = 1e-12  # Small constant for min-max normalization
EPS_LOG = 1e-30   # Small constant for log stability

# =============================================================================
# Extended Downstream Targets (beyond coverage Table V)
# =============================================================================

# Extra regression targets: all continuous, non-ordinal columns
EXTRA_REGRESSION_TARGETS = {
    # Speed & Performance (continuous)
    "velocidade_mediana_mean":         "Median Speed (mean)",
    "velocidade_mediana_std":          "Median Speed (std)",
    "pct_limite_mean":                 "Speed Cap Ratio (mean %)",
    # Socioeconomic (continuous)
    "renda_media_mean":                "Mean Income",
    "renda_media_std":                 "Income Variability",
    # Market concentration (continuous 0-1 index)
    "HHI SMP_2024":                    "HHI Mobile (2024)",
    "HHI SCM_2024":                    "HHI Fixed (2024)",
    # Infrastructure quality (continuous %)
    "pct_fibra_backhaul":              "Fiber Backhaul (%)",
    "pct_escolas_internet":            "Schools w/ Internet (%)",
    "pct_escolas_fibra":               "Schools w/ Fiber (%)",
    # Service density (continuous per-capita)
    "Densidade_Banda Larga Fixa_2025": "Broadband Density (2025)",
    "Densidade_Telefonia Móvel_2025":  "Mobile Density (2025)",
}

EXTRA_REGRESSION_TARGET_NAMES = list(EXTRA_REGRESSION_TARGETS.keys())

# Classification targets: binary and multiclass derived from data
# Each value is (display_label, n_classes_description)
CLASSIFICATION_TARGETS = {
    # Binary (natural thresholds)
    "has_5g":                  ("5G Presence", "binary"),
    "has_fiber_backhaul":      ("Fiber Backhaul Presence", "binary"),
    "has_high_speed_internet": ("High-Speed Internet", "binary"),
    # 3-class (tercile-binned continuous columns)
    "urbanization_level":      ("Urbanization Level", "3-class"),
    "broadband_speed_tier":    ("Broadband Speed Tier", "3-class"),
    "income_tier":             ("Income Tier", "3-class"),
    # 4-class (composite / quartile-binned)
    "income_speed_class":      ("Income-Speed Quadrant", "4-class"),
    "mobile_penetration_tier": ("Mobile Penetration Tier", "4-class"),
    # 5-class (quintile-binned)
    "infra_density_tier":      ("Infrastructure Density", "5-class"),
    "road_coverage_4g_tier":   ("Road 4G Coverage", "5-class"),
}

CLASSIFICATION_TARGET_NAMES = list(CLASSIFICATION_TARGETS.keys())

# Combined target lists for iteration
ALL_REGRESSION_TARGET_NAMES = COVERAGE_TARGET_NAMES + EXTRA_REGRESSION_TARGET_NAMES
ALL_CLASSIFICATION_TARGET_NAMES = CLASSIFICATION_TARGET_NAMES

# =============================================================================
# Downstream Evaluation Models
# =============================================================================

DOWNSTREAM_MODELS = ["krr", "knn", "rf", "lr", "gbt"]
