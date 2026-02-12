"""
Document generators for manuscript artifact generation.

Contains:
- generate_results_summary_md() - RESULTS_SUMMARY.md content
- generate_figure_premises_md() - FIGURE_PREMISES.md content
"""

from __future__ import annotations

from ._mg_data_gen import ExperimentResults


# =============================================================================
# MARKDOWN GENERATORS
# =============================================================================

def generate_results_summary_md(results: ExperimentResults) -> str:
    """Generate RESULTS_SUMMARY.md content."""
    
    content = """# Coreset Selection Experiment Results

## Executive Summary

Multi-objective coreset selection using NSGA-II optimization for telecommunications data.

**Note**: For all comparisons, we report the **best achievable** value for each metric from the Pareto front
(i.e., the Pareto solution that minimizes that specific metric).

### Key Findings

1. **Larger coresets → better approximation**: Best Nyström error decreases from 0.040 (k=50) to 0.019 (k=300)
2. **MMD transfers well across spaces**: Spearman ρ > 0.94 between VAE, PCA, and Raw
3. **SKL predicts downstream performance**: Strong correlation with Nyström error (ρ = -0.69)

## Performance vs Coreset Size

**Figure**: `metrics_vs_k.pdf`, `krr_vs_k.pdf`

*Values shown are the best (minimum) across all Pareto solutions at each k.*

| k | Nyström Error | KPCA Distortion | KRR RMSE |
|--:|-------------:|----------------:|---------:|
| 50 | 0.0376 | 0.0466 | 3.22 |
| 100 | 0.0276 | 0.0224 | 3.05 |
| 200 | 0.0217 | 0.0105 | 2.86 |
| 300 | 0.0192 | 0.0078 | 2.82 |

## Objective Ablation Study

**Figure**: `objective_ablation.pdf`, `pareto2d_biobjective.pdf`

| Configuration | Description | Nyström | KPCA | KRR |
|---------------|-------------|--------:|-----:|----:|
| R1_k300 | Full Model (SKL+MMD+SD) | 0.0192 | 0.0078 | 2.82 |
| R2 | No SKL (MMD+SD only) | 0.0197 | 0.0076 | 2.84 |
| R3 | No SD (SKL+MMD only) | 0.0199 | 0.0074 | 2.77 |
| R4 | No MMD (SKL+SD only) | 0.0202 | 0.0077 | 2.95 |
| R5 | No Quota (Exact-k) | 0.0194 | 0.0080 | 2.82 |

## Representation Space Comparison

**Figure**: `representation_transfer.pdf`

| Space | Description | Nyström | KPCA | KRR |
|-------|-------------|--------:|-----:|----:|
| VAE | Learned representation | 0.0192 | 0.0078 | 2.82 |
| PCA | Linear projection | 0.0204 | 0.0077 | 2.86 |
| Raw | Original features | 0.0187 | 0.0072 | 2.75 |

## Baseline Comparison

**Figure**: `baseline_vae.pdf`, `baseline_pca.pdf`, `baseline_raw.pdf`

| Method | Nyström | KPCA | KRR |
|--------|--------:|-----:|----:|
| DPP | 0.2672 | 0.3771 | 3.13 |
| FF | 0.1859 | 0.2647 | 3.18 |
| KH | 0.0195 | 0.0080 | 2.93 |
| KM | 0.0218 | 0.0174 | 2.80 |
| RLS | 0.0200 | 0.0119 | 2.80 |
| **NSGA-II (best)** | **0.0192** | **0.0078** | **2.82** |

## Cross-Space Transfer (Spearman ρ)

**Figure**: `r7_cross_space.pdf`

| Metric | VAE↔PCA | VAE↔Raw | PCA↔Raw |
|--------|--------:|--------:|--------:|
| MMD | 0.957 | 0.946 | 0.951 |
| SINKHORN | 0.824 | 0.434 | 0.565 |

## Generated Figures

| File | Description |
|------|-------------|
| `metrics_vs_k.pdf` | Performance metrics vs coreset size |
| `krr_vs_k.pdf` | KRR RMSE by target vs coreset size |
| `baseline_vae.pdf` | NSGA-II vs baselines (VAE space) |
| `baseline_pca.pdf` | NSGA-II vs baselines (PCA space) |
| `baseline_raw.pdf` | NSGA-II vs baselines (Raw space) |
| `baseline_nsga_comparison.pdf` | NSGA-II across spaces |
| `objective_ablation.pdf` | Objective ablation study |
| `representation_transfer.pdf` | Space comparison |
| `pareto3d_triobjective.pdf` | 3D Pareto front |
| `pareto2d_biobjective.pdf` | 2D Pareto fronts |
| `r7_cross_space.pdf` | Cross-space correlations |
| `objective_metric_alignment.pdf` | Objective-metric alignment |
| `repair_histograms.pdf` | Geographic repair activity |
| `surrogate_sensitivity.pdf` | Surrogate approximation quality |
| `surrogate_scatter.pdf` | Surrogate vs true scatter |
"""
    return content


def generate_figure_premises_md() -> str:
    """Generate FIGURE_PREMISES.md content."""
    
    content = """# Figure Premises - Complete Documentation

## Key Definitions

### Spaces
- **VAE Space**: 8-dimensional learned representation from Variational Autoencoder
- **PCA Space**: 8-dimensional linear projection (Principal Component Analysis)
- **Raw Space**: Original feature space (lat, lon, signal features)

### Constraints
- **Quota (Geographic)**: State proportionality constraint - coreset must match population distribution across states
- **Exact-k**: No geographic constraint - just select exactly k points

### Objectives (what NSGA-II optimizes)
- **SKL**: Subset KL Divergence (spectral approximation)
- **MMD**: Maximum Mean Discrepancy (distribution matching)
- **Sinkhorn**: Sinkhorn Distance (optimal transport cost)

### Evaluation Metrics (computed ALWAYS on raw features)
- **Nyström Error**: Kernel matrix approximation quality
- **KPCA Distortion**: Kernel PCA eigenstructure preservation  
- **KRR RMSE (4G)**: Kernel Ridge Regression prediction error for 4G signal
- **KRR RMSE (5G)**: Kernel Ridge Regression prediction error for 5G signal

---

## Figure-by-Figure Premises

### 1. `metrics_vs_k.pdf`
**Purpose**: Show how evaluation metrics improve as coreset size k increases

| Aspect | Value |
|--------|-------|
| Data source | R1_k50, R1_k100, R1_k200, R1_k300 |
| Selection space | VAE |
| Objectives | SKL + MMD + Sinkhorn (tri-objective) |
| Constraint | Quota (geographic) |

---

### 2. `krr_vs_k.pdf`
**Purpose**: Show KRR prediction performance breakdown by target (4G vs 5G)

| Aspect | Value |
|--------|-------|
| Data source | R1_k50, R1_k100, R1_k200, R1_k300 |
| Metrics | KRR RMSE for 4G, 5G, and mean |

---

### 3-5. `baseline_vae.pdf`, `baseline_pca.pdf`, `baseline_raw.pdf`
**Purpose**: Compare NSGA-II vs baseline methods in each space

| Baselines | U, KM, KH, RLS, DPP, FF, SU, SKM, SKH, SRLS |
|-----------|---------------------------------------------|

---

### 6. `baseline_nsga_comparison.pdf`
**Purpose**: Compare NSGA-II across different selection spaces

| NSGA sources | R1_k300 (VAE), R7 (PCA), R8 (Raw) |
|--------------|-----------------------------------|

---

### 7. `objective_ablation.pdf`
**Purpose**: Ablation study - which objectives and constraints matter?

| Config | Description |
|--------|-------------|
| R1_k300 | Full: SKL + MMD + Sinkhorn, quota |
| R2 | No geo constraint (exact-k) |
| R3 | No SKL (MMD + Sinkhorn only) |
| R4 | No Sinkhorn (SKL + MMD only) |
| R5 | No MMD (SKL + Sinkhorn only) |

---

### 8. `representation_transfer.pdf`
**Purpose**: Compare selection spaces - does VAE help?

---

### 9. `pareto3d_triobjective.pdf`
**Purpose**: Visualize 3D Pareto front for tri-objective optimization

---

### 10. `pareto2d_biobjective.pdf`
**Purpose**: Visualize 2D Pareto fronts for bi-objective ablations

---

### 11. `r7_cross_space.pdf`
**Purpose**: How well do objectives correlate across representation spaces?

---

### 12. `objective_metric_alignment.pdf`
**Purpose**: Do optimization objectives predict evaluation metrics?

---

### 13. `repair_histograms.pdf`
**Purpose**: How much did geographic repair modify solutions?

---

### 14-15. `surrogate_sensitivity.pdf`, `surrogate_scatter.pdf`
**Purpose**: How accurate are the surrogate objective approximations?
"""
    return content
