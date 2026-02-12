"""
Constants used throughout the coreset selection package.

Contains:
- Month abbreviations for date parsing
- Brazilian state codes
- Default parameter values
"""

from __future__ import annotations

from typing import Dict, List

# Month abbreviations (Portuguese)
_MONTH_ABBR: Dict[str, str] = {
    "jan": "01",
    "fev": "02",
    "mar": "03",
    "abr": "04",
    "mai": "05",
    "jun": "06",
    "jul": "07",
    "ago": "08",
    "set": "09",
    "out": "10",
    "nov": "11",
    "dez": "12",
}

# Month abbreviations (English)
_MONTH_ABBR_EN: Dict[str, str] = {
    "jan": "01",
    "feb": "02",
    "mar": "03",
    "apr": "04",
    "may": "05",
    "jun": "06",
    "jul": "07",
    "aug": "08",
    "sep": "09",
    "oct": "10",
    "nov": "11",
    "dec": "12",
}

# Brazilian state codes (UF)
BRAZILIAN_STATES: List[str] = [
    "AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO",
    "MA", "MT", "MS", "MG", "PA", "PB", "PR", "PE", "PI",
    "RJ", "RN", "RS", "RO", "RR", "SC", "SP", "SE", "TO",
]

# Region groupings
BRAZILIAN_REGIONS: Dict[str, List[str]] = {
    "Norte": ["AC", "AP", "AM", "PA", "RO", "RR", "TO"],
    "Nordeste": ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"],
    "Centro-Oeste": ["DF", "GO", "MT", "MS"],
    "Sudeste": ["ES", "MG", "RJ", "SP"],
    "Sul": ["PR", "RS", "SC"],
}

# Default numerical parameters
DEFAULT_SEED: int = 123
DEFAULT_ALPHA_GEO: float = 1.0
DEFAULT_RFF_DIM: int = 2000
DEFAULT_SINKHORN_ETA: float = 0.05  # η for ε = η * median(||r_i - r_j||²)
DEFAULT_SINKHORN_ANCHORS: int = 200  # A = 200 anchors per manuscript
DEFAULT_SINKHORN_ITERATIONS: int = 100  # 100 Sinkhorn iterations per manuscript

# NSGA-II defaults: P=200, T=1000
DEFAULT_POP_SIZE: int = 200
DEFAULT_N_GEN: int = 1000
DEFAULT_CROSSOVER_PROB: float = 0.9
DEFAULT_MUTATION_PROB: float = 0.2

# VAE defaults (match manuscript Table 1)
DEFAULT_VAE_LATENT_DIM: int = 32
DEFAULT_VAE_HIDDEN_DIM: int = 128
DEFAULT_VAE_EPOCHS: int = 1500
DEFAULT_VAE_EARLY_STOPPING_PATIENCE: int = 50
DEFAULT_VAE_BATCH_SIZE: int = 256
DEFAULT_VAE_LR: float = 1e-3
DEFAULT_VAE_KL_WEIGHT: float = 0.1
