#!/usr/bin/env python
"""
Build fresh replicate caches for the adaptive-tau experiment.

A **replicate cache** is a pre-computed, self-contained snapshot of all
data transformations needed before coreset optimisation can begin.  Each
cache stores:

  1. **Preprocessed features** -- imputation of missing values, log1p
     transform of skewed columns, and z-score standardisation.
  2. **VAE embedding** -- a variational-autoencoder latent representation
     (1 500 epochs, latent_dim=32) trained on the full preprocessed data.
  3. **PCA embedding** -- a deterministic PCA projection (n_components=32)
     used as a secondary representation space.
  4. **Evaluation split** -- a stratified held-out set (|E| = 2 000) used
     exclusively at evaluation time so that optimisation never sees the
     evaluation observations.

Caching these artefacts up-front serves two purposes:

  * **Reproducibility** -- every replicate of every experiment draws from
    the same preprocessed data and embeddings, keyed by a unique seed.
  * **Efficiency** -- VAE training dominates wall-clock time; caching it
    once avoids redundant GPU hours across dozens of downstream runs.

This script creates 5 replicate caches with seeds 7001--7005, chosen to
be entirely disjoint from the original seed family 2026--2030.

Experiment context
------------------
k = 100, population-share constraint, VAE optimisation space, adaptive
tau scheduling (probe / bisect / production policy).

Usage (typically run on LABGELE server)::

    python -u scripts/build_caches_adaptive_tau.py
    python -u scripts/build_caches_adaptive_tau.py --reps 3   # single rep
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

# ---------------------------------------------------------------------------
# Ensure the project root is importable regardless of working directory.
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from coreset_selection.cli._config import build_base_config
from coreset_selection.data.cache import build_replicate_cache

# -- Experiment configuration ------------------------------------------------

EXPERIMENT_NAME: str = "adaptive_tau_k100_ps_vae"
EXPERIMENT_DIR: Path = PROJECT_ROOT / "experiments" / EXPERIMENT_NAME

# Seed mapping: replicate index -> global seed.
# Seeds 7001--7005 are chosen to have *no overlap* with the original
# experiment family (seeds 2026--2030) so that results are statistically
# independent and can safely be pooled in bootstrap analyses.
SEEDS: Dict[int, int] = {
    0: 7001,
    1: 7002,
    2: 7003,
    3: 7004,
    4: 7005,
}

CACHE_DIR: Path = EXPERIMENT_DIR / "replicate_cache"


def main() -> None:
    """Parse CLI arguments, build replicate caches, and write a seed manifest.

    For each requested replicate the function:

    1. Constructs a ``DataConfig`` via :func:`build_base_config`.
    2. Adjusts the config seed so that the *effective* seed inside
       :func:`build_replicate_cache` equals the value in :data:`SEEDS`.
    3. Calls :func:`build_replicate_cache`, which trains the VAE, fits
       PCA, and persists everything under ``CACHE_DIR/rep<id>/``.
    4. Writes a JSON manifest recording the seed-to-replicate mapping for
       downstream audit and reproduction.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Build replicate caches for adaptive-tau experiment",
    )
    parser.add_argument(
        "--reps", nargs="+", type=int, default=list(range(5)),
        help="Which replicas to build (default: 0 1 2 3 4)",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Compute device for VAE training (cpu or cuda)",
    )
    args: argparse.Namespace = parser.parse_args()

    # Create experiment directory structure
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(EXPERIMENT_DIR / "results", exist_ok=True)

    print(f"{'='*70}")
    print(f"  BUILDING CACHES FOR: {EXPERIMENT_NAME}")
    print(f"  Cache dir: {CACHE_DIR}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Reps: {args.reps}")
    print(f"{'='*70}\n")

    for rep_id in args.reps:
        seed: int = SEEDS[rep_id]
        print(f"\n{'='*60}")
        print(f"  REPLICA {rep_id} / SEED {seed}")
        print(f"{'='*60}\n")

        t0: float = time.time()

        # Build base config pointing to our experiment cache dir
        base_cfg = build_base_config(
            output_dir=str(EXPERIMENT_DIR / "results"),
            cache_dir=str(CACHE_DIR),
            data_dir=str(PROJECT_ROOT / "data"),
            seed=seed,
            device=args.device,
        )

        # ------------------------------------------------------------------
        # Seed arithmetic: compensate for the internal offset.
        #
        # ``build_replicate_cache`` computes its effective seed as
        #     effective_seed = cfg.seed + rep_id
        #
        # We want the effective seed for replicate ``rep_id`` to be exactly
        # ``seed`` (the value from the SEEDS table).  Therefore we pass
        #     cfg.seed = seed - rep_id
        # so that  (seed - rep_id) + rep_id == seed.
        # ------------------------------------------------------------------
        cfg = replace(base_cfg, seed=seed - rep_id)

        try:
            cache_path = build_replicate_cache(cfg, rep_id)
            elapsed: float = time.time() - t0
            print(f"\n[Rep {rep_id}] Cache saved to {cache_path} ({elapsed:.0f}s)")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"\n[Rep {rep_id}] FAILED after {elapsed:.0f}s: {e}")
            import traceback
            traceback.print_exc()

    # ------------------------------------------------------------------
    # Persist a JSON manifest so that any later analysis can trace which
    # seed produced which replicate without re-reading this script.
    # ------------------------------------------------------------------
    manifest: Dict[str, object] = {
        "experiment": EXPERIMENT_NAME,
        "seeds": {str(k): v for k, v in SEEDS.items()},
        "description": (
            "k=100, population-share constraint, VAE space, "
            "adaptive tau with probe/bisect/production policy"
        ),
    }
    manifest_path: Path = EXPERIMENT_DIR / "seed_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSeed manifest saved to {manifest_path}")

    print(f"\n{'='*60}")
    print(f"  DONE — caches in {CACHE_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
