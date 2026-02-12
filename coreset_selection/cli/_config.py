"""Configuration building helpers for the CLI."""

from __future__ import annotations

from typing import List, Optional

from ..config.dataclasses import (
    ExperimentConfig,
    FilesConfig,
    VAEConfig,
    GeoConfig,
    SinkhornConfig,
    MMDConfig,
    SolverConfig,
    EvalConfig,
    BaselinesConfig,
)


def build_base_config(
    output_dir: str = "runs_out",
    cache_dir: str = "replicate_cache",
    data_dir: str = "data",
    seed: int = 123,
    device: str = "cpu",
) -> ExperimentConfig:
    """
    Build a base experiment configuration.

    Parameters
    ----------
    output_dir : str
        Output directory for results
    cache_dir : str
        Directory for replicate caches
    data_dir : str
        Directory containing input data files
    seed : int
        Base random seed
    device : str
        Compute device ('cpu' or 'cuda')

    Returns
    -------
    ExperimentConfig
        Base configuration
    """
    files_cfg = FilesConfig(
        data_dir=data_dir,
        output_dir=output_dir,
        cache_dir=cache_dir,
        cache_path="",  # Set per replicate
        use_brazil_telecom=True,
        main_data_file="smp_main.csv",
        metadata_file="metadata.csv",
        population_file="city_populations.csv",
    )

    vae_cfg = VAEConfig(
        latent_dim=32,
        hidden_dim=128,
        epochs=1500,
        batch_size=256,
        lr=1e-3,
        kl_weight=0.1,
        early_stopping_patience=200,
    )

    geo_cfg = GeoConfig(
        alpha_geo=1.0,
        min_one_per_group=True,
        constraint_mode="population_share",
        use_quota_constraints=False,
        tau_population=0.02,
        tau_municipality=0.02,
        include_geo_objective=False,
    )

    sinkhorn_cfg = SinkhornConfig(
        n_anchors=200,
        eta=0.05,
        max_iter=100,
        stop_thr=1e-6,
        anchor_method="kmeans",
    )

    mmd_cfg = MMDConfig(
        rff_dim=2000,
        bandwidth_mult=1.0,
    )

    solver_cfg = SolverConfig(
        k=300,
        crossover_prob=0.9,
        mutation_prob=0.2,
        objectives=("mmd", "sinkhorn"),
        enabled=True,
        verbose=False,
    )

    eval_cfg = EvalConfig(
        enabled=True,
        eval_size=2000,
        eval_train_frac=0.8,
    )

    baselines_cfg = BaselinesConfig(
        enabled=True,
        methods=["uniform", "kmeans", "herding", "farthest_first", "rls", "dpp", "kernel_thinning"],
    )

    return ExperimentConfig(
        run_id="R0",
        rep_id=0,
        seed=seed,
        device=device,
        files=files_cfg,
        vae=vae_cfg,
        geo=geo_cfg,
        sinkhorn=sinkhorn_cfg,
        mmd=mmd_cfg,
        solver=solver_cfg,
        eval=eval_cfg,
        baselines=baselines_cfg,
    )


def _parse_int_list(s: Optional[str]) -> Optional[List[int]]:
    if not s:
        return None
    out: List[int] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out
