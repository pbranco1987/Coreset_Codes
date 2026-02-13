"""
Phase 1 deliverable — verify all manuscript constants.

Imports every constant from config/constants.py and asserts it matches the
values specified in the manuscript:
  - Table I   (hyperparameters)
  - Table II  (run matrix / run specs)
  - Table V   (coverage targets)
  - Section VII (dataset dimensions, evaluation protocol)

Phase 6: Tests that depend on the manuscript's exact dataset dimensions
(D_FEATURES=621) are gated behind the ``STRICT_MANUSCRIPT_MODE`` environment
variable.  Set ``STRICT_MANUSCRIPT_MODE=1`` to enable them.  This allows
the repo to work with expanded CSVs that have additional categorical /
ordinal features without spurious test failures.
"""

import os

import numpy as np
import pytest

_STRICT_MANUSCRIPT = os.environ.get("STRICT_MANUSCRIPT_MODE", "0") == "1"


class TestDatasetConstants:
    """Verify dataset dimensions (Section 5.1)."""

    def test_n_municipalities(self):
        from coreset_selection.config.constants import N_MUNICIPALITIES
        assert N_MUNICIPALITIES == 5569

    def test_g_states(self):
        from coreset_selection.config.constants import G_STATES
        assert G_STATES == 27

    @pytest.mark.skipif(
        not _STRICT_MANUSCRIPT,
        reason="D_FEATURES=621 only enforced in strict manuscript mode "
               "(set STRICT_MANUSCRIPT_MODE=1 to enable)",
    )
    def test_d_features(self):
        from coreset_selection.config.constants import D_FEATURES
        assert D_FEATURES == 621


class TestCardinalityGrid:
    """Verify cardinality grid K (Table 4 / Table II)."""

    def test_k_grid(self):
        from coreset_selection.config.constants import K_GRID
        assert K_GRID == [50, 100, 200, 300, 400, 500]

    def test_k_primary_removed(self):
        """K_PRIMARY was removed — k is now user-defined via --k."""
        import coreset_selection.config.constants as c
        assert not hasattr(c, "K_PRIMARY")


class TestNSGA2Parameters:
    """Verify NSGA-II defaults (Section 5.3, Algorithm 3, Table I)."""

    def test_pop_size(self):
        from coreset_selection.config.constants import NSGA2_POP_SIZE
        assert NSGA2_POP_SIZE == 200

    def test_n_generations(self):
        from coreset_selection.config.constants import NSGA2_N_GENERATIONS
        assert NSGA2_N_GENERATIONS == 1000

    def test_crossover_prob(self):
        from coreset_selection.config.constants import NSGA2_CROSSOVER_PROB
        assert NSGA2_CROSSOVER_PROB == 0.9

    def test_mutation_prob(self):
        from coreset_selection.config.constants import NSGA2_MUTATION_PROB
        assert NSGA2_MUTATION_PROB == 0.2


class TestObjectiveParameters:
    """Verify objective function parameters (Sections 5.4–5.6, Table I)."""

    def test_rff_dim(self):
        from coreset_selection.config.constants import RFF_DIM_DEFAULT
        assert RFF_DIM_DEFAULT == 2000

    def test_sinkhorn_n_anchors(self):
        from coreset_selection.config.constants import SINKHORN_N_ANCHORS
        assert SINKHORN_N_ANCHORS == 200

    def test_sinkhorn_eta(self):
        from coreset_selection.config.constants import SINKHORN_ETA
        assert SINKHORN_ETA == 0.05

    def test_sinkhorn_max_iter(self):
        from coreset_selection.config.constants import SINKHORN_MAX_ITER
        assert SINKHORN_MAX_ITER == 100


class TestGeoParameters:
    """Verify geographic constraint parameters (Section 5.1, Table I)."""

    def test_alpha_geo(self):
        from coreset_selection.config.constants import ALPHA_GEO
        assert ALPHA_GEO == 1.0


class TestVAEParameters:
    """Verify VAE parameters (Section 5.8.1, Table I)."""

    def test_latent_dim(self):
        from coreset_selection.config.constants import VAE_LATENT_DIM
        assert VAE_LATENT_DIM == 32

    def test_epochs(self):
        from coreset_selection.config.constants import VAE_EPOCHS
        assert VAE_EPOCHS == 1500

    def test_batch_size(self):
        from coreset_selection.config.constants import VAE_BATCH_SIZE
        assert VAE_BATCH_SIZE == 256

    def test_lr(self):
        from coreset_selection.config.constants import VAE_LR
        assert VAE_LR == 1e-3


class TestPCAParameters:
    """Verify PCA parameters (Section 5.8.2)."""

    def test_pca_dim(self):
        from coreset_selection.config.constants import PCA_DIM
        assert PCA_DIM == 32

    def test_pca_config_default(self):
        from coreset_selection.config.dataclasses import PCAConfig
        cfg = PCAConfig()
        assert cfg.n_components == 32


class TestEvalParameters:
    """Verify evaluation parameters (Section 5.9)."""

    def test_eval_size(self):
        from coreset_selection.config.constants import EVAL_SIZE
        assert EVAL_SIZE == 2000

    def test_eval_train_frac(self):
        from coreset_selection.config.constants import EVAL_TRAIN_FRAC
        assert EVAL_TRAIN_FRAC == 0.8

    def test_kpca_components(self):
        from coreset_selection.config.constants import KPCA_COMPONENTS
        assert KPCA_COMPONENTS == 20


class TestCoverageTargetsTableV:
    """Verify coverage targets match manuscript Table V exactly."""

    def test_exactly_10_targets(self):
        from coreset_selection.config.constants import COVERAGE_TARGETS_TABLE_V
        assert len(COVERAGE_TARGETS_TABLE_V) == 10

    def test_table_v_keys(self):
        from coreset_selection.config.constants import COVERAGE_TARGETS_TABLE_V
        expected_keys = [
            "cov_area_4G", "cov_area_5G",
            "cov_hh_4G", "cov_res_4G",
            "cov_area_4G_5G", "cov_area_all",
            "cov_hh_4G_5G", "cov_hh_all",
            "cov_res_4G_5G", "cov_res_all",
        ]
        assert list(COVERAGE_TARGETS_TABLE_V.keys()) == expected_keys

    def test_table_v_labels(self):
        from coreset_selection.config.constants import COVERAGE_TARGETS_TABLE_V
        expected_labels = [
            "Area (4G)", "Area (5G)",
            "Households (4G)", "Residents (4G)",
            "Area (4G + 5G)", "Area (All)",
            "Households (4G + 5G)", "Households (All)",
            "Residents (4G + 5G)", "Residents (All)",
        ]
        assert list(COVERAGE_TARGETS_TABLE_V.values()) == expected_labels

    def test_coverage_target_names_order(self):
        from coreset_selection.config.constants import (
            COVERAGE_TARGET_NAMES,
            COVERAGE_TARGETS_TABLE_V,
        )
        assert COVERAGE_TARGET_NAMES == list(COVERAGE_TARGETS_TABLE_V.keys())


class TestRunSpecs:
    """Verify run specifications match manuscript Table II."""

    def test_all_run_ids_present(self):
        from coreset_selection.config.run_specs import get_run_specs
        specs = get_run_specs()
        expected = ["R0", "R1", "R2", "R3", "R4", "R5", "R6",
                    "R7", "R8", "R9", "R10", "R11", "R12"]
        assert sorted(specs.keys()) == sorted(expected)

    def test_r1_primary(self):
        from coreset_selection.config.run_specs import get_run_specs
        r1 = get_run_specs()["R1"]
        assert r1.space == "raw"
        assert r1.objectives == ("mmd", "sinkhorn")
        assert r1.constraint_mode == "population_share"
        assert r1.sweep_k is not None
        assert list(r1.sweep_k) == [50, 100, 200, 300, 400, 500]
        assert r1.n_reps == 1
        assert r1.n_reps_by_k == {300: 5}
        assert r1.get_n_reps_for_k(300) == 5
        assert r1.get_n_reps_for_k(50) == 1
        assert r1.get_n_reps_for_k(500) == 1
        assert r1.max_n_reps == 5

    def test_r5_joint(self):
        from coreset_selection.config.run_specs import get_run_specs
        r5 = get_run_specs()["R5"]
        assert r5.constraint_mode == "joint"
        assert r5.n_reps == 1
        assert r5.objectives == ("mmd", "sinkhorn")

    def test_r7_skl_vae(self):
        from coreset_selection.config.run_specs import get_run_specs
        r7 = get_run_specs()["R7"]
        assert r7.space == "vae"
        assert r7.objectives == ("mmd", "sinkhorn", "skl")

    def test_r6_no_constraints(self):
        from coreset_selection.config.run_specs import get_run_specs
        r6 = get_run_specs()["R6"]
        assert r6.constraint_mode == "none"

    def test_r4_municipality_share_quota(self):
        from coreset_selection.config.run_specs import get_run_specs
        r4 = get_run_specs()["R4"]
        assert r4.constraint_mode == "municipality_share_quota"

    def test_r8_pca(self):
        from coreset_selection.config.run_specs import get_run_specs
        r8 = get_run_specs()["R8"]
        assert r8.space == "pca"
        assert r8.requires_pca is True

    def test_r9_vae(self):
        from coreset_selection.config.run_specs import get_run_specs
        r9 = get_run_specs()["R9"]
        assert r9.space == "vae"
        assert r9.requires_vae is True


class TestGeoConfigDefaults:
    """Verify GeoConfig dataclass fields."""

    def test_geo_config_fields(self):
        from coreset_selection.config.dataclasses import GeoConfig
        cfg = GeoConfig()
        assert hasattr(cfg, "constraint_mode")
        assert hasattr(cfg, "alpha_geo")
        assert hasattr(cfg, "tau_population")
        assert hasattr(cfg, "tau_municipality")
        assert cfg.alpha_geo == 1.0
        assert cfg.tau_population == 0.02
        assert cfg.tau_municipality == 0.02


class TestEffortGrid:
    """Verify effort sweep grid matches the plan."""

    def test_effort_grid_constant(self):
        from coreset_selection.config.constants import EFFORT_GRID
        assert len(EFFORT_GRID) == 7
        assert EFFORT_GRID[0] == {"pop_size": 20, "n_gen": 100}
        assert EFFORT_GRID[-1] == {"pop_size": 400, "n_gen": 2000}

    def test_effort_sweep_grid_dataclass(self):
        from coreset_selection.config.dataclasses import EffortSweepGrid
        g = EffortSweepGrid()
        grid = g.grid()
        # Cartesian product: 4 pop_sizes × 3 n_gens = 12 (default)
        assert len(grid) == 12
        # Verify first and last entries
        assert grid[0] == (50, 100)
        assert grid[-1] == (200, 700)


class TestSolverConfigDefaults:
    """Verify SolverConfig defaults match Table I."""

    def test_defaults(self):
        from coreset_selection.config.dataclasses import SolverConfig
        cfg = SolverConfig()
        assert cfg.pop_size == 200
        assert cfg.n_gen == 1000
        assert cfg.crossover_prob == 0.9
        assert cfg.mutation_prob == 0.2
        assert cfg.objectives == ("mmd", "sinkhorn")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
