"""
Preprocessing tests — data loading, split persistence, and target column removal.

Split from tests/test_preprocessing.py.
"""

import os
import tempfile

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Phase 4.2: Split persistence and validation
# ---------------------------------------------------------------------------

class TestSplitPersistence:
    """Verify splits can be saved, loaded, and validated."""

    @staticmethod
    def _make_splits(N=500, G=5, seed=42):
        rng = np.random.default_rng(seed)
        group_ids = rng.integers(0, G, size=N)
        all_idx = np.arange(N)
        rng.shuffle(all_idx)

        # 80/20 split
        split = int(0.8 * N)
        train_idx = np.sort(all_idx[:split])
        val_idx = np.sort(all_idx[split:])

        # Eval set: 100 points
        eval_idx = np.sort(rng.choice(N, size=100, replace=False))
        rng2 = np.random.default_rng(seed + 1)
        rng2.shuffle(eval_idx)
        e_split = int(0.8 * len(eval_idx))
        eval_train_idx = np.sort(eval_idx[:e_split])
        eval_test_idx = np.sort(eval_idx[e_split:])

        return train_idx, val_idx, eval_idx, eval_train_idx, eval_test_idx, group_ids

    def test_save_and_load_roundtrip(self):
        """Saved splits should load back identically."""
        from coreset_selection.data.split_persistence import save_splits, load_splits

        train, val, ev, ev_tr, ev_te, _ = self._make_splits()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_splits(
                tmpdir, rep_id=0,
                train_idx=train, val_idx=val,
                eval_idx=ev, eval_train_idx=ev_tr, eval_test_idx=ev_te,
                seed=42,
            )
            loaded = load_splits(tmpdir, rep_id=0)

        np.testing.assert_array_equal(loaded["train_idx"], train)
        np.testing.assert_array_equal(loaded["val_idx"], val)
        np.testing.assert_array_equal(loaded["eval_idx"], ev)
        np.testing.assert_array_equal(loaded["eval_train_idx"], ev_tr)
        np.testing.assert_array_equal(loaded["eval_test_idx"], ev_te)
        assert loaded["seed"][0] == 42

    def test_rerun_same_seed_same_indices(self):
        """Re-running with the same seed must produce identical splits."""
        t1, v1, e1, et1, ete1, _ = self._make_splits(seed=99)
        t2, v2, e2, et2, ete2, _ = self._make_splits(seed=99)

        np.testing.assert_array_equal(t1, t2)
        np.testing.assert_array_equal(v1, v2)
        np.testing.assert_array_equal(e1, e2)

    def test_validate_splits_pass(self):
        """Validation should pass for correctly constructed splits."""
        from coreset_selection.data.split_persistence import validate_splits

        N = 500
        train, val, ev, ev_tr, ev_te, gids = self._make_splits(N=N)

        results = validate_splits(
            train, val, ev, ev_tr, ev_te, N=N,
            expected_eval_size=100,
            group_ids=gids,
        )

        assert results["train_val_disjoint"]
        assert results["indices_in_bounds"]
        assert results["eval_subsplit_disjoint"]

    def test_validate_splits_detect_overlap(self):
        """Validation should fail when train and val overlap."""
        from coreset_selection.data.split_persistence import validate_splits

        N = 100
        train = np.arange(80)
        val = np.arange(75, 100)  # overlaps with train at [75..79]
        ev = np.arange(50, 60)
        ev_tr = np.arange(50, 58)
        ev_te = np.arange(58, 60)

        results = validate_splits(train, val, ev, ev_tr, ev_te, N=N, expected_eval_size=10)

        assert not results["train_val_disjoint"]


# ---------------------------------------------------------------------------
# Phase 4/6: Explicit target column removal tests
# ---------------------------------------------------------------------------

class TestExplicitTargetRemoval:
    """Phase 4/6: remove_target_columns with explicit_targets parameter."""

    def test_explicit_targets_removed(self):
        """Explicitly declared target columns should be removed."""
        from coreset_selection.data.target_columns import remove_target_columns

        X = np.random.default_rng(0).normal(size=(10, 4))
        names = ["feat_a", "my_target", "feat_b", "feat_c"]

        X_clean, kept, removed = remove_target_columns(
            X, names, explicit_targets=["my_target"],
        )
        assert "my_target" in removed
        assert "my_target" not in kept
        assert X_clean.shape[1] == 3

    def test_explicit_targets_union_with_regex(self):
        """Explicit targets should be unioned with regex-detected targets."""
        from coreset_selection.data.target_columns import remove_target_columns

        X = np.random.default_rng(0).normal(size=(10, 5))
        names = ["feat_a", "target", "feat_b", "my_custom_y", "feat_c"]
        # "target" matches regex; "my_custom_y" is explicit
        X_clean, kept, removed = remove_target_columns(
            X, names, explicit_targets=["my_custom_y"],
        )
        assert "target" in removed       # regex-detected
        assert "my_custom_y" in removed   # explicit
        assert X_clean.shape[1] == 3

    def test_no_explicit_targets_backward_compatible(self):
        """Without explicit_targets, behavior is identical to before."""
        from coreset_selection.data.target_columns import remove_target_columns

        X = np.random.default_rng(0).normal(size=(10, 3))
        names = ["feat_a", "feat_b", "feat_c"]

        X_clean, kept, removed = remove_target_columns(X, names)
        assert removed == []
        assert X_clean.shape[1] == 3
