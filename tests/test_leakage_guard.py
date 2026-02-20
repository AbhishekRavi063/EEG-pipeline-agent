"""
Unit tests for leakage guard: validate that train/test splits are leakage-safe.

Ensures:
- No overlap between train and test indices
- LOSO: test subject not in training
- Cross-session: session boundary respected
"""

import pytest
import numpy as np

from bci_framework.utils.leakage_guard import (
    assert_no_overlap,
    assert_loso_isolated,
    assert_cross_session_respected,
    assert_no_leakage_split,
    assert_no_fit_on_full_dataset,
)


def test_assert_no_overlap_pass():
    train = np.array([0, 1, 2])
    test = np.array([3, 4, 5])
    assert_no_overlap(train, test)


def test_assert_no_overlap_fail():
    train = np.array([0, 1, 2])
    test = np.array([2, 3, 4])  # 2 overlaps
    with pytest.raises(AssertionError, match="must not overlap"):
        assert_no_overlap(train, test)


def test_assert_loso_isolated_pass():
    train_subj = np.array([1, 1, 2, 2, 3, 3])
    test_subj = np.array([4, 4, 4])
    assert_loso_isolated(train_subj, test_subj)


def test_assert_loso_isolated_fail():
    train_subj = np.array([1, 1, 2, 2])
    test_subj = np.array([2, 2])  # subject 2 in both
    with pytest.raises(AssertionError, match="LOSO violated"):
        assert_loso_isolated(train_subj, test_subj)


def test_assert_cross_session_respected_pass():
    train_idx = np.array([0, 1, 2])
    test_idx = np.array([3, 4, 5])
    boundary = 3
    assert_cross_session_respected(train_idx, test_idx, boundary)


def test_assert_cross_session_respected_fail_train():
    train_idx = np.array([0, 1, 3])  # 3 >= boundary
    test_idx = np.array([3, 4])
    boundary = 3
    with pytest.raises(AssertionError, match="Cross-session"):
        assert_cross_session_respected(train_idx, test_idx, boundary)


def test_assert_no_leakage_split_loso_pass():
    subject_ids = np.array([1, 1, 2, 2, 3, 3])
    train_idx = np.array([0, 1, 2, 3])  # subjects 1, 2
    test_idx = np.array([4, 5])         # subject 3
    assert_no_leakage_split(
        train_idx, test_idx,
        subject_ids=subject_ids,
        evaluation_mode="loso",
    )


def test_assert_no_leakage_split_loso_fail():
    subject_ids = np.array([1, 1, 2, 2])
    train_idx = np.array([0, 1, 2])
    test_idx = np.array([3])  # subject 2 in test but also in train (index 2)
    with pytest.raises(AssertionError):
        assert_no_leakage_split(
            train_idx, test_idx,
            subject_ids=subject_ids,
            evaluation_mode="loso",
        )


def test_assert_no_leakage_split_overlap_fail():
    train_idx = np.array([0, 1])
    test_idx = np.array([1, 2])
    with pytest.raises(AssertionError, match="overlap"):
        assert_no_leakage_split(train_idx, test_idx)


def test_assert_no_fit_on_full_dataset_pass():
    assert_no_fit_on_full_dataset(n_train=80, n_test=20, n_total_trials=100)


def test_assert_no_fit_on_full_dataset_fail():
    # Fitting on all 100 trials while claiming 20 test trials = leakage
    with pytest.raises(AssertionError):
        assert_no_fit_on_full_dataset(n_train=100, n_test=20, n_total_trials=100)


def test_ea_alignment_target_not_in_fit():
    """Euclidean Alignment must not use target subject in alignment computation (LOSO)."""
    from bci_framework.features.euclidean_alignment import EARiemannTangentOAS

    np.random.seed(42)
    n_trials, n_ch, n_samp = 20, 4, 100
    X = np.random.randn(n_trials, n_ch, n_samp).astype(np.float64)
    # Simulate: subject 1 = target (holdout), subject 2 = source. If we pass subject_ids that include target, assert.
    subject_ids = np.array([1] * 10 + [2] * 10)  # 10 trials from subject 1 (target), 10 from 2
    fe = EARiemannTangentOAS(fs=250.0)
    with pytest.raises(AssertionError, match="must not use target subject"):
        fe.fit(X, np.zeros(n_trials, dtype=np.int64), subject_ids=subject_ids, loso_target_subject=1)
