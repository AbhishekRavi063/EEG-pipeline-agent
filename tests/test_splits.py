"""Unit tests for trial-wise splits and LOSO (no leakage)."""

import numpy as np
import pytest

from bci_framework.utils.splits import (
    trial_train_test_split,
    subject_to_trial_indices,
    loso_splits,
    get_train_test_trials,
)


def test_trial_train_test_split():
    train_idx, test_idx = trial_train_test_split(100, train_ratio=0.8, shuffle=True, random_state=42)
    assert len(train_idx) == 80
    assert len(test_idx) == 20
    assert len(np.unique(np.concatenate([train_idx, test_idx]))) == 100
    assert len(set(train_idx) & set(test_idx)) == 0


def test_trial_train_test_split_no_shuffle():
    train_idx, test_idx = trial_train_test_split(100, train_ratio=0.8, shuffle=False)
    np.testing.assert_array_equal(train_idx, np.arange(80))
    np.testing.assert_array_equal(test_idx, np.arange(80, 100))


def test_subject_to_trial_indices():
    subject_ids = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
    train_idx, test_idx = subject_to_trial_indices(subject_ids, subject_id_holdout=2)
    np.testing.assert_array_equal(test_idx, np.array([3, 4, 5]))
    assert set(train_idx) == {0, 1, 2, 6, 7, 8}


def test_loso_splits():
    subject_ids = np.array([1, 1, 2, 2, 3, 3])
    splits = loso_splits(subject_ids)
    assert len(splits) == 3
    for train_idx, test_idx in splits:
        assert len(train_idx) + len(test_idx) == 6
        assert len(set(train_idx) & set(test_idx)) == 0


def test_get_train_test_trials_subject_wise():
    train_idx, test_idx = get_train_test_trials(
        100,
        subject_ids=None,
        evaluation_mode="subject_wise",
        train_ratio=0.8,
        random_state=42,
    )
    assert len(train_idx) == 80 and len(test_idx) == 20


def test_get_train_test_trials_loso():
    subject_ids = np.repeat([1, 2, 3], 10)
    train_idx, test_idx = get_train_test_trials(
        30,
        subject_ids=subject_ids,
        evaluation_mode="loso",
        loso_subject=None,
        random_state=42,
    )
    assert len(test_idx) == 10
    assert len(train_idx) == 20
