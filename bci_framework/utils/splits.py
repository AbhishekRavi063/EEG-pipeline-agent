"""
Trial-wise splitting and evaluation protocols.
No sample-level split; no preprocessing/CSP fitting on test data.
Supports subject-wise and leave-one-subject-out (LOSO).
"""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def trial_train_test_split(
    n_trials: int,
    train_ratio: float = 0.8,
    shuffle: bool = True,
    random_state: int | None = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split at trial level (not sample level). Returns train_indices, test_indices.
    Use these to index trials only; never fit preprocessing/features on test indices.
    """
    rng = np.random.default_rng(random_state)
    indices = np.arange(n_trials)
    if shuffle:
        indices = rng.permutation(indices)
    n_train = int(n_trials * train_ratio)
    n_train = max(1, min(n_train, n_trials - 1))
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    return train_idx, test_idx


def subject_to_trial_indices(
    subject_ids: np.ndarray,
    subject_id_holdout: int | str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For LOSO: train = all subjects except holdout; test = holdout subject only.
    subject_ids: (n_trials,) array of subject id per trial.
    subject_id_holdout: which subject to leave out for test.
    Returns (train_indices, test_indices).
    """
    if subject_id_holdout is None:
        return np.arange(len(subject_ids)), np.array([], dtype=np.int64)
    train_mask = subject_ids != subject_id_holdout
    test_mask = ~train_mask
    return np.where(train_mask)[0], np.where(test_mask)[0]


def loso_splits(
    subject_ids: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Leave-one-subject-out: one list entry per subject as test, rest as train.
    Each element is (train_indices, test_indices).
    """
    unique = np.unique(subject_ids)
    return [
        subject_to_trial_indices(subject_ids, holdout)
        for holdout in unique
    ]


def sequential_calibration_split(
    n_trials: int,
    n_calibration: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    First N trials = calibration (pipeline selection); rest = live stream.
    Like real BCI: initial segment for calibration, then stream with selected pipeline.
    Returns (train_indices, test_indices) with train = [0..n_cal-1], test = [n_cal..n_trials-1].
    """
    n_cal = max(1, min(n_calibration, n_trials - 1))
    train_idx = np.arange(0, n_cal)
    test_idx = np.arange(n_cal, n_trials)
    return train_idx, test_idx


def get_train_test_trials(
    n_trials: int,
    subject_ids: np.ndarray | None = None,
    evaluation_mode: str = "subject_wise",
    train_ratio: float = 0.8,
    loso_subject: int | str | None = None,
    random_state: int | None = 42,
    split_mode: str = "train_test",
    n_calibration_trials: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get train and test trial indices with no leakage.
    - split_mode "sequential": first n_calibration_trials = calibration, rest = live stream (no shuffle).
    - split_mode "train_test" (default): use evaluation_mode and train_ratio.
    - evaluation_mode "subject_wise": trial-wise split within subject (shuffle, then 80/20).
    - evaluation_mode "cross_subject" / "loso": use loso_subject as holdout.
    - No preprocessing or CSP fitting on test_indices.
    """
    if split_mode == "sequential":
        return sequential_calibration_split(n_trials, n_calibration_trials)
    if subject_ids is not None and loso_subject is not None:
        return subject_to_trial_indices(subject_ids, loso_subject)
    if subject_ids is not None and evaluation_mode in ("cross_subject", "loso"):
        unique = np.unique(subject_ids)
        if len(unique) < 2:
            return trial_train_test_split(n_trials, train_ratio, True, random_state)
        return subject_to_trial_indices(subject_ids, unique[0])
    return trial_train_test_split(n_trials, train_ratio, True, random_state)
