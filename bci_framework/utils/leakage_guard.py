"""
Leakage guard: validate train/test splits before training to prevent data leakage.

Ensures:
- Preprocessing (CSP, scaling, covariance, ICA, domain adaptation) is fitted ONLY on training data.
- LOSO: test subject is completely isolated from training.
- Cross-session: sessions are separated before any fitting.
- No overlap between train and test indices.

Use assert_no_leakage_split() before fitting pipelines in evaluation code.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def assert_no_overlap(
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    context: str = "",
) -> None:
    """
    Assert train and test indices do not overlap.
    Raises AssertionError if any index appears in both.
    """
    train_set = set(np.asarray(train_indices).ravel().tolist())
    test_set = set(np.asarray(test_indices).ravel().tolist())
    overlap = train_set & test_set
    assert len(overlap) == 0, (
        f"[LEAKAGE] Train and test indices must not overlap. "
        f"Overlap size: {len(overlap)}. {context}"
    )
    logger.debug("assert_no_overlap OK (train=%d, test=%d) %s", len(train_set), len(test_set), context)


def assert_loso_isolated(
    train_subject_ids: np.ndarray,
    test_subject_ids: np.ndarray,
    context: str = "",
) -> None:
    """
    Assert LOSO: no test subject appears in training.
    train_subject_ids and test_subject_ids are 1D arrays of subject id per trial.
    """
    train_subjects = set(np.unique(train_subject_ids).tolist())
    test_subjects = set(np.unique(test_subject_ids).tolist())
    leaking = test_subjects & train_subjects
    assert len(leaking) == 0, (
        f"[LEAKAGE] LOSO violated: test subject(s) {leaking} appear in training. "
        f"{context}"
    )
    logger.debug(
        "assert_loso_isolated OK (train_subjects=%s, test_subjects=%s) %s",
        sorted(train_subjects),
        sorted(test_subjects),
        context,
    )


def assert_cross_session_respected(
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    session_boundary_index: int,
    context: str = "",
) -> None:
    """
    Assert cross-session split: all train indices < session_boundary_index,
    all test indices >= session_boundary_index (or vice versa depending on convention).
    Convention: train = [0, session_boundary_index), test = [session_boundary_index, n_trials).
    """
    train_idx = np.asarray(train_indices).ravel()
    test_idx = np.asarray(test_indices).ravel()
    assert np.all(train_idx < session_boundary_index), (
        f"[LEAKAGE] Cross-session: all train indices must be < {session_boundary_index}. "
        f"Max train index: {int(np.max(train_idx))}. {context}"
    )
    assert np.all(test_idx >= session_boundary_index), (
        f"[LEAKAGE] Cross-session: all test indices must be >= {session_boundary_index}. "
        f"Min test index: {int(np.min(test_idx))}. {context}"
    )
    logger.debug(
        "assert_cross_session_respected OK (boundary=%d) %s",
        session_boundary_index,
        context,
    )


def assert_no_fit_on_full_dataset(
    n_train: int,
    n_test: int,
    n_total_trials: int,
    context: str = "",
) -> None:
    """
    Assert that fit is not being called on the full dataset.
    When there is a test set (n_test > 0), n_train must be < n_total_trials.
    """
    assert n_train <= n_total_trials and n_test <= n_total_trials, (
        f"[LEAKAGE] Train or test size exceeds total trials. "
        f"n_train={n_train}, n_test={n_test}, n_total={n_total_trials}. {context}"
    )
    if n_test > 0:
        assert n_train < n_total_trials, (
            f"[LEAKAGE] Fitting must not use all trials when test set exists. "
            f"n_train={n_train} must be < n_total={n_total_trials}. {context}"
        )
    logger.debug("assert_no_fit_on_full_dataset OK (train=%d, test=%d, total=%d) %s", n_train, n_test, n_total_trials, context)


def assert_no_leakage_split(
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    subject_ids: np.ndarray | None = None,
    evaluation_mode: str = "subject_wise",
    cross_session_boundary: int | None = None,
    context: str = "",
) -> None:
    """
    Single entry point: validate that the split is leakage-safe.

    - Always checks train/test index overlap.
    - If subject_ids is provided and evaluation_mode in ("loso", "cross_subject"),
      checks that no test subject appears in training.
    - If cross_session_boundary is provided, checks session boundary.

    Raises AssertionError if any check fails.
    """
    assert_no_overlap(train_indices, test_indices, context=context)
    if subject_ids is not None and evaluation_mode in ("loso", "cross_subject"):
        subj = np.asarray(subject_ids).ravel()
        train_subj = subj[train_indices]
        test_subj = subj[test_indices]
        assert_loso_isolated(train_subj, test_subj, context=context)
    if cross_session_boundary is not None:
        assert_cross_session_respected(
            train_indices,
            test_indices,
            cross_session_boundary,
            context=context,
        )
    logger.debug("assert_no_leakage_split passed %s", context)


def validate_pipeline_fit_contract(
    X_fit: np.ndarray,
    X_predict: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> None:
    """
    Optional runtime check: ensure the data passed to fit() matches train indices
    and the data passed to predict() matches test indices (by shape).
    Caller must pass X_fit = X[train_indices], X_predict = X[test_indices].
    """
    assert X_fit.shape[0] == len(train_indices), (
        f"[LEAKAGE] fit() data size must match train indices: "
        f"X_fit.shape[0]={X_fit.shape[0]} vs len(train_indices)={len(train_indices)}"
    )
    assert X_predict.shape[0] == len(test_indices), (
        f"[LEAKAGE] predict() data size must match test indices: "
        f"X_predict.shape[0]={X_predict.shape[0]} vs len(test_indices)={len(test_indices)}"
    )
