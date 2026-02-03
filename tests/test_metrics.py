"""Unit tests for BCI metrics (accuracy, kappa, ITR, F1, ROC-AUC)."""

import numpy as np
import pytest

from bci_framework.utils.metrics import (
    accuracy,
    cohen_kappa,
    f1_macro,
    itr_bits_per_trial,
    itr_bits_per_minute,
    compute_all_metrics,
)


def test_accuracy():
    y_true = np.array([0, 1, 2, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0])
    assert accuracy(y_true, y_pred) == 0.8


def test_cohen_kappa():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    k = cohen_kappa(y_true, y_pred, n_classes=2)
    assert 0 <= k <= 1


def test_itr_bits_per_trial():
    bpt = itr_bits_per_trial(0.8, n_classes=4, trial_duration_sec=3.0)
    assert bpt >= 0
    assert itr_bits_per_trial(0.25, 4, 3.0) == 0.0
    assert itr_bits_per_trial(1.0, 4, 3.0) >= 0  # perfect accuracy => log2(n_classes) or 0


def test_itr_bits_per_minute():
    bpm = itr_bits_per_minute(0.8, n_classes=4, trial_duration_sec=3.0)
    assert bpm >= 0


def test_compute_all_metrics():
    n = 20
    y_true = np.random.randint(0, 4, size=n)
    y_pred = y_true.copy()
    y_pred[::4] = (y_pred[::4] + 1) % 4
    y_proba = np.eye(4)[y_pred]
    out = compute_all_metrics(y_true, y_pred, y_proba, n_classes=4, trial_duration_sec=3.0)
    assert "accuracy" in out
    assert "kappa" in out
    assert "f1_macro" in out
    assert "itr_bits_per_trial" in out
    assert "roc_auc_macro" in out
