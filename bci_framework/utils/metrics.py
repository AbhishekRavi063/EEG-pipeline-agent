"""
Advanced BCI metrics: Accuracy, Kappa, ITR, F1, ROC-AUC, Mutual Information.
"""

from typing import Any

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def cohen_kappa(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    try:
        from sklearn.metrics import cohen_kappa_score
        return float(cohen_kappa_score(y_true, y_pred, labels=list(range(n_classes))))
    except Exception:
        return 0.0


def f1_macro(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    try:
        from sklearn.metrics import f1_score
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0, labels=list(range(n_classes))))
    except Exception:
        return 0.0


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Per-class recall averaged (macro); robust to class imbalance."""
    try:
        from sklearn.metrics import balanced_accuracy_score
        return float(balanced_accuracy_score(y_true, y_pred, adjusted=False))
    except Exception:
        return 0.0


def roc_auc_ovr(y_true: np.ndarray, y_proba: np.ndarray, n_classes: int) -> float:
    try:
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        return float(roc_auc_score(y_bin, y_proba, average="macro", multi_class="ovr"))
    except Exception:
        return 0.0


def itr_bits_per_trial(
    accuracy_rate: float,
    n_classes: int,
    trial_duration_sec: float,
) -> float:
    """
    Information Transfer Rate in bits per trial (Wolpaw et al.).
    accuracy_rate: 0..1
    n_classes: number of classes
    trial_duration_sec: duration of one trial in seconds
    """
    if accuracy_rate <= 0 or n_classes < 2:
        return 0.0
    if accuracy_rate >= 1.0:
        return float(np.log2(n_classes))
    p = accuracy_rate
    b = np.log2(n_classes) + p * np.log2(p) + (1 - p) * np.log2((1 - p) / (n_classes - 1))
    return max(0.0, float(b))


def itr_bits_per_minute(
    accuracy_rate: float,
    n_classes: int,
    trial_duration_sec: float,
) -> float:
    """ITR in bits per minute (60 / trial_duration_sec * bits_per_trial)."""
    bpt = itr_bits_per_trial(accuracy_rate, n_classes, trial_duration_sec)
    return float(bpt * 60.0 / trial_duration_sec) if trial_duration_sec > 0 else 0.0


def mutual_information(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    try:
        from sklearn.metrics import mutual_info_score
        return float(mutual_info_score(y_true, y_pred))
    except Exception:
        return 0.0


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    n_classes: int,
    trial_duration_sec: float = 3.0,
) -> dict[str, float]:
    """Compute accuracy, kappa, F1, ROC-AUC, ITR, mutual information."""
    acc = accuracy(y_true, y_pred)
    out = {
        "accuracy": acc,
        "balanced_accuracy": balanced_accuracy(y_true, y_pred, n_classes),
        "kappa": cohen_kappa(y_true, y_pred, n_classes),
        "f1_macro": f1_macro(y_true, y_pred, n_classes),
        "itr_bits_per_trial": itr_bits_per_trial(acc, n_classes, trial_duration_sec),
        "itr_bits_per_minute": itr_bits_per_minute(acc, n_classes, trial_duration_sec),
        "mutual_information": mutual_information(y_true, y_pred, n_classes),
    }
    if y_proba is not None and y_proba.shape[1] >= n_classes:
        out["roc_auc_macro"] = roc_auc_ovr(y_true, y_proba, n_classes)
    else:
        out["roc_auc_macro"] = 0.0
    return out
