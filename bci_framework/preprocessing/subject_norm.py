"""Subject-wise standardization for cross-subject pipelines.

Apply immediately after loading each subject so that normalization
is per subject (no leakage across subjects).
"""

from __future__ import annotations

import numpy as np


def subject_standardize(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Standardize per subject across trials.
    X: shape (trials, channels, samples).
    Uses mean and std over (trials, samples) per channel.
    """
    X = np.asarray(X, dtype=np.float64)
    mean = X.mean(axis=(0, 2), keepdims=True)
    std = X.std(axis=(0, 2), keepdims=True) + eps
    return ((X - mean) / std).astype(np.float64)


def subject_standardize_per_subject(
    X: np.ndarray,
    subject_ids: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Standardize each subject's trials separately (no cross-subject leakage).
    X: (trials, channels, samples), subject_ids: (trials,).
    Returns X with each subject block standardized.
    """
    X = np.asarray(X, dtype=np.float64)
    out = X.copy()
    for sid in np.unique(subject_ids):
        mask = subject_ids == sid
        if np.sum(mask) == 0:
            continue
        out[mask] = subject_standardize(X[mask], eps=eps)
    return out
