"""Z-score (subject-wise) domain adapter: re-center source to target statistics."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .base_adapter import DomainAdapter

logger = logging.getLogger(__name__)


class ZScoreAdapter(DomainAdapter):
    """
    Subject-wise normalization: compute mean/std of source;
    if X_target provided, re-center source to target mean/std.
    """

    name = "zscore"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._source_mean: np.ndarray | None = None
        self._source_std: np.ndarray | None = None
        self._target_mean: np.ndarray | None = None
        self._target_std: np.ndarray | None = None
        self._use_target_stats = False

    def fit(
        self,
        X_source: np.ndarray,
        X_target: np.ndarray | None = None,
    ) -> "ZScoreAdapter":
        logger.info("[TRANSFER] Adapter fit called")
        self._validate_input(X_source)
        X_source = np.asarray(X_source, dtype=np.float64)
        self._source_mean = np.mean(X_source, axis=0)
        self._source_std = np.std(X_source, axis=0)
        self._source_std[self._source_std < 1e-10] = 1.0

        self._use_target_stats = False
        if X_target is not None and len(X_target) > 0:
            self._validate_input(X_target)
            X_target = np.asarray(X_target, dtype=np.float64)
            self._target_mean = np.mean(X_target, axis=0)
            self._target_std = np.std(X_target, axis=0)
            self._target_std[self._target_std < 1e-10] = 1.0
            self._use_target_stats = True

        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._validate_input(X)
        if not self._fitted or self._source_mean is None:
            return np.asarray(X, dtype=np.float64)
        X = np.asarray(X, dtype=np.float64)
        if self._use_target_stats and self._target_mean is not None and self._target_std is not None:
            # Normalize to source scale then re-apply target scale
            z = (X - self._source_mean) / self._source_std
            return z * self._target_std + self._target_mean
        # Standard z-score (source stats only)
        return (X - self._source_mean) / self._source_std
