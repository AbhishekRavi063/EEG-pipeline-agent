"""Common Average Reference (CAR) spatial filter."""

from __future__ import annotations

import numpy as np

from .base import SpatialFilterBase


class CARSpatialFilter(SpatialFilterBase):
    """CAR: subtract mean across channels. Online-safe, no fit required."""

    name = "car"

    def fit(
        self, X: np.ndarray, y: np.ndarray | None = None, info: dict | None = None
    ) -> "CARSpatialFilter":
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        data = np.asarray(X, dtype=np.float64)
        # (n_trials, n_channels, n_samples) -> mean over axis=1 (channels)
        mean = data.mean(axis=1, keepdims=True)
        return data - mean
