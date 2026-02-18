"""
Base interface for spatial filter plugins.

Spatial filters operate on (n_trials, n_channels, n_samples) and produce
(n_trials, n_out_channels, n_samples). They are part of the modular pipeline
and support both offline (fit/transform) and online (transform only with precomputed state).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class SpatialFilterBase(ABC):
    """
    Abstract base for spatial filtering (CAR, Laplacian, CSP, GeDai/lead-field).

    Plugins are registered in SPATIAL_FILTER_REGISTRY and selected via config
    (spatial_filter.method). Online mode requires precomputed filters; transform
    must be causal (no future samples).
    """

    name: str = "base"

    def __init__(self, fs: float, **kwargs: Any) -> None:
        self.fs = fs
        self.params = kwargs
        self._fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray | None = None, info: dict[str, Any] | None = None) -> "SpatialFilterBase":
        """
        Fit filter from data. X: (n_trials, n_channels, n_samples).
        info: optional dict with channel_names, montage, etc.
        """
        return self

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply spatial filter. X: (n_trials, n_channels, n_samples).
        Returns (n_trials, n_out_channels, n_samples). Must be causal for online.
        """
        pass

    def fit_transform(
        self, X: np.ndarray, y: np.ndarray | None = None, info: dict[str, Any] | None = None
    ) -> np.ndarray:
        """Fit and transform."""
        self.fit(X, y=y, info=info)
        return self.transform(X)

    def is_online_safe(self) -> bool:
        """True if transform is causal and can run in real-time (matrix mult only)."""
        return True

    def set_channel_names(self, channel_names: list[str]) -> None:
        """Optional: set channel names for montage-dependent filters (Laplacian, GeDai)."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(fs={self.fs}, name={self.name!r})"
