"""Base class for feature extractors. All return standardized feature vectors."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class FeatureExtractorBase(ABC):
    """Abstract base for EEG feature extraction. Output: (n_trials, n_features)."""

    name: str = "base"

    def __init__(self, fs: float, **kwargs: Any) -> None:
        self.fs = fs
        self.params = kwargs
        self._fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "FeatureExtractorBase":
        """Fit extractor on (n_trials, n_channels, n_samples), labels (n_trials,)."""
        return self

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform to (n_trials, n_features)."""
        pass

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and transform."""
        self.fit(X, y)
        return self.transform(X)

    @property
    def n_features_out(self) -> int | None:
        """Number of output features if known (e.g. after fit)."""
        return None
