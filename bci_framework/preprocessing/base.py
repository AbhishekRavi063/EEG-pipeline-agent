"""Base class for preprocessing steps. All preprocessing methods inherit from this."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class PreprocessingBase(ABC):
    """Abstract base for EEG preprocessing. Steps can be stacked in sequence."""

    name: str = "base"

    def __init__(self, fs: float, **kwargs: Any) -> None:
        self.fs = fs
        self.params = kwargs
        self._fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "PreprocessingBase":
        """Fit preprocessing to data if needed (e.g. ICA). X: (n_trials, n_channels, n_samples)."""
        return self

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply preprocessing. X: (n_trials, n_channels, n_samples) -> same shape."""
        pass

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """Fit and transform."""
        self.fit(X, y)
        return self.transform(X)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(fs={self.fs}, params={self.params})"


class AdvancedPreprocessingBase(PreprocessingBase):
    """Specialised base for optional / research-grade preprocessing."""

    supports_online: bool = True

    def __init__(self, fs: float, **kwargs: Any) -> None:
        super().__init__(fs, **kwargs)
        self.supports_online = bool(kwargs.pop("supports_online", self.supports_online))

    def is_online_supported(self) -> bool:
        """Return True if the step can run in online/real-time mode."""
        return self.supports_online
