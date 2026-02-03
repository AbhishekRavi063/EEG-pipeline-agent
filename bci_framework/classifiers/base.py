"""Base class for classifiers. Unified API: fit, predict, predict_proba."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class ClassifierBase(ABC):
    """Abstract base for classifiers. fit(X, y), predict(X), predict_proba(X)."""

    name: str = "base"

    def __init__(self, n_classes: int = 4, **kwargs: Any) -> None:
        self.n_classes = n_classes
        self.params = kwargs

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "ClassifierBase":
        """Fit on features (n_samples, n_features) and labels (n_samples,)."""
        return self

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class indices (n_samples,)."""
        pass

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities (n_samples, n_classes). Default: one-hot from predict."""
        pred = self.predict(X)
        proba = np.zeros((len(pred), self.n_classes), dtype=np.float64)
        for i, p in enumerate(pred):
            proba[i, int(p)] = 1.0
        return proba
