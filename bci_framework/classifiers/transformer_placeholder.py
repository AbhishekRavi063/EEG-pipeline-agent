"""Placeholder for Transformer-based classifier."""

import numpy as np

from .base import ClassifierBase


class TransformerClassifier(ClassifierBase):
    """Placeholder: uses LDA under the hood until Transformer is implemented."""

    name = "transformer"

    def __init__(self, n_classes: int = 4, **kwargs: object) -> None:
        super().__init__(n_classes=n_classes, **kwargs)
        self._fallback = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "TransformerClassifier":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        self._fallback = LinearDiscriminantAnalysis()
        if sample_weight is not None:
            self._fallback.fit(X, y, sample_weight=sample_weight)
        else:
            self._fallback.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._fallback is None:
            raise RuntimeError("Transformer placeholder not fitted")
        return self._fallback.predict(X).astype(np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._fallback is None:
            raise RuntimeError("Transformer placeholder not fitted")
        return self._fallback.predict_proba(X).astype(np.float64)
