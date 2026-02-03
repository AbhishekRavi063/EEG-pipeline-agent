"""LDA classifier."""

import numpy as np

from .base import ClassifierBase


class LDAClassifier(ClassifierBase):
    """Linear Discriminant Analysis."""

    name = "lda"

    def __init__(self, n_classes: int = 4, **kwargs: object) -> None:
        super().__init__(n_classes=n_classes, **kwargs)
        self._clf = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LDAClassifier":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        self._clf = LinearDiscriminantAnalysis()
        self._clf.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("LDA not fitted")
        return self._clf.predict(X).astype(np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("LDA not fitted")
        return self._clf.predict_proba(X).astype(np.float64)
