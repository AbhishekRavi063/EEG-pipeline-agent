"""Random Forest classifier."""

import numpy as np

from .base import ClassifierBase


class RandomForestClassifier(ClassifierBase):
    """Random Forest."""

    name = "random_forest"

    def __init__(
        self,
        n_classes: int = 4,
        n_estimators: int = 200,
        max_depth: int = 10,
        min_samples_leaf: int = 5,
        **kwargs: object,
    ) -> None:
        super().__init__(n_classes=n_classes, **kwargs)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self._clf = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "RandomForestClassifier":
        from sklearn.ensemble import RandomForestClassifier as RF
        self._clf = RF(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42,
        )
        if sample_weight is not None:
            self._clf.fit(X, y, sample_weight=sample_weight)
        else:
            self._clf.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("Random Forest not fitted")
        return self._clf.predict(X).astype(np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("Random Forest not fitted")
        return self._clf.predict_proba(X).astype(np.float64)
