"""SVM classifier."""

import numpy as np

from .base import ClassifierBase


class SVMClassifier(ClassifierBase):
    """Support Vector Machine (RBF by default)."""

    name = "svm"

    def __init__(
        self,
        n_classes: int = 4,
        C: float = 1.0,
        kernel: str = "rbf",
        gamma: str = "scale",
        **kwargs: object,
    ) -> None:
        super().__init__(n_classes=n_classes, **kwargs)
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self._clf = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "SVMClassifier":
        from sklearn.svm import SVC
        self._clf = SVC(
            C=self.C,
            kernel=self.kernel,
            gamma=self.gamma,
            probability=True,
            random_state=42,
        )
        if sample_weight is not None:
            self._clf.fit(X, y, sample_weight=sample_weight)
        else:
            self._clf.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("SVM not fitted")
        return self._clf.predict(X).astype(np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("SVM not fitted")
        return self._clf.predict_proba(X).astype(np.float64)
