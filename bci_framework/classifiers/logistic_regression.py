"""Regularized multinomial Logistic Regression for tangent-space / high-dim features."""

from __future__ import annotations

import logging

import numpy as np

from .base import ClassifierBase

logger = logging.getLogger(__name__)


class LogisticRegressionClassifier(ClassifierBase):
    """L2-regularized multinomial Logistic Regression (stable for Riemann tangent features).
    Supports tune_C: inner CV on source only (never target) for C in {0.1, 1, 10}."""

    name = "logistic_regression"

    def __init__(
        self,
        n_classes: int = 4,
        C: float = 1.0,
        tune_C: bool = False,
        C_grid: list[float] | None = None,
        max_iter: int = 1000,
        solver: str = "lbfgs",
        cv_folds: int = 3,
        platt_scaling: bool = False,
        **kwargs: object,
    ) -> None:
        super().__init__(n_classes=n_classes, **kwargs)
        self.C = C
        self.tune_C = bool(tune_C)
        self.C_grid = C_grid or [0.01, 0.1, 1.0, 10.0]
        self.max_iter = max_iter
        self.solver = solver
        self.cv_folds = cv_folds
        self.platt_scaling = bool(platt_scaling)
        self._clf = None
        self._platt_A: np.ndarray | None = None
        self._platt_B: np.ndarray | None = None
        self._selected_C: float | None = None  # set when tune_C is used (for LOSO reporting)

    def _select_best_C(self, X: np.ndarray, y: np.ndarray) -> float:
        """Inner CV on source only; select best C. Never use target data."""
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.linear_model import LogisticRegression as LR

        n_splits = min(self.cv_folds, len(X) // 2, len(np.unique(y)) or 2)
        n_splits = max(2, n_splits)
        best_C = self.C
        best_score = -1.0
        for C in self.C_grid:
            kwargs: dict = {
                "penalty": "l2",
                "C": C,
                "max_iter": self.max_iter,
                "solver": self.solver,
                "random_state": 42,
            }
            try:
                clf = LR(multi_class="multinomial", **kwargs)
            except TypeError:
                clf = LR(**kwargs)
            kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            scores = cross_val_score(clf, X, y, cv=kf, scoring="accuracy")
            mean_score = float(np.mean(scores))
            if mean_score > best_score:
                best_score = mean_score
                best_C = C
        return best_C

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "LogisticRegressionClassifier":
        from sklearn.linear_model import LogisticRegression as LR

        C = self.C
        if self.tune_C and len(X) >= 6:
            C = self._select_best_C(X, y)
            self._selected_C = C
            logger.info("[LogisticRegression] fold selected C=%.4f (from grid %s)", C, self.C_grid)
        else:
            self._selected_C = C

        kwargs: dict = {
            "penalty": "l2",
            "C": C,
            "max_iter": self.max_iter,
            "solver": self.solver,
            "random_state": 42,
        }
        try:
            self._clf = LR(multi_class="multinomial", **kwargs)
        except TypeError:
            self._clf = LR(**kwargs)
        if sample_weight is not None:
            self._clf.fit(X, y, sample_weight=sample_weight)
        else:
            self._clf.fit(X, y)
        if self.platt_scaling and len(X) >= 4:
            self._fit_platt(X, y)
        else:
            self._platt_A = None
            self._platt_B = None
        return self

    def _fit_platt(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit Platt scaling (sigmoid) per class on training data only."""
        from scipy.optimize import minimize
        proba = self._clf.predict_proba(X)
        n_classes = proba.shape[1]
        A, B = np.ones(n_classes, dtype=np.float64), np.zeros(n_classes, dtype=np.float64)
        for k in range(n_classes):
            p = np.clip(proba[:, k], 1e-7, 1 - 1e-7)
            y_bin = (y == k).astype(np.float64)

            def neg_ll(params: np.ndarray) -> float:
                a, b = params[0], params[1]
                q = 1.0 / (1.0 + np.exp(np.clip(-a * p - b, -500, 500)))
                q = np.clip(q, 1e-7, 1 - 1e-7)
                return -float(np.sum(y_bin * np.log(q) + (1 - y_bin) * np.log(1 - q)))

            res = minimize(neg_ll, [1.0, 0.0], method="L-BFGS-B", bounds=[(0.01, 100), (-10, 10)])
            if res.success:
                A[k], B[k] = res.x[0], res.x[1]
        self._platt_A = A
        self._platt_B = B

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("LogisticRegression not fitted")
        return self._clf.predict(X).astype(np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("LogisticRegression not fitted")
        proba = self._clf.predict_proba(X).astype(np.float64)
        if self._platt_A is not None and self._platt_B is not None:
            for k in range(proba.shape[1]):
                p = np.clip(proba[:, k], 1e-7, 1 - 1e-7)
                proba[:, k] = 1.0 / (1.0 + np.exp(-self._platt_A[k] * p - self._platt_B[k]))
            row_sum = proba.sum(axis=1, keepdims=True)
            proba = np.where(row_sum > 0, proba / row_sum, proba)
        return proba
