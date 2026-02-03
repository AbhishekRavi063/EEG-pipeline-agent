"""Common Spatial Patterns (CSP) for motor imagery."""

import numpy as np

from .base import FeatureExtractorBase


class CSPFeatures(FeatureExtractorBase):
    """CSP feature extraction. Returns log-variance of CSP-filtered signals."""

    name = "csp"

    def __init__(self, fs: float, n_components: int = 4, **kwargs: object) -> None:
        super().__init__(fs, n_components=n_components, **kwargs)
        self.n_components = n_components
        self._filters = None
        self._n_features = 2 * n_components

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CSPFeatures":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        # Binary: take two most frequent classes for CSP
        uniq = np.unique(y)
        if len(uniq) < 2:
            self._filters = np.eye(X.shape[1])[:, : self.n_components * 2]
            self._fitted = True
            return self
        c1, c2 = uniq[0], uniq[1]
        X1 = X[y == c1]
        X2 = X[y == c2]
        cov1 = np.mean([np.cov(x) for x in X1], axis=0)
        cov2 = np.mean([np.cov(x) for x in X2], axis=0)
        cov1 += 1e-6 * np.eye(cov1.shape[0])
        cov2 += 1e-6 * np.eye(cov2.shape[0])
        D, W = np.linalg.eigh(cov1 + cov2)
        P = np.diag(1.0 / np.sqrt(D + 1e-10)) @ W.T
        S1 = P @ cov1 @ P.T
        S2 = P @ cov2 @ P.T
        D2, B = np.linalg.eigh(S1)
        idx = np.argsort(D2)
        B = B[:, idx]
        W = (B.T @ P)[: self.n_components * 2]
        self._filters = W.T
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._filters is None:
            raise RuntimeError("CSP not fitted")
        # X: (n_trials, n_channels, n_samples)
        out = []
        for i in range(X.shape[0]):
            proj = self._filters.T @ X[i]
            var = np.var(proj, axis=1) + 1e-10
            feat = np.log(var)
            out.append(feat)
        return np.array(out, dtype=np.float64)

    @property
    def n_features_out(self) -> int | None:
        return self._n_features if self._filters is not None else None
