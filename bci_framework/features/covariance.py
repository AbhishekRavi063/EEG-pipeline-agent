"""Covariance matrices (flattened upper triangle) for Riemannian/MDM pipelines."""

import numpy as np

from .base import FeatureExtractorBase


class CovarianceFeatures(FeatureExtractorBase):
    """
    Per-trial covariance matrix, flattened to upper triangle.
    For use with MDM classifier (Riemannian geometry on SPD matrices).
    """

    name = "covariance"

    def __init__(self, fs: float, **kwargs: object) -> None:
        super().__init__(fs, **kwargs)
        self._n_channels: int | None = None

    def _cov(self, x: np.ndarray) -> np.ndarray:
        """(n_channels, n_samples) -> (n_channels, n_channels) SPD. v3.2: 1e-5 regularization."""
        c = np.cov(x)
        c += 1e-5 * np.eye(c.shape[0])
        return c

    def _upper_tri_to_vec(self, M: np.ndarray) -> np.ndarray:
        n = M.shape[0]
        idx = np.triu_indices(n)
        return M[idx].astype(np.float64)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CovarianceFeatures":
        self._n_channels = X.shape[1]
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        out = []
        for i in range(X.shape[0]):
            c = self._cov(X[i])
            out.append(self._upper_tri_to_vec(c))
        return np.array(out)

    @property
    def n_features_out(self) -> int | None:
        if self._n_channels is None:
            return None
        n = self._n_channels
        return n * (n + 1) // 2
