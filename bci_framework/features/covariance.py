"""Covariance matrices (flattened upper triangle) for Riemannian/MDM pipelines."""

import logging
from typing import Any

import numpy as np

from .base import FeatureExtractorBase

logger = logging.getLogger(__name__)


def _cov_oas(x: np.ndarray, n_samples: int | None = None) -> np.ndarray:
    """Ledoit-Wolf OAS shrinkage for SPD; fallback to empirical + reg."""
    try:
        from pyriemann.estimation import Covariances
        est = Covariances(estimator="oas")
        # est.fit_transform expects (n_trials, n_channels, n_samples)
        C = est.fit_transform(x[np.newaxis, ...])
        return C[0]
    except Exception:
        c = np.cov(x)
        n = c.shape[0]
        reg = 1e-5 * np.eye(n)
        return c + reg


class CovarianceFeatures(FeatureExtractorBase):
    """
    Per-trial covariance matrix, flattened to upper triangle.
    For use with MDM classifier (Riemannian geometry on SPD matrices).
    Optional: use pyriemann OAS estimator when available.
    """

    name = "covariance"

    def __init__(self, fs: float, use_oas: bool = True, **kwargs: object) -> None:
        super().__init__(fs, **kwargs)
        self._n_channels: int | None = None
        self._use_oas = bool(use_oas)

    def _cov(self, x: np.ndarray) -> np.ndarray:
        """(n_channels, n_samples) -> (n_channels, n_channels) SPD."""
        if self._use_oas:
            return _cov_oas(x)
        c = np.cov(x)
        c += 1e-5 * np.eye(c.shape[0])
        return (c + c.T) / 2.0

    def _upper_tri_to_vec(self, M: np.ndarray) -> np.ndarray:
        n = M.shape[0]
        idx = np.triu_indices(n)
        return M[idx].astype(np.float64)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CovarianceFeatures":
        if X.ndim != 3:
            raise ValueError("CovarianceFeatures expects (n_trials, n_channels, n_samples)")
        self._n_channels = X.shape[1]
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        out = []
        for i in range(X.shape[0]):
            c = self._cov(X[i])
            c = (c + c.T) / 2.0
            evals = np.linalg.eigvalsh(c)
            if np.any(evals <= 0):
                c += (1e-5 - np.min(evals)) * np.eye(c.shape[0])
            vec = self._upper_tri_to_vec(c)
            assert not np.any(np.isnan(vec)), "NaN in covariance feature"
            out.append(vec)
        arr = np.array(out, dtype=np.float64)
        assert arr.ndim == 2 and arr.shape[0] == X.shape[0], "Feature shape = (n_trials, n_features)"
        return arr

    @property
    def n_features_out(self) -> int | None:
        if self._n_channels is None:
            return None
        n = self._n_channels
        return n * (n + 1) // 2
