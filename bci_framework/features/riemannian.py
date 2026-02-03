"""Riemannian covariance features (tangent space)."""

import numpy as np

from .base import FeatureExtractorBase


class RiemannianFeatures(FeatureExtractorBase):
    """Covariance matrices + tangent space projection. Optional: pyriemann."""

    name = "riemannian"

    def __init__(self, fs: float, n_components: int = 10, **kwargs: object) -> None:
        super().__init__(fs, n_components=n_components, **kwargs)
        self.n_components = n_components
        self._ref_cov = None

    def _cov(self, x: np.ndarray) -> np.ndarray:
        c = np.cov(x)
        c += 1e-6 * np.eye(c.shape[0])
        return c

    def _logm(self, M: np.ndarray) -> np.ndarray:
        from scipy.linalg import logm
        return np.real(logm(M))

    def _upper_tri_to_vec(self, M: np.ndarray) -> np.ndarray:
        n = M.shape[0]
        idx = np.triu_indices(n)
        return M[idx]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RiemannianFeatures":
        covs = [self._cov(x) for x in X]
        self._ref_cov = np.mean(covs, axis=0)
        self._ref_cov += 1e-6 * np.eye(self._ref_cov.shape[0])
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._ref_cov is None:
            self._ref_cov = np.eye(X.shape[1])
        out = []
        for i in range(X.shape[0]):
            c = self._cov(X[i])
            try:
                from scipy.linalg import sqrtm, inv
                ref_sqrt = sqrtm(self._ref_cov)
                ref_inv_sqrt = inv(ref_sqrt)
                M = ref_inv_sqrt @ c @ ref_inv_sqrt
                log_M = self._logm(M)
                vec = self._upper_tri_to_vec(log_M)
                out.append(vec.astype(np.float64))
            except Exception:
                vec = self._upper_tri_to_vec(c)
                out.append(vec.astype(np.float64))
        return np.array(out)

    @property
    def n_features_out(self) -> int | None:
        return None
