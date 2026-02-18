"""CSP as a spatial filter (projection to discriminative components)."""

from __future__ import annotations

import numpy as np

from .base import SpatialFilterBase


def _reg_cov_static(x: np.ndarray, reg: float = 1e-5) -> np.ndarray:
    """(n_channels, n_samples) -> (n_channels, n_channels) with regularization."""
    c = np.cov(x)
    c += reg * np.eye(c.shape[0])
    return c


class CSPSpatialFilter(SpatialFilterBase):
    """
    Common Spatial Patterns as spatial filter: project to n_components
    discriminative components. Output shape (n_trials, n_components*2, n_samples).
    Offline: fit on calibration; online: use precomputed filters (matrix mult only).
    """

    name = "csp"
    _reg = 1e-5

    def __init__(self, fs: float, n_components: int = 4, **kwargs: object) -> None:
        super().__init__(fs, n_components=n_components, **kwargs)
        self.n_components = n_components
        self._filters: np.ndarray | None = None  # (n_channels, n_components*2)

    def _reg_cov(self, x: np.ndarray) -> np.ndarray:
        return _reg_cov_static(x, self._reg)

    def fit(
        self, X: np.ndarray, y: np.ndarray | None = None, info: dict | None = None
    ) -> "CSPSpatialFilter":
        if X.ndim != 3:
            raise ValueError(
                f"CSP spatial expects 3D array (n_trials, n_channels, n_samples); got ndim={X.ndim}"
            )
        if y is None:
            self._filters = np.eye(X.shape[1])[:, : min(self.n_components * 2, X.shape[1])]
            self._fitted = True
            return self
        uniq = np.unique(y)
        if len(uniq) < 2:
            self._filters = np.eye(X.shape[1])[:, : min(self.n_components * 2, X.shape[1])]
            self._fitted = True
            return self
        c1, c2 = uniq[0], uniq[1]
        X1 = X[y == c1]
        X2 = X[y == c2]
        cov1 = np.mean([self._reg_cov(x) for x in X1], axis=0)
        cov2 = np.mean([self._reg_cov(x) for x in X2], axis=0)
        cov1 += 1e-5 * np.eye(cov1.shape[0])
        cov2 += 1e-5 * np.eye(cov2.shape[0])
        D, W = np.linalg.eigh(cov1 + cov2)
        P = np.diag(1.0 / np.sqrt(D + 1e-10)) @ W.T
        S1 = P @ cov1 @ P.T
        S2 = P @ cov2 @ P.T
        D2, B = np.linalg.eigh(S1)
        idx = np.argsort(D2)
        B = B[:, idx]
        n_comp = min(self.n_components * 2, X.shape[1])
        W_csp = (B.T @ P)[:n_comp]
        self._filters = W_csp.T  # (n_channels, n_comp)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._filters is None:
            raise RuntimeError("CSPSpatialFilter not fitted")
        if X.ndim != 3:
            raise ValueError(
                f"CSP spatial expects 3D array (n_trials, n_channels, n_samples); got ndim={X.ndim}"
            )
        # X (n_trials, n_channels, n_samples) -> (n_trials, n_comp, n_samples)
        # _filters (n_channels, n_comp); project over channels: (n_comp, n_ch) @ (b, n_ch, s) -> (b, n_comp, s)
        out = np.einsum("ic,bcs->bis", self._filters.T, X)
        return out.astype(np.float64)

    def is_online_safe(self) -> bool:
        return True  # transform is matrix multiply
