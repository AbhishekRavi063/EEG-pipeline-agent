"""Common Spatial Patterns (CSP) for motor imagery.

For 4-class (or multiclass) MI, uses MNE's multiclass CSP when available;
otherwise falls back to binary CSP (first two classes only), which yields
chance-level 4-class accuracy. See README §18 Performance Diagnosis.
"""

import logging
from typing import Any

import numpy as np

from .base import FeatureExtractorBase

logger = logging.getLogger(__name__)


def _mne_csp_available() -> bool:
    try:
        from mne.decoding import CSP
        return True
    except ImportError:
        return False


class CSPFeatures(FeatureExtractorBase):
    """CSP feature extraction. Returns log-variance of CSP-filtered signals.
    Uses MNE multiclass CSP when n_classes > 2 and MNE is available; otherwise binary (classes 0 vs 1)."""

    name = "csp"
    _reg = 1e-5  # covariance regularization (v3.2)

    def __init__(self, fs: float, n_components: int = 4, **kwargs: object) -> None:
        super().__init__(fs, n_components=n_components, **kwargs)
        self.n_components = n_components
        self._filters = None
        self._n_features = 2 * n_components
        self._use_mne: bool = False
        self._mne_csp: Any = None

    @staticmethod
    def _regularized_cov(x: np.ndarray) -> np.ndarray:
        """(n_channels, n_samples) -> (n_channels, n_channels) with regularization."""
        c = np.cov(x)
        c += CSPFeatures._reg * np.eye(c.shape[0])
        return c

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CSPFeatures":
        if X.ndim != 3:
            raise ValueError(f"CSP expects 3D array (n_trials, n_channels, n_samples); got ndim={X.ndim}")
        uniq = np.unique(y)
        n_classes = len(uniq)
        if n_classes < 2:
            self._filters = np.eye(X.shape[1])[:, : self.n_components * 2]
            self._fitted = True
            return self
        # Multiclass: use MNE CSP when available so 4-class MI is not at chance
        if n_classes > 2 and _mne_csp_available():
            from mne.decoding import CSP as MNE_CSP
            self._mne_csp = MNE_CSP(n_components=self.n_components, reg=None, log=True)
            self._mne_csp.fit(X, y)
            self._n_features = self._mne_csp.transform(X[:1]).shape[1]
            self._use_mne = True
            self._fitted = True
            logger.info("CSP: using MNE multiclass CSP (n_classes=%d)", n_classes)
            return self
        # Binary: original implementation (classes 0 vs 1)
        c1, c2 = uniq[0], uniq[1]
        X1 = X[y == c1]
        X2 = X[y == c2]
        cov1 = np.mean([self._regularized_cov(x) for x in X1], axis=0)
        cov2 = np.mean([self._regularized_cov(x) for x in X2], axis=0)
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
        if not self._fitted:
            raise RuntimeError("CSP not fitted")
        if X.ndim != 3:
            raise ValueError(f"CSP expects 3D array (n_trials, n_channels, n_samples); got ndim={X.ndim}")
        if self._use_mne and self._mne_csp is not None:
            return np.asarray(self._mne_csp.transform(X), dtype=np.float64)
        # Binary path
        out = []
        for i in range(X.shape[0]):
            proj = self._filters.T @ X[i]
            var = np.var(proj, axis=1) + 1e-10
            feat = np.log(var)
            out.append(feat)
        return np.array(out, dtype=np.float64)

    @property
    def n_features_out(self) -> int | None:
        if self._use_mne and self._mne_csp is not None:
            return self._n_features
        return self._n_features if self._filters is not None else None
