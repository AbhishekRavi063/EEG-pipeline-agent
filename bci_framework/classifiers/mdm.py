"""MDM (Minimum Distance to Mean) classifier on covariance matrices — Riemannian geometry."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .base import ClassifierBase

logger = logging.getLogger(__name__)


def _vec_to_upper_tri(vec: np.ndarray, n: int) -> np.ndarray:
    """Flattened upper triangle -> (n, n) symmetric matrix."""
    M = np.zeros((n, n), dtype=np.float64)
    idx = np.triu_indices(n)
    M[idx] = vec
    M.T[idx] = vec
    return M


def _riemannian_distance(C1: np.ndarray, C2: np.ndarray) -> float:
    """Geodesic distance between two SPD matrices: ||log(C1^{-1/2} C2 C1^{-1/2})||_F."""
    from scipy.linalg import sqrtm, inv, logm
    C1 = np.asarray(C1, dtype=np.float64)
    C2 = np.asarray(C2, dtype=np.float64)
    C1 += 1e-10 * np.eye(C1.shape[0])
    C2 += 1e-10 * np.eye(C2.shape[0])
    s1 = sqrtm(C1)
    s1_inv = inv(s1)
    M = s1_inv @ C2 @ s1_inv
    M = (M + M.T) / 2
    logM = logm(M)
    logM = np.real(logM)
    return float(np.sqrt(np.sum(logM ** 2)))


def _geometric_mean_scipy(covs: np.ndarray) -> np.ndarray:
    """Log-Euclidean mean of SPD matrices: exp(mean(log(C_i))). (n_matrices, n, n) -> (n, n)."""
    from scipy.linalg import logm, expm
    covs = np.asarray(covs, dtype=np.float64)
    n = covs.shape[1]
    log_sum = np.zeros((n, n))
    for i in range(covs.shape[0]):
        C = covs[i] + 1e-10 * np.eye(n)
        log_sum += np.real(logm(C))
    mean_log = log_sum / covs.shape[0]
    return np.real(expm(mean_log))


def _geometric_mean_pyriemann(covs: np.ndarray) -> np.ndarray:
    """Use pyriemann if available for proper Karcher mean."""
    try:
        from pyriemann.utils.mean import mean_riemann
        return mean_riemann(covs)
    except Exception:
        return _geometric_mean_scipy(covs)


class MDMClassifier(ClassifierBase):
    """
    Minimum Distance to Mean on covariance matrices (Riemannian).
    Expects X (n_samples, n_flat) where n_flat = n_channels*(n_channels+1)//2
    (flattened upper triangle of per-trial covariance).
    """

    name = "mdm"

    def __init__(self, n_classes: int = 4, metric: str = "riemann", **kwargs: Any) -> None:
        super().__init__(n_classes=n_classes, metric=metric, **kwargs)
        self.metric = metric
        self._centroids: list[np.ndarray] = []  # one (n, n) per class
        self._n_channels: int | None = None

    def _flat_to_cov(self, vec: np.ndarray) -> np.ndarray:
        n = self._n_channels
        if n is None:
            # Infer n from vec length: n*(n+1)//2 = len(vec) -> n = (-1 + sqrt(1+8*L))/2
            L = len(vec)
            n = int(round((-1 + (1 + 8 * L) ** 0.5) / 2))
            self._n_channels = n
        return _vec_to_upper_tri(vec, n)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "MDMClassifier":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64).ravel()
        n_flat = X.shape[1]
        n = int(round((-1 + (1 + 8 * n_flat) ** 0.5) / 2))
        if n * (n + 1) // 2 != n_flat:
            raise ValueError(
                f"MDM expects flattened upper triangle: n_flat = n*(n+1)/2; got {n_flat}"
            )
        self._n_channels = n

        try:
            from pyriemann.utils.mean import mean_riemann
            mean_fun = mean_riemann
            logger.debug("MDM using pyriemann geometric mean")
        except ImportError:
            mean_fun = _geometric_mean_scipy
            logger.debug("MDM using scipy log-Euclidean mean")

        self._centroids = []
        for k in range(self.n_classes):
            mask = y == k
            if not np.any(mask):
                # No sample for this class: use identity
                self._centroids.append(np.eye(n, dtype=np.float64))
                continue
            covs = np.array([self._flat_to_cov(X[i]) for i in np.where(mask)[0]])
            self._centroids.append(mean_fun(covs))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._centroids:
            raise RuntimeError("MDM not fitted")
        X = np.asarray(X, dtype=np.float64)
        n = self._n_channels
        pred = np.zeros(X.shape[0], dtype=np.int64)
        for i in range(X.shape[0]):
            C = self._flat_to_cov(X[i])
            best = 0
            best_d = float("inf")
            for k, M in enumerate(self._centroids):
                d = _riemannian_distance(C, M)
                if d < best_d:
                    best_d = d
                    best = k
            pred[i] = best
        return pred

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Softmax-like from negative Riemannian distances."""
        if not self._centroids:
            raise RuntimeError("MDM not fitted")
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        dists = np.zeros((n_samples, self.n_classes), dtype=np.float64)
        for i in range(n_samples):
            C = self._flat_to_cov(X[i])
            for k, M in enumerate(self._centroids):
                dists[i, k] = _riemannian_distance(C, M)
        # Convert distance to pseudo-probability: exp(-d) then normalize
        proba = np.exp(-dists)
        proba /= proba.sum(axis=1, keepdims=True)
        return proba.astype(np.float64)
