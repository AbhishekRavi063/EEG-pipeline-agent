"""Riemannian transport domain adapter: align covariance matrices in SPD space (covariance features only)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .base_adapter import DomainAdapter

logger = logging.getLogger(__name__)


def _vec_to_upper_tri(vec: np.ndarray, n: int) -> np.ndarray:
    """Flattened upper triangle -> (n, n) symmetric matrix."""
    M = np.zeros((n, n), dtype=np.float64)
    idx = np.triu_indices(n)
    M[idx] = vec
    M.T[idx] = vec
    return M


def _upper_tri_to_vec(M: np.ndarray) -> np.ndarray:
    n = M.shape[0]
    idx = np.triu_indices(n)
    return M[idx].astype(np.float64)


def _geometric_mean_scipy(covs: np.ndarray) -> np.ndarray:
    """Log-Euclidean mean of SPD matrices."""
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
    try:
        from pyriemann.utils.mean import mean_riemann
        return mean_riemann(covs)
    except Exception:
        return _geometric_mean_scipy(covs)


class RiemannianTransportAdapter(DomainAdapter):
    """
    Transport source covariance matrices to target Riemannian centroid.
    Only valid when features are covariance matrices (flattened upper triangle).
    C_aligned = G_target^{1/2} @ G_source^{-1/2} @ C @ G_source^{-1/2} @ G_target^{1/2}
    """

    name = "riemann_transport"

    def __init__(self, feature_name: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.feature_name = ((feature_name or kwargs.get("feature_name")) or "covariance").lower()
        self._n_channels: int | None = None
        self._G_source_inv_sqrt: np.ndarray | None = None
        self._G_target_sqrt: np.ndarray | None = None
        self._had_target = False

    def _is_covariance_feature(self, n_flat: int) -> bool:
        """Check if n_flat = n*(n+1)/2 for some n (covariance upper triangle)."""
        if self.feature_name == "covariance":
            return True
        n = int(round((-1 + (1 + 8 * n_flat) ** 0.5) / 2))
        return n * (n + 1) // 2 == n_flat

    def _flat_to_cov(self, vec: np.ndarray) -> np.ndarray:
        n = self._n_channels
        if n is None:
            n_flat = len(vec)
            n = int(round((-1 + (1 + 8 * n_flat) ** 0.5) / 2))
            if n * (n + 1) // 2 != n_flat:
                raise ValueError(
                    "RiemannianTransportAdapter requires covariance features (flattened upper triangle)."
                )
            self._n_channels = n
        return _vec_to_upper_tri(vec, n)

    def fit(
        self,
        X_source: np.ndarray,
        X_target: np.ndarray | None = None,
    ) -> "RiemannianTransportAdapter":
        logger.info("[TRANSFER] Adapter fit called")
        self._validate_input(X_source)
        X_source = np.asarray(X_source, dtype=np.float64)
        n_flat = X_source.shape[1]
        if not self._is_covariance_feature(n_flat):
            logger.warning(
                "RiemannianTransportAdapter: feature %s does not look like covariance (n_flat=%s); pass-through.",
                self.feature_name, n_flat,
            )
            self._last_identity_diff = None
            self._fitted = True
            return self

        n = int(round((-1 + (1 + 8 * n_flat) ** 0.5) / 2))
        self._n_channels = n

        covs_source = np.array([self._flat_to_cov(X_source[i]) for i in range(X_source.shape[0])])
        G_source = _geometric_mean_pyriemann(covs_source)
        G_source += 1e-10 * np.eye(n)
        from scipy.linalg import sqrtm, inv
        self._G_source_inv_sqrt = np.real(inv(np.real(sqrtm(G_source))))

        self._had_target = False
        if X_target is not None and len(X_target) > 0:
            self._validate_input(X_target)
            X_target = np.asarray(X_target, dtype=np.float64)
            covs_target = np.array([self._flat_to_cov(X_target[i]) for i in range(X_target.shape[0])])
            G_target = _geometric_mean_pyriemann(covs_target)
            G_target += 1e-10 * np.eye(n)
            self._G_target_sqrt = np.real(sqrtm(G_target))
            self._had_target = True
        else:
            self._G_target_sqrt = np.real(sqrtm(G_source))

        # Diagnostic: transport matrix T = G_target_sqrt @ G_source_inv_sqrt (maps source to target)
        T = self._G_target_sqrt @ self._G_source_inv_sqrt
        identity_diff = float(np.mean(np.abs(T - np.eye(T.shape[0]))))
        self._last_identity_diff = identity_diff
        logger.info(
            "[DEBUG] Riemann transport matrix deviation from identity: %.8f",
            identity_diff,
        )

        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._validate_input(X)
        if not self._fitted or self._G_source_inv_sqrt is None or self._n_channels is None:
            return np.asarray(X, dtype=np.float64)
        X = np.asarray(X, dtype=np.float64)
        n_flat = X.shape[1]
        if n_flat != self._n_channels * (self._n_channels + 1) // 2:
            return X
        G_target_sqrt = self._G_target_sqrt if self._G_target_sqrt is not None else np.eye(self._n_channels)
        out = []
        for i in range(X.shape[0]):
            C = self._flat_to_cov(X[i])
            C_aligned = G_target_sqrt @ self._G_source_inv_sqrt @ C @ self._G_source_inv_sqrt.T @ G_target_sqrt.T
            C_aligned = (C_aligned + C_aligned.T) / 2
            out.append(_upper_tri_to_vec(C_aligned))
        return np.array(out)
