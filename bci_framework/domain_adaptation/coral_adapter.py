"""CORAL (Correlation Alignment) domain adapter: covariance alignment in feature space.

Memory-safe: operates only on (n_trials, n_features) with d <= MAX_FEATURE_DIM.
Uses streamed covariance, float32, eigh; never on raw (n_trials, n_channels, n_samples).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .base_adapter import DomainAdapter

logger = logging.getLogger(__name__)

# Hard upper bound: keeps covariance matrix ≤ 256×256 = 65k elements (safe on 8–16GB RAM)
MAX_FEATURE_DIM = 256
# In safe_mode: stricter limit
SAFE_MODE_MAX_FEATURE_DIM = 128


def _compute_cov_streamed(X: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Streamed covariance: (Xc.T @ Xc)/(n-1) + eps*I. Avoids extra np.cov buffers."""
    n = X.shape[0]
    mean = np.mean(X, axis=0, keepdims=True)
    Xc = X - mean
    C = (Xc.T @ Xc) / max(n - 1, 1)
    C = C + eps * np.eye(C.shape[0], dtype=C.dtype)
    return C


def _sqrt_inv_sqrt_eigh(C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """C = Q @ diag(ev) @ Q.T. Return C^{1/2}, C^{-1/2} via eigh (lower memory than SVD)."""
    evals, evecs = np.linalg.eigh(C)
    evals = np.maximum(evals, 1e-10)
    sqrt_ev = np.sqrt(evals)
    inv_sqrt_ev = 1.0 / sqrt_ev
    C_sqrt = (evecs * sqrt_ev) @ evecs.T
    C_inv_sqrt = (evecs * inv_sqrt_ev) @ evecs.T
    return C_sqrt, C_inv_sqrt


def _log_memory(tag: str) -> None:
    """Log process RSS for 8–16GB machine monitoring."""
    try:
        import psutil
        process = psutil.Process()
        rss_gb = process.memory_info().rss / 1e9
        logger.info("[MEM] %s RSS: %.2f GB", tag, rss_gb)
    except Exception:
        pass


class CORALAdapter(DomainAdapter):
    """
    CORAL: whiten source with source covariance, re-color with target covariance.
    X_aligned = X_source @ C_source^{-1/2} @ C_target^{1/2}

    Memory-safe: 2D input only; feature dimension capped (256 default, 128 in safe_mode);
    streamed covariance; float32; eigh-based; fail-safe on guard or numerical error.
    """

    name = "coral"

    def __init__(
        self,
        epsilon: float = 1e-3,
        safe_mode: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.epsilon = float(epsilon)
        self.safe_mode = bool(safe_mode)
        self._max_dim = SAFE_MODE_MAX_FEATURE_DIM if safe_mode else MAX_FEATURE_DIM
        self._C_source_inv_sqrt: np.ndarray | None = None
        self._C_target_sqrt: np.ndarray | None = None
        self._had_target = False
        self._fallback_identity = False
        self._dtype = np.float32 if safe_mode else np.float64

    def fit(
        self,
        X_source: np.ndarray,
        X_target: np.ndarray | None = None,
    ) -> "CORALAdapter":
        logger.info("[TRANSFER] Adapter fit called")
        self._validate_input(X_source)
        n, d = X_source.shape
        logger.info("[TRANSFER] CORAL feature dim: %s", d)
        logger.info("[TRANSFER] CORAL safe_mode: %s", self.safe_mode)

        if d > self._max_dim:
            msg = (
                f"Feature dimension {d} exceeds safe limit ({self._max_dim}) for CORAL. "
                "Reduce dimensionality first (e.g. CSP components, PCA)."
            )
            if self.safe_mode:
                logger.warning("[TRANSFER] %s Skipping adaptation (identity).", msg)
                logger.warning("[TRANSFER] CORAL using identity fallback")
                self._fallback_identity = True
                self._C_source_inv_sqrt = np.eye(d, dtype=self._dtype)
                self._C_target_sqrt = np.eye(d, dtype=self._dtype)
                self._fitted = True
                return self
            raise RuntimeError(msg)

        _log_memory("CORAL fit start")
        X_source = np.asarray(X_source, dtype=self._dtype)
        C_source = _compute_cov_streamed(X_source, self.epsilon)
        C_target = C_source.copy()

        if X_target is not None and len(X_target) > 0:
            self._validate_input(X_target)
            X_target = np.asarray(X_target, dtype=self._dtype)
            if X_target.shape[1] != d:
                logger.warning(
                    "[TRANSFER] CORAL target feature dim %s != source %s; skipping adaptation.",
                    X_target.shape[1], d,
                )
                logger.warning("[TRANSFER] CORAL using identity fallback")
                self._fallback_identity = True
                self._C_source_inv_sqrt = np.eye(d, dtype=self._dtype)
                self._C_target_sqrt = np.eye(d, dtype=self._dtype)
                self._fitted = True
                return self
            C_target = _compute_cov_streamed(X_target, self.epsilon)
            self._had_target = True

        try:
            C_source_sqrt, C_source_inv_sqrt = _sqrt_inv_sqrt_eigh(C_source.astype(np.float64))
            _, C_target_sqrt = _sqrt_inv_sqrt_eigh(C_target.astype(np.float64))
            self._C_source_inv_sqrt = C_source_inv_sqrt.astype(self._dtype)
            self._C_target_sqrt = C_target_sqrt.astype(self._dtype)
            self._fallback_identity = False
        except Exception as e:
            logger.warning(
                "[TRANSFER] CORAL ill-conditioned — falling back to identity transform. %s",
                e,
            )
            logger.warning("[TRANSFER] CORAL using identity fallback")
            self._fallback_identity = True
            self._C_source_inv_sqrt = np.eye(d, dtype=self._dtype)
            self._C_target_sqrt = np.eye(d, dtype=self._dtype)

        _log_memory("CORAL fit end")
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._validate_input(X)
        if not self._fitted or self._C_source_inv_sqrt is None:
            return np.asarray(X, dtype=self._dtype)
        if self._fallback_identity:
            return np.asarray(X, dtype=self._dtype)
        X = np.asarray(X, dtype=self._dtype)
        C_target_sqrt = self._C_target_sqrt if self._C_target_sqrt is not None else np.eye(
            X.shape[1], dtype=self._dtype
        )
        return X @ self._C_source_inv_sqrt @ C_target_sqrt
