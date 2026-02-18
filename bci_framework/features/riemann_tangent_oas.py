"""Riemannian tangent-space features with OAS shrinkage covariance and source-only reference.

For cross-subject LOSO: reference (Riemann mean) is computed from SOURCE only in fit().
Tangent projection uses pyriemann. Low-memory: float32 output optional.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .base import FeatureExtractorBase

logger = logging.getLogger(__name__)


def _bandpass_eeg(X: np.ndarray, sfreq: float, l_freq: float = 8.0, h_freq: float = 30.0) -> np.ndarray:
    """Apply bandpass to (trials, channels, samples). Uses MNE if available else scipy."""
    try:
        from mne.filter import filter_data
        out = np.zeros_like(X, dtype=np.float64)
        for i in range(X.shape[0]):
            out[i] = filter_data(X[i], sfreq=sfreq, l_freq=l_freq, h_freq=h_freq, verbose=False)
        return out
    except ImportError:
        from scipy.signal import butter, filtfilt
        nyq = 0.5 * sfreq
        low = l_freq / nyq
        high = min(h_freq / nyq, 0.99)
        b, a = butter(5, [low, high], btype="band")
        out = np.zeros_like(X, dtype=np.float64)
        for i in range(X.shape[0]):
            for c in range(X.shape[1]):
                out[i, c] = filtfilt(b, a, X[i, c])
        return out


def compute_covariances_oas(X: np.ndarray) -> np.ndarray:
    """
    OAS shrinkage covariance per trial.
    X: (trials, channels, samples) -> (trials, channels, channels).
    """
    from sklearn.covariance import OAS
    covs = []
    for i in range(X.shape[0]):
        estimator = OAS()
        estimator.fit(X[i].T)  # (samples, channels)
        covs.append(estimator.covariance_)
    return np.stack(covs).astype(np.float64)


def compute_reference_cov(covs: np.ndarray) -> np.ndarray:
    """Riemannian mean of covariance matrices (reference). Source-only in LOSO."""
    try:
        from pyriemann.utils.mean import mean_covariance
        return mean_covariance(covs, metric="riemann")
    except ImportError:
        from scipy.linalg import logm, expm
        covs = np.asarray(covs, dtype=np.float64)
        n = covs.shape[1]
        log_sum = np.zeros((n, n))
        for i in range(covs.shape[0]):
            C = covs[i] + 1e-10 * np.eye(n)
            log_sum += np.real(logm(C))
        mean_log = log_sum / covs.shape[0]
        return np.real(expm(mean_log))


def project_to_tangent(covs: np.ndarray, reference_cov: np.ndarray) -> np.ndarray:
    """Project covariances to tangent space at reference. Returns (n_trials, n_features)."""
    try:
        from pyriemann.tangentspace import TangentSpace
        ts = TangentSpace(metric="riemann")
        ts.fit(reference_cov[np.newaxis, :, :])
        return ts.transform(covs).astype(np.float64)
    except ImportError:
        from scipy.linalg import sqrtm, logm
        ref_sqrt = np.real(sqrtm(reference_cov + 1e-10 * np.eye(reference_cov.shape[0])))
        ref_inv_sqrt = np.linalg.inv(ref_sqrt)
        n = reference_cov.shape[0]
        n_flat = n * (n + 1) // 2
        out = []
        for i in range(covs.shape[0]):
            C = covs[i] + 1e-10 * np.eye(n)
            M = ref_inv_sqrt @ C @ ref_inv_sqrt
            M = (M + M.T) / 2
            log_M = np.real(logm(M))
            idx = np.triu_indices(n)
            out.append(log_M[idx])
        return np.array(out, dtype=np.float64)


class RiemannTangentOAS(FeatureExtractorBase):
    """
    OAS shrinkage covariance -> Riemann mean (reference from source only) -> tangent projection.
    fit(X_source, y): computes reference from X_source only (LOSO-safe).
    transform(X): projects to tangent space at that reference.
    """

    name = "riemann_tangent_oas"

    def __init__(
        self,
        fs: float,
        apply_bandpass: bool = True,
        force_float32: bool = False,
        z_score_tangent: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(fs, **kwargs)
        self.apply_bandpass = bool(apply_bandpass)
        self.force_float32 = bool(force_float32)
        self.z_score_tangent = bool(z_score_tangent)
        self._ref_cov: np.ndarray | None = None
        self._n_channels: int | None = None
        self._scaler: Any = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RiemannTangentOAS":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 2:
            raise ValueError("RiemannTangentOAS expects (n_trials, n_channels, n_samples)")
        if self.apply_bandpass:
            X = _bandpass_eeg(X, self.fs, 8.0, 30.0)
        source_covs = compute_covariances_oas(X)
        self._ref_cov = compute_reference_cov(source_covs)
        self._n_channels = X.shape[1]
        self._scaler = None
        if self.z_score_tangent:
            from sklearn.preprocessing import StandardScaler
            F = project_to_tangent(source_covs, self._ref_cov)
            self._scaler = StandardScaler()
            self._scaler.fit(F)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._ref_cov is None:
            raise RuntimeError("RiemannTangentOAS not fitted")
        X = np.asarray(X, dtype=np.float64)
        if self.apply_bandpass:
            X = _bandpass_eeg(X, self.fs, 8.0, 30.0)
        covs = compute_covariances_oas(X)
        F = project_to_tangent(covs, self._ref_cov)
        if self._scaler is not None:
            F = self._scaler.transform(F)
        if self.force_float32:
            F = F.astype(np.float32)
        return F

    @property
    def n_features_out(self) -> int | None:
        if self._n_channels is None:
            return None
        n = self._n_channels
        return n * (n + 1) // 2
