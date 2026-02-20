"""
Euclidean Alignment (EA) + OAS covariance + tangent space + scaling.

For LOSO: alignment is computed ONLY from source subjects (fit data).
Target subject must never be used in alignment — assertion in fit().
During inner CV, alignment is recomputed from training split only.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .base import FeatureExtractorBase
from .riemann_tangent_oas import (
    compute_covariances_oas,
    compute_reference_cov,
    project_to_tangent,
)

logger = logging.getLogger(__name__)


def _sqrtm_sym(C: np.ndarray) -> np.ndarray:
    """Symmetric positive-definite square root."""
    from scipy.linalg import sqrtm
    S = np.real(sqrtm(C + 1e-10 * np.eye(C.shape[0])))
    return (S + S.T) / 2


def euclidean_align_trials(X: np.ndarray, C_global_inv_sqrt: np.ndarray) -> np.ndarray:
    """
    X: (n_trials, n_channels, n_samples).
    C_global_inv_sqrt: (n_channels, n_channels).
    Returns X_aligned = C_global_inv_sqrt @ X (same shape). Covariance of aligned trial
    is C_global^{-1/2} C_trial C_global^{-1/2}; left-multiply suffices for that.
    """
    out = np.zeros_like(X, dtype=np.float64)
    for i in range(X.shape[0]):
        # (n_ch, n_ch) @ (n_ch, n_samp) -> (n_ch, n_samp)
        out[i] = C_global_inv_sqrt @ X[i]
    return out


class EARiemannTangentOAS(FeatureExtractorBase):
    """
    Euclidean Alignment (source-only) -> OAS covariance -> tangent space -> StandardScaler.
    fit(X, y, subject_ids=..., loso_target_subject=...): asserts target not in subject_ids,
    computes per-subject mean covariance over source, C_global = mean of those, then OAS+tangent.
    transform(X): applies EA then OAS+tangent+scale.
    """

    name = "ea_riemann_tangent_oas"

    def __init__(
        self,
        fs: float,
        apply_bandpass: bool = True,
        force_float32: bool = False,
        z_score_tangent: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(fs, **kwargs)
        self.apply_bandpass = bool(apply_bandpass)
        self.force_float32 = bool(force_float32)
        self.z_score_tangent = bool(z_score_tangent)
        self._C_global_inv_sqrt: np.ndarray | None = None
        self._ref_cov: np.ndarray | None = None
        self._n_channels: int | None = None
        self._scaler: Any = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: np.ndarray | None = None,
        loso_target_subject: int | None = None,
        **kwargs: Any,
    ) -> "EARiemannTangentOAS":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 3:
            raise ValueError("EARiemannTangentOAS expects (n_trials, n_channels, n_samples)")
        n_trials, n_ch, n_samp = X.shape

        # Debug assertion: target subject must never be in alignment computation
        if loso_target_subject is not None and subject_ids is not None:
            subject_ids = np.asarray(subject_ids).ravel()
            if len(subject_ids) == n_trials:
                unique_in_fit = np.unique(subject_ids)
                assert int(loso_target_subject) not in unique_in_fit, (
                    "EA alignment must not use target subject. "
                    "loso_target_subject=%s found in fit subject_ids." % loso_target_subject
                )
                logger.debug("[EA] Assertion OK: target subject %s not in fit data", loso_target_subject)

        if self.apply_bandpass:
            from .riemann_tangent_oas import _bandpass_eeg
            X = _bandpass_eeg(X, self.fs, 8.0, 30.0)

        # Per-subject mean covariance, then global mean (source-only)
        covs = compute_covariances_oas(X)
        if subject_ids is not None and len(subject_ids) == n_trials:
            subject_ids = np.asarray(subject_ids).ravel()
            uniq = np.unique(subject_ids)
            per_subj = []
            for s in uniq:
                mask = subject_ids == s
                per_subj.append(compute_reference_cov(covs[mask]))
            C_global = np.mean(np.stack(per_subj), axis=0)
        else:
            C_global = compute_reference_cov(covs)
        self._C_global_inv_sqrt = np.linalg.inv(_sqrtm_sym(C_global))
        self._n_channels = n_ch

        # Align then tangent
        X_aligned = euclidean_align_trials(X, self._C_global_inv_sqrt)
        covs_aligned = compute_covariances_oas(X_aligned)
        self._ref_cov = compute_reference_cov(covs_aligned)
        F = project_to_tangent(covs_aligned, self._ref_cov)
        self._scaler = None
        if self.z_score_tangent:
            from sklearn.preprocessing import StandardScaler
            self._scaler = StandardScaler()
            self._scaler.fit(F)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._C_global_inv_sqrt is None or self._ref_cov is None:
            raise RuntimeError("EARiemannTangentOAS not fitted")
        X = np.asarray(X, dtype=np.float64)
        if self.apply_bandpass:
            from .riemann_tangent_oas import _bandpass_eeg
            X = _bandpass_eeg(X, self.fs, 8.0, 30.0)
        X_aligned = euclidean_align_trials(X, self._C_global_inv_sqrt)
        covs = compute_covariances_oas(X_aligned)
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
