"""Base interface for domain adaptation (transfer learning) — sklearn-like API."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class DomainAdapter(ABC):
    """
    Abstract domain adapter: align source feature distribution to target (unlabeled).
    Used offline in cross-subject (e.g. LOSO) settings only.

    Architectural contract (mandatory): adapters operate ONLY on (n_trials, n_features).
    They must NEVER receive (n_trials, n_channels, n_samples). Dimensionality reduction
    (e.g. CSP, Riemannian) must happen before domain adaptation.
    """

    name: str = "base"

    def __init__(self, **kwargs: Any) -> None:
        self._fitted = False

    def _validate_input(self, X: np.ndarray) -> None:
        """Enforce 2D feature matrix. Call in fit() and transform() of every adapter."""
        if X.ndim != 2:
            raise ValueError(
                "Domain adaptation expects 2D feature matrix (n_trials, n_features). "
                f"Got shape {X.shape}. Reduce dimensionality first (e.g. CSP, PSD, covariance)."
            )

    @abstractmethod
    def fit(
        self,
        X_source: np.ndarray,
        X_target: np.ndarray | None = None,
    ) -> "DomainAdapter":
        """
        Fit adapter from source (and optionally target) features.
        X_source: (n_samples_source, n_features) — may have labels used elsewhere; adapter does not use them.
        X_target: (n_samples_target, n_features) — unlabeled target data only.
        If X_target is None, no adaptation (store source stats only or no-op).
        """
        return self

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features (source or target) to aligned space. (n_samples, n_features) -> (n_samples, n_features)."""
        pass

    def fit_transform(
        self,
        X_source: np.ndarray,
        X_target: np.ndarray | None = None,
    ) -> np.ndarray:
        """Fit and transform source features."""
        self.fit(X_source, X_target)
        return self.transform(X_source)

    @property
    def is_fitted(self) -> bool:
        return self._fitted
