"""Raw EEG passthrough (flatten) for classifiers that expect raw (e.g. EEGNet)."""

import numpy as np

from .base import FeatureExtractorBase


class RawFeatures(FeatureExtractorBase):
    """Flatten (n_trials, n_channels, n_samples) -> (n_trials, n_channels * n_samples)."""

    name = "raw"

    def __init__(self, fs: float, **kwargs: object) -> None:
        super().__init__(fs, **kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RawFeatures":
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X.reshape(X.shape[0], -1).astype(np.float64)

    @property
    def n_features_out(self) -> int | None:
        return None
