"""Deep feature extractor placeholder (optional CNN encoder)."""

import numpy as np

from .base import FeatureExtractorBase


class DeepFeatureExtractor(FeatureExtractorBase):
    """Placeholder for CNN encoder. Returns handcrafted fallback until implemented."""

    name = "deep"

    def __init__(self, fs: float, **kwargs: object) -> None:
        super().__init__(fs, **kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DeepFeatureExtractor":
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        # Placeholder: return channel-wise mean + std as simple features
        # Replace with trained CNN encoder when available
        n_trials = X.shape[0]
        feats = []
        for i in range(n_trials):
            mean_per_ch = np.mean(X[i], axis=1)
            std_per_ch = np.std(X[i], axis=1) + 1e-10
            feats.append(np.concatenate([mean_per_ch, std_per_ch]))
        return np.array(feats, dtype=np.float64)

    @property
    def n_features_out(self) -> int | None:
        return None
