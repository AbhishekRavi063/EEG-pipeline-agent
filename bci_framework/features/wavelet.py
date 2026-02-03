"""Wavelet-based features (e.g. energy per band per channel)."""

import numpy as np

from .base import FeatureExtractorBase


class WaveletFeatures(FeatureExtractorBase):
    """Wavelet decomposition energy per level per channel."""

    name = "wavelet"

    def __init__(self, fs: float, wavelet: str = "db4", level: int = 4, **kwargs: object) -> None:
        super().__init__(fs, wavelet=wavelet, level=level, **kwargs)
        self.wavelet = wavelet
        self.level = level

    def fit(self, X: np.ndarray, y: np.ndarray) -> "WaveletFeatures":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        try:
            import pywt
        except ImportError:
            return np.var(X.reshape(X.shape[0], -1), axis=1, keepdims=True)
        out = []
        for i in range(X.shape[0]):
            ch_feats = []
            for c in range(X.shape[1]):
                coeffs = pywt.wavedec(X[i, c, :].astype(np.float64), self.wavelet, level=self.level)
                energies = [np.sum(np.square(c)) for c in coeffs]
                ch_feats.extend(energies)
            out.append(np.array(ch_feats, dtype=np.float64))
        return np.array(out)

    @property
    def n_features_out(self) -> int | None:
        return None
