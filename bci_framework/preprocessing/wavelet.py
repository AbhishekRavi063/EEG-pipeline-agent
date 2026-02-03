"""Wavelet denoising for EEG."""

import numpy as np

from .base import AdvancedPreprocessingBase


class WaveletDenoising(AdvancedPreprocessingBase):
    """Wavelet denoising (e.g. db4, level 4)."""

    name = "wavelet"
    supports_online: bool = False

    def __init__(
        self,
        fs: float,
        wavelet: str = "db4",
        level: int = 4,
        **kwargs: object,
    ) -> None:
        super().__init__(fs, wavelet=wavelet, level=level, **kwargs)
        self.wavelet = wavelet
        self.level = level

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "WaveletDenoising":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        try:
            import pywt
        except ImportError:
            return X
        out = np.zeros_like(X, dtype=np.float64)
        for i in range(X.shape[0]):
            for c in range(X.shape[1]):
                coeffs = pywt.wavedec(X[i, c, :].astype(np.float64), self.wavelet, level=self.level)
                sigma = np.median(np.abs(coeffs[-1])) / 0.6745 if coeffs[-1].size else 1.0
                uthresh = sigma * np.sqrt(2 * np.log(X.shape[2])) if X.shape[2] > 0 else sigma
                coeffs_t = [pywt.threshold(c, uthresh, mode="soft") for c in coeffs]
                out[i, c, :] = pywt.waverec(coeffs_t, self.wavelet)[: X.shape[2]]
        return out
