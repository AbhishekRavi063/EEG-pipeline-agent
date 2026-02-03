"""Notch filter (50/60 Hz) for line noise."""

from scipy.signal import iirnotch, filtfilt, lfilter

from .base import PreprocessingBase
import numpy as np


class NotchFilter(PreprocessingBase):
    """Notch filter at 50 or 60 Hz to remove line noise."""

    name = "notch"

    def __init__(
        self,
        fs: float,
        freq: float = 50.0,
        quality: float = 30.0,
        causal: bool = False,
        **kwargs: object,
    ) -> None:
        super().__init__(fs, freq=freq, quality=quality, causal=causal, **kwargs)
        self.freq = freq
        self.quality = quality
        self.causal = causal
        self._b, self._a = None, None
        self._build_filter()

    def _build_filter(self) -> None:
        w0 = self.freq / (0.5 * self.fs)
        w0 = min(max(w0, 1e-6), 1 - 1e-6)
        self._b, self._a = iirnotch(w0, self.quality)

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "NotchFilter":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        out = np.zeros_like(X, dtype=np.float64)
        filt_fn = lfilter if self.causal else filtfilt
        for i in range(X.shape[0]):
            for c in range(X.shape[1]):
                out[i, c, :] = filt_fn(self._b, self._a, X[i, c, :].astype(np.float64))
        return out
