"""Bandpass filter (1–40 Hz) for EEG."""

from scipy.signal import butter, filtfilt, lfilter

from .base import PreprocessingBase
import numpy as np


class BandpassFilter(PreprocessingBase):
    """Bandpass filter. Default 1–40 Hz for broad EEG."""

    name = "bandpass"

    def __init__(
        self,
        fs: float,
        lowcut: float = 1.0,
        highcut: float = 40.0,
        order: int = 5,
        causal: bool = False,
        **kwargs: object,
    ) -> None:
        super().__init__(fs, lowcut=lowcut, highcut=highcut, order=order, causal=causal, **kwargs)
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order
        self.causal = causal
        self._b = self._a = None
        self._build_filter()

    def _build_filter(self) -> None:
        nyq = 0.5 * self.fs
        low = self.lowcut / nyq
        high = self.highcut / nyq
        low = min(max(low, 1e-6), 1 - 1e-6)
        high = min(max(high, 1e-6), 1 - 1e-6)
        self._b, self._a = butter(self.order, [low, high], btype="band")

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "BandpassFilter":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        out = np.zeros_like(X, dtype=np.float64)
        filt_fn = lfilter if self.causal else filtfilt
        for i in range(X.shape[0]):
            for c in range(X.shape[1]):
                out[i, c, :] = filt_fn(self._b, self._a, X[i, c, :].astype(np.float64))
        return out
