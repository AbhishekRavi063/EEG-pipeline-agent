"""Motor band filter (8–30 Hz) for motor imagery."""

from scipy.signal import butter, filtfilt

from .base import PreprocessingBase
import numpy as np


class MotorBandFilter(PreprocessingBase):
    """Bandpass 8–30 Hz (mu and beta) for motor imagery."""

    name = "motor_band"

    def __init__(
        self,
        fs: float,
        lowcut: float = 8.0,
        highcut: float = 30.0,
        order: int = 5,
        **kwargs: object,
    ) -> None:
        super().__init__(fs, lowcut=lowcut, highcut=highcut, order=order, **kwargs)
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order
        self._b, self._a = None, None
        self._build_filter()

    def _build_filter(self) -> None:
        nyq = 0.5 * self.fs
        low = self.lowcut / nyq
        high = self.highcut / nyq
        low = min(max(low, 1e-6), 1 - 1e-6)
        high = min(max(high, 1e-6), 1 - 1e-6)
        self._b, self._a = butter(self.order, [low, high], btype="band")

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "MotorBandFilter":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        out = np.zeros_like(X, dtype=np.float64)
        for i in range(X.shape[0]):
            for c in range(X.shape[1]):
                out[i, c, :] = filtfilt(self._b, self._a, X[i, c, :].astype(np.float64))
        return out
