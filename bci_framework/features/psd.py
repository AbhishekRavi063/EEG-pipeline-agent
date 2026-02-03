"""Power Spectral Density (PSD) features."""

import numpy as np

from .base import FeatureExtractorBase


class PSDFeatures(FeatureExtractorBase):
    """PSD in band [fmin, fmax] per channel, flattened."""

    name = "psd"

    def __init__(
        self,
        fs: float,
        fmin: float = 1.0,
        fmax: float = 40.0,
        n_fft: int = 256,
        **kwargs: object,
    ) -> None:
        super().__init__(fs, fmin=fmin, fmax=fmax, n_fft=n_fft, **kwargs)
        self.fmin = fmin
        self.fmax = fmax
        self.n_fft = n_fft
        self._freq_bins = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PSDFeatures":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        # X: (n_trials, n_channels, n_samples)
        out = []
        for i in range(X.shape[0]):
            ch_feats = []
            for c in range(X.shape[1]):
                spec = np.abs(np.fft.rfft(X[i, c, :], n=self.n_fft)) ** 2
                freqs = np.fft.rfftfreq(self.n_fft, 1.0 / self.fs)
                mask = (freqs >= self.fmin) & (freqs <= self.fmax)
                ch_feats.append(spec[mask])
            out.append(np.concatenate(ch_feats))
        return np.array(out, dtype=np.float64)

    @property
    def n_features_out(self) -> int | None:
        return None  # Depends on n_fft and band
