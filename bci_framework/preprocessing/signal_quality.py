"""Signal quality monitoring and basic channel interpolation."""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np

from .base import AdvancedPreprocessingBase

logger = logging.getLogger(__name__)


def _kurtosis(data: np.ndarray) -> np.ndarray:
    mean = np.mean(data, axis=-1, keepdims=True)
    centered = data - mean
    var = np.mean(centered**2, axis=-1, keepdims=True) + 1e-12
    return np.mean(centered**4, axis=-1) / (var.squeeze(-1) ** 2) - 3.0


class SignalQualityMonitor(AdvancedPreprocessingBase):
    """
    Detects bad channels via variance/kurtosis z-scores and optionally interpolates them.

    Intended as a lightweight industry-grade health check rather than a full ICA-based
    rejection pipeline.
    """

    name = "signal_quality"
    supports_online = True

    def __init__(
        self,
        fs: float,
        variance_z: float = 5.0,
        kurtosis_z: float = 5.0,
        interpolate: bool = True,
        **kwargs: object,
    ) -> None:
        super().__init__(fs, variance_z=variance_z, kurtosis_z=kurtosis_z, interpolate=interpolate, **kwargs)
        self.variance_z = variance_z
        self.kurtosis_z = kurtosis_z
        self.interpolate = interpolate
        self._var_mean = None
        self._var_std = None
        self._kurt_mean = None
        self._kurt_std = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "SignalQualityMonitor":
        data = np.asarray(X, dtype=np.float64)
        var = np.var(data, axis=2)
        kurt = _kurtosis(data)
        self._var_mean = np.mean(var, axis=0)
        self._var_std = np.std(var, axis=0) + 1e-9
        self._kurt_mean = np.mean(kurt, axis=0)
        self._kurt_std = np.std(kurt, axis=0) + 1e-9
        logger.info(
            "SignalQualityMonitor: fitted baseline (variance_z=%.1f, kurtosis_z=%.1f)",
            self.variance_z,
            self.kurtosis_z,
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._var_mean is None or self._kurt_mean is None:
            return X
        data = np.asarray(X, dtype=np.float64).copy()
        var = np.var(data, axis=2)
        kurt = _kurtosis(data)
        var_z = (var - self._var_mean) / self._var_std
        kurt_z = (kurt - self._kurt_mean) / self._kurt_std
        mask = (var_z > self.variance_z) | (kurt_z > self.kurtosis_z)
        if not np.any(mask):
            return data

        logger.debug("SignalQualityMonitor: detected noisy channels (count=%d)", int(np.sum(mask)))
        if not self.interpolate:
            return data

        for trial in range(data.shape[0]):
            bad_channels = np.where(mask[trial])[0] if mask.ndim == 2 else np.where(mask)[0]
            for ch in bad_channels:
                good = [i for i in range(data.shape[1]) if i != ch]
                if not good:
                    continue
                data[trial, ch, :] = np.mean(data[trial, good, :], axis=0)
        return data
