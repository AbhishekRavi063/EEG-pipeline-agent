"""Preprocessing orchestration for mandatory and advanced EEG pipelines."""

from __future__ import annotations

import logging
import time
from typing import Any, Iterable, List, Tuple

import numpy as np

from .asr import ASRArtifactRemoval
from .base import AdvancedPreprocessingBase, PreprocessingBase
from .ica import ICAArtifactRemoval
from .reference import CommonAverageReference, LaplacianReference
from .signal_quality import SignalQualityMonitor
from .wavelet import WaveletDenoising
from .bandpass import BandpassFilter
from .notch import NotchFilter

logger = logging.getLogger(__name__)

ADVANCED_PREPROCESSING_REGISTRY = {
    "ica": ICAArtifactRemoval,
    "wavelet": WaveletDenoising,
    "asr": ASRArtifactRemoval,
    "signal_quality": SignalQualityMonitor,
}


class MandatoryPreprocessingPipeline:
    """Applies notch, bandpass, and re-referencing in the correct scientific order."""

    def __init__(
        self,
        fs: float,
        config: dict[str, Any],
        channel_names: list[str] | None = None,
    ) -> None:
        self.fs = fs
        self.config = config
        self.channel_names = channel_names or []
        prep_cfg = config.get("preprocessing", {})
        task = (config.get("task") or "").lower()

        notch_freq = float(prep_cfg.get("notch_freq", 50.0))
        notch_q = float(prep_cfg.get("notch_quality", 30.0))
        band_low = float(prep_cfg.get("bandpass_low", 0.5))
        band_high = float(prep_cfg.get("bandpass_high", 40.0))

        if task == "motor_imagery" and prep_cfg.get("adaptive_motor_band", True):
            band_low = float(prep_cfg.get("motor_band_low", 8.0))
            band_high = float(prep_cfg.get("motor_band_high", 30.0))
            logger.info(
                "MandatoryPreprocessingPipeline: applying adaptive motor band (%s–%s Hz)",
                band_low,
                band_high,
            )

        band_order = int(prep_cfg.get("bandpass_order", 5))

        reference_mode = (prep_cfg.get("reference", "car") or "car").lower()
        lap_neigh = prep_cfg.get("laplacian_neighbours") or prep_cfg.get("laplacian_neighbors")

        causal = (config.get("mode", "offline").lower() == "online") or bool(prep_cfg.get("force_causal_filters", False))

        self.notch = NotchFilter(fs=fs, freq=notch_freq, quality=notch_q, causal=causal)
        self.bandpass = BandpassFilter(fs=fs, lowcut=band_low, highcut=band_high, order=band_order, causal=causal)

        if reference_mode == "laplacian":
            self.reference: PreprocessingBase = LaplacianReference(fs=fs, neighbours=lap_neigh)
            if hasattr(self.reference, "set_channel_names") and channel_names:
                self.reference.set_channel_names(channel_names)
        else:
            self.reference = CommonAverageReference(fs=fs)

        self.steps: list[tuple[str, PreprocessingBase]] = [
            ("notch", self.notch),
            ("bandpass", self.bandpass),
            ("reference", self.reference),
        ]

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        data = np.asarray(X, dtype=np.float64)
        for name, step in self.steps:
            start = time.perf_counter()
            data = step.fit_transform(data, y)
            logger.info("Mandatory step %s fit/transform completed in %.3f ms", name, (time.perf_counter() - start) * 1e3)
        return data

    def transform(self, X: np.ndarray) -> np.ndarray:
        data = np.asarray(X, dtype=np.float64)
        for name, step in self.steps:
            start = time.perf_counter()
            data = step.transform(data)
            logger.debug(
                "Mandatory step %s transform completed in %.3f ms",
                name,
                (time.perf_counter() - start) * 1e3,
            )
        return data

    def process(self, data: np.ndarray, y: np.ndarray | None = None, fit: bool = False) -> np.ndarray:
        return self.fit(data, y) if fit else self.transform(data)


class PreprocessingManager:
    """Orchestrates mandatory and optional preprocessing stages."""

    def __init__(
        self,
        fs: float,
        config: dict[str, Any],
        channel_names: list[str] | None = None,
    ) -> None:
        self.fs = fs
        self.config = config
        self.channel_names = channel_names or []
        self.mode = (config.get("mode") or "offline").lower()
        self.task = (config.get("task") or "generic").lower()

        self.mandatory = MandatoryPreprocessingPipeline(fs=fs, config=config, channel_names=self.channel_names)
        self.advanced_steps: list[tuple[str, AdvancedPreprocessingBase]] = []

        adv_cfg = config.get("advanced_preprocessing", {})
        enabled: Iterable[str] = adv_cfg.get("enabled", []) or []
        disabled_for_mode: list[str] = []

        for name in enabled:
            cls = ADVANCED_PREPROCESSING_REGISTRY.get(name)
            if cls is None:
                logger.warning("Advanced preprocessing step %s is not registered; skipping.", name)
                continue
            params = adv_cfg.get(name, {})
            step = cls(fs=fs, **params)
            if hasattr(step, "set_channel_names") and self.channel_names:
                step.set_channel_names(self.channel_names)
            if self.mode == "online" and isinstance(step, AdvancedPreprocessingBase) and not step.is_online_supported():
                disabled_for_mode.append(name)
                continue
            self.advanced_steps.append((name, step))

        if disabled_for_mode:
            logger.info(
                "PreprocessingManager: disabled advanced steps in online mode: %s",
                ", ".join(disabled_for_mode),
            )

        enabled_names = [name for name, _ in self.advanced_steps]
        logger.info(
            "PreprocessingManager initialised (mode=%s, mandatory=[notch, bandpass, reference], advanced=%s)",
            self.mode,
            enabled_names or "none",
        )

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        data = self.mandatory.fit(X, y)
        for name, step in self.advanced_steps:
            start = time.perf_counter()
            data = step.fit_transform(data, y)
            logger.info(
                "Advanced step %s fit/transform completed in %.3f ms",
                name,
                (time.perf_counter() - start) * 1e3,
            )
        return data

    def transform(self, X: np.ndarray) -> np.ndarray:
        data = self.mandatory.transform(X)
        for name, step in self.advanced_steps:
            start = time.perf_counter() if self.mode == "online" else None
            data = step.transform(data)
            if start is not None:
                logger.debug(
                    "Advanced step %s transform took %.3f ms",
                    name,
                    (time.perf_counter() - start) * 1e3,
                )
        return data

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        return self.fit(X, y)

    def process(self, data: np.ndarray, y: np.ndarray | None = None, fit: bool = False) -> np.ndarray:
        return self.fit(data, y) if fit else self.transform(data)

    def enabled_advanced_steps(self) -> List[str]:
        return [name for name, _ in self.advanced_steps]
