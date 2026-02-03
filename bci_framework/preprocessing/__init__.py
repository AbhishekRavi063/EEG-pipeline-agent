"""Preprocessing orchestration for EEG pipelines."""

from .asr import ASRArtifactRemoval
from .base import AdvancedPreprocessingBase, PreprocessingBase
from .bandpass import BandpassFilter
from .manager import (
    ADVANCED_PREPROCESSING_REGISTRY,
    MandatoryPreprocessingPipeline,
    PreprocessingManager,
)
from .notch import NotchFilter
from .reference import CommonAverageReference, LaplacianReference
from .signal_quality import SignalQualityMonitor
from .wavelet import WaveletDenoising

__all__ = [
    "PreprocessingBase",
    "AdvancedPreprocessingBase",
    "BandpassFilter",
    "NotchFilter",
    "CommonAverageReference",
    "LaplacianReference",
    "MandatoryPreprocessingPipeline",
    "PreprocessingManager",
    "ADVANCED_PREPROCESSING_REGISTRY",
    "WaveletDenoising",
    "ASRArtifactRemoval",
    "SignalQualityMonitor",
]
