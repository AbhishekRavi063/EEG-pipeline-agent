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
from .spatial_filters import (
    SPATIAL_FILTER_REGISTRY,
    SpatialFilterBase,
    get_spatial_filter,
)
from .wavelet import WaveletDenoising
from .subject_norm import subject_standardize, subject_standardize_per_subject

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
    "SPATIAL_FILTER_REGISTRY",
    "SpatialFilterBase",
    "get_spatial_filter",
    "WaveletDenoising",
    "ASRArtifactRemoval",
    "SignalQualityMonitor",
    "subject_standardize",
    "subject_standardize_per_subject",
]
