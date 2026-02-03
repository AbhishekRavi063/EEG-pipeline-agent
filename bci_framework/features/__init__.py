"""Feature extraction for EEG. Returns standardized feature vectors."""

from .base import FeatureExtractorBase
from .csp import CSPFeatures
from .psd import PSDFeatures
from .wavelet import WaveletFeatures
from .riemannian import RiemannianFeatures
from .deep import DeepFeatureExtractor
from .raw import RawFeatures

FEATURE_REGISTRY: dict[str, type[FeatureExtractorBase]] = {
    "csp": CSPFeatures,
    "psd": PSDFeatures,
    "wavelet": WaveletFeatures,
    "riemannian": RiemannianFeatures,
    "deep": DeepFeatureExtractor,
    "raw": RawFeatures,
}

__all__ = [
    "FeatureExtractorBase",
    "CSPFeatures",
    "PSDFeatures",
    "WaveletFeatures",
    "RiemannianFeatures",
    "DeepFeatureExtractor",
    "RawFeatures",
    "FEATURE_REGISTRY",
]
