"""Feature extraction for EEG. Returns standardized feature vectors."""

from .base import FeatureExtractorBase
from .csp import CSPFeatures
from .psd import PSDFeatures
from .wavelet import WaveletFeatures
from .riemannian import RiemannianFeatures
from .covariance import CovarianceFeatures
from .riemann_tangent_oas import RiemannTangentOAS
from .filter_bank_riemann import FilterBankRiemann
from .deep import DeepFeatureExtractor
from .raw import RawFeatures
from .euclidean_alignment import EARiemannTangentOAS

FEATURE_REGISTRY: dict[str, type[FeatureExtractorBase]] = {
    "csp": CSPFeatures,
    "psd": PSDFeatures,
    "wavelet": WaveletFeatures,
    "riemannian": RiemannianFeatures,
    "covariance": CovarianceFeatures,
    "riemann_tangent_oas": RiemannTangentOAS,
    "filter_bank_riemann": FilterBankRiemann,
    "deep": DeepFeatureExtractor,
    "raw": RawFeatures,
    "ea_riemann_tangent_oas": EARiemannTangentOAS,
}

__all__ = [
    "FeatureExtractorBase",
    "CSPFeatures",
    "PSDFeatures",
    "WaveletFeatures",
    "RiemannianFeatures",
    "CovarianceFeatures",
    "RiemannTangentOAS",
    "FilterBankRiemann",
    "DeepFeatureExtractor",
    "RawFeatures",
    "EARiemannTangentOAS",
    "FEATURE_REGISTRY",
]
