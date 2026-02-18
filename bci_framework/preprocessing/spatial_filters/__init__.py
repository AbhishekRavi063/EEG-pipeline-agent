"""Spatial filter plugin system: CAR, Laplacian, CSP, GeDai/lead-field."""

from .base import SpatialFilterBase
from .car import CARSpatialFilter
from .laplacian import LaplacianSpatialFilter
from .csp_spatial import CSPSpatialFilter
from .gedai_leadfield import GeDaiLeadfieldSpatialFilter

# Registry: config-driven selection without code change
SPATIAL_FILTER_REGISTRY: dict[str, type] = {
    "car": CARSpatialFilter,
    "laplacian": LaplacianSpatialFilter,
    "csp": CSPSpatialFilter,
    "gedai": GeDaiLeadfieldSpatialFilter,
}


def get_spatial_filter(
    name: str,
    fs: float,
    **kwargs: object,
) -> SpatialFilterBase:
    """Factory: get spatial filter by name from config."""
    cls = SPATIAL_FILTER_REGISTRY.get(name)
    if cls is None:
        raise KeyError(
            f"Unknown spatial filter '{name}'. Available: {list(SPATIAL_FILTER_REGISTRY.keys())}"
        )
    return cls(fs=fs, **kwargs)


__all__ = [
    "SpatialFilterBase",
    "CARSpatialFilter",
    "LaplacianSpatialFilter",
    "CSPSpatialFilter",
    "GeDaiLeadfieldSpatialFilter",
    "SPATIAL_FILTER_REGISTRY",
    "get_spatial_filter",
]
