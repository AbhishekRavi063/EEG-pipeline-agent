"""
Spatial filter resolution (v3.1): strict vs auto modes, single log per reason.
Resolves requested method with FrameworkCapabilities; returns actual method to use.
"""

from __future__ import annotations

import logging
from typing import Any

SPATIAL_LOGGER = logging.getLogger("SPATIAL")
_reason_logged: set[str] = set()


def _normalize_method(method: str) -> str:
    """Backward compat: laplacian -> laplacian_auto, gedai -> gedai_auto."""
    m = (method or "").strip().lower()
    if not m:
        return m
    if m == "laplacian":
        return "laplacian_auto"
    if m == "gedai":
        return "gedai_auto"
    return m


def resolve_spatial_method(
    method: str,
    capabilities: Any,
) -> tuple[str, str]:
    """
    Resolve requested spatial method with capabilities (strict vs auto).
    Returns (actual_method_for_registry, actual_used_method_for_logging).
    Raises RuntimeError in strict mode if unsupported.
    """
    global _reason_logged
    method = _normalize_method(method)
    cap = capabilities
    has_laplacian = getattr(cap, "laplacian_supported", False)
    has_gedai = getattr(cap, "gedai_supported", False)

    # Laplacian
    if method in ("laplacian_strict", "laplacian_auto"):
        if method == "laplacian_strict":
            if not has_laplacian:
                reason = getattr(cap, "reason", {}).get("laplacian", "No montage available")
                raise RuntimeError(f"Laplacian requested (strict) but not supported: {reason}")
            return "laplacian", "laplacian"
        if method == "laplacian_auto":
            if has_laplacian:
                return "laplacian", "laplacian"
            key = "laplacian_unavailable"
            if key not in _reason_logged:
                SPATIAL_LOGGER.warning("[SPATIAL] Laplacian unavailable (no montage). Using CAR.")
                _reason_logged.add(key)
            return "car", "car"

    # GEDAI
    if method in ("gedai_strict", "gedai_auto"):
        if method == "gedai_strict":
            if not has_gedai:
                reason = getattr(cap, "reason", {}).get("gedai", "No leadfield found")
                raise RuntimeError(f"GEDAI requested (strict) but not supported: {reason}")
            return "gedai", "gedai"
        if method == "gedai_auto":
            if has_gedai:
                return "gedai", "gedai"
            key = "gedai_unavailable"
            if key not in _reason_logged:
                SPATIAL_LOGGER.warning("[SPATIAL] GEDAI unavailable (no leadfield). Using CAR.")
                _reason_logged.add(key)
            return "car", "car"

    return method, method


def method_for_registry(method: str) -> str:
    """Map strict/auto method names to SPATIAL_FILTER_REGISTRY keys (laplacian_auto -> laplacian)."""
    m = _normalize_method(method)
    if m == "laplacian_auto" or m == "laplacian_strict":
        return "laplacian"
    if m == "gedai_auto" or m == "gedai_strict":
        return "gedai"
    return m
