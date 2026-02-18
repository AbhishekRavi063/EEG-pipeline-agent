"""
Spatial Capability Resolution (v3.1).
Detect montage/neighbour and leadfield availability once; used to resolve strict/auto spatial modes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SPATIAL_LOGGER = logging.getLogger("SPATIAL")


@dataclass
class FrameworkCapabilities:
    """Result of one-time spatial capability detection."""

    has_montage: bool = False
    has_channel_positions: bool = False
    laplacian_supported: bool = False
    gedai_supported: bool = False
    reason: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "has_montage": self.has_montage,
            "has_channel_positions": self.has_channel_positions,
            "laplacian_supported": self.laplacian_supported,
            "gedai_supported": self.gedai_supported,
            "reason": dict(self.reason),
        }


def detect_spatial_capabilities(
    raw: Any = None,
    channel_names: list[str] | None = None,
    config: dict[str, Any] | None = None,
) -> FrameworkCapabilities:
    """
    Inspect MNE Raw (or fallback) and config to determine which spatial methods are usable.
    Call once after dataset load. If raw is None (e.g. MOABB numpy path), Laplacian is not supported
    unless montage is provided elsewhere.
    """
    cap = FrameworkCapabilities()
    config = config or {}

    # --- Laplacian: require montage or valid channel positions (MNE Raw) ---
    if raw is not None:
        try:
            montage_obj = None
            if hasattr(raw, "get_montage") and callable(raw.get_montage):
                montage_obj = raw.get_montage()
            info = getattr(raw, "info", None)
            dig = getattr(info, "dig", None) if info is not None else None
            if info is not None and hasattr(info, "get"):
                dig = info.get("dig", dig)

            if montage_obj is not None:
                cap.has_montage = True
                if hasattr(montage_obj, "get_positions") and callable(montage_obj.get_positions):
                    pos = montage_obj.get_positions()
                    if isinstance(pos, dict) and (pos.get("ch_pos") or len(pos) > 0):
                        cap.has_channel_positions = True
                else:
                    cap.has_channel_positions = True
            if dig is not None and len(dig) > 0:
                cap.has_channel_positions = True
                if not cap.has_montage:
                    cap.has_montage = True

            if cap.has_montage or cap.has_channel_positions:
                cap.laplacian_supported = True
            else:
                cap.reason["laplacian"] = "No electrode montage / neighbour info"
        except Exception as e:
            cap.reason["laplacian"] = str(e)
    else:
        cap.reason["laplacian"] = "No electrode montage / neighbour info (no Raw object)"

    # --- GEDAI: require leadfield matrix path ---
    leadfield_path = None
    for section in ("spatial_filter", "advanced_preprocessing", "preprocessing"):
        sect = config.get(section) or {}
        leadfield_path = sect.get("leadfield_path") or sect.get("gedai", {}).get("leadfield_path")
        if leadfield_path:
            break
    if leadfield_path:
        p = Path(leadfield_path)
        if p.exists() and (p.suffix in (".npy", ".pt", ".mat") or p.is_file()):
            cap.gedai_supported = True
        else:
            cap.gedai_supported = False
            cap.reason["gedai"] = "Leadfield path not found or not a file"
    else:
        cap.reason["gedai"] = "No leadfield matrix found"

    return cap
