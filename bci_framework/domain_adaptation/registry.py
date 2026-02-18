"""Registry for domain adaptation adapters (transfer learning)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .base_adapter import DomainAdapter
from .zscore_adapter import ZScoreAdapter
from .coral_adapter import CORALAdapter
from .riemann_transport_adapter import RiemannianTransportAdapter

logger = logging.getLogger(__name__)


class NoAdapter(DomainAdapter):
    """Pass-through (no adaptation). Still enforces 2D contract."""

    name = "none"

    def fit(
        self,
        X_source: np.ndarray,
        X_target: Any = None,
    ) -> "NoAdapter":
        logger.info("[TRANSFER] Adapter fit called")
        self._validate_input(X_source)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._validate_input(X)
        return np.asarray(X, dtype=np.float64)


ADAPTER_REGISTRY: dict[str, type[DomainAdapter]] = {
    "none": NoAdapter,
    "zscore": ZScoreAdapter,
    "coral": CORALAdapter,
    "riemann_transport": RiemannianTransportAdapter,
}


def get_adapter(name: str, feature_name: str | None = None, **kwargs: Any) -> DomainAdapter:
    """Factory: get domain adapter by name."""
    cls = ADAPTER_REGISTRY.get((name or "none").lower())
    if cls is None:
        raise KeyError(
            f"Unknown domain adapter '{name}'. Available: {list(ADAPTER_REGISTRY.keys())}"
        )
    if feature_name and hasattr(cls, "feature_name"):
        return cls(feature_name=feature_name, **kwargs)
    return cls(**kwargs)
