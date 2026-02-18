"""Domain adaptation (transfer learning) for cross-subject BCI — offline, modular, optional."""

from .base_adapter import DomainAdapter
from .zscore_adapter import ZScoreAdapter
from .coral_adapter import CORALAdapter
from .riemann_transport_adapter import RiemannianTransportAdapter
from .registry import NoAdapter, ADAPTER_REGISTRY, get_adapter

__all__ = [
    "DomainAdapter",
    "NoAdapter",
    "ZScoreAdapter",
    "CORALAdapter",
    "RiemannianTransportAdapter",
    "ADAPTER_REGISTRY",
    "get_adapter",
]
