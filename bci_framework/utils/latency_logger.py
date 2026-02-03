"""
Pipeline runtime / latency logging per window.
Tracks milliseconds per window to enforce latency budgets dynamically.
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LatencyRecord:
    """Single latency measurement."""
    pipeline_name: str
    latency_ms: float
    window_index: int
    timestamp: float = field(default_factory=time.perf_counter)


class PipelineLatencyLogger:
    """
    Thread-safe latency logger per pipeline. Keeps rolling window of last N
    measurements; supports percentiles and budget checks.
    """

    def __init__(self, window_size: int = 100) -> None:
        self.window_size = window_size
        self._records: dict[str, deque[LatencyRecord]] = {}
        self._lock = threading.RLock()

    def log(self, pipeline_name: str, latency_ms: float, window_index: int = 0) -> None:
        with self._lock:
            if pipeline_name not in self._records:
                self._records[pipeline_name] = deque(maxlen=self.window_size)
            self._records[pipeline_name].append(
                LatencyRecord(pipeline_name=pipeline_name, latency_ms=latency_ms, window_index=window_index)
            )

    def get_stats(self, pipeline_name: str) -> dict[str, float] | None:
        import numpy as np
        with self._lock:
            recs = self._records.get(pipeline_name)
            if not recs:
                return None
            ms = [r.latency_ms for r in recs]
            return {
                "mean_ms": float(np.mean(ms)),
                "p50_ms": float(np.percentile(ms, 50)),
                "p95_ms": float(np.percentile(ms, 95)),
                "p99_ms": float(np.percentile(ms, 99)),
                "max_ms": float(np.max(ms)),
                "n_samples": len(ms),
            }

    def exceeds_budget(self, pipeline_name: str, budget_ms: float) -> bool:
        stats = self.get_stats(pipeline_name)
        if stats is None:
            return False
        return stats["p95_ms"] > budget_ms or stats["mean_ms"] > budget_ms

    def summary(self) -> dict[str, dict[str, float]]:
        with self._lock:
            return {name: self.get_stats(name) or {} for name in self._records}
