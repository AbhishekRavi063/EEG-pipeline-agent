"""
Thread-safe pub-sub for streaming and GUI synchronization.
Producers (pipeline/streaming) publish; GUI subscribes without blocking evaluation.
"""

import logging
import queue
import threading
from collections import defaultdict
from typing import Any, Callable

logger = logging.getLogger(__name__)


class PubSub:
    """
    Thread-safe publish/subscribe. Subscribers are called in publisher thread;
    for GUI, publish to a queue and let GUI thread drain it (see subscribe_queue).
    """

    def __init__(self) -> None:
        self._subs: dict[str, list[Callable[..., None]]] = defaultdict(list)
        self._lock = threading.RLock()

    def subscribe(self, topic: str, callback: Callable[..., None]) -> None:
        with self._lock:
            self._subs[topic].append(callback)

    def unsubscribe(self, topic: str, callback: Callable[..., None]) -> None:
        with self._lock:
            if callback in self._subs[topic]:
                self._subs[topic].remove(callback)

    def publish(self, topic: str, *args: Any, **kwargs: Any) -> None:
        with self._lock:
            callbacks = list(self._subs[topic])
        for cb in callbacks:
            try:
                cb(*args, **kwargs)
            except Exception as e:
                logger.debug("PubSub callback %s failed: %s", topic, e)

    def clear_topic(self, topic: str) -> None:
        with self._lock:
            self._subs[topic].clear()


# Standard topics for BCI platform
TOPIC_RAW_EEG = "eeg/raw"
TOPIC_FILTERED_EEG = "eeg/filtered"
TOPIC_PREDICTION = "eeg/prediction"
TOPIC_LATENCY = "eeg/latency_ms"
TOPIC_PIPELINE_METRICS = "pipelines/metrics"
TOPIC_DRIFT_ALERT = "agent/drift"


def subscribe_queue(topic: str, pubsub: PubSub, q: queue.Queue[Any]) -> None:
    """Subscribe to topic and push messages into queue (for GUI thread to drain)."""
    def put(*args: Any, **kwargs: Any) -> None:
        q.put((args, kwargs))
    pubsub.subscribe(topic, put)
