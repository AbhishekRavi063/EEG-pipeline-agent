"""
Unified EEG streaming API: abstract base for offline and real-time sources.

Pipeline operates on continuous causal windows from get_chunk().
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generator

import numpy as np


class EEGStream(ABC):
    """
    Abstract stream: yields chunks for pipeline processing.
    get_chunk() returns (n_channels, n_samples) or (1, n_channels, n_samples).
    """

    @property
    def n_channels(self) -> int:
        """Number of EEG channels."""
        raise NotImplementedError

    @property
    def fs(self) -> float:
        """Sampling rate in Hz."""
        raise NotImplementedError

    @abstractmethod
    def get_chunk(self) -> np.ndarray | None:
        """
        Return next causal window: (n_channels, window_samples) or
        (1, n_channels, window_samples). None if no data / stream ended.
        """
        pass

    def iter_chunks(self) -> Generator[np.ndarray, None, None]:
        """Yield chunks until stream ends."""
        while True:
            chunk = self.get_chunk()
            if chunk is None:
                break
            yield chunk

    def get_metadata(self) -> dict[str, Any]:
        """Optional: channel names, subject id, label for current chunk."""
        return {}
