"""
Real-time EEG streaming: buffer and sliding window.
EEG Stream Buffer → Sliding Window → Pipeline Inference → GUI Output.
"""

import logging
from collections import deque
from typing import Any, Generator

import numpy as np

logger = logging.getLogger(__name__)


class EEGStreamBuffer:
    """
    Ring buffer for continuous EEG samples.
    Append channel-wise samples; read windows for pipeline inference.
    """

    def __init__(
        self,
        n_channels: int,
        max_samples: int,
        dtype: type = np.float64,
    ) -> None:
        self.n_channels = n_channels
        self.max_samples = max_samples
        self.dtype = dtype
        self._buffer = np.zeros((n_channels, max_samples), dtype=dtype)
        self._n_valid = 0

    def append(self, chunk: np.ndarray) -> None:
        """Append chunk (n_channels, n_new_samples)."""
        if chunk.ndim == 1:
            chunk = chunk.reshape(-1, 1)
        if chunk.shape[0] != self.n_channels:
            raise ValueError("Chunk channels mismatch")
        n_new = chunk.shape[1]
        if n_new >= self.max_samples:
            self._buffer[:] = chunk[:, -self.max_samples:]
            self._n_valid = self.max_samples
            return
        if self._n_valid + n_new <= self.max_samples:
            self._buffer[:, self._n_valid : self._n_valid + n_new] = chunk
            self._n_valid += n_new
        else:
            shift = self._n_valid + n_new - self.max_samples
            self._buffer[:, :-n_new] = self._buffer[:, shift : shift + (self.max_samples - n_new)]
            self._buffer[:, -n_new:] = chunk
            self._n_valid = self.max_samples

    def get_window(self, n_samples: int) -> np.ndarray | None:
        """Get last n_samples as (n_channels, n_samples). None if not enough data."""
        if self._n_valid < n_samples:
            return None
        return self._buffer[:, self._n_valid - n_samples : self._n_valid].copy()

    def clear(self) -> None:
        self._n_valid = 0


def sliding_window_chunks(
    data: np.ndarray,
    window_samples: int,
    overlap_ratio: float = 0.5,
    axis: int = -1,
) -> Generator[tuple[np.ndarray, int, int], None, None]:
    """
    Yield overlapping windows over the last axis (time).
    data: (..., n_samples) e.g. (n_trials, n_channels, n_samples).
    overlap_ratio: 0.5 = 50% overlap.
    Yields (chunk, start_idx, end_idx).
    """
    n_samples = data.shape[axis]
    step = max(1, int(window_samples * (1 - overlap_ratio)))
    start = 0
    while start + window_samples <= n_samples:
        if axis == -1:
            chunk = data[..., start : start + window_samples].copy()
        else:
            sl = [slice(None)] * data.ndim
            sl[axis] = slice(start, start + window_samples)
            chunk = data[tuple(sl)].copy()
        yield chunk, start, start + window_samples
        start += step


def stream_chunk(
    trial_data: np.ndarray,
    window_samples: int,
    overlap_ratio: float = 0.5,
) -> Generator[np.ndarray, None, None]:
    """
    Stream one trial as overlapping chunks for real-time simulation.
    trial_data: (n_channels, n_samples).
    Yields chunks (n_channels, window_samples).
    """
    n_samples = trial_data.shape[1]
    step = max(1, int(window_samples * (1 - overlap_ratio)))
    start = 0
    while start + window_samples <= n_samples:
        yield trial_data[:, start : start + window_samples].copy()
        start += step
