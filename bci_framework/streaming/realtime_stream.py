"""
Real-time EEG stream: LSL / OpenBCI / Emotiv (placeholder).

For production: connect to actual hardware or LSL. Pipeline uses same get_chunk() API.
Total latency target < 300 ms; precomputed filters only (matrix mult).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .base_stream import EEGStream
from .buffer import CircularRingBuffer

logger = logging.getLogger(__name__)


class RealTimeEEGStream(EEGStream):
    """
    Real-time stream: reads from ring buffer filled by LSL/OpenBCI/Emotiv.
    get_chunk() returns latest window from buffer. Placeholder: can be fed
    by replay or simulated data until hardware is connected.
    """

    def __init__(
        self,
        n_channels: int,
        fs: float,
        window_size_sec: float = 1.0,
        buffer_length_sec: float = 10.0,
        channel_names: list[str] | None = None,
    ) -> None:
        self._n_channels = n_channels
        self.fs_ = fs
        self.window_size_sec = window_size_sec
        self.channel_names_ = channel_names or []
        self._buffer = CircularRingBuffer(
            n_channels=n_channels,
            buffer_length_sec=buffer_length_sec,
            fs=fs,
        )
        self._window_samples = int(fs * window_size_sec)

    @property
    def n_channels(self) -> int:
        return self._n_channels

    @property
    def fs(self) -> float:
        return self.fs_

    def push_samples(self, samples: np.ndarray) -> None:
        """
        Push new samples from hardware. samples: (n_channels, n_samples).
        Called by LSL/OpenBCI callback or replay script.
        """
        self._buffer.append(samples)

    def get_chunk(self) -> np.ndarray | None:
        """Return latest causal window if enough data in buffer."""
        return self._buffer.get_window(self.window_size_sec)

    def get_metadata(self) -> dict[str, Any]:
        return {"channel_names": self.channel_names_}

    def clear(self) -> None:
        self._buffer.clear()


# Placeholder for future LSL integration
def create_lsl_stream(
    stream_name: str | None = None,
    window_size_sec: float = 1.0,
    buffer_length_sec: float = 10.0,
) -> RealTimeEEGStream | None:
    """
    Create real-time stream from Lab Streaming Layer. Returns None if LSL not available.
    Future: pip install pylsl, resolve stream, wrap in RealTimeEEGStream.
    """
    try:
        import pylsl
    except ImportError:
        logger.info("pylsl not installed; real-time LSL stream unavailable")
        return None
    # Resolve EEG stream
    streams = pylsl.resolve_streams()
    eeg_streams = [s for s in streams if "EEG" in s.name() or "eeg" in s.name().lower()]
    if not eeg_streams:
        logger.warning("No EEG LSL stream found")
        return None
    info = eeg_streams[0]
    n_ch = info.channel_count()
    fs = info.nominal_srate() or 250.0
    stream = RealTimeEEGStream(
        n_channels=n_ch,
        fs=fs,
        window_size_sec=window_size_sec,
        buffer_length_sec=buffer_length_sec,
        channel_names=list(info.desc().child("channels").child("channel").values()),
    )
    # TODO: start background thread that pulls from pylsl and calls stream.push_samples()
    logger.info("LSL stream placeholder: connect push_samples() to LSL inlet")
    return stream
