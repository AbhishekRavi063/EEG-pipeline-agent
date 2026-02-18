"""
Offline EEG stream: datasets, MOABB, or preloaded arrays.

Sliding window over trials with configurable window_size and overlap.
"""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np

from .base_stream import EEGStream


class OfflineEEGStream(EEGStream):
    """
    Stream EEG from offline dataset (e.g. DatasetLoader, MOABB adapter).
    Yields causal overlapping windows (window_size_sec, overlap_sec).
    """

    def __init__(
        self,
        data: np.ndarray,
        fs: float,
        window_size_sec: float = 1.0,
        overlap_sec: float = 0.5,
        channel_names: list[str] | None = None,
        labels: np.ndarray | None = None,
        trial_boundaries: list[tuple[int, int]] | None = None,
    ) -> None:
        """
        data: (n_trials, n_channels, n_samples) or (n_channels, n_samples).
        If 2D, treated as single continuous block.
        trial_boundaries: list of (start_sample, end_sample) for 2D data.
        """
        self._data = np.asarray(data, dtype=np.float64)
        self.fs_ = fs
        self.window_size_sec = window_size_sec
        self.overlap_sec = overlap_sec
        self.channel_names_ = channel_names or []
        self.labels_ = labels  # (n_trials,) if data is 3D
        self.trial_boundaries_ = trial_boundaries

        self._window_samples = int(fs * window_size_sec)
        self._step_samples = max(1, int(fs * (window_size_sec - overlap_sec)))
        self._n_channels = self._data.shape[1] if self._data.ndim >= 2 else self._data.shape[0]
        if self._data.ndim == 2:
            self._data = self._data[np.newaxis, ...]  # (1, n_ch, n_samp)
        self._n_trials = self._data.shape[0]
        self._trial_idx = 0
        self._start = 0
        self._current_label: int | None = None

    @property
    def n_channels(self) -> int:
        return self._n_channels

    @property
    def fs(self) -> float:
        return self.fs_

    def get_chunk(self) -> np.ndarray | None:
        while self._trial_idx < self._n_trials:
            trial_data = self._data[self._trial_idx]
            n_samp = trial_data.shape[1]
            if self._start + self._window_samples <= n_samp:
                chunk = trial_data[:, self._start : self._start + self._window_samples].copy()
                self._current_label = (
                    int(self.labels_[self._trial_idx])
                    if self.labels_ is not None and self._trial_idx < len(self.labels_)
                    else None
                )
                self._start += self._step_samples
                # (n_channels, window_samples)
                return chunk
            self._trial_idx += 1
            self._start = 0
        return None

    def get_metadata(self) -> dict[str, Any]:
        out = {"channel_names": self.channel_names_}
        if self._current_label is not None:
            out["label"] = self._current_label
        return out

    def iter_trials(self) -> Iterator[tuple[np.ndarray, int | None]]:
        """Yield (trial_data (n_channels, n_samples), label) for each trial."""
        for i in range(self._n_trials):
            label = int(self.labels_[i]) if self.labels_ is not None and i < len(self.labels_) else None
            yield self._data[i], label
