"""Base interface for EEG dataset loaders."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Iterator

import numpy as np


@dataclass
class EEGDataset:
    """Container for loaded EEG data and labels."""

    data: np.ndarray  # (n_trials, n_channels, n_samples)
    labels: np.ndarray  # (n_trials,) class indices; use -1 for unlabeled (e.g. E session without markers)
    fs: float
    channel_names: list[str]
    class_names: list[str]
    subject_id: str | int | None = None
    trial_metadata: list[dict] | None = None
    subject_ids_per_trial: np.ndarray | None = None  # (n_trials,) for LOSO
    n_trials_from_t: int | None = None  # first N trials from T session (rest from E); for T/E indicator in UI

    @property
    def n_trials(self) -> int:
        return self.data.shape[0]

    @property
    def n_channels(self) -> int:
        return self.data.shape[1]

    @property
    def n_samples(self) -> int:
        return self.data.shape[2]

    def __len__(self) -> int:
        return self.n_trials

    def iter_trials(self) -> Iterator[tuple[np.ndarray, int]]:
        """Yield (trial_data, label) for each trial."""
        for i in range(self.n_trials):
            yield self.data[i], self.labels[i]

    def get_trial_data(self, trial_id: int) -> tuple[np.ndarray, int]:
        """Return (trial_data (n_channels, n_samples), label) for trial_id."""
        if trial_id < 0 or trial_id >= self.n_trials:
            raise IndexError(f"trial_id {trial_id} out of range [0, {self.n_trials})")
        return self.data[trial_id], int(self.labels[trial_id])

    def stream_chunk(
        self,
        trial_id: int,
        window_samples: int,
        overlap_ratio: float = 0.5,
    ) -> Generator[tuple[np.ndarray, int], None, None]:
        """Stream one trial as overlapping chunks (n_channels, window_samples), label."""
        trial_data, label = self.get_trial_data(trial_id)
        n_samples = trial_data.shape[1]
        step = max(1, int(window_samples * (1 - overlap_ratio)))
        start = 0
        while start + window_samples <= n_samples:
            chunk = trial_data[:, start : start + window_samples].copy()
            yield chunk, label
            start += step


class DatasetLoader(ABC):
    """Abstract base for dataset loaders. Add new datasets by implementing this interface."""

    name: str = "base"
    default_data_dir: str = "./data"

    @abstractmethod
    def load(
        self,
        data_dir: str | Path | None = None,
        subjects: list[int] | list[str] | None = None,
        download_if_missing: bool = True,
    ) -> EEGDataset | dict[int | str, EEGDataset]:
        """
        Load dataset. Returns single EEGDataset or dict of subject_id -> EEGDataset
        depending on evaluation mode.
        """
        pass

    @abstractmethod
    def get_subject_ids(self) -> list[int] | list[str]:
        """Return list of available subject identifiers."""
        pass

    def get_trial_data(
        self,
        dataset: EEGDataset,
        subject_id: int | str | None,
        trial_id: int,
    ) -> tuple[np.ndarray, int]:
        """Return (trial_data, label). Default uses dataset.get_trial_data(trial_id)."""
        return dataset.get_trial_data(trial_id)

    def stream_chunk(
        self,
        dataset: EEGDataset,
        trial_id: int,
        window_size_samples: int,
        overlap_ratio: float = 0.5,
    ) -> Generator[tuple[np.ndarray, int], None, None]:
        """Stream trial as overlapping chunks. Default uses dataset.stream_chunk."""
        yield from dataset.stream_chunk(trial_id, window_size_samples, overlap_ratio)
