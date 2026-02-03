"""Referencing utilities for mandatory EEG preprocessing."""

from __future__ import annotations

import logging
from typing import Iterable, Mapping

import numpy as np

from .base import PreprocessingBase

logger = logging.getLogger(__name__)


class CommonAverageReference(PreprocessingBase):
    """Subtract the average across all channels (CAR)."""

    name = "common_average_reference"

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "CommonAverageReference":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        data = np.asarray(X, dtype=np.float64)
        mean = data.mean(axis=1, keepdims=True)
        return data - mean


class LaplacianReference(PreprocessingBase):
    """
    Simple surface Laplacian approximation.

    If neighbours are not provided, falls back to CAR while emitting a warning.
    Neighbours can be supplied either as channel-index keys (int -> list[int])
    or channel-name keys (str -> list[str]). When names are provided a mapping
    from names to indices must be supplied via `set_channel_names`.
    """

    name = "laplacian_reference"

    def __init__(
        self,
        fs: float,
        neighbours: Mapping[int, Iterable[int]] | Mapping[str, Iterable[str]] | None = None,
    ) -> None:
        super().__init__(fs, neighbours=neighbours)
        self._neighbour_spec = neighbours
        self._neighbours: dict[int, list[int]] | None = None
        self._channel_index: dict[str, int] | None = None
        self._warned_fallback = False

    def set_channel_names(self, names: list[str]) -> None:
        """Optional: provide channel-name mapping for neighbour lookup."""
        self._channel_index = {name: idx for idx, name in enumerate(names)}
        if isinstance(self._neighbour_spec, Mapping) and names:
            self._neighbours = self._resolve_neighbours(self._neighbour_spec)

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "LaplacianReference":
        if self._neighbours is None and isinstance(self._neighbour_spec, Mapping):
            # Attempt to resolve using channel count (assume indices)
            self._neighbours = self._resolve_neighbours(self._neighbour_spec, n_channels=X.shape[1])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        data = np.asarray(X, dtype=np.float64)
        neighbours = self._neighbours
        if neighbours is None:
            if not self._warned_fallback:
                logger.warning(
                    "LaplacianReference: neighbours not provided — falling back to common-average reference."
                )
                self._warned_fallback = True
            mean = data.mean(axis=1, keepdims=True)
            return data - mean

        out = np.empty_like(data, dtype=np.float64)
        global_mean = data.mean(axis=1)
        for ch_idx in range(data.shape[1]):
            neigh = neighbours.get(ch_idx)
            if not neigh:
                # If no neighbours defined, treat as CAR for that channel
                out[:, ch_idx, :] = data[:, ch_idx, :] - global_mean
                continue
            neigh_vals = np.mean(data[:, neigh, :], axis=1)
            out[:, ch_idx, :] = data[:, ch_idx, :] - neigh_vals
        return out

    def _resolve_neighbours(
        self,
        spec: Mapping[int, Iterable[int]] | Mapping[str, Iterable[str]],
        n_channels: int | None = None,
    ) -> dict[int, list[int]]:
        resolved: dict[int, list[int]] = {}
        if not spec:
            return resolved

        if all(isinstance(k, int) for k in spec.keys()):
            for idx, neigh in spec.items():
                idx_int = int(idx)
                indices = [int(n) for n in neigh if 0 <= int(n) and (n_channels is None or int(n) < n_channels)]
                if indices:
                    resolved[idx_int] = indices
            return resolved

        if self._channel_index is None:
            logger.debug("LaplacianReference: channel names not set; cannot resolve neighbours by name yet.")
            return resolved

        for name, neigh in spec.items():
            if name not in self._channel_index:
                continue
            idx = self._channel_index[name]
            indices = []
            for n in neigh:
                idx_neigh = self._channel_index.get(n)
                if idx_neigh is not None:
                    indices.append(idx_neigh)
            if indices:
                resolved[idx] = indices
        return resolved
