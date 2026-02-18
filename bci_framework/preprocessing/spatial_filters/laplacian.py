"""Surface Laplacian spatial filter (local average reference)."""

from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping

import numpy as np

from .base import SpatialFilterBase

logger = logging.getLogger(__name__)


class LaplacianSpatialFilter(SpatialFilterBase):
    """
    Surface Laplacian: subtract local ring average per channel.
    Neighbours: channel_index -> list of neighbour indices. Online-safe.
    """

    name = "laplacian"

    def __init__(
        self,
        fs: float,
        neighbours: Mapping[int, Iterable[int]] | Mapping[str, Iterable[str]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(fs, neighbours=neighbours, **kwargs)
        self._neighbour_spec = neighbours
        self._neighbours: dict[int, list[int]] | None = None
        self._channel_index: dict[str, int] | None = None
        self._warned_fallback = False

    def set_channel_names(self, channel_names: list[str]) -> None:
        self._channel_index = {name: idx for idx, name in enumerate(channel_names)}
        if self._neighbour_spec and channel_names:
            self._neighbours = self._resolve_neighbours(self._neighbour_spec)

    def fit(
        self, X: np.ndarray, y: np.ndarray | None = None, info: dict | None = None
    ) -> "LaplacianSpatialFilter":
        if self._neighbours is None and self._neighbour_spec:
            self._neighbours = self._resolve_neighbours(
                self._neighbour_spec, n_channels=X.shape[1]
            )
        self._fitted = True
        return self

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
                indices = [
                    int(n)
                    for n in neigh
                    if 0 <= int(n) and (n_channels is None or int(n) < n_channels)
                ]
                if indices:
                    resolved[int(idx)] = indices
            return resolved
        if self._channel_index is None:
            return resolved
        for name, neigh in spec.items():
            if name not in self._channel_index:
                continue
            indices = [
                self._channel_index[n]
                for n in neigh
                if self._channel_index.get(n) is not None
            ]
            if indices:
                resolved[self._channel_index[name]] = indices
        return resolved

    def transform(self, X: np.ndarray) -> np.ndarray:
        data = np.asarray(X, dtype=np.float64)
        neighbours = self._neighbours
        if neighbours is None:
            if not self._warned_fallback:
                logger.warning(
                    "LaplacianSpatialFilter: no neighbours — falling back to CAR."
                )
                self._warned_fallback = True
            mean = data.mean(axis=1, keepdims=True)
            return data - mean
        out = np.empty_like(data, dtype=np.float64)
        for ch_idx in range(data.shape[1]):
            neigh = neighbours.get(ch_idx)
            if not neigh:
                out[:, ch_idx, :] = data[:, ch_idx, :] - data.mean(axis=1)
                continue
            neigh_vals = np.mean(data[:, neigh, :], axis=1)
            out[:, ch_idx, :] = data[:, ch_idx, :] - neigh_vals
        return out
