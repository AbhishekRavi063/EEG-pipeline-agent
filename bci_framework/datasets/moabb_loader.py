"""
MOABB dataset adapter: wrap MOABB datasets into DatasetLoader API.

Supports Motor Imagery, P300, SSVEP paradigms. Normalizes channel layouts,
sampling rates, and labels for cross-dataset AutoML benchmarking.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from .base import EEGDataset, DatasetLoader
from bci_framework.utils.capability_checker import detect_spatial_capabilities

logger = logging.getLogger(__name__)

MOABB_AVAILABLE = False
try:
    import moabb
    from moabb.datasets import base as moabb_base
    from moabb.paradigms import MotorImagery, P300, SSVEP
    MOABB_AVAILABLE = True
except ImportError:
    pass


class MOABBDatasetLoader(DatasetLoader):
    """
    Adapter: load MOABB datasets via paradigm.get_data() and convert to EEGDataset.
    Enables cross-dataset benchmarking and consistent interface with BCI_IV_2a etc.
    """

    name = "MOABB"
    default_data_dir = "./data/moabb"

    def __init__(
        self,
        dataset_name: str = "BNCI2014_001",
        paradigm: str = "motor_imagery",
        **paradigm_kwargs: Any,
    ) -> None:
        """
        dataset_name: MOABB dataset class name (e.g. BNCI2014001, PhysionetMI).
        paradigm: "motor_imagery" | "p300" | "ssvep".
        paradigm_kwargs: fmin, fmax, tmin, tmax, n_classes, etc.
        """
        if not MOABB_AVAILABLE:
            raise ImportError("MOABB not installed. pip install moabb")
        self.dataset_name = dataset_name
        self.paradigm_name = paradigm.lower()
        self.paradigm_kwargs = paradigm_kwargs
        self._dataset_class: Any = None
        self._paradigm_instance: Any = None

    def _get_dataset(self) -> Any:
        if self._dataset_class is not None:
            return self._dataset_class
        try:
            from moabb import datasets
            # Support both BNCI2014_001 and BNCI2014001 (legacy)
            name = self.dataset_name.replace(" ", "_")
            self._dataset_class = getattr(datasets, name, None)
            if self._dataset_class is None and "_" not in name and len(name) >= 10:
                # Try with underscores: BNCI2014001 -> BNCI2014_001
                alt = name[:8] + "_" + name[8:]
                self._dataset_class = getattr(datasets, alt, None)
        except Exception:
            pass
        if self._dataset_class is None:
            raise ValueError(
                f"Unknown MOABB dataset: {self.dataset_name}. "
                "Use a class name from moabb.datasets (e.g. BNCI2014_001, PhysionetMI)."
            )
        return self._dataset_class

    def _get_paradigm(self) -> Any:
        if self._paradigm_instance is not None:
            return self._paradigm_instance
        if self.paradigm_name == "motor_imagery":
            self._paradigm_instance = MotorImagery(**self.paradigm_kwargs)
        elif self.paradigm_name == "p300":
            self._paradigm_instance = P300(**self.paradigm_kwargs)
        elif self.paradigm_name == "ssvep":
            self._paradigm_instance = SSVEP(**self.paradigm_kwargs)
        else:
            raise ValueError(f"Unknown paradigm: {self.paradigm_name}")
        return self._paradigm_instance

    def get_subject_ids(self) -> list[int]:
        ds_cls = self._get_dataset()
        return ds_cls().subject_list

    def load(
        self,
        data_dir: str | Path | None = None,
        subjects: list[int] | list[str] | None = None,
        download_if_missing: bool = True,
        **kwargs: Any,
    ) -> EEGDataset | dict[int | str, EEGDataset]:
        paradigm = self._get_paradigm()
        dataset = self._get_dataset()()
        if subjects is not None:
            subject_list = [int(s) for s in subjects]
        else:
            subject_list = dataset.subject_list

        try:
            X, y, meta = paradigm.get_data(dataset, subject_list)
        except Exception as e:
            logger.exception("MOABB get_data failed: %s", e)
            raise

        # X: (n_trials, n_channels, n_samples) from MotorImagery
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 2:
            X = X[np.newaxis, ...]
        y_raw = np.asarray(y).ravel()
        # MOABB may return string labels (e.g. 'left_hand', 'feet'); map to int 0,1,...
        uniq = np.unique(y_raw)
        label_to_int = None
        if uniq.dtype.kind in ("U", "O", "S") or not np.issubdtype(uniq.dtype, np.integer):
            label_to_int = {v: i for i, v in enumerate(sorted(uniq.tolist()))}
            y = np.array([label_to_int[v] for v in y_raw], dtype=np.int64)
        else:
            y = np.asarray(y_raw, dtype=np.int64)

        if hasattr(meta, "columns") and "subject" in meta.columns:
            subject_ids = np.asarray(meta["subject"].values, dtype=int)
        else:
            subject_ids = np.zeros(X.shape[0], dtype=int)

        fs = getattr(paradigm, "resample", None) or 250.0
        channel_names = getattr(paradigm, "channels", None)
        if channel_names is None:
            channel_names = [f"Ch{i}" for i in range(X.shape[1])]
        if label_to_int is not None:
            class_names = [str(k) for k in sorted(label_to_int.keys(), key=lambda k: label_to_int[k])]
        else:
            class_names = [str(c) for c in np.unique(y)]

        # v3.1: detect spatial capabilities once (MOABB has no Raw/montage)
        config = kwargs.get("config") or {}
        self.capabilities = detect_spatial_capabilities(raw=None, channel_names=list(channel_names), config=config)

        def _make_ds(data: np.ndarray, labels: np.ndarray, subject_id: Any, subj_ids: np.ndarray | None) -> EEGDataset:
            return EEGDataset(
                data=data,
                labels=labels,
                fs=float(fs),
                channel_names=list(channel_names),
                class_names=class_names,
                subject_id=subject_id,
                subject_ids_per_trial=subj_ids,
                capabilities=self.capabilities,
            )

        if len(subject_list) == 1:
            return _make_ds(
                X, y,
                subject_list[0],
                subject_ids if np.any(subject_ids != 0) else None,
            )

        has_subject_ids = len(np.unique(subject_ids)) > 1 or np.any(subject_ids != 0)
        if not has_subject_ids:
            return _make_ds(X, y, None, None)

        out: dict[int | str, EEGDataset] = {}
        for sid in subject_list:
            mask = subject_ids == sid
            if not np.any(mask):
                continue
            out[sid] = _make_ds(X[mask], y[mask], sid, None)
        if len(out) == 1:
            return next(iter(out.values()))
        return out  # type: ignore
