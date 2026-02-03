"""BCI Competition IV Dataset 2a loader (4-class motor imagery, 22 channels, 250 Hz)."""

import logging
from pathlib import Path
from typing import Any

import numpy as np

from .base import EEGDataset, DatasetLoader

logger = logging.getLogger(__name__)

# BCI IV 2a trigger codes (GDF)
CLASS_TRIGGERS = [769, 770, 771, 772]  # left hand, right hand, both feet, tongue
CLASS_NAMES = ["left_hand", "right_hand", "feet", "tongue"]
N_EEG_CHANNELS = 22
FS = 250.0


class BCICompetitionIV2aLoader(DatasetLoader):
    """
    Load BCI Competition IV Dataset 2a.
    Supports GDF (via MNE) or pre-converted mat/numpy.
    """

    name = "BCI_IV_2a"
    default_data_dir = "./data/BCI_IV_2a"
    subject_ids = list(range(1, 10))  # 9 subjects

    def get_subject_ids(self) -> list[int]:
        return list(self.subject_ids)

    def load(
        self,
        data_dir: str | Path | None = None,
        subjects: list[int] | list[str] | None = None,
        download_if_missing: bool = True,
        trial_duration_seconds: float | None = None,
        **kwargs: Any,
    ) -> EEGDataset | dict[int | str, EEGDataset]:
        data_path = Path(data_dir or self.default_data_dir)
        subject_list = subjects or self.subject_ids
        trial_sec = trial_duration_seconds if trial_duration_seconds is not None else 3.0

        if download_if_missing and not data_path.exists():
            self._download(data_path)

        result: dict[int, EEGDataset] = {}
        for subj in subject_list:
            sid = int(subj) if isinstance(subj, str) and subj.isdigit() else subj
            ds = self._load_subject(data_path, sid, trial_duration_seconds=trial_sec)
            if ds is not None:
                result[sid] = ds

        if len(result) == 1:
            return next(iter(result.values()))
        return result  # type: ignore

    def _load_subject(self, data_path: Path, subject_id: int, trial_duration_seconds: float = 3.0) -> EEGDataset | None:
        # Try GDF first (A01T.gdf, A01E.gdf)
        prefix = f"A{subject_id:02d}"
        train_gdf = data_path / f"{prefix}T.gdf"
        test_gdf = data_path / f"{prefix}E.gdf"

        if train_gdf.exists():
            return self._load_gdf_subject(train_gdf, test_gdf if test_gdf.exists() else train_gdf, subject_id, trial_duration_seconds)

        # Fallback: try mat files (e.g. from BNCI or converted)
        train_mat = data_path / f"{prefix}T.mat"
        if train_mat.exists():
            return self._load_mat_subject(data_path, subject_id)

        logger.warning("No data found for subject %s in %s", subject_id, data_path)
        return None

    def _load_gdf_subject(
        self, train_path: Path, test_path: Path, subject_id: int, trial_duration_seconds: float = 3.0
    ) -> EEGDataset | None:
        try:
            import mne
        except ImportError:
            logger.error("MNE is required for GDF. Install: pip install mne")
            return None

        all_data: list[np.ndarray] = []
        all_labels: list[int] = []
        loaded_sessions: list[tuple[str, int]] = []  # (filename, n_trials) for logging

        for path in (train_path, test_path):
            if not path.exists():
                logger.info("Session file not found: %s (skipped)", path.name)
                continue
            n_before = len(all_data)
            raw = mne.io.read_raw_gdf(str(path), preload=True, verbose=False)
            # Use only EEG channels (first 22)
            picks = mne.pick_types(raw.info, eeg=True, exclude="bads")[:N_EEG_CHANNELS]
            if len(picks) < N_EEG_CHANNELS:
                picks = list(range(N_EEG_CHANNELS))
            data, times = raw.get_data(picks=picks, return_times=True)
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            if events.size == 0:
                # Try stimulus channel (raw trigger codes 769, 770, 771, 772)
                stim = raw.get_data(picks="stim") if "stim" in raw.ch_names else None
                if stim is not None and stim.size > 0:
                    events = _events_from_stim_channel(np.squeeze(stim), raw.info["sfreq"])
            n_events_total = events.shape[0] if events.size else 0
            n_class_events = 0
            for event in events:
                trigger = int(event[2])
                # MNE may remap 769,770,771,772 to 1,2,3,4; accept both
                if trigger in CLASS_TRIGGERS:
                    class_idx = CLASS_TRIGGERS.index(trigger)
                elif 1 <= trigger <= 4:
                    class_idx = trigger - 1
                else:
                    continue
                n_class_events += 1
                t_start = int(event[0])
                n_samp = int(trial_duration_seconds * raw.info["sfreq"])
                t_end = min(t_start + n_samp, data.shape[1])
                if t_end - t_start >= int(1.5 * raw.info["sfreq"]):
                    trial = data[:, t_start:t_end].astype(np.float64)
                    if trial.shape[1] < int(2 * FS):
                        pad = int(2 * FS) - trial.shape[1]
                        trial = np.pad(trial, ((0, 0), (0, pad)), mode="edge")
                    all_data.append(trial)
                    all_labels.append(class_idx)
            if n_events_total == 0:
                # E (evaluation) file often has no markers; segment continuous EEG into fixed-length unlabeled trials
                sfreq = float(raw.info["sfreq"])
                n_samp = int(trial_duration_seconds * sfreq)
                step = n_samp  # non-overlapping windows
                for start in range(0, data.shape[1] - n_samp + 1, step):
                    trial = data[:, start : start + n_samp].astype(np.float64)
                    all_data.append(trial)
                    all_labels.append(-1)  # -1 = unlabeled (no ground truth)
                n_from_file = len(all_data) - n_before
                loaded_sessions.append((path.name, n_from_file))
                logger.info(
                    "  %s: 0 events → segmented into %d unlabeled trials (%.1f s windows, pipeline will predict only)",
                    path.name, n_from_file, trial_duration_seconds,
                )
            else:
                n_from_file = len(all_data) - n_before
                loaded_sessions.append((path.name, n_from_file))
                logger.info(
                    "  %s: %d events, %d class triggers → %d trials",
                    path.name, n_events_total, n_class_events, n_from_file,
                )

        if loaded_sessions:
            logger.info(
                "Loaded subject %s: %s",
                subject_id,
                ", ".join("%s (%d trials)" % (name, n) for name, n in loaded_sessions),
            )
        if not all_data:
            return None
        X = np.stack(all_data)
        y = np.array(all_labels, dtype=np.int64)
        ch_names = raw.ch_names[:N_EEG_CHANNELS] if hasattr(raw, "ch_names") else [f"EEG{i+1}" for i in range(N_EEG_CHANNELS)]
        # First session = T; rest = E. Used for T/E indicator in UI.
        n_trials_from_t = loaded_sessions[0][1] if loaded_sessions else None
        return EEGDataset(
            data=X,
            labels=y,
            fs=float(raw.info["sfreq"]),
            channel_names=ch_names,
            class_names=CLASS_NAMES,
            subject_id=subject_id,
            n_trials_from_t=n_trials_from_t,
        )

    def _load_mat_subject(self, data_path: Path, subject_id: int) -> EEGDataset | None:
        try:
            from scipy.io import loadmat
        except ImportError:
            logger.error("scipy required for .mat files")
            return None
        prefix = f"A{subject_id:02d}"
        train_mat = data_path / f"{prefix}T.mat"
        test_mat = data_path / f"{prefix}E.mat"
        all_data: list[np.ndarray] = []
        all_labels: list[int] = []
        for p in (train_mat, test_mat):
            if not p.exists():
                continue
            mat: Any = loadmat(p, struct_as_record=False, squeeze_me=True)
            # Common layouts: data (samples, channels) or (trials, channels, samples)
            if "data" in mat:
                d = mat["data"]
                if d.ndim == 2:
                    # (samples, channels) -> need to segment by labels
                    if "class_labels" in mat:
                        lbl = mat["class_labels"].flatten()
                    elif "labels" in mat:
                        lbl = mat["labels"].flatten()
                    else:
                        continue
                    # Assume trials are concatenated; need trial length
                    n_ch = d.shape[1]
                    n_per_trial = d.shape[0] // len(lbl) if len(lbl) else 0
                    if n_per_trial > 0:
                        for i in range(len(lbl)):
                            t = d[i * n_per_trial:(i + 1) * n_per_trial, :].T
                            all_data.append(t.astype(np.float64))
                            lab = int(lbl[i])
                            all_labels.append(lab if lab in (1, 2, 3, 4) else lab - 1)
                elif d.ndim == 3:
                    # (trials, channels, samples)
                    for i in range(d.shape[0]):
                        all_data.append(d[i].astype(np.float64))
                    if "class_labels" in mat:
                        lbl = mat["class_labels"].flatten()
                    else:
                        lbl = mat.get("labels", np.zeros(d.shape[0]))
                    for lab in lbl:
                        l = int(lab)
                        all_labels.append(l if 0 <= l <= 3 else l - 1)
            elif "X" in mat:
                X = mat["X"]
                y = mat["y"].flatten() if "y" in mat else mat["labels"].flatten()
                if X.ndim == 2:
                    n = X.shape[0]
                    feat = X.shape[1]
                    # Might be (trials, features) - not raw EEG; skip or adapt
                    logger.warning("Mat file has 2D X (features?), using as single trial block")
                    all_data.append(X.T.astype(np.float64))
                    all_labels.extend(y.astype(int).tolist())
                elif X.ndim == 3:
                    for i in range(X.shape[0]):
                        all_data.append(X[i].astype(np.float64))
                    all_labels.extend(y.astype(int).tolist())
        if not all_data:
            return None
        X = np.stack(all_data)
        y = np.array(all_labels, dtype=np.int64)
        y = np.clip(y, 0, 3)
        ch_names = [f"EEG{i+1}" for i in range(X.shape[1])]
        return EEGDataset(
            data=X,
            labels=y,
            fs=FS,
            channel_names=ch_names,
            class_names=CLASS_NAMES,
            subject_id=subject_id,
        )

    def _download(self, data_path: Path) -> None:
        """Attempt to download BCI IV 2a. User may need to place files manually."""
        data_path.mkdir(parents=True, exist_ok=True)
        readme = data_path / "README.txt"
        readme.write_text(
            "BCI Competition IV Dataset 2a\n"
            "Download from: https://bnci-horizon-2020.eu/database/data-sets\n"
            "or https://www.bbci.de/competition/iv/\n"
            "Place A01T.gdf, A01E.gdf, ... A09T.gdf, A09E.gdf in this folder.\n"
        )
        logger.info("Created %s. Please download BCI IV 2a GDF files into it.", data_path)


def _events_from_stim_channel(stim: np.ndarray, sfreq: float) -> np.ndarray:
    """Simple event detection from stimulus channel."""
    events = []
    prev = 0
    for i, v in enumerate(stim):
        v = int(v)
        if v in CLASS_TRIGGERS and v != prev:
            events.append([int(i * sfreq), 0, v])
            prev = v
        elif v == 0:
            prev = 0
    return np.array(events) if events else np.zeros((0, 3))
