"""
Configurable synthetic motor-imagery-like EEG for pipeline testing and CI.
Generates multi-channel signals with class-dependent spectral/amplitude structure.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

from .base import EEGDataset, DatasetLoader

logger = logging.getLogger(__name__)


def generate_synthetic_mi_eeg(
    n_trials: int = 120,
    n_channels: int = 22,
    n_samples: int = 750,
    n_classes: int = 4,
    fs: float = 250.0,
    class_balance: bool = True,
    mu_freq: float = 12.0,
    beta_freq: float = 20.0,
    snr_db: float = 0.0,
    random_state: int | None = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic motor-imagery-like EEG.
    Each class gets slightly different band power (mu/beta) and amplitude to simulate discriminability.
    Returns (data, labels) with data (n_trials, n_channels, n_samples).
    """
    rng = np.random.default_rng(random_state)
    t = np.arange(n_samples) / fs
    data = np.zeros((n_trials, n_channels, n_samples), dtype=np.float64)
    if class_balance:
        y = np.repeat(np.arange(n_classes), n_trials // n_classes)
        if len(y) < n_trials:
            y = np.concatenate([y, rng.integers(0, n_classes, size=n_trials - len(y))])
        rng.shuffle(y)
    else:
        y = rng.integers(0, n_classes, size=n_trials)

    for i in range(n_trials):
        c = y[i]
        amp_mu = 1.0 + 0.3 * np.sin(c * 0.7)
        amp_beta = 1.0 + 0.2 * np.cos(c * 0.5)
        for ch in range(n_channels):
            signal = (
                amp_mu * np.sin(2 * np.pi * mu_freq * t + rng.uniform(0, 2 * np.pi))
                + amp_beta * 0.5 * np.sin(2 * np.pi * beta_freq * t + rng.uniform(0, 2 * np.pi))
            )
            noise = rng.standard_normal(n_samples)
            if snr_db is not None and snr_db != 0:
                sig_power = np.mean(signal ** 2) + 1e-12
                noise_power = sig_power / (10 ** (snr_db / 10))
                noise = noise * np.sqrt(noise_power / (np.mean(noise ** 2) + 1e-12))
            data[i, ch, :] = signal + noise
    return data, y.astype(np.int64)


class SyntheticEEGLoader(DatasetLoader):
    """
    Load synthetic motor-imagery-like EEG for testing and CI.
    No files required; config-driven generation.
    """

    name = "synthetic_eeg"
    default_data_dir = "./data/synthetic"

    def get_subject_ids(self) -> list[int]:
        return [1]

    def load(
        self,
        data_dir: str | Path | None = None,
        subjects: list[int] | list[str] | None = None,
        download_if_missing: bool = True,
        n_trials: int = 120,
        n_channels: int = 22,
        n_samples: int = 750,
        n_classes: int = 4,
        fs: float = 250.0,
        random_state: int | None = 42,
        **kwargs: Any,
    ) -> EEGDataset:
        data, labels = generate_synthetic_mi_eeg(
            n_trials=n_trials,
            n_channels=n_channels,
            n_samples=n_samples,
            n_classes=n_classes,
            fs=fs,
            random_state=random_state,
            **kwargs,
        )
        class_names = ["left_hand", "right_hand", "feet", "tongue"][:n_classes]
        return EEGDataset(
            data=data,
            labels=labels,
            fs=fs,
            channel_names=[f"EEG{i+1}" for i in range(n_channels)],
            class_names=class_names,
            subject_id=1,
        )


def generate_synthetic_mi_eeg_for_ci(
    n_trials: int = 40,
    n_channels: int = 8,
    n_samples: int = 500,
    n_classes: int = 4,
    fs: float = 250.0,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Shorter synthetic dataset for CI (faster tests).
    """
    return generate_synthetic_mi_eeg(
        n_trials=n_trials,
        n_channels=n_channels,
        n_samples=n_samples,
        n_classes=n_classes,
        fs=fs,
        random_state=random_state,
    )
