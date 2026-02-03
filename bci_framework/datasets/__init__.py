"""Dataset loaders for EEG motor imagery data."""

from .base import DatasetLoader, EEGDataset
from .bci_iv_2a import BCICompetitionIV2aLoader
from .synthetic_eeg import SyntheticEEGLoader, generate_synthetic_mi_eeg, generate_synthetic_mi_eeg_for_ci

__all__ = [
    "DatasetLoader",
    "EEGDataset",
    "BCICompetitionIV2aLoader",
    "SyntheticEEGLoader",
    "generate_synthetic_mi_eeg",
    "generate_synthetic_mi_eeg_for_ci",
]

DATASET_REGISTRY: dict[str, type] = {
    "BCI_IV_2a": BCICompetitionIV2aLoader,
    "synthetic_eeg": SyntheticEEGLoader,
}


def get_dataset_loader(name: str) -> type[DatasetLoader]:
    """Get dataset loader class by name."""
    if name not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Available: {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[name]
