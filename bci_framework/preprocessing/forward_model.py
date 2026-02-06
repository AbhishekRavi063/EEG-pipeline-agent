"""
MNE-based forward model pipeline for GEDAI leadfield generation.

Generates physics-correct leadfield matrices for EEG source localization.
Uses MNE's head model and forward solution computation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

FORWARD_MODEL_AVAILABLE = False
try:
    import mne
    from mne.channels import make_standard_montage
    FORWARD_MODEL_AVAILABLE = True
except ImportError:
    pass


# BCI IV 2a standard channel names (10-20 system, 22 channels)
BCI_IV_2A_CHANNELS = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz"
]


def generate_leadfield_bci_iv_2a(
    output_path: str | Path,
    n_sources: int = 2000,
    conductivity: tuple[float, float, float] = (0.33, 0.004125, 0.33),  # (brain, skull, scalp)
    spacing: str = "ico4",
    save_format: str = "npy",
    overwrite: bool = False,
) -> np.ndarray:
    """
    Generate physics-correct leadfield matrix for BCI Competition IV 2a dataset.

    Uses MNE's 3-layer boundary element model (BEM) and forward solution.

    Parameters
    ----------
    output_path : str | Path
        Path to save leadfield matrix (.npy or .pt)
    n_sources : int
        Number of source dipoles (default: 2000, typical for ico4)
    conductivity : tuple[float, float, float]
        Conductivities (S/m) for brain, skull, scalp layers
    spacing : str
        Source space spacing ('ico4' = ~2562 sources, 'ico3' = ~642 sources)
    save_format : str
        Format: 'npy' (numpy) or 'pt' (PyTorch)
    overwrite : bool
        Overwrite existing file

    Returns
    -------
    leadfield : np.ndarray
        Leadfield matrix shape (n_channels, n_sources) or (n_channels, n_channels)
        if reduced to channel space.

    Notes
    -----
    The leadfield L relates source activity s to sensor measurements x:
        x = L @ s

    For GEDAI, we typically use a reduced leadfield in channel space:
        L_reduced = L @ L.T  (n_channels, n_channels)

    This script generates the full leadfield (n_channels, n_sources) and optionally
    reduces it to channel space for GEDAI compatibility.
    """
    if not FORWARD_MODEL_AVAILABLE:
        raise ImportError(
            "MNE is required for forward model generation. Install: pip install mne"
        )

    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        logger.info("Leadfield already exists at %s (use overwrite=True to regenerate)", output_path)
        if save_format == "npy":
            return np.load(output_path)
        else:
            import torch
            return torch.load(output_path, map_location="cpu", weights_only=False).numpy()

    logger.info("Generating forward model for BCI IV 2a (22 channels, %s spacing)...", spacing)

    # Create info object with BCI IV 2a montage
    info = mne.create_info(
        ch_names=BCI_IV_2A_CHANNELS,
        sfreq=250.0,  # BCI IV 2a sampling rate
        ch_types="eeg",
    )

    # Set standard 10-20 montage
    montage = make_standard_montage("standard_1020")
    info.set_montage(montage)

    # Create source space
    logger.info("Creating source space (%s)...", spacing)
    src = mne.setup_source_space(
        subject="fsaverage",  # Use fsaverage template
        spacing=spacing,
        add_dist="patch",
        n_jobs=1,
    )

    # Create BEM model (3-layer: brain, skull, scalp)
    logger.info("Creating BEM model (conductivity: brain=%.3f, skull=%.3f, scalp=%.3f)...", *conductivity)
    bem = mne.make_bem_model(
        subject="fsaverage",
        ico=4,
        conductivity=conductivity,
        subjects_dir=None,  # Use MNE's default fsaverage
    )
    bem_sol = mne.make_bem_solution(bem)

    # Compute forward solution
    logger.info("Computing forward solution...")
    fwd = mne.make_forward_solution(
        info,
        trans="fsaverage",  # Identity transformation (MNE template)
        src=src,
        bem=bem_sol,
        meg=False,
        eeg=True,
        mindist=5.0,  # Minimum distance from inner skull (mm)
        n_jobs=1,
    )

    # Extract leadfield matrix (n_channels, n_sources * 3) where 3 = x, y, z orientations
    leadfield = fwd["sol"]["data"]  # Shape: (n_channels, n_sources * 3)

    # Reduce to channel space for GEDAI compatibility
    # GEDAI expects (n_channels, n_channels) leadfield
    # We compute: L_reduced = L @ L.T (covariance-like structure)
    logger.info("Reducing leadfield to channel space (%d channels)...", len(BCI_IV_2A_CHANNELS))
    leadfield_reduced = leadfield @ leadfield.T  # (n_channels, n_channels)

    # Normalize to unit scale (optional, but helps numerical stability)
    leadfield_reduced = leadfield_reduced / np.linalg.norm(leadfield_reduced, ord="fro")

    # Save
    if save_format == "npy":
        np.save(output_path, leadfield_reduced.astype(np.float64))
        logger.info("Saved leadfield to %s (shape: %s)", output_path, leadfield_reduced.shape)
    else:
        import torch
        torch.save(torch.as_tensor(leadfield_reduced, dtype=torch.float64), output_path)
        logger.info("Saved leadfield to %s (shape: %s)", output_path, leadfield_reduced.shape)

    return leadfield_reduced


def generate_leadfield_from_montage(
    channel_names: list[str],
    output_path: str | Path,
    montage_name: str = "standard_1020",
    **kwargs: Any,
) -> np.ndarray:
    """
    Generate leadfield for custom channel montage.

    Parameters
    ----------
    channel_names : list[str]
        List of channel names (must match montage)
    output_path : str | Path
        Output path
    montage_name : str
        MNE montage name (e.g., 'standard_1020', 'standard_1005')
    **kwargs
        Passed to generate_leadfield_bci_iv_2a

    Returns
    -------
    leadfield : np.ndarray
        Leadfield matrix (n_channels, n_channels)
    """
    if not FORWARD_MODEL_AVAILABLE:
        raise ImportError("MNE is required. Install: pip install mne")

    info = mne.create_info(
        ch_names=channel_names,
        sfreq=250.0,
        ch_types="eeg",
    )
    montage = make_standard_montage(montage_name)
    info.set_montage(montage)

    # Use same forward model generation as BCI IV 2a
    # (This is a simplified version; full implementation would use custom montage)
    logger.warning(
        "Custom montage leadfield generation not fully implemented. "
        "Using BCI IV 2a template. For custom montages, use MNE directly."
    )
    return generate_leadfield_bci_iv_2a(output_path, **kwargs)


if __name__ == "__main__":
    # CLI for generating leadfield
    import argparse

    parser = argparse.ArgumentParser(description="Generate GEDAI leadfield for BCI IV 2a")
    parser.add_argument("--output", "-o", default="./data/leadfield_bci_iv_2a.npy", help="Output path")
    parser.add_argument("--format", choices=["npy", "pt"], default="npy", help="File format")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing file")
    args = parser.parse_args()

    leadfield = generate_leadfield_bci_iv_2a(
        output_path=args.output,
        save_format=args.format,
        overwrite=args.overwrite,
    )
    print(f"Leadfield shape: {leadfield.shape}")
    print(f"Saved to: {args.output}")
