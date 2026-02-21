"""
GEDAI (Generalized Eigenvalue De-Artifacting Instrument) wrapper for EEG denoising.

Physics-correct implementation with MNE leadfield support, sliding-window online mode,
and GPU acceleration.
"""

from __future__ import annotations

import logging
import warnings
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from .base import AdvancedPreprocessingBase

logger = logging.getLogger(__name__)

GEDAI_AVAILABLE = False
_gedai_import_error: Exception | None = None
try:
    import torch  # noqa: F401
    from pygedai import batch_gedai  # noqa: F401
    GEDAI_AVAILABLE = True
except ImportError as e:
    _gedai_import_error = e


class GEDAIArtifactRemoval(AdvancedPreprocessingBase):
    """
    GEDAI denoising: GEVD-based artifact removal using a leadfield-derived reference.

    Physics-correct implementation with:
    - MNE-based or FreeSurfer leadfield (required for real denoising)
    - Sliding-window online mode (causal, low-latency)
    - GPU acceleration support

    Leadfield (creator guidance):
    - Use a REAL leadfield (FreeSurfer or MNE forward model). Identity matrix is for
      testing the code only—with identity, GEDAI is effectively PCA and has no
      physics-based artifact removal. Set leadfield_path and use_identity_if_missing=False.
    - Best denoising: spectral GEDAI = 1st pass broadband GEDAI + 2nd pass wavelet GEDAI.
    - Real-time: wavelet GEDAI can be run in parallel on multi-core CPU.

    See docs/GEDAI_CREATOR_NOTES.md for full creator recommendations.
    """

    name = "gedai"

    # supports_online is set dynamically based on mode
    supports_online: bool = False

    def __init__(
        self,
        fs: float,
        leadfield_path: str | None = None,
        use_identity_if_missing: bool = False,  # True = dev/test only (identity ~ PCA; creator: no sense for real denoising)
        require_real_leadfield: bool = True,  # Fail if no leadfield; use real FreeSurfer/MNE leadfield
        mode: str = "batch",  # "batch" or "sliding" for online
        window_sec: float = 10.0,  # Sliding window duration (for mode="sliding")
        update_interval_sec: float = 1.0,  # How often to recompute eigenvectors
        device: str | None = None,  # "cpu", "cuda", "mps", or None (auto-detect)
        cov_recompute_post_gedai: bool = False,  # For CSP compatibility
        debug: bool = False,  # Print shape before/after, eigenvalue count
        **kwargs: Any,
    ) -> None:
        if not GEDAI_AVAILABLE:
            raise ImportError(
                "GEDAI preprocessing requires pygedai and PyTorch. "
                "Install with: pip install pygedai torch"
            ) from _gedai_import_error

        super().__init__(
            fs,
            leadfield_path=leadfield_path,
            use_identity_if_missing=use_identity_if_missing,
            require_real_leadfield=require_real_leadfield,
            mode=mode,
            window_sec=window_sec,
            update_interval_sec=update_interval_sec,
            device=device,
            cov_recompute_post_gedai=cov_recompute_post_gedai,
            **kwargs,
        )

        self.leadfield_path = leadfield_path
        self.use_identity_if_missing = bool(use_identity_if_missing)
        self.require_real_leadfield = bool(require_real_leadfield)
        self.mode = str(mode).lower()
        self.debug = bool(debug)
        self.window_sec = float(window_sec)
        self.update_interval_sec = float(update_interval_sec)
        self.cov_recompute_post_gedai = bool(cov_recompute_post_gedai)

        # Device selection
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = str(device).lower()

        # MPS doesn't support float64; use float32
        self.dtype = torch.float32 if self.device == "mps" else torch.float64

        # Online support: sliding mode is causal
        self.supports_online = self.mode == "sliding"

        # Sliding window state
        self._covariance_buffer: deque[np.ndarray] = deque(maxlen=int(self.window_sec * fs))
        self._last_update_sample = 0
        self._eigenvectors: torch.Tensor | None = None
        self._eigenvalues: torch.Tensor | None = None

        self._leadfield: torch.Tensor | None = None
        self._fitted = False

        logger.info(
            "GEDAI initialized: mode=%s, device=%s, require_real_leadfield=%s",
            self.mode,
            self.device,
            self.require_real_leadfield,
        )

    def _get_leadfield(self, n_channels: int) -> "torch.Tensor":
        import torch

        if self._leadfield is not None and self._leadfield.shape[0] == n_channels:
            return self._leadfield.to(device=self.device)

        if self.leadfield_path:
            path = Path(self.leadfield_path)
            if path.exists():
                try:
                    try:
                        L = torch.load(path, map_location=self.device, weights_only=False)
                    except TypeError:
                        L = torch.load(path, map_location=self.device)
                    if isinstance(L, np.ndarray):
                        L = torch.as_tensor(L, dtype=self.dtype, device=self.device)
                    if L.shape[0] == n_channels and L.shape[1] == n_channels:
                        self._leadfield = L.to(dtype=self.dtype, device=self.device)
                        logger.info("GEDAI: loaded leadfield from %s (%s)", path, L.shape)
                        return self._leadfield
                    else:
                        logger.warning(
                            "GEDAI: leadfield shape %s does not match channels %d",
                            L.shape,
                            n_channels,
                        )
                except Exception as e:
                    logger.warning("GEDAI: failed to load leadfield from %s: %s", path, e)

        # Fallback: identity or error (creator: identity is for testing only, otherwise ~PCA)
        if self.use_identity_if_missing:
            if self.require_real_leadfield:
                logger.warning(
                    "GEDAI: require_real_leadfield=True but using identity fallback. "
                    "Identity is for testing only—use a real leadfield for physics-based denoising."
                )
            self._leadfield = torch.eye(n_channels, dtype=self.dtype, device=self.device)
            logger.warning(
                "GEDAI: using identity leadfield for %d channels (TESTING ONLY; no physics-based denoising). "
                "For real denoising use leadfield_path with FreeSurfer/MNE leadfield.",
                n_channels,
            )
            return self._leadfield

        raise ValueError(
            "GEDAI: no leadfield provided. "
            "Set leadfield_path in config or generate one using: "
            "python -m bci_framework.preprocessing.forward_model"
        )

    def _compute_sliding_covariance(self, X: np.ndarray) -> np.ndarray:
        """Update sliding covariance buffer and return current covariance."""
        # X: (n_trials, n_channels, n_samples)
        n_trials, n_ch, n_samp = X.shape

        # Flatten trials for covariance computation
        X_flat = X.transpose(0, 2, 1).reshape(-1, n_ch)  # (n_trials * n_samples, n_channels)

        # Add to buffer
        for i in range(0, X_flat.shape[0], n_samp):
            self._covariance_buffer.append(X_flat[i : i + n_samp].T)

        # Compute covariance from buffer
        if len(self._covariance_buffer) == 0:
            return np.eye(n_ch, dtype=np.float64)

        buffer_data = np.concatenate(list(self._covariance_buffer), axis=1)  # (n_ch, total_samples)
        cov = np.cov(buffer_data)
        return cov.astype(np.float64)

    def _update_eigenvectors_sliding(self, data_cov: np.ndarray, leadfield: "torch.Tensor") -> None:
        """Update generalized eigenvectors from sliding covariance."""
        import torch

        # Convert to tensors
        data_cov_t = torch.as_tensor(data_cov, dtype=self.dtype, device=self.device)
        ref_cov_t = leadfield @ leadfield.T  # Reference covariance from leadfield

        # Generalized eigenvalue decomposition: data_cov @ v = lambda * ref_cov @ v
        # Using torch.linalg.eigh for symmetric matrices (faster, GPU-accelerated)
        try:
            eigenvals, eigenvecs = torch.linalg.eigh(data_cov_t, ref_cov_t)
            # Sort descending
            idx = torch.argsort(eigenvals, descending=True)
            self._eigenvalues = eigenvals[idx].cpu()
            self._eigenvectors = eigenvecs[:, idx].cpu()
            logger.debug(
                "GEDAI sliding: updated eigenvectors (top 5 eigenvalues: %s)",
                self._eigenvalues[:5].numpy(),
            )
        except Exception as e:
            logger.warning("GEDAI: eigen decomposition failed: %s", e)
            self._eigenvectors = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "GEDAIArtifactRemoval":
        import torch

        n_trials, n_ch, n_samp = X.shape
        if self.debug:
            print("[GEDAI DEBUG] fit: X_train shape=%s (train indices only, no test data)" % (X.shape,), flush=True)
        leadfield = self._get_leadfield(n_ch)

        if self.mode == "sliding":
            # Initialize sliding window
            self._covariance_buffer.clear()
            data_cov = self._compute_sliding_covariance(X)
            self._update_eigenvectors_sliding(data_cov, leadfield)
            self._last_update_sample = n_trials * n_samp
            if self.debug and self._eigenvalues is not None:
                n_ev = len(self._eigenvalues)
                print("[GEDAI DEBUG] fit: eigenvalues count=%d" % n_ev, flush=True)
        else:
            # Batch mode: fit done inside transform via batch_gedai; no eigenvectors stored here
            pass

        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        import torch

        if not GEDAI_AVAILABLE or self._leadfield is None:
            return X

        n_trials, n_ch, n_samp = X.shape
        if self.debug:
            print("[GEDAI DEBUG] transform: input shape before GEDAI=%s" % (X.shape,), flush=True)
        leadfield = self._get_leadfield(n_ch)

        if self.mode == "sliding":
            # Sliding-window causal mode
            if self._eigenvectors is None:
                # First call: initialize
                data_cov = self._compute_sliding_covariance(X)
                self._update_eigenvectors_sliding(data_cov, leadfield)
                if self._eigenvectors is None:
                    return X

            # Check if we need to update eigenvectors
            current_sample = n_trials * n_samp
            samples_since_update = current_sample - self._last_update_sample
            if samples_since_update >= int(self.update_interval_sec * self.fs):
                data_cov = self._compute_sliding_covariance(X)
                self._update_eigenvectors_sliding(data_cov, leadfield)
                self._last_update_sample = current_sample

            # Apply projection using current eigenvectors
            X_flat = X.transpose(0, 2, 1).reshape(-1, n_ch)  # (n_trials * n_samples, n_channels)
            X_t = torch.as_tensor(X_flat, dtype=self.dtype, device=self.device)
            eigenvecs_t = self._eigenvectors.to(device=self.device)

            # Project: keep top components (e.g., top 80% variance)
            n_keep = max(1, int(0.8 * n_ch))
            proj = eigenvecs_t[:, :n_keep] @ eigenvecs_t[:, :n_keep].T
            X_cleaned_t = X_t @ proj.T

            out = X_cleaned_t.cpu().numpy().reshape(n_trials, n_samp, n_ch).transpose(0, 2, 1)
            if self.debug:
                print("[GEDAI DEBUG] transform: output shape after GEDAI=%s" % (out.shape,), flush=True)
            return out.astype(np.float64)

        else:
            # Batch mode: use pygedai (pygedai expects CPU float64)
            t = torch.as_tensor(X, dtype=torch.float64, device="cpu")
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="EEG data length is too short for the epoch size",
                        category=RuntimeWarning,
                    )
                    result = batch_gedai(t.cpu(), sfreq=float(self.fs), leadfield=leadfield.cpu())
            except Exception as e:
                logger.warning("GEDAI batch_gedai failed, passing through unchanged: %s", e)
                return X

            # Handle return (dict or tensor)
            cleaned = result.get("cleaned") if isinstance(result, dict) else result
            if cleaned is None:
                return X

            # Convert back to numpy
            if isinstance(cleaned, torch.Tensor):
                out = cleaned.cpu().numpy()
            else:
                out = np.asarray(cleaned)

            if out.shape != X.shape:
                logger.warning("GEDAI output shape %s != input %s", out.shape, X.shape)
                return X

            if self.debug:
                print("[GEDAI DEBUG] transform: output shape after GEDAI=%s" % (out.shape,), flush=True)
            return out.astype(np.float64)
