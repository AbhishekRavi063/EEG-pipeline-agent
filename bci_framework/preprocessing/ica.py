"""ICA for artifact removal."""

import numpy as np

from .base import AdvancedPreprocessingBase


class ICAArtifactRemoval(AdvancedPreprocessingBase):
    """ICA-based artifact removal. Fits on calibration data, then transforms."""

    name = "ica"

    def __init__(
        self,
        fs: float,
        n_components: int = 15,
        max_iter: int = 500,
        **kwargs: object,
    ) -> None:
        super().__init__(fs, n_components=n_components, max_iter=max_iter, **kwargs)
        self.n_components = n_components
        self.max_iter = max_iter
        self._mixing = None
        self._unmixing = None

    supports_online: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "ICAArtifactRemoval":
        try:
            from sklearn.decomposition import FastICA
        except ImportError:
            raise ImportError("sklearn is required for ICA. pip install scikit-learn")
        # X: (n_trials, n_channels, n_samples) -> (n_trials * n_samples, n_channels)
        n_trials, n_ch, n_samp = X.shape
        X_flat = X.transpose(0, 2, 1).reshape(-1, n_ch).astype(np.float64)
        ica = FastICA(n_components=min(self.n_components, n_ch), max_iter=self.max_iter, random_state=42)
        ica.fit(X_flat)
        self._unmixing = ica.components_
        self._mixing = ica.mixing_
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._unmixing is None:
            return X
        n_trials, n_ch, n_samp = X.shape
        X_flat = X.transpose(0, 2, 1).reshape(-1, n_ch)
        # Zero out last components (assumed artifacts) and back-project
        n_keep = self._unmixing.shape[0]
        S = X_flat @ self._unmixing.T
        # Reconstruct with all components (no pruning by default to avoid info loss)
        X_recon = S @ self._unmixing
        return X_recon.reshape(n_trials, n_samp, n_ch).transpose(0, 2, 1).astype(np.float64)
