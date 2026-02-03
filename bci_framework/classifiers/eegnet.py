"""EEGNet (PyTorch CNN) for EEG classification."""

import time
from typing import Any

import numpy as np

from .base import ClassifierBase


class EEGNetClassifier(ClassifierBase):
    """EEGNet: compact CNN for EEG. Expects raw (n_samples, n_channels, n_time)."""

    name = "eegnet"

    def __init__(
        self,
        n_classes: int = 4,
        n_channels: int = 22,
        samples: int = 1000,
        dropout: float = 0.25,
        **kwargs: Any,
    ) -> None:
        super().__init__(n_classes=n_classes, **kwargs)
        self.n_channels = n_channels
        self.samples = samples
        self.dropout = dropout
        self._model = None
        self._device = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EEGNetClassifier":
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import TensorDataset, DataLoader
        except ImportError:
            raise ImportError("PyTorch required for EEGNet. pip install torch")
        from .eegnet_model import EEGNet

        self._device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            else "cpu"
        )
        # X: (n_trials, n_features) from CSP/PSD or (n_trials, n_ch, n_time) raw
        if X.ndim == 2:
            # Reshape if features were flattened from (ch, time)
            n_feat = X.shape[1]
            if n_feat == self.n_channels * self.samples:
                X = X.reshape(-1, self.n_channels, self.samples)
            else:
                # Use a small dummy temporal dim for compatibility
                pad = self.n_channels * self.samples - n_feat
                if pad > 0:
                    X = np.pad(X, ((0, 0), (0, pad)), mode="constant", constant_values=0)
                X = X.reshape(-1, self.n_channels, self.samples)
        X_t = torch.from_numpy(X.astype(np.float32)).unsqueeze(1)
        y_t = torch.from_numpy(y.astype(np.int64))
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
        self._model = EEGNet(
            n_classes=self.n_classes,
            n_channels=self.n_channels,
            n_samples=self.samples,
            dropout=self.dropout,
        ).to(self._device)
        opt = torch.optim.Adam(self._model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        self._model.train()
        for _ in range(30):
            for xb, yb in loader:
                xb, yb = xb.to(self._device), yb.to(self._device)
                opt.zero_grad()
                logits = self._model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                opt.step()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("EEGNet not fitted")
        import torch
        if X.ndim == 2:
            n_feat = X.shape[1]
            if n_feat == self.n_channels * self.samples:
                X = X.reshape(-1, self.n_channels, self.samples)
            else:
                pad = self.n_channels * self.samples - n_feat
                if pad > 0:
                    X = np.pad(X, ((0, 0), (0, pad)), mode="constant", constant_values=0)
                X = X.reshape(-1, self.n_channels, self.samples)
        X_t = torch.from_numpy(X.astype(np.float32)).unsqueeze(1).to(self._device)
        self._model.eval()
        with torch.no_grad():
            logits = self._model(X_t)
        return logits.argmax(dim=1).cpu().numpy().astype(np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("EEGNet not fitted")
        import torch
        if X.ndim == 2:
            n_feat = X.shape[1]
            if n_feat == self.n_channels * self.samples:
                X = X.reshape(-1, self.n_channels, self.samples)
            else:
                pad = self.n_channels * self.samples - n_feat
                if pad > 0:
                    X = np.pad(X, ((0, 0), (0, pad)), mode="constant", constant_values=0)
                X = X.reshape(-1, self.n_channels, self.samples)
        X_t = torch.from_numpy(X.astype(np.float32)).unsqueeze(1).to(self._device)
        self._model.eval()
        with torch.no_grad():
            logits = self._model(X_t)
        return torch.softmax(logits, dim=1).cpu().numpy().astype(np.float64)
