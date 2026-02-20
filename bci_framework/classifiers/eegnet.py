"""EEGNet (PyTorch CNN) for EEG classification."""

import logging
import time
from typing import Any

import numpy as np

from .base import ClassifierBase

logger = logging.getLogger(__name__)

# Training config for LOSO/cross-subject (conference-level: more epochs, cosine LR, early stop)
EEGNET_EPOCHS_DEFAULT = 150
EEGNET_LR = 1e-3
EEGNET_BATCH_SIZE = 32
EEGNET_WEIGHT_DECAY = 1e-4
EEGNET_EARLY_STOP_PATIENCE = 20
EEGNET_DROPOUT_DEFAULT = 0.5


class EEGNetClassifier(ClassifierBase):
    """EEGNet: compact CNN for EEG. Expects raw (n_trials, n_channels, n_time) or flattened (n_trials, n_ch*n_time)."""

    name = "eegnet"

    def __init__(
        self,
        n_classes: int = 4,
        n_channels: int = 22,
        samples: int = 1000,
        dropout: float = 0.5,
        epochs: int = 150,
        early_stopping_patience: int = 20,
        use_cosine_scheduler: bool = True,
        use_balanced_sampler: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(n_classes=n_classes, **kwargs)
        self.n_channels = n_channels
        self.samples = samples
        self.dropout = dropout
        self.epochs = int(epochs) if epochs is not None else EEGNET_EPOCHS_DEFAULT
        self.early_stopping_patience = int(early_stopping_patience)
        self.use_cosine_scheduler = bool(use_cosine_scheduler)
        self.use_balanced_sampler = bool(use_balanced_sampler)
        self._model = None
        self._device = None

    def _ensure_shape(self, X: np.ndarray) -> np.ndarray:
        """Ensure X is (n_trials, n_channels, n_samples). Crop or pad if flattened."""
        target = self.n_channels * self.samples
        if X.ndim == 3:
            if X.shape[2] != self.samples:
                X = X[:, :, : self.samples] if X.shape[2] >= self.samples else np.pad(
                    X, ((0, 0), (0, 0), (0, self.samples - X.shape[2])), mode="constant", constant_values=0
                )
            return X.astype(np.float32)
        n_feat = X.shape[1]
        if n_feat == target:
            return X.reshape(-1, self.n_channels, self.samples).astype(np.float32)
        if n_feat > target:
            X = X[:, :target]
        else:
            X = np.pad(X, ((0, 0), (0, target - n_feat)), mode="constant", constant_values=0)
        return X.reshape(-1, self.n_channels, self.samples).astype(np.float32)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "EEGNetClassifier":
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
        except ImportError:
            raise ImportError("PyTorch required for EEGNet. pip install torch")
        from .eegnet_model import EEGNet

        y = np.asarray(y, dtype=np.int64).ravel()
        assert np.all(y >= 0) and np.all(y < self.n_classes), "EEGNet: labels must be in [0, n_classes-1]"
        X = self._ensure_shape(X)
        assert X.ndim == 3 and X.shape[1] == self.n_channels and X.shape[2] == self.samples, (
            f"EEGNet fit: expected (n, {self.n_channels}, {self.samples}); got {X.shape}"
        )
        n_epochs = self.epochs
        logger.info(
            "[EEGNet] fit: input shape=%s, n_classes=%d, unique_labels=%s, epochs=%d, dropout=%.2f",
            X.shape, self.n_classes, np.unique(y).tolist(), n_epochs, self.dropout,
        )

        self._device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            else "cpu"
        )
        X_t = torch.from_numpy(X).unsqueeze(1)
        y_t = torch.from_numpy(y)
        dataset = TensorDataset(X_t, y_t)
        if self.use_balanced_sampler and self.n_classes > 1:
            classes, counts = np.unique(y, return_counts=True)
            class_weight = 1.0 / (np.array(counts, dtype=np.float64) + 1e-6)
            sample_weights = class_weight[np.searchsorted(classes, y)]
            sampler = WeightedRandomSampler(
                torch.from_numpy(sample_weights.astype(np.float32)),
                num_samples=len(y),
                replacement=True,
            )
            loader = DataLoader(dataset, batch_size=EEGNET_BATCH_SIZE, sampler=sampler, num_workers=0)
        else:
            loader = DataLoader(dataset, batch_size=EEGNET_BATCH_SIZE, shuffle=True, num_workers=0)
        self._model = EEGNet(
            n_classes=self.n_classes,
            n_channels=self.n_channels,
            n_samples=self.samples,
            dropout=self.dropout,
        ).to(self._device)
        opt = torch.optim.AdamW(
            self._model.parameters(),
            lr=EEGNET_LR,
            weight_decay=EEGNET_WEIGHT_DECAY,
        )
        scheduler = None
        if self.use_cosine_scheduler:
            try:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-6)
            except Exception:
                scheduler = None
        criterion = nn.CrossEntropyLoss()
        self._model.train()
        best_val_acc = -1.0
        best_epoch = 0
        patience_counter = 0
        n_val = max(1, min(100, len(y) // 4))
        val_idx = np.random.RandomState(42).choice(len(y), size=min(n_val, len(y)), replace=False)
        train_idx = np.array([i for i in range(len(y)) if i not in set(val_idx)])
        if len(train_idx) < 2:
            train_idx = np.arange(len(y))
            val_idx = np.arange(len(y))
        X_val = X[val_idx]
        y_val = y[val_idx]
        X_train_fit = X[train_idx]
        y_train_fit = y[train_idx]
        X_val_t = torch.from_numpy(self._ensure_shape(X_val)).unsqueeze(1).to(self._device)
        y_val_np = y_val

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            n_batches = 0
            for xb, yb in loader:
                xb, yb = xb.to(self._device), yb.to(self._device)
                opt.zero_grad()
                logits = self._model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
                n_batches += 1
            if scheduler is not None:
                scheduler.step()
            if (epoch + 1) % 20 == 0 or epoch == 0:
                logger.info("[EEGNet] epoch %d/%d loss=%.4f", epoch + 1, n_epochs, epoch_loss / max(n_batches, 1))
            self._model.eval()
            with torch.no_grad():
                val_logits = self._model(X_val_t)
                val_pred = val_logits.argmax(dim=1).cpu().numpy()
                val_acc = float(np.mean(val_pred == y_val_np))
            self._model.train()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                patience_counter = 0
            else:
                patience_counter += 1
            if self.early_stopping_patience > 0 and patience_counter >= self.early_stopping_patience:
                logger.info("[EEGNet] early stopping at epoch %d (best val acc=%.4f at epoch %d)", epoch + 1, best_val_acc, best_epoch)
                break
        with torch.no_grad():
            train_logits = self._model(torch.from_numpy(X).unsqueeze(1).to(self._device))
            train_pred = train_logits.argmax(dim=1).cpu().numpy()
            train_acc = float(np.mean(train_pred == y))
        logger.info("[EEGNet] train accuracy after %d epochs: %.4f", n_epochs, train_acc)
        logger.info("[EEGNet] validation accuracy: %.4f (best epoch %d)", best_val_acc, best_epoch)
        if train_acc < 0.40:
            logger.warning("[EEGNet] train accuracy < 40%%; check data/labels/config (%.4f)", train_acc)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("EEGNet not fitted")
        import torch
        X = self._ensure_shape(X)
        X_t = torch.from_numpy(X).unsqueeze(1).to(self._device)
        self._model.eval()
        with torch.no_grad():
            logits = self._model(X_t)
        return logits.argmax(dim=1).cpu().numpy().astype(np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("EEGNet not fitted")
        import torch
        X = self._ensure_shape(X)
        X_t = torch.from_numpy(X).unsqueeze(1).to(self._device)
        self._model.eval()
        with torch.no_grad():
            logits = self._model(X_t)
        return torch.softmax(logits, dim=1).cpu().numpy().astype(np.float64)
