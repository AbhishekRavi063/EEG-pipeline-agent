"""RSA + MLP: small regularized MLP on tangent features. CPU-only, float32, no GPU."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from .base import ClassifierBase

logger = logging.getLogger(__name__)

MAX_PARAMS = 400_000
HIDDEN1 = 256
HIDDEN2 = 64
DROPOUT = 0.2
LR = 1e-3
WEIGHT_DECAY = 5e-4
BATCH_SIZE = 64
MAX_EPOCHS = 200
PATIENCE = 15
VAL_FRACTION = 0.2


def _count_parameters(module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class RSAMLPClassifier(ClassifierBase):
    """
    Small regularized MLP for RSA tangent features.
    Linear(in → 256) → ReLU → Dropout(0.2) → Linear(256 → 64) → ReLU → Dropout(0.2) → Linear(64 → n_classes).
    CPU only, float32, no batch norm. Trained only on source; validation from source (20%).
    """

    name = "rsa_mlp"

    def __init__(
        self,
        n_classes: int = 4,
        hidden1: int = HIDDEN1,
        hidden2: int = HIDDEN2,
        single_hidden: int | None = None,
        dropout: float = DROPOUT,
        lr: float = LR,
        weight_decay: float = WEIGHT_DECAY,
        batch_size: int = BATCH_SIZE,
        max_epochs: int = MAX_EPOCHS,
        patience: int = PATIENCE,
        val_fraction: float = VAL_FRACTION,
        **kwargs: Any,
    ) -> None:
        super().__init__(n_classes=n_classes, **kwargs)
        self.single_hidden = int(single_hidden) if single_hidden is not None else None
        self.hidden1 = int(hidden1)
        self.hidden2 = int(hidden2)
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.val_fraction = float(val_fraction)
        self._input_dim: int | None = None
        self._model = None
        self._device = None
        self._train_loss_curve: list[float] = []
        self._val_loss_curve: list[float] = []
        self._best_epoch: int = 0
        self._param_count: int = 0
        self._train_time_sec: float = 0.0

    def _build_model(self, input_dim: int):
        import torch
        import torch.nn as nn

        if self.single_hidden is not None:
            # 1 hidden layer: in -> single_hidden -> n_classes
            class _MLPSingle(nn.Module):
                def __init__(self, in_dim: int, h: int, n_classes: int, drop: float):
                    super().__init__()
                    self.l1 = nn.Linear(in_dim, h)
                    self.l2 = nn.Linear(h, n_classes)
                    self.dropout = nn.Dropout(drop)

                def forward(self, x):
                    x = self.dropout(torch.relu(self.l1(x)))
                    return self.l2(x)

            model = _MLPSingle(
                input_dim,
                self.single_hidden,
                self.n_classes,
                self.dropout,
            )
        else:
            class _MLP(nn.Module):
                def __init__(self, in_dim: int, h1: int, h2: int, n_classes: int, drop: float):
                    super().__init__()
                    self.l1 = nn.Linear(in_dim, h1)
                    self.l2 = nn.Linear(h1, h2)
                    self.l3 = nn.Linear(h2, n_classes)
                    self.dropout = nn.Dropout(drop)

                def forward(self, x):
                    x = self.dropout(torch.relu(self.l1(x)))
                    x = self.dropout(torch.relu(self.l2(x)))
                    return self.l3(x)

            model = _MLP(
                input_dim,
                self.hidden1,
                self.hidden2,
                self.n_classes,
                self.dropout,
            )
        model = model.to(torch.float32)
        self._device = torch.device("cpu")
        model = model.to(self._device)
        n_params = _count_parameters(model)
        self._param_count = n_params
        assert n_params < MAX_PARAMS, (
            f"RSAMLP: total_params={n_params} >= {MAX_PARAMS}. Reduce hidden sizes."
        )
        logger.info("[RSAMLP] params=%d (max %d)", n_params, MAX_PARAMS)
        return model

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "RSAMLPClassifier":
        import torch
        import torch.nn as nn
        from sklearn.model_selection import train_test_split

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64).ravel()
        if X.ndim != 2 or len(y) != len(X):
            raise ValueError("X must be (n_samples, n_features), y length n_samples")
        input_dim = X.shape[1]
        self._input_dim = input_dim

        # Validation split from source only (20%)
        if len(X) < 20:
            X_tr, X_val, y_tr, y_val = X, X[:1], y, y[:1]
        else:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X, y, test_size=self.val_fraction, stratify=y, random_state=42
            )

        self._model = self._build_model(input_dim)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        best_val_loss = float("inf")
        best_state: dict | None = None
        self._best_epoch = 0
        epochs_no_improve = 0
        self._train_loss_curve = []
        self._val_loss_curve = []

        t0 = time.perf_counter()
        n_tr = len(X_tr)
        n_val = len(X_val)

        for epoch in range(self.max_epochs):
            self._model.train()
            perm = np.random.permutation(n_tr) if n_tr > 1 else np.arange(n_tr)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n_tr, self.batch_size):
                end = min(start + self.batch_size, n_tr)
                idx = perm[start:end]
                x_b = torch.from_numpy(X_tr[idx]).to(self._device, dtype=torch.float32)
                y_b = torch.from_numpy(y_tr[idx]).to(self._device, dtype=torch.long)
                optimizer.zero_grad()
                logits = self._model(x_b)
                loss = criterion(logits, y_b)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            train_loss = epoch_loss / max(n_batches, 1)
            self._train_loss_curve.append(train_loss)

            self._model.eval()
            with torch.no_grad():
                x_val = torch.from_numpy(X_val).to(self._device, dtype=torch.float32)
                y_val_t = torch.from_numpy(y_val).to(self._device, dtype=torch.long)
                logits = self._model(x_val)
                val_loss = criterion(logits, y_val_t).item()
            self._val_loss_curve.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}
                self._best_epoch = epoch + 1
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if (epoch + 1) % 20 == 0 or epoch == 0:
                logger.info(
                    "[RSAMLP] epoch %d train_loss=%.4f val_loss=%.4f",
                    epoch + 1, train_loss, val_loss,
                )
            if epochs_no_improve >= self.patience:
                logger.info("[RSAMLP] early stop at epoch %d (patience=%d)", epoch + 1, self.patience)
                break

        if best_state is not None:
            self._model.load_state_dict(best_state)
        self._train_time_sec = time.perf_counter() - t0
        logger.info(
            "[RSAMLP] best_epoch=%d params=%d train_time=%.2fs",
            self._best_epoch, self._param_count, self._train_time_sec,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("RSAMLP not fitted")
        import torch
        self._model.eval()
        X = np.asarray(X, dtype=np.float32)
        with torch.no_grad():
            x_t = torch.from_numpy(X).to(self._device, dtype=torch.float32)
            logits = self._model(x_t)
            pred = logits.argmax(dim=1)
        return pred.cpu().numpy().astype(np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("RSAMLP not fitted")
        import torch
        import torch.nn.functional as F
        self._model.eval()
        X = np.asarray(X, dtype=np.float32)
        with torch.no_grad():
            x_t = torch.from_numpy(X).to(self._device, dtype=torch.float32)
            logits = self._model(x_t)
            proba = F.softmax(logits, dim=1)
        return proba.cpu().numpy().astype(np.float64)
