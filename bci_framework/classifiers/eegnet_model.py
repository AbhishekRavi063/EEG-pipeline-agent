"""EEGNet architecture (PyTorch)."""

from typing import Any

try:
    import torch
    import torch.nn as nn
except ImportError:
    nn = None
    torch = None


class EEGNet(nn.Module if nn else object):
    """EEGNet: temporal + depthwise conv + separable conv."""

    def __init__(
        self,
        n_classes: int = 4,
        n_channels: int = 22,
        n_samples: int = 1000,
        dropout: float = 0.25,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        **kwargs: Any,
    ) -> None:
        if nn is None:
            raise ImportError("PyTorch required")
        super().__init__()
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.n_samples = n_samples
        # Block 1: temporal conv
        self.conv1 = nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        # Depthwise conv
        self.depthwise = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.elu = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)
        # Block 2: separable conv
        self.separable = nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)
        # Flatten and linear
        with torch.no_grad():
            x = torch.zeros(1, 1, n_channels, n_samples)
            x = self.pool1(self.elu(self.bn2(self.depthwise(self.elu(self.bn1(self.conv1(x)))))))
            x = self.pool2(self.elu(self.bn3(self.separable(x))))
            self._flat = x.numel()
        self.fc = nn.Linear(self._flat, n_classes)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)
        x = self.separable(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x
