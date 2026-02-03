"""
Drift detection and recalibration triggers.
EEG is non-stationary; rolling window accuracy triggers recalibration.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DriftDetectorConfig:
    """Config for drift detection."""

    window_size: int = 20
    min_trials_for_drift: int = 10
    accuracy_drop_threshold: float = 0.15
    min_accuracy_absolute: float = 0.35
    consecutive_low_windows: int = 2


class DriftDetector:
    """
    Rolling-window accuracy check. Triggers recalibration when:
    - Accuracy drops below baseline by accuracy_drop_threshold, or
    - Accuracy falls below min_accuracy_absolute for consecutive_low_windows.
    """

    def __init__(self, config: dict[str, Any] | DriftDetectorConfig | None = None) -> None:
        if isinstance(config, DriftDetectorConfig):
            self.cfg = config
        else:
            cfg = config or {}
            self.cfg = DriftDetectorConfig(
                window_size=cfg.get("window_trials", cfg.get("window_size", 20)),
                min_trials_for_drift=cfg.get("min_trials_for_drift", 10),
                accuracy_drop_threshold=cfg.get("accuracy_drop_threshold", 0.15),
                min_accuracy_absolute=cfg.get("min_accuracy_absolute", 0.35),
                consecutive_low_windows=cfg.get("consecutive_low_windows", 2),
            )
        self._baseline_accuracy: float | None = None
        self._recent_correct: deque[bool] = deque(maxlen=self.cfg.window_size)
        self._recent_accuracies: deque[float] = deque(maxlen=self.cfg.consecutive_low_windows)
        self._trial_count = 0
        self._drift_triggered = False

    def set_baseline(self, accuracy: float) -> None:
        """Set baseline accuracy (e.g. from calibration)."""
        self._baseline_accuracy = accuracy
        logger.info("Drift baseline accuracy set to %.3f", accuracy)

    def update(self, correct: bool) -> bool:
        """
        Update with one prediction result. Returns True if drift detected (recalibrate).
        """
        self._trial_count += 1
        self._recent_correct.append(correct)
        if len(self._recent_correct) < self.cfg.min_trials_for_drift:
            return False
        window_acc = float(np.mean(self._recent_correct))
        self._recent_accuracies.append(window_acc)
        self._drift_triggered = False
        if self._baseline_accuracy is not None:
            drop = self._baseline_accuracy - window_acc
            if drop >= self.cfg.accuracy_drop_threshold:
                self._drift_triggered = True
                logger.warning(
                    "Drift: accuracy dropped from %.3f to %.3f (delta %.3f)",
                    self._baseline_accuracy, window_acc, drop,
                )
        if window_acc < self.cfg.min_accuracy_absolute and len(self._recent_accuracies) >= self.cfg.consecutive_low_windows:
            if all(a < self.cfg.min_accuracy_absolute for a in self._recent_accuracies):
                self._drift_triggered = True
                logger.warning(
                    "Drift: accuracy %.3f below minimum %.3f for %d windows",
                    window_acc, self.cfg.min_accuracy_absolute, self.cfg.consecutive_low_windows,
                )
        return self._drift_triggered

    def reset(self) -> None:
        """Reset state (e.g. after recalibration)."""
        self._recent_correct.clear()
        self._recent_accuracies.clear()
        self._drift_triggered = False

    @property
    def drift_triggered(self) -> bool:
        return self._drift_triggered

    @property
    def current_window_accuracy(self) -> float | None:
        if len(self._recent_correct) < self.cfg.min_trials_for_drift:
            return None
        return float(np.mean(self._recent_correct))
