"""
Online calibration + selection + live streaming for subject-specific EEG pipelines.
First N trials → calibration → select best pipeline; remaining trials → live with selected pipeline.
Supports drift monitoring and optional re-calibration.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bci_framework.pipelines import Pipeline
from bci_framework.agent.pipeline_agent import PipelineSelectionAgent, PipelineMetrics
from bci_framework.agent.drift_detector import DriftDetector

logger = logging.getLogger(__name__)


@dataclass
class OnlineTrialRecord:
    """Single trial for calibration buffer or live log."""
    trial_index: int
    data: np.ndarray  # (1, n_channels, n_samples) or (n_channels, n_samples)
    label: int | None = None
    predicted: int | None = None
    proba: np.ndarray | None = None
    correct: bool | None = None


class OnlinePipelineSelector:
    """
    Online calibration + selection + live streaming.
    - add_trial(trial_data, trial_label): add trial to calibration buffer or (if live) update drift.
    - calibrate(): run pipelines on first N trials, prune, select best (called automatically when buffer full).
    - is_live_phase(): True after calibration done; remaining trials processed with selected pipeline.
    - predict(trial_data): run selected pipeline on chunk (live phase only).
    - update_drift(true_label, predicted_label): update drift detector; returns True if drift triggered.
    """

    def __init__(
        self,
        pipelines: list[Pipeline],
        config: dict[str, Any],
        n_classes: int,
    ) -> None:
        self.pipelines = pipelines
        self.config = config
        self.n_classes = n_classes
        agent_cfg = config.get("agent", {})
        self.calibration_window_trials = agent_cfg.get("calibration_window_trials", 5)
        thresholds = agent_cfg.get("prune_thresholds", {})
        self.min_accuracy = thresholds.get("min_accuracy_online", thresholds.get("min_accuracy", 0.5))
        self.max_latency_ms = thresholds.get("max_latency_ms_online", thresholds.get("max_latency_ms", 300))
        self._batch_agent = PipelineSelectionAgent(config)
        self._batch_agent.calibration_trials = self.calibration_window_trials
        self._batch_agent.min_accuracy = self.min_accuracy
        self._batch_agent.max_latency_ms = self.max_latency_ms
        drift_cfg = dict(agent_cfg.get("drift", {}))
        self._drift_min_accuracy = drift_cfg.get("min_accuracy_absolute_online", drift_cfg.get("min_accuracy_absolute", 0.6))
        drift_cfg.setdefault("min_accuracy_absolute", self._drift_min_accuracy)
        self._calibration_buffer: list[tuple[np.ndarray, int]] = []  # (X_i, y_i)
        self._calibrated = False
        self._selected_pipeline: Pipeline | None = None
        self._metrics: dict[str, PipelineMetrics] = {}
        self._trial_index = 0
        self._live_trial_index = 0
        self._drift_detector = DriftDetector(drift_cfg)
        self._live_predictions: list[OnlineTrialRecord] = []

    def add_trial(self, trial_data: np.ndarray, trial_label: int | None = None) -> None:
        """
        Add one trial. If still in calibration phase, append to buffer; when buffer reaches
        calibration_window_trials, run calibrate() once. Labels required for calibration.
        """
        trial_data = np.asarray(trial_data, dtype=np.float64)
        if trial_data.ndim == 2:
            trial_data = trial_data[np.newaxis, ...]  # (1, n_ch, n_samp)
        self._trial_index += 1
        if not self._calibrated:
            # Only use labeled trials (label >= 0) for calibration; unlabeled (-1) from E file are skipped
            if trial_label is not None and trial_label >= 0:
                self._calibration_buffer.append((trial_data.copy(), int(trial_label)))
            if len(self._calibration_buffer) >= self.calibration_window_trials:
                self.calibrate()
        else:
            self._live_trial_index += 1  # count live trials

    def calibrate(self) -> Pipeline | None:
        """
        Run all pipelines on calibration buffer, prune underperformers, select best.
        Sets selected_pipeline and drift baseline. Called automatically when buffer full.
        """
        if len(self._calibration_buffer) < 2:
            logger.warning("Online calibration: need at least 2 trials with labels")
            return None
        X_cal = np.concatenate([b[0] for b in self._calibration_buffer], axis=0)
        y_cal = np.array([b[1] for b in self._calibration_buffer], dtype=np.int64)
        n = len(y_cal)
        logger.info("Online calibration: running %d pipelines on first %d trials", len(self.pipelines), n)
        self._metrics = self._batch_agent.run_calibration(
            self.pipelines, X_cal, y_cal, self.n_classes, max_parallel=0
        )
        kept = self._batch_agent.prune(self.pipelines)
        self._batch_agent.select_top_n(kept)
        self._selected_pipeline = self._batch_agent.select_best()
        if self._selected_pipeline is not None:
            m = self._metrics.get(self._selected_pipeline.name)
            if m is not None:
                self._drift_detector.set_baseline(m.accuracy)
            self._calibrated = True
            logger.info("Online calibration done. Selected pipeline: %s", self._selected_pipeline.name)
        return self._selected_pipeline

    def is_live_phase(self) -> bool:
        """True after calibration has run and selected pipeline is set."""
        return self._calibrated and self._selected_pipeline is not None

    def predict(self, trial_data: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Run selected pipeline on one trial. Returns (pred_labels, proba) or (pred_labels, None).
        Use only when is_live_phase() is True.
        """
        if self._selected_pipeline is None:
            raise RuntimeError("No pipeline selected; run calibration first")
        trial_data = np.asarray(trial_data, dtype=np.float64)
        if trial_data.ndim == 2:
            trial_data = trial_data[np.newaxis, ...]
        pred = self._selected_pipeline.predict(trial_data)
        proba = None
        try:
            proba = self._selected_pipeline.predict_proba(trial_data)
        except Exception:
            pass
        return pred, proba

    def update_drift(self, true_label: int | None, predicted_label: int) -> bool:
        """
        Update drift detector with one outcome. Returns True if drift detected (re-calibration recommended).
        """
        if true_label is None:
            return False
        correct = int(true_label) == int(predicted_label)
        return self._drift_detector.update(correct)

    @property
    def selected_pipeline(self) -> Pipeline | None:
        return self._selected_pipeline

    def get_calibration_metrics(self) -> dict[str, dict[str, Any]]:
        """Metrics per pipeline from last calibration (for GUI)."""
        out = {}
        for name, m in self._metrics.items():
            out[name] = {
                "accuracy": m.accuracy,
                "kappa": m.kappa,
                "latency_ms": m.latency_ms,
                "stability": m.stability,
                "confidence": m.confidence,
            }
        return out

    def get_drift_detector(self) -> DriftDetector:
        return self._drift_detector

    def get_rolling_accuracy(self) -> float | None:
        """Rolling accuracy over recent live trials (for GUI)."""
        return self._drift_detector.current_window_accuracy

    def append_live_prediction(
        self,
        trial_index: int,
        data: np.ndarray,
        label: int | None,
        predicted: int,
        proba: np.ndarray | None,
        correct: bool | None,
    ) -> None:
        """Append one live trial record for snapshot logging."""
        self._live_predictions.append(
            OnlineTrialRecord(
                trial_index=trial_index,
                data=data,
                label=label,
                predicted=predicted,
                proba=proba,
                correct=correct,
            )
        )

    def get_live_predictions(self) -> list[OnlineTrialRecord]:
        return list(self._live_predictions)

    def trial_count(self) -> int:
        return self._trial_index

    def live_trial_count(self) -> int:
        return self._live_trial_index
