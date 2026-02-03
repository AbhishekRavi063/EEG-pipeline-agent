"""Pipeline Selection Agent: calibration, pruning, top-N, best pipeline, drift re-eval, Optuna HP tuning."""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from bci_framework.pipelines import Pipeline
from bci_framework.utils.metrics import (
    accuracy as _acc,
    cohen_kappa as _kappa,
    f1_macro,
    roc_auc_ovr,
    itr_bits_per_minute,
    compute_all_metrics,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineMetrics:
    """Per-pipeline metrics for selection (accuracy, kappa, ITR, F1, ROC-AUC, latency, stability, confidence)."""

    name: str
    accuracy: float
    kappa: float
    latency_ms: float
    stability: float  # 1 - variance of accuracy over time
    confidence: float
    accuracies_over_time: list[float] = field(default_factory=list)
    f1_macro: float = 0.0
    roc_auc_macro: float = 0.0
    itr_bits_per_minute: float = 0.0
    trial_duration_sec: float = 3.0


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return _acc(y_true, y_pred)


def _confidence(proba: np.ndarray) -> float:
    if proba is None or proba.size == 0:
        return 0.0
    return float(np.mean(np.max(proba, axis=1)))


class PipelineSelectionAgent:
    """
    Phase 1: Exploration – run all pipelines for calibration.
    Phase 2: Pruning – remove low accuracy, high latency, unstable.
    Phase 3: Exploitation – run only top N.
    Phase 4: Continuous adaptation – re-evaluate periodically.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        agent_cfg = config.get("agent", {})
        self.calibration_trials = agent_cfg.get("calibration_trials", 50)
        self.top_n = agent_cfg.get("top_n_pipelines", 3)
        thresholds = agent_cfg.get("prune_thresholds", {})
        self.min_accuracy = thresholds.get("min_accuracy", 0.45)
        self.max_latency_ms = thresholds.get("max_latency_ms", 500)
        self.max_stability_variance = thresholds.get("max_stability_variance", 0.05)
        self.re_eval_interval = agent_cfg.get("re_evaluate_interval_trials", 100)
        self.latency_budget_ms = thresholds.get("latency_budget_ms", 300)
        self._metrics: dict[str, PipelineMetrics] = {}
        self._top_pipelines: list[Pipeline] = []
        self._best_pipeline: Pipeline | None = None
        self._phase = "exploration"
        self._trial_count = 0
        self._drift_detector = None
        self._trial_duration_sec = agent_cfg.get("trial_duration_sec", 3.0)

    def run_calibration(
        self,
        pipelines: list[Pipeline],
        X: np.ndarray,
        y: np.ndarray,
        n_classes: int,
        max_parallel: int = 5,
    ) -> dict[str, PipelineMetrics]:
        """Phase 1: Run all pipelines on calibration data, compute metrics."""
        self._phase = "exploration"
        n = min(len(X), self.calibration_trials)
        X_cal, y_cal = X[:n], y[:n]
        self._metrics = {}
        for pipe in pipelines:
            try:
                pipe.fit(X_cal, y_cal)
                pred, latency_ms = pipe.predict_with_latency(X_cal)
                proba = pipe.predict_proba(X_cal)
                acc = _accuracy(y_cal, pred)
                kappa = _kappa(y_cal, pred, n_classes)
                conf = _confidence(proba)
                latency_per_trial = latency_ms / max(1, len(X_cal))
                all_metrics = compute_all_metrics(y_cal, pred, proba, n_classes, self._trial_duration_sec)
                chunk_size = max(5, n // 5)
                accs = []
                for start in range(0, n - chunk_size, chunk_size):
                    end = start + chunk_size
                    a = _accuracy(y_cal[start:end], pred[start:end])
                    accs.append(a)
                stability = 1.0 - (np.var(accs) if len(accs) > 1 else 0.0)
                self._metrics[pipe.name] = PipelineMetrics(
                    name=pipe.name,
                    accuracy=acc,
                    kappa=kappa,
                    latency_ms=latency_per_trial,
                    stability=float(stability),
                    confidence=conf,
                    accuracies_over_time=accs,
                    f1_macro=all_metrics.get("f1_macro", 0.0),
                    roc_auc_macro=all_metrics.get("roc_auc_macro", 0.0),
                    itr_bits_per_minute=all_metrics.get("itr_bits_per_minute", 0.0),
                    trial_duration_sec=self._trial_duration_sec,
                )
                logger.info(
                    "Pipeline %s: acc=%.3f kappa=%.3f latency=%.1fms stability=%.3f",
                    pipe.name, acc, kappa, latency_ms, stability,
                )
            except Exception as e:
                logger.warning("Pipeline %s failed: %s", pipe.name, e)
                self._metrics[pipe.name] = PipelineMetrics(
                    name=pipe.name,
                    accuracy=0.0,
                    kappa=0.0,
                    latency_ms=999.0,
                    stability=0.0,
                    confidence=0.0,
                    trial_duration_sec=self._trial_duration_sec,
                )
        return self._metrics

    def prune(self, pipelines: list[Pipeline]) -> list[Pipeline]:
        """Phase 2: Remove pipelines below thresholds."""
        self._phase = "pruning"
        kept = []
        for p in pipelines:
            m = self._metrics.get(p.name)
            if m is None:
                kept.append(p)
                continue
            if m.accuracy < self.min_accuracy:
                logger.info("Pruned %s: accuracy %.3f < %.3f", p.name, m.accuracy, self.min_accuracy)
                continue
            latency_limit = getattr(self, "latency_budget_ms", None) or self.max_latency_ms
            if m.latency_ms > latency_limit:
                logger.info("Pruned %s: latency %.1f > %.1f ms", p.name, m.latency_ms, latency_limit)
                continue
            if (1 - m.stability) > self.max_stability_variance:
                logger.info("Pruned %s: stability variance too high", p.name)
                continue
            kept.append(p)
        return kept

    def select_top_n(self, pipelines: list[Pipeline]) -> list[Pipeline]:
        """Phase 3: Keep top N by composite score (accuracy + kappa + stability - latency penalty).
        Tie-break: when scores are equal, prefer lowest latency (faster pipeline)."""
        self._phase = "exploitation"
        scored = []
        for p in pipelines:
            m = self._metrics.get(p.name)
            if m is None:
                scored.append((0.0, float("inf"), p))
                continue
            score = m.accuracy * 0.4 + m.kappa * 0.3 + m.stability * 0.2 - (m.latency_ms / 1000) * 0.1
            score = max(0, score)
            scored.append((score, m.latency_ms, p))
        # Sort by score descending, then by latency ascending (tie-break: lowest latency wins)
        scored.sort(key=lambda x: (-x[0], x[1]))
        self._top_pipelines = [p for _, _, p in scored[: self.top_n]]
        return self._top_pipelines

    def select_best(self) -> Pipeline | None:
        """Set best pipeline = top-1 from top N."""
        if self._top_pipelines:
            self._best_pipeline = self._top_pipelines[0]
            logger.info("Best pipeline: %s", self._best_pipeline.name)
        return self._best_pipeline

    def get_best_pipeline(self) -> Pipeline | None:
        return self._best_pipeline

    def get_metrics(self) -> dict[str, PipelineMetrics]:
        return self._metrics

    def get_top_pipelines(self) -> list[Pipeline]:
        return self._top_pipelines

    def should_re_evaluate(self) -> bool:
        """Phase 4: Whether to re-run evaluation (e.g. every N trials)."""
        return self._trial_count > 0 and self._trial_count % self.re_eval_interval == 0

    def increment_trials(self, n: int = 1) -> None:
        self._trial_count += n

    def get_metrics_dict(self) -> dict[str, dict[str, Any]]:
        """For snapshot logger: pipeline_name -> metrics dict."""
        out = {}
        for name, m in self._metrics.items():
            out[name] = {
                "accuracy": m.accuracy,
                "kappa": m.kappa,
                "latency_ms": m.latency_ms,
                "stability": m.stability,
                "confidence": m.confidence,
                "f1_macro": getattr(m, "f1_macro", 0.0),
                "roc_auc_macro": getattr(m, "roc_auc_macro", 0.0),
                "itr_bits_per_minute": getattr(m, "itr_bits_per_minute", 0.0),
            }
        return out

    def get_drift_detector(self):
        """Return drift detector (create from config if not set)."""
        if self._drift_detector is None:
            from bci_framework.agent.drift_detector import DriftDetector
            self._drift_detector = DriftDetector(self.config.get("agent", {}).get("drift", {}))
        return self._drift_detector

    def set_drift_baseline(self, accuracy: float) -> None:
        """Set baseline accuracy for drift detection (e.g. after calibration)."""
        self.get_drift_detector().set_baseline(accuracy)
