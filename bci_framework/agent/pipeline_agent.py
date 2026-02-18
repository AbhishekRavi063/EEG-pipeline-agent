"""Pipeline Selection Agent: calibration, pruning, top-N, best pipeline, drift re-eval, Optuna HP tuning."""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from bci_framework.pipelines import Pipeline
from bci_framework.utils.splits import k_fold_trial_indices
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
    """Per-pipeline metrics for selection (train/cv accuracy, kappa, ITR, F1, ROC-AUC, latency, stability, confidence)."""

    name: str
    accuracy: float  # train accuracy (full calibration fit)
    kappa: float
    latency_ms: float
    stability: float  # 1 - variance of accuracy over time
    confidence: float
    accuracies_over_time: list[float] = field(default_factory=list)
    f1_macro: float = 0.0
    roc_auc_macro: float = 0.0
    itr_bits_per_minute: float = 0.0
    trial_duration_sec: float = 3.0
    # v2: cross-validation and overfitting
    cv_accuracy: float | None = None  # mean k-fold CV accuracy (used for ranking when set)
    train_accuracy: float | None = None  # same as accuracy when CV is used; explicit for overfit gap
    # v3.2: failed = True if pipeline raised during calibration (do not select)
    failed: bool = False
    cv_score_std: float | None = None  # std of CV fold accuracies when using StratifiedKFold


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return _acc(y_true, y_pred)


def _confidence(proba: np.ndarray) -> float:
    if proba is None or proba.size == 0:
        return 0.0
    return float(np.mean(np.max(proba, axis=1)))


def _clone_pipeline(pipe: Pipeline) -> Pipeline:
    """Return a new pipeline with same config (for CV folds)."""
    config = getattr(pipe, "config", {})
    ch = getattr(pipe.preprocessing_manager, "channel_names", None) or []
    return Pipeline(
        name=pipe.name,
        feature_name=pipe.feature_name,
        classifier_name=pipe.classifier_name,
        fs=pipe.fs,
        n_classes=pipe.n_classes,
        config=config,
        channel_names=ch if ch else None,
    )


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
        # v3.2: no hard CV threshold; selection is argmax CV
        self.min_accuracy = thresholds.get("min_accuracy", 0.0)
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
        # v2: cross-validation pipeline selection
        self.cv_folds = agent_cfg.get("cv_folds", 5)
        self.overfit_penalty_weight = agent_cfg.get("overfit_penalty_weight", 0.2)
        self.prefer_linear_models = agent_cfg.get("prefer_linear_models", True)
        # Smart adaptive pruning (backward compatible: if quick_screening missing, behave like old system)
        qs = agent_cfg.get("quick_screening")
        if qs is None or not isinstance(qs, dict):
            self._quick_screening_enabled = False
            self._quick_subset_fraction = 0.2
            self._quick_min_accuracy_threshold = 0.35
            self._quick_max_pipelines_after_screening = 5
            self._quick_random_state = 42
            self._quick_ranking_metric = "balanced_accuracy"
            self._quick_min_pipelines_to_keep = 3
            self._quick_max_pipelines_to_keep = 5
            self._quick_num_repeats = 1
            self._quick_dynamic_top_k = True
            self._quick_exclude_pipelines: list[str] = []
        else:
            self._quick_screening_enabled = bool(qs.get("enabled", False))
            self._quick_subset_fraction = float(qs.get("subset_fraction", 0.2))
            self._quick_min_accuracy_threshold = float(qs.get("min_accuracy_threshold", 0.35))
            self._quick_max_pipelines_after_screening = int(qs.get("max_pipelines_after_screening", 5))
            self._quick_random_state = int(qs.get("random_state", 42))
            self._quick_ranking_metric = str(qs.get("ranking_metric", "balanced_accuracy")).lower()
            self._quick_min_pipelines_to_keep = int(qs.get("min_pipelines_to_keep", 3))
            self._quick_max_pipelines_to_keep = int(qs.get("max_pipelines_to_keep", 5))
            self._quick_num_repeats = max(1, int(qs.get("num_repeats", 1)))
            self._quick_dynamic_top_k = bool(qs.get("dynamic_top_k", True))
            excl = qs.get("exclude_pipelines", [])
            self._quick_exclude_pipelines = [str(x).lower() for x in (excl if isinstance(excl, list) else [])]
        self._early_cv_stop = bool(agent_cfg.get("early_cv_stop", False))
        ph = agent_cfg.get("progressive_halving")
        if ph is None or not isinstance(ph, dict):
            self._progressive_halving_enabled = False
            self._progressive_halving_reduction_factor = 2
        else:
            self._progressive_halving_enabled = bool(ph.get("enabled", False))
            self._progressive_halving_reduction_factor = max(1, int(ph.get("reduction_factor", 2)))
        self.quick_scores: dict[str, float] = {}
        self.quick_score_std: dict[str, float] = {}
        self.rejected_pipelines: list[str] = []
        self.early_stopped_pipelines: list[str] = []
        self.progressive_halving_used: bool = False
        self.screening_correlation_with_cv: float | None = None
        self.pruning_runtime_stats: dict[str, Any] = {}

    def _screening_metric(
        self, y_true: np.ndarray, y_pred: np.ndarray, n_classes: int
    ) -> float:
        """Compute ranking metric for screening: balanced_accuracy, kappa, or accuracy."""
        try:
            from sklearn.metrics import balanced_accuracy_score
        except ImportError:
            balanced_accuracy_score = None
        metric = getattr(self, "_quick_ranking_metric", "balanced_accuracy")
        if metric == "balanced_accuracy" and balanced_accuracy_score is not None:
            return float(balanced_accuracy_score(y_true, y_pred))
        if metric == "kappa":
            return float(_kappa(y_true, y_pred, n_classes))
        return _accuracy(y_true, y_pred)

    def _quick_screen_pipelines(
        self,
        pipelines: list[Pipeline],
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_classes: int = 2,
        trial_ids: np.ndarray | None = None,
    ) -> list[Pipeline]:
        """
        Cheap screening on a subset of training data; dynamic top-K ranking, no hard threshold.
        Uses StratifiedShuffleSplit (trial-level if trial_ids provided). Transfer disabled.
        Excluded pipelines (e.g. GEDAI) skip screening and are always included in full CV.
        On error or if all scored pipelines would be dropped, returns original list.
        """
        if not getattr(self, "_quick_screening_enabled", False):
            return pipelines
        original = list(pipelines)
        self.quick_scores = {}
        self.quick_score_std = {}
        self.rejected_pipelines = []
        try:
            from sklearn.model_selection import StratifiedShuffleSplit
        except ImportError:
            logger.warning("Quick screening failed (StratifiedShuffleSplit not available); using full list")
            return original
        try:
            n = len(X_train)
            n_subset = max(2, int(n * self._quick_subset_fraction))
            # Ensure enough samples per class for StratifiedShuffleSplit (min 2 per class)
            n_subset = max(n_subset, n_classes * 2) if n_classes else n_subset
            if n_subset >= n:
                return pipelines
            # Subset indices (stratified so each class has enough for split)
            try:
                from sklearn.model_selection import train_test_split
                _, subset_idx = train_test_split(
                    np.arange(n), test_size=n_subset / n, stratify=y_train, random_state=self._quick_random_state
                )
                subset_idx = np.array(subset_idx)
            except Exception:
                rng = np.random.default_rng(self._quick_random_state)
                subset_idx = rng.choice(n, size=min(n_subset, n), replace=False)
                subset_idx = np.sort(subset_idx)
            X_sub = X_train[subset_idx]
            y_sub = y_train[subset_idx]
            trial_sub = trial_ids[subset_idx] if trial_ids is not None and len(trial_ids) == n else None if trial_ids is not None and len(trial_ids) == n else None

            # Train/val split: trial-level to prevent leakage when trial_ids provided
            if trial_sub is not None:
                unique_trials = np.unique(trial_sub)
                if len(unique_trials) < 2:
                    logger.warning("Quick screening: too few unique trials for split; using sample-level split")
                    trial_sub = None
                else:
                    y_per_trial = np.array([y_sub[trial_sub == t][0] for t in unique_trials])
                    try:
                        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=self._quick_random_state)
                        train_trials, val_trials = next(sss.split(unique_trials, y_per_trial))
                        train_trial_set = set(unique_trials[train_trials])
                        train_idx = np.array([i for i in range(len(trial_sub)) if trial_sub[i] in train_trial_set])
                        val_idx = np.array([i for i in range(len(trial_sub)) if trial_sub[i] not in train_trial_set])
                    except Exception:
                        trial_sub = None
            if trial_sub is None:
                sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=self._quick_random_state)
                train_idx, val_idx = next(sss.split(np.arange(len(y_sub)), y_sub))
                logger.warning(
                    "Quick screening split: sample-level (no trial IDs provided; cannot verify trial leakage)"
                )
            else:
                logger.debug("Quick screening split verified: no trial leakage detected")

            X_sub_train, X_sub_val = X_sub[train_idx], X_sub[val_idx]
            y_sub_train, y_sub_val = y_sub[train_idx], y_sub[val_idx]

            # Pipelines excluded from screening (e.g. require GEDAI leadfield)
            exclude_keys = getattr(self, "_quick_exclude_pipelines", []) or []
            excluded: list[Pipeline] = []
            to_score: list[Pipeline] = []
            for pipe in pipelines:
                if any(k in pipe.name.lower() for k in exclude_keys):
                    excluded.append(pipe)
                    logger.info("Pipeline %s excluded from screening (special resource required)", pipe.name)
                else:
                    to_score.append(pipe)

            if not to_score:
                return original

            # Multi-repeat scoring for stability
            num_repeats = getattr(self, "_quick_num_repeats", 1)
            for pipe in to_score:
                repeat_scores: list[float] = []
                for rep in range(num_repeats):
                    rs = self._quick_random_state + rep
                    sss_rep = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=rs)
                    if trial_sub is not None:
                        tr_idx, v_idx = next(sss_rep.split(np.arange(len(y_sub)), y_sub))
                    else:
                        tr_idx, v_idx = next(sss_rep.split(np.arange(len(y_sub)), y_sub))
                    X_tr, X_va = X_sub[tr_idx], X_sub[v_idx]
                    y_tr, y_va = y_sub[tr_idx], y_sub[v_idx]
                    try:
                        pipe_cv = _clone_pipeline(pipe)
                        pipe_cv._transfer_enabled = False
                        pipe_cv.fit(X_tr, y_tr)
                        pred = pipe_cv.predict(X_va)
                        repeat_scores.append(self._screening_metric(y_va, pred, n_classes))
                    except Exception as e:
                        logger.debug("Quick screening repeat failed for %s: %s", pipe.name, e)
                        repeat_scores.append(0.0)
                mean_s = float(np.mean(repeat_scores))
                std_s = float(np.std(repeat_scores)) if len(repeat_scores) > 1 else 0.0
                self.quick_scores[pipe.name] = mean_s
                self.quick_score_std[pipe.name] = std_s
                if std_s > 0.1:
                    logger.warning("Quick screening unstable for %s (std=%.3f)", pipe.name, std_s)

            # Dynamic top-K: no hard threshold
            total_scored = len(to_score)
            min_keep = getattr(self, "_quick_min_pipelines_to_keep", 3)
            max_keep = getattr(self, "_quick_max_pipelines_to_keep", 5)
            dynamic = getattr(self, "_quick_dynamic_top_k", True)
            if dynamic:
                top_k = min(max_keep, max(min_keep, int(0.5 * total_scored)))
            else:
                top_k = min(getattr(self, "_quick_max_pipelines_after_screening", 5), total_scored)
            top_k = max(1, min(top_k, total_scored))

            scored_list: list[tuple[float, Pipeline]] = [(self.quick_scores[p.name], p) for p in to_score]
            scored_list.sort(key=lambda x: -x[0])
            kept_scored = [p for _, p in scored_list[:top_k]]
            for p in scored_list[top_k:]:
                self.rejected_pipelines.append(p[1].name)
            kept = excluded + kept_scored
            logger.info(
                "Quick screening: pipelines before=%d, after=%d (excluded=%d, top_k=%d); rejected=%s",
                len(pipelines), len(kept), len(excluded), top_k, self.rejected_pipelines,
            )
            return kept
        except Exception as e:
            logger.warning("Quick screening failed (%s); using full pipeline list", e)
            self.quick_scores = {}
            self.quick_score_std = {}
            self.rejected_pipelines = []
            return original

    def _progressive_halving_stage(
        self,
        pipelines: list[Pipeline],
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        sw_cal: np.ndarray | None,
        source_subject_ids: np.ndarray | None,
        n_classes: int,
    ) -> list[Pipeline]:
        """
        Evaluate remaining pipelines on 50%% of calibration data; keep top 1/reduction_factor for full CV.
        Only called when progressive_halving.enabled is True.
        """
        if not pipelines:
            return pipelines
        rng = np.random.default_rng(self._quick_random_state)
        n = len(X_cal)
        half = max(2, n // 2)
        perm = rng.permutation(n)
        train_idx = perm[:half]
        val_idx = perm[half:]
        X_tr, X_val = X_cal[train_idx], X_cal[val_idx]
        y_tr, y_val = y_cal[train_idx], y_cal[val_idx]
        sw_tr = sw_cal[train_idx] if sw_cal is not None else None
        sid_tr = source_subject_ids[train_idx] if source_subject_ids is not None and len(source_subject_ids) == n else None
        scored: list[tuple[float, Pipeline]] = []
        for pipe in pipelines:
            try:
                pipe_cv = _clone_pipeline(pipe)
                pipe_cv._transfer_enabled = False
                pipe_cv.fit(X_tr, y_tr, sample_weight=sw_tr, subject_ids=sid_tr)
                pred = pipe_cv.predict(X_val)
                acc = _accuracy(y_val, pred)
                scored.append((acc, pipe))
            except Exception as e:
                logger.debug("Progressive halving skip %s: %s", pipe.name, e)
                scored.append((0.0, pipe))
        scored.sort(key=lambda x: -x[0])
        k = max(1, len(pipelines) // self._progressive_halving_reduction_factor)
        kept = [p for _, p in scored[:k]]
        logger.info("Progressive halving: %d -> %d pipelines for full CV", len(pipelines), len(kept))
        self.progressive_halving_used = True
        return kept

    def run_calibration(
        self,
        pipelines: list[Pipeline],
        X: np.ndarray,
        y: np.ndarray,
        n_classes: int,
        max_parallel: int = 5,
        X_target_cal: np.ndarray | None = None,
        loso_fold_info: dict[str, Any] | None = None,
        sample_weight: np.ndarray | None = None,
        trial_ids: np.ndarray | None = None,
    ) -> dict[str, PipelineMetrics]:
        """Phase 1: Run all pipelines on calibration data; use k-fold CV for ranking when cv_folds > 1.
        When X_target_cal is provided (LOSO/transfer), adapter fits on source + unlabeled target (no target labels).
        loso_fold_info: optional {"target_subject": int, "source_subjects": list} for LOSO validation logging.
        sample_weight: optional (n_cal,) weights for classifier fit (e.g. subject similarity)."""
        self._phase = "exploration"
        n = min(len(X), self.calibration_trials)
        X_cal, y_cal = X[:n], y[:n]
        sw_cal = sample_weight[:n] if sample_weight is not None and len(sample_weight) >= n else None
        self._metrics = {}
        self.early_stopped_pipelines = []
        self.pruning_runtime_stats = {}
        t_cal_start = time.perf_counter()
        pipelines_before_pruning = len(pipelines)

        # Quick screening: reduce pipelines before full CV (no transfer, no test data)
        trial_ids_cal = trial_ids[:n] if trial_ids is not None and len(trial_ids) >= n else None
        pipelines_after_screen = self._quick_screen_pipelines(
            pipelines, X_cal, y_cal, n_classes=n_classes, trial_ids=trial_ids_cal
        )
        pipelines_after_pruning = len(pipelines_after_screen)
        if pipelines_after_pruning < pipelines_before_pruning:
            logger.info(
                "Quick screening reduced pipelines from %d to %d",
                pipelines_before_pruning, pipelines_after_pruning,
            )
        pipelines = pipelines_after_screen

        # Optional progressive halving: evaluate on 50%% of data, keep top 1/reduction_factor
        if getattr(self, "_progressive_halving_enabled", False) and len(pipelines) > 1:
            pipelines = self._progressive_halving_stage(
                pipelines, X_cal, y_cal, sw_cal, None, n_classes,
            )

        source_subject_ids = None
        if loso_fold_info is not None:
            target_subject = loso_fold_info.get("target_subject")
            source_subjects = loso_fold_info.get("source_subjects", [])
            source_subject_ids = loso_fold_info.get("source_subject_ids")
            if target_subject is not None and source_subjects is not None:
                assert target_subject not in source_subjects, (
                    f"LOSO violation: target_subject {target_subject} must not be in source_subjects {source_subjects}"
                )
                logger.info("[FOLD] Target subject: %s", target_subject)
                logger.info("[FOLD] Source subjects: %s", source_subjects)
            logger.info("[DATA] Source trials: %d", len(X_cal))
            logger.info("[DATA] Target train (unlabeled): %d", len(X_target_cal) if X_target_cal is not None else 0)

        use_cv = self.cv_folds > 1 and n >= self.cv_folds
        try:
            from sklearn.model_selection import StratifiedKFold
            skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42) if use_cv and n >= self.cv_folds else None
        except ImportError:
            skf = None
        folds = list(skf.split(np.arange(n), y_cal)) if (use_cv and skf is not None) else (
            k_fold_trial_indices(n, n_folds=self.cv_folds, shuffle=True, random_state=42) if use_cv else []
        )
        total_folds = len(folds)
        completed_cv_scores: dict[str, float] = {}
        cv_fits_executed = 0

        for pipe in pipelines:
            try:
                cv_acc_mean = None
                cv_acc_std = None
                if use_cv and folds:
                    cv_accs = []
                    for fold_idx, (train_idx, val_idx) in enumerate(folds):
                        X_tr, y_tr = X_cal[train_idx], y_cal[train_idx]
                        X_val, y_val = X_cal[val_idx], y_cal[val_idx]
                        sw_tr = sw_cal[train_idx] if sw_cal is not None else None
                        sid_tr = source_subject_ids[train_idx] if source_subject_ids is not None and len(source_subject_ids) == len(X_cal) else None
                        pipe_cv = _clone_pipeline(pipe)
                        if getattr(pipe_cv, "_transfer_enabled", False) and X_target_cal is not None and len(X_target_cal) > 0:
                            pipe_cv.fit(X_tr, y_tr, X_target=X_target_cal, sample_weight=sw_tr, subject_ids=sid_tr)
                        else:
                            pipe_cv.fit(X_tr, y_tr, sample_weight=sw_tr, subject_ids=sid_tr)
                        pred_val = pipe_cv.predict(X_val)
                        cv_accs.append(_accuracy(y_val, pred_val))
                        # MOABB-style: free fold arrays on LOSO/transfer to reduce peak RAM
                        if getattr(pipe_cv, "_transfer_enabled", False) and X_target_cal is not None:
                            import gc
                            del pipe_cv, X_tr, y_tr, X_val, y_val, pred_val
                            gc.collect()
                        # Early CV stop: if this pipeline cannot beat best completed score, stop folds
                        if getattr(self, "_early_cv_stop", False) and total_folds > 0:
                            remaining_folds = total_folds - (fold_idx + 1)
                            max_possible = (sum(cv_accs) + remaining_folds * 1.0) / total_folds
                            best_score_so_far = max(completed_cv_scores.values()) if completed_cv_scores else 0.0
                            if max_possible < best_score_so_far:
                                self.early_stopped_pipelines.append(pipe.name)
                                logger.info("Early CV stop: %s (max_possible=%.3f < best=%.3f)", pipe.name, max_possible, best_score_so_far)
                                break
                    cv_acc_mean = float(np.mean(cv_accs))
                    cv_acc_std = float(np.std(cv_accs)) if len(cv_accs) > 1 else None
                    if len(cv_accs) == total_folds:
                        completed_cv_scores[pipe.name] = cv_acc_mean
                    cv_fits_executed += len(cv_accs)
                cv_fits_executed += 1  # final fit on full calibration

                if getattr(pipe, "_transfer_enabled", False) and X_target_cal is not None and len(X_target_cal) > 0:
                    pipe.fit(X_cal, y_cal, X_target=X_target_cal, sample_weight=sw_cal, subject_ids=source_subject_ids)
                else:
                    pipe.fit(X_cal, y_cal, sample_weight=sw_cal, subject_ids=source_subject_ids)
                pred, latency_ms = pipe.predict_with_latency(X_cal)
                proba = pipe.predict_proba(X_cal)
                train_acc = _accuracy(y_cal, pred)
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
                    accuracy=train_acc,
                    kappa=kappa,
                    latency_ms=latency_per_trial,
                    stability=float(stability),
                    confidence=conf,
                    accuracies_over_time=accs,
                    f1_macro=all_metrics.get("f1_macro", 0.0),
                    roc_auc_macro=all_metrics.get("roc_auc_macro", 0.0),
                    itr_bits_per_minute=all_metrics.get("itr_bits_per_minute", 0.0),
                    trial_duration_sec=self._trial_duration_sec,
                    cv_accuracy=cv_acc_mean,
                    train_accuracy=train_acc,
                    failed=False,
                    cv_score_std=cv_acc_std,
                )
                logger.info(
                    "Pipeline %s: train_acc=%.3f cv_acc=%s kappa=%.3f latency=%.1fms stability=%.3f",
                    pipe.name, train_acc, f"{cv_acc_mean:.3f}" if cv_acc_mean is not None else "N/A",
                    kappa, latency_per_trial, stability,
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
                    failed=True,
                )
        t_cal_end = time.perf_counter()
        total_runtime = t_cal_end - t_cal_start
        self.pruning_runtime_stats = {
            "pipelines_before": pipelines_before_pruning,
            "pipelines_after": pipelines_after_pruning,
            "cv_fits_executed": cv_fits_executed,
            "runtime_seconds": round(total_runtime, 3),
        }
        logger.info(
            "Calibration runtime: %.2f s; pipelines before=%d after=%d; cv_fits=%d",
            total_runtime, pipelines_before_pruning, pipelines_after_pruning, cv_fits_executed,
        )

        # Ranking correlation: screening vs full CV
        if getattr(self, "_quick_screening_enabled", False) and self.quick_scores and self._metrics:
            try:
                from scipy.stats import spearmanr
                names_with_both = [nm for nm in self._metrics if nm in self.quick_scores and not getattr(self._metrics[nm], "failed", False)]
                if len(names_with_both) >= 2:
                    cv_acc = self._ranking_accuracy
                    screen_rank = sorted(names_with_both, key=lambda x: -self.quick_scores[x])
                    cv_rank = sorted(names_with_both, key=lambda x: -(cv_acc(self._metrics[x]) or 0.0))
                    rank_screen = {nm: i for i, nm in enumerate(screen_rank)}
                    rank_cv = {nm: i for i, nm in enumerate(cv_rank)}
                    r_screen = [rank_screen[n] for n in names_with_both]
                    r_cv = [rank_cv[n] for n in names_with_both]
                    corr, _ = spearmanr(r_screen, r_cv)
                    self.screening_correlation_with_cv = float(corr) if np.isfinite(corr) else None
                    if self.screening_correlation_with_cv is not None and self.screening_correlation_with_cv < 0.3:
                        logger.warning(
                            "Low screening-to-CV ranking correlation (%.3f). Screening may be unreliable.",
                            self.screening_correlation_with_cv,
                        )
                    else:
                        logger.info("Screening–CV Spearman correlation: %.3f", self.screening_correlation_with_cv or 0.0)
            except Exception as e:
                logger.debug("Could not compute screening–CV correlation: %s", e)
                self.screening_correlation_with_cv = None

        # Perfect-accuracy sanity check for potential leakage
        for name, m in self._metrics.items():
            if getattr(m, "failed", False):
                continue
            acc = m.cv_accuracy if m.cv_accuracy is not None else m.accuracy
            if acc is not None and acc >= 1.0 - 1e-6:
                try:
                    y_shuf = np.array(y_cal, copy=True)
                    rng = np.random.default_rng(42)
                    rng.shuffle(y_shuf)
                    pipe = next((p for p in pipelines if p.name == name), None)
                    if pipe is not None:
                        pipe_sanity = _clone_pipeline(pipe)
                        pipe_sanity._transfer_enabled = False
                        pipe_sanity.fit(X_cal, y_shuf)
                        pred_sanity = pipe_sanity.predict(X_cal)
                        shuffle_acc = _accuracy(y_shuf, pred_sanity)
                        if shuffle_acc > 0.7:
                            logger.critical(
                                "Potential leakage detected: pipeline %s has accuracy 1.0 but shuffled-label accuracy %.3f > 0.7",
                                name, shuffle_acc,
                            )
                except Exception as e:
                    logger.debug("Leakage sanity check skipped for %s: %s", name, e)
                break
        return self._metrics

    def prune(self, pipelines: list[Pipeline]) -> list[Pipeline]:
        """Phase 2: Remove pipelines below thresholds (use CV accuracy for ranking when available)."""
        self._phase = "pruning"
        kept = []
        for p in pipelines:
            m = self._metrics.get(p.name)
            if m is None:
                kept.append(p)
                continue
            acc_for_threshold = m.cv_accuracy if m.cv_accuracy is not None else m.accuracy
            if acc_for_threshold < self.min_accuracy:
                logger.info("Pruned %s: accuracy %.3f < %.3f", p.name, acc_for_threshold, self.min_accuracy)
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

    def _ranking_accuracy(self, m: PipelineMetrics) -> float:
        """Accuracy used for ranking: CV when available, else train."""
        return m.cv_accuracy if m.cv_accuracy is not None else m.accuracy

    def _is_linear_model(self, pipe: Pipeline) -> bool:
        """True if pipeline uses LDA or linear SVM (prefer when tie)."""
        if not getattr(self, "prefer_linear_models", False):
            return False
        c = (pipe.classifier_name or "").lower()
        if c == "lda":
            return True
        if c == "svm":
            kernel = (self.config.get("classifiers", {}).get("svm") or {}).get("kernel", "rbf")
            return str(kernel).lower() in ("linear",)
        return False

    def select_top_n(self, pipelines: list[Pipeline]) -> list[Pipeline]:
        """Phase 3: Keep top N by composite score (CV acc, kappa, stability, latency, overfit penalty).
        score = 0.4*cv_accuracy + 0.3*kappa + 0.2*stability - 0.1*latency - 0.2*(train - cv).
        Tie-break: prefer linear models (LDA, linear SVM), then lowest latency."""
        self._phase = "exploitation"
        rank_acc = self._ranking_accuracy
        overfit_w = getattr(self, "overfit_penalty_weight", 0.2)
        scored = []
        for p in pipelines:
            m = self._metrics.get(p.name)
            if m is None:
                scored.append((0.0, not self._is_linear_model(p), float("inf"), p))
                continue
            acc = rank_acc(m)
            overfit_gap = (m.train_accuracy or m.accuracy) - (m.cv_accuracy if m.cv_accuracy is not None else m.accuracy)
            overfit_gap = max(0.0, overfit_gap)
            score = (
                0.4 * acc
                + 0.3 * m.kappa
                + 0.2 * m.stability
                - 0.1 * (m.latency_ms / 1000.0)
                - overfit_w * overfit_gap
            )
            score = max(0.0, score)
            # tie-break: linear first (False < True), then latency
            scored.append((score, not self._is_linear_model(p), m.latency_ms, p))
        scored.sort(key=lambda x: (-x[0], x[1], x[2]))
        self._top_pipelines = [p for _, _, _, p in scored[: self.top_n]]
        return self._top_pipelines

    def select_best(self, pipelines: list[Pipeline] | None = None) -> Pipeline | None:
        """v3.2: Always select argmax CV. No hard threshold. pipelines = full list from run_calibration (optional)."""
        candidate_list = pipelines if pipelines is not None else self._top_pipelines
        if not candidate_list:
            self._best_pipeline = None
            return None
        valid = [
            p for p in candidate_list
            if self._metrics.get(p.name) is not None
            and not getattr(self._metrics[p.name], "failed", False)
            and (self._metrics[p.name].cv_accuracy is not None or self._metrics[p.name].accuracy is not None)
        ]
        if len(valid) == 0:
            raise RuntimeError("No valid pipelines completed.")
        best = max(valid, key=lambda p: self._ranking_accuracy(self._metrics[p.name]))
        self._best_pipeline = best
        logger.info("[AGENT] Selected best pipeline based on highest CV score: %s", best.name)
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
                "cv_accuracy": getattr(m, "cv_accuracy", None),
                "train_accuracy": getattr(m, "train_accuracy", None),
                "cv_score_std": getattr(m, "cv_score_std", None),
            }
        return out

    def get_adaptive_pruning_info(self) -> dict[str, Any]:
        """For metrics.json: adaptive_pruning block (and backward-compat flat keys)."""
        qs_enabled = getattr(self, "_quick_screening_enabled", False)
        remaining = list(self._metrics.keys()) if self._metrics else []
        quick_screening_flat = {
            "enabled": qs_enabled,
            "scores": dict(getattr(self, "quick_scores", {})),
            "rejected": list(getattr(self, "rejected_pipelines", [])),
            "remaining": remaining,
        }
        if hasattr(self, "quick_score_std") and self.quick_score_std:
            quick_screening_flat["std"] = dict(self.quick_score_std)
        out: dict[str, Any] = {
            "quick_screening": quick_screening_flat,
            "early_stopped_pipelines": list(getattr(self, "early_stopped_pipelines", [])),
            "progressive_halving_used": getattr(self, "progressive_halving_used", False),
        }
        if qs_enabled or out["early_stopped_pipelines"] or out["progressive_halving_used"] or getattr(self, "pruning_runtime_stats", {}):
            qs_block: dict[str, Any] = {
                "enabled": qs_enabled,
                "scores": dict(getattr(self, "quick_scores", {})),
                "correlation_with_full_cv": getattr(self, "screening_correlation_with_cv", None),
            }
            if hasattr(self, "quick_score_std") and self.quick_score_std:
                qs_block["std"] = dict(self.quick_score_std)
            out["adaptive_pruning"] = {
                "quick_screening": qs_block,
                "runtime_stats": dict(getattr(self, "pruning_runtime_stats", {})),
                "early_stopped_pipelines": out["early_stopped_pipelines"],
                "progressive_halving_used": out["progressive_halving_used"],
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
