"""
Preprocessing evaluation: impact of ICA vs baseline bandpass on cross-subject motor imagery decoding.

CRITICAL: LOSO removed. GEDAI disabled. Only subject-wise GroupKFold evaluation.
Dataset: Physionet Motor Movement/Imagery (EEGBCI) via MOABB, motor imagery only, trial 0–3 s, 250 Hz.
Classifier: FIXED — Bandpass 8–30 Hz, OAS covariance, tangent space, StandardScaler, LR (C ∈ [0.01, 0.1, 1, 10, 100]).
Conditions: A = bandpass only, B = bandpass + ICA. ICA fit only on training subjects per fold.
Evaluation: GroupKFold(n_splits=5), groups=subject_id. random_state=42 everywhere.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
PHYSIONET_DATASET = "PhysionetMI"
TRIAL_TMIN = 0.0
TRIAL_TMAX = 3.0
N_GROUP_FOLDS = 5
N_PERM = 10_000
N_BOOTSTRAP = 2000
C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]

# A = bandpass only, B = bandpass + ICA, C = bandpass + GEDAI (professor: paired comparisons A/B/C)
CONDITION_LABELS = {"A": "bandpass_only", "B": "bandpass_ica", "C": "bandpass_gedai"}
CONDITION_ENABLED = {
    "A": [],
    "B": ["ica"],
    "C": ["gedai"],
}
ALL_PIPELINES = ["A", "B", "C"]


def load_physionet_mi(
    subjects: list[int] | None = None,
    tmin: float = TRIAL_TMIN,
    tmax: float = TRIAL_TMAX,
):
    """
    Load Physionet MI (MOABB), motor imagery only, trial window 0–3 s, 250 Hz.
    Returns X, y, subject_ids, fs, ch_names, n_classes, capabilities, class_distribution.
    class_distribution: dict[subject_id] -> list of per-class counts (for metadata).
    """
    from bci_framework.datasets.moabb_loader import MOABBDatasetLoader

    loader = MOABBDatasetLoader(
        dataset_name=PHYSIONET_DATASET,
        paradigm="motor_imagery",
        resample=250,
        tmin=tmin,
        tmax=tmax,
    )
    if subjects is None:
        subjects = loader.get_subject_ids()
    subjects = list(subjects)
    result = loader.load(subjects=subjects, download_if_missing=True)

    if isinstance(result, dict):
        parts_x, parts_y, parts_sid = [], [], []
        class_distribution = {}
        for sid in subjects:
            ds = result.get(sid)
            if ds is None:
                continue
            x = np.asarray(ds.data, dtype=np.float32)
            y = np.asarray(ds.labels, dtype=np.int64).ravel()
            labeled = y >= 0
            x, y = x[labeled], y[labeled]
            parts_x.append(x)
            parts_y.append(y)
            parts_sid.append(np.full(len(y), sid, dtype=np.int64))
            n_cl = len(ds.class_names)
            counts = [int(np.sum(y == c)) for c in range(n_cl)]
            class_distribution[int(sid)] = counts
        X = np.concatenate(parts_x, axis=0)
        X = X.astype(np.float32, copy=False)
        y = np.concatenate(parts_y, axis=0)
        subject_ids = np.concatenate(parts_sid, axis=0)
        ds0 = result[subjects[0]]
    else:
        ds0 = result
        X = np.asarray(ds0.data, dtype=np.float32)
        y = np.asarray(ds0.labels, dtype=np.int64).ravel()
        labeled = y >= 0
        X = X[labeled]
        y = y[labeled]
        X = X.astype(np.float32, copy=False)
        subject_ids = np.full(len(y), subjects[0], dtype=np.int64)
        n_cl = len(ds0.class_names)
        class_distribution = {int(subjects[0]): [int(np.sum(y == c)) for c in range(n_cl)]}

    n_subjects = len(np.unique(subject_ids))
    total_trials = len(y)
    logger.info(
        "Dataset: n_subjects=%d, total_trials=%d, n_classes=%d, class_distribution_per_subject=%s",
        n_subjects, total_trials, len(ds0.class_names), class_distribution,
    )
    return X, y, subject_ids, ds0.fs, ds0.channel_names, len(ds0.class_names), loader.capabilities, class_distribution


def get_fixed_pipeline_config(condition: str, gedai_debug: bool = False) -> dict[str, Any]:
    """Config for FIXED pipeline: bandpass 8–30 Hz, OAS, tangent, StandardScaler, LR. A/B/C (bandpass, ICA, GEDAI)."""
    from bci_framework.utils.config_loader import load_config, get_config

    if condition not in CONDITION_ENABLED:
        raise ValueError("Condition must be one of %s. Got: %s" % (list(CONDITION_ENABLED.keys()), condition))
    load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    cfg = copy.deepcopy(get_config())
    cfg["preprocessing"] = cfg.get("preprocessing", {}) | {
        "bandpass_low": 8.0,
        "bandpass_high": 30.0,
        "adaptive_motor_band": True,
        "motor_band_low": 8.0,
        "motor_band_high": 30.0,
    }
    cfg["pipelines"] = {
        "auto_generate": False,
        "max_combinations": 1,
        "explicit": [["riemann_tangent_oas", "logistic_regression"]],
    }
    cfg["spatial_filter"] = {"enabled": True, "method": "laplacian_auto", "auto_select": False}
    cfg["features"] = cfg.get("features", {}) | {
        "riemann_tangent_oas": {"z_score_tangent": True, "apply_bandpass": False},
    }
    cfg["classifiers"] = cfg.get("classifiers", {}) | {
        "logistic_regression": {
            "tune_C": True,
            "C_grid": C_GRID,
            "cv_folds": 5,
            "random_state": RANDOM_STATE,
            "max_iter": 5000,
        },
    }
    cfg["transfer"] = {"enabled": False}
    enabled = CONDITION_ENABLED[condition]
    adv = copy.deepcopy(cfg.get("advanced_preprocessing", {}))
    cfg["advanced_preprocessing"] = {**adv, "enabled": list(enabled)}
    if gedai_debug and "gedai" in enabled:
        gedai_cfg = cfg["advanced_preprocessing"].get("gedai") or {}
        cfg["advanced_preprocessing"]["gedai"] = {**gedai_cfg, "debug": True}
    return cfg


def build_single_pipeline(config: dict, fs: float, n_classes: int, channel_names: list) -> Any:
    """Build one Pipeline instance (fixed feature + classifier)."""
    from bci_framework.pipelines import PipelineRegistry

    registry = PipelineRegistry(config)
    pipelines = registry.build_pipelines(fs=fs, n_classes=n_classes, channel_names=channel_names)
    if not pipelines:
        raise RuntimeError("No pipeline built; check config pipelines.explicit")
    return pipelines[0]


def compute_fold_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    n_classes: int,
) -> dict[str, float]:
    """Accuracy, macro ROC AUC, Cohen's kappa."""
    from bci_framework.utils.metrics import accuracy, cohen_kappa, roc_auc_ovr

    acc = accuracy(y_true, y_pred)
    kappa = cohen_kappa(y_true, y_pred, n_classes)
    auc = roc_auc_ovr(y_true, y_proba, n_classes) if y_proba is not None else 0.0
    return {"accuracy": acc, "roc_auc_macro": auc, "cohen_kappa": kappa}


def run_one_fold(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    condition: str,
    fs: float,
    n_classes: int,
    channel_names: list,
    spatial_capabilities: dict | None,
    random_state: int = RANDOM_STATE,
    return_fit_subjects: bool = False,
    gedai_debug: bool = False,
) -> dict[str, Any]:
    """
    Fit pipeline (preprocessing + feature + classifier) on train only; evaluate on test.
    STRICT: If condition B (ICA), fit subjects must not include any test subject → raise if violated.
    """
    train_subjects = set(np.unique(subject_ids[train_idx]))
    test_subjects = set(np.unique(subject_ids[test_idx]))
    overlap = train_subjects & test_subjects
    assert len(overlap) == 0, (
        "Subject leakage: train and test must be disjoint. Overlap: %s. Experiment invalid." % overlap
    )

    config = get_fixed_pipeline_config(condition, gedai_debug=gedai_debug)
    config["spatial_capabilities"] = spatial_capabilities
    pipe = build_single_pipeline(config, fs, n_classes, channel_names)

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    fit_subjects = list(np.unique(subject_ids[train_idx]))

    logger.info("Condition %s fit subjects: %s (test subjects must NOT be in this list)", condition, fit_subjects)
    if condition in ("B", "C") and test_subjects:
        if test_subjects.intersection(set(fit_subjects)):
            raise RuntimeError(
                "Preprocessing fit must not include test subject. Fit subjects: %s; test subjects: %s. "
                "If any fold accidentally includes test subject in preprocessing fitting → raise error." % (fit_subjects, list(test_subjects))
            )

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test) if hasattr(pipe, "predict_proba") else None
    if y_proba is None and hasattr(pipe, "classifier") and hasattr(pipe.classifier, "predict_proba"):
        try:
            X_feat = pipe.transform(X_test)
            y_proba = pipe.classifier.predict_proba(X_feat)
        except Exception:
            y_proba = None

    metrics = compute_fold_metrics(y_test, y_pred, y_proba, n_classes)
    out = {"metrics": metrics, "n_test": len(y_test)}
    if return_fit_subjects:
        out["fit_subjects"] = fit_subjects
        out["test_subjects"] = list(test_subjects)

    # Memory-safe: release large arrays so GC can reclaim before next fold
    del X_train, X_test, pipe
    if y_proba is not None:
        del y_proba
    import gc
    gc.collect()
    return out


def run_group_kfold(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    condition: str,
    fs: float,
    n_classes: int,
    channel_names: list,
    spatial_capabilities: dict | None,
    n_splits: int = N_GROUP_FOLDS,
    random_state: int = RANDOM_STATE,
    gedai_debug: bool = False,
) -> list[dict[str, Any]]:
    """GroupKFold(n_splits=5), groups=subject_id. STRICT: assert no subject in both train and test per fold."""
    from sklearn.model_selection import GroupKFold

    gkf = GroupKFold(n_splits=n_splits)
    rows = []
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=subject_ids)):
        print("  [GroupKFold] condition %s fold %d/%d ..." % (condition, fold_idx + 1, n_splits), flush=True)
        train_subjects = set(np.unique(subject_ids[train_idx]))
        test_subjects = set(np.unique(subject_ids[test_idx]))
        if len(train_subjects & test_subjects) != 0:
            raise RuntimeError(
                "GroupKFold subject leakage: fold %d has overlap %s. "
                "For every fold: assert len(set(train_subjects) & set(test_subjects)) == 0." % (fold_idx, train_subjects & test_subjects)
            )
        fold_out = run_one_fold(
            X, y, subject_ids,
            train_idx, test_idx,
            condition, fs, n_classes, channel_names, spatial_capabilities,
            random_state=random_state,
            return_fit_subjects=True,
            gedai_debug=gedai_debug,
        )
        rows.append({
            "fold": fold_idx,
            "metrics": fold_out["metrics"],
            "n_test": fold_out["n_test"],
            "train_subjects": list(train_subjects),
            "test_subjects": list(test_subjects),
        })
        # Memory-safe: encourage GC between folds to avoid accumulation
        import gc
        gc.collect()
    return rows


def aggregate_fold_metrics(rows: list[dict], metric_key: str) -> tuple[float, float, list[float]]:
    """Mean, std, and list of per-fold values."""
    values = [r["metrics"][metric_key] for r in rows if "metrics" in r and metric_key in r["metrics"]]
    if not values:
        return 0.0, 0.0, []
    arr = np.array(values, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr)), values


def statistical_comparison(
    baseline_values: list[float],
    other_values: list[float],
    n_perm: int = N_PERM,
    n_bootstrap: int = N_BOOTSTRAP,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Paired permutation test (10k, two-sided), mean difference, p-value, Cohen's d (paired), bootstrap 95% CI (2000)."""
    from bci_framework.utils.table_comparison import (
        _permutation_test_paired,
        _cohens_d_paired,
        _bootstrap_ci_paired,
    )

    if len(baseline_values) != len(other_values) or len(baseline_values) < 2:
        return {"mean_delta": None, "p_value": None, "cohens_d": None, "bootstrap_ci_95": None}
    mean_delta, p_value = _permutation_test_paired(baseline_values, other_values, n_perm=n_perm, random_state=random_state)
    cohens_d = _cohens_d_paired(baseline_values, other_values)
    ci_lo, ci_hi = _bootstrap_ci_paired(baseline_values, other_values, n_boot=n_bootstrap, random_state=random_state)
    return {
        "mean_delta": float(mean_delta),
        "p_value": float(p_value),
        "cohens_d": float(cohens_d) if cohens_d is not None else None,
        "bootstrap_ci_95": [ci_lo, ci_hi],
    }


def run_full_evaluation(
    subjects: list[int] | None = None,
    out_dir: Path | None = None,
    n_perm: int = N_PERM,
    n_bootstrap: int = N_BOOTSTRAP,
    random_state: int = RANDOM_STATE,
    baseline_only: bool = False,
    pipelines: list[str] | None = None,
    preloaded_data: tuple | None = None,
) -> dict[str, Any]:
    """
    Load Physionet MI (or use preloaded_data), run pipelines A/B/C with GroupKFold.
    If baseline_only, run only A. Otherwise run pipelines (default A,B,C).
    Paired comparisons: all pairs (A vs B, A vs C, B vs C).
    Save subject_level_results.csv and preprocessing_evaluation_metadata.json.

    preloaded_data: optional (X, y, subject_ids, fs, ch_names, n_classes, capabilities, class_distribution)
    to avoid loading the dataset twice (memory-safe for full Physionet 109 subjects).
    """
    if preloaded_data is not None:
        X, y, subject_ids, fs, ch_names, n_classes, capabilities, class_distribution = preloaded_data
    else:
        X, y, subject_ids, fs, ch_names, n_classes, capabilities, class_distribution = load_physionet_mi(subjects=subjects)
    subjects_list = list(np.unique(subject_ids))
    n_subjects = len(subjects_list)
    total_trials = len(y)

    logger.info("Loaded Physionet MI: n_subjects=%d, total_trials=%d, n_classes=%d", n_subjects, total_trials, n_classes)

    if baseline_only:
        conditions = ["A"]
    elif pipelines:
        conditions = [c for c in pipelines if c in CONDITION_ENABLED]
        if not conditions:
            conditions = ["A"]
    else:
        conditions = ALL_PIPELINES

    results_gkf = {}
    for cond in conditions:
        results_gkf[cond] = run_group_kfold(
            X, y, subject_ids, cond, fs, n_classes, ch_names, capabilities,
            n_splits=N_GROUP_FOLDS, random_state=random_state,
        )

    # Paired comparisons: all pairs (professor: A < B < C < D style)
    def compare_pair(res_a: list[dict], res_b: list[dict], key: str = "accuracy") -> dict:
        va = [r["metrics"][key] for r in res_a]
        vb = [r["metrics"][key] for r in res_b]
        if len(va) != len(vb) or len(va) < 2:
            return {}
        return statistical_comparison(va, vb, n_perm=n_perm, n_bootstrap=n_bootstrap, random_state=random_state)

    paired_comparisons = {}
    for i, ca in enumerate(conditions):
        for cb in conditions[i + 1:]:
            key = "%s_vs_%s" % (ca, cb)
            paired_comparisons[key] = compare_pair(results_gkf[ca], results_gkf[cb])

    # Backward compat: comparison_ica_vs_baseline
    comparison_ica_vs_baseline = paired_comparisons.get("A_vs_B", {})

    # Summary table: Condition | Mean Accuracy | Std | Mean AUC | Mean Kappa
    summary_table = []
    for cond in conditions:
        rows = results_gkf[cond]
        mean_acc, std_acc, _ = aggregate_fold_metrics(rows, "accuracy")
        mean_auc, std_auc, _ = aggregate_fold_metrics(rows, "roc_auc_macro")
        mean_kappa, std_kappa, _ = aggregate_fold_metrics(rows, "cohen_kappa")
        summary_table.append({
            "Condition": CONDITION_LABELS[cond],
            "Mean_Accuracy": round(mean_acc, 4),
            "Std_Accuracy": round(std_acc, 4),
            "Mean_AUC": round(mean_auc, 4),
            "Mean_Kappa": round(mean_kappa, 4),
        })

    # Stability: std across folds per condition
    stability = {}
    for cond in conditions:
        _, std_acc, _ = aggregate_fold_metrics(results_gkf[cond], "accuracy")
        stability[cond] = {"std_across_folds": std_acc}

    metadata = {
        "dataset_name": PHYSIONET_DATASET,
        "n_subjects": n_subjects,
        "total_trials": total_trials,
        "class_distribution": class_distribution,
        "random_state": random_state,
        "n_splits": N_GROUP_FOLDS,
        "hyperparameter_grid": {"C": C_GRID},
        "permutation_n": n_perm,
        "trial_window": [TRIAL_TMIN, TRIAL_TMAX],
    }

    out = {
        "metadata": metadata,
        "results_gkf": results_gkf,
        "summary_table": summary_table,
        "paired_comparisons": paired_comparisons,
        "comparison_ica_vs_baseline": comparison_ica_vs_baseline,
        "stability": stability,
    }

    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        import csv
        csv_path = out_dir / "subject_level_results.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["fold_id", "condition", "accuracy", "auc_macro", "kappa"])
            for cond in conditions:
                for r in results_gkf[cond]:
                    m = r.get("metrics", {})
                    w.writerow([
                        r.get("fold"),
                        cond,
                        m.get("accuracy"),
                        m.get("roc_auc_macro"),
                        m.get("cohen_kappa"),
                    ])
        logger.info("Wrote %s", csv_path)

        meta_json = {
            **metadata,
            "summary_table": summary_table,
            "paired_comparisons": paired_comparisons,
            "comparison_ica_vs_baseline": comparison_ica_vs_baseline,
            "comparison_results": comparison_ica_vs_baseline,
            "stability": stability,
        }
        json_path = out_dir / "preprocessing_evaluation_metadata.json"
        with open(json_path, "w") as f:
            json.dump(meta_json, f, indent=2)
        logger.info("Wrote %s", json_path)

    return out
