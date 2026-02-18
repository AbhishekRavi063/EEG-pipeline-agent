"""
Run LOSO validation with modular patches enabled (BNCI2014_001).

Execution order (enable sequentially, rollback if step drops LOSO >3%):
  1. Procrustes  2. Band weighting  3. Subject weighting  4. Temporal  5. Class-conditional
  Plus: C grid [0.01, 0.1, 1, 10], outlier detection, Platt scaling (always when patches on).

RSA_STABLE: minimal optimal RSA-only configuration (no patches). Use --rsa-stable.

Usage:
  python scripts/run_loso_with_patches.py                    # all patches
  python scripts/run_loso_with_patches.py --no-patches       # baseline (RSA only)
  python scripts/run_loso_with_patches.py --rsa-stable       # locked RSA-only, report, save loso_rsa_stable.json
  python scripts/run_loso_with_patches.py --classifier rsa_mlp  # RSA+MLP hybrid; revert if LOSO < baseline - 3%%
  python scripts/run_loso_with_patches.py --subjects 1 2 3   # quick (default)
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bci_framework.features.filter_bank_riemann import MODE_RSA_STABLE
from bci_framework.preprocessing import subject_standardize_per_subject
from tests.test_loso_transfer_validation import (
    get_base_config,
    load_moabb_loso,
    run_one_loso_fold,
    MI_WINDOW_PRIMARY,
    _get_memory_gb,
    _check_class_balance,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Locked minimal optimal RSA configuration (RSA_STABLE)
MODE = MODE_RSA_STABLE

EXPECTED_FEATURE_DIM = 1265
MAX_MEMORY_GB = 2.0
MIN_WITHIN_SUBJECT = 0.75
MIN_LOSO_3SUBJECTS = 0.40
# RSA+MLP hybrid
MAX_MEMORY_GB_MLP = 3.0
MIN_WITHIN_HYBRID = 0.85
BASELINE_DROP_TOLERANCE = 0.03
BASELINE_IMPROVEMENT_MIN = 0.02  # Accept hybrid if loso_mean >= baseline_loso_mean + 2%


def get_rsa_stable_config() -> dict:
    """Minimal RSA-only config: Filter Bank Riemann, correct covariance, RSA whitening, global ref, tangent, LR with C grid. No patches."""
    cfg = get_base_config(transfer_enabled=False, transfer_method="none", safe_mode=False)
    cfg["pipelines"]["explicit"] = [["filter_bank_riemann", "logistic_regression"]]
    cfg["features"]["filter_bank_riemann"] = {
        "z_score_tangent": True,
        "force_float32": True,
        "rsa": True,
        "use_procrustes": False,
        "use_class_conditional": False,
        "use_temporal": False,
        "use_band_weighting": False,
        "use_subject_weighting": False,
        "use_outlier_detection": False,
        "rsa_stable_mode": True,
    }
    cfg["classifiers"]["logistic_regression"] = {
        "tune_C": True,
        "C": 1.0,
        "C_grid": [0.01, 0.1, 1.0, 10.0],
        "cv_folds": 3,
        "platt_scaling": False,
    }
    return cfg


def get_rsa_mlp_config() -> dict:
    """RSA (same as stable) + rsa_mlp classifier. Same feature pipeline; classifier = rsa_mlp."""
    cfg = get_rsa_stable_config()
    cfg["pipelines"]["explicit"] = [["filter_bank_riemann", "rsa_mlp"]]
    cfg["classifiers"]["rsa_mlp"] = {}
    return cfg


def get_cc_rsa_config() -> dict:
    """RSA + Class-Conditional alignment (source labels only; target unsupervised). Same classifier as stable."""
    cfg = get_base_config(transfer_enabled=False, transfer_method="none", safe_mode=False)
    cfg["pipelines"]["explicit"] = [["filter_bank_riemann", "logistic_regression"]]
    cfg["features"]["filter_bank_riemann"] = {
        "z_score_tangent": True,
        "force_float32": True,
        "rsa": True,
        "use_procrustes": False,
        "use_class_conditional": False,
        "use_temporal": False,
        "use_band_weighting": False,
        "use_subject_weighting": False,
        "use_outlier_detection": False,
        "rsa_stable_mode": False,
        "use_class_conditional_rsa": True,
    }
    cfg["classifiers"]["logistic_regression"] = {
        "tune_C": True,
        "C": 1.0,
        "C_grid": [0.01, 0.1, 1.0, 10.0],
        "cv_folds": 3,
        "platt_scaling": False,
    }
    # Use enough source trials so 2+ subjects are present (CC-RSA needs ≥2 source subjects)
    cfg["agent"] = cfg.get("agent", {}) | {"calibration_trials": 2000}
    return cfg


def assert_rsa_stable_pipeline(pipe) -> None:
    """Abort if any disabled module is accidentally activated."""
    fe = getattr(pipe, "feature_extractor", None)
    if fe is None:
        return
    assert getattr(fe, "rsa", False), "RSA_STABLE: use_rsa must be True"
    assert not getattr(fe, "use_procrustes", True), "RSA_STABLE: use_procrustes must be False"
    assert not getattr(fe, "use_class_conditional", True), "RSA_STABLE: use_class_conditional must be False"
    assert not getattr(fe, "use_temporal", True), "RSA_STABLE: use_temporal must be False"
    assert not getattr(fe, "use_band_weighting", True), "RSA_STABLE: use_band_weighting must be False"
    assert not getattr(fe, "use_subject_weighting", True), "RSA_STABLE: use_subject_weighting must be False"
    assert not getattr(fe, "use_outlier_detection", True), "RSA_STABLE: use_outlier_detection must be False"
    clf = getattr(pipe, "classifier", None)
    if clf is not None:
        assert not getattr(clf, "platt_scaling", True), "RSA_STABLE: platt_scaling must be False"


def get_patched_config(use_patches: bool) -> dict:
    cfg = get_base_config(transfer_enabled=False, transfer_method="none", safe_mode=False)
    cfg["pipelines"]["explicit"] = [["filter_bank_riemann", "logistic_regression"]]
    cfg["features"]["filter_bank_riemann"] = {
        "z_score_tangent": True,
        "force_float32": True,
        "rsa": True,
    }
    cfg["classifiers"]["logistic_regression"] = {
        "tune_C": True,
        "C": 1.0,
        "cv_folds": 3,
    }
    if use_patches:
        cfg["features"]["filter_bank_riemann"] = {
            "z_score_tangent": True,
            "force_float32": True,
            "rsa": True,
            "use_procrustes": True,
            "use_class_conditional": True,
            "use_temporal": True,
            "use_band_weighting": True,
            "use_subject_weighting": True,
            "use_outlier_detection": True,
        }
        cfg["classifiers"]["logistic_regression"] = {
            "tune_C": True,
            "C": 1.0,
            "C_grid": [0.01, 0.1, 1.0, 10.0],
            "cv_folds": 3,
            "platt_scaling": True,
        }
    else:
        cfg["classifiers"]["logistic_regression"]["C_grid"] = [0.1, 1.0, 10.0]
    return cfg


def run_rsa_stable(subjects: list[int], dataset: str, run_full_nine: bool) -> None:
    """RSA_STABLE: minimal RSA-only config, stability assertions, report, save loso_rsa_stable.json."""
    config = get_rsa_stable_config()
    config["spatial_capabilities"] = None  # set after load

    X, y, subject_ids, fs, ch_names, n_cl, capabilities = load_moabb_loso(
        dataset, subjects, tmin=MI_WINDOW_PRIMARY[0], tmax=MI_WINDOW_PRIMARY[1]
    )
    X = subject_standardize_per_subject(X, subject_ids)
    X = np.asarray(X, dtype=np.float32)
    _check_class_balance(y, 4)
    config["spatial_capabilities"] = capabilities

    from sklearn.model_selection import train_test_split
    from bci_framework.pipelines.registry import PipelineRegistry

    registry = PipelineRegistry(config)
    pipelines = registry.build_pipelines(fs=fs, n_classes=n_cl, channel_names=ch_names)
    assert pipelines, "RSA_STABLE: no pipelines built"
    assert_rsa_stable_pipeline(pipelines[0])

    # Within-subject (subject 1) — require >= 75%
    s1 = subjects[0]
    mask_s1 = subject_ids == s1
    X_s1, y_s1 = X[mask_s1], y[mask_s1]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_s1, y_s1, test_size=0.30, stratify=y_s1, random_state=42
    )
    pipe = pipelines[0]
    sub_ids_tr = np.full(len(X_tr), s1, dtype=np.int64)
    pipe.fit(X_tr, y_tr, subject_ids=sub_ids_tr)
    within_acc = float(np.mean(pipe.predict(X_te) == y_te))
    logger.info("[WITHIN-SUBJECT] accuracy=%.4f (require >= 75%%)", within_acc)
    if within_acc < MIN_WITHIN_SUBJECT:
        logger.error("[ABORT] Within-subject %.2f%% < 75%%", within_acc * 100)
        sys.exit(1)

    # LOSO
    fold_accs = []
    fold_memory = []
    mem_peak = 0.0
    all_extras = []
    selected_C_per_fold = []
    for holdout in subjects:
        mem_start = _get_memory_gb()
        out = run_one_loso_fold(
            holdout, subjects, X, y, subject_ids, fs, ch_names, n_cl,
            copy.deepcopy(config), fold_memory_usage=fold_memory, return_diagnostics=True,
        )
        cv_acc, test_acc, _debug, extras = out
        fold_accs.append(test_acc)
        selected_C_per_fold.append(extras.get("selected_C"))
        mem_peak = max(mem_peak, _get_memory_gb() - mem_start)
        all_extras.append(extras)
        del out
        gc.collect()

    loso_mean = float(np.mean(fold_accs))
    e = all_extras[0] if all_extras else {}
    feature_dim = e.get("feature_dim")
    dist_before = e.get("rsa_distance_before")
    dist_after = e.get("rsa_distance_after")
    # Use first fold with valid diagnostics (full-source); fallback to any fold
    for ex in all_extras:
        db, da = ex.get("rsa_distance_before"), ex.get("rsa_distance_after")
        if db is not None and da is not None and db > 0:
            dist_before, dist_after = db, da
            break

    # distance_reduction_percent = (before - after) / before * 100
    distance_reduction_percent = None
    if dist_before is not None and dist_after is not None and dist_before > 0:
        distance_reduction_percent = float((dist_before - dist_after) / dist_before * 100.0)
        if distance_reduction_percent < 20.0:
            logger.warning(
                "[RSA] distance_reduction_percent=%.1f%% < 20%%. Something may be wrong.",
                distance_reduction_percent,
            )

    # Stability assertions
    assert feature_dim == EXPECTED_FEATURE_DIM, (
        f"RSA_STABLE: feature_dim={feature_dim} != {EXPECTED_FEATURE_DIM}"
    )
    if dist_before is not None and dist_after is not None and dist_before > 0:
        assert dist_after < dist_before, (
            f"RSA_STABLE: distance_after_whitening ({dist_after}) >= distance_before ({dist_before})"
        )
    assert mem_peak < MAX_MEMORY_GB, (
        f"RSA_STABLE: memory peak {mem_peak:.2f} GB >= {MAX_MEMORY_GB} GB"
    )
    if len(subjects) <= 3 and loso_mean < MIN_LOSO_3SUBJECTS:
        logger.error("[ABORT] LOSO (3 subjects) %.2f%% < 40%%", loso_mean * 100)
        sys.exit(1)

    # Benchmark report (formatted diagnostics)
    report = {
        "configuration": "RSA-only",
        "mode": MODE,
        "within_subject_accuracy": within_acc,
        "loso_mean": loso_mean,
        "per_subject": {int(s): float(a) for s, a in zip(subjects, fold_accs)},
        "selected_C_per_fold": selected_C_per_fold,
        "feature_dimension": feature_dim,
        "memory_peak_gb": round(mem_peak, 3),
        "distance_before_whitening": round(float(dist_before), 4) if dist_before is not None else None,
        "distance_after_whitening": round(float(dist_after), 4) if dist_after is not None else None,
        "distance_reduction_percent": round(float(distance_reduction_percent), 2) if distance_reduction_percent is not None else None,
    }
    logger.info("")
    logger.info("=== RSA STABLE BENCHMARK REPORT ===")
    logger.info("Configuration: %s", report["configuration"])
    logger.info("Within-subject accuracy: %.4f", report["within_subject_accuracy"])
    logger.info("LOSO mean: %.4f", report["loso_mean"])
    logger.info("Per-subject: %s", report["per_subject"])
    logger.info("Selected C per fold: %s", report["selected_C_per_fold"])
    logger.info("Feature dimension: %s", report["feature_dimension"])
    logger.info("Memory peak: %.3f GB", report["memory_peak_gb"])
    if report["distance_before_whitening"] is not None:
        logger.info("distance_before_whitening: %.4f", report["distance_before_whitening"])
    else:
        logger.info("distance_before_whitening: N/A")
    if report["distance_after_whitening"] is not None:
        logger.info("distance_after_whitening: %.4f", report["distance_after_whitening"])
    else:
        logger.info("distance_after_whitening: N/A")
    logger.info("distance_reduction_percent: %s", report["distance_reduction_percent"] if report["distance_reduction_percent"] is not None else "N/A")

    out_path = ROOT / "results" / "loso_rsa_stable.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Report saved to %s", out_path)

    if run_full_nine:
        logger.info("Running full 9-subject LOSO...")
        all_nine = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        X9, y9, sid9, fs9, ch9, n_cl9, cap9 = load_moabb_loso(
            dataset, all_nine, tmin=MI_WINDOW_PRIMARY[0], tmax=MI_WINDOW_PRIMARY[1]
        )
        X9 = subject_standardize_per_subject(X9, sid9)
        X9 = np.asarray(X9, dtype=np.float32)
        _check_class_balance(y9, 4)
        cfg9 = copy.deepcopy(config)
        cfg9["spatial_capabilities"] = cap9
        fold_accs9 = []
        selected_C_9 = []
        for holdout in all_nine:
            out9 = run_one_loso_fold(
                holdout, all_nine, X9, y9, sid9, fs9, ch9, n_cl9, copy.deepcopy(cfg9),
                return_diagnostics=True,
            )
            _, test_acc, _, extras9 = out9
            fold_accs9.append(test_acc)
            selected_C_9.append(extras9.get("selected_C"))
            gc.collect()
        loso_9_mean = float(np.mean(fold_accs9))
        full_report = {
            "configuration": "RSA-only",
            "mode": MODE,
            "loso_mean_9subjects": loso_9_mean,
            "per_subject": {int(s): float(a) for s, a in zip(all_nine, fold_accs9)},
            "selected_C_per_fold": selected_C_9,
            "feature_dimension": EXPECTED_FEATURE_DIM,
        }
        with open(out_path, "w") as f:
            json.dump(full_report, f, indent=2)
        logger.info("Full 9-subject LOSO mean: %.4f", loso_9_mean)
        logger.info("Updated %s", out_path)


def run_rsa_mlp(subjects: list[int], dataset: str) -> None:
    """RSA + MLP hybrid: baseline LOSO first, then MLP; revert if LOSO < baseline - 3%."""
    from sklearn.model_selection import train_test_split
    from bci_framework.pipelines.registry import PipelineRegistry

    # Step 1: Baseline (RSA + logistic) LOSO
    logger.info("=== Baseline (RSA + logistic) LOSO ===")
    config_baseline = get_rsa_stable_config()
    X, y, subject_ids, fs, ch_names, n_cl, capabilities = load_moabb_loso(
        dataset, subjects, tmin=MI_WINDOW_PRIMARY[0], tmax=MI_WINDOW_PRIMARY[1]
    )
    X = subject_standardize_per_subject(X, subject_ids)
    X = np.asarray(X, dtype=np.float32)
    _check_class_balance(y, 4)
    config_baseline["spatial_capabilities"] = capabilities

    baseline_fold_accs = []
    for holdout in subjects:
        out = run_one_loso_fold(
            holdout, subjects, X, y, subject_ids, fs, ch_names, n_cl,
            copy.deepcopy(config_baseline), return_diagnostics=False,
        )
        baseline_fold_accs.append(out[1])
        gc.collect()
    baseline_loso_mean = float(np.mean(baseline_fold_accs))
    logger.info("Baseline LOSO mean: %.4f", baseline_loso_mean)

    # Step 2: RSA + MLP
    logger.info("=== RSA + MLP hybrid LOSO ===")
    config = get_rsa_mlp_config()
    config["spatial_capabilities"] = capabilities
    registry = PipelineRegistry(config)
    pipelines = registry.build_pipelines(fs=fs, n_classes=n_cl, channel_names=ch_names)
    assert pipelines, "RSA+MLP: no pipelines built"
    assert pipelines[0].classifier_name == "rsa_mlp"

    s1 = subjects[0]
    mask_s1 = subject_ids == s1
    X_s1, y_s1 = X[mask_s1], y[mask_s1]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_s1, y_s1, test_size=0.30, stratify=y_s1, random_state=42
    )
    sub_ids_tr = np.full(len(X_tr), s1, dtype=np.int64)
    pipe = pipelines[0]
    pipe.fit(X_tr, y_tr, subject_ids=sub_ids_tr)
    within_acc = float(np.mean(pipe.predict(X_te) == y_te))
    logger.info("[WITHIN-SUBJECT] accuracy=%.4f (require >= 85%% for accept)", within_acc)

    fold_accs = []
    best_epochs_per_fold: list[int | None] = []
    mem_peak = 0.0
    all_extras = []
    for holdout in subjects:
        mem_start = _get_memory_gb()
        out = run_one_loso_fold(
            holdout, subjects, X, y, subject_ids, fs, ch_names, n_cl,
            copy.deepcopy(config), fold_memory_usage=[], return_diagnostics=True,
        )
        _, test_acc, _, extras = out
        fold_accs.append(test_acc)
        best_epochs_per_fold.append(extras.get("mlp_best_epoch"))
        mem_peak = max(mem_peak, _get_memory_gb() - mem_start)
        all_extras.append(extras)
        be = extras.get("mlp_best_epoch")
        pc = extras.get("mlp_param_count")
        tt = extras.get("mlp_train_time_sec")
        if be is not None or pc is not None:
            logger.info("[FOLD holdout=%s] best_epoch=%s params=%s train_time_sec=%s",
                        holdout, be, pc, tt)
        del out
        gc.collect()

    loso_mean = float(np.mean(fold_accs))
    e = all_extras[0] if all_extras else {}
    feature_dim = e.get("feature_dim")
    param_count = e.get("mlp_param_count")

    if mem_peak >= MAX_MEMORY_GB_MLP:
        logger.error("[ABORT] Memory peak %.2f GB >= %s GB", mem_peak, MAX_MEMORY_GB_MLP)
        sys.exit(1)

    delta = loso_mean - baseline_loso_mean
    if loso_mean < baseline_loso_mean - BASELINE_DROP_TOLERANCE:
        logger.error(
            "HYBRID UNDERPERFORMED — REVERT TO RSA_STABLE (LOSO %.2f%% < baseline %.2f%% - 3%%)",
            loso_mean * 100, baseline_loso_mean * 100,
        )
        sys.exit(1)

    if loso_mean < baseline_loso_mean + BASELINE_IMPROVEMENT_MIN:
        logger.error(
            "[ABORT] LOSO %.2f%% < baseline + 2%% (%.2f%%)",
            loso_mean * 100, (baseline_loso_mean + BASELINE_IMPROVEMENT_MIN) * 100,
        )
        sys.exit(1)
    if within_acc < MIN_WITHIN_HYBRID:
        logger.error("[ABORT] Within-subject %.2f%% < 85%%", within_acc * 100)
        sys.exit(1)

    report = {
        "configuration": "RSA + MLP (relaxed)",
        "within_subject_accuracy": within_acc,
        "loso_mean": loso_mean,
        "per_subject": {int(s): float(a) for s, a in zip(subjects, fold_accs)},
        "baseline_loso_mean": baseline_loso_mean,
        "delta": round(delta, 4),
        "feature_dimension": feature_dim,
        "param_count": param_count,
        "best_epochs_per_fold": best_epochs_per_fold,
        "memory_peak_gb": round(mem_peak, 3),
    }
    logger.info("")
    logger.info("Configuration: RSA + MLP (relaxed)")
    logger.info("Baseline LOSO: %.4f", report["baseline_loso_mean"])
    logger.info("Hybrid LOSO: %.4f", report["loso_mean"])
    logger.info("Delta: %.4f", report["delta"])
    logger.info("Within-subject: %.4f", report["within_subject_accuracy"])
    logger.info("Params: %s", report["param_count"])
    logger.info("Best epochs per fold: %s", report["best_epochs_per_fold"])
    logger.info("Memory peak: %.3f GB", report["memory_peak_gb"])

    out_path = ROOT / "results" / "loso_rsa_mlp_relaxed.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Report saved to %s", out_path)


MODE_CC_TRANSFER = "RSA_CC_TRANSFER"


def run_cc_rsa(subjects: list[int], dataset: str) -> None:
    """CC-RSA: Class-Conditional RSA alignment. Baseline LOSO first; accept if LOSO >= baseline + 2%."""
    from sklearn.model_selection import train_test_split
    from bci_framework.pipelines.registry import PipelineRegistry

    logger.info("=== Baseline (RSA_STABLE) LOSO ===")
    config_baseline = get_rsa_stable_config()
    X, y, subject_ids, fs, ch_names, n_cl, capabilities = load_moabb_loso(
        dataset, subjects, tmin=MI_WINDOW_PRIMARY[0], tmax=MI_WINDOW_PRIMARY[1]
    )
    X = subject_standardize_per_subject(X, subject_ids)
    X = np.asarray(X, dtype=np.float32)
    _check_class_balance(y, 4)
    config_baseline["spatial_capabilities"] = capabilities

    baseline_fold_accs = []
    for holdout in subjects:
        out = run_one_loso_fold(
            holdout, subjects, X, y, subject_ids, fs, ch_names, n_cl,
            copy.deepcopy(config_baseline), return_diagnostics=False,
        )
        baseline_fold_accs.append(out[1])
        gc.collect()
    baseline_loso_mean = float(np.mean(baseline_fold_accs))
    logger.info("Baseline LOSO mean: %.4f", baseline_loso_mean)

    logger.info("=== CC-RSA LOSO ===")
    config = get_cc_rsa_config()
    config["spatial_capabilities"] = capabilities
    registry = PipelineRegistry(config)
    pipelines = registry.build_pipelines(fs=fs, n_classes=n_cl, channel_names=ch_names)
    assert pipelines, "CC-RSA: no pipelines built"

    s1 = subjects[0]
    mask_s1 = subject_ids == s1
    X_s1, y_s1 = X[mask_s1], y[mask_s1]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_s1, y_s1, test_size=0.30, stratify=y_s1, random_state=42
    )
    sub_ids_tr = np.full(len(X_tr), s1, dtype=np.int64)
    pipe = pipelines[0]
    pipe.fit(X_tr, y_tr, subject_ids=sub_ids_tr)
    within_acc = float(np.mean(pipe.predict(X_te) == y_te))
    logger.info("[WITHIN-SUBJECT] accuracy=%.4f (require >= 85%%)", within_acc)

    fold_accs = []
    mem_peak = 0.0
    all_extras = []
    for holdout in subjects:
        mem_start = _get_memory_gb()
        out = run_one_loso_fold(
            holdout, subjects, X, y, subject_ids, fs, ch_names, n_cl,
            copy.deepcopy(config), fold_memory_usage=[], return_diagnostics=True,
        )
        _, test_acc, _, extras = out
        fold_accs.append(test_acc)
        mem_peak = max(mem_peak, _get_memory_gb() - mem_start)
        all_extras.append(extras)
        del out
        gc.collect()

    loso_mean = float(np.mean(fold_accs))
    e = all_extras[0] if all_extras else {}
    cb = e.get("cc_class_spread_before")
    ca = e.get("cc_class_spread_after")
    class_spread_reduction = None
    if cb is not None and ca is not None and cb > 0:
        class_spread_reduction = float((cb - ca) / cb * 100.0)

    if mem_peak >= MAX_MEMORY_GB_MLP:
        logger.error("[ABORT] Memory peak %.2f GB >= 3 GB", mem_peak)
        sys.exit(1)

    delta = loso_mean - baseline_loso_mean
    if loso_mean < baseline_loso_mean + BASELINE_IMPROVEMENT_MIN:
        logger.error(
            "CC-RSA did not improve transfer. Reverting to RSA_STABLE. (LOSO %.2f%% < baseline + 2%% = %.2f%%)",
            loso_mean * 100, (baseline_loso_mean + BASELINE_IMPROVEMENT_MIN) * 100,
        )
        sys.exit(1)
    if within_acc < MIN_WITHIN_HYBRID:
        logger.error("[ABORT] Within-subject %.2f%% < 85%%", within_acc * 100)
        sys.exit(1)

    report = {
        "configuration": "RSA + Class-Conditional Alignment",
        "mode": MODE_CC_TRANSFER,
        "baseline_loso_mean": baseline_loso_mean,
        "cc_rsa_loso_mean": loso_mean,
        "delta": round(delta, 4),
        "within_subject_accuracy": within_acc,
        "class_spread_reduction_percent": round(class_spread_reduction, 2) if class_spread_reduction is not None else None,
        "memory_peak_gb": round(mem_peak, 3),
        "per_subject": {int(s): float(a) for s, a in zip(subjects, fold_accs)},
    }
    logger.info("")
    logger.info("Configuration: RSA + Class-Conditional Alignment")
    logger.info("Baseline LOSO: %.4f", report["baseline_loso_mean"])
    logger.info("CC-RSA LOSO: %.4f", report["cc_rsa_loso_mean"])
    logger.info("Delta: %.4f", report["delta"])
    logger.info("Within-subject: %.4f", report["within_subject_accuracy"])
    logger.info("Class spread reduction: %s%%", report["class_spread_reduction_percent"] if report["class_spread_reduction_percent"] is not None else "N/A")
    logger.info("Memory peak: %.3f GB", report["memory_peak_gb"])

    out_path = ROOT / "results" / "loso_cc_rsa.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Report saved to %s", out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-patches", action="store_true", help="Run baseline (RSA only)")
    ap.add_argument("--rsa-stable", action="store_true", help="Locked RSA-only config; report; save loso_rsa_stable.json")
    ap.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3], help="Subject IDs")
    ap.add_argument("--dataset", type=str, default="BNCI2014_001")
    ap.add_argument("--no-full-nine", action="store_true", help="With --rsa-stable: skip full 9-subject LOSO (only run given --subjects)")
    ap.add_argument("--classifier", type=str, default="logistic", choices=["logistic", "rsa_mlp"], help="Classifier: logistic (default) or rsa_mlp (RSA+MLP hybrid)")
    ap.add_argument("--class-conditional", action="store_true", help="Run CC-RSA (Class-Conditional RSA); accept if LOSO >= baseline + 2%%")
    args = ap.parse_args()
    use_patches = not args.no_patches and not args.rsa_stable

    if getattr(args, "class_conditional", False):
        run_cc_rsa(subjects=args.subjects, dataset=args.dataset)
        return

    if args.classifier == "rsa_mlp":
        run_rsa_mlp(subjects=args.subjects, dataset=args.dataset)
        return

    if args.rsa_stable:
        run_rsa_stable(
            subjects=args.subjects,
            dataset=args.dataset,
            run_full_nine=not args.no_full_nine,
        )
        return

    config = get_patched_config(use_patches)
    dataset = args.dataset
    subjects = args.subjects

    X, y, subject_ids, fs, ch_names, n_cl, capabilities = load_moabb_loso(
        dataset, subjects, tmin=MI_WINDOW_PRIMARY[0], tmax=MI_WINDOW_PRIMARY[1]
    )
    X = subject_standardize_per_subject(X, subject_ids)
    X = np.asarray(X, dtype=np.float32)
    _check_class_balance(y, 4)
    config["spatial_capabilities"] = capabilities

    # Within-subject sanity (subject 1)
    from sklearn.model_selection import train_test_split
    s1 = subjects[0]
    mask_s1 = subject_ids == s1
    X_s1, y_s1 = X[mask_s1], y[mask_s1]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_s1, y_s1, test_size=0.30, stratify=y_s1, random_state=42
    )
    from bci_framework.pipelines.registry import PipelineRegistry
    registry = PipelineRegistry(config)
    pipelines = registry.build_pipelines(fs=fs, n_classes=n_cl, channel_names=ch_names)
    pipe = pipelines[0]
    pipe.fit(X_tr, y_tr)
    within_acc = float(np.mean(pipe.predict(X_te) == y_te))
    logger.info("[WITHIN-SUBJECT] accuracy=%.4f (require >= 75%%)", within_acc)
    if within_acc < 0.75:
        logger.error("[ABORT] Within-subject %.2f%% < 75%%", within_acc * 100)
        sys.exit(1)

    # LOSO
    fold_accs = []
    fold_memory = []
    mem_peak = 0.0
    all_extras = []
    for holdout in subjects:
        mem_start = _get_memory_gb()
        out = run_one_loso_fold(
            holdout, subjects, X, y, subject_ids, fs, ch_names, n_cl,
            copy.deepcopy(config), fold_memory_usage=fold_memory, return_diagnostics=True,
        )
        cv_acc, test_acc, _debug, extras = out
        fold_accs.append(test_acc)
        mem_peak = max(mem_peak, _get_memory_gb() - mem_start)
        all_extras.append(extras)
        del out
        gc.collect()

    loso_mean = float(np.mean(fold_accs))
    feature_dim = all_extras[0].get("feature_dim") if all_extras else None
    if loso_mean < 0.38:
        logger.error("[ABORT] LOSO mean %.2f%% < 38%%", loso_mean * 100)
        sys.exit(1)
    if mem_peak >= 4.0:
        logger.error("[ABORT] Memory peak %.2f GB >= 4 GB", mem_peak)
        sys.exit(1)

    # Final report
    e = all_extras[0] if all_extras else {}
    report = {
        "within_subject": within_acc,
        "loso_mean": loso_mean,
        "per_subject": {int(s): float(a) for s, a in zip(subjects, fold_accs)},
        "distance_before": e.get("rsa_distance_before"),
        "distance_after_whitening": e.get("rsa_distance_after"),
        "distance_after_procrustes": e.get("rsa_distance_after_procrustes"),
        "band_weights": e.get("band_weights"),
        "subject_weights": e.get("subject_weights"),
        "feature_dimension": feature_dim,
        "memory_peak_gb": round(mem_peak, 3),
    }
    logger.info("")
    logger.info("=== FINAL REPORT ===")
    logger.info("Within-subject: %.4f", report["within_subject"])
    logger.info("LOSO mean: %.4f", report["loso_mean"])
    logger.info("Per-subject: %s", report["per_subject"])
    logger.info("Distance reductions: before=%.4f after_whitening=%.4f after_procrustes=%s",
                report["distance_before"] or 0, report["distance_after_whitening"] or 0,
                report["distance_after_procrustes"])
    logger.info("Band weights: %s", report["band_weights"])
    logger.info("Subject weights: %s", report["subject_weights"])
    logger.info("Feature dimension: %s", report["feature_dimension"])
    logger.info("Memory peak: %.3f GB", report["memory_peak_gb"])

    out_path = ROOT / "results" / "loso_patches_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({k: v for k, v in report.items() if v is not None}, f, indent=2)
    logger.info("Report saved to %s", out_path)


if __name__ == "__main__":
    main()
