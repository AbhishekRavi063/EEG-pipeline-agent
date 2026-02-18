"""
Full CC-RSA validation: Phase 1 (9-subject LOSO), Phase 2 (cross-dataset),
Phase 3 (robustness), Phase 4 (optional MLP).

Usage:
  PYTHONPATH=. python scripts/run_cc_rsa_full_validation.py --phase 1
  PYTHONPATH=. python scripts/run_cc_rsa_full_validation.py --phase 1,2,3
  PYTHONPATH=. python scripts/run_cc_rsa_full_validation.py  # all phases
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bci_framework.preprocessing import subject_standardize_per_subject
from tests.test_loso_transfer_validation import (
    MI_WINDOW_PRIMARY,
    _check_class_balance,
    _get_memory_gb,
    load_moabb_loso,
    run_one_loso_fold,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = ROOT / "results"
DATASET_A = "BNCI2014_001"
DATASET_B = "BNCI2014_002"
MEAN_DELTA_MIN = 0.02
P_VALUE_MAX = 0.05
CHANCE_LEVEL = 0.25
STD_DELTA_MAX = 0.10
MAX_MEMORY_GB = 3.0
MIN_WITHIN_SUBJECT = 0.85
ROBUST_DELTA_DROP_MAX = 0.03
CC_RSA_MLP_DELTA_MIN = 0.01
FINAL_MODE = "RSA_CC_TRANSFER_V1"


def _get_subject_list(dataset: str, max_subjects: int | None = 9) -> list[int]:
    from bci_framework.datasets.moabb_loader import MOABBDatasetLoader
    loader = MOABBDatasetLoader(dataset_name=dataset, paradigm="motor_imagery", resample=250)
    all_subjects = loader.get_subject_ids()
    out = list(all_subjects)[: max_subjects if max_subjects else len(all_subjects)]
    return out


def get_rsa_stable_config() -> dict:
    from tests.test_loso_transfer_validation import get_base_config
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
    # Fair comparison: use same calibration size as CC-RSA when running full validation
    cfg["agent"] = cfg.get("agent", {}) | {"calibration_trials": 2000}
    return cfg


def get_cc_rsa_config(calibration_trials: int = 2000) -> dict:
    from tests.test_loso_transfer_validation import get_base_config
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
    cfg["agent"] = cfg.get("agent", {}) | {"calibration_trials": calibration_trials}
    return cfg


def get_cc_rsa_mlp_config(calibration_trials: int = 2000) -> dict:
    cfg = get_cc_rsa_config(calibration_trials=calibration_trials)
    cfg["pipelines"]["explicit"] = [["filter_bank_riemann", "rsa_mlp"]]
    cfg["classifiers"]["rsa_mlp"] = {
        "single_hidden": 128,
        "dropout": 0.2,
        "weight_decay": 5e-4,
    }
    return cfg


def run_phase1() -> dict:
    """Full 9-subject LOSO: baseline vs CC-RSA, t-test, stability, save loso_cc_rsa_full.json."""
    from scipy import stats
    from bci_framework.pipelines.registry import PipelineRegistry
    from sklearn.model_selection import train_test_split

    subjects = _get_subject_list(DATASET_A, 9)
    logger.info("Phase 1: Full 9-subject LOSO on %s, subjects %s", DATASET_A, subjects)

    X, y, subject_ids, fs, ch_names, n_cl, capabilities = load_moabb_loso(
        DATASET_A, subjects, tmin=MI_WINDOW_PRIMARY[0], tmax=MI_WINDOW_PRIMARY[1]
    )
    X = subject_standardize_per_subject(X, subject_ids)
    X = np.asarray(X, dtype=np.float32)
    _check_class_balance(y, 4)

    config_baseline = get_rsa_stable_config()
    config_baseline["spatial_capabilities"] = capabilities
    config_cc = get_cc_rsa_config()
    config_cc["spatial_capabilities"] = capabilities

    # Baseline LOSO
    logger.info("=== Baseline (RSA_STABLE) LOSO ===")
    baseline_accs = []
    baseline_times = []
    for holdout in subjects:
        t0 = time.perf_counter()
        out = run_one_loso_fold(
            holdout, subjects, X, y, subject_ids, fs, ch_names, n_cl,
            copy.deepcopy(config_baseline), return_diagnostics=False,
        )
        baseline_accs.append(out[1])
        baseline_times.append(time.perf_counter() - t0)
        gc.collect()
    mean_baseline = float(np.mean(baseline_accs))

    # Within-subject (for report)
    s1 = subjects[0]
    mask_s1 = subject_ids == s1
    X_s1, y_s1 = X[mask_s1], y[mask_s1]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_s1, y_s1, test_size=0.30, stratify=y_s1, random_state=42
    )
    registry = PipelineRegistry(config_cc)
    pipelines = registry.build_pipelines(fs=fs, n_classes=n_cl, channel_names=ch_names)
    sub_ids_tr = np.full(len(X_tr), s1, dtype=np.int64)
    pipelines[0].fit(X_tr, y_tr, subject_ids=sub_ids_tr)
    within_acc = float(np.mean(pipelines[0].predict(X_te) == y_te))
    if within_acc < MIN_WITHIN_SUBJECT:
        logger.error("[ABORT] Within-subject %.2f%% < 85%%", within_acc * 100)
        sys.exit(1)

    # CC-RSA LOSO
    logger.info("=== CC-RSA LOSO ===")
    cc_rsa_accs = []
    cc_rsa_times = []
    memory_peaks = []
    best_epochs = []
    for holdout in subjects:
        t0 = time.perf_counter()
        mem_start = _get_memory_gb()
        out = run_one_loso_fold(
            holdout, subjects, X, y, subject_ids, fs, ch_names, n_cl,
            copy.deepcopy(config_cc), return_diagnostics=True,
        )
        cc_rsa_accs.append(out[1])
        cc_rsa_times.append(time.perf_counter() - t0)
        memory_peaks.append(_get_memory_gb() - mem_start)
        extras = out[3] if len(out) > 3 else {}
        best_epochs.append(extras.get("mlp_best_epoch"))
        del out
        gc.collect()

    mean_cc_rsa = float(np.mean(cc_rsa_accs))
    deltas = [cc_rsa_accs[i] - baseline_accs[i] for i in range(len(subjects))]
    mean_delta = float(np.mean(deltas))
    std_delta = float(np.std(deltas)) if len(deltas) > 1 else 0.0
    memory_peak_max = float(max(memory_peaks)) if memory_peaks else 0.0

    # Paired t-test: H0 mean_delta = 0
    t_stat, p_value = stats.ttest_rel(cc_rsa_accs, baseline_accs)
    t_statistic = float(t_stat)
    p_value = float(p_value)

    # Acceptance
    if mean_delta < MEAN_DELTA_MIN or p_value >= P_VALUE_MAX:
        logger.error(
            "[ABORT] CC-RSA did not meet acceptance: mean_delta=%.4f (need >= %.2f), p=%.4f (need < %.2f). Revert to RSA_STABLE.",
            mean_delta, MEAN_DELTA_MIN, p_value, P_VALUE_MAX,
        )
        sys.exit(1)

    # Stability
    if any(a < CHANCE_LEVEL for a in cc_rsa_accs):
        logger.warning("[STABILITY] At least one subject below 25%% chance")
    if std_delta >= STD_DELTA_MAX:
        logger.warning("[STABILITY] std_delta=%.4f >= 10%%", std_delta)
    if memory_peak_max >= MAX_MEMORY_GB:
        logger.error("[ABORT] Memory peak %.2f GB >= %.1f GB", memory_peak_max, MAX_MEMORY_GB)
        sys.exit(1)

    per_subject = [
        {
            "subject_id": int(s),
            "baseline_accuracy": round(baseline_accs[i], 4),
            "cc_rsa_accuracy": round(cc_rsa_accs[i], 4),
            "delta": round(deltas[i], 4),
            "training_time_sec": round(cc_rsa_times[i], 2),
            "memory_peak_gb": round(memory_peaks[i], 3),
            "best_epoch": best_epochs[i],
        }
        for i, s in enumerate(subjects)
    ]

    report = {
        "configuration": "RSA_CC_TRANSFER",
        "dataset": DATASET_A,
        "n_subjects": len(subjects),
        "mean_baseline": round(mean_baseline, 4),
        "mean_cc_rsa": round(mean_cc_rsa, 4),
        "mean_delta": round(mean_delta, 4),
        "std_delta": round(std_delta, 4),
        "t_statistic": round(t_statistic, 4),
        "p_value": round(p_value, 6),
        "within_subject_mean": round(within_acc, 4),
        "memory_peak_max_gb": round(memory_peak_max, 3),
        "per_subject": per_subject,
        "acceptance_met": True,
        "stability_ok": all(a >= CHANCE_LEVEL for a in cc_rsa_accs) and std_delta < STD_DELTA_MAX,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "loso_cc_rsa_full.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Phase 1 report saved to %s", out_path)
    logger.info("mean_baseline=%.4f mean_cc_rsa=%.4f mean_delta=%.4f std_delta=%.4f p_value=%.4f",
                report["mean_baseline"], report["mean_cc_rsa"], report["mean_delta"],
                report["std_delta"], report["p_value"])
    return report


def run_phase2() -> dict:
    """Cross-dataset: train CC-RSA on BNCI2014_001 (all 9), test on BNCI2014_002 (all subjects)."""
    from bci_framework.pipelines.registry import PipelineRegistry

    subjects_a = _get_subject_list(DATASET_A, 9)
    subjects_b = _get_subject_list(DATASET_B, None)

    logger.info("Phase 2: Train on %s (subjects %s), test on %s (subjects %s)",
                DATASET_A, subjects_a, DATASET_B, subjects_b)

    # Load A
    X_a, y_a, subject_ids_a, fs_a, ch_names_a, n_cl_a, capabilities_a = load_moabb_loso(
        DATASET_A, subjects_a, tmin=MI_WINDOW_PRIMARY[0], tmax=MI_WINDOW_PRIMARY[1]
    )
    X_a = subject_standardize_per_subject(X_a, subject_ids_a)
    X_a = np.asarray(X_a, dtype=np.float32)
    _check_class_balance(y_a, 4)

    config = get_cc_rsa_config()
    config["spatial_capabilities"] = capabilities_a
    registry = PipelineRegistry(config)
    pipelines = registry.build_pipelines(fs=fs_a, n_classes=n_cl_a, channel_names=ch_names_a)
    pipe = pipelines[0]
    pipe.fit(X_a, y_a, subject_ids=subject_ids_a)

    # Load B and evaluate per subject
    accs_b = []
    for sb in subjects_b:
        try:
            X_b, y_b, subject_ids_b, fs_b, ch_names_b, n_cl_b, _ = load_moabb_loso(
                DATASET_B, [sb], tmin=MI_WINDOW_PRIMARY[0], tmax=MI_WINDOW_PRIMARY[1]
            )
        except Exception as e:
            logger.warning("Skip subject %s on B: %s", sb, e)
            continue
        if n_cl_b != n_cl_a or fs_b != fs_a or len(ch_names_b) != len(ch_names_a):
            logger.warning("Skip subject %s: incompatible n_classes/fs/channels", sb)
            continue
        X_b = subject_standardize_per_subject(X_b, subject_ids_b)
        X_b = np.asarray(X_b, dtype=np.float32)
        y_pred = pipe.predict(X_b)
        acc = float(np.mean(y_pred == y_b))
        accs_b.append((int(sb), acc))

    if not accs_b:
        logger.error("[ABORT] No subjects from %s could be evaluated", DATASET_B)
        sys.exit(1)

    per_subject_b = [{"subject_id": s, "accuracy": round(a, 4)} for s, a in accs_b]
    accs_only = [a for _, a in accs_b]
    mean_transfer = float(np.mean(accs_only))
    std_transfer = float(np.std(accs_only)) if len(accs_only) > 1 else 0.0

    interpretation = (
        "poor transfer" if mean_transfer < 0.30 else
        "moderate" if mean_transfer < 0.35 else
        "strong" if mean_transfer < 0.40 else
        "excellent"
    )

    report = {
        "train_dataset": DATASET_A,
        "test_dataset": DATASET_B,
        "mean_transfer_accuracy": round(mean_transfer, 4),
        "std_transfer_accuracy": round(std_transfer, 4),
        "interpretation": interpretation,
        "per_subject": per_subject_b,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "cross_dataset_cc_rsa.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Phase 2 report saved to %s", out_path)
    logger.info("mean_transfer_accuracy=%.4f (%s)", mean_transfer, interpretation)
    return report


def run_phase3(phase1_report: dict | None) -> dict:
    """Robustness: reduced channels (20%% drop), reduced trials (70%%), noise injection."""
    from scipy import stats

    subjects = _get_subject_list(DATASET_A, 9)
    X, y, subject_ids, fs, ch_names, n_cl, capabilities = load_moabb_loso(
        DATASET_A, subjects, tmin=MI_WINDOW_PRIMARY[0], tmax=MI_WINDOW_PRIMARY[1]
    )
    X = subject_standardize_per_subject(X, subject_ids)
    X = np.asarray(X, dtype=np.float32)
    _check_class_balance(y, 4)

    results = {}

    # 1) Reduced channels: drop 20%
    rng = np.random.RandomState(42)
    n_ch = X.shape[1]
    keep = int(np.ceil(0.8 * n_ch))
    sel = rng.choice(n_ch, size=keep, replace=False)
    sel = np.sort(sel)
    X_red = X[:, sel, :]
    ch_names_red = [ch_names[i] for i in sel]
    config_b = get_rsa_stable_config()
    config_b["spatial_capabilities"] = capabilities
    config_cc = get_cc_rsa_config()
    config_cc["spatial_capabilities"] = capabilities

    baseline_red = []
    cc_rsa_red = []
    for holdout in subjects:
        out_b = run_one_loso_fold(
            holdout, subjects, X_red, y, subject_ids, fs, ch_names_red, n_cl,
            copy.deepcopy(config_b), return_diagnostics=False,
        )
        baseline_red.append(out_b[1])
        out_c = run_one_loso_fold(
            holdout, subjects, X_red, y, subject_ids, fs, ch_names_red, n_cl,
            copy.deepcopy(config_cc), return_diagnostics=False,
        )
        cc_rsa_red.append(out_c[1])
        gc.collect()
    mean_delta_red = float(np.mean(cc_rsa_red) - np.mean(baseline_red))
    full_delta = phase1_report["mean_delta"] if phase1_report else 0.0
    delta_drop = full_delta - mean_delta_red
    robust_channels = delta_drop < ROBUST_DELTA_DROP_MAX
    results["reduced_channels_20pct"] = {
        "mean_baseline": round(float(np.mean(baseline_red)), 4),
        "mean_cc_rsa": round(float(np.mean(cc_rsa_red)), 4),
        "mean_delta": round(mean_delta_red, 4),
        "delta_drop_vs_full": round(delta_drop, 4),
        "robust": robust_channels,
    }

    # 2) Reduced trials: 70% training
    min_source = min(
        np.sum(subject_ids != holdout) for holdout in subjects
    )
    cal_trials_70 = int(0.7 * min_source)
    config_cc_70 = get_cc_rsa_config(calibration_trials=cal_trials_70)
    config_cc_70["spatial_capabilities"] = capabilities
    config_b_70 = get_rsa_stable_config()
    config_b_70["spatial_capabilities"] = capabilities
    config_b_70["agent"] = config_b_70.get("agent", {}) | {"calibration_trials": cal_trials_70}

    baseline_70 = []
    cc_rsa_70 = []
    for holdout in subjects:
        out_b = run_one_loso_fold(
            holdout, subjects, X, y, subject_ids, fs, ch_names, n_cl,
            copy.deepcopy(config_b_70), return_diagnostics=False,
        )
        baseline_70.append(out_b[1])
        out_c = run_one_loso_fold(
            holdout, subjects, X, y, subject_ids, fs, ch_names, n_cl,
            copy.deepcopy(config_cc_70), return_diagnostics=False,
        )
        cc_rsa_70.append(out_c[1])
        gc.collect()
    results["reduced_trials_70pct"] = {
        "mean_baseline": round(float(np.mean(baseline_70)), 4),
        "mean_cc_rsa": round(float(np.mean(cc_rsa_70)), 4),
        "mean_delta": round(float(np.mean(cc_rsa_70)) - np.mean(baseline_70), 4),
        "calibration_trials_used": cal_trials_70,
    }

    # 3) Noise injection: one fold
    eps = 1e-4
    X_noisy = X + np.random.RandomState(43).randn(*X.shape).astype(np.float32) * eps
    config_cc_noise = get_cc_rsa_config()
    config_cc_noise["spatial_capabilities"] = capabilities
    out_noise = run_one_loso_fold(
        subjects[0], subjects, X_noisy, y, subject_ids, fs, ch_names, n_cl,
        copy.deepcopy(config_cc_noise), return_diagnostics=False,
    )
    results["noise_injection"] = {
        "epsilon": eps,
        "holdout_subject": subjects[0],
        "cc_rsa_accuracy": round(float(out_noise[1]), 4),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "robustness_cc_rsa.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Phase 3 report saved to %s", out_path)
    return results


def run_phase4(phase1_report: dict | None) -> dict:
    """Optional: CC-RSA + small MLP (128 units) vs CC-RSA logistic. Accept if mean_delta >= +0.01."""
    subjects = _get_subject_list(DATASET_A, 9)
    X, y, subject_ids, fs, ch_names, n_cl, capabilities = load_moabb_loso(
        DATASET_A, subjects, tmin=MI_WINDOW_PRIMARY[0], tmax=MI_WINDOW_PRIMARY[1]
    )
    X = subject_standardize_per_subject(X, subject_ids)
    X = np.asarray(X, dtype=np.float32)
    _check_class_balance(y, 4)

    config_logistic = get_cc_rsa_config()
    config_logistic["spatial_capabilities"] = capabilities
    config_mlp = get_cc_rsa_mlp_config()
    config_mlp["spatial_capabilities"] = capabilities

    cc_logistic_accs = []
    cc_mlp_accs = []
    for holdout in subjects:
        out_l = run_one_loso_fold(
            holdout, subjects, X, y, subject_ids, fs, ch_names, n_cl,
            copy.deepcopy(config_logistic), return_diagnostics=False,
        )
        cc_logistic_accs.append(out_l[1])
        out_m = run_one_loso_fold(
            holdout, subjects, X, y, subject_ids, fs, ch_names, n_cl,
            copy.deepcopy(config_mlp), return_diagnostics=False,
        )
        cc_mlp_accs.append(out_m[1])
        gc.collect()

    mean_logistic = float(np.mean(cc_logistic_accs))
    mean_mlp = float(np.mean(cc_mlp_accs))
    delta_mlp_vs_logistic = mean_mlp - mean_logistic
    accept_mlp = delta_mlp_vs_logistic >= CC_RSA_MLP_DELTA_MIN

    report = {
        "mean_cc_rsa_logistic": round(mean_logistic, 4),
        "mean_cc_rsa_mlp_128": round(mean_mlp, 4),
        "delta_mlp_over_logistic": round(delta_mlp_vs_logistic, 4),
        "accept_mlp": accept_mlp,
        "per_subject_logistic": [round(a, 4) for a in cc_logistic_accs],
        "per_subject_mlp": [round(a, 4) for a in cc_mlp_accs],
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "cc_rsa_mlp_optional.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Phase 4: CC-RSA logistic=%.4f CC-RSA MLP(128)=%.4f delta=%.4f accept_mlp=%s",
                mean_logistic, mean_mlp, delta_mlp_vs_logistic, accept_mlp)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Full CC-RSA validation (Phases 1–4)")
    ap.add_argument("--phase", type=str, default="1,2,3,4",
                    help="Comma-separated phases to run, e.g. 1,2,3,4 or 1")
    args = ap.parse_args()
    phases = [int(p.strip()) for p in args.phase.split(",") if p.strip()]

    phase1_report = None
    phase2_report = None

    if 1 in phases:
        phase1_report = run_phase1()
    if 2 in phases:
        phase2_report = run_phase2()
    if 3 in phases:
        run_phase3(phase1_report)
    if 4 in phases:
        run_phase4(phase1_report)

    # Final stop condition
    if phase1_report and phase2_report:
        if (phase1_report.get("mean_delta", 0) >= MEAN_DELTA_MIN
                and phase1_report.get("p_value", 1) < P_VALUE_MAX
                and phase2_report.get("mean_transfer_accuracy", 0) >= 0.35):
            logger.info("Final stop condition met: lock %s", FINAL_MODE)
            lock_path = RESULTS_DIR / "FINAL_MODE.txt"
            with open(lock_path, "w") as f:
                f.write(FINAL_MODE + "\n")
            logger.info("Wrote %s", lock_path)


if __name__ == "__main__":
    main()
