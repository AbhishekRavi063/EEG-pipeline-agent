#!/usr/bin/env python3
"""
Run calibration + evaluation WITH and WITHOUT transfer learning, using the same data
and the new adaptive pruning (quick screening, dynamic top-K, balanced accuracy, etc.).

Output: comparison table with best pipeline, CV accuracy, test accuracy, runtime, pruning stats.
"""

import copy
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bci_framework.utils.config_loader import load_config, get_config
from bci_framework.datasets import get_dataset_loader
from bci_framework.pipelines import PipelineRegistry
from bci_framework.agent import PipelineSelectionAgent


def load_bci2a_subjects_1_2():
    """Load BCI IV 2a subjects 1 and 2; concatenate and build subject_ids_per_trial."""
    load_config(ROOT / "bci_framework" / "config.yaml")
    config = get_config()
    ds_cfg = config.get("dataset", {})
    data_path = ROOT / (ds_cfg.get("data_dir", "./data/BCI_IV_2a").lstrip("./"))
    loader = get_dataset_loader("BCI_IV_2a")()
    result = loader.load(
        data_dir=str(data_path),
        subjects=[1, 2],
        download_if_missing=ds_cfg.get("download_if_missing", False),
        trial_duration_seconds=ds_cfg.get("trial_duration_seconds", 3.0),
    )
    if isinstance(result, dict) and len(result) >= 2:
        ds1, ds2 = result[1], result[2]
        X1 = np.asarray(ds1.data, dtype=np.float64)
        y1 = np.asarray(ds1.labels, dtype=np.int64).ravel()
        X2 = np.asarray(ds2.data, dtype=np.float64)
        y2 = np.asarray(ds2.labels, dtype=np.int64).ravel()
        X = np.concatenate([X1, X2], axis=0)
        y = np.concatenate([y1, y2], axis=0)
        subject_ids = np.array([1] * len(X1) + [2] * len(X2), dtype=np.int64)
        fs = ds1.fs
        n_classes = len(ds1.class_names)
        channel_names = getattr(ds1, "channel_names", None)
        dataset = ds1  # for capabilities
    elif isinstance(result, dict) and len(result) == 1:
        dataset = next(iter(result.values()))
        X = np.asarray(dataset.data, dtype=np.float64)
        y = np.asarray(dataset.labels, dtype=np.int64).ravel()
        subject_ids = np.ones(len(y), dtype=np.int64)
        fs = dataset.fs
        n_classes = len(dataset.class_names)
        channel_names = getattr(dataset, "channel_names", None)
    else:
        dataset = result if hasattr(result, "data") else None
        X = np.asarray(dataset.data, dtype=np.float64)
        y = np.asarray(dataset.labels, dtype=np.int64).ravel()
        subject_ids = np.ones(len(y), dtype=np.int64)
        fs = dataset.fs
        n_classes = len(dataset.class_names)
        channel_names = getattr(dataset, "channel_names", None)
    return X, y, subject_ids, fs, n_classes, channel_names, config, dataset


def run_one(transfer_enabled: bool, X, y, subject_ids, fs, n_classes, channel_names, config, dataset):
    """Run calibration + select best; with or without transfer (subject 1 = source, subject 2 = target)."""
    cfg = copy.deepcopy(config)
    cfg["transfer"] = dict(cfg.get("transfer", {}))
    cfg["transfer"]["enabled"] = transfer_enabled
    if transfer_enabled:
        cfg["transfer"]["method"] = cfg["transfer"].get("method") or "coral"
        cfg["transfer"].setdefault("safe_mode", True)
    cfg["spatial_capabilities"] = getattr(dataset, "capabilities", None)
    # Disable GEDAI so pipelines run without leadfield (optional: set to ["signal_quality", "gedai", "ica", "wavelet"] if leadfield available)
    if cfg.get("advanced_preprocessing"):
        cfg["advanced_preprocessing"] = dict(cfg["advanced_preprocessing"])
        cfg["advanced_preprocessing"]["enabled"] = ["signal_quality", "ica", "wavelet"]

    # Subject 1 = source (calibration), Subject 2 = target (unlabeled cal + test)
    subject_ids = np.asarray(subject_ids).ravel()
    sub1 = np.where(subject_ids == 1)[0]
    sub2 = np.where(subject_ids == 2)[0]
    if len(sub1) == 0 or len(sub2) == 0:
        # Fallback: first half = source, second half = target
        n = len(y)
        sub1, sub2 = np.arange(n // 2), np.arange(n // 2, n)
    X1, y1 = X[sub1], y[sub1]
    X2, y2 = X[sub2], y[sub2]
    mask1 = y1 >= 0
    mask2 = y2 >= 0
    X1, y1 = X1[mask1], y1[mask1]
    X2, y2 = X2[mask2], y2[mask2]

    n_cal_src = min(len(X1), cfg["agent"].get("calibration_trials", 50))
    frac = cfg.get("transfer", {}).get("target_unlabeled_fraction", 0.2)
    n_unlabeled = max(1, min(len(X2) - 1, int(len(X2) * frac)))
    X_source_cal = X1[:n_cal_src]
    y_source_cal = y1[:n_cal_src]
    X_target_cal = X2[:n_unlabeled] if transfer_enabled else None
    X_target_test = X2[n_unlabeled:]
    y_target_test = y2[n_unlabeled:]

    registry = PipelineRegistry(cfg)
    pipelines = registry.build_pipelines(fs=fs, n_classes=n_classes, channel_names=channel_names)
    agent = PipelineSelectionAgent(cfg)

    t0 = time.perf_counter()
    metrics = agent.run_calibration(
        pipelines,
        X_source_cal,
        y_source_cal,
        n_classes=n_classes,
        max_parallel=0,
        X_target_cal=X_target_cal,
        loso_fold_info={"target_subject": 2, "source_subjects": [1]} if transfer_enabled else None,
    )
    t_sec = time.perf_counter() - t0

    pipelines_kept = agent.prune(pipelines)
    agent.select_top_n(pipelines_kept)
    try:
        best = agent.select_best(pipelines)
    except RuntimeError:
        best = None

    cv_acc = None
    test_acc = None
    if best is not None:
        m = metrics.get(best.name)
        if m:
            cv_acc = getattr(m, "cv_accuracy", None) or m.accuracy
        if len(X_target_test) > 0:
            y_pred = best.predict(X_target_test)
            test_acc = float(np.mean(y_pred == y_target_test))

    stats = getattr(agent, "pruning_runtime_stats", {})
    corr = getattr(agent, "screening_correlation_with_cv", None)
    return {
        "transfer_enabled": transfer_enabled,
        "best_pipeline": best.name if best else None,
        "cv_accuracy": cv_acc,
        "test_accuracy": test_acc,
        "runtime_sec": round(t_sec, 3),
        "pipelines_before": stats.get("pipelines_before"),
        "pipelines_after": stats.get("pipelines_after"),
        "cv_fits_executed": stats.get("cv_fits_executed"),
        "screening_correlation": corr,
        "n_source_cal": len(X_source_cal),
        "n_target_cal": len(X_target_cal) if X_target_cal is not None else 0,
        "n_target_test": len(X_target_test),
    }


def main():
    print("Loading BCI IV 2a (subjects 1 & 2)...")
    X, y, subject_ids, fs, n_classes, channel_names, config, dataset = load_bci2a_subjects_1_2()
    subject_ids = np.asarray(subject_ids).ravel()[:len(y)]
    print(f"  Trials: {len(y)}, subjects: {np.unique(subject_ids)}")

    print("\n" + "=" * 70)
    print("1) WITHOUT transfer learning (source calibration only)")
    print("=" * 70)
    out_no = run_one(False, X, y, subject_ids, fs, n_classes, channel_names, config, dataset)
    print(f"   Best pipeline:        {out_no['best_pipeline']}")
    print(f"   CV accuracy:          {out_no['cv_accuracy']:.4f}" if out_no['cv_accuracy'] is not None else "   CV accuracy:          N/A")
    print(f"   Test accuracy (S2):  {out_no['test_accuracy']:.4f}" if out_no['test_accuracy'] is not None else "   Test accuracy:       N/A")
    print(f"   Runtime:              {out_no['runtime_sec']} s")
    print(f"   Pipelines before:     {out_no['pipelines_before']}")
    print(f"   Pipelines after:      {out_no['pipelines_after']}")
    if out_no.get("screening_correlation") is not None:
        print(f"   Screening–CV corr:     {out_no['screening_correlation']:.3f}")

    print("\n" + "=" * 70)
    print("2) WITH transfer learning (source + unlabeled target adaptation)")
    print("=" * 70)
    out_yes = run_one(True, X, y, subject_ids, fs, n_classes, channel_names, config, dataset)
    print(f"   Best pipeline:        {out_yes['best_pipeline']}")
    print(f"   CV accuracy:          {out_yes['cv_accuracy']:.4f}" if out_yes['cv_accuracy'] is not None else "   CV accuracy:          N/A")
    print(f"   Test accuracy (S2):  {out_yes['test_accuracy']:.4f}" if out_yes['test_accuracy'] is not None else "   Test accuracy:       N/A")
    print(f"   Runtime:              {out_yes['runtime_sec']} s")
    print(f"   Pipelines before:     {out_yes['pipelines_before']}")
    print(f"   Pipelines after:      {out_yes['pipelines_after']}")
    if out_yes.get("screening_correlation") is not None:
        print(f"   Screening–CV corr:     {out_yes['screening_correlation']:.3f}")

    print("\n" + "=" * 70)
    print("COMPARISON (with new adaptive pruning recommendations)")
    print("=" * 70)
    print(f"   {'Metric':<28}  {'No transfer':>14}  {'With transfer':>14}")
    print("   " + "-" * 60)
    print(f"   {'Best pipeline':<28}  {str(out_no['best_pipeline'] or 'N/A'):>14}  {str(out_yes['best_pipeline'] or 'N/A'):>14}")
    cv_no = f"{out_no['cv_accuracy']:.4f}" if out_no['cv_accuracy'] is not None else "N/A"
    cv_yes = f"{out_yes['cv_accuracy']:.4f}" if out_yes['cv_accuracy'] is not None else "N/A"
    print(f"   {'CV accuracy':<28}  {cv_no:>14}  {cv_yes:>14}")
    te_no = f"{out_no['test_accuracy']:.4f}" if out_no['test_accuracy'] is not None else "N/A"
    te_yes = f"{out_yes['test_accuracy']:.4f}" if out_yes['test_accuracy'] is not None else "N/A"
    print(f"   {'Test accuracy (subject 2)':<28}  {te_no:>14}  {te_yes:>14}")
    print(f"   {'Runtime (s)':<28}  {out_no['runtime_sec']:>14}  {out_yes['runtime_sec']:>14}")
    print(f"   {'Pipelines after screening':<28}  {str(out_no['pipelines_after']):>14}  {str(out_yes['pipelines_after']):>14}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
