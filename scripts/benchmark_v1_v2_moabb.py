#!/usr/bin/env python3
"""
Benchmark v1 vs v2 on a MOABB dataset (downloads automatically).

Uses the same data for both configs; reports calibration accuracy, best pipeline,
and test-set metrics when available. Requires: pip install moabb

Example:
  PYTHONPATH=. python scripts/benchmark_v1_v2_moabb.py
  PYTHONPATH=. python scripts/benchmark_v1_v2_moabb.py --dataset PhysionetMI --subjects 1 2
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from bci_framework.utils.config_loader import load_config, get_config
from bci_framework.utils.splits import get_train_test_trials
from bci_framework.pipelines import PipelineRegistry
from bci_framework.agent import PipelineSelectionAgent


def _check_moabb():
    try:
        import moabb
        from moabb.paradigms import MotorImagery
        return True
    except ImportError:
        return False


def load_moabb_data(dataset_name: str, subjects: list[int], paradigm_kwargs: dict, concatenate_subjects: bool = True):
    """Load MOABB dataset via our adapter; returns X, y, fs, channel_names, n_classes.
    If concatenate_subjects is True and multiple subjects requested, concatenate all trials (bigger dataset)."""
    from bci_framework.datasets.moabb_loader import MOABBDatasetLoader
    loader = MOABBDatasetLoader(
        dataset_name=dataset_name,
        paradigm="motor_imagery",
        **paradigm_kwargs,
    )
    result = loader.load(subjects=subjects, download_if_missing=True)
    if isinstance(result, dict) and len(result) > 1 and concatenate_subjects:
        # Multi-subject: concatenate all subjects for a bigger dataset
        parts_x, parts_y = [], []
        for sid in subjects:
            ds = result.get(sid)
            if ds is None:
                continue
            x = np.asarray(ds.data, dtype=np.float64)
            y = np.asarray(ds.labels, dtype=np.int64).ravel()
            labeled = y >= 0
            parts_x.append(x[labeled])
            parts_y.append(y[labeled])
        if not parts_x:
            ds = result.get(subjects[0]) or next(iter(result.values()))
            X = np.asarray(ds.data, dtype=np.float64)
            y = np.asarray(ds.labels, dtype=np.int64).ravel()
            labeled = y >= 0
            X, y = X[labeled], y[labeled]
        else:
            X = np.concatenate(parts_x, axis=0)
            y = np.concatenate(parts_y, axis=0)
        ds_ref = result.get(subjects[0]) or next(iter(result.values()))
        return X, y, ds_ref.fs, ds_ref.channel_names, len(ds_ref.class_names), getattr(loader, "capabilities", None)
    if isinstance(result, dict):
        ds = result.get(subjects[0]) or next(iter(result.values()))
    else:
        ds = result
    X = np.asarray(ds.data, dtype=np.float64)
    y = np.asarray(ds.labels, dtype=np.int64).ravel()
    labeled = y >= 0
    X, y = X[labeled], y[labeled]
    return X, y, ds.fs, ds.channel_names, len(ds.class_names), getattr(loader, "capabilities", None)


def load_moabb_data_loso(
    dataset_name: str,
    subjects: list[int],
    paradigm_kwargs: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, list, int, Any]:
    """Load MOABB data with subject_ids per trial for LOSO/transfer. Returns (X, y, subject_ids, fs, channel_names, n_classes, capabilities)."""
    from bci_framework.datasets.moabb_loader import MOABBDatasetLoader
    loader = MOABBDatasetLoader(
        dataset_name=dataset_name,
        paradigm="motor_imagery",
        **paradigm_kwargs,
    )
    result = loader.load(subjects=subjects, download_if_missing=True)
    if not isinstance(result, dict) or len(result) < 2:
        # Single subject or single dataset: no subject_ids
        if isinstance(result, dict):
            ds = result.get(subjects[0]) or next(iter(result.values()))
        else:
            ds = result
        X = np.asarray(ds.data, dtype=np.float64)
        y = np.asarray(ds.labels, dtype=np.int64).ravel()
        labeled = y >= 0
        X, y = X[labeled], y[labeled]
        subject_ids = np.zeros(len(X), dtype=np.int64)
        if isinstance(result, dict):
            subject_ids[:] = list(result.keys())[0]
        return X, y, subject_ids, ds.fs, ds.channel_names, len(ds.class_names), getattr(loader, "capabilities", None)
    parts_x, parts_y, parts_sid = [], [], []
    for sid in subjects:
        ds = result.get(sid)
        if ds is None:
            continue
        x = np.asarray(ds.data, dtype=np.float64)
        y = np.asarray(ds.labels, dtype=np.int64).ravel()
        labeled = y >= 0
        parts_x.append(x[labeled])
        parts_y.append(y[labeled])
        parts_sid.append(np.full(np.sum(labeled), sid, dtype=np.int64))
    X = np.concatenate(parts_x, axis=0)
    y = np.concatenate(parts_y, axis=0)
    subject_ids = np.concatenate(parts_sid, axis=0)
    ds_ref = result.get(subjects[0]) or next(iter(result.values()))
    return X, y, subject_ids, ds_ref.fs, ds_ref.channel_names, len(ds_ref.class_names), getattr(loader, "capabilities", None)


def get_base_config(disable_advanced: bool = True):
    config_path = ROOT / "bci_framework" / "config.yaml"
    if config_path.exists():
        load_config(config_path)
        cfg = copy.deepcopy(get_config())
    else:
        cfg = {}
    cfg.setdefault("mode", "offline")
    cfg.setdefault("task", "motor_imagery")
    cfg.setdefault("preprocessing", {})
    cfg["preprocessing"].setdefault("reference", "car")
    cfg.setdefault("spatial_filter", {})
    if disable_advanced:
        cfg["advanced_preprocessing"] = {"enabled": []}
    cfg.setdefault("pipelines", {"auto_generate": True, "max_combinations": 12, "explicit": []})
    cfg.setdefault("agent", {
        "calibration_trials": 50,
        "cv_folds": 5,
        "overfit_penalty_weight": 0.2,
        "prefer_linear_models": True,
        "top_n_pipelines": 3,
        "prune_thresholds": {
            "min_accuracy": 0.2,
            "max_latency_ms": 500,
            "latency_budget_ms": 300,
            "max_stability_variance": 0.2,
        },
    })
    cfg.setdefault("features", {})
    cfg.setdefault("classifiers", {})
    cfg.setdefault("transfer", {"enabled": False, "method": "none", "target_unlabeled_fraction": 0.2})
    return cfg


def _pipeline_family(name: str) -> str:
    """Classify pipeline by feature/classifier: csp, riemannian, deep."""
    n = (name or "").lower()
    if "csp" in n:
        return "CSP"
    if "riemannian" in n or "covariance" in n or "mdm" in n:
        return "Riemannian"
    if "deep" in n or "eegnet" in n or "transformer" in n:
        return "Deep"
    return "Other"


def _print_result(_label: str, r: dict) -> None:
    """Print benchmark result summary (train/CV/test, overfitting gap, by family)."""
    print(f"  Pipelines: {r['n_pipelines']}, sample: {r['sample_names'][:2]}")
    print(f"  Best: {r['best_name']}")
    if r.get("train_accuracy") is not None:
        print(f"  Train accuracy:  {r['train_accuracy']:.4f}")
    if r.get("cv_accuracy") is not None:
        print(f"  CV accuracy:     {r['cv_accuracy']:.4f}")
    if r.get("overfitting_gap") is not None:
        print(f"  Overfitting gap: {r['overfitting_gap']:.4f} (train - cv)")
    print(f"  Test accuracy:   {(r['test_accuracy'] or 0):.4f}" if r.get("test_accuracy") is not None else "  Test: N/A")
    by_fam = r.get("by_family") or {}
    if by_fam:
        print("  Best by family (CSP / Riemannian / Deep):")
        for fam in ("CSP", "Riemannian", "Deep", "Other"):
            if fam not in by_fam:
                continue
            b = by_fam[fam]
            cv_s = f", cv={b['cv_accuracy']:.3f}" if b.get("cv_accuracy") is not None else ""
            test_s = f", test={b['test_accuracy']:.3f}" if b.get("test_accuracy") is not None else ""
            print(f"    {fam}: {b['name']} (train={b['train_accuracy']:.3f}{cv_s}{test_s})")


def _print_full_v3_result(r: dict, config: dict) -> None:
    """Print full v3 transfer-learning result: best, by family, and per-pipeline table."""
    print()
    print("  --- Best pipeline ---")
    print(f"  Best: {r['best_name']}")
    if r.get("train_accuracy") is not None:
        print(f"  Train accuracy:  {r['train_accuracy']:.4f}")
    if r.get("cv_accuracy") is not None:
        print(f"  CV accuracy:     {r['cv_accuracy']:.4f}")
    if r.get("overfitting_gap") is not None:
        print(f"  Overfitting gap: {r['overfitting_gap']:.4f} (train - cv)")
    if r.get("test_accuracy") is not None:
        print(f"  Test accuracy:  {r['test_accuracy']:.4f}")
    else:
        print("  Test accuracy:  N/A")
    transfer_cfg = config.get("transfer", {})
    print(f"  Transfer: enabled={transfer_cfg.get('enabled')}, method={transfer_cfg.get('method')}, target_unlabeled_frac={transfer_cfg.get('target_unlabeled_fraction')}")

    by_fam = r.get("by_family") or {}
    if by_fam:
        print()
        print("  --- Best by family ---")
        for fam in ("CSP", "Riemannian", "Deep", "Other"):
            if fam not in by_fam:
                continue
            b = by_fam[fam]
            cv_s = f", cv={b['cv_accuracy']:.3f}" if b.get("cv_accuracy") is not None else ""
            test_s = f", test={b['test_accuracy']:.3f}" if b.get("test_accuracy") is not None else ""
            print(f"    {fam}: {b['name']} (train={b['train_accuracy']:.3f}{cv_s}{test_s})")

    metrics = r.get("metrics") or {}
    if metrics:
        print()
        print("  --- All pipelines (train / cv / test) ---")
        rows = []
        for name, m in metrics.items():
            ta = getattr(m, "train_accuracy", None) or getattr(m, "accuracy", None)
            cva = getattr(m, "cv_accuracy", None)
            test_a = getattr(m, "test_accuracy", None)
            rows.append((name, ta, cva, test_a))
        # Sort by test accuracy desc, then cv, then train
        rows.sort(key=lambda x: (x[3] if x[3] is not None else -1, x[2] if x[2] is not None else -1, x[1] if x[1] is not None else -1), reverse=True)
        for name, ta, cva, test_a in rows[:25]:
            ta_s = f"{ta:.3f}" if ta is not None else "N/A"
            cv_s = f"{cva:.3f}" if cva is not None else "N/A"
            test_s = f"{test_a:.3f}" if test_a is not None else "N/A"
            print(f"    {name}: train={ta_s}, cv={cv_s}, test={test_s}")
        if len(rows) > 25:
            print(f"    ... and {len(rows) - 25} more")
    print()


def run_benchmark(
    config,
    X_train,
    y_train,
    X_test,
    y_test,
    n_classes,
    channel_names,
    X_target_cal: np.ndarray | None = None,
    capabilities: Any = None,
    loso_fold_info: dict | None = None,
):
    """Run pipeline selection and evaluation. X_target_cal: unlabeled target trials for transfer (v3 LOSO)."""
    n_train = len(X_train)
    cal_trials = min(n_train, max(50, min(400, n_train // 4)))
    config = copy.deepcopy(config)
    config.setdefault("agent", {})["calibration_trials"] = cal_trials
    if capabilities is not None:
        config["spatial_capabilities"] = capabilities

    # v3.2: transfer debug logging
    if X_target_cal is not None and len(X_target_cal) > 0:
        tc = config.get("transfer", {})
        print("[TRANSFER]")
        print(f"  Source trials: {len(X_train)}")
        print(f"  Target unlabeled trials: {len(X_target_cal)}")
        print(f"  Regularization epsilon: {tc.get('regularization', 1e-3)}")
        print(f"  Safe mode (dim cap, float32): {tc.get('safe_mode', False)}")

    # v3.1: report spatial capabilities before running pipelines
    caps = config.get("spatial_capabilities")
    if caps is not None:
        has_montage = getattr(caps, "has_montage", False) or (isinstance(caps, dict) and caps.get("has_montage", False))
        lap = getattr(caps, "laplacian_supported", False) or (isinstance(caps, dict) and caps.get("laplacian_supported", False))
        ged = getattr(caps, "gedai_supported", False) or (isinstance(caps, dict) and caps.get("gedai_supported", False))
        print("Spatial capabilities:")
        print("  Montage:    " + ("Yes" if has_montage else "No"))
        print("  Laplacian: " + ("Supported" if lap else "Not Supported"))
        print("  GEDAI:      " + ("Supported" if ged else "Not Supported"))
    else:
        print("Spatial capabilities: (not detected)")

    registry = PipelineRegistry(config)
    pipelines = registry.build_pipelines(
        fs=250.0, n_classes=n_classes, channel_names=channel_names
    )
    agent = PipelineSelectionAgent(config)
    n_cal = min(len(X_train), agent.calibration_trials)
    metrics = agent.run_calibration(
        pipelines, X_train[:n_cal], y_train[:n_cal],
        n_classes=n_classes, max_parallel=1,
        X_target_cal=X_target_cal,
        loso_fold_info=loso_fold_info,
    )
    kept = agent.prune(pipelines)
    agent.select_top_n(kept)
    try:
        best = agent.select_best(pipelines)  # v3.2: argmax CV over all valid pipelines
    except RuntimeError as e:
        best = None
        print(f"  Warning: {e}")
    if best is None:
        return {
            "best_name": None,
            "train_accuracy": None,
            "cv_accuracy": None,
            "test_accuracy": None,
            "overfitting_gap": None,
            "n_pipelines": len(pipelines),
            "sample_names": [p.name for p in pipelines[:3]],
            "metrics": metrics,
            "pipelines": pipelines,
            "by_family": {},
        }
    m = metrics[best.name]
    train_acc = getattr(m, "train_accuracy", None) or m.accuracy
    cv_acc = getattr(m, "cv_accuracy", None)
    test_acc = None
    if len(X_test) > 0:
        y_pred = best.predict(X_test)
        test_acc = float(np.mean(y_pred == y_test))
    overfit_gap = (train_acc - cv_acc) if cv_acc is not None else None

    # Best per family (CSP, Riemannian, Deep)
    by_family: dict[str, dict] = {}
    for p in pipelines:
        fam = _pipeline_family(p.name)
        mm = metrics.get(p.name)
        if mm is None:
            continue
        ta = getattr(mm, "train_accuracy", None) or mm.accuracy
        cva = getattr(mm, "cv_accuracy", None)
        rank_val = cva if cva is not None else ta
        if fam not in by_family or (by_family[fam].get("cv_accuracy") or by_family[fam].get("train_accuracy") or 0) < rank_val:
            test_fam = None
            if len(X_test) > 0:
                try:
                    test_fam = float(np.mean(p.predict(X_test) == y_test))
                except Exception:
                    pass
            by_family[fam] = {
                "name": p.name,
                "train_accuracy": ta,
                "cv_accuracy": cva,
                "test_accuracy": test_fam,
            }

    return {
        "best_name": best.name,
        "train_accuracy": train_acc,
        "cv_accuracy": cv_acc,
        "best_accuracy": train_acc,  # backward compat
        "test_accuracy": test_acc,
        "overfitting_gap": overfit_gap,
        "n_pipelines": len(pipelines),
        "sample_names": [p.name for p in pipelines[:3]],
        "metrics": metrics,
        "pipelines": pipelines,
        "by_family": by_family,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark v1 vs v2 on MOABB dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        default="BNCI2014_001",
        help="MOABB dataset (e.g. BNCI2014_001, PhysionetMI)",
    )
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        default=[1],
        help="Subject IDs (default: 1)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train ratio for split (default 0.8)",
    )
    parser.add_argument(
        "--loso",
        action="store_true",
        help="Run LOSO + v3 transfer comparison (need >=2 subjects)",
    )
    parser.add_argument(
        "--loso-holdout",
        type=int,
        default=None,
        help="Subject ID to hold out for LOSO (default: last in --subjects)",
    )
    parser.add_argument(
        "--transfer-method",
        type=str,
        default="coral",
        choices=["zscore", "coral", "riemann_transport"],
        help="Domain adapter for v3 when --loso (default: coral)",
    )
    parser.add_argument(
        "--transfer-safe-mode",
        action="store_true",
        help="Enable transfer safe_mode (CORAL dim cap 128, float32; for 8–16GB RAM)",
    )
    parser.add_argument(
        "--v3-only",
        action="store_true",
        help="With --loso: run only v3 transfer learning (skip v1/v2).",
    )
    args = parser.parse_args()

    if not _check_moabb():
        print("MOABB is not installed. Install with: pip install moabb")
        print("Then run: PYTHONPATH=. python scripts/benchmark_v1_v2_moabb.py")
        return 1

    # LOSO + v3 transfer path
    if args.loso and len(args.subjects) >= 2:
        print("Loading MOABB dataset (LOSO):", args.dataset, "subjects:", args.subjects)
        try:
            X, y, subject_ids, fs, channel_names, n_classes, capabilities = load_moabb_data_loso(
                args.dataset, args.subjects, paradigm_kwargs={"resample": 250}
            )
        except Exception as e:
            print("Failed to load MOABB data:", e)
            return 1
        holdout = args.loso_holdout if args.loso_holdout is not None else args.subjects[-1]
        src_mask = subject_ids != holdout
        tgt_mask = subject_ids == holdout
        X_source = X[src_mask]
        y_source = y[src_mask]
        X_target_all = X[tgt_mask]
        y_target_all = y[tgt_mask]
        frac = 0.2
        n_tgt = len(X_target_all)
        n_unlabeled = max(1, int(n_tgt * frac))
        X_target_unlabeled = X_target_all[:n_unlabeled]
        X_test = X_target_all[n_unlabeled:]
        y_test = y_target_all[n_unlabeled:]
        X_train = X_source
        y_train = y_source
        print(f"  LOSO holdout subject: {holdout}")
        print(f"  Source: {len(X_train)} trials, target unlabeled: {n_unlabeled}, target test: {len(X_test)}")
        print(f"  Loaded: {X.shape[0]} trials, {n_classes} classes, fs={fs}")

        config_v2 = get_base_config()
        config_v2["spatial_filter"]["enabled"] = True
        config_v2["spatial_filter"]["method"] = "laplacian"
        config_v2["transfer"] = {"enabled": False, "method": "none", "target_unlabeled_fraction": frac}
        config_v3 = get_base_config()
        config_v3["spatial_filter"]["enabled"] = True
        config_v3["spatial_filter"]["method"] = "laplacian"
        config_v3["transfer"] = {
            "enabled": True,
            "method": args.transfer_method,
            "target_unlabeled_fraction": frac,
            "regularization": 1e-3,
            "safe_mode": args.transfer_safe_mode,
        }

        if args.v3_only:
            print()
            print("=" * 60)
            print("V3 — Transfer learning & cross-subject (LOSO)")
            print("=" * 60)
            r3 = run_benchmark(
                config_v3, X_train, y_train, X_test, y_test, n_classes, channel_names,
                X_target_cal=X_target_unlabeled,
                capabilities=capabilities,
                loso_fold_info={"target_subject": holdout, "source_subjects": [s for s in args.subjects if s != holdout]},
            )
            _print_full_v3_result(r3, config_v3)
            print("=" * 60)
            print("V3 transfer learning (cross-subject) summary")
            print("=" * 60)
            t3 = r3.get("test_accuracy")
            print(f"  Test accuracy (holdout subject): {t3:.4f}" if t3 is not None else "  Test accuracy: N/A")
            print(f"  Transfer method: {args.transfer_method}")
            print(f"  Holdout subject: {holdout}")
            print(f"  Source trials: {len(X_train)}, target test trials: {len(X_test)}")
            print()
            print("  Latest architecture: Dataset → capabilities → Registry (spatial filter/auto) →")
            print("  Preprocessing (strict/auto) → Features → Domain adaptation (v3) → Classifier → Agent.")
            print("  See docs/ARCHITECTURE_V3.md for full summary.")
            return 0

        print()
        print("=" * 60)
        print("v2 baseline (no transfer)")
        print("=" * 60)
        r2 = run_benchmark(
            config_v2, X_train, y_train, X_test, y_test, n_classes, channel_names,
            capabilities=capabilities,
            loso_fold_info={"target_subject": holdout, "source_subjects": [s for s in args.subjects if s != holdout]},
        )
        _print_result("v2", r2)

        print()
        print("=" * 60)
        print("v3 with transfer (domain adaptation)")
        print("=" * 60)
        r3 = run_benchmark(
            config_v3, X_train, y_train, X_test, y_test, n_classes, channel_names,
            X_target_cal=X_target_unlabeled,
            capabilities=capabilities,
            loso_fold_info={"target_subject": holdout, "source_subjects": [s for s in args.subjects if s != holdout]},
        )
        _print_result("v3", r3)

        print()
        print("=" * 60)
        print("Transfer learning summary (LOSO)")
        print("=" * 60)
        t2 = r2.get("test_accuracy")
        t3 = r3.get("test_accuracy")
        if t2 is not None and t3 is not None:
            diff = t3 - t2
            pct = (diff / t2 * 100) if t2 > 0 else 0
            print(f"  v2 test accuracy:  {t2:.4f}")
            print(f"  v3 test accuracy:  {t3:.4f}")
            print(f"  Transfer improvement: {diff:+.4f} ({pct:+.1f}%)")
        return 0

    print("Loading MOABB dataset:", args.dataset, "subjects:", args.subjects)
    print("(First run may download data; this can take a few minutes.)")
    try:
        X, y, fs, channel_names, n_classes, capabilities = load_moabb_data(
            args.dataset,
            args.subjects,
            paradigm_kwargs={"resample": 250},
        )
    except Exception as e:
        print("Failed to load MOABB data:", e)
        return 1

    print(f"  Loaded: {X.shape[0]} trials, {X.shape[1]} ch, {X.shape[2]} samples, {n_classes} classes, fs={fs}")

    idx_train, idx_test = get_train_test_trials(
        len(X), train_ratio=args.train_ratio, random_state=42
    )
    X_train, y_train = X[idx_train], y[idx_train]
    X_test, y_test = X[idx_test], y[idx_test]
    print(f"  Split: {len(X_train)} train, {len(X_test)} test")

    # v1-style
    config_v1 = get_base_config()
    config_v1["spatial_filter"]["enabled"] = False
    config_v1["spatial_filter"].pop("method", None)

    # v2-style (CAR spatial layer)
    config_v2 = get_base_config()
    config_v2["spatial_filter"]["enabled"] = True
    config_v2["spatial_filter"]["method"] = "car"

    print()
    print("=" * 60)
    print("v1-style (spatial_filter.enabled = false)")
    print("=" * 60)
    r1 = run_benchmark(
        config_v1, X_train, y_train, X_test, y_test,
        n_classes, channel_names,
        capabilities=capabilities,
    )
    _print_result("v1", r1)

    print()
    print("=" * 60)
    print("v2-style (spatial_filter.enabled = true, CV ranking, laplacian)")
    print("=" * 60)
    config_v2["spatial_filter"]["method"] = "laplacian"
    r2 = run_benchmark(
        config_v2, X_train, y_train, X_test, y_test,
        n_classes, channel_names,
        capabilities=capabilities,
    )
    _print_result("v2", r2)

    print()
    print("=" * 60)
    print("Benchmark summary (MOABB dataset:", args.dataset, ")")
    print("=" * 60)
    def _fmt(v):
        return f"{v:.4f}" if v is not None else "N/A"
    print(f"  v1 train / CV / test: {_fmt(r1.get('train_accuracy') or r1.get('best_accuracy'))} / {_fmt(r1.get('cv_accuracy'))} / {_fmt(r1.get('test_accuracy'))}")
    print(f"  v2 train / CV / test: {_fmt(r2.get('train_accuracy') or r2.get('best_accuracy'))} / {_fmt(r2.get('cv_accuracy'))} / {_fmt(r2.get('test_accuracy'))}")
    if r1.get("overfitting_gap") is not None:
        print(f"  v1 overfitting gap: {r1['overfitting_gap']:.4f}")
    if r2.get("overfitting_gap") is not None:
        print(f"  v2 overfitting gap: {r2['overfitting_gap']:.4f}")
    if r1.get("test_accuracy") is not None and r2.get("test_accuracy") is not None:
        print(f"  Test difference (v2 - v1): {r2['test_accuracy'] - r1['test_accuracy']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
