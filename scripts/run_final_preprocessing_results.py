#!/usr/bin/env python3
"""
Run final preprocessing evaluation and report results in the required format.

Runs assertions (no subject leakage, ICA fit scope, baseline reproducibility), then
full GroupKFold evaluation, statistical comparison, and prints/saves final results.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reduce log noise during run
logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)


def run_assertions(X, y, subject_ids, fs, n_classes, ch_names, capabilities, baseline_only=False, pipelines=None):
    """STEP 6: Assert no subject leakage, fit scope for B/C (if not baseline_only), baseline reproducibility. Stop if any fails."""
    from bci_framework.evaluation.preprocessing_evaluation import (
        run_group_kfold,
        run_one_fold,
        RANDOM_STATE,
        CONDITION_ENABLED,
    )
    import numpy as np

    conditions = ["A"] if baseline_only else (pipelines or ["A", "B", "C"])
    fit_conditions = [c for c in conditions if CONDITION_ENABLED.get(c)]

    # Assert 1: No subject leakage (every fold has disjoint train/test subjects)
    for cond in conditions:
        print("    Condition %s ..." % cond, flush=True)
        rows = run_group_kfold(
            X, y, subject_ids, cond, fs, n_classes, ch_names, capabilities,
            n_splits=5, random_state=RANDOM_STATE,
        )
        for r in rows:
            train_s = set(r.get("train_subjects") or [])
            test_s = set(r.get("test_subjects") or [])
            assert len(train_s & test_s) == 0, "Assertion failed: subject leakage in fold."
    print("CHECK: No subject leakage (all folds).")
    if fit_conditions:
        print("  Assertion 2: Fit scope (one fold per fitted condition)...", flush=True)
        from sklearn.model_selection import GroupKFold
        gkf = GroupKFold(n_splits=5)
        for train_idx, test_idx in gkf.split(X, y, groups=subject_ids):
            test_subjects = set(np.unique(subject_ids[test_idx]))
            for cond in fit_conditions:
                out = run_one_fold(
                    X, y, subject_ids, train_idx, test_idx, cond, fs, n_classes, ch_names, capabilities,
                    return_fit_subjects=True,
                )
                fit_s = set(out.get("fit_subjects") or [])
                assert not test_subjects.intersection(fit_s), (
                    "Assertion failed: Condition %s fit included test subject." % cond
                )
            break
        print("CHECK: Fit scope (test subjects not in fit).")
    print("  Assertion 3: baseline reproducibility (run A twice)...", flush=True)

    # Assert 3: Baseline reproducibility (run A twice, mean acc difference < 0.1%)
    r1 = run_group_kfold(X, y, subject_ids, "A", fs, n_classes, ch_names, capabilities, n_splits=5, random_state=RANDOM_STATE)
    r2 = run_group_kfold(X, y, subject_ids, "A", fs, n_classes, ch_names, capabilities, n_splits=5, random_state=RANDOM_STATE)
    acc1 = np.mean([x["metrics"]["accuracy"] for x in r1])
    acc2 = np.mean([x["metrics"]["accuracy"] for x in r2])
    diff_pct = abs(acc1 - acc2) * 100
    assert diff_pct < 0.1, "Assertion failed: baseline reproducibility (diff = %.3f%%)." % diff_pct
    print("CHECK: Baseline reproducibility (diff < 0.1%%).")


def run_diagnostic_stratified_vs_group(X, y, subject_ids, fs, n_classes, ch_names, capabilities):
    """STEP 7 (optional): StratifiedKFold vs GroupKFold mean accuracy and inflation."""
    from bci_framework.evaluation.preprocessing_evaluation import (
        get_fixed_pipeline_config,
        build_single_pipeline,
        run_one_fold,
        compute_fold_metrics,
        RANDOM_STATE,
    )
    from sklearn.model_selection import StratifiedKFold, GroupKFold
    import numpy as np

    config = get_fixed_pipeline_config("A")
    config["spatial_capabilities"] = capabilities

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    stratified_accs = []
    for train_idx, test_idx in skf.split(X, y):
        pipe = build_single_pipeline(config, fs, n_classes, ch_names)
        pipe.fit(X[train_idx], y[train_idx])
        y_pred = pipe.predict(X[test_idx])
        y_proba = pipe.predict_proba(X[test_idx]) if hasattr(pipe, "predict_proba") else None
        if y_proba is None and hasattr(pipe, "classifier") and hasattr(pipe.classifier, "predict_proba"):
            try:
                y_proba = pipe.classifier.predict_proba(pipe.transform(X[test_idx]))
            except Exception:
                y_proba = None
        m = compute_fold_metrics(y[test_idx], y_pred, y_proba, n_classes)
        stratified_accs.append(m["accuracy"])

    gkf = GroupKFold(n_splits=5)
    group_accs = []
    for train_idx, test_idx in gkf.split(X, y, groups=subject_ids):
        out = run_one_fold(
            X, y, subject_ids, train_idx, test_idx, "A", fs, n_classes, ch_names, capabilities,
            random_state=RANDOM_STATE,
        )
        group_accs.append(out["metrics"]["accuracy"])

    mean_strat = float(np.mean(stratified_accs))
    mean_grp = float(np.mean(group_accs))
    inflation = (mean_strat - mean_grp) * 100
    return mean_strat, mean_grp, inflation


def main():
    import argparse
    import numpy as np
    from bci_framework.evaluation.preprocessing_evaluation import (
        load_physionet_mi,
        run_full_evaluation,
        statistical_comparison,
        RANDOM_STATE,
        N_PERM,
        PHYSIONET_DATASET,
    )

    p = argparse.ArgumentParser()
    p.add_argument("--n-subjects", type=int, default=None,
                   help="Cap subjects (default: all 109). Use 15 for faster runs, omit for full dataset.")
    p.add_argument("--out-dir", type=Path, default=None, help="Output dir (default: results/preprocessing_evaluation)")
    p.add_argument("--baseline-only", action="store_true",
                   help="Run only A (bandpass 8-30 Hz). No ICA, no GEDAI.")
    p.add_argument("--pipelines", type=str, default=None,
                   help="Comma-separated pipelines e.g. A,B,C (default: A,B,C). Use A,B to skip GEDAI if pygedai missing.")
    args = p.parse_args()
    out_dir = args.out_dir or (ROOT / "results" / "preprocessing_evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_only = getattr(args, "baseline_only", False)
    pipelines = [x.strip().upper() for x in args.pipelines.split(",")] if args.pipelines else None

    # If C (GEDAI) would run but pygedai is not installed, skip C so the script doesn't crash
    if not baseline_only and ("C" in (pipelines or ["A", "B", "C"])):
        try:
            from pygedai import batch_gedai  # noqa: F401
        except ImportError:
            want = pipelines or ["A", "B", "C"]
            pipelines = [p for p in want if p != "C"]
            if not pipelines:
                pipelines = ["A", "B"]
            print("Note: GEDAI (C) skipped — pygedai not installed. Running pipelines: %s" % ",".join(pipelines), flush=True)

    subjects = list(range(1, (args.n_subjects or 999) + 1)) if args.n_subjects else None
    print("Loading Physionet MI%s..." % (" (first %d subjects)" % args.n_subjects if args.n_subjects else ""))
    X, y, subject_ids, fs, ch_names, n_classes, capabilities, class_distribution = load_physionet_mi(subjects=subjects)
    n_subjects = len(np.unique(subject_ids))
    total_trials = len(y)
    print("Dataset loaded: X.dtype=%s (float32 for memory efficiency)" % X.dtype, flush=True)

    conds_str = " A only" if baseline_only else (" " + ",".join(pipelines or ["A", "B", "C"]))
    print("Running assertions (no leakage%s, reproducibility)..." % ("" if baseline_only else ", fit scope"), flush=True)
    print("  Assertion 1: no subject leakage (GroupKFold%s, 5 folds each)..." % conds_str, flush=True)
    run_assertions(X, y, subject_ids, fs, n_classes, ch_names, capabilities,
                   baseline_only=baseline_only, pipelines=pipelines)

    pipelines_run = pipelines if pipelines else (["A"] if baseline_only else ["A", "B", "C"])
    print("Running full GroupKFold evaluation (pipelines: %s). This may take several minutes..." % ",".join(pipelines_run), flush=True)
    preloaded = (X, y, subject_ids, fs, ch_names, n_classes, capabilities, class_distribution)
    out = run_full_evaluation(subjects=subjects, out_dir=out_dir, random_state=RANDOM_STATE, n_perm=N_PERM,
                              baseline_only=baseline_only, pipelines=pipelines, preloaded_data=preloaded)

    results_gkf = out["results_gkf"]
    paired = out.get("paired_comparisons") or {}
    conditions_run = list(results_gkf.keys())

    # Baseline vs Baseline permutation sanity (only when multiple conditions)
    if len(conditions_run) > 1:
        results_a = results_gkf["A"]
        vals_a = [r["metrics"]["accuracy"] for r in results_a]
        sanity = statistical_comparison(vals_a, vals_a, n_perm=N_PERM, random_state=RANDOM_STATE)
        assert sanity.get("p_value") is not None and sanity["p_value"] >= 0.99, "Baseline vs baseline p should be >= 0.99"

    def mean_std(rows, key):
        vals = [r["metrics"][key] for r in rows]
        return float(np.mean(vals)), float(np.std(vals))

    LABELS = {"A": "BASELINE (Bandpass 8–30 Hz)", "B": "ICA (Bandpass + ICA)", "C": "GEDAI (Bandpass + GEDAI)"}

    print()
    print("=====================================")
    print("FINAL RESULTS — GROUP 5-FOLD (Paired comparisons)")
    print("Dataset: PhysionetMI")
    print("Random state: 42")
    print("Number of subjects: %d" % n_subjects)
    print("Total trials: %d" % total_trials)
    print()

    for cond in conditions_run:
        rows = results_gkf[cond]
        acc, std_acc = mean_std(rows, "accuracy")
        auc, std_auc = mean_std(rows, "roc_auc_macro")
        kap, std_kap = mean_std(rows, "cohen_kappa")
        print("%s" % LABELS.get(cond, cond))
        print("Accuracy: %.4f ± %.4f" % (acc, std_acc))
        print("Macro AUC: %.4f ± %.4f" % (auc, std_auc))
        print("Kappa: %.4f ± %.4f" % (kap, std_kap))
        print()

    print("PAIRED COMPARISONS (Accuracy, fold-level permutation test):")
    if not paired:
        print("  N/A (baseline-only or single pipeline)")
    else:
        for key in sorted(paired.keys()):
            c = paired[key]
            md = (c.get("mean_delta") or 0) * 100
            pv = c.get("p_value")
            cd = c.get("cohens_d")
            ci = c.get("bootstrap_ci_95") or [None, None]
            ci_pp = [ci[0] * 100 if ci[0] is not None else None, ci[1] * 100 if ci[1] is not None else None]
            print("  %s:" % key)
            print("    Mean diff: %+.2f pp | p-value: %.4f | Cohen's d: %.4f" % (md, pv or 0, cd or 0))
            print("    Bootstrap 95%% CI: [%s, %s]" % (
                "%.2f" % ci_pp[0] if ci_pp[0] is not None else "N/A",
                "%.2f" % ci_pp[1] if ci_pp[1] is not None else "N/A",
            ))
    print()

    print("Stability (std across folds):")
    stability = out.get("stability", {})
    for cond in conditions_run:
        s = stability.get(cond, {})
        std_a = s.get("std_across_folds", 0)
        print("  %s: %.4f" % (LABELS.get(cond, cond), std_a))
    print()
    print("=====================================")

    # Optional diagnostic
    print()
    print("DIAGNOSTIC (StratifiedKFold vs GroupKFold):")
    try:
        mean_strat, mean_grp, inflation = run_diagnostic_stratified_vs_group(X, y, subject_ids, fs, n_classes, ch_names, capabilities)
        print("Random StratifiedKFold accuracy mean: %.4f" % mean_strat)
        print("GroupKFold accuracy mean: %.4f" % mean_grp)
        print("Inflation difference: %.2f pp" % inflation)
    except Exception as e:
        print("Diagnostic skipped: %s" % e)

    print()
    print("Saved: %s" % (out_dir / "subject_level_results.csv"))
    print("Saved: %s" % (out_dir / "preprocessing_evaluation_metadata.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
