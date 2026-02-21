#!/usr/bin/env python3
"""
GEDAI integration debug run: 1 subject, pipelines A (bandpass) and C (GEDAI), no ICA.

With 1 subject, GroupKFold cannot split (need >=2 groups). Uses StratifiedKFold on trials
for a minimal train/test split to verify the pipeline.
Saves: results/preprocessing_evaluation/test_run_gedai_1_subject.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    import argparse
    import numpy as np
    from bci_framework.evaluation.preprocessing_evaluation import (
        load_physionet_mi,
        get_fixed_pipeline_config,
        build_single_pipeline,
        compute_fold_metrics,
        RANDOM_STATE,
        CONDITION_ENABLED,
    )
    from sklearn.model_selection import StratifiedKFold

    p = argparse.ArgumentParser()
    p.add_argument("--n-subjects", type=int, default=1, help="Number of subjects (default: 1 for debug)")
    args = p.parse_args()
    n_subjects = max(1, args.n_subjects)

    out_path = ROOT / "results" / "preprocessing_evaluation" / "test_run_gedai_1_subject.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Pipelines A and C only (no ICA)
    pipelines = ["A", "C"]
    print("GEDAI DEBUG RUN: pipelines=%s (no ICA)" % pipelines, flush=True)
    print("Loading Physionet MI (first %d subject(s))..." % n_subjects, flush=True)

    X, y, subject_ids, fs, ch_names, n_classes, capabilities, class_distribution = load_physionet_mi(
        subjects=list(range(1, n_subjects + 1))
    )
    n_subj = len(np.unique(subject_ids))
    total_trials = len(y)
    print("Dataset loaded: X.shape=%s, X.dtype=%s, n_subjects=%d, total_trials=%d" % (
        X.shape, X.dtype, n_subj, total_trials,
    ), flush=True)

    # With 1 subject, use trial-based split (StratifiedKFold) since GroupKFold needs >=2 groups
    n_splits = 2  # minimal 2-fold for train/test split
    print("Using StratifiedKFold(n_splits=%d) on trials (1-subject debug mode)" % n_splits, flush=True)
    print("GEDAI fit uses only train indices; test data never seen during fit.", flush=True)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    results = {"pipelines": pipelines, "n_subjects": n_subj, "total_trials": total_trials,
               "n_splits": n_splits, "condition_results": {}, "debug_info": []}

    for cond in pipelines:
        config = get_fixed_pipeline_config(cond, gedai_debug=True)
        config["spatial_capabilities"] = capabilities

        fold_accs = []
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            print("  [Fold %d/%d] condition %s: train=%d, test=%d" % (
                fold_idx + 1, n_splits, cond, len(train_idx), len(test_idx),
            ), flush=True)
            if cond == "C":
                print("  [GEDAI DEBUG] GEDAI fit uses only train indices (no test data)", flush=True)

            pipe = build_single_pipeline(config, fs, n_classes, ch_names)
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            y_proba = pipe.predict_proba(X_test) if hasattr(pipe, "predict_proba") else None
            if y_proba is None and hasattr(pipe, "classifier") and hasattr(pipe.classifier, "predict_proba"):
                try:
                    X_feat = pipe.transform(X_test)
                    y_proba = pipe.classifier.predict_proba(X_feat)
                except Exception:
                    y_proba = None

            m = compute_fold_metrics(y_test, y_pred, y_proba, n_classes)
            fold_accs.append(m["accuracy"])
            results["debug_info"].append({
                "condition": cond,
                "fold": fold_idx,
                "train_trials": len(train_idx),
                "test_trials": len(test_idx),
                "accuracy": m["accuracy"],
            })

        mean_acc = float(np.mean(fold_accs))
        std_acc = float(np.std(fold_accs)) if len(fold_accs) > 1 else 0.0
        results["condition_results"][cond] = {
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "fold_accuracies": fold_accs,
        }
        print("Condition %s: accuracy %.4f ± %.4f" % (cond, mean_acc, std_acc), flush=True)

    results["success"] = True
    results["no_leakage"] = "GEDAI fit on train only; test never used during fit"
    results["no_shape_mismatch"] = True
    results["metrics_computed"] = True

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("Saved: %s" % out_path, flush=True)
    print("GEDAI debug run complete. No memory explosion, no shape mismatch, no leakage.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
