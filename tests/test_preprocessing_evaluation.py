"""
Validation tests for preprocessing evaluation (ICA vs baseline, GroupKFold only).

Test 1 — No subject leakage: no subject in both train and test in any fold.
Test 2 — ICA fit scope: ICA fit subjects do not include test subjects.
Test 3 — Baseline reproducibility: run baseline twice → mean accuracy difference < 0.1%.
Test 4 — Permutation sanity: baseline vs baseline → p ≥ 0.99.
Test 5 — Random trial-level k-fold (diagnostic): StratifiedKFold ignoring subject IDs;
         assert mean accuracy > GroupKFold; log inflation magnitude.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING)


def _load_small():
    """Load a small subset for fast tests."""
    from bci_framework.evaluation.preprocessing_evaluation import load_physionet_mi
    try:
        out = load_physionet_mi(subjects=[1, 2, 3])
        X, y, subject_ids, fs, ch_names, n_classes, capabilities, _ = out
    except Exception as e:
        pytest.skip("Physionet MI load failed (moabb/data): %s" % e)
    return X, y, subject_ids, fs, ch_names, n_classes, capabilities


# ---------- Test 1: No Subject Leakage ----------
def test_no_subject_leakage_group_kfold():
    """For each GroupKFold fold: assert len(set(train_subjects) & set(test_subjects)) == 0."""
    from bci_framework.evaluation.preprocessing_evaluation import run_group_kfold, load_physionet_mi, RANDOM_STATE

    try:
        X, y, subject_ids, fs, ch_names, n_classes, capabilities, _ = load_physionet_mi(subjects=[1, 2, 3, 4, 5])
    except Exception as e:
        pytest.skip("Physionet MI load failed: %s" % e)
    for cond in ["A", "B"]:
        rows = run_group_kfold(
            X, y, subject_ids, cond, fs, n_classes, ch_names, capabilities,
            n_splits=5, random_state=RANDOM_STATE,
        )
        for r in rows:
            train_subjects = set(r.get("train_subjects") or [])
            test_subjects = set(r.get("test_subjects") or [])
            assert len(train_subjects & test_subjects) == 0, "GroupKFold subject leakage"


# ---------- Test 2: ICA Fit Scope ----------
def test_ica_fit_scope():
    """Verify ICA (condition B) fit subjects do not include test subjects. Log which subjects were used for ICA fitting."""
    from bci_framework.evaluation.preprocessing_evaluation import run_one_fold, load_physionet_mi
    from sklearn.model_selection import GroupKFold

    try:
        X, y, subject_ids, fs, ch_names, n_classes, capabilities, _ = load_physionet_mi(subjects=[1, 2, 3, 4])
    except Exception as e:
        pytest.skip("Physionet MI load failed: %s" % e)
    gkf = GroupKFold(n_splits=2)
    for train_idx, test_idx in gkf.split(X, y, groups=subject_ids):
        test_subjects = set(np.unique(subject_ids[test_idx]))
        out = run_one_fold(
            X, y, subject_ids, train_idx, test_idx, "B", fs, n_classes, ch_names, capabilities,
            return_fit_subjects=True,
        )
        fit_subjects = set(out.get("fit_subjects") or [])
        assert not test_subjects.intersection(fit_subjects), (
            "ICA fit must not include test subject. Fit subjects: %s" % list(fit_subjects)
        )
        break


# ---------- Test 3: Baseline Reproducibility ----------
def test_baseline_reproducibility():
    """Run baseline (condition A) twice with GroupKFold. Mean accuracy difference must be < 0.1%."""
    from bci_framework.evaluation.preprocessing_evaluation import run_group_kfold, load_physionet_mi, RANDOM_STATE

    try:
        X, y, subject_ids, fs, ch_names, n_classes, capabilities, _ = load_physionet_mi(subjects=[1, 2, 3, 4, 5])
    except Exception as e:
        pytest.skip("Physionet MI load failed: %s" % e)
    r1 = run_group_kfold(X, y, subject_ids, "A", fs, n_classes, ch_names, capabilities, n_splits=5, random_state=RANDOM_STATE)
    r2 = run_group_kfold(X, y, subject_ids, "A", fs, n_classes, ch_names, capabilities, n_splits=5, random_state=RANDOM_STATE)
    acc1 = np.mean([r["metrics"]["accuracy"] for r in r1])
    acc2 = np.mean([r["metrics"]["accuracy"] for r in r2])
    diff_pct = abs(acc1 - acc2) * 100
    assert diff_pct < 0.1, (
        "Baseline reproducibility failed: mean accuracy difference = %.3f%%. Must be < 0.1%%." % diff_pct
    )


# ---------- Test 4: Permutation Test Sanity ----------
def test_permutation_baseline_vs_baseline():
    """Comparing baseline vs baseline: p-value must be ≥ 0.99."""
    from bci_framework.evaluation.preprocessing_evaluation import (
        statistical_comparison,
        run_group_kfold,
        load_physionet_mi,
        RANDOM_STATE,
        N_PERM,
    )

    try:
        X, y, subject_ids, fs, ch_names, n_classes, capabilities, _ = load_physionet_mi(subjects=[1, 2, 3, 4, 5])
    except Exception as e:
        pytest.skip("Physionet MI load failed: %s" % e)
    r = run_group_kfold(X, y, subject_ids, "A", fs, n_classes, ch_names, capabilities, n_splits=5, random_state=RANDOM_STATE)
    vals = [x["metrics"]["accuracy"] for x in r]
    comp = statistical_comparison(vals, vals, n_perm=N_PERM, random_state=RANDOM_STATE)
    p = comp.get("p_value")
    assert p is not None and p >= 0.99, (
        "Permutation test sanity failed: baseline vs baseline should give p ≥ 0.99, got p = %s." % p
    )


# ---------- Test 5: Performance Inflation (Diagnostic) ----------
def test_performance_inflation_stratified_vs_group_kfold():
    """StratifiedKFold (ignoring subject IDs) should give higher mean accuracy than GroupKFold. Log inflation magnitude."""
    from bci_framework.evaluation.preprocessing_evaluation import (
        load_physionet_mi,
        run_one_fold,
        get_fixed_pipeline_config,
        build_single_pipeline,
        compute_fold_metrics,
        RANDOM_STATE,
    )
    from sklearn.model_selection import StratifiedKFold, GroupKFold

    try:
        X, y, subject_ids, fs, ch_names, n_classes, capabilities, _ = load_physionet_mi(subjects=[1, 2, 3, 4, 5])
    except Exception as e:
        pytest.skip("Physionet MI load failed: %s" % e)

    config = get_fixed_pipeline_config("A")
    config["spatial_capabilities"] = capabilities

    # Trial-level StratifiedKFold (same subject can be in train and test → inflated)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    random_accs = []
    for train_idx, test_idx in skf.split(X, y):
        pipe = build_single_pipeline(config, fs, n_classes, ch_names)
        pipe.fit(X[train_idx], y[train_idx])
        y_pred = pipe.predict(X[test_idx])
        y_proba = getattr(pipe, "predict_proba", None)
        if y_proba is not None:
            y_proba = y_proba(X[test_idx])
        if y_proba is None and hasattr(pipe, "classifier") and hasattr(pipe.classifier, "predict_proba"):
            try:
                y_proba = pipe.classifier.predict_proba(pipe.transform(X[test_idx]))
            except Exception:
                y_proba = None
        m = compute_fold_metrics(y[test_idx], y_pred, y_proba, n_classes)
        random_accs.append(m["accuracy"])

    # GroupKFold (no subject in both train and test)
    gkf = GroupKFold(n_splits=5)
    group_accs = []
    for train_idx, test_idx in gkf.split(X, y, groups=subject_ids):
        out = run_one_fold(
            X, y, subject_ids, train_idx, test_idx, "A", fs, n_classes, ch_names, capabilities,
            random_state=RANDOM_STATE,
        )
        group_accs.append(out["metrics"]["accuracy"])

    mean_stratified = float(np.mean(random_accs))
    mean_group = float(np.mean(group_accs))
    inflation = (mean_stratified - mean_group) * 100
    # Log inflation magnitude (diagnostic)
    import logging
    logging.getLogger(__name__).info(
        "Performance inflation (diagnostic): StratifiedKFold mean acc = %.2f%%, GroupKFold = %.2f%%, inflation = %.2f pp",
        mean_stratified * 100, mean_group * 100, inflation,
    )
    assert mean_stratified > mean_group, (
        "StratifiedKFold (%.2f%%) should be > GroupKFold (%.2f%%). Inflation magnitude: %.2f pp." % (mean_stratified * 100, mean_group * 100, inflation)
    )
