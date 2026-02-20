#!/usr/bin/env python3
"""
Minimal within-subject diagnostic: Subject 1 only, 5-fold CV, Bandpass 8-30 Hz + CSP (4) + LDA.

No AutoML, no domain adaptation, no LOSO.
If accuracy < 60% → pipeline or data window is broken.

Usage:
  PYTHONPATH=. python scripts/diagnose_within_subject_csp_lda.py
  PYTHONPATH=. python scripts/diagnose_within_subject_csp_lda.py --tmin 0.5 --tmax 2.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description="Within-subject CSP+LDA diagnostic (5-fold CV)")
    ap.add_argument("--dataset", default="BNCI2014_001")
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--tmin", type=float, default=0.0, help="MI window start (s) after cue")
    ap.add_argument("--tmax", type=float, default=4.0, help="MI window end (s) after cue")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from sklearn.model_selection import StratifiedKFold
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.preprocessing import StandardScaler

    from bci_framework.datasets.moabb_loader import MOABBDatasetLoader
    from bci_framework.preprocessing.manager import PreprocessingManager

    # Use MNE's CSP (multiclass); our bci_framework CSP is binary-only (classes 0 vs 1)
    try:
        from mne.decoding import CSP as MNE_CSP
        USE_MNE_CSP = True
    except ImportError:
        USE_MNE_CSP = False
        from bci_framework.features.csp import CSPFeatures

    np.random.seed(args.seed)

    # ---- Load data (same as LOSO path but single subject) ----
    print("=" * 60)
    print("MI WINDOW: tmin = %.2f s, tmax = %.2f s (relative to cue)" % (args.tmin, args.tmax))
    print("(For BNCI2014_001: cue at 2s; 0-4s = 2-6s in raw time = MI period)")
    print("=" * 60)

    loader = MOABBDatasetLoader(
        dataset_name=args.dataset,
        paradigm="motor_imagery",
        resample=250,
        tmin=args.tmin,
        tmax=args.tmax,
    )
    result = loader.load(subjects=[args.subject], download_if_missing=True)
    if isinstance(result, dict):
        ds = result.get(args.subject)
    else:
        ds = result
    if ds is None:
        print("ERROR: No data for subject", args.subject)
        return 1

    X = np.asarray(ds.data, dtype=np.float64)
    y = np.asarray(ds.labels, dtype=np.int64).ravel()
    # Drop unlabeled
    labeled = y >= 0
    X = X[labeled]
    y = y[labeled]
    fs = ds.fs
    ch_names = ds.channel_names
    n_classes = len(ds.class_names)
    n_trials, n_ch, n_samp = X.shape
    print("Loaded: subject=%d, trials=%d, channels=%d, samples_per_trial=%d, fs=%.0f, classes=%d" % (
        args.subject, n_trials, n_ch, n_samp, fs, n_classes))
    print("Class names:", getattr(ds, "class_names", list(range(n_classes))))

    # ---- Minimal preprocessing: notch 50 Hz, bandpass 8-30 Hz, CAR ----
    config = {
        "mode": "offline",
        "task": "motor_imagery",
        "preprocessing": {
            "notch_freq": 50,
            "notch_quality": 30,
            "bandpass_low": 8,
            "bandpass_high": 30,
            "bandpass_order": 5,
            "adaptive_motor_band": False,
        },
        "spatial_filter": {"enabled": True, "method": "car"},
        "advanced_preprocessing": {"enabled": []},
    }
    prep = PreprocessingManager(fs=fs, config=config, channel_names=ch_names)
    X_prep = prep.fit_transform(X)
    print("Preprocessed: bandpass 8-30 Hz, CAR. Shape:", X_prep.shape)

    # ---- 5-fold stratified CV: CSP (4 comps) + LDA ----
    # X_prep: (n_trials, n_channels, n_samples); MNE expects (n_trials, n_channels, n_times)
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    if USE_MNE_CSP:
        # MNE CSP: multiclass (n_components per pair or total); transform returns log-var features
        csp = MNE_CSP(n_components=4, reg=None, log=True)
    else:
        csp = CSPFeatures(fs=fs, n_components=4)
    lda = LinearDiscriminantAnalysis()
    scaler = StandardScaler()
    fold_accs = []
    all_y_true = []
    all_y_pred = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_prep, y)):
        X_tr, X_te = X_prep[train_idx], X_prep[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        # Fit CSP on train only (MNE = multiclass, our CSP = binary classes 0 vs 1 only)
        csp.fit(X_tr, y_tr)
        F_tr = csp.transform(X_tr)
        F_te = csp.transform(X_te)
        # Scale on train only
        F_tr_s = scaler.fit_transform(F_tr)
        F_te_s = scaler.transform(F_te)
        # LDA
        lda.fit(F_tr_s, y_tr)
        y_pred = lda.predict(F_te_s)
        acc = accuracy_score(y_te, y_pred)
        fold_accs.append(acc)
        all_y_true.extend(y_te.tolist())
        all_y_pred.extend(y_pred.tolist())
        print("  Fold %d: train=%d test=%d accuracy=%.4f" % (fold + 1, len(y_tr), len(y_te), acc))

    mean_acc = float(np.mean(fold_accs))
    std_acc = float(np.std(fold_accs)) if len(fold_accs) > 1 else 0.0
    print()
    print("=" * 60)
    print("RESULT: Within-subject %d-fold CV accuracy = %.4f ± %.4f" % (args.n_folds, mean_acc, std_acc))
    print("=" * 60)
    if mean_acc >= 0.60:
        print("OK: Pipeline reaches >= 60% (expected for within-subject MI).")
    else:
        print("WARNING: Accuracy < 60%%. Pipeline or MI window may be wrong.")
        print("  Try: --tmin 0.5 --tmax 2.5 (shorter MI window)")
        print("  Or:  --tmin 1.0 --tmax 4.0 (skip first second)")
    print()
    print("Confusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(all_y_true, all_y_pred)
    print(cm)
    print("(Classes: %s)" % (getattr(ds, "class_names", list(range(n_classes)))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
