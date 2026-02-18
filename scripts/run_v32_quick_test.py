#!/usr/bin/env python3
"""Quick test of v3.2 architecture: LOSO + CORAL pre-CSP + argmax CV selection. ~1–2 min."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from bci_framework.utils.config_loader import load_config, get_config
from bci_framework.datasets.moabb_loader import MOABBDatasetLoader
from bci_framework.pipelines import PipelineRegistry
from bci_framework.agent import PipelineSelectionAgent

def main():
    import copy
    config_path = ROOT / "bci_framework" / "config.yaml"
    load_config(config_path)
    config = copy.deepcopy(get_config())
    config["pipelines"] = {"auto_generate": True, "max_combinations": 6, "explicit": []}
    config["agent"] = {
        "calibration_trials": 40,
        "cv_folds": 3,
        "prune_thresholds": {"min_accuracy": 0.0, "max_latency_ms": 500},
    }
    config["transfer"] = {
        "enabled": True,
        "method": "coral",
        "target_unlabeled_fraction": 0.2,
        "regularization": 1e-3,
        "safe_mode": True,
    }
    config["spatial_filter"] = {"enabled": True, "method": "laplacian_auto", "auto_select": False}
    config["advanced_preprocessing"] = {"enabled": []}  # no GEDAI (avoids pygedai dependency)

    print("Loading MOABB (BNCI2014_001, subjects 1, 2)...")
    loader = MOABBDatasetLoader(dataset_name="BNCI2014_001", paradigm="motor_imagery", resample=250)
    result = loader.load(subjects=[1, 2], download_if_missing=True)
    config["spatial_capabilities"] = loader.capabilities

    ds1 = result[1]
    ds2 = result[2]
    X1 = np.asarray(ds1.data, dtype=np.float64)
    y1 = np.asarray(ds1.labels, dtype=np.int64).ravel()
    X2 = np.asarray(ds2.data, dtype=np.float64)
    y2 = np.asarray(ds2.labels, dtype=np.int64).ravel()
    mask1 = y1 >= 0
    mask2 = y2 >= 0
    X1, y1 = X1[mask1], y1[mask1]
    X2, y2 = X2[mask2], y2[mask2]

    X_train, y_train = X1, y1
    n_unlabeled = max(1, int(len(X2) * 0.2))
    X_target_cal = X2[:n_unlabeled]
    X_test = X2[n_unlabeled:]
    y_test = y2[n_unlabeled:]

    fs = ds1.fs
    n_classes = len(ds1.class_names)
    channel_names = ds1.channel_names

    print("[TRANSFER] Source trials:", len(X_train), "Target unlabeled:", len(X_target_cal), "Target test:", len(X_test))
    print("Spatial capabilities: Laplacian", getattr(loader.capabilities, "laplacian_supported", False))

    registry = PipelineRegistry(config)
    pipelines = registry.build_pipelines(fs=fs, n_classes=n_classes, channel_names=channel_names)
    print("Pipelines built:", len(pipelines), [p.name for p in pipelines])

    agent = PipelineSelectionAgent(config)
    n_cal = min(len(X_train), agent.calibration_trials)
    metrics = agent.run_calibration(
        pipelines, X_train[:n_cal], y_train[:n_cal],
        n_classes=n_classes, max_parallel=1,
        X_target_cal=X_target_cal,
    )
    kept = agent.prune(pipelines)
    agent.select_top_n(kept)
    try:
        best = agent.select_best(pipelines)
    except RuntimeError as e:
        best = None
        print("Select best error:", e)

    print("\n" + "=" * 60)
    print("V3.2 QUICK TEST RESULT")
    print("=" * 60)
    if best is not None:
        m = metrics.get(best.name)
        cv = getattr(m, "cv_accuracy", None)
        print("Best pipeline:", best.name)
        print("CV accuracy:", f"{cv:.4f}" if cv is not None else "N/A")
        y_pred = best.predict(X_test)
        test_acc = float(np.mean(y_pred == y_test))
        print("Test accuracy (holdout subject 2):", f"{test_acc:.4f}")
    else:
        print("Best pipeline: None")
    print("=" * 60)
    return 0 if best is not None else 1

if __name__ == "__main__":
    sys.exit(main())
