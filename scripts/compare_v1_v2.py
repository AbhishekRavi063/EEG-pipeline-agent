#!/usr/bin/env python3
"""
Run and compare BCI AutoML v1-style vs v2-style config on the same synthetic data.

v1: spatial_filter.enabled = false (legacy reference: CAR in mandatory pipeline)
v2: spatial_filter.enabled = true, method = "car" (spatial filter layer; same effect as CAR)

Expectation: Same calibration accuracy and compatible pipeline names; v2 names include spatial tag.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from bci_framework.utils.config_loader import load_config, get_config
from bci_framework.pipelines import PipelineRegistry
from bci_framework.agent import PipelineSelectionAgent


def get_base_config():
    """Minimal config for synthetic run (no dataset load)."""
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
    # Disable advanced steps that need external data (GEDAI needs leadfield) for a clean comparison
    cfg["advanced_preprocessing"] = {"enabled": []}
    cfg.setdefault("pipelines", {"auto_generate": True, "max_combinations": 8, "explicit": []})
    cfg.setdefault("agent", {
        "calibration_trials": 30,
        "top_n_pipelines": 2,
        "prune_thresholds": {"min_accuracy": 0.2, "max_latency_ms": 500, "latency_budget_ms": 300, "max_stability_variance": 0.2},
    })
    cfg.setdefault("features", {})
    cfg.setdefault("classifiers", {})
    return cfg


def run_calibration(config, X, y, n_classes, channel_names):
    registry = PipelineRegistry(config)
    pipelines = registry.build_pipelines(fs=250.0, n_classes=n_classes, channel_names=channel_names)
    agent = PipelineSelectionAgent(config)
    metrics = agent.run_calibration(pipelines, X, y, n_classes=n_classes, max_parallel=1)
    kept = agent.prune(pipelines)
    agent.select_top_n(kept)
    best = agent.select_best()
    return {
        "n_pipelines": len(pipelines),
        "pipeline_names": [p.name for p in pipelines[:5]],
        "metrics": {k: (m.accuracy, m.kappa, m.latency_ms) for k, m in metrics.items()},
        "best_name": best.name if best else None,
        "best_accuracy": metrics[best.name].accuracy if best and best.name in metrics else None,
    }


def main():
    np.random.seed(42)
    n_trials, n_ch, n_samp = 60, 22, 750
    X = np.random.randn(n_trials, n_ch, n_samp).astype(np.float64) * 1e-5
    y = np.random.randint(0, 4, size=n_trials)
    channel_names = [f"Ch{i}" for i in range(n_ch)]
    n_classes = 4

    # --- v1 style: no spatial filter layer (legacy reference only)
    config_v1 = get_base_config()
    config_v1["spatial_filter"]["enabled"] = False
    config_v1["spatial_filter"].pop("method", None)

    # --- v2 style: spatial filter layer enabled (method = car)
    config_v2 = get_base_config()
    config_v2["spatial_filter"]["enabled"] = True
    config_v2["spatial_filter"]["method"] = "car"

    print("=" * 60)
    print("v1-style (spatial_filter.enabled = false, legacy reference)")
    print("=" * 60)
    result_v1 = run_calibration(config_v1, X, y, n_classes, channel_names)
    print(f"  Pipelines built: {result_v1['n_pipelines']}")
    print(f"  Sample names: {result_v1['pipeline_names'][:3]}")
    print(f"  Best: {result_v1['best_name']} (acc={result_v1['best_accuracy']:.3f})" if result_v1["best_accuracy"] is not None else "  Best: none")

    print()
    print("=" * 60)
    print("v2-style (spatial_filter.enabled = true, method = car)")
    print("=" * 60)
    result_v2 = run_calibration(config_v2, X, y, n_classes, channel_names)
    print(f"  Pipelines built: {result_v2['n_pipelines']}")
    print(f"  Sample names: {result_v2['pipeline_names'][:3]}")
    print(f"  Best: {result_v2['best_name']} (acc={result_v2['best_accuracy']:.3f})" if result_v2["best_accuracy"] is not None else "  Best: none")

    print()
    print("=" * 60)
    print("Comparison")
    print("=" * 60)
    print(f"  v1 best accuracy: {result_v1['best_accuracy']:.4f}" if result_v1["best_accuracy"] is not None else "  v1 best: N/A")
    print(f"  v2 best accuracy: {result_v2['best_accuracy']:.4f}" if result_v2["best_accuracy"] is not None else "  v2 best: N/A")
    # With same data/seed, CAR reference vs CAR spatial filter should give same numerical result
    if result_v1["best_accuracy"] is not None and result_v2["best_accuracy"] is not None:
        diff = abs(result_v1["best_accuracy"] - result_v2["best_accuracy"])
        print(f"  Difference (v2 - v1): {result_v2['best_accuracy'] - result_v1['best_accuracy']:.4f}")
        if diff < 0.01:
            print("  OK: v1 and v2 (CAR) give effectively same accuracy.")
        else:
            print("  Note: small differences possible due to pipeline order/count.")
    print("  Naming: v1 = baseline_<feature>_<clf>; v2 = baseline_<spatial>_<feature>_<clf> (e.g. baseline_car_csp_lda).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
