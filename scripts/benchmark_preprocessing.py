#!/usr/bin/env python3
"""
Standardized preprocessing benchmark script.

Compares baseline, ICA, ASR, Wavelet, and GEDAI (with/without real leadfield)
on BCI IV 2a dataset. Generates metrics, plots, and JSON reports.

Usage:
    python scripts/benchmark_preprocessing.py --subject 1 --output results/benchmark/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from bci_framework.datasets import BCICompetitionIV2aLoader
from bci_framework.pipelines import Pipeline, PipelineRegistry
from bci_framework.utils.config_loader import load_config
from bci_framework.utils.metrics import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


PREPROCESSING_CONFIGS = {
    "baseline": {
        "enabled": ["signal_quality"],
        "description": "Baseline: notch + bandpass + CAR + signal quality",
    },
    "ica": {
        "enabled": ["signal_quality", "ica"],
        "description": "ICA artifact removal",
    },
    "asr": {
        "enabled": ["signal_quality", "asr"],
        "description": "ASR adaptive artifact removal",
    },
    "wavelet": {
        "enabled": ["signal_quality", "wavelet"],
        "description": "Wavelet denoising",
    },
    "gedai_identity": {
        "enabled": ["signal_quality", "gedai"],
        "gedai": {
            "use_identity_if_missing": True,
            "require_real_leadfield": False,
        },
        "description": "GEDAI with identity leadfield (dev mode)",
    },
    "gedai_real": {
        "enabled": ["signal_quality", "gedai"],
        "gedai": {
            "leadfield_path": str(ROOT / "data" / "leadfield_bci_iv_2a.npy"),
            "use_identity_if_missing": False,
            "require_real_leadfield": True,
        },
        "description": "GEDAI with physics-correct leadfield",
    },
}


def benchmark_preprocessing(
    subject_id: int,
    output_dir: Path,
    config_path: Path | None = None,
    n_trials_calibration: int = 50,
) -> dict[str, Any]:
    """
    Benchmark all preprocessing methods.

    Parameters
    ----------
    subject_id : int
        BCI IV 2a subject ID (1-9)
    output_dir : Path
        Directory to save results
    config_path : Path | None
        Base config file (default: bci_framework/config.yaml)
    n_trials_calibration : int
        Number of trials for calibration

    Returns
    -------
    results : dict
        Benchmark results with metrics per preprocessing method
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    logger.info("Loading BCI IV 2a subject %d...", subject_id)
    loader = BCICompetitionIV2aLoader()
    dataset = loader.load(
        subjects=[subject_id],
        trial_duration_seconds=3.0,
        download_if_missing=True,
    )
    if dataset is None:
        raise ValueError(f"Failed to load subject {subject_id}")

    X = dataset.data
    y = dataset.labels
    fs = dataset.fs
    n_classes = len(set(y[y >= 0]))  # Exclude unlabeled (-1)

    # Split train/test
    n_train = int(0.8 * len(X))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    # Limit calibration trials
    X_cal = X_train[:n_trials_calibration]
    y_cal = y_train[:n_trials_calibration]

    logger.info("Dataset: %d trials (train: %d, test: %d, calibration: %d)", len(X), n_train, len(X_test), len(X_cal))

    # Load base config
    base_config = load_config(config_path) if config_path else load_config()

    results: dict[str, Any] = {
        "subject_id": subject_id,
        "n_trials_total": len(X),
        "n_trials_calibration": len(X_cal),
        "n_trials_test": len(X_test),
        "fs": float(fs),
        "n_classes": n_classes,
        "preprocessing_methods": {},
    }

    # Benchmark each preprocessing method
    for method_name, method_config in PREPROCESSING_CONFIGS.items():
        logger.info("=" * 60)
        logger.info("Benchmarking: %s", method_name)
        logger.info("Description: %s", method_config["description"])

        try:
            # Create config with this preprocessing
            test_config = base_config.copy()
            test_config["advanced_preprocessing"] = {
                "enabled": method_config["enabled"],
            }
            # Merge gedai config if present
            if "gedai" in method_config:
                test_config["advanced_preprocessing"]["gedai"] = method_config["gedai"]

            # Build pipeline (use CSP+LDA as standard)
            registry = PipelineRegistry(config=test_config)
            pipelines = registry.build_pipelines(
                fs=fs,
                n_classes=n_classes,
                channel_names=dataset.channel_names,
            )

            # Find CSP+LDA pipeline
            csp_lda = None
            for pipe in pipelines:
                if pipe.feature_name == "csp" and pipe.classifier_name == "lda":
                    csp_lda = pipe
                    break

            if csp_lda is None:
                logger.warning("CSP+LDA pipeline not found for %s, skipping", method_name)
                continue

            # Fit on calibration
            start_time = time.perf_counter()
            csp_lda.fit(X_cal, y_cal)
            fit_time = time.perf_counter() - start_time

            # Predict on test set
            start_time = time.perf_counter()
            y_pred = csp_lda.predict(X_test)
            predict_time = time.perf_counter() - start_time

            # Compute metrics
            metrics = compute_metrics(y_test[y_test >= 0], y_pred[y_test >= 0], n_classes=n_classes)

            # Latency per trial
            latency_ms = (predict_time / len(X_test)) * 1000

            # Store results
            method_results = {
                "description": method_config["description"],
                "accuracy": float(metrics["accuracy"]),
                "kappa": float(metrics["kappa"]),
                "f1_macro": float(metrics.get("f1_macro", 0.0)),
                "roc_auc_macro": float(metrics.get("roc_auc_macro", 0.0)),
                "latency_ms": latency_ms,
                "fit_time_sec": fit_time,
                "predict_time_sec": predict_time,
            }

            results["preprocessing_methods"][method_name] = method_results

            logger.info(
                "Results: acc=%.3f, kappa=%.3f, latency=%.2f ms",
                method_results["accuracy"],
                method_results["kappa"],
                method_results["latency_ms"],
            )

        except Exception as e:
            logger.error("Failed to benchmark %s: %s", method_name, e, exc_info=True)
            results["preprocessing_methods"][method_name] = {"error": str(e)}

    # Save results
    results_file = output_dir / f"benchmark_subject_{subject_id}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Benchmark results saved to: %s", results_file)

    # Print summary table
    print("\n" + "=" * 80)
    print("PREPROCESSING BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Method':<20} {'Accuracy':<10} {'Kappa':<10} {'Latency (ms)':<15} {'F1':<10}")
    print("-" * 80)

    for method_name, method_results in results["preprocessing_methods"].items():
        if "error" in method_results:
            print(f"{method_name:<20} {'ERROR':<10}")
            continue
        print(
            f"{method_name:<20} "
            f"{method_results['accuracy']:<10.3f} "
            f"{method_results['kappa']:<10.3f} "
            f"{method_results['latency_ms']:<15.2f} "
            f"{method_results['f1_macro']:<10.3f}"
        )

    print("=" * 80)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark preprocessing methods")
    parser.add_argument("--subject", "-s", type=int, default=1, help="Subject ID (1-9)")
    parser.add_argument("--output", "-o", type=str, default="./results/benchmark", help="Output directory")
    parser.add_argument("--config", "-c", type=str, default=None, help="Config file path")
    parser.add_argument("--trials", "-t", type=int, default=50, help="Calibration trials")
    args = parser.parse_args()

    output_path = Path(args.output)
    config_path = Path(args.config) if args.config else None

    results = benchmark_preprocessing(
        subject_id=args.subject,
        output_dir=output_path,
        config_path=config_path,
        n_trials_calibration=args.trials,
    )
