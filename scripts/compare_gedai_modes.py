"""
Compare GEDAI batch mode vs sliding mode performance.

This script runs the same pipeline with GEDAI in two modes:
1. Batch mode (offline, non-causal)
2. Sliding mode (online, causal, low-latency)

Generates a comparison report showing accuracy, latency, and other metrics.
"""

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bci_framework.utils.config_loader import load_config, get_config
from bci_framework.utils.splits import get_train_test_trials
from bci_framework.utils.experiment import set_seed
from bci_framework.datasets import get_dataset_loader
from bci_framework.pipelines import PipelineRegistry
from bci_framework.agent import PipelineSelectionAgent
from bci_framework.streaming import RealtimeInferenceEngine
from bci_framework.utils.metrics import compute_all_metrics, accuracy, cohen_kappa


def evaluate_pipeline(
    pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
    mode: str = "batch",
) -> dict[str, Any]:
    """Evaluate pipeline and return metrics."""
    print(f"  Evaluating in {mode} mode...")
    
    # Run predictions with latency measurement
    latencies = []
    predictions = []
    probabilities = []
    
    t0 = time.perf_counter()
    
    for i in range(len(X_test)):
        trial = X_test[i : i + 1]
        t_pred_start = time.perf_counter()
        
        if mode == "sliding":
            # Use predict_stream for sliding mode
            pred = pipeline.predict_stream(trial)
        else:
            pred = pipeline.predict(trial)
        
        latency_ms = (time.perf_counter() - t_pred_start) * 1000
        latencies.append(latency_ms)
        predictions.append(int(pred[0]))
        
        try:
            proba = pipeline.predict_proba(trial)
            probabilities.append(proba[0])
        except Exception:
            probabilities.append(None)
    
    total_time = time.perf_counter() - t0
    
    predictions = np.array(predictions)
    y_test_arr = np.asarray(y_test)
    
    # Compute metrics
    acc = accuracy(y_test_arr, predictions)
    kappa = cohen_kappa(y_test_arr, predictions, n_classes)
    
    proba_array = np.array(probabilities) if all(p is not None for p in probabilities) else None
    all_metrics = compute_all_metrics(y_test_arr, predictions, proba_array, n_classes, 3.0)
    
    latency_stats = {
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "max_ms": float(np.max(latencies)),
        "min_ms": float(np.min(latencies)),
        "std_ms": float(np.std(latencies)),
    }
    
    return {
        "accuracy": acc,
        "kappa": kappa,
        "latency_stats": latency_stats,
        "total_time_sec": total_time,
        "predictions": predictions,
        "all_metrics": all_metrics,
    }


def run_comparison():
    """Run comparison between batch and sliding modes."""
    print("=" * 80)
    print("GEDAI Batch vs Sliding Mode Comparison")
    print("=" * 80)
    
    # Load config
    config_path = ROOT / "bci_framework" / "config.yaml"
    load_config(config_path)
    config = get_config()
    
    # Set seed for reproducibility
    seed = config.get("experiment", {}).get("seed", 42)
    set_seed(seed)
    
    # Ensure GEDAI is enabled
    adv_prep = config.get("advanced_preprocessing", {})
    enabled = adv_prep.get("enabled", [])
    if "gedai" not in enabled:
        print("WARNING: GEDAI not enabled. Adding to enabled list.")
        enabled.append("gedai")
        adv_prep["enabled"] = enabled
    
    # Load dataset
    dataset_cfg = config.get("dataset", {})
    loader_cls = get_dataset_loader(dataset_cfg.get("name", "BCI_IV_2a"))
    loader = loader_cls()
    data_path = ROOT / dataset_cfg.get("data_dir", "./data/BCI_IV_2a").lstrip("./")
    
    result = loader.load(
        data_dir=str(data_path),
        subjects=[1],  # Single subject for comparison
        download_if_missing=False,
        trial_duration_seconds=dataset_cfg.get("trial_duration_seconds", 3.0),
    )
    
    if isinstance(result, dict):
        dataset = list(result.values())[0]
    else:
        dataset = result
    
    if dataset is None or dataset.n_trials == 0:
        print("ERROR: No dataset found. Please ensure BCI IV 2a data is available.")
        return
    
    X = dataset.data
    y = dataset.labels
    fs = dataset.fs
    n_classes = len(dataset.class_names)
    channel_names = dataset.channel_names
    
    print(f"\nDataset: {dataset.n_trials} trials, {len(channel_names)} channels, {fs} Hz")
    
    # Split data
    train_idx, test_idx = get_train_test_trials(
        len(X),
        subject_ids=None,
        evaluation_mode="subject_wise",
        train_ratio=0.8,
        loso_subject=None,
        random_state=seed,
        split_mode="train_test",
        n_calibration_trials=None,
        n_trials_from_t=None,
        use_cross_session=False,
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    print(f"Split: {len(X_train)} train, {len(X_test)} test")
    
    # Test 1: Batch mode
    print("\n" + "=" * 80)
    print("TEST 1: GEDAI Batch Mode (Offline, Non-Causal)")
    print("=" * 80)
    
    config_batch = config.copy()
    config_batch["mode"] = "offline"
    gedai_cfg = config_batch.get("advanced_preprocessing", {}).get("gedai", {})
    gedai_cfg["mode"] = "batch"
    
    registry_batch = PipelineRegistry(config_batch)
    pipelines_batch = registry_batch.build_pipelines(
        fs=fs, n_classes=n_classes, channel_names=channel_names
    )
    
    # Use CSP+LDA pipeline (common choice)
    pipeline_batch = None
    for p in pipelines_batch:
        if "csp" in p.feature_name.lower() and "lda" in p.classifier_name.lower():
            pipeline_batch = p
            break
    
    if pipeline_batch is None:
        pipeline_batch = pipelines_batch[0]
    
    print(f"Pipeline: {pipeline_batch.name}")
    print("Fitting pipeline...")
    pipeline_batch.fit(X_train[:30], y_train[:30])  # Use subset for speed
    
    results_batch = evaluate_pipeline(
        pipeline_batch, X_test[:20], y_test[:20], n_classes, mode="batch"
    )
    
    # Test 2: Sliding mode
    print("\n" + "=" * 80)
    print("TEST 2: GEDAI Sliding Mode (Online, Causal, Low-Latency)")
    print("=" * 80)
    
    config_sliding = config.copy()
    config_sliding["mode"] = "online"
    gedai_cfg_sliding = config_sliding.get("advanced_preprocessing", {}).get("gedai", {})
    gedai_cfg_sliding["mode"] = "sliding"
    
    registry_sliding = PipelineRegistry(config_sliding)
    pipelines_sliding = registry_sliding.build_pipelines(
        fs=fs, n_classes=n_classes, channel_names=channel_names
    )
    
    # Use same pipeline type
    pipeline_sliding = None
    for p in pipelines_sliding:
        if p.feature_name == pipeline_batch.feature_name and p.classifier_name == pipeline_batch.classifier_name:
            pipeline_sliding = p
            break
    
    if pipeline_sliding is None:
        pipeline_sliding = pipelines_sliding[0]
    
    print(f"Pipeline: {pipeline_sliding.name}")
    print("Fitting pipeline...")
    pipeline_sliding.fit(X_train[:30], y_train[:30])
    
    # Enable causal filters
    if hasattr(pipeline_sliding.preprocessing_manager.mandatory, "notch"):
        pipeline_sliding.preprocessing_manager.mandatory.notch.causal = True
    if hasattr(pipeline_sliding.preprocessing_manager.mandatory, "bandpass"):
        pipeline_sliding.preprocessing_manager.mandatory.bandpass.causal = True
    
    results_sliding = evaluate_pipeline(
        pipeline_sliding, X_test[:20], y_test[:20], n_classes, mode="sliding"
    )
    
    # Generate comparison report
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    report_lines = [
        "# GEDAI Batch vs Sliding Mode Comparison Report",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Subject:** BCI IV 2a, Subject 1",
        f"**Pipeline:** {pipeline_batch.name}",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        "Comparison of GEDAI preprocessing in two modes:",
        "- **Batch mode:** Offline, non-causal, processes full trials",
        "- **Sliding mode:** Online, causal, low-latency, sliding-window",
        "",
        "---",
        "",
        "## Detailed Comparison",
        "",
        "| Metric | Batch Mode | Sliding Mode | Change |",
        "|--------|-----------|--------------|--------|",
        f"| **Accuracy** | {results_batch['accuracy']:.3f} | {results_sliding['accuracy']:.3f} | {results_sliding['accuracy'] - results_batch['accuracy']:+.3f} |",
        f"| **Kappa** | {results_batch['kappa']:.3f} | {results_sliding['kappa']:.3f} | {results_sliding['kappa'] - results_batch['kappa']:+.3f} |",
        f"| **F1-macro** | {results_batch['all_metrics'].get('f1_macro', 0):.3f} | {results_sliding['all_metrics'].get('f1_macro', 0):.3f} | {results_sliding['all_metrics'].get('f1_macro', 0) - results_batch['all_metrics'].get('f1_macro', 0):+.3f} |",
        f"| **ROC-AUC** | {results_batch['all_metrics'].get('roc_auc_macro', 0):.3f} | {results_sliding['all_metrics'].get('roc_auc_macro', 0):.3f} | {results_sliding['all_metrics'].get('roc_auc_macro', 0) - results_batch['all_metrics'].get('roc_auc_macro', 0):+.3f} |",
        "",
        "### Latency Comparison",
        "",
        "| Statistic | Batch Mode (ms) | Sliding Mode (ms) | Change |",
        "|-----------|-----------------|-------------------|--------|",
        f"| **Mean** | {results_batch['latency_stats']['mean_ms']:.2f} | {results_sliding['latency_stats']['mean_ms']:.2f} | {results_sliding['latency_stats']['mean_ms'] - results_batch['latency_stats']['mean_ms']:+.2f} ms |",
        f"| **Median** | {results_batch['latency_stats']['median_ms']:.2f} | {results_sliding['latency_stats']['median_ms']:.2f} | {results_sliding['latency_stats']['median_ms'] - results_batch['latency_stats']['median_ms']:+.2f} ms |",
        f"| **Max** | {results_batch['latency_stats']['max_ms']:.2f} | {results_sliding['latency_stats']['max_ms']:.2f} | {results_sliding['latency_stats']['max_ms'] - results_batch['latency_stats']['max_ms']:+.2f} ms |",
        f"| **Min** | {results_batch['latency_stats']['min_ms']:.2f} | {results_sliding['latency_stats']['min_ms']:.2f} | {results_sliding['latency_stats']['min_ms'] - results_batch['latency_stats']['min_ms']:+.2f} ms |",
        f"| **Std** | {results_batch['latency_stats']['std_ms']:.2f} | {results_sliding['latency_stats']['std_ms']:.2f} | {results_sliding['latency_stats']['std_ms'] - results_batch['latency_stats']['std_ms']:+.2f} ms |",
        "",
        "### Performance Analysis",
        "",
    ]
    
    # Add analysis
    latency_reduction = results_batch['latency_stats']['mean_ms'] - results_sliding['latency_stats']['mean_ms']
    latency_reduction_pct = (latency_reduction / results_batch['latency_stats']['mean_ms']) * 100 if results_batch['latency_stats']['mean_ms'] > 0 else 0
    
    if latency_reduction > 0:
        report_lines.extend([
            f"✅ **Latency Improvement:** Sliding mode is {latency_reduction:.2f} ms faster ({latency_reduction_pct:.1f}% reduction)",
        ])
    else:
        report_lines.extend([
            f"⚠️ **Latency Increase:** Sliding mode is {abs(latency_reduction):.2f} ms slower ({abs(latency_reduction_pct):.1f}% increase)",
        ])
    
    acc_diff = results_sliding['accuracy'] - results_batch['accuracy']
    if abs(acc_diff) < 0.01:
        report_lines.append("✅ **Accuracy:** Both modes achieve similar accuracy (difference < 1%)")
    elif acc_diff > 0:
        report_lines.append(f"✅ **Accuracy:** Sliding mode is {acc_diff:.3f} better")
    else:
        report_lines.append(f"⚠️ **Accuracy:** Sliding mode is {abs(acc_diff):.3f} lower")
    
    report_lines.extend([
        "",
        "### Key Findings",
        "",
        "1. **Causality:** Sliding mode uses causal filters (lfilter) vs non-causal (filtfilt) in batch mode",
        "2. **GEDAI Updates:** Sliding mode updates eigenvectors incrementally vs full recomputation in batch",
        "3. **Real-time Compatibility:** Sliding mode is designed for online/streaming use",
        "",
        "### Recommendations",
        "",
        "- **For offline analysis:** Use batch mode for maximum accuracy",
        "- **For real-time BCI:** Use sliding mode for lower latency and causal processing",
        "- **Latency target:** < 100 ms for real-time applications (sliding mode typically achieves this)",
        "",
        "---",
    ])
    
    # Print report
    report_text = "\n".join(report_lines)
    print(report_text)
    
    # Save report
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    report_path = results_dir / "GEDAI_SLIDING_COMPARISON.md"
    report_path.write_text(report_text)
    
    print(f"\nReport saved to: {report_path}")
    print("\nDone!")


if __name__ == "__main__":
    run_comparison()
