"""
Minimal example: Real-time sliding-window EEG streaming replay.

This script demonstrates how to use the sliding-window streaming infrastructure
to replay recorded EEG data in real-time with continuous inference.

Usage:
    PYTHONPATH=. python examples/realtime_replay.py
"""

import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bci_framework.streaming import CircularRingBuffer, RealtimeInferenceEngine
from bci_framework.pipelines import PipelineRegistry
from bci_framework.utils.config_loader import load_config, get_config
from bci_framework.datasets import get_dataset_loader


def main():
    """Run minimal real-time replay example."""
    print("=" * 60)
    print("Real-Time Sliding-Window EEG Streaming Example")
    print("=" * 60)
    
    # Load config
    config_path = ROOT / "bci_framework" / "config.yaml"
    load_config(config_path)
    config = get_config()
    
    # Enable sliding mode
    config["mode"] = "online"
    config["dataset"]["streaming_mode"] = "sliding"
    config["streaming"]["mode"] = "sliding"
    
    # Load dataset (single subject for demo)
    dataset_cfg = config.get("dataset", {})
    loader_cls = get_dataset_loader(dataset_cfg.get("name", "BCI_IV_2a"))
    loader = loader_cls()
    data_path = ROOT / dataset_cfg.get("data_dir", "./data/BCI_IV_2a").lstrip("./")
    
    result = loader.load(
        data_dir=str(data_path),
        subjects=[1],  # Single subject for demo
        download_if_missing=False,
        trial_duration_seconds=dataset_cfg.get("trial_duration_seconds", 3.0),
    )
    
    if isinstance(result, dict):
        dataset = list(result.values())[0]
    else:
        dataset = result
    
    if dataset is None or dataset.n_trials == 0:
        print("ERROR: No dataset found. Please ensure BCI IV 2a data is available.")
        print(f"Expected location: {data_path}")
        return
    
    X = dataset.data
    y = dataset.labels
    fs = dataset.fs
    n_classes = len(dataset.class_names)
    channel_names = dataset.channel_names
    
    print(f"\nDataset loaded: {dataset.n_trials} trials, {len(channel_names)} channels, {fs} Hz")
    
    # Split data (simple 80/20)
    n_train = int(0.8 * len(X))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    
    print(f"Split: {len(X_train)} train, {len(X_test)} test")
    
    # Build and fit a simple pipeline
    print("\nBuilding pipeline...")
    registry = PipelineRegistry(config)
    pipelines = registry.build_pipelines(
        fs=fs,
        n_classes=n_classes,
        channel_names=channel_names,
    )
    
    # Use first pipeline (CSP+LDA typically)
    pipeline = pipelines[0]
    print(f"Fitting pipeline: {pipeline.name}")
    pipeline.fit(X_train[:20], y_train[:20])  # Use subset for speed
    print("Pipeline fitted ✓")
    
    # Configure streaming parameters
    stream_cfg = config.get("streaming", {})
    window_size_sec = stream_cfg.get("window_size_sec", 1.5)
    update_interval_sec = stream_cfg.get("update_interval_sec", 0.1)
    buffer_length_sec = stream_cfg.get("buffer_length_sec", 10.0)
    
    print(f"\nStreaming configuration:")
    print(f"  Window size: {window_size_sec}s")
    print(f"  Update interval: {update_interval_sec}s ({1/update_interval_sec:.1f} Hz)")
    print(f"  Buffer length: {buffer_length_sec}s")
    
    # Initialize inference engine
    inference_engine = RealtimeInferenceEngine(
        pipeline=pipeline,
        fs=fs,
        window_size_sec=window_size_sec,
        update_interval_sec=update_interval_sec,
        buffer_length_sec=buffer_length_sec,
        n_channels=len(channel_names),
    )
    
    # Stream test trials sample-by-sample
    print(f"\nStreaming {len(X_test)} test trials...")
    print("-" * 60)
    
    n_samples_per_trial = int(3.0 * fs)  # 3 seconds per trial
    sample_interval = 1.0 / fs  # Time between samples
    
    n_predictions = 0
    correct_predictions = 0
    
    for trial_idx in range(min(5, len(X_test))):  # Demo: first 5 trials
        trial_data = X_test[trial_idx : trial_idx + 1]  # (1, n_channels, n_samples)
        trial_label = y_test[trial_idx]
        
        # Convert to (n_channels, n_samples)
        trial_channel_data = trial_data[0]
        
        print(f"\nTrial {trial_idx + 1}: true_label={dataset.class_names[trial_label]}")
        
        # Stream samples one by one
        for sample_idx in range(n_samples_per_trial):
            if sample_idx >= trial_channel_data.shape[1]:
                break
            
            # Push single sample
            sample = trial_channel_data[:, sample_idx]
            inference_engine.push_samples(sample)
            
            # Check if inference should run
            result = inference_engine.update()
            if result is not None:
                prediction, latency_ms = result
                pred_label = int(prediction[0])
                n_predictions += 1
                
                if pred_label == trial_label:
                    correct_predictions += 1
                
                if n_predictions % 10 == 0:  # Print every 10th prediction
                    print(
                        f"  Sample {sample_idx}: pred={dataset.class_names[pred_label]}, "
                        f"latency={latency_ms:.2f}ms"
                    )
            
            # Real-time pacing
            time.sleep(sample_interval)
        
        # Get latency stats
        stats = inference_engine.get_latency_stats()
        print(f"  Trial complete: {stats['n_predictions']} predictions")
    
    # Final statistics
    print("\n" + "=" * 60)
    print("Streaming Complete")
    print("=" * 60)
    
    stats = inference_engine.get_latency_stats()
    print(f"\nLatency Statistics:")
    print(f"  Mean: {stats['mean_ms']:.2f} ms")
    print(f"  Median: {stats['median_ms']:.2f} ms")
    print(f"  Max: {stats['max_ms']:.2f} ms")
    print(f"  Min: {stats['min_ms']:.2f} ms")
    print(f"  Total predictions: {stats['n_predictions']}")
    
    if n_predictions > 0:
        accuracy = correct_predictions / n_predictions
        print(f"\nPrediction Accuracy: {accuracy:.1%} ({correct_predictions}/{n_predictions})")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
