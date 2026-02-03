"""Programmatic offline calibration example using the mandatory preprocessing pipeline."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np

from bci_framework.utils.config_loader import load_config, get_config
from bci_framework.datasets import get_dataset_loader
from bci_framework.pipelines import PipelineRegistry
from bci_framework.agent import PipelineSelectionAgent
from bci_framework.utils.splits import get_train_test_trials


def run_offline(subject: int) -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "bci_framework" / "config.yaml"
    load_config(config_path)
    config = copy.deepcopy(get_config())
    config["mode"] = "offline"

    dataset_cfg = config.get("dataset", {})
    ds_name = dataset_cfg.get("name", "BCI_IV_2a")
    loader_cls = get_dataset_loader(ds_name)
    loader = loader_cls()

    data_dir = root / dataset_cfg.get("data_dir", "./data/BCI_IV_2a").lstrip("./")
    result = loader.load(
        data_dir=str(data_dir),
        subjects=[subject],
        download_if_missing=dataset_cfg.get("download_if_missing", True),
        trial_duration_seconds=dataset_cfg.get("trial_duration_seconds", 3.0),
    )
    if isinstance(result, dict):
        dataset = result.get(subject) or next(iter(result.values()))
    else:
        dataset = result
    if dataset is None or getattr(dataset, "n_trials", 0) == 0:
        raise RuntimeError("Dataset is empty; download BCI IV 2a files before running this example.")

    X = dataset.data
    y = dataset.labels
    fs = dataset.fs
    n_classes = len(dataset.class_names)

    registry = PipelineRegistry(config)
    pipelines = registry.build_pipelines(fs=fs, n_classes=n_classes, channel_names=dataset.channel_names)
    print(f"Constructed {len(pipelines)} pipelines (advanced steps = {pipelines[0].advanced_preprocessing if pipelines else []}).")

    idx_train, _ = get_train_test_trials(len(X), train_ratio=0.8, random_state=42)
    X_train = X[idx_train]
    y_train = y[idx_train]
    labeled = y_train >= 0
    X_cal = X_train[labeled]
    y_cal = y_train[labeled]
    if len(X_cal) == 0:
        raise RuntimeError("No labeled trials available for calibration.")

    agent = PipelineSelectionAgent(config)
    n_cal = min(agent.calibration_trials, len(X_cal))
    metrics = agent.run_calibration(pipelines, X_cal[:n_cal], y_cal[:n_cal], n_classes=n_classes, max_parallel=0)
    kept = agent.prune(pipelines)
    agent.select_top_n(kept)
    best = agent.select_best()

    if best is None:
        print("No pipeline met the thresholds.")
        return

    m = metrics.get(best.name)
    acc = m.accuracy if m else float("nan")
    print(f"Best pipeline: {best.name}")
    print(f"  Advanced steps: {best.advanced_preprocessing}")
    print(f"  Calibration accuracy: {acc:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline calibration example")
    parser.add_argument("--subject", type=int, default=1, help="Subject ID to load (default: 1)")
    args = parser.parse_args()
    run_offline(args.subject)


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    main()
