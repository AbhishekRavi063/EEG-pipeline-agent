"""Programmatic online streaming example with mandatory + optional preprocessing."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np

from bci_framework.utils.config_loader import load_config, get_config
from bci_framework.datasets import get_dataset_loader
from bci_framework.pipelines import PipelineRegistry
from bci_framework.agent import OnlinePipelineSelector


def run_online(subject: int, calibration_trials: int = 5, stream_trials: int = 10) -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "bci_framework" / "config.yaml"
    load_config(config_path)
    config = copy.deepcopy(get_config())
    config["mode"] = "online"

    dataset_cfg = config.get("dataset", {})
    loader_cls = get_dataset_loader(dataset_cfg.get("name", "BCI_IV_2a"))
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
    selector = OnlinePipelineSelector(pipelines=pipelines, config=config, n_classes=n_classes)

    print(f"Online selector initialised with {len(pipelines)} pipelines.")
    print(f"Advanced steps enabled: {pipelines[0].advanced_preprocessing if pipelines else []}")

    trials_added = 0
    idx = 0
    while trials_added < calibration_trials and idx < len(X):
        label = int(y[idx])
        if label >= 0:
            selector.add_trial(X[idx : idx + 1], label)
            trials_added += 1
        idx += 1

    if not selector.is_live_phase():
        selector.calibrate()

    selected = selector.selected_pipeline
    if selected is None:
        print("No pipeline selected (insufficient labeled trials).")
        return

    print(f"Selected pipeline: {selected.name} (advanced steps: {selected.advanced_preprocessing})")

    streamed = 0
    current = idx
    while streamed < stream_trials and current < len(X):
        trial = X[current : current + 1]
        label = int(y[current])
        pred, proba = selector.predict(trial)
        print(
            f"Trial {current}: predicted={int(pred[0])}"
            + (f", true={label}" if label >= 0 else ", true=unlabeled")
        )
        streamed += 1
        current += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Online streaming example")
    parser.add_argument("--subject", type=int, default=1, help="Subject ID to load (default: 1)")
    parser.add_argument("--cal-trials", type=int, default=5, help="Labeled trials for calibration")
    parser.add_argument("--stream-trials", type=int, default=10, help="Trials to stream after calibration")
    args = parser.parse_args()
    run_online(args.subject, calibration_trials=args.cal_trials, stream_trials=args.stream_trials)


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    main()
