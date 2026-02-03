"""Pipeline test with synthetic EEG (no dataset)."""

from pathlib import Path

import numpy as np
import pytest

from bci_framework.utils.config_loader import load_config, get_config
from bci_framework.utils.splits import get_train_test_trials
from bci_framework.pipelines import PipelineRegistry
from bci_framework.agent import PipelineSelectionAgent

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "bci_framework" / "config.yaml"


@pytest.fixture
def config():
    if _CONFIG_PATH.exists():
        load_config(_CONFIG_PATH)
        return get_config()
    return {
        "mode": "offline",
        "task": "motor_imagery",
        "pipelines": {"auto_generate": True, "max_combinations": 5, "explicit": []},
        "agent": {"calibration_trials": 20, "top_n_pipelines": 2, "prune_thresholds": {"min_accuracy": 0.3, "max_latency_ms": 500, "latency_budget_ms": 300}},
        "preprocessing": {
            "notch_freq": 50,
            "bandpass_low": 0.5,
            "bandpass_high": 40,
            "reference": "car",
            "adaptive_motor_band": False,
        },
        "advanced_preprocessing": {"enabled": []},
        "features": {},
        "classifiers": {},
    }


@pytest.fixture
def synthetic_eeg():
    np.random.seed(42)
    n_trials, n_ch, n_samp = 80, 22, 500
    X = np.random.randn(n_trials, n_ch, n_samp).astype(np.float64) * 1e-5
    y = np.random.randint(0, 4, size=n_trials)
    return X, y


def test_pipeline_registry_builds(config):
    registry = PipelineRegistry(config)
    channels = [f"Ch{i}" for i in range(22)]
    pipelines = registry.build_pipelines(fs=250.0, n_classes=4, channel_names=channels)
    assert len(pipelines) >= 1
    p = pipelines[0]
    assert hasattr(p, "fit") and hasattr(p, "predict")
    assert p.name


def test_trial_split_no_leakage(synthetic_eeg):
    X, y = synthetic_eeg
    train_idx, test_idx = get_train_test_trials(len(X), train_ratio=0.8, random_state=42)
    assert len(set(train_idx) & set(test_idx)) == 0
    assert len(train_idx) + len(test_idx) == len(X)


def test_calibration_and_prune(config, synthetic_eeg):
    X, y = synthetic_eeg
    train_idx, test_idx = get_train_test_trials(len(X), train_ratio=0.8, random_state=42)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    registry = PipelineRegistry(config)
    channels = [f"Ch{i}" for i in range(22)]
    pipelines = registry.build_pipelines(fs=250.0, n_classes=4, channel_names=channels)
    pipelines = pipelines[:3]
    agent = PipelineSelectionAgent(config)
    metrics = agent.run_calibration(pipelines, X_train, y_train, n_classes=4, max_parallel=0)
    assert len(metrics) == len(pipelines)
    kept = agent.prune(pipelines)
    best = agent.select_best()
    assert best is None or best in agent.get_top_pipelines()
