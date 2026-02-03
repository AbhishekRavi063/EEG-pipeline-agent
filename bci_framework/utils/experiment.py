"""
Experiment tracking and reproducibility: seed control, experiment ID, optional MLflow.
"""

import logging
import os
import random
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_EXPERIMENT_ID: str | None = None
_MLFLOW_ACTIVE = False


def set_seed(seed: int = 42) -> None:
    """Set global seeds for numpy, random, and optional PyTorch."""
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_experiment_id() -> str:
    """Return current experiment ID (create one if not set)."""
    global _EXPERIMENT_ID
    if _EXPERIMENT_ID is None:
        _EXPERIMENT_ID = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    return _EXPERIMENT_ID


def set_experiment_id(experiment_id: str) -> None:
    """Set experiment ID (e.g. from config or CLI)."""
    global _EXPERIMENT_ID
    _EXPERIMENT_ID = experiment_id


def log_experiment_params(params: dict[str, Any]) -> None:
    """Log params to MLflow if active, else to logger."""
    if _MLFLOW_ACTIVE:
        try:
            import mlflow
            mlflow.log_params(params)
        except Exception as e:
            logger.debug("MLflow log_params failed: %s", e)
    logger.info("Experiment %s params: %s", get_experiment_id(), params)


def log_experiment_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Log metrics to MLflow if active, else to logger."""
    if _MLFLOW_ACTIVE:
        try:
            import mlflow
            mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            logger.debug("MLflow log_metrics failed: %s", e)
    if step is not None:
        logger.debug("Step %d metrics: %s", step, metrics)
    else:
        logger.debug("Metrics: %s", metrics)


def enable_mlflow(
    tracking_uri: str | Path | None = None,
    experiment_name: str = "bci_automl",
) -> None:
    """Enable MLflow tracking. Call before run."""
    global _MLFLOW_ACTIVE
    try:
        import mlflow
        if tracking_uri:
            mlflow.set_tracking_uri(str(tracking_uri))
        mlflow.set_experiment(experiment_name)
        _MLFLOW_ACTIVE = True
        logger.info("MLflow tracking enabled")
    except ImportError:
        logger.warning("MLflow not installed; install with pip install mlflow")
