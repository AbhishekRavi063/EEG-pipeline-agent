"""
Stability-aware reporting: mean, std, coefficient of variation, fold variance per pipeline.

Used for transparent reporting; no exaggeration of composite scoring.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def compute_stability_metrics(
    table: list[dict[str, Any]],
    metric_columns: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """
    For each metric column: mean across subjects, std, coefficient of variation (std/mean),
    fold variance (variance = std^2). Transparent stability reporting.
    """
    from bci_framework.utils.subject_table import TABLE_METRIC_COLUMNS
    metric_columns = metric_columns or TABLE_METRIC_COLUMNS
    out: dict[str, dict[str, float]] = {}
    for col in metric_columns:
        values = [row[col] for row in table if row.get(col) is not None]
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float64)
        mean = float(np.mean(arr))
        std = float(np.std(arr)) if len(arr) > 1 else 0.0
        var = std ** 2
        cv = (std / mean) if mean != 0 else 0.0
        out[col] = {
            "mean": mean,
            "std": std,
            "coefficient_of_variation": float(cv),
            "fold_variance": var,
        }
    return out


def log_and_export_stability(
    table: list[dict[str, Any]],
    pipeline_name: str,
    results_dir: str | Path,
    experiment_id: str,
    metric_columns: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """
    Compute stability metrics, log them, and save to results/<experiment_id>/stability_<pipeline_name>.json.
    """
    from bci_framework.utils.subject_table import TABLE_METRIC_COLUMNS
    metric_columns = metric_columns or TABLE_METRIC_COLUMNS
    metrics = compute_stability_metrics(table, metric_columns)
    for col, s in metrics.items():
        logger.info(
            "[STABILITY] %s %s: mean=%.4f std=%.4f CV=%.4f var=%.6f",
            pipeline_name, col, s["mean"], s["std"], s["coefficient_of_variation"], s["fold_variance"],
        )
    path = Path(results_dir) / experiment_id / f"stability_{pipeline_name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Wrote stability metrics: %s", path)
    return metrics
