"""
Subject-level results table (Table_X): one row per subject, columns = outcome measures.

Used for multi-subject aggregation and A/B pipeline comparison (professor's suggestion).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Standard metrics to include in Table_X (same order for CSV)
TABLE_METRIC_COLUMNS = [
    "accuracy",
    "balanced_accuracy",
    "roc_auc_macro",
    "kappa",
    "f1_macro",
    "itr_bits_per_minute",
    "n_trials_test",
]


def build_subject_table(
    rows: list[dict[str, Any]],
    pipeline_name: str | None = None,
    metric_columns: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Build a canonical subject-level table from per-subject result rows.

    Each row should have at least subject_id and either:
      - test_metrics dict (from run_one_loso_fold extras), or
      - explicit metric keys (accuracy, balanced_accuracy, ...).

    Returns list of dicts with keys: subject_id, pipeline_name (optional), + metric_columns.
    """
    metric_columns = metric_columns or TABLE_METRIC_COLUMNS
    out = []
    for r in rows:
        subject_id = r.get("subject_id")
        if subject_id is None:
            continue
        metrics = r.get("test_metrics") or r
        row = {"subject_id": subject_id}
        if pipeline_name is not None:
            row["pipeline_name"] = pipeline_name
        for col in metric_columns:
            if col in metrics:
                row[col] = metrics[col]
            elif col == "n_trials_test":
                row[col] = r.get("n_trials_test") or metrics.get("n_trials_test")
            else:
                row[col] = None
        out.append(row)
    return out


def save_table_csv(
    table: list[dict[str, Any]],
    path: str | Path,
    metric_columns: list[str] | None = None,
) -> None:
    """Write table to CSV (subject_id, pipeline_name if present, then metric columns)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_columns = metric_columns or TABLE_METRIC_COLUMNS
    cols = ["subject_id"]
    if table and "pipeline_name" in table[0]:
        cols.append("pipeline_name")
    cols += [c for c in metric_columns if table and table[0].get(c) is not None]
    if not cols:
        cols = ["subject_id"] + metric_columns
    lines = [",".join(str(c) for c in cols)]
    for row in table:
        lines.append(",".join(str(row.get(c, "")) for c in cols))
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote subject table CSV: %s (%d rows)", path, len(table))


def save_table_json(
    table: list[dict[str, Any]],
    path: str | Path,
    meta: dict[str, Any] | None = None,
) -> None:
    """Write table to JSON; optional meta (dataset, pipeline_name, etc.)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"table": table}
    if meta:
        payload["meta"] = meta
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote subject table JSON: %s (%d rows)", path, len(table))


def load_table_json(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load table from JSON; returns (table, meta)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    table = data.get("table", data) if isinstance(data, dict) else data
    meta = data.get("meta", {}) if isinstance(data, dict) else {}
    return table, meta


def subject_table_summary(
    table: list[dict[str, Any]],
    metric_columns: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """
    Compute mean ± std across subjects for each metric (publication-ready summary).
    Returns {metric: {"mean": float, "std": float}}.
    """
    metric_columns = metric_columns or TABLE_METRIC_COLUMNS
    import numpy as np
    summary: dict[str, dict[str, float]] = {}
    for col in metric_columns:
        values = [row[col] for row in table if row.get(col) is not None]
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float64)
        summary[col] = {"mean": float(np.mean(arr)), "std": float(np.std(arr)) if len(arr) > 1 else 0.0}
    return summary


def save_subject_level_results(
    table: list[dict[str, Any]],
    results_dir: str | Path,
    experiment_id: str,
    filename: str = "subject_level_results.csv",
    metric_columns: list[str] | None = None,
    include_summary: bool = True,
) -> Path:
    """
    Save publication-ready subject-level table to results/<experiment_id>/<filename>.
    Default filename: subject_level_results.csv. Adds mean ± std summary at end of file.
    """
    path = Path(results_dir) / experiment_id / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_columns = metric_columns or TABLE_METRIC_COLUMNS
    cols = ["subject_id"]
    if table and table[0].get("pipeline_name") is not None:
        cols.append("pipeline_name")
    cols += [c for c in metric_columns if table and any(row.get(c) is not None for row in table)]
    if not cols:
        cols = ["subject_id"] + metric_columns
    lines = [",".join(str(c) for c in cols)]
    for row in table:
        lines.append(",".join(str(row.get(c, "")) for c in cols))
    if include_summary and table:
        summary = subject_table_summary(table, metric_columns)
        lines.append("")
        lines.append("# Summary (mean ± std across subjects)")
        for metric, s in summary.items():
            lines.append("# %s: %.6f ± %.6f" % (metric, s["mean"], s["std"]))
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote subject-level results: %s (%d rows + summary)", path, len(table))
    return path
