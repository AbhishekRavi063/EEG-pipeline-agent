"""
Compare two subject-level tables (Table_1 vs Table_2) with paired statistical tests.

Returns p-value and summary for pipeline A vs B (professor's suggestion).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _align_tables(
    table_1: list[dict[str, Any]],
    table_2: list[dict[str, Any]],
    metric: str,
) -> tuple[list[float], list[float], list[Any]]:
    """
    Align two tables by subject_id and extract metric vectors.
    Returns (values_1, values_2, subject_ids) for subjects present in both.
    """
    by_id_1 = {r["subject_id"]: r for r in table_1 if "subject_id" in r}
    by_id_2 = {r["subject_id"]: r for r in table_2 if "subject_id" in r}
    common = sorted(set(by_id_1) & set(by_id_2))
    v1, v2, subject_ids = [], [], []
    for sid in common:
        a = by_id_1[sid].get(metric)
        b = by_id_2[sid].get(metric)
        if a is not None and b is not None:
            v1.append(float(a))
            v2.append(float(b))
            subject_ids.append(sid)
    return v1, v2, subject_ids


def compare_tables(
    table_1: list[dict[str, Any]],
    table_2: list[dict[str, Any]],
    metric: str = "accuracy",
    test: str = "ttest",
    name_1: str = "Pipeline_A",
    name_2: str = "Pipeline_B",
) -> dict[str, Any]:
    """
    Paired comparison of the same metric across two tables (same subjects).

    test: "ttest" (paired t-test) or "wilcoxon" (Wilcoxon signed-rank).
    Returns dict with: statistic, p_value, mean_1, mean_2, std_1, std_2,
    mean_delta, std_delta, n_subjects, significant (bool at alpha=0.05), test_used.
    """
    v1, v2, subject_ids = _align_tables(table_1, table_2, metric)
    n = len(v1)
    if n < 2:
        return {
            "metric": metric,
            "n_subjects": n,
            "test_used": test,
            "name_1": name_1,
            "name_2": name_2,
            "mean_1": sum(v1) / n if n else None,
            "mean_2": sum(v2) / n if n else None,
            "statistic": None,
            "p_value": None,
            "mean_delta": None,
            "std_delta": None,
            "significant": False,
            "error": "Need at least 2 subjects in common with valid metric",
        }
    import numpy as np
    v1_arr = np.asarray(v1, dtype=np.float64)
    v2_arr = np.asarray(v2, dtype=np.float64)
    mean_1 = float(np.mean(v1_arr))
    mean_2 = float(np.mean(v2_arr))
    std_1 = float(np.std(v1_arr)) if n > 1 else 0.0
    std_2 = float(np.std(v2_arr)) if n > 1 else 0.0
    deltas = v2_arr - v1_arr  # Table_2 - Table_1
    mean_delta = float(np.mean(deltas))
    std_delta = float(np.std(deltas)) if n > 1 else 0.0

    statistic = None
    p_value = None
    test_used = test
    if test == "ttest":
        try:
            from scipy import stats
            stat, p = stats.ttest_rel(v2_arr, v1_arr)
            statistic = float(stat)
            p_value = float(p)
        except Exception as e:
            logger.warning("Paired t-test failed: %s", e)
            test_used = "ttest_failed"
    elif test == "wilcoxon":
        try:
            from scipy import stats
            stat, p = stats.wilcoxon(v2_arr, v1_arr, alternative="two-sided")
            statistic = float(stat)
            p_value = float(p)
        except Exception as e:
            logger.warning("Wilcoxon failed: %s", e)
            test_used = "wilcoxon_failed"

    return {
        "metric": metric,
        "n_subjects": n,
        "test_used": test_used,
        "name_1": name_1,
        "name_2": name_2,
        "mean_1": round(mean_1, 6),
        "mean_2": round(mean_2, 6),
        "std_1": round(std_1, 6),
        "std_2": round(std_2, 6),
        "mean_delta": round(mean_delta, 6),
        "std_delta": round(std_delta, 6),
        "statistic": (round(statistic, 6) if statistic == statistic else None) if statistic is not None else None,
        "p_value": (round(p_value, 6) if p_value == p_value else None) if p_value is not None else None,
        "significant": p_value is not None and p_value == p_value and p_value < 0.05,
    }


def compare_tables_multi_metric(
    table_1: list[dict[str, Any]],
    table_2: list[dict[str, Any]],
    metrics: list[str] | None = None,
    test: str = "ttest",
    name_1: str = "Pipeline_A",
    name_2: str = "Pipeline_B",
) -> dict[str, dict[str, Any]]:
    """Run comparison for multiple metrics; returns {metric: comparison_result}."""
    metrics = metrics or ["accuracy", "balanced_accuracy", "roc_auc_macro", "kappa", "f1_macro"]
    results = {}
    for m in metrics:
        results[m] = compare_tables(table_1, table_2, metric=m, test=test, name_1=name_1, name_2=name_2)
    return results
