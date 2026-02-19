"""
Compare two subject-level tables (Table_1 vs Table_2) with paired statistical tests.

Returns p-value and summary for pipeline A vs B (professor's suggestion).
Supports: paired t-test (parametric), Wilcoxon signed-rank (non-parametric),
paired permutation test (distribution-free), and DeLong test for comparing
AUC of two ROC curves directly (MLstatkit).
Also: MLstatkit Bootstrapping for metric CIs (bootstrap_metric_ci) and
AUC2OR for AUC-to-odds-ratio (auc_to_odds_ratio).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Default number of permutations for permutation test
DEFAULT_N_PERM = 10_000


def delong_test_auc(
    y_true: list[int] | "np.ndarray",
    prob_a: list[float] | "np.ndarray",
    prob_b: list[float] | "np.ndarray",
    n_classes: int | None = None,
    return_ci: bool = False,
    return_auc: bool = True,
) -> dict[str, Any]:
    """
    Compare AUC of two pipelines directly using DeLong's test (MLstatkit).
    Use when you have the same test set and prediction scores from pipeline A and B.

    For binary: pass y_true in {0, 1}, prob_a and prob_b as 1D (probability of class 1).
    For multi-class: pass y_true in [0..n_classes-1], prob_a and prob_b as (n_samples, n_classes).
    Then one DeLong test is run per one-vs-rest class; returned p_value is the minimum (conservative).

    Returns dict with: p_value, z, auc_a, auc_b, test_used ("delong"), and optionally
    per_class (list of {p_value, z, auc_a, auc_b} per class), ci_a, ci_b.
    If MLstatkit is not installed, returns {"error": "...", "test_used": "delong_unavailable"}.
    """
    try:
        from MLstatkit import Delong_test
    except ImportError:
        return {
            "test_used": "delong_unavailable",
            "error": "MLstatkit not installed. pip install MLstatkit",
            "p_value": None,
            "z": None,
            "auc_a": None,
            "auc_b": None,
        }
    import numpy as np
    y_true = np.asarray(y_true)
    prob_a = np.asarray(prob_a)
    prob_b = np.asarray(prob_b)
    if prob_a.ndim == 1:
        prob_a = prob_a.reshape(-1, 1)
    if prob_b.ndim == 1:
        prob_b = prob_b.reshape(-1, 1)
    if n_classes is None:
        n_classes = int(np.max(y_true)) + 1 if y_true.size else 2
    n_classes = min(n_classes, prob_a.shape[1], prob_b.shape[1])
    per_class = []
    p_values = []
    for c in range(n_classes):
        y_bin = (y_true == c).astype(np.int64)
        if np.sum(y_bin) < 2 or np.sum(1 - y_bin) < 2:
            continue
        pa_c = np.asarray(prob_a[:, c], dtype=np.float64).ravel()
        pb_c = np.asarray(prob_b[:, c], dtype=np.float64).ravel()
        out = Delong_test(
            y_bin, pa_c, pb_c,
            return_ci=False, return_auc=True, verbose=0,
        )
        # (z, p_value, auc_A, auc_B) when return_auc=True, return_ci=False
        z, p = float(out[0]), float(out[1])
        auc_a_c = float(out[2]) if len(out) > 2 else None
        auc_b_c = float(out[3]) if len(out) > 3 else None
        per_class.append({"p_value": p, "z": z, "auc_a": auc_a_c, "auc_b": auc_b_c})
        p_values.append(p)
    if not p_values:
        return {
            "test_used": "delong",
            "error": "Not enough positive/negative samples in any class for DeLong",
            "p_value": None,
            "z": None,
            "auc_a": None,
            "auc_b": None,
        }
    # Conservative: report minimum p-value across classes; z from the same class
    min_idx = int(np.argmin(p_values))
    p_value = p_values[min_idx]
    z = per_class[min_idx].get("z")
    # Macro AUC from sklearn-style (we don't recompute here; use first class or average if available)
    auc_a = per_class[min_idx].get("auc_a")
    auc_b = per_class[min_idx].get("auc_b")
    if all(p.get("auc_a") is not None for p in per_class) and all(p.get("auc_b") is not None for p in per_class):
        auc_a = float(np.mean([p["auc_a"] for p in per_class]))
        auc_b = float(np.mean([p["auc_b"] for p in per_class]))
    result = {
        "test_used": "delong",
        "p_value": float(p_value),
        "z": float(z) if z is not None else None,
        "auc_a": auc_a,
        "auc_b": auc_b,
        "per_class": per_class,
        "n_classes": n_classes,
    }
    return result


def bootstrap_metric_ci(
    y_true: list[int] | "np.ndarray",
    y_prob: list[float] | "np.ndarray",
    metric_str: str = "roc_auc",
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    threshold: float = 0.5,
    random_state: int = 0,
) -> dict[str, Any]:
    """
    Bootstrap confidence interval for a single metric (MLstatkit).
    Use when you have one set of predictions and want e.g. AUC [CI_low, CI_high].

    y_true: binary labels in {0, 1}. For multi-class, binarize per one-vs-rest and call per class.
    y_prob: predicted probability of the positive class (1D), or pass one column for multi-class.
    metric_str: 'roc_auc', 'pr_auc', 'f1', 'accuracy', 'precision', 'recall', 'average_precision'.
    Returns dict with: original_score, ci_lower, ci_upper, metric_str, error (if MLstatkit missing).
    """
    try:
        from MLstatkit import Bootstrapping
    except ImportError:
        return {
            "original_score": None,
            "ci_lower": None,
            "ci_upper": None,
            "metric_str": metric_str,
            "error": "MLstatkit not installed. pip install MLstatkit",
        }
    import numpy as np
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    if y_true.size != y_prob.size:
        return {
            "original_score": None,
            "ci_lower": None,
            "ci_upper": None,
            "metric_str": metric_str,
            "error": "y_true and y_prob length mismatch",
        }
    try:
        orig, ci_lo, ci_hi = Bootstrapping(
            y_true, y_prob,
            metric_str=metric_str,
            n_bootstraps=n_bootstraps,
            confidence_level=confidence_level,
            threshold=threshold,
            random_state=random_state,
        )
        return {
            "original_score": float(orig),
            "ci_lower": float(ci_lo),
            "ci_upper": float(ci_hi),
            "metric_str": metric_str,
            "error": None,
        }
    except Exception as e:
        logger.warning("Bootstrapping failed: %s", e)
        return {
            "original_score": None,
            "ci_lower": None,
            "ci_upper": None,
            "metric_str": metric_str,
            "error": str(e),
        }


def auc_to_odds_ratio(auc: float, return_all: bool = False) -> dict[str, Any] | float:
    """
    Convert AUC to odds ratio (MLstatkit AUC2OR). Useful for interpreting AUC in clinical/effect-size terms.

    auc: value in (0, 1).
    return_all: if True, return dict with z, d (Cohen's d), ln_or, OR; else return float OR only.
    If MLstatkit is not installed, returns 0.0 or dict with error.
    """
    try:
        from MLstatkit import AUC2OR
    except ImportError:
        if return_all:
            return {"OR": None, "z": None, "d": None, "ln_or": None, "error": "MLstatkit not installed"}
        return 0.0
    try:
        if return_all:
            z, d, ln_or, OR = AUC2OR(auc, return_all=True)
            return {"OR": float(OR), "z": float(z), "d": float(d), "ln_or": float(ln_or), "error": None}
        return float(AUC2OR(auc))
    except Exception as e:
        logger.warning("AUC2OR failed: %s", e)
        if return_all:
            return {"OR": None, "z": None, "d": None, "ln_or": None, "error": str(e)}
        return 0.0


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


def _permutation_test_paired(
    v1: list[float],
    v2: list[float],
    n_perm: int = DEFAULT_N_PERM,
    random_state: int | None = None,
) -> tuple[float, float]:
    """
    Paired permutation test: under H0 the sign of (v2_i - v1_i) is random.
    Statistic: mean(v2 - v1). Two-sided p-value (proportion of permutations
    with |permuted mean| >= |observed mean|).
    Returns (statistic, p_value).
    """
    import numpy as np
    v1_arr = np.asarray(v1, dtype=np.float64)
    v2_arr = np.asarray(v2, dtype=np.float64)
    diffs = v2_arr - v1_arr
    observed = float(np.mean(diffs))
    rng = np.random.default_rng(random_state)
    n = len(diffs)
    # (n_perm, n) array of random signs; multiply by diffs and mean over axis=1
    signs = rng.choice([-1, 1], size=(n_perm, n))
    perm_means = np.mean(diffs * signs, axis=1)
    n_extreme = int(np.sum(np.abs(perm_means) >= abs(observed)))
    p_value = (n_extreme + 1) / (n_perm + 1.0)
    return observed, p_value


def compare_tables(
    table_1: list[dict[str, Any]],
    table_2: list[dict[str, Any]],
    metric: str = "accuracy",
    test: str = "ttest",
    name_1: str = "Pipeline_A",
    name_2: str = "Pipeline_B",
    n_perm: int = DEFAULT_N_PERM,
) -> dict[str, Any]:
    """
    Paired comparison of the same metric across two tables (same subjects).

    test: "ttest" (paired t-test), "wilcoxon" (Wilcoxon signed-rank),
    or "permutation" (non-parametric permutation test; distribution-free).
    n_perm: number of permutations when test="permutation" (default 10000).
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
    elif test == "permutation":
        try:
            stat, p = _permutation_test_paired(v1, v2, n_perm=n_perm, random_state=42)
            statistic = float(stat)
            p_value = float(p)
            test_used = "permutation"
        except Exception as e:
            logger.warning("Permutation test failed: %s", e)
            test_used = "permutation_failed"

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
    n_perm: int = DEFAULT_N_PERM,
) -> dict[str, dict[str, Any]]:
    """Run comparison for multiple metrics; returns {metric: comparison_result}."""
    metrics = metrics or ["accuracy", "balanced_accuracy", "roc_auc_macro", "kappa", "f1_macro"]
    results = {}
    for m in metrics:
        results[m] = compare_tables(
            table_1, table_2, metric=m, test=test, name_1=name_1, name_2=name_2, n_perm=n_perm
        )
    return results
