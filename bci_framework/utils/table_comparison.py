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

import csv
import logging
from pathlib import Path
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


def _cohens_d_paired(v1: list[float], v2: list[float]) -> float | None:
    """Cohen's d for paired samples: mean(delta) / std(delta)."""
    import numpy as np
    v1_arr = np.asarray(v1, dtype=np.float64)
    v2_arr = np.asarray(v2, dtype=np.float64)
    d = v2_arr - v1_arr
    n = len(d)
    if n < 2:
        return None
    std_d = float(np.std(d))
    if std_d <= 0:
        return 0.0
    return float(np.mean(d) / std_d)


def _bootstrap_ci_paired(
    v1: list[float],
    v2: list[float],
    statistic: str = "mean_delta",
    n_boot: int = 2000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> tuple[float | None, float | None]:
    """Bootstrap 95% CI for paired difference (mean_delta). Returns (ci_low, ci_high)."""
    import numpy as np
    v1_arr = np.asarray(v1, dtype=np.float64)
    v2_arr = np.asarray(v2, dtype=np.float64)
    d = v2_arr - v1_arr
    n = len(d)
    if n < 2:
        return None, None
    rng = np.random.default_rng(random_state)
    boot_means = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means.append(float(np.mean(d[idx])))
    boot_means = np.array(boot_means)
    alpha = 1 - confidence
    low = float(np.percentile(boot_means, 100 * alpha / 2))
    high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return low, high


def compare_tables(
    table_1: list[dict[str, Any]],
    table_2: list[dict[str, Any]],
    metric: str = "accuracy",
    test: str = "ttest",
    name_1: str = "Pipeline_A",
    name_2: str = "Pipeline_B",
    n_perm: int = DEFAULT_N_PERM,
    include_effect_size: bool = False,
    include_bootstrap_ci: bool = False,
    n_bootstrap: int = 2000,
) -> dict[str, Any]:
    """
    Paired comparison of the same metric across two tables (same subjects).
    Tests operate on aligned subject IDs only.

    test: "ttest" (paired t-test), "wilcoxon" (Wilcoxon signed-rank),
    or "permutation" (paired permutation test: shuffle sign of (B-A); default, distribution-free).
    n_perm: number of permutations when test="permutation" (default 10000).
    Returns dict with: statistic, p_value, mean_1, mean_2, std_1, std_2,
    mean_delta, std_delta, n_subjects, significant (bool at alpha=0.05), test_used.
    If include_effect_size: add cohens_d. If include_bootstrap_ci: add bootstrap_ci_95_low, bootstrap_ci_95_high.
    """
    v1, v2, subject_ids = _align_tables(table_1, table_2, metric)
    n = len(v1)
    if n < 2:
        out = {
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
        if include_effect_size:
            out["cohens_d"] = None
        if include_bootstrap_ci:
            out["bootstrap_ci_95_low"] = out["bootstrap_ci_95_high"] = None
        return out
    import numpy as np
    v1_arr = np.asarray(v1, dtype=np.float64)
    v2_arr = np.asarray(v2, dtype=np.float64)
    mean_1 = float(np.mean(v1_arr))
    mean_2 = float(np.mean(v2_arr))
    std_1 = float(np.std(v1_arr)) if n > 1 else 0.0
    std_2 = float(np.std(v2_arr)) if n > 1 else 0.0
    deltas = v2_arr - v1_arr  # Table_2 - Table_1 (B - A)
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

    result: dict[str, Any] = {
        "metric": metric,
        "n_subjects": n,
        "subject_ids": subject_ids,
        "test_used": test_used,
        "name_1": name_1,
        "name_2": name_2,
        "mean_1": round(mean_1, 6),
        "mean_2": round(mean_2, 6),
        "std_1": round(std_1, 6),
        "std_2": round(std_2, 6),
        "mean_delta": round(mean_delta, 6),
        "std_delta": round(std_delta, 6),
        "statistic": (round(statistic, 6) if statistic is not None and statistic == statistic else None) if statistic is not None else None,
        "p_value": (round(p_value, 6) if p_value is not None and p_value == p_value else None) if p_value is not None else None,
        "significant": p_value is not None and p_value == p_value and p_value < 0.05,
    }
    if include_effect_size:
        result["cohens_d"] = round(_cohens_d_paired(v1, v2) or 0.0, 6)
    if include_bootstrap_ci:
        ci_lo, ci_hi = _bootstrap_ci_paired(v1, v2, n_boot=n_bootstrap, random_state=42)
        result["bootstrap_ci_95_low"] = round(ci_lo, 6) if ci_lo is not None else None
        result["bootstrap_ci_95_high"] = round(ci_hi, 6) if ci_hi is not None else None
    return result


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


def compare_tables_multi_metric_research(
    table_1: list[dict[str, Any]],
    table_2: list[dict[str, Any]],
    metrics: list[str] | None = None,
    name_1: str = "Pipeline_A",
    name_2: str = "Pipeline_B",
    n_perm: int = DEFAULT_N_PERM,
    n_bootstrap: int = 2000,
) -> dict[str, Any]:
    """
    Research-grade comparison: for each metric run permutation (default), Wilcoxon, t-test,
    plus effect size (Cohen's d) and bootstrap 95% CI. Operates on aligned subject IDs only.
    Returns dict: {metric: {permutation: {...}, wilcoxon: {...}, ttest: {...}, cohens_d, bootstrap_ci_95}}.
    """
    import numpy as np
    metrics = metrics or ["accuracy", "balanced_accuracy", "roc_auc_macro", "kappa", "f1_macro"]
    out: dict[str, Any] = {"name_1": name_1, "name_2": name_2, "metrics": {}}
    for m in metrics:
        v1, v2, subject_ids = _align_tables(table_1, table_2, m)
        n = len(v1)
        if n < 2:
            out["metrics"][m] = {"error": "Need at least 2 subjects", "n_subjects": n}
            continue
        perm = compare_tables(
            table_1, table_2, metric=m, test="permutation",
            name_1=name_1, name_2=name_2, n_perm=n_perm,
            include_effect_size=True, include_bootstrap_ci=True, n_bootstrap=n_bootstrap,
        )
        wilcoxon = compare_tables(
            table_1, table_2, metric=m, test="wilcoxon",
            name_1=name_1, name_2=name_2, n_perm=n_perm,
        )
        ttest = compare_tables(
            table_1, table_2, metric=m, test="ttest",
            name_1=name_1, name_2=name_2, n_perm=n_perm,
        )
        out["metrics"][m] = {
            "n_subjects": n,
            "subject_ids": subject_ids,
            "permutation": perm,
            "wilcoxon": wilcoxon,
            "ttest": ttest,
            "cohens_d": perm.get("cohens_d"),
            "bootstrap_ci_95": [perm.get("bootstrap_ci_95_low"), perm.get("bootstrap_ci_95_high")],
            "mean_1": perm.get("mean_1"),
            "mean_2": perm.get("mean_2"),
            "mean_delta": perm.get("mean_delta"),
            "p_value_permutation": perm.get("p_value"),
            "significant_05": perm.get("significant"),
        }
    return out


def export_pipeline_comparison_report(
    comparison: dict[str, Any],
    path_json: str | None = None,
    path_csv: str | None = None,
    path_latex: str | None = None,
) -> None:
    """
    Export pipeline comparison to publication-ready files.
    comparison: output of compare_tables_multi_metric_research (or compare_tables_multi_metric with permutation).
    """
    import csv
    from pathlib import Path
    if path_json:
        import json
        Path(path_json).parent.mkdir(parents=True, exist_ok=True)
        with open(path_json, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2)
        logger.info("Wrote pipeline comparison JSON: %s", path_json)
    if path_csv:
        _write_comparison_csv(comparison, path_csv)
    if path_latex:
        _write_comparison_latex(comparison, path_latex)


def _write_comparison_csv(comparison: dict[str, Any], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    metrics = comparison.get("metrics", comparison) if "metrics" in comparison else comparison
    if isinstance(metrics, dict) and metrics and isinstance(next(iter(metrics.values())), dict):
        rows = []
        for metric, res in metrics.items():
            if isinstance(res, dict) and "error" not in res:
                perm = res.get("permutation") or res
                rows.append({
                    "metric": metric,
                    "mean_A": perm.get("mean_1"),
                    "mean_B": perm.get("mean_2"),
                    "mean_delta": perm.get("mean_delta"),
                    "p_value_permutation": res.get("p_value_permutation") or perm.get("p_value"),
                    "cohens_d": res.get("cohens_d") or perm.get("cohens_d"),
                    "significant_05": res.get("significant_05") or perm.get("significant"),
                })
        if rows:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            logger.info("Wrote pipeline comparison CSV: %s", path)
        return
    # Fallback: single-metric style
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "mean_1", "mean_2", "mean_delta", "p_value", "significant"])
        for metric, res in (metrics if isinstance(metrics, dict) else {}).items():
            if isinstance(res, dict):
                w.writerow([
                    metric,
                    res.get("mean_1"),
                    res.get("mean_2"),
                    res.get("mean_delta"),
                    res.get("p_value"),
                    res.get("significant"),
                ])
    logger.info("Wrote pipeline comparison CSV: %s", path)


def _write_comparison_latex(comparison: dict[str, Any], path: str) -> None:
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    name_1 = comparison.get("name_1", "Pipeline_A")
    name_2 = comparison.get("name_2", "Pipeline_B")
    metrics = comparison.get("metrics", comparison) if "metrics" in comparison else comparison
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Pipeline comparison: %s vs %s (paired permutation test, aligned subjects).}" % (name_1, name_2),
        "\\label{tab:pipeline_comparison}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Metric & Mean (%s) & Mean (%s) & $\\Delta$ & $p$ (perm.) & Cohen's $d$ & Sig. \\\\" % (name_1, name_2),
        "\\midrule",
    ]
    if isinstance(metrics, dict):
        for metric, res in metrics.items():
            if isinstance(res, dict) and "error" not in res:
                m1 = res.get("mean_1") if res.get("mean_1") is not None else "---"
                m2 = res.get("mean_2") if res.get("mean_2") is not None else "---"
                delta = res.get("mean_delta") if res.get("mean_delta") is not None else "---"
                p = res.get("p_value_permutation") or (res.get("permutation") or {}).get("p_value")
                p = "%.4f" % p if p is not None else "---"
                d = res.get("cohens_d")
                d = "%.3f" % d if d is not None else "---"
                sig = "Yes" if res.get("significant_05") else "No"
                lines.append("%s & %s & %s & %s & %s & %s & %s \\\\" % (metric.replace("_", " "), m1, m2, delta, p, d, sig))
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote pipeline comparison LaTeX: %s", path)


def build_ablation_table(
    all_tables: dict[str, list],
    reference_name: str = "Riemann_MDM",
    metric: str = "accuracy",
    n_perm: int = DEFAULT_N_PERM,
) -> tuple[list[dict], str]:
    """
    Build ablation table: for each method compute mean ± std LOSO and p vs reference (permutation), Cohen's d.
    all_tables: {method_name: list[dict]} with subject-level rows (subject_id, accuracy, ...).
    Returns (rows for CSV, LaTeX table string).
    """
    import numpy as np
    from pathlib import Path

    ref_table = all_tables.get(reference_name)
    if not ref_table:
        reference_name = next(iter(all_tables), "")
        ref_table = all_tables.get(reference_name)

    order = [
        "CSP_LDA", "Riemann_MDM", "Tangent_LR", "FilterBankRiemann", "EEGNet",
        "EA_Tangent_LR",
        "AutoML", "AutoML_Ensemble",
    ]
    rows = []
    for name in order:
        table = all_tables.get(name)
        if not table:
            continue
        accs = []
        for r in table:
            a = r.get(metric)
            if a is not None and str(a).strip() and not str(a).startswith("#"):
                try:
                    accs.append(float(a))
                except (TypeError, ValueError):
                    pass
        if not accs:
            continue
        mean_acc = float(np.mean(accs))
        std_acc = float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0
        p_val = None
        cohens_d = None
        if ref_table and name != reference_name:
            comp = compare_tables(
                table, ref_table, metric=metric, test="permutation",
                name_1=name, name_2=reference_name, n_perm=n_perm,
                include_effect_size=True,
            )
            p_val = comp.get("p_value")
            cohens_d = comp.get("cohens_d")
        rows.append({
            "method": name,
            "mean_loso": mean_acc,
            "std_loso": std_acc,
            "p_vs_ref": p_val,
            "effect_size": cohens_d,
        })

    # LaTeX
    latex_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{LOSO ablation (reference: %s). Mean $\\pm$ std accuracy; $p$ from paired permutation; Cohen's $d$.}" % reference_name,
        "\\label{tab:ablation}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Method & Mean LOSO (\\%%) & Std & $p$ vs %s & Cohen's $d$ \\\\" % reference_name,
        "\\midrule",
    ]
    for r in rows:
        mean_pct = r["mean_loso"] * 100
        std_pct = r["std_loso"] * 100
        p_str = "%.4f" % r["p_vs_ref"] if r["p_vs_ref"] is not None else "---"
        d_str = "%.3f" % r["effect_size"] if r["effect_size"] is not None else "---"
        latex_lines.append(
            "%s & %.2f & %.2f & %s & %s \\\\"
            % (r["method"].replace("_", " "), mean_pct, std_pct, p_str, d_str)
        )
    latex_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    latex_str = "\n".join(latex_lines)
    return rows, latex_str
