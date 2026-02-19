"""
Quick test of MLstatkit integration: DeLong, Bootstrapping (CI), AUC2OR.

Run (from repo root, with venv activated and pip install -r requirements.txt):
  PYTHONPATH=. python scripts/test_mlstatkit_impl.py

Without MLstatkit you get clear errors/N/A; our permutation test still runs.
With MLstatkit you get real p-values, CIs, and OR.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import importlib.util
spec = importlib.util.spec_from_file_location(
    "table_comparison",
    ROOT / "bci_framework" / "utils" / "table_comparison.py",
)
tc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tc)
delong_test_auc = tc.delong_test_auc
bootstrap_metric_ci = tc.bootstrap_metric_ci
auc_to_odds_ratio = tc.auc_to_odds_ratio
compare_tables = tc.compare_tables

def main():
    np.random.seed(42)
    print("=" * 60)
    print("1. DeLong test (compare AUC of two pipelines on same test set)")
    print("=" * 60)
    # Simulate: same 20 test trials, scores from pipeline A and B
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 2)  # 20 trials, binary
    prob_a = np.clip(np.random.rand(20) + 0.1 * (y_true - 0.5), 0.05, 0.95)
    prob_b = np.clip(prob_a + np.random.randn(20) * 0.15, 0.05, 0.95)  # B slightly different
    r = delong_test_auc(y_true, prob_a, prob_b, n_classes=2)
    print("  test_used:", r.get("test_used"))
    print("  p_value:  ", r.get("p_value"))
    print("  auc_a:    ", r.get("auc_a"))
    print("  auc_b:    ", r.get("auc_b"))
    if r.get("error"):
        print("  error:    ", r["error"])
    print()

    print("2. Bootstrap CI (confidence interval for one metric)")
    print("=" * 60)
    # Same binary predictions, get AUC with 95% CI
    r_ci = bootstrap_metric_ci(y_true, prob_a, metric_str="roc_auc", n_bootstraps=500, confidence_level=0.95)
    print("  metric:        ", r_ci.get("metric_str"))
    print("  original_score:", r_ci.get("original_score"))
    lo, hi = r_ci.get("ci_lower"), r_ci.get("ci_upper")
    print("  CI:            [%s, %s]" % (lo if lo is not None else "N/A", hi if hi is not None else "N/A"))
    if r_ci.get("error"):
        print("  error:         ", r_ci["error"])
    # F1 as well
    r_f1 = bootstrap_metric_ci(y_true, prob_a, metric_str="f1", threshold=0.5, n_bootstraps=500)
    print("  F1 original:  ", r_f1.get("original_score"))
    lo, hi = r_f1.get("ci_lower"), r_f1.get("ci_upper")
    print("  F1 CI:         [%s, %s]" % (lo if lo is not None else "N/A", hi if hi is not None else "N/A"))
    print()

    print("3. AUC to odds ratio (interpret AUC as effect size)")
    print("=" * 60)
    auc_val = 0.72
    or_only = auc_to_odds_ratio(auc_val)
    or_all = auc_to_odds_ratio(auc_val, return_all=True)
    print("  AUC = %.2f" % auc_val)
    print("  OR (odds ratio) = %.4f" % or_only)
    if isinstance(or_all, dict):
        print("  return_all: z=%.4f, d=%.4f, ln_or=%.4f, OR=%.4f" % (
            or_all.get("z") or 0, or_all.get("d") or 0, or_all.get("ln_or") or 0, or_all.get("OR") or 0
        ))
    print()

    print("4. Subject-level comparison (our permutation test, not MLstatkit)")
    print("=" * 60)
    table_a = [{"subject_id": i, "accuracy": 0.6 + i * 0.02, "roc_auc_macro": 0.65 + i * 0.02} for i in range(1, 6)]
    table_b = [{"subject_id": i, "accuracy": 0.62 + i * 0.02, "roc_auc_macro": 0.67 + i * 0.02} for i in range(1, 6)]
    comp = compare_tables(table_a, table_b, metric="accuracy", test="permutation")
    print("  metric:     ", comp.get("metric"))
    print("  mean_A:     ", comp.get("mean_1"))
    print("  mean_B:     ", comp.get("mean_2"))
    print("  p_value:    ", comp.get("p_value"))
    print("  test_used:  ", comp.get("test_used"))
    print("  significant:", comp.get("significant"))
    print()
    print("Done. All MLstatkit helpers (DeLong, Bootstrap, AUC2OR) and our permutation test ran successfully.")

if __name__ == "__main__":
    main()
