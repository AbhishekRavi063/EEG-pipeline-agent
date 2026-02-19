# Why we use these statistical tools (and what they’re for)

## How to run the check

From the repo root with your environment active (e.g. `source venv/bin/activate` and `pip install -r requirements.txt`):

```bash
PYTHONPATH=. python scripts/test_mlstatkit_impl.py
```

If MLstatkit is not installed, DeLong / Bootstrap / AUC2OR will report an error and N/A; the **permutation test** (our own) still runs. With MLstatkit installed you get real numbers for all.

---

## 1. DeLong test (`delong_test_auc`) — **what for**

- **What it does:** Compares the **AUC of two ROC curves** when both pipelines are evaluated on the **same test set** and you have **prediction scores** (probabilities) from both.
- **Why we use it:** The professor asked for a proper test to compare AUC curves directly. DeLong is the standard method: it accounts for the fact that the two AUCs are estimated on the same data (correlated), and gives a **p-value** for “is pipeline A’s AUC different from pipeline B’s AUC?”
- **When to use:** When you have `y_true`, `prob_a`, `prob_b` for the same trials (e.g. after running pipeline A and B on the same holdout set or same subjects). Use this for **AUC comparison**, not for other metrics.

---

## 2. Bootstrap confidence interval (`bootstrap_metric_ci`) — **what for**

- **What it does:** For **one** set of predictions, it computes a **confidence interval** for a metric (e.g. ROC-AUC, F1, accuracy) by resampling the test set many times and recomputing the metric.
- **Why we use it:** Reporting “AUC = 0.72 [0.65–0.79]” is more informative than “AUC = 0.72” alone. It shows **uncertainty** and is standard in papers and reports.
- **When to use:** When you have one pipeline’s predictions (`y_true`, `y_prob`) and want to report that metric with a CI (e.g. in a table or for a single subject/model).

---

## 3. AUC to odds ratio (`auc_to_odds_ratio`) — **what for**

- **What it does:** Converts an **AUC value** into an **odds ratio (OR)** (and optionally z, Cohen’s d, ln(OR)) under a standard model. So you can say “AUC 0.72 corresponds to OR ≈ 2.3.”
- **Why we use it:** In clinical/epidemiology style reporting, people often interpret discrimination in terms of odds ratios. AUC2OR lets you **translate AUC into that language** for papers or reports.
- **When to use:** When you want to interpret or report AUC as an effect size (e.g. “equivalent to an odds ratio of …”).

---

## 4. Our permutation test (in `compare_tables` with `test="permutation"`) — **what for**

- **What it does:** For **subject-level tables** (one number per subject for pipeline A and one for pipeline B), it tests whether the **mean difference** between A and B is significant by **randomly flipping the sign** of (B − A) per subject many times and seeing how often you get a difference as large as the one you observed.
- **Why we use it:** It’s **distribution-free** (no normality assumption), works well with few subjects, and is the default for the **Pipeline A vs B** comparison in the web UI and scripts. We use **our** implementation (not MLstatkit’s) because we need a **paired** test on subject-level metrics (accuracy, AUC per subject, etc.), not on raw predictions.
- **When to use:** Whenever you compare two pipelines using **tables** where each row is a subject and columns are metrics (accuracy, roc_auc_macro, etc.). That’s the main “A vs B” comparison.

---

## Summary

| Tool              | What it’s for                          | When you use it                                      |
|-------------------|----------------------------------------|------------------------------------------------------|
| **DeLong**        | Compare AUC of two pipelines           | Same test set, you have scores from both A and B      |
| **Bootstrap CI**  | Uncertainty interval for one metric     | Report “metric = x [low, high]” for one pipeline      |
| **AUC2OR**        | Interpret AUC as odds ratio             | Reporting / writing up results in OR language        |
| **Permutation**   | A vs B on subject-level metrics         | Tables with one value per subject per pipeline       |

All of these are already wired in; the test script checks that they run (and that we fail clearly when MLstatkit is missing).
