# GEDAI Comparison Report

**Date:** 2026-02-05  
**Setup:** BCI IV 2a, subject 1, 28 trials (22 train / 6 test), `--no-gui`, offline mode.

---

## 1. Environment and runs

- **Venv activated;** installed: `pygedai`, `pandas`, `mat73` (pygedai’s optional deps).
- **Baseline (no GEDAI):** `advanced_preprocessing.enabled: ["signal_quality", "ica", "wavelet"]`.
- **With GEDAI:** same list plus `"gedai"`; leadfield = identity (`use_identity_if_missing: true`).
- **Fix applied:** GEDAI wrapper now supports both dict and tensor return from `batch_gedai` (pygedai 1.0.3 returns a tensor).

---

## 2. Baseline results (without GEDAI)

**Experiment ID:** `20260205_173030_0ea66aae`  
**Pipelines:** 20 (all completed).  
**Pipeline selection time:** ~10.2 s.

### Best pipeline (selected)

| Metric        | Value   |
|---------------|---------|
| Pipeline      | `baseline_signal_quality-ica-wavelet_csp_lda` |
| Accuracy      | 1.00    |
| Cohen's Kappa | 1.00    |
| Latency (ms)  | ~3.06   |
| Stability     | 1.00    |
| F1 (macro)    | 1.00    |
| ROC-AUC (macro) | 1.00  |
| ITR (bits/min)| 40      |

### Other pipelines (sample)

| Pipeline (feature_classifier)     | Accuracy | Kappa | Latency (ms) |
|----------------------------------|----------|-------|--------------|
| csp_lda                          | 1.000    | 1.000 | 3.06         |
| csp_random_forest                 | 1.000    | 1.000 | ~3.1         |
| csp_svm                          | 0.864    | 0.365 | 3.08         |
| riemannian_random_forest          | 1.000    | 1.000 | 3.82         |
| riemannian_lda                   | 1.000    | 1.000 | ~3.5         |
| riemannian_svm                   | 0.864    | 0.365 | 3.76         |
| psd_lda                          | 1.000    | 1.000 | ~3.2         |
| wavelet_lda                     | 1.000    | 1.000 | ~3.2         |

Results are under `results/20260205_173030_0ea66aae/<pipeline_name>/metrics.json`.

---

## 3. Results with GEDAI

**Experiment ID (fixed run):** `20260205_173347_9dd528ed` (run partial; timed out after several pipelines).

- **Preprocessing:** GEDAI step used identity leadfield (22 channels). pygedai issued warnings that epoch length is short for its lowest frequency bands (expected for 3 s trials).
- **GEDAI cost:** ~2.9–3.0 s per batch (22 trials) in preprocessing; dominates per-trial latency when enabled.
- **From logs (partial run):**
  - `baseline_signal_quality-ica-wavelet-gedai_riemannian_random_forest`: **acc = 1.000**, kappa = 1.000, **latency ≈ 3047 ms**.
  - `baseline_signal_quality-ica-wavelet-gedai_riemannian_svm`: acc = 0.818, kappa = 0.000, latency ≈ 3036 ms.

So with GEDAI, **accuracy can match baseline** (e.g. 1.0 for Riemannian + Random Forest), but **latency is ~3000 ms per trial** (vs ~3 ms without GEDAI), so real-time use with GEDAI in this setup is not feasible.

---

## 4. Comparison summary

| Aspect           | Without GEDAI              | With GEDAI (identity leadfield)   |
|------------------|---------------------------|-----------------------------------|
| Best accuracy    | 1.00 (e.g. CSP+LDA)       | 1.00 (e.g. Riemannian+RF)         |
| Per-trial latency| ~3–4 ms                   | ~3000 ms (GEDAI-dominated)        |
| Pipeline selection time | ~10 s              | ~60+ s (partial run)              |
| Real-time viable | Yes                       | No (latency too high)             |
| Snapshot logs    | 20 pipelines saved        | Run timed out; no full snapshots |

**Conclusion:** On this small, clean BCI IV 2a subject-1 slice, adding GEDAI (with identity leadfield) did not improve accuracy over the existing chain (signal_quality + ICA + wavelet). It increased latency by roughly three orders of magnitude, making it unsuitable for real-time BCI in the current configuration. GEDAI may still be useful for offline, heavily contaminated data or with a proper leadfield and longer epochs; the wrapper is in place for such experiments.

---

## 5. How to reproduce

```bash
cd "EEG Agent"
source venv/bin/activate
pip install pygedai pandas mat73   # mat73 optional, used by pygedai ref_cov

# Baseline (no GEDAI)
# In config: advanced_preprocessing.enabled: ["signal_quality", "ica", "wavelet"]
PYTHONPATH=. python main.py --subject 1 --no-gui

# With GEDAI
# In config: advanced_preprocessing.enabled: ["signal_quality", "ica", "wavelet", "gedai"]
PYTHONPATH=. python main.py --subject 1 --no-gui
```

Compare metrics in `results/<experiment_id>/<pipeline_name>/metrics.json` and in the console logs (accuracy, kappa, latency).
