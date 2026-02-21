# Memory Audit: Preprocessing Evaluation Pipeline

## Executive Summary

The full Physionet dataset (109 subjects) consumed ~47 GB RAM and crashed on 16 GB systems. Root causes identified:

1. **Duplicate dataset load** (2x) — primary cause
2. **MOABB loader internal duplication** (~2x during load)
3. **No explicit cleanup** of large arrays between folds
4. **Pipeline intermediates** (X_train, X_flat for ICA) retained during fit

Estimated peak: 2× base dataset + 1× train slice + 1× ICA X_flat + overhead ≈ 40–50 GB for 109 subjects.

---

## A. Exact Location(s) of Memory Accumulation

### 1. Duplicate Dataset Load (CRITICAL)

| Location | Issue |
|----------|-------|
| `scripts/run_final_preprocessing_results.py` line 162 | `load_physionet_mi(subjects)` — first load |
| `bci_framework/evaluation/preprocessing_evaluation.py` line 326 | `load_physionet_mi(subjects)` — second load inside `run_full_evaluation()` |

**Impact:** Two full copies of X, y, subject_ids in memory simultaneously. For 109 subjects (~20k trials, 64 ch, 750 samples): ~7.7 GB per copy → **15.4 GB**.

### 2. MOABB Loader Duplication

| Location | Issue |
|----------|-------|
| `bci_framework/datasets/moabb_loader.py` line 176 | `out[sid] = _make_ds(X[mask], y[mask], ...)` — `X[mask]` creates a copy per subject |
| `bci_framework/evaluation/preprocessing_evaluation.py` line 83 | `np.concatenate(parts_x)` — creates another full copy |

**Impact:** During load: paradigm X (1×) + 109 subject slices (1× combined) + concatenated result (1×) = **~3× peak** during `load_physionet_mi()`. After return, only the concatenated array remains (1×).

### 3. Cross-Validation Loop — Per-Fold Accumulation

| Location | Issue |
|----------|-------|
| `preprocessing_evaluation.py` line 204 | `X_train = X[train_idx]` — creates copy (~80% of data, ~6 GB for 109 subjects) |
| `preprocessing_evaluation.py` line 205 | `X_test = X[test_idx]` — creates copy (~20%) |
| `preprocessing/ica.py` line 34 | `X_flat = X.transpose(0,2,1).reshape(-1, n_ch)` — another full-sized array for ICA fit |
| Pipeline `fit()` | Feature extractor covariance matrices, tangent features — moderate |

**Impact:** Per fold: X_train (~6 GB) + X_flat for ICA (~6 GB) + pipeline intermediates. Not explicitly freed; relies on Python GC. With 2× dataset in memory, peak ≈ 15 + 6 + 6 = **27 GB** per fold before GC.

### 4. Rows Storage — OK

| Location | Check |
|----------|-------|
| `preprocessing_evaluation.py` line 265–271 | `rows.append({"fold", "metrics", "n_test", "train_subjects", "test_subjects"})` — only scalars and small lists. **OK.** |

### 5. Permutation / Bootstrap — OK

| Location | Check |
|----------|-------|
| `table_comparison.py` line 256 | `signs = rng.choice([-1,1], size=(n_perm, n))` — (10000, 5) ≈ 400 KB. **OK.** |
| Bootstrap loop | `boot_means` list of 2000 floats. **OK.** |

### 6. Parallel Processing — OK

| Location | Check |
|----------|-------|
| `classifiers/logistic_regression.py` | `cross_val_score(..., n_jobs=None)` — default 1. **OK.** |
| No `joblib` or `n_jobs=-1` in preprocessing evaluation path. **OK.** |

### 7. Result Storage — OK

| Location | Check |
|----------|-------|
| CSV / JSON | Only scalars (accuracy, auc, kappa, fold_id). **OK.** |

---

## B. Why Memory Scales With Subject Count

- **Dataset size:** n_trials × n_channels × n_samples × 8 bytes  
  Physionet 109 subjects: ~20k trials × 64 × 750 × 8 ≈ **7.7 GB** per copy.

- **Duplicate load:** 2 copies → **15.4 GB**.

- **Per-fold copies:** `X[train_idx]` is a copy; size ∝ n_trials. For 109 subjects, train ≈ 16k trials → **~6 GB** per fold.

- **ICA X_flat:** (n_trials × n_samples) × n_channels × 8. Same order as X_train → **~6 GB** for condition B.

- **Total:** 15 + 6 + 6 + overhead ≈ **28–35 GB** sustained; with GC lag and intermediates, peaks can reach **40–50 GB**.

---

## C. Concrete Code Fixes Applied

### Fix 1: Eliminate Duplicate Dataset Load

- **Change:** Add `preloaded_data` parameter to `run_full_evaluation()`. When provided, skip internal load.
- **Change:** Script loads once, passes `preloaded_data` to `run_full_evaluation()`.
- **Effect:** Removes ~7.7 GB (one full dataset copy).

### Fix 2: Explicit Memory Cleanup in `run_one_fold`

- **Change:** Add `del X_train, X_test, pipe` and `gc.collect()` at end of `run_one_fold()`.
- **Effect:** Encourages prompt release of fold-level arrays.

### Fix 3: Optional GC Between Folds in `run_group_kfold`

- **Change:** Add `gc.collect()` after each fold.
- **Effect:** Reduces peak memory across folds.

### Fix 4: (Future) MOABB Loader Optimization

- **Idea:** Return single `EEGDataset` with full X and `subject_ids_per_trial` instead of per-subject dict.
- **Effect:** Avoid per-subject slices and concatenation; requires loader API change.

---

## D. Minimal Memory-Safe Fold Loop

```python
def run_one_fold(..., return_fit_subjects: bool = False) -> dict[str, Any]:
    ...
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    pipe = build_single_pipeline(...)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test) if hasattr(pipe, "predict_proba") else None
    metrics = compute_fold_metrics(y_test, y_pred, y_proba, n_classes)
    out = {"metrics": metrics, "n_test": len(y_test), ...}

    # Memory-safe: explicitly release large arrays before return
    del X_train, X_test, pipe
    import gc
    gc.collect()
    return out
```

---

## Estimated Memory After Fixes

| Component | Before | After |
|-----------|--------|-------|
| Dataset copies | 2× (~15 GB) | 1× (~7.7 GB) |
| Per-fold X_train | ~6 GB (GC delayed) | ~6 GB (released promptly) |
| ICA X_flat | ~6 GB | ~6 GB |
| **Peak estimate** | **40–50 GB** | **~20–25 GB** |

Target: run full 109-subject evaluation on 32 GB RAM; 16 GB may still be tight during ICA folds.
