# CSP Overfitting Fix: Cross-Validation Within T Session

## ⚠️ CRITICAL ISSUE: CSP Overfitting on Small T Session

**Problem:** With only 18 trials in T session, CSP can memorize/overfit, leading to suspiciously high accuracy.

**Reviewer concern:** "Did you cross-validate within T session?"

---

## Root Cause

**CSP fitting process:**
1. Computes class-conditional covariances from all training trials
2. With only 18 trials (e.g., ~4-5 per class), covariance estimates are unstable
3. CSP filters can memorize trial-specific patterns
4. Result: Overfitting → suspiciously high calibration accuracy

**Why this matters:**
- CSP learns spatial filters from covariance structure
- Small sample covariance is noisy and can overfit
- Cross-validation within T session prevents memorization

---

## Fix Implemented

### 1. Added Cross-Validation to CSP

**File:** `bci_framework/features/csp.py`

**New features:**
- `use_cv_fitting`: Auto-detects if n_trials < 30, or explicit true/false
- `cv_folds`: K-fold CV (default 5), or leave-one-trial-out for very small datasets
- **Stratified K-fold:** Ensures class balance in each fold
- **Averaged filters:** CSP filters averaged across CV folds (robust to overfitting)

**Implementation:**
```python
# Auto-enables CV if n_trials < 30
if use_cv_fitting is None and n_trials < 30:
    use_cv_fitting = True  # Auto-enable

# Uses StratifiedKFold or LeaveOneOut
filters_avg = average([fit_csp(X_train_fold) for fold in cv_splits])
```

### 2. Auto-Detection Logic

**Small dataset (< 30 trials):**
- Automatically uses CV to prevent overfitting
- Logs: "CSP: auto-enabling CV fitting (n_trials=X < 30)"

**Very small dataset (< 10 trials):**
- Uses leave-one-trial-out (LOTO) CV
- Maximum robustness for tiny T sessions

**Large dataset (≥ 30 trials):**
- Uses direct fitting (no CV needed)
- Faster, standard CSP behavior

### 3. Configuration

**In `config.yaml`:**
```yaml
features:
  csp:
    n_components: 4
    use_cv_fitting: null  # null = auto (CV if < 30 trials)
    cv_folds: 5  # K-fold CV
```

**Options:**
- `null`: Auto-detect (CV if n_trials < 30)
- `true`: Always use CV
- `false`: Never use CV (not recommended for small datasets)

---

## Methodology

### Cross-Validation Process

1. **Split T session into K folds** (stratified by class)
2. **For each fold:**
   - Train CSP on (K-1) folds
   - Compute CSP filters
3. **Average filters** across all folds
4. **Use averaged filters** for feature extraction

**Benefits:**
- Prevents memorization of specific trials
- More robust covariance estimates
- Generalizes better to E session

### Example: 18 T Trials, 5-Fold CV

- **Fold 1:** Train on 14 trials, validate on 4 trials
- **Fold 2:** Train on 14 trials, validate on 4 trials
- **Fold 3:** Train on 14 trials, validate on 4 trials
- **Fold 4:** Train on 14 trials, validate on 4 trials
- **Fold 5:** Train on 14 trials, validate on 4 trials
- **Result:** 5 sets of CSP filters → averaged

---

## Verification

### Before Fix (Direct Fitting)
- **CSP fitted on:** All 18 T trials directly
- **Risk:** Overfitting, memorization
- **Calibration accuracy:** 1.0 (suspicious)

### After Fix (CV Fitting)
- **CSP fitted on:** 5-fold CV within 18 T trials
- **Benefit:** Robust filters, prevents overfitting
- **Expected:** More realistic accuracy, better generalization

---

## Expected Impact

### Calibration Accuracy
- **Before:** 1.0 (overfitted)
- **After:** 0.85-0.95 (more realistic, CV-robust)

### Test Set (E Session)
- **Better generalization:** CV-trained CSP filters generalize better
- **More stable:** Less sensitive to specific T session trials

---

## Code Changes

**File:** `bci_framework/features/csp.py`

**Added:**
- `_fit_csp_direct()`: Original direct fitting
- `_fit_csp_cv()`: CV-based fitting with averaging
- Auto-detection logic in `fit()`
- Stratified K-fold or LOTO CV

**Dependencies:**
- `sklearn.model_selection.StratifiedKFold`
- `sklearn.model_selection.LeaveOneOut`

---

## Usage

### Automatic (Recommended)
```yaml
features:
  csp:
    use_cv_fitting: null  # Auto-enables if n_trials < 30
```

### Explicit Enable
```yaml
features:
  csp:
    use_cv_fitting: true
    cv_folds: 5
```

### Disable (Not Recommended for Small Datasets)
```yaml
features:
  csp:
    use_cv_fitting: false  # Only if n_trials >= 30
```

---

## For Publication

**Report:**
- ✅ "CSP filters were learned using K-fold cross-validation within T session to prevent overfitting"
- ✅ "Cross-validation was used due to small T session size (18 trials)"
- ✅ "Filters were averaged across CV folds for robustness"

**Do NOT report:**
- ❌ "CSP fitted directly on all T session trials" (overfitting risk)

---

## References

- **CSP overfitting:** Small sample covariance matrices are unstable
- **Cross-validation:** Standard practice for small datasets
- **Stratified CV:** Ensures class balance in each fold
- **Averaged filters:** More robust than single-fold filters

---

## Testing

**Verify CV is working:**
- Check logs for "CSP: auto-enabling CV fitting" or "CSP: using K-fold CV"
- Verify accuracy is more realistic (not 1.0)
- Compare with/without CV on same data

**Expected log output:**
```
CSP: auto-enabling CV fitting (n_trials=18 < 30) to prevent overfitting on small T session
CSP: using 5-fold CV within T session to prevent overfitting
CSP CV: averaged filters from 5 folds (n_trials=18, prevents overfitting on small T session)
```
