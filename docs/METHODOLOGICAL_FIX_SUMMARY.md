# Methodological Fix Summary: Cross-Session Split

## Issue Identified

**Problem:** Getting 1.0 accuracy on BCI IV 2a Subject 1 is suspiciously high and indicates potential data leakage.

**Root Cause:** Default `subject_wise` split with `shuffle=True` **mixes T and E sessions**, causing:
- Training can include trials from E session
- Testing can include trials from T session
- Session-specific biases leak across train/test
- Unrealistically high accuracy

## Fix Implemented

### 1. Added Cross-Session Split Function

**File:** `bci_framework/utils/splits.py`

- New function: `cross_session_split(n_trials_from_t, n_trials_total)`
- **Train:** T session only (indices 0..n_trials_from_t-1)
- **Test:** E session only (indices n_trials_from_t..n_trials_total-1)
- **No shuffle, no mixing**

### 2. Updated `get_train_test_trials()`

- Added `n_trials_from_t` parameter
- Added `use_cross_session` flag
- Added `evaluation_mode: "cross_session"` option
- Added warning when mixing sessions

### 3. Updated `main.py`

- Passes `n_trials_from_t` from dataset to split function
- Logs warning when mixing sessions
- Enables cross-session split when configured

### 4. Updated `config.yaml`

- Added `use_cross_session_split: true/false` option
- Added documentation about methodological correctness

## Verification Checklist

✅ **Cross-session split implemented:** T session train, E session test  
✅ **No trial overlap:** Verified - train and test indices are disjoint  
✅ **CSP trained only on train:** Verified - `Pipeline.fit()` only called on `X_train` (line 392)  
✅ **Preprocessing fitted only on train:** Verified - `preprocess(..., fit=True)` only on train  
✅ **Test set evaluation:** Confusion matrix uses `y_test` and `pipe.predict(X_test)` (line 435-439)  

## Results Comparison

### Mixed Sessions (OLD - INCORRECT)
- **Split:** 80/20 trial-wise with shuffle (mixes T and E)
- **Train:** 22 trials (mixed T/E)
- **Test:** 6 trials (mixed T/E)
- **Accuracy:** 1.0 (suspiciously high - data leakage)

### Cross-Session Split (NEW - CORRECT)
- **Split:** T session train, E session test (no mixing)
- **Train:** 18 trials (T session only)
- **Test:** 10 trials (E session only)
- **Calibration accuracy:** 1.0 (on 18 T trials - still high but methodologically correct)
- **Test set:** Evaluated separately (see confusion matrix)

## Important Notes

### Calibration vs Test Set Accuracy

**Current implementation:**
- **Calibration accuracy** (in `metrics.json`): Evaluated on training trials (for pipeline selection)
- **Test set accuracy** (in confusion matrix): Evaluated on E session (for final evaluation)

**For publication, report:**
- ✅ **Test set accuracy** (from confusion matrix on E session)
- ⚠️ **Calibration accuracy** is for pipeline selection only, not final performance

### Why Calibration Still Shows 1.0

Even with cross-session split, calibration accuracy on 18 T trials can be 1.0 because:
1. **Small calibration set:** Only 18 trials (easy to overfit)
2. **Subject 1 is clean:** BCI IV 2a Subject 1 has relatively clean data
3. **Calibration is for selection:** Not the final test set evaluation

**The test set (E session) accuracy is what matters for publication.**

## Usage

### Enable Cross-Session Split

**Option 1: Config flag**
```yaml
dataset:
  use_cross_session_split: true
```

**Option 2: Evaluation mode**
```yaml
dataset:
  evaluation_mode: "cross_session"
```

### Verify Split

Check logs for:
```
Cross-session split: T session (train) = 18 trials, E session (test) = 10 trials
```

## Code Changes

**Files modified:**
1. `bci_framework/utils/splits.py` - Added `cross_session_split()` and parameters
2. `main.py` - Added warning and passes `n_trials_from_t`
3. `bci_framework/config.yaml` - Added `use_cross_session_split` option
4. `docs/METHODOLOGICAL_WARNING.md` - Documentation

**Files created:**
1. `docs/METHODOLOGICAL_FIX_SUMMARY.md` - This file

## Next Steps

1. ✅ **Cross-session split implemented** - T/E split now available
2. ⚠️ **Test set accuracy logging** - Currently only in confusion matrix, not in metrics.json
3. 📝 **Documentation** - Methodological warning added
4. 🔍 **Review test set results** - Check confusion matrices for actual E session performance

## For Publication

**Report:**
- ✅ Cross-session split methodology (T session train, E session test)
- ✅ Test set accuracy from E session (not calibration accuracy)
- ✅ Confusion matrix on test set
- ✅ No data leakage (sessions separated)

**Do NOT report:**
- ❌ Mixed-session accuracy (1.0 from shuffled T/E)
- ❌ Calibration accuracy as final performance

## References

- BCI Competition IV Dataset 2a standard evaluation protocol
- Cross-session evaluation prevents session-specific biases
- Reviewers expect T/E split for BCI IV 2a
