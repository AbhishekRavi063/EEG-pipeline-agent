# Methodological Warning: Cross-Session Split for BCI IV 2a

## ⚠️ CRITICAL ISSUE: Suspiciously High Accuracy

**Problem:** Getting 1.0 accuracy on BCI IV 2a Subject 1 is suspiciously high and will be questioned by reviewers.

## Root Cause

The default `subject_wise` split with `shuffle=True` **MIXES T and E sessions**:

1. **T session (training):** Contains labeled trials (class markers)
2. **E session (evaluation):** Contains unlabeled trials (no markers in official dataset)
3. **Current behavior:** Both sessions are loaded, concatenated, then shuffled and split 80/20
4. **Problem:** Training can include trials from E session, testing can include trials from T session
5. **Result:** Data leakage → suspiciously high accuracy

## Methodologically Correct Split

**Cross-session split (T/E):**
- **Train:** T session only (all labeled trials from A01T.gdf)
- **Test:** E session only (all trials from A01E.gdf)
- **No mixing, no shuffle across sessions**

This is the **standard evaluation protocol** for BCI IV 2a and what reviewers expect.

## Verification Checklist

✅ **Cross-session split:** T session for train, E session for test  
✅ **No data leakage:** No trial overlap between train and test  
✅ **CSP trained only on train:** Verified - `Pipeline.fit()` only called on `X_train`  
✅ **Preprocessing fitted only on train:** Verified - `preprocess(..., fit=True)` only on train  

## How to Fix

### Option 1: Enable Cross-Session Split (Recommended)

In `config.yaml`:
```yaml
dataset:
  evaluation_mode: "cross_session"  # T session train, E session test
  # OR:
  use_cross_session_split: true  # Same effect
```

### Option 2: Use Sequential Split (No Shuffle)

```yaml
dataset:
  split_mode: "sequential"
  stream_calibration_trials: 20  # First 20 trials = calibration, rest = test
```

This preserves session order (T first, then E).

## Current Implementation Status

### ✅ What's Correct

1. **CSP fitting:** `Pipeline.fit()` is only called on `X_train` (line 392 in main.py)
2. **Preprocessing fitting:** `preprocess(..., fit=True)` only on train data
3. **No sample-level leakage:** Split is at trial level, not sample level

### ⚠️ What's Wrong

1. **Session mixing:** Default split shuffles T and E sessions together
2. **No cross-session option:** Previously missing (now added)
3. **No warning:** Previously no warning about suspicious accuracy

## Code Changes Made

1. **Added `cross_session_split()` function** in `bci_framework/utils/splits.py`
2. **Added `use_cross_session` parameter** to `get_train_test_trials()`
3. **Added warning in main.py** when mixing sessions
4. **Updated config.yaml** with `use_cross_session_split` option

## Expected Results After Fix

With cross-session split (T/E):
- **Lower accuracy:** Typically 60-80% (not 100%)
- **More realistic:** Matches published BCI IV 2a results
- **Reviewer-acceptable:** Follows standard evaluation protocol

## References

- BCI Competition IV Dataset 2a: Standard evaluation uses T session for training, E session for testing
- No official labels in E session, but trials can be used for evaluation if labels are available
- Cross-session evaluation is the standard protocol to avoid session-specific biases

## Action Required

**Before publication or review:**

1. ✅ Enable cross-session split: `use_cross_session_split: true`
2. ✅ Re-run experiments with T/E split
3. ✅ Report cross-session accuracy (not mixed-session accuracy)
4. ✅ Document split methodology in paper

**Current results (1.0 accuracy) are NOT valid for publication** - they use mixed sessions which causes data leakage.
