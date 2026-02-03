# BCI AutoML Platform — Suggestions & Improvements

## Verification (full run)

After clearing previous results and running:

```bash
rm -rf results/*
PYTHONPATH=. python main.py --synthetic --no-gui
```

**Expected behavior:**

1. **Experiment ID** — Logged at start; results under `results/<experiment_id>/`.
2. **Trial-wise split** — Train/test indices with no sample-level leakage; log: "Trial-wise split: train N, test M (no leakage)".
3. **Calibration** — All pipelines run on calibration trials; metrics (accuracy, kappa, latency_ms, stability) logged per pipeline.
4. **Pruning** — Underperforming pipelines pruned (low accuracy, high latency, unstable); log: "Pruned <name>: ...".
5. **Best pipeline** — Top-1 selected; log: "Best pipeline: <name>".
6. **Snapshots** — For each fitted pipeline: `raw_eeg.png`, `filtered_eeg.png`, `features.png`, `accuracy_curve.png`, `confusion_matrix.png`, `metrics.json`, `model_checkpoint.pkl`.
7. **Drift baseline** — Set from best pipeline accuracy for later adaptation.
8. **Live simulation (--no-gui)** — 20 trial predictions logged; final: "Done. Best pipeline: ... Results in ...".

**Unit tests:** `PYTHONPATH=. pytest tests/ -v` — all tests should pass.

---

## Suggested improvements

### 1. **Optional: save only best pipeline checkpoint**

To save disk, add config `logging.save_checkpoints_selected_only: true` and in main save `.pkl` only for the best pipeline (skip saving all fitted pipelines when this is true).

### 2. **Progress bar for calibration**

Use `tqdm` or a simple counter log (e.g. "Pipeline 5/20 ...") during `run_calibration` so long runs show progress.

### 3. **Validate config at startup**

Check required keys (e.g. `dataset.name`, `agent.prune_thresholds`) and log a warning if missing; fail fast for invalid paths.

### 4. **GUI: use pub-sub for updates**

Wire streaming/pipeline thread to publish to `TOPIC_RAW_EEG`, `TOPIC_PREDICTION`; GUI subscribes via queue and updates plots without blocking evaluation.

### 5. **Latency logger in main loop**

In live simulation, call `PipelineLatencyLogger.log(best_pipeline.name, latency_ms)` after each prediction and optionally log a warning if `exceeds_budget(budget_ms)`.

### 6. **Synthetic EEG in main with --synthetic**

Use `generate_synthetic_mi_eeg()` or `SyntheticEEGLoader` from `datasets.synthetic_eeg` when `--synthetic` is set so synthetic data is generated via the same dataset interface (optional cleanup).

### 7. **CI: use synthetic_eeg in pipeline test**

In `tests/test_pipeline_synthetic.py`, optionally use `generate_synthetic_mi_eeg_for_ci()` for smaller, faster CI runs.

### 8. **README: one-command run**

Add a single copy-paste command for first-time run (e.g. create venv, install deps, run with --synthetic --no-gui) so users can verify the pipeline quickly.

---

## Current status

- **Working as expected:** Trial-wise split, calibration, pruning, best selection, snapshot logging (plots + JSON + checkpoints), versioned results, drift baseline, live simulation (no-GUI and GUI), unit tests.
- **Optional enhancements:** Above list can be implemented incrementally without breaking existing behavior.
