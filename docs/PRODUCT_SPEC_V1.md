# BCI AutoML Platform v1.0 — Product-Grade Spec Summary

## Addressed Gaps (from research prototype → product)

### 1. Data leakage prevention
- **Trial-wise split only:** `utils/splits.py`: `trial_train_test_split`, `get_train_test_trials`. No sample-level split; preprocessing and CSP are fitted on train trials only; test is never seen during fit.
- **Subject-wise vs LOSO:** `evaluation_mode: subject_wise | cross_subject | loso`; `loso_holdout_subject` and `--loso` for leave-one-subject-out.
- **Main flow:** `main.py` uses `get_train_test_trials()` for train/test indices; no leakage.

### 2. Calibration protocol
- **Config:** `config.yaml` → `calibration.duration_trials`, `calibration.baseline_min_trials`, `calibration.sliding_window_adaptation`.
- **Agent:** Calibration phase runs all pipelines on first N trials; baseline used for drift detection.

### 3. Real-time streaming architecture
- **Buffer:** `utils/streaming.py`: `EEGStreamBuffer` (ring buffer), `sliding_window_chunks`, `stream_chunk`.
- **Config:** `streaming.window_size_sec`, `streaming.overlap_ratio` (e.g. 50%).
- **Dataset:** `EEGDataset.get_trial_data()`, `EEGDataset.stream_chunk(trial_id, window_samples, overlap_ratio)`; `DatasetLoader.get_trial_data` / `stream_chunk` in base.

### 4. Hyperparameter optimization (scaffold)
- **Module:** `agent/hyperparameter_optimizer.py`: `_grid_search`, `optuna_optimize` (Optuna). Config: `agent.hyperparameter_optimization.enabled`, `backend`, `n_trials`, `param_ranges`.
- **Integration:** Can be wired in main or agent to tune preprocessing/classifier params per pipeline.

### 5. Model persistence
- **Snapshot logger:** `logging/snapshot.py`: `save_pipeline_model()` (`.pkl`), `load_pipeline_model()`, `save_pipeline_torch()` (`.pt`).
- **Config:** `logging.save_model_checkpoints`, `logging.versioned_experiments`.
- **Main:** Saves best pipeline and optionally all fitted pipelines; versioned under `results/<experiment_id>/` when enabled.

### 6. Drift detection
- **Module:** `agent/drift_detector.py`: `DriftDetector` with rolling-window accuracy, baseline, thresholds (`accuracy_drop_threshold`, `min_accuracy_absolute`, `consecutive_low_windows`).
- **Config:** `agent.drift.*`.
- **Main:** After calibration, `agent.set_drift_baseline(best_accuracy)`; during live run, `detector.update(correct)` can trigger recalibration.

### 7. Human-in-the-loop placeholder
- **Module:** `utils/feedback.py`: `HumanFeedbackAPI` with `submit_correction(trial_id, predicted, corrected)`, `get_pending_corrections()`, `get_correction_accuracy()`.
- **Config:** `future.human_in_the_loop_feedback`.

### 8. Compute / latency budget
- **Config:** `compute.max_parallel_pipelines` (e.g. 5 for M4 Pro); `prune_thresholds.latency_budget_ms` (100–300 ms); pruning uses `latency_budget_ms` when set.
- **Agent:** Prunes pipelines exceeding latency budget.

### 9. Advanced metrics
- **Module:** `utils/metrics.py`: `accuracy`, `cohen_kappa`, `f1_macro`, `roc_auc_ovr`, `itr_bits_per_trial`, `itr_bits_per_minute`, `mutual_information`, `compute_all_metrics`.
- **Agent:** Pipeline metrics include `f1_macro`, `roc_auc_macro`, `itr_bits_per_minute`; snapshot JSON and leaderboard use these.

### 10. Experiment tracking & reproducibility
- **Module:** `utils/experiment.py`: `set_seed()`, `get_experiment_id()`, `set_experiment_id()`, `log_experiment_params`, `log_experiment_metrics`, `enable_mlflow()`.
- **Config:** `experiment.seed`, `experiment.experiment_id`, `experiment.mlflow.*`.
- **Main:** Sets seed and experiment ID; optional MLflow; versioned results when `versioned_experiments` is true.

### 11. Plugin / registry
- **Existing:** `PREPROCESSING_REGISTRY`, `FEATURE_REGISTRY`, `CLASSIFIER_REGISTRY`, `DATASET_REGISTRY`. New methods: inherit base class and add to registry in module `__init__.py`.
- **Config-driven:** Pipelines from config (explicit or auto-generated); no code change for new combinations.

### 12. Deployment modes
- **CLI:** `main.py --no-gui`, `--experiment-id`, `--loso`.
- **GUI:** Default; live EEG, leaderboard, prediction.
- **Headless:** `--no-gui` for calibration + logging + optional live sim without GUI.

### 13. Testing & CI
- **Tests:** `tests/test_splits.py`, `test_streaming.py`, `test_metrics.py`, `test_pipeline_synthetic.py` (synthetic EEG, calibration, prune).
- **CI:** `.github/workflows/ci.yml` template (pytest on push/PR).

## Config overview (config.yaml)

- **dataset:** split_level, loso_holdout_subject, evaluation_mode.
- **calibration:** duration_trials, baseline_min_trials, sliding_window_adaptation.
- **streaming:** window_size_sec, overlap_ratio.
- **agent:** prune_thresholds (min_accuracy, max_latency_ms, latency_budget_ms), drift.*, hyperparameter_optimization.*, trial_duration_sec.
- **logging:** save_model_checkpoints, versioned_experiments.
- **experiment:** seed, experiment_id, mlflow.
- **compute:** max_parallel_pipelines, use_gpu.

## Minor additions (post-v1)

- **Streaming & GUI sync:** Thread-safe `utils/pubsub.py` (PubSub, TOPIC_RAW_EEG, etc.); `subscribe_queue()` for GUI thread to drain without blocking pipeline.
- **Latency logging:** `utils/latency_logger.py` — `PipelineLatencyLogger` logs ms per window, `get_stats()`, `exceeds_budget()` for dynamic budget checks.
- **Default HP ranges:** `config.yaml` → `agent.hyperparameter_optimization.param_ranges` (bandpass_highcut, csp_n_components, lda_shrinkage, svm_C, svm_gamma) for out-of-the-box tuning.
- **Leaderboard panel:** `gui/leaderboard.py` — `build_leaderboard_table()`, `render_leaderboard_matplotlib()` (sortable table, pruned/selected flags).
- **Explainability stub:** `gui/explainability.py` — `plot_csp_patterns()`, `shap_importance_stub()` for CSP maps and SHAP placeholder.
- **Example notebook:** `examples/bci_automl_minimal_example.ipynb` — load data, calibration, leaderboard, best pipeline, snapshot saving, CSP plot.

## Optional / future

- Optuna integration in main loop (per-pipeline HP tuning).
- RL-based pipeline selector.
- Real hardware (OpenBCI, Emotiv), MCP/LLM, cloud training.
- Data augmentation (noise, time shift, GAN placeholder).
- API mode for external control.
