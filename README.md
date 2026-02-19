  BCI AutoML Platform — Full Project Overview

A modular, research-grade Motor Imagery BCI framework in Python. It loads EEG data (BCI IV 2a or MOABB), runs a configurable pipeline (preprocessing, spatial filter, features, optional domain adaptation, classifier), uses an AutoML-style agent to select the best pipeline, and provides desktop and web GUIs plus multi-subject tables and statistical comparison (Pipeline A vs B with p-values). Designed for no data leakage, reproducibility, modularity (config-driven; pluggable datasets, features, classifiers), and research workflow (subject-level tables, p-values, publishable code).

Quick start

  1. Install: cd into the project, create a venv, activate it, run pip install -r requirements.txt. Put BCI IV 2a GDF files in data/BCI_IV_2a/ or run scripts/download_bci_iv_2a.sh.

  2. Run single-subject (desktop GUI): PYTHONPATH=. python main.py --subject 1
     Use --subject 2, --subject 3, … up to 9 to pick subject A02, A03, … A09. Subject 1 = A01.

  3. Run with web UI: PYTHONPATH=. python main.py --subject 1 --web
     Then open http://127.0.0.1:8765 (Home) and http://127.0.0.1:8765/compare (Pipeline A vs B). To skip GEDAI/leadfield: ./venv/bin/python scripts/run_web_interface.py (subject is still chosen in config or when you start main.py with --subject N).

  4. No GUI (calibration and logs only): PYTHONPATH=. python main.py --no-gui --subject 1

  5. Compare pipelines on multiple subjects: use the Compare page in the web UI; set "Subjects" to e.g. 1 2 3 and run the comparison. Or use scripts/run_multi_subject_tables.py with subjects in config.


1. What This Project Does

  Load EEG data (BCI Competition IV 2a, MOABB datasets, or synthetic).

  Pipeline: Preprocessing, spatial filter (CAR, Laplacian, CSP, GEDAI), feature extraction (CSP, Riemannian, filter-bank, etc.), optional domain adaptation (zscore, CORAL, Riemann transport), classifier (LDA, SVM, MDM, EEGNet, etc.).

  Agent: Calibration on training trials, pruning (accuracy/latency/stability), cross-validation, composite score, select best pipeline.

  Modes: Offline (batch) and online (first N trials = calibration, rest = live stream).

  GUIs: Desktop (Matplotlib/Tk) and Web (Plotly, zoomable EEG; Pipeline A vs B comparison page with dropdowns and automatic statistical comparison).

  Research: LOSO, transfer learning, subject-level tables (Table_1, Table_2, …) with outcome measures per subject (AUC, balanced accuracy, etc.), paired statistical comparison (permutation test, t-test, Wilcoxon; p-values) — implementing the professor's workflow (see below).

No data leakage (trial-wise or LOSO split); snapshot logging (plots, metrics.json, checkpoints); config-driven; extensible via plugins; memory-conscious (streaming, per-subject LOSO when building tables).


2. Professor's Suggestions — Implemented

The following research workflow was requested and is fully implemented:

  Loop through subjects and get an overall table with each subject in a row and outcome measures in columns (e.g. AUC, balanced accuracy, test accuracy, kappa, F1, ITR). That is Table_1 (one table per pipeline configuration).

  Run exactly the same pipeline but with one setting changed (e.g. different classifier, or replace ICA with GEDAI). That generates Table_2. Every Table_X contains the same subjects (same rows), so you have distributions across the same subjects for each pipeline.

  Compare those distributions with statistics — get p-values to see whether the Table_1 pipeline is significantly different from the Table_2 pipeline (permutation test recommended, or paired t-test / Wilcoxon).

  Workflow: First explore the results to see which comparisons are interesting and statistically significant; then choose a few for the paper; you can publish the code with the paper.

Web-based pipeline comparison (implemented):

  Two parallel pipelines A and B in the same UI, with the same choice of parameters in dropdown menus (feature, classifier, spatial filter).

  You keep or change one (or a few) settings between A and B (e.g. Pipeline A = LDA, Pipeline B = SVM).

  The system automatically runs the statistical comparison between those two pipelines on the same subjects and shows Table_A, Table_B, and the comparison (mean A, mean B, delta, p-value, significant Yes/No at α=0.05).

Where it lives:

  Tables: bci_framework/utils/subject_table.py — build_subject_table, TABLE_METRIC_COLUMNS (accuracy, balanced_accuracy, roc_auc_macro, kappa, f1_macro, itr_bits_per_minute, n_trials_test). Scripts and web backend use this to build Table_1, Table_2.

  Comparison: bci_framework/utils/table_comparison.py — compare_tables, compare_tables_multi_metric (permutation test, paired t-test, Wilcoxon). MLstatkit: DeLong (delong_test_auc), bootstrap CIs (bootstrap_metric_ci), AUC to odds ratio (auc_to_odds_ratio).

  Runner: bci_framework/evaluation/multi_subject_runner.py — run_table_for_config (LOSO per subject), run_ab_comparison (Table_A + Table_B + comparison).

  Web UI: bci_framework/gui/static/compare.html — dropdowns for Pipeline A and B, "Copy A to B", Run comparison, POST /api/compare_pipelines, displays two tables and comparison table.

  Scripts: scripts/run_multi_subject_tables.py (Table_1, optional Table_2 + comparison); scripts/explore_pipeline_comparisons.py (batch configs, pairwise comparison).


3. Architecture (End-to-End)

main.py:
  Load config, Seed, Load dataset, Split (train/test or LOSO)
  Build pipelines (registry from config)
  Agent: calibration (optional quick screening, then CV), prune, select best
  Snapshot logs (per pipeline)
  Stream test set through best pipeline (trial-by-trial or sliding)
  GUI (desktop or web)

Pipeline (per trial):

  Raw EEG (n_trials, n_channels, n_samples)
  Signal quality (optional)
  Notch (50/60 Hz) + Bandpass (e.g. 8–30 Hz motor)
  Spatial filter (CAR, Laplacian, CSP, GEDAI)
  Optional advanced (ICA, wavelet, GEDAI)
  Feature extraction (CSP, Riemannian, Covariance, Filter-bank Riemann, etc.)
  Domain adaptation (none, zscore, coral, riemann_transport) — optional
  Classifier (LDA, SVM, MDM, EEGNet, Logistic Regression, RSA-MLP, etc.)
  Prediction + metrics

Modularity: Datasets, features, classifiers, spatial filters, and preprocessing plugins implement base interfaces and are registered; pipelines are built from config (explicit list or auto combinations). No core code change needed to add a new dataset, feature, or classifier — implement the interface and register.

Memory: Streaming uses buffers (configurable window); LOSO table runs process one holdout subject at a time (train on others, test on holdout) and append one row per subject; full raw data is loaded once per run for the requested subjects, then released after tables are built.


4. Directory Layout

EEG Agent/
  main.py — Entry: load, calibrate, best, stream, GUI
  requirements.txt
  README.md — This file
  bci_framework/
    config.yaml — Master config (dataset, preprocessing, pipelines, agent, gui, logging)
    datasets/ — base, bci_iv_2a, moabb_loader, synthetic_eeg
    preprocessing/ — manager, notch, bandpass, spatial_filters/, gedai, ica, wavelet
    features/ — csp, riemannian, covariance, filter_bank_riemann, etc.
    domain_adaptation/ — zscore, coral, riemann_transport
    classifiers/ — lda, svm, mdm, eegnet, logistic_regression, rsa_mlp, etc.
    pipelines/ — pipeline.py, registry.py
    agent/ — pipeline_agent.py (calibration, pruning, quick_screening, early_cv_stop), drift_detector
    streaming/ — offline_stream, realtime_stream, buffer
    logging/ — snapshot.py (plots, metrics.json, checkpoints, adaptive_pruning)
    gui/ — app.py (desktop), web_server.py, static/index.html, compare.html
    evaluation/ — multi_subject_runner (LOSO, run_ab_comparison)
    utils/ — config_loader, splits, metrics, subject_table, table_comparison
  scripts/ — run_multi_subject_tables, explore_pipeline_comparisons, benchmark_v1_v2_moabb, run_web_interface, download_bci_iv_2a.sh
  tests/
  examples/ — run_offline.py, run_online.py, notebooks
  results/ — experiment_id/pipeline_name/ (plots, metrics.json)
  data/BCI_IV_2a/ — A01T.gdf, A01E.gdf, etc. (or MOABB cache)


5. Dataset (BCI IV 2a and MOABB)

  BCI IV 2a: 4 classes (left hand, right hand, both feet, tongue). 22 EEG channels, 250 Hz, 9 subjects (A01–A09). One trial = one cued 3 s segment. Training (T) and evaluation (E) sessions.

  Setup: Place GDF files in ./data/BCI_IV_2a/ or run ./scripts/download_bci_iv_2a.sh. Channel names: when the file has generic names (EEG1, EEG2, …), the loader uses standard 10–20 names (Fz, FC3, Cz, …) for the UI.

  MOABB: e.g. BNCI2014_001, PhysionetMI — used for LOSO and multi-subject tables via moabb_loader.py.

  Config: dataset.name, data_dir, subjects, trial_duration_seconds, train_test_split, split_mode, evaluation_mode, use_cross_session_split.


6. Installation and Quick Run

cd "EEG Agent"
python -m venv venv
  source venv/bin/activate   (Windows: venv\Scripts\activate)
pip install -r requirements.txt

  Default: calibration + live stream + desktop GUI
PYTHONPATH=. python main.py --subject 1

  Web UI (browser)
  PYTHONPATH=. python main.py --subject 1 --web
  Home: http://127.0.0.1:8765   Compare: http://127.0.0.1:8765/compare

  Web without GEDAI (no leadfield)
  ./venv/bin/python scripts/run_web_interface.py

  No GUI
  PYTHONPATH=. python main.py --no-gui --subject 1

  Online: first N trials = calibration, rest = live stream
  PYTHONPATH=. python main.py --web --online --subject 1


7. Web UI — User Guide

How to run the web UI

  With GEDAI disabled (no leadfield; recommended for quick runs):
  ./venv/bin/python scripts/run_web_interface.py

  With default config (needs leadfield if GEDAI enabled):
  ./venv/bin/python main.py --subject 1 --web

After calibration (about 15–20 s with GEDAI disabled), open http://127.0.0.1:8765 (Home) and http://127.0.0.1:8765/compare (Pipeline A vs B). Keep the terminal running.

Home page (Live EEG)

Purpose: Single-subject stream — one pipeline, trial by trial.

  Status bar: Subject, dataset, pipeline name, trial (e.g. 3/10), prediction, trial source (T/E), rolling accuracy.

  Raw EEG: One trial, unprocessed; channel names from dataset (e.g. Fz, Cz when using BCI IV 2a). May show line noise, drift.

  Processed EEG: Same trial after pipeline (notch, bandpass, CAR/Laplacian, optional ICA/wavelet). Smoother trace = actual processing, not display smoothing.

  Active frequency bands (raw / processed): Power in delta, theta, alpha, beta, gamma and noise bands (drift, line_50/60, emg).

  Accuracy over time: Accuracy as trials progress (one point per trial).

  Pipeline comparison: Bar chart of calibration CV accuracy; best pipeline highlighted. Table: Accuracy, Kappa, Latency (ms), Stability, Composite score. Formula: 0.4×accuracy + 0.3×kappa + 0.2×stability − 0.1×(latency in s).

Subjects: Home uses one subject (e.g. --subject 1). Data from dataset files on disk (not a DB). Transfer learning: controlled in config; used in backend when enabled; not shown in the web UI (only in logs and metrics.json).

Compare page (Pipeline A vs B)

Purpose: Two pipelines on the same subjects, Table_A, Table_B, and statistical comparison (p-values).

  1. Set Dataset (e.g. BNCI2014_001) and Subjects (e.g. 1 2 3).
  2. Configure Pipeline A and Pipeline B via Simple dropdowns (Feature, Classifier, Spatial) or Advanced (config paths, Override for B JSON).
  3. Click Run comparison — backend runs LOSO for A and B on the same subjects, builds Table_A and Table_B, runs the selected test (permutation, t-test, or Wilcoxon).
  4. Page shows Table A, Table B, and Comparison table: per metric — Mean A, Mean B, Delta, p-value, Significant (α=0.05).

Tip: "Copy A to B", then change one dropdown for B (e.g. Classifier B = SVM) to compare "same pipeline, different classifier."

Dropdown options (Compare page)

  Feature: filter_bank_riemann, csp, riemannian, covariance, riemann_tangent_oas.

  Classifier: logistic_regression, lda, svm, random_forest, rsa_mlp, mdm.

  Spatial: laplacian_auto, car.

  Test: Permutation (recommended; distribution-free), Paired t-test (parametric), or Wilcoxon signed-rank (non-parametric).

Advanced (Compare page)

Leave empty to use Simple dropdowns.

  Pipeline A/B config path: Use a full config file instead of dropdowns.

  Override for B (JSON): "B = A + this JSON." Example: {"pipelines":{"explicit":[["filter_bank_riemann","svm"]]}} — B = same as A but classifier SVM.

Quick reference (Web UI)

  Purpose — Home: One subject, one pipeline, trial-by-trial. Compare: Two pipelines (A vs B) on same subjects; tables + p-values.

  Subjects — Home: One. Compare: Multiple (e.g. 1 2 3).

  Shows — Home: Raw/processed EEG, band power, accuracy, pipeline bar. Compare: Table A, Table B, comparison (Mean A/B, delta, p-value, significant?).

  URL — Home: http://127.0.0.1:8765. Compare: http://127.0.0.1:8765/compare.


8. Pipeline A vs B — Flow (Compare)

  Backend loads data once for all listed subjects.

  For each subject, LOSO: train on other subjects, test on this subject — one row (subject_id, accuracy, balanced_accuracy, roc_auc_macro, kappa, f1_macro, itr_bits_per_minute, n_trials_test) for Pipeline A — Table A.

  Same for Pipeline B — Table B (same subject IDs).

  Comparison: For each metric, align by subject_id, run selected test (permutation, t-test, or Wilcoxon) — p-value, mean A, mean B, delta, significant (Yes/No at α=0.05). Displayed in the same page.


9. Configuration (config.yaml)

  dataset — name, data_dir, subjects, trial_duration_seconds, train_test_split, split_mode, evaluation_mode, use_cross_session_split

  spatial_filter — enabled, method (car, laplacian_auto, csp, gedai), auto_select, methods_for_automl

  preprocessing — notch_freq, bandpass_low/high, adaptive_motor_band, reference

  advanced_preprocessing — enabled: [signal_quality, gedai, ica, wavelet], gedai.leadfield_path

  pipelines — auto_generate, max_combinations, explicit: [[feature, clf], ...]

  transfer — enabled, method (none, zscore, coral, riemann_transport), target_unlabeled_fraction

  agent — calibration_trials, cv_folds, prune_thresholds (min_accuracy, max_latency_ms), quick_screening, early_cv_stop, progressive_halving

  streaming — mode, trial_duration_sec, real_time_timing, stream_full_test_set

  gui — refresh_rate_ms, eeg_channels_display, window_seconds, web_port


10. Main Flow (main.py)

  1. Load config and dataset.
  2. Split (trial-wise 80/20, or sequential, or LOSO).
  3. Build pipelines from config (registry).
  4. Agent: calibration (optional quick screening, then full CV), prune, select best by composite score.
  5. Snapshot: save plots, metrics.json, optional checkpoints per pipeline (including adaptive_pruning stats when used).
  6. Stream test set through best pipeline; optional real-time pacing.
  7. GUI: desktop or web (WebSocket pushes state: raw_buffer, filtered_buffer, channel_names, pipeline_metrics, accuracy_history, etc.).


11. Domain Adaptation (Transfer Learning)

When transfer.enabled is true (e.g. LOSO): adapter fits on source features + unlabeled target calibration; test target never used for fitting. Methods: none, zscore, coral, riemann_transport. Pipeline: features, adapter, classifier; predict always goes through adapter. If transfer is on, X_target (unlabeled) is required at fit time; clear errors if missing.


12. Agent (Pruning and Selection)

  Quick screening (optional): Evaluate pipelines on a stratified subset; keep top-K for full CV; no transfer on clones; StratifiedShuffleSplit; optional trial-level split to avoid leakage.

  Early CV stop: Stop a pipeline's CV if it cannot beat the best score so far.

  Progressive halving (optional): Reduce pipelines on a fraction of data before full CV.

  Prune: By min_accuracy, max_latency_ms, stability variance; exclude list (e.g. GEDAI) can skip screening and go to full CV.

  Select best: Composite score; tie-break by latency or prefer linear models. Ranking correlation (screening vs full CV) and runtime stats (pipelines_before/after, cv_fits, runtime_seconds) logged; perfect-accuracy check (shuffled labels) for leakage warning.


13. Scripts (Summary)

  run_multi_subject_tables.py — Build Table_1 (and Table_2) over subjects (LOSO); optional comparison report (Table_1 vs Table_2, p-values).

  explore_pipeline_comparisons.py — Multiple configs, tables, pairwise comparison.

  benchmark_v1_v2_moabb.py — v1/v2/v3 on MOABB; optional --loso, --transfer-method.

  run_web_interface.py — Start web UI with GEDAI disabled (no leadfield).

  download_bci_iv_2a.sh — Download BCI IV 2a GDF files.


14. Adding New Methods (Modularity)

  Dataset: Implement DatasetLoader in datasets/, register in datasets/__init__.py.

  Feature / classifier: Implement base interface in features/ or classifiers/, register in __init__.py.

  Preprocessing: Add class in preprocessing/, register in manager. New pipelines are picked up when auto_generate is true or via explicit in config.


15. Snapshot Logging

Per pipeline under results/experiment_id/pipeline_name/: raw EEG plot, filtered EEG plot, metrics.json (accuracy, kappa, latency, transfer info, spatial_filter_used, etc.). When adaptive pruning is used: adaptive_pruning block (quick_screening, runtime_stats, early_stopped_pipelines, correlation_with_full_cv). Optional model checkpoints (.pkl).


16. Cross-Check: Features from UI to Backend

  Subject-level tables (Table_1, Table_2): Compare page shows Table A, Table B (subjects × metrics). Backend: run_table_for_config, LOSO per subject, build_subject_table; one row per subject; metrics = accuracy, balanced_accuracy, roc_auc_macro, kappa, f1_macro, itr_bits_per_minute, n_trials_test.

  Statistical comparison (p-values): Compare page: Mean A, Mean B, Delta, p-value, Significant. Backend: compare_tables_multi_metric (permutation test, t-test, Wilcoxon); same subjects in both tables. AUC curves: DeLong test (MLstatkit) available via delong_test_auc when you have prediction scores for the same test set from both pipelines.

  Web A vs B with same params, change one: Dropdowns Pipeline A and B; "Copy A to B"; change one (e.g. classifier). Backend: run_ab_comparison with pipeline_a / pipeline_b dicts (feature, classifier, spatial) or config paths / override_b.

  Channel names (montage): Raw/processed EEG legend: Fz, Cz, … (not EEG1, EEG2). Backend: BCI IV 2a loader uses standard 10–20 list when file has generic names; channel_names passed to WebApp and sent via WebSocket.

  Transfer learning: Not shown in UI. Backend: Config; pipeline requires X_target when transfer on; predict always via adapter; logged in metrics.json.

  Adaptive pruning: Not in UI. Backend: Agent quick_screening, early_cv_stop, progressive_halving; runtime and correlation in metrics.json (adaptive_pruning).

  Modularity: Config-driven dropdowns (feature, classifier, spatial). Backend: Registries for datasets, features, classifiers; pipelines from config; no core change for new methods.

  Memory: Compare runs LOSO per subject; tables built incrementally. Backend: Data loaded once for requested subjects; per-holdout fold runs; tables = list of dicts (one row per subject).


17. License

Use and extend for research and product development as needed. Publish code with the paper as suggested in the professor's workflow.
