# BCI AutoML Platform v1.0

A modular, extensible **Motor Imagery BCI AutoML** platform in Python (research and product-grade). It loads a standard public EEG motor imagery dataset, implements multiple preprocessing / feature extraction / classification pipelines, automatically evaluates and prunes underperforming pipelines, selects the best pipeline for real-time simulation, and provides a desktop GUI with live EEG and performance plots. All pipelines (selected and rejected) get snapshot logs and graphs for research.

## Architecture

```
EEG → Preprocessing → Feature Extraction → Classifier → Metrics
```

The project is structured as a **reusable research and product framework** so new methods can be added by creating new Python files and registering them.

### Directory Layout

```
/bci_framework
  /datasets       # Dataset loaders (BCI IV 2a; add more via DatasetLoader)
/preprocessing  # Mandatory baseline pipeline + advanced registry (ICA, wavelet, ASR)
  /features       # CSP, PSD, wavelet, Riemannian, deep placeholder
  /classifiers    # LDA, SVM, Random Forest, EEGNet, Transformer placeholder
  /pipelines      # Pipeline = prep chain + feature + classifier; PipelineRegistry
  /agent          # Pipeline Selection Agent (explore, prune, exploit, adapt)
  /gui            # Desktop GUI + Web UI (live EEG, accuracy, zoomable Plotly view)
  /logging        # Snapshot logger (plots + JSON per pipeline)
  /utils          # Config loader, registry helpers
  config.yaml     # Pipelines, thresholds, paths (no code change needed)
main.py           # Entry: load data → calibrate → prune → best → live sim → GUI → logs
```

Each module is **independent and pluggable**.

## Dataset

- **BCI Competition IV Dataset 2a only** (4-class motor imagery, 22 channels, 250 Hz). The app uses this dataset from `./data/BCI_IV_2a/`.

**What does “each trial” mean? (Subject → Session → Run → Trial)**

- **Subject** — One person (e.g. A01). The dataset has 9 subjects.
- **Session** — One recording file: **training (T)** or **evaluation (E)** (e.g. `A01T.gdf`, `A01E.gdf`). Each session has 6 **runs**.
- **Run** — One continuous part of a session. Each run has **about 48 trials**.
- **Trial** — **One 3-second segment** where the user was cued to imagine one movement (left hand, right hand, both feet, or tongue). So: **one trial = one cue + 3 seconds of EEG** for that mental task. The classifier predicts the class (1 of 4) from that 3 s of data.

So “trials in a run” = the ~48 such segments in that run; “trials in a subject” = all segments from that subject (e.g. ~288 per session, ~576 if both T and E are loaded).

- **Subjects:** **9** (A01–A09). Config: `dataset.subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9]`; use `--subject 1` for a single-subject run.
- **Trials per subject:** Two GDF files per subject — **training (T)** e.g. `A01T.gdf` and **evaluation (E)** e.g. `A01E.gdf`. Each session has **about 288 trials** (6 runs × 48 trials). If both T and E are present, one subject yields **~576 trials**; if only one file is present, **~288 trials** (actual count depends on how many valid class triggers are in the file).

**Which sessions are used for subject 1?** — We use **both** sessions. The loader reads **A01T.gdf** (training) first, then **A01E.gdf** (evaluation). In the official BCI IV 2a dataset, **T has class markers**; **E has no markers**. The system handles this: **T** trials are loaded with labels (0–3); **E** is segmented into fixed-length (e.g. 3 s) **unlabeled trials** (label **-1**). Calibration and pipeline selection use **only labeled trials** (from T). During the stream, **all trials** (T + E) are run through the selected pipeline: for labeled trials we compute accuracy and drift; for unlabeled (E) we only show predictions (no ground truth).
- Each trial is a **3-second** motor imagery segment (BCI IV 2a protocol). The EEG plots show **one trial at a time** (0–3 s on the x-axis); "full subject" means all trials of the subject are streamed, not one long recording. To use 4 seconds per trial, set `dataset.trial_duration_seconds: 4` in `bci_framework/config.yaml` (GDF must have enough data after each cue).
- If the folder is empty, run `./scripts/download_bci_iv_2a.sh` to download the GDF files (~420 MB), or place GDF/MAT files manually (see below).
- Supports **subject-wise** and **cross-subject** evaluation.
- New datasets can be added by implementing the `DatasetLoader` interface in `datasets/base.py`.

### BCI IV 2a Setup

1. Download from [BNCI Horizon](https://bnci-horizon-2020.eu/database/data-sets) or [BCI Competition IV](https://www.bbci.de/competition/iv/).
2. Place GDF files (e.g. `A01T.gdf`, `A01E.gdf`, … `A09T.gdf`, `A09E.gdf`) in `./data/BCI_IV_2a/` (or path set in `config.yaml`).

## Installation

```bash
cd "EEG Agent"
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Usage

From the **project root** (`EEG Agent`). Use `python3` if `python` is not found (e.g. on macOS):

```bash
# Activate venv first (optional): source venv/bin/activate

# Default: calibration, pruning, best pipeline, live simulation, GUI
PYTHONPATH=. python main.py
# Or: PYTHONPATH=. python3 main.py

# Single subject (e.g. subject 1)
PYTHONPATH=. python main.py --subject 1

# Custom config
PYTHONPATH=. python main.py --config path/to/config.yaml

# No GUI (only calibration + logs)
PYTHONPATH=. python main.py --no-gui

# No GUI (calibration + logs only)
PYTHONPATH=. python main.py --no-gui

# Online mode: first N trials calibration, rest = live stream (full subject; with drift monitoring)
PYTHONPATH=. python main.py --online
# Uses all loaded trials: first few for pipeline selection, then streams the rest with the selected pipeline.
# For one full subject (~576 trials), ensure both A0XT.gdf and A0XE.gdf are present so the loader returns all trials.
# Online without GUI
PYTHONPATH=. python main.py --online --no-gui

# Web interface: zoomable EEG view in browser (instead of desktop GUI)
PYTHONPATH=. python main.py --web
# Web + online mode (default: subject 1 / A01)
PYTHONPATH=. python main.py --web --online
# Use a different subject: add --subject N (e.g. --subject 3 for A03)
# The browser should open automatically to http://127.0.0.1:8765. If it doesn't, open that URL manually (use 127.0.0.1, not localhost).
# If you see "connection refused" or "address already in use": run `./scripts/kill_web_server.sh` to free port 8765, then run again. The app will also try 8766, 8767, … if 8765 is in use.
```

Programmatic examples are available in `examples/run_offline.py` and `examples/run_online.py`.

### Full flow (how pipeline selection and live stream work)

The **same dataset** is split once; the **training part** is used to choose the pipeline (offline), and the **test part** is what you see as the "live stream" (simulated trial-by-trial with the selected pipeline).

1. **Load data** — BCI IV 2a from `./data/BCI_IV_2a/` (downloads if missing).
2. **Split (before any stream)** — Trial-wise 80% train / 20% test. Train and test are fixed; no data is streamed yet.
3. **Pipeline selection (offline, once)** — All pipelines are **fitted and evaluated on training trials only** (batch, not stream). Best pipeline is chosen by accuracy, kappa, latency, stability. This happens **before** any live stream.
4. **Snapshots** — Plots and checkpoints saved per pipeline under `results/<experiment_id>/`.
5. **Live stream + GUI** — **Only the test set** is streamed, trial by trial, through the **selected best pipeline**. The GUI shows raw EEG, processed EEG, and predictions. Trials are paced at `trial_duration_sec` (e.g. 3 s) per trial.

So: **pipeline selection does not happen during the stream.** It happens once at the start on the training split; then the stream is only the test split run through that single selected pipeline.

So: **how much data/time for selecting the pipeline?** — Up to **50 training trials** (or all train trials if fewer); time = time to fit and evaluate all pipelines on that data (typically tens of seconds to a couple of minutes). After that, the GUI opens and live streams the test set with the selected pipeline.

**Split options (not only 80/20):**

| Config / CLI | Effect |
|--------------|--------|
| **Default** | Trial-wise **80% train, 20% test** within the same subject (`split_mode: train_test`, `train_test_split: 0.8`). |
| **Sequential (stream-first)** | First N trials = calibration (pipeline selection), **rest = live stream** with selected pipeline. Set `split_mode: "sequential"` and `stream_calibration_trials: 20` in `config.yaml`. Like real BCI: initial segment (1,2,3,…,N) for selection; trials N+1, N+2, … streamed in real time with the best pipeline. No shuffle. |
| **Change ratio** | In `config.yaml`: `train_test_split: 0.9` → 90% train, 10% test (only when `split_mode: train_test`). |
| **LOSO** | Leave-one-subject-out: train on all subjects except one, test on that subject. Set `evaluation_mode: "loso"` or `"cross_subject"`, or run `python main.py --loso 3` to hold out subject 3. |
| **Single subject** | `python main.py --subject 1` uses only subject 1; split is still trial-wise or sequential within that subject. |

**Is 80/20 enough?** — For single-subject BCI it’s a common choice: enough test trials to get a stable accuracy estimate and run a short live stream, while using most data for calibration. If you have very few trials (e.g. &lt;30), consider a higher train ratio (e.g. `train_test_split: 0.9`) so calibration sees more data; if you care about generalisation across subjects, use LOSO.

**Sequential (stream-first) strategy — is it efficient?** — Yes, for deployment-like flow: use the **first few trials** (e.g. 1–20) for pipeline selection, then **stream the rest** (21, 22, …) with the selected pipeline only. Set `split_mode: "sequential"` and `stream_calibration_trials: 20` in `config.yaml`. Tradeoff: fewer calibration trials can make pipeline choice noisier; more calibration gives a stabler choice but less "live" stream. For real-time feel, sequential is efficient.

Or run as a package (from the directory that contains `bci_framework`):

```bash
python -m bci_framework.main
```

## Configuration

Edit `bci_framework/config.yaml` to:

- Set **dataset path**, subjects, train/test split.
- Configure **preprocessing**: mandatory notch/bandpass/reference settings and optional advanced modules (ICA, wavelet, ASR).
- Tune **feature** and **classifier** options (CSP, PSD, LDA, SVM, etc.).
- Control **pipelines**: `auto_generate` combinations or list `explicit` pipelines.
- Set **agent** thresholds: `min_accuracy`, `max_latency_ms`, `max_stability_variance`, `top_n_pipelines`, `re_evaluate_interval_trials`.
- Set **GUI** refresh rate and **logging** paths.

No code changes are required for these options.

## Pipeline Design

Each pipeline is:

**Mandatory preprocessing** → **Optional advanced plugins** → **Feature extraction** → **Classifier**

- **Mandatory preprocessing:** Notch (50/60 Hz), bandpass (0.5–40 Hz or 8–30 Hz for motor imagery), and re-referencing (CAR or Laplacian) applied by `MandatoryPreprocessingPipeline`.
- **Signal quality monitoring:** Variance/kurtosis z-score checks plus optional channel interpolation via `signal_quality`.
- **Advanced preprocessing (optional):** ICA, Wavelet denoising, ASR/rASR, and future artefact modules configured in `advanced_preprocessing.enabled`.
- **Features:** CSP, PSD, Wavelet, Riemannian covariance, Deep placeholder (and raw passthrough for EEGNet).
- **Classifiers:** LDA, SVM, Random Forest, EEGNet (PyTorch), Transformer placeholder.

Unified classifier API: `fit(X, y)`, `predict(X)`, `predict_proba(X)`.

See `docs/preprocessing_architecture.md` for a deep dive into the mandatory vs optional preprocessing flow and online/offline constraints.

## Pipeline Selection Agent

1. **Exploration:** Run all pipelines on a short calibration period.
2. **Pruning:** Remove pipelines with accuracy &lt; threshold, high latency, or unstable predictions.
3. **Exploitation:** Keep top N pipelines; select the best for deployment.
4. **Continuous adaptation:** Re-evaluate periodically (configurable).

Metrics: **Accuracy**, **Cohen’s Kappa**, **Latency**, **Stability** (1 − variance of accuracy over time), **Confidence**.

**How is the best pipeline chosen when several have 1.0 accuracy?** — The selector uses a **composite score**: `0.4×accuracy + 0.3×kappa + 0.2×stability − 0.1×(latency_sec)`. When multiple pipelines tie (e.g. all 1.0 accuracy), the one with **lowest latency** is selected (tie-break by speed).

## Real-time streaming (after best pipeline selected)

After the best pipeline is chosen, the **full test set** is streamed through it like a real BCI session:

- **Full dataset:** All test trials are processed (config: `streaming.stream_full_test_set: true`).
- **Real-time timing:** A delay of `trial_duration_sec` (e.g. 3 s) is applied between trials so total runtime matches real-time (e.g. 24 trials × 3 s ≈ 72 s).
- **GUI:** Window title shows progress (e.g. "Trial 5/24 - left_hand") and live raw/filtered EEG, accuracy, and pipeline comparison.

To disable real-time pacing (run as fast as possible), set `streaming.real_time_timing: false` in `config.yaml`. To stream only the first N trials, set `stream_full_test_set: false` (and the code uses the first 50 trials).

## GUI

- **Live plots:** Raw EEG channels, filtered EEG, feature visualization (e.g. CSP), pipeline accuracy over time, pipeline comparison bar chart.
- **Live prediction:** Selected pipeline runs on streamed chunks; GUI shows class (e.g. left hand, right hand).

Uses **Matplotlib** (TkAgg). No web dashboard.

## Snapshot Logging

For **every** pipeline (selected and rejected):

- Folder: `results/<pipeline_name>/`
- Contents: Raw EEG plot, filtered EEG plot, feature visualization, accuracy curve, confusion matrix, `metrics.json` (metrics and timestamps).

Mandatory for research and reproducibility.

## Adding New Methods

1. **New preprocessing plugin:** Add a class in `preprocessing/` inheriting `AdvancedPreprocessingBase`, implement `fit` / `transform`, and register it in `ADVANCED_PREPROCESSING_REGISTRY` (see `preprocessing/manager.py`).
2. **New feature extractor:** Add a class in `features/` inheriting `FeatureExtractorBase`, register in `features/__init__.py`.
3. **New classifier:** Add a class in `classifiers/` inheriting `ClassifierBase` with `fit`, `predict`, `predict_proba`, register in `classifiers/__init__.py`.
4. **New dataset:** Implement `DatasetLoader` in `datasets/`, implement `load()` and `get_subject_ids()`, register in `datasets/__init__.py`.

New pipelines are then picked up automatically by `PipelineRegistry` when `auto_generate` is true, or add them to `explicit` in `config.yaml`.

## Performance (M4 Pro, 16 GB)

- Pipeline count is capped (e.g. `max_combinations: 20` in config) to avoid running too many heavy pipelines.
- Preprocessing is shared across pipelines where possible (same chain reused).
- Multiprocessing can be added where safe (e.g. per-pipeline evaluation) if needed.

## Future Placeholders

- **Reinforcement learning** pipeline selector (config: `reinforcement_learning_selector`).
- **Real EEG hardware** (OpenBCI, Emotiv) input (config: `real_eeg_hardware`).
- **MCP integration** for LLM control (config: `mcp_llm_integration`).
- **Cloud training** (config: `cloud_training`).

## Research Notes

- BCI IV 2a: 4 classes (left hand, right hand, both feet, tongue), 22 EEG channels, 250 Hz, 9 subjects.
- CSP is most effective with motor band (8–30 Hz) or broad bandpass (1–40 Hz).
- For production, consider subject-specific calibration and continuous adaptation (drift re-evaluation).
- Snapshot logs under `results/` allow offline comparison of all pipelines and hyperparameters.

## Product-Grade Additions (v1.0)

- **No data leakage:** Trial-wise split and LOSO; no preprocessing/CSP fitting on test data.
- **Streaming:** `EEGStreamBuffer`, sliding window (configurable overlap), `dataset.stream_chunk()`.
- **Advanced metrics:** ITR, F1, ROC-AUC, mutual information; logged per pipeline.
- **Model persistence:** Save/load best and all pipelines (`.pkl` / `.pt`); versioned experiments.
- **Drift detection:** Rolling-window accuracy and recalibration triggers; baseline from calibration.
- **Experiment tracking:** Seed control, experiment ID, optional MLflow.
- **Latency budget:** Prune pipelines above 100–300 ms (configurable).
- **Human-in-the-loop placeholder:** Feedback API for correction labels.
- **Tests & CI:** Unit tests for splits, streaming, metrics, pipeline; GitHub Actions template.

See `docs/PRODUCT_SPEC_V1.md` for the full product spec and addressed gaps.

**Minor additions (streaming/GUI sync, latency logging, leaderboard, explainability, example notebook):**

- **Pub-sub:** `utils/pubsub.py` — thread-safe publish/subscribe for streaming and GUI (no blocking).
- **Latency logger:** `utils/latency_logger.py` — per-pipeline ms per window, budget checks.
- **Default HP ranges:** `config.yaml` → `agent.hyperparameter_optimization.param_ranges` (CSP, bandpass, LDA, SVM).
- **Leaderboard:** `gui/leaderboard.py` — sortable table of top-N pipelines with metrics and pruning flags.
- **Explainability stub:** `gui/explainability.py` — `plot_csp_patterns()`, `shap_importance_stub()`.
- **Example notebook:** `examples/bci_automl_minimal_example.ipynb` — load data, calibration, leaderboard, snapshots.

## License

Use and extend for research and product development as needed.
