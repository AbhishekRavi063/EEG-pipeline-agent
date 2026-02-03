# Usage Guide

## Quick Start

1. **Install:**  
   `pip install -r requirements.txt`

2. **Data:**  
   Put BCI IV 2a GDF files in `./data/BCI_IV_2a/` (or set `dataset.data_dir` in `config.yaml`).

3. **Run:**  
   From project root:  
   `PYTHONPATH=. python main.py`

4. **No GUI:**  
   `PYTHONPATH=. python main.py --no-gui`

## Adding a New Preprocessing Method

1. Create `bci_framework/preprocessing/my_filter.py`.
2. Subclass `PreprocessingBase`, implement `fit(X, y=None)` and `transform(X)`.
3. In `bci_framework/preprocessing/__init__.py`, import the class and add it to `PREPROCESSING_REGISTRY` with a string key (e.g. `"my_filter"`).
4. Add parameters to `config.yaml` under `preprocessing.my_filter` if needed.
5. Pipelines will pick it up when `pipelines.auto_generate` is true, or add an explicit pipeline in `pipelines.explicit`.

## Adding a New Feature Extractor

1. Create `bci_framework/features/my_feature.py`.
2. Subclass `FeatureExtractorBase`, implement `fit(X, y)` and `transform(X)` (output shape `(n_trials, n_features)`).
3. Register in `bci_framework/features/__init__.py` in `FEATURE_REGISTRY`.
4. Optionally add config under `features.my_feature` in `config.yaml`.

## Adding a New Classifier

1. Create `bci_framework/classifiers/my_clf.py`.
2. Subclass `ClassifierBase`, implement `fit(X, y)`, `predict(X)`, and optionally `predict_proba(X)`.
3. Register in `bci_framework/classifiers/__init__.py` in `CLASSIFIER_REGISTRY`.
4. Add config under `classifiers.my_clf` in `config.yaml` if needed.

## Adding a New Dataset

1. Create `bci_framework/datasets/my_dataset.py`.
2. Implement `DatasetLoader`: `load()`, `get_subject_ids()`.
3. Return an `EEGDataset` (or dict of subject_id → EEGDataset).
4. Register in `bci_framework/datasets/__init__.py` in `DATASET_REGISTRY` with a name (e.g. `"MY_DATASET"`).
5. Set `dataset.name: "MY_DATASET"` and `dataset.data_dir` in `config.yaml`.

## Config Highlights

- **dataset.data_dir:** Where to find (or download) data.
- **dataset.subjects:** Which subjects to load.
- **pipelines.auto_generate / max_combinations / explicit:** How to build pipelines.
- **agent.calibration_trials, top_n_pipelines, prune_thresholds:** Selection behavior.
- **logging.results_dir, save_all_pipelines:** Where and whether to save snapshots.

## Results Layout

After a run, `results/` (or your `logging.results_dir`) contains:

- `results/<pipeline_name>/raw_eeg.png`
- `results/<pipeline_name>/filtered_eeg.png`
- `results/<pipeline_name>/features.png`
- `results/<pipeline_name>/accuracy_curve.png`
- `results/<pipeline_name>/confusion_matrix.png`
- `results/<pipeline_name>/metrics.json`

Use these for reporting and comparison.
