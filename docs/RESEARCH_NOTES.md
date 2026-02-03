# Research Notes – BCI Motor Imagery AutoML

## Dataset: BCI Competition IV 2a

- **Classes:** Left hand, right hand, both feet, tongue (4-class motor imagery).
- **Channels:** 22 EEG, 3 EOG (framework uses 22 EEG).
- **Sampling rate:** 250 Hz.
- **Subjects:** 9 (A01–A09).
- **Sessions:** Training (T) and evaluation (E) GDF files per subject.
- **Trials:** ~288 per session (48 per run × 6 runs).
- **Trigger codes:** 769, 770, 771, 772 for the four classes.

## Pipeline Design Choices

- **Preprocessing order:** Bandpass/notch first to limit bandwidth before ICA or wavelet.
- **CSP:** Fitted on two most frequent classes for binary-like separation; extend to OneVsRestCSP for full 4-class.
- **EEGNet:** Expects raw (or flattened raw) when used with `raw` feature extractor; otherwise receives CSP/PSD features (adapter in classifier).
- **Riemannian:** Tangent space from mean reference covariance; optional `pyriemann` for full Riemannian geometry.

## Metrics

- **Accuracy:** Fraction correct.
- **Cohen’s Kappa:** Agreement beyond chance.
- **Latency:** Mean prediction time per trial (ms).
- **Stability:** 1 − variance of accuracy over calibration chunks.
- **Confidence:** Mean max probability over trials.

## Pruning and Selection

- Pipelines below `min_accuracy`, above `max_latency_ms`, or with high accuracy variance are pruned.
- Top N are kept; best is chosen by composite score (accuracy, kappa, stability, latency penalty).
- Re-evaluation interval can trigger periodic re-calibration (drift adaptation).

## Reproducibility

- All pipelines (selected and rejected) get snapshot logs under `results/<pipeline_name>/`.
- Use `metrics.json` and plots for papers and method comparison.
- Fix random seeds in config or code (e.g. sklearn/PyTorch) for full reproducibility.

## Future Work

- RL-based pipeline selector.
- Real hardware (OpenBCI, Emotiv) streaming.
- MCP/LLM integration for high-level control.
- Cloud training and hyperparameter search.
