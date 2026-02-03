# EEG Preprocessing Architecture

This framework adopts the preprocessing pipeline organisation used in clinical and research-grade EEG systems such as MNE, BrainFlow, and BCI2000. The flow is split into **mandatory baseline conditioning** and **optional advanced modules** that can be toggled per experiment.

```
┌─────────────────────────────┐
│ Raw EEG trials (n_ch × n_s) │
└──────────────┬──────────────┘
               │
               ▼
        [Mandatory Pipeline]
               │
   ┌───────────┴────────────┐
   │  Notch (50/60 Hz)      │  Removes power-line interference
   │  Bandpass (0.5–40 Hz*) │  Physics-informed signal band
   │  Re-reference (CAR/    │  Spatial normalisation, drift removal
   │               Laplacian)│
   └───────────┬────────────┘
               │
               ▼
   Signal quality monitoring (bad channel detection / interpolation)
               │
               ▼
        [Advanced Plugins]
   ┌───────────┼────────────┐
   │ ICA / ASR │ Wavelet    │  Optional artefact attenuation
   │ Quality   │ Filters    │
   └───────────┴────────────┘
               │
               ▼
      Epoching & baseline correction (e.g. −0.2–0 s)
               │
               ▼
       Spatial filtering (CAR/Laplacian/CSP/FBCSP)
               │
               ▼
        Feature extraction
               │
               ▼
          Classification

*Motor imagery tasks automatically tighten the bandpass to 8–30 Hz to track μ/β rhythms.
```

## Why these steps are mandatory

| Stage        | Purpose | Reference practice |
|--------------|---------|--------------------|
| **Notch**    | Suppresses 50/60 Hz mains interference and harmonics that can dwarf cortical rhythms. | Present in almost all FDA/CE-grade EEG amplifiers and BCI systems. |
| **Bandpass** | Rejects DC drift and high-frequency EMG noise, keeping the neurophysiological band of interest. | MNE defaults (0.5–40 Hz) and motor imagery toolboxes (8–30 Hz) match this design. |
| **Re-reference** | Removes common-mode artefacts and equalises channel baselines; CAR is universal, Laplacian emphasises local sources for motor cortex. | BCI2000, OpenBCI, and clinical EEG adopt CAR/Laplacian before any feature computation. We apply referencing after bandpass to avoid spreading line noise; some amplifiers reference earlier, and Laplacian pipelines may re-reference twice. |

These operations are hardware-agnostic and must always run to deliver calibrated features, so they are encapsulated in `MandatoryPreprocessingPipeline`.

### Task-specific high-pass guidance

| Task | Recommended high-pass |
|------|----------------------|
| ERP / cognitive potentials | 0.1–0.3 Hz |
| Motor imagery (MI)         | 8 Hz (μ/β focus) |
| Emotion / affective EEG    | 0.5 Hz |
| Sleep / clinical slow-wave | 0.1 Hz |

The default (0.5–40 Hz) covers most research scenarios, with automatic MI override. Adjust `bandpass_low`/`bandpass_high` per task for publication-grade setups.

During epoching we optionally apply a **baseline correction window (−0.2 s → 0 s)**, which is critical for ERP/affective experiments; motor imagery trials can disable it via config if they operate on cue-locked segments only.

## Optional / research plugins

| Step                   | Description                                  | Online compatible |
|------------------------|----------------------------------------------|-------------------|
| **ICA artefact removal** | Removes blink/ECG components learned during calibration. | ❌ (non-causal, heavy) |
| **ASR (batch)**        | Classic artifact subspace reconstruction.    | ❌ |
| **rASR**               | Recursive / streaming ASR variant (latency trade-off). | ✅ |
| **Wavelet denoising**  | Soft-threshold wavelet coefficients to reduce broadband noise. | ❌ (needs full-window context) |
| **Signal quality monitor** | Variance/kurtosis z-score, bad-channel interpolation, dropout handling. | ✅ |
| **Adaptive filters / channel interpolation** | Extendable via `ADVANCED_PREPROCESSING_REGISTRY`. | depends on implementation |

Advanced steps are configured in `advanced_preprocessing.enabled`. At runtime the manager disables any module that declares `supports_online = False` when `mode: online`. This mirrors industry deployments where low-latency constraints forbid non-causal processing.

## Spatial filtering stage

CSP/FBCSP and Laplacian derivations act as **spatial filters** bridging preprocessing and features. In this framework the spatial step may run twice:

1. **Baseline referencing** (CAR/Laplacian) inside the mandatory pipeline.
2. **Data-driven filters** (CSP/FBCSP) as part of the feature extractor, ensuring reviewers can trace every spatial transform.

## Clinical vs industry vs research

- **Clinical monitoring:** Mandatory pipeline only; audit trails and deterministic latency. Optional steps rarely enabled outside offline review.
- **Research / PhD work:** Combination of ICA and wavelet denoising for high-quality trial averaging; offline mode allows full-window algorithms and batch recalibration.
- **Industrial neurotech / BCI products:** Online mode with causal filters, ASR variants that run in streaming fashion, strict timing telemetry recorded (`PreprocessingManager` logs per-step runtime).

## Online filtering constraints

- **FIR causal filtering** (overlap-save or sample-by-sample) or **IIR Butterworth causal** filtering are enforced in `mode: online`.
- **Zero-phase filtfilt** is deliberately deactivated online to avoid non-causal leakage.
- The `force_causal_filters` flag can override behaviour for advanced deployments.

## Configuration summary

```yaml
mode: "offline"        # or "online"
task: "motor_imagery"  # enables adaptive motor band

preprocessing:
  notch_freq: 50
  notch_quality: 30
  bandpass_low: 0.5
  bandpass_high: 40
  adaptive_motor_band: true
  motor_band_low: 8
  motor_band_high: 30
  reference: "car"     # or "laplacian"

advanced_preprocessing:
  enabled: ["signal_quality", "ica", "wavelet"]
  signal_quality:
    variance_z: 5.0
    kurtosis_z: 5.0
    interpolate: true
  ica:
    n_components: 15
  wavelet:
    wavelet: "db4"
  asr:
    cutoff: 20.0
    window_sec: 0.5
```

Set `mode: online` to automatically drop ICA / wavelet / batch ASR while keeping causal-safe modules (signal quality monitor, rASR) active.

## Example execution scripts

- **Offline calibration and replay:** `python examples/run_offline.py`
- **Online calibration + live stream:** `python examples/run_online.py --subject 1`

See the scripts under `examples/` for programmatic usage with the new preprocessing manager.
