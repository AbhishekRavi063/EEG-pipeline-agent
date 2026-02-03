# BCI AutoML Platform — Examples

## Minimal example notebook

**`bci_automl_minimal_example.ipynb`** demonstrates:

1. Load dataset (synthetic motor-imagery-like EEG)
2. Trial-wise split (no leakage)
3. Run calibration and pruning
4. Show pipeline leaderboard (sortable table)
5. Select best pipeline
6. Snapshot saving (plots + JSON)
7. Optional: CSP pattern visualization (explainability stub)

Run from project root:

```bash
cd "EEG Agent"
jupyter notebook examples/bci_automl_minimal_example.ipynb
```

Or from `examples/` with `PYTHONPATH=..` so that `bci_framework` is importable.

## Synthetic EEG for CI

Use `generate_synthetic_mi_eeg_for_ci()` from `bci_framework.datasets.synthetic_eeg` in tests for fast, deterministic pipeline tests without real data.
