#!/usr/bin/env python3
"""
Denoising comparison: bandpass only vs bandpass+ICA vs bandpass+GEDAI.

Same pipeline (feature + classifier), only advanced_preprocessing.enabled varies.
Requested by professor: "Change the denoising parameters (simple bandpass vs ICA vs GEDAI)
and see how this impacts the classification performance."

Usage:
  PYTHONPATH=. python scripts/run_denoising_comparison.py --dataset BNCI2014_001 --n-subjects 5
  PYTHONPATH=. python scripts/run_denoising_comparison.py --dataset PhysionetMI --n-subjects 10 --out results/denoising_comparison.json
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Same pipeline for all conditions; only denoising changes
DENOISING_CONDITIONS = [
    ("bandpass_only", []),
    ("bandpass_ica", ["ica"]),
    ("bandpass_gedai", ["gedai"]),
]


def get_base_config(dataset_name: str):
    """Config with one pipeline (Tangent_LR), LOSO-friendly; advanced_preprocessing.enabled set by caller."""
    from bci_framework.evaluation.multi_subject_runner import get_baseline_config

    cfg = get_baseline_config("Tangent_LR")
    cfg = copy.deepcopy(cfg)
    # Ensure we use MOABB when dataset is not BCI_IV_2a
    if dataset_name != "BCI_IV_2a":
        cfg["dataset"] = cfg.get("dataset", {}) | {
            "name": "MOABB",
            "dataset_name": dataset_name,
            "paradigm": "motor_imagery",
        }
    return cfg


def run_denoising_comparison(
    dataset: str = "BNCI2014_001",
    n_subjects: int = 5,
    out_path: Path | None = None,
) -> dict:
    """
    Run LOSO for each denoising condition; return subject-level accuracies and summary.
    dataset: MOABB dataset name (BNCI2014_001, PhysionetMI, etc.)
    n_subjects: cap subjects (e.g. 5 for quick run, 9 for BNCI2014_001, 20 for PhysionetMI).
    """
    from bci_framework.evaluation.multi_subject_runner import run_table_for_config

    # Subject list: BNCI2014_001 uses 1..9; PhysionetMI and others use 1..N
    if dataset == "BNCI2014_001":
        subjects = list(range(1, min(n_subjects + 1, 10)))
    else:
        subjects = list(range(1, n_subjects + 1))

    base = get_base_config(dataset)
    # Preserve full advanced_preprocessing (ica/gedai params) from base; only override enabled
    adv = base.get("advanced_preprocessing", {})
    if not isinstance(adv, dict):
        adv = {}
    adv = copy.deepcopy(adv)

    results = {}
    for label, enabled_list in DENOISING_CONDITIONS:
        cfg = copy.deepcopy(base)
        cfg["advanced_preprocessing"] = {**adv, "enabled": enabled_list}
        # GEDAI: allow identity if no leadfield (testing)
        if "gedai" in enabled_list:
            cfg["advanced_preprocessing"].setdefault("gedai", {})
            cfg["advanced_preprocessing"]["gedai"]["use_identity_if_missing"] = True
            cfg["advanced_preprocessing"]["gedai"]["require_real_leadfield"] = False

        logger.info("Running condition %s (enabled=%s) ...", label, enabled_list)
        rows = run_table_for_config(cfg, dataset, subjects, pipeline_name=label)
        accs = [r.get("test_metrics", {}).get("accuracy", r.get("accuracy")) for r in rows]
        accs = [a for a in accs if a is not None]
        if not accs:
            accs = []
        mean_acc = sum(accs) / len(accs) if accs else 0.0
        results[label] = {
            "mean_accuracy": mean_acc,
            "std_accuracy": (sum((a - mean_acc) ** 2 for a in accs) / len(accs)) ** 0.5 if len(accs) > 1 else 0.0,
            "n_subjects": len(accs),
            "subject_accuracies": accs,
        }
        logger.info("  %s: %.2f%% ± %.2f%% (n=%d)", label, mean_acc * 100, results[label]["std_accuracy"] * 100, len(accs))

    out = {
        "dataset": dataset,
        "subjects": subjects,
        "conditions": list(r["mean_accuracy"] for r in results.values()),
        "results": results,
    }
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        logger.info("Wrote %s", out_path)
    return out


def main():
    p = argparse.ArgumentParser(description="Compare bandpass vs ICA vs GEDAI denoising on classification.")
    p.add_argument("--dataset", default="BNCI2014_001", help="MOABB dataset (e.g. BNCI2014_001, PhysionetMI)")
    p.add_argument("--n-subjects", type=int, default=5, help="Max subjects (default 5; use 9 for full BNCI2014_001)")
    p.add_argument("--out", type=Path, default=None, help="Output JSON path (e.g. results/denoising_comparison.json)")
    args = p.parse_args()
    run_denoising_comparison(dataset=args.dataset, n_subjects=args.n_subjects, out_path=args.out)


if __name__ == "__main__":
    main()
