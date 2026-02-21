#!/usr/bin/env python3
"""
Run preprocessing evaluation: Physionet MI, GroupKFold only, conditions A (baseline) and B (ICA).
No LOSO. No GEDAI. Output: summary table, ICA vs Baseline comparison, CSV + JSON.

Usage:
  PYTHONPATH=. python scripts/run_preprocessing_evaluation.py --out-dir results/preprocessing_evaluation
  PYTHONPATH=. python scripts/run_preprocessing_evaluation.py --n-subjects 20 --out-dir results/preprocessing_evaluation
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    from bci_framework.evaluation.preprocessing_evaluation import run_full_evaluation, RANDOM_STATE, N_PERM

    p = argparse.ArgumentParser(description="Preprocessing evaluation: Physionet MI, GroupKFold, A (baseline) vs B (ICA).")
    p.add_argument("--out-dir", type=Path, default=None, help="Output directory for CSV and JSON")
    p.add_argument("--n-subjects", type=int, default=None, help="Cap number of subjects (default: all)")
    p.add_argument("--n-perm", type=int, default=N_PERM, help="Permutations for paired test")
    p.add_argument("--seed", type=int, default=RANDOM_STATE, help="Random state")
    args = p.parse_args()

    subjects = list(range(1, args.n_subjects + 1)) if args.n_subjects else None
    out = run_full_evaluation(
        subjects=subjects,
        out_dir=args.out_dir,
        n_perm=args.n_perm,
        random_state=args.seed,
    )

    logger.info("Summary table:")
    for row in out.get("summary_table", []):
        logger.info(
            "  %s | Mean Acc %.4f ± %.4f | Mean AUC %.4f | Mean Kappa %.4f",
            row["Condition"],
            row["Mean_Accuracy"],
            row["Std_Accuracy"],
            row["Mean_AUC"],
            row["Mean_Kappa"],
        )
    comp = out.get("comparison_ica_vs_baseline", {})
    if comp:
        logger.info(
            "ICA vs Baseline: mean_delta=%.4f, p=%.4f, cohens_d=%.4f, 95%% CI=%s",
            comp.get("mean_delta"),
            comp.get("p_value"),
            comp.get("cohens_d"),
            comp.get("bootstrap_ci_95"),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
