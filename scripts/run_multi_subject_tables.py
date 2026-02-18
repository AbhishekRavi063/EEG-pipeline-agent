#!/usr/bin/env python3
"""
Multi-subject tables (Table_1, Table_2) and optional A/B statistical comparison.

Professor's suggestion: run pipeline on all subjects → Table_X (rows=subjects, cols=metrics).
Run same pipeline with one setting changed → Table_2. Compare Table_1 vs Table_2 (p-values).

Usage:
  PYTHONPATH=. python scripts/run_multi_subject_tables.py --dataset BNCI2014_001 --subjects 1 2 3 --config bci_framework/config.yaml --output-dir results/tables
  PYTHONPATH=. python scripts/run_multi_subject_tables.py --dataset BNCI2014_001 --subjects 1 2 3 --config config_a.yaml --config-b config_b.yaml --output-dir results/tables

Requires: MOABB (pip install moabb). Uses LOSO: train on N-1 subjects, test on holdout.
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

from bci_framework.evaluation import run_table_for_config, load_config_for_tables
from bci_framework.utils.subject_table import (
    build_subject_table,
    save_table_csv,
    save_table_json,
    load_table_json,
    TABLE_METRIC_COLUMNS,
)
from bci_framework.utils.table_comparison import compare_tables_multi_metric

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    ap = argparse.ArgumentParser(description="Multi-subject tables (Table_1, Table_2) and A/B comparison")
    ap.add_argument("--dataset", default="BNCI2014_001", help="MOABB dataset name")
    ap.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3], help="Subject IDs")
    ap.add_argument("--config", type=str, default=None, help="Config path for Table_1 (default: bci_framework/config.yaml)")
    ap.add_argument("--config-b", type=str, default=None, help="Config path for Table_2 (if set, run A/B comparison)")
    ap.add_argument("--override-b", type=str, default=None, help="JSON (or path to .json) to merge over config A for Pipeline B (e.g. one setting changed). Example: '{\"pipelines\":{\"explicit\":[[\"filter_bank_riemann\",\"rsa_mlp\"]]}}'")
    ap.add_argument("--output-dir", type=str, default="results/multi_subject_tables", help="Output directory")
    ap.add_argument("--test", choices=["ttest", "wilcoxon"], default="ttest", help="Paired test for comparison")
    ap.add_argument("--name-a", type=str, default="Table_1", help="Label for first pipeline")
    ap.add_argument("--name-b", type=str, default="Table_2", help="Label for second pipeline")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config_a = load_config_for_tables(args.config)
    rows_1 = run_table_for_config(config_a, args.dataset, args.subjects, pipeline_name=args.name_a)
    table_1 = build_subject_table(rows_1, pipeline_name=args.name_a)
    save_table_csv(table_1, out_dir / "Table_1.csv")
    save_table_json(table_1, out_dir / "Table_1.json", meta={"dataset": args.dataset, "subjects": args.subjects, "pipeline": args.name_a})
    logger.info("Table_1: %d subjects → %s", len(table_1), out_dir / "Table_1.csv")

    if args.config_b is None and args.override_b is None:
        logger.info("Done. No --config-b or --override-b; only Table_1 generated.")
        return 0

    if args.override_b is not None:
        raw = args.override_b.strip()
        if raw.startswith("{"):
            override_b = json.loads(raw)
        else:
            override_b = json.loads(Path(args.override_b).read_text())
        config_b = copy.deepcopy(config_a)
        for k, v in override_b.items():
            if isinstance(v, dict) and isinstance(config_b.get(k), dict):
                config_b[k] = {**config_b.get(k, {}), **v}
            else:
                config_b[k] = v
    else:
        config_b = load_config_for_tables(args.config_b)
    rows_2 = run_table_for_config(config_b, args.dataset, args.subjects, pipeline_name=args.name_b)
    table_2 = build_subject_table(rows_2, pipeline_name=args.name_b)
    save_table_csv(table_2, out_dir / "Table_2.csv")
    save_table_json(table_2, out_dir / "Table_2.json", meta={"dataset": args.dataset, "subjects": args.subjects, "pipeline": args.name_b})
    logger.info("Table_2: %d subjects → %s", len(table_2), out_dir / "Table_2.csv")

    comparison = compare_tables_multi_metric(
        table_1, table_2,
        metrics=TABLE_METRIC_COLUMNS,
        test=args.test,
        name_1=args.name_a,
        name_2=args.name_b,
    )
    report_path = out_dir / "comparison_report.json"
    with open(report_path, "w") as f:
        json.dump(comparison, f, indent=2)
    logger.info("Comparison report → %s", report_path)
    for metric, res in comparison.items():
        if res.get("p_value") is not None:
            sig = " *" if res.get("significant") else ""
            logger.info(
                "  %s: %s=%.4f vs %s=%.4f  delta=%.4f  p=%.4f%s",
                metric, args.name_a, res["mean_1"], args.name_b, res["mean_2"],
                res["mean_delta"], res["p_value"], sig,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
