#!/usr/bin/env python3
"""
Batch exploration: run multiple pipeline configs across same subjects, then compare all pairs.

Produces Table_1, Table_2, ... Table_K and a comparison report (which A vs B are significant).
Professor's suggestion: "Explore results, see which are interesting and statistically significant."

Usage:
  PYTHONPATH=. python scripts/explore_pipeline_comparisons.py --dataset BNCI2014_001 --subjects 1 2 3 --configs config_a.yaml config_b.yaml config_c.yaml --output-dir results/explore
  PYTHONPATH=. python scripts/explore_pipeline_comparisons.py --dataset BNCI2014_001 --subjects 1 2 3 --config-list list.txt --output-dir results/explore
"""

from __future__ import annotations

import argparse
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
    TABLE_METRIC_COLUMNS,
)
from bci_framework.utils.table_comparison import compare_tables

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch pipeline tables and pairwise comparison report")
    ap.add_argument("--dataset", default="BNCI2014_001", help="MOABB dataset name")
    ap.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3], help="Subject IDs")
    ap.add_argument("--configs", type=str, nargs="+", default=[], help="Config paths (Table_1, Table_2, ...)")
    ap.add_argument("--config-list", type=str, default=None, help="File with one config path per line (alternative to --configs)")
    ap.add_argument("--output-dir", type=str, default="results/explore_pipelines", help="Output directory")
    ap.add_argument("--test", choices=["ttest", "wilcoxon"], default="ttest", help="Paired test")
    ap.add_argument("--metric", type=str, default="accuracy", help="Primary metric for comparison matrix")
    args = ap.parse_args()

    config_paths = list(args.configs)
    if args.config_list:
        config_paths = [p.strip() for p in Path(args.config_list).read_text().splitlines() if p.strip()]
    if not config_paths:
        logger.error("Provide --configs or --config-list")
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tables: list[tuple[str, list[dict]]] = []
    for i, cfg_path in enumerate(config_paths):
        name = f"Table_{i + 1}"
        config = load_config_for_tables(cfg_path)
        rows = run_table_for_config(config, args.dataset, args.subjects, pipeline_name=name)
        table = build_subject_table(rows, pipeline_name=name)
        tables.append((name, table))
        save_table_csv(table, out_dir / f"{name}.csv")
        save_table_json(table, out_dir / f"{name}.json", meta={"dataset": args.dataset, "config": cfg_path})
        logger.info("%s: %d subjects → %s", name, len(table), out_dir / f"{name}.csv")

    # Pairwise comparison matrix (primary metric)
    n = len(tables)
    matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append({"p_value": None, "mean_delta": 0.0, "significant": False})
            else:
                res = compare_tables(
                    tables[i][1], tables[j][1],
                    metric=args.metric,
                    test=args.test,
                    name_1=tables[i][0],
                    name_2=tables[j][0],
                )
                row.append(res)
        matrix.append(row)

    # Summary: which pairs are significant
    summary = []
    for i in range(n):
        for j in range(i + 1, n):
            res = matrix[i][j]
            summary.append({
                "pipeline_a": tables[i][0],
                "pipeline_b": tables[j][0],
                "metric": args.metric,
                "mean_a": res["mean_1"],
                "mean_b": res["mean_2"],
                "mean_delta": res["mean_delta"],
                "p_value": res["p_value"],
                "significant": res["significant"],
            })
            if res["p_value"] is not None:
                sig = " *" if res["significant"] else ""
                logger.info(
                    "%s vs %s: %s %.4f vs %.4f  delta=%.4f  p=%.4f%s",
                    tables[i][0], tables[j][0], args.metric,
                    res["mean_1"], res["mean_2"], res["mean_delta"], res["p_value"], sig,
                )

    report = {
        "dataset": args.dataset,
        "subjects": args.subjects,
        "configs": config_paths,
        "metric": args.metric,
        "test": args.test,
        "pairwise_summary": summary,
        "matrix_indices": [t[0] for t in tables],
        "comparison_matrix": [[{"p_value": m.get("p_value"), "significant": m.get("significant")} for m in r] for r in matrix],
    }
    report_path = out_dir / "comparison_summary.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Comparison summary → %s", report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
