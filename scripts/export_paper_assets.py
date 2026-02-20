#!/usr/bin/env python3
"""
Export publication-ready paper assets from a completed research experiment.

Reads results/<experiment_id>/ and produces:
  - Table_1_subject_metrics.csv (subject-level metrics)
  - Table_2_pipeline_comparison.csv (AutoML vs baseline comparison)
  - LaTeX versions
  - JSON summaries

No GUI dependencies. Usage:
  PYTHONPATH=. python scripts/export_paper_assets.py --experiment-id <id> [--results-dir results] [--out-dir paper_assets]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description="Export paper assets from research experiment")
    ap.add_argument("--experiment-id", required=True, help="Experiment ID (subdir under results-dir)")
    ap.add_argument("--results-dir", default="results", help="Results root")
    ap.add_argument("--out-dir", default="paper_assets", help="Output directory for exported assets")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    exp_dir = results_dir / args.experiment_id
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not exp_dir.exists():
        print("Error: experiment dir not found:", exp_dir, file=sys.stderr)
        return 1

    # Table 1: subject-level metrics (from subject_level_results.csv or baseline)
    table_1_path = exp_dir / "subject_level_results.csv"
    if table_1_path.exists():
        rows = []
        header = None
        with open(table_1_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p.strip() for p in line.split(",")]
                if header is None:
                    header = parts
                    continue
                if len(parts) == len(header):
                    rows.append(dict(zip(header, parts)))
        if rows:
            out_csv = out_dir / "Table_1_subject_metrics.csv"
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            print("Wrote", out_csv)
            # LaTeX snippet
            out_tex = out_dir / "Table_1_subject_metrics.tex"
            _write_subject_table_latex(rows, out_tex, "Subject-level metrics (LOSO)")
            print("Wrote", out_tex)
    else:
        print("No subject_level_results.csv found; skipping Table_1")

    # Table 2: pipeline comparison
    comp_json = exp_dir / "pipeline_comparison_report.json"
    comp_csv = exp_dir / "pipeline_comparison_table.csv"
    if comp_json.exists():
        with open(comp_json, encoding="utf-8") as f:
            comparison = json.load(f)
        out_csv = out_dir / "Table_2_pipeline_comparison.csv"
        if comp_csv.exists():
            out_csv.write_text(comp_csv.read_text(), encoding="utf-8")
        else:
            _write_comparison_csv_from_report(comparison, out_csv)
        print("Wrote", out_csv)
        out_tex = out_dir / "Table_2_pipeline_comparison.tex"
        _write_comparison_latex_from_report(comparison, out_tex)
        print("Wrote", out_tex)
        out_json = out_dir / "pipeline_comparison_summary.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2)
        print("Wrote", out_json)
    elif comp_csv.exists():
        (out_dir / "Table_2_pipeline_comparison.csv").write_text(comp_csv.read_text(), encoding="utf-8")
        print("Wrote Table_2_pipeline_comparison.csv (from CSV only)")
    else:
        print("No pipeline_comparison_report.json or .csv found; skipping Table_2")

    # metadata
    meta_path = exp_dir / "metadata.json"
    if meta_path.exists():
        (out_dir / "metadata.json").write_text(meta_path.read_text(), encoding="utf-8")
        print("Wrote metadata.json")
    print("Paper assets in", out_dir)
    return 0


def _write_subject_table_latex(rows: list[dict], path: Path, caption: str) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{%s}" % caption,
        "\\label{tab:subject_metrics}",
        "\\begin{tabular}{" + "l" + "r" * (len(cols) - 1) + "}",
        "\\toprule",
        " & ".join(c.replace("_", " ").title() for c in cols) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(str(row.get(c, "")) for c in cols) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_comparison_csv_from_report(comparison: dict, path: Path) -> None:
    metrics = comparison.get("metrics", {})
    if not metrics:
        return
    rows = []
    for metric, res in metrics.items():
        if isinstance(res, dict) and "error" not in res:
            rows.append({
                "metric": metric,
                "mean_A": res.get("mean_1"),
                "mean_B": res.get("mean_2"),
                "mean_delta": res.get("mean_delta"),
                "p_value_permutation": res.get("p_value_permutation"),
                "cohens_d": res.get("cohens_d"),
                "significant_05": res.get("significant_05"),
            })
    if rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


def _write_comparison_latex_from_report(comparison: dict, path: Path) -> None:
    name_1 = comparison.get("name_1", "A")
    name_2 = comparison.get("name_2", "B")
    metrics = comparison.get("metrics", {})
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Pipeline comparison: %s vs %s (paired permutation test).}" % (name_1, name_2),
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Metric & Mean (%s) & Mean (%s) & $\\Delta$ & $p$ (perm.) & Cohen's $d$ & Sig. \\\\" % (name_1, name_2),
        "\\midrule",
    ]
    for metric, res in (metrics or {}).items():
        if isinstance(res, dict) and "error" not in res:
            m1 = res.get("mean_1", "---")
            m2 = res.get("mean_2", "---")
            delta = res.get("mean_delta", "---")
            p = res.get("p_value_permutation")
            p = "%.4f" % p if p is not None else "---"
            d = res.get("cohens_d")
            d = "%.3f" % d if d is not None else "---"
            sig = "Yes" if res.get("significant_05") else "No"
            lines.append("%s & %s & %s & %s & %s & %s & %s \\\\" % (metric.replace("_", " "), m1, m2, delta, p, d, sig))
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
