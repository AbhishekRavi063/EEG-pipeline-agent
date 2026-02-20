#!/usr/bin/env python3
"""
Unified research entry point: leakage-safe LOSO evaluation and paired pipeline comparison.

Runs:
  1. Fixed baselines (CSP+LDA, Riemann+MDM, EEGNet) under identical LOSO
  2. AutoML pipeline selection under same LOSO
  3. Subject-level tables (publication-ready CSV with mean ± std)
  4. Paired statistical comparison (AutoML vs best baseline): permutation, Wilcoxon, t-test, Cohen's d, bootstrap 95% CI
  5. Stability metrics per pipeline
  6. metadata.json (dataset, LOSO, seed, config hash, git commit, timestamp)

No GUI. Single command for full reproducibility.
Usage:
  PYTHONPATH=. python run_research_experiment.py --dataset BNCI2014_001 --subjects 1 2 3
  PYTHONPATH=. python run_research_experiment.py --dataset BNCI2014_001 --subjects 1 2 3 4 5 6 7 8 9 --seed 42 --out-dir results
  PYTHONPATH=. python run_research_experiment.py --dataset BNCI2014_001 --subjects 1 2 3 4 5 6 7 8 9 --fast   # 2-3x faster (fewer pipelines, no ensemble, lighter EEGNet)
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    ap = argparse.ArgumentParser(description="Research experiment: LOSO baselines + AutoML + comparison")
    ap.add_argument("--dataset", default="BNCI2014_001", help="MOABB dataset name")
    ap.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3], help="Subject IDs")
    ap.add_argument("--seed", type=int, default=42, help="Global seed")
    ap.add_argument("--out-dir", default="results", help="Results root (experiment_id subdir created)")
    ap.add_argument("--skip-automl", action="store_true", help="Only run baselines (faster)")
    ap.add_argument("--skip-baselines", action="store_true", help="Only run AutoML")
    ap.add_argument("--fast", action="store_true", help="Faster run: fewer AutoML pipelines, no ensemble, 2-fold CV, lighter EEGNet")
    args = ap.parse_args()

    from bci_framework.utils.experiment import (
        set_seed,
        get_experiment_id,
        set_experiment_id,
        build_research_metadata,
        save_metadata,
    )
    from bci_framework.evaluation.multi_subject_runner import (
        get_default_loso_config,
        get_automl_loso_config,
        run_baselines_loso,
        run_table_for_config,
        get_baseline_config,
        BASELINE_PRESETS,
    )
    from bci_framework.utils.subject_table import (
        build_subject_table,
        TABLE_METRIC_COLUMNS,
        save_subject_level_results,
        save_table_csv,
    )
    from bci_framework.utils.table_comparison import (
        compare_tables_multi_metric_research,
        export_pipeline_comparison_report,
        build_ablation_table,
    )
    from bci_framework.utils.stability_metrics import log_and_export_stability

    set_seed(args.seed)
    experiment_id = get_experiment_id()
    set_experiment_id(experiment_id)
    results_dir = Path(args.out_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    exp_dir = results_dir / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Experiment ID: %s | Dataset: %s | Subjects: %s | Seed: %s%s", experiment_id, args.dataset, args.subjects, args.seed, " | fast=True" if args.fast else "")

    # ---- Baselines (identical LOSO) ----
    baseline_tables: dict[str, list] = {}
    if not args.skip_baselines:
        logger.info("Running fixed baselines (LOSO)%s...", " [fast]" if args.fast else "")
        baseline_rows = run_baselines_loso(args.dataset, args.subjects, fast=args.fast)
        for name, rows in baseline_rows.items():
            table = build_subject_table(rows, pipeline_name=name)
            baseline_tables[name] = table
            save_subject_level_results(
                table, results_dir, experiment_id,
                filename=f"baseline_{name}_subject_table.csv",
                include_summary=True,
            )
            log_and_export_stability(table, name, results_dir, experiment_id)
        # Single combined baseline_subject_table.csv: all baselines stacked with pipeline_name column
        combined_baseline = []
        for name, table in baseline_tables.items():
            for row in table:
                r = copy.deepcopy(row)
                r["pipeline_name"] = name
                combined_baseline.append(r)
        save_table_csv(combined_baseline, exp_dir / "baseline_subject_table.csv")
        logger.info("Baselines done. Wrote baseline_*_subject_table.csv and baseline_subject_table.csv")
    else:
        # Load baseline tables if we need them for comparison and only ran AutoML
        for name in BASELINE_PRESETS:
            path = exp_dir / f"baseline_{name}_subject_table.csv"
            if path.exists():
                import csv
                with open(path, encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    baseline_tables[name] = [row for row in reader if not row.get("subject_id", "").startswith("#")]
            else:
                baseline_tables[name] = []

    # ---- AutoML LOSO ----
    automl_table: list[dict] = []
    ensemble_table: list[dict] = []
    if not args.skip_automl:
        logger.info("Running AutoML selection (LOSO)%s...", " [fast]" if args.fast else "")
        automl_config = get_automl_loso_config(fast=args.fast)
        automl_rows = run_table_for_config(automl_config, args.dataset, args.subjects, pipeline_name="AutoML")
        automl_table = build_subject_table(automl_rows, pipeline_name="AutoML")
        save_subject_level_results(automl_table, results_dir, experiment_id, filename="subject_level_results.csv")
        log_and_export_stability(automl_table, "AutoML", results_dir, experiment_id)
        logger.info("AutoML done. Wrote subject_level_results.csv")

        # ---- AutoML Ensemble (top-2 probability averaging); skipped in --fast ----
        if not args.fast:
            logger.info("Running AutoML Ensemble (top-2, probability averaging)...")
            ensemble_config = copy.deepcopy(get_automl_loso_config(fast=False))
            ensemble_config.setdefault("agent", {})
            ensemble_config["agent"]["ensemble_top_k"] = 2
            ensemble_config["agent"]["top_n_pipelines"] = 2
            ensemble_rows = run_table_for_config(ensemble_config, args.dataset, args.subjects, pipeline_name="AutoML_Ensemble")
            ensemble_table = build_subject_table(ensemble_rows, pipeline_name="AutoML_Ensemble")
            save_subject_level_results(
                ensemble_table, results_dir, experiment_id, filename="automl_ensemble_subject_table.csv", include_summary=True
            )
            log_and_export_stability(ensemble_table, "AutoML_Ensemble", results_dir, experiment_id)
            logger.info("AutoML Ensemble done. Wrote automl_ensemble_subject_table.csv")
    else:
        # Load AutoML / ensemble tables if only baselines were run this time
        path = exp_dir / "subject_level_results.csv"
        if path.exists():
            import csv
            with open(path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                automl_table = [row for row in reader if not str(row.get("subject_id", "")).startswith("#")]
        path_ens = exp_dir / "automl_ensemble_subject_table.csv"
        if path_ens.exists():
            import csv
            with open(path_ens, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                ensemble_table = [row for row in reader if not str(row.get("subject_id", "")).startswith("#")]
        if not path.exists():
            automl_table = []

    # ---- Ablation table (all methods vs Riemann_MDM) ----
    all_for_ablation: dict[str, list] = dict(baseline_tables)
    if automl_table:
        all_for_ablation["AutoML"] = automl_table
    if ensemble_table:
        all_for_ablation["AutoML_Ensemble"] = ensemble_table
    if all_for_ablation:
        ablate_rows, ablate_latex = build_ablation_table(
            all_for_ablation, reference_name="Riemann_MDM", metric="accuracy"
        )
        if ablate_rows:
            import csv
            with open(exp_dir / "ablation_table.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["method", "mean_loso", "std_loso", "p_vs_ref", "effect_size"])
                w.writeheader()
                w.writerows(ablate_rows)
            (exp_dir / "ablation_latex.tex").write_text(ablate_latex, encoding="utf-8")
            logger.info("Wrote ablation_table.csv and ablation_latex.tex")

    # ---- Paired comparison: AutoML vs best baseline ----
    if automl_table and baseline_tables:
        # Best baseline by mean accuracy
        best_baseline_name = None
        best_mean_acc = -1.0
        for name, table in baseline_tables.items():
            if not table:
                continue
            accs = [float(r.get("accuracy", 0) or 0) for r in table if r.get("accuracy") is not None]
            if accs and (sum(accs) / len(accs)) > best_mean_acc:
                best_mean_acc = sum(accs) / len(accs)
                best_baseline_name = name
        if best_baseline_name and baseline_tables[best_baseline_name]:
            comparison = compare_tables_multi_metric_research(
                automl_table,
                baseline_tables[best_baseline_name],
                metrics=TABLE_METRIC_COLUMNS,
                name_1="AutoML",
                name_2=best_baseline_name,
            )
            export_pipeline_comparison_report(
                comparison,
                path_json=str(exp_dir / "pipeline_comparison_report.json"),
                path_csv=str(exp_dir / "pipeline_comparison_table.csv"),
                path_latex=str(exp_dir / "pipeline_comparison_latex.tex"),
            )
            logger.info("Comparison (AutoML vs %s) written to pipeline_comparison_*", best_baseline_name)
        else:
            logger.warning("No baseline table available for comparison")
    else:
        logger.info("Skipping comparison (need both AutoML and baseline tables)")

    # ---- Metadata ----
    config_ref = get_default_loso_config()
    metadata = build_research_metadata(
        dataset=args.dataset,
        evaluation_mode="loso",
        seed=args.seed,
        config=config_ref,
        extra={"subjects": args.subjects, "experiment_id": experiment_id},
    )
    save_metadata(metadata, exp_dir / "metadata.json")
    logger.info("Done. Results in %s", exp_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
