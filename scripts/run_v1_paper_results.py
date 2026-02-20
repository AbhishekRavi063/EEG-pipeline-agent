#!/usr/bin/env python3
"""
v1.0 paper results: FAST CLEAN PAPER RESULTS (NO LEAKAGE).

Rules: No hyperparameter tuning based on LOSO; no test-set-informed changes;
all improvements defined before evaluation; strict LOSO only; seed 42.

STEP 1: Freeze benchmark as v1_clean_benchmark (5 baselines + AutoML).
STEP 2: Euclidean Alignment implemented in bci_framework/features/euclidean_alignment.py.
STEP 3: Run EA+Tangent_LR as v1_ea_alignment.
STEP 4: Generate final paper table (7 methods), LaTeX/CSV/JSON.
STEP 5: Sanity assertions + leakage tests.
STEP 6: Freeze note.

Usage:
  PYTHONPATH=. python scripts/run_v1_paper_results.py --out-dir results
  PYTHONPATH=. python scripts/run_v1_paper_results.py --out-dir results --skip-benchmark  # EA only
  PYTHONPATH=. python scripts/run_v1_paper_results.py --out-dir results --skip-ea        # Benchmark only
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

V1_BENCHMARK_ID = "v1_clean_benchmark"
V1_EA_ID = "v1_ea_alignment"
V1_FINAL_TABLE_ID = "v1_final_paper_tables"
SEED = 42
DEFAULT_SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8, 9]


def _run_step1_benchmark(out_dir: Path, dataset: str, subjects: list[int]) -> None:
    """STEP 1: Run 9-subject LOSO benchmark (5 baselines + AutoML), export all."""
    from bci_framework.utils.experiment import set_seed, set_experiment_id, build_research_metadata, save_metadata
    from bci_framework.evaluation.multi_subject_runner import (
        get_default_loso_config,
        get_automl_loso_config,
        run_baselines_loso,
        run_table_for_config,
        get_baseline_config,
        BASELINE_PRESETS,
    )
    from bci_framework.utils.subject_table import build_subject_table, TABLE_METRIC_COLUMNS, save_subject_level_results, save_table_csv
    from bci_framework.utils.table_comparison import (
        compare_tables_multi_metric_research,
        export_pipeline_comparison_report,
        build_ablation_table,
    )
    from bci_framework.utils.stability_metrics import log_and_export_stability

    set_seed(SEED)
    set_experiment_id(V1_BENCHMARK_ID)
    exp_dir = out_dir / V1_BENCHMARK_ID
    exp_dir.mkdir(parents=True, exist_ok=True)

    logger.info("STEP 1: v1_clean_benchmark | Dataset=%s Subjects=%s Seed=%s", dataset, subjects, SEED)

    # Baselines (do not modify pipelines)
    baseline_tables: dict[str, list] = {}
    baseline_rows = run_baselines_loso(dataset, subjects, fast=False)
    for name, rows in baseline_rows.items():
        table = build_subject_table(rows, pipeline_name=name)
        baseline_tables[name] = table
        save_subject_level_results(table, out_dir, V1_BENCHMARK_ID, filename=f"baseline_{name}_subject_table.csv", include_summary=True)
        log_and_export_stability(table, name, out_dir, V1_BENCHMARK_ID)
    combined = []
    for name, table in baseline_tables.items():
        for row in table:
            r = copy.deepcopy(row)
            r["pipeline_name"] = name
            combined.append(r)
    save_table_csv(combined, exp_dir / "baseline_subject_table.csv")
    logger.info("Baselines done.")

    # AutoML (no ensemble)
    automl_config = get_automl_loso_config(fast=False)
    automl_rows = run_table_for_config(automl_config, dataset, subjects, pipeline_name="AutoML")
    automl_table = build_subject_table(automl_rows, pipeline_name="AutoML")
    save_subject_level_results(automl_table, out_dir, V1_BENCHMARK_ID, filename="subject_level_results.csv")
    log_and_export_stability(automl_table, "AutoML", out_dir, V1_BENCHMARK_ID)
    logger.info("AutoML done.")

    # Ablation (all vs Riemann_MDM)
    all_for_ablation = dict(baseline_tables)
    all_for_ablation["AutoML"] = automl_table
    ablate_rows, ablate_latex = build_ablation_table(all_for_ablation, reference_name="Riemann_MDM", metric="accuracy")
    if ablate_rows:
        with open(exp_dir / "ablation_table.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["method", "mean_loso", "std_loso", "p_vs_ref", "effect_size"])
            w.writeheader()
            w.writerows(ablate_rows)
        (exp_dir / "ablation_latex.tex").write_text(ablate_latex, encoding="utf-8")

    # AutoML vs best baseline (permutation, bootstrap CI)
    best_baseline_name = None
    best_mean_acc = -1.0
    for name, table in baseline_tables.items():
        if not table:
            continue
        accs = [float(r.get("accuracy", 0) or 0) for r in table if r.get("accuracy") is not None]
        if accs and (sum(accs) / len(accs)) > best_mean_acc:
            best_mean_acc = sum(accs) / len(accs)
            best_baseline_name = name
    if best_baseline_name and automl_table:
        comparison = compare_tables_multi_metric_research(
            automl_table, baseline_tables[best_baseline_name],
            metrics=TABLE_METRIC_COLUMNS, name_1="AutoML", name_2=best_baseline_name,
        )
        export_pipeline_comparison_report(
            comparison,
            path_json=str(exp_dir / "pipeline_comparison_report.json"),
            path_csv=str(exp_dir / "pipeline_comparison_table.csv"),
            path_latex=str(exp_dir / "pipeline_comparison_latex.tex"),
        )
        logger.info("Comparison (AutoML vs %s) written.", best_baseline_name)

    # Metadata
    config_ref = get_default_loso_config()
    metadata = build_research_metadata(
        dataset=dataset, evaluation_mode="loso", seed=SEED, config=config_ref,
        extra={"subjects": subjects, "experiment_id": V1_BENCHMARK_ID, "experiment_name": V1_BENCHMARK_ID},
    )
    save_metadata(metadata, exp_dir / "metadata.json")
    logger.info("STEP 1 done. Results in %s", exp_dir)


def _run_step3_ea_alignment(out_dir: Path, dataset: str, subjects: list[int]) -> None:
    """STEP 3: Run EA+Tangent_LR 9-subject LOSO, save as v1_ea_alignment."""
    from bci_framework.utils.experiment import set_seed, set_experiment_id, build_research_metadata, save_metadata
    from bci_framework.evaluation.multi_subject_runner import get_ea_tangent_lr_config, run_table_for_config, get_baseline_config, run_baselines_loso
    from bci_framework.utils.subject_table import build_subject_table, save_subject_level_results
    from bci_framework.utils.table_comparison import compare_tables_multi_metric_research, export_pipeline_comparison_report
    from bci_framework.utils.stability_metrics import log_and_export_stability

    set_seed(SEED)
    set_experiment_id(V1_EA_ID)
    exp_dir = out_dir / V1_EA_ID
    exp_dir.mkdir(parents=True, exist_ok=True)

    logger.info("STEP 3: v1_ea_alignment | EA+Tangent_LR LOSO")

    cfg = get_ea_tangent_lr_config()
    rows = run_table_for_config(cfg, dataset, subjects, pipeline_name="EA_Tangent_LR")
    table = build_subject_table(rows, pipeline_name="EA_Tangent_LR")
    save_subject_level_results(table, out_dir, V1_EA_ID, filename="ea_tangent_lr_subject_table.csv", include_summary=True)
    log_and_export_stability(table, "EA_Tangent_LR", out_dir, V1_EA_ID)

    # Load Riemann_MDM from benchmark for Δ and p-value
    benchmark_dir = out_dir / V1_BENCHMARK_ID
    ref_table = []
    ref_path = benchmark_dir / "baseline_Riemann_MDM_subject_table.csv"
    if ref_path.exists():
        with open(ref_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            ref_table = [r for r in reader if not str(r.get("subject_id", "")).startswith("#")]
    if ref_table and table:
        comparison = compare_tables_multi_metric_research(
            table, ref_table, name_1="EA_Tangent_LR", name_2="Riemann_MDM",
        )
        export_pipeline_comparison_report(
            comparison,
            path_json=str(exp_dir / "ea_vs_riemann_mdm_report.json"),
            path_csv=str(exp_dir / "ea_vs_riemann_mdm_table.csv"),
            path_latex=str(exp_dir / "ea_vs_riemann_mdm_latex.tex"),
        )
        acc_metric = comparison.get("metrics", {}).get("accuracy", {})
        logger.info(
            "EA_Tangent_LR vs Riemann_MDM: Δ=%.4f p=%.4f Cohen's d=%.3f",
            acc_metric.get("mean_delta") or 0,
            acc_metric.get("p_value_permutation") or 0,
            acc_metric.get("cohens_d") or 0,
        )

    metadata = build_research_metadata(
        dataset=dataset, evaluation_mode="loso", seed=SEED, config=cfg,
        extra={"subjects": subjects, "experiment_id": V1_EA_ID, "experiment_name": V1_EA_ID},
    )
    save_metadata(metadata, exp_dir / "metadata.json")
    logger.info("STEP 3 done. Results in %s", exp_dir)


def _run_step4_final_tables(out_dir: Path) -> None:
    """STEP 4: Final paper table (7 methods): LaTeX, CSV, JSON."""
    from bci_framework.utils.table_comparison import build_ablation_table

    exp_dir = out_dir / V1_FINAL_TABLE_ID
    exp_dir.mkdir(parents=True, exist_ok=True)
    benchmark_dir = out_dir / V1_BENCHMARK_ID
    ea_dir = out_dir / V1_EA_ID

    all_tables: dict[str, list] = {}
    for name in ["CSP_LDA", "Riemann_MDM", "Tangent_LR", "FilterBankRiemann", "EEGNet"]:
        path = benchmark_dir / f"baseline_{name}_subject_table.csv"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                all_tables[name] = [r for r in reader if not str(r.get("subject_id", "")).startswith("#")]
    # AutoML
    path = benchmark_dir / "subject_level_results.csv"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            all_tables["AutoML"] = [r for r in reader if not str(r.get("subject_id", "")).startswith("#")]
    # EA_Tangent_LR
    path = ea_dir / "ea_tangent_lr_subject_table.csv"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            all_tables["EA_Tangent_LR"] = [r for r in reader if not str(r.get("subject_id", "")).startswith("#")]

    if not all_tables:
        logger.warning("STEP 4: No tables found; run STEP 1 and STEP 3 first.")
        return

    ablate_rows, ablate_latex = build_ablation_table(all_tables, reference_name="Riemann_MDM", metric="accuracy")
    if not ablate_rows:
        logger.warning("STEP 4: build_ablation_table returned no rows.")
        return

    # CSV
    with open(exp_dir / "final_paper_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["method", "mean_loso", "std_loso", "p_vs_ref", "effect_size"])
        w.writeheader()
        w.writerows(ablate_rows)
    # LaTeX
    (exp_dir / "final_paper_table.tex").write_text(ablate_latex, encoding="utf-8")
    # JSON (with reproducibility metadata)
    payload = {
        "experiment_name": V1_FINAL_TABLE_ID,
        "reference": "Riemann_MDM",
        "metric": "accuracy",
        "seed": SEED,
        "methods": ablate_rows,
        "source_experiments": [V1_BENCHMARK_ID, V1_EA_ID],
    }
    with open(exp_dir / "final_paper_table.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("STEP 4 done. Final table in %s (CSV, LaTeX, JSON)", exp_dir)


def _run_step5_sanity_check(out_dir: Path) -> None:
    """STEP 5: Assertions and leakage tests."""
    import numpy as np

    errors = []

    # Assert: no train/test overlap (covered by leakage tests)
    # Assert: alignment does not use target (assertion inside EA feature fit)
    # Assert: seed reproducibility (we use set_seed(42))
    # Assert: feature dimension consistent across folds (log only; optional)

    logger.info("STEP 5: Running leakage tests (pytest)...")
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_leakage_guard.py", "-v", "--tb=short"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        errors.append("Leakage tests failed: " + (result.stderr or result.stdout or "")[:500])
    else:
        logger.info("Leakage tests passed.")

    if errors:
        for e in errors:
            logger.error("%s", e)
        raise RuntimeError("STEP 5 sanity check had errors")
    logger.info("STEP 5 done. Sanity check OK.")


def _run_step6_freeze_note(out_dir: Path) -> None:
    """STEP 6: Write freeze note."""
    exp_dir = out_dir / V1_FINAL_TABLE_ID
    exp_dir.mkdir(parents=True, exist_ok=True)
    note = """# v1.0 Paper Results — FROZEN

- Do NOT re-run with different grids.
- Do NOT adjust components (e.g. CSP n_components).
- Do NOT adjust bands (e.g. FilterBankRiemann).
- Benchmark: v1_clean_benchmark (5 baselines + AutoML).
- EA alignment: v1_ea_alignment (EA+Tangent_LR, fixed C grid).
- Final table: 7 methods, p vs Riemann_MDM, Cohen's d.
- Seed: 42. LOSO only. No test-set-informed changes.
"""
    (exp_dir / "FREEZE_NOTE.md").write_text(note, encoding="utf-8")
    logger.info("STEP 6 done. Freeze note written to %s/FREEZE_NOTE.md", exp_dir)


def main() -> int:
    ap = argparse.ArgumentParser(description="v1.0 paper results (no leakage)")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results", help="Results root")
    ap.add_argument("--dataset", default="BNCI2014_001")
    ap.add_argument("--subjects", type=int, nargs="+", default=DEFAULT_SUBJECTS)
    ap.add_argument("--skip-benchmark", action="store_true", help="Skip STEP 1 (use existing v1_clean_benchmark)")
    ap.add_argument("--skip-ea", action="store_true", help="Skip STEP 3 (EA+Tangent_LR)")
    ap.add_argument("--skip-final-table", action="store_true", help="Skip STEP 4")
    ap.add_argument("--skip-tests", action="store_true", help="Skip STEP 5 leakage tests")
    args = ap.parse_args()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        if not args.skip_benchmark:
            _run_step1_benchmark(args.out_dir, args.dataset, args.subjects)
        else:
            logger.info("Skipping STEP 1 (benchmark)")

        if not args.skip_ea:
            _run_step3_ea_alignment(args.out_dir, args.dataset, args.subjects)
        else:
            logger.info("Skipping STEP 3 (EA+Tangent_LR)")

        if not args.skip_final_table:
            _run_step4_final_tables(args.out_dir)
        else:
            logger.info("Skipping STEP 4 (final table)")

        if not args.skip_tests:
            _run_step5_sanity_check(args.out_dir)
        else:
            logger.info("Skipping STEP 5 (tests)")

        _run_step6_freeze_note(args.out_dir)

        logger.info("v1 paper results pipeline complete.")
        return 0
    except Exception as e:
        logger.exception("%s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
