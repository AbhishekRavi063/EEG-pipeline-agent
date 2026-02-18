"""
Multi-subject table runner: LOSO over subjects, return subject-level metrics.

Used by scripts/run_multi_subject_tables.py and the web A/B comparison API.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]


def get_default_loso_config() -> dict:
    """
    Default config for multi-subject tables that produces meaningful results (no GEDAI/leadfield).
    Matches the RSA-stable LOSO setup used in run_loso_with_patches.py --rsa-stable.
    """
    from bci_framework.utils.config_loader import load_config, get_config

    load_config(ROOT / "bci_framework" / "config.yaml")
    cfg = copy.deepcopy(get_config())
    # Disable advanced preprocessing that requires leadfield or may fail on MOABB
    cfg["advanced_preprocessing"] = {"enabled": []}
    # Single explicit pipeline: filter_bank_riemann + logistic_regression (RSA-stable style)
    cfg["pipelines"] = {
        "auto_generate": False,
        "max_combinations": 6,
        "explicit": [["filter_bank_riemann", "logistic_regression"]],
    }
    cfg["features"] = cfg.get("features", {}) | {
        "filter_bank_riemann": {
            "z_score_tangent": True,
            "force_float32": True,
            "rsa": True,
            "use_procrustes": False,
            "use_class_conditional": False,
            "use_temporal": False,
            "use_band_weighting": False,
            "use_subject_weighting": False,
            "use_outlier_detection": False,
            "rsa_stable_mode": True,
        },
        "riemann_tangent_oas": {"z_score_tangent": True, "force_float32": False},
    }
    cfg["classifiers"] = cfg.get("classifiers", {}) | {
        "logistic_regression": {
            "tune_C": True,
            "C": 1.0,
            "C_grid": [0.01, 0.1, 1.0, 10.0],
            "cv_folds": 3,
            "platt_scaling": False,
        },
    }
    cfg["agent"] = {
        "calibration_trials": 80,
        "cv_folds": 3,
        "trial_duration_sec": 3.0,
        "prune_thresholds": {"min_accuracy": 0.0, "max_latency_ms": 500},
    }
    cfg["transfer"] = {
        "enabled": False,
        "method": "none",
        "target_unlabeled_fraction": 0.3,
    }
    cfg["spatial_filter"] = {"enabled": True, "method": "laplacian_auto", "auto_select": False}
    cfg["experiment"] = {"calibration_fraction": 0.3, "subject_weighting": False}
    return cfg


def config_from_preset(feature: str, classifier: str, spatial: str) -> dict:
    """Build config from preset (feature, classifier, spatial) for dropdown-driven comparison."""
    cfg = get_default_loso_config()
    cfg = copy.deepcopy(cfg)
    cfg["pipelines"] = {
        "auto_generate": False,
        "max_combinations": 6,
        "explicit": [[feature, classifier]],
    }
    cfg["spatial_filter"] = {"enabled": True, "method": spatial, "auto_select": False}
    return cfg


def load_config_for_tables(path: str | Path | None) -> dict:
    """Load config from path; if None, use default LOSO config that yields meaningful results."""
    from bci_framework.utils.config_loader import load_config, get_config

    if path is not None:
        path = Path(path)
        if path.exists():
            load_config(path)
            cfg = copy.deepcopy(get_config())
            # When loading from file, ensure LOSO-friendly agent defaults so pruning doesn't drop all pipelines
            cfg.setdefault("agent", {})
            cfg["agent"].setdefault("prune_thresholds", {})
            if cfg["agent"]["prune_thresholds"].get("min_accuracy", 0.5) > 0.35:
                cfg["agent"]["prune_thresholds"]["min_accuracy"] = 0.0
            return cfg
    return get_default_loso_config()


def run_table_for_config(
    config: dict,
    dataset: str,
    subjects: list[int],
    pipeline_name: str = "Pipeline",
) -> list[dict]:
    """
    Run LOSO for each subject with given config; return subject-level rows with test_metrics.
    """
    from bci_framework.preprocessing import subject_standardize_per_subject
    from tests.test_loso_transfer_validation import load_moabb_loso, run_one_loso_fold, MI_WINDOW_PRIMARY

    X, y, subject_ids, fs, ch_names, n_cl, capabilities = load_moabb_loso(
        dataset, subjects, tmin=MI_WINDOW_PRIMARY[0], tmax=MI_WINDOW_PRIMARY[1]
    )
    X = subject_standardize_per_subject(X, subject_ids)
    config = copy.deepcopy(config)
    config["spatial_capabilities"] = capabilities

    rows = []
    for holdout in subjects:
        out = run_one_loso_fold(
            holdout, subjects, X, y, subject_ids, fs, ch_names, n_cl,
            config, return_diagnostics=True,
        )
        cv_mean, test_acc, debug_info, extras = out[0], out[1], out[2], out[3]
        test_metrics = dict(extras.get("test_metrics") or {})
        test_metrics["accuracy"] = test_acc
        row = {
            "subject_id": holdout,
            "test_metrics": test_metrics,
            "n_trials_test": extras.get("n_trials_test"),
        }
        rows.append(row)
        logger.info("Subject %s: test_accuracy=%.4f", holdout, test_acc)
    return rows


def run_ab_comparison(
    dataset: str,
    subjects: list[int],
    config_path_a: str | Path | None = None,
    config_path_b: str | Path | None = None,
    override_b: dict[str, Any] | None = None,
    pipeline_a: dict[str, str] | None = None,
    pipeline_b: dict[str, str] | None = None,
    name_a: str = "Pipeline_A",
    name_b: str = "Pipeline_B",
    test: str = "ttest",
) -> dict[str, Any]:
    """
    Run Table_A and Table_B (same subjects), then compare. Returns dict with
    table_a, table_b, comparison (per-metric p-value and means).
    If pipeline_a / pipeline_b are set (each: feature, classifier, spatial), use preset dropdown configs.
    Else if override_b is set, config_b = config_a merged with override_b.
    """
    from bci_framework.utils.subject_table import build_subject_table, TABLE_METRIC_COLUMNS
    from bci_framework.utils.table_comparison import compare_tables_multi_metric

    if pipeline_a is not None:
        config_a = config_from_preset(
            pipeline_a.get("feature", "filter_bank_riemann"),
            pipeline_a.get("classifier", "logistic_regression"),
            pipeline_a.get("spatial", "laplacian_auto"),
        )
    else:
        config_a = load_config_for_tables(config_path_a)
    rows_a = run_table_for_config(config_a, dataset, subjects, pipeline_name=name_a)
    table_a = build_subject_table(rows_a, pipeline_name=name_a)

    run_b = False
    config_b = None
    if pipeline_b is not None:
        config_b = config_from_preset(
            pipeline_b.get("feature", "filter_bank_riemann"),
            pipeline_b.get("classifier", "logistic_regression"),
            pipeline_b.get("spatial", "laplacian_auto"),
        )
        run_b = True
    elif config_path_b is not None or override_b is not None:
        if override_b is not None:
            config_b = copy.deepcopy(config_a)
            for k, v in override_b.items():
                if isinstance(v, dict) and isinstance(config_b.get(k), dict):
                    config_b[k] = {**config_b.get(k, {}), **v}
                else:
                    config_b[k] = v
        else:
            config_b = load_config_for_tables(config_path_b)
        run_b = True

    if run_b and config_b is not None:
        rows_b = run_table_for_config(config_b, dataset, subjects, pipeline_name=name_b)
        table_b = build_subject_table(rows_b, pipeline_name=name_b)
        comparison = compare_tables_multi_metric(
            table_a, table_b, metrics=TABLE_METRIC_COLUMNS, test=test, name_1=name_a, name_2=name_b
        )
    else:
        table_b = []
        comparison = {}

    return {
        "table_a": table_a,
        "table_b": table_b,
        "comparison": comparison,
        "dataset": dataset,
        "subjects": subjects,
        "name_a": name_a,
        "name_b": name_b,
    }
