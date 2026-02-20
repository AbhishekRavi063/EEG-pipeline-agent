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


def get_automl_loso_config(fast: bool = False) -> dict:
    """
    Config for AutoML LOSO: agent selects best pipeline per fold from multiple options.
    Same LOSO conditions as baselines; pipelines auto-generated (capped).
    If fast=True: fewer pipelines (4), fewer calibration trials (50), 2-fold CV (~2–3× faster).
    """
    cfg = get_default_loso_config()
    cfg = copy.deepcopy(cfg)
    if fast:
        cfg["pipelines"] = {
            "auto_generate": True,
            "max_combinations": 4,
            "explicit": [],
        }
        cfg["agent"]["calibration_trials"] = 50
        cfg["agent"]["cv_folds"] = 2
        cfg["agent"]["prune_thresholds"] = {"min_accuracy": 0.0, "max_latency_ms": 500}
    else:
        cfg["pipelines"] = {
            "auto_generate": True,
            "max_combinations": 8,
            "explicit": [],
        }
        cfg["agent"]["prune_thresholds"] = {"min_accuracy": 0.0, "max_latency_ms": 500}
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
    # CSP_LDA: more components for cross-subject (paper target 38-45%)
    if feature == "csp" and classifier == "lda":
        cfg.setdefault("features", {}).setdefault("csp", {})
        cfg["features"]["csp"]["n_components"] = 8
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


# Fixed baselines for research comparison (identical LOSO conditions)
# Riemann_MDM: MDM expects flattened covariances, not tangent space -> use "covariance" feature
# Tangent_LR: OAS cov -> tangent space -> StandardScaler -> LogisticRegression (C tuned inner CV)
# FilterBankRiemann: 5 bands (4-8..24-32 Hz), OAS cov, tangent concat, scale, LR
BASELINE_PRESETS = {
    "CSP_LDA": {"feature": "csp", "classifier": "lda", "spatial": "laplacian_auto"},
    "Riemann_MDM": {"feature": "covariance", "classifier": "mdm", "spatial": "laplacian_auto"},
    "Tangent_LR": {"feature": "riemann_tangent_oas", "classifier": "logistic_regression", "spatial": "laplacian_auto"},
    "FilterBankRiemann": {"feature": "filter_bank_riemann", "classifier": "logistic_regression", "spatial": "laplacian_auto"},
    "EEGNet": {"feature": "raw", "classifier": "eegnet", "spatial": "car"},
}


def get_baseline_config(name: str, fast: bool = False) -> dict:
    """Get config for a fixed baseline (CSP_LDA, Riemann_MDM, Tangent_LR, FilterBankRiemann, EEGNet)."""
    preset = BASELINE_PRESETS.get(name)
    if not preset:
        raise KeyError("Unknown baseline %r. Use one of %s" % (name, list(BASELINE_PRESETS)))
    cfg = config_from_preset(
        preset["feature"],
        preset["classifier"],
        preset["spatial"],
    )
    # Tangent_LR: C grid including 100 and 1000 (paper target 38-42%)
    if name == "Tangent_LR":
        cfg.setdefault("classifiers", {})
        cfg["classifiers"].setdefault("logistic_regression", {})
        cfg["classifiers"]["logistic_regression"]["C_grid"] = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        cfg["classifiers"]["logistic_regression"]["tune_C"] = True
        cfg["classifiers"]["logistic_regression"]["cv_folds"] = 3
        cfg.setdefault("features", {})
        cfg["features"].setdefault("riemann_tangent_oas", {})
        cfg["features"]["riemann_tangent_oas"]["z_score_tangent"] = True
    # FilterBankRiemann: bands 4-8, 8-12, 12-16, 16-24, 24-32 Hz; OAS; RSA for cross-subject (paper target 40-45%)
    if name == "FilterBankRiemann":
        cfg.setdefault("features", {})
        cfg["features"].setdefault("filter_bank_riemann", {})
        cfg["features"]["filter_bank_riemann"]["bands"] = [(4, 8), (8, 12), (12, 16), (16, 24), (24, 32)]
        cfg["features"]["filter_bank_riemann"]["use_oas"] = True
        cfg["features"]["filter_bank_riemann"]["z_score_tangent"] = True
        cfg["features"]["filter_bank_riemann"]["rsa"] = True  # cross-subject alignment (subject_ids passed in LOSO)
        cfg["features"]["filter_bank_riemann"]["rsa_stable_mode"] = False
        cfg.setdefault("classifiers", {})
        cfg["classifiers"].setdefault("logistic_regression", {})
        cfg["classifiers"]["logistic_regression"]["C_grid"] = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        cfg["classifiers"]["logistic_regression"]["tune_C"] = True
        cfg["classifiers"]["logistic_regression"]["cv_folds"] = 3
    # EEGNet (non-fast): 200 epochs, dropout 0.4 for paper target ≥35%
    if name == "EEGNet":
        cfg.setdefault("classifiers", {}).setdefault("eegnet", {})
        if not fast:
            cfg["classifiers"]["eegnet"]["epochs"] = 200
            cfg["classifiers"]["eegnet"]["dropout"] = 0.4
    if fast and name == "EEGNet":
        cfg.setdefault("classifiers", {}).setdefault("eegnet", {})
        cfg["classifiers"]["eegnet"]["epochs"] = 60
        cfg["classifiers"]["eegnet"]["early_stopping_patience"] = 10
    return cfg


# Fixed C grid for v1 paper (no tuning based on LOSO)
EA_TANGENT_LR_C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


def get_ea_tangent_lr_config() -> dict:
    """Config for EA + Tangent_LR (v1 paper: Euclidean Alignment then OAS cov, tangent, LR with fixed C grid)."""
    cfg = config_from_preset(
        "ea_riemann_tangent_oas",
        "logistic_regression",
        "laplacian_auto",
    )
    cfg.setdefault("features", {}).setdefault("ea_riemann_tangent_oas", {})
    cfg["features"]["ea_riemann_tangent_oas"]["z_score_tangent"] = True
    cfg.setdefault("classifiers", {}).setdefault("logistic_regression", {})
    cfg["classifiers"]["logistic_regression"]["C_grid"] = EA_TANGENT_LR_C_GRID
    cfg["classifiers"]["logistic_regression"]["tune_C"] = True
    cfg["classifiers"]["logistic_regression"]["cv_folds"] = 3
    return cfg


def run_baselines_loso(
    dataset: str,
    subjects: list[int],
    baselines: list[str] | None = None,
    fast: bool = False,
) -> dict[str, list[dict]]:
    """
    Run fixed baselines (CSP+LDA, Riemann+MDM, etc.) under identical LOSO conditions.
    Returns {baseline_name: list[dict]} where each list is subject-level rows (same format as run_table_for_config).
    If fast=True, EEGNet uses fewer epochs (60) and earlier stopping.
    """
    baselines = baselines or list(BASELINE_PRESETS)
    out: dict[str, list[dict]] = {}
    for name in baselines:
        cfg = get_baseline_config(name, fast=fast)
        rows = run_table_for_config(cfg, dataset, subjects, pipeline_name=name)
        out[name] = rows
    return out
