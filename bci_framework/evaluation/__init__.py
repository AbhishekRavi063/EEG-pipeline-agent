"""Evaluation and multi-subject table runners for pipeline A/B comparison."""

from .multi_subject_runner import (
    run_table_for_config,
    load_config_for_tables,
    run_ab_comparison,
    get_default_loso_config,
    get_automl_loso_config,
    config_from_preset,
    get_baseline_config,
    get_ea_tangent_lr_config,
    run_baselines_loso,
    BASELINE_PRESETS,
)

__all__ = [
    "run_table_for_config",
    "load_config_for_tables",
    "run_ab_comparison",
    "get_default_loso_config",
    "get_automl_loso_config",
    "config_from_preset",
    "get_baseline_config",
    "get_ea_tangent_lr_config",
    "run_baselines_loso",
    "BASELINE_PRESETS",
]
