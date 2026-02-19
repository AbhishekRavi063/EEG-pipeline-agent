"""Utility functions and helpers for the BCI framework."""

from .config_loader import load_config, get_config
from .registry import Registry
from .splits import trial_train_test_split, get_train_test_trials, loso_splits
from .streaming import EEGStreamBuffer, sliding_window_chunks, stream_chunk
from .experiment import set_seed, get_experiment_id, set_experiment_id, log_experiment_params, enable_mlflow
from .pubsub import PubSub, TOPIC_RAW_EEG, TOPIC_FILTERED_EEG, TOPIC_PREDICTION, TOPIC_LATENCY, subscribe_queue
from .latency_logger import PipelineLatencyLogger, LatencyRecord
from .subject_table import (
    build_subject_table,
    save_table_csv,
    save_table_json,
    load_table_json,
    TABLE_METRIC_COLUMNS,
)
from .table_comparison import (
    compare_tables,
    compare_tables_multi_metric,
    delong_test_auc,
    bootstrap_metric_ci,
    auc_to_odds_ratio,
)

__all__ = [
    "load_config",
    "get_config",
    "Registry",
    "trial_train_test_split",
    "get_train_test_trials",
    "loso_splits",
    "EEGStreamBuffer",
    "sliding_window_chunks",
    "stream_chunk",
    "set_seed",
    "get_experiment_id",
    "set_experiment_id",
    "log_experiment_params",
    "enable_mlflow",
    "PubSub",
    "TOPIC_RAW_EEG",
    "TOPIC_FILTERED_EEG",
    "TOPIC_PREDICTION",
    "TOPIC_LATENCY",
    "subscribe_queue",
    "PipelineLatencyLogger",
    "LatencyRecord",
    "build_subject_table",
    "save_table_csv",
    "save_table_json",
    "load_table_json",
    "TABLE_METRIC_COLUMNS",
    "compare_tables",
    "compare_tables_multi_metric",
    "delong_test_auc",
    "bootstrap_metric_ci",
    "auc_to_odds_ratio",
]
