"""
Hyperparameter optimization for pipelines (Optuna or grid search).
Tunes preprocessing params, feature params, classifier params.
"""

import logging
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


def _grid_search(
    param_grid: dict[str, list[Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    pipeline_factory: Callable[[dict[str, Any]], Any],
    score_metric: str = "accuracy",
    n_classes: int = 4,
) -> tuple[dict[str, Any], float]:
    """Exhaustive grid search over param_grid. Returns best params and best score."""
    from itertools import product
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    best_score = -1.0
    best_params: dict[str, Any] = {}
    for combo in product(*values):
        params = dict(zip(keys, combo))
        try:
            pipe = pipeline_factory(params)
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_val)
            proba = pipe.predict_proba(X_val) if score_metric in ("roc_auc",) else None
            if score_metric == "accuracy":
                score = float(np.mean(y_val == pred))
            elif score_metric == "kappa":
                from sklearn.metrics import cohen_kappa_score
                score = float(cohen_kappa_score(y_val, pred, labels=list(range(n_classes))))
            elif score_metric == "f1":
                from sklearn.metrics import f1_score
                score = float(f1_score(y_val, pred, average="macro", zero_division=0))
            elif score_metric == "roc_auc" and proba is not None:
                from sklearn.metrics import roc_auc_score
                from sklearn.preprocessing import label_binarize
                y_bin = label_binarize(y_val, classes=list(range(n_classes)))
                score = float(roc_auc_score(y_bin, proba, average="macro", multi_class="ovr"))
            else:
                score = float(np.mean(y_val == pred))
            if score > best_score:
                best_score = score
                best_params = dict(params)
        except Exception as e:
            logger.debug("Grid combo %s failed: %s", params, e)
            continue
    return best_params, best_score


def optuna_optimize(
    study_name: str,
    pipeline_factory: Callable[[dict[str, Any]], Any],
    param_suggestions: dict[str, tuple[str, Any, Any, Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int = 4,
    n_trials: int = 20,
    timeout_sec: float | None = None,
    seed: int = 42,
) -> tuple[dict[str, Any], float]:
    """
    Optuna-based hyperparameter search.
    pipeline_factory(params) -> pipeline with fit/predict.
    param_suggestions: name -> (suggest_type, low, high, log) e.g. ("float", 0.01, 1.0, True).
    Returns (best_params, best_score).
    """
    try:
        import optuna
    except ImportError:
        logger.warning("Optuna not installed; falling back to grid search")
        return {}, 0.0
    from sklearn.metrics import cohen_kappa_score

    def objective(trial: "optuna.Trial") -> float:
        params = {}
        for name, (suggest_type, a, b, *extra) in param_suggestions.items():
            if suggest_type == "float":
                params[name] = trial.suggest_float(name, a, b, log=bool(extra[0]) if extra else False)
            elif suggest_type == "int":
                params[name] = trial.suggest_int(name, int(a), int(b))
            elif suggest_type == "categorical":
                params[name] = trial.suggest_categorical(name, list(a) if hasattr(a, "__iter__") and not isinstance(a, str) else [a, b])
        try:
            pipe = pipeline_factory(params)
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_val)
            return float(cohen_kappa_score(y_val, pred, labels=list(range(n_classes))))
        except Exception as e:
            logger.debug("Trial params %s failed: %s", params, e)
            return 0.0

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", study_name=study_name, seed=seed)
    study.optimize(objective, n_trials=n_trials, timeout=timeout_sec, show_progress_bar=False)
    if study.best_trial is None:
        return {}, 0.0
    best_params = dict(study.best_trial.params)
    best_score = float(study.best_value)
    return best_params, best_score
