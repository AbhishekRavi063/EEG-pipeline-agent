"""Pipeline: EEG -> Preprocessing -> Feature Extraction -> Classifier -> Metrics."""

import time
from typing import Any

import numpy as np

from bci_framework.preprocessing import PreprocessingManager
from bci_framework.features import FEATURE_REGISTRY
from bci_framework.classifiers import CLASSIFIER_REGISTRY


class Pipeline:
    """Combines mandatory/advanced preprocessing, feature extraction, and classifier."""

    def __init__(
        self,
        name: str,
        feature_name: str,
        classifier_name: str,
        fs: float,
        n_classes: int,
        config: dict[str, Any] | None = None,
        channel_names: list[str] | None = None,
    ) -> None:
        self.name = name
        self.feature_name = feature_name
        self.classifier_name = classifier_name
        self.fs = fs
        self.n_classes = n_classes
        config = config or {}
        feat_cfg = config.get("features", {})
        clf_cfg = config.get("classifiers", {})

        self.preprocessing_manager = PreprocessingManager(
            fs=fs,
            config=config,
            channel_names=channel_names,
        )

        fcls = FEATURE_REGISTRY[feature_name]
        params = {k: v for k, v in feat_cfg.get(feature_name, {}).items() if k != "fs"}
        self.feature_extractor = fcls(fs=fs, **params)

        ccls = CLASSIFIER_REGISTRY[classifier_name]
        params = {k: v for k, v in clf_cfg.get(classifier_name, {}).items() if k != "n_classes"}
        self.classifier = ccls(n_classes=n_classes, **params)

        self._fitted = False

    def preprocess(self, X: np.ndarray, y: np.ndarray | None = None, fit: bool = False) -> np.ndarray:
        """Run mandatory + advanced preprocessing."""
        if fit:
            return self.preprocessing_manager.fit_transform(X, y)
        return self.preprocessing_manager.transform(X)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Pipeline":
        """Fit preprocessing, feature extractor, classifier in sequence."""
        X_pre = self.preprocess(X, y=y, fit=True)
        self.feature_extractor.fit(X_pre, y)
        X_feat = self.feature_extractor.transform(X_pre)
        self.classifier.fit(X_feat, y)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Preprocess and extract features (no classifier)."""
        X_pre = self.preprocess(X)
        return self.feature_extractor.transform(X_pre)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Full pipeline: preprocess -> features -> predict."""
        X_feat = self.transform(X)
        return self.classifier.predict(X_feat)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_feat = self.transform(X)
        return self.classifier.predict_proba(X_feat)

    def predict_with_latency(self, X: np.ndarray) -> tuple[np.ndarray, float]:
        """Return (predictions, latency_ms)."""
        t0 = time.perf_counter()
        pred = self.predict(X)
        latency_ms = (time.perf_counter() - t0) * 1000
        return pred, latency_ms

    def predict_stream(self, X: np.ndarray) -> np.ndarray:
        """
        Predict on streaming window data (same as predict, but explicitly for streaming).
        
        This method is identical to predict() but exists for clarity when used
        in streaming contexts. It ensures preprocessing is causal (no future samples).
        
        Parameters
        ----------
        X : np.ndarray
            Shape (n_trials, n_channels, n_samples) - typically n_trials=1 for streaming
        
        Returns
        -------
        predictions : np.ndarray
            Shape (n_trials,) with predicted class indices
        """
        # For streaming, ensure we're using causal preprocessing
        # The PreprocessingManager already handles this based on mode="online"
        return self.predict(X)

    @property
    def advanced_preprocessing(self) -> list[str]:
        return self.preprocessing_manager.enabled_advanced_steps()

    def __repr__(self) -> str:
        return (
            f"Pipeline(name={self.name!r}, "
            f"advanced_steps={self.advanced_preprocessing}, "
            f"feature={self.feature_name!r}, classifier={self.classifier_name!r})"
        )
