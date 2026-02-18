"""Pipeline: EEG -> Preprocessing -> Spatial -> Feature Extraction -> Domain Adaptation -> Classifier.

Strict order (memory-safe): Domain adaptation runs ONLY on (n_trials, n_features) after
dimensionality reduction. Never Preprocessing -> DA -> CSP.
"""

import logging
import time
from typing import Any

import numpy as np

from bci_framework.preprocessing import PreprocessingManager
from bci_framework.features import FEATURE_REGISTRY
from bci_framework.classifiers import CLASSIFIER_REGISTRY
from bci_framework.domain_adaptation import get_adapter, ADAPTER_REGISTRY

logger = logging.getLogger(__name__)


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
        self.config = config
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

        # v3: domain adapter runs ONLY after feature extraction (memory-safe: 2D features only)
        transfer_cfg = config.get("transfer", {}) or {}
        self._transfer_enabled = bool(transfer_cfg.get("enabled", False))
        transfer_method = (transfer_cfg.get("method") or "none").lower()
        if transfer_method not in ADAPTER_REGISTRY:
            transfer_method = "none"
        if self._transfer_enabled:
            kwargs = {
                k: v for k, v in transfer_cfg.items()
                if k not in ("enabled", "method", "target_unlabeled_fraction")
            }
            if transfer_method == "coral":
                kwargs.setdefault("epsilon", transfer_cfg.get("regularization", 1e-3))
                kwargs.setdefault("safe_mode", transfer_cfg.get("safe_mode", False))
            self.domain_adapter = get_adapter(
                transfer_method,
                feature_name=feature_name,
                **kwargs,
            )
        else:
            self.domain_adapter = get_adapter("none")
        self._fitted = False
        # Diagnostic (transfer): set during fit() for LOSO validation
        self._debug_diff_source: float | None = None
        self._debug_diff_target: float | None = None

    def preprocess(self, X: np.ndarray, y: np.ndarray | None = None, fit: bool = False) -> np.ndarray:
        """Run mandatory + advanced preprocessing."""
        if fit:
            return self.preprocessing_manager.fit_transform(X, y)
        return self.preprocessing_manager.transform(X)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_target: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
        subject_ids: np.ndarray | None = None,
    ) -> "Pipeline":
        """Fit: Preprocessing -> Feature extraction -> Domain adaptation (2D only) -> Classifier.
        When transfer is enabled, X_target (unlabeled target data) is required; the adapter
        is fitted on source + target features and the classifier is always fitted on adapted source.
        """
        if self._transfer_enabled and (X_target is None or len(X_target) == 0):
            raise ValueError(
                "Transfer enabled but no target unlabeled data provided. "
                "Call fit(X_source, y_source, X_target=X_target_unlabeled)."
            )
        X_pre = self.preprocess(X, y=y, fit=True)
        fe = self.feature_extractor
        if getattr(fe, "rsa", False) and subject_ids is not None and len(subject_ids) == len(X_pre):
            fe.fit(X_pre, y, subject_ids=subject_ids)
        else:
            fe.fit(X_pre, y)
        X_feat = self.feature_extractor.transform(X_pre)
        if X_feat is not None and hasattr(X_feat, "shape") and self.feature_name in ("riemann_tangent_oas", "filter_bank_riemann"):
            logger.info("Feature dim (%s): %s", self.feature_name, X_feat.shape[1])
        # Low-memory: force float32 to halve memory (safe_low_memory.force_float32)
        if (self.config.get("safe_low_memory") or {}).get("force_float32"):
            X_feat = np.asarray(X_feat, dtype=np.float32)

        if self._transfer_enabled and X_target is not None and len(X_target) > 0:
            logger.info("[CHECK] Target test labels used in fit: False (adapter receives unlabeled target only)")
            X_target_pre = self.preprocess(X_target)
            X_target_feat = self.feature_extractor.transform(X_target_pre)
            if (self.config.get("safe_low_memory") or {}).get("force_float32"):
                X_target_feat = np.asarray(X_target_feat, dtype=np.float32)
            # Diagnostic: force strong domain shift to verify transfer is active (config only)
            scale = (self.config.get("transfer") or {}).get("diagnostic_scale_target")
            if scale is not None and float(scale) != 1.0:
                X_target_feat = np.asarray(X_target_feat, dtype=np.float64) * float(scale)
                logger.info("[DEBUG] Artificial shift applied to target features: scale=%.2f", float(scale))
            try:
                F_source = np.asarray(X_feat, dtype=np.float64).copy()
                self.domain_adapter.fit(X_feat, X_target_feat)
                X_feat = self.domain_adapter.transform(X_feat)
                F_source_adapted = np.asarray(X_feat, dtype=np.float64)
                diff_source = float(np.mean(np.abs(F_source_adapted - F_source)))
                logger.info("[DEBUG] Mean source feature difference after adaptation: %.8f", diff_source)
                F_target_test_adapted = self.domain_adapter.transform(X_target_feat)
                diff_target = float(np.mean(np.abs(F_target_test_adapted - X_target_feat)))
                logger.info("[DEBUG] Mean target (cal) feature difference after adaptation: %.8f", diff_target)
                self._debug_diff_source = diff_source
                self._debug_diff_target = diff_target
            except (ValueError, RuntimeError) as e:
                logger.warning("[TRANSFER] Adaptation failed, continuing with baseline: %s", e)
                self._debug_diff_source = self._debug_diff_target = None

        logger.info("[DEBUG] Classifier fit input shape: %s", getattr(X_feat, "shape", None))
        self.classifier.fit(X_feat, y, sample_weight=sample_weight)
        self._fitted = True
        return self

    def _features_to_classifier(self, X_feat: np.ndarray) -> np.ndarray:
        """Apply domain adapter when transfer is enabled. Classifier always receives adapted features."""
        if (self.config.get("safe_low_memory") or {}).get("force_float32"):
            X_feat = np.asarray(X_feat, dtype=np.float32)
        if self._transfer_enabled:
            if not self.domain_adapter.is_fitted:
                raise RuntimeError(
                    "Transfer enabled but adapter not fitted. Call fit(X_source, y_source, X_target=X_target_unlabeled) first."
                )
            try:
                return self.domain_adapter.transform(X_feat)
            except (ValueError, RuntimeError) as e:
                logger.warning("[TRANSFER] Transform failed, using baseline features: %s", e)
                return X_feat
        return X_feat

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Preprocess -> Feature extraction -> Domain adapter (2D) -> (no classifier)."""
        X_pre = self.preprocess(X)
        X_feat = self.feature_extractor.transform(X_pre)
        return self._features_to_classifier(X_feat)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Full pipeline: preprocess -> features -> [adapter] -> predict."""
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
