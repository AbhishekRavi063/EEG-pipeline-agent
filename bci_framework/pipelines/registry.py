"""PipelineRegistry: auto-generate or load explicit pipeline combinations."""

import logging
from typing import Any, Iterable

from bci_framework.features import FEATURE_REGISTRY
from bci_framework.classifiers import CLASSIFIER_REGISTRY
from bci_framework.preprocessing.spatial_filters.resolver import _normalize_method, resolve_spatial_method, method_for_registry

from .pipeline import Pipeline


def _compose_pipeline_name(
    feature: str,
    classifier: str,
    advanced: Iterable[str],
    spatial_filter: str | None = None,
) -> str:
    parts = ["baseline"]
    if spatial_filter:
        parts.append(spatial_filter)
    if advanced:
        parts.append("-".join(advanced))
    parts.append(feature)
    parts.append(classifier)
    return "_".join(filter(None, parts))


class PipelineRegistry:
    """Build pipelines from config: auto combinations or explicit list."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.pipelines_config = config.get("pipelines", {})
        self.max_combinations = self.pipelines_config.get("max_combinations", 20)

    def build_pipelines(
        self,
        fs: float,
        n_classes: int,
        channel_names: list[str] | None = None,
    ) -> list[Pipeline]:
        """Return list of Pipeline instances."""
        explicit = self.pipelines_config.get("explicit", [])
        auto = self.pipelines_config.get("auto_generate", True)

        pipelines: list[Pipeline] = []

        if explicit:
            for spec in explicit:
                try:
                    feat, clf = self._parse_explicit_spec(spec)
                except ValueError as exc:
                    self.logger.warning("Invalid explicit pipeline spec %s: %s", spec, exc)
                    continue
                if feat not in FEATURE_REGISTRY or clf not in CLASSIFIER_REGISTRY:
                    continue
                pipe = Pipeline(
                    name=f"{feat}_{clf}",
                    feature_name=feat,
                    classifier_name=clf,
                    fs=fs,
                    n_classes=n_classes,
                    config=self.config,
                    channel_names=channel_names,
                )
                sf = self._get_spatial_filter_name()
                pipe.name = _compose_pipeline_name(
                    feat, clf, pipe.advanced_preprocessing, spatial_filter=sf
                )
                pipelines.append(pipe)
            return pipelines

        if not auto:
            return []

        # v2: priority pipelines (Riemannian + Laplacian) always included
        # Riemann tangent OAS + logistic regression: cross-subject baseline (source-only reference, no transfer)
        priority_specs: list[tuple[str, str, str]] = [
            ("laplacian", "csp", "lda"),
            ("", "riemannian", "lda"),
            ("", "covariance", "mdm"),
            ("", "riemann_tangent_oas", "logistic_regression"),
        ]
        seen: set[tuple[str, str, str]] = set()
        for sf_method, feat, clf in priority_specs:
            if feat not in FEATURE_REGISTRY or clf not in CLASSIFIER_REGISTRY:
                continue
            key = (sf_method or "", feat, clf)
            if key in seen:
                continue
            seen.add(key)
            config_use = self._config_for_spatial(sf_method) if sf_method else self.config
            pipe = Pipeline(
                name=f"{feat}_{clf}",
                feature_name=feat,
                classifier_name=clf,
                fs=fs,
                n_classes=n_classes,
                config=config_use,
                channel_names=channel_names,
            )
            pipe.name = _compose_pipeline_name(
                feat, clf, pipe.advanced_preprocessing, spatial_filter=sf_method or None
            )
            pipelines.append(pipe)

        feat_names = list(FEATURE_REGISTRY.keys())
        clf_names = list(CLASSIFIER_REGISTRY.keys())
        spatial_methods = self._get_spatial_methods_for_automl()

        count = len(pipelines)
        for feat in feat_names:
            if count >= self.max_combinations:
                break
            for clf in clf_names:
                if count >= self.max_combinations:
                    break
                for sf_method in spatial_methods:
                    if count >= self.max_combinations:
                        break
                    key = (sf_method or "", feat, clf)
                    if key in seen:
                        continue
                    seen.add(key)
                    config_use = self._config_for_spatial(sf_method)
                    pipe = Pipeline(
                        name=f"{feat}_{clf}",
                        feature_name=feat,
                        classifier_name=clf,
                        fs=fs,
                        n_classes=n_classes,
                        config=config_use,
                        channel_names=channel_names,
                    )
                    pipe.name = _compose_pipeline_name(
                        feat, clf, pipe.advanced_preprocessing, spatial_filter=sf_method
                    )
                    pipelines.append(pipe)
                    count += 1
        return pipelines

    def _get_spatial_filter_name(self) -> str | None:
        """Return config spatial_filter.method if enabled, else None (for display)."""
        sf = self.config.get("spatial_filter", {})
        if sf.get("enabled") and sf.get("method"):
            return str(sf["method"])
        return None

    def _resolve_spatial_method_for_automl(self, method: str) -> str | None:
        """Resolve one spatial method with capabilities. Returns None to skip (strict), or method to use (may be 'car')."""
        capabilities = self.config.get("spatial_capabilities")
        method = _normalize_method(method)
        if capabilities is None:
            return method_for_registry(method) or method
        try:
            resolved, _ = resolve_spatial_method(method, capabilities)
            return resolved
        except RuntimeError:
            return None

    def _get_spatial_methods_for_automl(self) -> list[str]:
        """Return list of spatial filter method names (resolved with capabilities; unsupported dropped in strict)."""
        sf = self.config.get("spatial_filter", {})
        methods_raw = []
        if sf.get("auto_select") and sf.get("methods_for_automl"):
            methods_raw = list(sf["methods_for_automl"])
        else:
            name = self._get_spatial_filter_name()
            methods_raw = [name] if name else [""]
        capabilities = self.config.get("spatial_capabilities")
        out = []
        for m in methods_raw:
            m = _normalize_method(m) or m
            if capabilities is None:
                out.append(method_for_registry(m) or m or "")
                continue
            try:
                resolved, _ = resolve_spatial_method(m, capabilities)
                out.append(resolved)
            except RuntimeError:
                continue
        return list(dict.fromkeys(out))  # preserve order, dedupe

    def _config_for_spatial(self, spatial_method: str) -> dict[str, Any]:
        """Return config with spatial_filter set to given method (for AutoML variants)."""
        if not spatial_method:
            return self.config
        import copy
        cfg = copy.deepcopy(self.config)
        sf = cfg.setdefault("spatial_filter", {})
        sf["enabled"] = True
        sf["method"] = spatial_method
        return cfg

    @staticmethod
    def _parse_explicit_spec(spec: Any) -> tuple[str, str]:
        """
        Backwards compatible parsing of explicit pipeline specs.
        Accepts [feature, classifier] (new) or [prep(s), feature, classifier] (legacy).
        """
        if isinstance(spec, (list, tuple)):
            if len(spec) >= 2 and isinstance(spec[0], str) and isinstance(spec[1], str) and len(spec) == 2:
                return spec[0], spec[1]
            if len(spec) >= 3:
                return spec[-2], spec[-1]
        raise ValueError(f"Invalid explicit pipeline specification: {spec}")
