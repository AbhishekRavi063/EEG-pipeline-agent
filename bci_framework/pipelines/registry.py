"""PipelineRegistry: auto-generate or load explicit pipeline combinations."""

import logging
from typing import Any, Iterable

from bci_framework.features import FEATURE_REGISTRY
from bci_framework.classifiers import CLASSIFIER_REGISTRY

from .pipeline import Pipeline


def _compose_pipeline_name(feature: str, classifier: str, advanced: Iterable[str]) -> str:
    parts = ["baseline"]
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
                pipe.name = _compose_pipeline_name(feat, clf, pipe.advanced_preprocessing)
                pipelines.append(pipe)
            return pipelines

        if not auto:
            return []

        feat_names = list(FEATURE_REGISTRY.keys())
        clf_names = list(CLASSIFIER_REGISTRY.keys())

        count = 0
        for feat in feat_names:
            if count >= self.max_combinations:
                break
            for clf in clf_names:
                if count >= self.max_combinations:
                    break
                pipe = Pipeline(
                    name=f"{feat}_{clf}",
                    feature_name=feat,
                    classifier_name=clf,
                    fs=fs,
                    n_classes=n_classes,
                    config=self.config,
                    channel_names=channel_names,
                )
                pipe.name = _compose_pipeline_name(feat, clf, pipe.advanced_preprocessing)
                pipelines.append(pipe)
                count += 1
        return pipelines

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
