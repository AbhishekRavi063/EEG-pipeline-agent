"""Pipeline: Preprocessing chain + Feature extractor + Classifier."""

from .pipeline import Pipeline
from .registry import PipelineRegistry

__all__ = ["Pipeline", "PipelineRegistry"]
