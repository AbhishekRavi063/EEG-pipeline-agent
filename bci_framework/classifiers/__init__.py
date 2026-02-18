"""Classifiers with unified API: fit(X, y), predict(X), predict_proba(X)."""

from .base import ClassifierBase
from .lda import LDAClassifier
from .svm import SVMClassifier
from .random_forest import RandomForestClassifier
from .eegnet import EEGNetClassifier
from .transformer_placeholder import TransformerClassifier
from .mdm import MDMClassifier
from .logistic_regression import LogisticRegressionClassifier
from .rsa_mlp import RSAMLPClassifier

CLASSIFIER_REGISTRY: dict[str, type[ClassifierBase]] = {
    "lda": LDAClassifier,
    "svm": SVMClassifier,
    "random_forest": RandomForestClassifier,
    "eegnet": EEGNetClassifier,
    "transformer": TransformerClassifier,
    "mdm": MDMClassifier,
    "logistic_regression": LogisticRegressionClassifier,
    "rsa_mlp": RSAMLPClassifier,
}

__all__ = [
    "ClassifierBase",
    "LDAClassifier",
    "SVMClassifier",
    "RandomForestClassifier",
    "EEGNetClassifier",
    "TransformerClassifier",
    "MDMClassifier",
    "LogisticRegressionClassifier",
    "RSAMLPClassifier",
    "CLASSIFIER_REGISTRY",
]
