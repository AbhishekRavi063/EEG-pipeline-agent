"""
Explainability placeholder: CSP spatial maps and SHAP stub.
Example visualization stubs to avoid future confusion.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def plot_csp_patterns(
    csp_filters: np.ndarray,
    channel_names: list[str],
    title: str = "CSP spatial patterns",
    save_path: str | Path | None = None,
) -> "matplotlib.figure.Figure | None":
    """
    Plot CSP spatial patterns (filter weights per channel).
    csp_filters: (n_components, n_channels) or (n_channels, n_components).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib required for CSP pattern plot")
        return None
    if csp_filters.ndim == 2:
        if csp_filters.shape[0] < csp_filters.shape[1]:
            W = csp_filters.T
        else:
            W = csp_filters
    else:
        return None
    n_ch = W.shape[0]
    n_comp = min(W.shape[1], 6)
    ch_names = channel_names or [f"Ch{i}" for i in range(n_ch)]
    fig, axes = plt.subplots(2, (n_comp + 1) // 2, figsize=(4 * ((n_comp + 1) // 2), 4))
    if n_comp == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    for i in range(n_comp):
        axes[i].bar(range(n_ch), W[:, i], color="steelblue")
        axes[i].set_xticks(range(n_ch))
        axes[i].set_xticklabels(ch_names[:n_ch], rotation=45, ha="right")
        axes[i].set_title(f"CSP {i+1}")
    for j in range(n_comp, len(axes)):
        axes[j].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(Path(save_path), dpi=100)
    return fig


def shap_importance_stub(
    feature_names: list[str],
    importance: np.ndarray | None = None,
    title: str = "Feature importance (SHAP placeholder)",
    save_path: str | Path | None = None,
) -> "matplotlib.figure.Figure | None":
    """
    Placeholder for SHAP-based feature importance.
    When SHAP is integrated, replace with real summary plot.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    n = len(feature_names)
    if importance is None:
        importance = np.random.rand(n)
    fig, ax = plt.subplots(figsize=(8, max(4, n * 0.3)))
    idx = np.argsort(importance)[::-1]
    ax.barh(range(n), importance[idx], color="steelblue")
    ax.set_yticks(range(n))
    ax.set_yticklabels([feature_names[i] for i in idx])
    ax.set_xlabel("Importance (SHAP placeholder)")
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(Path(save_path), dpi=100)
    return fig
