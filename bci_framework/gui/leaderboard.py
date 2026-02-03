"""
Pipeline leaderboard panel: sortable table of top-N pipelines with metrics and pruning flags.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def build_leaderboard_table(
    pipeline_metrics: dict[str, dict[str, Any]],
    top_n: int = 10,
    sort_by: str = "accuracy",
    ascending: bool = False,
    pruned_names: set[str] | None = None,
    selected_name: str | None = None,
) -> list[dict[str, Any]]:
    """
    Build sortable leaderboard rows from pipeline metrics.
    Returns list of dicts with keys: name, accuracy, kappa, latency_ms, stability, pruned, selected, ...
    """
    pruned_names = pruned_names or set()
    rows = []
    for name, m in pipeline_metrics.items():
        if isinstance(m, dict):
            row = dict(name=name, pruned=name in pruned_names, selected=name == selected_name, **m)
        else:
            row = dict(
                name=name,
                accuracy=getattr(m, "accuracy", 0),
                kappa=getattr(m, "kappa", 0),
                latency_ms=getattr(m, "latency_ms", 0),
                stability=getattr(m, "stability", 0),
                pruned=name in pruned_names,
                selected=name == selected_name,
            )
        rows.append(row)
    if not rows:
        return []
    keys = list(rows[0].keys())
    if sort_by not in keys:
        sort_by = "accuracy"
    rows.sort(key=lambda r: r.get(sort_by, 0) if isinstance(r.get(sort_by), (int, float)) else 0, reverse=not ascending)
    return rows[:top_n]


def render_leaderboard_matplotlib(
    ax,
    pipeline_metrics: dict[str, dict[str, Any]],
    top_n: int = 10,
    sort_by: str = "accuracy",
    pruned_names: set[str] | None = None,
    selected_name: str | None = None,
    col_names: list[str] | None = None,
) -> None:
    """
    Render a simple table on a matplotlib axis (leaderboard panel).
    """
    rows = build_leaderboard_table(
        pipeline_metrics, top_n=top_n, sort_by=sort_by,
        pruned_names=pruned_names, selected_name=selected_name,
    )
    if not rows:
        ax.set_title("Pipeline leaderboard (no data)")
        return
    col_names = col_names or ["name", "accuracy", "kappa", "latency_ms", "stability"]
    col_names = [c for c in col_names if c in rows[0]]
    cell_text = []
    for r in rows:
        cell_text.append([str(r.get(c, ""))[:12] for c in col_names])
    table = ax.table(
        cellText=cell_text,
        colLabels=col_names,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax.axis("off")
    ax.set_title("Pipeline leaderboard (sort by " + sort_by + ")")
