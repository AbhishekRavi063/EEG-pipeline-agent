"""Desktop GUI: live EEG, filtered EEG, features, accuracy, pipeline comparison, leaderboard, explainability; web UI."""

from .app import BCIApp
from .leaderboard import build_leaderboard_table, render_leaderboard_matplotlib
from .explainability import plot_csp_patterns, shap_importance_stub
from .web_server import WebApp, WebSocketManager, start_server_thread, DEFAULT_WEB_PORT

__all__ = [
    "BCIApp",
    "WebApp",
    "WebSocketManager",
    "start_server_thread",
    "DEFAULT_WEB_PORT",
    "build_leaderboard_table",
    "render_leaderboard_matplotlib",
    "plot_csp_patterns",
    "shap_importance_stub",
]
