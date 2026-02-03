"""Configuration loader for YAML-based framework config."""

from pathlib import Path
from typing import Any

import yaml

_CONFIG: dict[str, Any] | None = None


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load config from YAML file. Uses default path if none given."""
    global _CONFIG
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r") as f:
        _CONFIG = yaml.safe_load(f)
    return _CONFIG


def get_config() -> dict[str, Any]:
    """Return loaded config. Loads default if not yet loaded."""
    global _CONFIG
    if _CONFIG is None:
        load_config()
    assert _CONFIG is not None
    return _CONFIG
