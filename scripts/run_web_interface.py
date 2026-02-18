#!/usr/bin/env python3
"""Run the web interface with GEDAI disabled so pipelines can fit (no leadfield required)."""

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import yaml

def main():
    config_path = ROOT / "bci_framework" / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    # Disable GEDAI so pipelines run without leadfield
    if config.get("advanced_preprocessing"):
        config["advanced_preprocessing"] = dict(config["advanced_preprocessing"])
        config["advanced_preprocessing"]["enabled"] = ["signal_quality", "ica", "wavelet"]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        temp_path = f.name
    try:
        cmd = [sys.executable, str(ROOT / "main.py"), "--config", temp_path, "--subject", "1", "--web"]
        subprocess.run(cmd, cwd=ROOT)
    finally:
        Path(temp_path).unlink(missing_ok=True)

if __name__ == "__main__":
    main()
