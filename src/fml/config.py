"""Load the YAML configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path("configs/base.yaml")


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Read a YAML config file into a dictionary."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
