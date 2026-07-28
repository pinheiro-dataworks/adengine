"""YAML config loading with minimal validation.

Every module receives a plain dict loaded from configs/*.yaml — no framework,
no implicit defaults hidden in code. If a key is missing, callers fail with a
KeyError at the point of use, which is the intended fail-fast behavior.
"""
from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {path} did not parse to a mapping")
    return cfg
