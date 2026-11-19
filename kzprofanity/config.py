"""Load config.yaml and resolve paths relative to the repo root."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path = ROOT / "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def path(cfg: dict, key: str) -> Path:
    """Absolute path for cfg['paths'][key]."""
    return ROOT / cfg["paths"][key]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def save_json(obj, p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
