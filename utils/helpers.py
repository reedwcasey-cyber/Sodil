"""Misc helpers."""
from __future__ import annotations
import json
import os
import pickle
from pathlib import Path
from typing import Any

CACHE_DIR = Path.home() / ".sodil_cache"
CACHE_DIR.mkdir(exist_ok=True)


def cache_write(key: str, data: Any) -> None:
    path = CACHE_DIR / f"{key}.pkl"
    with open(path, "wb") as f:
        pickle.dump(data, f)


def cache_read(key: str) -> Any:
    path = CACHE_DIR / f"{key}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def fmt_pct(value: float, decimals: int = 1) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.{decimals}f}%"


def fmt_dollar(value: float) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign}${abs(value):,.2f}"
