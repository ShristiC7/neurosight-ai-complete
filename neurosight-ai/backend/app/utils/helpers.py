"""
NeuroSight AI — Utility Helpers
Shared utility functions used across the application.
"""
import re
import uuid
from datetime import datetime, timezone
from typing import Any


def utcnow() -> datetime:
    """Return timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def is_valid_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
        return True
    except ValueError:
        return False


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a float value between min and max."""
    return max(min_val, min(max_val, value))


def normalize(value: float, min_val: float, max_val: float) -> float:
    """Normalize value to [0, 1] given known min/max."""
    span = max_val - min_val
    if span == 0:
        return 0.0
    return clamp((value - min_val) / span, 0.0, 1.0)


def safe_divide(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    """Divide safely, returning fallback on division by zero."""
    return numerator / denominator if denominator != 0 else fallback


def round_to(value: float, decimals: int = 2) -> float:
    """Round a float to N decimal places."""
    return round(value, decimals)


def slugify(text: str) -> str:
    """Convert string to URL-safe slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    return text.strip("-")


def truncate(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate a string to max_length characters."""
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten a nested dict: {"a": {"b": 1}} -> {"a.b": 1}"""
    items: list = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def chunk_list(lst: list, size: int) -> list[list]:
    """Split a list into chunks of given size."""
    return [lst[i : i + size] for i in range(0, len(lst), size)]


def moving_average(values: list[float], window: int = 5) -> list[float]:
    """Compute simple moving average over a list of floats."""
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        window_vals = values[start : i + 1]
        result.append(sum(window_vals) / len(window_vals))
    return result


def percentage_change(old: float, new: float) -> float:
    """Return percentage change from old to new value."""
    if old == 0:
        return 0.0
    return ((new - old) / abs(old)) * 100
