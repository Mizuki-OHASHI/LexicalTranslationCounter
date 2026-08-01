"""Environment variable helpers shared across LTC backends and CLIs."""

from __future__ import annotations

import os

TRUTHY = ("1", "true", "yes", "on")


def get_bool_env(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in TRUTHY


def get_positive_int_env(name, default=None):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer: {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be positive: {value!r}")
    return parsed
