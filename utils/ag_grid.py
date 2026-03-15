"""Shared AG Grid helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def literal_field_dash_grid_options(options: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Treat AG Grid field names as literal keys, not dotted object paths."""
    merged = dict(options or {})
    merged["suppressFieldDotNotation"] = True
    return merged
