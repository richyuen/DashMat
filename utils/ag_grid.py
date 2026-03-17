"""Shared AG Grid helpers."""

from __future__ import annotations

from collections.abc import Mapping


_CLIPBOARD_MINUS_NORMALIZER = {"function": "dashmatProcessCellForClipboard(params)"}


def literal_field_dash_grid_options(options: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Treat AG Grid field names as literal keys, not dotted object paths."""
    merged = dict(options or {})
    merged["suppressFieldDotNotation"] = True
    merged.setdefault("processCellForClipboard", _CLIPBOARD_MINUS_NORMALIZER)
    return merged
