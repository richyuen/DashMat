"""Serialization and payload-normalization helpers.

Centralizes stable serialization for cache keys and callback payloads.
"""

from __future__ import annotations

import json
from typing import Any


def _normalize_for_json(value: Any) -> Any:
    """Convert objects to a deterministic JSON-serializable structure."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, dict):
        # Normalize keys as strings for deterministic ordering.
        normalized = {}
        for key in sorted(value.keys(), key=lambda k: str(k)):
            normalized[str(key)] = _normalize_for_json(value[key])
        return normalized

    if isinstance(value, (list, tuple)):
        return [_normalize_for_json(item) for item in value]

    if isinstance(value, set):
        return sorted((_normalize_for_json(item) for item in value), key=lambda x: str(x))

    # Numpy scalars and other custom objects.
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except Exception:
            pass

    # Fallback keeps behavior stable for unknown payload types.
    return str(value)


def canonical_json_dumps(value: Any) -> str:
    """Serialize payloads into stable JSON for cache keys and callback calls."""
    return json.dumps(
        _normalize_for_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def parse_mapping_payload(value: Any) -> dict:
    """Parse dict-like payloads passed as dict or JSON string."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def normalize_date_range_payload(value: Any) -> dict | None:
    """Normalize date range payload into {'start': str, 'end': str} or None."""
    if value is None:
        return None

    if isinstance(value, str):
        raw = value.strip()
        if not raw or raw in {"null", "None"}:
            return None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return normalize_date_range_payload(parsed)

    if isinstance(value, dict):
        start = value.get("start")
        end = value.get("end")
        if start is None or end is None:
            return None
        return {"start": str(start), "end": str(end)}

    # Backward compatibility path for legacy [start, end] payloads.
    if isinstance(value, (list, tuple)) and len(value) == 2:
        start, end = value
        if start is None or end is None:
            return None
        return {"start": str(start), "end": str(end)}

    return None


def mapping_payload_for_cache(value: Any) -> str:
    """Return canonical mapping JSON string for cacheable function args."""
    return canonical_json_dumps(parse_mapping_payload(value))


def date_range_payload_for_cache(value: Any) -> str:
    """Return canonical date-range JSON string for cacheable function args."""
    normalized = normalize_date_range_payload(value)
    if normalized is None:
        return "null"
    return canonical_json_dumps(normalized)
