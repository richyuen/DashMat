"""Shared helpers for add-series validation callbacks."""

from __future__ import annotations

from utils.returns import json_to_df


def get_existing_columns(raw_data) -> set[str]:
    """Return existing series names from raw-data JSON payload."""
    if not raw_data:
        return set()
    try:
        return set(json_to_df(raw_data).columns)
    except Exception:
        return set()


def find_duplicate_series(selected_series, raw_data) -> list[str]:
    """Return selected series that already exist in the current dataset."""
    if not selected_series:
        return []
    existing = get_existing_columns(raw_data)
    return [series for series in selected_series if series in existing]


def import_selected_disabled(selected_sheets) -> bool:
    """Return True when Import Selected should be disabled."""
    return not bool(selected_sheets)
