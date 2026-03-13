"""Shared helpers for add-series validation callbacks."""

from __future__ import annotations

from utils.raw_dataset import get_dataset_key, get_raw_dataset_df


def get_existing_columns(raw_data) -> set[str]:
    """Return existing series names from the shared raw-data store payload."""
    if not raw_data:
        return set()
    try:
        dataset_key = get_dataset_key(raw_data)
        if not dataset_key:
            return set()
        return set(get_raw_dataset_df(dataset_key).columns)
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
