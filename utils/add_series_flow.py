"""Shared helpers for add-series validation callbacks."""

from __future__ import annotations

from utils.raw_dataset import get_dataset_key, get_raw_dataset_df


def get_existing_columns(raw_data=None, raw_meta=None) -> set[str]:
    """Return existing series names from raw-data metadata or the raw-data store."""
    if isinstance(raw_meta, dict):
        columns = raw_meta.get("columns")
        if isinstance(columns, list) and columns:
            return set(columns)
        try:
            dataset_key = raw_meta.get("dataset_key")
            if dataset_key:
                return set(get_raw_dataset_df(dataset_key).columns)
        except Exception:
            return set()
    if not raw_data:
        return set()
    try:
        dataset_key = get_dataset_key(raw_data)
        if not dataset_key:
            return set()
        return set(get_raw_dataset_df(dataset_key).columns)
    except Exception:
        return set()


def find_duplicate_series(selected_series, raw_data=None, raw_meta=None) -> list[str]:
    """Return selected series that already exist in the current dataset."""
    if not selected_series:
        return []
    existing = get_existing_columns(raw_data=raw_data, raw_meta=raw_meta)
    return [series for series in selected_series if series in existing]


def import_selected_disabled(selected_sheets) -> bool:
    """Return True when Import Selected should be disabled."""
    return not bool(selected_sheets)
