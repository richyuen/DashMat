from __future__ import annotations

from typing import Any

import pandas as pd

from utils.returns import (
    align_monthly_index_to_month_end,
    align_monthly_series_to_month_end,
    json_to_df,
    merge_returns,
)
from utils.artifact_store import get_default_artifact_store, normalize_raw_data_descriptor, write_raw_data_frame


def normalize_saved_series_store(store: Any) -> dict[str, dict[str, Any]]:
    """Normalize legacy/new saved-series provenance payloads."""
    if isinstance(store, dict):
        normalized: dict[str, dict[str, Any]] = {}
        for name, meta in store.items():
            key = str(name).strip()
            if not key:
                continue
            if isinstance(meta, dict):
                normalized[key] = dict(meta)
            else:
                normalized[key] = {}
        return normalized

    if isinstance(store, (list, tuple, set)):
        return {
            str(name): {
                "origin_page": "portopt",
                "origin_result": str(name),
                "series_type": "portfolio",
            }
            for name in store
            if str(name).strip()
        }

    return {}


def saved_series_store_names(store: Any) -> list[str]:
    return list(normalize_saved_series_store(store).keys())


def save_series_to_raw_data(
    *,
    raw_data: str | None,
    periodicity: str,
    series: pd.Series,
    base_name: str,
    saved_series_store: Any,
    origin_page: str,
    origin_result: str,
    series_type: str,
    prior_saved_name: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Persist a result series into the shared raw-data store."""
    clean_series = series.dropna()
    if clean_series.empty:
        raise ValueError("No series data available to save.")

    existing_df = json_to_df(raw_data) if raw_data else pd.DataFrame()
    resolved_periodicity = str(periodicity or "daily")
    normalized_store = normalize_saved_series_store(saved_series_store)

    previous_name = str(prior_saved_name).strip() if prior_saved_name else ""
    previous_name = previous_name or None
    base_name = str(base_name or "SavedSeries").strip() or "SavedSeries"

    if resolved_periodicity == "monthly":
        existing_df = align_monthly_index_to_month_end(existing_df)
        clean_series = align_monthly_series_to_month_end(clean_series)

    existing_columns = set(existing_df.columns)
    if previous_name and previous_name in existing_columns:
        target_name = previous_name
        action = "overwritten"
    else:
        target_name = _resolve_unique_column_name(base_name, existing_columns)
        action = "saved"

    frame = clean_series.to_frame(name=target_name)
    existing_without_target = existing_df.drop(columns=[target_name], errors="ignore")
    merged_df = merge_returns(existing_without_target, frame)

    if previous_name and previous_name != target_name:
        normalized_store.pop(previous_name, None)

    normalized_store[target_name] = {
        **normalized_store.get(target_name, {}),
        "origin_page": origin_page,
        "origin_result": origin_result,
        "series_type": series_type,
    }

    resolved_session_id = str(session_id or "").strip()
    raw_descriptor = normalize_raw_data_descriptor(raw_data)
    if not resolved_session_id and raw_descriptor is not None:
        current_descriptor = get_default_artifact_store().get_descriptor(raw_descriptor["raw_data_key"])
        if current_descriptor is not None:
            resolved_session_id = str(current_descriptor.session_id or "").strip()
    if not resolved_session_id:
        raise ValueError("Session id is required to save series.")
    raw_payload, _ = write_raw_data_frame(
        df=merged_df,
        session_id=resolved_session_id,
        original_periodicity=resolved_periodicity,
    )

    return {
        "raw_data": raw_payload,
        "saved_series_store": normalized_store,
        "saved_name": target_name,
        "action": action,
    }


def _resolve_unique_column_name(base_name: str, existing_columns: set[str]) -> str:
    if base_name not in existing_columns:
        return base_name

    suffix = 1
    while True:
        candidate = f"{base_name}_{suffix}"
        if candidate not in existing_columns:
            return candidate
        suffix += 1
