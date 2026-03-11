from __future__ import annotations

from hashlib import md5
from typing import Any

import pandas as pd

from utils.artifact_store import ArtifactStore, get_dataframe_artifact, get_default_artifact_store
from utils.serialization import canonical_json_dumps
from utils.returns import df_to_json


def normalize_saved_series_cache_store(store_data: Any) -> dict[str, Any]:
    if isinstance(store_data, dict):
        return dict(store_data)
    return {}


def saved_series_cache_is_fresh(
    store_data: Any,
    raw_end: pd.Timestamp | None,
    *,
    store: ArtifactStore | None = None,
) -> bool:
    if raw_end is None:
        return False

    normalized = normalize_saved_series_cache_store(store_data)
    series_max_dates = normalized.get("series_max_dates")
    cache_key = normalized.get("cache_key")
    if not isinstance(cache_key, str) or not cache_key:
        return False
    if not isinstance(series_max_dates, dict) or not series_max_dates:
        return False
    if load_saved_series_cache_frame(normalized, store=store).empty:
        return False

    for max_date_raw in series_max_dates.values():
        max_date = pd.to_datetime(max_date_raw, errors="coerce")
        if pd.isna(max_date) or raw_end > max_date:
            return False
    return True


def load_saved_series_cache_frame(
    store_data: Any,
    *,
    store: ArtifactStore | None = None,
) -> pd.DataFrame:
    normalized = normalize_saved_series_cache_store(store_data)
    series_data = normalized.get("series_data")
    if isinstance(series_data, dict):
        frames = []
        for series_name, payload in series_data.items():
            if not isinstance(payload, dict):
                continue
            returns_json = payload.get("returns_json")
            if not isinstance(returns_json, str) or not returns_json:
                continue
            try:
                frame = pd.read_json(returns_json, orient="split")
            except ValueError:
                continue
            if frame.empty:
                continue
            if series_name in frame.columns:
                frames.append(frame[[series_name]])
            else:
                frames.append(frame.iloc[:, :1].rename(columns={frame.columns[0]: str(series_name)}))
        if not frames:
            return pd.DataFrame()
        merged = pd.concat(frames, axis=1).sort_index()
        merged.index = pd.to_datetime(merged.index)
        merged.index.name = "Date"
        return merged

    cache_key = normalized.get("cache_key")
    if not isinstance(cache_key, str) or not cache_key:
        return pd.DataFrame()

    frame = get_dataframe_artifact(cache_key, store=store)
    if frame.empty:
        return pd.DataFrame()
    frame = frame.sort_index()
    frame.index = pd.to_datetime(frame.index)
    frame.index.name = frame.index.name or "Date"
    return frame


def series_json_from_saved_series_cache(
    store_data: Any,
    series_name: str,
    *,
    store: ArtifactStore | None = None,
) -> str:
    frame = load_saved_series_cache_frame(store_data, store=store)
    if frame.empty or series_name not in frame.columns:
        return ""
    return df_to_json(frame[[series_name]].dropna())


def build_saved_series_cache_descriptor(
    *,
    session_id: str,
    saved_df: pd.DataFrame,
    series_max_dates: dict[str, str],
    raw_data_json: Any,
    store: ArtifactStore | None = None,
) -> dict[str, Any] | None:
    if not session_id or saved_df is None or saved_df.empty:
        return None

    saved_df = saved_df.sort_index()
    saved_df.index = pd.to_datetime(saved_df.index)
    saved_df.index.name = saved_df.index.name or "Date"
    raw_hash = md5(canonical_json_dumps(raw_data_json or "").encode("utf-8")).hexdigest()
    payload = {
        "columns": list(saved_df.columns),
        "series_max_dates": dict(series_max_dates or {}),
        "raw_hash": raw_hash,
    }
    artifact_store = store or get_default_artifact_store()
    descriptor = artifact_store.put_dataframe(
        df=saved_df,
        artifact_type="saved_series_cache",
        session_id=str(session_id),
        payload=payload,
        metadata={
            "columns": list(saved_df.columns),
            "series_max_dates": dict(series_max_dates or {}),
            "raw_hash": raw_hash,
            "min_date": str(pd.Timestamp(saved_df.index.min()).date()) if len(saved_df.index) else None,
            "max_date": str(pd.Timestamp(saved_df.index.max()).date()) if len(saved_df.index) else None,
        },
    )
    return {
        "cache_key": descriptor.key,
        "series_names": list(saved_df.columns),
        "series_max_dates": dict(series_max_dates or {}),
        "date_min": descriptor.metadata.get("min_date"),
        "date_max": descriptor.metadata.get("max_date"),
        "row_count": descriptor.row_count or 0,
        "raw_hash": raw_hash,
    }
