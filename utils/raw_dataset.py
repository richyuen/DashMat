from __future__ import annotations

import hashlib
from io import StringIO
from typing import Any

import pandas as pd

import cache_config


RAW_DATA_STORE_SCHEMA_VERSION = 2
_DATASET_DF_CACHE_PREFIX = "raw_dataset_df:"
_DATASET_JSON_CACHE_PREFIX = "raw_dataset_json:"
_FALLBACK_DATASET_DF: dict[str, pd.DataFrame] = {}
_FALLBACK_DATASET_JSON: dict[str, str] = {}


def _dataset_df_cache_key(dataset_key: str) -> str:
    return f"{_DATASET_DF_CACHE_PREFIX}{dataset_key}"


def _dataset_json_cache_key(dataset_key: str) -> str:
    return f"{_DATASET_JSON_CACHE_PREFIX}{dataset_key}"


def _cache_get(key: str):
    value = cache_config.cache.get(key)
    if value is not None:
        return value
    if key.startswith(_DATASET_DF_CACHE_PREFIX):
        return _FALLBACK_DATASET_DF.get(key.removeprefix(_DATASET_DF_CACHE_PREFIX))
    if key.startswith(_DATASET_JSON_CACHE_PREFIX):
        return _FALLBACK_DATASET_JSON.get(key.removeprefix(_DATASET_JSON_CACHE_PREFIX))
    return None


def _cache_set(key: str, value) -> None:
    cache_config.cache.set(key, value, timeout=0)
    if key.startswith(_DATASET_DF_CACHE_PREFIX):
        dataset_key = key.removeprefix(_DATASET_DF_CACHE_PREFIX)
        if isinstance(value, pd.DataFrame):
            _FALLBACK_DATASET_DF[dataset_key] = value
    elif key.startswith(_DATASET_JSON_CACHE_PREFIX):
        dataset_key = key.removeprefix(_DATASET_JSON_CACHE_PREFIX)
        if isinstance(value, str):
            _FALLBACK_DATASET_JSON[dataset_key] = value


def clear_raw_dataset_cache() -> None:
    _FALLBACK_DATASET_DF.clear()
    _FALLBACK_DATASET_JSON.clear()


def _is_cached_dataset_key(value: str) -> bool:
    if not value:
        return False
    return (
        _cache_get(_dataset_json_cache_key(value)) is not None
        or _cache_get(_dataset_df_cache_key(value)) is not None
    )


def dataset_key_from_json(raw_data_json: str) -> str:
    text = str(raw_data_json or "")
    if not text:
        raise ValueError("raw_data_json is required")
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _df_to_raw_json(df: pd.DataFrame) -> str:
    out = df.copy()
    if out.index.name is None:
        out.index.name = "Date"
    return out.to_json(date_format="iso", orient="split")


def _json_to_dataset_df(raw_data_json: str) -> pd.DataFrame:
    df = pd.read_json(StringIO(raw_data_json), orient="split")
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df


def build_raw_data_store_payload(df_or_json: pd.DataFrame | str) -> dict[str, Any]:
    raw_data_json = _df_to_raw_json(df_or_json) if isinstance(df_or_json, pd.DataFrame) else str(df_or_json or "")
    if not raw_data_json:
        raise ValueError("raw dataset payload cannot be empty")
    dataset_key = dataset_key_from_json(raw_data_json)
    payload = {
        "schema_version": RAW_DATA_STORE_SCHEMA_VERSION,
        "dataset_key": dataset_key,
        "raw_data_json": raw_data_json,
    }
    cache_raw_dataset(payload)
    return payload


def normalize_raw_data_store(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = str(value or "")
        if not text:
            return None
        return {
            "schema_version": RAW_DATA_STORE_SCHEMA_VERSION,
            "dataset_key": dataset_key_from_json(text),
            "raw_data_json": text,
        }
    if not isinstance(value, dict):
        raise ValueError("raw-data store must be a dict payload")

    dataset_key = str(value.get("dataset_key") or "").strip()
    raw_data_json = str(value.get("raw_data_json") or "")
    if not dataset_key or not raw_data_json:
        raise ValueError("raw-data store payload requires dataset_key and raw_data_json")
    return {
        "schema_version": int(value.get("schema_version") or RAW_DATA_STORE_SCHEMA_VERSION),
        "dataset_key": dataset_key,
        "raw_data_json": raw_data_json,
    }


def cache_raw_dataset(payload: dict[str, Any] | None) -> str | None:
    normalized = normalize_raw_data_store(payload)
    if normalized is None:
        return None

    dataset_key = normalized["dataset_key"]
    raw_data_json = normalized["raw_data_json"]
    json_cache_key = _dataset_json_cache_key(dataset_key)
    df_cache_key = _dataset_df_cache_key(dataset_key)

    cached_json = _cache_get(json_cache_key)
    if cached_json is None:
        _cache_set(json_cache_key, raw_data_json)

    cached_df = _cache_get(df_cache_key)
    if cached_df is None:
        _cache_set(df_cache_key, _json_to_dataset_df(raw_data_json))

    return dataset_key


def get_dataset_key(raw_data_store: dict[str, Any] | None) -> str | None:
    normalized = normalize_raw_data_store(raw_data_store)
    if normalized is None:
        return None
    cache_raw_dataset(normalized)
    return normalized["dataset_key"]


def resolve_dataset_key(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return get_dataset_key(value)
    if isinstance(value, str):
        text = str(value or "").strip()
        if not text:
            return None
        if _is_cached_dataset_key(text):
            return text
        if text.startswith("{"):
            return get_dataset_key(text)
        return text
    raise ValueError("dataset reference must be a payload dict or string")


def has_raw_dataset(raw_data_store: dict[str, Any] | None) -> bool:
    return bool(get_dataset_key(raw_data_store))


def get_raw_data_json_from_store(raw_data_store: dict[str, Any] | None) -> str | None:
    dataset_key = get_dataset_key(raw_data_store)
    if not dataset_key:
        return None
    return get_raw_dataset_json(dataset_key)


def get_raw_dataset_json(dataset_key: str) -> str:
    resolved_key = resolve_dataset_key(dataset_key)
    if not resolved_key:
        raise ValueError("dataset_key is required")
    raw_data_json = _cache_get(_dataset_json_cache_key(resolved_key))
    if not isinstance(raw_data_json, str) or not raw_data_json:
        raise KeyError(f"raw dataset JSON missing for dataset_key={resolved_key}")
    return raw_data_json


def get_raw_dataset_df(dataset_key: str) -> pd.DataFrame:
    resolved_key = resolve_dataset_key(dataset_key)
    if not resolved_key:
        raise ValueError("dataset_key is required")
    cached_df = _cache_get(_dataset_df_cache_key(resolved_key))
    if isinstance(cached_df, pd.DataFrame):
        return cached_df
    raw_data_json = get_raw_dataset_json(resolved_key)
    df = _json_to_dataset_df(raw_data_json)
    _cache_set(_dataset_df_cache_key(resolved_key), df)
    return df
