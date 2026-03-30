from __future__ import annotations

import hashlib

import pandas as pd
from dash import Input, Output, State
from dash.exceptions import PreventUpdate

import cache_config
from dbengine import engine as DB_ENGINE, engine_MRD as MRD_ENGINE
from utils.core_categories import load_cma_returns_for_benches
from utils.returns import df_to_json
from utils.serialization import canonical_json_dumps
from utils.shared_metrics import MARKET_BETA_SERIES, RISK_FREE_SERIES

SAVED_SERIES_CONFIG = {
    RISK_FREE_SERIES: {},
    MARKET_BETA_SERIES: {"start_date": "1988-01-04"},
}

SHARED_BENCHMARK_STAMP_FIELDS = (
    "risk_free_max_date",
    "spx_max_date",
    "risk_free_hash",
    "spx_hash",
)


def build_saved_series_cache_series_data(saved_df: pd.DataFrame) -> dict:
    if saved_df.empty:
        return {}

    saved_df = saved_df.sort_index()
    series_data = {}
    for series_name, config in SAVED_SERIES_CONFIG.items():
        if series_name not in saved_df.columns:
            continue

        series_returns = saved_df[series_name].dropna().sort_index()
        start_date = config.get("start_date")
        if start_date:
            series_returns = series_returns.loc[
                series_returns.index >= pd.Timestamp(start_date)
            ]
        if series_returns.empty:
            continue

        series_max = pd.to_datetime(series_returns.index.max())
        series_data[series_name] = {
            "max_date": series_max.strftime("%Y-%m-%d"),
            "returns_json": df_to_json(series_returns.to_frame(series_name)),
        }

    return series_data


def _saved_series_entry_from_store(store_data, series_name: str) -> dict:
    if not isinstance(store_data, dict):
        return {}
    series_data = store_data.get("series_data")
    if not isinstance(series_data, dict):
        return {}
    entry = series_data.get(series_name)
    return dict(entry) if isinstance(entry, dict) else {}


def extract_shared_benchmark_payload(store_data) -> dict:
    risk_free_entry = _saved_series_entry_from_store(store_data, RISK_FREE_SERIES)
    spx_entry = _saved_series_entry_from_store(store_data, MARKET_BETA_SERIES)
    return {
        "risk_free_json": str(risk_free_entry.get("returns_json") or ""),
        "spx_json": str(spx_entry.get("returns_json") or ""),
        "risk_free_max_date": str(risk_free_entry.get("max_date") or ""),
        "spx_max_date": str(spx_entry.get("max_date") or ""),
    }


def _shared_benchmark_payload_json_hash(payload_json: str) -> str:
    normalized = str(payload_json or "")
    if not normalized:
        return ""
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def build_shared_benchmark_stamp(shared_benchmark_payload) -> dict:
    payload = shared_benchmark_payload if isinstance(shared_benchmark_payload, dict) else {}
    risk_free_json = str(payload.get("risk_free_json") or "")
    spx_json = str(payload.get("spx_json") or "")
    return {
        "risk_free_max_date": str(payload.get("risk_free_max_date") or ""),
        "spx_max_date": str(payload.get("spx_max_date") or ""),
        "risk_free_hash": _shared_benchmark_payload_json_hash(risk_free_json),
        "spx_hash": _shared_benchmark_payload_json_hash(spx_json),
    }


def coerce_shared_benchmark_stamp(value) -> dict:
    if isinstance(value, dict) and isinstance(value.get("series_data"), dict):
        return build_shared_benchmark_stamp(extract_shared_benchmark_payload(value))
    if not isinstance(value, dict):
        return {field: "" for field in SHARED_BENCHMARK_STAMP_FIELDS}
    return {
        field: str(value.get(field) or "")
        for field in SHARED_BENCHMARK_STAMP_FIELDS
    }


def _shared_benchmark_payload_cache_key(shared_benchmark_stamp) -> str:
    normalized_stamp = coerce_shared_benchmark_stamp(shared_benchmark_stamp)
    digest = hashlib.md5(
        canonical_json_dumps(normalized_stamp).encode("utf-8")
    ).hexdigest()
    return f"analyticstool.shared_benchmark_payload:{digest}"


def cache_shared_benchmark_payload(shared_benchmark_stamp, shared_benchmark_payload) -> None:
    normalized_payload = {
        "risk_free_json": str((shared_benchmark_payload or {}).get("risk_free_json") or ""),
        "spx_json": str((shared_benchmark_payload or {}).get("spx_json") or ""),
    }
    cache_config.cache.set(
        _shared_benchmark_payload_cache_key(shared_benchmark_stamp),
        normalized_payload,
        timeout=0,
    )


@cache_config.cache.memoize(timeout=0)
def load_shared_benchmark_payload_from_stamp(
    risk_free_max_date: str,
    spx_max_date: str,
    risk_free_hash: str,
    spx_hash: str,
) -> dict:
    del risk_free_max_date, spx_max_date, risk_free_hash, spx_hash
    try:
        saved_df = load_cma_returns_for_benches(
            DB_ENGINE,
            list(SAVED_SERIES_CONFIG.keys()),
            MRD_ENGINE,
        )
    except Exception:
        return {"risk_free_json": "", "spx_json": ""}

    series_data = build_saved_series_cache_series_data(saved_df)
    extracted = extract_shared_benchmark_payload({"series_data": series_data})
    return {
        "risk_free_json": extracted["risk_free_json"],
        "spx_json": extracted["spx_json"],
    }


def resolve_shared_benchmark_payload(shared_benchmark_source) -> dict:
    if isinstance(shared_benchmark_source, dict) and isinstance(
        shared_benchmark_source.get("series_data"), dict
    ):
        extracted = extract_shared_benchmark_payload(shared_benchmark_source)
        return {
            "risk_free_json": extracted["risk_free_json"],
            "spx_json": extracted["spx_json"],
        }

    normalized_stamp = coerce_shared_benchmark_stamp(shared_benchmark_source)
    if not any(normalized_stamp.values()):
        return {"risk_free_json": "", "spx_json": ""}

    cached_payload = cache_config.cache.get(
        _shared_benchmark_payload_cache_key(normalized_stamp)
    )
    if isinstance(cached_payload, dict):
        return {
            "risk_free_json": str(cached_payload.get("risk_free_json") or ""),
            "spx_json": str(cached_payload.get("spx_json") or ""),
        }

    return load_shared_benchmark_payload_from_stamp(
        normalized_stamp["risk_free_max_date"],
        normalized_stamp["spx_max_date"],
        normalized_stamp["risk_free_hash"],
        normalized_stamp["spx_hash"],
    )


def risk_free_json_from_source(shared_benchmark_source) -> str:
    return str(resolve_shared_benchmark_payload(shared_benchmark_source).get("risk_free_json") or "")


def spx_json_from_source(shared_benchmark_source) -> str:
    return str(resolve_shared_benchmark_payload(shared_benchmark_source).get("spx_json") or "")


def compute_saved_series_stamp(current_stamp, db_engine=DB_ENGINE, mrd_engine=MRD_ENGINE):
    normalized_current = coerce_shared_benchmark_stamp(current_stamp)
    if normalized_current.get("risk_free_hash") and normalized_current.get("spx_hash"):
        raise PreventUpdate

    try:
        saved_df = load_cma_returns_for_benches(
            db_engine,
            list(SAVED_SERIES_CONFIG.keys()),
            mrd_engine,
        )
    except Exception:
        raise PreventUpdate

    if saved_df.empty:
        return None

    series_data = build_saved_series_cache_series_data(saved_df)
    if not series_data:
        return None

    shared_benchmark_payload = extract_shared_benchmark_payload({"series_data": series_data})
    next_stamp = build_shared_benchmark_stamp(shared_benchmark_payload)
    cache_shared_benchmark_payload(next_stamp, shared_benchmark_payload)
    if next_stamp == normalized_current:
        raise PreventUpdate
    return next_stamp


def register_shared_benchmark_callbacks(app, db_engine=DB_ENGINE, mrd_engine=MRD_ENGINE) -> None:
    @app.callback(
        Output("dashmat-saved-series-stamp-store", "data"),
        Input("_pages_location", "pathname"),
        State("dashmat-saved-series-stamp-store", "data"),
        prevent_initial_call=False,
    )
    def refresh_saved_series_stamp_store(pathname, current_stamp):
        normalized_path = str(pathname or "").split("?")[0].rstrip("/") or "/"
        if normalized_path not in {"/analyticstool", "/portopt", "/regression"}:
            raise PreventUpdate
        return compute_saved_series_stamp(current_stamp, db_engine=db_engine, mrd_engine=mrd_engine)
