from __future__ import annotations

from datetime import datetime, timezone
import json
import time
from typing import Any
from uuid import uuid4

import pandas as pd
from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

from utils.core_categories import load_cma_returns_for_benches_with_meta
from utils.portfolio_series import load_portfolio_series
from utils.raw_data_imports import load_factor_series, load_fund_series, load_performance_series
from utils.date_range_flow import ACCOUNT_LIST_MAX_END_SENTINEL, compute_date_range_candidates
from utils.raw_dataset import build_raw_data_store_payload, get_dataset_key, get_raw_dataset_df, resolve_dataset_key
from utils.perf_timing import timed_block
from utils.returns import (
    align_monthly_index_to_month_end,
    build_raw_data_metadata,
    merge_returns,
    resample_returns,
)
from utils.serialization import canonical_json_dumps
from utils.underlying_category_imports import load_underlying_category_series


ACCOUNT_LIST_SCHEMA_VERSION = 2

_DATE_RANGE_SENTINEL_CONFIG = {
    "at-date-range-store": {
        "periodicity": "at-periodicity-value-store",
        "series": "at-series-select",
    },
    "po-date-range-store": {
        "periodicity": "po-periodicity-value-store",
        "series": "po-series-select",
    },
    "reg-date-range-store": {
        "periodicity": "reg-periodicity-value-store",
        "series": "reg-series-select",
    },
}

AT_STORE_IDS = {
    "selected": "at-series-select",
    "selected_value": "at-series-select-value-store",
    "bench": "at-benchmark-assignments-store",
    "long_short": "at-long-short-store",
    "order": "at-series-order-store",
    "vol": "at-vol-scaling-assignments-store",
}

PO_STORE_IDS = {
    "selected": "po-series-select",
    "selected_value": "po-series-select-value-store",
    "bench": "po-benchmark-assignments-store",
    "cmabench": "po-cmabench-assignments-store",
    "long_short": "po-long-short-store",
    "order": "po-series-order-store",
    "vol": "po-vol-scaling-assignments-store",
    "min_wt": "po-min-wt-store",
    "max_wt": "po-max-wt-store",
    "force_max": "po-force-max-store",
}

REG_STORE_IDS = {
    "selected": "reg-series-select",
    "selected_value": "reg-series-select-value-store",
    "bench": "reg-benchmark-assignments-store",
    "long_short": "reg-long-short-store",
    "order": "reg-series-order-store",
    "vol": "reg-vol-scaling-assignments-store",
    "dep": "reg-dependent-var-store",
    "lag": "reg-lag-store",
    "min_beta": "reg-min-beta-store",
    "max_beta": "reg-max-beta-store",
    "enable": "reg-enable-constraint-store",
}

AT_EXTRA_CONTROL_STORE_IDS = [
    "at-periodicity-value-store",
    "at-returns-type-value-store",
    "at-active-tab-store",
    "at-rolling-window-store",
    "at-rolling-metric-store",
    "at-rolling-return-type-store",
    "at-rolling-chart-switch-store",
    "at-drawdown-chart-switch-store",
    "at-growth-chart-switch-store",
    "at-use-risk-free-store",
    "at-monthly-view-store",
    "at-partial-period-store",
    "at-monthly-series-store",
    "at-factor-mode-store",
    "at-factor-quantiles-store",
    "at-factor-transform-store",
    "at-factor-series-store",
    "at-factor-qq-reference-store",
    "at-conditional-view-store",
    "at-conditional-comparator-store",
    "at-conditional-threshold-store",
    "at-conditional-window-conversion-store",
    "at-conditional-step-store",
    "at-conditional-step-unit-store",
    "at-conditional-display-mode-store",
    "at-regime-definition-store",
    "at-date-range-store",
    "at-vol-scaler-value-store",
]

PO_EXTRA_CONTROL_STORE_IDS = [
    "po-periodicity-value-store",
    "po-vol-scaler-value-store",
    "po-use-risk-free-store",
    "po-returns-basis-store",
    "po-reporting-basis-store",
    "po-partial-period-store",
    "po-date-range-store",
    "po-opt-window-store",
    "po-window-size-store",
    "po-opt-step-store",
    "po-opt-step-unit-store",
    "po-opt-model-store",
    "po-portfolio-name-store",
    "po-exp-wt-cov-store",
    "po-halflife-store",
    "po-cov-shrinkage-store",
    "po-cov-shrinkage-target-store",
    "po-missing-data-store",
    "po-fill-in-sample-store",
    "po-ex-ante-mode-store",
    "po-bl-tau-store",
    "po-objective-store",
    "po-active-tab-store",
    "po-weight-chart-switch-store",
    "po-attribution-chart-switch-store",
    "po-risk-chart-switch-store",
    "po-turnover-chart-switch-store",
    "po-frontier-chart-switch-store",
]

REG_EXTRA_CONTROL_STORE_IDS = [
    "reg-periodicity-value-store",
    "reg-vol-scaler-value-store",
    "reg-use-risk-free-store",
    "reg-partial-period-store",
    "reg-date-range-store",
    "reg-model-store",
    "reg-regression-name-store",
    "reg-force-zero-intercept-store",
    "reg-robust-se-store",
    "reg-exp-wt-store",
    "reg-halflife-store",
    "reg-window-type-store",
    "reg-window-size-store",
    "reg-opt-step-store",
    "reg-opt-step-unit-store",
    "reg-fill-in-sample-store",
    "reg-missing-data-store",
    "reg-alpha-store",
    "reg-l1-ratio-store",
    "reg-active-tab-store",
]

ACCOUNT_LIST_SERIES_DIALOG_STORE_IDS = [
    AT_STORE_IDS["selected"],
    AT_STORE_IDS["bench"],
    AT_STORE_IDS["long_short"],
    AT_STORE_IDS["order"],
    AT_STORE_IDS["vol"],
    PO_STORE_IDS["selected"],
    PO_STORE_IDS["bench"],
    PO_STORE_IDS["cmabench"],
    PO_STORE_IDS["long_short"],
    PO_STORE_IDS["order"],
    PO_STORE_IDS["vol"],
    PO_STORE_IDS["min_wt"],
    PO_STORE_IDS["max_wt"],
    PO_STORE_IDS["force_max"],
    REG_STORE_IDS["selected"],
    REG_STORE_IDS["bench"],
    REG_STORE_IDS["long_short"],
    REG_STORE_IDS["order"],
    REG_STORE_IDS["vol"],
    REG_STORE_IDS["dep"],
    REG_STORE_IDS["lag"],
    REG_STORE_IDS["min_beta"],
    REG_STORE_IDS["max_beta"],
    REG_STORE_IDS["enable"],
]

ACCOUNT_LIST_EXTRA_CONTROL_STORE_IDS = (
    AT_EXTRA_CONTROL_STORE_IDS
    + PO_EXTRA_CONTROL_STORE_IDS
    + REG_EXTRA_CONTROL_STORE_IDS
)

ACCOUNT_LIST_CAPTURE_STORE_IDS = ACCOUNT_LIST_SERIES_DIALOG_STORE_IDS + ACCOUNT_LIST_EXTRA_CONTROL_STORE_IDS

ACCOUNT_LIST_LOAD_MERGE_STORE_IDS = [
    AT_STORE_IDS["selected"],
    AT_STORE_IDS["bench"],
    AT_STORE_IDS["long_short"],
    AT_STORE_IDS["order"],
    AT_STORE_IDS["vol"],
    PO_STORE_IDS["selected"],
    PO_STORE_IDS["bench"],
    PO_STORE_IDS["cmabench"],
    PO_STORE_IDS["long_short"],
    PO_STORE_IDS["order"],
    PO_STORE_IDS["vol"],
    PO_STORE_IDS["min_wt"],
    PO_STORE_IDS["max_wt"],
    PO_STORE_IDS["force_max"],
    REG_STORE_IDS["selected"],
    REG_STORE_IDS["bench"],
    REG_STORE_IDS["long_short"],
    REG_STORE_IDS["order"],
    REG_STORE_IDS["vol"],
    REG_STORE_IDS["dep"],
    REG_STORE_IDS["lag"],
    REG_STORE_IDS["min_beta"],
    REG_STORE_IDS["max_beta"],
    REG_STORE_IDS["enable"],
]

ACCOUNT_LIST_ENTRY_FRAME_CACHE_TTL_SECONDS = 60.0
ACCOUNT_LIST_PREFETCH_MAX_ENTRIES = 8
ACCOUNT_LIST_PREFETCH_MAX_MS = 750.0

_ACCOUNT_LIST_ENTRY_FRAME_CACHE: dict[str, tuple[float, pd.DataFrame, str]] = {}


def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)


def _table_name(db_engine: Engine, base: str) -> str:
    if db_engine.dialect.name == "sqlite":
        return f"[{base}]"
    return f"[dbo].[{base}]"


def _table_exists(db_engine: Engine, table_name: str) -> bool:
    insp = inspect(db_engine)
    if db_engine.dialect.name == "sqlite":
        return insp.has_table(table_name)
    return insp.has_table(table_name, schema="dbo") or insp.has_table(table_name)


def _rowcount_is_known_miss(rowcount: Any) -> bool:
    try:
        return int(rowcount) == 0
    except Exception:
        return False


def _rowcount_is_unknown(rowcount: Any) -> bool:
    try:
        return int(rowcount) < 0
    except Exception:
        return rowcount is None


def _timestamps_equal(left: Any, right: Any) -> bool:
    left_ts = pd.to_datetime(left, errors="coerce", utc=True)
    right_ts = pd.to_datetime(right, errors="coerce", utc=True)
    if pd.isna(left_ts) or pd.isna(right_ts):
        return False
    left_norm = pd.Timestamp(left_ts).tz_convert("UTC").tz_localize(None)
    right_norm = pd.Timestamp(right_ts).tz_convert("UTC").tz_localize(None)
    if left_norm == right_norm:
        return True
    return left_norm.floor("s") == right_norm.floor("s")


def _iso_or_none(value: Any) -> str | None:
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return None
    ts_norm = pd.Timestamp(ts).tz_convert("UTC").tz_localize(None)
    if int(ts_norm.microsecond) > 0:
        return ts_norm.strftime("%Y-%m-%d %H:%M:%S.%f")
    return ts_norm.strftime("%Y-%m-%d %H:%M:%S")


def _dedupe_strings(values: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        text_val = str(value or "").strip()
        if not text_val:
            continue
        key = text_val.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text_val)
    return out


def normalize_db_import_provenance_store(store: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(store, dict):
        return {}

    normalized: dict[str, dict[str, Any]] = {}
    for raw_key, raw_value in store.items():
        if not isinstance(raw_value, dict):
            continue
        loader_type = str(raw_value.get("loader_type") or "").strip().lower()
        if not loader_type:
            continue
        entry_id = str(raw_value.get("entry_id") or raw_key or "").strip() or uuid4().hex
        emitted_series = _dedupe_strings(raw_value.get("emitted_series"))
        if not emitted_series:
            continue
        primary_series = str(raw_value.get("primary_series") or "").strip() or emitted_series[0]
        normalized[entry_id] = {
            "entry_id": entry_id,
            "loader_type": loader_type,
            "loader_args": raw_value.get("loader_args") if isinstance(raw_value.get("loader_args"), dict) else {},
            "emitted_series": emitted_series,
            "primary_series": primary_series,
        }
    return normalized


def add_db_import_provenance_entry(
    store: Any,
    *,
    loader_type: str,
    loader_args: dict[str, Any],
    emitted_series: list[str],
    primary_series: str | None = None,
) -> dict[str, dict[str, Any]]:
    normalized = normalize_db_import_provenance_store(store)
    emitted = _dedupe_strings(emitted_series)
    if not emitted:
        return normalized
    entry_id = uuid4().hex
    normalized[entry_id] = {
        "entry_id": entry_id,
        "loader_type": str(loader_type or "").strip().lower(),
        "loader_args": dict(loader_args or {}),
        "emitted_series": emitted,
        "primary_series": str(primary_series or emitted[0]).strip() or emitted[0],
    }
    return normalized


def rename_db_import_provenance_series(store: Any, rename_map: dict[str, str]) -> dict[str, dict[str, Any]]:
    normalized = normalize_db_import_provenance_store(store)
    clean_map = {
        str(old).strip(): str(new).strip()
        for old, new in (rename_map or {}).items()
        if str(old).strip() and str(new).strip()
    }
    if not clean_map:
        return normalized

    updated: dict[str, dict[str, Any]] = {}
    for entry_id, entry in normalized.items():
        next_entry = dict(entry)
        next_entry["emitted_series"] = [clean_map.get(series, series) for series in _dedupe_strings(entry.get("emitted_series"))]
        primary_series = str(entry.get("primary_series") or "").strip()
        next_entry["primary_series"] = clean_map.get(primary_series, primary_series or next_entry["emitted_series"][0])
        updated[entry_id] = next_entry
    return updated


def remove_db_import_provenance_series(store: Any, deleted_series: list[str] | set[str]) -> dict[str, dict[str, Any]]:
    normalized = normalize_db_import_provenance_store(store)
    deleted = {str(name).strip() for name in (deleted_series or []) if str(name).strip()}
    if not deleted:
        return normalized

    updated: dict[str, dict[str, Any]] = {}
    for entry_id, entry in normalized.items():
        remaining = [series for series in entry.get("emitted_series", []) if series not in deleted]
        if not remaining:
            continue
        next_entry = dict(entry)
        next_entry["emitted_series"] = remaining
        primary_series = str(entry.get("primary_series") or "").strip()
        next_entry["primary_series"] = primary_series if primary_series in remaining else remaining[0]
        updated[entry_id] = next_entry
    return updated


def prune_db_import_provenance(store: Any, existing_columns: list[str] | set[str] | tuple[str, ...]) -> dict[str, dict[str, Any]]:
    normalized = normalize_db_import_provenance_store(store)
    allowed = {str(name).strip() for name in (existing_columns or []) if str(name).strip()}
    stale = [
        series
        for entry in normalized.values()
        for series in entry.get("emitted_series", [])
        if series not in allowed
    ]
    return remove_db_import_provenance_series(normalized, stale)


def account_list_tables_available(db_engine: Engine) -> bool:
    return _table_exists(db_engine, "DMAccountLists") and _table_exists(db_engine, "DMAccountListsArchive")


def users_table_available(db_engine: Engine) -> bool:
    return _table_exists(db_engine, "Users")


def _normalize_snapshot_list(value: Any, allowed: set[str]) -> list[str]:
    return [item for item in _dedupe_strings(value) if item in allowed]


def _normalize_snapshot_mapping(value: Any, allowed: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key or "").strip()
        if not key or key not in allowed:
            continue
        out[key] = raw_value
    return out


def _session_value(snapshot: Any, key: str, default: Any) -> Any:
    if isinstance(snapshot, dict) and key in snapshot:
        return snapshot.get(key)
    return default


def _saved_control_values(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): value.get(key)
        for key in ACCOUNT_LIST_CAPTURE_STORE_IDS
        if str(key) in value
    }


def _apply_max_end_sentinel(control_values: dict, raw_data_store: Any) -> dict:
    """Replace end dates that match max available with sentinel."""
    dataset_key = resolve_dataset_key(raw_data_store)
    if not dataset_key:
        return control_values
    result = dict(control_values)
    for store_id, cfg in _DATE_RANGE_SENTINEL_CONFIG.items():
        dr = result.get(store_id)
        if not isinstance(dr, dict) or not dr.get("end"):
            continue
        periodicity = result.get(cfg["periodicity"]) or "daily"
        selected = result.get(cfg["series"]) or ()
        if not selected:
            continue
        candidates = compute_date_range_candidates(dataset_key, periodicity, tuple(selected))
        max_end = candidates.get("max_end")
        if max_end and dr["end"] == max_end:
            result[store_id] = {**dr, "end": ACCOUNT_LIST_MAX_END_SENTINEL}
    return result


def _series_names_from_entries(entries: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        for series in entry.get("emitted_series", []):
            key = str(series or "").strip()
            if not key or key.lower() in seen:
                continue
            seen.add(key.lower())
            out.append(key)
    return out


def build_account_list_payload(
    provenance_store: Any,
    session_snapshot: Any,
    raw_data_store: Any = None,
) -> dict[str, Any]:
    entries = list(normalize_db_import_provenance_store(provenance_store).values())
    control_values = _saved_control_values(session_snapshot)
    if raw_data_store is not None:
        control_values = _apply_max_end_sentinel(control_values, raw_data_store)
    return {
        "schema_version": ACCOUNT_LIST_SCHEMA_VERSION,
        "captured_at": _now_utc().strftime("%Y-%m-%d %H:%M:%S"),
        "series_entries": entries,
        "control_values": control_values,
    }


def normalize_account_list_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, str):
        raw = payload.strip()
        if not raw:
            return {}
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return {}
    if not isinstance(payload, dict):
        return {}

    entries = list(normalize_db_import_provenance_store({
        str((entry or {}).get("entry_id") or idx): entry
        for idx, entry in enumerate(payload.get("series_entries") or [])
        if isinstance(entry, dict)
    }).values())

    return {
        "schema_version": int(payload.get("schema_version") or ACCOUNT_LIST_SCHEMA_VERSION),
        "captured_at": str(payload.get("captured_at") or ""),
        "series_entries": entries,
        "control_values": _saved_control_values(payload.get("control_values")),
    }


def account_list_preview_rows(payload: Any) -> list[dict[str, Any]]:
    normalized = normalize_account_list_payload(payload)
    if not normalized:
        return []
    series_names = set(_series_names_from_entries(normalized.get("series_entries", [])))
    control_values = normalized.get("control_values") if isinstance(normalized.get("control_values"), dict) else {}
    at_selected = set(_normalize_snapshot_list(control_values.get(AT_STORE_IDS["selected"]), series_names))
    po_selected = set(_normalize_snapshot_list(control_values.get(PO_STORE_IDS["selected"]), series_names))
    reg_selected = set(_normalize_snapshot_list(control_values.get(REG_STORE_IDS["selected"]), series_names))
    reg_dep = str(control_values.get(REG_STORE_IDS["dep"]) or "").strip()
    if reg_dep:
        reg_selected.add(reg_dep)

    preview: list[dict[str, Any]] = []
    for entry in normalized.get("series_entries", []):
        source_type = str(entry.get("loader_type") or "").strip()
        for series in entry.get("emitted_series", []):
            preview.append(
                {
                    "Series": series,
                    "SourceType": source_type,
                    "AT": series in at_selected,
                    "PO": series in po_selected,
                    "REG": series in reg_selected,
                }
            )
    return preview


def _load_account_list_row_by_id(conn, db_engine: Engine, account_list_id: Any, username: str | None = None) -> dict[str, Any] | None:
    table_name = _table_name(db_engine, "DMAccountLists")
    q = text(
        f"SELECT AccountListID, Username, ListName, ConfigJson, UPDATE_DATE, UPDATE_BY "
        f"FROM {table_name} WHERE AccountListID = :account_list_id"
    )
    row = conn.execute(q, {"account_list_id": account_list_id}).mappings().first()
    if not row:
        return None
    output = dict(row)
    if username and str(output.get("Username") or "").strip().lower() != str(username or "").strip().lower():
        return None
    return output


def _account_list_summary_row(row: dict[str, Any]) -> dict[str, Any]:
    payload = normalize_account_list_payload(row.get("ConfigJson"))
    preview_rows = account_list_preview_rows(payload)
    return {
        "AccountListID": int(row.get("AccountListID")),
        "Username": str(row.get("Username") or "").strip(),
        "ListName": str(row.get("ListName") or "").strip(),
        "UPDATE_DATE": _iso_or_none(row.get("UPDATE_DATE")),
        "UPDATE_BY": str(row.get("UPDATE_BY") or "").strip(),
        "SeriesCount": len({str(item.get("Series") or "").strip() for item in preview_rows if str(item.get("Series") or "").strip()}),
        "PreviewRows": preview_rows,
        "ConfigJson": canonical_json_dumps(payload),
    }


def _account_list_list_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "AccountListID": int(row.get("AccountListID")),
        "Username": str(row.get("Username") or "").strip(),
        "ListName": str(row.get("ListName") or "").strip(),
        "UPDATE_DATE": _iso_or_none(row.get("UPDATE_DATE")),
        "UPDATE_BY": str(row.get("UPDATE_BY") or "").strip(),
        "SeriesCount": None,
    }


def _account_list_detail_row(row: dict[str, Any]) -> dict[str, Any]:
    payload = normalize_account_list_payload(row.get("ConfigJson"))
    return {
        **_account_list_list_row(row),
        "ConfigJson": canonical_json_dumps(payload),
    }


def save_account_list(
    db_engine: Engine,
    *,
    username: str,
    update_by: str,
    list_name: str,
    payload: Any,
) -> tuple[bool, str, dict[str, Any] | None]:
    if not account_list_tables_available(db_engine):
        return False, "Account-list tables are unavailable.", None
    clean_username = str(username or "").strip()
    clean_name = str(list_name or "").strip()
    clean_update_by = str(update_by or "").strip() or clean_username or "unknown"
    if not clean_username:
        return False, "Username is required.", None
    if not clean_name:
        return False, "List name is required.", None

    normalized_payload = normalize_account_list_payload(payload)
    if not normalized_payload.get("series_entries"):
        return False, "Select at least one DB-backed series before saving an account list.", None

    table_name = _table_name(db_engine, "DMAccountLists")
    now_val = _now_utc()
    with db_engine.begin() as conn:
        result = conn.execute(
            text(
                f"INSERT INTO {table_name} ("
                "Username, ListName, ConfigJson, UPDATE_DATE, UPDATE_BY"
                ") VALUES ("
                ":Username, :ListName, :ConfigJson, :UPDATE_DATE, :UPDATE_BY"
                ")"
            ),
            {
                "Username": clean_username,
                "ListName": clean_name,
                "ConfigJson": canonical_json_dumps(normalized_payload),
                "UPDATE_DATE": now_val,
                "UPDATE_BY": clean_update_by,
            },
        )
        inserted_id = getattr(result, "lastrowid", None)
        if inserted_id is None:
            if db_engine.dialect.name == "sqlite":
                inserted_id = conn.execute(text("SELECT last_insert_rowid()")).scalar()
            else:
                inserted_id = conn.execute(
                    text(
                        f"SELECT TOP 1 AccountListID FROM {table_name} "
                        "WHERE Username = :Username AND ListName = :ListName AND UPDATE_DATE = :UPDATE_DATE "
                        "ORDER BY AccountListID DESC"
                    ),
                    {
                        "Username": clean_username,
                        "ListName": clean_name,
                        "UPDATE_DATE": now_val,
                    },
                ).scalar()
        saved = _load_account_list_row_by_id(conn, db_engine, inserted_id, clean_username)
        if saved is None:
            return False, "Unable to reload saved account list.", None
        return True, f"Saved account list `{clean_name}`.", _account_list_summary_row(saved)


def list_account_list_users(db_engine: Engine, current_username: str) -> list[dict[str, str]]:
    if not users_table_available(db_engine):
        return []
    clean_username = str(current_username or "").strip()
    table_name = _table_name(db_engine, "Users")
    query_sql = (
        f"SELECT Username, Role FROM {table_name} "
        "WHERE Username IS NOT NULL AND LTRIM(RTRIM(Username)) <> '' "
    )
    params: dict[str, Any] = {}
    if clean_username:
        query_sql += "AND LOWER(LTRIM(RTRIM(Username))) <> LOWER(LTRIM(RTRIM(:current_username))) "
        params["current_username"] = clean_username
    query_sql += "ORDER BY Username"
    with db_engine.connect() as conn:
        rows = conn.execute(text(query_sql), params).mappings().all()
    return [
        {
            "Username": str(row.get("Username") or "").strip(),
            "Role": str(row.get("Role") or "").strip(),
        }
        for row in rows
        if str(row.get("Username") or "").strip()
    ]


def list_account_lists(db_engine: Engine, username: str) -> list[dict[str, Any]]:
    if not account_list_tables_available(db_engine):
        return []
    clean_username = str(username or "").strip()
    if not clean_username:
        return []
    table_name = _table_name(db_engine, "DMAccountLists")
    q = text(
        f"SELECT AccountListID, Username, ListName, UPDATE_DATE, UPDATE_BY "
        f"FROM {table_name} "
        "WHERE LOWER(LTRIM(RTRIM(Username))) = LOWER(LTRIM(RTRIM(:username))) "
        "ORDER BY UPDATE_DATE DESC, AccountListID DESC"
    )
    with timed_block("account_list.list_rows", username=clean_username):
        with db_engine.connect() as conn:
            rows = conn.execute(q, {"username": clean_username}).mappings().all()
    return [_account_list_list_row(dict(row)) for row in rows]


def load_account_list_by_id(db_engine: Engine, account_list_id: Any, username: str) -> dict[str, Any] | None:
    if not account_list_tables_available(db_engine):
        return None
    clean_username = str(username or "").strip()
    if not clean_username:
        return None
    with timed_block("account_list.load_detail", account_list_id=account_list_id):
        with db_engine.connect() as conn:
            row = _load_account_list_row_by_id(conn, db_engine, account_list_id, clean_username)
    return _account_list_detail_row(row) if row else None


def _archive_account_list_row(conn, db_engine: Engine, row: dict[str, Any]) -> None:
    archive_table = _table_name(db_engine, "DMAccountListsArchive")
    conn.execute(
        text(
            f"INSERT INTO {archive_table} ("
            "AccountListID, Username, ListName, ConfigJson, UPDATE_DATE, UPDATE_BY, ARCHIVE_DATE"
            ") VALUES ("
            ":AccountListID, :Username, :ListName, :ConfigJson, :UPDATE_DATE, :UPDATE_BY, :ARCHIVE_DATE"
            ")"
        ),
        {
            "AccountListID": row.get("AccountListID"),
            "Username": row.get("Username"),
            "ListName": row.get("ListName"),
            "ConfigJson": row.get("ConfigJson"),
            "UPDATE_DATE": row.get("UPDATE_DATE"),
            "UPDATE_BY": row.get("UPDATE_BY"),
            "ARCHIVE_DATE": _now_utc(),
        },
    )


def delete_account_list(
    db_engine: Engine,
    *,
    account_list_id: Any,
    username: str,
    expected_update_date: str | None = None,
) -> tuple[bool, str]:
    if not account_list_tables_available(db_engine):
        return False, "Account-list tables are unavailable."
    clean_username = str(username or "").strip()
    if not clean_username:
        return False, "Username is required."
    table_name = _table_name(db_engine, "DMAccountLists")
    is_mssql = db_engine.dialect.name == "mssql"
    with db_engine.begin() as conn:
        current = _load_account_list_row_by_id(conn, db_engine, account_list_id, clean_username)
        if current is None:
            return False, "Account list no longer exists."
        if expected_update_date and not _timestamps_equal(current.get("UPDATE_DATE"), expected_update_date):
            return False, "Account list changed in another session. Reload before deleting."

        _archive_account_list_row(conn, db_engine, current)
        delete_sql = (
            f"DELETE FROM {table_name} WHERE AccountListID = :account_list_id "
            "AND LOWER(LTRIM(RTRIM(Username))) = LOWER(LTRIM(RTRIM(:username))) "
        )
        params: dict[str, Any] = {"account_list_id": account_list_id, "username": clean_username}
        if not is_mssql:
            delete_sql += "AND UPDATE_DATE = :ExpectedDbUpdateDate"
            params["ExpectedDbUpdateDate"] = current.get("UPDATE_DATE")
        result = conn.execute(text(delete_sql), params)
        if _rowcount_is_known_miss(result.rowcount):
            return False, "Account list changed in another session. Reload before deleting."
        if _rowcount_is_unknown(result.rowcount):
            refreshed = _load_account_list_row_by_id(conn, db_engine, account_list_id, clean_username)
            if refreshed is not None:
                return False, "Account list changed in another session. Reload before deleting."
        return True, f"Deleted account list `{current.get('ListName')}`."


def send_account_list(
    db_engine: Engine,
    *,
    account_list_id: Any,
    sender_username: str,
    recipient_username: str,
    expected_update_date: str | None = None,
) -> tuple[bool, str]:
    if not account_list_tables_available(db_engine):
        return False, "Account-list tables are unavailable."
    if not users_table_available(db_engine):
        return False, "Users table is unavailable."

    clean_sender = str(sender_username or "").strip()
    clean_recipient = str(recipient_username or "").strip()
    if not clean_sender:
        return False, "Username is required."
    if not clean_recipient:
        return False, "Select a user to send to."
    if clean_sender.lower() == clean_recipient.lower():
        return False, "Choose a different user."

    users_table = _table_name(db_engine, "Users")
    account_lists_table = _table_name(db_engine, "DMAccountLists")
    now_val = _now_utc()

    with db_engine.begin() as conn:
        current = _load_account_list_row_by_id(conn, db_engine, account_list_id, clean_sender)
        if current is None:
            return False, "Account list no longer exists."
        if expected_update_date and not _timestamps_equal(current.get("UPDATE_DATE"), expected_update_date):
            return False, "Account list changed in another session. Reload before sending."

        recipient_exists = conn.execute(
            text(
                f"SELECT 1 FROM {users_table} "
                "WHERE LOWER(LTRIM(RTRIM(Username))) = LOWER(LTRIM(RTRIM(:recipient_username)))"
            ),
            {"recipient_username": clean_recipient},
        ).scalar()
        if recipient_exists is None:
            return False, "Selected user no longer exists."

        result = conn.execute(
            text(
                f"INSERT INTO {account_lists_table} (Username, ListName, ConfigJson, UPDATE_DATE, UPDATE_BY) "
                f"SELECT :recipient_username, ListName, ConfigJson, :update_date, :update_by "
                f"FROM {account_lists_table} "
                "WHERE AccountListID = :account_list_id "
                "AND LOWER(LTRIM(RTRIM(Username))) = LOWER(LTRIM(RTRIM(:sender_username))) "
                "AND UPDATE_DATE = :source_update_date"
            ),
            {
                "recipient_username": clean_recipient,
                "update_date": now_val,
                "update_by": clean_sender,
                "account_list_id": account_list_id,
                "sender_username": clean_sender,
                "source_update_date": current.get("UPDATE_DATE"),
            },
        )
        if _rowcount_is_known_miss(result.rowcount):
            return False, "Account list changed in another session. Reload before sending."
        if _rowcount_is_unknown(result.rowcount):
            copied = conn.execute(
                text(
                    f"SELECT 1 FROM {account_lists_table} "
                    "WHERE LOWER(LTRIM(RTRIM(Username))) = LOWER(LTRIM(RTRIM(:recipient_username))) "
                    "AND ListName = :list_name AND UPDATE_DATE = :update_date AND UPDATE_BY = :update_by"
                ),
                {
                    "recipient_username": clean_recipient,
                    "list_name": current.get("ListName"),
                    "update_date": now_val,
                    "update_by": clean_sender,
                },
            ).scalar()
            if copied is None:
                return False, "Account list changed in another session. Reload before sending."

        return True, f"Sent account list `{current.get('ListName')}` to `{clean_recipient}`."


def _normalize_monthly_df_if_needed(df: pd.DataFrame, periodicity: str) -> pd.DataFrame:
    if periodicity == "monthly":
        return align_monthly_index_to_month_end(df)
    return df


def _merge_imported_with_existing(
    existing_df: pd.DataFrame | None,
    existing_periodicity: str | None,
    new_df: pd.DataFrame,
    new_periodicity: str,
) -> tuple[pd.DataFrame, str]:
    if existing_df is None or existing_df.empty:
        merged = _normalize_monthly_df_if_needed(new_df, new_periodicity)
        return merged, new_periodicity

    resolved_existing = str(existing_periodicity or "daily")
    working_existing = existing_df
    working_new = new_df
    if resolved_existing == "monthly" and new_periodicity == "daily":
        working_new = resample_returns(working_new, "monthly")
        combined_periodicity = "monthly"
    elif new_periodicity == "monthly" and resolved_existing == "daily":
        working_existing = resample_returns(working_existing, "monthly")
        combined_periodicity = "monthly"
    else:
        combined_periodicity = resolved_existing
    working_existing = _normalize_monthly_df_if_needed(working_existing, combined_periodicity)
    working_new = _normalize_monthly_df_if_needed(working_new, combined_periodicity)
    return merge_returns(working_existing, working_new), combined_periodicity


def _entry_load_priority_key(entry: dict[str, Any]) -> tuple[int, str]:
    emitted = _dedupe_strings(entry.get("emitted_series"))
    primary_series = str(entry.get("primary_series") or "").strip()
    return (-len(emitted), primary_series.lower())


def _entry_frame_cache_now() -> float:
    return time.monotonic()


def _entry_frame_cache_key(entry: dict[str, Any]) -> str:
    payload = {
        "loader_type": str(entry.get("loader_type") or "").strip().lower(),
        "loader_args": entry.get("loader_args") if isinstance(entry.get("loader_args"), dict) else {},
    }
    return canonical_json_dumps(payload)


def _clear_account_list_entry_frame_cache() -> None:
    _ACCOUNT_LIST_ENTRY_FRAME_CACHE.clear()


def _load_entry_frame_uncached(
    entry: dict[str, Any],
    *,
    db_engine: Engine,
    mrd_engine: Engine,
    perf_engine: Engine,
) -> tuple[pd.DataFrame, str]:
    loader_type = str(entry.get("loader_type") or "").strip().lower()
    loader_args = entry.get("loader_args") if isinstance(entry.get("loader_args"), dict) else {}

    if loader_type == "cma_bench":
        benches = _dedupe_strings(loader_args.get("selected_benches"))
        df, _meta = load_cma_returns_for_benches_with_meta(db_engine, benches, mrd_engine)
        return df, "daily"
    if loader_type == "raw_factor":
        result = load_factor_series(mrd_engine, loader_args.get("rows") or [])
        return result.returns_df, result.periodicity or "monthly"
    if loader_type == "raw_funds":
        result = load_fund_series(mrd_engine, loader_args.get("rows") or [])
        return result.returns_df, result.periodicity or "monthly"
    if loader_type == "raw_performance":
        result = load_performance_series(perf_engine, loader_args.get("rows") or [])
        return result.returns_df, result.periodicity or "monthly"
    if loader_type == "underlying_category":
        result = load_underlying_category_series(db_engine, loader_args.get("rows") or [])
        return result.returns_df, result.periodicity or "daily"
    if loader_type in {"portfolio_peer", "portfolio_index", "portfolio_other"}:
        mode = loader_type.split("_", 1)[1]
        result = load_portfolio_series(
            db_engine,
            mode,
            loader_args.get("rows") or [],
            performance_engine=perf_engine,
        )
        return result.returns_df, result.periodicity or "monthly"
    return pd.DataFrame(), "monthly"


def _load_entry_frame(
    entry: dict[str, Any],
    *,
    db_engine: Engine,
    mrd_engine: Engine,
    perf_engine: Engine,
) -> tuple[pd.DataFrame, str]:
    cache_key = _entry_frame_cache_key(entry)
    now = _entry_frame_cache_now()
    cached = _ACCOUNT_LIST_ENTRY_FRAME_CACHE.get(cache_key)
    if cached is not None:
        cached_at, cached_df, cached_periodicity = cached
        if (now - cached_at) <= ACCOUNT_LIST_ENTRY_FRAME_CACHE_TTL_SECONDS:
            return cached_df.copy(), cached_periodicity

    df, periodicity = _load_entry_frame_uncached(
        entry,
        db_engine=db_engine,
        mrd_engine=mrd_engine,
        perf_engine=perf_engine,
    )
    _ACCOUNT_LIST_ENTRY_FRAME_CACHE[cache_key] = (now, df.copy(), periodicity)
    return df, periodicity


def prefetch_account_list_entry_frames(
    payload: Any,
    *,
    db_engine: Engine,
    mrd_engine: Engine,
    perf_engine: Engine,
) -> dict[str, Any]:
    normalized_payload = normalize_account_list_payload(payload)
    entries_to_load = sorted(
        normalized_payload.get("series_entries", []),
        key=_entry_load_priority_key,
    )
    started_at = _entry_frame_cache_now()
    warmed_count = 0
    attempted_count = 0
    budget_exhausted = False

    for entry in entries_to_load:
        if attempted_count >= ACCOUNT_LIST_PREFETCH_MAX_ENTRIES:
            budget_exhausted = True
            break
        elapsed_ms = (_entry_frame_cache_now() - started_at) * 1000.0
        if elapsed_ms >= ACCOUNT_LIST_PREFETCH_MAX_MS:
            budget_exhausted = True
            break
        if not _dedupe_strings(entry.get("emitted_series")):
            continue
        attempted_count += 1
        _load_entry_frame(
            entry,
            db_engine=db_engine,
            mrd_engine=mrd_engine,
            perf_engine=perf_engine,
        )
        warmed_count += 1

    return {
        "entry_count": len(entries_to_load),
        "attempted_count": attempted_count,
        "warmed_count": warmed_count,
        "budget_exhausted": budget_exhausted,
    }


def _merge_selected_list(current: Any, saved: Any, restorable_series: set[str]) -> list[str]:
    current_list = _dedupe_strings(current)
    out = list(current_list)
    for series in _dedupe_strings(saved):
        if series in restorable_series and series not in out:
            out.append(series)
    return out


def _merge_order_list(current: Any, saved: Any, merged_columns: list[str], restorable_series: set[str]) -> list[str]:
    merged_set = set(merged_columns)
    out: list[str] = []
    for series in _dedupe_strings(current):
        if series in merged_set and series not in out:
            out.append(series)
    for series in _dedupe_strings(saved):
        if series in restorable_series and series in merged_set and series not in out:
            out.append(series)
    for series in merged_columns:
        if series not in out:
            out.append(series)
    return out


def _merge_boolean_map(current: Any, saved: Any, restorable_series: set[str]) -> dict[str, Any]:
    out = dict(current or {}) if isinstance(current, dict) else {}
    if isinstance(saved, dict):
        for key, value in saved.items():
            clean_key = str(key or "").strip()
            if clean_key and clean_key in restorable_series:
                out[clean_key] = value
    return out


def _normalize_benchmark_map(current: Any, saved: Any, restorable_series: set[str], available_series: set[str]) -> dict[str, str]:
    out = dict(current or {}) if isinstance(current, dict) else {}
    if not isinstance(saved, dict):
        return out
    for key, value in saved.items():
        clean_key = str(key or "").strip()
        if clean_key not in restorable_series:
            continue
        benchmark = str(value or "None").strip() or "None"
        if benchmark != "None" and benchmark not in available_series:
            benchmark = "None"
        out[clean_key] = benchmark
    return out


def _filter_monthly_series_list(saved: Any, available_series: set[str]) -> list[str]:
    return _normalize_snapshot_list(saved, available_series)


def _filter_factor_series_value(saved: Any, available_series: set[str]) -> str | None:
    text_val = str(saved or "").strip()
    if not text_val:
        return None
    if text_val.startswith("raw::"):
        raw_name = text_val.split("::", 1)[1]
        return text_val if raw_name in available_series else None
    return text_val


def _filtered_extra_control_value(store_id: str, saved_value: Any, available_series: set[str]) -> Any:
    if store_id == "at-monthly-series-store":
        return _filter_monthly_series_list(saved_value, available_series)
    if store_id == "at-factor-series-store":
        return _filter_factor_series_value(saved_value, available_series)
    return saved_value


def build_account_list_session_payload(
    *,
    payload: Any,
    current_raw_data: dict[str, Any] | None,
    current_original_periodicity: str | None,
    current_provenance: Any,
    current_session_snapshot: Any,
    apply_settings: bool,
    db_engine: Engine,
    mrd_engine: Engine,
    perf_engine: Engine,
) -> tuple[dict[str, Any], dict[str, Any]]:
    with timed_block("account_list.build_session_payload", apply_settings=bool(apply_settings)):
        normalized_payload = normalize_account_list_payload(payload)
        if not normalized_payload.get("series_entries"):
            raise ValueError("Saved account list has no DB-backed series.")

        dataset_key = get_dataset_key(current_raw_data) if current_raw_data else None
        existing_df = get_raw_dataset_df(dataset_key) if dataset_key else pd.DataFrame()
        existing_set = set(existing_df.columns)
        merged_df = existing_df.copy()
        combined_periodicity = str(current_original_periodicity or "daily")
        updated_provenance = normalize_db_import_provenance_store(current_provenance)

        added_series: list[str] = []
        skipped_conflicts: list[str] = []
        missing_series: list[str] = []
        added_set: set[str] = set()

        entries_to_load = sorted(
            normalized_payload.get("series_entries", []),
            key=_entry_load_priority_key,
        )

        for entry in entries_to_load:
            entry_emitted = _dedupe_strings(entry.get("emitted_series"))
            if not entry_emitted:
                continue
            pending_series = [
                series
                for series in entry_emitted
                if series not in existing_set and series not in added_set
            ]
            if not pending_series:
                skipped_conflicts.extend(entry_emitted)
                continue

            with timed_block(
                "account_list.build_session_payload.load_entry_frame",
                loader_type=str(entry.get("loader_type") or ""),
                emitted_count=len(entry_emitted),
            ):
                entry_df, entry_periodicity = _load_entry_frame(
                    entry,
                    db_engine=db_engine,
                    mrd_engine=mrd_engine,
                    perf_engine=perf_engine,
                )
            if entry_df.empty:
                missing_series.extend(entry.get("emitted_series", []))
                continue

            candidate_cols = [col for col in entry_df.columns if col not in existing_set and col not in added_set]
            conflict_cols = [col for col in entry_df.columns if col in existing_set or col in added_set]
            if conflict_cols:
                skipped_conflicts.extend(conflict_cols)
            if not candidate_cols:
                continue

            filtered_df = entry_df[candidate_cols].copy()
            with timed_block(
                "account_list.build_session_payload.merge_entry_frame",
                incoming_series=len(candidate_cols),
                periodicity=entry_periodicity,
            ):
                merged_df, combined_periodicity = _merge_imported_with_existing(
                    merged_df if not merged_df.empty else None,
                    combined_periodicity,
                    filtered_df,
                    entry_periodicity,
                )
            added_series.extend(candidate_cols)
            added_set.update(candidate_cols)
            existing_set = set(merged_df.columns)
            updated_provenance = add_db_import_provenance_entry(
                updated_provenance,
                loader_type=str(entry.get("loader_type") or ""),
                loader_args=dict(entry.get("loader_args") or {}),
                emitted_series=candidate_cols,
                primary_series=str(entry.get("primary_series") or candidate_cols[0]).strip() or candidate_cols[0],
            )
            missing_series.extend([series for series in entry.get("emitted_series", []) if series not in entry_df.columns])

        with timed_block(
            "account_list.build_session_payload.assemble",
            added_series=len(added_series),
            apply_settings=bool(apply_settings),
        ):
            updated_provenance = prune_db_import_provenance(updated_provenance, list(merged_df.columns))
            available_series = set(merged_df.columns)
            merged_columns = list(merged_df.columns)
            current_snapshot = current_session_snapshot if isinstance(current_session_snapshot, dict) else {}

            control_values = normalized_payload.get("control_values") if isinstance(normalized_payload.get("control_values"), dict) else {}

            at_selected = _merge_selected_list(current_snapshot.get(AT_STORE_IDS["selected"]), control_values.get(AT_STORE_IDS["selected"]), available_series)
            at_order = _merge_order_list(current_snapshot.get(AT_STORE_IDS["order"]), control_values.get(AT_STORE_IDS["order"]), merged_columns, available_series)
            at_bench = _normalize_benchmark_map(current_snapshot.get(AT_STORE_IDS["bench"]), control_values.get(AT_STORE_IDS["bench"]), available_series, available_series)
            at_ls = _merge_boolean_map(current_snapshot.get(AT_STORE_IDS["long_short"]), control_values.get(AT_STORE_IDS["long_short"]), available_series)
            at_vol = _merge_boolean_map(current_snapshot.get(AT_STORE_IDS["vol"]), control_values.get(AT_STORE_IDS["vol"]), available_series)

            po_selected = _merge_selected_list(current_snapshot.get(PO_STORE_IDS["selected"]), control_values.get(PO_STORE_IDS["selected"]), available_series)
            po_order = _merge_order_list(current_snapshot.get(PO_STORE_IDS["order"]), control_values.get(PO_STORE_IDS["order"]), merged_columns, available_series)
            po_bench = _normalize_benchmark_map(current_snapshot.get(PO_STORE_IDS["bench"]), control_values.get(PO_STORE_IDS["bench"]), available_series, available_series)
            po_cmabench = dict(current_snapshot.get(PO_STORE_IDS["cmabench"]) or {})
            for key, value in dict(control_values.get(PO_STORE_IDS["cmabench"]) or {}).items():
                clean_key = str(key or "").strip()
                if clean_key in available_series:
                    po_cmabench[clean_key] = str(value or "").strip()
            po_ls = _merge_boolean_map(current_snapshot.get(PO_STORE_IDS["long_short"]), control_values.get(PO_STORE_IDS["long_short"]), available_series)
            po_vol = _merge_boolean_map(current_snapshot.get(PO_STORE_IDS["vol"]), control_values.get(PO_STORE_IDS["vol"]), available_series)
            po_min = _merge_boolean_map(current_snapshot.get(PO_STORE_IDS["min_wt"]), control_values.get(PO_STORE_IDS["min_wt"]), available_series)
            po_max = _merge_boolean_map(current_snapshot.get(PO_STORE_IDS["max_wt"]), control_values.get(PO_STORE_IDS["max_wt"]), available_series)
            po_force = _merge_boolean_map(current_snapshot.get(PO_STORE_IDS["force_max"]), control_values.get(PO_STORE_IDS["force_max"]), available_series)

            reg_selected = _merge_selected_list(current_snapshot.get(REG_STORE_IDS["selected"]), control_values.get(REG_STORE_IDS["selected"]), available_series)
            reg_order = _merge_order_list(current_snapshot.get(REG_STORE_IDS["order"]), control_values.get(REG_STORE_IDS["order"]), merged_columns, available_series)
            reg_bench = _normalize_benchmark_map(current_snapshot.get(REG_STORE_IDS["bench"]), control_values.get(REG_STORE_IDS["bench"]), available_series, available_series)
            reg_ls = _merge_boolean_map(current_snapshot.get(REG_STORE_IDS["long_short"]), control_values.get(REG_STORE_IDS["long_short"]), available_series)
            reg_vol = _merge_boolean_map(current_snapshot.get(REG_STORE_IDS["vol"]), control_values.get(REG_STORE_IDS["vol"]), available_series)
            reg_lag = _merge_boolean_map(current_snapshot.get(REG_STORE_IDS["lag"]), control_values.get(REG_STORE_IDS["lag"]), available_series)
            reg_min = _merge_boolean_map(current_snapshot.get(REG_STORE_IDS["min_beta"]), control_values.get(REG_STORE_IDS["min_beta"]), available_series)
            reg_max = _merge_boolean_map(current_snapshot.get(REG_STORE_IDS["max_beta"]), control_values.get(REG_STORE_IDS["max_beta"]), available_series)
            reg_enable = _merge_boolean_map(current_snapshot.get(REG_STORE_IDS["enable"]), control_values.get(REG_STORE_IDS["enable"]), available_series)
            current_dep = str(current_snapshot.get(REG_STORE_IDS["dep"]) or "").strip()
            saved_dep = str(control_values.get(REG_STORE_IDS["dep"]) or "").strip()
            if saved_dep and saved_dep in available_series:
                reg_dep = saved_dep
            elif current_dep and current_dep in available_series:
                reg_dep = current_dep
            else:
                reg_dep = None

            notice = {
                "message": (
                    f"Loaded account list with {len(added_series)} added series, "
                    f"{len(_dedupe_strings(skipped_conflicts))} skipped conflicts, and {len(_dedupe_strings(missing_series))} missing series."
                ),
                "color": "orange" if skipped_conflicts or missing_series else "green",
            }

            raw_data_payload = build_raw_data_store_payload(merged_df)
            raw_data_dataset_key = str(raw_data_payload.get("dataset_key") or "").strip() or None
            session_payload = {
                "dashmat-raw-data-store": raw_data_payload,
                "dashmat-raw-data-identity-store": {
                    "dataset_key": raw_data_dataset_key,
                    "has_data": bool(raw_data_dataset_key),
                },
                "dashmat-raw-data-meta-store": build_raw_data_metadata(raw_data_payload, combined_periodicity),
                "dashmat-original-periodicity-store": combined_periodicity,
                "dashmat-db-import-provenance-store": updated_provenance,
                "dashmat-account-list-notice-store": notice,
                AT_STORE_IDS["selected"]: at_selected,
                AT_STORE_IDS["selected_value"]: at_selected,
                AT_STORE_IDS["bench"]: at_bench,
                AT_STORE_IDS["long_short"]: at_ls,
                AT_STORE_IDS["order"]: at_order,
                AT_STORE_IDS["vol"]: at_vol,
                PO_STORE_IDS["selected"]: po_selected,
                PO_STORE_IDS["selected_value"]: po_selected,
                PO_STORE_IDS["bench"]: po_bench,
                PO_STORE_IDS["cmabench"]: po_cmabench,
                PO_STORE_IDS["long_short"]: po_ls,
                PO_STORE_IDS["order"]: po_order,
                PO_STORE_IDS["vol"]: po_vol,
                PO_STORE_IDS["min_wt"]: po_min,
                PO_STORE_IDS["max_wt"]: po_max,
                PO_STORE_IDS["force_max"]: po_force,
                REG_STORE_IDS["selected"]: reg_selected,
                REG_STORE_IDS["selected_value"]: reg_selected,
                REG_STORE_IDS["bench"]: reg_bench,
                REG_STORE_IDS["long_short"]: reg_ls,
                REG_STORE_IDS["order"]: reg_order,
                REG_STORE_IDS["vol"]: reg_vol,
                REG_STORE_IDS["dep"]: reg_dep,
                REG_STORE_IDS["lag"]: reg_lag,
                REG_STORE_IDS["min_beta"]: reg_min,
                REG_STORE_IDS["max_beta"]: reg_max,
                REG_STORE_IDS["enable"]: reg_enable,
            }

            if apply_settings:
                for store_id in ACCOUNT_LIST_EXTRA_CONTROL_STORE_IDS:
                    if store_id not in control_values:
                        continue
                    session_payload[store_id] = _filtered_extra_control_value(
                        store_id,
                        control_values.get(store_id),
                        available_series,
                    )

    return session_payload, {
        "added_series": added_series,
        "skipped_conflicts": _dedupe_strings(skipped_conflicts),
        "missing_series": _dedupe_strings(missing_series),
        "notice": notice,
    }
