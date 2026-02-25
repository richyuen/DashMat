"""Regime definition storage and validation helpers for AnalyticsTool."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any

import pandas as pd
from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

from utils.serialization import canonical_json_dumps, parse_mapping_payload


REGIME_METHOD_OPTIONS = [
    {"value": "1", "label": "1 - HMM on PC1"},
    {"value": "2", "label": "2 - Quantiles on PC1"},
    {"value": "3", "label": "3 - Quantiles on Single Series"},
]

REGIME_RETURN_BASIS_OPTIONS = [
    {"value": "total", "label": "Total"},
    {"value": "excess", "label": "Excess"},
]


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


def _iso_or_none(value: Any) -> str | None:
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return None
    ts_norm = pd.Timestamp(ts).tz_convert("UTC").tz_localize(None)
    if int(ts_norm.microsecond) > 0:
        return ts_norm.strftime("%Y-%m-%d %H:%M:%S.%f")
    return ts_norm.strftime("%Y-%m-%d %H:%M:%S")


def _parse_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        return default
    return int(parsed)


def _parse_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        return default
    return float(parsed)


def _parse_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text_val = str(value).strip().lower()
    if text_val in {"true", "1", "yes", "y", "on"}:
        return True
    if text_val in {"false", "0", "no", "n", "off"}:
        return False
    return default


def _parse_series_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        parts = [str(v).strip() for v in value]
    else:
        parts = [v.strip() for v in str(value).split(",")]
    out: list[str] = []
    seen: set[str] = set()
    for part in parts:
        if not part:
            continue
        key = part.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(part)
    return out


def _parse_config(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        text_val = value.strip()
        if not text_val:
            return {}
        try:
            parsed = json.loads(text_val)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _timestamps_equal(left: Any, right: Any) -> bool:
    left_ts = pd.to_datetime(left, errors="coerce", utc=True)
    right_ts = pd.to_datetime(right, errors="coerce", utc=True)
    if pd.isna(left_ts) or pd.isna(right_ts):
        return False
    left_norm = pd.Timestamp(left_ts).tz_convert("UTC").tz_localize(None)
    right_norm = pd.Timestamp(right_ts).tz_convert("UTC").tz_localize(None)
    if left_norm == right_norm:
        return True
    # Allow second-level match for clients that truncate fractional seconds.
    return left_norm.floor("s") == right_norm.floor("s")


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


def validate_regime_definition_payload(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """Validate and normalize a regime definition payload."""
    name = str(payload.get("RegimeName", "")).strip()
    if not name:
        return None, "Regime name is required."

    method_type = _parse_int(payload.get("MethodType"))
    if method_type not in {1, 2, 3}:
        return None, "Method type is invalid."

    config_payload = _parse_config(payload.get("ConfigJson"))
    config_payload.update(_parse_config(payload.get("Config")))

    return_basis = str(
        config_payload.get("return_basis")
        or payload.get("ReturnBasis")
        or "total"
    ).strip().lower()
    if return_basis not in {"total", "excess"}:
        return None, "Return basis must be total or excess."

    num_regimes = _parse_int(
        config_payload.get("num_regimes", payload.get("NumRegimes")),
        3,
    )
    if num_regimes is None:
        return None, "Number of regimes is required."
    if method_type == 1 and not (2 <= num_regimes <= 6):
        return None, "HMM on PC1 supports 2 to 6 regimes."
    if method_type in {2, 3} and not (2 <= num_regimes <= 10):
        return None, "Quantile methods support 2 to 10 regimes."

    min_observations = _parse_int(
        config_payload.get("min_observations", payload.get("MinObservations")),
        60,
    )
    if min_observations is None or min_observations < 20:
        return None, "Minimum observations must be at least 20."

    pca_standardize = _parse_bool(
        config_payload.get("pca_standardize", payload.get("PcaStandardize")),
        True,
    )

    universe_series = _parse_series_list(
        config_payload.get("universe_series", payload.get("UniverseSeries"))
    )
    single_series = str(
        config_payload.get("single_series", payload.get("SingleSeries", "")) or ""
    ).strip()

    if method_type in {1, 2} and not universe_series:
        return None, "At least one universe series is required for PC1 methods."
    if method_type == 3 and not single_series:
        return None, "A single series is required for single-series quantiles."

    vol_scaler = _parse_float(
        config_payload.get("vol_scaler", payload.get("VolScaler")),
        0.0,
    )
    if vol_scaler is None or vol_scaler < 0:
        return None, "Vol scaler must be non-negative."

    benchmark_assignments = parse_mapping_payload(
        config_payload.get("benchmark_assignments", payload.get("BenchmarkAssignmentsJson"))
    )
    long_short_assignments = parse_mapping_payload(
        config_payload.get("long_short_assignments", payload.get("LongShortAssignmentsJson"))
    )
    vol_scaling_assignments = parse_mapping_payload(
        config_payload.get("vol_scaling_assignments", payload.get("VolScalingAssignmentsJson"))
    )

    description_raw = payload.get("Description")
    description = str(description_raw).strip() if description_raw is not None else ""
    if not description:
        description = None

    normalized_config: dict[str, Any] = {
        "schema_version": 1,
        "num_regimes": int(num_regimes),
        "return_basis": return_basis,
        "benchmark_assignments": benchmark_assignments,
        "long_short_assignments": long_short_assignments,
        "vol_scaling_assignments": vol_scaling_assignments,
        "vol_scaler": float(vol_scaler),
        "min_observations": int(min_observations),
        "pca_standardize": bool(pca_standardize),
    }
    if method_type in {1, 2}:
        normalized_config["universe_series"] = universe_series
    if method_type == 3:
        normalized_config["single_series"] = single_series
    if method_type in {2, 3}:
        normalized_config["quantile_window"] = "in_sample_full_range"

    normalized = {
        "RegimeName": name,
        "Description": description,
        "MethodType": int(method_type),
        "Config": normalized_config,
        "ConfigJson": canonical_json_dumps(normalized_config),
        "UPDATE_DATE": _iso_or_none(payload.get("UPDATE_DATE")),
        "UPDATE_BY": str(payload.get("UPDATE_BY", "") or "").strip() or None,
    }
    return normalized, None


def regime_tables_available(db_engine: Engine) -> bool:
    return _table_exists(db_engine, "RegimeDefinitions") and _table_exists(db_engine, "RegimeDefinitionsArchive")


def _load_definition_row_by_name(conn, db_engine: Engine, regime_name: str) -> dict[str, Any] | None:
    table_name = _table_name(db_engine, "RegimeDefinitions")
    select_prefix = "SELECT RegimeName, Description, MethodType, ConfigJson, UPDATE_DATE, UPDATE_BY "
    row = conn.execute(
        text(select_prefix + f"FROM {table_name} WHERE RegimeName = :name"),
        {"name": regime_name},
    ).mappings().first()
    if row:
        return dict(row)

    row = conn.execute(
        text(
            select_prefix
            + f"FROM {table_name} "
            "WHERE LOWER(LTRIM(RTRIM(RegimeName))) = LOWER(LTRIM(RTRIM(:name)))"
        ),
        {"name": regime_name},
    ).mappings().first()
    return dict(row) if row else None


def _normalize_db_definition_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized, _error = validate_regime_definition_payload(
        {
            "RegimeName": row.get("RegimeName"),
            "Description": row.get("Description"),
            "MethodType": row.get("MethodType"),
            "ConfigJson": row.get("ConfigJson"),
            "UPDATE_DATE": row.get("UPDATE_DATE"),
            "UPDATE_BY": row.get("UPDATE_BY"),
        }
    )
    if not normalized:
        return {}
    normalized["source"] = "db"
    return normalized


def load_regime_definitions(db_engine: Engine) -> list[dict[str, Any]]:
    if not regime_tables_available(db_engine):
        return []
    table_name = _table_name(db_engine, "RegimeDefinitions")
    q = text(
        f"SELECT RegimeName, Description, MethodType, ConfigJson, UPDATE_DATE, UPDATE_BY "
        f"FROM {table_name} ORDER BY RegimeName"
    )
    with db_engine.connect() as conn:
        rows = conn.execute(q).mappings().all()
    output: list[dict[str, Any]] = []
    for row in rows:
        item = _normalize_db_definition_row(dict(row))
        if item:
            output.append(item)
    return output


def _archive_definition_row(conn, db_engine: Engine, row: dict[str, Any]) -> None:
    archive_table = _table_name(db_engine, "RegimeDefinitionsArchive")
    conn.execute(
        text(
            f"INSERT INTO {archive_table} ("
            "RegimeName, Description, MethodType, ConfigJson, UPDATE_DATE, UPDATE_BY, ARCHIVE_DATE"
            ") VALUES ("
            ":RegimeName, :Description, :MethodType, :ConfigJson, :UPDATE_DATE, :UPDATE_BY, :ARCHIVE_DATE"
            ")"
        ),
        {
            "RegimeName": row.get("RegimeName"),
            "Description": row.get("Description"),
            "MethodType": row.get("MethodType"),
            "ConfigJson": row.get("ConfigJson"),
            "UPDATE_DATE": row.get("UPDATE_DATE"),
            "UPDATE_BY": row.get("UPDATE_BY"),
            "ARCHIVE_DATE": _now_utc(),
        },
    )


def save_regime_definition(
    db_engine: Engine,
    payload: dict[str, Any],
    update_by: str,
    original_name: str | None = None,
    expected_update_date: str | None = None,
) -> tuple[bool, str, dict[str, Any] | None]:
    if not regime_tables_available(db_engine):
        return False, "Regime definition tables are unavailable.", None

    normalized, error = validate_regime_definition_payload(payload)
    if error:
        return False, error, None
    assert normalized is not None

    table_name = _table_name(db_engine, "RegimeDefinitions")
    target_name = normalized["RegimeName"]
    now_val = _now_utc()
    update_by_val = str(update_by or "").strip() or "unknown"
    update_original = str(original_name or "").strip() or None

    with db_engine.begin() as conn:
        if update_original:
            current = _load_definition_row_by_name(conn, db_engine, update_original)
            if current is None:
                return False, "Definition no longer exists. Reload and try again.", None
            if expected_update_date and not _timestamps_equal(current.get("UPDATE_DATE"), expected_update_date):
                return False, "Definition changed in another session. Reload before saving.", None

            if target_name.lower() != str(current.get("RegimeName", "")).lower():
                existing_target = _load_definition_row_by_name(conn, db_engine, target_name)
                if existing_target is not None:
                    return False, f"Regime `{target_name}` already exists.", None

            _archive_definition_row(conn, db_engine, current)
            result = conn.execute(
                text(
                    f"UPDATE {table_name} SET "
                    "RegimeName = :RegimeName, "
                    "Description = :Description, "
                    "MethodType = :MethodType, "
                    "ConfigJson = :ConfigJson, "
                    "UPDATE_DATE = :UPDATE_DATE, "
                    "UPDATE_BY = :UPDATE_BY "
                    "WHERE LOWER(LTRIM(RTRIM(RegimeName))) = LOWER(LTRIM(RTRIM(:OriginalName))) "
                    "AND UPDATE_DATE = :ExpectedDbUpdateDate"
                ),
                {
                    "RegimeName": target_name,
                    "Description": normalized["Description"],
                    "MethodType": normalized["MethodType"],
                    "ConfigJson": normalized["ConfigJson"],
                    "UPDATE_DATE": now_val,
                    "UPDATE_BY": update_by_val,
                    "OriginalName": update_original,
                    "ExpectedDbUpdateDate": current.get("UPDATE_DATE"),
                },
            )
            if _rowcount_is_known_miss(result.rowcount):
                return False, "Definition changed in another session. Reload before saving.", None
            if _rowcount_is_unknown(result.rowcount):
                refreshed = _load_definition_row_by_name(conn, db_engine, target_name)
                if refreshed is None or not _timestamps_equal(refreshed.get("UPDATE_DATE"), now_val):
                    return False, "Definition changed in another session. Reload before saving.", None
        else:
            existing = _load_definition_row_by_name(conn, db_engine, target_name)
            if existing is not None:
                return False, f"Regime `{target_name}` already exists.", None

            conn.execute(
                text(
                    f"INSERT INTO {table_name} ("
                    "RegimeName, Description, MethodType, ConfigJson, UPDATE_DATE, UPDATE_BY"
                    ") VALUES ("
                    ":RegimeName, :Description, :MethodType, :ConfigJson, :UPDATE_DATE, :UPDATE_BY"
                    ")"
                ),
                {
                    "RegimeName": target_name,
                    "Description": normalized["Description"],
                    "MethodType": normalized["MethodType"],
                    "ConfigJson": normalized["ConfigJson"],
                    "UPDATE_DATE": now_val,
                    "UPDATE_BY": update_by_val,
                },
            )

        saved = _load_definition_row_by_name(conn, db_engine, target_name)
        if saved is None:
            return False, "Unable to reload saved definition.", None
        normalized_saved = _normalize_db_definition_row(saved)
        if not normalized_saved:
            return False, "Saved definition is invalid.", None
        return True, f"Saved regime definition `{target_name}`.", normalized_saved


def delete_regime_definition(
    db_engine: Engine,
    regime_name: str,
    expected_update_date: str | None = None,
) -> tuple[bool, str]:
    if not regime_tables_available(db_engine):
        return False, "Regime definition tables are unavailable."

    name = str(regime_name or "").strip()
    if not name:
        return False, "Select a regime definition to delete."

    table_name = _table_name(db_engine, "RegimeDefinitions")
    with db_engine.begin() as conn:
        current = _load_definition_row_by_name(conn, db_engine, name)
        if current is None:
            return False, "Definition no longer exists."
        if expected_update_date and not _timestamps_equal(current.get("UPDATE_DATE"), expected_update_date):
            return False, "Definition changed in another session. Reload before deleting."

        _archive_definition_row(conn, db_engine, current)
        result = conn.execute(
            text(
                f"DELETE FROM {table_name} "
                "WHERE LOWER(LTRIM(RTRIM(RegimeName))) = LOWER(LTRIM(RTRIM(:name))) "
                "AND UPDATE_DATE = :ExpectedDbUpdateDate"
            ),
            {"name": name, "ExpectedDbUpdateDate": current.get("UPDATE_DATE")},
        )
        if _rowcount_is_known_miss(result.rowcount):
            return False, "Definition changed in another session. Reload before deleting."
        if _rowcount_is_unknown(result.rowcount):
            refreshed = _load_definition_row_by_name(conn, db_engine, name)
            if refreshed is not None:
                return False, "Definition changed in another session. Reload before deleting."
        return True, f"Deleted regime definition `{name}`."
