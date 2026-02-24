"""Factor definition storage and compute helpers for AnalyticsTool."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import bindparam, inspect, text
from sqlalchemy.engine import Engine

import cache_config
from utils.serialization import canonical_json_dumps, normalize_date_range_payload


FACTOR_AGG_TYPE_OPTIONS = [
    {"value": "1", "label": "1 - COMPOUND_RETURN"},
    {"value": "2", "label": "2 - LAST_VALUE"},
    {"value": "3", "label": "3 - PERIOD_MEAN"},
    {"value": "4", "label": "4 - ANNUALIZED_VOL"},
    {"value": "5", "label": "5 - ALREADY_PERIODIC"},
    {"value": "6", "label": "6 - QUARTERLY_INTERP"},
    {"value": "7", "label": "7 - RETURN_FROM_LEVELS"},
]

OUTPUT_TRANSFORM_OPTIONS = [
    {"value": "0", "label": "0 - NONE"},
    {"value": "1", "label": "1 - PCT_CHANGE"},
    {"value": "2", "label": "2 - SIMPLE_DIFF"},
]

_PERIODICITY_CODES = {
    "weekly_monday": "W-MON",
    "weekly_tuesday": "W-TUE",
    "weekly_wednesday": "W-WED",
    "weekly_thursday": "W-THU",
    "weekly_friday": "W-FRI",
    "monthly": "ME",
}


def _mrd_account_table(engine: Engine) -> str:
    if engine.dialect.name == "sqlite":
        return "[CORE_DATA.ACCOUNT]"
    return "[CORE_DATA].[ACCOUNT]"


def _mrd_factor_table(engine: Engine) -> str:
    if engine.dialect.name == "sqlite":
        return "[CORE_DATA.ACCOUNT_FACTOR_DATA]"
    return "[CORE_DATA].[ACCOUNT_FACTOR_DATA]"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)


def _iso_or_none(value: Any) -> str | None:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _parse_components(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        parts = [str(v).strip() for v in value]
    else:
        parts = [p.strip() for p in str(value).split(",")]
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


def _parse_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        return default
    return int(parsed)


def validate_factor_definition_payload(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """Validate and normalize a factor definition payload."""
    name = str(payload.get("FactorName", "")).strip()
    if not name:
        return None, "Factor name is required."

    long_components = _parse_components(payload.get("LongComponent"))
    short_components = _parse_components(payload.get("ShortComponent"))
    if not long_components:
        return None, "At least one Long component is required."

    long_agg = _parse_int(payload.get("LongAggType"))
    if long_agg not in {1, 2, 3, 4, 5, 6, 7}:
        return None, "Long aggregation type is invalid."

    short_agg = _parse_int(payload.get("ShortAggType"))
    if short_components:
        if short_agg not in {1, 2, 3, 4, 5, 6, 7}:
            return None, "Short aggregation type is invalid when Short components are provided."
    else:
        short_agg = None

    long_lag = _parse_int(payload.get("LongLag"), 0)
    if long_lag is None or long_lag < 0:
        return None, "Long lag must be a non-negative integer."

    output_transform = _parse_int(payload.get("OutputTransform"), 0)
    if output_transform not in {0, 1, 2}:
        return None, "Output transform is invalid."

    description_raw = payload.get("Description")
    description = str(description_raw).strip() if description_raw is not None else ""
    if not description:
        description = None

    return (
        {
            "FactorName": name,
            "LongComponentList": long_components,
            "ShortComponentList": short_components,
            "LongComponent": ", ".join(long_components),
            "ShortComponent": ", ".join(short_components) if short_components else None,
            "Description": description,
            "LongAggType": long_agg,
            "ShortAggType": short_agg,
            "LongLag": long_lag,
            "OutputTransform": output_transform,
            "UPDATE_DATE": _iso_or_none(payload.get("UPDATE_DATE")),
            "UPDATE_BY": str(payload.get("UPDATE_BY", "") or "").strip() or None,
        },
        None,
    )


def factor_tables_available(db_engine: Engine) -> bool:
    insp = inspect(db_engine)
    return insp.has_table("FactorDefinitions") and insp.has_table("FactorDefinitionsArchive")


def _normalize_db_definition_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized, _ = validate_factor_definition_payload(
        {
            "FactorName": row.get("FactorName"),
            "LongComponent": row.get("LongComponent"),
            "ShortComponent": row.get("ShortComponent"),
            "Description": row.get("Description"),
            "LongAggType": row.get("LongAggType"),
            "ShortAggType": row.get("ShortAggType"),
            "LongLag": row.get("LongLag"),
            "OutputTransform": row.get("OutputTransform"),
            "UPDATE_DATE": row.get("UPDATE_DATE"),
            "UPDATE_BY": row.get("UPDATE_BY"),
        }
    )
    if not normalized:
        return {}
    normalized["source"] = "db"
    return normalized


def load_factor_definitions(db_engine: Engine) -> list[dict[str, Any]]:
    if not factor_tables_available(db_engine):
        return []
    q = text(
        "SELECT FactorName, LongComponent, ShortComponent, Description, "
        "LongAggType, ShortAggType, LongLag, OutputTransform, UPDATE_DATE, UPDATE_BY "
        "FROM FactorDefinitions ORDER BY FactorName"
    )
    with db_engine.connect() as conn:
        rows = conn.execute(q).mappings().all()
    output: list[dict[str, Any]] = []
    for row in rows:
        item = _normalize_db_definition_row(dict(row))
        if item:
            output.append(item)
    return output


def _archive_factor_definition_row(conn, row: dict[str, Any]) -> None:
    archive_q = text(
        "INSERT INTO FactorDefinitionsArchive ("
        "FactorName, LongComponent, ShortComponent, Description, "
        "LongAggType, ShortAggType, LongLag, OutputTransform, UPDATE_DATE, UPDATE_BY, ARCHIVE_DATE"
        ") VALUES ("
        ":FactorName, :LongComponent, :ShortComponent, :Description, "
        ":LongAggType, :ShortAggType, :LongLag, :OutputTransform, :UPDATE_DATE, :UPDATE_BY, :ARCHIVE_DATE"
        ")"
    )
    conn.execute(
        archive_q,
        {
            "FactorName": row.get("FactorName"),
            "LongComponent": row.get("LongComponent"),
            "ShortComponent": row.get("ShortComponent"),
            "Description": row.get("Description"),
            "LongAggType": row.get("LongAggType"),
            "ShortAggType": row.get("ShortAggType"),
            "LongLag": row.get("LongLag"),
            "OutputTransform": row.get("OutputTransform"),
            "UPDATE_DATE": row.get("UPDATE_DATE"),
            "UPDATE_BY": row.get("UPDATE_BY"),
            "ARCHIVE_DATE": _now_utc(),
        },
    )


def _load_definition_row_by_name(conn, factor_name: str) -> dict[str, Any] | None:
    q = text(
        "SELECT FactorName, LongComponent, ShortComponent, Description, "
        "LongAggType, ShortAggType, LongLag, OutputTransform, UPDATE_DATE, UPDATE_BY "
        "FROM FactorDefinitions WHERE LOWER(FactorName) = LOWER(:name)"
    )
    row = conn.execute(q, {"name": factor_name}).mappings().first()
    return dict(row) if row else None


def _timestamps_equal(left: Any, right: Any) -> bool:
    left_ts = pd.to_datetime(left, errors="coerce")
    right_ts = pd.to_datetime(right, errors="coerce")
    if pd.isna(left_ts) or pd.isna(right_ts):
        return False
    return pd.Timestamp(left_ts) == pd.Timestamp(right_ts)


def save_factor_definition(
    db_engine: Engine,
    payload: dict[str, Any],
    update_by: str,
    original_name: str | None = None,
    expected_update_date: str | None = None,
) -> tuple[bool, str, dict[str, Any] | None]:
    if not factor_tables_available(db_engine):
        return False, "Factor definition tables are unavailable.", None

    normalized, error = validate_factor_definition_payload(payload)
    if error:
        return False, error, None
    assert normalized is not None

    target_name = normalized["FactorName"]
    now_val = _now_utc()
    update_by_val = str(update_by or "").strip() or "unknown"
    update_original = str(original_name or "").strip() or None

    with db_engine.begin() as conn:
        if update_original:
            current = _load_definition_row_by_name(conn, update_original)
            if current is None:
                return False, "Definition no longer exists. Reload and try again.", None

            if expected_update_date and not _timestamps_equal(current.get("UPDATE_DATE"), expected_update_date):
                return False, "Definition changed in another session. Reload before saving.", None

            if target_name.lower() != str(current.get("FactorName", "")).lower():
                existing_target = _load_definition_row_by_name(conn, target_name)
                if existing_target is not None:
                    return False, f"Factor `{target_name}` already exists.", None

            _archive_factor_definition_row(conn, current)

            update_q = text(
                "UPDATE FactorDefinitions SET "
                "FactorName = :FactorName, "
                "LongComponent = :LongComponent, "
                "ShortComponent = :ShortComponent, "
                "Description = :Description, "
                "LongAggType = :LongAggType, "
                "ShortAggType = :ShortAggType, "
                "LongLag = :LongLag, "
                "OutputTransform = :OutputTransform, "
                "UPDATE_DATE = :UPDATE_DATE, "
                "UPDATE_BY = :UPDATE_BY "
                "WHERE LOWER(FactorName) = LOWER(:OriginalName) "
                "AND UPDATE_DATE = :ExpectedDbUpdateDate"
            )
            result = conn.execute(
                update_q,
                {
                    "FactorName": target_name,
                    "LongComponent": normalized["LongComponent"],
                    "ShortComponent": normalized["ShortComponent"],
                    "Description": normalized["Description"],
                    "LongAggType": normalized["LongAggType"],
                    "ShortAggType": normalized["ShortAggType"],
                    "LongLag": normalized["LongLag"],
                    "OutputTransform": normalized["OutputTransform"],
                    "UPDATE_DATE": now_val,
                    "UPDATE_BY": update_by_val,
                    "OriginalName": update_original,
                    "ExpectedDbUpdateDate": current.get("UPDATE_DATE"),
                },
            )
            if int(result.rowcount or 0) != 1:
                return False, "Definition changed in another session. Reload before saving.", None
        else:
            existing = _load_definition_row_by_name(conn, target_name)
            if existing is not None:
                return False, f"Factor `{target_name}` already exists.", None

            insert_q = text(
                "INSERT INTO FactorDefinitions ("
                "FactorName, LongComponent, ShortComponent, Description, "
                "LongAggType, ShortAggType, LongLag, OutputTransform, UPDATE_DATE, UPDATE_BY"
                ") VALUES ("
                ":FactorName, :LongComponent, :ShortComponent, :Description, "
                ":LongAggType, :ShortAggType, :LongLag, :OutputTransform, :UPDATE_DATE, :UPDATE_BY"
                ")"
            )
            conn.execute(
                insert_q,
                {
                    "FactorName": target_name,
                    "LongComponent": normalized["LongComponent"],
                    "ShortComponent": normalized["ShortComponent"],
                    "Description": normalized["Description"],
                    "LongAggType": normalized["LongAggType"],
                    "ShortAggType": normalized["ShortAggType"],
                    "LongLag": normalized["LongLag"],
                    "OutputTransform": normalized["OutputTransform"],
                    "UPDATE_DATE": now_val,
                    "UPDATE_BY": update_by_val,
                },
            )

        saved = _load_definition_row_by_name(conn, target_name)
        if saved is None:
            return False, "Unable to reload saved definition.", None
        normalized_saved = _normalize_db_definition_row(saved)
        if not normalized_saved:
            return False, "Saved definition is invalid.", None
        return True, f"Saved factor definition `{target_name}`.", normalized_saved


def delete_factor_definition(
    db_engine: Engine,
    factor_name: str,
    expected_update_date: str | None = None,
) -> tuple[bool, str]:
    if not factor_tables_available(db_engine):
        return False, "Factor definition tables are unavailable."

    name = str(factor_name or "").strip()
    if not name:
        return False, "Select a factor definition to delete."

    with db_engine.begin() as conn:
        current = _load_definition_row_by_name(conn, name)
        if current is None:
            return False, "Definition no longer exists."
        if expected_update_date and not _timestamps_equal(current.get("UPDATE_DATE"), expected_update_date):
            return False, "Definition changed in another session. Reload before deleting."

        _archive_factor_definition_row(conn, current)

        delete_q = text(
            "DELETE FROM FactorDefinitions "
            "WHERE LOWER(FactorName) = LOWER(:name) AND UPDATE_DATE = :ExpectedDbUpdateDate"
        )
        result = conn.execute(
            delete_q,
            {"name": name, "ExpectedDbUpdateDate": current.get("UPDATE_DATE")},
        )
        if int(result.rowcount or 0) != 1:
            return False, "Definition changed in another session. Reload before deleting."
        return True, f"Deleted factor definition `{name}`."


def _load_sec_factor_account_rows(mrd_engine: Engine) -> pd.DataFrame:
    account_table = _mrd_account_table(mrd_engine)
    try:
        q = text(
            f"SELECT ACCT_ID, ACCT_NAME, FACTOR_NAME, ACCT_CD, SOURCE_SYSTEM "
            f"FROM {account_table} "
            "WHERE ACCT_TYPE_CD = 'SEC_FACTOR' "
            "AND COALESCE(SOURCE_SYSTEM, '') <> 'PERF' "
            "ORDER BY ACCT_NAME, FACTOR_NAME, ACCT_ID"
        )
        with mrd_engine.connect() as conn:
            rows = conn.execute(q).fetchall()
    except Exception:
        q = text(
            f"SELECT ACCT_ID, ACCT_NAME, FACTOR_NAME, ACCT_CD, '' AS SOURCE_SYSTEM "
            f"FROM {account_table} "
            "WHERE ACCT_TYPE_CD = 'SEC_FACTOR' "
            "ORDER BY ACCT_NAME, FACTOR_NAME, ACCT_ID"
        )
        with mrd_engine.connect() as conn:
            rows = conn.execute(q).fetchall()
    if not rows:
        return pd.DataFrame(columns=["ACCT_ID", "TOKEN", "ACCT_CD", "SOURCE_SYSTEM"])
    out = pd.DataFrame(rows, columns=["ACCT_ID", "ACCT_NAME", "FACTOR_NAME", "ACCT_CD", "SOURCE_SYSTEM"])
    out["ACCT_ID"] = pd.to_numeric(out["ACCT_ID"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["ACCT_ID"])
    out["TOKEN"] = out["ACCT_NAME"].astype(str).str.strip() + " " + out["FACTOR_NAME"].astype(str).str.strip()
    out["TOKEN"] = out["TOKEN"].str.replace(r"\s+", " ", regex=True).str.strip()
    out = out.sort_values(["TOKEN", "ACCT_ID"])
    return out[["ACCT_ID", "TOKEN", "ACCT_CD", "SOURCE_SYSTEM"]]


@cache_config.cache.memoize(timeout=0)
def get_sec_factor_component_meta_cached(mrd_engine: Engine) -> dict[str, dict[str, Any]]:
    rows = _load_sec_factor_account_rows(mrd_engine)
    if rows.empty:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for token, frame in rows.groupby("TOKEN", sort=True):
        acct_ids = [int(v) for v in frame["ACCT_ID"].tolist()]
        if not acct_ids:
            continue
        out[str(token)] = {
            "token": str(token),
            "acct_ids": acct_ids,
            "first_acct_id": int(min(acct_ids)),
            "count": int(len(acct_ids)),
        }
    return out


@cache_config.cache.memoize(timeout=0)
def get_sec_factor_component_options_cached(mrd_engine: Engine) -> list[dict[str, str]]:
    meta = get_sec_factor_component_meta_cached(mrd_engine)
    options: list[dict[str, str]] = []
    for token in sorted(meta.keys()):
        count = int(meta[token].get("count", 1))
        label = token if count <= 1 else f"{token} ({count} matches)"
        options.append({"value": token, "label": label})
    return options


def resolve_component_tokens_to_acct_ids(mrd_engine: Engine, tokens: list[str]) -> list[int]:
    meta = get_sec_factor_component_meta_cached(mrd_engine)
    ids: list[int] = []
    for token in _parse_components(tokens):
        item = meta.get(token)
        if not item:
            continue
        first_id = item.get("first_acct_id")
        if first_id is None:
            continue
        ids.append(int(first_id))
    return ids


@cache_config.cache.memoize(timeout=0)
def _load_sec_factor_values_cached(
    mrd_engine: Engine,
    acct_ids: tuple[int, ...],
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    ids = [int(v) for v in acct_ids if v is not None]
    if not ids:
        return pd.DataFrame(columns=["ACCT_ID", "REFERENCE_DATE", "FACTOR_VALUE"])

    factor_table = _mrd_factor_table(mrd_engine)
    clauses = ["ACCT_ID IN :acct_ids"]
    params: dict[str, Any] = {"acct_ids": ids}

    if start_date:
        clauses.append("REFERENCE_DATE >= :start_date")
        params["start_date"] = pd.Timestamp(start_date).strftime("%Y-%m-%d")
    if end_date:
        clauses.append("REFERENCE_DATE <= :end_date")
        params["end_date"] = pd.Timestamp(end_date).strftime("%Y-%m-%d")

    q = text(
        f"SELECT ACCT_ID, REFERENCE_DATE, FACTOR_VALUE "
        f"FROM {factor_table} WHERE {' AND '.join(clauses)} "
        "ORDER BY REFERENCE_DATE, ACCT_ID"
    ).bindparams(bindparam("acct_ids", expanding=True))

    with mrd_engine.connect() as conn:
        rows = conn.execute(q, params).fetchall()
    if not rows:
        return pd.DataFrame(columns=["ACCT_ID", "REFERENCE_DATE", "FACTOR_VALUE"])
    out = pd.DataFrame(rows, columns=["ACCT_ID", "REFERENCE_DATE", "FACTOR_VALUE"])
    out["ACCT_ID"] = pd.to_numeric(out["ACCT_ID"], errors="coerce").astype("Int64")
    out["REFERENCE_DATE"] = pd.to_datetime(out["REFERENCE_DATE"], errors="coerce")
    out["FACTOR_VALUE"] = pd.to_numeric(out["FACTOR_VALUE"], errors="coerce")
    out = out.dropna(subset=["ACCT_ID", "REFERENCE_DATE"])
    return out


def _resolve_window(
    date_range_payload: dict[str, Any] | str | None,
    periodicity: str,
    long_lag: int,
    output_transform: int,
    long_agg_type: int,
    short_agg_type: int | None,
) -> tuple[str | None, str | None, str | None, str | None]:
    normalized = normalize_date_range_payload(date_range_payload)
    if not normalized:
        return None, None, None, None

    start = pd.Timestamp(normalized["start"])
    end = pd.Timestamp(normalized["end"])
    if pd.isna(start) or pd.isna(end):
        return None, None, None, None

    buffer_periods = max(0, int(long_lag or 0)) + (1 if output_transform in {1, 2} else 0) + 2
    if periodicity == "monthly":
        buffer_days = 35 * buffer_periods
    elif str(periodicity or "").startswith("weekly_"):
        buffer_days = 10 * buffer_periods
    else:
        buffer_days = max(8, 2 * buffer_periods)

    if long_agg_type in {5, 6, 7} or short_agg_type in {5, 6, 7}:
        buffer_days += 370
    tail_days = 0
    if long_agg_type == 5 or short_agg_type == 5:
        tail_days = max(tail_days, 40)
    if long_agg_type == 6 or short_agg_type == 6:
        tail_days = max(tail_days, 100)

    fetch_start = (start - timedelta(days=buffer_days)).strftime("%Y-%m-%d")
    fetch_end = (end + timedelta(days=tail_days)).strftime("%Y-%m-%d")
    return fetch_start, fetch_end, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _resample_code(periodicity: str) -> str | None:
    if periodicity in {"daily", "daily_trading"}:
        return None
    return _PERIODICITY_CODES.get(periodicity)


def _target_index_for_mapping(series: pd.Series, periodicity: str) -> pd.DatetimeIndex:
    base = series.dropna().sort_index()
    if base.empty:
        return pd.DatetimeIndex([])
    code = _resample_code(periodicity)
    if code is None:
        return pd.DatetimeIndex(base.index)
    sampled = base.resample(code).last().dropna()
    return pd.DatetimeIndex(sampled.index)


def _aggregate_component_series(series: pd.Series, agg_type: int, periodicity: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    if values.empty:
        return pd.Series(dtype=float)

    code = _resample_code(periodicity)

    def _compound(window: pd.Series) -> float:
        clean = pd.to_numeric(window, errors="coerce").dropna()
        if clean.empty:
            return np.nan
        return float(np.prod(1.0 + clean.values) - 1.0)

    def _annualized_vol(window: pd.Series) -> float:
        clean = pd.to_numeric(window, errors="coerce").dropna()
        if len(clean) < 2:
            return np.nan
        if (clean <= -1.0).any():
            return np.nan
        log_ret = np.log1p(clean.values)
        if len(log_ret) < 2:
            return np.nan
        return float(np.std(log_ret, ddof=1) * np.sqrt(252.0))

    if agg_type == 1:
        out = values if code is None else values.resample(code).apply(_compound)
    elif agg_type == 2:
        out = values if code is None else values.resample(code).last()
    elif agg_type == 3:
        out = values if code is None else values.resample(code).mean()
    elif agg_type == 4:
        if code is None:
            out = pd.Series(np.nan, index=values.index, dtype=float)
        else:
            out = values.resample(code).apply(_annualized_vol)
    elif agg_type == 5:
        monthly_vals = values.resample("ME").last().dropna()
        target_idx = _target_index_for_mapping(values, periodicity)
        mapped_idx = target_idx + pd.offsets.MonthEnd(0)
        mapped = monthly_vals.reindex(mapped_idx)
        mapped.index = target_idx
        out = mapped
    elif agg_type == 6:
        quarter_vals = values.resample("QE").last().dropna()
        target_idx = _target_index_for_mapping(values, periodicity)
        q_idx = pd.DatetimeIndex(
            [pd.Timestamp(dt).to_period("Q").to_timestamp(how="end").normalize() for dt in target_idx]
        )
        mapped = quarter_vals.reindex(q_idx)
        mapped.index = target_idx
        out = mapped
    elif agg_type == 7:
        levels = values if code is None else values.resample(code).last()
        out = levels.pct_change(fill_method=None)
    else:
        return pd.Series(dtype=float)

    out = out.dropna()
    out.index = pd.to_datetime(out.index)
    out.index.name = "Date"
    return out.astype(float)


def _build_component_series(values_df: pd.DataFrame, acct_ids: list[int]) -> pd.Series:
    if values_df.empty or not acct_ids:
        return pd.Series(dtype=float)
    subset = values_df.loc[values_df["ACCT_ID"].isin(acct_ids), ["REFERENCE_DATE", "ACCT_ID", "FACTOR_VALUE"]]
    if subset.empty:
        return pd.Series(dtype=float)
    pivot = subset.pivot_table(
        index="REFERENCE_DATE",
        columns="ACCT_ID",
        values="FACTOR_VALUE",
        aggfunc="last",
    ).sort_index()
    if pivot.empty:
        return pd.Series(dtype=float)
    if pivot.shape[1] == 1:
        out = pivot.iloc[:, 0]
    else:
        out = pivot.mean(axis=1, skipna=True)
    out = pd.to_numeric(out, errors="coerce").dropna()
    out.index = pd.to_datetime(out.index)
    out.index.name = "Date"
    return out.astype(float)


def _clip_series_to_window(series: pd.Series, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.Series:
    if not isinstance(series, pd.Series) or series.empty:
        return series

    idx = series.index
    if isinstance(idx, pd.DatetimeIndex):
        return series[(idx >= start_ts) & (idx <= end_ts)]

    coerced_idx = pd.to_datetime(idx, errors="coerce")
    valid_mask = ~pd.isna(coerced_idx)
    if not bool(np.any(valid_mask)):
        return series

    filtered = series.iloc[np.flatnonzero(valid_mask)].copy()
    filtered.index = pd.DatetimeIndex(coerced_idx[valid_mask])
    return filtered[(filtered.index >= start_ts) & (filtered.index <= end_ts)]


def _compute_factor_series_core(
    mrd_engine: Engine,
    definition: dict[str, Any],
    periodicity: str,
    date_range_payload: dict[str, Any] | str | None,
) -> dict[str, pd.Series]:
    normalized, error = validate_factor_definition_payload(definition)
    if error or not normalized:
        return {
            "long": pd.Series(dtype=float),
            "short": pd.Series(dtype=float),
            "long_lagged": pd.Series(dtype=float),
            "combined": pd.Series(dtype=float),
            "final": pd.Series(dtype=float),
        }

    fetch_start, fetch_end, clip_start, clip_end = _resolve_window(
        date_range_payload,
        periodicity or "daily",
        int(normalized["LongLag"]),
        int(normalized["OutputTransform"]),
        int(normalized["LongAggType"]),
        normalized["ShortAggType"],
    )

    long_ids = resolve_component_tokens_to_acct_ids(mrd_engine, normalized["LongComponentList"])
    short_ids = resolve_component_tokens_to_acct_ids(mrd_engine, normalized["ShortComponentList"])
    all_ids = sorted(set(long_ids + short_ids))
    if not all_ids:
        return {
            "long": pd.Series(dtype=float),
            "short": pd.Series(dtype=float),
            "long_lagged": pd.Series(dtype=float),
            "combined": pd.Series(dtype=float),
            "final": pd.Series(dtype=float),
        }

    values = _load_sec_factor_values_cached(
        mrd_engine,
        tuple(all_ids),
        fetch_start,
        fetch_end,
    )
    if values.empty:
        return {
            "long": pd.Series(dtype=float),
            "short": pd.Series(dtype=float),
            "long_lagged": pd.Series(dtype=float),
            "combined": pd.Series(dtype=float),
            "final": pd.Series(dtype=float),
        }

    long_daily = _build_component_series(values, long_ids)
    short_daily = _build_component_series(values, short_ids)

    long_agg = _aggregate_component_series(long_daily, int(normalized["LongAggType"]), periodicity or "daily")
    short_agg = pd.Series(dtype=float)
    if short_ids and normalized["ShortAggType"] is not None:
        short_agg = _aggregate_component_series(short_daily, int(normalized["ShortAggType"]), periodicity or "daily")

    long_lag = int(normalized["LongLag"] or 0)
    long_lagged = long_agg.shift(long_lag) if long_lag > 0 else long_agg

    combined = long_lagged.copy()
    if not short_agg.empty:
        combined = long_lagged - short_agg

    output_transform = int(normalized["OutputTransform"] or 0)
    final = combined.copy()
    if output_transform == 1:
        final = final.pct_change(fill_method=None)
    elif output_transform == 2:
        final = final.diff()

    if clip_start and clip_end:
        start_ts = pd.Timestamp(clip_start)
        end_ts = pd.Timestamp(clip_end)
        long_agg = _clip_series_to_window(long_agg, start_ts, end_ts)
        short_agg = _clip_series_to_window(short_agg, start_ts, end_ts)
        long_lagged = _clip_series_to_window(long_lagged, start_ts, end_ts)
        combined = _clip_series_to_window(combined, start_ts, end_ts)
        final = _clip_series_to_window(final, start_ts, end_ts)

    return {
        "long": long_agg.replace([np.inf, -np.inf], np.nan).dropna(),
        "short": short_agg.replace([np.inf, -np.inf], np.nan).dropna(),
        "long_lagged": long_lagged.replace([np.inf, -np.inf], np.nan).dropna(),
        "combined": combined.replace([np.inf, -np.inf], np.nan).dropna(),
        "final": final.replace([np.inf, -np.inf], np.nan).dropna(),
    }


@cache_config.cache.memoize(timeout=0)
def compute_factor_series_cached(
    mrd_engine: Engine,
    definition_payload_json: str,
    periodicity: str,
    date_range_payload_json: str,
) -> pd.Series:
    try:
        definition = json.loads(str(definition_payload_json or "{}"))
        if not isinstance(definition, dict):
            definition = {}
    except Exception:
        return pd.Series(dtype=float)
    try:
        date_range_payload = json.loads(str(date_range_payload_json or "null"))
    except Exception:
        date_range_payload = date_range_payload_json
    parts = _compute_factor_series_core(
        mrd_engine,
        definition=definition,
        periodicity=periodicity,
        date_range_payload=date_range_payload,
    )
    return parts.get("final", pd.Series(dtype=float))


def compute_factor_series(
    mrd_engine: Engine,
    definition: dict[str, Any],
    periodicity: str,
    date_range_payload: dict[str, Any] | str | None,
) -> pd.Series:
    payload = canonical_json_dumps(
        {
            "FactorName": definition.get("FactorName"),
            "LongComponent": definition.get("LongComponent"),
            "ShortComponent": definition.get("ShortComponent"),
            "LongAggType": definition.get("LongAggType"),
            "ShortAggType": definition.get("ShortAggType"),
            "LongLag": definition.get("LongLag"),
            "OutputTransform": definition.get("OutputTransform"),
        }
    )
    date_payload = canonical_json_dumps(normalize_date_range_payload(date_range_payload))
    return compute_factor_series_cached(mrd_engine, payload, periodicity, date_payload)


def compute_factor_preview_lines(
    mrd_engine: Engine,
    definition: dict[str, Any],
    periodicity: str,
    date_range_payload: dict[str, Any] | str | None,
    max_rows: int = 6,
) -> list[str]:
    parts = _compute_factor_series_core(
        mrd_engine,
        definition=definition,
        periodicity=periodicity,
        date_range_payload=date_range_payload,
    )
    final = parts.get("final", pd.Series(dtype=float))
    if final.empty:
        return []

    frame = pd.DataFrame(index=final.index.copy())
    frame["Final"] = final
    long_lagged = parts.get("long_lagged", pd.Series(dtype=float))
    short_series = parts.get("short", pd.Series(dtype=float))
    combined = parts.get("combined", pd.Series(dtype=float))
    if not long_lagged.empty:
        frame["Long"] = long_lagged.reindex(frame.index)
    if not short_series.empty:
        frame["Short"] = short_series.reindex(frame.index)
    if not combined.empty:
        frame["Combined"] = combined.reindex(frame.index)
    frame = frame.dropna(how="all")
    if frame.empty:
        return []

    frame = frame.sort_index().iloc[:max_rows]
    lines: list[str] = []
    for dt, row in frame.iterrows():
        dt_str = pd.Timestamp(dt).strftime("%Y-%m-%d")
        parts_line = [dt_str]
        for col in frame.columns:
            value = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
            if pd.isna(value):
                parts_line.append("")
            else:
                parts_line.append(f"{float(value):.10g}")
        lines.append(":".join([parts_line[0], "|".join(parts_line[1:])]))
    header = "Date:Final"
    if "Long" in frame.columns:
        header += "|Long"
    if "Short" in frame.columns:
        header += "|Short"
    if "Combined" in frame.columns:
        header += "|Combined"
    return [header] + lines
