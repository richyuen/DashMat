"""Shared SEC_FACTOR loaders with AA-compatible return shaping."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pandas_market_calendars as mcal
from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine

import cache_config


def _mrd_account_table(engine: Engine) -> str:
    if engine.dialect.name == "sqlite":
        return "[CORE_DATA.ACCOUNT]"
    return "[CORE_DATA].[ACCOUNT]"


def _mrd_factor_table(engine: Engine) -> str:
    if engine.dialect.name == "sqlite":
        return "[CORE_DATA.ACCOUNT_FACTOR_DATA]"
    return "[CORE_DATA].[ACCOUNT_FACTOR_DATA]"


def _normalize_spaces(value: str) -> str:
    return " ".join(str(value or "").strip().split())


def normalize_sec_factor_name(name: str | None) -> str | None:
    """Normalize SEC_FACTOR name aliases to canonical `ACCT_NAME_FACTOR_NAME`."""
    text_value = _normalize_spaces(str(name or ""))
    if not text_value:
        return None

    underscore_idx = text_value.rfind("_")
    space_idx = text_value.rfind(" ")
    split_idx = max(underscore_idx, space_idx)

    if split_idx < 0:
        acct_name = text_value
        factor_name = "TRIndex"
    else:
        acct_name = _normalize_spaces(text_value[:split_idx])
        factor_name = _normalize_spaces(text_value[split_idx + 1 :])
        if not factor_name:
            factor_name = "TRIndex"

    if not acct_name:
        return None
    return f"{acct_name}_{factor_name}"


def _split_canonical_name(canonical_name: str) -> tuple[str, str] | None:
    text_value = str(canonical_name or "").strip()
    if "_" not in text_value:
        return None
    acct_name, factor_name = text_value.rsplit("_", 1)
    acct_name = _normalize_spaces(acct_name)
    factor_name = _normalize_spaces(factor_name)
    if not acct_name or not factor_name:
        return None
    return acct_name, factor_name


@cache_config.cache.memoize(timeout=0)
def _load_sec_factor_account_rows_cached(mrd_engine: Engine) -> pd.DataFrame:
    account_table = _mrd_account_table(mrd_engine)
    try:
        query = text(
            f"SELECT ACCT_ID, ACCT_NAME, FACTOR_NAME, ACCT_CD, SOURCE_SYSTEM "
            f"FROM {account_table} "
            "WHERE ACCT_TYPE_CD = 'SEC_FACTOR' "
            "ORDER BY ACCT_NAME, FACTOR_NAME, ACCT_ID"
        )
        with mrd_engine.connect() as conn:
            rows = conn.execute(query).fetchall()
    except Exception:
        query = text(
            f"SELECT ACCT_ID, ACCT_NAME, FACTOR_NAME, ACCT_CD, '' AS SOURCE_SYSTEM "
            f"FROM {account_table} "
            "WHERE ACCT_TYPE_CD = 'SEC_FACTOR' "
            "ORDER BY ACCT_NAME, FACTOR_NAME, ACCT_ID"
        )
        with mrd_engine.connect() as conn:
            rows = conn.execute(query).fetchall()

    if not rows:
        return pd.DataFrame(
            columns=[
                "ACCT_ID",
                "ACCT_NAME",
                "FACTOR_NAME",
                "ACCT_CD",
                "SOURCE_SYSTEM",
                "CANONICAL_NAME",
                "CANONICAL_KEY",
                "TOKEN",
                "SOURCE_RANK",
            ]
        )

    out = pd.DataFrame(rows, columns=["ACCT_ID", "ACCT_NAME", "FACTOR_NAME", "ACCT_CD", "SOURCE_SYSTEM"])
    out["ACCT_ID"] = pd.to_numeric(out["ACCT_ID"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["ACCT_ID"])
    if out.empty:
        return pd.DataFrame(
            columns=[
                "ACCT_ID",
                "ACCT_NAME",
                "FACTOR_NAME",
                "ACCT_CD",
                "SOURCE_SYSTEM",
                "CANONICAL_NAME",
                "CANONICAL_KEY",
                "TOKEN",
                "SOURCE_RANK",
            ]
        )

    out["ACCT_NAME"] = out["ACCT_NAME"].astype(str).map(_normalize_spaces)
    out["FACTOR_NAME"] = out["FACTOR_NAME"].astype(str).map(_normalize_spaces)
    out["ACCT_CD"] = out["ACCT_CD"].astype(str).map(_normalize_spaces)
    out["SOURCE_SYSTEM"] = out["SOURCE_SYSTEM"].fillna("").astype(str).map(_normalize_spaces)
    out["CANONICAL_NAME"] = (
        out["ACCT_NAME"].astype(str) + "_" + out["FACTOR_NAME"].astype(str)
    ).map(normalize_sec_factor_name)
    out = out.dropna(subset=["CANONICAL_NAME"])
    out["CANONICAL_KEY"] = out["CANONICAL_NAME"].astype(str).str.lower()
    out["TOKEN"] = (out["ACCT_NAME"].astype(str) + " " + out["FACTOR_NAME"].astype(str)).map(_normalize_spaces)
    out["SOURCE_RANK"] = out["SOURCE_SYSTEM"].astype(str).str.upper().map(lambda v: 0 if v == "BB" else 1)
    out = out.sort_values(["CANONICAL_KEY", "SOURCE_RANK", "ACCT_ID"]).reset_index(drop=True)
    return out


def get_sec_factor_account_rows(mrd_engine: Engine, exclude_perf: bool = False) -> pd.DataFrame:
    rows = _load_sec_factor_account_rows_cached(mrd_engine)
    if rows.empty:
        return rows.copy()
    if not exclude_perf:
        return rows.copy()
    filtered = rows[rows["SOURCE_SYSTEM"].astype(str).str.upper() != "PERF"].copy()
    return filtered.reset_index(drop=True)


def _pick_account_row_for_canonical(
    rows: pd.DataFrame,
    canonical_name: str,
    collision_policy: str,
) -> pd.Series | None:
    canonical_key = str(canonical_name or "").strip().lower()
    if not canonical_key:
        return None
    matched = rows[rows["CANONICAL_KEY"] == canonical_key]
    if matched.empty:
        return None

    policy = str(collision_policy or "bb_then_lowest").strip().lower()
    if policy == "lowest":
        chosen = matched.sort_values(["ACCT_ID"]).iloc[0]
        return chosen
    if policy == "error" and len(matched) > 1:
        return None

    # Default: BB preferred, then lowest ACCT_ID.
    chosen = matched.sort_values(["SOURCE_RANK", "ACCT_ID"]).iloc[0]
    return chosen


def resolve_sec_factor_accounts_by_names(
    mrd_engine: Engine,
    names: list[str] | tuple[str, ...],
    collision_policy: str = "bb_then_lowest",
    exclude_perf: bool = False,
) -> dict[str, int]:
    requested_canonical: list[str] = []
    seen: set[str] = set()
    for raw_name in names or []:
        canonical = normalize_sec_factor_name(str(raw_name or ""))
        if not canonical:
            continue
        key = canonical.lower()
        if key in seen:
            continue
        seen.add(key)
        requested_canonical.append(canonical)

    if not requested_canonical:
        return {}

    rows = get_sec_factor_account_rows(mrd_engine, exclude_perf=exclude_perf)
    if rows.empty:
        return {}

    resolved: dict[str, int] = {}
    for canonical in requested_canonical:
        selected_row = _pick_account_row_for_canonical(rows, canonical, collision_policy)
        if selected_row is None:
            continue
        resolved[canonical] = int(selected_row["ACCT_ID"])
    return resolved


@cache_config.cache.memoize(timeout=0)
def _load_sec_factor_points_cached(
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
        params["start_date"] = str(pd.Timestamp(start_date).strftime("%Y-%m-%d"))
    if end_date:
        clauses.append("REFERENCE_DATE <= :end_date")
        params["end_date"] = str(pd.Timestamp(end_date).strftime("%Y-%m-%d"))

    query = text(
        f"SELECT ACCT_ID, REFERENCE_DATE, FACTOR_VALUE "
        f"FROM {factor_table} "
        f"WHERE {' AND '.join(clauses)} "
        "ORDER BY REFERENCE_DATE, ACCT_ID"
    ).bindparams(bindparam("acct_ids", expanding=True))

    with mrd_engine.connect() as conn:
        rows = conn.execute(query, params).fetchall()

    if not rows:
        return pd.DataFrame(columns=["ACCT_ID", "REFERENCE_DATE", "FACTOR_VALUE"])

    out = pd.DataFrame(rows, columns=["ACCT_ID", "REFERENCE_DATE", "FACTOR_VALUE"])
    out["ACCT_ID"] = pd.to_numeric(out["ACCT_ID"], errors="coerce").astype("Int64")
    out["REFERENCE_DATE"] = pd.to_datetime(out["REFERENCE_DATE"], errors="coerce")
    out["FACTOR_VALUE"] = pd.to_numeric(out["FACTOR_VALUE"], errors="coerce")
    out = out.dropna(subset=["ACCT_ID", "REFERENCE_DATE"])
    return out


def load_sec_factor_levels(
    mrd_engine: Engine,
    acct_ids: list[int] | tuple[int, ...],
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    ids = sorted({int(v) for v in (acct_ids or []) if v is not None})
    if not ids:
        return pd.DataFrame()

    start_norm = None if not start_date else str(pd.Timestamp(start_date).strftime("%Y-%m-%d"))
    end_norm = None if not end_date else str(pd.Timestamp(end_date).strftime("%Y-%m-%d"))
    points = _load_sec_factor_points_cached(mrd_engine, tuple(ids), start_norm, end_norm)
    if points.empty:
        return pd.DataFrame()

    levels = points.pivot_table(
        index="REFERENCE_DATE",
        columns="ACCT_ID",
        values="FACTOR_VALUE",
        aggfunc="last",
    ).sort_index()
    levels.index = pd.to_datetime(levels.index, errors="coerce")
    levels = levels.loc[~pd.isna(levels.index)]
    return levels


def _next_trading_day_lookup(start: pd.Timestamp, end: pd.Timestamp) -> dict[pd.Timestamp, pd.Timestamp]:
    nyse = mcal.get_calendar("NYSE")
    valid_days = nyse.valid_days(
        start_date=start - pd.Timedelta(days=7),
        end_date=end + pd.Timedelta(days=7),
    ).tz_localize(None)
    valid_days = pd.DatetimeIndex(valid_days).sort_values().unique()
    lookup: dict[pd.Timestamp, pd.Timestamp] = {}
    for idx in range(len(valid_days) - 1):
        lookup[pd.Timestamp(valid_days[idx])] = pd.Timestamp(valid_days[idx + 1])
    return lookup


def _infer_daily_start_from_observation_dates(observation_dates: pd.DatetimeIndex) -> pd.Timestamp | None:
    obs = pd.DatetimeIndex(observation_dates).sort_values().unique()
    if len(obs) < 3:
        return None
    next_lookup = _next_trading_day_lookup(pd.Timestamp(obs.min()), pd.Timestamp(obs.max()))
    for idx in range(len(obs) - 2):
        d0 = pd.Timestamp(obs[idx])
        d1 = pd.Timestamp(obs[idx + 1])
        d2 = pd.Timestamp(obs[idx + 2])
        if next_lookup.get(d0) == d1 and next_lookup.get(d1) == d2:
            return d1
    return None


def build_aa_compatible_returns_from_levels(
    levels_df: pd.DataFrame,
    acct_id_to_names: dict[int, list[str]],
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    if levels_df is None or levels_df.empty or not acct_id_to_names:
        return pd.DataFrame(), {}

    index_levels = levels_df.copy()
    index_levels.index = pd.to_datetime(index_levels.index, errors="coerce")
    index_levels = index_levels.loc[~pd.isna(index_levels.index)].sort_index()
    if index_levels.empty:
        return pd.DataFrame(), {}

    daily_returns = pd.DataFrame(index=index_levels.index, columns=index_levels.columns, dtype=float)
    for acct_id in index_levels.columns:
        levels_series = pd.to_numeric(index_levels[acct_id], errors="coerce").dropna()
        if levels_series.empty:
            continue
        series_returns = levels_series.pct_change(fill_method=None)
        daily_returns[acct_id] = series_returns.reindex(index_levels.index)
    daily_returns = daily_returns.dropna(how="all")

    out = pd.DataFrame(index=daily_returns.index)
    series_meta: dict[str, dict[str, object]] = {}
    for acct_id, names in acct_id_to_names.items():
        acct_id_int = int(acct_id)
        if acct_id_int not in daily_returns.columns:
            continue
        returns_series = daily_returns[acct_id_int]
        first_return = returns_series.dropna().index.min()
        obs_dates = (
            index_levels[acct_id_int].dropna().index
            if acct_id_int in index_levels.columns
            else pd.DatetimeIndex([])
        )
        daily_start = _infer_daily_start_from_observation_dates(obs_dates)
        starts_daily = bool(
            first_return is not None
            and daily_start is not None
            and pd.Timestamp(first_return) == pd.Timestamp(daily_start)
        )
        for name in names:
            out[name] = returns_series
            series_meta[name] = {
                "first_return_date": first_return.strftime("%Y-%m-%d") if first_return is not None else None,
                "daily_start_date": daily_start.strftime("%Y-%m-%d") if daily_start is not None else None,
                "starts_daily": starts_daily,
            }

    if out.empty:
        return out, series_meta

    full_index = pd.date_range(out.index.min(), out.index.max(), freq="D")
    out = out.reindex(full_index)
    for col in out.columns:
        last = out[col].last_valid_index()
        if last is None:
            continue
        daily_start_raw = (
            series_meta.get(col, {}).get("daily_start_date")
            if isinstance(series_meta.get(col, {}), dict)
            else None
        )
        if not daily_start_raw:
            continue
        daily_start = pd.to_datetime(daily_start_raw, errors="coerce")
        if pd.isna(daily_start):
            continue
        mask = (out.index >= daily_start) & (out.index <= last)
        out.loc[mask, col] = out.loc[mask, col].fillna(0.0)
    out = out.dropna(how="all")

    has_daily_phase = any(
        bool((meta_row or {}).get("daily_start_date")) or bool((meta_row or {}).get("starts_daily"))
        for meta_row in series_meta.values()
        if isinstance(meta_row, dict)
    )
    out.attrs["periodicity_hint"] = "daily" if has_daily_phase else "monthly"
    out.index.name = "Date"
    return out, series_meta


def load_sec_factor_returns_by_acct_ids_aa(
    mrd_engine: Engine,
    name_to_acct_id: dict[str, int] | None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    mapping = {}
    for raw_name, raw_id in (name_to_acct_id or {}).items():
        name = str(raw_name or "").strip()
        if not name:
            continue
        parsed_id = pd.to_numeric(pd.Series([raw_id]), errors="coerce").iloc[0]
        if pd.isna(parsed_id):
            continue
        mapping[name] = int(parsed_id)
    if not mapping:
        return pd.DataFrame(), {}

    levels = load_sec_factor_levels(
        mrd_engine,
        list(mapping.values()),
        start_date=start_date,
        end_date=end_date,
    )
    if levels.empty:
        return pd.DataFrame(), {}

    acct_to_names: dict[int, list[str]] = {}
    for name, acct_id in mapping.items():
        acct_to_names.setdefault(int(acct_id), []).append(name)

    out, meta = build_aa_compatible_returns_from_levels(levels, acct_to_names)
    if out.empty:
        return out, meta

    ordered_cols = [name for name in mapping.keys() if name in out.columns]
    out = out.reindex(columns=ordered_cols)
    out_meta = {name: meta.get(name, {}) for name in ordered_cols}
    return out, out_meta


def load_sec_factor_returns_by_names_aa(
    mrd_engine: Engine,
    names: list[str] | tuple[str, ...],
    collision_policy: str = "bb_then_lowest",
    exclude_perf: bool = False,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    requested_names: list[str] = []
    requested_to_canonical: dict[str, str] = {}
    seen_requested: set[str] = set()
    canonical_names: list[str] = []
    seen_canonical: set[str] = set()

    for raw_name in names or []:
        name = str(raw_name or "").strip()
        if not name:
            continue
        requested_key = name.lower()
        if requested_key in seen_requested:
            continue
        seen_requested.add(requested_key)

        canonical = normalize_sec_factor_name(name)
        if not canonical:
            continue
        requested_names.append(name)
        requested_to_canonical[name] = canonical
        canonical_key = canonical.lower()
        if canonical_key in seen_canonical:
            continue
        seen_canonical.add(canonical_key)
        canonical_names.append(canonical)

    if not requested_names:
        return pd.DataFrame(), {}

    resolved = resolve_sec_factor_accounts_by_names(
        mrd_engine,
        canonical_names,
        collision_policy=collision_policy,
        exclude_perf=exclude_perf,
    )
    if not resolved:
        return pd.DataFrame(), {}

    name_to_acct_id: dict[str, int] = {}
    for name in requested_names:
        canonical = requested_to_canonical.get(name)
        if not canonical:
            continue
        acct_id = resolved.get(canonical)
        if acct_id is None:
            continue
        name_to_acct_id[name] = int(acct_id)

    if not name_to_acct_id:
        return pd.DataFrame(), {}
    return load_sec_factor_returns_by_acct_ids_aa(
        mrd_engine,
        name_to_acct_id,
        start_date=start_date,
        end_date=end_date,
    )

