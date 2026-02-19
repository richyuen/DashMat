"""Helpers for raw database import workflows (factor/funds/performance)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine

import cache_config
from utils.constants import INDEX_BENCHMARK_SUFFIX

RawImportMode = Literal["factor", "funds", "performance"]

FACTOR_RETURN_DEFAULTS = {
    "TRIndex",
    "GRIndex",
    "PRIndex",
    "TRLocal",
    "GRLocal",
    "PRLocal",
}
RAW_PREVIEW_ROWS = 6
FACTOR_PREVIEW_BASE_ROWS = RAW_PREVIEW_ROWS + 1


@dataclass(frozen=True)
class RawImportResult:
    returns_df: pd.DataFrame
    benchmark_assignments: dict[str, str]
    periodicity: str


def _mrd_table_name(engine: Engine, table_name: str) -> str:
    if engine.dialect.name == "sqlite":
        return f"[CORE_DATA.{table_name}]"
    return f"[CORE_DATA].[{table_name}]"


def _perf_table_name(table_name: str) -> str:
    return f"[{table_name}]"


def _infer_periodicity(df: pd.DataFrame) -> str:
    if df.empty:
        return "monthly"
    idx = pd.DatetimeIndex(df.index).sort_values().unique()
    if len(idx) < 2:
        return "monthly"
    min_delta = pd.Series(idx).diff().dropna().min()
    if min_delta is not None and min_delta <= pd.Timedelta(days=7):
        return "daily"
    return "monthly"


def _format_value(value) -> str:
    num = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(num):
        return ""
    return f"{float(num):.10g}"


def factor_defaults_to_returns(factor_name: str | None) -> bool:
    return str(factor_name or "").strip() in FACTOR_RETURN_DEFAULTS


def _top_n_query(engine: Engine, base_sql: str, order_sql: str, top_n: int) -> str:
    if engine.dialect.name == "sqlite":
        return f"{base_sql} {order_sql} LIMIT {int(top_n)}"
    return f"{base_sql.replace('SELECT', f'SELECT TOP {int(top_n)}', 1)} {order_sql}"


def build_preview_row_from_controls(
    mode: str | None,
    series_key: str | None,
    table_choice: str | None,
    fee_choice: str | None,
    include_benchmark: bool | None,
    convert_to_returns: bool | None,
    divide_by: float | int | str | None,
) -> dict | None:
    mode_key = str(mode or "").strip().lower()
    key = str(series_key or "").strip()
    if mode_key not in {"factor", "funds", "performance"} or not key:
        return None

    row: dict[str, object] = {"mode": mode_key, "series_key": key}
    if mode_key == "factor":
        convert = bool(convert_to_returns)
        row["convert_to_returns"] = convert
        if convert:
            row["divide_by"] = 100.0
        else:
            div = pd.to_numeric(pd.Series([divide_by]), errors="coerce").iloc[0]
            row["divide_by"] = float(div) if not pd.isna(div) else divide_by
        return row

    if mode_key == "funds":
        row["table_choice"] = "monthly" if str(table_choice or "").lower() == "monthly" else "daily"
        row["fee_choice"] = "net" if str(fee_choice or "").lower().startswith("n") else "gross"
        return row

    row["table_choice"] = "monthly" if str(table_choice or "").lower() == "monthly" else "daily"
    row["fee_choice"] = "N" if str(fee_choice or "").upper().startswith("N") else "G"
    row["include_benchmark"] = bool(include_benchmark)
    return row


def _get_factor_option_rows(mrd_engine: Engine) -> list[dict]:
    account_table = _mrd_table_name(mrd_engine, "ACCOUNT")

    try:
        q = text(
            f"SELECT ACCT_ID, ACCT_NAME, ACCT_CD, FACTOR_NAME, SOURCE_SYSTEM "
            f"FROM {account_table} "
            "WHERE ACCT_TYPE_CD = 'SEC_FACTOR' "
            "AND COALESCE(SOURCE_SYSTEM, '') <> 'PERF' "
            "ORDER BY ACCT_NAME, FACTOR_NAME, SOURCE_SYSTEM, ACCT_CD, ACCT_ID"
        )
        with mrd_engine.connect() as conn:
            rows = conn.execute(q).fetchall()
    except Exception:
        # Backward-compatible fallback when SOURCE_SYSTEM is absent on ACCOUNT.
        factor_table = _mrd_table_name(mrd_engine, "ACCOUNT_FACTOR_DATA")
        q = text(
            f"SELECT a.ACCT_ID, a.ACCT_NAME, a.ACCT_CD, a.FACTOR_NAME, "
            f"MIN(COALESCE(fd.SOURCE_SYSTEM, '')) AS SOURCE_SYSTEM "
            f"FROM {account_table} a "
            f"LEFT JOIN {factor_table} fd ON fd.ACCT_ID = a.ACCT_ID "
            "WHERE a.ACCT_TYPE_CD = 'SEC_FACTOR' "
            "GROUP BY a.ACCT_ID, a.ACCT_NAME, a.ACCT_CD, a.FACTOR_NAME "
            "HAVING MIN(COALESCE(fd.SOURCE_SYSTEM, '')) <> 'PERF' "
            "ORDER BY a.ACCT_NAME, a.FACTOR_NAME, SOURCE_SYSTEM, a.ACCT_CD, a.ACCT_ID"
        )
        with mrd_engine.connect() as conn:
            rows = conn.execute(q).fetchall()

    out: list[dict] = []
    for acct_id, acct_name, acct_cd, factor_name, source_system in rows:
        acct_id_s = str(acct_id)
        import_name = f"{str(acct_name)}_{str(factor_name)}"
        source = str(source_system or "")
        label = f"{import_name} [{source}: {str(acct_cd)}]"
        out.append(
            {
                "value": acct_id_s,
                "label": label,
                "acct_id": int(acct_id),
                "acct_name": str(acct_name),
                "acct_cd": str(acct_cd),
                "factor_name": str(factor_name),
                "source_system": source,
                "import_name": import_name,
            }
        )
    return out


@cache_config.cache.memoize(timeout=0)
def get_factor_option_meta_cached(mrd_engine: Engine) -> dict[str, dict]:
    rows = _get_factor_option_rows(mrd_engine)
    return {str(r["value"]): r for r in rows}


@cache_config.cache.memoize(timeout=0)
def get_factor_options_cached(mrd_engine: Engine) -> list[dict]:
    meta = get_factor_option_meta_cached(mrd_engine)
    ordered = sorted(meta.items(), key=lambda item: str(item[1].get("label", "")))
    return [{"value": key, "label": str(row["label"])} for key, row in ordered]


def _get_fund_option_rows(mrd_engine: Engine) -> list[dict]:
    account_table = _mrd_table_name(mrd_engine, "ACCOUNT")
    q = text(
        f"SELECT ACCT_ID, ACCT_NAME, ACCT_CD "
        f"FROM {account_table} "
        "WHERE ACCT_TYPE_CD IN :acct_types "
        "AND SOURCE_SYSTEM = 'MSTAR' "
        "ORDER BY ACCT_NAME, ACCT_ID"
    ).bindparams(bindparam("acct_types", expanding=True))
    with mrd_engine.connect() as conn:
        rows = conn.execute(
            q,
            {"acct_types": ["OE", "SLEEVE", "TRUST"]},
        ).fetchall()
    out: list[dict] = []
    for acct_id, acct_name, acct_cd in rows:
        out.append(
            {
                "value": str(acct_id),
                "label": str(acct_name),
                "acct_id": int(acct_id),
                "acct_name": str(acct_name),
                "acct_cd": str(acct_cd),
                "import_name": str(acct_name),
            }
        )
    return out


@cache_config.cache.memoize(timeout=0)
def get_fund_option_meta_cached(mrd_engine: Engine) -> dict[str, dict]:
    rows = _get_fund_option_rows(mrd_engine)
    return {str(r["value"]): r for r in rows}


@cache_config.cache.memoize(timeout=0)
def get_fund_options_cached(mrd_engine: Engine) -> list[dict]:
    meta = get_fund_option_meta_cached(mrd_engine)
    ordered = sorted(meta.items(), key=lambda item: str(item[1].get("label", "")))
    return [{"value": key, "label": str(row["label"])} for key, row in ordered]


def _get_performance_option_rows(perf_engine: Engine) -> list[dict]:
    account_table = _perf_table_name("ACCOUNT")
    q = text(
        f"SELECT ACCT_ID, ACCT_CD "
        f"FROM {account_table} "
        "ORDER BY ACCT_CD, ACCT_ID"
    )
    with perf_engine.connect() as conn:
        rows = conn.execute(q).fetchall()

    out: list[dict] = []
    for acct_id, acct_cd in rows:
        code = str(acct_cd)
        out.append(
            {
                "value": str(acct_id),
                "label": code,
                "acct_id": int(acct_id),
                "acct_cd": code,
                "import_name": code,
            }
        )
    return out


@cache_config.cache.memoize(timeout=0)
def get_performance_option_meta_cached(perf_engine: Engine) -> dict[str, dict]:
    rows = _get_performance_option_rows(perf_engine)
    return {str(r["value"]): r for r in rows}


@cache_config.cache.memoize(timeout=0)
def get_performance_options_cached(perf_engine: Engine) -> list[dict]:
    meta = get_performance_option_meta_cached(perf_engine)
    ordered = sorted(meta.items(), key=lambda item: str(item[1].get("label", "")))
    return [{"value": key, "label": str(row["label"])} for key, row in ordered]


def _series_to_preview_lines(
    series: pd.Series,
    benchmark_series: pd.Series | None = None,
) -> list[str]:
    if series is None or series.empty:
        return []

    s = series.copy()
    s.index = pd.to_datetime(s.index, errors="coerce")
    s = s.loc[~pd.isna(s.index)]
    s = s[~s.index.duplicated(keep="last")].sort_index().dropna()
    if s.empty:
        return []
    s = s.iloc[:RAW_PREVIEW_ROWS]

    bm = None
    if benchmark_series is not None:
        bm = benchmark_series.copy()
        bm.index = pd.to_datetime(bm.index, errors="coerce")
        bm = bm.loc[~pd.isna(bm.index)]
        bm = bm[~bm.index.duplicated(keep="last")].sort_index()
        bm = bm.reindex(s.index)

    lines: list[str] = []
    for dt, value in s.items():
        dt_str = dt.strftime("%Y-%m-%d")
        val = _format_value(value)
        if bm is None:
            lines.append(f"{dt_str}:{val}")
        else:
            lines.append(f"{dt_str}:{val}|{_format_value(bm.loc[dt])}")
    return lines


@cache_config.cache.memoize(timeout=0)
def _load_factor_preview_points_cached(mrd_engine: Engine, acct_id: int) -> pd.DataFrame:
    factor_table = _mrd_table_name(mrd_engine, "ACCOUNT_FACTOR_DATA")
    sql = _top_n_query(
        mrd_engine,
        f"SELECT REFERENCE_DATE, FACTOR_VALUE FROM {factor_table} WHERE ACCT_ID = :acct_id",
        "ORDER BY REFERENCE_DATE",
        FACTOR_PREVIEW_BASE_ROWS,
    )
    with mrd_engine.connect() as conn:
        rows = conn.execute(text(sql), {"acct_id": int(acct_id)}).fetchall()
    if not rows:
        return pd.DataFrame(columns=["REFERENCE_DATE", "FACTOR_VALUE"])
    df = pd.DataFrame(rows, columns=["REFERENCE_DATE", "FACTOR_VALUE"])
    df["REFERENCE_DATE"] = pd.to_datetime(df["REFERENCE_DATE"], errors="coerce")
    df["FACTOR_VALUE"] = pd.to_numeric(df["FACTOR_VALUE"], errors="coerce")
    return df.dropna(subset=["REFERENCE_DATE"]).sort_values("REFERENCE_DATE")


@cache_config.cache.memoize(timeout=0)
def _load_fund_preview_points_cached(
    mrd_engine: Engine,
    acct_id: int,
    table_choice: str,
) -> pd.DataFrame:
    table_name = "ACCOUNT_RETURNS" if str(table_choice).lower() == "daily" else "ACCOUNT_RETURNS_M"
    returns_table = _mrd_table_name(mrd_engine, table_name)
    sql = _top_n_query(
        mrd_engine,
        f"SELECT REFERENCE_DATE, GROSS, NET "
        f"FROM {returns_table} "
        "WHERE ACCT_ID = :acct_id AND SOURCE_SYSTEM = 'MSTAR'",
        "ORDER BY REFERENCE_DATE",
        RAW_PREVIEW_ROWS,
    )
    with mrd_engine.connect() as conn:
        rows = conn.execute(text(sql), {"acct_id": int(acct_id)}).fetchall()
    if not rows:
        return pd.DataFrame(columns=["REFERENCE_DATE", "GROSS", "NET"])
    df = pd.DataFrame(rows, columns=["REFERENCE_DATE", "GROSS", "NET"])
    df["REFERENCE_DATE"] = pd.to_datetime(df["REFERENCE_DATE"], errors="coerce")
    df["GROSS"] = pd.to_numeric(df["GROSS"], errors="coerce")
    df["NET"] = pd.to_numeric(df["NET"], errors="coerce")
    return df.dropna(subset=["REFERENCE_DATE"]).sort_values("REFERENCE_DATE")


@cache_config.cache.memoize(timeout=0)
def _load_performance_preview_points_cached(
    perf_engine: Engine,
    acct_id: int,
    table_choice: str,
    fee_choice: str,
) -> pd.DataFrame:
    fee_type = "G" if str(fee_choice).upper().startswith("G") else "N"
    account_benchmark_table = _perf_table_name("ACCOUNT_BENCHMARK")
    table_lower = str(table_choice).lower()

    if table_lower == "monthly":
        returns_table = _perf_table_name("MONTHLY_RETURN")
        base_sql = (
            f"SELECT dr.Effective_Date AS REFERENCE_DATE, "
            "dr.mth1_ror AS PORT_RET, dr.mth1_ror_index AS BENCH_RET "
            f"FROM {returns_table} dr "
            f"JOIN {account_benchmark_table} ab ON ab.BENCHMARK_ID = dr.BENCHMARK_ACCT_ID "
            "WHERE dr.ACCT_ID = :acct_id "
            "AND dr.FEE_TYPE = :fee_type "
            "AND dr.IS_LATEST = 1 "
            "AND dr.Return_Type = 'Ann' "
            "AND ab.PRECEDENCE = 1"
        )
    else:
        returns_table = _perf_table_name("DAILY_RETURN")
        base_sql = (
            f"SELECT dr.Effective_Date AS REFERENCE_DATE, "
            "dr.Daily_ror AS PORT_RET, dr.Daily_ror_index AS BENCH_RET "
            f"FROM {returns_table} dr "
            f"JOIN {account_benchmark_table} ab ON ab.BENCHMARK_ID = dr.BENCHMARK_ACCT_ID "
            "WHERE dr.ACCT_ID = :acct_id "
            "AND dr.FEE_TYPE = :fee_type "
            "AND dr.IS_LATEST = 1 "
            "AND ab.PRECEDENCE = 1"
        )

    sql = _top_n_query(perf_engine, base_sql, "ORDER BY dr.Effective_Date", RAW_PREVIEW_ROWS)
    with perf_engine.connect() as conn:
        rows = conn.execute(text(sql), {"acct_id": int(acct_id), "fee_type": fee_type}).fetchall()
    if not rows:
        return pd.DataFrame(columns=["REFERENCE_DATE", "PORT_RET", "BENCH_RET"])
    df = pd.DataFrame(rows, columns=["REFERENCE_DATE", "PORT_RET", "BENCH_RET"])
    df["REFERENCE_DATE"] = pd.to_datetime(df["REFERENCE_DATE"], errors="coerce")
    df["PORT_RET"] = pd.to_numeric(df["PORT_RET"], errors="coerce") / 100.0
    df["BENCH_RET"] = pd.to_numeric(df["BENCH_RET"], errors="coerce") / 100.0
    return df.dropna(subset=["REFERENCE_DATE"]).sort_values("REFERENCE_DATE")


@cache_config.cache.memoize(timeout=0)
def get_factor_preview_lines_cached(
    mrd_engine: Engine,
    acct_id: str,
    convert_to_returns: bool = False,
    divide_by: float = 100.0,
) -> list[str]:
    acct_id_int = int(acct_id)
    if not bool(convert_to_returns):
        divide_value = pd.to_numeric(pd.Series([divide_by]), errors="coerce").iloc[0]
        if pd.isna(divide_value) or float(divide_value) == 0.0:
            return []

    points = _load_factor_preview_points_cached(mrd_engine, acct_id_int)
    if points.empty:
        return []

    series = pd.Series(
        points["FACTOR_VALUE"].values,
        index=pd.DatetimeIndex(points["REFERENCE_DATE"]),
        dtype=float,
    )
    series = series[~series.index.duplicated(keep="last")].sort_index()
    if bool(convert_to_returns):
        series = series.pct_change(fill_method=None).dropna()
    else:
        divide_value = pd.to_numeric(pd.Series([divide_by]), errors="coerce").iloc[0]
        series = (series / float(divide_value)).dropna()

    return _series_to_preview_lines(series)


@cache_config.cache.memoize(timeout=0)
def get_fund_preview_lines_cached(
    mrd_engine: Engine,
    acct_id: str,
    table_choice: str,
    fee_choice: str,
) -> list[str]:
    acct_id_int = int(acct_id)
    table_key = "monthly" if str(table_choice).lower() == "monthly" else "daily"
    value_col = "GROSS" if str(fee_choice).lower().startswith("g") else "NET"
    points = _load_fund_preview_points_cached(mrd_engine, acct_id_int, table_key)
    if points.empty:
        return []

    series = pd.Series(
        points[value_col].values,
        index=pd.DatetimeIndex(points["REFERENCE_DATE"]),
        dtype=float,
    )
    series = series[~series.index.duplicated(keep="last")].sort_index().dropna()
    return _series_to_preview_lines(series)


@cache_config.cache.memoize(timeout=0)
def get_performance_preview_lines_cached(
    perf_engine: Engine,
    acct_id: str,
    table_choice: str,
    fee_choice: str,
    include_benchmark: bool,
) -> list[str]:
    acct_id_int = int(acct_id)
    table_key = "monthly" if str(table_choice).lower() == "monthly" else "daily"
    fee_key = "G" if str(fee_choice).upper().startswith("G") else "N"
    points = _load_performance_preview_points_cached(perf_engine, acct_id_int, table_key, fee_key)
    if points.empty:
        return []

    port_series = pd.Series(
        points["PORT_RET"].values,
        index=pd.DatetimeIndex(points["REFERENCE_DATE"]),
        dtype=float,
    )
    port_series = port_series[~port_series.index.duplicated(keep="last")].sort_index().dropna()
    if not bool(include_benchmark):
        return _series_to_preview_lines(port_series)

    bench_series = pd.Series(
        points["BENCH_RET"].values,
        index=pd.DatetimeIndex(points["REFERENCE_DATE"]),
        dtype=float,
    )
    bench_series = bench_series[~bench_series.index.duplicated(keep="last")].sort_index()
    return _series_to_preview_lines(port_series, bench_series)


def get_preview_lines_for_row(
    row: dict,
    mrd_engine: Engine,
    perf_engine: Engine,
) -> list[str]:
    mode = str((row or {}).get("mode", "")).strip().lower()
    series_key = str((row or {}).get("series_key", "")).strip()
    if not series_key:
        return []

    if mode == "factor":
        convert_to_returns = bool((row or {}).get("convert_to_returns", False))
        divide_by = 100.0
        if not convert_to_returns:
            divide_by = pd.to_numeric(pd.Series([(row or {}).get("divide_by", 100)]), errors="coerce").iloc[0]
        return get_factor_preview_lines_cached(
            mrd_engine,
            series_key,
            convert_to_returns,
            divide_by,
        )
    if mode == "funds":
        return get_fund_preview_lines_cached(
            mrd_engine,
            series_key,
            str((row or {}).get("table_choice", "daily")),
            str((row or {}).get("fee_choice", "gross")),
        )
    if mode == "performance":
        return get_performance_preview_lines_cached(
            perf_engine,
            series_key,
            str((row or {}).get("table_choice", "daily")),
            str((row or {}).get("fee_choice", "G")),
            bool((row or {}).get("include_benchmark", False)),
        )
    return []


@cache_config.cache.memoize(timeout=0)
def _load_factor_points_cached(mrd_engine: Engine, acct_ids: tuple[int, ...]) -> pd.DataFrame:
    ids = [int(v) for v in acct_ids if v is not None]
    if not ids:
        return pd.DataFrame(columns=["ACCT_ID", "REFERENCE_DATE", "FACTOR_VALUE"])
    factor_table = _mrd_table_name(mrd_engine, "ACCOUNT_FACTOR_DATA")
    q = text(
        f"SELECT ACCT_ID, REFERENCE_DATE, FACTOR_VALUE "
        f"FROM {factor_table} "
        "WHERE ACCT_ID IN :acct_ids "
        "ORDER BY REFERENCE_DATE, ACCT_ID"
    ).bindparams(bindparam("acct_ids", expanding=True))
    with mrd_engine.connect() as conn:
        rows = conn.execute(q, {"acct_ids": ids}).fetchall()
    if not rows:
        return pd.DataFrame(columns=["ACCT_ID", "REFERENCE_DATE", "FACTOR_VALUE"])
    df = pd.DataFrame(rows, columns=["ACCT_ID", "REFERENCE_DATE", "FACTOR_VALUE"])
    df["ACCT_ID"] = pd.to_numeric(df["ACCT_ID"], errors="coerce").astype("Int64")
    df["REFERENCE_DATE"] = pd.to_datetime(df["REFERENCE_DATE"], errors="coerce")
    df["FACTOR_VALUE"] = pd.to_numeric(df["FACTOR_VALUE"], errors="coerce")
    return df.dropna(subset=["ACCT_ID", "REFERENCE_DATE"])


@cache_config.cache.memoize(timeout=0)
def _load_fund_points_cached(
    mrd_engine: Engine,
    acct_ids: tuple[int, ...],
    table_choice: str,
) -> pd.DataFrame:
    ids = [int(v) for v in acct_ids if v is not None]
    if not ids:
        return pd.DataFrame(columns=["ACCT_ID", "REFERENCE_DATE", "GROSS", "NET"])
    table_name = "ACCOUNT_RETURNS" if str(table_choice).lower() == "daily" else "ACCOUNT_RETURNS_M"
    returns_table = _mrd_table_name(mrd_engine, table_name)
    q = text(
        f"SELECT ACCT_ID, REFERENCE_DATE, GROSS, NET "
        f"FROM {returns_table} "
        "WHERE ACCT_ID IN :acct_ids "
        "AND SOURCE_SYSTEM = 'MSTAR' "
        "ORDER BY REFERENCE_DATE, ACCT_ID"
    ).bindparams(bindparam("acct_ids", expanding=True))
    with mrd_engine.connect() as conn:
        rows = conn.execute(q, {"acct_ids": ids}).fetchall()
    if not rows:
        return pd.DataFrame(columns=["ACCT_ID", "REFERENCE_DATE", "GROSS", "NET"])
    df = pd.DataFrame(rows, columns=["ACCT_ID", "REFERENCE_DATE", "GROSS", "NET"])
    df["ACCT_ID"] = pd.to_numeric(df["ACCT_ID"], errors="coerce").astype("Int64")
    df["REFERENCE_DATE"] = pd.to_datetime(df["REFERENCE_DATE"], errors="coerce")
    df["GROSS"] = pd.to_numeric(df["GROSS"], errors="coerce")
    df["NET"] = pd.to_numeric(df["NET"], errors="coerce")
    return df.dropna(subset=["ACCT_ID", "REFERENCE_DATE"])


@cache_config.cache.memoize(timeout=0)
def _load_performance_points_cached(
    perf_engine: Engine,
    acct_ids: tuple[int, ...],
    table_choice: str,
    fee_choice: str,
) -> pd.DataFrame:
    ids = [int(v) for v in acct_ids if v is not None]
    if not ids:
        return pd.DataFrame(columns=["ACCT_ID", "REFERENCE_DATE", "PORT_RET", "BENCH_RET"])

    fee_type = "G" if str(fee_choice).upper().startswith("G") else "N"
    account_benchmark_table = _perf_table_name("ACCOUNT_BENCHMARK")
    table_lower = str(table_choice).lower()

    if table_lower == "monthly":
        returns_table = _perf_table_name("MONTHLY_RETURN")
        q = text(
            f"SELECT dr.ACCT_ID, dr.Effective_Date AS REFERENCE_DATE, "
            "dr.mth1_ror AS PORT_RET, dr.mth1_ror_index AS BENCH_RET "
            f"FROM {returns_table} dr "
            f"JOIN {account_benchmark_table} ab ON ab.BENCHMARK_ID = dr.BENCHMARK_ACCT_ID "
            "WHERE dr.ACCT_ID IN :acct_ids "
            "AND dr.FEE_TYPE = :fee_type "
            "AND dr.IS_LATEST = 1 "
            "AND dr.Return_Type = 'Ann' "
            "AND ab.PRECEDENCE = 1 "
            "ORDER BY dr.Effective_Date, dr.ACCT_ID"
        ).bindparams(bindparam("acct_ids", expanding=True))
    else:
        returns_table = _perf_table_name("DAILY_RETURN")
        q = text(
            f"SELECT dr.ACCT_ID, dr.Effective_Date AS REFERENCE_DATE, "
            "dr.Daily_ror AS PORT_RET, dr.Daily_ror_index AS BENCH_RET "
            f"FROM {returns_table} dr "
            f"JOIN {account_benchmark_table} ab ON ab.BENCHMARK_ID = dr.BENCHMARK_ACCT_ID "
            "WHERE dr.ACCT_ID IN :acct_ids "
            "AND dr.FEE_TYPE = :fee_type "
            "AND dr.IS_LATEST = 1 "
            "AND ab.PRECEDENCE = 1 "
            "ORDER BY dr.Effective_Date, dr.ACCT_ID"
        ).bindparams(bindparam("acct_ids", expanding=True))

    with perf_engine.connect() as conn:
        rows = conn.execute(
            q,
            {"acct_ids": ids, "fee_type": fee_type},
        ).fetchall()
    if not rows:
        return pd.DataFrame(columns=["ACCT_ID", "REFERENCE_DATE", "PORT_RET", "BENCH_RET"])
    df = pd.DataFrame(rows, columns=["ACCT_ID", "REFERENCE_DATE", "PORT_RET", "BENCH_RET"])
    df["ACCT_ID"] = pd.to_numeric(df["ACCT_ID"], errors="coerce").astype("Int64")
    df["REFERENCE_DATE"] = pd.to_datetime(df["REFERENCE_DATE"], errors="coerce")
    df["PORT_RET"] = pd.to_numeric(df["PORT_RET"], errors="coerce") / 100.0
    df["BENCH_RET"] = pd.to_numeric(df["BENCH_RET"], errors="coerce") / 100.0
    return df.dropna(subset=["ACCT_ID", "REFERENCE_DATE"])


def load_factor_series(
    mrd_engine: Engine,
    staged_rows: list[dict] | None,
) -> RawImportResult:
    rows = [r for r in (staged_rows or []) if isinstance(r, dict)]
    if not rows:
        return RawImportResult(pd.DataFrame(), {}, "monthly")

    acct_ids = sorted({int(str(r.get("series_key", "0"))) for r in rows if str(r.get("series_key", "")).strip()})
    points = _load_factor_points_cached(mrd_engine, tuple(acct_ids))
    if points.empty:
        return RawImportResult(pd.DataFrame(), {}, "monthly")

    series_map: dict[str, pd.Series] = {}
    ordered_cols: list[str] = []
    for row in rows:
        series_key = str(row.get("series_key", "")).strip()
        import_name = str(row.get("import_name", "")).strip()
        if not series_key or not import_name:
            continue
        if import_name in series_map:
            raise ValueError(f"Duplicate staged series name `{import_name}`.")

        acct_id = int(series_key)
        subset = points.loc[points["ACCT_ID"] == acct_id, ["REFERENCE_DATE", "FACTOR_VALUE"]].copy()
        if subset.empty:
            raise ValueError(f"No factor data rows found for ACCT_ID `{acct_id}`.")

        subset = subset.sort_values("REFERENCE_DATE")
        series = pd.Series(
            subset["FACTOR_VALUE"].values,
            index=pd.DatetimeIndex(subset["REFERENCE_DATE"]),
            name=import_name,
            dtype=float,
        )
        series = series[~series.index.duplicated(keep="last")]

        convert = bool(row.get("convert_to_returns", False))
        if convert:
            series = series.pct_change(fill_method=None).dropna()
        else:
            divide_by = pd.to_numeric(pd.Series([row.get("divide_by", 100)]), errors="coerce").iloc[0]
            if pd.isna(divide_by) or float(divide_by) == 0.0:
                raise ValueError(f"Invalid divide-by value for `{import_name}`.")
            series = (series / float(divide_by)).dropna()

        if series.empty:
            raise ValueError(f"No usable factor values for `{import_name}`.")
        series_map[import_name] = series.rename(import_name)
        ordered_cols.append(import_name)

    if not ordered_cols:
        return RawImportResult(pd.DataFrame(), {}, "monthly")
    out = pd.concat([series_map[c] for c in ordered_cols], axis=1)
    out = out.sort_index().dropna(how="all")
    out.index.name = "Date"
    return RawImportResult(out, {}, _infer_periodicity(out))


def load_fund_series(
    mrd_engine: Engine,
    staged_rows: list[dict] | None,
) -> RawImportResult:
    rows = [r for r in (staged_rows or []) if isinstance(r, dict)]
    if not rows:
        return RawImportResult(pd.DataFrame(), {}, "monthly")

    series_map: dict[str, pd.Series] = {}
    ordered_cols: list[str] = []
    table_periods: list[str] = []
    for table_choice in ("daily", "monthly"):
        ids = sorted(
            {
                int(str(r.get("series_key", "0")))
                for r in rows
                if str(r.get("mode", "")).lower() == "funds"
                and str(r.get("table_choice", "daily")).lower() == table_choice
                and str(r.get("series_key", "")).strip()
            }
        )
        if not ids:
            continue
        points = _load_fund_points_cached(mrd_engine, tuple(ids), table_choice)
        if points.empty:
            continue

        for row in rows:
            if str(row.get("mode", "")).lower() != "funds":
                continue
            if str(row.get("table_choice", "daily")).lower() != table_choice:
                continue

            series_key = str(row.get("series_key", "")).strip()
            import_name = str(row.get("import_name", "")).strip()
            if not series_key or not import_name:
                continue
            if import_name in series_map:
                raise ValueError(f"Duplicate staged series name `{import_name}`.")

            acct_id = int(series_key)
            fee_choice = str(row.get("fee_choice", "gross")).lower()
            value_col = "GROSS" if fee_choice.startswith("g") else "NET"

            subset = points.loc[points["ACCT_ID"] == acct_id, ["REFERENCE_DATE", value_col]].copy()
            if subset.empty:
                raise ValueError(
                    f"No fund rows found for `{import_name}` ({table_choice}, {value_col.lower()})."
                )
            subset = subset.sort_values("REFERENCE_DATE")
            series = pd.Series(
                subset[value_col].values,
                index=pd.DatetimeIndex(subset["REFERENCE_DATE"]),
                name=import_name,
                dtype=float,
            )
            series = series[~series.index.duplicated(keep="last")].dropna()
            if series.empty:
                raise ValueError(f"No usable fund values for `{import_name}`.")
            series_map[import_name] = series.rename(import_name)
            ordered_cols.append(import_name)
            table_periods.append(table_choice)

    if not ordered_cols:
        return RawImportResult(pd.DataFrame(), {}, "monthly")
    out = pd.concat([series_map[c] for c in ordered_cols], axis=1)
    out = out.sort_index().dropna(how="all")
    out.index.name = "Date"
    periodicity = "daily" if any(p == "daily" for p in table_periods) else _infer_periodicity(out)
    return RawImportResult(out, {}, periodicity)


def load_performance_series(
    perf_engine: Engine,
    staged_rows: list[dict] | None,
) -> RawImportResult:
    rows = [r for r in (staged_rows or []) if isinstance(r, dict)]
    if not rows:
        return RawImportResult(pd.DataFrame(), {}, "monthly")

    series_map: dict[str, pd.Series] = {}
    ordered_cols: list[str] = []
    benchmark_assignments: dict[str, str] = {}
    table_periods: list[str] = []

    for table_choice in ("daily", "monthly"):
        for fee_choice in ("G", "N"):
            ids = sorted(
                {
                    int(str(r.get("series_key", "0")))
                    for r in rows
                    if str(r.get("mode", "")).lower() == "performance"
                    and str(r.get("table_choice", "daily")).lower() == table_choice
                    and str(r.get("fee_choice", "G")).upper().startswith(fee_choice)
                    and str(r.get("series_key", "")).strip()
                }
            )
            if not ids:
                continue
            points = _load_performance_points_cached(perf_engine, tuple(ids), table_choice, fee_choice)
            if points.empty:
                continue

            for row in rows:
                if str(row.get("mode", "")).lower() != "performance":
                    continue
                if str(row.get("table_choice", "daily")).lower() != table_choice:
                    continue
                if not str(row.get("fee_choice", "G")).upper().startswith(fee_choice):
                    continue

                series_key = str(row.get("series_key", "")).strip()
                import_name = str(row.get("import_name", "")).strip()
                include_benchmark = bool(row.get("include_benchmark", False))
                if not series_key or not import_name:
                    continue
                if import_name in series_map:
                    raise ValueError(f"Duplicate staged series name `{import_name}`.")

                acct_id = int(series_key)
                subset = points.loc[points["ACCT_ID"] == acct_id, ["REFERENCE_DATE", "PORT_RET", "BENCH_RET"]].copy()
                if subset.empty:
                    raise ValueError(
                        f"No performance rows found for `{import_name}` ({table_choice}, {fee_choice})."
                    )
                subset = subset.sort_values("REFERENCE_DATE")

                port_series = pd.Series(
                    subset["PORT_RET"].values,
                    index=pd.DatetimeIndex(subset["REFERENCE_DATE"]),
                    name=import_name,
                    dtype=float,
                )
                port_series = port_series[~port_series.index.duplicated(keep="last")].dropna()
                if port_series.empty:
                    raise ValueError(f"No usable performance values for `{import_name}`.")
                series_map[import_name] = port_series.rename(import_name)
                ordered_cols.append(import_name)

                if include_benchmark:
                    bm_name = f"{import_name}{INDEX_BENCHMARK_SUFFIX}"
                    if bm_name in series_map:
                        raise ValueError(f"Duplicate benchmark series name `{bm_name}`.")
                    bm_series = pd.Series(
                        subset["BENCH_RET"].values,
                        index=pd.DatetimeIndex(subset["REFERENCE_DATE"]),
                        name=bm_name,
                        dtype=float,
                    )
                    bm_series = bm_series[~bm_series.index.duplicated(keep="last")].dropna()
                    if bm_series.empty:
                        raise ValueError(f"No usable performance benchmark values for `{import_name}`.")
                    series_map[bm_name] = bm_series.rename(bm_name)
                    ordered_cols.append(bm_name)
                    benchmark_assignments[import_name] = bm_name

                table_periods.append(table_choice)

    if not ordered_cols:
        return RawImportResult(pd.DataFrame(), {}, "monthly")
    out = pd.concat([series_map[c] for c in ordered_cols], axis=1)
    out = out.sort_index().dropna(how="all")
    out.index.name = "Date"
    periodicity = "daily" if any(p == "daily" for p in table_periods) else _infer_periodicity(out)
    return RawImportResult(out, benchmark_assignments, periodicity)
