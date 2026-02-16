"""CMA portfolio-series helpers for peer/index/other import workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine

from utils.constants import (
    INDEX_BENCHMARK_TYPE_OPTIONS,
    INDEX_BENCHMARK_DESC,
    INDEX_BENCHMARK_SUFFIX,
    INDEX_PORTFOLIO_TYPE_OPTIONS,
    PEER_BENCHMARK_TYPE_OPTIONS,
    PEER_PORTFOLIO_TYPE_OPTIONS,
    PORTFOLIO_TS_VALUE_MODE,
)

PortfolioMode = Literal["peer", "index", "other"]
OTHER_SUPPORTED_SOURCES = {"AltTS", "Perf"}
PERF_GROSS_FEE_TYPE = "G"


def _option_db_values(options: list[dict]) -> set[str]:
    values: set[str] = set()
    for opt in options:
        if not isinstance(opt, dict):
            continue
        db_val = opt.get("db_value")
        if db_val is None:
            continue
        values.add(str(db_val).strip())
    return {v for v in values if v}


PEER_PORTFOLIO_BENCHMARK_OVERLAP = _option_db_values(PEER_BENCHMARK_TYPE_OPTIONS).intersection(
    _option_db_values(PEER_PORTFOLIO_TYPE_OPTIONS)
)
INDEX_PORTFOLIO_BENCHMARK_OVERLAP = _option_db_values(INDEX_BENCHMARK_TYPE_OPTIONS).intersection(
    _option_db_values(INDEX_PORTFOLIO_TYPE_OPTIONS)
)
INDEX_BENCHMARK_DESC_VALUES = _option_db_values(INDEX_BENCHMARK_TYPE_OPTIONS)


def _first_portfolio_type(mode: PortfolioMode) -> str:
    if mode == "peer":
        options = PEER_PORTFOLIO_TYPE_OPTIONS
    elif mode == "index":
        options = INDEX_PORTFOLIO_TYPE_OPTIONS
    else:
        return ""

    for opt in options:
        if not isinstance(opt, dict):
            continue
        val = str(opt.get("db_value", "")).strip()
        if val:
            return val
    return ""


def _effective_portfolio_series_name(portfolio: str, mode: PortfolioMode, selected_type: str | None) -> str:
    p = str(portfolio or "").strip()
    if mode not in {"peer", "index"}:
        return p

    selected = str(selected_type or "").strip()
    first_type = _first_portfolio_type(mode)
    if not selected or selected == first_type:
        return p
    return f"{p}_{selected}"


def _should_convert_levels_to_returns(series: pd.Series) -> bool:
    mode = str(PORTFOLIO_TS_VALUE_MODE or "auto").strip().lower()
    if mode == "levels":
        return True
    if mode == "returns":
        return False

    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return False

    # Levels are expected to be strictly positive and usually above return scale.
    if (clean <= 0).any():
        return False
    frac_abs_gt_half = float((clean.abs() > 0.5).mean())
    median_abs = float(clean.abs().median())
    return frac_abs_gt_half > 0.8 or median_abs > 2.0


def _normalize_values_to_returns(series: pd.Series) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce").sort_index()
    if _should_convert_levels_to_returns(clean):
        clean = clean.pct_change(fill_method=None)
    return clean.dropna()


@dataclass(frozen=True)
class PortfolioImportResult:
    returns_df: pd.DataFrame
    benchmark_assignments: dict[str, str]
    periodicity: str


def get_portfolio_options(engine: Engine, mode: PortfolioMode) -> list[dict]:
    """Return dropdown options in `PortfolioName [Portfolio]` format."""
    if mode not in {"peer", "index", "other"}:
        return []

    if mode == "other":
        q = text(
            "SELECT p.Portfolio, p.PortfolioName "
            "FROM Portfolios p "
            "WHERE p.PortfolioSuite = 'IndNoAttr' "
            "AND COALESCE(p.PortfolioVintage, '') IN ('AltTS', 'Perf') "
            "ORDER BY p.PortfolioID, p.Portfolio"
        )
    elif mode == "peer":
        where_clause = (
            "COALESCE(s.PeerTDOrder, 0) > 0 "
            "OR COALESCE(s.PeerModelOrder, 0) > 0 "
            "OR COALESCE(s.PeerAllocOrder, 0) > 0 "
            "OR COALESCE(s.Peer529Order, 0) > 0"
        )
        order_clause = "p.PortfolioName, p.Portfolio"
    else:
        where_clause = "COALESCE(s.IndexDailyOrder, 0) > 0 AND COALESCE(p.PortfolioSuite, '') <> 'IndNoAttr'"
        order_clause = "p.PortfolioName, p.Portfolio"

    if mode in {"peer", "index"}:
        q = text(
            "SELECT p.Portfolio, p.PortfolioName "
            "FROM Portfolios p "
            "JOIN Suites s ON s.SuiteShort = p.PortfolioSuite "
            f"WHERE {where_clause} "
            f"ORDER BY {order_clause}"
        )
    with engine.connect() as conn:
        rows = conn.execute(q).fetchall()

    options: list[dict] = []
    for portfolio, portfolio_name in rows:
        if not portfolio:
            continue
        p = str(portfolio)
        p_name = str(portfolio_name or p)
        options.append({"value": p, "label": f"{p_name} [{p}]"})
    return options


def has_portfolio_benchmark(engine: Engine, mode: PortfolioMode, portfolio: str | None) -> bool:
    p = str(portfolio or "").strip()
    if mode not in {"peer", "index", "other"} or not p:
        return False

    if mode == "index":
        descs = sorted(INDEX_BENCHMARK_DESC_VALUES or {INDEX_BENCHMARK_DESC})
        q = text(
            "SELECT COUNT(1) "
            "FROM IndexTS "
            "WHERE Portfolio = :portfolio "
            "AND Item = :item "
            "AND [Desc] IN :descs"
        ).bindparams(bindparam("descs", expanding=True))
        with engine.connect() as conn:
            count = conn.execute(
                q,
                {"portfolio": p, "item": "PortRet", "descs": descs},
            ).scalar()
        return int(count or 0) > 0

    meta_q = text(
        "SELECT PeerVintage, PortfolioVintage "
        "FROM Portfolios "
        "WHERE Portfolio = :portfolio"
    )
    with engine.connect() as conn:
        row = conn.execute(meta_q, {"portfolio": p}).first()
    if not row:
        return False
    vintage = str(row[0] or "").strip()
    source = str(row[1] or "").strip()

    if mode == "other":
        if source == "AltTS":
            q = text(
                "SELECT COUNT(1) "
                "FROM AltTS "
                "WHERE Portfolio = :portfolio "
                "AND Item = :item"
            )
            with engine.connect() as conn:
                count = conn.execute(q, {"portfolio": p, "item": "BenchRet"}).scalar()
            return int(count or 0) > 0
        if source == "Perf":
            # Perf benchmark availability is metadata-driven.
            return bool(vintage)
        return False

    if vintage:
        peer_q = text(
            "SELECT COUNT(1) "
            "FROM PeerTS "
            "WHERE Item = :item "
            "AND (Portfolio = :portfolio OR Portfolio = :vintage)"
        )
        params = {"item": "MeanRet", "portfolio": p, "vintage": vintage}
    else:
        peer_q = text(
            "SELECT COUNT(1) "
            "FROM PeerTS "
            "WHERE Item = :item "
            "AND Portfolio = :portfolio"
        )
        params = {"item": "MeanRet", "portfolio": p}

    with engine.connect() as conn:
        count = conn.execute(peer_q, params).scalar()
    return int(count or 0) > 0


def _read_series(
    engine: Engine,
    table_name: str,
    portfolio: str,
    item: str,
    desc: str,
    normalize_values: bool = True,
) -> pd.Series:
    q = text(
        f"SELECT Date, Value "
        f"FROM {table_name} "
        "WHERE Portfolio = :portfolio "
        "AND Item = :item "
        "AND [Desc] = :desc "
        "ORDER BY Date"
    )
    with engine.connect() as conn:
        rows = conn.execute(
            q,
            {
                "portfolio": portfolio,
                "item": item,
                "desc": desc,
            },
        ).fetchall()

    return _rows_to_series(rows, portfolio, normalize_values=normalize_values)


def _read_series_no_desc(
    engine: Engine,
    table_name: str,
    portfolio: str,
    item: str,
    normalize_values: bool = True,
) -> pd.Series:
    q = text(
        f"SELECT Date, Value "
        f"FROM {table_name} "
        "WHERE Portfolio = :portfolio "
        "AND Item = :item "
        "ORDER BY Date"
    )
    with engine.connect() as conn:
        rows = conn.execute(
            q,
            {
                "portfolio": portfolio,
                "item": item,
            },
        ).fetchall()

    return _rows_to_series(rows, portfolio, normalize_values=normalize_values)


def _performance_table_name(table_name: str) -> str:
    return f"[{table_name}]"


def _read_perf_series(
    performance_engine: Engine,
    portfolio: str,
    value_column: str,
) -> pd.Series:
    if value_column not in {"Daily_ror", "Daily_ror_index"}:
        raise ValueError(f"Unsupported performance value column: {value_column}")

    account_table = _performance_table_name("ACCOUNT")
    account_benchmark_table = _performance_table_name("ACCOUNT_BENCHMARK")
    daily_return_table = _performance_table_name("DAILY_RETURN")

    q = text(
        f"SELECT dr.Effective_Date AS Date, dr.{value_column} AS Value "
        f"FROM {daily_return_table} dr "
        f"JOIN {account_table} a ON a.ACCT_ID = dr.ACCT_ID "
        f"JOIN {account_benchmark_table} ab ON ab.BENCHMARK_ID = dr.BENCHMARK_ACCT_ID "
        "WHERE a.ACCT_CD = :portfolio "
        "AND dr.IS_LATEST = 1 "
        "AND ab.PRECEDENCE = 1 "
        "AND dr.FEE_TYPE = :fee_type "
        "ORDER BY dr.Effective_Date"
    )
    with performance_engine.connect() as conn:
        rows = conn.execute(
            q,
            {
                "portfolio": portfolio,
                "fee_type": PERF_GROSS_FEE_TYPE,
            },
        ).fetchall()

    series = _rows_to_series(rows, portfolio, normalize_values=False)
    if series.empty:
        return series
    series = (series / 100.0).dropna()
    return series.rename(portfolio)


def _rows_to_series(rows, series_name: str, normalize_values: bool = True) -> pd.Series:
    if not rows:
        return pd.Series(dtype=float, name=series_name)

    df = pd.DataFrame(rows, columns=["Date", "Value"])
    df["Date"] = pd.to_datetime(df["Date"])
    series = pd.Series(
        pd.to_numeric(df["Value"], errors="coerce").values,
        index=pd.DatetimeIndex(df["Date"]),
        name=series_name,
        dtype=float,
    ).sort_index()
    series = series[~series.index.duplicated(keep="last")]
    if normalize_values:
        series = _normalize_values_to_returns(series)
    else:
        series = series.dropna()
    series = series.rename(series_name)
    return series


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


def load_portfolio_series(
    engine: Engine,
    mode: PortfolioMode,
    staged_rows: list[dict] | None,
    performance_engine: Engine | None = None,
) -> PortfolioImportResult:
    """Load staged portfolio requests from CMA tables."""
    rows = [r for r in (staged_rows or []) if isinstance(r, dict)]
    if mode not in {"peer", "index", "other"} or not rows:
        return PortfolioImportResult(pd.DataFrame(), {}, "monthly")
    perf_engine = performance_engine or engine

    requested = [str(r.get("portfolio", "")).strip() for r in rows if r.get("portfolio")]
    if not requested:
        return PortfolioImportResult(pd.DataFrame(), {}, "monthly")

    portfolio_query = text(
        "SELECT Portfolio, PortfolioName, PeerVintage, PortfolioVintage, IncepDate "
        "FROM Portfolios "
        "WHERE Portfolio IN :portfolios"
    ).bindparams(bindparam("portfolios", expanding=True))

    with engine.connect() as conn:
        meta_rows = conn.execute(
            portfolio_query,
            {"portfolios": sorted(set(requested))},
        ).fetchall()

    metadata: dict[str, dict[str, object]] = {}
    for portfolio, portfolio_name, peer_vintage, portfolio_vintage, incep_date in meta_rows:
        p = str(portfolio or "").strip()
        if not p:
            continue
        metadata[p] = {
            "portfolio_name": str(portfolio_name or p),
            "peer_vintage": str(peer_vintage).strip() if peer_vintage is not None else "",
            "portfolio_vintage": str(portfolio_vintage).strip() if portfolio_vintage is not None else "",
            "incep_date": pd.to_datetime(incep_date) if incep_date else None,
        }

    series_map: dict[str, pd.Series] = {}
    ordered_cols: list[str] = []
    benchmark_assignments: dict[str, str] = {}
    peer_bench_type_by_vintage: dict[str, str] = {}

    for row in rows:
        portfolio = str(row.get("portfolio", "")).strip()
        if not portfolio:
            continue
        if portfolio not in metadata:
            raise ValueError(f"Unknown portfolio key: {portfolio}")

        ret_desc = str(row.get("type", "")).strip()
        include_benchmark = bool(row.get("include_benchmark", False))
        benchmark_type = str(row.get("benchmark_type", "")).strip()

        if not ret_desc:
            raise ValueError(f"Missing type for portfolio: {portfolio}")
        portfolio_col = _effective_portfolio_series_name(portfolio, mode, ret_desc)

        if mode == "peer":
            port_series = _read_series(engine, "PeerTS", portfolio, "PortRet", ret_desc)
        elif mode == "index":
            port_series = _read_series(engine, "IndexTS", portfolio, "PortRet", ret_desc)
        else:
            source = str(metadata[portfolio].get("portfolio_vintage", "")).strip()
            if source == "AltTS":
                # AltTS stores arithmetic returns directly.
                port_series = _read_series_no_desc(
                    engine,
                    "AltTS",
                    portfolio,
                    "PortRet",
                    normalize_values=False,
                )
            elif source == "Perf":
                port_series = _read_perf_series(perf_engine, portfolio, "Daily_ror")
            else:
                raise ValueError(
                    f"PortfolioVintage `{source}` is not implemented for portfolio `{portfolio}`."
                )

        incep = metadata[portfolio].get("incep_date")
        if incep is not None:
            incep_ts = pd.Timestamp(incep)
            port_series = port_series.loc[port_series.index >= incep_ts]

        if port_series.empty:
            raise ValueError(f"No rows found for portfolio `{portfolio}` with type `{ret_desc}`.")

        if portfolio_col not in series_map:
            series_map[portfolio_col] = port_series.rename(portfolio_col)
            ordered_cols.append(portfolio_col)

        if not include_benchmark:
            continue

        if mode == "peer":
            vintage = str(metadata[portfolio].get("peer_vintage", "")).strip()
            if not vintage:
                raise ValueError(f"PeerVintage is missing for portfolio `{portfolio}`.")
            if not benchmark_type:
                raise ValueError(f"Missing benchmark type for portfolio `{portfolio}`.")

            if benchmark_type in PEER_PORTFOLIO_BENCHMARK_OVERLAP:
                # Special peer benchmark mode:
                # when benchmark type matches a peer portfolio type db_value,
                # use the matching portfolio return series in PeerTS.
                bm_col = f"{portfolio}_{benchmark_type}"
                if bm_col not in series_map:
                    bm_series = _read_series(engine, "PeerTS", portfolio, "PortRet", benchmark_type)
                    if bm_series.empty:
                        raise ValueError(
                            f"No peer benchmark rows for portfolio `{portfolio}` and type `{benchmark_type}`."
                        )
                    series_map[bm_col] = bm_series.rename(bm_col)
                    ordered_cols.append(bm_col)
                benchmark_assignments[portfolio_col] = bm_col
            else:
                prev_type = peer_bench_type_by_vintage.get(vintage)
                if prev_type and prev_type != benchmark_type:
                    raise ValueError(
                        f"Peer vintage `{vintage}` requested with multiple benchmark types "
                        f"(`{prev_type}` and `{benchmark_type}`)."
                    )
                peer_bench_type_by_vintage[vintage] = benchmark_type

                if vintage not in series_map:
                    bm_series = _read_series(engine, "PeerTS", vintage, "MeanRet", benchmark_type)
                    if bm_series.empty:
                        raise ValueError(
                            f"No peer benchmark rows for vintage `{vintage}` and type `{benchmark_type}`."
                        )
                    series_map[vintage] = bm_series.rename(vintage)
                    ordered_cols.append(vintage)
                benchmark_assignments[portfolio_col] = vintage
        elif mode == "index":
            if not benchmark_type:
                raise ValueError(f"Missing benchmark type for portfolio `{portfolio}`.")

            if benchmark_type in INDEX_PORTFOLIO_BENCHMARK_OVERLAP:
                bm_col = _effective_portfolio_series_name(portfolio, mode, benchmark_type)
                if bm_col not in series_map:
                    bm_series = _read_series(engine, "IndexTS", portfolio, "PortRet", benchmark_type)
                    if bm_series.empty:
                        raise ValueError(
                            f"No index benchmark rows for portfolio `{portfolio}` and type `{benchmark_type}`."
                        )
                    series_map[bm_col] = bm_series.rename(bm_col)
                    ordered_cols.append(bm_col)
                benchmark_assignments[portfolio_col] = bm_col
            else:
                if benchmark_type != INDEX_BENCHMARK_DESC:
                    raise ValueError(
                        f"Unsupported index benchmark type `{benchmark_type}` for portfolio `{portfolio}`."
                    )
                bm_col = f"{portfolio}{INDEX_BENCHMARK_SUFFIX}"
                if bm_col not in series_map:
                    bm_series = _read_series(engine, "IndexTS", portfolio, "PortRet", INDEX_BENCHMARK_DESC)
                    if bm_series.empty:
                        raise ValueError(f"No index benchmark rows for portfolio `{portfolio}`.")
                    series_map[bm_col] = bm_series.rename(bm_col)
                    ordered_cols.append(bm_col)
                benchmark_assignments[portfolio_col] = bm_col
        else:
            source = str(metadata[portfolio].get("portfolio_vintage", "")).strip()
            bm_col = f"{portfolio}{INDEX_BENCHMARK_SUFFIX}"
            if bm_col not in series_map:
                if source == "AltTS":
                    bm_series = _read_series_no_desc(
                        engine,
                        "AltTS",
                        portfolio,
                        "BenchRet",
                        normalize_values=False,
                    )
                    if bm_series.empty:
                        raise ValueError(f"No AltTS benchmark rows for portfolio `{portfolio}`.")
                elif source == "Perf":
                    vintage = str(metadata[portfolio].get("peer_vintage", "")).strip()
                    if not vintage:
                        raise ValueError(f"PeerVintage is missing for portfolio `{portfolio}`.")
                    bm_series = _read_perf_series(perf_engine, portfolio, "Daily_ror_index")
                    if bm_series.empty:
                        raise ValueError(f"No Perf benchmark rows for portfolio `{portfolio}`.")
                else:
                    if source in OTHER_SUPPORTED_SOURCES:
                        raise ValueError(f"No benchmark source rows for portfolio `{portfolio}`.")
                    raise ValueError(
                        f"PortfolioVintage `{source}` is not implemented for portfolio `{portfolio}`."
                    )
                series_map[bm_col] = bm_series.rename(bm_col)
                ordered_cols.append(bm_col)
            benchmark_assignments[portfolio_col] = bm_col

    if not ordered_cols:
        return PortfolioImportResult(pd.DataFrame(), {}, "monthly")

    out = pd.concat([series_map[c] for c in ordered_cols], axis=1)
    out = out.sort_index().dropna(how="all")
    out.index.name = "Date"
    periodicity = _infer_periodicity(out)
    return PortfolioImportResult(out, benchmark_assignments, periodicity)
