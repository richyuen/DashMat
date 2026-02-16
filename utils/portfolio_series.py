"""CMA portfolio-series helpers for peer/index import workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine

from utils.constants import (
    INDEX_BENCHMARK_DESC,
    INDEX_BENCHMARK_SUFFIX,
    PEER_BENCHMARK_TYPE_OPTIONS,
    PEER_PORTFOLIO_TYPE_OPTIONS,
)

PortfolioMode = Literal["peer", "index"]


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


@dataclass(frozen=True)
class PortfolioImportResult:
    returns_df: pd.DataFrame
    benchmark_assignments: dict[str, str]
    periodicity: str


def get_portfolio_options(engine: Engine, mode: PortfolioMode) -> list[dict]:
    """Return dropdown options in `PortfolioName [Portfolio]` format."""
    if mode not in {"peer", "index"}:
        return []

    if mode == "peer":
        where_clause = (
            "COALESCE(s.PeerTDOrder, 0) > 0 "
            "OR COALESCE(s.PeerModelOrder, 0) > 0 "
            "OR COALESCE(s.PeerAllocOrder, 0) > 0 "
            "OR COALESCE(s.Peer529Order, 0) > 0"
        )
        order_clause = (
            "COALESCE(s.PeerTDOrder, s.PeerModelOrder, s.PeerAllocOrder, s.Peer529Order, 999999), "
            "p.Portfolio"
        )
    else:
        where_clause = "COALESCE(s.IndexMonthlyOrder, 0) > 0"
        order_clause = "COALESCE(s.IndexMonthlyOrder, 999999), p.Portfolio"

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


def _read_series(
    engine: Engine,
    table_name: str,
    portfolio: str,
    item: str,
    desc: str,
) -> pd.Series:
    q = text(
        f"SELECT Date, Value "
        f"FROM {table_name} "
        "WHERE Portfolio = :portfolio "
        "AND Item = :item "
        "AND Desc = :desc "
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

    if not rows:
        return pd.Series(dtype=float, name=portfolio)

    df = pd.DataFrame(rows, columns=["Date", "Value"])
    df["Date"] = pd.to_datetime(df["Date"])
    series = pd.Series(
        pd.to_numeric(df["Value"], errors="coerce").values,
        index=pd.DatetimeIndex(df["Date"]),
        name=portfolio,
        dtype=float,
    ).sort_index()
    series = series[~series.index.duplicated(keep="last")]
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
) -> PortfolioImportResult:
    """Load staged portfolio requests from CMA tables."""
    rows = [r for r in (staged_rows or []) if isinstance(r, dict)]
    if mode not in {"peer", "index"} or not rows:
        return PortfolioImportResult(pd.DataFrame(), {}, "monthly")

    requested = [str(r.get("portfolio", "")).strip() for r in rows if r.get("portfolio")]
    if not requested:
        return PortfolioImportResult(pd.DataFrame(), {}, "monthly")

    portfolio_query = text(
        "SELECT Portfolio, PortfolioName, PeerVintage, IncepDate "
        "FROM Portfolios "
        "WHERE Portfolio IN :portfolios"
    ).bindparams(bindparam("portfolios", expanding=True))

    with engine.connect() as conn:
        meta_rows = conn.execute(
            portfolio_query,
            {"portfolios": sorted(set(requested))},
        ).fetchall()

    metadata: dict[str, dict[str, object]] = {}
    for portfolio, portfolio_name, peer_vintage, incep_date in meta_rows:
        p = str(portfolio or "").strip()
        if not p:
            continue
        metadata[p] = {
            "portfolio_name": str(portfolio_name or p),
            "peer_vintage": str(peer_vintage).strip() if peer_vintage is not None else "",
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

        if mode == "peer":
            port_series = _read_series(engine, "PeerTS", portfolio, "PortRet", ret_desc)
        else:
            port_series = _read_series(engine, "IndexTS", portfolio, "PortRet", ret_desc)

        incep = metadata[portfolio].get("incep_date")
        if incep is not None:
            incep_ts = pd.Timestamp(incep)
            port_series = port_series.loc[port_series.index >= incep_ts]

        if port_series.empty:
            raise ValueError(f"No rows found for portfolio `{portfolio}` with type `{ret_desc}`.")

        if portfolio not in series_map:
            series_map[portfolio] = port_series.rename(portfolio)
            ordered_cols.append(portfolio)

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
                # use portfolio-specific benchmark lookup in PeerTS.
                bm_col = f"{portfolio}_{benchmark_type}"
                if bm_col not in series_map:
                    bm_series = _read_series(engine, "PeerTS", portfolio, "MeanRet", benchmark_type)
                    if bm_series.empty:
                        raise ValueError(
                            f"No peer benchmark rows for portfolio `{portfolio}` and type `{benchmark_type}`."
                        )
                    series_map[bm_col] = bm_series.rename(bm_col)
                    ordered_cols.append(bm_col)
                benchmark_assignments[portfolio] = bm_col
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
                benchmark_assignments[portfolio] = vintage
        else:
            bm_col = f"{portfolio}{INDEX_BENCHMARK_SUFFIX}"
            if bm_col not in series_map:
                bm_series = _read_series(engine, "IndexTS", portfolio, "PortRet", INDEX_BENCHMARK_DESC)
                if bm_series.empty:
                    raise ValueError(f"No index benchmark rows for portfolio `{portfolio}`.")
                series_map[bm_col] = bm_series.rename(bm_col)
                ordered_cols.append(bm_col)
            benchmark_assignments[portfolio] = bm_col

    if not ordered_cols:
        return PortfolioImportResult(pd.DataFrame(), {}, "monthly")

    out = pd.concat([series_map[c] for c in ordered_cols], axis=1)
    out = out.sort_index().dropna(how="all")
    out.index.name = "Date"
    periodicity = _infer_periodicity(out)
    return PortfolioImportResult(out, benchmark_assignments, periodicity)
