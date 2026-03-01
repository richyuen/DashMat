"""Helpers for importing underlying category returns from PeerTS."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine

import cache_config


UNDERLYING_CATEGORY_ITEM = "PeerRet"
UNDERLYING_CATEGORY_BASE_OPTIONS = (
    {"value": "Core", "label": "Core"},
    {"value": "Base", "label": "Base"},
)
UNDERLYING_CATEGORY_TYPE_OPTIONS = (
    {"value": "TD", "label": "TD"},
    {"value": "Alloc", "label": "Alloc"},
    {"value": "529", "label": "529"},
    {"value": "Model", "label": "Model"},
)
UNDERLYING_CATEGORY_TYPE_ORDER = tuple(opt["value"] for opt in UNDERLYING_CATEGORY_TYPE_OPTIONS)


@dataclass(frozen=True)
class UnderlyingCategoryImportResult:
    returns_df: pd.DataFrame
    periodicity: str


def _normalize_base_value(base_value: str | None) -> str:
    value = str(base_value or "").strip()
    return value if value in {"Core", "Base"} else ""


def _normalize_type_values(type_values: list[str] | tuple[str, ...] | None) -> list[str]:
    selected = {str(value or "").strip() for value in (type_values or []) if str(value or "").strip()}
    return [value for value in UNDERLYING_CATEGORY_TYPE_ORDER if value in selected]


def _normalize_desc_values(desc_values: list[str] | tuple[str, ...] | None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in desc_values or []:
        desc = str(value or "").strip()
        if not desc or desc in seen:
            continue
        seen.add(desc)
        out.append(desc)
    return out


def build_underlying_portfolio_codes(base_value: str | None, type_values: list[str] | tuple[str, ...] | None) -> list[str]:
    base = _normalize_base_value(base_value)
    if not base:
        return []
    return [f"{base}{type_value}" for type_value in _normalize_type_values(type_values)]


def build_underlying_series_name(desc: str, portfolio: str) -> str:
    return f"{str(desc).strip()} [{str(portfolio).strip()}]"


@cache_config.cache.memoize(timeout=0)
def get_underlying_category_meta_cached(engine: Engine) -> dict[str, object]:
    q = text(
        "SELECT DISTINCT Portfolio, [Desc] "
        "FROM PeerTS "
        "WHERE Item = :item "
        "ORDER BY Portfolio, [Desc]"
    )
    with engine.connect() as conn:
        rows = conn.execute(q, {"item": UNDERLYING_CATEGORY_ITEM}).fetchall()

    portfolio_to_descs: dict[str, list[str]] = {}
    valid_pairs: set[tuple[str, str]] = set()
    for portfolio, desc in rows:
        portfolio_key = str(portfolio or "").strip()
        desc_key = str(desc or "").strip()
        if not portfolio_key or not desc_key:
            continue
        portfolio_to_descs.setdefault(portfolio_key, []).append(desc_key)
        valid_pairs.add((portfolio_key, desc_key))

    normalized_portfolio_to_descs = {
        portfolio: tuple(sorted(set(descs), key=str.casefold))
        for portfolio, descs in portfolio_to_descs.items()
    }
    return {
        "portfolio_to_descs": normalized_portfolio_to_descs,
        "valid_pairs": frozenset(valid_pairs),
    }


def get_underlying_category_desc_options(
    engine: Engine,
    base_value: str | None,
    type_values: list[str] | tuple[str, ...] | None,
) -> list[dict[str, str]]:
    metadata = get_underlying_category_meta_cached(engine)
    portfolio_to_descs = metadata.get("portfolio_to_descs", {})
    descs: set[str] = set()
    for portfolio in build_underlying_portfolio_codes(base_value, type_values):
        descs.update(portfolio_to_descs.get(portfolio, ()))
    return [{"value": desc, "label": desc} for desc in sorted(descs, key=str.casefold)]


def expand_underlying_category_rows(
    engine: Engine,
    base_value: str | None,
    type_values: list[str] | tuple[str, ...] | None,
    desc_values: list[str] | tuple[str, ...] | None,
) -> list[dict[str, str]]:
    metadata = get_underlying_category_meta_cached(engine)
    valid_pairs = metadata.get("valid_pairs", frozenset())
    portfolios = build_underlying_portfolio_codes(base_value, type_values)
    descs = _normalize_desc_values(desc_values)

    out: list[dict[str, str]] = []
    for desc in descs:
        for portfolio in portfolios:
            if (portfolio, desc) not in valid_pairs:
                continue
            series_name = build_underlying_series_name(desc, portfolio)
            out.append(
                {
                    "Series": series_name,
                    "Portfolio": portfolio,
                    "Desc": desc,
                    "series_name": series_name,
                    "portfolio": portfolio,
                    "desc": desc,
                }
            )
    return out


def _rows_to_return_series(rows: list[tuple[object, object]], series_name: str) -> pd.Series:
    if not rows:
        return pd.Series(dtype=float, name=series_name)

    frame = pd.DataFrame(rows, columns=["Date", "Value"])
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    series = pd.Series(
        pd.to_numeric(frame["Value"], errors="coerce").values,
        index=pd.DatetimeIndex(frame["Date"]),
        dtype=float,
        name=series_name,
    ).sort_index()
    series = series.loc[~pd.isna(series.index)]
    series = series[~series.index.duplicated(keep="last")].dropna()
    if series.empty:
        return pd.Series(dtype=float, name=series_name)
    return series.pct_change(fill_method=None).dropna().rename(series_name)


def load_underlying_category_series(
    engine: Engine,
    staged_rows: list[dict] | None,
) -> UnderlyingCategoryImportResult:
    rows = [dict(row) for row in (staged_rows or []) if isinstance(row, dict)]
    requested: list[tuple[str, str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for row in rows:
        portfolio = str(row.get("portfolio") or row.get("Portfolio") or "").strip()
        desc = str(row.get("desc") or row.get("Desc") or "").strip()
        if not portfolio or not desc:
            continue
        pair = (portfolio, desc)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        series_name = str(row.get("series_name") or row.get("Series") or build_underlying_series_name(desc, portfolio)).strip()
        requested.append((portfolio, desc, series_name))

    if not requested:
        return UnderlyingCategoryImportResult(pd.DataFrame(), "daily")

    portfolios = sorted({portfolio for portfolio, _, _ in requested})
    descs = sorted({desc for _, desc, _ in requested})
    q = text(
        "SELECT Date, Portfolio, [Desc], Value "
        "FROM PeerTS "
        "WHERE Item = :item "
        "AND Portfolio IN :portfolios "
        "AND [Desc] IN :descs "
        "ORDER BY Portfolio, [Desc], Date"
    ).bindparams(
        bindparam("portfolios", expanding=True),
        bindparam("descs", expanding=True),
    )
    with engine.connect() as conn:
        source_rows = conn.execute(
            q,
            {
                "item": UNDERLYING_CATEGORY_ITEM,
                "portfolios": portfolios,
                "descs": descs,
            },
        ).fetchall()

    grouped_rows: dict[tuple[str, str], list[tuple[object, object]]] = {}
    requested_pairs = {(portfolio, desc) for portfolio, desc, _ in requested}
    for date_value, portfolio, desc, value in source_rows:
        pair = (str(portfolio or "").strip(), str(desc or "").strip())
        if pair not in requested_pairs:
            continue
        grouped_rows.setdefault(pair, []).append((date_value, value))

    series_map: dict[str, pd.Series] = {}
    ordered_names: list[str] = []
    for portfolio, desc, series_name in requested:
        rows_for_pair = grouped_rows.get((portfolio, desc), [])
        if not rows_for_pair:
            raise ValueError(f"No underlying category rows found for `{desc}` in `{portfolio}`.")

        series = _rows_to_return_series(rows_for_pair, series_name)
        if series.empty:
            raise ValueError(f"No return observations were produced for `{desc}` in `{portfolio}`.")

        series_map[series_name] = series.rename(series_name)
        ordered_names.append(series_name)

    out = pd.concat([series_map[name] for name in ordered_names], axis=1)
    out = out.sort_index().dropna(how="all")
    out.index.name = "Date"
    return UnderlyingCategoryImportResult(out, "daily")
