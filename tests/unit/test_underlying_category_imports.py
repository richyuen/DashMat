from __future__ import annotations

from datetime import date

import numpy as np
from sqlalchemy import create_engine, text

from utils.underlying_category_imports import (
    build_underlying_portfolio_codes,
    expand_underlying_category_rows,
    get_underlying_category_desc_options,
    load_underlying_category_series,
)


def _seed_underlying_engine():
    engine = create_engine("sqlite:///:memory:", future=True)
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE PeerTS ("
                "Date DATE NOT NULL, "
                "Portfolio TEXT NOT NULL, "
                "Item TEXT NOT NULL, "
                "Desc TEXT NOT NULL, "
                "Value REAL NOT NULL)"
            )
        )

        levels_by_pair = {
            ("CoreTD", "Large Cap"): [100.0, 110.0, 121.0],
            ("CoreTD", "Small Cap"): [200.0, 210.0, 231.0],
            ("CoreAlloc", "Large Cap"): [50.0, 55.0, 60.5],
            ("CoreAlloc", "Intl Equity"): [80.0, 84.0, 88.2],
            ("Base529", "Large Cap"): [300.0, 315.0, 330.75],
            ("Base529", "College Bond"): [400.0, 420.0, 441.0],
            ("BaseModel", "Growth"): [120.0, 126.0, 132.3],
        }
        dates = [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)]
        for (portfolio, desc), levels in levels_by_pair.items():
            for dt, value in zip(dates, levels):
                conn.execute(
                    text(
                        "INSERT INTO PeerTS (Date, Portfolio, Item, Desc, Value) "
                        "VALUES (:d, :portfolio, 'PeerRet', :desc, :value)"
                    ),
                    {
                        "d": dt,
                        "portfolio": portfolio,
                        "desc": desc,
                        "value": value,
                    },
                )
    return engine


def test_build_underlying_portfolio_codes_preserves_fixed_type_order():
    assert build_underlying_portfolio_codes("Core", ["529", "TD", "Bogus", "TD"]) == ["CoreTD", "Core529"]
    assert build_underlying_portfolio_codes("Base", ["Model", "Alloc"]) == ["BaseAlloc", "BaseModel"]
    assert build_underlying_portfolio_codes(None, ["TD"]) == []


def test_get_underlying_category_desc_options_filters_by_base_and_type_union():
    engine = _seed_underlying_engine()

    core_options = get_underlying_category_desc_options(engine, "Core", ["TD", "Alloc"])
    assert [opt["value"] for opt in core_options] == ["Intl Equity", "Large Cap", "Small Cap"]

    base_options = get_underlying_category_desc_options(engine, "Base", ["529"])
    assert [opt["value"] for opt in base_options] == ["College Bond", "Large Cap"]


def test_expand_underlying_category_rows_creates_one_row_per_valid_pair():
    engine = _seed_underlying_engine()

    rows = expand_underlying_category_rows(
        engine,
        "Core",
        ["Alloc", "TD"],
        ["Large Cap", "Intl Equity", "Missing"],
    )

    assert [row["Series"] for row in rows] == [
        "Large Cap [CoreTD]",
        "Large Cap [CoreAlloc]",
        "Intl Equity [CoreAlloc]",
    ]
    assert [row["Portfolio"] for row in rows] == ["CoreTD", "CoreAlloc", "CoreAlloc"]


def test_load_underlying_category_series_converts_levels_to_daily_returns():
    engine = _seed_underlying_engine()

    result = load_underlying_category_series(
        engine,
        [
            {"portfolio": "CoreTD", "desc": "Large Cap", "series_name": "Large Cap [CoreTD]"},
            {"portfolio": "Base529", "desc": "College Bond", "series_name": "College Bond [Base529]"},
        ],
    )

    assert result.periodicity == "daily"
    df = result.returns_df
    assert list(df.columns) == ["Large Cap [CoreTD]", "College Bond [Base529]"]
    assert np.allclose(df["Large Cap [CoreTD]"].to_numpy(), [0.10, 0.10])
    assert np.allclose(df["College Bond [Base529]"].to_numpy(), [0.05, 0.05])


def test_load_underlying_category_series_raises_for_missing_pair():
    engine = _seed_underlying_engine()

    try:
        load_underlying_category_series(
            engine,
            [{"portfolio": "CoreTD", "desc": "Does Not Exist", "series_name": "Does Not Exist [CoreTD]"}],
        )
    except ValueError as exc:
        assert "No underlying category rows found" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing underlying category pair")
