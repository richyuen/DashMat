from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine, text

from tools.db.backfill_underlying_category_peerts import backfill_underlying_category_peerts
from tools.db.init_local_cma_db import (
    UNDERLYING_CATEGORY_ITEM,
    UNDERLYING_CATEGORY_PORTFOLIO_DESCS,
    _build_account_list_seed_rows,
    build_underlying_category_seed_rows,
)


def _sample_daily_df() -> pd.DataFrame:
    idx = pd.date_range("2024-01-02", periods=5, freq="B")
    return pd.DataFrame(
        {
            "SPX": [0.0100, -0.0040, 0.0060, 0.0020, -0.0030],
            "RMID": [0.0090, -0.0030, 0.0050, 0.0010, -0.0020],
            "R2000": [0.0120, -0.0060, 0.0080, 0.0030, -0.0040],
            "EAFE": [0.0070, -0.0020, 0.0040, 0.0010, -0.0010],
            "EM": [0.0080, -0.0040, 0.0060, 0.0020, -0.0020],
            "MSCIUSREIT": [0.0060, -0.0010, 0.0030, 0.0010, -0.0010],
            "BCAgg": [0.0010, 0.0000, 0.0010, 0.0000, 0.0010],
            "BCHY": [0.0040, -0.0010, 0.0030, 0.0010, -0.0010],
            "BCGAgg": [0.0010, 0.0000, 0.0010, 0.0000, 0.0010],
            "BCGC13": [0.0004, 0.0004, 0.0004, 0.0004, 0.0004],
        },
        index=idx,
    )


def test_build_underlying_category_seed_rows_cover_expected_portfolios_and_descs():
    rows = build_underlying_category_seed_rows(_sample_daily_df())

    assert rows
    assert {row["Item"] for row in rows} == {UNDERLYING_CATEGORY_ITEM}
    assert {row["Portfolio"] for row in rows} == set(UNDERLYING_CATEGORY_PORTFOLIO_DESCS)

    pairs = {(row["Portfolio"], row["Desc"]) for row in rows}
    assert ("CoreTD", "High Yield") in pairs
    assert ("Base529", "Core Bond") in pairs
    assert ("Base529", "Real Assets") not in pairs


def test_build_underlying_category_seed_rows_are_deterministic_levels():
    first = build_underlying_category_seed_rows(_sample_daily_df())
    second = build_underlying_category_seed_rows(_sample_daily_df())

    assert first == second

    first_large_cap = next(
        row for row in first if row["Portfolio"] == "CoreTD" and row["Desc"] == "Large Cap"
    )
    assert first_large_cap["Value"] > 90.0


def test_backfill_underlying_category_peerts_is_idempotent():
    engine = create_engine("sqlite:///:memory:", future=True)
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE PeerTS ("
                "Date DATE NOT NULL, "
                "Portfolio TEXT NOT NULL, "
                "Item TEXT NOT NULL, "
                "Desc TEXT NOT NULL, "
                "Value REAL NOT NULL, "
                "PRIMARY KEY (Date, Portfolio, Item, Desc))"
            )
        )

    daily_df = _sample_daily_df()
    first = backfill_underlying_category_peerts(engine, daily_df=daily_df)
    second = backfill_underlying_category_peerts(engine, daily_df=daily_df)

    assert first["generated"] > 0
    assert first["inserted"] == first["generated"]
    assert first["existing"] == 0
    assert second["inserted"] == 0
    assert second["existing"] == second["generated"]

    with engine.connect() as conn:
        row_count = conn.execute(
            text("SELECT COUNT(*) FROM PeerTS WHERE Item = :item"),
            {"item": UNDERLYING_CATEGORY_ITEM},
        ).scalar_one()

    assert int(row_count) == first["generated"]


def test_account_list_seed_rows_use_control_values_schema():
    rows = _build_account_list_seed_rows()

    assert rows

    for row in rows:
        payload = row["ConfigJson"]
        assert '"schema_version":2' in payload
        assert '"control_values"' in payload
        assert '"settings"' not in payload
        assert '"at-series-select"' in payload
        assert '"po-series-select"' in payload
        assert '"reg-series-select"' in payload
