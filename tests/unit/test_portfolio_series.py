from __future__ import annotations

from datetime import date

import pandas as pd
from sqlalchemy import create_engine, text

from utils.constants import INDEX_BENCHMARK_SUFFIX
from utils.portfolio_series import get_portfolio_options, load_portfolio_series


def _seed_db():
    engine = create_engine("sqlite:///:memory:", future=True)
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE Suites ("
                "SuiteID INTEGER PRIMARY KEY, "
                "SuiteShort TEXT UNIQUE NOT NULL, "
                "SuiteLong TEXT NOT NULL, "
                "IndexMonthlyOrder INTEGER, "
                "PeerTDOrder INTEGER, "
                "PeerModelOrder INTEGER, "
                "PeerAllocOrder INTEGER, "
                "Peer529Order INTEGER)"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE Portfolios ("
                "PortfolioID INTEGER PRIMARY KEY, "
                "PortfolioName TEXT NOT NULL, "
                "Portfolio TEXT UNIQUE NOT NULL, "
                "PortfolioSuite TEXT NOT NULL, "
                "PeerVintage TEXT, "
                "IncepDate DATE)"
            )
        )
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
        conn.execute(
            text(
                "CREATE TABLE IndexTS ("
                "Date DATE NOT NULL, "
                "Portfolio TEXT NOT NULL, "
                "Item TEXT NOT NULL, "
                "Desc TEXT NOT NULL, "
                "Value REAL NOT NULL)"
            )
        )

        conn.execute(
            text(
                "INSERT INTO Suites "
                "(SuiteID, SuiteShort, SuiteLong, IndexMonthlyOrder, PeerTDOrder, PeerModelOrder, PeerAllocOrder, Peer529Order) "
                "VALUES "
                "(1, 'TD', 'Target Date', NULL, 1, NULL, NULL, NULL), "
                "(2, 'RISK', 'Risk Based', 1, NULL, NULL, NULL, NULL)"
            )
        )
        conn.execute(
            text(
                "INSERT INTO Portfolios "
                "(PortfolioID, PortfolioName, Portfolio, PortfolioSuite, PeerVintage, IncepDate) "
                "VALUES "
                "(1, 'TDF 2030 A', 'TD2030A', 'TD', '2030', '2020-02-29'), "
                "(2, 'TDF 2030 B', 'TD2030B', 'TD', '2030', '2020-01-31'), "
                "(3, 'Risk 60', 'Risk60', 'RISK', '', '2020-01-31')"
            )
        )

        dates = [date(2020, 1, 31), date(2020, 2, 29), date(2020, 3, 31)]
        for dt in dates:
            conn.execute(
                text("INSERT INTO PeerTS (Date, Portfolio, Item, Desc, Value) VALUES (:d, 'TD2030A', 'PortRet', 'Actual', 0.01)"),
                {"d": dt},
            )
            conn.execute(
                text("INSERT INTO PeerTS (Date, Portfolio, Item, Desc, Value) VALUES (:d, 'TD2030B', 'PortRet', 'Actual', 0.011)"),
                {"d": dt},
            )
            conn.execute(
                text("INSERT INTO PeerTS (Date, Portfolio, Item, Desc, Value) VALUES (:d, '2030', 'MeanRet', 'Estimated', 0.008)"),
                {"d": dt},
            )
            conn.execute(
                text("INSERT INTO PeerTS (Date, Portfolio, Item, Desc, Value) VALUES (:d, 'TD2030A', 'MeanRet', 'Calculated', 0.0075)"),
                {"d": dt},
            )
            conn.execute(
                text("INSERT INTO PeerTS (Date, Portfolio, Item, Desc, Value) VALUES (:d, 'TD2030B', 'MeanRet', 'Calculated', 0.0079)"),
                {"d": dt},
            )
            conn.execute(
                text("INSERT INTO IndexTS (Date, Portfolio, Item, Desc, Value) VALUES (:d, 'Risk60', 'PortRet', 'Actual', 0.012)"),
                {"d": dt},
            )
            conn.execute(
                text("INSERT INTO IndexTS (Date, Portfolio, Item, Desc, Value) VALUES (:d, 'Risk60', 'PortRet', 'Benchmark', 0.009)"),
                {"d": dt},
            )
    return engine


def test_get_portfolio_options_labels_and_mode_filters():
    engine = _seed_db()

    peer_options = get_portfolio_options(engine, "peer")
    assert {"value": "TD2030A", "label": "TDF 2030 A [TD2030A]"} in peer_options
    assert {"value": "TD2030B", "label": "TDF 2030 B [TD2030B]"} in peer_options
    assert not any(opt["value"] == "Risk60" for opt in peer_options)

    index_options = get_portfolio_options(engine, "index")
    assert {"value": "Risk60", "label": "Risk 60 [Risk60]"} in index_options
    assert not any(opt["value"] == "TD2030A" for opt in index_options)


def test_load_portfolio_series_peer_dedup_and_incep_cutoff():
    engine = _seed_db()
    result = load_portfolio_series(
        engine,
        "peer",
        [
            {"portfolio": "TD2030A", "type": "Actual", "include_benchmark": True, "benchmark_type": "Estimated"},
            {"portfolio": "TD2030B", "type": "Actual", "include_benchmark": True, "benchmark_type": "Estimated"},
        ],
    )

    assert list(result.returns_df.columns) == ["TD2030A", "2030", "TD2030B"]
    assert result.benchmark_assignments == {"TD2030A": "2030", "TD2030B": "2030"}
    assert pd.Timestamp("2020-01-31") not in result.returns_df["TD2030A"].dropna().index
    assert pd.Timestamp("2020-01-31") in result.returns_df["2030"].dropna().index
    assert result.periodicity == "monthly"


def test_load_portfolio_series_index_benchmark_suffix():
    engine = _seed_db()
    result = load_portfolio_series(
        engine,
        "index",
        [
            {"portfolio": "Risk60", "type": "Actual", "include_benchmark": True, "benchmark_type": "Benchmark"},
        ],
    )

    bm_name = f"Risk60{INDEX_BENCHMARK_SUFFIX}"
    assert list(result.returns_df.columns) == ["Risk60", bm_name]
    assert result.benchmark_assignments == {"Risk60": bm_name}
    assert result.periodicity == "monthly"


def test_load_portfolio_series_peer_calculated_benchmark_uses_portfolio_key():
    engine = _seed_db()
    result = load_portfolio_series(
        engine,
        "peer",
        [
            {"portfolio": "TD2030A", "type": "Actual", "include_benchmark": True, "benchmark_type": "Calculated"},
        ],
    )

    bm_name = "TD2030A_Calculated"
    assert list(result.returns_df.columns) == ["TD2030A", bm_name]
    assert result.benchmark_assignments == {"TD2030A": bm_name}
