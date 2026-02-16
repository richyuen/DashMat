from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from utils.constants import INDEX_BENCHMARK_SUFFIX
from utils.portfolio_series import get_portfolio_options, has_portfolio_benchmark, load_portfolio_series


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
                "PortfolioVintage TEXT, "
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
                "CREATE TABLE AltTS ("
                "Date DATE NOT NULL, "
                "Portfolio TEXT NOT NULL, "
                "Item TEXT NOT NULL, "
                "Value REAL NOT NULL)"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE [ACCOUNT] ("
                "ACCT_ID INTEGER PRIMARY KEY, "
                "ACCT_CD TEXT NOT NULL)"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE [ACCOUNT_BENCHMARK] ("
                "BENCHMARK_ID INTEGER PRIMARY KEY, "
                "PRECEDENCE INTEGER NOT NULL)"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE [DAILY_RETURN] ("
                "Effective_Date DATE NOT NULL, "
                "Daily_ror REAL NOT NULL, "
                "Daily_ror_index REAL NOT NULL, "
                "ACCT_ID INTEGER NOT NULL, "
                "BENCHMARK_ACCT_ID INTEGER NOT NULL, "
                "FEE_TYPE TEXT NOT NULL, "
                "IS_LATEST INTEGER NOT NULL)"
            )
        )

        conn.execute(
            text(
                "INSERT INTO Suites "
                "(SuiteID, SuiteShort, SuiteLong, IndexMonthlyOrder, PeerTDOrder, PeerModelOrder, PeerAllocOrder, Peer529Order) "
                "VALUES "
                "(1, 'TD', 'Target Date', NULL, 1, NULL, NULL, NULL), "
                "(2, 'RISK', 'Risk Based', 1, NULL, NULL, NULL, NULL), "
                "(3, 'IndNoAttr', 'Index No Attribution', NULL, NULL, NULL, NULL, NULL)"
            )
        )
        conn.execute(
            text(
                "INSERT INTO Portfolios "
                "(PortfolioID, PortfolioName, Portfolio, PortfolioSuite, PeerVintage, PortfolioVintage, IncepDate) "
                "VALUES "
                "(1, 'TDF 2030 A', 'TD2030A', 'TD', '2030', '', '2020-02-29'), "
                "(2, 'TDF 2030 B', 'TD2030B', 'TD', '2030', '', '2020-01-31'), "
                "(3, 'Risk 60', 'Risk60', 'RISK', '', '', '2020-01-31'), "
                "(4, 'Alternative Trend', 'ALTTRN', 'IndNoAttr', '', 'AltTS', '2020-01-31'), "
                "(5, 'Alternative Macro', 'ALTMAC', 'IndNoAttr', '', 'AltTS', '2020-01-31'), "
                "(6, 'Alternative No Benchmark', 'ALTNOBM', 'IndNoAttr', '', 'AltTS', '2020-01-31'), "
                "(7, 'Performance Trend', 'PERFTRN', 'IndNoAttr', '2030', 'Perf', '2020-01-31'), "
                "(8, 'Performance No Benchmark', 'PERFNOBM', 'IndNoAttr', '', 'Perf', '2020-01-31')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO [ACCOUNT] (ACCT_ID, ACCT_CD) VALUES "
                "(101, 'PERFTRN'), "
                "(102, 'PERFNOBM')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO [ACCOUNT_BENCHMARK] (BENCHMARK_ID, PRECEDENCE) VALUES "
                "(901, 1), "
                "(902, 2)"
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
            conn.execute(
                text("INSERT INTO AltTS (Date, Portfolio, Item, Value) VALUES (:d, 'ALTTRN', 'PortRet', :v)"),
                {"d": dt, "v": 1.20},
            )
            conn.execute(
                text("INSERT INTO AltTS (Date, Portfolio, Item, Value) VALUES (:d, 'ALTTRN', 'BenchRet', :v)"),
                {"d": dt, "v": 0.90},
            )
            conn.execute(
                text("INSERT INTO AltTS (Date, Portfolio, Item, Value) VALUES (:d, 'ALTMAC', 'PortRet', :v)"),
                {"d": dt, "v": 0.015},
            )
            conn.execute(
                text("INSERT INTO AltTS (Date, Portfolio, Item, Value) VALUES (:d, 'ALTMAC', 'BenchRet', :v)"),
                {"d": dt, "v": 0.010},
            )
            conn.execute(
                text("INSERT INTO AltTS (Date, Portfolio, Item, Value) VALUES (:d, 'ALTNOBM', 'PortRet', :v)"),
                {"d": dt, "v": 0.007},
            )

            # Perf rows: only (PRECEDENCE=1, IS_LATEST=1, FEE_TYPE='G') should survive.
            conn.execute(
                text(
                    "INSERT INTO [DAILY_RETURN] "
                    "(Effective_Date, Daily_ror, Daily_ror_index, ACCT_ID, BENCHMARK_ACCT_ID, FEE_TYPE, IS_LATEST) "
                    "VALUES (:d, 1.50, 1.10, 101, 901, 'G', 1)"
                ),
                {"d": dt},
            )
            conn.execute(
                text(
                    "INSERT INTO [DAILY_RETURN] "
                    "(Effective_Date, Daily_ror, Daily_ror_index, ACCT_ID, BENCHMARK_ACCT_ID, FEE_TYPE, IS_LATEST) "
                    "VALUES (:d, 1.80, 1.30, 101, 902, 'G', 1)"
                ),
                {"d": dt},
            )
            conn.execute(
                text(
                    "INSERT INTO [DAILY_RETURN] "
                    "(Effective_Date, Daily_ror, Daily_ror_index, ACCT_ID, BENCHMARK_ACCT_ID, FEE_TYPE, IS_LATEST) "
                    "VALUES (:d, 1.90, 1.40, 101, 901, 'N', 1)"
                ),
                {"d": dt},
            )
            conn.execute(
                text(
                    "INSERT INTO [DAILY_RETURN] "
                    "(Effective_Date, Daily_ror, Daily_ror_index, ACCT_ID, BENCHMARK_ACCT_ID, FEE_TYPE, IS_LATEST) "
                    "VALUES (:d, 2.10, 1.60, 101, 901, 'G', 0)"
                ),
                {"d": dt},
            )
            conn.execute(
                text(
                    "INSERT INTO [DAILY_RETURN] "
                    "(Effective_Date, Daily_ror, Daily_ror_index, ACCT_ID, BENCHMARK_ACCT_ID, FEE_TYPE, IS_LATEST) "
                    "VALUES (:d, 0.60, 0.40, 102, 901, 'G', 1)"
                ),
                {"d": dt},
            )
    return engine


def test_get_portfolio_options_labels_and_mode_filters():
    engine = _seed_db()

    peer_options = get_portfolio_options(engine, "peer")
    assert [opt["value"] for opt in peer_options] == ["TD2030A", "TD2030B"]
    assert {"value": "TD2030A", "label": "TDF 2030 A [TD2030A]"} in peer_options
    assert {"value": "TD2030B", "label": "TDF 2030 B [TD2030B]"} in peer_options
    assert not any(opt["value"] == "Risk60" for opt in peer_options)

    index_options = get_portfolio_options(engine, "index")
    assert {"value": "Risk60", "label": "Risk 60 [Risk60]"} in index_options
    assert not any(opt["value"] == "TD2030A" for opt in index_options)

    other_options = get_portfolio_options(engine, "other")
    assert [opt["value"] for opt in other_options] == ["ALTTRN", "ALTMAC", "ALTNOBM", "PERFTRN", "PERFNOBM"]
    assert {"value": "ALTTRN", "label": "Alternative Trend [ALTTRN]"} in other_options
    assert {"value": "PERFTRN", "label": "Performance Trend [PERFTRN]"} in other_options
    assert not any(opt["value"] == "Risk60" for opt in other_options)
    assert {"value": "PERFNOBM", "label": "Performance No Benchmark [PERFNOBM]"} in other_options


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


def test_load_portfolio_series_other_uses_altts_port_and_bench_items():
    engine = _seed_db()
    result = load_portfolio_series(
        engine,
        "other",
        [
            {"portfolio": "ALTTRN", "type": "Actual", "include_benchmark": True, "benchmark_type": "Actual"},
        ],
    )

    bm_name = f"ALTTRN{INDEX_BENCHMARK_SUFFIX}"
    assert list(result.returns_df.columns) == ["ALTTRN", bm_name]
    assert result.benchmark_assignments == {"ALTTRN": bm_name}
    # AltTS values are returns, so they should not be converted via pct_change.
    assert (result.returns_df["ALTTRN"].dropna() == 1.20).all()
    assert (result.returns_df[bm_name].dropna() == 0.90).all()


def test_load_portfolio_series_other_two_portfolios_get_two_benchmarks():
    engine = _seed_db()
    result = load_portfolio_series(
        engine,
        "other",
        [
            {"portfolio": "ALTTRN", "type": "Actual", "include_benchmark": True, "benchmark_type": "Actual"},
            {"portfolio": "ALTMAC", "type": "Actual", "include_benchmark": True, "benchmark_type": "Actual"},
        ],
    )

    assert list(result.returns_df.columns) == ["ALTTRN", "ALTTRN_BM", "ALTMAC", "ALTMAC_BM"]
    assert result.benchmark_assignments == {"ALTTRN": "ALTTRN_BM", "ALTMAC": "ALTMAC_BM"}


def test_load_portfolio_series_other_perf_filters_and_scales_returns():
    engine = _seed_db()
    result = load_portfolio_series(
        engine,
        "other",
        [
            {"portfolio": "PERFTRN", "type": "Actual", "include_benchmark": False, "benchmark_type": ""},
        ],
        performance_engine=engine,
    )

    assert list(result.returns_df.columns) == ["PERFTRN"]
    assert np.allclose(result.returns_df["PERFTRN"].dropna().values, 0.015)


def test_load_portfolio_series_other_perf_benchmark_name_and_series():
    engine = _seed_db()
    result = load_portfolio_series(
        engine,
        "other",
        [
            {"portfolio": "PERFTRN", "type": "Actual", "include_benchmark": True, "benchmark_type": "Actual"},
        ],
        performance_engine=engine,
    )

    bm_name = f"PERFTRN{INDEX_BENCHMARK_SUFFIX}"
    assert list(result.returns_df.columns) == ["PERFTRN", bm_name]
    assert result.benchmark_assignments == {"PERFTRN": bm_name}
    assert np.allclose(result.returns_df[bm_name].dropna().values, 0.011)


def test_load_portfolio_series_other_perf_requires_peer_vintage_for_benchmark():
    engine = _seed_db()
    with pytest.raises(ValueError, match="PeerVintage is missing"):
        load_portfolio_series(
            engine,
            "other",
            [
                {"portfolio": "PERFNOBM", "type": "Actual", "include_benchmark": True, "benchmark_type": "Actual"},
            ],
            performance_engine=engine,
        )


def test_has_portfolio_benchmark_respects_source_logic():
    engine = _seed_db()
    assert has_portfolio_benchmark(engine, "peer", "TD2030A")
    assert has_portfolio_benchmark(engine, "index", "Risk60")
    assert has_portfolio_benchmark(engine, "other", "ALTTRN")
    assert not has_portfolio_benchmark(engine, "other", "ALTNOBM")
    assert has_portfolio_benchmark(engine, "other", "PERFTRN")
    assert not has_portfolio_benchmark(engine, "other", "PERFNOBM")
