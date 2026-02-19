from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import create_engine, text

from utils.raw_data_imports import (
    factor_defaults_to_returns,
    get_factor_options_cached,
    get_factor_preview_lines_cached,
    get_fund_options_cached,
    get_fund_preview_lines_cached,
    get_performance_options_cached,
    get_performance_preview_lines_cached,
    load_factor_series,
    load_fund_series,
    load_performance_series,
)


def _seed_mrd_engine():
    engine = create_engine("sqlite:///:memory:", future=True)
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE [CORE_DATA.ACCOUNT] ("
                "ACCT_ID INTEGER PRIMARY KEY, "
                "ACCT_NAME TEXT NOT NULL, "
                "ACCT_CD TEXT NOT NULL, "
                "ACCT_TYPE_CD TEXT NOT NULL, "
                "FACTOR_NAME TEXT NOT NULL, "
                "SOURCE_SYSTEM TEXT NOT NULL)"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE [CORE_DATA.ACCOUNT_FACTOR_DATA] ("
                "ACCT_ID INTEGER NOT NULL, "
                "REFERENCE_DATE DATE NOT NULL, "
                "FACTOR_VALUE REAL NOT NULL, "
                "SOURCE_SYSTEM TEXT NOT NULL)"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE [CORE_DATA.ACCOUNT_RETURNS] ("
                "ACCT_ID INTEGER NOT NULL, "
                "REFERENCE_DATE DATE NOT NULL, "
                "GROSS REAL NOT NULL, "
                "NET REAL NOT NULL, "
                "SOURCE_SYSTEM TEXT NOT NULL)"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE [CORE_DATA.ACCOUNT_RETURNS_M] ("
                "ACCT_ID INTEGER NOT NULL, "
                "REFERENCE_DATE DATE NOT NULL, "
                "GROSS REAL NOT NULL, "
                "NET REAL NOT NULL, "
                "SOURCE_SYSTEM TEXT NOT NULL)"
            )
        )

        conn.execute(
            text(
                "INSERT INTO [CORE_DATA.ACCOUNT] "
                "(ACCT_ID, ACCT_NAME, ACCT_CD, ACCT_TYPE_CD, FACTOR_NAME, SOURCE_SYSTEM) VALUES "
                "(1, 'SPX', 'SPTR Index', 'SEC_FACTOR', 'TRIndex', 'BB'), "
                "(2, 'UST10Y', 'USGG10YR Index', 'SEC_FACTOR', 'Yield', 'BB'), "
                "(3, 'PERF_EXCL', 'EXCL Index', 'SEC_FACTOR', 'TRIndex', 'PERF'), "
                "(10, 'Fund A', 'FUNDA', 'OE', 'Ret', 'MSTAR'), "
                "(11, 'Fund B', 'FUNDB', 'TRUST', 'Ret', 'MSTAR'), "
                "(12, 'Fund Excluded', 'FUNDX', 'OE', 'Ret', 'OTHER')"
            )
        )

        factor_dates = [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)]
        factor_values_1 = [100.0, 101.0, 103.0]
        factor_values_2 = [5.0, 5.2, 5.4]
        factor_values_3 = [50.0, 52.0, 54.0]
        for dt, v1, v2, v3 in zip(factor_dates, factor_values_1, factor_values_2, factor_values_3):
            conn.execute(
                text(
                    "INSERT INTO [CORE_DATA.ACCOUNT_FACTOR_DATA] "
                    "(ACCT_ID, REFERENCE_DATE, FACTOR_VALUE, SOURCE_SYSTEM) VALUES "
                    "(:acct_id, :dt, :value, :source)"
                ),
                {"acct_id": 1, "dt": dt, "value": v1, "source": "BB"},
            )
            conn.execute(
                text(
                    "INSERT INTO [CORE_DATA.ACCOUNT_FACTOR_DATA] "
                    "(ACCT_ID, REFERENCE_DATE, FACTOR_VALUE, SOURCE_SYSTEM) VALUES "
                    "(:acct_id, :dt, :value, :source)"
                ),
                {"acct_id": 2, "dt": dt, "value": v2, "source": "BB"},
            )
            conn.execute(
                text(
                    "INSERT INTO [CORE_DATA.ACCOUNT_FACTOR_DATA] "
                    "(ACCT_ID, REFERENCE_DATE, FACTOR_VALUE, SOURCE_SYSTEM) VALUES "
                    "(:acct_id, :dt, :value, :source)"
                ),
                {"acct_id": 3, "dt": dt, "value": v3, "source": "PERF"},
            )

        daily_dates = [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)]
        for dt in daily_dates:
            conn.execute(
                text(
                    "INSERT INTO [CORE_DATA.ACCOUNT_RETURNS] "
                    "(ACCT_ID, REFERENCE_DATE, GROSS, NET, SOURCE_SYSTEM) VALUES "
                    "(10, :dt, 0.0100, 0.0092, 'MSTAR')"
                ),
                {"dt": dt},
            )
            conn.execute(
                text(
                    "INSERT INTO [CORE_DATA.ACCOUNT_RETURNS] "
                    "(ACCT_ID, REFERENCE_DATE, GROSS, NET, SOURCE_SYSTEM) VALUES "
                    "(11, :dt, 0.0040, 0.0038, 'MSTAR')"
                ),
                {"dt": dt},
            )
            conn.execute(
                text(
                    "INSERT INTO [CORE_DATA.ACCOUNT_RETURNS] "
                    "(ACCT_ID, REFERENCE_DATE, GROSS, NET, SOURCE_SYSTEM) VALUES "
                    "(12, :dt, 0.0200, 0.0190, 'OTHER')"
                ),
                {"dt": dt},
            )

        monthly_dates = [date(2020, 1, 31), date(2020, 2, 29)]
        for dt in monthly_dates:
            conn.execute(
                text(
                    "INSERT INTO [CORE_DATA.ACCOUNT_RETURNS_M] "
                    "(ACCT_ID, REFERENCE_DATE, GROSS, NET, SOURCE_SYSTEM) VALUES "
                    "(10, :dt, 0.0220, 0.0200, 'MSTAR')"
                ),
                {"dt": dt},
            )
            conn.execute(
                text(
                    "INSERT INTO [CORE_DATA.ACCOUNT_RETURNS_M] "
                    "(ACCT_ID, REFERENCE_DATE, GROSS, NET, SOURCE_SYSTEM) VALUES "
                    "(11, :dt, 0.0110, 0.0102, 'MSTAR')"
                ),
                {"dt": dt},
            )
    return engine


def _seed_perf_engine():
    engine = create_engine("sqlite:///:memory:", future=True)
    with engine.begin() as conn:
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
                "ACCT_ID INTEGER NOT NULL, "
                "BENCHMARK_ACCT_ID INTEGER NOT NULL, "
                "FEE_TYPE TEXT NOT NULL, "
                "IS_LATEST INTEGER NOT NULL, "
                "Daily_ror REAL NOT NULL, "
                "Daily_ror_index REAL NOT NULL)"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE [MONTHLY_RETURN] ("
                "Effective_Date DATE NOT NULL, "
                "ACCT_ID INTEGER NOT NULL, "
                "BENCHMARK_ACCT_ID INTEGER NOT NULL, "
                "FEE_TYPE TEXT NOT NULL, "
                "IS_LATEST INTEGER NOT NULL, "
                "Return_Type TEXT NOT NULL, "
                "mth1_ror REAL NOT NULL, "
                "mth1_ror_index REAL NOT NULL)"
            )
        )

        conn.execute(
            text(
                "INSERT INTO [ACCOUNT] (ACCT_ID, ACCT_CD) VALUES "
                "(100, 'PERF1'), "
                "(101, 'PERF2')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO [ACCOUNT_BENCHMARK] (BENCHMARK_ID, PRECEDENCE) VALUES "
                "(901, 1), "
                "(902, 2)"
            )
        )

        for dt in [date(2020, 1, 1), date(2020, 1, 2)]:
            conn.execute(
                text(
                    "INSERT INTO [DAILY_RETURN] "
                    "(Effective_Date, ACCT_ID, BENCHMARK_ACCT_ID, FEE_TYPE, IS_LATEST, Daily_ror, Daily_ror_index) VALUES "
                    "(:dt, 100, 901, 'G', 1, 1.50, 1.10), "
                    "(:dt, 100, 901, 'N', 1, 1.30, 1.00), "
                    "(:dt, 100, 902, 'G', 1, 2.50, 2.10), "
                    "(:dt, 100, 901, 'G', 0, 2.30, 2.00), "
                    "(:dt, 101, 901, 'G', 1, 0.90, 0.70)"
                ),
                {"dt": dt},
            )

        for dt in [date(2020, 1, 31), date(2020, 2, 29)]:
            conn.execute(
                text(
                    "INSERT INTO [MONTHLY_RETURN] "
                    "(Effective_Date, ACCT_ID, BENCHMARK_ACCT_ID, FEE_TYPE, IS_LATEST, Return_Type, mth1_ror, mth1_ror_index) VALUES "
                    "(:dt, 100, 901, 'G', 1, 'Ann', 2.00, 1.70), "
                    "(:dt, 100, 901, 'N', 1, 'Ann', 1.70, 1.40), "
                    "(:dt, 100, 901, 'G', 1, 'Cum', 7.00, 6.00), "
                    "(:dt, 100, 902, 'G', 1, 'Ann', 6.00, 5.00), "
                    "(:dt, 100, 901, 'G', 0, 'Ann', 5.00, 4.00), "
                    "(:dt, 101, 901, 'N', 1, 'Ann', 1.20, 1.00)"
                ),
                {"dt": dt},
            )
    return engine


def test_factor_defaults():
    assert factor_defaults_to_returns("TRIndex") is True
    assert factor_defaults_to_returns("Yield") is False


def test_factor_options_preview_and_load():
    mrd = _seed_mrd_engine()
    options = get_factor_options_cached(mrd)
    labels = {opt["label"] for opt in options}
    assert "SPX_TRIndex [BB: SPTR Index]" in labels
    assert "UST10Y_Yield [BB: USGG10YR Index]" in labels
    assert all("PERF_EXCL_TRIndex" not in label for label in labels)

    lines = get_factor_preview_lines_cached(mrd, "1")
    assert lines[0] == "2020-01-01:100"

    result = load_factor_series(
        mrd,
        [
            {
                "mode": "factor",
                "series_key": "1",
                "import_name": "SPX_TRIndex",
                "convert_to_returns": True,
                "divide_by": 100,
            },
            {
                "mode": "factor",
                "series_key": "2",
                "import_name": "UST10Y_Yield",
                "convert_to_returns": False,
                "divide_by": 100,
            },
        ],
    )

    assert list(result.returns_df.columns) == ["SPX_TRIndex", "UST10Y_Yield"]
    assert result.returns_df["SPX_TRIndex"].dropna().iloc[0] == pytest.approx(0.01)
    assert result.returns_df["UST10Y_Yield"].dropna().iloc[0] == pytest.approx(0.05)
    assert result.periodicity == "daily"


def test_funds_options_preview_and_load():
    mrd = _seed_mrd_engine()
    options = get_fund_options_cached(mrd)
    values = {opt["label"] for opt in options}
    assert "Fund A" in values
    assert "Fund B" in values
    assert "Fund Excluded" not in values

    lines = get_fund_preview_lines_cached(mrd, "10", "daily", "gross")
    assert lines[0] == "2020-01-01:0.01"

    result = load_fund_series(
        mrd,
        [
            {
                "mode": "funds",
                "series_key": "10",
                "import_name": "Fund A",
                "table_choice": "daily",
                "fee_choice": "gross",
            },
            {
                "mode": "funds",
                "series_key": "11",
                "import_name": "Fund B",
                "table_choice": "monthly",
                "fee_choice": "net",
            },
        ],
    )

    assert list(result.returns_df.columns) == ["Fund A", "Fund B"]
    assert result.returns_df["Fund A"].dropna().iloc[0] == pytest.approx(0.01)
    assert result.periodicity == "daily"


def test_performance_options_preview_and_load_filters():
    perf = _seed_perf_engine()
    options = get_performance_options_cached(perf)
    assert {"value": "100", "label": "PERF1"} in options
    assert {"value": "101", "label": "PERF2"} in options

    lines = get_performance_preview_lines_cached(perf, "100", "daily", "N", True)
    assert lines[0] == "2020-01-01:0.013|0.01"

    daily_result = load_performance_series(
        perf,
        [
            {
                "mode": "performance",
                "series_key": "100",
                "import_name": "PERF1",
                "table_choice": "daily",
                "fee_choice": "G",
                "include_benchmark": True,
            }
        ],
    )
    assert list(daily_result.returns_df.columns) == ["PERF1", "PERF1_BM"]
    assert daily_result.returns_df["PERF1"].dropna().iloc[0] == pytest.approx(0.015)
    assert daily_result.returns_df["PERF1_BM"].dropna().iloc[0] == pytest.approx(0.011)
    assert daily_result.benchmark_assignments == {"PERF1": "PERF1_BM"}
    assert daily_result.periodicity == "daily"

    monthly_result = load_performance_series(
        perf,
        [
            {
                "mode": "performance",
                "series_key": "101",
                "import_name": "PERF2",
                "table_choice": "monthly",
                "fee_choice": "N",
                "include_benchmark": False,
            }
        ],
    )
    assert list(monthly_result.returns_df.columns) == ["PERF2"]
    assert monthly_result.returns_df["PERF2"].dropna().iloc[0] == pytest.approx(0.012)
    assert monthly_result.benchmark_assignments == {}
    assert monthly_result.periodicity == "monthly"
