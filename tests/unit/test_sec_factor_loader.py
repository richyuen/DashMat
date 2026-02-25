from __future__ import annotations

from datetime import date

import pandas as pd
from pandas.testing import assert_series_equal
from sqlalchemy import create_engine, text

from utils.sec_factor_loader import (
    load_sec_factor_returns_by_acct_ids_aa,
    load_sec_factor_returns_by_names_aa,
    normalize_sec_factor_name,
    resolve_sec_factor_accounts_by_names,
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
                "INSERT INTO [CORE_DATA.ACCOUNT] "
                "(ACCT_ID, ACCT_NAME, ACCT_CD, ACCT_TYPE_CD, FACTOR_NAME, SOURCE_SYSTEM) VALUES "
                "(1, 'ACCX', 'ACCX_PERF', 'SEC_FACTOR', 'TRIndex', 'PERF'), "
                "(2, 'ACCX', 'ACCX_BB', 'SEC_FACTOR', 'TRIndex', 'BB')"
            )
        )

        series_dates = [
            date(2024, 1, 2),
            date(2024, 1, 3),
            date(2024, 1, 4),
            date(2024, 1, 8),
        ]
        perf_levels = [200.0, 202.0, 204.0, 210.0]
        bb_levels = [100.0, 101.0, 102.0, 104.0]
        for dt, perf_value, bb_value in zip(series_dates, perf_levels, bb_levels):
            conn.execute(
                text(
                    "INSERT INTO [CORE_DATA.ACCOUNT_FACTOR_DATA] "
                    "(ACCT_ID, REFERENCE_DATE, FACTOR_VALUE, SOURCE_SYSTEM) VALUES "
                    "(:acct_id, :dt, :value, :source)"
                ),
                {"acct_id": 1, "dt": dt, "value": perf_value, "source": "PERF"},
            )
            conn.execute(
                text(
                    "INSERT INTO [CORE_DATA.ACCOUNT_FACTOR_DATA] "
                    "(ACCT_ID, REFERENCE_DATE, FACTOR_VALUE, SOURCE_SYSTEM) VALUES "
                    "(:acct_id, :dt, :value, :source)"
                ),
                {"acct_id": 2, "dt": dt, "value": bb_value, "source": "BB"},
            )
    return engine


def test_normalize_sec_factor_name_supports_aliases():
    assert normalize_sec_factor_name("SPX_TRIndex") == "SPX_TRIndex"
    assert normalize_sec_factor_name("SPX TRIndex") == "SPX_TRIndex"
    assert normalize_sec_factor_name(" SPX   TRIndex ") == "SPX_TRIndex"
    assert normalize_sec_factor_name("SPX") == "SPX_TRIndex"


def test_resolve_sec_factor_accounts_prefers_bb_then_lowest():
    mrd_engine = _seed_mrd_engine()
    resolved = resolve_sec_factor_accounts_by_names(
        mrd_engine,
        ["ACCX_TRIndex"],
        collision_policy="bb_then_lowest",
        exclude_perf=False,
    )
    assert resolved["ACCX_TRIndex"] == 2


def test_load_returns_by_names_accepts_underscore_space_and_default_trindex_aliases():
    mrd_engine = _seed_mrd_engine()
    df_us, _meta_us = load_sec_factor_returns_by_names_aa(mrd_engine, ["ACCX_TRIndex"])
    df_sp, _meta_sp = load_sec_factor_returns_by_names_aa(mrd_engine, ["ACCX TRIndex"])
    df_no, _meta_no = load_sec_factor_returns_by_names_aa(mrd_engine, ["ACCX"])

    assert "ACCX_TRIndex" in df_us.columns
    assert "ACCX TRIndex" in df_sp.columns
    assert "ACCX" in df_no.columns

    s_us = df_us["ACCX_TRIndex"]
    s_sp = df_sp["ACCX TRIndex"]
    s_no = df_no["ACCX"]
    assert_series_equal(s_us, s_sp, check_names=False)
    assert_series_equal(s_us, s_no, check_names=False)

    # Daily-phase calendar fill applies from inferred daily start onward.
    assert pd.Timestamp("2024-01-05") in s_us.index
    assert float(s_us.loc[pd.Timestamp("2024-01-05")]) == 0.0


def test_load_returns_by_acct_ids_keeps_requested_output_name():
    mrd_engine = _seed_mrd_engine()
    df_alias, _meta = load_sec_factor_returns_by_acct_ids_aa(
        mrd_engine,
        {"Alias_ACCX": 2},
    )
    assert list(df_alias.columns) == ["Alias_ACCX"]
    assert pd.Timestamp("2024-01-05") in df_alias.index
    assert float(df_alias.loc[pd.Timestamp("2024-01-05"), "Alias_ACCX"]) == 0.0

