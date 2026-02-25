from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

import utils.factor_definitions as factor_defs
from utils.factor_definitions import (
    _factor_table_name,
    compute_factor_preview_lines,
    compute_factor_series,
    delete_factor_definition,
    factor_tables_available,
    get_sec_factor_component_meta_cached,
    resolve_component_tokens_to_acct_ids,
    save_factor_definition,
    validate_factor_definition_payload,
)


def _seed_db_engine():
    engine = create_engine("sqlite:///:memory:", future=True)
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE FactorDefinitions ("
                "FactorName TEXT PRIMARY KEY, "
                "LongComponent TEXT NOT NULL, "
                "ShortComponent TEXT NULL, "
                "Description TEXT NULL, "
                "LongAggType INTEGER NOT NULL, "
                "ShortAggType INTEGER NULL, "
                "LongLag INTEGER NOT NULL, "
                "OutputTransform INTEGER NOT NULL, "
                "UPDATE_DATE DATETIME NOT NULL, "
                "UPDATE_BY TEXT NOT NULL)"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE FactorDefinitionsArchive ("
                "FactorName TEXT NOT NULL, "
                "LongComponent TEXT NOT NULL, "
                "ShortComponent TEXT NULL, "
                "Description TEXT NULL, "
                "LongAggType INTEGER NOT NULL, "
                "ShortAggType INTEGER NULL, "
                "LongLag INTEGER NOT NULL, "
                "OutputTransform INTEGER NOT NULL, "
                "UPDATE_DATE DATETIME NOT NULL, "
                "UPDATE_BY TEXT NOT NULL, "
                "ARCHIVE_DATE DATETIME NOT NULL)"
            )
        )
    return engine


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
                "(1, 'ACC1', 'ACC1_TR', 'SEC_FACTOR', 'TRIndex', 'BB'), "
                "(2, 'ACC1', 'ACC1_TR2', 'SEC_FACTOR', 'TRIndex', 'BB'), "
                "(3, 'ACC2', 'ACC2_TR', 'SEC_FACTOR', 'TRIndex', 'BB')"
            )
        )

        for i in range(10):
            dt = date(2024, 1, 1) + timedelta(days=i)
            conn.execute(
                text(
                    "INSERT INTO [CORE_DATA.ACCOUNT_FACTOR_DATA] "
                    "(ACCT_ID, REFERENCE_DATE, FACTOR_VALUE, SOURCE_SYSTEM) VALUES "
                    "(:acct_id, :dt, :value, 'BB')"
                ),
                {"acct_id": 1, "dt": dt, "value": 0.01 + i * 0.001},
            )
            conn.execute(
                text(
                    "INSERT INTO [CORE_DATA.ACCOUNT_FACTOR_DATA] "
                    "(ACCT_ID, REFERENCE_DATE, FACTOR_VALUE, SOURCE_SYSTEM) VALUES "
                    "(:acct_id, :dt, :value, 'BB')"
                ),
                {"acct_id": 2, "dt": dt, "value": 0.5 + i * 0.01},
            )
            conn.execute(
                text(
                    "INSERT INTO [CORE_DATA.ACCOUNT_FACTOR_DATA] "
                    "(ACCT_ID, REFERENCE_DATE, FACTOR_VALUE, SOURCE_SYSTEM) VALUES "
                    "(:acct_id, :dt, :value, 'BB')"
                ),
                {"acct_id": 3, "dt": dt, "value": 0.005 + i * 0.0005},
            )
    return engine


def test_factor_table_name_uses_dbo_for_sql_server():
    class _MockDialect:
        name = "mssql"

    class _MockEngine:
        dialect = _MockDialect()

    assert _factor_table_name(_MockEngine(), "FactorDefinitions") == "[dbo].[FactorDefinitions]"


def test_factor_table_exists_checks_dbo_schema_first_for_sql_server(monkeypatch):
    calls: list[tuple[str, str | None]] = []

    class _MockInspector:
        def has_table(self, table_name, schema=None):
            calls.append((table_name, schema))
            return schema is None

    class _MockDialect:
        name = "mssql"

    class _MockEngine:
        dialect = _MockDialect()

    monkeypatch.setattr(factor_defs, "inspect", lambda _engine: _MockInspector())

    assert factor_defs._factor_table_exists(_MockEngine(), "FactorDefinitions") is True
    assert calls == [("FactorDefinitions", "dbo"), ("FactorDefinitions", None)]


def test_validate_factor_definition_payload():
    normalized, error = validate_factor_definition_payload(
        {
            "FactorName": "Carry",
            "LongComponent": ["ACC1 TRIndex"],
            "ShortComponent": [],
            "LongAggType": 2,
            "ShortAggType": None,
            "LongLag": 1,
            "OutputTransform": 0,
        }
    )
    assert error is None
    assert normalized is not None
    assert normalized["FactorName"] == "Carry"
    assert normalized["ShortAggType"] is None


def test_save_update_delete_factor_definition_archives_versions():
    db_engine = _seed_db_engine()
    assert factor_tables_available(db_engine) is True

    ok, _, saved = save_factor_definition(
        db_engine,
        {
            "FactorName": "Carry",
            "LongComponent": ["ACC1 TRIndex"],
            "ShortComponent": [],
            "LongAggType": 2,
            "ShortAggType": None,
            "LongLag": 0,
            "OutputTransform": 0,
        },
        update_by="Admin:tester",
    )
    assert ok is True
    assert saved is not None

    update_ok, _, updated = save_factor_definition(
        db_engine,
        {
            "FactorName": "Carry",
            "LongComponent": ["ACC1 TRIndex"],
            "ShortComponent": ["ACC2 TRIndex"],
            "LongAggType": 2,
            "ShortAggType": 2,
            "LongLag": 1,
            "OutputTransform": 2,
        },
        update_by="Admin:tester",
        original_name="Carry",
        expected_update_date=saved["UPDATE_DATE"],
    )
    assert update_ok is True
    assert updated is not None
    assert updated["LongLag"] == 1

    delete_ok, _ = delete_factor_definition(
        db_engine,
        "Carry",
        expected_update_date=updated["UPDATE_DATE"],
    )
    assert delete_ok is True

    with db_engine.connect() as conn:
        live_count = conn.execute(text("SELECT COUNT(*) FROM FactorDefinitions")).scalar_one()
        archive_count = conn.execute(text("SELECT COUNT(*) FROM FactorDefinitionsArchive")).scalar_one()
    assert live_count == 0
    assert archive_count == 2


def test_case_insensitive_factor_name_uniqueness():
    db_engine = _seed_db_engine()
    ok, _, _ = save_factor_definition(
        db_engine,
        {
            "FactorName": "Quality",
            "LongComponent": ["ACC1 TRIndex"],
            "ShortComponent": [],
            "LongAggType": 2,
            "ShortAggType": None,
            "LongLag": 0,
            "OutputTransform": 0,
        },
        update_by="Admin:tester",
    )
    assert ok is True

    duplicate_ok, message, _ = save_factor_definition(
        db_engine,
        {
            "FactorName": "quality",
            "LongComponent": ["ACC1 TRIndex"],
            "ShortComponent": [],
            "LongAggType": 2,
            "ShortAggType": None,
            "LongLag": 0,
            "OutputTransform": 0,
        },
        update_by="Admin:tester",
    )
    assert duplicate_ok is False
    assert "already exists" in message.lower()


def test_component_resolution_prefers_first_match_and_computes_series():
    mrd_engine = _seed_mrd_engine()

    meta = get_sec_factor_component_meta_cached(mrd_engine)
    assert meta["ACC1 TRIndex"]["count"] == 2
    resolved_ids = resolve_component_tokens_to_acct_ids(mrd_engine, ["ACC1 TRIndex"])
    assert resolved_ids == [1]

    definition = {
        "FactorName": "Spread",
        "LongComponent": ["ACC1 TRIndex"],
        "ShortComponent": ["ACC2 TRIndex"],
        "LongAggType": 2,
        "ShortAggType": 2,
        "LongLag": 1,
        "OutputTransform": 0,
    }
    series = compute_factor_series(
        mrd_engine,
        definition,
        "daily",
        {"start": "2024-01-01", "end": "2024-01-10"},
    )
    assert isinstance(series, pd.Series)
    assert not series.empty

    preview_lines = compute_factor_preview_lines(
        mrd_engine,
        definition,
        "daily",
        {"start": "2024-01-01", "end": "2024-01-10"},
        max_rows=6,
    )
    assert preview_lines
    assert preview_lines[0].startswith("Date:Final")


def test_factor_preview_handles_empty_short_component_series():
    mrd_engine = _seed_mrd_engine()
    with mrd_engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO [CORE_DATA.ACCOUNT] "
                "(ACCT_ID, ACCT_NAME, ACCT_CD, ACCT_TYPE_CD, FACTOR_NAME, SOURCE_SYSTEM) VALUES "
                "(4, 'ACC4', 'ACC4_TR', 'SEC_FACTOR', 'TRIndex', 'BB')"
            )
        )

    definition = {
        "FactorName": "LongOnlyWithEmptyShortData",
        "LongComponent": ["ACC1 TRIndex"],
        "ShortComponent": ["ACC4 TRIndex"],
        "LongAggType": 2,
        "ShortAggType": 2,
        "LongLag": 0,
        "OutputTransform": 0,
    }

    series = compute_factor_series(
        mrd_engine,
        definition,
        "daily",
        {"start": "2024-01-01", "end": "2024-01-10"},
    )
    assert not series.empty

    preview_lines = compute_factor_preview_lines(
        mrd_engine,
        definition,
        "daily",
        {"start": "2024-01-01", "end": "2024-01-10"},
        max_rows=6,
    )
    assert preview_lines
    assert preview_lines[0] == "Date:Final|Long|Combined"
