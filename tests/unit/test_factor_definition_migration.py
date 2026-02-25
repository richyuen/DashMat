from __future__ import annotations

from sqlalchemy import create_engine, text

import tools.db.migrate_factor_definitions as factor_migration
from tools.db.migrate_factor_definitions import (
    _factor_table_name,
    SAMPLE_FACTOR_SPECS,
    ensure_factor_definition_tables_and_seed,
    seed_sample_factor_definitions,
)


def _seed_mrd_engine(with_tokens: bool = True):
    engine = create_engine("sqlite:///:memory:", future=True)
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE [CORE_DATA.ACCOUNT] ("
                "ACCT_ID INTEGER PRIMARY KEY, "
                "ACCT_NAME TEXT NOT NULL, "
                "FACTOR_NAME TEXT NOT NULL, "
                "ACCT_TYPE_CD TEXT NOT NULL, "
                "SOURCE_SYSTEM TEXT NOT NULL)"
            )
        )
        if with_tokens:
            conn.execute(
                text(
                    "INSERT INTO [CORE_DATA.ACCOUNT] "
                    "(ACCT_ID, ACCT_NAME, FACTOR_NAME, ACCT_TYPE_CD, SOURCE_SYSTEM) VALUES "
                    "(1, 'SPX', 'TRIndex', 'SEC_FACTOR', 'BB'), "
                    "(2, 'EAFE', 'TRIndex', 'SEC_FACTOR', 'BB'), "
                    "(3, 'EM', 'TRIndex', 'SEC_FACTOR', 'BB'), "
                    "(4, 'BCAgg', 'TRIndex', 'SEC_FACTOR', 'BB'), "
                    "(5, 'PERF_EXCL', 'TRIndex', 'SEC_FACTOR', 'PERF')"
                )
            )
    return engine


def test_migration_factor_table_name_uses_dbo_for_sql_server():
    class _MockDialect:
        name = "mssql"

    class _MockEngine:
        dialect = _MockDialect()

    assert _factor_table_name(_MockEngine(), "FactorDefinitions") == "[dbo].[FactorDefinitions]"


def test_migration_factor_table_exists_checks_dbo_schema_first(monkeypatch):
    calls: list[tuple[str, str | None]] = []

    class _MockInspector:
        def has_table(self, table_name, schema=None):
            calls.append((table_name, schema))
            return schema is None

    class _MockDialect:
        name = "mssql"

    class _MockEngine:
        dialect = _MockDialect()

    monkeypatch.setattr(factor_migration, "inspect", lambda _engine: _MockInspector())

    assert factor_migration._factor_table_exists(_MockEngine(), "FactorDefinitions") is True
    assert calls == [("FactorDefinitions", "dbo"), ("FactorDefinitions", None)]


def test_migration_factor_table_indexes_reads_dbo_schema_first(monkeypatch):
    calls: list[tuple[str, str | None]] = []

    class _MockInspector:
        def get_indexes(self, table_name, schema=None):
            calls.append((table_name, schema))
            if schema == "dbo":
                return [{"name": "idx_factor_defs_name"}]
            return []

    class _MockDialect:
        name = "mssql"

    class _MockEngine:
        dialect = _MockDialect()

    monkeypatch.setattr(factor_migration, "inspect", lambda _engine: _MockInspector())

    indexes = factor_migration._factor_table_indexes(_MockEngine(), "FactorDefinitions")
    assert indexes == {"idx_factor_defs_name"}
    assert calls == [("FactorDefinitions", "dbo")]


def test_ensure_factor_definition_tables_and_seed_creates_and_inserts_samples():
    db_engine = create_engine("sqlite:///:memory:", future=True)
    mrd_engine = _seed_mrd_engine(with_tokens=True)

    stats = ensure_factor_definition_tables_and_seed(
        db_engine,
        mrd_engine,
        update_by="unit_test",
    )

    assert stats["created_factor_definitions"] is True
    assert stats["created_factor_definitions_archive"] is True
    assert stats["token_count"] == 4
    assert stats["inserted"] > 0
    assert stats["updated"] == 0
    assert stats["archived"] == 0

    with db_engine.connect() as conn:
        live_count = int(conn.execute(text("SELECT COUNT(*) FROM FactorDefinitions")).scalar_one())
        archive_count = int(conn.execute(text("SELECT COUNT(*) FROM FactorDefinitionsArchive")).scalar_one())
        sample_prefix_count = int(
            conn.execute(text("SELECT COUNT(*) FROM FactorDefinitions WHERE FactorName LIKE 'SAMPLE_%'")).scalar_one()
        )

    assert live_count == stats["inserted"]
    assert archive_count == 0
    assert sample_prefix_count == live_count


def test_seed_factor_definitions_is_idempotent_and_archives_on_change():
    db_engine = create_engine("sqlite:///:memory:", future=True)
    mrd_engine = _seed_mrd_engine(with_tokens=True)

    first = ensure_factor_definition_tables_and_seed(db_engine, mrd_engine, update_by="unit_test")
    assert first["inserted"] > 0

    second = seed_sample_factor_definitions(db_engine, mrd_engine, update_by="unit_test")
    assert second["inserted"] == 0
    assert second["updated"] == 0
    assert second["archived"] == 0
    assert second["unchanged"] == first["inserted"]

    with db_engine.begin() as conn:
        factor_name = conn.execute(text("SELECT FactorName FROM FactorDefinitions ORDER BY FactorName LIMIT 1")).scalar_one()
        conn.execute(
            text("UPDATE FactorDefinitions SET Description = 'Manual override' WHERE FactorName = :factor_name"),
            {"factor_name": factor_name},
        )

    third = seed_sample_factor_definitions(db_engine, mrd_engine, update_by="unit_test")
    assert third["updated"] >= 1
    assert third["archived"] >= 1

    with db_engine.connect() as conn:
        archive_count = int(conn.execute(text("SELECT COUNT(*) FROM FactorDefinitionsArchive")).scalar_one())
        description = conn.execute(
            text("SELECT Description FROM FactorDefinitions WHERE FactorName = :factor_name"),
            {"factor_name": factor_name},
        ).scalar_one()

    assert archive_count >= 1
    assert description != "Manual override"


def test_seed_skips_when_no_sec_factor_tokens_available():
    db_engine = create_engine("sqlite:///:memory:", future=True)
    mrd_engine = _seed_mrd_engine(with_tokens=False)

    stats = ensure_factor_definition_tables_and_seed(db_engine, mrd_engine, update_by="unit_test")
    assert stats["token_count"] == 0
    assert stats["eligible"] == 0
    assert stats["inserted"] == 0
    assert stats["skipped"] == len(SAMPLE_FACTOR_SPECS)

    with db_engine.connect() as conn:
        live_count = int(conn.execute(text("SELECT COUNT(*) FROM FactorDefinitions")).scalar_one())
        archive_count = int(conn.execute(text("SELECT COUNT(*) FROM FactorDefinitionsArchive")).scalar_one())
    assert live_count == 0
    assert archive_count == 0
