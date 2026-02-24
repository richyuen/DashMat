"""Create RegimeDefinitions and RegimeDefinitionsArchive if missing."""

from __future__ import annotations

from pathlib import Path
import sys

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dbengine import DATABASE_URL, engine  # noqa: E402


def _table_name(db_engine: Engine, base: str) -> str:
    if db_engine.dialect.name == "sqlite":
        return f"[{base}]"
    return f"[dbo].[{base}]"


def _create_tables_if_missing(db_engine: Engine) -> dict[str, bool]:
    inspector = inspect(db_engine)
    if db_engine.dialect.name == "sqlite":
        has_defs = inspector.has_table("RegimeDefinitions")
        has_archive = inspector.has_table("RegimeDefinitionsArchive")
    else:
        has_defs = inspector.has_table("RegimeDefinitions", schema="dbo") or inspector.has_table("RegimeDefinitions")
        has_archive = (
            inspector.has_table("RegimeDefinitionsArchive", schema="dbo")
            or inspector.has_table("RegimeDefinitionsArchive")
        )

    defs_table = _table_name(db_engine, "RegimeDefinitions")
    archive_table = _table_name(db_engine, "RegimeDefinitionsArchive")
    created_defs = False
    created_archive = False

    with db_engine.begin() as conn:
        if not has_defs:
            conn.execute(
                text(
                    f"""
                    CREATE TABLE {defs_table} (
                        RegimeName VARCHAR(128) NOT NULL PRIMARY KEY,
                        Description VARCHAR(4000) NULL,
                        MethodType INTEGER NOT NULL,
                        ConfigJson {'TEXT' if db_engine.dialect.name == 'sqlite' else 'VARCHAR(MAX)'} NOT NULL,
                        UPDATE_DATE DATETIME NOT NULL,
                        UPDATE_BY VARCHAR(128) NOT NULL
                    )
                    """
                )
            )
            created_defs = True

        if not has_archive:
            conn.execute(
                text(
                    f"""
                    CREATE TABLE {archive_table} (
                        RegimeName VARCHAR(128) NOT NULL,
                        Description VARCHAR(4000) NULL,
                        MethodType INTEGER NOT NULL,
                        ConfigJson {'TEXT' if db_engine.dialect.name == 'sqlite' else 'VARCHAR(MAX)'} NOT NULL,
                        UPDATE_DATE DATETIME NOT NULL,
                        UPDATE_BY VARCHAR(128) NOT NULL,
                        ARCHIVE_DATE DATETIME NOT NULL
                    )
                    """
                )
            )
            created_archive = True

    inspector = inspect(db_engine)
    with db_engine.begin() as conn:
        if db_engine.dialect.name == "sqlite":
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_regime_defs_name "
                    "ON RegimeDefinitions (RegimeName)"
                )
            )
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_regime_defs_archive_name_date "
                    "ON RegimeDefinitionsArchive (RegimeName, ARCHIVE_DATE)"
                )
            )
        else:
            defs_indexes = {
                idx.get("name")
                for idx in (inspector.get_indexes("RegimeDefinitions", schema="dbo") or inspector.get_indexes("RegimeDefinitions"))
            }
            if "idx_regime_defs_name" not in defs_indexes:
                conn.execute(text(f"CREATE INDEX idx_regime_defs_name ON {defs_table} (RegimeName)"))

            archive_indexes = {
                idx.get("name")
                for idx in (
                    inspector.get_indexes("RegimeDefinitionsArchive", schema="dbo")
                    or inspector.get_indexes("RegimeDefinitionsArchive")
                )
            }
            if "idx_regime_defs_archive_name_date" not in archive_indexes:
                conn.execute(
                    text(
                        f"CREATE INDEX idx_regime_defs_archive_name_date "
                        f"ON {archive_table} (RegimeName, ARCHIVE_DATE)"
                    )
                )

    return {
        "created_regime_definitions": created_defs,
        "created_regime_definitions_archive": created_archive,
    }


def ensure_regime_definition_tables(db_engine: Engine) -> dict[str, bool]:
    return _create_tables_if_missing(db_engine)


def main() -> None:
    stats = ensure_regime_definition_tables(engine)
    print(f"Regime definition migration complete for {DATABASE_URL}")
    print(f"Created RegimeDefinitions: {stats['created_regime_definitions']}")
    print(f"Created RegimeDefinitionsArchive: {stats['created_regime_definitions_archive']}")


if __name__ == "__main__":
    main()
