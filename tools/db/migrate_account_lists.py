"""Create DMAccountLists and DMAccountListsArchive if missing."""

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


def ensure_account_list_tables(db_engine: Engine) -> dict[str, bool]:
    inspector = inspect(db_engine)
    if db_engine.dialect.name == "sqlite":
        has_live = inspector.has_table("DMAccountLists")
        has_archive = inspector.has_table("DMAccountListsArchive")
    else:
        has_live = inspector.has_table("DMAccountLists", schema="dbo") or inspector.has_table("DMAccountLists")
        has_archive = inspector.has_table("DMAccountListsArchive", schema="dbo") or inspector.has_table("DMAccountListsArchive")

    live_table = _table_name(db_engine, "DMAccountLists")
    archive_table = _table_name(db_engine, "DMAccountListsArchive")
    created_live = False
    created_archive = False

    with db_engine.begin() as conn:
        if not has_live:
            if db_engine.dialect.name == "sqlite":
                conn.execute(
                    text(
                        f"""
                        CREATE TABLE {live_table} (
                            AccountListID INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                            Username TEXT NOT NULL,
                            ListName TEXT NOT NULL,
                            ConfigJson TEXT NOT NULL,
                            UPDATE_DATE DATETIME NOT NULL,
                            UPDATE_BY TEXT NOT NULL
                        )
                        """
                    )
                )
            else:
                conn.execute(
                    text(
                        f"""
                        CREATE TABLE {live_table} (
                            AccountListID BIGINT IDENTITY(1,1) NOT NULL PRIMARY KEY,
                            Username NVARCHAR(128) NOT NULL,
                            ListName NVARCHAR(256) NOT NULL,
                            ConfigJson NVARCHAR(MAX) NOT NULL,
                            UPDATE_DATE DATETIME2(0) NOT NULL,
                            UPDATE_BY NVARCHAR(128) NOT NULL
                        )
                        """
                    )
                )
            created_live = True

        if not has_archive:
            if db_engine.dialect.name == "sqlite":
                conn.execute(
                    text(
                        f"""
                        CREATE TABLE {archive_table} (
                            AccountListID INTEGER NOT NULL,
                            Username TEXT NOT NULL,
                            ListName TEXT NOT NULL,
                            ConfigJson TEXT NOT NULL,
                            UPDATE_DATE DATETIME NOT NULL,
                            UPDATE_BY TEXT NOT NULL,
                            ARCHIVE_DATE DATETIME NOT NULL
                        )
                        """
                    )
                )
            else:
                conn.execute(
                    text(
                        f"""
                        CREATE TABLE {archive_table} (
                            AccountListID BIGINT NOT NULL,
                            Username NVARCHAR(128) NOT NULL,
                            ListName NVARCHAR(256) NOT NULL,
                            ConfigJson NVARCHAR(MAX) NOT NULL,
                            UPDATE_DATE DATETIME2(0) NOT NULL,
                            UPDATE_BY NVARCHAR(128) NOT NULL,
                            ARCHIVE_DATE DATETIME2(0) NOT NULL
                        )
                        """
                    )
                )
            created_archive = True

        if db_engine.dialect.name == "sqlite":
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_dm_account_lists_username_name_date "
                    "ON DMAccountLists (Username, ListName, UPDATE_DATE DESC)"
                )
            )
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_dm_account_lists_username_date "
                    "ON DMAccountLists (Username, UPDATE_DATE DESC, AccountListID DESC)"
                )
            )
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_dm_account_lists_archive_username_name_date "
                    "ON DMAccountListsArchive (Username, ListName, ARCHIVE_DATE DESC)"
                )
            )
        else:
            live_indexes = {
                idx.get("name")
                for idx in (inspector.get_indexes("DMAccountLists", schema="dbo") or inspector.get_indexes("DMAccountLists"))
            }
            if "idx_dm_account_lists_username_name_date" not in live_indexes:
                conn.execute(
                    text(
                        f"CREATE INDEX idx_dm_account_lists_username_name_date "
                        f"ON {live_table} (Username, ListName, UPDATE_DATE DESC)"
                    )
                )
            if "idx_dm_account_lists_username_date" not in live_indexes:
                conn.execute(
                    text(
                        f"CREATE INDEX idx_dm_account_lists_username_date "
                        f"ON {live_table} (Username, UPDATE_DATE DESC, AccountListID DESC)"
                    )
                )

            archive_indexes = {
                idx.get("name")
                for idx in (
                    inspector.get_indexes("DMAccountListsArchive", schema="dbo")
                    or inspector.get_indexes("DMAccountListsArchive")
                )
            }
            if "idx_dm_account_lists_archive_username_name_date" not in archive_indexes:
                conn.execute(
                    text(
                        f"CREATE INDEX idx_dm_account_lists_archive_username_name_date "
                        f"ON {archive_table} (Username, ListName, ARCHIVE_DATE DESC)"
                    )
                )

    return {
        "created_dm_account_lists": created_live,
        "created_dm_account_lists_archive": created_archive,
    }


def main() -> None:
    stats = ensure_account_list_tables(engine)
    print(f"Account-list migration complete for {DATABASE_URL}")
    print(f"Created DMAccountLists: {stats['created_dm_account_lists']}")
    print(f"Created DMAccountListsArchive: {stats['created_dm_account_lists_archive']}")


if __name__ == "__main__":
    main()
