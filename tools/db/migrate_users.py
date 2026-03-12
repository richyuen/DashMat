"""Create Users table if missing."""

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


def ensure_users_table(db_engine: Engine) -> dict[str, bool]:
    inspector = inspect(db_engine)
    if db_engine.dialect.name == "sqlite":
        has_users = inspector.has_table("Users")
    else:
        has_users = inspector.has_table("Users", schema="dbo") or inspector.has_table("Users")

    users_table = _table_name(db_engine, "Users")
    created_users = False

    with db_engine.begin() as conn:
        if not has_users:
            if db_engine.dialect.name == "sqlite":
                conn.execute(
                    text(
                        f"""
                        CREATE TABLE {users_table} (
                            Username TEXT NOT NULL PRIMARY KEY,
                            Role TEXT NOT NULL
                        )
                        """
                    )
                )
            else:
                conn.execute(
                    text(
                        f"""
                        CREATE TABLE {users_table} (
                            Username NVARCHAR(128) NOT NULL PRIMARY KEY,
                            Role NVARCHAR(128) NOT NULL
                        )
                        """
                    )
                )
            created_users = True

    return {"created_users": created_users}


def main() -> None:
    stats = ensure_users_table(engine)
    print(f"Users migration complete for {DATABASE_URL}")
    print(f"Created Users: {stats['created_users']}")


if __name__ == "__main__":
    main()
