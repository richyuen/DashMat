"""Backfill PeerTS underlying category sample rows for local testing.

This script inserts missing PeerTS rows with Item='PeerRet' and is safe to run
multiple times. It does not overwrite or delete existing rows.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dbengine import DATABASE_URL, engine
from tools.db.init_local_cma_db import (
    UNDERLYING_CATEGORY_ITEM,
    build_underlying_category_seed_rows,
    _load_daily_returns,
)


def _existing_underlying_keys(conn) -> set[tuple[object, str, str, str]]:
    rows = conn.execute(
        text(
            "SELECT Date, Portfolio, Item, [Desc] "
            "FROM PeerTS "
            "WHERE Item = :item"
        ),
        {"item": UNDERLYING_CATEGORY_ITEM},
    ).fetchall()
    return {
        (pd.Timestamp(row[0]).date(), str(row[1]), str(row[2]), str(row[3]))
        for row in rows
    }


def backfill_underlying_category_peerts(
    target_engine: Engine | None = None,
    daily_df: pd.DataFrame | None = None,
) -> dict[str, int]:
    target_engine = target_engine or engine
    if not inspect(target_engine).has_table("PeerTS"):
        raise RuntimeError("PeerTS table does not exist. Run tools/db/init_local_cma_db.py first.")

    generated_rows = build_underlying_category_seed_rows(daily_df if daily_df is not None else _load_daily_returns())
    if not generated_rows:
        return {"generated": 0, "inserted": 0, "existing": 0}

    with target_engine.begin() as conn:
        existing_keys = _existing_underlying_keys(conn)
        rows_to_insert = [
            row
            for row in generated_rows
            if (row["Date"], row["Portfolio"], row["Item"], row["Desc"]) not in existing_keys
        ]
        if rows_to_insert:
            conn.execute(
                text(
                    "INSERT INTO PeerTS (Date, Portfolio, Item, [Desc], Value) "
                    "VALUES (:Date, :Portfolio, :Item, :Desc, :Value)"
                ),
                rows_to_insert,
            )

    return {
        "generated": len(generated_rows),
        "inserted": len(rows_to_insert),
        "existing": len(generated_rows) - len(rows_to_insert),
    }


def main() -> None:
    stats = backfill_underlying_category_peerts(engine)
    print(f"Backfilled PeerTS underlying category sample rows in {DATABASE_URL}")
    print(f"Generated rows: {stats['generated']}")
    print(f"Inserted rows: {stats['inserted']}")
    print(f"Existing rows skipped: {stats['existing']}")


if __name__ == "__main__":
    main()
