from __future__ import annotations

from sqlalchemy import create_engine, inspect, text

from tools.db.migrate_account_lists import ensure_account_list_tables


def test_ensure_account_list_tables_creates_missing_tables_and_indexes():
    db_engine = create_engine("sqlite:///:memory:", future=True)

    first = ensure_account_list_tables(db_engine)
    assert first["created_dm_account_lists"] is True
    assert first["created_dm_account_lists_archive"] is True

    second = ensure_account_list_tables(db_engine)
    assert second["created_dm_account_lists"] is False
    assert second["created_dm_account_lists_archive"] is False

    inspector = inspect(db_engine)
    assert inspector.has_table("DMAccountLists")
    assert inspector.has_table("DMAccountListsArchive")

    with db_engine.connect() as conn:
        index_rows = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='index' ORDER BY name")
        ).fetchall()
    index_names = {row[0] for row in index_rows}
    assert "idx_dm_account_lists_username_name_date" in index_names
    assert "idx_dm_account_lists_username_date" in index_names
    assert "idx_dm_account_lists_archive_username_name_date" in index_names
