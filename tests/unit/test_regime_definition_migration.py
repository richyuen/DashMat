from __future__ import annotations

from sqlalchemy import create_engine, inspect, text

from tools.db.migrate_regime_definitions import ensure_regime_definition_tables


def test_ensure_regime_definition_tables_creates_missing_tables_and_indexes():
    db_engine = create_engine("sqlite:///:memory:", future=True)

    first = ensure_regime_definition_tables(db_engine)
    assert first["created_regime_definitions"] is True
    assert first["created_regime_definitions_archive"] is True

    second = ensure_regime_definition_tables(db_engine)
    assert second["created_regime_definitions"] is False
    assert second["created_regime_definitions_archive"] is False

    inspector = inspect(db_engine)
    assert inspector.has_table("RegimeDefinitions")
    assert inspector.has_table("RegimeDefinitionsArchive")

    with db_engine.connect() as conn:
        index_rows = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='index' ORDER BY name")
        ).fetchall()
    index_names = {row[0] for row in index_rows}
    assert "idx_regime_defs_name" in index_names
    assert "idx_regime_defs_archive_name_date" in index_names
