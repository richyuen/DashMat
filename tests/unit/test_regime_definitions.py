from __future__ import annotations

from sqlalchemy import create_engine, text

import utils.regime_definitions as regime_defs
from utils.regime_definitions import (
    delete_regime_definition,
    load_regime_definitions,
    regime_tables_available,
    save_regime_definition,
    validate_regime_definition_payload,
)


def _seed_db_engine():
    engine = create_engine("sqlite:///:memory:", future=True)
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE RegimeDefinitions ("
                "RegimeName TEXT PRIMARY KEY, "
                "Description TEXT NULL, "
                "MethodType INTEGER NOT NULL, "
                "ConfigJson TEXT NOT NULL, "
                "UPDATE_DATE DATETIME NOT NULL, "
                "UPDATE_BY TEXT NOT NULL)"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE RegimeDefinitionsArchive ("
                "RegimeName TEXT NOT NULL, "
                "Description TEXT NULL, "
                "MethodType INTEGER NOT NULL, "
                "ConfigJson TEXT NOT NULL, "
                "UPDATE_DATE DATETIME NOT NULL, "
                "UPDATE_BY TEXT NOT NULL, "
                "ARCHIVE_DATE DATETIME NOT NULL)"
            )
        )
    return engine


def _definition_payload(name: str, method_type: int = 1) -> dict:
    config = {
        "num_regimes": 3,
        "return_basis": "total",
        "min_observations": 40,
        "pca_standardize": True,
        "vol_scaler": 0.0,
        "benchmark_assignments": {},
        "long_short_assignments": {},
        "vol_scaling_assignments": {},
    }
    if method_type in {1, 2}:
        config["universe_series"] = ["Asset_A", "Asset_B"]
    else:
        config["single_series"] = "Asset_A"
    return {
        "RegimeName": name,
        "Description": f"{name} description",
        "MethodType": method_type,
        "Config": config,
    }


def test_regime_timestamp_compare_handles_timezone_and_fractional_precision():
    assert regime_defs._timestamps_equal("2026-02-25 12:34:56", "2026-02-25T12:34:56Z")
    assert regime_defs._timestamps_equal("2026-02-25 12:34:56.123", "2026-02-25 12:34:56")


def test_regime_rowcount_helpers_handle_unknown_sql_server_values():
    assert regime_defs._rowcount_is_known_miss(0) is True
    assert regime_defs._rowcount_is_known_miss(-1) is False
    assert regime_defs._rowcount_is_unknown(-1) is True
    assert regime_defs._rowcount_is_unknown(None) is True
    assert regime_defs._rowcount_is_unknown(1) is False


def test_regime_lookup_by_name_falls_back_to_trimmed_case_insensitive_match():
    db_engine = _seed_db_engine()
    with db_engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO RegimeDefinitions ("
                "RegimeName, Description, MethodType, ConfigJson, UPDATE_DATE, UPDATE_BY"
                ") VALUES ("
                ":name, :description, :method_type, :config_json, :update_date, :update_by"
                ")"
            ),
            {
                "name": "CycleA  ",
                "description": "desc",
                "method_type": 1,
                "config_json": '{"num_regimes":3,"return_basis":"total","min_observations":40,"pca_standardize":true}',
                "update_date": "2026-02-25 00:00:00",
                "update_by": "tester",
            },
        )
    with db_engine.connect() as conn:
        row = regime_defs._load_definition_row_by_name(conn, db_engine, "cyclea")
    assert row is not None
    assert str(row.get("RegimeName", "")).strip().lower() == "cyclea"


def test_regime_definition_matches_payload_ignores_update_timestamp():
    payload = _definition_payload("CycleA", method_type=1)
    normalized, error = validate_regime_definition_payload(payload)
    assert error is None and normalized is not None
    row = {
        "RegimeName": "CycleA",
        "Description": normalized.get("Description"),
        "MethodType": normalized.get("MethodType"),
        "ConfigJson": normalized.get("ConfigJson"),
        "UPDATE_DATE": "2020-01-01 00:00:00",
        "UPDATE_BY": "x",
    }
    assert regime_defs._regime_definition_matches_payload(row, normalized, "CycleA") is True


def test_validate_regime_definition_payload():
    normalized, error = validate_regime_definition_payload(_definition_payload("QCycle", method_type=1))
    assert error is None
    assert normalized is not None
    assert normalized["RegimeName"] == "QCycle"
    assert normalized["MethodType"] == 1
    assert normalized["Config"]["num_regimes"] == 3


def test_save_update_delete_regime_definition_archives_versions():
    db_engine = _seed_db_engine()
    assert regime_tables_available(db_engine) is True

    ok, _msg, saved = save_regime_definition(
        db_engine,
        _definition_payload("CycleA", method_type=1),
        update_by="Admin:tester",
    )
    assert ok is True
    assert saved is not None

    update_ok, _msg, updated = save_regime_definition(
        db_engine,
        _definition_payload("CycleA", method_type=3),
        update_by="Admin:tester",
        original_name="CycleA",
        expected_update_date=saved["UPDATE_DATE"],
    )
    assert update_ok is True
    assert updated is not None
    assert updated["MethodType"] == 3

    delete_ok, _msg = delete_regime_definition(
        db_engine,
        "CycleA",
        expected_update_date=updated["UPDATE_DATE"],
    )
    assert delete_ok is True

    with db_engine.connect() as conn:
        live_count = conn.execute(text("SELECT COUNT(*) FROM RegimeDefinitions")).scalar_one()
        archive_count = conn.execute(text("SELECT COUNT(*) FROM RegimeDefinitionsArchive")).scalar_one()
    assert live_count == 0
    assert archive_count == 2


def test_case_insensitive_regime_name_uniqueness():
    db_engine = _seed_db_engine()
    ok, _msg, _saved = save_regime_definition(
        db_engine,
        _definition_payload("MacroCycle", method_type=2),
        update_by="Admin:tester",
    )
    assert ok is True

    dup_ok, dup_msg, _dup_saved = save_regime_definition(
        db_engine,
        _definition_payload("macrocycle", method_type=2),
        update_by="Admin:tester",
    )
    assert dup_ok is False
    assert "already exists" in dup_msg.lower()


def test_load_regime_definitions_returns_normalized_rows():
    db_engine = _seed_db_engine()
    save_regime_definition(
        db_engine,
        _definition_payload("LoadedCycle", method_type=1),
        update_by="Admin:tester",
    )

    rows = load_regime_definitions(db_engine)
    assert len(rows) == 1
    assert rows[0]["RegimeName"] == "LoadedCycle"
    assert rows[0]["source"] == "db"
