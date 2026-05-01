from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine

import utils.account_lists as account_lists
from tools.db.migrate_account_lists import ensure_account_list_tables
from tools.db.migrate_users import ensure_users_table
from utils.account_lists import (
    AT_STORE_IDS,
    REG_STORE_IDS,
    add_db_import_provenance_entry,
    build_account_list_payload,
    build_account_list_session_payload,
    delete_account_list,
    list_account_lists,
    list_account_list_users,
    load_account_list_by_id,
    normalize_account_list_payload,
    normalize_db_import_provenance_store,
    save_account_list,
    send_account_list,
)
from utils.returns import df_to_json, json_to_df


def _seed_db_engine():
    engine = create_engine("sqlite:///:memory:", future=True)
    ensure_account_list_tables(engine)
    ensure_users_table(engine)
    return engine


def test_ensure_users_table_is_idempotent():
    engine = create_engine("sqlite:///:memory:", future=True)

    first = ensure_users_table(engine)
    second = ensure_users_table(engine)

    assert first == {"created_users": True}
    assert second == {"created_users": False}


def test_build_account_list_payload_filters_to_db_backed_series():
    provenance = add_db_import_provenance_entry(
        {},
        loader_type="cma_bench",
        loader_args={"selected_benches": ["SPX_TRIndex"]},
        emitted_series=["SPX_TRIndex"],
    )
    snapshot = {
        AT_STORE_IDS["selected"]: ["SPX_TRIndex", "UploadedOnly"],
        AT_STORE_IDS["order"]: ["UploadedOnly", "SPX_TRIndex"],
        AT_STORE_IDS["bench"]: {"SPX_TRIndex": "None", "UploadedOnly": "None"},
        "at-correlation-controls-store": {"view": "pca", "pca_basis": "covariance"},
        REG_STORE_IDS["dep"]: "UploadedOnly",
    }

    payload = build_account_list_payload(provenance, snapshot)
    normalized = normalize_account_list_payload(payload)

    assert normalized["control_values"][AT_STORE_IDS["selected"]] == ["SPX_TRIndex", "UploadedOnly"]
    assert normalized["control_values"][AT_STORE_IDS["order"]] == ["UploadedOnly", "SPX_TRIndex"]
    assert normalized["control_values"]["at-correlation-controls-store"] == {
        "view": "pca",
        "pca_basis": "covariance",
    }
    assert normalized["control_values"][REG_STORE_IDS["dep"]] == "UploadedOnly"


def test_build_account_list_payload_replaces_latest_end_with_sentinel(monkeypatch):
    provenance = add_db_import_provenance_entry(
        {},
        loader_type="cma_bench",
        loader_args={"selected_benches": ["SPX_TRIndex"]},
        emitted_series=["SPX_TRIndex"],
    )
    snapshot = {
        AT_STORE_IDS["selected"]: ["SPX_TRIndex"],
        "at-periodicity-value-store": "daily_trading",
        "at-date-range-store": {"start": "2024-01-15", "end": "2024-03-29"},
    }
    raw_data_store = {"dataset_key": "raw-key"}

    monkeypatch.setattr(account_lists, "resolve_dataset_key", lambda _raw_data: "raw-key")
    monkeypatch.setattr(
        account_lists,
        "compute_date_range_candidates",
        lambda dataset_key, periodicity, selected: {
            "max_start": "2024-01-01",
            "max_end": "2024-03-29",
        },
    )

    payload = build_account_list_payload(provenance, snapshot, raw_data_store)

    assert payload["control_values"]["at-date-range-store"]["start"] == "2024-01-15"
    assert payload["control_values"]["at-date-range-store"]["end"] == account_lists.ACCOUNT_LIST_MAX_END_SENTINEL


def test_save_list_load_and_delete_account_list_support_duplicate_names():
    db_engine = _seed_db_engine()
    with db_engine.begin() as conn:
        conn.exec_driver_sql(
            "INSERT INTO Users (Username, Role) VALUES ('tester', 'Admin'), ('recipient', 'Analyst')"
        )
    provenance = add_db_import_provenance_entry(
        {},
        loader_type="cma_bench",
        loader_args={"selected_benches": ["SPX_TRIndex"]},
        emitted_series=["SPX_TRIndex"],
    )
    payload = build_account_list_payload(provenance, {AT_STORE_IDS["selected"]: ["SPX_TRIndex"]})

    ok1, _msg1, saved1 = save_account_list(
        db_engine,
        username="tester",
        update_by="tester",
        list_name="My List",
        payload=payload,
    )
    ok2, _msg2, saved2 = save_account_list(
        db_engine,
        username="tester",
        update_by="tester",
        list_name="My List",
        payload=payload,
    )

    assert ok1 is True
    assert ok2 is True
    assert saved1 is not None and saved2 is not None
    assert saved1["AccountListID"] != saved2["AccountListID"]

    rows = list_account_lists(db_engine, "tester")
    assert len(rows) == 2
    assert rows[0]["ListName"] == "My List"
    assert "ConfigJson" not in rows[0]
    assert rows[0]["SeriesCount"] is None

    loaded = load_account_list_by_id(db_engine, saved1["AccountListID"], "tester")
    assert loaded is not None
    assert "ConfigJson" in loaded
    assert loaded["UPDATE_BY"] == "tester"

    delete_ok, _delete_msg = delete_account_list(
        db_engine,
        account_list_id=saved1["AccountListID"],
        username="tester",
        expected_update_date=saved1["UPDATE_DATE"],
    )
    assert delete_ok is True
    assert len(list_account_lists(db_engine, "tester")) == 1


def test_list_account_list_users_excludes_current_username():
    db_engine = _seed_db_engine()
    with db_engine.begin() as conn:
        conn.exec_driver_sql(
            "INSERT INTO Users (Username, Role) VALUES ('tester', 'Admin'), ('alice', 'Analyst'), ('bob', 'Viewer')"
        )

    assert list_account_list_users(db_engine, "tester") == [
        {"Username": "alice", "Role": "Analyst"},
        {"Username": "bob", "Role": "Viewer"},
    ]


def test_list_account_lists_does_not_require_config_json_parsing():
    db_engine = _seed_db_engine()
    with db_engine.begin() as conn:
        conn.exec_driver_sql("INSERT INTO Users (Username, Role) VALUES ('tester', 'Admin')")
        conn.exec_driver_sql(
            """
            INSERT INTO DMAccountLists (Username, ListName, ConfigJson, UPDATE_DATE, UPDATE_BY)
            VALUES ('tester', 'Broken Config', 'not-json', '2026-03-20 10:00:00', 'tester')
            """
        )

    rows = list_account_lists(db_engine, "tester")

    assert rows == [
        {
            "AccountListID": 1,
            "Username": "tester",
            "ListName": "Broken Config",
            "UPDATE_DATE": "2026-03-20 10:00:00",
            "UPDATE_BY": "tester",
            "SeriesCount": None,
        }
    ]


def test_load_entry_frame_uses_short_lived_cache(monkeypatch):
    calls = {"count": 0}
    now = {"value": 100.0}
    frame = pd.DataFrame({"SPX_TRIndex": [0.01, 0.02]}, index=pd.to_datetime(["2025-01-01", "2025-01-02"]))

    def fake_uncached(entry, **_kwargs):
        calls["count"] += 1
        return frame, "daily"

    monkeypatch.setattr(account_lists, "_load_entry_frame_uncached", fake_uncached)
    monkeypatch.setattr(account_lists, "_entry_frame_cache_now", lambda: now["value"])
    account_lists._clear_account_list_entry_frame_cache()

    entry = {
        "loader_type": "cma_bench",
        "loader_args": {"selected_benches": ["SPX_TRIndex"]},
        "emitted_series": ["SPX_TRIndex"],
    }

    first_df, first_periodicity = account_lists._load_entry_frame(entry, db_engine=None, mrd_engine=None, perf_engine=None)
    second_df, second_periodicity = account_lists._load_entry_frame(entry, db_engine=None, mrd_engine=None, perf_engine=None)

    now["value"] += account_lists.ACCOUNT_LIST_ENTRY_FRAME_CACHE_TTL_SECONDS + 1.0
    third_df, third_periodicity = account_lists._load_entry_frame(entry, db_engine=None, mrd_engine=None, perf_engine=None)

    assert calls["count"] == 2
    assert first_periodicity == second_periodicity == third_periodicity == "daily"
    pd.testing.assert_frame_equal(first_df, second_df)
    pd.testing.assert_frame_equal(first_df, third_df)
    account_lists._clear_account_list_entry_frame_cache()


def test_send_account_list_copies_record_to_recipient():
    db_engine = _seed_db_engine()
    with db_engine.begin() as conn:
        conn.exec_driver_sql(
            "INSERT INTO Users (Username, Role) VALUES ('tester', 'Admin'), ('recipient', 'Analyst')"
        )
    provenance = add_db_import_provenance_entry(
        {},
        loader_type="cma_bench",
        loader_args={"selected_benches": ["SPX_TRIndex"]},
        emitted_series=["SPX_TRIndex"],
    )
    payload = build_account_list_payload(provenance, {AT_STORE_IDS["selected"]: ["SPX_TRIndex"]})
    ok, _message, saved = save_account_list(
        db_engine,
        username="tester",
        update_by="tester",
        list_name="Send Me",
        payload=payload,
    )

    assert ok is True
    assert saved is not None

    sent_ok, sent_message = send_account_list(
        db_engine,
        account_list_id=saved["AccountListID"],
        sender_username="tester",
        recipient_username="recipient",
        expected_update_date=saved["UPDATE_DATE"],
    )

    assert sent_ok is True
    assert "recipient" in sent_message
    recipient_rows = list_account_lists(db_engine, "recipient")
    assert len(recipient_rows) == 1
    assert recipient_rows[0]["ListName"] == "Send Me"
    assert recipient_rows[0]["UPDATE_BY"] == "tester"
    assert len(list_account_lists(db_engine, "tester")) == 1


def test_send_account_list_rejects_self_and_unknown_recipient():
    db_engine = _seed_db_engine()
    with db_engine.begin() as conn:
        conn.exec_driver_sql("INSERT INTO Users (Username, Role) VALUES ('tester', 'Admin')")
    provenance = add_db_import_provenance_entry(
        {},
        loader_type="cma_bench",
        loader_args={"selected_benches": ["SPX_TRIndex"]},
        emitted_series=["SPX_TRIndex"],
    )
    payload = build_account_list_payload(provenance, {AT_STORE_IDS["selected"]: ["SPX_TRIndex"]})
    ok, _message, saved = save_account_list(
        db_engine,
        username="tester",
        update_by="tester",
        list_name="Mine",
        payload=payload,
    )

    assert ok is True
    assert saved is not None

    self_ok, self_message = send_account_list(
        db_engine,
        account_list_id=saved["AccountListID"],
        sender_username="tester",
        recipient_username="tester",
        expected_update_date=saved["UPDATE_DATE"],
    )
    missing_ok, missing_message = send_account_list(
        db_engine,
        account_list_id=saved["AccountListID"],
        sender_username="tester",
        recipient_username="missing",
        expected_update_date=saved["UPDATE_DATE"],
    )

    assert self_ok is False
    assert self_message == "Choose a different user."
    assert missing_ok is False
    assert missing_message == "Selected user no longer exists."


def test_build_account_list_session_payload_skips_conflicts_and_keeps_existing_benchmark(monkeypatch):
    current_df = pd.DataFrame(
        {
            "A": [0.01, 0.02],
            "B": [0.03, 0.04],
        },
        index=pd.to_datetime(["2025-01-31", "2025-02-28"]),
    )
    payload = {
        "schema_version": 2,
        "series_entries": [
            {
                "entry_id": "entry-1",
                "loader_type": "cma_bench",
                "loader_args": {"selected_benches": ["B", "C"]},
                "emitted_series": ["B", "C"],
                "primary_series": "B",
            }
        ],
        "control_values": {
            AT_STORE_IDS["selected"]: ["C"],
            AT_STORE_IDS["order"]: ["C"],
            AT_STORE_IDS["bench"]: {"C": "B"},
            AT_STORE_IDS["long_short"]: {"C": False},
            AT_STORE_IDS["vol"]: {"C": True},
            "at-periodicity-value-store": "monthly",
            "at-partial-period-store": "full",
            "at-correlation-controls-store": {"view": "pca", "pca_basis": "correlation"},
            REG_STORE_IDS["dep"]: "C",
        },
    }

    def fake_load_entry_frame(entry, **_kwargs):
        df = pd.DataFrame(
            {
                "B": [0.05, 0.06],
                "C": [0.07, 0.08],
            },
            index=current_df.index,
        )
        return df, "daily"

    monkeypatch.setattr(account_lists, "_load_entry_frame", fake_load_entry_frame)

    session_payload, stats = build_account_list_session_payload(
        payload=payload,
        current_raw_data=df_to_json(current_df),
        current_original_periodicity="daily",
        current_provenance={},
        current_session_snapshot={
            AT_STORE_IDS["selected"]: ["A"],
            AT_STORE_IDS["order"]: ["A", "B"],
            AT_STORE_IDS["bench"]: {"A": "None"},
            REG_STORE_IDS["dep"]: "A",
        },
        apply_settings=True,
        db_engine=None,  # not used because loader is monkeypatched
        mrd_engine=None,
        perf_engine=None,
    )

    assert stats["added_series"] == ["C"]
    assert stats["skipped_conflicts"] == ["B"]
    assert session_payload[AT_STORE_IDS["selected"]] == ["A", "C"]
    assert session_payload[AT_STORE_IDS["bench"]]["C"] == "B"
    assert session_payload[REG_STORE_IDS["dep"]] == "A"
    assert session_payload["at-periodicity-value-store"] == "monthly"
    assert session_payload["at-partial-period-store"] == "full"
    assert session_payload["at-correlation-controls-store"] == {"view": "pca", "pca_basis": "correlation"}
    normalized_provenance = normalize_db_import_provenance_store(session_payload["dashmat-db-import-provenance-store"])
    assert any("C" in entry["emitted_series"] for entry in normalized_provenance.values())


def test_build_account_list_session_payload_skips_extra_controls_when_apply_settings_is_off(monkeypatch):
    payload = {
        "schema_version": 2,
        "series_entries": [
            {
                "entry_id": "entry-1",
                "loader_type": "cma_bench",
                "loader_args": {"selected_benches": ["C"]},
                "emitted_series": ["C"],
                "primary_series": "C",
            }
        ],
        "control_values": {
            AT_STORE_IDS["selected"]: ["C"],
            AT_STORE_IDS["order"]: ["C"],
            "at-active-tab-store": "conditional_returns",
        },
    }

    monkeypatch.setattr(
        account_lists,
        "_load_entry_frame",
        lambda *_args, **_kwargs: (
            pd.DataFrame({"C": [0.01, 0.02]}, index=pd.to_datetime(["2025-01-31", "2025-02-28"])),
            "daily",
        ),
    )

    session_payload, _stats = build_account_list_session_payload(
        payload=payload,
        current_raw_data=None,
        current_original_periodicity="daily",
        current_provenance={},
        current_session_snapshot={},
        apply_settings=False,
        db_engine=None,
        mrd_engine=None,
        perf_engine=None,
    )

    assert session_payload[AT_STORE_IDS["selected"]] == ["C"]
    assert "at-active-tab-store" not in session_payload


def test_build_account_list_session_payload_skips_redundant_fetch_for_saved_benchmark(monkeypatch):
    payload = {
        "schema_version": 2,
        "series_entries": [
            {
                "entry_id": "bench-only",
                "loader_type": "cma_bench",
                "loader_args": {"selected_benches": ["BM"]},
                "emitted_series": ["BM"],
                "primary_series": "BM",
            },
            {
                "entry_id": "asset-with-bench",
                "loader_type": "raw_performance",
                "loader_args": {"rows": [{"import_name": "Asset", "include_benchmark": True}]},
                "emitted_series": ["Asset", "BM"],
                "primary_series": "Asset",
            },
        ],
        "control_values": {
            AT_STORE_IDS["selected"]: ["Asset", "BM"],
            AT_STORE_IDS["order"]: ["Asset", "BM"],
        },
    }

    calls: list[str] = []

    def fake_load_entry_frame(entry, **_kwargs):
        calls.append(entry["entry_id"])
        if entry["entry_id"] == "asset-with-bench":
            df = pd.DataFrame(
                {"Asset": [0.01, 0.02], "BM": [0.03, 0.04]},
                index=pd.to_datetime(["2025-01-31", "2025-02-28"]),
            )
            return df, "daily"
        raise AssertionError("Redundant benchmark-only entry should not be fetched")

    monkeypatch.setattr(account_lists, "_load_entry_frame", fake_load_entry_frame)

    session_payload, stats = build_account_list_session_payload(
        payload=payload,
        current_raw_data=None,
        current_original_periodicity="daily",
        current_provenance={},
        current_session_snapshot={},
        apply_settings=True,
        db_engine=None,
        mrd_engine=None,
        perf_engine=None,
    )

    assert calls == ["asset-with-bench"]
    assert stats["added_series"] == ["Asset", "BM"]
    assert stats["skipped_conflicts"] == ["BM"]
    loaded_df = json_to_df(session_payload["dashmat-raw-data-store"]["raw_data_json"])
    assert list(loaded_df.columns) == ["Asset", "BM"]
