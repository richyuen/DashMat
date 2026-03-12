from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine

import utils.account_lists as account_lists
from tools.db.migrate_account_lists import ensure_account_list_tables
from utils.account_lists import (
    AT_STORE_IDS,
    REG_STORE_IDS,
    add_db_import_provenance_entry,
    build_account_list_payload,
    build_account_list_session_payload,
    delete_account_list,
    list_account_lists,
    load_account_list_by_id,
    normalize_account_list_payload,
    normalize_db_import_provenance_store,
    save_account_list,
)
from utils.returns import df_to_json


def _seed_db_engine():
    engine = create_engine("sqlite:///:memory:", future=True)
    ensure_account_list_tables(engine)
    return engine


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
        REG_STORE_IDS["dep"]: "UploadedOnly",
    }

    payload = build_account_list_payload(provenance, snapshot)
    normalized = normalize_account_list_payload(payload)

    assert normalized["settings"]["at"]["selected"] == ["SPX_TRIndex"]
    assert normalized["settings"]["at"]["order"] == ["SPX_TRIndex"]
    assert normalized["settings"]["reg"]["dependent_var"] is None


def test_save_list_load_and_delete_account_list_support_duplicate_names():
    db_engine = _seed_db_engine()
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
        update_by="Admin:tester",
        list_name="My List",
        payload=payload,
    )
    ok2, _msg2, saved2 = save_account_list(
        db_engine,
        username="tester",
        update_by="Admin:tester",
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

    loaded = load_account_list_by_id(db_engine, saved1["AccountListID"], "tester")
    assert loaded is not None
    assert loaded["SeriesCount"] == 1

    delete_ok, _delete_msg = delete_account_list(
        db_engine,
        account_list_id=saved1["AccountListID"],
        username="tester",
        expected_update_date=saved1["UPDATE_DATE"],
    )
    assert delete_ok is True
    assert len(list_account_lists(db_engine, "tester")) == 1


def test_build_account_list_session_payload_skips_conflicts_and_keeps_existing_benchmark(monkeypatch):
    current_df = pd.DataFrame(
        {
            "A": [0.01, 0.02],
            "B": [0.03, 0.04],
        },
        index=pd.to_datetime(["2025-01-31", "2025-02-28"]),
    )
    payload = {
        "schema_version": 1,
        "series_entries": [
            {
                "entry_id": "entry-1",
                "loader_type": "cma_bench",
                "loader_args": {"selected_benches": ["B", "C"]},
                "emitted_series": ["B", "C"],
                "primary_series": "B",
            }
        ],
        "settings": {
            "at": {
                "selected": ["C"],
                "order": ["C"],
                "benchmark": {"C": "B"},
                "long_short": {"C": False},
                "scale_vol": {"C": True},
            },
            "po": {
                "selected": [],
                "order": [],
                "benchmark": {},
                "cmabench": {},
                "long_short": {},
                "scale_vol": {},
                "min_wt": {},
                "max_wt": {},
                "force_max": {},
            },
            "reg": {
                "selected": [],
                "order": [],
                "benchmark": {},
                "long_short": {},
                "scale_vol": {},
                "lag": {},
                "min_beta": {},
                "max_beta": {},
                "enable_constraint": {},
                "dependent_var": "C",
            },
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
        db_engine=None,  # not used because loader is monkeypatched
        mrd_engine=None,
        perf_engine=None,
    )

    assert stats["added_series"] == ["C"]
    assert stats["skipped_conflicts"] == ["B"]
    assert session_payload[AT_STORE_IDS["selected"]] == ["A", "C"]
    assert session_payload[AT_STORE_IDS["bench"]]["C"] == "B"
    assert session_payload[REG_STORE_IDS["dep"]] == "A"
    normalized_provenance = normalize_db_import_provenance_store(session_payload["dashmat-db-import-provenance-store"])
    assert any("C" in entry["emitted_series"] for entry in normalized_provenance.values())
