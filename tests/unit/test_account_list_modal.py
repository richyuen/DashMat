from __future__ import annotations

import pytest
from dash import no_update
from dash.exceptions import PreventUpdate

import utils.account_list_modal as modal_module


def test_account_list_loader_visible_only_for_loading_status():
    assert modal_module.build_account_list_load_state("loading") == {"status": "loading"}
    assert modal_module.account_list_loader_visible({"status": "loading"}) is True
    assert modal_module.account_list_loader_visible({"status": "error"}) is False
    assert modal_module.account_list_loader_visible(None) is False
    assert modal_module.account_list_loader_wrapper_style({"status": "idle"}) == {"display": "none"}
    assert modal_module.account_list_loader_wrapper_style({"status": "loading"}) == {
        "position": "fixed",
        "inset": 0,
        "zIndex": 4100,
    }


def test_load_selected_account_list_session_requires_click():
    with pytest.raises(PreventUpdate):
        modal_module.load_selected_account_list_session(
            n_clicks=0,
            selected_id=1,
            selected_detail=None,
            rows=[],
            apply_settings=True,
            raw_data=None,
            original_periodicity="daily",
            provenance_store={},
            load_snapshot={},
            userinfo={"username": "tester"},
            db_engine=None,
            mrd_engine=None,
            perf_engine=None,
        )


def test_load_selected_account_list_session_handles_missing_selection():
    payload, notice, load_state = modal_module.load_selected_account_list_session(
        n_clicks=2,
        selected_id=None,
        selected_detail=None,
        rows=[],
        apply_settings=True,
        raw_data=None,
        original_periodicity="daily",
        provenance_store={},
        load_snapshot={},
        userinfo={"username": "tester"},
        db_engine=None,
        mrd_engine=None,
        perf_engine=None,
    )

    assert payload is no_update
    assert notice == {"message": "Select an account list to load.", "color": "orange"}
    assert load_state == {"status": "error"}


def test_load_selected_account_list_session_handles_missing_row(monkeypatch):
    monkeypatch.setattr(modal_module, "load_account_list_by_id", lambda *_args, **_kwargs: None)

    payload, notice, load_state = modal_module.load_selected_account_list_session(
        n_clicks=4,
        selected_id=8,
        selected_detail=None,
        rows=[],
        apply_settings=True,
        raw_data=None,
        original_periodicity="daily",
        provenance_store={},
        load_snapshot={},
        userinfo={"username": "tester"},
        db_engine=None,
        mrd_engine=None,
        perf_engine=None,
    )

    assert payload is no_update
    assert notice == {"message": "Saved account list no longer exists.", "color": "red"}
    assert load_state == {"status": "error"}


def test_load_selected_account_list_session_handles_loader_error(monkeypatch):
    monkeypatch.setattr(
        modal_module,
        "load_account_list_by_id",
        lambda *_args, **_kwargs: {"ConfigJson": {"series_entries": [{"entry_id": "x"}]}},
    )

    def raise_build_error(**_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(modal_module, "build_account_list_session_payload", raise_build_error)

    payload, notice, load_state = modal_module.load_selected_account_list_session(
        n_clicks=5,
        selected_id=9,
        selected_detail=None,
        rows=[],
        apply_settings=False,
        raw_data=None,
        original_periodicity="daily",
        provenance_store={},
        load_snapshot={},
        userinfo={"username": "tester"},
        db_engine=None,
        mrd_engine=None,
        perf_engine=None,
    )

    assert payload is no_update
    assert notice == {"message": "Unable to load account list: boom", "color": "red"}
    assert load_state == {"status": "error"}


def test_load_selected_account_list_session_reports_success(monkeypatch):
    monkeypatch.setattr(
        modal_module,
        "load_account_list_by_id",
        lambda *_args, **_kwargs: {"ConfigJson": {"series_entries": [{"entry_id": "x"}]}},
    )

    seen = {}

    def fake_build(**kwargs):
        seen["snapshot"] = kwargs["current_session_snapshot"]
        return {"dashmat-raw-data-store": "json"}, {"added_series": ["A"]}

    monkeypatch.setattr(
        modal_module,
        "build_account_list_session_payload",
        fake_build,
    )

    payload, notice, load_state = modal_module.load_selected_account_list_session(
        n_clicks=6,
        selected_id=10,
        selected_detail=None,
        rows=[],
        apply_settings=True,
        raw_data=None,
        original_periodicity="daily",
        provenance_store={},
        load_snapshot={"at-series-select": ["A"], "unused-store": 1},
        userinfo={"username": "tester"},
        db_engine=None,
        mrd_engine=None,
        perf_engine=None,
    )

    assert payload == {"dashmat-raw-data-store": "json"}
    assert notice is no_update
    assert load_state == {"status": "success"}
    assert seen["snapshot"] == {"at-series-select": ["A"]}


def test_load_selected_account_list_session_repeated_failures_return_same_error_state(monkeypatch):
    monkeypatch.setattr(modal_module, "load_account_list_by_id", lambda *_args, **_kwargs: None)

    first = modal_module.load_selected_account_list_session(
        n_clicks=7,
        selected_id=11,
        selected_detail=None,
        rows=[],
        apply_settings=True,
        raw_data=None,
        original_periodicity="daily",
        provenance_store={},
        load_snapshot={},
        userinfo={"username": "tester"},
        db_engine=None,
        mrd_engine=None,
        perf_engine=None,
    )
    second = modal_module.load_selected_account_list_session(
        n_clicks=8,
        selected_id=11,
        selected_detail=None,
        rows=[],
        apply_settings=True,
        raw_data=None,
        original_periodicity="daily",
        provenance_store={},
        load_snapshot={},
        userinfo={"username": "tester"},
        db_engine=None,
        mrd_engine=None,
        perf_engine=None,
    )

    assert first[2] == {"status": "error"}
    assert second[2] == {"status": "error"}
    assert first[1] == second[1] == {"message": "Saved account list no longer exists.", "color": "red"}


def test_account_list_update_by_uses_username_only():
    assert modal_module._account_list_update_by({"role": "Admin", "username": "tester"}) == "tester"
    assert modal_module._account_list_update_by({"role": "Admin"}) == "unknown"


def test_account_list_send_user_options_and_control_state():
    options = modal_module.account_list_send_user_options(
        [{"Username": "alice", "Role": "Analyst"}, {"Username": "bob", "Role": "Viewer"}]
    )

    assert options == [
        {"label": "alice", "value": "alice"},
        {"label": "bob", "value": "bob"},
    ]

    hidden_state = modal_module.account_list_send_controls_state("save", None, options, None)
    empty_state = modal_module.account_list_send_controls_state("load", None, [], None)
    ready_state = modal_module.account_list_send_controls_state("load", 1, options, "alice")

    assert hidden_state[0] == {"display": "none"}
    assert empty_state == ({}, True, True, "No other users available")
    assert ready_state == ({}, False, False, "Select a user")


def test_resolve_selected_account_list_detail_reuses_matching_detail(monkeypatch):
    reused_detail = {
        "AccountListID": 7,
        "UPDATE_DATE": "2026-03-20 09:00:00",
        "ConfigJson": {"series_entries": []},
    }

    monkeypatch.setattr(
        modal_module,
        "load_selected_account_list_detail",
        lambda **_kwargs: pytest.fail("detail should be reused"),
    )

    resolved = modal_module.resolve_selected_account_list_detail(
        selected_id=7,
        selected_detail=reused_detail,
        rows=[{"AccountListID": 7, "UPDATE_DATE": "2026-03-20 09:00:00"}],
        userinfo={"username": "tester"},
        db_engine=None,
    )

    assert resolved == reused_detail


def test_resolve_selected_account_list_detail_refetches_when_row_version_changes(monkeypatch):
    monkeypatch.setattr(
        modal_module,
        "load_selected_account_list_detail",
        lambda **_kwargs: {"AccountListID": 7, "UPDATE_DATE": "2026-03-20 10:00:00", "ConfigJson": {}},
    )

    resolved = modal_module.resolve_selected_account_list_detail(
        selected_id=7,
        selected_detail={"AccountListID": 7, "UPDATE_DATE": "2026-03-20 09:00:00", "ConfigJson": {}},
        rows=[{"AccountListID": 7, "UPDATE_DATE": "2026-03-20 10:00:00"}],
        userinfo={"username": "tester"},
        db_engine=None,
    )

    assert resolved["UPDATE_DATE"] == "2026-03-20 10:00:00"


def test_render_selected_account_list_preview_parses_selected_detail():
    preview_rows = modal_module.render_selected_account_list_preview(
        {
            "AccountListID": 4,
            "ConfigJson": {
                "series_entries": [
                    {
                        "entry_id": "row-1",
                        "loader_type": "cma_bench",
                        "loader_args": {"selected_benches": ["SPX_TRIndex"]},
                        "emitted_series": ["SPX_TRIndex"],
                        "primary_series": "SPX_TRIndex",
                    }
                ],
                "control_values": {"at-series-select": ["SPX_TRIndex"]},
            },
        }
    )

    assert preview_rows == [
        {"Series": "SPX_TRIndex", "SourceType": "cma_bench", "AT": True, "PO": False, "REG": False}
    ]


def test_normalize_account_list_load_snapshot_keeps_only_merge_keys():
    snapshot = modal_module.normalize_account_list_load_snapshot(
        {
            "at-series-select": ["A"],
            "po-series-select": ["B"],
            "reg-series-select": ["C"],
            "unrelated-store": 123,
        }
    )

    assert snapshot == {
        "at-series-select": ["A"],
        "po-series-select": ["B"],
        "reg-series-select": ["C"],
    }


def test_account_list_load_merge_store_ids_count_is_expected():
    assert len(modal_module.ACCOUNT_LIST_LOAD_MERGE_STORE_IDS) == 24


def test_prefetch_selected_account_list_entries_resets_when_modal_not_ready():
    reset_state = modal_module.prefetch_selected_account_list_entries(
        opened=False,
        mode="load",
        selected_detail={"AccountListID": 1, "UPDATE_DATE": "2026-03-20 12:00:00", "ConfigJson": {}},
        current_prefetch={"account_list_id": 1, "update_date": "2026-03-20 12:00:00", "status": "ready"},
        db_engine=None,
        mrd_engine=None,
        perf_engine=None,
    )

    assert reset_state == {"account_list_id": None, "update_date": None, "status": "idle"}


def test_prefetch_selected_account_list_entries_dedupes_same_selected_version(monkeypatch):
    monkeypatch.setattr(
        modal_module,
        "prefetch_account_list_entry_frames",
        lambda *args, **kwargs: pytest.fail("prefetch should be deduped"),
    )

    result = modal_module.prefetch_selected_account_list_entries(
        opened=True,
        mode="load",
        selected_detail={"AccountListID": 2, "UPDATE_DATE": "2026-03-20 12:00:00", "ConfigJson": {}},
        current_prefetch={"account_list_id": 2, "update_date": "2026-03-20 12:00:00", "status": "ready"},
        db_engine=None,
        mrd_engine=None,
        perf_engine=None,
    )

    assert result is no_update


def test_prefetch_selected_account_list_entries_marks_ready(monkeypatch):
    seen = {}

    def fake_prefetch(payload, **_kwargs):
        seen["payload"] = payload
        return {"warmed_count": 1}

    monkeypatch.setattr(modal_module, "prefetch_account_list_entry_frames", fake_prefetch)

    result = modal_module.prefetch_selected_account_list_entries(
        opened=True,
        mode="load",
        selected_detail={"AccountListID": 3, "UPDATE_DATE": "2026-03-20 12:00:00", "ConfigJson": {"series_entries": []}},
        current_prefetch=None,
        db_engine=None,
        mrd_engine=None,
        perf_engine=None,
    )

    assert seen["payload"] == {"series_entries": []}
    assert result == {"account_list_id": 3, "update_date": "2026-03-20 12:00:00", "status": "ready"}


def test_prefetch_selected_account_list_entries_marks_error(monkeypatch):
    monkeypatch.setattr(
        modal_module,
        "prefetch_account_list_entry_frames",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    result = modal_module.prefetch_selected_account_list_entries(
        opened=True,
        mode="load",
        selected_detail={"AccountListID": 4, "UPDATE_DATE": "2026-03-20 12:00:00", "ConfigJson": {}},
        current_prefetch=None,
        db_engine=None,
        mrd_engine=None,
        perf_engine=None,
    )

    assert result == {"account_list_id": 4, "update_date": "2026-03-20 12:00:00", "status": "error"}
