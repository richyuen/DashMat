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
            apply_settings=True,
            raw_data=None,
            original_periodicity="daily",
            provenance_store={},
            session_snapshot={},
            userinfo={"username": "tester"},
            db_engine=None,
            mrd_engine=None,
            perf_engine=None,
        )


def test_load_selected_account_list_session_handles_missing_selection():
    payload, notice, load_state = modal_module.load_selected_account_list_session(
        n_clicks=2,
        selected_id=None,
        apply_settings=True,
        raw_data=None,
        original_periodicity="daily",
        provenance_store={},
        session_snapshot={},
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
        apply_settings=True,
        raw_data=None,
        original_periodicity="daily",
        provenance_store={},
        session_snapshot={},
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
        apply_settings=False,
        raw_data=None,
        original_periodicity="daily",
        provenance_store={},
        session_snapshot={},
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
    monkeypatch.setattr(
        modal_module,
        "build_account_list_session_payload",
        lambda **_kwargs: ({"dashmat-raw-data-store": "json"}, {"added_series": ["A"]}),
    )

    payload, notice, load_state = modal_module.load_selected_account_list_session(
        n_clicks=6,
        selected_id=10,
        apply_settings=True,
        raw_data=None,
        original_periodicity="daily",
        provenance_store={},
        session_snapshot={},
        userinfo={"username": "tester"},
        db_engine=None,
        mrd_engine=None,
        perf_engine=None,
    )

    assert payload == {"dashmat-raw-data-store": "json"}
    assert notice is no_update
    assert load_state == {"status": "success"}


def test_load_selected_account_list_session_repeated_failures_return_same_error_state(monkeypatch):
    monkeypatch.setattr(modal_module, "load_account_list_by_id", lambda *_args, **_kwargs: None)

    first = modal_module.load_selected_account_list_session(
        n_clicks=7,
        selected_id=11,
        apply_settings=True,
        raw_data=None,
        original_periodicity="daily",
        provenance_store={},
        session_snapshot={},
        userinfo={"username": "tester"},
        db_engine=None,
        mrd_engine=None,
        perf_engine=None,
    )
    second = modal_module.load_selected_account_list_session(
        n_clicks=8,
        selected_id=11,
        apply_settings=True,
        raw_data=None,
        original_periodicity="daily",
        provenance_store={},
        session_snapshot={},
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
