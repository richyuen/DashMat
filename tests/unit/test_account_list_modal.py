from __future__ import annotations

import pytest
from dash import no_update
from dash.exceptions import PreventUpdate

import utils.account_list_modal as modal_module


def test_account_list_loader_visible_only_for_loading_status():
    assert modal_module.build_account_list_load_state("loading", 3) == {"status": "loading", "token": 3}
    assert modal_module.account_list_loader_visible({"status": "loading", "token": 3}) is True
    assert modal_module.account_list_loader_visible({"status": "error", "token": 3}) is False
    assert modal_module.account_list_loader_visible(None) is False
    assert modal_module.account_list_loader_wrapper_style({"status": "idle", "token": None}) == {"display": "none"}
    assert modal_module.account_list_loader_wrapper_style({"status": "loading", "token": 3}) == {
        "position": "fixed",
        "inset": 0,
        "zIndex": 4100,
    }


def test_load_selected_account_list_session_requires_click():
    with pytest.raises(PreventUpdate):
        modal_module.load_selected_account_list_session(
            n_clicks=0,
            selected_id=1,
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
    assert load_state == {"status": "error", "token": 2}


def test_load_selected_account_list_session_handles_missing_row(monkeypatch):
    monkeypatch.setattr(modal_module, "load_account_list_by_id", lambda *_args, **_kwargs: None)

    payload, notice, load_state = modal_module.load_selected_account_list_session(
        n_clicks=4,
        selected_id=8,
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
    assert load_state == {"status": "error", "token": 4}


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
    assert load_state == {"status": "error", "token": 5}


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
    assert load_state == {"status": "success", "token": 6}
