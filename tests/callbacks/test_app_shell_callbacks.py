from __future__ import annotations

import pytest
from dash.exceptions import PreventUpdate


def test_update_app_nav_links_for_test_role():
    import app as app_module

    home_href, analytics_href, portopt_href, regression_href = app_module.update_app_nav_links({"role": "Test"})
    assert home_href == "/"
    assert analytics_href == "/restricted?target=Analytics%20Tool"
    assert portopt_href == "/restricted?target=Portfolio%20Optimization"
    assert regression_href == "/restricted?target=Regression%20Analysis"


def test_update_app_nav_links_for_non_test_role():
    import app as app_module

    home_href, analytics_href, portopt_href, regression_href = app_module.update_app_nav_links({"role": "Admin"})
    assert home_href == "/"
    assert analytics_href == "/analyticstool"
    assert portopt_href == "/portopt"
    assert regression_href == "/regression"


def test_restricted_href_for_path_resolves_for_test_role():
    import app as app_module

    assert app_module._restricted_href_for_path("/analyticstool", {"role": "Test"}) == (
        "/restricted?target=Analytics%20Tool"
    )
    assert app_module._restricted_href_for_path("/portopt", {"role": "Test"}) == (
        "/restricted?target=Portfolio%20Optimization"
    )
    assert app_module._restricted_href_for_path("/regression", {"role": "Test"}) == (
        "/restricted?target=Regression%20Analysis"
    )


def test_restricted_href_for_path_skips_non_test_or_other_paths():
    import app as app_module

    assert app_module._restricted_href_for_path("/analyticstool", {"role": "Admin"}) is None
    assert app_module._restricted_href_for_path("/regression", {"role": "Admin"}) is None
    assert app_module._restricted_href_for_path("/", {"role": "Test"}) is None


def test_guard_protected_pages_redirects_or_prevent_update():
    import app as app_module

    assert app_module.guard_protected_pages("/analyticstool", {"role": "Test"}) == (
        "/restricted?target=Analytics%20Tool"
    )
    assert app_module.guard_protected_pages("/regression", {"role": "Test"}) == (
        "/restricted?target=Regression%20Analysis"
    )
    with pytest.raises(PreventUpdate):
        app_module.guard_protected_pages("/", {"role": "Test"})


def test_app_shell_hosts_shared_account_list_modal_and_store():
    from pathlib import Path

    app_text = Path("app.py").read_text(encoding="utf-8")
    account_list_text = Path("utils/account_list_modal.py").read_text(encoding="utf-8")
    welcome_text = Path("utils/dashmat_welcome_modal.py").read_text(encoding="utf-8")

    assert 'dcc.Store(id="dashmat-db-import-provenance-store"' in app_text
    assert 'dcc.Store(id="dashmat-account-list-session-apply-store"' not in app_text
    assert 'dcc.Store(id="dashmat-account-list-load-state-store"' not in app_text
    assert 'dcc.Store(id="dashmat-account-list-selected-detail-store"' not in app_text
    assert 'dcc.Store(id="dashmat-account-list-load-snapshot-store"' not in app_text
    assert "build_account_list_components" in app_text
    assert "register_account_list_callbacks" in app_text
    assert 'dcc.Store(id="dashmat-account-list-session-apply-store"' in account_list_text
    assert 'dcc.Store(id="dashmat-account-list-load-state-store"' in account_list_text
    assert 'dcc.Store(id="dashmat-account-list-selected-detail-store"' in account_list_text
    assert 'dcc.Store(id="dashmat-account-list-load-snapshot-store"' in account_list_text
    assert 'id="dashmat-account-list-modal"' in account_list_text
    assert 'id="dashmat-account-list-load-overlay"' in account_list_text
    assert 'id="dashmat-account-list-load-overlay-shell"' in account_list_text
    assert 'id="dashmat-account-list-grid"' in account_list_text
    assert 'id="dashmat-account-list-preview-grid"' in account_list_text
    assert 'id="dashmat-account-list-send-user-select"' in account_list_text
    assert 'id="dashmat-account-list-send-button"' in account_list_text
    assert 'dashmat-series-modal-grid' in account_list_text
    assert 'ACCOUNT_LIST_MODAL_LOAD_CLASS' in account_list_text
    assert 'Output("dashmat-account-list-selected-detail-store", "data")' in account_list_text
    assert 'Output("dashmat-account-list-load-snapshot-store", "data")' in account_list_text
    assert 'Output("dashmat-account-list-load-state-store", "data", allow_duplicate=True)' in account_list_text
    assert 'Output("dashmat-account-list-load-overlay", "visible")' in account_list_text
    assert 'Output("dashmat-account-list-load-overlay-shell", "style")' in account_list_text
    assert 'Input("dashmat-account-list-load-button", "n_clicks")' in account_list_text
    assert 'Input("dashmat-account-list-modal", "opened")' in account_list_text
    assert 'Input("dashmat-account-list-selected-id-store", "data")' in account_list_text
    assert '"welcome-load-account-list-btn"' in welcome_text
    assert '"Load Account List"' in welcome_text
    assert 'Input("at-welcome-load-account-list-btn", "n_clicks", allow_optional=True)' in account_list_text
    assert 'Input("po-welcome-load-account-list-btn", "n_clicks", allow_optional=True)' in account_list_text
    assert 'Input("reg-welcome-load-account-list-btn", "n_clicks", allow_optional=True)' in account_list_text
    assert 'sessionStorage.setItem(entry[0], entry[1]);' in account_list_text
    assert 'timing name=account_list.load_snapshot_capture' in account_list_text
    assert 'timing name=account_list.session_apply' in account_list_text
    assert 'changed_key_count=%s' in account_list_text
    assert 'window.location.reload();' in account_list_text
    assert "configure_timing_logger" in app_text


def test_account_list_modal_does_not_open_when_optional_menu_input_mounts(monkeypatch):
    from utils import account_list_modal as modal_module

    with pytest.raises(PreventUpdate):
        modal_module.toggle_account_list_modal(
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            triggered_id="at-menu-load-account-list",
        )


def test_account_list_load_view_keeps_static_grid_structure():
    from utils.account_list_modal import ACCOUNT_LIST_MODAL_LOAD_CLASS, render_account_list_modal_view

    rows = [
        {
            "AccountListID": 1,
            "ListName": "Seed",
            "UPDATE_DATE": "2026-01-01 10:00:00",
            "UPDATE_BY": "seed",
            "SeriesCount": None,
        }
    ]
    selected_detail = {
        "AccountListID": 1,
        "ConfigJson": {
            "series_entries": [
                {
                    "entry_id": "seed-1",
                    "loader_type": "cma_bench",
                    "loader_args": {"selected_benches": ["SPX_TRIndex"]},
                    "emitted_series": ["SPX_TRIndex"],
                    "primary_series": "SPX_TRIndex",
                }
            ],
            "control_values": {"at-series-select": ["SPX_TRIndex"]},
        },
    }

    out_empty = render_account_list_modal_view(True, "load", rows, None, None)
    out_selected = render_account_list_modal_view(True, "load", rows, 1, selected_detail)

    assert out_empty[1] == ACCOUNT_LIST_MODAL_LOAD_CLASS
    assert out_selected[1] == ACCOUNT_LIST_MODAL_LOAD_CLASS
    assert out_empty[3] == {}
    assert out_selected[3] == {}
    assert out_empty[4][0]["ListName"] == "Seed"
    assert out_empty[5] == []
    assert out_selected[5][0]["Series"] == "SPX_TRIndex"
