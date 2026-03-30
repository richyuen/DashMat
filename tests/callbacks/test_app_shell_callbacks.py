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


def test_refresh_raw_data_meta_store_uses_identity_store(monkeypatch):
    import app as app_module

    captured = {}

    def fake_builder(dataset_key, original_periodicity):
        captured["dataset_key"] = dataset_key
        captured["original_periodicity"] = original_periodicity
        return {"dataset_key": dataset_key, "original_periodicity": original_periodicity}

    monkeypatch.setattr(app_module, "_build_raw_data_metadata_cached", fake_builder)

    result = app_module.refresh_raw_data_meta_store(
        {"dataset_key": "seed-key", "has_data": True}, "daily", None,
    )

    assert captured == {"dataset_key": "seed-key", "original_periodicity": "daily"}
    assert result == {"dataset_key": "seed-key", "original_periodicity": "daily"}


def test_refresh_raw_data_meta_store_warms_cache_from_raw_data_store(monkeypatch):
    """Cold-cache resilience: the callback re-hydrates the server cache from
    the raw-data-store payload so downstream key-only callbacks succeed."""
    import app as app_module
    from utils.raw_dataset import (
        build_raw_data_store_payload,
        clear_raw_dataset_cache,
        get_raw_dataset_df,
    )
    import pandas as pd

    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    df = pd.DataFrame({"A": [0.01, 0.02, 0.03]}, index=idx)
    payload = build_raw_data_store_payload(df)
    dataset_key = payload["dataset_key"]

    # Simulate cold cache (server restart / eviction)
    clear_raw_dataset_cache()
    from cache_config import cache as cache_proxy
    cache_proxy.clear()

    # Before the fix, get_raw_dataset_df would raise KeyError here
    with pytest.raises(KeyError):
        get_raw_dataset_df(dataset_key)

    # The callback warms the cache from the raw-data-store payload
    monkeypatch.setattr(
        app_module,
        "_build_raw_data_metadata_cached",
        lambda dk, p: {"dataset_key": dk},
    )
    app_module.refresh_raw_data_meta_store(
        {"dataset_key": dataset_key, "has_data": True},
        "daily",
        payload,
    )

    # Now key-only lookup succeeds
    result = get_raw_dataset_df(dataset_key)
    assert list(result.columns) == ["A"]
    assert len(result) == 3


def test_raw_data_meta_store_is_memory_backed():
    """The meta store must be memory-backed so it starts as None on reload,
    preventing a race where stale session data propagates a dataset key
    before the server cache is warm."""
    from pathlib import Path

    app_text = Path("app.py").read_text(encoding="utf-8")
    assert 'dcc.Store(id="dashmat-raw-data-meta-store", data=None, storage_type="memory")' in app_text


def test_shared_saved_series_stamp_store_is_memory_backed():
    from pathlib import Path

    app_text = Path("app.py").read_text(encoding="utf-8")

    assert 'dcc.Store(id="dashmat-saved-series-stamp-store", data=None, storage_type="memory")' in app_text
    assert 'dcc.Store(id="dashmat-saved-series-cache-store", data=None, storage_type="session")' not in app_text
    assert "register_shared_benchmark_callbacks" in app_text


def test_app_shell_hosts_shared_account_list_modal_and_store():
    from pathlib import Path

    app_text = Path("app.py").read_text(encoding="utf-8")
    account_list_text = Path("utils/account_list_modal.py").read_text(encoding="utf-8")
    welcome_text = Path("utils/dashmat_welcome_modal.py").read_text(encoding="utf-8")

    assert 'dcc.Store(id="dashmat-db-import-provenance-store"' in app_text
    assert 'dcc.Store(id="dashmat-raw-data-identity-store"' in app_text
    assert 'dcc.Store(id="dashmat-account-list-session-apply-store"' not in app_text
    assert 'dcc.Store(id="dashmat-account-list-load-state-store"' not in app_text
    assert 'dcc.Store(id="dashmat-account-list-selected-detail-store"' not in app_text
    assert 'dcc.Store(id="dashmat-account-list-load-snapshot-store"' not in app_text
    assert 'dcc.Store(id="dashmat-account-list-prefetch-store"' not in app_text
    assert "build_account_list_components" in app_text
    assert "register_account_list_callbacks" in app_text
    assert 'dcc.Store(id="dashmat-account-list-session-apply-store"' in account_list_text
    assert 'dcc.Store(id="dashmat-account-list-load-state-store"' in account_list_text
    assert 'dcc.Store(id="dashmat-account-list-selected-detail-store"' in account_list_text
    assert 'dcc.Store(id="dashmat-account-list-load-snapshot-store"' in account_list_text
    assert 'dcc.Store(id="dashmat-account-list-prefetch-store"' in account_list_text
    assert 'dcc.Store(id="dashmat-account-list-prefetch-trigger-store"' in account_list_text
    assert 'dcc.Store(id="dashmat-account-list-modal-view-trigger-store"' in account_list_text
    assert 'dcc.Store(id="dashmat-account-list-load-timing-dummy"' in account_list_text
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
    assert 'Output("dashmat-account-list-prefetch-store", "data")' in account_list_text
    assert 'Output("dashmat-account-list-prefetch-trigger-store", "data")' in account_list_text
    assert 'Output("dashmat-account-list-modal-view-trigger-store", "data")' in account_list_text
    assert 'Output("dashmat-account-list-load-snapshot-store", "data")' in account_list_text
    assert 'Output("dashmat-account-list-load-state-store", "data", allow_duplicate=True)' in account_list_text
    assert 'Output("dashmat-account-list-load-timing-dummy", "data")' in account_list_text
    assert 'Output("dashmat-account-list-load-overlay", "visible")' in account_list_text
    assert 'Output("dashmat-account-list-load-overlay-shell", "style")' in account_list_text
    assert 'Input("dashmat-account-list-load-button", "n_clicks")' in account_list_text
    assert 'Input("dashmat-account-list-modal", "opened")' in account_list_text
    assert 'Input("dashmat-account-list-selected-id-store", "data")' in account_list_text
    assert 'Input("dashmat-raw-data-store", "data")' in app_text
    assert 'State("dashmat-raw-data-identity-store", "data")' in app_text
    assert 'Input("dashmat-raw-data-identity-store", "data")' in app_text
    assert '"welcome-load-account-list-btn"' in welcome_text
    assert '"Load Account List"' in welcome_text
    assert 'Input("at-welcome-load-account-list-btn", "n_clicks", allow_optional=True)' in account_list_text
    assert 'Input("po-welcome-load-account-list-btn", "n_clicks", allow_optional=True)' in account_list_text
    assert 'Input("reg-welcome-load-account-list-btn", "n_clicks", allow_optional=True)' in account_list_text
    assert 'sessionStorage.setItem(entry[0], entry[1]);' in account_list_text
    assert 'timing name=account_list.load_snapshot_capture' in account_list_text
    assert '"account_list.prefetch_entry_frames"' in account_list_text
    assert 'timing name=account_list.session_apply' in account_list_text
    assert 'timing name=account_list.click_to_ready' in account_list_text
    assert 'live_apply_analyticstool' in account_list_text
    assert 'live_applying' in account_list_text
    assert 'click_to_live_apply_commit_ms=' in account_list_text
    assert 'changed_key_count=%s' in account_list_text
    assert 'dashmat-account-list-load-timing' in account_list_text
    assert 'State("_pages_location", "pathname")' in account_list_text
    assert 'window.dash_clientside.set_props("at-state-ready-store", {data: false});' in account_list_text
    assert '__AT_STORE_IDS__.forEach(function(id)' in account_list_text
    assert 'window.dash_clientside.set_props(id, {data: changedEntryMap[id]});' in account_list_text
    assert 'State("dashmat-account-list-load-state-store", "data")' in account_list_text
    assert 'window.location.reload();' in account_list_text
    assert "configure_timing_logger" in app_text


def test_shared_saved_series_stamp_contract_matches_pages():
    from pathlib import Path

    analytics_text = Path("pages/analyticstool.py").read_text(encoding="utf-8")
    portopt_text = Path("pages/portopt.py").read_text(encoding="utf-8")
    regression_text = Path("pages/regression.py").read_text(encoding="utf-8")

    assert 'dashmat-saved-series-cache-store' not in analytics_text
    assert 'dashmat-saved-series-stamp-store' in analytics_text
    assert 'dashmat-saved-series-cache-store' not in regression_text
    assert 'dashmat-saved-series-stamp-store' in regression_text
    assert 'dashmat-saved-series-cache-store' not in portopt_text
    assert 'dashmat-saved-series-stamp-store' in portopt_text
    assert 'at-shared-benchmark-stamp-store' not in analytics_text


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
