from __future__ import annotations

import pytest
from dash import no_update
from dash.exceptions import PreventUpdate


def _find_component_by_id(node, target_id):
    if node is None:
        return None
    if getattr(node, "id", None) == target_id:
        return node
    children = getattr(node, "children", None)
    if isinstance(children, (list, tuple)):
        for child in children:
            found = _find_component_by_id(child, target_id)
            if found is not None:
                return found
        return None
    return _find_component_by_id(children, target_id)


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


def test_app_layout_includes_session_and_artifact_stores():
    import app as app_module

    assert _find_component_by_id(app_module.app.layout, "dashmat-session-id-store") is not None
    assert _find_component_by_id(app_module.app.layout, "dashmat-raw-data-artifact-store") is not None
    assert _find_component_by_id(app_module.app.layout, "dashmat-session-export-request-store") is not None
    assert _find_component_by_id(app_module.app.layout, "dashmat-session-import-request-store") is not None
    assert _find_component_by_id(app_module.app.layout, "dashmat-save-session-download") is not None


def test_ensure_dashmat_session_id_reuses_existing_value():
    import app as app_module

    assert app_module.ensure_dashmat_session_id("/analyticstool", "session-123") is no_update


def test_ensure_dashmat_session_id_generates_uuid_when_missing():
    import app as app_module

    value = app_module.ensure_dashmat_session_id("/analyticstool", None)

    assert isinstance(value, str)
    assert len(value) >= 32


def test_refresh_raw_data_artifact_store_returns_none_without_raw_data():
    import app as app_module

    assert app_module.refresh_raw_data_artifact_store(None, "daily", "session-123") is None


def test_refresh_raw_data_artifact_store_delegates_to_artifact_writer(monkeypatch):
    import app as app_module

    expected = {"raw_data_key": "abc", "has_data": True}
    captured = {}

    def fake_store_raw_data_artifact(**kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(app_module, "store_raw_data_artifact", fake_store_raw_data_artifact)

    result = app_module.refresh_raw_data_artifact_store("{}", "daily", "session-123")

    assert result == expected
    assert captured["session_id"] == "session-123"
    assert captured["raw_data_json"] == "{}"
    assert captured["original_periodicity"] == "daily"


def test_export_workspace_session_bundle_returns_download_payload(monkeypatch):
    import app as app_module

    captured = {}

    def fake_build(workspace_session):
        captured["workspace_session"] = workspace_session
        return {"version": 2, "workspace_session": workspace_session, "artifact_refs": [], "artifacts": []}

    monkeypatch.setattr(app_module, "build_workspace_session_bundle", fake_build)

    result = app_module.export_workspace_session_bundle({"workspace_session": {"po-results-store": "{}"}})

    assert result["filename"] == "dashmat_session.json"
    assert result["type"] == "application/json"
    assert captured["workspace_session"] == {"po-results-store": "{}"}


def test_import_workspace_session_bundle_delegates_to_restore(monkeypatch):
    import app as app_module

    expected = {"workspace_session": {"dashmat-session-id-store": "\"session-new\""}}
    captured = {}

    def fake_restore(bundle):
        captured["bundle"] = bundle
        return expected

    monkeypatch.setattr(app_module, "restore_workspace_session_bundle", fake_restore)

    result = app_module.import_workspace_session_bundle({"bundle": {"version": 2}})

    assert result == expected
    assert captured["bundle"] == {"version": 2}
