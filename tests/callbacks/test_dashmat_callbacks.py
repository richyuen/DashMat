from __future__ import annotations

from types import SimpleNamespace

import pytest
from dash.exceptions import PreventUpdate


def test_dm_search_for_module_selection_keeps_bare_default_url():
    import app  # noqa: F401
    import pages.dashmat as dashmat

    assert dashmat.dm_search_for_module_selection("analyticstool", "") is None


def test_dm_search_for_module_selection_sets_non_default_query():
    import app  # noqa: F401
    import pages.dashmat as dashmat

    assert dashmat.dm_search_for_module_selection("portopt", "") == "?module=portopt"


def test_dm_search_for_module_selection_clears_query_when_returning_to_default():
    import app  # noqa: F401
    import pages.dashmat as dashmat

    assert dashmat.dm_search_for_module_selection("analyticstool", "?module=portopt") == ""


def test_dm_sync_module_from_query_normalizes_default_and_named_modules():
    import app  # noqa: F401
    import pages.dashmat as dashmat

    assert dashmat.dm_sync_module_from_query(None) == "analyticstool"
    assert dashmat.dm_sync_module_from_query("?module=regression") == "regression"


def test_dm_route_non_file_imports_requires_positive_click(monkeypatch):
    import app  # noqa: F401
    import pages.dashmat as dashmat

    monkeypatch.setattr(dashmat, "callback_context", SimpleNamespace(triggered_id="dm-welcome-add-db-btn"))

    with pytest.raises(PreventUpdate):
        dashmat.dm_route_non_file_imports(0, None, None, None, None, None, None, None, "analyticstool")


def test_dm_route_non_file_imports_builds_intent_for_real_click(monkeypatch):
    import app  # noqa: F401
    import pages.dashmat as dashmat

    monkeypatch.setattr(dashmat, "callback_context", SimpleNamespace(triggered_id="dm-welcome-add-db-btn"))

    intent, nav_target = dashmat.dm_route_non_file_imports(1, None, None, None, None, None, None, None, "portopt")

    assert intent["target_module"] == "portopt"
    assert intent["action"] == "open_import_modal"
    assert intent["flow"] == "db"
    assert nav_target == "/portopt"


def test_dm_update_workspace_cta_hides_restore_button_without_session_data():
    import app  # noqa: F401
    import pages.dashmat as dashmat

    href, label, style = dashmat.dm_update_workspace_cta("portopt", None)

    assert href == "/portopt"
    assert label == "Restore existing session in Portfolio Optimization"
    assert style == {"display": "none"}


def test_dm_update_workspace_cta_shows_restore_button_with_session_data():
    import app  # noqa: F401
    import pages.dashmat as dashmat

    href, label, style = dashmat.dm_update_workspace_cta("regression", {"data": "present"})

    assert href == "/regression"
    assert label == "Restore existing session in Regression"
    assert style == {}
