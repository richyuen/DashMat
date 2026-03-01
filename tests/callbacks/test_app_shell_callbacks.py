from __future__ import annotations

import pytest
from dash.exceptions import PreventUpdate


def test_update_global_nav_links_for_test_role():
    import app as app_module

    home_href, analytics_href, portopt_href, regression_href = app_module.update_global_nav_links({"role": "Test"}, None)
    assert home_href == "/"
    assert analytics_href == "/restricted?target=Analytics%20Tool"
    assert portopt_href == "/restricted?target=Portfolio%20Optimization"
    assert regression_href == "/restricted?target=Regression"


def test_update_global_nav_links_for_non_test_role_without_data():
    import app as app_module

    home_href, analytics_href, portopt_href, regression_href = app_module.update_global_nav_links({"role": "Admin"}, None)
    assert home_href == "/"
    assert analytics_href == "/landing?module=analyticstool"
    assert portopt_href == "/landing?module=portopt"
    assert regression_href == "/landing?module=regression"


def test_update_global_nav_links_for_non_test_role_with_data():
    import app as app_module

    home_href, analytics_href, portopt_href, regression_href = app_module.update_global_nav_links(
        {"role": "Admin"},
        {"mock": True},
    )
    assert home_href == "/"
    assert analytics_href == "/workbench?module=analyticstool"
    assert portopt_href == "/workbench?module=portopt"
    assert regression_href == "/workbench?module=regression"


def test_update_raw_data_summary_returns_none_without_data():
    import app as app_module

    assert app_module.update_raw_data_summary(None, "daily") is None


def test_update_raw_data_summary_builds_expected_payload(raw_json):
    import app as app_module

    summary = app_module.update_raw_data_summary(raw_json, "daily")

    assert summary["columns"] == ["Asset_A", "Asset_B", "Asset_C", "Asset_D"]
    assert summary["original_periodicity"] == "daily"
    assert summary["available_periodicity_values"][0] == "daily_trading"
    assert summary["raw_data_hash"]


def test_restricted_href_for_path_resolves_for_test_role():
    import app as app_module

    assert app_module._restricted_href_for_path("/landing", {"role": "Test"}) == (
        "/restricted?target=DashMat"
    )
    assert app_module._restricted_href_for_path("/workbench", {"role": "Test"}) is None


def test_restricted_href_for_path_skips_non_test_or_other_paths():
    import app as app_module

    assert app_module._restricted_href_for_path("/landing", {"role": "Admin"}) is None
    assert app_module._restricted_href_for_path("/workbench", {"role": "Admin"}) is None
    assert app_module._restricted_href_for_path("/", {"role": "Test"}) is None


def test_guard_protected_pages_redirects_or_prevent_update():
    import app as app_module

    assert app_module.guard_protected_pages("/workbench", "?module=analyticstool", {"role": "Test"}) == (
        "/restricted?target=Analytics%20Tool"
    )
    assert app_module.guard_protected_pages("/workbench", "?module=regression", {"role": "Test"}) == (
        "/restricted?target=Regression"
    )
    assert app_module.guard_protected_pages("/landing", "", {"role": "Test"}) == (
        "/restricted?target=DashMat"
    )
    with pytest.raises(PreventUpdate):
        app_module.guard_protected_pages("/", "", {"role": "Test"})
