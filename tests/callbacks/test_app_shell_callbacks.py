from __future__ import annotations

import pytest
from dash.exceptions import PreventUpdate


def test_update_app_nav_links_for_test_role():
    import app as app_module

    home_href, analytics_href, portopt_href, regression_href = app_module.update_app_nav_links({"role": "Test"})
    assert home_href == "/"
    assert analytics_href == "/restricted?target=Analytics%20Tool"
    assert portopt_href == "/restricted?target=Portfolio%20Optimization"
    assert regression_href == "/restricted?target=Regression"


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
        "/restricted?target=Regression"
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
        "/restricted?target=Regression"
    )
    with pytest.raises(PreventUpdate):
        app_module.guard_protected_pages("/", {"role": "Test"})
