from __future__ import annotations


def test_update_home_nav_links_for_test_role():
    import app  # noqa: F401
    import pages.home as home

    analytics_href, portopt_href, regression_href = home.update_home_nav_links({"role": "Test"})
    assert analytics_href == "/restricted?target=Analytics%20Tool"
    assert portopt_href == "/restricted?target=Portfolio%20Optimization"
    assert regression_href == "/restricted?target=Regression"


def test_update_home_nav_links_for_non_test_role():
    import app  # noqa: F401
    import pages.home as home

    analytics_href, portopt_href, regression_href = home.update_home_nav_links({"role": "Admin"})
    assert analytics_href == "/landing?module=analyticstool"
    assert portopt_href == "/landing?module=portopt"
    assert regression_href == "/landing?module=regression"
