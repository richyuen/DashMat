from __future__ import annotations


def test_update_app_nav_links_for_test_role():
    import app as app_module

    home_href, analytics_href, portopt_href = app_module.update_app_nav_links({"role": "Test"})
    assert home_href == "/"
    assert analytics_href == "/restricted?target=Analytics%20Tool"
    assert portopt_href == "/restricted?target=Portfolio%20Optimization"


def test_update_app_nav_links_for_non_test_role():
    import app as app_module

    home_href, analytics_href, portopt_href = app_module.update_app_nav_links({"role": "Admin"})
    assert home_href == "/"
    assert analytics_href == "/analyticstool"
    assert portopt_href == "/portopt"


def test_server_guard_redirects_protected_routes():
    import app as app_module

    client = app_module.app.server.test_client()

    analytics_response = client.get("/analyticstool", follow_redirects=False)
    assert analytics_response.status_code == 302
    assert analytics_response.headers["Location"] == "/restricted?target=Analytics%20Tool"

    portopt_response = client.get("/portopt", follow_redirects=False)
    assert portopt_response.status_code == 302
    assert portopt_response.headers["Location"] == "/restricted?target=Portfolio%20Optimization"
