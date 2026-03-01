from __future__ import annotations

import pytest
from dash.exceptions import PreventUpdate


def test_wb_render_active_module_returns_analytics_layout():
    import app  # noqa: F401
    import pages.workbench as workbench

    assert workbench.wb_render_active_module("analyticstool", "/workbench") is workbench.analyticstool.layout


def test_wb_render_active_module_returns_portopt_layout():
    import app  # noqa: F401
    import pages.workbench as workbench

    assert workbench.wb_render_active_module("portopt", "/workbench") is workbench.portopt.layout


def test_wb_render_active_module_returns_regression_layout():
    import app  # noqa: F401
    import pages.workbench as workbench

    assert workbench.wb_render_active_module("regression", "/workbench") is workbench.regression.layout


def test_wb_render_active_module_ignores_non_workbench_path():
    import app  # noqa: F401
    import pages.workbench as workbench

    with pytest.raises(PreventUpdate):
        workbench.wb_render_active_module("analyticstool", "/landing")


def test_wb_redirect_empty_state_stays_when_fresh_matching_intent_exists():
    import app  # noqa: F401
    import pages.workbench as workbench
    from utils.route_intent import ACTION_OPEN_IMPORT_MODAL, build_route_intent

    route_intent = build_route_intent("analyticstool", ACTION_OPEN_IMPORT_MODAL, flow="db")

    with pytest.raises(PreventUpdate):
        workbench.wb_redirect_empty_state(
            None,
            "analyticstool",
            route_intent,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            "/workbench",
            None,
            None,
            None,
        )


def test_wb_redirect_empty_state_redirects_when_intent_is_consumed():
    import app  # noqa: F401
    import pages.workbench as workbench
    from utils.route_intent import ACTION_OPEN_IMPORT_MODAL, build_route_intent

    route_intent = build_route_intent("analyticstool", ACTION_OPEN_IMPORT_MODAL, flow="db")

    assert workbench.wb_redirect_empty_state(
        None,
        "analyticstool",
        route_intent,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        "/workbench",
        route_intent["token"],
        None,
        None,
    ) == ("/landing", "?module=analyticstool")


def test_wb_redirect_empty_state_redirects_without_matching_intent():
    import app  # noqa: F401
    import pages.workbench as workbench
    from utils.route_intent import ACTION_OPEN_IMPORT_MODAL, build_route_intent

    route_intent = build_route_intent("portopt", ACTION_OPEN_IMPORT_MODAL, flow="db")

    assert workbench.wb_redirect_empty_state(
        None,
        "analyticstool",
        route_intent,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        "/workbench",
        None,
        None,
        None,
    ) == ("/landing", "?module=analyticstool")


def test_wb_redirect_empty_state_stays_when_active_import_modal_is_open():
    import app  # noqa: F401
    import pages.workbench as workbench
    from utils.route_intent import ACTION_OPEN_IMPORT_MODAL, build_route_intent

    route_intent = build_route_intent("analyticstool", ACTION_OPEN_IMPORT_MODAL, flow="db")

    with pytest.raises(PreventUpdate):
        workbench.wb_redirect_empty_state(
            None,
            "analyticstool",
            route_intent,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            "/workbench",
            route_intent["token"],
            None,
            None,
        )
