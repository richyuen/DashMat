"""Unified workbench page hosting all DashMat workspace modules."""

from __future__ import annotations

from urllib.parse import parse_qs

from dash import Input, Output, State, callback, dcc, html, no_update, register_page
from dash.exceptions import PreventUpdate

from pages import analyticstool, portopt, regression
from utils.page_paths import LANDING_PATH, WORKBENCH_PATH, landing_href, normalize_module
from utils.route_intent import ACTION_OPEN_IMPORT_MODAL


register_page(__name__, path=WORKBENCH_PATH, name="Workbench", title="DashMat Workbench")


def _route_intent_targets_active_module(route_intent, active_module):
    if not isinstance(route_intent, dict):
        return False
    if str(route_intent.get("target_module") or "").strip().lower() != normalize_module(active_module):
        return False
    if str(route_intent.get("action") or "").strip().lower() != ACTION_OPEN_IMPORT_MODAL:
        return False
    return bool(str(route_intent.get("token") or "").strip())


layout = html.Div(
    [
        html.Div(
            id="wb-analytics-wrapper",
            children=analyticstool.layout,
            style={"display": "flex", "flex": "1", "minHeight": 0},
        ),
        html.Div(
            id="wb-portopt-wrapper",
            children=portopt.layout,
            style={"display": "none", "flex": "1", "minHeight": 0},
        ),
        html.Div(
            id="wb-regression-wrapper",
            children=regression.layout,
            style={"display": "none", "flex": "1", "minHeight": 0},
        ),
    ],
    style={"display": "flex", "flex": "1", "minHeight": 0},
)


@callback(
    Output("wb-active-module-store", "data"),
    Input("_pages_location", "search"),
    prevent_initial_call=False,
)
def wb_sync_active_module(search):
    query = parse_qs(str(search or "").lstrip("?"))
    return normalize_module((query.get("module") or [None])[0])


@callback(
    Output("_pages_location", "search", allow_duplicate=True),
    Input("_pages_location", "pathname"),
    Input("_pages_location", "search"),
    prevent_initial_call="initial_duplicate",
)
def wb_canonicalize_query(pathname, search):
    page_path = str(pathname or "").split("?")[0].rstrip("/") or "/"
    if page_path != WORKBENCH_PATH:
        raise PreventUpdate
    query = parse_qs(str(search or "").lstrip("?"))
    module_name = normalize_module((query.get("module") or [None])[0])
    expected_search = f"?module={module_name}"
    if str(search or "") == expected_search:
        raise PreventUpdate
    return expected_search


@callback(
    Output("wb-analytics-wrapper", "style"),
    Output("wb-portopt-wrapper", "style"),
    Output("wb-regression-wrapper", "style"),
    Input("wb-active-module-store", "data"),
    prevent_initial_call=False,
)
def wb_update_wrapper_visibility(active_module):
    active_module = normalize_module(active_module)
    visible = {"display": "flex", "flex": "1", "minHeight": 0}
    hidden = {"display": "none", "flex": "1", "minHeight": 0}
    return (
        visible if active_module == "analyticstool" else hidden,
        visible if active_module == "portopt" else hidden,
        visible if active_module == "regression" else hidden,
    )


@callback(
    Output("wb-previous-module-store", "data"),
    Output("wb-analytics-activation-store", "data"),
    Output("wb-portopt-activation-store", "data"),
    Output("wb-regression-activation-store", "data"),
    Input("wb-active-module-store", "data"),
    State("wb-previous-module-store", "data"),
    State("wb-analytics-activation-store", "data"),
    State("wb-portopt-activation-store", "data"),
    State("wb-regression-activation-store", "data"),
    prevent_initial_call=False,
)
def wb_dispatch_activation(active_module, previous_module, at_token, po_token, reg_token):
    active_module = normalize_module(active_module)
    if active_module == previous_module:
        raise PreventUpdate
    at_token = int(at_token or 0)
    po_token = int(po_token or 0)
    reg_token = int(reg_token or 0)
    if active_module == "analyticstool":
        at_token += 1
    elif active_module == "portopt":
        po_token += 1
    else:
        reg_token += 1
    return active_module, at_token, po_token, reg_token


@callback(
    Output("_pages_location", "pathname", allow_duplicate=True),
    Output("_pages_location", "search", allow_duplicate=True),
    Input("dashmat-raw-data-store", "data"),
    Input("wb-active-module-store", "data"),
    Input("dashmat-route-intent-store", "data"),
    Input("at-db-add-modal", "opened", allow_optional=True),
    Input("at-raw-db-add-modal", "opened", allow_optional=True),
    Input("at-portfolio-add-modal", "opened", allow_optional=True),
    Input("at-underlying-add-modal", "opened", allow_optional=True),
    Input("po-db-add-modal", "opened", allow_optional=True),
    Input("po-raw-db-add-modal", "opened", allow_optional=True),
    Input("po-portfolio-add-modal", "opened", allow_optional=True),
    Input("po-underlying-add-modal", "opened", allow_optional=True),
    Input("reg-db-add-modal", "opened", allow_optional=True),
    Input("reg-raw-db-add-modal", "opened", allow_optional=True),
    Input("reg-portfolio-add-modal", "opened", allow_optional=True),
    Input("reg-underlying-add-modal", "opened", allow_optional=True),
    State("_pages_location", "pathname"),
    prevent_initial_call="initial_duplicate",
)
def wb_redirect_empty_state(
    raw_data,
    active_module,
    route_intent,
    at_db_open,
    at_raw_open,
    at_portfolio_open,
    at_underlying_open,
    po_db_open,
    po_raw_open,
    po_portfolio_open,
    po_underlying_open,
    reg_db_open,
    reg_raw_open,
    reg_portfolio_open,
    reg_underlying_open,
    pathname,
):
    page_path = str(pathname or "").split("?")[0].rstrip("/") or "/"
    if page_path != WORKBENCH_PATH:
        raise PreventUpdate
    if raw_data:
        raise PreventUpdate
    active_module = normalize_module(active_module)
    if _route_intent_targets_active_module(route_intent, active_module):
        raise PreventUpdate
    module_modal_states = {
        "analyticstool": (at_db_open, at_raw_open, at_portfolio_open, at_underlying_open),
        "portopt": (po_db_open, po_raw_open, po_portfolio_open, po_underlying_open),
        "regression": (reg_db_open, reg_raw_open, reg_portfolio_open, reg_underlying_open),
    }
    if any(bool(value) for value in module_modal_states.get(active_module, ())):
        raise PreventUpdate
    return LANDING_PATH, f"?module={active_module}"
