"""Unified workbench page hosting all DashMat workspace modules."""

from __future__ import annotations

from urllib.parse import parse_qs

from dash import Input, Output, State, callback, html, register_page
from dash.exceptions import PreventUpdate

from pages import analyticstool, portopt, regression
from utils.page_paths import LANDING_PATH, WORKBENCH_PATH, normalize_module
from utils.route_intent import ACTION_OPEN_IMPORT_MODAL


register_page(__name__, path=WORKBENCH_PATH, name="Workbench", title="DashMat Workbench")


def _workbench_path(pathname):
    return str(pathname or "").split("?")[0].rstrip("/") or "/"


def _fresh_open_import_token(route_intent, active_module, consumed_token):
    if not isinstance(route_intent, dict):
        return None
    if str(route_intent.get("target_module") or "").strip().lower() != normalize_module(active_module):
        return None
    if str(route_intent.get("action") or "").strip().lower() != ACTION_OPEN_IMPORT_MODAL:
        return None
    token = str(route_intent.get("token") or "").strip()
    if not token or token == str(consumed_token or ""):
        return None
    return token


layout = html.Div(
    [
        html.Div(id="wb-active-module-host", style={"display": "flex", "flex": "1", "minHeight": 0}),
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
    page_path = _workbench_path(pathname)
    if page_path != WORKBENCH_PATH:
        raise PreventUpdate
    query = parse_qs(str(search or "").lstrip("?"))
    module_name = normalize_module((query.get("module") or [None])[0])
    expected_search = f"?module={module_name}"
    if str(search or "") == expected_search:
        raise PreventUpdate
    return expected_search


@callback(
    Output("wb-active-module-host", "children"),
    Input("wb-active-module-store", "data"),
    Input("_pages_location", "pathname"),
    prevent_initial_call=False,
)
def wb_render_active_module(active_module, pathname):
    if _workbench_path(pathname) != WORKBENCH_PATH:
        raise PreventUpdate
    active_module = normalize_module(active_module)
    if active_module == "portopt":
        return portopt.layout
    if active_module == "regression":
        return regression.layout
    return analyticstool.layout


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
    State("at-route-intent-consumed-token-store", "data", allow_optional=True),
    State("po-route-intent-consumed-token-store", "data", allow_optional=True),
    State("reg-route-intent-consumed-token-store", "data", allow_optional=True),
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
    at_consumed_token,
    po_consumed_token,
    reg_consumed_token,
):
    page_path = _workbench_path(pathname)
    if page_path != WORKBENCH_PATH:
        raise PreventUpdate
    if raw_data:
        raise PreventUpdate
    active_module = normalize_module(active_module)
    consumed_tokens = {
        "analyticstool": at_consumed_token,
        "portopt": po_consumed_token,
        "regression": reg_consumed_token,
    }
    module_modal_states = {
        "analyticstool": (at_db_open, at_raw_open, at_portfolio_open, at_underlying_open),
        "portopt": (po_db_open, po_raw_open, po_portfolio_open, po_underlying_open),
        "regression": (reg_db_open, reg_raw_open, reg_portfolio_open, reg_underlying_open),
    }
    if _fresh_open_import_token(route_intent, active_module, consumed_tokens.get(active_module)):
        raise PreventUpdate
    if any(bool(value) for value in module_modal_states.get(active_module, ())):
        raise PreventUpdate
    return LANDING_PATH, f"?module={active_module}"
