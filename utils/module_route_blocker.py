from __future__ import annotations

import dash_mantine_components as dmc
from dash import Input, Output, State, clientside_callback, dcc, html
from dash.dependencies import ClientsideFunction


MODULE_ROUTE_PATHS = frozenset(("/analyticstool", "/portopt", "/regression"))


def is_module_route(pathname: str | None) -> bool:
    normalized = str(pathname or "").split("?", 1)[0].rstrip("/") or "/"
    return normalized in MODULE_ROUTE_PATHS


def route_blocker_shell_style(active: bool) -> dict[str, object]:
    if not active:
        return {"display": "none"}
    return {"position": "fixed", "inset": 0, "zIndex": 2400}


def build_module_route_blocker_components() -> list:
    return [
        dcc.Store(
            id="dashmat-route-blocker-store",
            data={"active": False, "pathname": None, "requestId": 0},
            storage_type="memory",
        ),
        dcc.Store(
            id="at-route-ready-store",
            data={"pathname": "/analyticstool", "requestId": 0, "ready": False},
            storage_type="memory",
        ),
        dcc.Store(
            id="po-route-ready-store",
            data={"pathname": "/portopt", "requestId": 0, "ready": False},
            storage_type="memory",
        ),
        dcc.Store(
            id="reg-route-ready-store",
            data={"pathname": "/regression", "requestId": 0, "ready": False},
            storage_type="memory",
        ),
        html.Div(
            id="dashmat-route-blocker-shell",
            style=route_blocker_shell_style(False),
            children=[
                dmc.LoadingOverlay(
                    id="dashmat-route-blocker-overlay",
                    visible=False,
                    zIndex=2400,
                    overlayProps={"radius": "sm", "blur": 2},
                    loaderProps={"variant": "bars", "size": "lg"},
                    style={"position": "absolute", "inset": 0},
                ),
            ],
        ),
    ]


def register_module_route_blocker_callbacks(app) -> None:
    clientside_callback(
        ClientsideFunction(namespace="dashmat_callbacks", function_name="moduleRouteBlockerState"),
        Output("dashmat-route-blocker-store", "data"),
        Input("_pages_location", "pathname"),
        State("dashmat-route-blocker-store", "data"),
        prevent_initial_call=False,
    )

    clientside_callback(
        ClientsideFunction(namespace="dashmat_callbacks", function_name="moduleRouteBlockerPresentation"),
        Output("dashmat-route-blocker-shell", "style"),
        Output("dashmat-route-blocker-overlay", "visible"),
        Input("dashmat-route-blocker-store", "data"),
        Input("at-route-ready-store", "data"),
        Input("po-route-ready-store", "data"),
        Input("reg-route-ready-store", "data"),
        prevent_initial_call=False,
    )
