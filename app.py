"""DashMat - Market Returns Time Series Dashboard."""

import dash
import dash_mantine_components as dmc
from dash import ClientsideFunction, Dash, Input, Output, State, clientside_callback, dcc, no_update, page_container
from dash_iconify import DashIconify
from dash.exceptions import PreventUpdate
from uuid import uuid4

from cache_config import init_cache
from utils.artifact_store import store_raw_data_artifact
from utils.perf_timing import record_payload_size, timed_block
from utils.returns import build_raw_data_metadata
from utils.serialization import canonical_json_dumps
from utils.workspace_session import build_workspace_session_bundle, restore_workspace_session_bundle

# Initialize the app with multi-page support
app = Dash(
    __name__,
    use_pages=True,
    suppress_callback_exceptions=True,
)

# Initialize cache for performance optimization (after app creation)
cache = init_cache(app.server)

USERINFO_DATA = {"role": "Admin"}
dmc.pre_render_color_scheme()


def _registry_path(page_key: str, fallback: str) -> str:
    page_meta = dash.page_registry.get(page_key, {})
    path = page_meta.get("path") if isinstance(page_meta, dict) else None
    if isinstance(path, str) and path:
        return path
    return fallback


def _restricted_href_for_path(pathname: str | None, userinfo: dict | None) -> str | None:
    if (userinfo or {}).get("role") != "Test":
        return None

    if pathname in ("/analyticstool", "/analyticstool/"):
        return "/restricted?target=Analytics%20Tool"
    if pathname in ("/portopt", "/portopt/"):
        return "/restricted?target=Portfolio%20Optimization"
    if pathname in ("/regression", "/regression/"):
        return "/restricted?target=Regression%20Analysis"
    return None

HOME_PATH = _registry_path("pages.home", "/")
ANALYTICS_PATH = _registry_path("pages.analyticstool", "/analyticstool")
PORTOPT_PATH = _registry_path("pages.portopt", "/portopt")
REGRESSION_PATH = _registry_path("pages.regression", "/regression")

# Layout wraps page content with MantineProvider
# Shared stores are defined here so they are accessible across all pages
_provider_kwargs = {"id": "mantine-provider", "children": [
    dcc.Store(id="dashmat-session-id-store", data=None, storage_type="session"),
    dcc.Store(id="dashmat-raw-data-store", data=None, storage_type="session"),
    dcc.Store(id="dashmat-raw-data-artifact-store", data=None, storage_type="session"),
    dcc.Store(id="dashmat-raw-data-meta-store", data=None, storage_type="session"),
    dcc.Store(id="dashmat-original-periodicity-store", data="daily", storage_type="session"),
    dcc.Store(id="dashmat-pending-new-series-store", data={}, storage_type="session"),
    dcc.Store(id="dashmat-saved-series-cache-store", data=None, storage_type="session"),
    dcc.Store(id="dashmat-session-export-request-store", data=None, storage_type="memory"),
    dcc.Store(id="dashmat-session-import-request-store", data=None, storage_type="memory"),
    dcc.Store(id="dashmat-session-import-result-store", data=None, storage_type="memory"),
    dcc.Store(id="dashmat-session-import-apply-dummy", data=None, storage_type="memory"),
    dcc.Download(id="dashmat-save-session-download"),
    dcc.Store(id="userinfo", data=USERINFO_DATA, storage_type="session"),
    dmc.AppShell(
        header={"height": 45},
        padding=0,
        children=[
            dmc.AppShellHeader(
                dmc.Group(
                    justify="space-between",
                    px="md",
                    h="100%",
                    children=[
                        dmc.Text("DashMat", fw=700),
                        dmc.Group(
                            gap="xs",
                            children=[
                                dmc.ColorSchemeToggle(
                                    id="global-color-scheme-toggle",
                                    variant="gradient",
                                    gradient={"from": "orange", "to": "blue", "deg": 135},
                                    lightIcon=DashIconify(icon="tabler:sun-filled", width=16),
                                    darkIcon=DashIconify(icon="tabler:moon-stars", width=16),
                                    autoContrast=True,
                                    radius="xl",
                                    size="md",
                                    style={"boxShadow": "0 1px 3px rgba(0,0,0,0.20)"},
                                ),
                                dmc.Menu(
                                    trigger="hover",
                                    openDelay=100,
                                    closeDelay=200,
                                    position="bottom-end",
                                    shadow="md",
                                    offset=6,
                                    children=[
                                        dmc.MenuTarget(
                                            dmc.Button(
                                                "Menu",
                                                size="sm",
                                                variant="subtle",
                                                color="gray",
                                                radius="sm",
                                            ),
                                        ),
                                        dmc.MenuDropdown(
                                            children=[
                                                dmc.MenuItem(
                                                    "Home",
                                                    id="global-navbar-pretrade-home",
                                                    href=HOME_PATH,
                                                ),
                                                dmc.MenuItem(
                                                    "Analytics Tool",
                                                    id="global-navbar-pretrade-analyticstool",
                                                    href=ANALYTICS_PATH,
                                                ),
                                                dmc.MenuItem(
                                                    "Portfolio Optimization",
                                                    id="global-navbar-pretrade-portopt",
                                                    href=PORTOPT_PATH,
                                                ),
                                                dmc.MenuItem(
                                                    "Regression",
                                                    id="global-navbar-pretrade-regression",
                                                    href=REGRESSION_PATH,
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ),
            dmc.AppShellMain(
                page_container,
                style={"paddingTop": "53px"},
            ),
        ],
    ),
]}
_provider_kwargs["defaultColorScheme"] = "light"
app.layout = dmc.MantineProvider(**_provider_kwargs)


@app.callback(
    Output("dashmat-session-id-store", "data"),
    Input("_pages_location", "pathname"),
    State("dashmat-session-id-store", "data"),
    prevent_initial_call=False,
)
def ensure_dashmat_session_id(_pathname, existing_session_id):
    if existing_session_id:
        return no_update
    return str(uuid4())


@app.callback(
    Output("global-navbar-pretrade-home", "href"),
    Output("global-navbar-pretrade-analyticstool", "href"),
    Output("global-navbar-pretrade-portopt", "href"),
    Output("global-navbar-pretrade-regression", "href"),
    Input("userinfo", "data"),
    prevent_initial_call=True,
)
def update_app_nav_links(userinfo):
    home_path = _registry_path("pages.home", "/")
    analytics_path = _registry_path("pages.analyticstool", "/analyticstool")
    portopt_path = _registry_path("pages.portopt", "/portopt")
    regression_path = _registry_path("pages.regression", "/regression")

    if (userinfo or {}).get("role") == "Test":
        return (
            home_path,
            "/restricted?target=Analytics%20Tool",
            "/restricted?target=Portfolio%20Optimization",
            "/restricted?target=Regression%20Analysis",
        )

    return home_path, analytics_path, portopt_path, regression_path


@app.callback(
    Output("_pages_location", "href"),
    Input("_pages_location", "pathname"),
    Input("userinfo", "data"),
    prevent_initial_call=False,
)
def guard_protected_pages(pathname, userinfo):
    restricted_href = _restricted_href_for_path(pathname, userinfo)
    if not restricted_href:
        raise PreventUpdate
    return restricted_href


@app.callback(
    Output("dashmat-raw-data-artifact-store", "data"),
    Input("dashmat-raw-data-store", "data"),
    Input("dashmat-original-periodicity-store", "data"),
    Input("dashmat-session-id-store", "data"),
    prevent_initial_call=False,
)
def refresh_raw_data_artifact_store(raw_data, original_periodicity, session_id):
    if not raw_data or not session_id:
        return None
    with timed_block(
        "refresh_raw_data_artifact_store",
        has_data=bool(raw_data),
        session_id=session_id,
    ):
        descriptor = store_raw_data_artifact(
            session_id=session_id,
            raw_data_json=raw_data,
            original_periodicity=original_periodicity,
        )
    record_payload_size(
        "dashmat_raw_data_artifact_store.output",
        descriptor,
        session_id=session_id,
    )
    return descriptor


@app.callback(
    Output("dashmat-save-session-download", "data"),
    Input("dashmat-session-export-request-store", "data"),
    prevent_initial_call=True,
)
def export_workspace_session_bundle(request_data):
    if not isinstance(request_data, dict):
        raise PreventUpdate
    workspace_session = request_data.get("workspace_session")
    if not isinstance(workspace_session, dict) or not workspace_session:
        raise PreventUpdate
    with timed_block("workspace_session.export", key_count=len(workspace_session)):
        bundle = build_workspace_session_bundle(workspace_session)
    return {
        "content": canonical_json_dumps(bundle),
        "filename": "dashmat_session.json",
        "type": "application/json",
    }


@app.callback(
    Output("dashmat-session-import-result-store", "data"),
    Input("dashmat-session-import-request-store", "data"),
    prevent_initial_call=True,
)
def import_workspace_session_bundle(request_data):
    if not isinstance(request_data, dict):
        raise PreventUpdate
    bundle = request_data.get("bundle")
    if not isinstance(bundle, dict):
        raise PreventUpdate
    with timed_block("workspace_session.import", artifact_count=len(bundle.get("artifacts") or [])):
        return restore_workspace_session_bundle(bundle)


@app.callback(
    Output("dashmat-raw-data-meta-store", "data"),
    Input("dashmat-raw-data-store", "data"),
    Input("dashmat-original-periodicity-store", "data"),
    prevent_initial_call=False,
)
def refresh_raw_data_meta_store(raw_data, original_periodicity):
    return build_raw_data_metadata(raw_data, original_periodicity)


clientside_callback(
    ClientsideFunction(namespace="dashmat_callbacks", function_name="applyLoadedWorkspaceSession"),
    Output("dashmat-session-import-apply-dummy", "data"),
    Input("dashmat-session-import-result-store", "data"),
    prevent_initial_call=True,
)

# Theme consumer callbacks are defined in page modules for charts.

if __name__ == "__main__":
    import sys
    debug = "--debug" in sys.argv
    app.run(debug=debug)
