"""DashMat - Market Returns Time Series Dashboard."""

import dash
import dash_mantine_components as dmc
from dash import Dash, Input, Output, dcc, html, page_container
from dash_iconify import DashIconify
from dash.exceptions import PreventUpdate
from cache_config import init_cache
from dbengine import engine as DB_ENGINE, engine_MRD as MRD_ENGINE, engine_PERFORMANCE as PERF_ENGINE
from utils.account_list_modal import (
    build_account_list_modal_components,
    register_account_list_callbacks,
)
from utils.module_route_blocker import (
    build_module_route_blocker_components,
    register_module_route_blocker_callbacks,
)
from utils.returns import build_raw_data_metadata

# Initialize the app with multi-page support
app = Dash(
    __name__,
    use_pages=True,
    suppress_callback_exceptions=True,
)

# Initialize cache for performance optimization (after app creation)
cache = init_cache(app.server)

USERINFO_DATA = {"role": "Admin", "username": "Admin User"}
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
    dcc.Store(id="userinfo", data=USERINFO_DATA, storage_type="session"),
    dcc.Store(id="dashmat-raw-data-store", data=None, storage_type="session"),
    dcc.Store(id="dashmat-raw-data-meta-store", data=None, storage_type="session"),
    dcc.Store(id="dashmat-original-periodicity-store", data="daily", storage_type="session"),
    dcc.Store(id="dashmat-pending-new-series-store", data={}, storage_type="session"),
    dcc.Store(id="dashmat-saved-series-cache-store", data=None, storage_type="session"),
    dcc.Store(id="dashmat-db-import-provenance-store", data={}, storage_type="session"),
    dcc.Store(id="dashmat-account-list-notice-store", data=None, storage_type="session"),
    dcc.Store(id="dashmat-account-list-modal-mode-store", data="load"),
    dcc.Store(id="dashmat-account-list-rows-store", data=[]),
    dcc.Store(id="dashmat-account-list-selected-id-store", data=None),
    dcc.Store(id="dashmat-account-list-session-snapshot-store", data={}),
    dcc.Store(id="dashmat-account-list-refresh-store", data=0),
    dcc.Store(id="dashmat-account-list-session-apply-store", data=None),
    dcc.Store(id="dashmat-account-list-load-state-store", data={"status": "idle"}),
    dcc.Store(id="dashmat-account-list-enter-submit-dummy", data=None),
    dcc.Store(id="dashmat-account-list-focus-dummy", data=None),
    *build_account_list_modal_components(),
    *build_module_route_blocker_components(),
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
    Output("dashmat-raw-data-meta-store", "data"),
    Input("dashmat-raw-data-store", "data"),
    Input("dashmat-original-periodicity-store", "data"),
    prevent_initial_call=False,
)
def refresh_raw_data_meta_store(raw_data, original_periodicity):
    return build_raw_data_metadata(raw_data, original_periodicity)


register_account_list_callbacks(
    app,
    db_engine=DB_ENGINE,
    mrd_engine=MRD_ENGINE,
    perf_engine=PERF_ENGINE,
)
register_module_route_blocker_callbacks(app)

# Theme consumer callbacks are defined in page modules for charts.

if __name__ == "__main__":
    import sys
    debug = "--debug" in sys.argv
    app.run(debug=debug)
