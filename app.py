"""DashMat - Market Returns Time Series Dashboard."""

from urllib.parse import parse_qs

import dash_mantine_components as dmc
from dash import Dash, Input, Output, dcc, page_container
from dash_iconify import DashIconify
from dash.exceptions import PreventUpdate
from cache_config import init_cache
from utils.date_range_flow import build_raw_data_summary
from utils.page_paths import (
    HOME_PATH,
    LANDING_PATH,
    landing_href,
    module_to_label,
    normalize_module,
    workbench_href,
    WORKBENCH_PATH,
)

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


def _restricted_href_for_path(pathname: str | None, userinfo: dict | None) -> str | None:
    if (userinfo or {}).get("role") != "Test":
        return None

    if pathname in (LANDING_PATH, f"{LANDING_PATH}/"):
        return "/restricted?target=DashMat"
    if pathname in (WORKBENCH_PATH, f"{WORKBENCH_PATH}/"):
        return None
    return None

# Layout wraps page content with MantineProvider
# Shared stores are defined here so they are accessible across all pages
_provider_kwargs = {"id": "mantine-provider", "children": [
    dcc.Store(id="dashmat-raw-data-store", data=None, storage_type="session"),
    dcc.Store(id="dashmat-original-periodicity-store", data="daily", storage_type="session"),
    dcc.Store(id="dashmat-raw-data-summary-store", data=None, storage_type="memory"),
    dcc.Store(id="dashmat-pending-new-series-store", data=[], storage_type="session"),
    dcc.Store(id="dashmat-saved-series-cache-store", data=None, storage_type="session"),
    dcc.Store(id="dashmat-route-intent-store", data=None, storage_type="session"),
    dcc.Store(id="wb-active-module-store", data="analyticstool", storage_type="memory"),
    dcc.Store(id="wb-previous-module-store", data=None, storage_type="memory"),
    dcc.Store(id="wb-analytics-activation-store", data=0, storage_type="memory"),
    dcc.Store(id="wb-portopt-activation-store", data=0, storage_type="memory"),
    dcc.Store(id="wb-regression-activation-store", data=0, storage_type="memory"),
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
                                                dcc.Link(
                                                    id="global-navbar-pretrade-home",
                                                    href=HOME_PATH,
                                                    refresh=False,
                                                    style={"textDecoration": "none", "color": "inherit"},
                                                    children=dmc.MenuItem("Home"),
                                                ),
                                                dcc.Link(
                                                    id="global-navbar-pretrade-analytics",
                                                    href=landing_href("analyticstool"),
                                                    refresh=False,
                                                    style={"textDecoration": "none", "color": "inherit"},
                                                    children=dmc.MenuItem("Analytics Tool"),
                                                ),
                                                dcc.Link(
                                                    id="global-navbar-pretrade-portopt",
                                                    href=landing_href("portopt"),
                                                    refresh=False,
                                                    style={"textDecoration": "none", "color": "inherit"},
                                                    children=dmc.MenuItem("Portfolio Optimization"),
                                                ),
                                                dcc.Link(
                                                    id="global-navbar-pretrade-regression",
                                                    href=landing_href("regression"),
                                                    refresh=False,
                                                    style={"textDecoration": "none", "color": "inherit"},
                                                    children=dmc.MenuItem("Regression"),
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
    Output("dashmat-raw-data-summary-store", "data"),
    Input("dashmat-raw-data-store", "data"),
    Input("dashmat-original-periodicity-store", "data"),
    prevent_initial_call=False,
)
def update_raw_data_summary(raw_data, original_periodicity):
    return build_raw_data_summary(raw_data, original_periodicity or "daily")


@app.callback(
    Output("global-navbar-pretrade-home", "href"),
    Output("global-navbar-pretrade-analytics", "href"),
    Output("global-navbar-pretrade-portopt", "href"),
    Output("global-navbar-pretrade-regression", "href"),
    Input("userinfo", "data"),
    Input("dashmat-raw-data-store", "data"),
    prevent_initial_call=True,
)
def update_global_nav_links(userinfo, raw_data):
    if (userinfo or {}).get("role") == "Test":
        return (
            HOME_PATH,
            "/restricted?target=Analytics%20Tool",
            "/restricted?target=Portfolio%20Optimization",
            "/restricted?target=Regression",
        )

    if raw_data:
        return (
            HOME_PATH,
            workbench_href("analyticstool"),
            workbench_href("portopt"),
            workbench_href("regression"),
        )

    return (
        HOME_PATH,
        landing_href("analyticstool"),
        landing_href("portopt"),
        landing_href("regression"),
    )


@app.callback(
    Output("_pages_location", "href"),
    Input("_pages_location", "pathname"),
    Input("_pages_location", "search"),
    Input("userinfo", "data"),
    prevent_initial_call=False,
)
def guard_protected_pages(pathname, search, userinfo):
    if (userinfo or {}).get("role") == "Test" and pathname in (WORKBENCH_PATH, f"{WORKBENCH_PATH}/"):
        module_name = normalize_module((parse_qs(str(search or "").lstrip("?")).get("module") or [None])[0])
        return f"/restricted?target={module_to_label(module_name).replace(' ', '%20')}"
    restricted_href = _restricted_href_for_path(pathname, userinfo)
    if not restricted_href:
        raise PreventUpdate
    return restricted_href

# Theme consumer callbacks are defined in page modules for charts.

if __name__ == "__main__":
    import sys
    debug = "--debug" in sys.argv
    app.run(debug=debug)
