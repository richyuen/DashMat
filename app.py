"""DashMat - Market Returns Time Series Dashboard."""

import dash
import dash_mantine_components as dmc
from flask import redirect, request
from dash import Dash, Input, Output, dcc, page_container
from cache_config import init_cache

# Initialize the app with multi-page support
app = Dash(
    __name__,
    use_pages=True,
    suppress_callback_exceptions=True,
)

# Initialize cache for performance optimization (after app creation)
cache = init_cache(app.server)

_PROTECTED_ROUTE_TARGETS = {
    "/analyticstool": "Analytics%20Tool",
    "/analyticstool/": "Analytics%20Tool",
    "/portopt": "Portfolio%20Optimization",
    "/portopt/": "Portfolio%20Optimization",
}


def _registry_path(page_key: str, fallback: str) -> str:
    page_meta = dash.page_registry.get(page_key, {})
    path = page_meta.get("path") if isinstance(page_meta, dict) else None
    if isinstance(path, str) and path:
        return path
    return fallback


@app.server.before_request
def _guard_protected_routes():
    target = _PROTECTED_ROUTE_TARGETS.get(request.path)
    if not target:
        return None

    # Test role for local validation; production role is sourced from Azure SSO/database.
    role = "Test"
    if role == "Test":
        return redirect(f"/restricted?target={target}", code=302)
    return None

HOME_PATH = _registry_path("pages.home", "/")
ANALYTICS_PATH = _registry_path("pages.analyticstool", "/analyticstool")
PORTOPT_PATH = _registry_path("pages.portopt", "/portopt")

# Layout wraps page content with MantineProvider
# Shared stores are defined here so they are accessible across all pages
app.layout = dmc.MantineProvider(
    id="mantine-provider",
    children=[
        dcc.Store(id="analyticstool-raw-data-store", data=None, storage_type="session"),
        dcc.Store(id="analyticstool-original-periodicity-store", data="daily", storage_type="session"),
        dcc.Store(id="analyticstool-pending-new-series-store", data=[], storage_type="session"),
        dcc.Store(id="analyticstool-saved-series-cache-store", data=None, storage_type="session"),
        dcc.Store(id="userinfo", data={"role": "Test"}, storage_type="session"),
        dcc.Store(id="theme-store", data="light", storage_type="local"),
        dmc.AppShell(
            header={"height": 48},
            padding=0,
            children=[
                dmc.AppShellHeader(
                    dmc.Group(
                        justify="space-between",
                        px="md",
                        h="100%",
                        children=[
                            dmc.Text("DashMat", fw=700),
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
                                                id="app-nav-home",
                                                href=HOME_PATH,
                                            ),
                                            dmc.MenuItem(
                                                "Analytics Tool",
                                                id="app-nav-analytics",
                                                href=ANALYTICS_PATH,
                                            ),
                                            dmc.MenuItem(
                                                "Portfolio Optimization",
                                                id="app-nav-portopt",
                                                href=PORTOPT_PATH,
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                ),
                dmc.AppShellMain(page_container),
            ],
        ),
    ]
)


@app.callback(
    Output("app-nav-home", "href"),
    Output("app-nav-analytics", "href"),
    Output("app-nav-portopt", "href"),
    Input("userinfo", "data"),
)
def update_app_nav_links(userinfo):
    home_path = _registry_path("pages.home", "/")
    analytics_path = _registry_path("pages.analyticstool", "/analyticstool")
    portopt_path = _registry_path("pages.portopt", "/portopt")

    if (userinfo or {}).get("role") == "Test":
        return (
            home_path,
            "/restricted?target=Analytics%20Tool",
            "/restricted?target=Portfolio%20Optimization",
        )

    return home_path, analytics_path, portopt_path

# Apply theme to MantineProvider
app.clientside_callback(
    "function(theme) { return theme || 'light'; }",
    Output("mantine-provider", "forceColorScheme"),
    Input("theme-store", "data"),
)

# Dark mode toggle callbacks are defined in each page module
# (analyticstool.py and portopt.py) to avoid referencing cross-page
# component IDs that don't exist when the other page is rendered.

if __name__ == "__main__":
    import sys
    debug = "--debug" in sys.argv
    app.run(debug=debug)
