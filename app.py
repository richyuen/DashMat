"""DashMat - Market Returns Time Series Dashboard."""

import dash
import dash_mantine_components as dmc
from dash import Dash, Input, Output, State, dcc, page_container
from dash.exceptions import PreventUpdate
from dash_iconify import DashIconify
from cache_config import init_cache

# Initialize the app with multi-page support
app = Dash(
    __name__,
    use_pages=True,
    suppress_callback_exceptions=True,
)

# Initialize cache for performance optimization (after app creation)
cache = init_cache(app.server)

USERINFO_DATA = {"role": "Admin"}
HAS_COLOR_SCHEME_TOGGLE = hasattr(dmc, "ColorSchemeToggle")
HAS_PRE_RENDER_COLOR_SCHEME = hasattr(dmc, "pre_render_color_scheme")


def _build_theme_toggle(toggle_id: str):
    if HAS_COLOR_SCHEME_TOGGLE:
        return dmc.ColorSchemeToggle(id=toggle_id)
    return dmc.ActionIcon(
        DashIconify(icon="tabler:moon", width=20),
        id=toggle_id,
        variant="outline",
        size="lg",
        color="blue",
    )


def _init_pre_render_color_scheme_helper():
    if not HAS_PRE_RENDER_COLOR_SCHEME:
        return None

    attempts = (
        lambda: dmc.pre_render_color_scheme(
            mantine_provider_id="mantine-provider",
            toggle_id="app-theme-toggle",
        ),
        lambda: dmc.pre_render_color_scheme(mantine_provider_id="mantine-provider"),
        lambda: dmc.pre_render_color_scheme("mantine-provider", "app-theme-toggle"),
        lambda: dmc.pre_render_color_scheme("mantine-provider"),
        lambda: dmc.pre_render_color_scheme(),
    )

    for attempt in attempts:
        try:
            return attempt()
        except TypeError:
            continue
        except Exception:
            return None
    return None


PRE_RENDER_COLOR_SCHEME = _init_pre_render_color_scheme_helper()


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
    return None

HOME_PATH = _registry_path("pages.home", "/")
ANALYTICS_PATH = _registry_path("pages.analyticstool", "/analyticstool")
PORTOPT_PATH = _registry_path("pages.portopt", "/portopt")

# Layout wraps page content with MantineProvider
# Shared stores are defined here so they are accessible across all pages
app.layout = dmc.MantineProvider(
    id="mantine-provider",
    forceColorScheme="light",
    children=[
        PRE_RENDER_COLOR_SCHEME,
        dcc.Store(id="dashmat-raw-data-store", data=None, storage_type="session"),
        dcc.Store(id="dashmat-original-periodicity-store", data="daily", storage_type="session"),
        dcc.Store(id="dashmat-pending-new-series-store", data=[], storage_type="session"),
        dcc.Store(id="dashmat-saved-series-cache-store", data=None, storage_type="session"),
        dcc.Store(id="userinfo", data=USERINFO_DATA, storage_type="session"),
        dcc.Store(id="theme-store", data="light", storage_type="local"),
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
                                    _build_theme_toggle("app-theme-toggle"),
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
                        ],
                    ),
                ),
                dmc.AppShellMain(
                    page_container,
                    style={"paddingTop": "53px"},
                ),
            ],
        ),
    ]
)


@app.callback(
    Output("app-nav-home", "href"),
    Output("app-nav-analytics", "href"),
    Output("app-nav-portopt", "href"),
    Input("userinfo", "data"),
    prevent_initial_call=True,
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


if HAS_COLOR_SCHEME_TOGGLE:
    # Keep existing chart callbacks compatible while ColorSchemeToggle drives MantineProvider.
    app.clientside_callback(
        "function(scheme) { return (scheme === 'dark' || scheme === 'light') ? scheme : window.dash_clientside.no_update; }",
        Output("theme-store", "data", allow_duplicate=True),
        Input("mantine-provider", "forceColorScheme"),
    )
else:
    # Legacy fallback for DMC versions before ColorSchemeToggle.
    app.clientside_callback(
        "function(theme) { return theme || 'light'; }",
        Output("mantine-provider", "forceColorScheme"),
        Input("theme-store", "data"),
    )

    app.clientside_callback(
        """
        function(n_clicks, current_theme) {
            if (!n_clicks) return window.dash_clientside.no_update;
            return current_theme === "dark" ? "light" : "dark";
        }
        """,
        Output("theme-store", "data", allow_duplicate=True),
        Input("app-theme-toggle", "n_clicks"),
        State("theme-store", "data"),
        prevent_initial_call=True,
    )

    @app.callback(
        Output("app-theme-toggle", "children"),
        Output("app-theme-toggle", "color"),
        Input("theme-store", "data"),
    )
    def _update_fallback_theme_icon(theme):
        if theme == "dark":
            return DashIconify(icon="tabler:sun", width=20), "yellow"
        return DashIconify(icon="tabler:moon", width=20), "blue"

# Theme consumer callbacks are defined in page modules for charts.

if __name__ == "__main__":
    import sys
    debug = "--debug" in sys.argv
    app.run(debug=debug)
