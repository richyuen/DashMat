"""DashMat - Market Returns Time Series Dashboard."""

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
        page_container,
    ]
)

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
