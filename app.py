"""DashMat - Market Returns Time Series Dashboard."""

import dash_mantine_components as dmc
from dash import Dash, dcc, page_container
from cache_config import init_cache

# Initialize the app with multi-page support
app = Dash(
    __name__,
    use_pages=True,
    suppress_callback_exceptions=True,
)

# Initialize cache for performance optimization (after app creation)
cache = init_cache(app.server)

# Layout wraps page content with MantineProvider
# Shared stores are defined here so they are accessible across all pages
app.layout = dmc.MantineProvider(
    children=[
        dcc.Store(id="analyticstool-raw-data-store", data=None, storage_type="session"),
        dcc.Store(id="analyticstool-original-periodicity-store", data="daily", storage_type="session"),
        dcc.Store(id="analyticstool-pending-new-series-store", data=[], storage_type="session"),
        page_container,
    ]
)

if __name__ == "__main__":
    import sys
    debug = "--debug" in sys.argv
    app.run(debug=debug)
