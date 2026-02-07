"""DashMat - Market Returns Time Series Dashboard."""

import diskcache
import dash_mantine_components as dmc
from dash import Dash, DiskcacheManager, dcc, page_container
from cache_config import init_cache

# Background callback manager using diskcache
cache_disk = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache_disk)

# Initialize the app with multi-page support
app = Dash(
    __name__,
    use_pages=True,
    suppress_callback_exceptions=True,
    background_callback_manager=background_callback_manager,
)

# Initialize cache for performance optimization (after app creation)
cache = init_cache(app.server)

# Layout wraps page content with MantineProvider
# Shared stores are defined here so they are accessible across all pages
app.layout = dmc.MantineProvider(
    children=[
        dcc.Store(id="raw-data-store", data=None, storage_type="session"),
        dcc.Store(id="original-periodicity-store", data="daily", storage_type="session"),
        dcc.Store(id="pending-new-series-store", data=[], storage_type="session"),
        page_container,
    ]
)

if __name__ == "__main__":
    import sys
    debug = "--debug" in sys.argv
    app.run(debug=debug)
