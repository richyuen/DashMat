"""DashMat - Market Returns Time Series Dashboard."""

import dash_mantine_components as dmc
from dash import Dash, page_container
from cache_config import init_cache

# Initialize the app with multi-page support
app = Dash(__name__, use_pages=True, suppress_callback_exceptions=True)

# Initialize cache for performance optimization (after app creation)
cache = init_cache(app.server)

# Layout wraps page content with MantineProvider
app.layout = dmc.MantineProvider(
    children=[
        page_container,
    ]
)

if __name__ == "__main__":
    import sys
    debug = "--debug" in sys.argv
    app.run(debug=debug)
