"""DashMat - Market Returns Time Series Dashboard."""

import dash_mantine_components as dmc
from dash import Dash, page_container
from flask_caching import Cache

# Initialize the app with multi-page support
app = Dash(__name__, use_pages=True)

# Configure cache for performance optimization
cache = Cache(app.server, config={
    'CACHE_TYPE': 'SimpleCache',
    'CACHE_DEFAULT_TIMEOUT': 300,  # 5 minutes
    'CACHE_THRESHOLD': 100,  # Maximum number of cached items
})

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
