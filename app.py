"""DashMat - Market Returns Time Series Dashboard."""

import dash_mantine_components as dmc
from dash import Dash, page_container

# Initialize the app with multi-page support
app = Dash(__name__, use_pages=True)

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
