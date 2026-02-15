"""Home portal page for DashMat."""

import dash_mantine_components as dmc
from dash import register_page, html, callback, Input, Output, State, clientside_callback
from dash_iconify import DashIconify

register_page(__name__, path="/", name="Home", title="DashMat")

layout = dmc.Container(
    size="lg",
    py="xl",
    children=[
        # Theme Toggle (Top Right)
        dmc.Group(
            justify="flex-end",
            mb="xl",
            children=[
                dmc.ActionIcon(
                    DashIconify(icon="tabler:sun", width=20),
                    id="dashmat-home-theme-toggle",
                    variant="outline",
                    size="lg",
                    color="yellow",
                ),
            ],
        ),
        dmc.Stack(
            align="center",
            gap="xl",
            children=[
                dmc.Title("DashMat", order=1),
                dmc.Text(
                    "Market Returns Time Series Dashboard",
                    size="xl",
                    c="dimmed",
                ),
                dmc.Paper(
                    shadow="md",
                    p="xl",
                    withBorder=True,
                    children=[
                        dmc.Stack(
                            gap="md",
                            children=[
                                dmc.Text(
                                    "Analyze and visualize market returns data with powerful tools:",
                                    size="lg",
                                ),
                                dmc.List(
                                    [
                                        dmc.ListItem("Upload Excel or CSV files with returns data"),
                                        dmc.ListItem("Select series and assign benchmarks"),
                                        dmc.ListItem("Toggle between total and excess returns"),
                                        dmc.ListItem("Convert periodicity (daily, weekly, monthly)"),
                                        dmc.ListItem("View comprehensive statistics"),
                                        dmc.ListItem("Export data to Excel"),
                                    ],
                                ),
                                dmc.Group(
                                    gap="md",
                                    mt="md",
                                    children=[
                                        dmc.Anchor(
                                            dmc.Button(
                                                "Analytics Tool",
                                                size="lg",
                                                variant="filled",
                                            ),
                                            id="dashmat-home-analytics-link",
                                            href="/analyticstool",
                                        ),
                                        dmc.Anchor(
                                            dmc.Button(
                                                "Portfolio Optimization",
                                                size="lg",
                                                variant="outline",
                                            ),
                                            id="dashmat-home-portopt-link",
                                            href="/portopt",
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)

# Clientside callback to toggle theme in local storage store
clientside_callback(
    """
    function(n_clicks, current_theme) {
        if (!n_clicks) return window.dash_clientside.no_update;
        return current_theme === "dark" ? "light" : "dark";
    }
    """,
    Output("dashmat-theme-store", "data", allow_duplicate=True),
    Input("dashmat-home-theme-toggle", "n_clicks"),
    State("dashmat-theme-store", "data"),
    prevent_initial_call=True,
)

# Server-side callback to update toggle icon and color
@callback(
    Output("dashmat-home-theme-toggle", "children"),
    Output("dashmat-home-theme-toggle", "color"),
    Input("dashmat-theme-store", "data"),
)
def update_toggle_icon(theme):
    if theme == "dark":
        return DashIconify(icon="tabler:sun", width=20), "yellow"
    return DashIconify(icon="tabler:moon", width=20), "blue"


@callback(
    Output("dashmat-home-analytics-link", "href"),
    Output("dashmat-home-portopt-link", "href"),
    Input("dashmat-userinfo", "data"),
    prevent_initial_call=True,
)
def update_home_nav_links(userinfo):
    if (userinfo or {}).get("role") == "Test":
        return (
            "/restricted?target=Analytics%20Tool",
            "/restricted?target=Portfolio%20Optimization",
        )
    return "/analyticstool", "/portopt"
