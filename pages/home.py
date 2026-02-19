"""Home portal page for DashMat."""

import dash_mantine_components as dmc
from dash import register_page, callback, Input, Output

register_page(__name__, path="/", name="Home", title="DashMat")

layout = dmc.Container(
    size="lg",
    py="xl",
    children=[
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
                                            id="home-analytics-link",
                                            href="/analyticstool",
                                        ),
                                        dmc.Anchor(
                                            dmc.Button(
                                                "Portfolio Optimization",
                                                size="lg",
                                                variant="outline",
                                            ),
                                            id="home-portopt-link",
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

@callback(
    Output("home-analytics-link", "href"),
    Output("home-portopt-link", "href"),
    Input("userinfo", "data"),
    prevent_initial_call=True,
)
def update_home_nav_links(userinfo):
    if (userinfo or {}).get("role") == "Test":
        return (
            "/restricted?target=Analytics%20Tool",
            "/restricted?target=Portfolio%20Optimization",
        )
    return "/analyticstool", "/portopt"
