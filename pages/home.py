"""Home portal page for DashMat."""

import dash_mantine_components as dmc
from dash import register_page

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
                                dmc.Anchor(
                                    dmc.Button(
                                        "Open Dashboard",
                                        size="lg",
                                        variant="filled",
                                    ),
                                    href="/dashboard",
                                    mt="md",
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)
