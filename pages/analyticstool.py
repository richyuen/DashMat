"""Analytics tool page - Market Returns Time Series Dashboard."""

from io import BytesIO, StringIO
import hashlib
import json

import dash_ag_grid as dag
import dash_mantine_components as dmc
from dash_iconify import DashIconify
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Input, Output, State, callback, dcc, html, no_update, register_page, ALL, clientside_callback, callback_context
from dash.exceptions import PreventUpdate

import cache_config
from utils.parsing import detect_periodicity, parse_uploaded_file
from utils.returns import (
    calculate_calendar_year_returns,
    calculate_excess_returns,
    calculate_rolling_returns,
    create_monthly_view,
    get_available_periodicities,
    get_working_returns,
    json_to_df,
    merge_returns,
    resample_returns,
    resample_returns_cached,
)
from utils.statistics import (
    calculate_drawdown,
    calculate_growth_of_dollar,
    calculate_statistics_cached,
    generate_correlogram_cached,
)

register_page(__name__, path="/analyticstool", name="Analytics Tool", title="Analytics Tool")

# Performance optimization constants
MAX_SCATTER_MATRIX_SIZE = 10  # Maximum series for scatter matrix (creates n² subplots)

# Statistics row order and formatting
STATS_CONFIG = [
    ("Start Date", None),
    ("End Date", None),
    ("Number of Periods", None),
    ("Cumulative Return", ".2%"),
    ("Annualized Return", ".2%"),
    ("Annualized Volatility", ".2%"),
    ("Sharpe Ratio", ".2f"),
    ("Sortino Ratio", ".2f"),
    ("Annualized Excess Return", ".2%"),
    ("Annualized Tracking Error", ".2%"),
    ("Information Ratio", ".2f"),
    ("Correlation", ".2f"),
    ("Hit Rate", ".2%"),
    ("Hit Rate (vs Benchmark)", ".2%"),
    ("Best Period Return", ".2%"),
    ("Worst Period Return", ".2%"),
    ("Maximum Drawdown", ".2%"),
    ("Skewness", ".2f"),
    ("Kurtosis", ".2f"),
    ("1Y Annualized Return", ".2%"),
    ("1Y Annualized Volatility", ".2%"),
    ("1Y Sharpe Ratio", ".2f"),
    ("1Y Sortino Ratio", ".2f"),
    ("1Y Excess Return", ".2%"),
    ("1Y Tracking Error", ".2%"),
    ("1Y Information Ratio", ".2f"),
    ("1Y Correlation", ".2f"),
    ("3Y Annualized Return", ".2%"),
    ("3Y Annualized Volatility", ".2%"),
    ("3Y Sharpe Ratio", ".2f"),
    ("3Y Sortino Ratio", ".2f"),
    ("3Y Excess Return", ".2%"),
    ("3Y Tracking Error", ".2%"),
    ("3Y Information Ratio", ".2f"),
    ("3Y Correlation", ".2f"),
    ("5Y Annualized Return", ".2%"),
    ("5Y Annualized Volatility", ".2%"),
    ("5Y Sharpe Ratio", ".2f"),
    ("5Y Sortino Ratio", ".2f"),
    ("5Y Excess Return", ".2%"),
    ("5Y Tracking Error", ".2%"),
    ("5Y Information Ratio", ".2f"),
    ("5Y Correlation", ".2f"),
]


def df_to_json(df: pd.DataFrame) -> str:
    """Convert DataFrame to JSON string for storage."""
    return df.to_json(date_format="iso", orient="split")


def build_welcome_screen():
    return dmc.Stack(
        align="center",
        justify="center",
        h=400,
        children=[
            DashIconify(icon="tabler:chart-arrows-vertical", width=60, color="#adb5bd"),
            dmc.Text("Welcome to Analytics Tool", size="xl", fw=500, c="dimmed", mt="md"),
            dmc.Text("Use the File menu to add data series.", size="sm", c="dimmed"),
            dmc.Button(
                "Add series from file", 
                leftSection=DashIconify(icon="tabler:upload"),
                variant="light",
                mt="lg",
                id="welcome-add-series-btn"
            )
        ]
    )

# Callback for the welcome button
clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks) {
            var uploadDiv = document.getElementById('upload-data');
            if (uploadDiv) {
                var input = uploadDiv.querySelector('input[type="file"]');
                if (input) {
                    input.click();
                }
            }
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("upload-trigger", "children", allow_duplicate=True),
    Input("welcome-add-series-btn", "n_clicks"),
    prevent_initial_call=True,
)


def build_main_layout(periodicity_options, periodicity_value, returns_type, vol_scaler,
                      active_tab, rolling_window, rolling_metric, rolling_return_type, rolling_chart_switch,
                      drawdown_chart_switch, growth_chart_switch, monthly_view, monthly_series,
                      monthly_series_options, monthly_select_disabled):
    
    # Calculate visibility styles
    rolling_grid_style = {"display": "block"} if rolling_chart_switch == "table" else {"display": "none"}
    rolling_chart_style = {"display": "block"} if rolling_chart_switch == "chart" else {"display": "none"}
    
    drawdown_grid_style = {"display": "block"} if drawdown_chart_switch == "table" else {"display": "none"}
    drawdown_chart_style = {"display": "block"} if drawdown_chart_switch == "chart" else {"display": "none"}
    
    growth_grid_style = {"display": "block"} if growth_chart_switch == "table" else {"display": "none"}
    growth_chart_style = {"display": "block"} if growth_chart_switch == "chart" else {"display": "none"}

    rolling_return_type_disabled = False if rolling_metric in ["total_return", "excess_return"] else True
    rolling_return_type_style = {} if not rolling_return_type_disabled else {"opacity": 0.5, "pointerEvents": "none"}

    return html.Div([
        # Controls Section (Collapsible, starts expanded)
        dmc.Accordion(
            value="controls",
            mb="md",
            variant="contained",
            children=[
                dmc.AccordionItem(
                    value="controls",
                    children=[
                        dmc.AccordionControl("Controls"),
                        dmc.AccordionPanel(
                            children=[
                                dmc.Group(
                                    mb="md",
                                    align="flex-start",
                                    children=[
                                        html.Div([
                                            dmc.Text("Series Selection", size="sm", mb=3, fw=500),
                                            dmc.Button(
                                                "Select Series",
                                                id="open-series-modal-button",
                                                variant="light",
                                                size="sm",
                                                w=200,
                                            ),
                                        ]),
                                        dmc.Select(
                                            id="periodicity-select",
                                            label="Periodicity",
                                            data=periodicity_options,
                                            value=periodicity_value,
                                            w=200,
                                            disabled=False,
                                        ),
                                        html.Div([
                                            dmc.Text("Returns Type", size="sm", mb=3, fw=500),
                                            dmc.SegmentedControl(
                                                id="returns-type-select",
                                                data=[
                                                    {"value": "total", "label": "Total"},
                                                    {"value": "excess", "label": "Excess"},
                                                ],
                                                value=returns_type,
                                                w=250,
                                            ),
                                        ]),
                                        html.Div([
                                            dmc.Text("Vol Scaler", size="sm", mb=3, fw=500),
                                            dmc.Tooltip(
                                                label="A value of 0% disables the volatility scaling.",
                                                position="top",
                                                withArrow=True,
                                                children=dmc.NumberInput(
                                                    id="vol-scaler-input",
                                                    value=vol_scaler,
                                                    min=0,
                                                    step=1,
                                                    suffix="%",
                                                    w=120,
                                                ),
                                            ),
                                        ]),
                                    ],
                                ),
                                html.Div([
                                    html.Div(
                                        id="date-picker-wrapper",
                                        children=[
                                            html.Div([
                                                dmc.DateInput(
                                                    id="start-date-picker",
                                                    label="Start Date",
                                                    value=None,
                                                    w=200,
                                                    valueFormat="YYYY-MM-DD",
                                                ),
                                            ], style={"marginRight": "15px"}),
                                            html.Div([
                                                dmc.DateInput(
                                                    id="end-date-picker",
                                                    label="End Date",
                                                    value=None,
                                                    w=200,
                                                    valueFormat="YYYY-MM-DD",
                                                ),
                                            ], style={"marginRight": "15px"}),
                                            html.Div([
                                                dmc.Button(
                                                    "Common Range",
                                                    id="common-range-button",
                                                    size="xs",
                                                    variant="outline",
                                                    disabled=True,
                                                    w=120,
                                                ),
                                            ], style={"marginRight": "10px", "alignSelf": "flex-end", "marginBottom": "2px"}),
                                            html.Div([
                                                dmc.Button(
                                                    "Max Range",
                                                    id="maximum-range-button",
                                                    size="xs",
                                                    variant="outline",
                                                    disabled=True,
                                                    w=120,
                                                ),
                                            ], style={"alignSelf": "flex-end", "marginBottom": "2px"}),
                                        ],
                                        style={"display": "flex", "opacity": 0.5, "pointerEvents": "none", "alignItems": "flex-start"},
                                    ),
                                ], style={"marginBottom": "1rem"}),
                            ]
                        ),
                    ],
                ),
            ],
        ),

        # Tabs with AG Grid and Statistics
        dmc.Tabs(
            id="main-tabs",
            value=active_tab,
            children=[
                dmc.TabsList(
                    children=[
                        dmc.TabsTab("Statistics", value="statistics"),
                        dmc.TabsTab("Returns", value="returns"),
                        dmc.TabsTab("Rolling", value="rolling"),
                        dmc.TabsTab("Calendar Year", value="calendar"),
                        dmc.TabsTab("Growth of $1", value="growth"),
                        dmc.TabsTab("Drawdown", value="drawdown"),
                        dmc.TabsTab("Correlogram", value="correlogram"),
                    ],
                ),
                dmc.TabsPanel(
                    value="returns",
                    pt="md",
                    children=[
                        dcc.Loading(
                            id="loading-returns",
                            type="default",
                            children=[
                                dag.AgGrid(
                                    id="returns-grid",
                                    columnDefs=[],
                                    rowData=[],
                                    defaultColDef={
                                        "sortable": True,
                                        "resizable": True,
                                    },
                                    style={"height": "600px"},
                                    dashGridOptions={
                                        "animateRows": True,
                                        "pagination": True,
                                        "paginationPageSize": 100,
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
                dmc.TabsPanel(
                    value="rolling",
                    pt="md",
                    children=[
                        dmc.Group(
                            mb="md",
                            children=[
                                dmc.Select(
                                    id="rolling-metric-select",
                                    data=[
                                        {"value": "total_return", "label": "Total Return"},
                                        {"value": "volatility", "label": "Volatility"},
                                        {"value": "sharpe_ratio", "label": "Sharpe Ratio"},
                                        {"value": "sortino_ratio", "label": "Sortino Ratio"},
                                        {"value": "excess_return", "label": "Excess Return"},
                                        {"value": "tracking_error", "label": "Tracking Error"},
                                        {"value": "information_ratio", "label": "Information Ratio"},
                                        {"value": "correlation", "label": "Correlation"},
                                    ],
                                    value=rolling_metric,
                                    w=150,
                                    size="sm",
                                    clearable=False,
                                ),
                                dmc.Select(
                                    id="rolling-window-select",
                                    data=[
                                        {"value": "3m", "label": "3-month"},
                                        {"value": "6m", "label": "6-month"},
                                        {"value": "1y", "label": "1-year"},
                                        {"value": "3y", "label": "3-year"},
                                        {"value": "5y", "label": "5-year"},
                                        {"value": "10y", "label": "10-year"},
                                    ],
                                    value=rolling_window,
                                    w=120,
                                    size="sm",
                                ),
                                dmc.SegmentedControl(
                                    id="rolling-return-type-select",
                                    data=[
                                        {"value": "cumulative", "label": "Cumulative"},
                                        {"value": "annualized", "label": "Annualized"},
                                    ],
                                    value=rolling_return_type,
                                    size="sm",
                                    disabled=rolling_return_type_disabled,
                                    style=rolling_return_type_style,
                                ),
                                dmc.SegmentedControl(
                                    id="rolling-chart-switch",
                                    data=[
                                        {"value": "table", "label": "Table"},
                                        {"value": "chart", "label": "Chart"},
                                    ],
                                    value=rolling_chart_switch,
                                    size="sm",
                                ),
                            ],
                        ),
                        html.Div(
                            id="rolling-grid-container",
                            style=rolling_grid_style,
                            children=[
                                dag.AgGrid(
                                    id="rolling-grid",
                                    columnDefs=[],
                                    rowData=[],
                                    defaultColDef={
                                        "sortable": True,
                                        "resizable": True,
                                    },
                                    style={"height": "550px"},
                                    dashGridOptions={
                                        "animateRows": True,
                                        "pagination": True,
                                        "paginationPageSize": 100,
                                    },
                                ),
                            ],
                        ),
                        html.Div(
                            id="rolling-chart-container",
                            style=rolling_chart_style,
                            children=[
                                html.Div(id="rolling-chart-wrapper"),
                            ],
                        ),
                    ],
                ),
                dmc.TabsPanel(
                    value="statistics",
                    pt="md",
                    children=[
                        dcc.Loading(
                            id="loading-statistics",
                            type="default",
                            children=[
                                dag.AgGrid(
                                    id="statistics-grid",
                                    columnDefs=[],
                                    rowData=[],
                                    defaultColDef={
                                        "resizable": True,
                                    },
                                    style={"height": "700px"},
                                    dashGridOptions={
                                        "animateRows": True,
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
                dmc.TabsPanel(
                    value="calendar",
                    pt="md",
                    children=[
                        dmc.Group(
                            mb="md",
                            children=[
                                dmc.SegmentedControl(
                                    id="monthly-view-checkbox",
                                    data=[
                                        {"value": "annual", "label": "Annual"},
                                        {"value": "monthly", "label": "Monthly"},
                                    ],
                                    value=monthly_view,
                                    size="sm",
                                ),
                                dmc.Select(
                                    id="monthly-series-select",
                                    data=monthly_series_options,
                                    value=monthly_series,
                                    w=200,
                                    size="sm",
                                    placeholder="Select series",
                                    disabled=monthly_select_disabled,
                                ),
                            ],
                        ),
                        dag.AgGrid(
                            id="calendar-grid",
                            columnDefs=[],
                            rowData=[],
                            defaultColDef={
                                "sortable": True,
                                "resizable": True,
                            },
                            style={"height": "600px"},
                            dashGridOptions={
                                "animateRows": True,
                            },
                        ),
                    ],
                ),
                dmc.TabsPanel(
                    value="correlogram",
                    pt="md",
                    children=[
                        html.Div(id="correlogram-container"),
                    ],
                ),
                dmc.TabsPanel(
                    value="growth",
                    pt="md",
                    children=[
                        dmc.Group(
                            mb="md",
                            children=[
                                dmc.SegmentedControl(
                                    id="growth-chart-switch",
                                    data=[
                                        {"value": "table", "label": "Table"},
                                        {"value": "chart", "label": "Chart"},
                                    ],
                                    value=growth_chart_switch,
                                    size="sm",
                                ),
                            ],
                        ),
                        html.Div(
                            id="growth-chart-container",
                            style=growth_chart_style,
                            children=[
                                html.Div(id="growth-charts-container"),
                            ],
                        ),
                        html.Div(
                            id="growth-grid-container",
                            style=growth_grid_style,
                            children=[
                                dag.AgGrid(
                                    id="growth-grid",
                                    columnDefs=[],
                                    rowData=[],
                                    defaultColDef={
                                        "sortable": True,
                                        "resizable": True,
                                    },
                                    style={"height": "600px"},
                                    dashGridOptions={
                                        "animateRows": True,
                                        "pagination": True,
                                        "paginationPageSize": 100,
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
                dmc.TabsPanel(
                    value="drawdown",
                    pt="md",
                    children=[
                        dmc.Group(
                            mb="md",
                            children=[
                                dmc.SegmentedControl(
                                    id="drawdown-chart-switch",
                                    data=[
                                        {"value": "table", "label": "Table"},
                                        {"value": "chart", "label": "Chart"},
                                    ],
                                    value=drawdown_chart_switch,
                                    size="sm",
                                ),
                            ],
                        ),
                        html.Div(
                            id="drawdown-chart-container",
                            style=drawdown_chart_style,
                            children=[
                                html.Div(id="drawdown-charts"),
                            ],
                        ),
                        html.Div(
                            id="drawdown-grid-container",
                            style=drawdown_grid_style,
                            children=[
                                dag.AgGrid(
                                    id="drawdown-grid",
                                    columnDefs=[],
                                    rowData=[],
                                    defaultColDef={
                                        "sortable": True,
                                        "resizable": True,
                                    },
                                    style={"height": "600px"},
                                    dashGridOptions={
                                        "animateRows": True,
                                        "pagination": True,
                                        "paginationPageSize": 100,
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ])


layout = dmc.Container(
    size="xl",
    py="md",
    children=[
        # Menu Bar
        dmc.Paper(
            shadow="xs",
            p="xs",
            mb="md",
            withBorder=True,
            children=[
                dmc.Group(
                    gap="xs",
                    children=[
                        # File Menu (left)
                        dmc.Menu(
                            trigger="hover",
                            openDelay=100,
                            closeDelay=200,
                            children=[
                                dmc.MenuTarget(
                                    dmc.Button("File", variant="subtle", size="sm"),
                                ),
                                dmc.MenuDropdown(
                                    children=[
                                        dmc.MenuItem(
                                            "Add series from file",
                                            id="menu-add-series",
                                        ),
                                        dmc.MenuItem(
                                            "Download Excel",
                                            id="menu-download-excel",
                                            disabled=True,
                                        ),
                                        dmc.MenuDivider(),
                                        dmc.MenuItem(
                                            "Exit",
                                            id="menu-exit",
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        # Edit Menu (left)
                        dmc.Menu(
                            trigger="hover",
                            openDelay=100,
                            closeDelay=200,
                            children=[
                                dmc.MenuTarget(
                                    dmc.Button("Edit", variant="subtle", size="sm"),
                                ),
                                dmc.MenuDropdown(
                                    children=[
                                        dmc.MenuItem(
                                            "Clear all series",
                                            id="menu-clear-all-series",
                                        ),
                                        dmc.MenuItem(
                                            "Clear session storage and refresh",
                                            id="menu-clear-local-storage",
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        # Spacer
                        dmc.Box(style={"flexGrow": 1}),
                        # Help Menu (right)
                        dmc.Menu(
                            trigger="hover",
                            openDelay=100,
                            closeDelay=200,
                            children=[
                                dmc.MenuTarget(
                                    dmc.Button("Help", variant="subtle", size="sm"),
                                ),
                                dmc.MenuDropdown(
                                    children=[
                                        dmc.MenuItem("(No help topics available)", disabled=True),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        
        # Hidden file upload (triggered by menu item) - Moved here for startup priority
        html.Div(
            dcc.Upload(
                id="upload-data",
                children=html.Div(id="upload-trigger"),
                multiple=False,
                accept=".csv,.xlsx,.xls",
            ),
            style={"display": "none"},
        ),

        # Series Selection Modal
        dmc.Modal(
            id="series-selection-modal",
            title="Select Series",
            size="xl",
            children=[
                # Alert for messages (with close button)
                dmc.Alert(
                    id="alert-message",
                    title="Info",
                    color="blue",
                    hide=True,
                    mb="md",
                    withCloseButton=True,
                ),
                html.Div(
                    id="series-selection-container",
                    children=[dmc.Text("Upload data to select series", size="sm", c="dimmed")],
                    style={"maxHeight": "60vh", "overflowY": "auto"},
                ),
                dmc.Group(
                    mt="md",
                    justify="flex-end",
                    children=[
                        dmc.Button("Cancel", id="modal-cancel-button", variant="outline", color="red"),
                        dmc.Button("OK", id="modal-ok-button", color="blue"),
                    ],
                ),
            ],
        ),

        # Welcome Screen (Initially Visible)
        html.Div(
            id="welcome-screen-container",
            children=build_welcome_screen(),
            style={"display": "block"}
        ),

        # Main App Container (Initially Hidden)
        html.Div(
            id="main-app-container",
            children=build_main_layout(
                periodicity_options=[{"value": "daily", "label": "Daily"}],
                periodicity_value="daily",
                returns_type="total",
                vol_scaler=0,
                active_tab="statistics",
                rolling_window="1y",
                rolling_metric="total_return",
                rolling_return_type="annualized",
                rolling_chart_switch="chart",
                drawdown_chart_switch="chart",
                growth_chart_switch="chart",
                monthly_view="annual",
                monthly_series=None,
                monthly_series_options=[],
                monthly_select_disabled=True
            ),
            style={"display": "none"}
        ),

        # Hidden stores for state management (using local storage for persistence)
        # These MUST remain in the static layout to be available for callbacks
        dcc.Store(id="raw-data-store", data=None, storage_type="session"),
        dcc.Store(id="original-periodicity-store", data="daily", storage_type="session"),
        dcc.Store(id="benchmark-assignments-store", data={}, storage_type="session"),
        dcc.Store(id="long-short-store", data={}, storage_type="session"),
        dcc.Store(id="periodicity-value-store", data="daily", storage_type="session"),
        dcc.Store(id="returns-type-value-store", data="total", storage_type="session"),
        dcc.Store(id="series-select-value-store", data=[], storage_type="session"),
        dcc.Store(id="series-order-store", data=[], storage_type="session"),
        dcc.Store(id="active-tab-store", data="statistics", storage_type="session"),
        dcc.Store(id="rolling-window-store", data="1y", storage_type="session"),
        dcc.Store(id="rolling-metric-store", data="total_return", storage_type="session"),
        dcc.Store(id="rolling-return-type-store", data="annualized", storage_type="session"),
        dcc.Store(id="rolling-chart-switch-store", data="chart", storage_type="session"),
        dcc.Store(id="drawdown-chart-switch-store", data="chart", storage_type="session"),
        dcc.Store(id="growth-chart-switch-store", data="chart", storage_type="session"),
        dcc.Store(id="monthly-view-store", data="annual", storage_type="session"),
        dcc.Store(id="monthly-series-store", data=None, storage_type="session"),
        dcc.Store(id="date-range-store", data=None, storage_type="session"),
        dcc.Store(id="vol-scaler-value-store", data=0, storage_type="session"),
        dcc.Store(id="vol-scaling-assignments-store", data={}, storage_type="session"),
        dcc.Store(id="download-enabled-store", data=False),
        dcc.Store(id="first-load-store", data=False, storage_type="session"),
        # Temporary stores for modal state
        dcc.Store(id="temp-series-select", data=[]),
        dcc.Store(id="temp-benchmark-assignments-store", data={}),
        dcc.Store(id="temp-long-short-store", data={}),
        dcc.Store(id="temp-vol-scaling-assignments-store", data={}),
        dcc.Store(id="temp-series-order-store", data=[]),
        dcc.Store(id="temp-deleted-series-store", data=[]),
        dcc.Download(id="download-excel"),
        dcc.Location(id="url-location", refresh=True),
        # Moved series-select and edit-mode to global scope
        dcc.Store(id="series-select", data=[], storage_type="session"),
        dcc.Store(id="series-edit-mode", data=None),

        # Store to trigger clientside focus on edit input
        dcc.Store(id="edit-box-focus-trigger", data=None),
        # Dummy div for clientside callback output
        html.Div(id="dummy-focus-output"),
    ],
)


@callback(
    Output("welcome-screen-container", "style"),
    Output("main-app-container", "style"),
    Output("periodicity-select", "data", allow_duplicate=True),
    Output("periodicity-select", "value", allow_duplicate=True),
    Output("returns-type-select", "value"),
    Output("vol-scaler-input", "value"),
    Output("main-tabs", "value"),
    Output("rolling-window-select", "value"),
    Output("rolling-metric-select", "value"),
    Output("rolling-return-type-select", "value"),
    Output("rolling-return-type-select", "disabled", allow_duplicate=True),
    Output("rolling-return-type-select", "style", allow_duplicate=True),
    Output("rolling-chart-switch", "value"),
    Output("drawdown-chart-switch", "value"),
    Output("growth-chart-switch", "value"),
    Output("monthly-view-checkbox", "value"),
    Output("series-select", "data"),
    Input("raw-data-store", "data"),
    State("original-periodicity-store", "data"),
    State("periodicity-value-store", "data"),
    State("series-select-value-store", "data"),
    State("returns-type-value-store", "data"),
    State("vol-scaler-value-store", "data"),
    State("active-tab-store", "data"),
    State("rolling-window-store", "data"),
    State("rolling-metric-store", "data"),
    State("rolling-return-type-store", "data"),
    State("rolling-chart-switch-store", "data"),
    State("drawdown-chart-switch-store", "data"),
    State("growth-chart-switch-store", "data"),
    State("monthly-view-store", "data"),
    State("monthly-series-store", "data"),
    prevent_initial_call="initial_duplicate",
)
def restore_application_state(raw_data, orig_periodicity, stored_periodicity, stored_series, stored_returns, stored_vol, stored_tab, stored_roll_win, stored_roll_metric, stored_roll_type, stored_roll_chart, stored_dd_chart, stored_gr_chart, stored_monthly_view, stored_monthly_series):
    if not raw_data:
        # Show welcome, hide main, reset defaults if needed
        return (
            {"display": "block"}, {"display": "none"},
            [{"value": "daily", "label": "Daily"}], "daily", "total", 0, "statistics",
            "1y", "total_return", "annualized", False, {}, "chart", "chart", "chart",
            "annual", []
        )

    try:
        df = json_to_df(raw_data)
        
        # Periodicity
        periodicity_options = get_available_periodicities(orig_periodicity or "daily")
        valid_periodicity = stored_periodicity if stored_periodicity in [p["value"] for p in periodicity_options] else (orig_periodicity or "daily")
        
        # Returns Type
        valid_returns = stored_returns if stored_returns in ["total", "excess"] else "total"
        
        # Vol Scaler
        valid_vol = stored_vol if stored_vol is not None else 0
        
        # Active Tab
        active_tab = stored_tab if stored_tab else "statistics"
        
        # Rolling
        roll_win = stored_roll_win if stored_roll_win else "1y"
        roll_metric = stored_roll_metric if stored_roll_metric else "total_return"
        roll_type = stored_roll_type if stored_roll_type else "annualized"
        roll_chart = stored_roll_chart if stored_roll_chart is not None else "chart"
        
        # Rolling Return Type Disabled Logic
        roll_type_disabled = False if roll_metric in ["total_return", "excess_return"] else True
        roll_type_style = {} if not roll_type_disabled else {"opacity": 0.5, "pointerEvents": "none"}
        
        # Drawdown
        dd_chart = stored_dd_chart if stored_dd_chart is not None else "chart"
        
        # Growth
        gr_chart = stored_gr_chart if stored_gr_chart is not None else "chart"
        
        # Monthly View
        monthly_view = stored_monthly_view if stored_monthly_view is not None else "annual"
        
        # Monthly Series Options & Selection
        current_selection = stored_series or []
        valid_selection = [s for s in current_selection if s in df.columns]
        
        monthly_series_options = [{"value": s, "label": s} for s in valid_selection]
        
        monthly_select_disabled = True
        monthly_series_val = None
        
        if monthly_view == "monthly":
            monthly_select_disabled = False
            if stored_monthly_series and stored_monthly_series in valid_selection:
                monthly_series_val = stored_monthly_series
            elif valid_selection:
                monthly_series_val = valid_selection[0]
        
        return (
            {"display": "none"}, {"display": "block"},
            periodicity_options, valid_periodicity, valid_returns, valid_vol, active_tab,
            roll_win, roll_metric, roll_type, roll_type_disabled, roll_type_style, roll_chart, dd_chart, gr_chart,
            monthly_view, valid_selection
        )

    except Exception:
        # Fallback to welcome screen on error
        return (
            {"display": "block"}, {"display": "none"},
            [{"value": "daily", "label": "Daily"}], "daily", "total", 0, "statistics",
            "1y", "total_return", "annualized", False, {}, "chart", "chart", "chart",
            "annual", []
        )


# Clientside callback to navigate to home on Exit
clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks) {
            window.location.href = '/';
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("url-location", "pathname"),
    Input("menu-exit", "n_clicks"),
    prevent_initial_call=True,
)


# Clientside callback to clear session storage and refresh page
clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks) {
            // Clear all sessionStorage keys specific to Analytics Tool
            const keysToRemove = [
                'series-select',
                'raw-data-store',
                'original-periodicity-store',
                'benchmark-assignments-store',
                'long-short-store',
                'periodicity-value-store',
                'returns-type-value-store',
                'series-select-value-store',
                'series-order-store',
                'active-tab-store',
                'rolling-window-store',
                'rolling-return-type-store',
                'rolling-chart-switch-store',
                'drawdown-chart-switch-store',
                'growth-chart-switch-store',
                'monthly-view-store',
                'monthly-series-store',
                'date-range-store',
                'vol-scaler-value-store',
                'vol-scaling-assignments-store'
            ];

            keysToRemove.forEach(key => {
                sessionStorage.removeItem(key);
            });

            // Refresh the page
            window.location.reload();
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("url-location", "pathname", allow_duplicate=True),
    Input("menu-clear-local-storage", "n_clicks"),
    prevent_initial_call=True,
)


# Clientside callback to trigger file upload from menu
clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks) {
            // Find the hidden upload input and click it
            var uploadDiv = document.getElementById('upload-data');
            if (uploadDiv) {
                var input = uploadDiv.querySelector('input[type="file"]');
                if (input) {
                    input.click();
                }
            }
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("upload-trigger", "children"),
    Input("menu-add-series", "n_clicks"),
    prevent_initial_call=True,
)


clientside_callback(
    """
    function(series_name) {
        if (series_name) {
            // Use a short delay to ensure the input is rendered
            setTimeout(function() {
                // The ID is a JSON string, e.g., {"type":"edit-series-input","series":"SPY"}
                // We construct the selector to find the input element.
                var selector = `[id*='"series":"${series_name}"'][id*='"type":"edit-series-input"']`;
                var inputElement = document.querySelector(selector);
                
                if (inputElement) {
                    inputElement.focus();
                    inputElement.select();
                }
            }, 50);
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("dummy-focus-output", "children"),
    Input("edit-box-focus-trigger", "data"),
    prevent_initial_call=True,
)


clientside_callback(
    """
    function(n_submit_list, input_ids) {
        if (!n_submit_list || n_submit_list.every(n => !n)) {
            return window.dash_clientside.no_update;
        }

        const ctx = window.dash_clientside.callback_context;
        const triggered = ctx.triggered[0];
        if (!triggered) {
            return window.dash_clientside.no_update;
        }

        const triggered_id_str = triggered.prop_id.split('.')[0]; // e.g., '{"type":"edit-series-input","series":"SPY"}'
        const triggered_id = JSON.parse(triggered_id_str);
        const series_name = triggered_id.series;

        if (series_name) {
            var saveButtonSelector = `[id*='"type":"save-edit-button"'][id*='"series":"${series_name}"']`;
            var saveButton = document.querySelector(saveButtonSelector);

            if (saveButton) {
                // Programmatically click the save button
                saveButton.click();
            }
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("dummy-focus-output", "children", allow_duplicate=True),  # Reuse dummy output
    Input({"type": "edit-series-input", "series": ALL}, "n_submit"),
    prevent_initial_call=True,
)


@callback(
    Output("series-selection-modal", "opened", allow_duplicate=True),
    Output("temp-series-select", "data", allow_duplicate=True),
    Output("temp-benchmark-assignments-store", "data", allow_duplicate=True),
    Output("temp-long-short-store", "data", allow_duplicate=True),
    Output("temp-series-order-store", "data", allow_duplicate=True),
    Output("temp-deleted-series-store", "data", allow_duplicate=True),
    Output("temp-vol-scaling-assignments-store", "data", allow_duplicate=True),
    Input("open-series-modal-button", "n_clicks"),
    State("series-select", "data"),
    State("benchmark-assignments-store", "data"),
    State("long-short-store", "data"),
    State("series-order-store", "data"),
    State("vol-scaling-assignments-store", "data"),
    prevent_initial_call=True,
)
def open_modal(n_clicks, current_select, current_bench, current_ls, current_order, current_vol_scaling):
    if not n_clicks:
        raise PreventUpdate
    return True, current_select, current_bench, current_ls, current_order, [], current_vol_scaling


@callback(
    Output("series-select", "data", allow_duplicate=True),
    Output("benchmark-assignments-store", "data", allow_duplicate=True),
    Output("long-short-store", "data", allow_duplicate=True),
    Output("series-order-store", "data", allow_duplicate=True),
    Output("series-selection-modal", "opened", allow_duplicate=True),
    Output("series-select-value-store", "data", allow_duplicate=True), # Sync persistence
    Output("raw-data-store", "data", allow_duplicate=True),
    Output("vol-scaling-assignments-store", "data", allow_duplicate=True),
    Input("modal-ok-button", "n_clicks"),
    State({"type": "series-include-checkbox", "series": ALL}, "checked"),
    State({"type": "series-include-checkbox", "series": ALL}, "id"),
    State("temp-benchmark-assignments-store", "data"),
    State("temp-long-short-store", "data"),
    State("temp-series-order-store", "data"),
    State("temp-deleted-series-store", "data"),
    State("raw-data-store", "data"),
    State("temp-vol-scaling-assignments-store", "data"),
    prevent_initial_call=True,
)
def on_modal_ok(n_clicks, checkbox_values, checkbox_ids, temp_bench, temp_ls, temp_order, temp_deleted, raw_data, temp_vol_scaling):
    if not n_clicks:
        raise PreventUpdate

    # Reconstruct selected series from checkbox states
    temp_select = []
    if checkbox_values and checkbox_ids:
        # Map checkbox values to series
        checkbox_map = {}
        for i, checkbox_id in enumerate(checkbox_ids):
            series = checkbox_id["series"]
            if i < len(checkbox_values):
                checkbox_map[series] = checkbox_values[i]
        
        # Use series order if available to maintain consistency
        order_to_use = temp_order if temp_order else (list(checkbox_map.keys()))
        for series in order_to_use:
             if checkbox_map.get(series, False):
                 temp_select.append(series)

    # Apply deletions to raw data
    updated_raw_data = raw_data
    if temp_deleted and raw_data:
        df = json_to_df(raw_data)
        # Filter out series that are actually in the columns
        series_to_drop = [s for s in temp_deleted if s in df.columns]
        if series_to_drop:
            df = df.drop(columns=series_to_drop)
            updated_raw_data = df_to_json(df)
            
            # Clean up assignments and order
            if temp_bench:
                temp_bench = {k: v for k, v in temp_bench.items() if k not in series_to_drop}
            if temp_ls:
                temp_ls = {k: v for k, v in temp_ls.items() if k not in series_to_drop}
            if temp_order:
                temp_order = [s for s in temp_order if s not in series_to_drop]
            if temp_vol_scaling:
                temp_vol_scaling = {k: v for k, v in temp_vol_scaling.items() if k not in series_to_drop}
            
            # Also remove from temp_select if present
            temp_select = [s for s in temp_select if s not in series_to_drop]

    return temp_select, temp_bench, temp_ls, temp_order, False, temp_select, updated_raw_data, temp_vol_scaling


@callback(
    Output("series-selection-modal", "opened", allow_duplicate=True),
    Input("modal-cancel-button", "n_clicks"),
    prevent_initial_call=True,
)
def on_modal_cancel(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    return False


@callback(
    Output("temp-series-order-store", "data", allow_duplicate=True),
    Output("temp-series-select", "data", allow_duplicate=True),
    Input({"type": "move-up-button", "series": ALL}, "n_clicks"),
    Input({"type": "move-down-button", "series": ALL}, "n_clicks"),
    State("temp-series-order-store", "data"),
    State("raw-data-store", "data"),
    State({"type": "series-include-checkbox", "series": ALL}, "checked"),
    State({"type": "series-include-checkbox", "series": ALL}, "id"),
    prevent_initial_call=True,
)
def reorder_series(up_clicks, down_clicks, current_order, raw_data, checkbox_values, checkbox_ids):
    """Reorder series when up/down buttons are clicked."""
    if raw_data is None or not current_order:
        raise PreventUpdate

    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    # Find which button was clicked
    triggered_id = ctx.triggered[0]["prop_id"]

    if not triggered_id:
        raise PreventUpdate

    # Parse the button ID
    try:
        button_data = eval(triggered_id.rsplit(".", 1)[0])
        button_type = button_data["type"]
        series_name = button_data["series"]
    except (SyntaxError, KeyError, ValueError):
        raise PreventUpdate

    # Reconstruct selected series from checkbox states
    current_selected = []
    if checkbox_values and checkbox_ids:
        checkbox_map = {}
        for i, checkbox_id in enumerate(checkbox_ids):
            s = checkbox_id["series"]
            if i < len(checkbox_values):
                checkbox_map[s] = checkbox_values[i]
        
        for s in current_order:
             if checkbox_map.get(s, False):
                 current_selected.append(s)

    # Find current index
    if series_name not in current_order:
        raise PreventUpdate

    current_idx = current_order.index(series_name)
    new_order = current_order.copy()

    # Move up or down
    if button_type == "move-up-button" and current_idx > 0:
        new_order[current_idx], new_order[current_idx - 1] = new_order[current_idx - 1], new_order[current_idx]
    elif button_type == "move-down-button" and current_idx < len(new_order) - 1:
        new_order[current_idx], new_order[current_idx + 1] = new_order[current_idx + 1], new_order[current_idx]
    else:
        raise PreventUpdate

    return new_order, current_selected


@callback(
    Output("raw-data-store", "data", allow_duplicate=True),
    Output("original-periodicity-store", "data", allow_duplicate=True),
    Output("benchmark-assignments-store", "data", allow_duplicate=True),
    Output("long-short-store", "data", allow_duplicate=True),
    Output("periodicity-value-store", "data", allow_duplicate=True),
    Output("returns-type-value-store", "data", allow_duplicate=True),
    Output("series-select-value-store", "data", allow_duplicate=True),
    Output("series-order-store", "data", allow_duplicate=True),
    Output("series-select", "data", allow_duplicate=True),
    Output("vol-scaler-value-store", "data", allow_duplicate=True),
    Output("vol-scaling-assignments-store", "data", allow_duplicate=True),
    Input("menu-clear-all-series", "n_clicks"),
    prevent_initial_call=True,
)
def clear_all_series(n_clicks):
    """Clear all loaded series and reset application state."""
    if n_clicks is None:
        raise PreventUpdate

    # Reset all stores to initial state
    return None, "daily", {}, {}, None, None, [], [], [], 0, {}





@callback(
    Output("periodicity-value-store", "data"),
    Input("periodicity-select", "value"),
    prevent_initial_call=True,
)
def save_periodicity(value):
    """Save periodicity selection to local storage."""
    return value


@callback(
    Output("returns-type-value-store", "data"),
    Input("returns-type-select", "value"),
    prevent_initial_call=True,
)
def save_returns_type(value):
    """Save returns type selection to local storage."""
    return value


@callback(
    Output("vol-scaler-value-store", "data"),
    Input("vol-scaler-input", "value"),
    prevent_initial_call=True,
)
def save_vol_scaler_value(value):
    """Save vol scaler value to local storage."""
    return value


@callback(
    Output("series-select-value-store", "data"),
    Input("series-select", "data"),
    prevent_initial_call=True,
)
def save_series_selection(value):
    """Save series selection to local storage."""
    return value or []


@callback(
    Output("active-tab-store", "data"),
    Input("main-tabs", "value"),
    prevent_initial_call=True,
)
def save_active_tab(value):
    """Save active tab to local storage."""
    return value or "statistics"





@callback(
    Output("rolling-window-store", "data"),
    Input("rolling-window-select", "value"),
    prevent_initial_call=True,
)
def save_rolling_window(value):
    """Save rolling window selection to local storage."""
    return value or "1y"


@callback(
    Output("rolling-metric-store", "data"),
    Input("rolling-metric-select", "value"),
    prevent_initial_call=True,
)
def save_rolling_metric(value):
    """Save rolling metric selection to local storage."""
    return value or "total_return"


@callback(
    Output("rolling-return-type-store", "data"),
    Input("rolling-return-type-select", "value"),
    prevent_initial_call=True,
)
def save_rolling_return_type(value):
    """Save rolling return type to local storage."""
    return value or "annualized"


@callback(
    Output("rolling-return-type-select", "disabled"),
    Output("rolling-return-type-select", "style"),
    Input("rolling-metric-select", "value"),
)
def update_rolling_controls_state(metric):
    """Enable/disable return type select based on metric."""
    if metric in ["total_return", "excess_return"]:
        return False, {}
    return True, {"opacity": 0.5, "pointerEvents": "none"}





@callback(
    Output("rolling-chart-switch-store", "data"),
    Input("rolling-chart-switch", "value"),
    prevent_initial_call=True,
)
def save_rolling_chart_switch(value):
    """Save rolling chart switch state to local storage."""
    return value if value is not None else "chart"





@callback(
    Output("rolling-grid-container", "style"),
    Output("rolling-chart-container", "style"),
    Input("rolling-chart-switch", "value"),
    prevent_initial_call=True,
)
def toggle_rolling_view(view_type):
    """Toggle between grid and chart view for rolling returns."""
    if view_type == "chart":
        return {"display": "none"}, {"display": "block"}
    else:
        return {"display": "block"}, {"display": "none"}


@callback(
    Output("drawdown-chart-switch-store", "data"),
    Input("drawdown-chart-switch", "value"),
    prevent_initial_call=True,
)
def save_drawdown_chart_switch(value):
    """Save drawdown chart switch state to local storage."""
    return value if value is not None else "chart"





@callback(
    Output("drawdown-grid-container", "style"),
    Output("drawdown-chart-container", "style"),
    Input("drawdown-chart-switch", "value"),
    prevent_initial_call=True,
)
def toggle_drawdown_view(view_type):
    """Toggle between grid and chart view for drawdown."""
    if view_type == "chart":
        return {"display": "none"}, {"display": "block"}
    else:
        return {"display": "block"}, {"display": "none"}


@callback(
    Output("growth-chart-switch-store", "data"),
    Input("growth-chart-switch", "value"),
    prevent_initial_call=True,
)
def save_growth_chart_switch(value):
    """Save growth chart switch state to local storage."""
    return value if value is not None else "chart"





@callback(
    Output("growth-grid-container", "style"),
    Output("growth-chart-container", "style"),
    Input("growth-chart-switch", "value"),
    prevent_initial_call=True,
)
def toggle_growth_view(view_type):
    """Toggle between grid and chart view for growth of $1."""
    if view_type == "chart":
        return {"display": "none"}, {"display": "block"}
    else:
        return {"display": "block"}, {"display": "none"}


@callback(
    Output("monthly-view-store", "data"),
    Input("monthly-view-checkbox", "value"),
    prevent_initial_call=True,
)
def save_monthly_view(value):
    """Save monthly view selection to local storage."""
    return value if value is not None else "annual"


@callback(
    Output("monthly-series-store", "data"),
    Input("monthly-series-select", "value"),
    prevent_initial_call=True,
)
def save_monthly_series(value):
    """Save monthly series selection to local storage."""
    return value





@callback(
    Output("raw-data-store", "data"),
    Output("original-periodicity-store", "data"),
    Output("periodicity-select", "data"),
    Output("periodicity-select", "value"),
    Output("periodicity-select", "disabled"),
    Output("temp-series-select", "data", allow_duplicate=True),
    Output("alert-message", "children"),
    Output("alert-message", "color"),
    Output("alert-message", "hide"),
    Output("periodicity-value-store", "data", allow_duplicate=True),
    Output("series-selection-modal", "opened", allow_duplicate=True),
    Output("temp-benchmark-assignments-store", "data", allow_duplicate=True),
    Output("temp-long-short-store", "data", allow_duplicate=True),
    Output("temp-series-order-store", "data", allow_duplicate=True),
    Output("first-load-store", "data"),
    Output("temp-deleted-series-store", "data", allow_duplicate=True),
    Output("temp-vol-scaling-assignments-store", "data", allow_duplicate=True),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    State("raw-data-store", "data"),
    State("original-periodicity-store", "data"),
    State("series-select", "data"),
    State("benchmark-assignments-store", "data"),
    State("long-short-store", "data"),
    State("series-order-store", "data"),
    State("first-load-store", "data"),
    State("vol-scaling-assignments-store", "data"),
    prevent_initial_call=True,
)
def handle_upload(contents, filename, existing_data, existing_periodicity, current_selection, current_bench, current_ls, current_order, first_load, current_vol_scaling):
    """Handle file upload, parse data, and update stores."""
    if contents is None:
        raise PreventUpdate

    try:
        # Parse the uploaded file
        new_df = parse_uploaded_file(contents, filename)
        new_periodicity = detect_periodicity(new_df)

        # Determine the effective periodicity
        if existing_data is not None:
            existing_df = json_to_df(existing_data)

            # Check periodicity compatibility
            if existing_periodicity == "monthly" and new_periodicity == "daily":
                return (
                    no_update, no_update, no_update, no_update, no_update,
                    no_update,
                    "Cannot append daily data to monthly data. Monthly data cannot be upsampled.",
                    "red",
                    False,
                    no_update, no_update, no_update, no_update, no_update,
                    no_update, no_update, no_update,
                )

            # If new data is monthly but existing is daily, convert existing to monthly
            if new_periodicity == "monthly" and existing_periodicity == "daily":
                existing_df = resample_returns(existing_df, "monthly")
                combined_periodicity = "monthly"
            else:
                combined_periodicity = existing_periodicity

            # Merge the data
            merged_df = merge_returns(existing_df, new_df)
        else:
            merged_df = new_df
            combined_periodicity = new_periodicity

        # Get available periodicities
        periodicity_options = get_available_periodicities(combined_periodicity)
        default_periodicity = "monthly"

        # Keep current selection and add new series
        new_series = [col for col in new_df.columns if col not in (current_selection or [])]
        updated_selection = (current_selection or []) + new_series

        # Determine alert state
        if not first_load:
            alert_msg = f"Loaded {len(new_df.columns)} series with {len(new_df)} rows from {filename}"
            alert_color = "green"
            alert_hide = False
            new_first_load = True
        else:
            alert_msg = no_update
            alert_color = no_update
            alert_hide = True
            new_first_load = True

        return (
            df_to_json(merged_df),
            combined_periodicity,
            periodicity_options,
            default_periodicity,
            False,
            updated_selection,
            alert_msg,
            alert_color,
            alert_hide,
            default_periodicity,
            True, # Open modal
            current_bench or {},
            current_ls or {},
            current_order or [],
            new_first_load,
            [], # Reset deleted series
            current_vol_scaling or {},
        )

    except Exception as e:
        return (
            no_update, no_update, no_update, no_update, no_update,
            no_update,
            f"Error loading file: {str(e)}",
            "red",
            False,
            no_update, no_update, no_update, no_update, no_update,
            no_update, no_update, no_update,
        )


@callback(
    Output("series-selection-container", "children"),
    Output("temp-series-order-store", "data", allow_duplicate=True),
    Input("raw-data-store", "data"),
    Input("temp-series-select", "data"),
    Input("temp-series-order-store", "data"),
    Input("series-edit-mode", "data"),
    Input("temp-deleted-series-store", "data"),
    State("temp-benchmark-assignments-store", "data"),
    State("temp-long-short-store", "data"),
    State("temp-vol-scaling-assignments-store", "data"),
    prevent_initial_call="initial_duplicate",
)
def update_series_selectors(raw_data, selected_series, series_order, edit_mode_series, deleted_series, current_assignments, long_short_assignments, vol_scaling_assignments):
    """Create series selection rows with checkbox, benchmark dropdown, long-short, reorder buttons, and delete button."""
    if raw_data is None:
        return [], []

    df = json_to_df(raw_data)
    
    # Filter out deleted series
    deleted_set = set(deleted_series or [])
    all_series = [s for s in list(df.columns) if s not in deleted_set]

    if not all_series:
        return [], []

    # Initialize or update series order
    if not series_order:
        series_order = all_series
    else:
        # Add any new series to the end
        for series in all_series:
            if series not in series_order:
                series_order.append(series)
        # Remove any deleted/filtered series
        series_order = [s for s in series_order if s in all_series]

    default_benchmark = "None"
    selected_series = selected_series or []

    # Create benchmark options with "None" as first option
    benchmark_options = [{"value": "None", "label": "None"}] + [{"value": s, "label": s} for s in all_series]

    # Calculate dynamic width based on longest series name
    # Use approximately 8 pixels per character in monospace font, with minimum of 150px
    max_series_length = max(len(s) for s in all_series) if all_series else 10
    series_width = max(150, max_series_length * 8 + 20)  # Add padding
    benchmark_width = int(series_width * 1.3)  # Make benchmark wider to prevent cutoff

    # Create column headers
    header_row = dmc.Group(
        mb="xs",
        gap="xs",
        children=[
            # Spacer for up/down arrows
            dmc.Box(w=20),
            # Spacer for checkbox
            dmc.Box(w=20),
            # Series label
            dmc.Text("Series", size="xs", fw=700, w=series_width, c="dimmed"),
            # Benchmark label
            dmc.Text("Benchmark", size="xs", fw=700, w=benchmark_width, c="dimmed"),
            # L/S label
            dmc.Text("L/S", size="xs", fw=700, w=50, c="dimmed"),
            # Scale Vol label
            dmc.Text("Scale Vol", size="xs", fw=700, w=60, c="dimmed"),
            # Spacer for delete button
            dmc.Box(w=30),
        ],
    )

    # Create a row for each series in the order specified
    series_rows = [header_row]
    for idx, series in enumerate(series_order):
        # Pre-compute ALL conditional values to avoid serialization issues
        if current_assignments:
            current_benchmark = current_assignments.get(series, default_benchmark)
        else:
            current_benchmark = default_benchmark

        if long_short_assignments:
            is_long_short = long_short_assignments.get(series, False)
        else:
            is_long_short = False

        if vol_scaling_assignments:
            is_scale_vol = vol_scaling_assignments.get(series, True) # Default True
        else:
            is_scale_vol = True

        is_selected = series in selected_series

        # Pre-compute benchmark value
        if current_benchmark in all_series or current_benchmark == "None":
            benchmark_value = current_benchmark
        else:
            benchmark_value = default_benchmark

        # Pre-compute disabled states for move buttons
        up_disabled = (idx == 0)
        down_disabled = (idx == len(series_order) - 1)

        # Pre-compute children for series name display based on edit mode
        is_editing = (series == edit_mode_series)

        if is_editing:
            series_name_children = [
                dmc.TextInput(
                    value=series,
                    id={"type": "edit-series-input", "series": series},
                    size="xs",
                    style={"flex": 1},
                ),
                dmc.ActionIcon(
                    DashIconify(icon="tabler:check", width=14),
                    id={"type": "save-edit-button", "series": series},
                    variant="subtle",
                    color="green",
                    size="xs",
                ),
                dmc.ActionIcon(
                    DashIconify(icon="tabler:x", width=14),
                    id={"type": "cancel-edit-button", "series": series},
                    variant="subtle",
                    color="red",
                    size="xs",
                ),
            ]
        else:
            series_name_children = [
                dmc.Text(
                    series,
                    size="sm",
                    style={"fontFamily": "monospace", "flex": 1},
                ),
                dmc.ActionIcon(
                    DashIconify(icon="tabler:pencil", width=14),
                    id={"type": "edit-series-button", "series": series},
                    variant="subtle",
                    color="gray",
                    size="xs",
                ),
            ]

        series_rows.append(
            dmc.Group(
                mb="xs",
                gap="xs",
                children=[
                    # Up/Down arrows for reordering
                    dmc.Stack(
                        gap=0,
                        children=[
                            dmc.ActionIcon(
                                "▲",
                                id={"type": "move-up-button", "series": series},
                                variant="subtle",
                                color="gray",
                                size="xs",
                                disabled=up_disabled,
                                style={"fontSize": "8px", "height": "12px", "minHeight": "12px"},
                            ),
                            dmc.ActionIcon(
                                "▼",
                                id={"type": "move-down-button", "series": series},
                                variant="subtle",
                                color="gray",
                                size="xs",
                                disabled=down_disabled,
                                style={"fontSize": "8px", "height": "12px", "minHeight": "12px"},
                            ),
                        ],
                    ),
                    # Checkbox to include series in analysis
                    dmc.Checkbox(
                        id={"type": "series-include-checkbox", "series": series},
                        checked=is_selected,
                        size="xs",
                    ),
                    # Series name with edit button OR edit textbox with check/X
                    dmc.Group(
                        gap=4,
                        w=series_width,
                        wrap="nowrap",
                        children=series_name_children,
                    ),
                    # Benchmark dropdown
                    dmc.Select(
                        id={"type": "benchmark-select", "series": series},
                        data=benchmark_options,
                        value=benchmark_value,
                        size="xs",
                        w=benchmark_width,
                        placeholder="Benchmark",
                    ),
                    # Long-Short switch
                    dmc.Switch(
                        id={"type": "long-short-checkbox", "series": series},
                        checked=is_long_short,
                        size="xs",
                        w=50,
                    ),
                    # Scale Vol switch
                    dmc.Switch(
                        id={"type": "scale-vol-checkbox", "series": series},
                        checked=is_scale_vol,
                        size="xs",
                        w=50,
                    ),
                    # Trash button to delete series
                    dmc.ActionIcon(
                        DashIconify(icon="tabler:trash-x", color="red", width=20),
                        id={"type": "delete-series-button", "series": series},
                        variant="subtle",
                        color="red",
                        size="sm",
                    ),
                ],
            )
        )

    return series_rows, series_order





@callback(
    Output("temp-deleted-series-store", "data", allow_duplicate=True),
    Output("temp-series-select", "data", allow_duplicate=True),
    Input({"type": "delete-series-button", "series": ALL}, "n_clicks"),
    State("temp-deleted-series-store", "data"),
    State({"type": "series-include-checkbox", "series": ALL}, "checked"),
    State({"type": "series-include-checkbox", "series": ALL}, "id"),
    prevent_initial_call=True,
)
def delete_series(n_clicks_list, deleted_series, checkbox_values, checkbox_ids):
    """Delete a series by adding it to the temporary deleted list."""
    if not n_clicks_list or all(n is None for n in n_clicks_list):
        raise PreventUpdate

    # Find which button was clicked
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    triggered_id = ctx.triggered[0]["prop_id"]
    # Parse the pattern-matching ID to get series name
    import json
    try:
        id_dict = json.loads(triggered_id.rsplit(".", 1)[0])
        series_to_delete = id_dict.get("series")
    except (json.JSONDecodeError, KeyError):
        raise PreventUpdate

    if not series_to_delete:
        raise PreventUpdate

    # Add to deleted list
    new_deleted = (deleted_series or []) + [series_to_delete]

    # Reconstruct selected series from checkbox states
    selected_series = []
    if checkbox_values and checkbox_ids:
        for i, checkbox_id in enumerate(checkbox_ids):
             if i < len(checkbox_values) and checkbox_values[i]:
                 selected_series.append(checkbox_id["series"])

    # Update selected series to remove deleted one (just for UI consistency)
    new_selected = [s for s in selected_series if s != series_to_delete]

    return new_deleted, new_selected


@callback(
    Output("series-edit-mode", "data"),
    Output("edit-box-focus-trigger", "data", allow_duplicate=True),
    Output("temp-series-select", "data", allow_duplicate=True),
    Input({"type": "edit-series-button", "series": ALL}, "n_clicks"),
    State({"type": "edit-series-button", "series": ALL}, "id"),
    State({"type": "series-include-checkbox", "series": ALL}, "checked"),
    State({"type": "series-include-checkbox", "series": ALL}, "id"),
    prevent_initial_call=True,
)
def enter_edit_mode(n_clicks_list, button_ids, checkbox_values, checkbox_ids):
    """Enter edit mode when pencil button is clicked."""
    if not n_clicks_list or all(n is None for n in n_clicks_list):
        raise PreventUpdate

    # Find which button was clicked
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    triggered_id = ctx.triggered[0]["prop_id"]
    import json
    try:
        id_dict = json.loads(triggered_id.rsplit(".", 1)[0])
        series_name = id_dict.get("series")
    except (json.JSONDecodeError, KeyError):
        raise PreventUpdate

    if not series_name:
        raise PreventUpdate

    # Reconstruct selected series from checkbox states
    current_selected = []
    if checkbox_values and checkbox_ids:
        for i, checkbox_id in enumerate(checkbox_ids):
             if i < len(checkbox_values) and checkbox_values[i]:
                 current_selected.append(checkbox_id["series"])

    return series_name, series_name, current_selected


@callback(
    Output("series-edit-mode", "data", allow_duplicate=True),
    Output("edit-box-focus-trigger", "data", allow_duplicate=True),
    Input({"type": "cancel-edit-button", "series": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def cancel_edit_mode(n_clicks_list):
    """Exit edit mode when X button is clicked."""
    if not n_clicks_list or all(n is None for n in n_clicks_list):
        raise PreventUpdate

    # Exit edit mode
    return None, None


@callback(
    Output("raw-data-store", "data", allow_duplicate=True),
    Output("temp-benchmark-assignments-store", "data", allow_duplicate=True),
    Output("temp-long-short-store", "data", allow_duplicate=True),
    Output("temp-vol-scaling-assignments-store", "data", allow_duplicate=True),
    Output("temp-series-select", "data", allow_duplicate=True),
    Output("temp-series-order-store", "data", allow_duplicate=True),
    Output("series-edit-mode", "data", allow_duplicate=True),
    Output("series-select-value-store", "data", allow_duplicate=True),
    Output("edit-box-focus-trigger", "data", allow_duplicate=True),
    Input({"type": "save-edit-button", "series": ALL}, "n_clicks"),
    State({"type": "save-edit-button", "series": ALL}, "id"),
    State({"type": "edit-series-input", "series": ALL}, "value"),
    State({"type": "edit-series-input", "series": ALL}, "id"),
    State("raw-data-store", "data"),
    State("temp-benchmark-assignments-store", "data"),
    State("temp-long-short-store", "data"),
    State("temp-vol-scaling-assignments-store", "data"),
    State({"type": "series-include-checkbox", "series": ALL}, "checked"),
    State({"type": "series-include-checkbox", "series": ALL}, "id"),
    State("temp-series-order-store", "data"),
    State("series-edit-mode", "data"),
    prevent_initial_call=True,
)
def save_edit(save_clicks_list, save_ids, input_values, input_ids, raw_data, benchmark_assignments, long_short_assignments, vol_scaling_assignments, checkbox_values, checkbox_ids, series_order, edit_mode_series):
    """Save the series rename when check button is clicked."""
    if not save_clicks_list or all(n is None for n in save_clicks_list):
        raise PreventUpdate

    # Find which button was clicked
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    triggered_id = ctx.triggered[0]["prop_id"]
    import json
    try:
        id_dict = json.loads(triggered_id.rsplit(".", 1)[0])
        old_name = id_dict.get("series")
    except (json.JSONDecodeError, KeyError):
        raise PreventUpdate

    if not old_name or not edit_mode_series or old_name != edit_mode_series:
        raise PreventUpdate

    # Find the new name from the input
    new_name = None
    for i, input_id in enumerate(input_ids):
        if input_id["series"] == old_name and i < len(input_values):
            new_name = input_values[i]
            break

    if not new_name:
        raise PreventUpdate

    new_name = new_name.strip()

    # If name unchanged, just exit edit mode
    if new_name == old_name or not new_name:
        return no_update, no_update, no_update, no_update, no_update, no_update, None, no_update, None

    # Check if new name already exists
    df = json_to_df(raw_data)
    if new_name in df.columns:
        # Don't allow duplicate names, exit edit mode
        return no_update, no_update, no_update, no_update, no_update, no_update, None, no_update, None

    # Rename column in DataFrame
    df = df.rename(columns={old_name: new_name})
    new_raw_data = df_to_json(df)

    # Update benchmark assignments
    new_benchmark_assignments = {}
    for series, benchmark in benchmark_assignments.items():
        series_key = new_name if series == old_name else series
        benchmark_value = new_name if benchmark == old_name else benchmark
        new_benchmark_assignments[series_key] = benchmark_value

    # Update long-short assignments
    new_long_short_assignments = {}
    for series, is_long_short in long_short_assignments.items():
        series_key = new_name if series == old_name else series
        new_long_short_assignments[series_key] = is_long_short

    # Update vol scaling assignments
    new_vol_scaling_assignments = {}
    if vol_scaling_assignments:
        for series, is_scaled in vol_scaling_assignments.items():
            series_key = new_name if series == old_name else series
            new_vol_scaling_assignments[series_key] = is_scaled

    # Reconstruct selected series from checkbox states
    current_selected = []
    if checkbox_values and checkbox_ids:
        for i, checkbox_id in enumerate(checkbox_ids):
             if i < len(checkbox_values) and checkbox_values[i]:
                 current_selected.append(checkbox_id["series"])

    # Update series selection (handling the rename)
    new_series_select = [new_name if s == old_name else s for s in current_selected]

    # Update series order
    new_series_order = [new_name if s == old_name else s for s in series_order]

    # Return updated data and exit edit mode
    return new_raw_data, new_benchmark_assignments, new_long_short_assignments, new_vol_scaling_assignments, new_series_select, new_series_order, None, new_series_select, None


@callback(
    Output("temp-benchmark-assignments-store", "data"),
    Input({"type": "benchmark-select", "series": ALL}, "value"),
    State({"type": "benchmark-select", "series": ALL}, "id"),
    State("raw-data-store", "data"),
    prevent_initial_call=True,
)
def update_benchmark_assignments(benchmark_values, benchmark_ids, raw_data):
    """Store benchmark assignments for all series."""
    if raw_data is None or not benchmark_values or not benchmark_ids:
        return {}

    # Map values to series using the pattern-matching IDs
    assignments = {}
    for i, benchmark_id in enumerate(benchmark_ids):
        series = benchmark_id["series"]
        if i < len(benchmark_values) and benchmark_values[i]:
            assignments[series] = benchmark_values[i]

    return assignments


@callback(
    Output("temp-long-short-store", "data"),
    Input({"type": "long-short-checkbox", "series": ALL}, "checked"),
    State({"type": "long-short-checkbox", "series": ALL}, "id"),
    State("raw-data-store", "data"),
    prevent_initial_call=True,
)
def update_long_short_assignments(checkbox_values, checkbox_ids, raw_data):
    """Store long-short checkbox assignments for all series."""
    if raw_data is None or checkbox_values is None or not checkbox_ids:
        return {}

    # Map values to series using the pattern-matching IDs
    assignments = {}
    for i, checkbox_id in enumerate(checkbox_ids):
        series = checkbox_id["series"]
        if i < len(checkbox_values):
            assignments[series] = checkbox_values[i] or False

    return assignments


@callback(
    Output("temp-vol-scaling-assignments-store", "data"),
    Input({"type": "scale-vol-checkbox", "series": ALL}, "checked"),
    State({"type": "scale-vol-checkbox", "series": ALL}, "id"),
    State("raw-data-store", "data"),
    prevent_initial_call=True,
)
def update_vol_scaling_assignments(checkbox_values, checkbox_ids, raw_data):
    """Store vol-scaling checkbox assignments for all series."""
    if raw_data is None or checkbox_values is None or not checkbox_ids:
        return {}

    # Map values to series using the pattern-matching IDs
    assignments = {}
    for i, checkbox_id in enumerate(checkbox_ids):
        series = checkbox_id["series"]
        if i < len(checkbox_values):
            assignments[series] = checkbox_values[i]

    return assignments


@callback(
    Output("start-date-picker", "value"),
    Output("end-date-picker", "value"),
    Output("date-picker-wrapper", "style"),
    Output("common-range-button", "disabled"),
    Output("maximum-range-button", "disabled"),
    Output("date-range-store", "data", allow_duplicate=True),
    Input("raw-data-store", "data"),
    Input("periodicity-select", "value"),
    Input("series-select", "data"),
    prevent_initial_call="initial_duplicate",
)
def initialize_date_range(raw_data, periodicity, selected_series):
    """Initialize date range to maximum range when data is loaded."""
    disabled_style = {"display": "flex", "opacity": 0.5, "pointerEvents": "none", "alignItems": "flex-start"}
    enabled_style = {"display": "flex", "alignItems": "flex-start"}

    if raw_data is None or not selected_series:
        return None, None, disabled_style, True, True, None

    try:
        df = resample_returns_cached(raw_data, periodicity or "daily")

        # Filter to selected series
        available_series = [s for s in selected_series if s in df.columns]
        if not available_series:
            return None, None, disabled_style, True, True, None

        # Get maximum range (earliest start, latest end)
        start_date = df.index.min().strftime("%Y-%m-%d")
        end_date = df.index.max().strftime("%Y-%m-%d")

        date_range = {"start": start_date, "end": end_date}

        return start_date, end_date, enabled_style, False, False, date_range

    except Exception:
        return None, None, disabled_style, True, True, None


@callback(
    Output("start-date-picker", "value", allow_duplicate=True),
    Output("end-date-picker", "value", allow_duplicate=True),
    Output("date-range-store", "data"),
    Input("common-range-button", "n_clicks"),
    Input("maximum-range-button", "n_clicks"),
    State("raw-data-store", "data"),
    State("periodicity-select", "value"),
    State("series-select", "data"),
    prevent_initial_call=True,
)
def update_date_range_buttons(common_clicks, max_clicks, raw_data, periodicity, selected_series):
    """Update date range based on button clicks."""
    if raw_data is None or not selected_series:
        raise PreventUpdate

    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    try:
        df = resample_returns_cached(raw_data, periodicity or "daily")

        # Filter to selected series
        available_series = [s for s in selected_series if s in df.columns]
        if not available_series:
            raise PreventUpdate

        if button_id == "common-range-button":
            # Common range: only dates where ALL selected series have data
            subset_df = df[available_series].dropna()
            if len(subset_df) == 0:
                raise PreventUpdate
            start_date = subset_df.index.min().strftime("%Y-%m-%d")
            end_date = subset_df.index.max().strftime("%Y-%m-%d")
        else:  # maximum-range-button
            # Maximum range: earliest start to latest end across all selected series
            start_date = df.index.min().strftime("%Y-%m-%d")
            end_date = df.index.max().strftime("%Y-%m-%d")

        date_range = {"start": start_date, "end": end_date}
        return start_date, end_date, date_range

    except Exception:
        raise PreventUpdate


@callback(
    Output("date-range-store", "data", allow_duplicate=True),
    Input("start-date-picker", "value"),
    Input("end-date-picker", "value"),
    prevent_initial_call=True,
)
def update_date_range_store(start_date, end_date):
    """Store date range when user manually changes dates."""
    if start_date and end_date:
        return {"start": start_date, "end": end_date}
    return no_update


@callback(
    Output("returns-grid", "columnDefs"),
    Output("returns-grid", "rowData"),
    Output("menu-download-excel", "disabled"),
    Input("raw-data-store", "data"),
    Input("periodicity-select", "value"),
    Input("series-select", "data"),
    Input("returns-type-select", "value"),
    Input("benchmark-assignments-store", "data"),
    Input("long-short-store", "data"),
    Input("date-range-store", "data"),
    Input("vol-scaler-value-store", "data"),
    Input("vol-scaling-assignments-store", "data"),
    prevent_initial_call=True,
)
def update_grid(raw_data, periodicity, selected_series, returns_type, benchmark_assignments, long_short_assignments, date_range, vol_scaler, vol_scaling_assignments):
    """Update the AG Grid based on selections (optimized with caching)."""
    if raw_data is None or not selected_series:
        return [], [], True

    try:
        # Use cached function to avoid repeated deserialization and computation
        display_df = calculate_excess_returns(
            raw_data,
            periodicity or "daily",
            tuple(selected_series),  # Convert to tuple for cache key
            str(benchmark_assignments),  # Convert to string for cache key
            returns_type,
            str(long_short_assignments),  # Convert to string for cache key
            str(date_range),  # Convert to string for cache key
            vol_scaler or 0,
            str(vol_scaling_assignments)
        )

        if display_df.empty:
            return [], [], True

        # Create column definitions
        column_defs = [
            {
                "field": "Date",
                "pinned": "left",
                "valueFormatter": {"function": "d3.timeFormat('%Y-%m-%d')(new Date(params.value))"},
                "width": 120,
            }
        ]

        for col in display_df.columns:
            column_defs.append({
                "field": col,
                "valueFormatter": {"function": "params.value != null ? d3.format('.2%')(params.value) : ''"},
                "width": 120,
            })

        # Convert to row data
        df_reset = display_df.reset_index()
        df_reset["Date"] = df_reset["Date"].dt.strftime("%Y-%m-%d")
        row_data = df_reset.to_dict("records")

        return column_defs, row_data, False

    except Exception:
        return [], [], True


@callback(
    Output("rolling-grid", "columnDefs"),
    Output("rolling-grid", "rowData"),
    Input("main-tabs", "value"),
    Input("raw-data-store", "data"),
    Input("periodicity-select", "value"),
    Input("series-select", "data"),
    Input("rolling-window-select", "value"),
    Input("rolling-return-type-select", "value"),
    Input("rolling-metric-select", "value"),
    Input("benchmark-assignments-store", "data"),
    Input("long-short-store", "data"),
    Input("date-range-store", "data"),
    Input("vol-scaler-value-store", "data"),
    Input("vol-scaling-assignments-store", "data"),
    prevent_initial_call=True,
)
def update_rolling_grid(active_tab, raw_data, periodicity, selected_series, rolling_window, rolling_return_type, rolling_metric, benchmark_assignments, long_short_assignments, date_range, vol_scaler, vol_scaling_assignments):
    """Update the Rolling Returns grid with rolling window calculations."""
    # Lazy loading: only calculate when rolling tab is active
    if active_tab != "rolling":
        raise PreventUpdate

    if raw_data is None or not selected_series:
        return [], []

    try:
        # Use shared calculate_rolling_returns function
        # We pass "total" for returns_type as it's ignored by the new logic in favor of rolling_metric
        rolling_df = calculate_rolling_returns(
            raw_data,
            periodicity,
            tuple(selected_series),
            "total",
            str(benchmark_assignments),
            str(long_short_assignments),
            str(date_range),
            rolling_window,
            rolling_return_type,
            rolling_metric or "total_return",
            vol_scaler or 0,
            str(vol_scaling_assignments)
        )

        if rolling_df.empty:
            return [], []

        # Determine formatter based on metric
        metric = rolling_metric or "total_return"
        if metric in ["total_return", "excess_return", "volatility", "tracking_error"]:
            formatter = ".2%"
        else:
            formatter = ".2f"

        # Create column definitions
        column_defs = [
            {
                "field": "Date",
                "pinned": "left",
                "valueFormatter": {"function": "d3.timeFormat('%Y-%m-%d')(new Date(params.value))"},
                "width": 120,
            }
        ]

        for col in rolling_df.columns:
            column_defs.append({
                "field": col,
                "valueFormatter": {"function": f"params.value != null ? d3.format('{formatter}')(params.value) : ''"},
                "width": 120,
            })

        # Convert to row data
        df_reset = rolling_df.reset_index()
        df_reset["Date"] = df_reset["Date"].dt.strftime("%Y-%m-%d")
        row_data = df_reset.to_dict("records")

        return column_defs, row_data

    except Exception:
        return [], []


@callback(
    Output("rolling-chart-wrapper", "children"),
    Input("main-tabs", "value"),
    Input("raw-data-store", "data"),
    Input("periodicity-select", "value"),
    Input("series-select", "data"),
    Input("rolling-window-select", "value"),
    Input("rolling-return-type-select", "value"),
    Input("rolling-metric-select", "value"),
    Input("benchmark-assignments-store", "data"),
    Input("long-short-store", "data"),
    Input("date-range-store", "data"),
    Input("vol-scaler-value-store", "data"),
    Input("vol-scaling-assignments-store", "data"),
    prevent_initial_call=True,
)
def update_rolling_chart(active_tab, raw_data, periodicity, selected_series, rolling_window, rolling_return_type, rolling_metric, benchmark_assignments, long_short_assignments, date_range, vol_scaler, vol_scaling_assignments):
    """Update the Rolling Returns chart with rolling window calculations."""
    # Create empty figure
    empty_fig = go.Figure()
    empty_fig.update_layout(
        title="",
        xaxis_title="",
        yaxis_title="",
        template="plotly_white",
    )
    empty_graph = dcc.Graph(figure=empty_fig, style={"height": "550px"})

    # Lazy loading: only calculate when rolling tab is active
    if active_tab != "rolling":
        raise PreventUpdate

    if raw_data is None or not selected_series:
        return empty_graph

    try:
        # Use shared calculate_rolling_returns function
        rolling_df = calculate_rolling_returns(
            raw_data,
            periodicity,
            tuple(selected_series),
            "total",
            str(benchmark_assignments),
            str(long_short_assignments),
            str(date_range),
            rolling_window,
            rolling_return_type,
            rolling_metric or "total_return",
            vol_scaler or 0,
            str(vol_scaling_assignments)
        )

        if rolling_df.empty:
            return empty_graph

        # Determine formatting
        metric = rolling_metric or "total_return"
        if metric in ["total_return", "excess_return", "volatility", "tracking_error"]:
            y_format = ".2%"
        else:
            y_format = ".2f"

        # Create the line chart
        fig = go.Figure()

        for col in rolling_df.columns:
            fig.add_trace(go.Scatter(
                x=rolling_df.index,
                y=rolling_df[col],
                mode='lines',
                name=col,
                hovertemplate=f'%{{y:{y_format}}}<extra></extra>',
            ))

        # Update layout
        window_label_map = {
            "3m": "3-Month",
            "6m": "6-Month",
            "1y": "1-Year",
            "3y": "3-Year",
            "5y": "5-Year",
            "10y": "10-Year",
        }
        window_label = window_label_map.get(rolling_window, "1-Year")
        
        metric_label_map = {
            "total_return": "Total Return",
            "volatility": "Volatility",
            "sharpe_ratio": "Sharpe Ratio",
            "sortino_ratio": "Sortino Ratio",
            "excess_return": "Excess Return",
            "tracking_error": "Tracking Error",
            "information_ratio": "Information Ratio",
            "correlation": "Correlation",
        }
        metric_label = metric_label_map.get(metric, "Total Return")
        
        return_type_label = "Annualized" if rolling_return_type == "annualized" else "Cumulative"
        
        if metric in ["total_return", "excess_return"]:
            title = f"Rolling {window_label} {return_type_label} {metric_label}"
        elif metric in ["volatility", "tracking_error"]:
            title = f"Rolling {window_label} Annualized {metric_label}"
        else:
            title = f"Rolling {window_label} {metric_label}"

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title=metric_label,
            yaxis_tickformat=y_format,
            template="plotly_white",
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        return dcc.Graph(figure=fig, style={"height": "550px"})

    except Exception:
        return empty_graph





@callback(
    Output("monthly-series-select", "disabled"),
    Output("monthly-series-select", "data"),
    Output("monthly-series-select", "value", allow_duplicate=True),
    Input("monthly-view-checkbox", "value"),
    Input("series-select", "data"),
    State("monthly-series-store", "data"),
    State("monthly-series-select", "value"),
    prevent_initial_call=True,
)
def update_monthly_series_select(monthly_view, selected_series, stored_monthly_series, current_value):
    """Enable/disable monthly series select and populate with available series."""
    # Check which input triggered the callback
    ctx = callback_context
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

    if not selected_series:
        return True, [], None

    # Create dropdown options from selected series
    options = [{"value": s, "label": s} for s in selected_series]

    # Disable when in annual view
    if monthly_view != "monthly":
        return True, options, no_update

    # Enable when in monthly view
    # Only update value when switching TO monthly view
    if triggered_id == "monthly-view-checkbox":
        # Use stored value when switching to monthly view
        if stored_monthly_series and stored_monthly_series in selected_series:
            default_value = stored_monthly_series
        else:
            default_value = selected_series[0] if selected_series else None
        return False, options, default_value

    # For series list changes while already in monthly view, preserve current value
    else:
        # Check if current value is still valid, otherwise use stored or first
        if current_value and current_value in selected_series:
            return False, options, no_update
        elif stored_monthly_series and stored_monthly_series in selected_series:
            return False, options, stored_monthly_series
        else:
            return False, options, selected_series[0] if selected_series else None


@callback(
    Output("calendar-grid", "columnDefs"),
    Output("calendar-grid", "rowData"),
    Input("main-tabs", "value"),
    Input("raw-data-store", "data"),
    Input("original-periodicity-store", "data"),
    Input("periodicity-select", "value"),
    Input("series-select", "data"),
    Input("returns-type-select", "value"),
    Input("benchmark-assignments-store", "data"),
    Input("long-short-store", "data"),
    Input("date-range-store", "data"),
    Input("monthly-view-checkbox", "value"),
    Input("monthly-series-select", "value"),
    Input("vol-scaler-value-store", "data"),
    Input("vol-scaling-assignments-store", "data"),
    prevent_initial_call=True,
)
def update_calendar_grid(active_tab, raw_data, original_periodicity, selected_periodicity, selected_series, returns_type, benchmark_assignments, long_short_assignments, date_range, monthly_view, monthly_series, vol_scaler, vol_scaling_assignments):
    """Update the Calendar Year Returns grid (lazy loaded)."""
    # Lazy loading: only calculate when calendar tab is active
    if active_tab != "calendar":
        raise PreventUpdate

    if raw_data is None or not selected_series:
        return [], []

    # Only calculate for daily or monthly original data
    if original_periodicity not in ["daily", "monthly"]:
        # Weekly data - don't calculate calendar year returns
        return [], []

    try:
        if monthly_view == "monthly" and monthly_series and monthly_series in selected_series:
            # Handle monthly view if selected
            return create_monthly_view(
                raw_data,
                monthly_series,
                original_periodicity,
                selected_periodicity,
                returns_type,
                benchmark_assignments,
                long_short_assignments,
                selected_series,
                date_range,
                vol_scaler or 0,
                str(vol_scaling_assignments)
            )

        else:
            # Calculate calendar returns for the selected periodicity
            calendar_returns = calculate_calendar_year_returns(
                raw_data,
                original_periodicity,
                selected_periodicity,
                selected_series,
                returns_type,
                benchmark_assignments,
                long_short_assignments,
                date_range,
                vol_scaler or 0,
                str(vol_scaling_assignments)
            )

            if calendar_returns.empty:
                return [], []

            # Get all years that have data for at least one series
            all_years = calendar_returns.index.unique().sort_values().tolist()

            if not all_years:
                return [], []

            # Build row data first to calculate max absolute value
            row_data = []
            for year in all_years:
                row = {"Year": int(year)}
                for series in selected_series:
                    if series in calendar_returns and year in calendar_returns[series].index:
                        row[series] = calendar_returns[series].loc[year]
                    else:
                        row[series] = None
                row_data.append(row)

            # Create column definitions with conditional formatting
            column_defs = [
                {
                    "field": "Year",
                    "pinned": "left",
                    "width": 100,
                }
            ]

            for series in selected_series:
                if series in calendar_returns:
                    column_defs.append({
                        "field": series,
                        "valueFormatter": {"function": "params.value != null ? d3.format('.2%')(params.value) : ''"},
                        "width": 120,
                    })

            return column_defs, row_data

    except Exception:
        return [], []


@callback(
    Output("statistics-grid", "columnDefs"),
    Output("statistics-grid", "rowData"),
    Input("raw-data-store", "data"),
    Input("periodicity-select", "value"),
    Input("series-select", "data"),
    Input("benchmark-assignments-store", "data"),
    Input("long-short-store", "data"),
    Input("date-range-store", "data"),
    Input("vol-scaler-value-store", "data"),
    Input("vol-scaling-assignments-store", "data"),
    prevent_initial_call=True,
)
def update_statistics(raw_data, periodicity, selected_series, benchmark_assignments, long_short_assignments, date_range, vol_scaler, vol_scaling_assignments):
    """Update the Statistics grid with transposed data (optimized with caching)."""
    if raw_data is None or not selected_series:
        return [], []

    try:
        # Use cached function to avoid repeated computation
        stats = calculate_statistics_cached(
            raw_data,
            periodicity or "daily",
            tuple(selected_series),
            str(benchmark_assignments),
            str(long_short_assignments),
            str(date_range),
            vol_scaler or 0,
            str(vol_scaling_assignments)
        )

        if not stats:
            return [], []

        # Transpose: rows become statistics, columns become series
        # First column is "Statistic" (pinned), then one column per series
        column_defs = [
            {"field": "Statistic", "pinned": "left", "width": 200},
        ]           
        for series_stats in stats:
            series_name = series_stats["Series"]
            column_defs.append({
                "field": series_name,
                "width": 120,
                # Dynamic formatting based on row - use expression instead of statements
                "valueFormatter": {
                    "function": "(!params.data._format || params.value == null) ? params.value : d3.format(params.data._format)(params.value)"
                },
            })

        # Build transposed rows - keep raw values for JavaScript formatting
        row_data = []
        for stat_name, fmt in STATS_CONFIG:
            row = {"Statistic": stat_name, "_format": fmt}
            for series_stats in stats:
                series_name = series_stats["Series"]
                value = series_stats.get(stat_name)
                # Check if value is NaN and replace with empty string
                if value is None or (isinstance(value, float) and pd.isna(value)):
                    row[series_name] = None
                else:
                    # Keep raw numeric values for JavaScript formatting
                    row[series_name] = value

            row_data.append(row)
            
        return column_defs, row_data

    except Exception:
        return [], []





@callback(
    Output("correlogram-container", "children"),
    Input("main-tabs", "value"),  # Lazy loading: only update when tab is active
    Input("raw-data-store", "data"),
    Input("periodicity-select", "value"),
    Input("series-select", "data"),
    Input("returns-type-select", "value"),
    Input("benchmark-assignments-store", "data"),
    Input("long-short-store", "data"),
    Input("date-range-store", "data"),
    Input("vol-scaler-value-store", "data"),
    Input("vol-scaling-assignments-store", "data"),
    prevent_initial_call=True,
)
def update_correlogram(active_tab, raw_data, periodicity, selected_series, returns_type, benchmark_assignments, long_short_assignments, date_range, vol_scaler, vol_scaling_assignments):
    """Update the Correlogram with custom pairs plot (lazy loaded, size-limited, cached)."""
    # Define empty figure
    empty_fig = go.Figure()
    empty_fig.add_annotation(
        text="Select at least 2 series to view correlogram",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color="gray"),
    )
    empty_fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        template="plotly_white",
    )
    empty_graph = dcc.Graph(figure=empty_fig, style={"height": "700px"})

    # Lazy loading: only generate when correlogram tab is active
    if active_tab != "correlogram":
        raise PreventUpdate

    if raw_data is None or not selected_series or len(selected_series) < 2:
        return empty_graph

    # Size limit: show simple correlation matrix heatmap if too many series
    if len(selected_series) > MAX_SCATTER_MATRIX_SIZE:
        try:
            result = generate_correlogram_cached(
                raw_data,
                periodicity or "daily",
                tuple(selected_series),
                returns_type,
                str(benchmark_assignments),
                str(long_short_assignments),
                str(date_range),
                vol_scaler or 0,
                str(vol_scaling_assignments)
            )

            if result is None:
                return empty_graph

            corr_matrix = result['corr_matrix']
            available_series = result['available_series']

            # Create a simple heatmap for correlation matrix
            heatmap_fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=available_series,
                y=available_series,
                colorscale='RdBu_r',
                zmid=0,
                zmin=-1,
                zmax=1,
                text=corr_matrix.values.round(2),
                texttemplate='%{text}',
                textfont={"size": 10},
                hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>',
            ))

            height = max(500, 30 * len(available_series) + 150)
            heatmap_fig.update_layout(
                title=f"Correlation Matrix ({returns_type.title()} Returns)",
                height=height,
                xaxis=dict(tickangle=45),
                yaxis=dict(autorange='reversed'),
                template="plotly_white",
            )

            return dcc.Graph(figure=heatmap_fig, style={"height": f"{height}px"})

        except Exception:
            return empty_graph

    try:
        # Use cached function to avoid repeated computation
        result = generate_correlogram_cached(
            raw_data,
            periodicity or "daily",
            tuple(selected_series),
            returns_type,
            str(benchmark_assignments),
            str(long_short_assignments),
            str(date_range),
            vol_scaler or 0,
            str(vol_scaling_assignments)
        )

        if result is None:
            return empty_graph

        display_df = result['display_df']
        corr_matrix = result['corr_matrix']
        available_series = result['available_series']
        n = result['n']

        if n < 2:
            return empty_graph

        # Create subplots
        fig = make_subplots(
            rows=n, cols=n,
            horizontal_spacing=0.02,
            vertical_spacing=0.02,
        )

        # Populate the grid
        for i, row_series in enumerate(available_series):
            for j, col_series in enumerate(available_series):
                row_idx = i + 1
                col_idx = j + 1

                if i == j:
                    # Diagonal: density chart (histogram with KDE-like appearance)
                    fig.add_trace(
                        go.Histogram(
                            x=display_df[row_series].dropna(),
                            histnorm='probability density',
                            marker_color='#228be6',
                            opacity=0.7,
                            showlegend=False,
                            nbinsx=30,  # Limit bins for performance
                        ),
                        row=row_idx, col=col_idx
                    )
                elif i > j:
                    # Lower triangle: scatter plot with sampling for large datasets
                    series_data = display_df[[col_series, row_series]].dropna()
                    if len(series_data) > 1000:
                        # Sample for performance if > 1000 points
                        series_data = series_data.sample(n=1000, random_state=42)

                    fig.add_trace(
                        go.Scattergl(  # Use Scattergl for better performance
                            x=series_data[col_series],
                            y=series_data[row_series],
                            mode='markers',
                            marker=dict(size=3, opacity=0.5, color='#228be6'),
                            showlegend=False,
                        ),
                        row=row_idx, col=col_idx
                    )
                else:
                    # Upper triangle: correlation value
                    corr_val = corr_matrix.loc[row_series, col_series]
                    # Color based on correlation
                    if corr_val >= 0.7:
                        color = '#1971c2'
                    elif corr_val >= 0.3:
                        color = '#228be6'
                    elif corr_val <= -0.7:
                        color = '#c92a2a'
                    elif corr_val <= -0.3:
                        color = '#e03131'
                    else:
                        color = '#868e96'

                    fig.add_trace(
                        go.Scatter(
                            x=[0.5], y=[0.5],
                            mode='text',
                            text=[f'{corr_val:.2f}'],
                            textfont=dict(size=14, color=color),
                            showlegend=False,
                        ),
                        row=row_idx, col=col_idx
                    )
                    # Hide axes for correlation cells
                    fig.update_xaxes(
                        showticklabels=False, showgrid=False,
                        zeroline=False, range=[0, 1],
                        row=row_idx, col=col_idx
                    )
                    fig.update_yaxes(
                        showticklabels=False, showgrid=False,
                        zeroline=False, range=[0, 1],
                        row=row_idx, col=col_idx
                    )

        # Update axis labels
        for i, series in enumerate(available_series):
            # Bottom row x-axis labels
            fig.update_xaxes(title_text=series, row=n, col=i+1, title_font=dict(size=10))
            # Left column y-axis labels
            fig.update_yaxes(title_text=series, row=i+1, col=1, title_font=dict(size=10))

        # Update layout
        height = max(500, 150 * n)
        fig.update_layout(
            title=f"Scatter Matrix ({returns_type.title()} Returns)",
            height=height,
            margin=dict(l=60, r=30, t=50, b=60),
            showlegend=False,
            template="plotly_white",
        )

        # Hide tick labels for inner plots
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)

        return dcc.Graph(figure=fig, style={"height": f"{height}px"})

    except Exception:
        return empty_graph


@callback(
    Output("growth-charts-container", "children"),
    Input("main-tabs", "value"),
    Input("growth-chart-switch", "value"),
    Input("raw-data-store", "data"),
    Input("periodicity-select", "value"),
    Input("series-select", "data"),
    Input("benchmark-assignments-store", "data"),
    Input("long-short-store", "data"),
    Input("date-range-store", "data"),
    Input("vol-scaler-value-store", "data"),
    Input("vol-scaling-assignments-store", "data"),
    prevent_initial_call=True,
)
def update_growth_charts(active_tab, chart_checked, raw_data, periodicity, selected_series, benchmark_assignments, long_short_assignments, date_range, vol_scaler, vol_scaling_assignments):
    """Update Growth of $1 charts (lazy loaded)."""
    # Lazy loading: only generate when growth tab is active and chart view is selected
    if active_tab != "growth" or chart_checked != "chart":
        raise PreventUpdate

    if raw_data is None or not selected_series:
        return dmc.Text("Select series to view growth charts", size="sm", c="dimmed")

    try:
        # Use get_working_returns to get aligned data + benchmarks
        df = get_working_returns(
            raw_data, periodicity or "daily", tuple(selected_series),
            str(benchmark_assignments), str(long_short_assignments), str(date_range),
            vol_scaler or 0, str(vol_scaling_assignments)
        )

        if df.empty:
            return dmc.Text("No data available for selected series", size="sm", c="dimmed")

        benchmark_dict = eval(str(benchmark_assignments)) if benchmark_assignments else {}
        long_short_dict = eval(str(long_short_assignments)) if long_short_assignments else {}

        # Filter to selected series only
        available_series = [s for s in selected_series if s in df.columns]
        if not available_series:
            return dmc.Text("No data available for selected series", size="sm", c="dimmed")

        # Determine the period offset based on periodicity
        periodicity_str = periodicity or "daily"
        if periodicity_str == "daily":
            period_offset = pd.DateOffset(days=1)
        elif periodicity_str == "monthly":
            period_offset = pd.tseries.offsets.MonthEnd(1)
        elif periodicity_str.startswith("weekly"):
            period_offset = pd.DateOffset(weeks=1)
        else:
            period_offset = pd.DateOffset(days=1)

        # Use shared calculate_growth_of_dollar function for the main chart
        # (It calls get_working_returns internally, but it's cached)
        growth_df = calculate_growth_of_dollar(
            raw_data,
            periodicity,
            tuple(selected_series),
            str(benchmark_assignments),
            str(long_short_assignments),
            str(date_range),
            vol_scaler or 0,
            str(vol_scaling_assignments)
        )

        # Create main growth figure
        main_fig = go.Figure()
        if not growth_df.empty:
            for series in growth_df.columns:
                main_fig.add_trace(go.Scatter(
                    x=growth_df.index,
                    y=growth_df[series],
                    mode='lines',
                    name=series,
                    line=dict(width=2),
                ))

        main_fig.update_layout(
            title="Growth of $1 - All Series",
            xaxis_title="Date",
            yaxis_title="Growth of $1",
            height=500,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
        )

        # Create individual series vs benchmark charts
        individual_charts = []
        for series in available_series:
            benchmark = benchmark_dict.get(series, available_series[0])
            is_long_short = long_short_dict.get(series, False)

            # Skip if benchmark is None or same as series
            if benchmark == "None" or benchmark == series:
                continue

            if benchmark not in df.columns:
                continue

            # Calculate growth for series - aligned to valid data
            series_returns = df[series].dropna()
            if series_returns.empty:
                continue
            
            series_start = series_returns.index[0]
            series_growth = (1 + series_returns).cumprod()

            # Determine effective start for benchmark
            # If benchmark starts earlier, clip to series start.
            # If benchmark starts later, use benchmark start.
            benchmark_full = df[benchmark].dropna()
            if benchmark_full.empty:
                continue
                
            benchmark_start = benchmark_full.index[0]
            effective_benchmark_start = max(series_start, benchmark_start)
            
            # Calculate growth for benchmark from effective start
            benchmark_returns = df[benchmark][df.index >= effective_benchmark_start].dropna()
            benchmark_growth = (1 + benchmark_returns).cumprod()

            # Prepend 1.0 for Series
            series_start_date = series_start - period_offset
            series_start_val = pd.Series([1.0], index=[series_start_date])
            series_growth = pd.concat([series_start_val, series_growth])
            
            # Prepend 1.0 for Benchmark
            if not benchmark_returns.empty:
                benchmark_start_date = effective_benchmark_start - period_offset
                benchmark_start_val = pd.Series([1.0], index=[benchmark_start_date])
                benchmark_growth = pd.concat([benchmark_start_val, benchmark_growth])

            # Create figure for this pair
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=series_growth.index,
                y=series_growth,
                mode='lines',
                name=series,
                line=dict(width=2),
            ))
            fig.add_trace(go.Scatter(
                x=benchmark_growth.index,
                y=benchmark_growth,
                mode='lines',
                name=benchmark,
                line=dict(width=2, dash='dash'),
            ))

            suffix = " (Long-Short)" if is_long_short else ""
            fig.update_layout(
                title=f"Growth of $1: {series} vs {benchmark}{suffix}",
                xaxis_title="Date",
                yaxis_title="Growth of $1",
                height=400,
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
            )

            individual_charts.append(dcc.Graph(figure=fig, style={"marginBottom": "2rem"}))

        # Combine all charts
        charts = [dcc.Graph(figure=main_fig, style={"marginBottom": "3rem"})] + individual_charts

        return html.Div(charts)

    except Exception as e:
        return dmc.Text(f"Error generating growth charts: {str(e)}", size="sm", c="red")


@callback(
    Output("growth-grid", "columnDefs"),
    Output("growth-grid", "rowData"),
    Input("main-tabs", "value"),
    Input("growth-chart-switch", "value"),
    Input("raw-data-store", "data"),
    Input("periodicity-select", "value"),
    Input("series-select", "data"),
    Input("benchmark-assignments-store", "data"),
    Input("long-short-store", "data"),
    Input("date-range-store", "data"),
    Input("vol-scaler-value-store", "data"),
    Input("vol-scaling-assignments-store", "data"),
    prevent_initial_call=True,
)
def update_growth_grid(active_tab, chart_checked, raw_data, periodicity, selected_series, benchmark_assignments, long_short_assignments, date_range, vol_scaler, vol_scaling_assignments):
    """Update Growth of $1 grid (lazy loaded)."""
    # Lazy loading: only generate when growth tab is active and table view is selected
    if active_tab != "growth" or chart_checked != "table":
        raise PreventUpdate

    if raw_data is None or not selected_series:
        return [], []

    try:
        # Use shared calculate_growth_of_dollar function
        growth_df = calculate_growth_of_dollar(
            raw_data,
            periodicity,
            tuple(selected_series),
            str(benchmark_assignments),
            str(long_short_assignments),
            str(date_range),
            vol_scaler or 0,
            str(vol_scaling_assignments)
        )

        if growth_df.empty:
            return [], []

        # Reset index to include Date as a column
        growth_df = growth_df.reset_index()
        if "Date" in growth_df.columns:
            growth_df["Date"] = growth_df["Date"].dt.strftime("%Y-%m-%d")
        elif "index" in growth_df.columns:
            growth_df["Date"] = growth_df["index"].dt.strftime("%Y-%m-%d")
            growth_df = growth_df.drop(columns=["index"])

        # Define column definitions
        column_defs = [
            {"field": "Date", "pinned": "left", "width": 120},
        ]

        for col in growth_df.columns:
            if col != "Date":
                column_defs.append({
                    "field": col,
                    "valueFormatter": {"function": "params.value != null ? d3.format('.4f')(params.value) : ''"},
                })

        # Convert to records
        row_data = growth_df.to_dict("records")

        return column_defs, row_data

    except Exception:
        return [], []


@callback(
    Output("drawdown-charts", "children"),
    Input("main-tabs", "value"),
    Input("drawdown-chart-switch", "value"),
    Input("raw-data-store", "data"),
    Input("periodicity-select", "value"),
    Input("series-select", "data"),
    Input("returns-type-select", "value"),
    Input("benchmark-assignments-store", "data"),
    Input("long-short-store", "data"),
    Input("date-range-store", "data"),
    Input("vol-scaler-value-store", "data"),
    Input("vol-scaling-assignments-store", "data"),
    prevent_initial_call=True,
)
def update_drawdown_charts(active_tab, chart_checked, raw_data, periodicity, selected_series, returns_type, benchmark_assignments, long_short_assignments, date_range, vol_scaler, vol_scaling_assignments):
    """Update Drawdown charts (lazy loaded)."""
    # Lazy loading: only generate when drawdown tab is active and chart view is selected
    if active_tab != "drawdown" or chart_checked != "chart":
        raise PreventUpdate

    if raw_data is None or not selected_series:
        return dmc.Text("Select series to view drawdown charts", size="sm", c="dimmed")

    try:
        # Use shared calculate_drawdown function
        drawdown_df = calculate_drawdown(
            raw_data,
            periodicity,
            tuple(selected_series),
            returns_type,
            str(benchmark_assignments),
            str(long_short_assignments),
            str(date_range),
            vol_scaler or 0,
            str(vol_scaling_assignments)
        )

        if drawdown_df.empty:
            return dmc.Text("No data available for selected series", size="sm", c="dimmed")

        long_short_dict = eval(str(long_short_assignments)) if long_short_assignments else {}

        # Create individual drawdown charts for each series
        charts = []
        for series in drawdown_df.columns:
            drawdown = drawdown_df[series].dropna()

            if drawdown.empty:
                continue

            # Create figure
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=drawdown.index,
                y=drawdown,
                mode='lines',
                name=series,
                line=dict(width=2),
                fill='tozeroy',
                fillcolor='rgba(255, 0, 0, 0.2)',
            ))

            is_long_short = long_short_dict.get(series, False)
            suffix = " (Long-Short)" if is_long_short else ""
            fig.update_layout(
                title=f"Drawdown: {series}{suffix}",
                xaxis_title="Date",
                yaxis_title="Drawdown",
                yaxis_tickformat=".2%",
                height=400,
                hovermode='x unified',
                template="plotly_white",
            )

            charts.append(dcc.Graph(figure=fig, style={"marginBottom": "2rem"}))

        return html.Div(charts)

    except Exception as e:
        return dmc.Text(f"Error generating drawdown charts: {str(e)}", size="sm", c="red")


@callback(
    Output("drawdown-grid", "columnDefs"),
    Output("drawdown-grid", "rowData"),
    Input("main-tabs", "value"),
    Input("drawdown-chart-switch", "value"),
    Input("raw-data-store", "data"),
    Input("periodicity-select", "value"),
    Input("series-select", "data"),
    Input("returns-type-select", "value"),
    Input("benchmark-assignments-store", "data"),
    Input("long-short-store", "data"),
    Input("date-range-store", "data"),
    Input("vol-scaler-value-store", "data"),
    Input("vol-scaling-assignments-store", "data"),
    prevent_initial_call=True,
)
def update_drawdown_grid(active_tab, chart_checked, raw_data, periodicity, selected_series, returns_type, benchmark_assignments, long_short_assignments, date_range, vol_scaler, vol_scaling_assignments):
    """Update Drawdown grid (lazy loaded)."""
    # Lazy loading: only generate when drawdown tab is active and table view is selected
    if active_tab != "drawdown" or chart_checked != "table":
        raise PreventUpdate

    if raw_data is None or not selected_series:
        return [], []

    try:
        # Use shared calculate_drawdown function
        drawdown_df = calculate_drawdown(
            raw_data,
            periodicity,
            tuple(selected_series),
            returns_type,
            str(benchmark_assignments),
            str(long_short_assignments),
            str(date_range),
            vol_scaler or 0,
            str(vol_scaling_assignments)
        )

        if drawdown_df.empty:
            return [], []

        # Reset index to include Date as a column
        drawdown_df = drawdown_df.reset_index()
        if "Date" in drawdown_df.columns:
            drawdown_df["Date"] = drawdown_df["Date"].dt.strftime("%Y-%m-%d")
        elif "index" in drawdown_df.columns:
            drawdown_df["Date"] = drawdown_df["index"].dt.strftime("%Y-%m-%d")
            drawdown_df = drawdown_df.drop(columns=["index"])

        # Define column definitions
        column_defs = [
            {"field": "Date", "pinned": "left", "width": 120},
        ]

        for col in drawdown_df.columns:
            if col != "Date":
                column_defs.append({
                    "field": col,
                    "valueFormatter": {"function": "params.value != null ? d3.format('.2%')(params.value) : ''"},
                })

        # Convert to records
        row_data = drawdown_df.to_dict("records")

        return column_defs, row_data

    except Exception:
        return [], []



@callback(
    Output("download-excel", "data"),
    Input("menu-download-excel", "n_clicks"),
    State("raw-data-store", "data"),
    State("original-periodicity-store", "data"),
    State("periodicity-select", "value"),
    State("series-select", "data"),
    State("returns-type-select", "value"),
    State("benchmark-assignments-store", "data"),
    State("long-short-store", "data"),
    State("date-range-store", "data"),
    State("rolling-window-store", "data"),
    State("rolling-return-type-store", "data"),
    State("monthly-view-store", "data"),
    State("monthly-series-store", "data"),
    State("vol-scaler-value-store", "data"),
    State("vol-scaling-assignments-store", "data"),
    prevent_initial_call=True,
)
def download_excel(n_clicks, raw_data, original_periodicity, selected_periodicity, selected_series, returns_type, benchmark_assignments, long_short_assignments, date_range, rolling_window, rolling_return_type, monthly_view, monthly_series, vol_scaler, vol_scaling_assignments):
    """Generate Excel file with Statistics, Returns, Rolling, Calendar Year, Growth, Drawdown, and Correlogram sheets."""
    if n_clicks is None or raw_data is None or not selected_series:
        raise PreventUpdate

    # Use cached functions to get data
    returns_df = calculate_excess_returns(
        raw_data,
        selected_periodicity or "daily",
        tuple(selected_series),
        str(benchmark_assignments),
        returns_type,
        str(long_short_assignments),
        str(date_range),
        vol_scaler or 0,
        str(vol_scaling_assignments)
    )

    if returns_df.empty:
        raise PreventUpdate

    # Get cached statistics
    stats = calculate_statistics_cached(
        raw_data,
        selected_periodicity or "daily",
        tuple(selected_series),
        str(benchmark_assignments),
        str(long_short_assignments),
        str(date_range),
        vol_scaler or 0,
        str(vol_scaling_assignments)
    )

    # Build statistics DataFrame (transposed: statistics as rows, series as columns)
    stats_data = {"Statistic": [stat_name for stat_name, _ in STATS_CONFIG]}
    for series_stats in stats:
        series_name = series_stats["Series"]
        stats_data[series_name] = [series_stats.get(stat_name) for stat_name, _ in STATS_CONFIG]
    stats_df = pd.DataFrame(stats_data)

    # Prepare correlogram data (correlation matrix)
    corr_df = returns_df.corr()
    corr_df.index.name = "Series"

    # Create Excel file in memory with multiple sheets
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        # Sheet 1: Statistics (moved to first position)
        stats_df.to_excel(writer, sheet_name="Statistics", index=False)

        # Sheet 2: Returns
        returns_df.to_excel(writer, sheet_name="Returns")

        # Sheet 3: Rolling (use current settings)
        try:
            # Use stored rolling options, default to 1y annualized if not set
            window = rolling_window if rolling_window else "1y"
            return_type = rolling_return_type if rolling_return_type else "annualized"

            rolling_df = calculate_rolling_returns(
                raw_data,
                selected_periodicity,
                tuple(selected_series),
                returns_type,
                str(benchmark_assignments),
                str(long_short_assignments),
                str(date_range),
                window,
                return_type,
                "total_return", # Default metric for excel
                vol_scaler or 0,
                str(vol_scaling_assignments)
            )
            if not rolling_df.empty:
                # Create sheet name based on window and type
                window_label_map = {
                    "3m": "3M",
                    "6m": "6M",
                    "1y": "1Y",
                    "3y": "3Y",
                    "5y": "5Y",
                    "10y": "10Y",
                }
                window_label = window_label_map.get(window, "1Y")
                type_label = "Ann" if return_type == "annualized" else "Cum"
                sheet_name = f"Rolling ({window_label} {type_label})"
                rolling_df.to_excel(writer, sheet_name=sheet_name)
        except Exception:
            pass  # Skip if rolling calculation fails

        # Sheet 4: Calendar Year Returns
        if original_periodicity in ["daily", "monthly"]:
            try:
                # Check if monthly view is selected
                if monthly_view == "monthly" and monthly_series and monthly_series in selected_series:
                    # Get monthly view data
                    column_defs, row_data = create_monthly_view(
                        raw_data,
                        monthly_series,
                        original_periodicity,
                        selected_periodicity,
                        returns_type,
                        benchmark_assignments,
                        long_short_assignments,
                        selected_series,
                        date_range,
                        vol_scaler or 0,
                        str(vol_scaling_assignments)
                    )

                    if row_data:
                        # Convert row data to DataFrame
                        calendar_df = pd.DataFrame(row_data)
                        calendar_df = calendar_df.set_index('Year_Label')
                        calendar_df.index.name = 'Year'
                        calendar_df.to_excel(writer, sheet_name="Calendar Year")
                else:
                    # Use standard calendar year returns (all series, one row per year)
                    calendar_df = calculate_calendar_year_returns(
                        raw_data,
                        original_periodicity,
                        selected_periodicity,
                        selected_series,
                        returns_type,
                        benchmark_assignments,
                        long_short_assignments,
                        date_range,
                        vol_scaler or 0,
                        str(vol_scaling_assignments)
                    )
                    if not calendar_df.empty:
                        calendar_df.to_excel(writer, sheet_name="Calendar Year")
            except Exception:
                pass  # Skip if calendar calculation fails

        # Sheet 5: Growth of $1
        try:
            growth_df = calculate_growth_of_dollar(
                raw_data,
                selected_periodicity,
                tuple(selected_series),
                str(benchmark_assignments),
                str(long_short_assignments),
                str(date_range),
                vol_scaler or 0,
                str(vol_scaling_assignments)
            )
            if not growth_df.empty:
                growth_df.to_excel(writer, sheet_name="Growth of $1")
        except Exception:
            pass  # Skip if growth calculation fails

        # Sheet 6: Drawdown
        try:
            drawdown_df = calculate_drawdown(
                raw_data,
                selected_periodicity,
                tuple(selected_series),
                returns_type,
                str(benchmark_assignments),
                str(long_short_assignments),
                str(date_range),
                vol_scaler or 0,
                str(vol_scaling_assignments)
            )
            if not drawdown_df.empty:
                drawdown_df.to_excel(writer, sheet_name="Drawdown")
        except Exception:
            pass  # Skip if drawdown calculation fails

        # Sheet 7: Correlogram
        corr_df.to_excel(writer, sheet_name="Correlogram")

    output.seek(0)

    # Generate filename
    periodicity_suffix = selected_periodicity.replace("_", "-") if selected_periodicity else "returns"
    returns_suffix = "excess" if returns_type == "excess" else "total"
    filename = f"dashmat_{periodicity_suffix}_{returns_suffix}.xlsx"

    return dcc.send_bytes(output.getvalue(), filename)
