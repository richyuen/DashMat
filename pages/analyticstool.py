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
    calculate_rolling_returns,
    create_monthly_view,
    get_available_periodicities,
    json_to_df,
    merge_returns,
    resample_returns,
    resample_returns_cached,
)
from utils.statistics import (
    calculate_all_statistics,
    calculate_drawdown,
    calculate_growth_of_dollar,
)

register_page(__name__, path="/dashboard", name="Dashboard", title="DashMat - Dashboard")

# Performance optimization constants
MAX_SCATTER_MATRIX_SIZE = 10  # Maximum series for scatter matrix (creates n² subplots)

# Statistics row order and formatting
STATS_CONFIG = [
    ("Start Date", None),
    ("End Date", None),
    ("Number of Periods", None),
    ("Cumulative Return", ".2%"),
    ("Annualized Return", ".2%"),
    ("Annualized Excess Return", ".2%"),
    ("Annualized Volatility", ".2%"),
    ("Annualized Tracking Error", ".2%"),
    ("Sharpe Ratio", ".2f"),
    ("Information Ratio", ".2f"),
    ("Hit Rate", ".2%"),
    ("Hit Rate (vs Benchmark)", ".2%"),
    ("Best Period Return", ".2%"),
    ("Worst Period Return", ".2%"),
    ("Maximum Drawdown", ".2%"),
    ("Skewness", ".2f"),
    ("Kurtosis", ".2f"),
    ("1Y Annualized Return", ".2%"),
    ("1Y Sharpe Ratio", ".2f"),
    ("1Y Excess Return", ".2%"),
    ("1Y Information Ratio", ".2f"),
    ("3Y Annualized Return", ".2%"),
    ("3Y Sharpe Ratio", ".2f"),
    ("3Y Excess Return", ".2%"),
    ("3Y Information Ratio", ".2f"),
    ("5Y Annualized Return", ".2%"),
    ("5Y Sharpe Ratio", ".2f"),
    ("5Y Excess Return", ".2%"),
    ("5Y Information Ratio", ".2f"),
]


def df_to_json(df: pd.DataFrame) -> str:
    """Convert DataFrame to JSON string for storage."""
    return df.to_json(date_format="iso", orient="split")




@cache_config.cache.memoize(timeout=0)
def calculate_excess_returns(json_str: str, periodicity: str, selected_series: tuple,
                             benchmark_assignments: str, returns_type: str, long_short_assignments: str,
                             date_range_str: str) -> pd.DataFrame:
    """Calculate excess returns with caching."""
    df = resample_returns_cached(json_str, periodicity)

    # Parse assignments
    benchmark_dict = eval(benchmark_assignments) if benchmark_assignments else {}
    long_short_dict = eval(long_short_assignments) if long_short_assignments else {}

    # Filter to selected series only
    available_series = [s for s in selected_series if s in df.columns]
    if not available_series:
        return pd.DataFrame()

    display_df = df[available_series].copy()

    # Calculate long-short returns for series with long-short enabled
    for series in available_series:
        is_long_short = long_short_dict.get(series, False)
        if is_long_short:
            benchmark = benchmark_dict.get(series, available_series[0])
            if benchmark == series:
                display_df[series] = np.nan
            elif benchmark == "None":
                # None benchmark: keep the series returns as-is (vs zero)
                display_df.loc[:, series] = df[series]
            elif benchmark in df.columns:
                # Long-short: always show difference, regardless of returns_type
                display_df.loc[:, series] = df[series] - df[benchmark]

    # Calculate excess returns if requested (for non-long-short series)
    if returns_type == "excess":
        for series in available_series:
            is_long_short = long_short_dict.get(series, False)
            if not is_long_short:  # Only apply to non-long-short series
                benchmark = benchmark_dict.get(series, available_series[0])
                if benchmark == series:
                    display_df[series] = np.nan
                elif benchmark == "None":
                    # None benchmark: keep the series returns as-is (vs zero)
                    display_df.loc[:, series] = df[series]
                elif benchmark in df.columns:
                    # Vectorized operation instead of per-series assignment
                    display_df.loc[:, series] = df[series] - df[benchmark]

    # Apply date range filter if provided (after all calculations)
    date_range = eval(date_range_str) if date_range_str and date_range_str != "None" else None
    if date_range:
        start_date = pd.to_datetime(date_range["start"])
        end_date = pd.to_datetime(date_range["end"])
        display_df = display_df[(display_df.index >= start_date) & (display_df.index <= end_date)]

    return display_df


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
                                            "Clear local storage and refresh",
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
                                    children=[
                                        dmc.Select(
                                            id="periodicity-select",
                                            label="Periodicity",
                                            data=[{"value": "daily", "label": "Daily"}],
                                            value="daily",
                                            w=200,
                                            disabled=True,
                                        ),
                                        html.Div([
                                            dmc.Text("Returns Type", size="sm", mb=5, c="dimmed"),
                                            dmc.SegmentedControl(
                                                id="returns-type-select",
                                                data=[
                                                    {"value": "total", "label": "Total"},
                                                    {"value": "excess", "label": "Excess"},
                                                ],
                                                value="total",
                                                w=200,
                                            ),
                                        ]),
                                    ],
                                ),
                                html.Div([
                                    html.Div(
                                        id="date-picker-wrapper",
                                        children=[
                                            html.Div([
                                                dmc.Text("Start Date", size="sm", mb=5, c="dimmed"),
                                                dmc.DateInput(
                                                    id="start-date-picker",
                                                    value=None,
                                                    w=200,
                                                    valueFormat="YYYY-MM-DD",
                                                ),
                                            ], style={"marginRight": "15px"}),
                                            html.Div([
                                                dmc.Text("End Date", size="sm", mb=5, c="dimmed"),
                                                dmc.DateInput(
                                                    id="end-date-picker",
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
                                                ),
                                            ], style={"marginRight": "10px", "alignSelf": "flex-end", "marginBottom": "2px"}),
                                            html.Div([
                                                dmc.Button(
                                                    "Maximum Range",
                                                    id="maximum-range-button",
                                                    size="xs",
                                                    variant="outline",
                                                    disabled=True,
                                                ),
                                            ], style={"alignSelf": "flex-end", "marginBottom": "2px"}),
                                        ],
                                        style={"display": "flex", "opacity": 0.5, "pointerEvents": "none", "alignItems": "flex-start"},
                                    ),
                                ], style={"marginBottom": "1rem"}),
                                dmc.Divider(mb="md"),
                                dmc.Text("Series Selection", size="sm", c="dimmed", mb="xs"),
                                html.Div(
                                    id="series-selection-container",
                                    children=[dmc.Text("Upload data to select series", size="sm", c="dimmed")],
                                ),
                                # Hidden store for series-select value (driven by checkboxes)
                                dcc.Store(id="series-select", data=[], storage_type="local"),
                                # Hidden store to track which series is being edited
                                dcc.Store(id="series-edit-mode", data=None),
                            ]
                        ),
                    ],
                ),
            ],
        ),
        # Alert for messages (with close button)
        dmc.Alert(
            id="alert-message",
            title="Info",
            color="blue",
            hide=True,
            mb="md",
            withCloseButton=True,
        ),
        # Tabs with AG Grid and Statistics
        dmc.Tabs(
            id="main-tabs",
            value="statistics",
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
                                    id="rolling-window-select",
                                    data=[
                                        {"value": "3m", "label": "3-month"},
                                        {"value": "6m", "label": "6-month"},
                                        {"value": "1y", "label": "1-year"},
                                        {"value": "3y", "label": "3-year"},
                                        {"value": "5y", "label": "5-year"},
                                        {"value": "10y", "label": "10-year"},
                                    ],
                                    value="1y",
                                    w=120,
                                    size="sm",
                                ),
                                dmc.SegmentedControl(
                                    id="rolling-return-type-select",
                                    data=[
                                        {"value": "cumulative", "label": "Cumulative"},
                                        {"value": "annualized", "label": "Annualized"},
                                    ],
                                    value="annualized",
                                    size="sm",
                                ),
                                dmc.SegmentedControl(
                                    id="rolling-chart-switch",
                                    data=[
                                        {"value": "table", "label": "Table"},
                                        {"value": "chart", "label": "Chart"},
                                    ],
                                    value="table",
                                    size="sm",
                                ),
                            ],
                        ),
                        dcc.Loading(
                            id="loading-rolling",
                            type="default",
                            children=[
                                html.Div(
                                    id="rolling-grid-container",
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
                                    style={"display": "none"},
                                    children=[
                                        dcc.Graph(
                                            id="rolling-chart",
                                            style={"height": "550px"},
                                        ),
                                    ],
                                ),
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
                                    value="annual",
                                    size="sm",
                                ),
                                dmc.Select(
                                    id="monthly-series-select",
                                    data=[],
                                    value=None,
                                    w=200,
                                    size="sm",
                                    placeholder="Select series",
                                    disabled=True,
                                ),
                            ],
                        ),
                        dcc.Loading(
                            id="loading-calendar",
                            type="default",
                            children=[
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
                    ],
                ),
                dmc.TabsPanel(
                    value="correlogram",
                    pt="md",
                    children=[
                        dcc.Loading(
                            id="loading-correlogram",
                            type="default",
                            children=[
                                dcc.Graph(
                                    id="correlogram-graph",
                                    style={"height": "700px"},
                                ),
                            ],
                        ),
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
                                    value="chart",
                                    size="sm",
                                ),
                            ],
                        ),
                        dcc.Loading(
                            id="loading-growth",
                            type="default",
                            children=[
                                html.Div(
                                    id="growth-chart-container",
                                    children=[
                                        html.Div(id="growth-charts-container"),
                                    ],
                                ),
                                html.Div(
                                    id="growth-grid-container",
                                    style={"display": "none"},
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
                                    value="chart",
                                    size="sm",
                                ),
                            ],
                        ),
                        dcc.Loading(
                            id="loading-drawdown",
                            type="default",
                            children=[
                                html.Div(
                                    id="drawdown-chart-container",
                                    children=[
                                        html.Div(id="drawdown-charts"),
                                    ],
                                ),
                                html.Div(
                                    id="drawdown-grid-container",
                                    style={"display": "none"},
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
            ],
        ),
        # Hidden stores for state management (using local storage for persistence)
        dcc.Store(id="raw-data-store", data=None, storage_type="local"),
        dcc.Store(id="original-periodicity-store", data="daily", storage_type="local"),
        dcc.Store(id="benchmark-assignments-store", data={}, storage_type="local"),
        dcc.Store(id="long-short-store", data={}, storage_type="local"),
        dcc.Store(id="periodicity-value-store", data="daily", storage_type="local"),
        dcc.Store(id="returns-type-value-store", data="total", storage_type="local"),
        dcc.Store(id="series-select-value-store", data=[], storage_type="local"),
        dcc.Store(id="series-order-store", data=[], storage_type="local"),
        dcc.Store(id="active-tab-store", data="statistics", storage_type="local"),
        dcc.Store(id="rolling-window-store", data="1y", storage_type="local"),
        dcc.Store(id="rolling-return-type-store", data="annualized", storage_type="local"),
        dcc.Store(id="rolling-chart-switch-store", data="table", storage_type="local"),
        dcc.Store(id="drawdown-chart-switch-store", data="chart", storage_type="local"),
        dcc.Store(id="growth-chart-switch-store", data="chart", storage_type="local"),
        dcc.Store(id="monthly-view-store", data="annual", storage_type="local"),
        dcc.Store(id="monthly-series-store", data=None, storage_type="local"),
        dcc.Store(id="date-range-store", data=None, storage_type="local"),
        dcc.Store(id="download-enabled-store", data=False),
        dcc.Download(id="download-excel"),
        dcc.Location(id="url-location", refresh=True),
        # Hidden file upload (triggered by menu item)
        html.Div(
            dcc.Upload(
                id="upload-data",
                children=html.Div(id="upload-trigger"),
                multiple=False,
                accept=".csv,.xlsx,.xls",
            ),
            style={"display": "none"},
        ),
        # Store to trigger clientside focus on edit input
        dcc.Store(id="edit-box-focus-trigger", data=None),
        # Dummy div for clientside callback output
        html.Div(id="dummy-focus-output"),
    ],
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


# Clientside callback to clear local storage and refresh page
clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks) {
            // Clear all localStorage keys specific to Analytics Tool
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
                'date-range-store'
            ];

            keysToRemove.forEach(key => {
                localStorage.removeItem(key);
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
    Output("series-order-store", "data", allow_duplicate=True),
    Input({"type": "move-up-button", "series": ALL}, "n_clicks"),
    Input({"type": "move-down-button", "series": ALL}, "n_clicks"),
    State("series-order-store", "data"),
    State("raw-data-store", "data"),
    prevent_initial_call=True,
)
def reorder_series(up_clicks, down_clicks, current_order, raw_data):
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

    # Parse the button ID (it will be like '{"type":"move-up-button","series":"SPY"}.n_clicks')
    try:
        button_data = eval(triggered_id.rsplit(".", 1)[0])
        button_type = button_data["type"]
        series_name = button_data["series"]
    except (SyntaxError, KeyError, ValueError):
        raise PreventUpdate

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

    return new_order


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
    Input("menu-clear-all-series", "n_clicks"),
    prevent_initial_call=True,
)
def clear_all_series(n_clicks):
    """Clear all loaded series and reset application state."""
    if n_clicks is None:
        raise PreventUpdate

    # Reset all stores to initial state
    return None, "daily", {}, {}, None, None, [], [], []


@callback(
    Output("periodicity-select", "data", allow_duplicate=True),
    Output("periodicity-select", "value", allow_duplicate=True),
    Output("periodicity-select", "disabled", allow_duplicate=True),
    Output("series-select", "data", allow_duplicate=True),
    Output("returns-type-select", "value", allow_duplicate=True),
    Input("raw-data-store", "data"),
    State("original-periodicity-store", "data"),
    State("periodicity-value-store", "data"),
    State("series-select-value-store", "data"),
    State("returns-type-value-store", "data"),
    prevent_initial_call="initial_duplicate",
)
def restore_state_from_storage(raw_data, original_periodicity, stored_periodicity, stored_series, stored_returns_type):
    """Restore UI state from local storage on page load."""
    if raw_data is None:
        return [], "daily", True, [], "total"

    try:
        df = json_to_df(raw_data)

        # Get available periodicities
        from utils.returns import get_available_periodicities
        periodicity_options = get_available_periodicities(original_periodicity or "daily")

        # Validate stored values
        valid_periodicity = stored_periodicity if stored_periodicity in [p["value"] for p in periodicity_options] else (original_periodicity or "daily")
        valid_series = [s for s in (stored_series or []) if s in df.columns]
        valid_returns_type = stored_returns_type if stored_returns_type in ["total", "excess"] else "total"

        return periodicity_options, valid_periodicity, False, valid_series, valid_returns_type

    except Exception:
        return [], "daily", True, [], "total"


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
    Output("main-tabs", "value"),
    Input("raw-data-store", "data"),
    State("active-tab-store", "data"),
    prevent_initial_call="initial_duplicate",
)
def restore_active_tab(raw_data, stored_tab):
    """Restore active tab from local storage on page load."""
    # Return stored tab if available, otherwise default to statistics
    return stored_tab if stored_tab else "statistics"


@callback(
    Output("rolling-window-store", "data"),
    Input("rolling-window-select", "value"),
    prevent_initial_call=True,
)
def save_rolling_window(value):
    """Save rolling window selection to local storage."""
    return value or "1y"


@callback(
    Output("rolling-return-type-store", "data"),
    Input("rolling-return-type-select", "value"),
    prevent_initial_call=True,
)
def save_rolling_return_type(value):
    """Save rolling return type to local storage."""
    return value or "annualized"


@callback(
    Output("rolling-window-select", "value"),
    Output("rolling-return-type-select", "value"),
    Input("raw-data-store", "data"),
    State("rolling-window-store", "data"),
    State("rolling-return-type-store", "data"),
    prevent_initial_call="initial_duplicate",
)
def restore_rolling_options(raw_data, stored_window, stored_return_type):
    """Restore rolling options from local storage on page load."""
    window = stored_window if stored_window else "1y"
    return_type = stored_return_type if stored_return_type else "annualized"
    return window, return_type


@callback(
    Output("rolling-chart-switch-store", "data"),
    Input("rolling-chart-switch", "value"),
    prevent_initial_call=True,
)
def save_rolling_chart_switch(value):
    """Save rolling chart switch state to local storage."""
    return value if value is not None else "table"


@callback(
    Output("rolling-chart-switch", "value"),
    Input("raw-data-store", "data"),
    State("rolling-chart-switch-store", "data"),
    prevent_initial_call="initial_duplicate",
)
def restore_rolling_chart_switch(raw_data, stored_chart_switch):
    """Restore rolling chart switch from local storage on page load."""
    return stored_chart_switch if stored_chart_switch is not None else "table"


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
    Output("drawdown-chart-switch", "value"),
    Input("raw-data-store", "data"),
    State("drawdown-chart-switch-store", "data"),
    prevent_initial_call="initial_duplicate",
)
def restore_drawdown_chart_switch(raw_data, stored_chart_switch):
    """Restore drawdown chart switch from local storage on page load."""
    return stored_chart_switch if stored_chart_switch is not None else "chart"


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
    Output("growth-chart-switch", "value"),
    Input("raw-data-store", "data"),
    State("growth-chart-switch-store", "data"),
    prevent_initial_call="initial_duplicate",
)
def restore_growth_chart_switch(raw_data, stored_chart_switch):
    """Restore growth chart switch from local storage on page load."""
    return stored_chart_switch if stored_chart_switch is not None else "chart"


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
    Output("monthly-view-checkbox", "value"),
    Input("raw-data-store", "data"),
    State("monthly-view-store", "data"),
    prevent_initial_call="initial_duplicate",
)
def restore_monthly_view(raw_data, stored_monthly_view):
    """Restore monthly view selection from local storage on page load."""
    return stored_monthly_view if stored_monthly_view is not None else "annual"


@callback(
    Output("raw-data-store", "data"),
    Output("original-periodicity-store", "data"),
    Output("periodicity-select", "data"),
    Output("periodicity-select", "value"),
    Output("periodicity-select", "disabled"),
    Output("series-select", "data", allow_duplicate=True),
    Output("alert-message", "children"),
    Output("alert-message", "color"),
    Output("alert-message", "hide"),
    Output("periodicity-value-store", "data", allow_duplicate=True),
    Output("series-select-value-store", "data", allow_duplicate=True),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    State("raw-data-store", "data"),
    State("original-periodicity-store", "data"),
    State("series-select", "data"),
    prevent_initial_call=True,
)
def handle_upload(contents, filename, existing_data, existing_periodicity, current_selection):
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
                    no_update, no_update,
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

        return (
            df_to_json(merged_df),
            combined_periodicity,
            periodicity_options,
            default_periodicity,
            False,
            updated_selection,
            f"Loaded {len(new_df.columns)} series with {len(new_df)} rows from {filename}",
            "green",
            False,
            default_periodicity,
            updated_selection,
        )

    except Exception as e:
        return (
            no_update, no_update, no_update, no_update, no_update,
            no_update,
            f"Error loading file: {str(e)}",
            "red",
            False,
            no_update, no_update,
        )


@callback(
    Output("series-selection-container", "children"),
    Output("series-order-store", "data", allow_duplicate=True),
    Input("raw-data-store", "data"),
    Input("series-select", "data"),
    Input("series-order-store", "data"),
    Input("series-edit-mode", "data"),
    State("benchmark-assignments-store", "data"),
    State("long-short-store", "data"),
    prevent_initial_call="initial_duplicate",
)
def update_series_selectors(raw_data, selected_series, series_order, edit_mode_series, current_assignments, long_short_assignments):
    """Create series selection rows with checkbox, benchmark dropdown, long-short, reorder buttons, and delete button."""
    if raw_data is None:
        return [], []

    df = json_to_df(raw_data)
    all_series = list(df.columns)

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
        # Remove any deleted series
        series_order = [s for s in series_order if s in all_series]

    default_benchmark = all_series[0] if all_series else None
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
    Output("series-select", "data", allow_duplicate=True),
    Input({"type": "series-include-checkbox", "series": ALL}, "checked"),
    State({"type": "series-include-checkbox", "series": ALL}, "id"),
    State("raw-data-store", "data"),
    State("series-order-store", "data"),
    prevent_initial_call=True,
)
def update_series_selection_from_checkboxes(checkbox_values, checkbox_ids, raw_data, series_order):
    """Update series selection based on checkbox states, preserving order."""
    if raw_data is None or checkbox_values is None or not checkbox_ids:
        return []

    # Map checkbox values to series using the pattern-matching IDs
    checkbox_map = {}
    for i, checkbox_id in enumerate(checkbox_ids):
        series = checkbox_id["series"]
        if i < len(checkbox_values):
            checkbox_map[series] = checkbox_values[i]

    df = json_to_df(raw_data)
    all_series = list(df.columns)

    # Use series_order if available, otherwise use DataFrame column order
    ordered_series = series_order if series_order else all_series

    # Build selected list in the correct order
    selected = []
    for series in ordered_series:
        if checkbox_map.get(series, False):
            selected.append(series)

    return selected


@callback(
    Output("raw-data-store", "data", allow_duplicate=True),
    Output("series-select", "data", allow_duplicate=True),
    Input({"type": "delete-series-button", "series": ALL}, "n_clicks"),
    State("raw-data-store", "data"),
    State("series-select", "data"),
    prevent_initial_call=True,
)
def delete_series(n_clicks_list, raw_data, selected_series):
    """Delete a series from the raw data when trash icon is clicked."""
    if raw_data is None or not n_clicks_list or all(n is None for n in n_clicks_list):
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

    df = json_to_df(raw_data)

    if series_to_delete not in df.columns:
        raise PreventUpdate

    # Remove the series
    df = df.drop(columns=[series_to_delete])

    # If no series left, return None
    if df.empty or len(df.columns) == 0:
        return None, []

    # Update selected series to remove deleted one
    new_selected = [s for s in (selected_series or []) if s != series_to_delete]

    return df_to_json(df), new_selected


@callback(
    Output("series-edit-mode", "data"),
    Output("edit-box-focus-trigger", "data", allow_duplicate=True),
    Input({"type": "edit-series-button", "series": ALL}, "n_clicks"),
    State({"type": "edit-series-button", "series": ALL}, "id"),
    prevent_initial_call=True,
)
def enter_edit_mode(n_clicks_list, button_ids):
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

    return series_name, series_name


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
    Output("benchmark-assignments-store", "data", allow_duplicate=True),
    Output("long-short-store", "data", allow_duplicate=True),
    Output("series-select", "data", allow_duplicate=True),
    Output("series-order-store", "data", allow_duplicate=True),
    Output("series-edit-mode", "data", allow_duplicate=True),
    Output("series-select-value-store", "data", allow_duplicate=True),
    Output("edit-box-focus-trigger", "data", allow_duplicate=True),
    Input({"type": "save-edit-button", "series": ALL}, "n_clicks"),
    State({"type": "save-edit-button", "series": ALL}, "id"),
    State({"type": "edit-series-input", "series": ALL}, "value"),
    State({"type": "edit-series-input", "series": ALL}, "id"),
    State("raw-data-store", "data"),
    State("benchmark-assignments-store", "data"),
    State("long-short-store", "data"),
    State("series-select", "data"),
    State("series-order-store", "data"),
    State("series-edit-mode", "data"),
    prevent_initial_call=True,
)
def save_edit(save_clicks_list, save_ids, input_values, input_ids, raw_data, benchmark_assignments, long_short_assignments, series_select, series_order, edit_mode_series):
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
        return no_update, no_update, no_update, no_update, no_update, None, no_update, None

    # Check if new name already exists
    df = json_to_df(raw_data)
    if new_name in df.columns:
        # Don't allow duplicate names, exit edit mode
        return no_update, no_update, no_update, no_update, no_update, None, no_update, None

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

    # Update series selection
    new_series_select = [new_name if s == old_name else s for s in series_select]

    # Update series order
    new_series_order = [new_name if s == old_name else s for s in series_order]

    # Return updated data and exit edit mode
    return new_raw_data, new_benchmark_assignments, new_long_short_assignments, new_series_select, new_series_order, None, new_series_select, None


@callback(
    Output("benchmark-assignments-store", "data"),
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
    Output("long-short-store", "data"),
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
    prevent_initial_call=True,
)
def update_grid(raw_data, periodicity, selected_series, returns_type, benchmark_assignments, long_short_assignments, date_range):
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
            str(date_range)  # Convert to string for cache key
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
    Input("returns-type-select", "value"),
    Input("benchmark-assignments-store", "data"),
    Input("long-short-store", "data"),
    Input("date-range-store", "data"),
    prevent_initial_call=True,
)
def update_rolling_grid(active_tab, raw_data, periodicity, selected_series, rolling_window, rolling_return_type, returns_type, benchmark_assignments, long_short_assignments, date_range):
    """Update the Rolling Returns grid with rolling window calculations."""
    # Lazy loading: only calculate when rolling tab is active
    if active_tab != "rolling":
        raise PreventUpdate

    if raw_data is None or not selected_series:
        return [], []

    try:
        # Use shared calculate_rolling_returns function
        rolling_df = calculate_rolling_returns(
            raw_data,
            periodicity,
            tuple(selected_series),
            returns_type,
            str(benchmark_assignments),
            str(long_short_assignments),
            str(date_range),
            rolling_window,
            rolling_return_type
        )

        if rolling_df.empty:
            return [], []

        if rolling_df.empty:
            return [], []

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
                "valueFormatter": {"function": "params.value != null ? d3.format('.2%')(params.value) : ''"},
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
    Output("rolling-chart", "figure"),
    Input("main-tabs", "value"),
    Input("rolling-chart-switch", "value"),
    Input("raw-data-store", "data"),
    Input("periodicity-select", "value"),
    Input("series-select", "data"),
    Input("rolling-window-select", "value"),
    Input("rolling-return-type-select", "value"),
    Input("returns-type-select", "value"),
    Input("benchmark-assignments-store", "data"),
    Input("long-short-store", "data"),
    Input("date-range-store", "data"),
    prevent_initial_call=True,
)
def update_rolling_chart(active_tab, chart_checked, raw_data, periodicity, selected_series, rolling_window, rolling_return_type, returns_type, benchmark_assignments, long_short_assignments, date_range):
    """Update the Rolling Returns chart with rolling window calculations."""
    # Create empty figure
    empty_fig = go.Figure()
    empty_fig.update_layout(
        title="",
        xaxis_title="",
        yaxis_title="",
        template="plotly_white",
    )

    # Lazy loading: only calculate when rolling tab is active and chart view is selected
    if active_tab != "rolling" or chart_checked != "chart":
        return empty_fig

    if raw_data is None or not selected_series:
        return empty_fig

    try:
        from utils.statistics import annualization_factor

        # Get the resampled data
        df = resample_returns_cached(raw_data, periodicity or "daily")

        # Apply date range filter if provided
        date_range_dict = eval(str(date_range)) if date_range and str(date_range) != "None" else None
        if date_range_dict:
            start_date = pd.to_datetime(date_range_dict["start"])
            end_date = pd.to_datetime(date_range_dict["end"])
            df = df[(df.index >= start_date) & (df.index <= end_date)]

        # Parse assignments
        benchmark_dict = eval(str(benchmark_assignments)) if benchmark_assignments else {}
        long_short_dict = eval(str(long_short_assignments)) if long_short_assignments else {}

        # Filter to selected series only
        available_series = [s for s in selected_series if s in df.columns]
        if not available_series:
            return empty_fig

        # Calculate periods per year and window size
        periods_per_year = annualization_factor(periodicity or "daily")

        # For daily data, use calendar days; for other periodicities, use number of periods
        use_calendar_days = (periodicity or "daily") == "daily"

        if use_calendar_days:
            # Map rolling window to calendar days
            window_map_days = {
                "3m": "91D",
                "6m": "183D",
                "1y": "365D",
                "3y": "1096D",
                "5y": "1826D",
                "10y": "3652D",
            }
            window_spec = window_map_days.get(rolling_window, "365D")

            # Extract the number of days from the window spec
            window_days_map = {
                "3m": 91,
                "6m": 183,
                "1y": 365,
                "3y": 1096,
                "5y": 1826,
                "10y": 3652,
            }
            min_calendar_days = window_days_map.get(rolling_window, 365)
            window_size = None
        else:
            # Map rolling window to number of periods
            window_map = {
                "3m": int(periods_per_year / 4),
                "6m": int(periods_per_year / 2),
                "1y": int(periods_per_year),
                "3y": int(periods_per_year * 3),
                "5y": int(periods_per_year * 5),
                "10y": int(periods_per_year * 10),
            }
            window_size = window_map.get(rolling_window, int(periods_per_year))
            window_size = max(1, window_size)
            window_spec = window_size

        # Map rolling window to number of years for annualization
        window_years_map = {
            "3m": 0.25,
            "6m": 0.5,
            "1y": 1.0,
            "3y": 3.0,
            "5y": 5.0,
            "10y": 10.0,
        }
        window_years = window_years_map.get(rolling_window, 1.0)

        # Calculate rolling returns for each series (same logic as grid)
        rolling_df = pd.DataFrame(index=df.index)

        for series in available_series:
            is_long_short = long_short_dict.get(series, False)
            benchmark = benchmark_dict.get(series, available_series[0])

            # Calculate rolling returns
            def calc_rolling_return(window):
                if len(window) == 0:
                    return np.nan
                if not use_calendar_days and len(window) < window_size:
                    return np.nan
                cum_ret = (1 + window).prod() - 1
                if rolling_return_type == "annualized":
                    if window_years <= 1.0:
                        return cum_ret
                    return (1 + cum_ret) ** (1 / window_years) - 1
                else:
                    return cum_ret

            if is_long_short:
                if benchmark == "None":
                    series_returns = df[series]
                elif benchmark == series:
                    rolling_df[series] = np.nan
                    continue
                elif benchmark in df.columns:
                    series_returns = df[series] - df[benchmark]
                else:
                    series_returns = df[series]

                if use_calendar_days:
                    rolling_returns = series_returns.rolling(window=window_spec).apply(
                        calc_rolling_return, raw=False
                    )
                else:
                    rolling_returns = series_returns.rolling(window=window_spec, min_periods=window_size).apply(
                        calc_rolling_return, raw=False
                    )
                rolling_df[series] = rolling_returns
            else:
                if returns_type == "excess":
                    if benchmark == "None":
                        if use_calendar_days:
                            rolling_returns = df[series].rolling(window=window_spec).apply(
                                calc_rolling_return, raw=False
                            )
                        else:
                            rolling_returns = df[series].rolling(window=window_spec, min_periods=window_size).apply(
                                calc_rolling_return, raw=False
                            )
                        rolling_df[series] = rolling_returns
                    elif benchmark == series:
                        rolling_df[series] = np.nan
                    elif benchmark in df.columns:
                        if use_calendar_days:
                            rolling_series = df[series].rolling(window=window_spec).apply(
                                calc_rolling_return, raw=False
                            )
                            rolling_bench = df[benchmark].rolling(window=window_spec).apply(
                                calc_rolling_return, raw=False
                            )
                        else:
                            rolling_series = df[series].rolling(window=window_spec, min_periods=window_size).apply(
                                calc_rolling_return, raw=False
                            )
                            rolling_bench = df[benchmark].rolling(window=window_spec, min_periods=window_size).apply(
                                calc_rolling_return, raw=False
                            )
                        rolling_df[series] = rolling_series - rolling_bench
                    else:
                        if use_calendar_days:
                            rolling_returns = df[series].rolling(window=window_spec).apply(
                                calc_rolling_return, raw=False
                            )
                        else:
                            rolling_returns = df[series].rolling(window=window_spec, min_periods=window_size).apply(
                                calc_rolling_return, raw=False
                            )
                        rolling_df[series] = rolling_returns
                else:
                    if use_calendar_days:
                        rolling_returns = df[series].rolling(window=window_spec).apply(
                            calc_rolling_return, raw=False
                        )
                    else:
                        rolling_returns = df[series].rolling(window=window_spec, min_periods=window_size).apply(
                            calc_rolling_return, raw=False
                        )
                    rolling_df[series] = rolling_returns

        # For calendar-based windows, filter out periods that don't have enough calendar days
        if use_calendar_days and len(rolling_df) > 0:
            first_date = df.index.min()
            valid_dates_mask = (rolling_df.index - first_date).days >= min_calendar_days - 1
            rolling_df = rolling_df[valid_dates_mask]

        # Drop rows with all NaN values
        rolling_df = rolling_df.dropna(how='all')

        if rolling_df.empty:
            return empty_fig

        # Create the line chart
        fig = go.Figure()

        for col in rolling_df.columns:
            fig.add_trace(go.Scatter(
                x=rolling_df.index,
                y=rolling_df[col],
                mode='lines',
                name=col,
                hovertemplate='%{y:.2%}<extra></extra>',
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
        return_type_label = "Annualized" if rolling_return_type == "annualized" else "Cumulative"

        fig.update_layout(
            title=f"Rolling {window_label} {return_type_label} Returns",
            xaxis_title="Date",
            yaxis_title="Return",
            yaxis_tickformat=".2%",
            template="plotly_white",
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
        )

        return fig

    except Exception:
        return empty_fig


@cache_config.cache.memoize(timeout=0)
def calculate_statistics_cached(json_str: str, periodicity: str, selected_series: tuple,
                                benchmark_assignments: str, long_short_assignments: str, date_range_str: str) -> list:
    """Calculate statistics with caching."""
    df = resample_returns_cached(json_str, periodicity)

    # Apply date range filter if provided
    date_range = eval(date_range_str) if date_range_str and date_range_str != "None" else None
    if date_range:
        start_date = pd.to_datetime(date_range["start"])
        end_date = pd.to_datetime(date_range["end"])
        df = df[(df.index >= start_date) & (df.index <= end_date)]

    benchmark_dict = eval(benchmark_assignments) if benchmark_assignments else {}
    long_short_dict = eval(long_short_assignments) if long_short_assignments else {}

    return calculate_all_statistics(
        df,
        list(selected_series),
        benchmark_dict,
        periodicity,
        long_short_dict,
    )


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
    prevent_initial_call=True,
)
def update_calendar_grid(active_tab, raw_data, original_periodicity, selected_periodicity, selected_series, returns_type, benchmark_assignments, long_short_assignments, date_range, monthly_view, monthly_series):
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
                date_range
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
                date_range
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
    prevent_initial_call=True,
)
def update_statistics(raw_data, periodicity, selected_series, benchmark_assignments, long_short_assignments, date_range):
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
            str(date_range)
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


@cache_config.cache.memoize(timeout=0)
def generate_correlogram_cached(json_str: str, periodicity: str, selected_series: tuple,
                                returns_type: str, benchmark_assignments: str, long_short_assignments: str,
                                date_range_str: str):
    """Generate correlogram with caching."""
    display_df = calculate_excess_returns(
        json_str, periodicity, selected_series, benchmark_assignments, returns_type, long_short_assignments, date_range_str
    )

    if display_df.empty:
        return None

    available_series = list(display_df.columns)
    n = len(available_series)

    # Calculate correlation matrix
    corr_matrix = display_df.corr()

    return {
        'display_df': display_df,
        'corr_matrix': corr_matrix,
        'available_series': available_series,
        'n': n
    }


@callback(
    Output("correlogram-graph", "figure"),
    Input("main-tabs", "value"),  # Lazy loading: only update when tab is active
    Input("raw-data-store", "data"),
    Input("periodicity-select", "value"),
    Input("series-select", "data"),
    Input("returns-type-select", "value"),
    Input("benchmark-assignments-store", "data"),
    Input("long-short-store", "data"),
    Input("date-range-store", "data"),
    prevent_initial_call=True,
)
def update_correlogram(active_tab, raw_data, periodicity, selected_series, returns_type, benchmark_assignments, long_short_assignments, date_range):
    """Update the Correlogram with custom pairs plot (lazy loaded, size-limited, cached)."""
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
    )

    # Lazy loading: only generate when correlogram tab is active
    if active_tab != "correlogram":
        raise PreventUpdate

    if raw_data is None or not selected_series or len(selected_series) < 2:
        return empty_fig

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
                str(date_range)
            )

            if result is None:
                return empty_fig

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

            heatmap_fig.update_layout(
                title=f"Correlation Matrix ({returns_type.title()} Returns)",
                height=max(500, 30 * len(available_series) + 150),
                xaxis=dict(tickangle=45),
                yaxis=dict(autorange='reversed'),
            )

            return heatmap_fig

        except Exception:
            return empty_fig

    try:
        # Use cached function to avoid repeated computation
        result = generate_correlogram_cached(
            raw_data,
            periodicity or "daily",
            tuple(selected_series),
            returns_type,
            str(benchmark_assignments),
            str(long_short_assignments),
            str(date_range)
        )

        if result is None:
            return empty_fig

        display_df = result['display_df']
        corr_matrix = result['corr_matrix']
        available_series = result['available_series']
        n = result['n']

        if n < 2:
            return empty_fig

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
        fig.update_layout(
            title=f"Scatter Matrix ({returns_type.title()} Returns)",
            height=max(500, 150 * n),
            margin=dict(l=60, r=30, t=50, b=60),
            showlegend=False,
        )

        # Hide tick labels for inner plots
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)

        return fig

    except Exception:
        return empty_fig


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
    prevent_initial_call=True,
)
def update_growth_charts(active_tab, chart_checked, raw_data, periodicity, selected_series, benchmark_assignments, long_short_assignments, date_range):
    """Update Growth of $1 charts (lazy loaded)."""
    # Lazy loading: only generate when growth tab is active and chart view is selected
    if active_tab != "growth" or chart_checked != "chart":
        raise PreventUpdate

    if raw_data is None or not selected_series:
        return dmc.Text("Select series to view growth charts", size="sm", c="dimmed")

    try:
        df = resample_returns_cached(raw_data, periodicity or "daily")

        # Apply date range filter if provided
        date_range_dict = eval(str(date_range)) if date_range and str(date_range) != "None" else None
        if date_range_dict:
            start_date = pd.to_datetime(date_range_dict["start"])
            end_date = pd.to_datetime(date_range_dict["end"])
            df = df[(df.index >= start_date) & (df.index <= end_date)]

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

        # Create the main growth chart
        main_chart_data = {}
        for series in available_series:
            is_long_short = long_short_dict.get(series, False)
            benchmark = benchmark_dict.get(series, available_series[0])

            if is_long_short:
                # For long-short, use the difference
                if benchmark == "None":
                    returns = df[series]
                elif benchmark == series:
                    # Skip series where benchmark is itself for long-short
                    continue
                elif benchmark in df.columns:
                    returns = df[series] - df[benchmark]
                else:
                    returns = df[series]
            else:
                # For non-long-short, use total returns
                returns = df[series]
            
            # Drop NaNs before calculation
            returns = returns.dropna()

            # Calculate cumulative growth (compounded)
            growth = (1 + returns).cumprod()

            # Prepend starting value of 1.0 at one period before first date
            if len(growth) > 0:
                first_date = growth.index[0]
                # Create a series with 1.0 at the start, one full period earlier
                start_date = first_date - period_offset
                start_value = pd.Series([1.0], index=[start_date])
                growth = pd.concat([start_value, growth])

            main_chart_data[series] = growth

        # Create main growth figure
        main_fig = go.Figure()
        for series, growth in main_chart_data.items():
            main_fig.add_trace(go.Scatter(
                x=growth.index,
                y=growth,
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
    prevent_initial_call=True,
)
def update_growth_grid(active_tab, chart_checked, raw_data, periodicity, selected_series, benchmark_assignments, long_short_assignments, date_range):
    """Update Growth of $1 grid (lazy loaded)."""
    # Lazy loading: only generate when growth tab is active and table view is selected
    if active_tab != "growth" or chart_checked != "table":
        return [], []

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
            str(date_range)
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
    prevent_initial_call=True,
)
def update_drawdown_charts(active_tab, chart_checked, raw_data, periodicity, selected_series, returns_type, benchmark_assignments, long_short_assignments, date_range):
    """Update Drawdown charts (lazy loaded)."""
    # Lazy loading: only generate when drawdown tab is active and chart view is selected
    if active_tab != "drawdown" or chart_checked != "chart":
        raise PreventUpdate

    if raw_data is None or not selected_series:
        return dmc.Text("Select series to view drawdown charts", size="sm", c="dimmed")

    try:
        df = resample_returns_cached(raw_data, periodicity or "daily")

        # Apply date range filter if provided
        date_range_dict = eval(str(date_range)) if date_range and str(date_range) != "None" else None
        if date_range_dict:
            start_date = pd.to_datetime(date_range_dict["start"])
            end_date = pd.to_datetime(date_range_dict["end"])
            df = df[(df.index >= start_date) & (df.index <= end_date)]

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

        # Create individual drawdown charts for each series
        charts = []
        for series in available_series:
            is_long_short = long_short_dict.get(series, False)
            benchmark = benchmark_dict.get(series, available_series[0])

            if is_long_short:
                # For long-short, use the difference
                if benchmark == "None":
                    returns = df[series]
                elif benchmark == series:
                    # Skip series where benchmark is itself for long-short
                    continue
                elif benchmark in df.columns:
                    returns = df[series] - df[benchmark]
                else:
                    returns = df[series]
                
                # Drop NaNs before calculation to handle different start dates
                returns = returns.dropna()

                # Calculate cumulative growth
                growth = (1 + returns).cumprod()
            elif returns_type == "excess" and benchmark != "None" and benchmark != series and benchmark in df.columns:
                # For excess returns, calculate drawdown of series relative to benchmark
                # Compound each separately, then calculate relative performance
                
                # Align data first by dropping NaNs where either series or benchmark is missing
                aligned_df = df[[series, benchmark]].dropna()
                
                series_growth = (1 + aligned_df[series]).cumprod()
                benchmark_growth = (1 + aligned_df[benchmark]).cumprod()

                # Relative growth (series vs benchmark)
                growth = series_growth / benchmark_growth
            else:
                # For total returns, use series returns directly
                returns = df[series].dropna()
                growth = (1 + returns).cumprod()

            # Prepend starting value of 1.0 to properly calculate drawdown from initial capital
            # This ensures that a negative first period return counts as a drawdown
            growth_array = np.concatenate([[1.0], growth.values])
            running_max_array = np.maximum.accumulate(growth_array)

            # Calculate drawdown (exclude the prepended 1.0)
            drawdown_array = (growth_array[1:] / running_max_array[1:]) - 1
            drawdown = pd.Series(drawdown_array, index=growth.index)

            # Prepend starting value of 0.0 drawdown at one period before first date
            if len(drawdown) > 0:
                first_date = drawdown.index[0]
                start_date = first_date - period_offset
                start_value = pd.Series([0.0], index=[start_date])
                drawdown = pd.concat([start_value, drawdown])

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
    prevent_initial_call=True,
)
def update_drawdown_grid(active_tab, chart_checked, raw_data, periodicity, selected_series, returns_type, benchmark_assignments, long_short_assignments, date_range):
    """Update Drawdown grid (lazy loaded)."""
    # Lazy loading: only generate when drawdown tab is active and table view is selected
    if active_tab != "drawdown" or chart_checked != "table":
        return [], []

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
            str(date_range)
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
    prevent_initial_call=True,
)
def download_excel(n_clicks, raw_data, original_periodicity, selected_periodicity, selected_series, returns_type, benchmark_assignments, long_short_assignments, date_range, rolling_window, rolling_return_type, monthly_view, monthly_series):
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
        str(date_range)
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
        str(date_range)
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
                return_type
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
                        date_range
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
                        date_range
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
                str(date_range)
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
                str(date_range)
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
