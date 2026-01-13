"""Analytics tool page - Market Returns Time Series Dashboard."""

from io import BytesIO, StringIO
import hashlib

import dash_ag_grid as dag
import dash_mantine_components as dmc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Input, Output, State, callback, dcc, html, no_update, register_page, ALL, clientside_callback
from dash.exceptions import PreventUpdate

import cache_config
from utils.parsing import detect_periodicity, parse_uploaded_file
from utils.returns import (
    get_available_periodicities,
    merge_returns,
    resample_returns,
)
from utils.statistics import calculate_all_statistics

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
    ("Hit Rate", ".1%"),
    ("Hit Rate (vs Benchmark)", ".1%"),
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


def _hash_json(json_str: str) -> str:
    """Create a hash of JSON string for cache key."""
    return hashlib.md5(json_str.encode()).hexdigest()


@cache_config.cache.memoize(timeout=300)
def json_to_df_cached(json_str: str) -> pd.DataFrame:
    """Convert JSON string back to DataFrame with caching.

    This is the primary performance bottleneck - caching this operation
    prevents repeated deserialization of the same data.
    """
    df = pd.read_json(StringIO(json_str), orient="split")
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df


def json_to_df(json_str: str) -> pd.DataFrame:
    """Convert JSON string back to DataFrame (cached wrapper)."""
    return json_to_df_cached(json_str)


@cache_config.cache.memoize(timeout=300)
def resample_returns_cached(json_str: str, periodicity: str) -> pd.DataFrame:
    """Resample returns with caching to avoid repeated computation."""
    df = json_to_df(json_str)
    if periodicity == "daily":
        return df
    return resample_returns(df, periodicity)


@cache_config.cache.memoize(timeout=300)
def calculate_excess_returns(json_str: str, periodicity: str, selected_series: tuple,
                             benchmark_assignments: str, returns_type: str, long_short_assignments: str) -> pd.DataFrame:
    """Calculate excess returns with caching."""
    df = resample_returns_cached(json_str, periodicity)

    # Filter to selected series only
    available_series = [s for s in selected_series if s in df.columns]
    if not available_series:
        return pd.DataFrame()

    display_df = df[available_series].copy()

    # Parse assignments
    benchmark_dict = eval(benchmark_assignments) if benchmark_assignments else {}
    long_short_dict = eval(long_short_assignments) if long_short_assignments else {}

    # Calculate long-short returns for series with long-short enabled
    for series in available_series:
        is_long_short = long_short_dict.get(series, False)
        if is_long_short:
            benchmark = benchmark_dict.get(series, available_series[0])
            if benchmark == series:
                display_df[series] = 0.0
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
                    display_df[series] = 0.0
                elif benchmark in df.columns:
                    # Vectorized operation instead of per-series assignment
                    display_df.loc[:, series] = df[series] - df[benchmark]

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
                                        dmc.Select(
                                            id="returns-type-select",
                                            label="Returns Type",
                                            data=[
                                                {"value": "total", "label": "Total Returns"},
                                                {"value": "excess", "label": "Excess Returns"},
                                            ],
                                            value="total",
                                            w=200,
                                        ),
                                    ],
                                ),
                                dmc.Divider(mb="md"),
                                dmc.MultiSelect(
                                    id="series-select",
                                    label="Select series to include in analysis",
                                    data=[],
                                    value=[],
                                    placeholder="Upload data to select series",
                                    searchable=True,
                                    clearable=True,
                                    mb="md",
                                ),
                                dmc.Text("Benchmark Assignment", size="sm", c="dimmed", mb="xs"),
                                html.Div(id="benchmark-assignment-container"),
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
            value="returns",
            children=[
                dmc.TabsList(
                    children=[
                        dmc.TabsTab("Returns", value="returns"),
                        dmc.TabsTab("Statistics", value="statistics"),
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


@callback(
    Output("raw-data-store", "data", allow_duplicate=True),
    Output("original-periodicity-store", "data", allow_duplicate=True),
    Output("benchmark-assignments-store", "data", allow_duplicate=True),
    Output("long-short-store", "data", allow_duplicate=True),
    Output("periodicity-value-store", "data", allow_duplicate=True),
    Output("returns-type-value-store", "data", allow_duplicate=True),
    Output("series-select-value-store", "data", allow_duplicate=True),
    Input("menu-clear-all-series", "n_clicks"),
    prevent_initial_call=True,
)
def clear_all_series(n_clicks):
    """Clear all loaded series and reset application state."""
    if n_clicks is None:
        raise PreventUpdate

    # Reset all stores to initial state
    return None, "daily", {}, {}, None, None, []


@callback(
    Output("periodicity-select", "data", allow_duplicate=True),
    Output("periodicity-select", "value", allow_duplicate=True),
    Output("periodicity-select", "disabled", allow_duplicate=True),
    Output("series-select", "data", allow_duplicate=True),
    Output("series-select", "value", allow_duplicate=True),
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
        return [], "daily", True, [], [], "total"

    try:
        df = json_to_df(raw_data)
        all_series = [{"value": col, "label": col} for col in df.columns]

        # Get available periodicities
        from utils.returns import get_available_periodicities
        periodicity_options = get_available_periodicities(original_periodicity or "daily")

        # Validate stored values
        valid_periodicity = stored_periodicity if stored_periodicity in [p["value"] for p in periodicity_options] else (original_periodicity or "daily")
        valid_series = [s for s in (stored_series or []) if s in df.columns]
        valid_returns_type = stored_returns_type if stored_returns_type in ["total", "excess"] else "total"

        return periodicity_options, valid_periodicity, False, all_series, valid_series, valid_returns_type

    except Exception:
        return [], "daily", True, [], [], "total"


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
    Input("series-select", "value"),
    prevent_initial_call=True,
)
def save_series_selection(value):
    """Save series selection to local storage."""
    return value or []


@callback(
    Output("raw-data-store", "data"),
    Output("original-periodicity-store", "data"),
    Output("periodicity-select", "data"),
    Output("periodicity-select", "value"),
    Output("periodicity-select", "disabled"),
    Output("series-select", "data"),
    Output("series-select", "value"),
    Output("alert-message", "children"),
    Output("alert-message", "color"),
    Output("alert-message", "hide"),
    Output("periodicity-value-store", "data", allow_duplicate=True),
    Output("series-select-value-store", "data", allow_duplicate=True),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    State("raw-data-store", "data"),
    State("original-periodicity-store", "data"),
    State("series-select", "value"),
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
                    no_update, no_update,
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

        # Update series selection options
        all_series = [{"value": col, "label": col} for col in merged_df.columns]

        # Keep current selection and add new series
        new_series = [col for col in new_df.columns if col not in (current_selection or [])]
        updated_selection = (current_selection or []) + new_series

        return (
            df_to_json(merged_df),
            combined_periodicity,
            periodicity_options,
            default_periodicity,
            False,
            all_series,
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
            no_update, no_update,
            f"Error loading file: {str(e)}",
            "red",
            False,
            no_update, no_update,
        )


@callback(
    Output("benchmark-assignment-container", "children"),
    Input("series-select", "value"),
    State("raw-data-store", "data"),
    State("benchmark-assignments-store", "data"),
    State("long-short-store", "data"),
)
def update_benchmark_selectors(selected_series, raw_data, current_assignments, long_short_assignments):
    """Create benchmark dropdown for each selected series."""
    if not selected_series or raw_data is None:
        return dmc.Text("Select series to assign benchmarks", size="sm", c="dimmed")

    df = json_to_df(raw_data)
    all_series = list(df.columns)
    default_benchmark = all_series[0] if all_series else None

    # Create a dropdown and checkbox for each selected series
    benchmark_selectors = []
    for series in selected_series:
        current_benchmark = current_assignments.get(series, default_benchmark) if current_assignments else default_benchmark
        is_long_short = long_short_assignments.get(series, False) if long_short_assignments else False
        benchmark_selectors.append(
            dmc.Group(
                mb="xs",
                children=[
                    dmc.Text(series, size="sm", w=150, style={"fontFamily": "monospace"}),
                    dmc.Select(
                        id={"type": "benchmark-select", "series": series},
                        data=[{"value": s, "label": s} for s in all_series],
                        value=current_benchmark if current_benchmark in all_series else default_benchmark,
                        size="xs",
                        w=200,
                        placeholder="Select benchmark",
                    ),
                    dmc.Checkbox(
                        id={"type": "long-short-checkbox", "series": series},
                        label="Long-Short",
                        checked=is_long_short,
                        size="xs",
                    ),
                ],
            )
        )

    return benchmark_selectors


@callback(
    Output("benchmark-assignments-store", "data"),
    Input({"type": "benchmark-select", "series": ALL}, "value"),
    State("series-select", "value"),
    prevent_initial_call=True,
)
def update_benchmark_assignments(benchmark_values, selected_series):
    """Store benchmark assignments."""
    if not selected_series or not benchmark_values:
        return {}

    assignments = {}
    for i, series in enumerate(selected_series):
        if i < len(benchmark_values) and benchmark_values[i]:
            assignments[series] = benchmark_values[i]

    return assignments


@callback(
    Output("long-short-store", "data"),
    Input({"type": "long-short-checkbox", "series": ALL}, "checked"),
    State("series-select", "value"),
    prevent_initial_call=True,
)
def update_long_short_assignments(checkbox_values, selected_series):
    """Store long-short checkbox assignments."""
    if not selected_series or checkbox_values is None:
        return {}

    assignments = {}
    for i, series in enumerate(selected_series):
        if i < len(checkbox_values):
            assignments[series] = checkbox_values[i] or False

    return assignments


@callback(
    Output("returns-grid", "columnDefs"),
    Output("returns-grid", "rowData"),
    Output("menu-download-excel", "disabled"),
    Input("raw-data-store", "data"),
    Input("periodicity-select", "value"),
    Input("series-select", "value"),
    Input("returns-type-select", "value"),
    Input("benchmark-assignments-store", "data"),
    Input("long-short-store", "data"),
    prevent_initial_call=True,
)
def update_grid(raw_data, periodicity, selected_series, returns_type, benchmark_assignments, long_short_assignments):
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
            str(long_short_assignments)  # Convert to string for cache key
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
                "valueFormatter": {"function": "params.value != null ? d3.format('.4%')(params.value) : ''"},
                "width": 120,
            })

        # Convert to row data
        df_reset = display_df.reset_index()
        df_reset["Date"] = df_reset["Date"].dt.strftime("%Y-%m-%d")
        row_data = df_reset.to_dict("records")

        return column_defs, row_data, False

    except Exception:
        return [], [], True


@cache_config.cache.memoize(timeout=300)
def calculate_statistics_cached(json_str: str, periodicity: str, selected_series: tuple,
                                benchmark_assignments: str, long_short_assignments: str) -> list:
    """Calculate statistics with caching."""
    df = resample_returns_cached(json_str, periodicity)
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
    Output("statistics-grid", "columnDefs"),
    Output("statistics-grid", "rowData"),
    Input("raw-data-store", "data"),
    Input("periodicity-select", "value"),
    Input("series-select", "value"),
    Input("benchmark-assignments-store", "data"),
    Input("long-short-store", "data"),
    prevent_initial_call=True,
)
def update_statistics(raw_data, periodicity, selected_series, benchmark_assignments, long_short_assignments):
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
            str(long_short_assignments)
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
                # Dynamic formatting based on row
                "valueFormatter": {
                    "function": """
                        const fmt = params.data._format;
                        if (!fmt || params.value == null) return params.value;
                        if (fmt.includes('%')) return d3.format(fmt)(params.value);
                        return d3.format(fmt)(params.value);
                    """
                },
            })

        # Build transposed rows
        row_data = []
        for stat_name, fmt in STATS_CONFIG:
            row = {"Statistic": stat_name, "_format": fmt}
            for series_stats in stats:
                series_name = series_stats["Series"]
                row[series_name] = series_stats.get(stat_name)
            row_data.append(row)

        return column_defs, row_data

    except Exception:
        return [], []


@cache_config.cache.memoize(timeout=300)
def generate_correlogram_cached(json_str: str, periodicity: str, selected_series: tuple,
                                returns_type: str, benchmark_assignments: str, long_short_assignments: str):
    """Generate correlogram with caching."""
    display_df = calculate_excess_returns(
        json_str, periodicity, selected_series, benchmark_assignments, returns_type, long_short_assignments
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
    Input("series-select", "value"),
    Input("returns-type-select", "value"),
    Input("benchmark-assignments-store", "data"),
    Input("long-short-store", "data"),
    prevent_initial_call=True,
)
def update_correlogram(active_tab, raw_data, periodicity, selected_series, returns_type, benchmark_assignments, long_short_assignments):
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

    # Size limit: warn if too many series
    if len(selected_series) > MAX_SCATTER_MATRIX_SIZE:
        warning_fig = go.Figure()
        warning_fig.add_annotation(
            text=f"Too many series selected ({len(selected_series)}). Please select {MAX_SCATTER_MATRIX_SIZE} or fewer series to view the scatter matrix.<br>Large scatter matrices can cause performance issues.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#fa5252"),
            align="center",
        )
        warning_fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )
        return warning_fig

    try:
        # Use cached function to avoid repeated computation
        result = generate_correlogram_cached(
            raw_data,
            periodicity or "daily",
            tuple(selected_series),
            returns_type,
            str(benchmark_assignments),
            str(long_short_assignments)
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
    Output("download-excel", "data"),
    Input("menu-download-excel", "n_clicks"),
    State("raw-data-store", "data"),
    State("periodicity-select", "value"),
    State("series-select", "value"),
    State("returns-type-select", "value"),
    State("benchmark-assignments-store", "data"),
    State("long-short-store", "data"),
    prevent_initial_call=True,
)
def download_excel(n_clicks, raw_data, periodicity, selected_series, returns_type, benchmark_assignments, long_short_assignments):
    """Generate Excel file with Returns, Statistics, and Correlogram sheets (optimized)."""
    if n_clicks is None or raw_data is None or not selected_series:
        raise PreventUpdate

    # Use cached functions to get data
    returns_df = calculate_excess_returns(
        raw_data,
        periodicity or "daily",
        tuple(selected_series),
        str(benchmark_assignments),
        returns_type,
        str(long_short_assignments)
    )

    if returns_df.empty:
        raise PreventUpdate

    # Get cached statistics
    stats = calculate_statistics_cached(
        raw_data,
        periodicity or "daily",
        tuple(selected_series),
        str(benchmark_assignments),
        str(long_short_assignments)
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
        # Sheet 1: Returns
        returns_df.to_excel(writer, sheet_name="Returns")

        # Sheet 2: Statistics
        stats_df.to_excel(writer, sheet_name="Statistics", index=False)

        # Sheet 3: Correlogram
        corr_df.to_excel(writer, sheet_name="Correlogram")

    output.seek(0)

    # Generate filename
    periodicity_suffix = periodicity.replace("_", "-") if periodicity else "returns"
    returns_suffix = "excess" if returns_type == "excess" else "total"
    filename = f"dashmat_{periodicity_suffix}_{returns_suffix}.xlsx"

    return dcc.send_bytes(output.getvalue(), filename)
