"""Dashboard page - Market Returns Time Series Dashboard."""

from io import BytesIO, StringIO

import dash_ag_grid as dag
import dash_mantine_components as dmc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Input, Output, State, callback, dcc, html, no_update, register_page, ALL, clientside_callback
from dash.exceptions import PreventUpdate

from utils.parsing import detect_periodicity, parse_uploaded_file
from utils.returns import (
    get_available_periodicities,
    merge_returns,
    resample_returns,
)
from utils.statistics import calculate_all_statistics

register_page(__name__, path="/dashboard", name="Dashboard", title="DashMat - Dashboard")

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


def json_to_df(json_str: str) -> pd.DataFrame:
    """Convert JSON string back to DataFrame."""
    df = pd.read_json(StringIO(json_str), orient="split")
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df


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
                                        dmc.MenuItem("(No actions available)", disabled=True),
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
                dmc.TabsPanel(
                    value="statistics",
                    pt="md",
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
                dmc.TabsPanel(
                    value="correlogram",
                    pt="md",
                    children=[
                        dcc.Graph(
                            id="correlogram-graph",
                            style={"height": "700px"},
                        ),
                    ],
                ),
            ],
        ),
        # Hidden stores for state management (using local storage for persistence)
        dcc.Store(id="raw-data-store", data=None, storage_type="local"),
        dcc.Store(id="original-periodicity-store", data="daily", storage_type="local"),
        dcc.Store(id="benchmark-assignments-store", data={}, storage_type="local"),
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
        default_periodicity = "daily" if combined_periodicity == "daily" else "monthly"

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
)
def update_benchmark_selectors(selected_series, raw_data, current_assignments):
    """Create benchmark dropdown for each selected series."""
    if not selected_series or raw_data is None:
        return dmc.Text("Select series to assign benchmarks", size="sm", c="dimmed")

    df = json_to_df(raw_data)
    all_series = list(df.columns)
    default_benchmark = all_series[0] if all_series else None

    # Create a dropdown for each selected series
    benchmark_selectors = []
    for series in selected_series:
        current_benchmark = current_assignments.get(series, default_benchmark) if current_assignments else default_benchmark
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
    Output("returns-grid", "columnDefs"),
    Output("returns-grid", "rowData"),
    Output("menu-download-excel", "disabled"),
    Input("raw-data-store", "data"),
    Input("periodicity-select", "value"),
    Input("series-select", "value"),
    Input("returns-type-select", "value"),
    Input("benchmark-assignments-store", "data"),
    prevent_initial_call=True,
)
def update_grid(raw_data, periodicity, selected_series, returns_type, benchmark_assignments):
    """Update the AG Grid based on selections."""
    if raw_data is None or not selected_series:
        return [], [], True

    try:
        df = json_to_df(raw_data)

        # Resample if needed
        if periodicity and periodicity != "daily":
            df = resample_returns(df, periodicity)

        # Filter to selected series only
        available_series = [s for s in selected_series if s in df.columns]
        if not available_series:
            return [], [], True

        display_df = df[available_series].copy()

        # Calculate excess returns if requested
        if returns_type == "excess":
            for series in available_series:
                benchmark = benchmark_assignments.get(series, available_series[0]) if benchmark_assignments else available_series[0]
                if benchmark == series:
                    # Excess return vs itself is 0
                    display_df[series] = 0.0
                elif benchmark in df.columns:
                    display_df[series] = df[series] - df[benchmark]

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


@callback(
    Output("statistics-grid", "columnDefs"),
    Output("statistics-grid", "rowData"),
    Input("raw-data-store", "data"),
    Input("periodicity-select", "value"),
    Input("series-select", "value"),
    Input("benchmark-assignments-store", "data"),
    prevent_initial_call=True,
)
def update_statistics(raw_data, periodicity, selected_series, benchmark_assignments):
    """Update the Statistics grid with transposed data (series as columns)."""
    if raw_data is None or not selected_series:
        return [], []

    try:
        df = json_to_df(raw_data)

        # Resample if needed
        if periodicity and periodicity != "daily":
            df = resample_returns(df, periodicity)

        # Calculate statistics for all selected series
        stats = calculate_all_statistics(
            df,
            selected_series,
            benchmark_assignments,
            periodicity or "daily",
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


@callback(
    Output("correlogram-graph", "figure"),
    Input("raw-data-store", "data"),
    Input("periodicity-select", "value"),
    Input("series-select", "value"),
    Input("returns-type-select", "value"),
    Input("benchmark-assignments-store", "data"),
    prevent_initial_call=True,
)
def update_correlogram(raw_data, periodicity, selected_series, returns_type, benchmark_assignments):
    """Update the Correlogram with custom pairs plot."""
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

    if raw_data is None or not selected_series or len(selected_series) < 2:
        return empty_fig

    try:
        df = json_to_df(raw_data)

        # Resample if needed
        if periodicity and periodicity != "daily":
            df = resample_returns(df, periodicity)

        # Filter to selected series only
        available_series = [s for s in selected_series if s in df.columns]
        n = len(available_series)
        if n < 2:
            return empty_fig

        display_df = df[available_series].copy()

        # Calculate excess returns if requested
        if returns_type == "excess":
            for series in available_series:
                benchmark = benchmark_assignments.get(series, available_series[0]) if benchmark_assignments else available_series[0]
                if benchmark == series:
                    display_df[series] = 0.0
                elif benchmark in df.columns:
                    display_df[series] = df[series] - df[benchmark]

        # Calculate correlation matrix
        corr_matrix = display_df.corr()

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
                        ),
                        row=row_idx, col=col_idx
                    )
                elif i > j:
                    # Lower triangle: scatter plot
                    fig.add_trace(
                        go.Scatter(
                            x=display_df[col_series],
                            y=display_df[row_series],
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
    prevent_initial_call=True,
)
def download_excel(n_clicks, raw_data, periodicity, selected_series, returns_type, benchmark_assignments):
    """Generate Excel file for download."""
    if n_clicks is None or raw_data is None or not selected_series:
        raise PreventUpdate

    df = json_to_df(raw_data)

    # Resample if needed
    if periodicity and periodicity != "daily":
        df = resample_returns(df, periodicity)

    # Filter to selected series
    available_series = [s for s in selected_series if s in df.columns]
    display_df = df[available_series].copy()

    # Calculate excess returns if requested
    if returns_type == "excess":
        for series in available_series:
            benchmark = benchmark_assignments.get(series, available_series[0]) if benchmark_assignments else available_series[0]
            if benchmark == series:
                # Excess return vs itself is 0
                display_df[series] = 0.0
            elif benchmark in df.columns:
                display_df[series] = df[series] - df[benchmark]

    # Create Excel file in memory
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        display_df.to_excel(writer, sheet_name="Returns")
    output.seek(0)

    # Generate filename
    periodicity_suffix = periodicity.replace("_", "-") if periodicity else "returns"
    returns_suffix = "excess" if returns_type == "excess" else "total"
    filename = f"returns_{periodicity_suffix}_{returns_suffix}.xlsx"

    return dcc.send_bytes(output.getvalue(), filename)
