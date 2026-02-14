"""Analytics tool page - Market Returns Time Series Dashboard."""

from dataclasses import dataclass
from io import BytesIO, StringIO
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
from utils.parsing import detect_periodicity, get_sheet_names, parse_uploaded_file
from utils.returns import (
    align_monthly_index_to_month_end,
    calculate_calendar_year_returns,
    calculate_excess_returns,
    calculate_rolling_returns,
    create_monthly_view,
    df_to_json,
    get_available_periodicities,
    get_working_returns,
    json_to_df,
    merge_returns,
    resample_returns,
    resample_returns_cached,
)
from utils.sample_data import get_sample_file_path
from utils.statistics import (
    calculate_drawdown,
    calculate_growth_of_dollar,
    calculate_statistics_cached,
    generate_correlogram_cached,
)
from utils.charting import apply_chart_theme
from utils.perf_timing import timed_block
from utils.serialization import date_range_payload_for_cache, mapping_payload_for_cache
from utils.shared_metrics import (
    MARKET_BETA_SERIES,
    RISK_FREE_SERIES,
    STATS_CONFIG,
    risk_free_json_from_store as _risk_free_json_from_store,
    spx_json_from_store as _spx_json_from_store,
)
from dbengine import AG_GRID_LICENSE_KEY, engine as DB_ENGINE, engine_MRD as MRD_ENGINE
from utils.core_categories import (
    clear_dropdown_caches,
    get_common_daily_range,
    get_core_category_options_cached,
    load_cma_returns_for_benches,
    load_cma_returns_for_benches_with_meta,
)

register_page(__name__, path="/analyticstool", name="Analytics Tool", title="Analytics Tool")

# Performance optimization constants

SAVED_SERIES_CONFIG = {
    RISK_FREE_SERIES: {},
    MARKET_BETA_SERIES: {"start_date": "1988-01-04"},
}


def _mapping_payload(value) -> str:
    return mapping_payload_for_cache(value)


def _date_range_payload(value) -> str:
    return date_range_payload_for_cache(value)


@dataclass(frozen=True)
class _AnalyticsComputeBundle:
    raw_data: str
    periodicity: str
    selected_series: tuple
    benchmark_payload: str
    long_short_payload: str
    date_range_payload: str
    vol_scaler: float
    vol_scaling_payload: str


def _build_analytics_compute_bundle(
    raw_data,
    periodicity,
    selected_series,
    benchmark_assignments,
    long_short_assignments,
    date_range,
    vol_scaler,
    vol_scaling_assignments,
) -> _AnalyticsComputeBundle:
    """Build canonicalized compute inputs once per callback."""
    return _AnalyticsComputeBundle(
        raw_data=raw_data,
        periodicity=periodicity or "daily",
        selected_series=tuple(selected_series or ()),
        benchmark_payload=_mapping_payload(benchmark_assignments),
        long_short_payload=_mapping_payload(long_short_assignments),
        date_range_payload=_date_range_payload(date_range),
        vol_scaler=vol_scaler or 0,
        vol_scaling_payload=_mapping_payload(vol_scaling_assignments),
    )


def _normalize_monthly_df_if_needed(df: pd.DataFrame, periodicity: str) -> pd.DataFrame:
    """Canonicalize monthly indexes only when the workflow is monthly."""
    if periodicity == "monthly":
        return align_monthly_index_to_month_end(df)
    return df


def build_welcome_screen():
    return dmc.Stack(
        align="center",
        justify="center",
        h=400,
        children=[
            DashIconify(icon="fluent-mdl2:chart", width=60, color="#adb5bd"),
            dmc.Text("Welcome to the Analytics Tool", size="xl", fw=500, c="dimmed", mt="md"),
            dmc.Text("Add a data series to begin", size="sm", c="dimmed"),
            dmc.Group(
                gap="sm",
                mt="lg",
                children=[
                    dmc.Button(
                        "Add from database",
                        leftSection=DashIconify(icon="tabler:database"),
                        variant="outline",
                        size="sm",
                        w=210,
                        id="welcome-add-db-btn",
                    ),
                    dmc.Button(
                        "Add series from file",
                        leftSection=DashIconify(icon="tabler:upload"),
                        variant="outline",
                        size="sm",
                        w=210,
                        id="welcome-add-series-btn",
                    ),
                ],
            ),
            dmc.Group(
                gap="md",
                mt="sm",
                children=[
                    dmc.Button(
                        "Sample Daily File",
                        leftSection=DashIconify(icon="tabler:download"),
                        id="download-sample-daily-btn",
                        size="sm",
                        variant="light",
                        w=210,
                    ),
                    dmc.Button(
                        "Sample Monthly File",
                        leftSection=DashIconify(icon="tabler:download"),
                        id="download-sample-monthly-btn",
                        size="sm",
                        variant="light",
                        w=210,
                    ),
                ],
            ),
        ]
    )

# Clientside callback to trigger upload from welcome button
clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks) {
            // Trigger the file input click with a small delay to allow overlay to render
            setTimeout(function() {
                var uploadDiv = document.getElementById('upload-data');
                if (uploadDiv) {
                    var input = uploadDiv.querySelector('input[type="file"]');
                    if (input) {
                        // Listen for window focus to detect cancel
                        var onFocus = function() {
                            window.removeEventListener('focus', onFocus);
                            setTimeout(function() {
                                if (!input.files || input.files.length === 0) {
                                    // User cancelled - hide the blocker
                                    var store = document.getElementById('ui-blocker-store');
                                    if (store && store._dashprivate_setValue) {
                                        store._dashprivate_setValue(false);
                                    }
                                    window.dash_clientside.set_props('ui-blocker-store', {data: false});
                                    window.dash_clientside.set_props('ui-blocker-timeout', {disabled: true});
                                }
                            }, 500);
                        };
                        window.addEventListener('focus', onFocus);
                        input.click();
                    }
                }
            }, 100);
            
            // Show Blocker (True), Enable Timeout (False)
            return [true, false];
        }
        return [window.dash_clientside.no_update, window.dash_clientside.no_update];
    }
    """,
    Output("ui-blocker-store", "data", allow_duplicate=True),
    Output("ui-blocker-timeout", "disabled", allow_duplicate=True),
    Input("welcome-add-series-btn", "n_clicks"),
    prevent_initial_call=True,
)

clientside_callback(
    "function(n) { return true; }",
    Output("help-modal", "opened"),
    Input("menu-help-guide", "n_clicks"),
    prevent_initial_call=True,
)

# Save session: download all sessionStorage as JSON
clientside_callback(
    """
    function(n_clicks) {
        if (!n_clicks) return window.dash_clientside.no_update;
        var data = {};
        for (var i = 0; i < sessionStorage.length; i++) {
            var key = sessionStorage.key(i);
            data[key] = sessionStorage.getItem(key);
        }
        var blob = new Blob([JSON.stringify(data)], {type: 'application/json'});
        var a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'dashmat_session.json';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        return window.dash_clientside.no_update;
    }
    """,
    Output("save-session-dummy", "data"),
    Input("menu-save-session", "n_clicks"),
    prevent_initial_call=True,
)

# Load session: trigger hidden upload file dialog
clientside_callback(
    """
    function(n_clicks) {
        if (!n_clicks) return window.dash_clientside.no_update;
        setTimeout(function() {
            var el = document.querySelector('#load-session-upload input[type="file"]');
            if (el) el.click();
        }, 100);
        return window.dash_clientside.no_update;
    }
    """,
    Output("load-session-dummy", "data"),
    Input("menu-load-session", "n_clicks"),
    prevent_initial_call=True,
)

# Load session: restore sessionStorage from uploaded file and reload
clientside_callback(
    """
    function(contents) {
        if (!contents) return window.dash_clientside.no_update;
        var raw = atob(contents.split(',')[1]);
        var data = JSON.parse(raw);
        sessionStorage.clear();
        for (var key in data) {
            sessionStorage.setItem(key, data[key]);
        }
        window.location.reload();
        return window.dash_clientside.no_update;
    }
    """,
    Output("load-session-dummy", "data", allow_duplicate=True),    Input("load-session-upload", "contents"),
    prevent_initial_call=True,
)


@callback(
    Output("menu-save-session", "disabled"),
    Input("welcome-screen-container", "style"),
)
def at_toggle_save_session(welcome_style):
    if not welcome_style:
        return True
    return welcome_style.get("display") != "none"


@callback(
    Output("db-add-modal", "opened", allow_duplicate=True),
    Output("db-add-series-select", "data", allow_duplicate=True),
    Output("db-add-series-select", "value", allow_duplicate=True),
    Input("menu-add-from-db", "n_clicks"),
    Input("welcome-add-db-btn", "n_clicks"),
    prevent_initial_call=True,
)
def open_db_add_modal(menu_clicks, welcome_clicks):
    if not menu_clicks and not welcome_clicks:
        raise PreventUpdate
    options = get_core_category_options_cached(DB_ENGINE)
    return True, options, []


@callback(
    Output("db-add-modal", "opened", allow_duplicate=True),
    Output("db-add-series-select", "value", allow_duplicate=True),
    Input("db-add-cancel-button", "n_clicks"),
    prevent_initial_call=True,
)
def close_db_add_modal(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    return False, []


@callback(
    Output("analyticstool-saved-series-cache-store", "data"),
    Input("analyticstool-raw-data-store", "data"),
    State("analyticstool-saved-series-cache-store", "data"),
)
def refresh_saved_series_cache(raw_data, cache_data):
    """Cache shared saved benchmark series and refresh if raw data extends beyond them."""
    if not raw_data:
        raise PreventUpdate

    try:
        raw_df = json_to_df(raw_data)
    except Exception:
        raise PreventUpdate

    if raw_df.empty:
        raise PreventUpdate

    raw_end = pd.to_datetime(raw_df.index.max())

    cache_is_fresh = isinstance(cache_data, dict) and isinstance(cache_data.get("series_data"), dict)
    if cache_is_fresh:
        for series_name in SAVED_SERIES_CONFIG:
            series_payload = cache_data["series_data"].get(series_name)
            if not isinstance(series_payload, dict):
                cache_is_fresh = False
                break
            payload_json = series_payload.get("returns_json")
            payload_max_raw = series_payload.get("max_date")
            payload_max = pd.to_datetime(payload_max_raw, errors="coerce")
            if not isinstance(payload_json, str) or pd.isna(payload_max) or raw_end > payload_max:
                cache_is_fresh = False
                break

    if cache_is_fresh:
        raise PreventUpdate

    try:
        saved_df = load_cma_returns_for_benches(
            DB_ENGINE,
            list(SAVED_SERIES_CONFIG.keys()),
            MRD_ENGINE,
        )
    except Exception:
        raise PreventUpdate

    if saved_df.empty:
        raise PreventUpdate

    saved_df = saved_df.sort_index()
    series_data = {}
    for series_name, config in SAVED_SERIES_CONFIG.items():
        if series_name not in saved_df.columns:
            continue

        series_returns = saved_df[series_name].dropna().sort_index()
        start_date = config.get("start_date")
        if start_date:
            series_returns = series_returns.loc[
                series_returns.index >= pd.Timestamp(start_date)
            ]
        if series_returns.empty:
            continue

        series_max = pd.to_datetime(series_returns.index.max())
        series_data[series_name] = {
            "max_date": series_max.strftime("%Y-%m-%d"),
            "returns_json": df_to_json(series_returns.to_frame(series_name)),
        }

    if not series_data:
        raise PreventUpdate

    return {"series_data": series_data}


@callback(
    Output("db-add-error-alert", "children"),
    Output("db-add-error-alert", "hide"),
    Output("db-add-ok-button", "disabled"),
    Input("db-add-series-select", "value"),
    Input("analyticstool-raw-data-store", "data"),
    Input("db-add-modal", "opened"),
    prevent_initial_call=True,
)
def validate_db_add_selection(selected_benches, raw_data, opened):
    if not opened:
        raise PreventUpdate

    if not selected_benches:
        return no_update, True, True

    existing_cols = set()
    if raw_data:
        try:
            existing_cols = set(json_to_df(raw_data).columns)
        except Exception:
            existing_cols = set()

    duplicates = [s for s in selected_benches if s in existing_cols]
    if duplicates:
        return f"Cannot add duplicate series: {', '.join(duplicates)}", False, True
    return no_update, True, False


def build_main_layout(periodicity_options, periodicity_value, returns_type, vol_scaler,
                      active_tab, rolling_window, rolling_metric, rolling_return_type, rolling_chart_switch,
                      drawdown_chart_switch, growth_chart_switch, monthly_view, monthly_series,
                      monthly_series_options, monthly_select_disabled):
    
    # Calculate visibility styles - use flex for full height
    flex_style = {"display": "flex", "flexDirection": "column", "flex": "1", "overflow": "hidden"}
    flex_scroll_style = {"display": "flex", "flexDirection": "column", "flex": "1", "overflow": "auto"}
    none_style = {"display": "none"}
    
    rolling_grid_style = flex_style if rolling_chart_switch == "table" else none_style
    rolling_chart_style = flex_style if rolling_chart_switch == "chart" else none_style
    
    drawdown_grid_style = flex_style if drawdown_chart_switch == "table" else none_style
    drawdown_chart_style = flex_scroll_style if drawdown_chart_switch == "chart" else none_style
    
    growth_grid_style = flex_style if growth_chart_switch == "table" else none_style
    growth_chart_style = flex_scroll_style if growth_chart_switch == "chart" else none_style

    rolling_return_type_disabled = False if rolling_metric in ["total_return", "excess_return"] else True
    rolling_return_type_style = {} if not rolling_return_type_disabled else {"opacity": 0.5, "pointerEvents": "none"}

    return html.Div(
        style={"display": "flex", "flexDirection": "column", "height": "100%", "overflow": "hidden"},
        children=[
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
                                                    "Common Daily",
                                                    id="common-daily-button",
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
            style={"flex": "1", "display": "flex", "flexDirection": "column", "overflow": "hidden"},
            children=[
                dmc.TabsList(
                    children=[
                        dmc.TabsTab("Statistics", value="statistics"),
                        dmc.TabsTab("Returns", value="returns"),
                        dmc.TabsTab("Rolling", value="rolling"),
                        dmc.TabsTab("Calendar Year", value="calendar"),
                        dmc.TabsTab("Growth of $1", value="growth"),
                        dmc.TabsTab("Drawdown", value="drawdown"),
                        dmc.TabsTab("Correlation", value="correlogram"),
                    ],
                ),
                dmc.TabsPanel(
                    value="returns",
                    pt="md",
                    style={"flex": "1", "display": "flex", "flexDirection": "column", "overflow": "hidden"},
                    children=[
                        dcc.Loading(
                            id="loading-returns",
                            type="default",
                            style={"flex": "1", "display": "flex", "flexDirection": "column", "overflow": "hidden"},
                            parent_style={"flex": "1", "display": "flex", "flexDirection": "column", "overflow": "hidden"},
                            children=[
                                dag.AgGrid(
                                    enableEnterpriseModules=True,
                                    licenseKey=AG_GRID_LICENSE_KEY,
                                    id="returns-grid",
                                    className='ag-theme-alpine',
                                    columnDefs=[],
                                    rowData=[],
                                    defaultColDef={
                                        "sortable": True,
                                        "resizable": True,
                                        "suppressHeaderMenuButton": True,
                                        "cellStyle": {"textAlign": "center"},
                                        "headerClass": "dashmat-center-header",
                                    },
                                    style={"height": "100%", "width": "100%"},
                                    dashGridOptions={
                                        "animateRows": True,
                                        "pagination": False,
                                        "suppressExcelExport": True,
                                        "enableRangeSelection": True,
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
                dmc.TabsPanel(
                    value="rolling",
                    pt="md",
                    style={"flex": "1", "display": "flex", "flexDirection": "column", "overflow": "hidden"},
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
                                    enableEnterpriseModules=True,
                                    licenseKey=AG_GRID_LICENSE_KEY,
                                    id="rolling-grid",
                                    className='ag-theme-alpine',
                                    columnDefs=[],
                                    rowData=[],
                                    defaultColDef={
                                        "sortable": True,
                                        "resizable": True,
                                        "suppressHeaderMenuButton": True,
                                        "cellStyle": {"textAlign": "center"},
                                        "headerClass": "dashmat-center-header",
                                    },
                                    style={"height": "100%", "width": "100%"},
                                    dashGridOptions={
                                        "animateRows": True,
                                        "pagination": False,
                                        "suppressExcelExport": True,
                                        "enableRangeSelection": True,
                                        "suppressCsvExport": True,
                                    },
                                ),
                            ],
                        ),
                        html.Div(
                            id="rolling-chart-container",
                            style=rolling_chart_style,
                            children=[
                                html.Div(id="rolling-chart-wrapper", style={"height": "100%", "width": "100%"}),
                            ],
                        ),
                    ],
                ),
                dmc.TabsPanel(
                    value="statistics",
                    pt="md",
                    style={"flex": "1", "display": "flex", "flexDirection": "column", "overflow": "hidden"},
                    children=[
                        dcc.Loading(
                            id="loading-statistics",
                            type="default",
                            style={"flex": "1", "display": "flex", "flexDirection": "column", "overflow": "hidden"},
                            parent_style={"flex": "1", "display": "flex", "flexDirection": "column", "overflow": "hidden"},
                            children=[
                                dag.AgGrid(
                                    enableEnterpriseModules=True,
                                    licenseKey=AG_GRID_LICENSE_KEY,
                                    id="statistics-grid",
                                    className='ag-theme-alpine',
                                    columnDefs=[],
                                    rowData=[],
                                    defaultColDef={
                                        "resizable": True,
                                        "suppressHeaderMenuButton": True,
                                        "cellStyle": {"textAlign": "center"},
                                        "headerClass": "dashmat-center-header",
                                    },
                                    style={"height": "100%", "width": "100%"},
                                    dashGridOptions={
                                        "animateRows": True,
                                        "suppressExcelExport": True,
                                        "enableRangeSelection": True,
                                        "suppressCsvExport": True,
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
                dmc.TabsPanel(
                    value="calendar",
                    pt="md",
                    style={"flex": "1", "display": "flex", "flexDirection": "column", "overflow": "hidden"},
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
                            enableEnterpriseModules=True,
                            licenseKey=AG_GRID_LICENSE_KEY,
                            id="calendar-grid",
                            className='ag-theme-alpine',
                            columnDefs=[],
                            rowData=[],
                            defaultColDef={
                                "sortable": True,
                                "resizable": True,
                                "suppressHeaderMenuButton": True,
                                "cellStyle": {"textAlign": "center"},
                                "headerClass": "dashmat-center-header",
                            },
                            style={"height": "100%", "width": "100%"},
                            dashGridOptions={
                                "animateRows": True,
                                "suppressExcelExport": True,
                                "enableRangeSelection": True,
                                "suppressCsvExport": True,
                            },
                        ),
                    ],
                ),
                dmc.TabsPanel(
                    value="correlogram",
                    pt="md",
                    style={"flex": "1", "display": "flex", "flexDirection": "column", "overflow": "auto"},
                    children=[
                        dmc.Group(
                            mb="md",
                            children=[
                                dmc.SegmentedControl(
                                    id="correlation-view-switch",
                                    data=[
                                        {"value": "correlation", "label": "Correlation"},
                                        {"value": "correlogram", "label": "Correlogram"},
                                    ],
                                    value="correlogram",
                                    size="sm",
                                ),
                                dmc.NumberInput(
                                    id="correlogram-block-width",
                                    label=None,
                                    value=100,
                                    min=50,
                                    step=50,
                                    suffix="px",
                                    w=100,
                                    size="sm",
                                ),
                            ],
                        ),
                        html.Div(id="correlogram-container", style={"flex": "1", "minHeight": "0", "overflow": "auto"}),
                    ],
                ),
                dmc.TabsPanel(
                    value="growth",
                    pt="md",
                    style={"flex": "1", "display": "flex", "flexDirection": "column", "overflow": "hidden"},
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
                                html.Div(id="growth-charts-container", style={"height": "100%", "width": "100%"}),
                            ],
                        ),
                        html.Div(
                            id="growth-grid-container",
                            style=growth_grid_style,
                            children=[
                                dag.AgGrid(
                                    enableEnterpriseModules=True,
                                    licenseKey=AG_GRID_LICENSE_KEY,
                                    id="growth-grid",
                                    className='ag-theme-alpine',
                                    columnDefs=[],
                                    rowData=[],
                                    defaultColDef={
                                        "sortable": True,
                                        "resizable": True,
                                        "suppressHeaderMenuButton": True,
                                        "cellStyle": {"textAlign": "center"},
                                        "headerClass": "dashmat-center-header",
                                    },
                                    style={"height": "100%", "width": "100%"},
                                    dashGridOptions={
                                        "animateRows": True,
                                        "pagination": False,
                                        "suppressExcelExport": True,
                                        "enableRangeSelection": True,
                                        "suppressCsvExport": True,
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
                dmc.TabsPanel(
                    value="drawdown",
                    pt="md",
                    style={"flex": "1", "display": "flex", "flexDirection": "column", "overflow": "hidden"},
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
                                html.Div(id="drawdown-charts", style={"height": "100%", "width": "100%"}),
                            ],
                        ),
                        html.Div(
                            id="drawdown-grid-container",
                            style=drawdown_grid_style,
                            children=[
                                dag.AgGrid(
                                    enableEnterpriseModules=True,
                                    licenseKey=AG_GRID_LICENSE_KEY,
                                    id="drawdown-grid",
                                    className='ag-theme-alpine',
                                    columnDefs=[],
                                    rowData=[],
                                    defaultColDef={
                                        "sortable": True,
                                        "resizable": True,
                                        "suppressHeaderMenuButton": True,
                                        "cellStyle": {"textAlign": "center"},
                                        "headerClass": "dashmat-center-header",
                                    },
                                    style={"height": "100%", "width": "100%"},
                                    dashGridOptions={
                                        "animateRows": True,
                                        "pagination": False,
                                        "suppressExcelExport": True,
                                        "enableRangeSelection": True,
                                        "suppressCsvExport": True,
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
    fluid=True,
    style={"height": "calc(100vh - 55px)", "display": "flex", "flexDirection": "column", "overflow": "hidden"}, # 45px for header + 10px bottom margin
    className='dashmat-page-container',
    children=[
        # Stores for state management
        dmc.Paper(
            shadow="xs",
            p="xs",
            mb="md",
            radius="md",
            withBorder=True,
            className="dashmat-menu-bar",
            children=[
                dmc.Group(
                    gap="xs",
                    children=[
                        # File Menu (left)
                        dmc.Menu(
                            trigger="hover",
                            openDelay=100,
                            closeDelay=200,
                            position="bottom-start",
                            shadow="md",
                            offset=6,
                            children=[
                                dmc.MenuTarget(
                                    dmc.Button(
                                        "File",
                                        variant="subtle",
                                        color="gray",
                                        size="sm",
                                        radius="sm",
                                    ),
                                ),
                                dmc.MenuDropdown(
                                    className="dashmat-menu-dropdown",
                                    children=[
                                        dmc.MenuItem(
                                            "Add series from database...",
                                            id="menu-add-from-db",
                                            leftSection=DashIconify(icon="tabler:database", width=14),
                                        ),
                                        dmc.MenuItem(
                                            "Add series from file...",
                                            id="menu-add-series",
                                            leftSection=DashIconify(icon="tabler:upload", width=14),
                                        ),
                                        dmc.MenuDivider(),
                                        dmc.MenuItem(
                                            "Save Session",
                                            id="menu-save-session",
                                            disabled=True,
                                            leftSection=DashIconify(icon="tabler:device-floppy", width=14),
                                        ),
                                        dmc.MenuItem(
                                            "Load Session",
                                            id="menu-load-session",
                                            leftSection=DashIconify(icon="tabler:folder-open", width=14),
                                        ),
                                        dmc.MenuDivider(),
                                        dmc.MenuItem(
                                            "Download Excel",
                                            id="menu-download-excel",
                                            disabled=True,
                                            leftSection=DashIconify(icon="tabler:file-spreadsheet", width=14),
                                        ),
                                        dmc.MenuDivider(),
                                        dmc.MenuItem(
                                            "Exit",
                                            id="menu-exit",
                                            color="red",
                                            leftSection=DashIconify(icon="tabler:door-exit", width=14),
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
                            position="bottom-start",
                            shadow="md",
                            offset=6,
                            children=[
                                dmc.MenuTarget(
                                    dmc.Button(
                                        "Edit",
                                        variant="subtle",
                                        color="gray",
                                        size="sm",
                                        radius="sm",
                                    ),
                                ),
                                dmc.MenuDropdown(
                                    className="dashmat-menu-dropdown",
                                    children=[
                                        dmc.MenuItem(
                                            "Clear session storage and refresh",
                                            id="menu-clear-local-storage",
                                            leftSection=DashIconify(icon="tabler:trash", width=14),
                                        ),
                                        dmc.MenuItem(
                                            "Clear server cache",
                                            id="menu-clear-server-cache",
                                            leftSection=DashIconify(icon="tabler:server-off", width=14),
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        # Page switch button
                        dmc.Button(
                            "Switch to Optimization",
                            id="menu-view-portfolio",
                            size="sm",
                            radius="md",
                            variant="gradient",
                            gradient={"from": "indigo", "to": "cyan", "deg": 90},
                            leftSection=DashIconify(icon="grommet-icons:optimize", width=16),
                        ),
                        # Spacer
                        dmc.Box(style={"flexGrow": 1}),
                        # Help button (opens User Guide)
                        dmc.Button(
                            "Help",
                            id="menu-help-guide",
                            variant="gradient",
                            gradient={"from": "teal", "to": "cyan", "deg": 90},
                            size="sm",
                            radius="xl",
                            className="dashmat-menu-trigger",
                            leftSection=DashIconify(icon="tabler:help-circle", width=14),
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
            title=dmc.Group(
                gap="xs",
                children=[
                    dmc.ThemeIcon(DashIconify(icon="tabler:list-check"), color="blue", variant="light", size="sm"),
                    dmc.Text("Select Series", fw=600, size="sm"),
                ],
            ),
            size="80vw",
            styles={"content": {"maxWidth": "1250px"}},
            centered=True,
            closeOnEscape=False,
            radius="lg",
            className='series-modal-dark dashmat-modal',
            overlayProps={"blur": 2, "opacity": 0.45},
            transitionProps={"transition": "fade", "duration": 180},
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
                    style={"maxHeight": "50vh"},
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

        # Add-from-database Modal
        dmc.Modal(
            id="db-add-modal",
            title=dmc.Group(
                gap="xs",
                children=[
                    dmc.ThemeIcon(DashIconify(icon="tabler:database"), color="indigo", variant="light", size="sm"),
                    dmc.Text("Add from database", fw=600, size="sm"),
                ],
            ),
            size="md",
            centered=True,
            closeOnClickOutside=True,
            withCloseButton=True,
            radius="lg",
            className="dashmat-modal",
            overlayProps={"blur": 2, "opacity": 0.45},
            transitionProps={"transition": "fade", "duration": 180},
            children=[
                dmc.Alert(
                    id="db-add-error-alert",
                    title="Cannot add series",
                    color="red",
                    hide=True,
                    mb="sm",
                ),
                dmc.MultiSelect(
                    id="db-add-series-select",
                    label="Select Series",
                    data=[],
                    value=[],
                    searchable=True,
                    clearSearchOnChange=False,
                    placeholder="Select one or more series",
                    nothingFoundMessage="No categories found",
                    w="100%",
                ),
                dmc.Group(
                    mt="md",
                    justify="flex-end",
                    children=[
                        dmc.Button("Cancel", id="db-add-cancel-button", variant="outline", color="red"),
                        dmc.Button("OK", id="db-add-ok-button", color="blue", disabled=True),
                    ],
                ),
            ],
        ),

        # Sheet Selection Modal (for multi-tab Excel files)
        dmc.Modal(
            id="sheet-select-modal",
            title=dmc.Group(
                gap="xs",
                children=[
                    dmc.ThemeIcon(DashIconify(icon="tabler:table"), color="teal", variant="light", size="sm"),
                    dmc.Text("Select Sheet", fw=600, size="sm"),
                ],
            ),
            size="sm",
            centered=True,
            closeOnClickOutside=False,
            radius="lg",
            className="dashmat-modal",
            overlayProps={"blur": 2, "opacity": 0.45},
            transitionProps={"transition": "fade", "duration": 180},
            children=[
                dmc.Text("This file contains multiple sheets. Select which sheet to import:", size="sm", mb="md"),
                dmc.Select(
                    id="sheet-select-dropdown",
                    data=[],
                    value=None,
                    w="100%",
                    size="sm",
                    placeholder="Select a sheet",
                ),
                dmc.Group(
                    mt="md",
                    justify="flex-end",
                    children=[
                        dmc.Button("Cancel", id="sheet-select-cancel-button", variant="outline", color="red"),
                        dmc.Button("OK", id="sheet-select-ok-button", color="blue"),
                    ],
                ),
            ],
        ),

        # Help Modal
        dmc.Modal(
            id="help-modal",
            title=dmc.Group(
                gap="xs",
                children=[
                    dmc.ThemeIcon(DashIconify(icon="tabler:book"), color="grape", variant="light", size="sm"),
                    dmc.Text("User Guide", fw=600, size="sm"),
                ],
            ),
            size="lg",
            centered=True,
            radius="lg",
            className="dashmat-modal",
            overlayProps={"blur": 2, "opacity": 0.45},
            transitionProps={"transition": "fade", "duration": 180},
            children=[
                dmc.Accordion(
                    variant="separated",
                    children=[
                        dmc.AccordionItem(
                            value="getting-started",
                            children=[
                                dmc.AccordionControl("Getting Started"),
                                dmc.AccordionPanel(dmc.Text(
                                    "Upload Excel (.xlsx, .xls) or CSV files containing returns data. "
                                    "Rows should be dates and columns should be series names. "
                                    "Values can be in decimal format (0.05) or percent format (5%). "
                                    "Percent signs are auto-detected and converted. "
                                    "Sample data files are available from File > Download sample data.",
                                    size="sm",
                                )),
                            ],
                        ),
                        dmc.AccordionItem(
                            value="series-selection",
                            children=[
                                dmc.AccordionControl("Series Selection"),
                                dmc.AccordionPanel(dmc.Text(
                                    "Click the Series Selection button to open the modal. "
                                    "Select, reorder (drag and drop), and rename series. "
                                    "Assign a benchmark to any series for relative analysis. "
                                    "Enable Long-Short to treat the series-benchmark difference as an absolute return stream. "
                                    "Enable Vol Scaling per series to scale returns to a target volatility.",
                                    size="sm",
                                )),
                            ],
                        ),
                        dmc.AccordionItem(
                            value="periodicity",
                            children=[
                                dmc.AccordionControl("Periodicity"),
                                dmc.AccordionPanel(dmc.Text(
                                    "Periodicity is auto-detected on upload. "
                                    "Daily data can be converted to Weekly (with Monday through Friday end-of-week options) or Monthly. "
                                    "Monthly data cannot be upsampled to daily. "
                                    "When appending daily data to an existing monthly dataset, daily data is automatically resampled to monthly.",
                                    size="sm",
                                )),
                            ],
                        ),
                        dmc.AccordionItem(
                            value="returns",
                            children=[
                                dmc.AccordionControl("Returns Mode"),
                                dmc.AccordionPanel(dmc.Stack(gap="xs", children=[
                                    dmc.Text(
                                        "The Returns Type control applies across pages that support return mode switching.",
                                        size="sm",
                                    ),
                                    dmc.Text(
                                        "Total Returns: use each selected series directly.",
                                        size="sm",
                                    ),
                                    dmc.Text(
                                        "Excess Returns: arithmetic difference (Series - Benchmark) per period.",
                                        size="sm",
                                    ),
                                    dmc.Text(
                                        "Long-Short: converts a series into the Series - Benchmark stream and treats it as an absolute return stream.",
                                        size="sm",
                                    ),
                                ])),
                            ],
                        ),
                        dmc.AccordionItem(
                            value="vol-scaler",
                            children=[
                                dmc.AccordionControl("Vol Scaler"),
                                dmc.AccordionPanel(dmc.Text(
                                    "Scale returns to a target annualized volatility (0-100%). "
                                    "When set to a non-zero value, each series with vol scaling enabled "
                                    "will have its returns scaled so that its annualized volatility matches the target.",
                                    size="sm",
                                )),
                            ],
                        ),
                        dmc.AccordionItem(
                            value="statistics-page",
                            children=[
                                dmc.AccordionControl("Statistics Page"),
                                dmc.AccordionPanel(dmc.Stack(gap="xs", children=[
                                    dmc.Text(
                                        "Statistics are calculated on the currently selected periodicity and date range. "
                                        "If a benchmark is assigned, series and benchmark are aligned to overlapping dates.",
                                        size="sm",
                                    ),
                                    dmc.Text(
                                        "Sharpe Ratio uses the session risk-free proxy BCTBill13: "
                                        "(Annualized Return of series - Annualized Return of BCTBill13) / Annualized Volatility. "
                                        "If history lengths differ, only overlapping dates are used.",
                                        size="sm",
                                    ),
                                    dmc.Text(
                                        "Relative metrics (Excess Return, Tracking Error, Information Ratio, Correlation, Hit Rate vs Benchmark) "
                                        "are blank when no valid benchmark exists.",
                                        size="sm",
                                    ),
                                    dmc.Divider(label="Base Metrics", labelPosition="center"),
                                    dmc.Text("Start Date: first date in the aligned sample used for calculations.", size="sm"),
                                    dmc.Text("End Date: last date in the aligned sample used for calculations.", size="sm"),
                                    dmc.Text("Number of Periods: count of aligned return observations.", size="sm"),
                                    dmc.Text("Cumulative Return: (product of (1 + r_t)) - 1.", size="sm"),
                                    dmc.Text(
                                        "Annualized Return: for daily/weekly uses calendar-day annualization; "
                                        "for monthly uses period-based annualization. For <= 1 year, cumulative return is shown.",
                                        size="sm",
                                    ),
                                    dmc.Text("Annualized Volatility: sample std(r_t) * sqrt(periods_per_year).", size="sm"),
                                    dmc.Text("Sharpe Ratio: (Annualized Return - Annualized Return(BCTBill13)) / Annualized Volatility.", size="sm"),
                                    dmc.Text(
                                        "Sortino Ratio: (Annualized Return - Annualized Return(BCTBill13)) / Annualized Downside Deviation, "
                                        "where downside includes only periods below 0.",
                                        size="sm",
                                    ),
                                    dmc.Text("Annualized Excess Return: Annualized Return(series) - Annualized Return(benchmark).", size="sm"),
                                    dmc.Text("Annualized Tracking Error: std(series - benchmark) * sqrt(periods_per_year).", size="sm"),
                                    dmc.Text("Information Ratio: Annualized Excess Return / Annualized Tracking Error.", size="sm"),
                                    dmc.Text("Correlation: Pearson correlation of series and benchmark returns.", size="sm"),
                                    dmc.Text("Hit Rate: fraction of periods with return > 0.", size="sm"),
                                    dmc.Text("Hit Rate (vs Benchmark): fraction of periods where series return > benchmark return.", size="sm"),
                                    dmc.Text("Best Period Return: maximum single-period return.", size="sm"),
                                    dmc.Text("Worst Period Return: minimum single-period return.", size="sm"),
                                    dmc.Text(
                                        "Maximum Drawdown: minimum of (Wealth / Running Peak - 1), with Wealth = cumulative product of (1 + r_t).",
                                        size="sm",
                                    ),
                                    dmc.Text("Skewness: return distribution skew (requires > 2 observations).", size="sm"),
                                    dmc.Text("Kurtosis: excess kurtosis of returns (requires > 3 observations).", size="sm"),
                                    dmc.Divider(label="Trailing Window Metrics", labelPosition="center"),
                                    dmc.Text(
                                        "1Y/3Y/5Y metrics use the same formulas above, applied to trailing windows. "
                                        "If insufficient history exists, values are blank.",
                                        size="sm",
                                    ),
                                    dmc.Text(
                                        "For daily data, trailing windows are calendar-day offsets from the latest date. "
                                        "Example if latest date is 2026-02-11: "
                                        "1Y uses 2025-02-12 to 2026-02-11 inclusive; "
                                        "3Y uses 2023-02-12 to 2026-02-11 inclusive.",
                                        size="sm",
                                    ),
                                ])),
                            ],
                        ),
                        dmc.AccordionItem(
                            value="returns-page",
                            children=[
                                dmc.AccordionControl("Returns Page"),
                                dmc.AccordionPanel(dmc.Stack(gap="xs", children=[
                                    dmc.Text(
                                        "Shows the selected series in a time-series table at the selected periodicity.",
                                        size="sm",
                                    ),
                                    dmc.Text(
                                        "Values reflect Total or Excess mode, date range filters, long-short settings, and optional vol scaling.",
                                        size="sm",
                                    ),
                                    dmc.Text(
                                        "Use Common Range to limit dates to periods where all selected series have data, "
                                        "Common Daily to jump to the overlap where all selected series are in daily phase "
                                        "and switch periodicity to Daily (Trading), or Maximum Range to use full available dates.",
                                        size="sm",
                                    ),
                                ])),
                            ],
                        ),
                        dmc.AccordionItem(
                            value="rolling-page",
                            children=[
                                dmc.AccordionControl("Rolling Page"),
                                dmc.AccordionPanel(dmc.Stack(gap="xs", children=[
                                    dmc.Text(
                                        "Calculates rolling metrics for each selected series over 3M, 6M, 1Y, 3Y, 5Y, or 10Y windows.",
                                        size="sm",
                                    ),
                                    dmc.Text(
                                        "Rolling Total Return / Excess Return can be shown as cumulative or annualized.",
                                        size="sm",
                                    ),
                                    dmc.Text(
                                        "Rolling Volatility / Tracking Error are annualized as std(window) * sqrt(periods_per_year).",
                                        size="sm",
                                    ),
                                    dmc.Text(
                                        "Rolling Sharpe / Sortino / Information Ratio use rolling windows of the selected stream "
                                        "(series, excess stream, or long-short stream as applicable).",
                                        size="sm",
                                    ),
                                    dmc.Text(
                                        "Rolling Correlation uses Pearson correlation between each series and its benchmark in each window.",
                                        size="sm",
                                    ),
                                    dmc.Text(
                                        "If a relative metric is selected without a valid benchmark, values are blank.",
                                        size="sm",
                                    ),
                                ])),
                            ],
                        ),
                        dmc.AccordionItem(
                            value="calendar-page",
                            children=[
                                dmc.AccordionControl("Calendar Year Page"),
                                dmc.AccordionPanel(dmc.Stack(gap="xs", children=[
                                    dmc.Text(
                                        "Annual view: groups returns by year and compounds within each year.",
                                        size="sm",
                                    ),
                                    dmc.Text(
                                        "Partial years are removed. For daily data, first year must start by Jan 4 and last year must end by Dec 28. "
                                        "For monthly data, full-year month coverage is required.",
                                        size="sm",
                                    ),
                                    dmc.Text(
                                        "Excess mode subtracts annual benchmark returns for non-long-short series with valid benchmarks.",
                                        size="sm",
                                    ),
                                    dmc.Text(
                                        "Monthly view shows one selected series with Jan-Dec columns by year.",
                                        size="sm",
                                    ),
                                ])),
                            ],
                        ),
                        dmc.AccordionItem(
                            value="growth-page",
                            children=[
                                dmc.AccordionControl("Growth of $1 Page"),
                                dmc.AccordionPanel(dmc.Stack(gap="xs", children=[
                                    dmc.Text(
                                        "Builds growth index series as cumulative product of (1 + r_t).",
                                        size="sm",
                                    ),
                                    dmc.Text(
                                        "Each series is prepended with a starting value of 1.0 one period before its first return.",
                                        size="sm",
                                    ),
                                    dmc.Text(
                                        "Displays an all-series growth chart plus per-series benchmark comparison charts when benchmarks are assigned.",
                                        size="sm",
                                    ),
                                    dmc.Text(
                                        "Long-short series are plotted as their long-short return streams.",
                                        size="sm",
                                    ),
                                ])),
                            ],
                        ),
                        dmc.AccordionItem(
                            value="drawdown-page",
                            children=[
                                dmc.AccordionControl("Drawdown Page"),
                                dmc.AccordionPanel(dmc.Stack(gap="xs", children=[
                                    dmc.Text(
                                        "Computes drawdown as (Current Wealth / Running Peak) - 1 for each series.",
                                        size="sm",
                                    ),
                                    dmc.Text(
                                        "Total mode uses each series cumulative wealth from returns.",
                                        size="sm",
                                    ),
                                    dmc.Text(
                                        "Excess mode (non-long-short) with valid benchmark uses geometric relative wealth "
                                        "(Growth(series) / Growth(benchmark)) before drawdown is computed.",
                                        size="sm",
                                    ),
                                    dmc.Text(
                                        "Each drawdown series is prepended with 0.0 one period before the first return.",
                                        size="sm",
                                    ),
                                ])),
                            ],
                        ),
                        dmc.AccordionItem(
                            value="correlation-page",
                            children=[
                                dmc.AccordionControl("Correlation Page"),
                                dmc.AccordionPanel(dmc.Stack(gap="xs", children=[
                                    dmc.Text(
                                        "Displays cross-series correlation using the currently selected return stream "
                                        "(Total or Excess), periodicity, date range, and scaling settings.",
                                        size="sm",
                                    ),
                                    dmc.Text(
                                        "Heatmap view shows the full correlation matrix with values from -1 to 1.",
                                        size="sm",
                                    ),
                                    dmc.Text(
                                        "Scatter Matrix (correlogram) view shows pairwise scatter plots, diagonal histograms, "
                                        "and upper-triangle correlation labels.",
                                        size="sm",
                                    ),
                                ])),
                            ],
                        ),
                        dmc.AccordionItem(
                            value="export",
                            children=[
                                dmc.AccordionControl("Export"),
                                dmc.AccordionPanel(dmc.Text(
                                    "Download all tabs as a multi-sheet Excel workbook via File > Download Excel. "
                                    "The export includes Statistics, Returns, Rolling, Calendar Year, Growth of $1, Drawdown, and Correlation sheets.",
                                    size="sm",
                                )),
                            ],
                        ),
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
        # analyticstool-raw-data-store and analyticstool-original-periodicity-store are defined in app.py (shared across pages)
        dcc.Store(id="benchmark-assignments-store", data={}, storage_type="session"),
        dcc.Store(id="long-short-store", data={}, storage_type="session"),
        dcc.Store(id="periodicity-value-store", data="daily_trading", storage_type="session"),
        dcc.Store(id="at-periodicity-load-sync-dummy", data=None),
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
        # Temp stores for sheet selection (stash upload while user picks a tab)
        dcc.Store(id="sheet-select-contents-store", data=None),
        dcc.Store(id="sheet-select-filename-store", data=None),
        dcc.Download(id="download-excel"),
        dcc.Download(id="download-sample-daily"),
        dcc.Download(id="download-sample-monthly"),
        # Save/Load session
        dcc.Store(id="save-session-dummy", data=None, storage_type="memory"),
        dcc.Store(id="load-session-dummy", data=None, storage_type="memory"),
        dcc.Store(id="server-cache-clear-result", data=None, storage_type="memory"),
        html.Div(
            dcc.Upload(
                id="load-session-upload",
                children=html.Div(),
                multiple=False,
                accept=".json",
            ),
            style={"display": "none"},
        ),
        dcc.Location(id="url-location", refresh=False),
        # Moved series-select and edit-mode to global scope
        dcc.Store(id="series-select", data=[], storage_type="session"),
        dcc.Store(id="series-edit-mode", data=None),

        # Store to trigger clientside focus on edit input
        dcc.Store(id="edit-box-focus-trigger", data=None),
        # Dummy div for clientside callback output
        html.Div(id="dummy-focus-output"),
        
        # Correlogram metadata for client-side sizing
        dcc.Store(id="correlogram-meta-store", data={}),

        # UI Blocker for file dialog (Overlay)
        dcc.Store(id="ui-blocker-store", data=False),
        dcc.Interval(id="ui-blocker-timeout", interval=15000, disabled=True), # 15 second timeout
        dmc.LoadingOverlay(
            id="ui-blocker-overlay",
            visible=False,
            zIndex=2000,
            overlayProps={"radius": "sm", "blur": 2},
            loaderProps={"variant": "bars"},
        ),

        # One-shot interval to trigger visibility check after session-storage hydration
        dcc.Interval(id="at-page-load-trigger", interval=50, max_intervals=1, n_intervals=0),
    ],
)


# Toggle welcome/main visibility based on analyticstool-raw-data-store.
# Uses a one-shot Interval to guarantee session-storage has hydrated on
# cross-page navigation, plus analyticstool-raw-data-store Input for same-page uploads.
clientside_callback(
    """
    function(n_intervals, data) {
        if (data) {
            return [{display: "none"}, {display: "flex", flexDirection: "column", flex: "1", overflow: "hidden"}];
        }
        return [{display: "block"}, {display: "none"}];
    }
    """,
    Output("welcome-screen-container", "style"),
    Output("main-app-container", "style"),
    Input("at-page-load-trigger", "n_intervals"),
    Input("analyticstool-raw-data-store", "data"),
)


@callback(
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
    Input("at-page-load-trigger", "n_intervals"),
    Input("analyticstool-raw-data-store", "data"),
    State("analyticstool-original-periodicity-store", "data"),
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
    State("analyticstool-pending-new-series-store", "data"),
    prevent_initial_call="initial_duplicate",
)
def restore_application_state(n_intervals, raw_data, orig_periodicity, stored_periodicity, stored_series, stored_returns, stored_vol, stored_tab, stored_roll_win, stored_roll_metric, stored_roll_type, stored_roll_chart, stored_dd_chart, stored_gr_chart, stored_monthly_view, stored_monthly_series, pending_series):
    if not raw_data:
        # Reset defaults (visibility handled by clientside callback)
        return (
            [{"value": "daily_trading", "label": "Daily (Trading)"}], "daily_trading", "total", 0, "statistics",
            "1y", "total_return", "annualized", False, {}, "chart", "chart", "chart",
            "annual", []
        )

    try:
        df = json_to_df(raw_data)
        
        # Periodicity
        periodicity_options = get_available_periodicities(orig_periodicity or "daily")
        valid_periodicity = stored_periodicity if stored_periodicity in [p["value"] for p in periodicity_options] else ("daily_trading" if orig_periodicity == "daily" else (orig_periodicity or "daily_trading"))
        
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
        if not valid_selection:
            valid_selection = list(df.columns)

        # Auto-add any pending new series from portfolio optimization
        for s in (pending_series or []):
            if s in df.columns and s not in valid_selection:
                valid_selection.append(s)
        
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
            periodicity_options, valid_periodicity, valid_returns, valid_vol, active_tab,
            roll_win, roll_metric, roll_type, roll_type_disabled, roll_type_style, roll_chart, dd_chart, gr_chart,
            monthly_view, valid_selection
        )

    except Exception:
        # Fallback to defaults on error (visibility handled by clientside callback)
        return (
            [{"value": "daily_trading", "label": "Daily (Trading)"}], "daily_trading", "total", 0, "statistics",
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


# Navigate to Portfolio Optimization page (client-side, preserves shared stores)
clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks) {
            window.location.pathname = '/portopt';
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("url-location", "pathname", allow_duplicate=True),
    Input("menu-view-portfolio", "n_clicks"),
    prevent_initial_call=True,
)


# Clientside callback to clear session storage and refresh page
clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks) {
            // Clear all sessionStorage keys for both pages
            const keysToRemove = [
                'analyticstool-raw-data-store',
                'analyticstool-original-periodicity-store',
                'analyticstool-pending-new-series-store',
                'analyticstool-saved-series-cache-store',
                'bctbill13-cache-store',
                'series-select',
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
                'vol-scaling-assignments-store',
                'po-series-select',
                'po-series-order-store',
                'po-benchmark-assignments-store',
                'po-long-short-store',
                'po-vol-scaling-assignments-store',
                'po-min-wt-store',
                'po-max-wt-store',
                'po-force-max-store',
                'po-periodicity-value-store',
                'po-vol-scaler-value-store',
                'po-date-range-store',
                'po-series-select-value-store',
                'po-opt-window-store',
                'po-window-size-store',
                'po-opt-step-store',
                'po-opt-model-store',
                'po-portfolio-name-store',
                'po-exp-wt-cov-store',
                'po-halflife-store',
                'po-missing-data-store',
                'po-fill-in-sample-store',
                'po-results-store',
                'po-active-tab-store'
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


@callback(
    Output("server-cache-clear-result", "data"),
    Input("menu-clear-server-cache", "n_clicks"),
    prevent_initial_call=True,
)
def clear_server_cache(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    cache_config.cache.clear()
    clear_dropdown_caches()
    return {"cleared": True, "timestamp": pd.Timestamp.utcnow().isoformat()}


# Clientside callback to trigger upload from menu
clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks) {
            // Trigger the file input click with a small delay to allow overlay to render
            setTimeout(function() {
                var uploadDiv = document.getElementById('upload-data');
                if (uploadDiv) {
                    var input = uploadDiv.querySelector('input[type="file"]');
                    if (input) {
                        // Listen for window focus to detect cancel
                        var onFocus = function() {
                            window.removeEventListener('focus', onFocus);
                            setTimeout(function() {
                                if (!input.files || input.files.length === 0) {
                                    // User cancelled - hide the blocker
                                    window.dash_clientside.set_props('ui-blocker-store', {data: false});
                                    window.dash_clientside.set_props('ui-blocker-timeout', {disabled: true});
                                }
                            }, 500);
                        };
                        window.addEventListener('focus', onFocus);
                        input.click();
                    }
                }
            }, 100);

            // Show Blocker (True), Enable Timeout (False)
            return [true, false];
        }
        return [window.dash_clientside.no_update, window.dash_clientside.no_update];
    }
    """,
    Output("ui-blocker-store", "data", allow_duplicate=True),
    Output("ui-blocker-timeout", "disabled", allow_duplicate=True),
    Input("menu-add-series", "n_clicks"),
    prevent_initial_call=True,
)


clientside_callback(
    """
    function(n) {
        // Hide Blocker (False), Disable Timeout (True)
        return [false, true];
    }
    """,
    Output("ui-blocker-store", "data", allow_duplicate=True),
    Output("ui-blocker-timeout", "disabled", allow_duplicate=True),
    Input("ui-blocker-timeout", "n_intervals"),
    prevent_initial_call=True,
)


clientside_callback(
    """
    function(is_loading) {
        return is_loading || false;
    }
    """,
    Output("ui-blocker-overlay", "visible"),
    Input("ui-blocker-store", "data"),
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
    Output("analyticstool-raw-data-store", "data", allow_duplicate=True),
    Output("vol-scaling-assignments-store", "data", allow_duplicate=True),
    Input("modal-ok-button", "n_clicks"),
    State("temp-series-select", "data"),
    State("temp-benchmark-assignments-store", "data"),
    State("temp-long-short-store", "data"),
    State("temp-series-order-store", "data"),
    State("temp-deleted-series-store", "data"),
    State("analyticstool-raw-data-store", "data"),
    State("temp-vol-scaling-assignments-store", "data"),
    prevent_initial_call=True,
)
def on_modal_ok(n_clicks, temp_select, temp_bench, temp_ls, temp_order, temp_deleted, raw_data, temp_vol_scaling):
    if not n_clicks:
        raise PreventUpdate

    temp_select = list(temp_select or [])
    if temp_order:
        selected_set = set(temp_select)
        temp_select = [series for series in temp_order if series in selected_set]

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
                remaining_cols = set(df.columns)
                cleaned_bench = {}
                for series, bench in temp_bench.items():
                    if series not in remaining_cols:
                        continue
                    bench_value = bench if isinstance(bench, str) else "None"
                    if bench_value != "None" and bench_value not in remaining_cols:
                        bench_value = "None"
                    cleaned_bench[series] = bench_value
                temp_bench = cleaned_bench
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
    Input("series-selection-grid", "virtualRowData", allow_optional=True),
    Input("series-selection-grid", "selectedRows", allow_optional=True),
    State("temp-series-order-store", "data"),
    State("temp-series-select", "data"),
    prevent_initial_call=True,
)
def reorder_series(virtual_rows, selected_rows, current_order, current_selected):
    """Keep temp order and selection aligned with AG Grid state."""
    ordered_series = []
    if isinstance(virtual_rows, (list, tuple)):
        ordered_series = [
            row.get("Series")
            for row in virtual_rows
            if isinstance(row, dict) and row.get("Series")
        ]
    elif current_order:
        ordered_series = list(current_order)
    if not ordered_series:
        raise PreventUpdate

    triggered_props = []
    try:
        if callback_context and callback_context.triggered:
            triggered_props = [t.get("prop_id", "") for t in callback_context.triggered]
    except Exception:
        triggered_props = []

    if isinstance(selected_rows, (list, tuple)):
        selected_set = {
            row.get("Series")
            for row in selected_rows
            if isinstance(row, dict) and row.get("Series")
        }
        selected_series = [series for series in ordered_series if series in selected_set]
        # Guard against transient empty selectedRows payloads during grid hydration.
        selected_rows_triggered = any(
            prop.startswith("series-selection-grid.selectedRows")
            for prop in triggered_props
        )
        if not selected_series and (current_selected or []) and not selected_rows_triggered:
            selected_fallback = set(current_selected or [])
            selected_series = [series for series in ordered_series if series in selected_fallback]
    else:
        selected_fallback = set(current_selected or [])
        selected_series = [series for series in ordered_series if series in selected_fallback]

    if ordered_series == (current_order or []) and selected_series == (current_selected or []):
        raise PreventUpdate
    return ordered_series, selected_series







# Clientside callback for periodicity selection storage
clientside_callback(
    "function(value) { return value; }",
    Output("periodicity-value-store", "data"),
    Input("periodicity-select", "value"),
    prevent_initial_call=True,
)


# Sync periodicity to PortOpt only on raw-data load/update events.
clientside_callback(
    """
    function(rawData, periodicityValue) {
        const ctx = window.dash_clientside.callback_context;
        const triggered = (ctx && ctx.triggered) ? ctx.triggered : [];
        const rawTriggered = triggered.some(
            t => t && t.prop_id && t.prop_id.indexOf("analyticstool-raw-data-store.") === 0
        );
        if (!rawTriggered || !rawData || !periodicityValue) {
            return window.dash_clientside.no_update;
        }
        sessionStorage.setItem("po-periodicity-value-store", JSON.stringify(periodicityValue));
        return periodicityValue;
    }
    """,
    Output("at-periodicity-load-sync-dummy", "data"),
    Input("analyticstool-raw-data-store", "data"),
    Input("periodicity-value-store", "data"),
    prevent_initial_call=True,
)


# Clientside callback for returns type selection storage
clientside_callback(
    "function(value) { return value; }",
    Output("returns-type-value-store", "data"),
    Input("returns-type-select", "value"),
    prevent_initial_call=True,
)


# Clientside callback for vol scaler value storage
clientside_callback(
    "function(value) { return value; }",
    Output("vol-scaler-value-store", "data"),
    Input("vol-scaler-input", "value"),
    prevent_initial_call=True,
)


# Clientside callback for series selection storage
clientside_callback(
    "function(value) { return value || []; }",
    Output("series-select-value-store", "data"),
    Input("series-select", "data"),
    prevent_initial_call=True,
)


# Clientside callback for active tab storage
clientside_callback(
    "function(value) { return value || 'statistics'; }",
    Output("active-tab-store", "data"),
    Input("main-tabs", "value"),
    prevent_initial_call=True,
)





# Clientside callback for rolling window selection storage
clientside_callback(
    "function(value) { return value || '1y'; }",
    Output("rolling-window-store", "data"),
    Input("rolling-window-select", "value"),
    prevent_initial_call=True,
)


# Clientside callback for rolling metric selection storage
clientside_callback(
    "function(value) { return value || 'total_return'; }",
    Output("rolling-metric-store", "data"),
    Input("rolling-metric-select", "value"),
    prevent_initial_call=True,
)


# Clientside callback for rolling return type storage
clientside_callback(
    "function(value) { return value || 'annualized'; }",
    Output("rolling-return-type-store", "data"),
    Input("rolling-return-type-select", "value"),
    prevent_initial_call=True,
)


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





# Clientside callback for rolling chart switch storage
clientside_callback(
    "function(value) { return value !== null && value !== undefined ? value : 'chart'; }",
    Output("rolling-chart-switch-store", "data"),
    Input("rolling-chart-switch", "value"),
    prevent_initial_call=True,
)





# Clientside callback for rolling view toggle
clientside_callback(
    """
    function(view_type) {
        const flex_style = {display: "flex", flexDirection: "column", flex: "1", overflow: "hidden"};
        if (view_type === "chart") {
            return [{display: "none"}, flex_style];
        } else {
            return [flex_style, {display: "none"}];
        }
    }
    """,
    Output("rolling-grid-container", "style"),
    Output("rolling-chart-container", "style"),
    Input("rolling-chart-switch", "value"),
    prevent_initial_call=True,
)


# Clientside callback for drawdown chart switch storage
clientside_callback(
    "function(value) { return value !== null && value !== undefined ? value : 'chart'; }",
    Output("drawdown-chart-switch-store", "data"),
    Input("drawdown-chart-switch", "value"),
    prevent_initial_call=True,
)





# Clientside callback for drawdown view toggle
clientside_callback(
    """
    function(view_type) {
        const flex_style = {display: "flex", flexDirection: "column", flex: "1", overflow: "hidden"};
        const flex_scroll_style = {display: "flex", flexDirection: "column", flex: "1", overflow: "auto"};
        if (view_type === "chart") {
            return [{display: "none"}, flex_scroll_style];
        } else {
            return [flex_style, {display: "none"}];
        }
    }
    """,
    Output("drawdown-grid-container", "style"),
    Output("drawdown-chart-container", "style"),
    Input("drawdown-chart-switch", "value"),
    prevent_initial_call=True,
)


# Clientside callback for growth chart switch storage
clientside_callback(
    "function(value) { return value !== null && value !== undefined ? value : 'chart'; }",
    Output("growth-chart-switch-store", "data"),
    Input("growth-chart-switch", "value"),
    prevent_initial_call=True,
)





# Clientside callback for growth view toggle
clientside_callback(
    """
    function(view_type) {
        const flex_style = {display: "flex", flexDirection: "column", flex: "1", overflow: "hidden"};
        const flex_scroll_style = {display: "flex", flexDirection: "column", flex: "1", overflow: "auto"};
        if (view_type === "chart") {
            return [{display: "none"}, flex_scroll_style];
        } else {
            return [flex_style, {display: "none"}];
        }
    }
    """,
    Output("growth-grid-container", "style"),
    Output("growth-chart-container", "style"),
    Input("growth-chart-switch", "value"),
    prevent_initial_call=True,
)


# Clientside callback for monthly view storage
clientside_callback(
    "function(value) { return value !== null && value !== undefined ? value : 'annual'; }",
    Output("monthly-view-store", "data"),
    Input("monthly-view-checkbox", "value"),
    prevent_initial_call=True,
)


# Clientside callback for monthly series selection storage
clientside_callback(
    "function(value) { return value; }",
    Output("monthly-series-store", "data"),
    Input("monthly-series-select", "value"),
    prevent_initial_call=True,
)





@callback(
    Output("analyticstool-raw-data-store", "data", allow_duplicate=True),
    Output("analyticstool-original-periodicity-store", "data", allow_duplicate=True),
    Output("periodicity-select", "data", allow_duplicate=True),
    Output("periodicity-select", "value", allow_duplicate=True),
    Output("periodicity-select", "disabled", allow_duplicate=True),
    Output("temp-series-select", "data", allow_duplicate=True),
    Output("alert-message", "children", allow_duplicate=True),
    Output("alert-message", "color", allow_duplicate=True),
    Output("alert-message", "hide", allow_duplicate=True),
    Output("periodicity-value-store", "data", allow_duplicate=True),
    Output("series-selection-modal", "opened", allow_duplicate=True),
    Output("temp-benchmark-assignments-store", "data", allow_duplicate=True),
    Output("temp-long-short-store", "data", allow_duplicate=True),
    Output("temp-series-order-store", "data", allow_duplicate=True),
    Output("first-load-store", "data", allow_duplicate=True),
    Output("temp-deleted-series-store", "data", allow_duplicate=True),
    Output("temp-vol-scaling-assignments-store", "data", allow_duplicate=True),
    Output("db-add-modal", "opened", allow_duplicate=True),
    Output("db-add-series-select", "value", allow_duplicate=True),
    Input("db-add-ok-button", "n_clicks"),
    State("db-add-series-select", "value"),
    State("analyticstool-raw-data-store", "data"),
    State("analyticstool-original-periodicity-store", "data"),
    State("series-select", "data"),
    State("benchmark-assignments-store", "data"),
    State("long-short-store", "data"),
    State("series-order-store", "data"),
    State("first-load-store", "data"),
    State("vol-scaling-assignments-store", "data"),
    prevent_initial_call=True,
)
def add_series_from_database(
    n_clicks,
    selected_benches,
    existing_data,
    existing_periodicity,
    current_selection,
    current_bench,
    current_ls,
    current_order,
    first_load,
    current_vol_scaling,
):
    if not n_clicks:
        raise PreventUpdate

    n_no = no_update
    if not selected_benches:
        return (
            n_no, n_no, n_no, n_no, n_no,
            n_no,
            "Select at least one series from the database.",
            "orange",
            False,
            n_no, n_no, n_no, n_no, n_no,
            n_no, n_no, n_no,
            True, n_no,
        )

    try:
        if existing_data:
            existing_cols = set(json_to_df(existing_data).columns)
            duplicates = [s for s in selected_benches if s in existing_cols]
            if duplicates:
                return (
                    n_no, n_no, n_no, n_no, n_no,
                    n_no,
                    f"Cannot add duplicate series: {', '.join(duplicates)}",
                    "red",
                    False,
                    n_no, n_no, n_no, n_no, n_no,
                    n_no, n_no, n_no,
                    True, n_no,
                )

        new_df, db_meta = load_cma_returns_for_benches_with_meta(
            DB_ENGINE, selected_benches, MRD_ENGINE
        )
        if new_df.empty:
            raise ValueError("No rows returned for selected FOFBench values.")

        # Database import is daily unless every selected series is monthly-only.
        new_periodicity = "daily"
        any_start_daily = False
        all_start_daily = True
        daily_transition_notes: list[str] = []
        for series_name in new_df.columns:
            meta = db_meta.get(series_name, {}) if isinstance(db_meta, dict) else {}
            starts_daily = bool(meta.get("starts_daily", True))
            any_start_daily = any_start_daily or starts_daily
            if not starts_daily:
                all_start_daily = False
                daily_start_date = meta.get("daily_start_date")
                if daily_start_date:
                    daily_transition_notes.append(f"{series_name}: {daily_start_date}")
                else:
                    daily_transition_notes.append(f"{series_name}: no daily phase detected")
        if not any_start_daily:
            new_periodicity = "monthly"

        if existing_data is not None:
            existing_df = json_to_df(existing_data)
            if existing_periodicity == "monthly" and new_periodicity == "daily":
                new_df = resample_returns(new_df, "monthly")
                combined_periodicity = "monthly"
            elif new_periodicity == "monthly" and existing_periodicity == "daily":
                existing_df = resample_returns(existing_df, "monthly")
                combined_periodicity = "monthly"
            else:
                combined_periodicity = existing_periodicity
            existing_df = _normalize_monthly_df_if_needed(existing_df, combined_periodicity)
            new_df = _normalize_monthly_df_if_needed(new_df, combined_periodicity)
            merged_df = merge_returns(existing_df, new_df)
        else:
            merged_df = new_df
            combined_periodicity = new_periodicity
            merged_df = _normalize_monthly_df_if_needed(merged_df, combined_periodicity)

        periodicity_options = get_available_periodicities(combined_periodicity)
        if combined_periodicity == "daily":
            default_periodicity = "daily_trading" if all_start_daily else "monthly"
        else:
            default_periodicity = combined_periodicity

        new_series = [col for col in new_df.columns if col not in (current_selection or [])]
        updated_selection = (current_selection or []) + new_series

        alert_msg = (
            f"Loaded {len(new_df.columns)} series with {len(new_df)} rows from database"
        )
        if daily_transition_notes:
            alert_msg = f"{alert_msg}. Series become daily on: {'; '.join(daily_transition_notes)}"
        alert_color = "orange" if daily_transition_notes else "green"
        alert_hide = False
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
            True,
            current_bench or {},
            current_ls or {},
            current_order or [],
            new_first_load,
            [],
            current_vol_scaling or {},
            False,
            [],
        )
    except Exception as e:
        return (
            n_no, n_no, n_no, n_no, n_no,
            n_no,
            f"Error loading database series: {str(e)}",
            "red",
            False,
            n_no, n_no, n_no, n_no, n_no,
            n_no, n_no, n_no,
            True, n_no,
        )


@callback(
    Output("analyticstool-raw-data-store", "data"),
    Output("analyticstool-original-periodicity-store", "data"),
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
    Output("ui-blocker-store", "data", allow_duplicate=True),
    Output("ui-blocker-timeout", "disabled", allow_duplicate=True),
    # Sheet-select modal outputs
    Output("sheet-select-modal", "opened", allow_duplicate=True),
    Output("sheet-select-dropdown", "data", allow_duplicate=True),
    Output("sheet-select-dropdown", "value", allow_duplicate=True),
    Output("sheet-select-contents-store", "data", allow_duplicate=True),
    Output("sheet-select-filename-store", "data", allow_duplicate=True),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    State("analyticstool-raw-data-store", "data"),
    State("analyticstool-original-periodicity-store", "data"),
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

    n_no = no_update
    # Sheet-select outputs default to no_update
    sheet_no = (n_no, n_no, n_no, n_no, n_no)

    try:
        # Check for multi-tab Excel files
        sheet_names = get_sheet_names(contents, filename)
        if len(sheet_names) > 1:
            # Stash contents and open the sheet-select modal
            dropdown_data = [{"value": s, "label": s} for s in sheet_names]
            return (
                n_no, n_no, n_no, n_no, n_no, n_no,
                n_no, n_no, True,  # hide alert
                n_no, n_no, n_no, n_no, n_no,
                n_no, n_no, n_no,
                False, True,  # hide blocker
                True, dropdown_data, sheet_names[0], contents, filename,  # open sheet modal
            )

        # Parse the uploaded file
        new_df = parse_uploaded_file(contents, filename)
        new_periodicity = detect_periodicity(new_df)

        # Determine the effective periodicity
        if existing_data is not None:
            existing_df = json_to_df(existing_data)

            # Check periodicity compatibility and resample if needed
            if existing_periodicity == "monthly" and new_periodicity == "daily":
                # Resample new daily data to monthly before appending
                new_df = resample_returns(new_df, "monthly")
                combined_periodicity = "monthly"
            elif new_periodicity == "monthly" and existing_periodicity == "daily":
                # If new data is monthly but existing is daily, convert existing to monthly
                existing_df = resample_returns(existing_df, "monthly")
                combined_periodicity = "monthly"
            else:
                combined_periodicity = existing_periodicity

            # Merge the data
            existing_df = _normalize_monthly_df_if_needed(existing_df, combined_periodicity)
            new_df = _normalize_monthly_df_if_needed(new_df, combined_periodicity)
            merged_df = merge_returns(existing_df, new_df)
        else:
            merged_df = new_df
            combined_periodicity = new_periodicity
            merged_df = _normalize_monthly_df_if_needed(merged_df, combined_periodicity)

        # Get available periodicities
        periodicity_options = get_available_periodicities(combined_periodicity)
        default_periodicity = "daily_trading" if combined_periodicity == "daily" else combined_periodicity

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
            False, True, # Hide blocker
            *sheet_no,
        )

    except Exception as e:
        return (
            n_no, n_no, n_no, n_no, n_no,
            n_no,
            f"Error loading file: {str(e)}",
            "red",
            False,
            n_no, n_no, n_no, n_no, n_no,
            n_no, n_no, n_no,
            False, True, # Hide blocker
            *sheet_no,
        )


# ---------------------------------------------------------------------------
# Sheet selection modal: confirm
# ---------------------------------------------------------------------------
@callback(
    Output("analyticstool-raw-data-store", "data", allow_duplicate=True),
    Output("analyticstool-original-periodicity-store", "data", allow_duplicate=True),
    Output("periodicity-select", "data", allow_duplicate=True),
    Output("periodicity-select", "value", allow_duplicate=True),
    Output("periodicity-select", "disabled", allow_duplicate=True),
    Output("temp-series-select", "data", allow_duplicate=True),
    Output("alert-message", "children", allow_duplicate=True),
    Output("alert-message", "color", allow_duplicate=True),
    Output("alert-message", "hide", allow_duplicate=True),
    Output("periodicity-value-store", "data", allow_duplicate=True),
    Output("series-selection-modal", "opened", allow_duplicate=True),
    Output("temp-benchmark-assignments-store", "data", allow_duplicate=True),
    Output("temp-long-short-store", "data", allow_duplicate=True),
    Output("temp-series-order-store", "data", allow_duplicate=True),
    Output("first-load-store", "data", allow_duplicate=True),
    Output("temp-deleted-series-store", "data", allow_duplicate=True),
    Output("temp-vol-scaling-assignments-store", "data", allow_duplicate=True),
    Output("ui-blocker-store", "data", allow_duplicate=True),
    Output("ui-blocker-timeout", "disabled", allow_duplicate=True),
    Output("sheet-select-modal", "opened", allow_duplicate=True),
    Output("sheet-select-contents-store", "data", allow_duplicate=True),
    Output("sheet-select-filename-store", "data", allow_duplicate=True),
    Output("upload-data", "contents", allow_duplicate=True),
    Input("sheet-select-ok-button", "n_clicks"),
    State("sheet-select-dropdown", "value"),
    State("sheet-select-contents-store", "data"),
    State("sheet-select-filename-store", "data"),
    State("analyticstool-raw-data-store", "data"),
    State("analyticstool-original-periodicity-store", "data"),
    State("series-select", "data"),
    State("benchmark-assignments-store", "data"),
    State("long-short-store", "data"),
    State("series-order-store", "data"),
    State("first-load-store", "data"),
    State("vol-scaling-assignments-store", "data"),
    prevent_initial_call=True,
)
def on_sheet_select_ok(n_clicks, selected_sheet, stashed_contents, stashed_filename,
                       existing_data, existing_periodicity, current_selection,
                       current_bench, current_ls, current_order, first_load, current_vol_scaling):
    """Parse the selected sheet and complete the import."""
    if not n_clicks or not stashed_contents:
        raise PreventUpdate

    n_no = no_update
    try:
        new_df = parse_uploaded_file(stashed_contents, stashed_filename, sheet_name=selected_sheet)
        new_periodicity = detect_periodicity(new_df)
        filename = stashed_filename

        if existing_data is not None:
            existing_df = json_to_df(existing_data)
            if existing_periodicity == "monthly" and new_periodicity == "daily":
                new_df = resample_returns(new_df, "monthly")
                combined_periodicity = "monthly"
            elif new_periodicity == "monthly" and existing_periodicity == "daily":
                existing_df = resample_returns(existing_df, "monthly")
                combined_periodicity = "monthly"
            else:
                combined_periodicity = existing_periodicity
            existing_df = _normalize_monthly_df_if_needed(existing_df, combined_periodicity)
            new_df = _normalize_monthly_df_if_needed(new_df, combined_periodicity)
            merged_df = merge_returns(existing_df, new_df)
        else:
            merged_df = new_df
            combined_periodicity = new_periodicity
            merged_df = _normalize_monthly_df_if_needed(merged_df, combined_periodicity)

        periodicity_options = get_available_periodicities(combined_periodicity)
        default_periodicity = "daily_trading" if combined_periodicity == "daily" else combined_periodicity

        new_series = [col for col in new_df.columns if col not in (current_selection or [])]
        updated_selection = (current_selection or []) + new_series

        if not first_load:
            alert_msg = f"Loaded {len(new_df.columns)} series with {len(new_df)} rows from {filename} (sheet: {selected_sheet})"
            alert_color = "green"
            alert_hide = False
            new_first_load = True
        else:
            alert_msg = n_no
            alert_color = n_no
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
            True,  # Open series-selection modal
            current_bench or {},
            current_ls or {},
            current_order or [],
            new_first_load,
            [],
            current_vol_scaling or {},
            False, True,  # Hide blocker
            False, None, None, None,  # Close sheet modal, clear stash, reset upload
        )

    except Exception as e:
        return (
            n_no, n_no, n_no, n_no, n_no,
            n_no,
            f"Error loading file: {str(e)}",
            "red",
            False,
            n_no, n_no, n_no, n_no, n_no,
            n_no, n_no, n_no,
            False, True,  # Hide blocker
            False, None, None, None,  # Close sheet modal, clear stash, reset upload
        )


# ---------------------------------------------------------------------------
# Sheet selection modal: cancel
# ---------------------------------------------------------------------------
@callback(
    Output("sheet-select-modal", "opened", allow_duplicate=True),
    Output("sheet-select-contents-store", "data", allow_duplicate=True),
    Output("sheet-select-filename-store", "data", allow_duplicate=True),
    Output("upload-data", "contents", allow_duplicate=True),
    Output("ui-blocker-store", "data", allow_duplicate=True),
    Output("ui-blocker-timeout", "disabled", allow_duplicate=True),
    Input("sheet-select-cancel-button", "n_clicks"),
    prevent_initial_call=True,
)
def on_sheet_select_cancel(n_clicks):
    """Cancel sheet selection and clear stashed data."""
    if not n_clicks:
        raise PreventUpdate
    return False, None, None, None, False, True


# Clear the file input so the same file can be re-uploaded
clientside_callback(
    """
    function(opened) {
        if (!opened) {
            var el = document.getElementById('upload-data');
            if (el) {
                var inp = el.querySelector('input[type="file"]');
                if (inp) inp.value = '';
            }
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("sheet-select-modal", "title", allow_duplicate=True),
    Input("sheet-select-modal", "opened"),
    prevent_initial_call=True,
)


@callback(
    Output("series-selection-container", "children"),
    Output("temp-series-order-store", "data", allow_duplicate=True),
    Input("analyticstool-raw-data-store", "data"),
    Input("temp-series-select", "data"),
    Input("temp-series-order-store", "data"),
    Input("temp-deleted-series-store", "data"),
    Input("series-selection-grid", "cellValueChanged", allow_optional=True),
    Input("temp-benchmark-assignments-store", "data"),
    Input("temp-long-short-store", "data"),
    Input("temp-vol-scaling-assignments-store", "data"),
    prevent_initial_call="initial_duplicate",
)
def update_series_selectors(
    raw_data,
    selected_series,
    series_order,
    deleted_series,
    _cell_change,
    current_assignments,
    long_short_assignments,
    vol_scaling_assignments,
):
    """Render Select Series as a single AG Grid with in-grid controls."""
    if raw_data is None:
        return [dmc.Text("Upload data to select series", size="sm", c="dimmed")], []

    df = json_to_df(raw_data)
    all_series = list(df.columns)
    if not all_series:
        return [dmc.Text("Upload data to select series", size="sm", c="dimmed")], []

    if not series_order:
        series_order = list(all_series)
    else:
        for series in all_series:
            if series not in series_order:
                series_order.append(series)
        series_order = [s for s in series_order if s in all_series]

    deleted_set = set(deleted_series or [])
    selected_set = set(selected_series or [])
    current_assignments = current_assignments or {}
    long_short_assignments = long_short_assignments or {}
    vol_scaling_assignments = vol_scaling_assignments or {}

    benchmark_values = ["None"] + list(all_series)
    row_data = []
    for series in series_order:
        benchmark_value = current_assignments.get(series, "None")
        if benchmark_value not in all_series and benchmark_value != "None":
            benchmark_value = "None"
        row_data.append(
            {
                "Series": series,
                "Benchmark": benchmark_value,
                "LongShort": bool(long_short_assignments.get(series, False)),
                "ScaleVol": bool(vol_scaling_assignments.get(series, True)),
                "Delete": series in deleted_set,
            }
        )

    selected_rows = [
        row
        for row in row_data
        if row["Series"] in selected_set and not row["Delete"]
    ]

    grid = dag.AgGrid(
        id="series-selection-grid",
        className="ag-theme-alpine dashmat-series-modal-grid",
        getRowId="params.data.Series",
        columnDefs=[
            {
                "headerName": "",
                "rowDrag": True,
                "editable": False,
                "sortable": False,
                "filter": False,
                "resizable": False,
                "width": 36,
                "pinned": "left",
                "valueGetter": {"function": "''"},
                "cellClass": "dashmat-series-center-cell",
            },
            {
                "headerName": "",
                "checkboxSelection": True,
                "headerCheckboxSelection": True,
                "editable": False,
                "sortable": False,
                "filter": False,
                "resizable": False,
                "width": 56,
                "pinned": "left",
                "cellClass": "dashmat-series-center-cell",
            },
            {
                "field": "Series",
                "editable": True,
                "minWidth": 150,
                "cellStyle": {"textAlign": "left", "fontFamily": "monospace"},
                "headerClass": "dashmat-left-header",
            },
            {
                "field": "Benchmark",
                "editable": True,
                "cellEditor": "agRichSelectCellEditor",
                "cellEditorPopup": True,
                "cellEditorParams": {
                    "values": benchmark_values,
                    "allowTyping": True,
                    "filterList": True,
                    "highlightMatch": True,
                },
                "minWidth": 150,
                "cellStyle": {"textAlign": "left"},
                "headerClass": "dashmat-left-header",
            },
            {
                "field": "LongShort",
                "headerName": "L/S",
                "editable": True,
                "cellRenderer": "agCheckboxCellRenderer",
                "cellEditor": "agCheckboxCellEditor",
                "width": 72,
                "cellClass": "dashmat-series-center-cell",
            },
            {
                "field": "ScaleVol",
                "headerName": "Scale Vol",
                "editable": True,
                "cellRenderer": "agCheckboxCellRenderer",
                "cellEditor": "agCheckboxCellEditor",
                "width": 106,
                "cellClass": "dashmat-series-center-cell",
            },
            {
                "field": "Delete",
                "editable": True,
                "cellRenderer": "agCheckboxCellRenderer",
                "cellEditor": "agCheckboxCellEditor",
                "width": 78,
                "cellClass": "dashmat-series-center-cell",
            },
        ],
        rowData=row_data,
        selectedRows=selected_rows,
        defaultColDef={
            "resizable": True,
            "sortable": False,
            "filter": False,
            "suppressHeaderMenuButton": True,
            "suppressMovable": True,
            "cellStyle": {"textAlign": "center"},
            "headerClass": "dashmat-center-header",
        },
        style={"height": "46vh", "width": "100%"},
        dashGridOptions={
            "rowSelection": "multiple",
            "rowMultiSelectWithClick": False,
            "suppressRowClickSelection": True,
            "suppressRowDeselection": True,
            "suppressMovableColumns": True,
            "rowDragManaged": True,
            "animateRows": True,
            "singleClickEdit": True,
            "stopEditingWhenCellsLoseFocus": True,
            "suppressExcelExport": True,
            "suppressCsvExport": True,
        },
        enableEnterpriseModules=True,
        licenseKey=AG_GRID_LICENSE_KEY,
    )
    return [grid], series_order


@callback(
    Output("temp-deleted-series-store", "data", allow_duplicate=True),
    Input("series-selection-grid", "cellValueChanged", allow_optional=True),
    State("series-selection-grid", "rowData", allow_optional=True),
    prevent_initial_call=True,
)
def delete_series(cell_change, row_data):
    """Sync staged deletions from the grid Delete checkbox column."""
    change = cell_change
    if isinstance(change, list):
        change = next((item for item in reversed(change) if isinstance(item, dict)), None)
    if not isinstance(change, dict) or change.get("colId") != "Delete":
        raise PreventUpdate

    rows = row_data or []
    deleted = [
        row.get("Series")
        for row in rows
        if isinstance(row, dict) and row.get("Series") and bool(row.get("Delete"))
    ]
    return deleted


@callback(
    Output("analyticstool-raw-data-store", "data", allow_duplicate=True),
    Output("temp-benchmark-assignments-store", "data", allow_duplicate=True),
    Output("temp-long-short-store", "data", allow_duplicate=True),
    Output("temp-vol-scaling-assignments-store", "data", allow_duplicate=True),
    Output("temp-series-select", "data", allow_duplicate=True),
    Output("temp-series-order-store", "data", allow_duplicate=True),
    Output("series-edit-mode", "data", allow_duplicate=True),
    Output("series-select-value-store", "data", allow_duplicate=True),
    Output("edit-box-focus-trigger", "data", allow_duplicate=True),
    Input("series-selection-grid", "cellValueChanged", allow_optional=True),
    State("analyticstool-raw-data-store", "data"),
    State("temp-benchmark-assignments-store", "data"),
    State("temp-long-short-store", "data"),
    State("temp-vol-scaling-assignments-store", "data"),
    State("temp-series-select", "data"),
    State("temp-series-order-store", "data"),
    prevent_initial_call=True,
)
def save_edit(
    cell_change,
    raw_data,
    benchmark_assignments,
    long_short_assignments,
    vol_scaling_assignments,
    selected_series,
    series_order,
):
    """Save an in-grid Series rename and cascade key updates across stores."""
    change = cell_change
    if isinstance(change, list):
        change = next((item for item in reversed(change) if isinstance(item, dict)), None)
    if not isinstance(change, dict) or change.get("colId") != "Series":
        raise PreventUpdate

    old_name = str(change.get("oldValue", "")).strip()
    new_name = str(change.get("newValue", "")).strip()
    if not old_name or not new_name or new_name == old_name:
        raise PreventUpdate

    # Check if new name already exists
    df = json_to_df(raw_data)
    if old_name not in df.columns or new_name in df.columns:
        raise PreventUpdate

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

    # Update series selection (handling the rename)
    selected_series = selected_series or []
    new_series_select = [new_name if s == old_name else s for s in selected_series]

    # Update series order
    series_order = series_order or list(df.columns)
    new_series_order = [new_name if s == old_name else s for s in series_order]

    # Return updated data and exit edit mode
    return new_raw_data, new_benchmark_assignments, new_long_short_assignments, new_vol_scaling_assignments, new_series_select, new_series_order, None, new_series_select, None


@callback(
    Output("temp-benchmark-assignments-store", "data"),
    Input("series-selection-grid", "cellValueChanged", allow_optional=True),
    State("series-selection-grid", "rowData", allow_optional=True),
    State("analyticstool-raw-data-store", "data"),
    prevent_initial_call=True,
)
def update_benchmark_assignments(cell_change, row_data, raw_data):
    """Store benchmark assignments for all series."""
    if raw_data is None or not row_data:
        return {}

    valid_series = set(json_to_df(raw_data).columns)
    assignments = {}
    for row in row_data:
        if not isinstance(row, dict):
            continue
        series = row.get("Series")
        benchmark = row.get("Benchmark", "None")
        if not series or series not in valid_series:
            continue
        if benchmark not in valid_series and benchmark != "None":
            benchmark = "None"
        assignments[series] = benchmark

    return assignments


@callback(
    Output("temp-long-short-store", "data"),
    Input("series-selection-grid", "cellValueChanged", allow_optional=True),
    State("series-selection-grid", "rowData", allow_optional=True),
    State("analyticstool-raw-data-store", "data"),
    prevent_initial_call=True,
)
def update_long_short_assignments(cell_change, row_data, raw_data):
    """Store long-short checkbox assignments for all series."""
    if raw_data is None or not row_data:
        return {}

    valid_series = set(json_to_df(raw_data).columns)
    assignments = {}
    for row in row_data:
        if not isinstance(row, dict):
            continue
        series = row.get("Series")
        if not series or series not in valid_series:
            continue
        assignments[series] = bool(row.get("LongShort", False))

    return assignments


@callback(
    Output("temp-vol-scaling-assignments-store", "data"),
    Input("series-selection-grid", "cellValueChanged", allow_optional=True),
    State("series-selection-grid", "rowData", allow_optional=True),
    State("analyticstool-raw-data-store", "data"),
    prevent_initial_call=True,
)
def update_vol_scaling_assignments(cell_change, row_data, raw_data):
    """Store vol-scaling checkbox assignments for all series."""
    if raw_data is None or not row_data:
        return {}

    valid_series = set(json_to_df(raw_data).columns)
    assignments = {}
    for row in row_data:
        if not isinstance(row, dict):
            continue
        series = row.get("Series")
        if not series or series not in valid_series:
            continue
        assignments[series] = bool(row.get("ScaleVol", True))

    return assignments


@callback(
    Output("start-date-picker", "value"),
    Output("end-date-picker", "value"),
    Output("date-picker-wrapper", "style"),
    Output("common-range-button", "disabled"),
    Output("common-daily-button", "disabled"),
    Output("maximum-range-button", "disabled"),
    Output("date-range-store", "data", allow_duplicate=True),
    Input("analyticstool-raw-data-store", "data"),
    Input("periodicity-select", "value"),
    Input("series-select", "data"),
    State("date-range-store", "data"),
    prevent_initial_call="initial_duplicate",
)
def initialize_date_range(raw_data, periodicity, selected_series, stored_range):
    """Initialize date range to maximum range when data is loaded."""
    disabled_style = {"display": "flex", "opacity": 0.5, "pointerEvents": "none", "alignItems": "flex-start"}
    enabled_style = {"display": "flex", "alignItems": "flex-start"}

    if raw_data is None or not selected_series:
        return None, None, disabled_style, True, True, True, None

    try:
        df = resample_returns_cached(raw_data, periodicity or "daily")

        # Filter to selected series
        available_series = [s for s in selected_series if s in df.columns]
        if not available_series:
            return None, None, disabled_style, True, True, True, None

        daily_df = resample_returns_cached(raw_data, "daily_trading")
        daily_available = [s for s in selected_series if s in daily_df.columns]
        has_common_daily = bool(get_common_daily_range(daily_df, daily_available)) if daily_available else False

        # Get maximum range (earliest start, latest end)
        data_start = df.index.min().strftime("%Y-%m-%d")
        data_end = df.index.max().strftime("%Y-%m-%d")

        # Use stored dates if they fall within the available data range
        if stored_range and stored_range.get("start") and stored_range.get("end"):
            s = stored_range["start"]
            e = stored_range["end"]
            if s >= data_start and e <= data_end:
                return s, e, enabled_style, False, not has_common_daily, False, {"start": s, "end": e}

        return data_start, data_end, enabled_style, False, not has_common_daily, False, {"start": data_start, "end": data_end}

    except Exception:
        return None, None, disabled_style, True, True, True, None


@callback(
    Output("start-date-picker", "value", allow_duplicate=True),
    Output("end-date-picker", "value", allow_duplicate=True),
    Output("date-range-store", "data"),
    Output("periodicity-select", "value", allow_duplicate=True),
    Output("periodicity-value-store", "data", allow_duplicate=True),
    Input("common-range-button", "n_clicks"),
    Input("common-daily-button", "n_clicks"),
    Input("maximum-range-button", "n_clicks"),
    State("analyticstool-raw-data-store", "data"),
    State("periodicity-select", "value"),
    State("series-select", "data"),
    prevent_initial_call=True,
)
def update_date_range_buttons(common_clicks, common_daily_clicks, max_clicks, raw_data, periodicity, selected_series):
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
            periodicity_value = no_update
        elif button_id == "common-daily-button":
            daily_df = resample_returns_cached(raw_data, "daily_trading")
            daily_available = [s for s in selected_series if s in daily_df.columns]
            common_daily = get_common_daily_range(daily_df, daily_available)
            if not common_daily:
                raise PreventUpdate
            start_date = common_daily[0].strftime("%Y-%m-%d")
            end_date = common_daily[1].strftime("%Y-%m-%d")
            periodicity_value = "daily_trading"
        else:  # maximum-range-button
            # Maximum range: earliest start to latest end across all selected series
            start_date = df.index.min().strftime("%Y-%m-%d")
            end_date = df.index.max().strftime("%Y-%m-%d")
            periodicity_value = no_update

        date_range = {"start": start_date, "end": end_date}
        return start_date, end_date, date_range, periodicity_value, periodicity_value

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
    Input("analyticstool-raw-data-store", "data"),
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
            _mapping_payload(benchmark_assignments),  # Convert to string for cache key
            returns_type,
            _mapping_payload(long_short_assignments),  # Convert to string for cache key
            _date_range_payload(date_range),  # Convert to string for cache key
            vol_scaler or 0,
            _mapping_payload(vol_scaling_assignments)
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
    Input("analyticstool-raw-data-store", "data"),
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
            _mapping_payload(benchmark_assignments),
            _mapping_payload(long_short_assignments),
            _date_range_payload(date_range),
            rolling_window,
            rolling_return_type,
            rolling_metric or "total_return",
            vol_scaler or 0,
            _mapping_payload(vol_scaling_assignments)
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
    Input("analyticstool-raw-data-store", "data"),
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
    State("theme-store", "data"),
    prevent_initial_call=True,
)
def update_rolling_chart(active_tab, raw_data, periodicity, selected_series, rolling_window, rolling_return_type, rolling_metric, benchmark_assignments, long_short_assignments, date_range, vol_scaler, vol_scaling_assignments, theme):
    """Update the Rolling Returns chart with rolling window calculations."""
    # Create empty figure
    empty_fig = go.Figure()
    empty_fig.update_layout(
        title="",
        xaxis_title="",
        yaxis_title="",
        template="plotly_white",
    )
    apply_chart_theme(empty_fig, theme)
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
            _mapping_payload(benchmark_assignments),
            _mapping_payload(long_short_assignments),
            _date_range_payload(date_range),
            rolling_window,
            rolling_return_type,
            rolling_metric or "total_return",
            vol_scaler or 0,
            _mapping_payload(vol_scaling_assignments)
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
        apply_chart_theme(fig, theme)

        return dcc.Graph(figure=fig, style={"height": "100%"})

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
    Input("analyticstool-raw-data-store", "data"),
    Input("analyticstool-original-periodicity-store", "data"),
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
                _mapping_payload(benchmark_assignments),
                _mapping_payload(long_short_assignments),
                selected_series,
                _date_range_payload(date_range),
                vol_scaler or 0,
                _mapping_payload(vol_scaling_assignments)
            )

        else:
            # Calculate calendar returns for the selected periodicity
            calendar_returns = calculate_calendar_year_returns(
                raw_data,
                original_periodicity,
                selected_periodicity,
                selected_series,
                returns_type,
                _mapping_payload(benchmark_assignments),
                _mapping_payload(long_short_assignments),
                _date_range_payload(date_range),
                vol_scaler or 0,
                _mapping_payload(vol_scaling_assignments)
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

            # Calculate max absolute value for conditional formatting gradient
            max_abs = 0
            for row in row_data:
                for key, val in row.items():
                    if key != "Year" and val is not None:
                        max_abs = max(max_abs, abs(val))

            # Build styleConditions for green/red gradient (10 bins)
            style_conditions = []
            if max_abs > 0:
                n_bins = 10
                for i in range(n_bins):
                    lo = max_abs * i / n_bins
                    hi = max_abs * (i + 1) / n_bins
                    alpha = round(0.1 + 0.6 * (i + 1) / n_bins, 2)
                    text_color = "#fff" if alpha > 0.4 else "inherit"
                    # Positive bins
                    if i == n_bins - 1:
                        style_conditions.append({
                            "condition": f"params.value >= {lo}",
                            "style": {"backgroundColor": f"rgba(34, 139, 34, {alpha})", "color": text_color, "textAlign": "center"},
                        })
                    else:
                        style_conditions.append({
                            "condition": f"params.value >= {lo} && params.value < {hi}",
                            "style": {"backgroundColor": f"rgba(34, 139, 34, {alpha})", "color": text_color, "textAlign": "center"},
                        })
                    if i == n_bins - 1:
                        style_conditions.append({
                            "condition": f"params.value <= {-lo}",
                            "style": {"backgroundColor": f"rgba(220, 38, 38, {alpha})", "color": text_color, "textAlign": "center"},
                        })
                    else:
                        style_conditions.append({
                            "condition": f"params.value <= {-lo} && params.value > {-hi}",
                            "style": {"backgroundColor": f"rgba(220, 38, 38, {alpha})", "color": text_color, "textAlign": "center"},
                        })

            cell_style = {"styleConditions": style_conditions, "defaultStyle": {"textAlign": "center"}} if style_conditions else {"textAlign": "center"}

            # Create column definitions with conditional formatting
            column_defs = [
                {
                    "field": "Year",
                    "pinned": "left",
                    "width": 100,
                    "cellStyle": {"textAlign": "center"},
                    "headerClass": "dashmat-center-header",
                }
            ]

            for series in selected_series:
                if series in calendar_returns:
                    col_def = {
                        "field": series,
                        "valueFormatter": {"function": "params.value != null ? d3.format('.2%')(params.value) : ''"},
                        "width": 120,
                        "headerClass": "dashmat-center-header",
                    }
                    if cell_style:
                        col_def["cellStyle"] = cell_style
                    column_defs.append(col_def)

            return column_defs, row_data

    except Exception:
        return [], []


@callback(
    Output("statistics-grid", "columnDefs"),
    Output("statistics-grid", "rowData"),
    Input("analyticstool-raw-data-store", "data"),
    Input("periodicity-select", "value"),
    Input("series-select", "data"),
    Input("benchmark-assignments-store", "data"),
    Input("long-short-store", "data"),
    Input("date-range-store", "data"),
    Input("vol-scaler-value-store", "data"),
    Input("vol-scaling-assignments-store", "data"),
    Input("analyticstool-saved-series-cache-store", "data"),
    prevent_initial_call=True,
)
def update_statistics(raw_data, periodicity, selected_series, benchmark_assignments, long_short_assignments, date_range, vol_scaler, vol_scaling_assignments, saved_series_store):
    """Update the Statistics grid with transposed data (optimized with caching)."""
    if raw_data is None or not selected_series:
        return [], []

    try:
        # Use cached function to avoid repeated computation
        stats = calculate_statistics_cached(
            raw_data,
            periodicity or "daily",
            tuple(selected_series),
            _mapping_payload(benchmark_assignments),
            _mapping_payload(long_short_assignments),
            _date_range_payload(date_range),
            vol_scaler or 0,
            _mapping_payload(vol_scaling_assignments),
            _risk_free_json_from_store(saved_series_store),
            _spx_json_from_store(saved_series_store),
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
    Output("correlogram-meta-store", "data"),
    Input("series-select", "data"),
    Input("main-tabs", "value"),
)
def update_correlogram_meta(selected_series, active_tab):
    """Update correlogram metadata (num_series) when tab is active."""
    if active_tab != "correlogram" or not selected_series:
        return no_update
    return {"num_series": len(selected_series)}


clientside_callback(
    """
    function(meta) {
        if (!meta || !meta.num_series || meta.num_series <= 1) {
            return dash_clientside.no_update;
        }

        var container = document.getElementById('correlogram-container');
        if (!container) {
            return dash_clientside.no_update;
        }
        
        var container_width = container.clientWidth;
        if (!container_width) return dash_clientside.no_update;

        // Default strategy: Clamp between 100 and 200, based on (Container - Buffer) / N
        // This ensures we fill the window if possible, but respect min 100px and max 200px defaults.
        var available_width = container_width - 40;
        var default_width = Math.floor(available_width / meta.num_series);
        
        if (default_width < 100) {
            default_width = 100;
        } else if (default_width > 200) {
            default_width = 200;
        }
        
        return default_width;
    }
    """,
    Output("correlogram-block-width", "value"),
    Input("correlogram-meta-store", "data"),
)


@callback(
    Output("correlogram-container", "children"),
    Input("main-tabs", "value"),  # Lazy loading: only update when tab is active
    Input("analyticstool-raw-data-store", "data"),
    Input("periodicity-select", "value"),
    Input("series-select", "data"),
    Input("returns-type-select", "value"),
    Input("benchmark-assignments-store", "data"),
    Input("long-short-store", "data"),
    Input("date-range-store", "data"),
    Input("vol-scaler-value-store", "data"),
    Input("vol-scaling-assignments-store", "data"),
    Input("correlation-view-switch", "value"),
    Input("correlogram-block-width", "value"),
    State("theme-store", "data"),
    prevent_initial_call=True,
)
def update_correlogram(active_tab, raw_data, periodicity, selected_series, returns_type, benchmark_assignments, long_short_assignments, date_range, vol_scaler, vol_scaling_assignments, correlation_view, block_width, theme):
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
    empty_graph = dcc.Graph(figure=empty_fig, style={"height": "100%"})

    # Lazy loading: only generate when correlogram tab is active
    if active_tab != "correlogram":
        raise PreventUpdate

    if raw_data is None or not selected_series or len(selected_series) < 2:
        return empty_graph

    try:
        result = generate_correlogram_cached(
            raw_data,
            periodicity or "daily",
            tuple(selected_series),
            returns_type,
            _mapping_payload(benchmark_assignments),
            _mapping_payload(long_short_assignments),
            _date_range_payload(date_range),
            vol_scaler or 0,
            _mapping_payload(vol_scaling_assignments)
        )

        if result is None:
            return empty_graph

        available_series = result['available_series']
        corr_matrix = result['corr_matrix']

        # 1. Correlation Matrix (Heatmap)
        if correlation_view == "correlation":
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
                xaxis=dict(tickangle=45),
                yaxis=dict(autorange='reversed'),
                template="plotly_white",
            )
            apply_chart_theme(heatmap_fig, theme)

            return dcc.Graph(figure=heatmap_fig, style={"height": "100%"})

        # 2. Correlogram (Scatter Matrix)
        else:
            display_df = result['display_df']
            n = result['n']

            if n < 2:
                return empty_graph

            # Create subplots
            fig = make_subplots(
                rows=n, cols=n,
                horizontal_spacing=0.02,
                vertical_spacing=0.02,
                print_grid=False,
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
                                hoverinfo='skip',
                            ),
                            row=row_idx, col=col_idx
                        )
                        # Hide axes for upper triangle
                        fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False, row=row_idx, col=col_idx)
                        fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False, row=row_idx, col=col_idx)

            # Scaling logic: Fixed size based on user input
            # Always square blocks (N * block_width)
            user_block_width = block_width if block_width else 100
            total_size_px = len(available_series) * user_block_width
            
            graph_style = {
                "width": f"{total_size_px}px",
                "height": f"{total_size_px}px",
            }
            
            # Set explicit size on figure layout
            fig.update_layout(width=total_size_px, height=total_size_px, autosize=False)

            fig.update_layout(
                title=f"Scatter Matrix ({returns_type.title()} Returns)",
                showlegend=False,
                template="plotly_white",
                margin=dict(l=20, r=20, t=50, b=20),
            )
            
            # Update axes labels only on edges
            for i in range(n):
                # Bottom row x-axes
                fig.update_xaxes(title_text=available_series[i], row=n, col=i+1, title_font=dict(size=10))
                # Left col y-axes
                fig.update_yaxes(title_text=available_series[i], row=i+1, col=1, title_font=dict(size=10))
                
                # Hide internal tick labels
                if i < n-1:
                     fig.update_xaxes(showticklabels=False, row=i+1)
                if i > 0:
                     fig.update_yaxes(showticklabels=False, col=i+1)


            apply_chart_theme(fig, theme)
            return dcc.Graph(figure=fig, style=graph_style)

    except Exception:
        return empty_graph


@callback(
    Output("growth-charts-container", "children"),
    Input("main-tabs", "value"),
    Input("growth-chart-switch", "value"),
    Input("analyticstool-raw-data-store", "data"),
    Input("periodicity-select", "value"),
    Input("series-select", "data"),
    Input("benchmark-assignments-store", "data"),
    Input("long-short-store", "data"),
    Input("date-range-store", "data"),
    Input("vol-scaler-value-store", "data"),
    Input("vol-scaling-assignments-store", "data"),
    State("theme-store", "data"),
    prevent_initial_call=True,
)
def update_growth_charts(active_tab, chart_checked, raw_data, periodicity, selected_series, benchmark_assignments, long_short_assignments, date_range, vol_scaler, vol_scaling_assignments, theme):
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
            _mapping_payload(benchmark_assignments), _mapping_payload(long_short_assignments), _date_range_payload(date_range),
            vol_scaler or 0, _mapping_payload(vol_scaling_assignments)
        )

        if df.empty:
            return dmc.Text("No data available for selected series", size="sm", c="dimmed")

        benchmark_dict = json.loads(benchmark_assignments) if isinstance(benchmark_assignments, str) else (benchmark_assignments if isinstance(benchmark_assignments, dict) else {})
        long_short_dict = json.loads(long_short_assignments) if isinstance(long_short_assignments, str) else (long_short_assignments if isinstance(long_short_assignments, dict) else {})

        # Filter to selected series only
        available_series = [s for s in selected_series if s in df.columns]
        if not available_series:
            return dmc.Text("No data available for selected series", size="sm", c="dimmed")

        # Determine the period offset based on periodicity
        from utils.returns import is_daily
        periodicity_str = periodicity or "daily"
        if is_daily(periodicity_str):
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
            _mapping_payload(benchmark_assignments),
            _mapping_payload(long_short_assignments),
            _date_range_payload(date_range),
            vol_scaler or 0,
            _mapping_payload(vol_scaling_assignments)
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
            benchmark = benchmark_dict.get(series, None)
            is_long_short = long_short_dict.get(series, False)

            # Skip if benchmark is None or same as series
            if benchmark is None or benchmark == "None" or benchmark == series:
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

            apply_chart_theme(fig, theme)
            individual_charts.append(dcc.Graph(figure=fig, style={"marginBottom": "2rem"}))

        # Combine all charts
        apply_chart_theme(main_fig, theme)
        charts = [dcc.Graph(figure=main_fig, style={"height": "100%", "marginBottom": "3rem"})] + individual_charts

        return html.Div(charts, style={"height": "100%"})

    except Exception as e:
        return dmc.Text(f"Error generating growth charts: {str(e)}", size="sm", c="red")


@callback(
    Output("growth-grid", "columnDefs"),
    Output("growth-grid", "rowData"),
    Input("main-tabs", "value"),
    Input("growth-chart-switch", "value"),
    Input("analyticstool-raw-data-store", "data"),
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
            _mapping_payload(benchmark_assignments),
            _mapping_payload(long_short_assignments),
            _date_range_payload(date_range),
            vol_scaler or 0,
            _mapping_payload(vol_scaling_assignments)
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
                    "width": 120,
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
    Input("analyticstool-raw-data-store", "data"),
    Input("periodicity-select", "value"),
    Input("series-select", "data"),
    Input("returns-type-select", "value"),
    Input("benchmark-assignments-store", "data"),
    Input("long-short-store", "data"),
    Input("date-range-store", "data"),
    Input("vol-scaler-value-store", "data"),
    Input("vol-scaling-assignments-store", "data"),
    State("theme-store", "data"),
    prevent_initial_call=True,
)
def update_drawdown_charts(active_tab, chart_checked, raw_data, periodicity, selected_series, returns_type, benchmark_assignments, long_short_assignments, date_range, vol_scaler, vol_scaling_assignments, theme):
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
            _mapping_payload(benchmark_assignments),
            _mapping_payload(long_short_assignments),
            _date_range_payload(date_range),
            vol_scaler or 0,
            _mapping_payload(vol_scaling_assignments)
        )

        if drawdown_df.empty:
            return dmc.Text("No data available for selected series", size="sm", c="dimmed")

        long_short_dict = json.loads(long_short_assignments) if isinstance(long_short_assignments, str) else (long_short_assignments if isinstance(long_short_assignments, dict) else {})

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

            apply_chart_theme(fig, theme)
            charts.append(dcc.Graph(figure=fig, style={"marginBottom": "2rem"}))

        return html.Div(charts)

    except Exception as e:
        return dmc.Text(f"Error generating drawdown charts: {str(e)}", size="sm", c="red")


@callback(
    Output("drawdown-grid", "columnDefs"),
    Output("drawdown-grid", "rowData"),
    Input("main-tabs", "value"),
    Input("drawdown-chart-switch", "value"),
    Input("analyticstool-raw-data-store", "data"),
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
            _mapping_payload(benchmark_assignments),
            _mapping_payload(long_short_assignments),
            _date_range_payload(date_range),
            vol_scaler or 0,
            _mapping_payload(vol_scaling_assignments)
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
                    "width": 120,
                })

        # Convert to records
        row_data = drawdown_df.to_dict("records")

        return column_defs, row_data

    except Exception:
        return [], []



@callback(
    Output("download-excel", "data"),
    Input("menu-download-excel", "n_clicks"),
    State("analyticstool-raw-data-store", "data"),
    State("analyticstool-original-periodicity-store", "data"),
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
    State("analyticstool-saved-series-cache-store", "data"),
    prevent_initial_call=True,
)
def download_excel(n_clicks, raw_data, original_periodicity, selected_periodicity, selected_series, returns_type, benchmark_assignments, long_short_assignments, date_range, rolling_window, rolling_return_type, monthly_view, monthly_series, vol_scaler, vol_scaling_assignments, saved_series_store):
    """Generate Excel file with Statistics, Returns, Rolling, Calendar Year, Growth, Drawdown, and Correlogram sheets."""
    if n_clicks is None or raw_data is None or not selected_series:
        raise PreventUpdate

    with timed_block(
        "analyticstool.download_excel.total",
        series_count=len(selected_series or ()),
        returns_type=returns_type,
    ):
        bundle = _build_analytics_compute_bundle(
            raw_data,
            selected_periodicity,
            selected_series,
            benchmark_assignments,
            long_short_assignments,
            date_range,
            vol_scaler,
            vol_scaling_assignments,
        )

        # Use cached functions to get data
        with timed_block("analyticstool.download_excel.returns"):
            returns_df = calculate_excess_returns(
                bundle.raw_data,
                bundle.periodicity,
                bundle.selected_series,
                bundle.benchmark_payload,
                returns_type,
                bundle.long_short_payload,
                bundle.date_range_payload,
                bundle.vol_scaler,
                bundle.vol_scaling_payload,
            )

        if returns_df.empty:
            raise PreventUpdate

        # Get cached statistics
        with timed_block("analyticstool.download_excel.statistics"):
            stats = calculate_statistics_cached(
                bundle.raw_data,
                bundle.periodicity,
                bundle.selected_series,
                bundle.benchmark_payload,
                bundle.long_short_payload,
                bundle.date_range_payload,
                bundle.vol_scaler,
                bundle.vol_scaling_payload,
                _risk_free_json_from_store(saved_series_store),
                _spx_json_from_store(saved_series_store),
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
        with timed_block("analyticstool.download_excel.workbook"):
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                # Sheet 1: Statistics (moved to first position)
                stats_df.to_excel(writer, sheet_name="Statistics", index=False)

                # Sheet 2: Returns
                returns_df.to_excel(writer, sheet_name="Returns")

                # Sheet 3: Rolling (use current settings)
                try:
                    with timed_block("analyticstool.download_excel.rolling"):
                        # Use stored rolling options, default to 1y annualized if not set
                        window = rolling_window if rolling_window else "1y"
                        return_type = rolling_return_type if rolling_return_type else "annualized"

                        rolling_df = calculate_rolling_returns(
                            bundle.raw_data,
                            bundle.periodicity,
                            bundle.selected_series,
                            returns_type,
                            bundle.benchmark_payload,
                            bundle.long_short_payload,
                            bundle.date_range_payload,
                            window,
                            return_type,
                            "total_return", # Default metric for excel
                            bundle.vol_scaler,
                            bundle.vol_scaling_payload,
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
                        with timed_block("analyticstool.download_excel.calendar"):
                            # Check if monthly view is selected
                            if monthly_view == "monthly" and monthly_series and monthly_series in selected_series:
                                # Get monthly view data
                                _, row_data = create_monthly_view(
                                    bundle.raw_data,
                                    monthly_series,
                                    original_periodicity,
                                    bundle.periodicity,
                                    returns_type,
                                    bundle.benchmark_payload,
                                    bundle.long_short_payload,
                                    bundle.selected_series,
                                    bundle.date_range_payload,
                                    bundle.vol_scaler,
                                    bundle.vol_scaling_payload,
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
                                    bundle.raw_data,
                                    original_periodicity,
                                    bundle.periodicity,
                                    bundle.selected_series,
                                    returns_type,
                                    bundle.benchmark_payload,
                                    bundle.long_short_payload,
                                    bundle.date_range_payload,
                                    bundle.vol_scaler,
                                    bundle.vol_scaling_payload,
                                )
                                if not calendar_df.empty:
                                    calendar_df.to_excel(writer, sheet_name="Calendar Year")
                    except Exception:
                        pass  # Skip if calendar calculation fails

                # Sheet 5: Growth of $1
                try:
                    with timed_block("analyticstool.download_excel.growth"):
                        growth_df = calculate_growth_of_dollar(
                            bundle.raw_data,
                            bundle.periodicity,
                            bundle.selected_series,
                            bundle.benchmark_payload,
                            bundle.long_short_payload,
                            bundle.date_range_payload,
                            bundle.vol_scaler,
                            bundle.vol_scaling_payload,
                        )
                        if not growth_df.empty:
                            growth_df.to_excel(writer, sheet_name="Growth of $1")
                except Exception:
                    pass  # Skip if growth calculation fails

                # Sheet 6: Drawdown
                try:
                    with timed_block("analyticstool.download_excel.drawdown"):
                        drawdown_df = calculate_drawdown(
                            bundle.raw_data,
                            bundle.periodicity,
                            bundle.selected_series,
                            returns_type,
                            bundle.benchmark_payload,
                            bundle.long_short_payload,
                            bundle.date_range_payload,
                            bundle.vol_scaler,
                            bundle.vol_scaling_payload,
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


# Sample file download callbacks
@callback(
    Output("download-sample-daily", "data"),
    Input("download-sample-daily-btn", "n_clicks"),
    prevent_initial_call=True,
)
def download_sample_daily(n_clicks):
    """Download stored sample daily returns file."""
    if n_clicks is None:
        raise PreventUpdate

    return dcc.send_file(str(get_sample_file_path("daily")))


@callback(
    Output("download-sample-monthly", "data"),
    Input("download-sample-monthly-btn", "n_clicks"),
    prevent_initial_call=True,
)
def download_sample_monthly(n_clicks):
    """Download stored sample monthly returns file."""
    if n_clicks is None:
        raise PreventUpdate

    return dcc.send_file(str(get_sample_file_path("monthly")))

