"""Portfolio Optimization page for DashMat."""

import json

import dash_mantine_components as dmc
from dash_iconify import DashIconify
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import (
    Input, Output, State, callback, dcc, html, no_update,
    register_page, ALL, clientside_callback, callback_context,
)
from dash.exceptions import PreventUpdate

from utils.parsing import detect_periodicity, parse_uploaded_file
from utils.returns import (
    df_to_json,
    get_available_periodicities,
    get_working_returns,
    json_to_df,
    merge_returns,
    resample_returns,
    resample_returns_cached,
    annualization_factor,
)
from utils.optimization import run_portfolio_optimization

register_page(__name__, path="/portfolio", name="Portfolio Optimization", title="Portfolio Optimization")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _periodicity_defaults(periodicity):
    """Return (window_size, opt_step, halflife) defaults for a periodicity."""
    if periodicity and periodicity.startswith("weekly"):
        return 52, 52, 13
    if periodicity == "monthly":
        return 12, 12, 6
    return 252, 252, 63


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def build_po_welcome_screen():
    return dmc.Stack(
        align="center",
        justify="center",
        h=400,
        children=[
            DashIconify(icon="tabler:chart-pie", width=60, color="#adb5bd"),
            dmc.Text("Portfolio Optimization", size="xl", fw=500, c="dimmed", mt="md"),
            dmc.Text("Add series from file to begin", size="sm", c="dimmed"),
            dmc.Button(
                "Add series from file",
                leftSection=DashIconify(icon="tabler:upload"),
                variant="outline",
                mt="lg",
                id="po-welcome-add-series-btn",
            ),
        ],
    )


def build_po_main_layout():
    return html.Div(
        style={"display": "flex", "flexDirection": "column", "height": "100%", "overflow": "hidden"},
        children=[
            # Controls Accordion
            dmc.Accordion(
                value="controls",
                mb="xs",
                variant="contained",
                children=[
                    dmc.AccordionItem(
                        value="controls",
                        children=[
                            dmc.AccordionControl("Controls"),
                            dmc.AccordionPanel(children=[
                                dmc.Group(
                                    mb="md",
                                    align="flex-start",
                                    children=[
                                        html.Div([
                                            dmc.Text("Series Selection", size="sm", mb=3, fw=500),
                                            dmc.Button(
                                                "Select Series",
                                                id="po-open-modal-button",
                                                variant="light",
                                                size="sm",
                                                w=200,
                                            ),
                                        ]),
                                        dmc.Select(
                                            id="po-periodicity-select",
                                            label="Periodicity",
                                            data=[{"value": "daily", "label": "Daily"}],
                                            value="daily",
                                            w=200,
                                            disabled=False,
                                        ),
                                        html.Div([
                                            dmc.Text("Vol Scaler", size="sm", mb=3, fw=500),
                                            dmc.Tooltip(
                                                label="A value of 0% disables the volatility scaling.",
                                                position="top",
                                                withArrow=True,
                                                children=dmc.NumberInput(
                                                    id="po-vol-scaler-input",
                                                    value=0,
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
                                        id="po-date-picker-wrapper",
                                        children=[
                                            html.Div([
                                                dmc.DateInput(
                                                    id="po-start-date-picker",
                                                    label="Start Date",
                                                    value=None,
                                                    w=200,
                                                    valueFormat="YYYY-MM-DD",
                                                ),
                                            ], style={"marginRight": "15px"}),
                                            html.Div([
                                                dmc.DateInput(
                                                    id="po-end-date-picker",
                                                    label="End Date",
                                                    value=None,
                                                    w=200,
                                                    valueFormat="YYYY-MM-DD",
                                                ),
                                            ], style={"marginRight": "15px"}),
                                            html.Div([
                                                dmc.Button(
                                                    "Common Range",
                                                    id="po-common-range-button",
                                                    size="xs",
                                                    variant="outline",
                                                    disabled=True,
                                                    w=120,
                                                ),
                                            ], style={"marginRight": "10px", "alignSelf": "flex-end", "marginBottom": "2px"}),
                                            html.Div([
                                                dmc.Button(
                                                    "Max Range",
                                                    id="po-maximum-range-button",
                                                    size="xs",
                                                    variant="outline",
                                                    disabled=True,
                                                    w=120,
                                                ),
                                            ], style={"alignSelf": "flex-end", "marginBottom": "2px"}),
                                        ],
                                        style={"display": "flex", "opacity": 0.5, "pointerEvents": "none", "alignItems": "flex-start"},
                                    ),
                                ]),
                            ]),
                        ],
                    ),
                ],
            ),

            # Optimization Accordion
            dmc.Accordion(
                value="optimization",
                mb="xs",
                variant="contained",
                children=[
                    dmc.AccordionItem(
                        value="optimization",
                        children=[
                            dmc.AccordionControl("Optimization"),
                            dmc.AccordionPanel(children=[
                                dmc.Group(
                                    align="flex-end",
                                    gap="md",
                                    children=[
                                        html.Div([
                                            dmc.Text("Exp Wt Cov", size="sm", mb=3, fw=500),
                                            dmc.Switch(
                                                id="po-exp-wt-cov-switch",
                                                checked=False,
                                                size="sm",
                                            ),
                                        ]),
                                        html.Div(
                                            id="po-halflife-wrapper",
                                            style={"display": "none"},
                                            children=[
                                                dmc.NumberInput(
                                                    id="po-halflife-input",
                                                    label="Halflife",
                                                    value=63,
                                                    min=1,
                                                    step=1,
                                                    w=100,
                                                    size="sm",
                                                ),
                                            ],
                                        ),
                                        html.Div([
                                            dmc.Text("Missing Data", size="sm", mb=3, fw=500),
                                            dmc.SegmentedControl(
                                                id="po-missing-data-select",
                                                data=[
                                                    {"value": "fill_na", "label": "Fill NA"},
                                                    {"value": "fill_0", "label": "Fill 0"},
                                                ],
                                                value="fill_na",
                                                size="sm",
                                            ),
                                        ]),
                                        dmc.TextInput(
                                            id="po-portfolio-name-input",
                                            label="Portfolio Name",
                                            value="OptResult",
                                            w=150,
                                            size="sm",
                                        ),
                                        html.Div([
                                            dmc.Text("Window", size="sm", mb=3, fw=500),
                                            dmc.SegmentedControl(
                                                id="po-opt-window-select",
                                                data=[
                                                    {"value": "expanding", "label": "Expanding"},
                                                    {"value": "rolling", "label": "Rolling"},
                                                    {"value": "full", "label": "Full"},
                                                ],
                                                value="full",
                                                size="sm",
                                            ),
                                        ]),
                                        dmc.NumberInput(
                                            id="po-window-size-input",
                                            label="Window Size",
                                            value=252,
                                            min=2,
                                            step=1,
                                            w=110,
                                            size="sm",
                                            disabled=True,
                                        ),
                                        dmc.NumberInput(
                                            id="po-opt-step-input",
                                            label="Opt Step",
                                            value=252,
                                            min=1,
                                            step=1,
                                            w=100,
                                            size="sm",
                                            disabled=True,
                                        ),
                                        html.Div([
                                            dmc.Text("Fill In-Sample", size="sm", mb=3, fw=500),
                                            dmc.SegmentedControl(
                                                id="po-fill-in-sample-select",
                                                data=[
                                                    {"value": "off", "label": "Off"},
                                                    {"value": "on", "label": "On"},
                                                ],
                                                value="off",
                                                size="sm",
                                                disabled=True,
                                            ),
                                        ]),
                                        dmc.Select(
                                            id="po-opt-model-select",
                                            label="Model",
                                            data=[
                                                {"value": "risk_parity", "label": "Risk Parity"},
                                                {"value": "factor_risk_parity", "label": "Factor Risk Parity"},
                                                {"value": "hrp", "label": "HRP"},
                                                {"value": "minimize_cvar", "label": "Minimize CVaR"},
                                                {"value": "equal_weight", "label": "Equal Weight"},
                                            ],
                                            value="risk_parity",
                                            w=180,
                                            size="sm",
                                            clearable=False,
                                        ),
                                        dmc.Button(
                                            "Run Optimization",
                                            id="po-run-button",
                                            color="blue",
                                            size="sm",
                                            leftSection=DashIconify(icon="tabler:player-play"),
                                            disabled=True,
                                        ),
                                    ],
                                ),
                            ]),
                        ],
                    ),
                ],
            ),

            # Portfolio selector and visualization tabs
            dmc.Group(
                mb="xs",
                gap="md",
                children=[
                    dmc.Select(
                        id="po-weight-portfolio-select",
                        label="Portfolio",
                        data=[],
                        value=None,
                        w=200,
                        size="sm",
                        clearable=False,
                    ),
                    dmc.ActionIcon(
                        DashIconify(icon="tabler:trash", width=18),
                        id="po-delete-portfolio-button",
                        color="red",
                        variant="subtle",
                        size="lg",
                        style={"alignSelf": "flex-end"},
                    ),
                    html.Div(
                        id="po-growth-multiselect-wrapper",
                        style={"display": "none"},
                        children=[
                            dmc.MultiSelect(
                                id="po-growth-portfolio-multiselect",
                                label="Compare",
                                data=[],
                                value=[],
                                w=400,
                                size="sm",
                            ),
                        ],
                    ),
                ],
            ),

            dmc.Tabs(
                id="po-vis-tabs",
                value="weight",
                style={"flex": "1", "display": "flex", "flexDirection": "column", "overflow": "hidden"},
                children=[
                    dmc.TabsList(children=[
                        dmc.TabsTab("Weights", value="weight"),
                        dmc.TabsTab("Growth of $1", value="growth"),
                        dmc.TabsTab("Attribution", value="attribution"),
                    ]),
                    dmc.TabsPanel(
                        value="weight",
                        pt="md",
                        style={"flex": "1", "display": "flex", "flexDirection": "column", "overflow": "auto"},
                        children=[
                            dcc.Loading(
                                type="default",
                                children=[html.Div(id="po-weight-chart-container")],
                            ),
                        ],
                    ),
                    dmc.TabsPanel(
                        value="growth",
                        pt="md",
                        style={"flex": "1", "display": "flex", "flexDirection": "column", "overflow": "auto"},
                        children=[
                            dcc.Loading(
                                type="default",
                                children=[html.Div(id="po-growth-chart-container")],
                            ),
                        ],
                    ),
                    dmc.TabsPanel(
                        value="attribution",
                        pt="md",
                        style={"flex": "1", "display": "flex", "flexDirection": "column", "overflow": "auto"},
                        children=[
                            dcc.Loading(
                                type="default",
                                children=[html.Div(id="po-attribution-chart-container")],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Page Layout
# ---------------------------------------------------------------------------

layout = dmc.Container(
    fluid=True,
    style={"height": "calc(100vh - 55px)", "display": "flex", "flexDirection": "column", "overflow": "hidden"},
    children=[
        # Menu bar
        dmc.Paper(
            shadow="xs",
            p="xs",
            mb="md",
            withBorder=True,
            children=[
                dmc.Group(
                    gap="xs",
                    children=[
                        dmc.Menu(
                            trigger="hover",
                            openDelay=100,
                            closeDelay=200,
                            children=[
                                dmc.MenuTarget(dmc.Button("File", variant="subtle", size="sm")),
                                dmc.MenuDropdown(children=[
                                    dmc.MenuItem("Add series from file", id="po-menu-add-series"),
                                    dmc.MenuDivider(),
                                    dmc.MenuItem("Exit", id="po-menu-exit"),
                                ]),
                            ],
                        ),
                        dmc.Menu(
                            trigger="hover",
                            openDelay=100,
                            closeDelay=200,
                            children=[
                                dmc.MenuTarget(dmc.Button("Edit", variant="subtle", size="sm")),
                                dmc.MenuDropdown(children=[
                                    dmc.MenuItem("Clear all series", id="po-menu-clear-all-series"),
                                    dmc.MenuItem("Clear session storage and refresh", id="po-menu-clear-local-storage"),
                                ]),
                            ],
                        ),
                        dmc.Menu(
                            trigger="hover",
                            openDelay=100,
                            closeDelay=200,
                            children=[
                                dmc.MenuTarget(dmc.Button("View", variant="subtle", size="sm")),
                                dmc.MenuDropdown(children=[
                                    dmc.MenuItem("Analytics Tool", id="po-menu-view-analytics"),
                                ]),
                            ],
                        ),
                        dmc.Box(style={"flexGrow": 1}),
                    ],
                ),
            ],
        ),

        # Hidden upload
        html.Div(
            dcc.Upload(
                id="po-upload-data",
                children=html.Div(id="po-upload-trigger"),
                multiple=False,
                accept=".csv,.xlsx,.xls",
            ),
            style={"display": "none"},
        ),

        # Series selection modal
        dmc.Modal(
            id="po-series-selection-modal",
            title="Select Series",
            size="auto",
            centered=True,
            transitionProps={"transition": "fade", "duration": 200},
            children=[
                dmc.Alert(
                    id="po-alert-message",
                    title="Info",
                    color="blue",
                    hide=True,
                    mb="md",
                    withCloseButton=True,
                ),
                html.Div(
                    id="po-series-selection-container",
                    children=[dmc.Text("Upload data to select series", size="sm", c="dimmed")],
                    style={"maxHeight": "70vh", "overflowY": "auto", "overflowX": "auto"},
                ),
                dmc.Group(
                    mt="md",
                    justify="flex-end",
                    children=[
                        dmc.Button("Cancel", id="po-modal-cancel-button", variant="outline", color="red"),
                        dmc.Button("OK", id="po-modal-ok-button", color="blue"),
                    ],
                ),
            ],
        ),

        # Progress modal
        dmc.Modal(
            id="po-progress-modal",
            opened=False,
            closeOnClickOutside=False,
            withCloseButton=False,
            size="sm",
            children=[
                dmc.Stack(
                    align="center",
                    children=[
                        dmc.RingProgress(
                            id="po-progress-ring",
                            sections=[{"value": 0, "color": "blue"}],
                            size=120,
                            thickness=12,
                            label=dmc.Text("0%", ta="center", size="lg", fw=700),
                        ),
                        dmc.Text(id="po-progress-text", children="Starting..."),
                        dmc.Button("Cancel", id="po-cancel-button", color="red", variant="outline"),
                    ],
                ),
            ],
        ),

        # Completion modal
        dmc.Modal(
            id="po-completion-modal",
            opened=False,
            closeOnClickOutside=False,
            size="sm",
            children=[
                dmc.Stack(
                    align="center",
                    children=[
                        dmc.RingProgress(
                            id="po-completion-ring",
                            sections=[{"value": 100, "color": "green"}],
                            size=120,
                            thickness=12,
                            label=DashIconify(icon="tabler:check", width=40, color="green"),
                        ),
                        dmc.Text(id="po-completion-text", children=""),
                        dmc.Button("Close", id="po-close-completion-button"),
                    ],
                ),
            ],
        ),

        # Welcome screen
        html.Div(
            id="po-welcome-screen",
            children=build_po_welcome_screen(),
            style={"display": "block"},
        ),

        # Main container (hidden until data loaded)
        html.Div(
            id="po-main-container",
            children=build_po_main_layout(),
            style={"display": "none"},
        ),

        # ---- Stores ----
        # Series/config stores
        dcc.Store(id="po-series-select", data=[], storage_type="session"),
        dcc.Store(id="po-series-order-store", data=[], storage_type="session"),
        dcc.Store(id="po-benchmark-assignments-store", data={}, storage_type="session"),
        dcc.Store(id="po-long-short-store", data={}, storage_type="session"),
        dcc.Store(id="po-vol-scaling-assignments-store", data={}, storage_type="session"),
        dcc.Store(id="po-min-wt-store", data={}, storage_type="session"),
        dcc.Store(id="po-max-wt-store", data={}, storage_type="session"),
        dcc.Store(id="po-force-max-store", data={}, storage_type="session"),
        # Temp modal stores
        dcc.Store(id="po-temp-series-select", data=[]),
        dcc.Store(id="po-temp-series-order-store", data=[]),
        dcc.Store(id="po-temp-deleted-series-store", data=[]),
        dcc.Store(id="po-temp-benchmark-assignments-store", data={}),
        dcc.Store(id="po-temp-long-short-store", data={}),
        dcc.Store(id="po-temp-vol-scaling-assignments-store", data={}),
        dcc.Store(id="po-temp-min-wt-store", data={}),
        dcc.Store(id="po-temp-max-wt-store", data={}),
        dcc.Store(id="po-temp-force-max-store", data={}),
        # Controls stores
        dcc.Store(id="po-periodicity-value-store", data="daily", storage_type="session"),
        dcc.Store(id="po-vol-scaler-value-store", data=0, storage_type="session"),
        dcc.Store(id="po-date-range-store", data=None, storage_type="session"),
        dcc.Store(id="po-series-select-value-store", data=[], storage_type="session"),
        # Optimization stores
        dcc.Store(id="po-opt-window-store", data="full", storage_type="session"),
        dcc.Store(id="po-window-size-store", data=252, storage_type="session"),
        dcc.Store(id="po-opt-step-store", data=252, storage_type="session"),
        dcc.Store(id="po-opt-model-store", data="risk_parity", storage_type="session"),
        dcc.Store(id="po-portfolio-name-store", data="OptResult", storage_type="session"),
        dcc.Store(id="po-exp-wt-cov-store", data=False, storage_type="session"),
        dcc.Store(id="po-halflife-store", data=63, storage_type="session"),
        dcc.Store(id="po-missing-data-store", data="fill_na", storage_type="session"),
        dcc.Store(id="po-fill-in-sample-store", data="off", storage_type="session"),
        # Results stores
        dcc.Store(id="po-results-store", data={}, storage_type="session"),
        dcc.Store(id="po-opt-status-store", data=None, storage_type="memory"),
        dcc.Store(id="po-active-tab-store", data="weight", storage_type="session"),
        # Navigation
        dcc.Location(id="po-url-location", refresh=False),
        # One-shot interval to trigger visibility check after session-storage hydration
        dcc.Interval(id="po-page-load-trigger", interval=50, max_intervals=1, n_intervals=0),
    ],
)


# ===========================================================================
# Clientside callbacks
# ===========================================================================

# Navigate to home on Exit
clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks) { window.location.href = '/'; }
        return window.dash_clientside.no_update;
    }
    """,
    Output("po-url-location", "pathname"),
    Input("po-menu-exit", "n_clicks"),
    prevent_initial_call=True,
)

# Navigate to Analytics Tool page (client-side, preserves shared stores)
clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks) { window.location.pathname = '/analyticstool'; }
        return window.dash_clientside.no_update;
    }
    """,
    Output("po-url-location", "pathname", allow_duplicate=True),
    Input("po-menu-view-analytics", "n_clicks"),
    prevent_initial_call=True,
)

# Clear session storage and refresh page
clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks) {
            const keysToRemove = [
                'raw-data-store',
                'original-periodicity-store',
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
            window.location.reload();
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("po-url-location", "pathname", allow_duplicate=True),
    Input("po-menu-clear-local-storage", "n_clicks"),
    prevent_initial_call=True,
)

# Trigger upload from menu
clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks) {
            setTimeout(function() {
                var uploadDiv = document.getElementById('po-upload-data');
                if (uploadDiv) {
                    var input = uploadDiv.querySelector('input[type="file"]');
                    if (input) { input.click(); }
                }
            }, 100);
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("po-url-location", "pathname", allow_duplicate=True),
    Input("po-menu-add-series", "n_clicks"),
    prevent_initial_call=True,
)

# Trigger upload from welcome button
clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks) {
            setTimeout(function() {
                var uploadDiv = document.getElementById('po-upload-data');
                if (uploadDiv) {
                    var input = uploadDiv.querySelector('input[type="file"]');
                    if (input) { input.click(); }
                }
            }, 100);
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("po-url-location", "pathname", allow_duplicate=True),
    Input("po-welcome-add-series-btn", "n_clicks"),
    prevent_initial_call=True,
)

# Store sync: periodicity
clientside_callback(
    "function(value) { return value; }",
    Output("po-periodicity-value-store", "data"),
    Input("po-periodicity-select", "value"),
    prevent_initial_call=True,
)

# Store sync: vol scaler
clientside_callback(
    "function(value) { return value; }",
    Output("po-vol-scaler-value-store", "data"),
    Input("po-vol-scaler-input", "value"),
    prevent_initial_call=True,
)

# Store sync: active tab
clientside_callback(
    "function(value) { return value || 'weight'; }",
    Output("po-active-tab-store", "data"),
    Input("po-vis-tabs", "value"),
    prevent_initial_call=True,
)

# Store sync: series selection
clientside_callback(
    "function(value) { return value || []; }",
    Output("po-series-select-value-store", "data"),
    Input("po-series-select", "data"),
    prevent_initial_call=True,
)

# Toggle halflife visibility based on exp wt cov switch
clientside_callback(
    """
    function(checked) {
        if (checked) {
            return {display: "block"};
        }
        return {display: "none"};
    }
    """,
    Output("po-halflife-wrapper", "style"),
    Input("po-exp-wt-cov-switch", "checked"),
    prevent_initial_call=True,
)

# Store sync: fill in-sample
clientside_callback(
    "function(value) { return value; }",
    Output("po-fill-in-sample-store", "data"),
    Input("po-fill-in-sample-select", "value"),
    prevent_initial_call=True,
)

# Toggle window size / opt step / fill in-sample disabled based on window type
clientside_callback(
    """
    function(value) {
        var disabled = (value === "full");
        return [disabled, disabled, disabled];
    }
    """,
    Output("po-window-size-input", "disabled"),
    Output("po-opt-step-input", "disabled"),
    Output("po-fill-in-sample-select", "disabled"),
    Input("po-opt-window-select", "value"),
    prevent_initial_call=True,
)

# Toggle portfolio selector visibility based on active tab
clientside_callback(
    """
    function(tab) {
        if (tab === "growth") {
            return [{display: "none"}, {display: "none"}, {display: "block"}];
        }
        return [{display: "block"}, {display: "block"}, {display: "none"}];
    }
    """,
    Output("po-weight-portfolio-select", "style"),
    Output("po-delete-portfolio-button", "style"),
    Output("po-growth-multiselect-wrapper", "style"),
    Input("po-vis-tabs", "value"),
    prevent_initial_call=True,
)


# ===========================================================================
# Server-side callbacks
# ===========================================================================

# ---------------------------------------------------------------------------
# Toggle welcome/main visibility.
# Uses a one-shot Interval to guarantee session-storage has hydrated on
# cross-page navigation, plus raw-data-store Input for same-page uploads.
# ---------------------------------------------------------------------------

clientside_callback(
    """
    function(n_intervals, data) {
        if (data) {
            return [{display: "none"}, {display: "flex", flexDirection: "column", flex: "1", overflow: "hidden"}];
        }
        return [{display: "block"}, {display: "none"}];
    }
    """,
    Output("po-welcome-screen", "style"),
    Output("po-main-container", "style"),
    Input("po-page-load-trigger", "n_intervals"),
    Input("raw-data-store", "data"),
)

# ---------------------------------------------------------------------------
# Restore application state when raw data loads
# ---------------------------------------------------------------------------

@callback(
    Output("po-periodicity-select", "data", allow_duplicate=True),
    Output("po-periodicity-select", "value", allow_duplicate=True),
    Output("po-vol-scaler-input", "value"),
    Output("po-series-select", "data"),
    Input("raw-data-store", "data"),
    State("original-periodicity-store", "data"),
    State("po-periodicity-value-store", "data"),
    State("po-series-select-value-store", "data"),
    State("po-vol-scaler-value-store", "data"),
    prevent_initial_call="initial_duplicate",
)
def po_restore_state(raw_data, orig_periodicity, stored_periodicity, stored_series, stored_vol):
    if not raw_data:
        return (
            [{"value": "daily", "label": "Daily"}],
            "daily",
            0,
            [],
        )
    try:
        df = json_to_df(raw_data)
        periodicity_options = get_available_periodicities(orig_periodicity or "daily")
        valid_periodicity = stored_periodicity if stored_periodicity in [p["value"] for p in periodicity_options] else (orig_periodicity or "daily")
        valid_vol = stored_vol if stored_vol is not None else 0
        current_selection = stored_series or []
        valid_selection = [s for s in current_selection if s in df.columns]
        if not valid_selection:
            valid_selection = list(df.columns)
        return (
            periodicity_options,
            valid_periodicity,
            valid_vol,
            valid_selection,
        )
    except Exception:
        return (
            [{"value": "daily", "label": "Daily"}],
            "daily",
            0,
            [],
        )


# ---------------------------------------------------------------------------
# Run button enable/disable
# ---------------------------------------------------------------------------

@callback(
    Output("po-run-button", "disabled"),
    Input("po-portfolio-name-input", "value"),
    Input("po-series-select", "data"),
)
def po_toggle_run_button(name, selected):
    if not name or not name.strip():
        return True
    if not selected or len(selected) < 2:
        return True
    return False


# ---------------------------------------------------------------------------
# Update default window/step/halflife when periodicity changes
# ---------------------------------------------------------------------------

@callback(
    Output("po-window-size-input", "value"),
    Output("po-opt-step-input", "value"),
    Output("po-halflife-input", "value"),
    Input("po-periodicity-select", "value"),
    prevent_initial_call=True,
)
def po_update_periodicity_defaults(periodicity):
    ws, step, hl = _periodicity_defaults(periodicity)
    return ws, step, hl


# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------

@callback(
    Output("raw-data-store", "data", allow_duplicate=True),
    Output("original-periodicity-store", "data", allow_duplicate=True),
    Output("po-periodicity-select", "data"),
    Output("po-periodicity-select", "value"),
    Output("po-periodicity-select", "disabled"),
    Output("po-temp-series-select", "data", allow_duplicate=True),
    Output("po-alert-message", "children"),
    Output("po-alert-message", "color"),
    Output("po-alert-message", "hide"),
    Output("po-periodicity-value-store", "data", allow_duplicate=True),
    Output("po-series-selection-modal", "opened", allow_duplicate=True),
    Output("po-temp-benchmark-assignments-store", "data", allow_duplicate=True),
    Output("po-temp-long-short-store", "data", allow_duplicate=True),
    Output("po-temp-series-order-store", "data", allow_duplicate=True),
    Output("po-temp-deleted-series-store", "data", allow_duplicate=True),
    Output("po-temp-vol-scaling-assignments-store", "data", allow_duplicate=True),
    Output("po-temp-min-wt-store", "data", allow_duplicate=True),
    Output("po-temp-max-wt-store", "data", allow_duplicate=True),
    Output("po-temp-force-max-store", "data", allow_duplicate=True),
    Input("po-upload-data", "contents"),
    State("po-upload-data", "filename"),
    State("raw-data-store", "data"),
    State("original-periodicity-store", "data"),
    State("po-series-select", "data"),
    State("po-benchmark-assignments-store", "data"),
    State("po-long-short-store", "data"),
    State("po-series-order-store", "data"),
    State("po-vol-scaling-assignments-store", "data"),
    State("po-min-wt-store", "data"),
    State("po-max-wt-store", "data"),
    State("po-force-max-store", "data"),
    prevent_initial_call=True,
)
def po_handle_upload(contents, filename, existing_data, existing_periodicity,
                     current_selection, current_bench, current_ls, current_order,
                     current_vol_scaling, current_min_wt, current_max_wt, current_force_max):
    if contents is None:
        raise PreventUpdate

    n_no = no_update
    try:
        new_df = parse_uploaded_file(contents, filename)
        new_periodicity = detect_periodicity(new_df)

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
            merged_df = merge_returns(existing_df, new_df)
        else:
            merged_df = new_df
            combined_periodicity = new_periodicity

        periodicity_options = get_available_periodicities(combined_periodicity)
        default_periodicity = combined_periodicity

        new_series = [col for col in new_df.columns if col not in (current_selection or [])]
        updated_selection = (current_selection or []) + new_series

        alert_msg = f"Loaded {len(new_df.columns)} series with {len(new_df)} rows from {filename}"

        return (
            df_to_json(merged_df),
            combined_periodicity,
            periodicity_options,
            default_periodicity,
            False,
            updated_selection,
            alert_msg, "green", False,
            default_periodicity,
            True,  # open modal
            current_bench or {},
            current_ls or {},
            current_order or [],
            [],
            current_vol_scaling or {},
            current_min_wt or {},
            current_max_wt or {},
            current_force_max or {},
        )
    except Exception as e:
        return (
            n_no, n_no, n_no, n_no, n_no, n_no,
            f"Error loading file: {str(e)}", "red", False,
            n_no, n_no, n_no, n_no, n_no, n_no, n_no, n_no, n_no, n_no,
        )


# ---------------------------------------------------------------------------
# Series selection modal: open
# ---------------------------------------------------------------------------

@callback(
    Output("po-series-selection-modal", "opened", allow_duplicate=True),
    Output("po-temp-series-select", "data", allow_duplicate=True),
    Output("po-temp-benchmark-assignments-store", "data", allow_duplicate=True),
    Output("po-temp-long-short-store", "data", allow_duplicate=True),
    Output("po-temp-series-order-store", "data", allow_duplicate=True),
    Output("po-temp-deleted-series-store", "data", allow_duplicate=True),
    Output("po-temp-vol-scaling-assignments-store", "data", allow_duplicate=True),
    Output("po-temp-min-wt-store", "data", allow_duplicate=True),
    Output("po-temp-max-wt-store", "data", allow_duplicate=True),
    Output("po-temp-force-max-store", "data", allow_duplicate=True),
    Input("po-open-modal-button", "n_clicks"),
    State("po-series-select", "data"),
    State("po-benchmark-assignments-store", "data"),
    State("po-long-short-store", "data"),
    State("po-series-order-store", "data"),
    State("po-vol-scaling-assignments-store", "data"),
    State("po-min-wt-store", "data"),
    State("po-max-wt-store", "data"),
    State("po-force-max-store", "data"),
    prevent_initial_call=True,
)
def po_open_modal(n_clicks, current_select, current_bench, current_ls, current_order,
                  current_vol_scaling, current_min_wt, current_max_wt, current_force_max):
    if not n_clicks:
        raise PreventUpdate
    return (True, current_select, current_bench, current_ls, current_order, [],
            current_vol_scaling, current_min_wt, current_max_wt, current_force_max)


# ---------------------------------------------------------------------------
# Series selection modal: render rows
# ---------------------------------------------------------------------------

@callback(
    Output("po-series-selection-container", "children"),
    Output("po-temp-series-order-store", "data", allow_duplicate=True),
    Input("raw-data-store", "data"),
    Input("po-temp-series-select", "data"),
    Input("po-temp-series-order-store", "data"),
    Input("po-temp-deleted-series-store", "data"),
    State("po-temp-benchmark-assignments-store", "data"),
    State("po-temp-long-short-store", "data"),
    State("po-temp-vol-scaling-assignments-store", "data"),
    State("po-temp-min-wt-store", "data"),
    State("po-temp-max-wt-store", "data"),
    State("po-temp-force-max-store", "data"),
    prevent_initial_call="initial_duplicate",
)
def po_update_series_selectors(raw_data, selected_series, series_order, deleted_series,
                               current_assignments, long_short_assignments,
                               vol_scaling_assignments, min_wt, max_wt, force_max):
    if raw_data is None:
        return [], []

    df = json_to_df(raw_data)
    deleted_set = set(deleted_series or [])
    all_series = [s for s in list(df.columns) if s not in deleted_set]

    if not all_series:
        return [], []

    if not series_order:
        series_order = all_series
    else:
        for s in all_series:
            if s not in series_order:
                series_order.append(s)
        series_order = [s for s in series_order if s in all_series]

    selected_series = selected_series or []
    min_wt = min_wt or {}
    max_wt = max_wt or {}
    force_max = force_max or {}

    benchmark_options = [{"value": "None", "label": "None"}] + [{"value": s, "label": s} for s in all_series]
    max_len = max(len(s) for s in all_series) if all_series else 10
    series_width = max(150, max_len * 8 + 20)
    benchmark_width = int(series_width * 1.3)

    header_row = dmc.Group(
        mb="xs",
        gap="xs",
        wrap="nowrap",
        children=[
            dmc.Box(w=20, miw=20),
            dmc.Box(w=20, miw=20),
            dmc.Text("Series", size="xs", fw=700, w=series_width, miw=series_width, c="dimmed"),
            dmc.Text("Benchmark", size="xs", fw=700, w=benchmark_width, miw=benchmark_width, c="dimmed"),
            dmc.Text("L/S", size="xs", fw=700, w=50, miw=50, c="dimmed"),
            dmc.Text("Scale Vol", size="xs", fw=700, w=70, miw=70, c="dimmed"),
            dmc.Text("Min Wt", size="xs", fw=700, w=80, miw=80, c="dimmed"),
            dmc.Text("Max Wt", size="xs", fw=700, w=80, miw=80, c="dimmed"),
            dmc.Text("Force", size="xs", fw=700, w=50, miw=50, c="dimmed"),
            dmc.Box(w=30, miw=30),
        ],
    )

    rows = [header_row]
    for idx, series in enumerate(series_order):
        bench_val = (current_assignments or {}).get(series, "None")
        if bench_val not in [s for s in all_series] and bench_val != "None":
            bench_val = "None"
        is_ls = (long_short_assignments or {}).get(series, False)
        is_scale_vol = (vol_scaling_assignments or {}).get(series, True)
        is_selected = series in selected_series
        min_wt_val = min_wt.get(series, 0)
        max_wt_val = max_wt.get(series, 100)
        force_max_val = force_max.get(series, False)

        up_disabled = (idx == 0)
        down_disabled = (idx == len(series_order) - 1)

        rows.append(
            dmc.Group(
                mb="xs",
                gap="xs",
                wrap="nowrap",
                children=[
                    dmc.Stack(
                        gap=0,
                        w=20,
                        miw=20,
                        children=[
                            dmc.ActionIcon(
                                "▲",
                                id={"type": "po-move-up-button", "series": series},
                                variant="subtle", color="gray", size="xs",
                                disabled=up_disabled,
                                style={"fontSize": "8px", "height": "12px", "minHeight": "12px"},
                            ),
                            dmc.ActionIcon(
                                "▼",
                                id={"type": "po-move-down-button", "series": series},
                                variant="subtle", color="gray", size="xs",
                                disabled=down_disabled,
                                style={"fontSize": "8px", "height": "12px", "minHeight": "12px"},
                            ),
                        ],
                    ),
                    dmc.Checkbox(
                        id={"type": "po-series-include-checkbox", "series": series},
                        checked=is_selected,
                        size="xs",
                    ),
                    dmc.Text(series, size="sm", w=series_width, miw=series_width, style={"fontFamily": "monospace"}),
                    dmc.Select(
                        id={"type": "po-benchmark-select", "series": series},
                        data=benchmark_options,
                        value=bench_val,
                        size="xs",
                        w=benchmark_width,
                        miw=benchmark_width,
                        placeholder="Benchmark",
                    ),
                    dmc.Switch(
                        id={"type": "po-long-short-checkbox", "series": series},
                        checked=is_ls,
                        size="xs",
                        w=50,
                        miw=50,
                    ),
                    dmc.Switch(
                        id={"type": "po-scale-vol-checkbox", "series": series},
                        checked=is_scale_vol,
                        size="xs",
                        w=70,
                        miw=70,
                    ),
                    dmc.NumberInput(
                        id={"type": "po-min-wt-input", "series": series},
                        value=min_wt_val,
                        min=0, max=100, step=1, suffix="%",
                        size="xs", w=80, miw=80,
                        disabled=force_max_val,
                    ),
                    dmc.NumberInput(
                        id={"type": "po-max-wt-input", "series": series},
                        value=max_wt_val,
                        min=0, max=100, step=1, suffix="%",
                        size="xs", w=80, miw=80,
                    ),
                    dmc.Switch(
                        id={"type": "po-force-max-switch", "series": series},
                        checked=force_max_val,
                        size="xs",
                        w=50,
                        miw=50,
                    ),
                    dmc.ActionIcon(
                        DashIconify(icon="tabler:trash-x", color="red", width=20),
                        id={"type": "po-delete-series-button", "series": series},
                        variant="subtle", color="red", size="sm",
                    ),
                ],
            )
        )

    return rows, series_order


# ---------------------------------------------------------------------------
# Modal: collect benchmark assignments
# ---------------------------------------------------------------------------

@callback(
    Output("po-temp-benchmark-assignments-store", "data"),
    Input({"type": "po-benchmark-select", "series": ALL}, "value"),
    State({"type": "po-benchmark-select", "series": ALL}, "id"),
    State("raw-data-store", "data"),
    prevent_initial_call=True,
)
def po_update_benchmarks(values, ids, raw_data):
    if raw_data is None or not values or not ids:
        return {}
    assignments = {}
    for i, bid in enumerate(ids):
        if i < len(values) and values[i]:
            assignments[bid["series"]] = values[i]
    return assignments


# ---------------------------------------------------------------------------
# Modal: collect long-short assignments
# ---------------------------------------------------------------------------

@callback(
    Output("po-temp-long-short-store", "data"),
    Input({"type": "po-long-short-checkbox", "series": ALL}, "checked"),
    State({"type": "po-long-short-checkbox", "series": ALL}, "id"),
    State("raw-data-store", "data"),
    prevent_initial_call=True,
)
def po_update_ls(values, ids, raw_data):
    if raw_data is None or values is None or not ids:
        return {}
    return {ids[i]["series"]: (values[i] or False) for i in range(min(len(ids), len(values)))}


# ---------------------------------------------------------------------------
# Modal: collect vol scaling assignments
# ---------------------------------------------------------------------------

@callback(
    Output("po-temp-vol-scaling-assignments-store", "data"),
    Input({"type": "po-scale-vol-checkbox", "series": ALL}, "checked"),
    State({"type": "po-scale-vol-checkbox", "series": ALL}, "id"),
    State("raw-data-store", "data"),
    prevent_initial_call=True,
)
def po_update_vol_scaling(values, ids, raw_data):
    if raw_data is None or values is None or not ids:
        return {}
    return {ids[i]["series"]: values[i] for i in range(min(len(ids), len(values)))}


# ---------------------------------------------------------------------------
# Modal: collect min/max/force_max weights
# ---------------------------------------------------------------------------

@callback(
    Output("po-temp-min-wt-store", "data"),
    Input({"type": "po-min-wt-input", "series": ALL}, "value"),
    State({"type": "po-min-wt-input", "series": ALL}, "id"),
    prevent_initial_call=True,
)
def po_update_min_wt(values, ids):
    if not values or not ids:
        return {}
    return {ids[i]["series"]: (values[i] or 0) for i in range(min(len(ids), len(values)))}


@callback(
    Output("po-temp-max-wt-store", "data"),
    Input({"type": "po-max-wt-input", "series": ALL}, "value"),
    State({"type": "po-max-wt-input", "series": ALL}, "id"),
    prevent_initial_call=True,
)
def po_update_max_wt(values, ids):
    if not values or not ids:
        return {}
    return {ids[i]["series"]: (values[i] or 100) for i in range(min(len(ids), len(values)))}


@callback(
    Output("po-temp-force-max-store", "data"),
    Input({"type": "po-force-max-switch", "series": ALL}, "checked"),
    State({"type": "po-force-max-switch", "series": ALL}, "id"),
    prevent_initial_call=True,
)
def po_update_force_max(values, ids):
    if not values or not ids:
        return {}
    return {ids[i]["series"]: (values[i] or False) for i in range(min(len(ids), len(values)))}


# ---------------------------------------------------------------------------
# Modal: delete series
# ---------------------------------------------------------------------------

@callback(
    Output("po-temp-deleted-series-store", "data", allow_duplicate=True),
    Output("po-temp-series-select", "data", allow_duplicate=True),
    Input({"type": "po-delete-series-button", "series": ALL}, "n_clicks"),
    State("po-temp-deleted-series-store", "data"),
    State({"type": "po-series-include-checkbox", "series": ALL}, "checked"),
    State({"type": "po-series-include-checkbox", "series": ALL}, "id"),
    prevent_initial_call=True,
)
def po_delete_series(n_clicks_list, deleted_series, checkbox_values, checkbox_ids):
    if not n_clicks_list or all(n is None for n in n_clicks_list):
        raise PreventUpdate
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    triggered_id = ctx.triggered[0]["prop_id"]
    try:
        id_dict = json.loads(triggered_id.rsplit(".", 1)[0])
        series_to_delete = id_dict.get("series")
    except (json.JSONDecodeError, KeyError):
        raise PreventUpdate
    if not series_to_delete:
        raise PreventUpdate

    new_deleted = (deleted_series or []) + [series_to_delete]
    selected = []
    if checkbox_values and checkbox_ids:
        for i, cid in enumerate(checkbox_ids):
            if i < len(checkbox_values) and checkbox_values[i]:
                selected.append(cid["series"])
    selected = [s for s in selected if s != series_to_delete]
    return new_deleted, selected


# ---------------------------------------------------------------------------
# Modal: reorder series
# ---------------------------------------------------------------------------

@callback(
    Output("po-temp-series-order-store", "data", allow_duplicate=True),
    Output("po-temp-series-select", "data", allow_duplicate=True),
    Input({"type": "po-move-up-button", "series": ALL}, "n_clicks"),
    Input({"type": "po-move-down-button", "series": ALL}, "n_clicks"),
    State("po-temp-series-order-store", "data"),
    State("raw-data-store", "data"),
    State({"type": "po-series-include-checkbox", "series": ALL}, "checked"),
    State({"type": "po-series-include-checkbox", "series": ALL}, "id"),
    prevent_initial_call=True,
)
def po_reorder_series(up_clicks, down_clicks, current_order, raw_data, checkbox_values, checkbox_ids):
    if raw_data is None or not current_order:
        raise PreventUpdate
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    triggered_id = ctx.triggered[0]["prop_id"]
    if not triggered_id:
        raise PreventUpdate
    try:
        button_data = json.loads(triggered_id.rsplit(".", 1)[0])
        button_type = button_data["type"]
        series_name = button_data["series"]
    except (json.JSONDecodeError, KeyError, ValueError):
        raise PreventUpdate

    current_selected = []
    if checkbox_values and checkbox_ids:
        checkbox_map = {}
        for i, cid in enumerate(checkbox_ids):
            if i < len(checkbox_values):
                checkbox_map[cid["series"]] = checkbox_values[i]
        for s in current_order:
            if checkbox_map.get(s, False):
                current_selected.append(s)

    if series_name not in current_order:
        raise PreventUpdate
    current_idx = current_order.index(series_name)
    new_order = current_order.copy()
    if button_type == "po-move-up-button" and current_idx > 0:
        new_order[current_idx], new_order[current_idx - 1] = new_order[current_idx - 1], new_order[current_idx]
    elif button_type == "po-move-down-button" and current_idx < len(new_order) - 1:
        new_order[current_idx], new_order[current_idx + 1] = new_order[current_idx + 1], new_order[current_idx]
    else:
        raise PreventUpdate
    return new_order, current_selected


# ---------------------------------------------------------------------------
# Modal: OK button
# ---------------------------------------------------------------------------

@callback(
    Output("po-series-select", "data", allow_duplicate=True),
    Output("po-benchmark-assignments-store", "data", allow_duplicate=True),
    Output("po-long-short-store", "data", allow_duplicate=True),
    Output("po-series-order-store", "data", allow_duplicate=True),
    Output("po-series-selection-modal", "opened", allow_duplicate=True),
    Output("po-series-select-value-store", "data", allow_duplicate=True),
    Output("raw-data-store", "data", allow_duplicate=True),
    Output("po-vol-scaling-assignments-store", "data", allow_duplicate=True),
    Output("po-min-wt-store", "data"),
    Output("po-max-wt-store", "data"),
    Output("po-force-max-store", "data"),
    Input("po-modal-ok-button", "n_clicks"),
    State({"type": "po-series-include-checkbox", "series": ALL}, "checked"),
    State({"type": "po-series-include-checkbox", "series": ALL}, "id"),
    State("po-temp-benchmark-assignments-store", "data"),
    State("po-temp-long-short-store", "data"),
    State("po-temp-series-order-store", "data"),
    State("po-temp-deleted-series-store", "data"),
    State("raw-data-store", "data"),
    State("po-temp-vol-scaling-assignments-store", "data"),
    State("po-temp-min-wt-store", "data"),
    State("po-temp-max-wt-store", "data"),
    State("po-temp-force-max-store", "data"),
    prevent_initial_call=True,
)
def po_on_modal_ok(n_clicks, checkbox_values, checkbox_ids, temp_bench, temp_ls,
                   temp_order, temp_deleted, raw_data, temp_vol_scaling,
                   temp_min_wt, temp_max_wt, temp_force_max):
    if not n_clicks:
        raise PreventUpdate

    temp_select = []
    if checkbox_values and checkbox_ids:
        checkbox_map = {}
        for i, cid in enumerate(checkbox_ids):
            if i < len(checkbox_values):
                checkbox_map[cid["series"]] = checkbox_values[i]
        order_to_use = temp_order if temp_order else list(checkbox_map.keys())
        for s in order_to_use:
            if checkbox_map.get(s, False):
                temp_select.append(s)

    updated_raw_data = raw_data
    if temp_deleted and raw_data:
        df = json_to_df(raw_data)
        to_drop = [s for s in temp_deleted if s in df.columns]
        if to_drop:
            df = df.drop(columns=to_drop)
            updated_raw_data = df_to_json(df)
            if temp_bench:
                temp_bench = {k: v for k, v in temp_bench.items() if k not in to_drop}
            if temp_ls:
                temp_ls = {k: v for k, v in temp_ls.items() if k not in to_drop}
            if temp_order:
                temp_order = [s for s in temp_order if s not in to_drop]
            if temp_vol_scaling:
                temp_vol_scaling = {k: v for k, v in temp_vol_scaling.items() if k not in to_drop}
            if temp_min_wt:
                temp_min_wt = {k: v for k, v in temp_min_wt.items() if k not in to_drop}
            if temp_max_wt:
                temp_max_wt = {k: v for k, v in temp_max_wt.items() if k not in to_drop}
            if temp_force_max:
                temp_force_max = {k: v for k, v in temp_force_max.items() if k not in to_drop}
            temp_select = [s for s in temp_select if s not in to_drop]

    return (temp_select, temp_bench, temp_ls, temp_order, False, temp_select,
            updated_raw_data, temp_vol_scaling, temp_min_wt, temp_max_wt, temp_force_max)


# ---------------------------------------------------------------------------
# Modal: Cancel button
# ---------------------------------------------------------------------------

@callback(
    Output("po-series-selection-modal", "opened", allow_duplicate=True),
    Input("po-modal-cancel-button", "n_clicks"),
    prevent_initial_call=True,
)
def po_on_modal_cancel(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    return False


# ---------------------------------------------------------------------------
# Edit menu: Clear all series
# ---------------------------------------------------------------------------

@callback(
    Output("raw-data-store", "data", allow_duplicate=True),
    Output("original-periodicity-store", "data", allow_duplicate=True),
    Output("po-series-select", "data", allow_duplicate=True),
    Output("po-benchmark-assignments-store", "data", allow_duplicate=True),
    Output("po-long-short-store", "data", allow_duplicate=True),
    Output("po-series-order-store", "data", allow_duplicate=True),
    Output("po-vol-scaling-assignments-store", "data", allow_duplicate=True),
    Output("po-min-wt-store", "data", allow_duplicate=True),
    Output("po-max-wt-store", "data", allow_duplicate=True),
    Output("po-force-max-store", "data", allow_duplicate=True),
    Output("po-periodicity-value-store", "data", allow_duplicate=True),
    Output("po-vol-scaler-value-store", "data", allow_duplicate=True),
    Output("po-series-select-value-store", "data", allow_duplicate=True),
    Output("po-results-store", "data", allow_duplicate=True),
    Input("po-menu-clear-all-series", "n_clicks"),
    prevent_initial_call=True,
)
def po_clear_all_series(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    return None, "daily", [], {}, {}, [], {}, {}, {}, {}, None, 0, [], {}


# ---------------------------------------------------------------------------
# Date range initialization
# ---------------------------------------------------------------------------

@callback(
    Output("po-start-date-picker", "value"),
    Output("po-end-date-picker", "value"),
    Output("po-date-picker-wrapper", "style"),
    Output("po-common-range-button", "disabled"),
    Output("po-maximum-range-button", "disabled"),
    Output("po-date-range-store", "data", allow_duplicate=True),
    Input("raw-data-store", "data"),
    Input("po-periodicity-select", "value"),
    Input("po-series-select", "data"),
    prevent_initial_call="initial_duplicate",
)
def po_init_date_range(raw_data, periodicity, selected_series):
    disabled_style = {"display": "flex", "opacity": 0.5, "pointerEvents": "none", "alignItems": "flex-start"}
    enabled_style = {"display": "flex", "alignItems": "flex-start"}

    if raw_data is None or not selected_series:
        return None, None, disabled_style, True, True, None

    try:
        df = resample_returns_cached(raw_data, periodicity or "daily")
        available = [s for s in selected_series if s in df.columns]
        if not available:
            return None, None, disabled_style, True, True, None

        start_date = df.index.min().strftime("%Y-%m-%d")
        end_date = df.index.max().strftime("%Y-%m-%d")
        return start_date, end_date, enabled_style, False, False, {"start": start_date, "end": end_date}
    except Exception:
        return None, None, disabled_style, True, True, None


# ---------------------------------------------------------------------------
# Date range buttons
# ---------------------------------------------------------------------------

@callback(
    Output("po-start-date-picker", "value", allow_duplicate=True),
    Output("po-end-date-picker", "value", allow_duplicate=True),
    Output("po-date-range-store", "data"),
    Input("po-common-range-button", "n_clicks"),
    Input("po-maximum-range-button", "n_clicks"),
    State("raw-data-store", "data"),
    State("po-periodicity-select", "value"),
    State("po-series-select", "data"),
    prevent_initial_call=True,
)
def po_date_range_buttons(common_clicks, max_clicks, raw_data, periodicity, selected_series):
    if raw_data is None or not selected_series:
        raise PreventUpdate
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    try:
        df = resample_returns_cached(raw_data, periodicity or "daily")
        available = [s for s in selected_series if s in df.columns]
        if not available:
            raise PreventUpdate
        if button_id == "po-common-range-button":
            subset = df[available].dropna()
            if len(subset) == 0:
                raise PreventUpdate
            start_date = subset.index.min().strftime("%Y-%m-%d")
            end_date = subset.index.max().strftime("%Y-%m-%d")
        else:
            start_date = df.index.min().strftime("%Y-%m-%d")
            end_date = df.index.max().strftime("%Y-%m-%d")
        return start_date, end_date, {"start": start_date, "end": end_date}
    except Exception:
        raise PreventUpdate


@callback(
    Output("po-date-range-store", "data", allow_duplicate=True),
    Input("po-start-date-picker", "value"),
    Input("po-end-date-picker", "value"),
    prevent_initial_call=True,
)
def po_update_date_range_store(start, end):
    if start and end:
        return {"start": start, "end": end}
    return no_update


# ===========================================================================
# Background optimization callback
# ===========================================================================

@callback(
    Output("po-results-store", "data", allow_duplicate=True),
    Output("raw-data-store", "data", allow_duplicate=True),
    Output("po-opt-status-store", "data"),
    Input("po-run-button", "n_clicks"),
    State("raw-data-store", "data"),
    State("original-periodicity-store", "data"),
    State("po-periodicity-select", "value"),
    State("po-series-select", "data"),
    State("po-benchmark-assignments-store", "data"),
    State("po-long-short-store", "data"),
    State("po-date-range-store", "data"),
    State("po-vol-scaler-value-store", "data"),
    State("po-vol-scaling-assignments-store", "data"),
    State("po-min-wt-store", "data"),
    State("po-max-wt-store", "data"),
    State("po-force-max-store", "data"),
    State("po-exp-wt-cov-switch", "checked"),
    State("po-halflife-input", "value"),
    State("po-portfolio-name-input", "value"),
    State("po-opt-window-select", "value"),
    State("po-window-size-input", "value"),
    State("po-opt-step-input", "value"),
    State("po-opt-model-select", "value"),
    State("po-missing-data-select", "value"),
    State("po-fill-in-sample-select", "value"),
    State("po-results-store", "data"),
    prevent_initial_call=True,
)
def po_run_optimization(n_clicks, raw_data, orig_periodicity, periodicity,
                        selected_series, benchmark_assignments, long_short_assignments,
                        date_range, vol_scaler, vol_scaling_assignments,
                        min_wt, max_wt, force_max, exp_wt_cov, halflife,
                        portfolio_name, opt_window, window_size, opt_step,
                        opt_model, missing_data, fill_in_sample_value, current_results):
    if not n_clicks or not raw_data or not selected_series:
        raise PreventUpdate

    try:
        # Compute working returns
        working_df = get_working_returns(
            raw_data,
            periodicity or "daily",
            tuple(selected_series),
            json.dumps(benchmark_assignments) if benchmark_assignments else "{}",
            json.dumps(long_short_assignments) if long_short_assignments else "{}",
            json.dumps(date_range) if date_range else "null",
            vol_scaler or 0,
            json.dumps(vol_scaling_assignments) if vol_scaling_assignments else "{}",
        )

        if working_df.empty:
            return (
                no_update, no_update,
                {"status": "error", "name": portfolio_name, "message": "No data available for selected series."},
            )

        # Filter to only selected series (exclude benchmark columns)
        opt_cols = [s for s in selected_series if s in working_df.columns]
        opt_df = working_df[opt_cols]

        # Build config
        config = {
            "model": opt_model or "risk_parity",
            "window_type": opt_window or "full",
            "window_size": int(window_size or 252),
            "opt_step": int(opt_step or 252),
            "exp_wt_cov": bool(exp_wt_cov),
            "halflife": int(halflife or 63),
            "missing_data": missing_data or "fill_na",
            "fill_in_sample": fill_in_sample_value == "on",
            "selected_series": opt_cols,
            "min_wt": min_wt or {},
            "max_wt": max_wt or {},
            "force_max": force_max or {},
        }

        # Run optimization
        window_results, portfolio_returns = run_portfolio_optimization(opt_df, config)

        # Determine unique portfolio name
        current_results = current_results or {}
        final_name = portfolio_name.strip() or "OptResult"
        existing_df = json_to_df(raw_data)

        # Avoid collisions with existing columns and existing results
        base_name = final_name
        counter = 1
        while final_name in existing_df.columns or final_name in current_results:
            final_name = f"{base_name}_{counter}"
            counter += 1

        # Add portfolio returns to raw data
        portfolio_series = portfolio_returns.reindex(existing_df.index)
        existing_df[final_name] = portfolio_series
        new_raw_data = df_to_json(existing_df)

        # Store results
        window_data = []
        for wr in window_results:
            window_data.append({
                "apply_start": wr.apply_start.isoformat(),
                "apply_end": wr.apply_end.isoformat(),
                "weights": wr.weights,
            })

        current_results[final_name] = {
            "window_weights": window_data,
            "returns_json": portfolio_returns.to_json(date_format="iso"),
            "config": config,
        }

        return (
            current_results,
            new_raw_data,
            {"status": "complete", "name": final_name},
        )

    except ValueError as e:
        return (
            no_update, no_update,
            {"status": "error", "name": portfolio_name, "message": str(e)},
        )
    except Exception as e:
        return (
            no_update, no_update,
            {"status": "error", "name": portfolio_name, "message": f"Optimization failed: {str(e)}"},
        )


# ---------------------------------------------------------------------------
# Completion modal
# ---------------------------------------------------------------------------

@callback(
    Output("po-completion-modal", "opened"),
    Output("po-completion-text", "children"),
    Output("po-completion-ring", "sections"),
    Input("po-opt-status-store", "data"),
    prevent_initial_call=True,
)
def po_show_completion(status):
    if not status:
        raise PreventUpdate
    if status.get("status") == "complete":
        return (
            True,
            f"Portfolio '{status['name']}' created successfully.",
            [{"value": 100, "color": "green"}],
        )
    elif status.get("status") == "error":
        return (
            True,
            status.get("message", "An error occurred."),
            [{"value": 100, "color": "red"}],
        )
    raise PreventUpdate


@callback(
    Output("po-completion-modal", "opened", allow_duplicate=True),
    Input("po-close-completion-button", "n_clicks"),
    prevent_initial_call=True,
)
def po_close_completion(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    return False


# ---------------------------------------------------------------------------
# Update portfolio dropdowns when results change
# ---------------------------------------------------------------------------

@callback(
    Output("po-weight-portfolio-select", "data"),
    Output("po-weight-portfolio-select", "value"),
    Output("po-growth-portfolio-multiselect", "data"),
    Output("po-growth-portfolio-multiselect", "value"),
    Input("po-results-store", "data"),
    State("po-weight-portfolio-select", "value"),
    State("po-growth-portfolio-multiselect", "value"),
    prevent_initial_call=True,
)
def po_update_portfolio_dropdowns(results, current_select, current_multi):
    if not results:
        return [], None, [], []
    names = list(results.keys())
    options = [{"value": n, "label": n} for n in names]
    # Always select the newest portfolio (last added)
    sel = names[-1] if names else None
    multi = [v for v in (current_multi or []) if v in names]
    if not multi and names:
        multi = [names[-1]]
    return options, sel, options, multi


# ---------------------------------------------------------------------------
# Delete portfolio
# ---------------------------------------------------------------------------

@callback(
    Output("po-results-store", "data", allow_duplicate=True),
    Output("raw-data-store", "data", allow_duplicate=True),
    Output("po-weight-portfolio-select", "value", allow_duplicate=True),
    Input("po-delete-portfolio-button", "n_clicks"),
    State("po-weight-portfolio-select", "value"),
    State("po-results-store", "data"),
    State("raw-data-store", "data"),
    prevent_initial_call=True,
)
def po_delete_portfolio(n_clicks, selected_portfolio, results, raw_data):
    if not n_clicks or not selected_portfolio or not results:
        raise PreventUpdate
    if selected_portfolio not in results:
        raise PreventUpdate

    # Remove from results
    new_results = {k: v for k, v in results.items() if k != selected_portfolio}

    # Remove column from raw data
    new_raw = raw_data
    if raw_data:
        df = json_to_df(raw_data)
        if selected_portfolio in df.columns:
            df = df.drop(columns=[selected_portfolio])
            new_raw = df_to_json(df)

    # Pick next selection
    remaining = list(new_results.keys())
    new_sel = remaining[-1] if remaining else None

    return new_results, new_raw, new_sel


# ===========================================================================
# Visualization callbacks
# ===========================================================================

# ---------------------------------------------------------------------------
# Weight chart (stacked area)
# ---------------------------------------------------------------------------

@callback(
    Output("po-weight-chart-container", "children"),
    Input("po-weight-portfolio-select", "value"),
    Input("po-results-store", "data"),
    Input("po-vis-tabs", "value"),
    prevent_initial_call=True,
)
def po_render_weight_chart(selected_portfolio, results, active_tab):
    if active_tab != "weight" or not selected_portfolio or not results:
        return html.Div()
    if selected_portfolio not in results:
        return html.Div()

    portfolio_data = results[selected_portfolio]
    window_weights = portfolio_data.get("window_weights", [])

    if not window_weights:
        return dmc.Text("No weight data available.", c="dimmed")

    # Get asset names from first window
    asset_names = list(window_weights[0]["weights"].keys())

    # Build per-period weights as a step function
    all_dates = []
    all_weights = {a: [] for a in asset_names}

    for ww in window_weights:
        start = pd.Timestamp(ww["apply_start"])
        end = pd.Timestamp(ww["apply_end"])
        weights = ww["weights"]
        all_dates.append(start)
        all_dates.append(end)
        for a in asset_names:
            all_weights[a].append(weights.get(a, 0) * 100)
            all_weights[a].append(weights.get(a, 0) * 100)

    fig = go.Figure()
    for a in asset_names:
        fig.add_trace(go.Scatter(
            x=all_dates,
            y=all_weights[a],
            name=a,
            mode="lines",
            stackgroup="one",
            line={"width": 0.5},
        ))

    fig.update_layout(
        title=f"Portfolio Weights: {selected_portfolio}",
        yaxis_title="Weight (%)",
        yaxis={"range": [0, 100]},
        hovermode="x unified",
        margin={"t": 40, "b": 40, "l": 60, "r": 20},
        height=500,
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.2},
    )

    return dcc.Graph(figure=fig, style={"height": "100%", "width": "100%"})


# ---------------------------------------------------------------------------
# Growth of $1 chart
# ---------------------------------------------------------------------------

@callback(
    Output("po-growth-chart-container", "children"),
    Input("po-growth-portfolio-multiselect", "value"),
    Input("po-results-store", "data"),
    Input("po-vis-tabs", "value"),
    prevent_initial_call=True,
)
def po_render_growth_chart(selected_portfolios, results, active_tab):
    if active_tab != "growth" or not selected_portfolios or not results:
        return html.Div()

    fig = go.Figure()
    for pname in selected_portfolios:
        if pname not in results:
            continue
        returns_json = results[pname].get("returns_json")
        if not returns_json:
            continue
        returns = pd.read_json(returns_json, typ="series")
        returns.index = pd.to_datetime(returns.index)
        returns = returns.sort_index()
        growth = (1 + returns).cumprod()
        fig.add_trace(go.Scatter(
            x=growth.index,
            y=growth.values,
            name=pname,
            mode="lines",
        ))

    fig.update_layout(
        title="Growth of $1",
        yaxis_title="Value ($)",
        hovermode="x unified",
        margin={"t": 40, "b": 40, "l": 60, "r": 20},
        height=500,
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.2},
    )

    return dcc.Graph(figure=fig, style={"height": "100%", "width": "100%"})


# ---------------------------------------------------------------------------
# Attribution chart (monthly stacked bar)
# ---------------------------------------------------------------------------

@callback(
    Output("po-attribution-chart-container", "children"),
    Input("po-weight-portfolio-select", "value"),
    Input("po-results-store", "data"),
    Input("po-vis-tabs", "value"),
    State("raw-data-store", "data"),
    State("original-periodicity-store", "data"),
    State("po-periodicity-select", "value"),
    State("po-benchmark-assignments-store", "data"),
    State("po-long-short-store", "data"),
    State("po-date-range-store", "data"),
    State("po-vol-scaler-value-store", "data"),
    State("po-vol-scaling-assignments-store", "data"),
    prevent_initial_call=True,
)
def po_render_attribution_chart(selected_portfolio, results, active_tab, raw_data,
                                 orig_periodicity, periodicity, bench, ls, date_range,
                                 vol_scaler, vol_scaling):
    if active_tab != "attribution" or not selected_portfolio or not results:
        return html.Div()
    if selected_portfolio not in results:
        return html.Div()

    portfolio_data = results[selected_portfolio]
    window_weights = portfolio_data.get("window_weights", [])
    config = portfolio_data.get("config", {})
    opt_series = config.get("selected_series", [])

    if not window_weights or not opt_series or not raw_data:
        return dmc.Text("No attribution data available.", c="dimmed")

    try:
        # Get the working returns for the component series
        working_df = get_working_returns(
            raw_data,
            periodicity or "daily",
            tuple(opt_series),
            json.dumps(bench) if bench else "{}",
            json.dumps(ls) if ls else "{}",
            json.dumps(date_range) if date_range else "null",
            vol_scaler or 0,
            json.dumps(vol_scaling) if vol_scaling else "{}",
        )

        # Build per-period weights DataFrame
        weights_df = pd.DataFrame(0.0, index=working_df.index, columns=opt_series)
        for ww in window_weights:
            start = pd.Timestamp(ww["apply_start"])
            end = pd.Timestamp(ww["apply_end"])
            mask = (weights_df.index >= start) & (weights_df.index <= end)
            for s in opt_series:
                weights_df.loc[mask, s] = ww["weights"].get(s, 0)

        # Compute attribution = weight * return
        attribution = weights_df * working_df[opt_series].fillna(0)

        # Resample to monthly for readability
        attribution_monthly = attribution.resample("ME").sum()
        attribution_monthly = attribution_monthly.dropna(how="all")

        if attribution_monthly.empty:
            return dmc.Text("No attribution data available.", c="dimmed")

        fig = go.Figure()
        for s in opt_series:
            if s in attribution_monthly.columns:
                fig.add_trace(go.Bar(
                    x=attribution_monthly.index,
                    y=attribution_monthly[s] * 100,
                    name=s,
                ))

        fig.update_layout(
            barmode="relative",
            title=f"Monthly Return Attribution: {selected_portfolio}",
            yaxis_title="Contribution (%)",
            hovermode="x unified",
            margin={"t": 40, "b": 40, "l": 60, "r": 20},
            height=500,
            legend={"orientation": "h", "yanchor": "bottom", "y": -0.2},
        )

        return dcc.Graph(figure=fig, style={"height": "100%", "width": "100%"})

    except Exception:
        return dmc.Text("Error computing attribution.", c="dimmed")
