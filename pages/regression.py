"""Regression Analysis page for DashMat."""

from __future__ import annotations

import json

import dash_ag_grid as dag
import dash_mantine_components as dmc
from dash_iconify import DashIconify
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import (
    Input, Output, State, callback, dcc, html, no_update,
    register_page, clientside_callback, callback_context,
)
from dash.exceptions import PreventUpdate

import cache_config
from utils.parsing import get_sheet_names
from utils.date_range_flow import (
    compute_date_range_candidates,
    resolve_button_range,
    resolve_initial_range,
)
from utils.upload_flow import (
    import_selected_workbook_sheets as _shared_import_selected_workbook_sheets,
    import_single_upload as _shared_import_single_upload,
    merge_uploaded_with_existing as _shared_merge_uploaded_with_existing,
)
from utils.returns import (
    df_to_json,
    get_available_periodicities,
    get_working_returns,
    json_to_df,
    annualization_factor,
)
from utils.statistics import calculate_statistics_cached
from utils.charting import apply_chart_theme
from utils.regression import run_regression, RegressionWindowResult
from utils.serialization import date_range_payload_for_cache, mapping_payload_for_cache
from utils.dashmat_welcome_modal import (
    PagePrefixConfig,
    build_db_add_modal,
    build_series_selection_modal,
    build_sheet_select_modal,
    build_welcome_screen as build_shared_welcome_screen,
    compute_close_db_add_modal,
    compute_open_db_add_modal,
    compute_validate_db_add_selection,
)
from utils.sample_data import get_sample_file_path
from utils.core_categories import load_cma_returns_for_benches_with_meta
from dbengine import engine as DB_ENGINE, engine_MRD as MRD_ENGINE

register_page(__name__, path="/regression", name="Regression", title="Regression")

REG_CONFIG = PagePrefixConfig(
    prefix="reg",
    page_icon="tabler:chart-dots-3",
    page_title="Regression Analysis",
    page_subtitle="Load returns data and run OLS, Ridge, Lasso, Style Analysis and more.",
    series_modal_size="90vw",
    series_modal_max_width="1650px",
    series_modal_transition_ms=200,
)

_MODEL_OPTIONS = [
    {"value": "ols", "label": "OLS"},
    {"value": "constrained_ols", "label": "Constrained OLS"},
    {"value": "style_analysis", "label": "Style Analysis"},
    {"value": "ridge", "label": "Ridge"},
    {"value": "lasso", "label": "Lasso"},
    {"value": "elastic_net", "label": "Elastic Net"},
]

_MISSING_DATA_OPTIONS = [
    {"value": "fill_na", "label": "Fill NA"},
    {"value": "fill_0", "label": "Fill 0"},
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mapping_payload(value) -> str:
    return mapping_payload_for_cache(value)


def _date_range_payload(value) -> str:
    return date_range_payload_for_cache(value)


def _reg_get_working_returns(raw_data, periodicity, selected_series,
                              benchmark_assignments, long_short_assignments,
                              date_range, vol_scaler, vol_scaling_assignments):
    series_tuple = tuple(selected_series or ())
    if not series_tuple or not raw_data:
        return pd.DataFrame()
    return get_working_returns(
        raw_data,
        periodicity or "daily",
        series_tuple,
        _mapping_payload(benchmark_assignments),
        _mapping_payload(long_short_assignments),
        _date_range_payload(date_range),
        vol_scaler or 0,
        _mapping_payload(vol_scaling_assignments),
    )


def _periodicity_defaults(periodicity):
    if periodicity and periodicity.startswith("weekly"):
        return 52, 4, 1, 13
    if periodicity == "monthly":
        return 12, 1, 1, 6
    return 252, 21, 1, 63


def _annualization_for_periodicity(periodicity) -> int:
    try:
        ann = int(annualization_factor(periodicity or "daily"))
        if ann > 0:
            return ann
    except Exception:
        pass
    p = periodicity or "daily"
    if str(p).startswith("weekly"):
        return 52
    if p == "monthly":
        return 12
    return 252


def _fmt(v, decimals=6):
    """Format a numeric value for display."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:.{decimals}f}"


# ---------------------------------------------------------------------------
# Layout builders
# ---------------------------------------------------------------------------

def build_reg_welcome_screen():
    return build_shared_welcome_screen(REG_CONFIG)


def build_reg_main_layout():
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
                                                id="reg-open-modal-button",
                                                variant="light",
                                                size="sm",
                                                w=200,
                                            ),
                                        ]),
                                        dmc.Select(
                                            id="reg-periodicity-select",
                                            label="Periodicity",
                                            data=[{"value": "daily", "label": "Daily"}],
                                            value="daily",
                                            w=200,
                                        ),
                                        html.Div([
                                            dmc.Text("Vol Scaler", size="sm", mb=3, fw=500),
                                            dmc.Tooltip(
                                                label="A value of 0% disables the volatility scaling.",
                                                position="top",
                                                withArrow=True,
                                                children=dmc.NumberInput(
                                                    id="reg-vol-scaler-input",
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
                                html.Div(
                                    id="reg-date-picker-wrapper",
                                    children=[
                                        html.Div([
                                            dmc.DateInput(
                                                id="reg-start-date-picker",
                                                label="Start Date",
                                                value=None,
                                                w=200,
                                                valueFormat="YYYY-MM-DD",
                                            ),
                                        ], style={"marginRight": "15px"}),
                                        html.Div([
                                            dmc.DateInput(
                                                id="reg-end-date-picker",
                                                label="End Date",
                                                value=None,
                                                w=200,
                                                valueFormat="YYYY-MM-DD",
                                            ),
                                        ], style={"marginRight": "15px"}),
                                        html.Div([
                                            dmc.Button(
                                                "Common Range",
                                                id="reg-common-range-button",
                                                size="xs",
                                                variant="outline",
                                                disabled=True,
                                                w=120,
                                            ),
                                        ], style={"marginRight": "10px", "alignSelf": "flex-end", "marginBottom": "2px"}),
                                        html.Div([
                                            dmc.Button(
                                                "Max Range",
                                                id="reg-maximum-range-button",
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
                        ],
                    ),
                ],
            ),

            # Regression Accordion
            dmc.Accordion(
                value="regression",
                mb="xs",
                variant="contained",
                children=[
                    dmc.AccordionItem(
                        value="regression",
                        children=[
                            dmc.AccordionControl("Regression"),
                            dmc.AccordionPanel(children=[
                                dmc.Group(
                                    gap="md",
                                    align="flex-end",
                                    mb="sm",
                                    children=[
                                        dmc.TextInput(
                                            id="reg-regression-name-input",
                                            label="Regression Name",
                                            value="OLS_1",
                                            w=130,
                                            size="sm",
                                        ),
                                        html.Div([
                                            dmc.Text("Model", size="sm", fw=500, mb=3),
                                            dmc.Select(
                                                id="reg-model-select",
                                                data=_MODEL_OPTIONS,
                                                value="ols",
                                                w=170,
                                                size="sm",
                                                clearable=False,
                                            ),
                                        ]),
                                        html.Div([
                                            dmc.Text("Force Zero Intercept", size="sm", fw=500, mb=3),
                                            html.Div(
                                                dmc.Switch(
                                                    id="reg-force-zero-intercept-switch",
                                                    checked=False,
                                                    size="sm",
                                                ),
                                                style={"height": "36px", "display": "flex", "alignItems": "center"},
                                            ),
                                        ]),
                                        html.Div([
                                            dmc.Text("Robust SE", size="sm", fw=500, mb=3),
                                            html.Div(
                                                dmc.Switch(
                                                    id="reg-robust-se-switch",
                                                    checked=False,
                                                    size="sm",
                                                ),
                                                style={"height": "36px", "display": "flex", "alignItems": "center"},
                                            ),
                                        ]),
                                        html.Div(
                                            id="reg-alpha-container",
                                            style={"display": "none"},
                                            children=dmc.NumberInput(
                                                id="reg-alpha-input",
                                                label="Alpha",
                                                value=1.0,
                                                min=0.0001,
                                                step=0.1,
                                                w=100,
                                                size="sm",
                                            ),
                                        ),
                                        html.Div(
                                            id="reg-l1-ratio-container",
                                            style={"display": "none"},
                                            children=dmc.NumberInput(
                                                id="reg-l1-ratio-input",
                                                label="L1 Ratio",
                                                value=0.5,
                                                min=0.0,
                                                max=1.0,
                                                step=0.1,
                                                w=100,
                                                size="sm",
                                            ),
                                        ),
                                        html.Div([
                                            dmc.Text("Exp Wt", size="sm", fw=500, mb=3),
                                            html.Div(
                                                dmc.Switch(
                                                    id="reg-exp-wt-switch",
                                                    checked=False,
                                                    size="sm",
                                                ),
                                                style={"height": "36px", "display": "flex", "alignItems": "center"},
                                            ),
                                        ]),
                                        dmc.NumberInput(
                                            id="reg-halflife-input",
                                            label="Half-Life",
                                            value=63,
                                            min=1,
                                            step=1,
                                            w=90,
                                            size="sm",
                                            disabled=True,
                                        ),
                                    ],
                                ),
                                dmc.Group(
                                    gap="md",
                                    align="flex-end",
                                    mb="sm",
                                    children=[
                                        html.Div([
                                            dmc.Text("Window", size="sm", mb=3, fw=500),
                                            dmc.SegmentedControl(
                                                id="reg-window-type-select",
                                                data=[
                                                    {"value": "full", "label": "Full"},
                                                    {"value": "expanding", "label": "Expanding"},
                                                    {"value": "rolling", "label": "Rolling"},
                                                ],
                                                value="full",
                                                size="sm",
                                            ),
                                        ]),
                                        html.Div([
                                            dmc.Text("Fill In-Sample", size="sm", mb=3, fw=500),
                                            dmc.SegmentedControl(
                                                id="reg-fill-in-sample-select",
                                                data=[
                                                    {"value": "off", "label": "Off"},
                                                    {"value": "on", "label": "On"},
                                                ],
                                                value="off",
                                                size="sm",
                                            ),
                                        ]),
                                        dmc.NumberInput(
                                            id="reg-window-size-input",
                                            label="Window Size (Periods)",
                                            value=36,
                                            min=2,
                                            step=1,
                                            w=160,
                                            size="sm",
                                            disabled=True,
                                        ),
                                        html.Div([
                                            dmc.Text("Opt Step", size="sm", mb=4, fw=500),
                                            dmc.Group(
                                                gap="xs",
                                                wrap="nowrap",
                                                children=[
                                                    dmc.NumberInput(
                                                        id="reg-opt-step-input",
                                                        value=1,
                                                        min=1,
                                                        step=1,
                                                        w=90,
                                                        size="sm",
                                                        disabled=True,
                                                    ),
                                                    dmc.Select(
                                                        id="reg-opt-step-unit-select",
                                                        data=[
                                                            {"value": "months", "label": "Months"},
                                                            {"value": "periods", "label": "Periods"},
                                                        ],
                                                        value="months",
                                                        w=100,
                                                        size="sm",
                                                        clearable=False,
                                                        disabled=True,
                                                    ),
                                                ],
                                            ),
                                        ]),
                                        html.Div([
                                            dmc.Text("Missing Data", size="sm", mb=3, fw=500),
                                            dmc.SegmentedControl(
                                                id="reg-missing-data-select",
                                                data=_MISSING_DATA_OPTIONS,
                                                value="fill_na",
                                                size="sm",
                                            ),
                                        ]),
                                    ],
                                ),
                                html.Div(
                                    id="reg-arima-garch-panel",
                                    style={"display": "none"},
                                    children=[
                                        dmc.Divider(label="ARIMA / GARCH (on residuals)", labelPosition="left", mb="sm", mt="xs"),
                                        dmc.Group(
                                            gap="md",
                                            align="flex-end",
                                            children=[
                                                dmc.Text("ARIMA:", size="sm", fw=500),
                                                dmc.NumberInput(id="reg-arima-p-input", label="p", value=0, min=0, max=5, step=1, w=70, size="sm"),
                                                dmc.NumberInput(id="reg-arima-d-input", label="d", value=0, min=0, max=5, step=1, w=70, size="sm"),
                                                dmc.NumberInput(id="reg-arima-q-input", label="q", value=0, min=0, max=5, step=1, w=70, size="sm"),
                                                dmc.Text("GARCH:", size="sm", fw=500, ml="md"),
                                                dmc.NumberInput(id="reg-garch-p-input", label="p", value=0, min=0, max=5, step=1, w=70, size="sm"),
                                                dmc.NumberInput(id="reg-garch-q-input", label="q", value=0, min=0, max=5, step=1, w=70, size="sm"),
                                            ],
                                        ),
                                    ],
                                ),
                                dmc.Group(
                                    gap="md",
                                    align="flex-end",
                                    mt="sm",
                                    children=[
                                        dmc.Button(
                                            "Run Regression",
                                            id="reg-run-button",
                                            variant="filled",
                                            color="blue",
                                            size="sm",
                                            leftSection=DashIconify(icon="tabler:chart-dots-3"),
                                        ),
                                        dmc.Text(id="reg-run-status-text", size="sm", c="dimmed"),
                                    ],
                                ),
                            ]),
                        ],
                    ),
                ],
            ),

            # Linear Constraints Accordion
            dmc.Accordion(
                value=None,
                mb="xs",
                variant="contained",
                children=[
                    dmc.AccordionItem(
                        value="linear-constraints",
                        children=[
                            dmc.AccordionControl("Linear Constraints"),
                            dmc.AccordionPanel(children=[
                                dmc.Text(
                                    "Define linear beta constraints. Each row enforces Min <= sum(coef_i * beta_i) <= Max.",
                                    size="xs", c="dimmed", mb="xs",
                                ),
                                dmc.Group(
                                    gap="xs",
                                    mb="sm",
                                    children=[
                                        dmc.Button(
                                            "Add Constraint",
                                            id="reg-add-constraint-btn",
                                            variant="outline",
                                            size="xs",
                                            leftSection=DashIconify(icon="tabler:plus"),
                                        ),
                                        dmc.Button(
                                            "Clear Constraints",
                                            id="reg-clear-constraints-btn",
                                            variant="outline",
                                            size="xs",
                                            color="red",
                                            leftSection=DashIconify(icon="tabler:trash"),
                                        ),
                                    ],
                                ),
                                dag.AgGrid(
                                    id="reg-linear-constraints-grid",
                                    className="ag-theme-alpine",
                                    columnDefs=[
                                        {"field": "Constraint", "editable": True, "width": 120, "headerClass": "dashmat-center-header"},
                                        {"field": "Min", "editable": True, "width": 90, "type": "numericColumn", "headerClass": "dashmat-center-header"},
                                        {"field": "Max", "editable": True, "width": 90, "type": "numericColumn", "headerClass": "dashmat-center-header"},
                                    ],
                                    rowData=[],
                                    defaultColDef={"resizable": True, "sortable": False, "suppressHeaderMenuButton": True, "cellStyle": {"textAlign": "center"}},
                                    style={"height": "160px"},
                                    dashGridOptions={"singleClickEdit": True, "suppressExcelExport": True, "suppressCsvExport": True},
                                ),
                            ]),
                        ],
                    ),
                ],
            ),

            # Results area
            html.Div(
                style={"flex": "1", "display": "flex", "flexDirection": "column", "overflow": "hidden", "minHeight": 0},
                children=[
                    dmc.Group(
                        gap="xs",
                        mb="xs",
                        align="flex-end",
                        children=[
                            dmc.Select(
                                id="reg-result-select",
                                label="Regression Result",
                                data=[],
                                value=None,
                                w=250,
                                size="sm",
                                placeholder="No results yet",
                                clearable=False,
                            ),
                            dmc.Button(
                                "Delete",
                                id="reg-delete-result-btn",
                                variant="outline",
                                color="red",
                                size="sm",
                                disabled=True,
                                leftSection=DashIconify(icon="tabler:trash"),
                            ),
                        ],
                    ),
                    dmc.Tabs(
                        id="reg-tabs",
                        value="anova",
                        style={"flex": "1", "display": "flex", "flexDirection": "column", "overflow": "hidden"},
                        children=[
                            dmc.TabsList([
                                dmc.TabsTab("ANOVA", value="anova"),
                                dmc.TabsTab("Rolling Summary", value="rolling"),
                                dmc.TabsTab("Weights", value="weights"),
                                dmc.TabsTab("Statistics", value="statistics"),
                                dmc.TabsTab("Returns", value="returns"),
                                dmc.TabsTab("Growth of $1", value="growth"),
                                dmc.TabsTab("Scatter", value="scatter"),
                            ]),
                            dmc.TabsPanel(value="anova", style={"overflow": "auto", "flex": "1"},
                                children=[html.Div(id="reg-anova-content", style={"padding": "8px"})]),
                            dmc.TabsPanel(value="rolling", style={"overflow": "auto", "flex": "1"},
                                children=[html.Div(id="reg-rolling-content", style={"padding": "8px"})]),
                            dmc.TabsPanel(value="weights", style={"overflow": "auto", "flex": "1"},
                                children=[html.Div(id="reg-weights-content", style={"padding": "8px"})]),
                            dmc.TabsPanel(value="statistics", style={"overflow": "auto", "flex": "1"},
                                children=[html.Div(id="reg-statistics-content", style={"padding": "8px"})]),
                            dmc.TabsPanel(value="returns", style={"overflow": "auto", "flex": "1"},
                                children=[html.Div(id="reg-returns-content", style={"padding": "8px"})]),
                            dmc.TabsPanel(value="growth", style={"overflow": "auto", "flex": "1"},
                                children=[html.Div(id="reg-growth-content", style={"padding": "8px"})]),
                            dmc.TabsPanel(value="scatter", style={"overflow": "auto", "flex": "1"},
                                children=[html.Div(id="reg-scatter-content", style={"padding": "8px"})]),
                        ],
                    ),
                ],
            ),
        ],
    )


# ===========================================================================
# Layout variable
# ===========================================================================

layout = dmc.Container(
    fluid=True,
    style={"minHeight": "calc(100vh - 55px)", "display": "flex", "flexDirection": "column", "overflow": "visible"},
    className="dashmat-page-container",
    children=[
        # Menu bar
        dmc.Paper(
            shadow="xs", p="xs", mb="md", radius="md", withBorder=True,
            className="dashmat-menu-bar",
            children=[
                dmc.Group(
                    gap="xs",
                    children=[
                        dmc.Menu(
                            trigger="click", openDelay=100, closeDelay=200,
                            position="bottom-start", shadow="md", offset=6,
                            children=[
                                dmc.MenuTarget(dmc.Button("File", variant="subtle", color="gray", size="sm", radius="sm")),
                                dmc.MenuDropdown(className="dashmat-menu-dropdown", children=[
                                    dmc.MenuItem("Add series (upload)", id="reg-menu-add-series",
                                                 leftSection=DashIconify(icon="tabler:upload", width=14)),
                                    dmc.MenuDivider(),
                                    dmc.MenuItem("Download sample data (daily)", id="reg-download-sample-daily-btn",
                                                 leftSection=DashIconify(icon="tabler:file-download", width=14)),
                                    dmc.MenuItem("Download sample data (monthly)", id="reg-download-sample-monthly-btn",
                                                 leftSection=DashIconify(icon="tabler:file-download", width=14)),
                                    dmc.MenuDivider(),
                                    dmc.MenuItem("Exit", id="reg-menu-exit", color="red",
                                                 leftSection=DashIconify(icon="tabler:door-exit", width=14)),
                                ]),
                            ],
                        ),
                        dmc.Menu(
                            trigger="click", openDelay=100, closeDelay=200,
                            position="bottom-start", shadow="md", offset=6,
                            children=[
                                dmc.MenuTarget(dmc.Button("Edit", variant="subtle", color="gray", size="sm", radius="sm")),
                                dmc.MenuDropdown(className="dashmat-menu-dropdown", children=[
                                    dmc.MenuItem("Clear all series", id="reg-menu-clear-series", color="red",
                                                 leftSection=DashIconify(icon="tabler:trash", width=14)),
                                ]),
                            ],
                        ),
                        dmc.Menu(
                            trigger="click", openDelay=100, closeDelay=200,
                            position="bottom-start", shadow="md", offset=6,
                            children=[
                                dmc.MenuTarget(dmc.Button("Help", variant="subtle", color="gray", size="sm", radius="sm")),
                                dmc.MenuDropdown(className="dashmat-menu-dropdown", children=[
                                    dmc.MenuItem("User Guide", id="reg-menu-help-guide",
                                                 leftSection=DashIconify(icon="tabler:help-circle", width=14)),
                                ]),
                            ],
                        ),
                    ],
                ),
            ],
        ),

        # Welcome screen
        html.Div(
            id="reg-welcome-screen",
            children=build_reg_welcome_screen(),
            style={"display": "flex", "flex": "1", "alignItems": "center", "justifyContent": "center"},
        ),

        # Main container
        html.Div(
            id="reg-main-container",
            children=build_reg_main_layout(),
            style={"display": "none", "flex": "1", "flexDirection": "column", "overflow": "hidden"},
        ),

        # Modals
        build_db_add_modal("reg"),
        build_series_selection_modal(REG_CONFIG),
        build_sheet_select_modal(REG_CONFIG.prefix),

        # Help modal
        dmc.Modal(
            id="reg-help-modal",
            title=dmc.Group(gap="xs", children=[
                dmc.ThemeIcon(DashIconify(icon="tabler:help-circle"), color="blue", variant="light", size="sm"),
                dmc.Text("Regression Analysis — User Guide", fw=600, size="sm"),
            ]),
            size="lg", centered=True, withCloseButton=True, radius="lg",
            className="dashmat-modal",
            overlayProps={"blur": 2, "opacity": 0.45},
            children=[
                dmc.Accordion(children=[
                    dmc.AccordionItem(value="overview", children=[
                        dmc.AccordionControl("Overview"),
                        dmc.AccordionPanel(dmc.Text(
                            "Run OLS, Constrained OLS, Style Analysis, Ridge, Lasso, or Elastic Net on return series. "
                            "Select Y (dependent) and X (independent) series, configure the model, then click Run.",
                            size="sm")),
                    ]),
                    dmc.AccordionItem(value="series", children=[
                        dmc.AccordionControl("Series Selection"),
                        dmc.AccordionPanel(dmc.Stack(gap="xs", children=[
                            dmc.Text("• Y: Exactly one series as the dependent variable.", size="sm"),
                            dmc.Text("• X: One or more series as independent variables.", size="sm"),
                            dmc.Text("• Lag: Shift an X series by N periods before regression.", size="sm"),
                            dmc.Text("• Min/Max Beta + Enable: Per-variable beta bounds (Constrained OLS).", size="sm"),
                        ])),
                    ]),
                    dmc.AccordionItem(value="models", children=[
                        dmc.AccordionControl("Models"),
                        dmc.AccordionPanel(dmc.Stack(gap="xs", children=[
                            dmc.Text("• OLS: Full ANOVA, t-stats, p-values, diagnostics.", size="sm"),
                            dmc.Text("• Style Analysis: Weights sum to 1, bounded [0,1], no intercept.", size="sm"),
                            dmc.Text("• ARIMA/GARCH: Post-regression residual modeling (OLS/Constrained OLS).", size="sm"),
                        ])),
                    ]),
                ]),
            ],
        ),

        # ---- Stores ----
        dcc.Store(id="reg-series-select", data=[], storage_type="session"),
        dcc.Store(id="reg-series-order-store", data=[], storage_type="session"),
        dcc.Store(id="reg-benchmark-assignments-store", data={}, storage_type="session"),
        dcc.Store(id="reg-long-short-store", data={}, storage_type="session"),
        dcc.Store(id="reg-vol-scaling-assignments-store", data={}, storage_type="session"),
        dcc.Store(id="reg-dependent-var-store", data=None, storage_type="session"),
        dcc.Store(id="reg-lag-store", data={}, storage_type="session"),
        dcc.Store(id="reg-min-beta-store", data={}, storage_type="session"),
        dcc.Store(id="reg-max-beta-store", data={}, storage_type="session"),
        dcc.Store(id="reg-enable-constraint-store", data={}, storage_type="session"),
        # Temp stores
        dcc.Store(id="reg-temp-series-select", data=[]),
        dcc.Store(id="reg-temp-series-order-store", data=[]),
        dcc.Store(id="reg-temp-deleted-series-store", data=[]),
        dcc.Store(id="reg-temp-benchmark-assignments-store", data={}),
        dcc.Store(id="reg-temp-long-short-store", data={}),
        dcc.Store(id="reg-temp-vol-scaling-assignments-store", data={}),
        dcc.Store(id="reg-temp-dependent-var-store", data=None),
        dcc.Store(id="reg-temp-lag-store", data={}),
        dcc.Store(id="reg-temp-min-beta-store", data={}),
        dcc.Store(id="reg-temp-max-beta-store", data={}),
        dcc.Store(id="reg-temp-enable-constraint-store", data={}),
        # Sheet select temp
        dcc.Store(id="reg-sheet-select-contents-store", data=None),
        dcc.Store(id="reg-sheet-select-filename-store", data=None),
        dcc.Store(id="reg-sheet-select-sheetnames-store", data=None),
        # Controls
        dcc.Store(id="reg-periodicity-value-store", data="daily", storage_type="session"),
        dcc.Store(id="reg-periodicity-load-sync-dummy", data=None),
        dcc.Store(id="reg-vol-scaler-value-store", data=0, storage_type="session"),
        dcc.Store(id="reg-date-range-store", data=None, storage_type="session"),
        dcc.Store(id="reg-series-select-value-store", data=[], storage_type="session"),
        # Regression settings
        dcc.Store(id="reg-model-store", data="ols", storage_type="session"),
        dcc.Store(id="reg-regression-name-store", data="OLS_1", storage_type="session"),
        dcc.Store(id="reg-force-zero-intercept-store", data=False, storage_type="session"),
        dcc.Store(id="reg-robust-se-store", data=False, storage_type="session"),
        dcc.Store(id="reg-exp-wt-store", data=False, storage_type="session"),
        dcc.Store(id="reg-halflife-store", data=63, storage_type="session"),
        dcc.Store(id="reg-window-type-store", data="full", storage_type="session"),
        dcc.Store(id="reg-window-size-store", data=36, storage_type="session"),
        dcc.Store(id="reg-opt-step-store", data=1, storage_type="session"),
        dcc.Store(id="reg-opt-step-unit-store", data="months", storage_type="session"),
        dcc.Store(id="reg-fill-in-sample-store", data="off", storage_type="session"),
        dcc.Store(id="reg-missing-data-store", data="fill_na", storage_type="session"),
        dcc.Store(id="reg-alpha-store", data=1.0, storage_type="session"),
        dcc.Store(id="reg-l1-ratio-store", data=0.5, storage_type="session"),
        dcc.Store(id="reg-linear-constraints-store", data=[], storage_type="session"),
        # Results
        dcc.Store(id="reg-results-store", data={}, storage_type="session"),
        dcc.Store(id="reg-active-tab-store", data="anova", storage_type="session"),
        # Uploads / downloads
        html.Div(
            dcc.Upload(id="reg-upload-data", children=html.Div(), multiple=False),
            style={"display": "none"},
        ),
        dcc.Download(id="reg-download-excel"),
        dcc.Download(id="reg-download-sample-daily"),
        dcc.Download(id="reg-download-sample-monthly"),
        dcc.Location(id="reg-url-location", refresh=False),
        dcc.Interval(id="reg-page-load-trigger", interval=50, max_intervals=1, n_intervals=0),
    ],
)


# ===========================================================================
# Clientside callbacks
# ===========================================================================

clientside_callback(
    "function(n) { return true; }",
    Output("reg-help-modal", "opened"),
    Input("reg-menu-help-guide", "n_clicks"),
    prevent_initial_call=True,
)

clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks) { window.location.href = '/'; }
        return window.dash_clientside.no_update;
    }
    """,
    Output("reg-url-location", "pathname"),
    Input("reg-menu-exit", "n_clicks"),
    prevent_initial_call=True,
)

clientside_callback(
    "function(checked) { return !checked; }",
    Output("reg-halflife-input", "disabled"),
    Input("reg-exp-wt-switch", "checked"),
    prevent_initial_call=False,
)

clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks) {
            setTimeout(function() {
                var uploadDiv = document.getElementById('reg-upload-data');
                if (uploadDiv) {
                    var input = uploadDiv.querySelector('input[type="file"]');
                    if (input) {
                        input.click();
                    }
                }
            }, 100);
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("reg-upload-data", "contents", allow_duplicate=True),
    Input("reg-menu-add-series", "n_clicks"),
    State("reg-upload-data", "contents"),
    prevent_initial_call=True,
)

clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks) {
            setTimeout(function() {
                var uploadDiv = document.getElementById('reg-upload-data');
                if (uploadDiv) {
                    var input = uploadDiv.querySelector('input[type="file"]');
                    if (input) {
                        input.click();
                    }
                }
            }, 100);
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("reg-upload-data", "contents", allow_duplicate=True),
    Input("reg-welcome-add-series-btn", "n_clicks"),
    State("reg-upload-data", "contents"),
    prevent_initial_call=True,
)

# Store syncs (controls → stores)
clientside_callback("function(v){return v;}",
    Output("reg-periodicity-value-store","data"), Input("reg-periodicity-select","value"), prevent_initial_call=True)
clientside_callback("function(v){return v;}",
    Output("reg-periodicity-select","value", allow_duplicate=True), Input("reg-periodicity-load-sync-dummy","data"), prevent_initial_call=True)
clientside_callback("function(v){return v??0;}",
    Output("reg-vol-scaler-value-store","data"), Input("reg-vol-scaler-input","value"), prevent_initial_call=True)
clientside_callback("function(v){return v;}",
    Output("reg-model-store","data"), Input("reg-model-select","value"), prevent_initial_call=True)
clientside_callback("function(v){return v;}",
    Output("reg-regression-name-store","data"), Input("reg-regression-name-input","value"), prevent_initial_call=True)
clientside_callback("function(v){return v;}",
    Output("reg-force-zero-intercept-store","data"), Input("reg-force-zero-intercept-switch","checked"), prevent_initial_call=True)
clientside_callback("function(v){return v;}",
    Output("reg-robust-se-store","data"), Input("reg-robust-se-switch","checked"), prevent_initial_call=True)
clientside_callback("function(v){return v;}",
    Output("reg-exp-wt-store","data"), Input("reg-exp-wt-switch","checked"), prevent_initial_call=True)
clientside_callback("function(v){return v??63;}",
    Output("reg-halflife-store","data"), Input("reg-halflife-input","value"), prevent_initial_call=True)
clientside_callback("function(v){return v;}",
    Output("reg-window-type-store","data"), Input("reg-window-type-select","value"), prevent_initial_call=True)
clientside_callback("function(v){return v??36;}",
    Output("reg-window-size-store","data"), Input("reg-window-size-input","value"), prevent_initial_call=True)
clientside_callback("function(v){return v??1;}",
    Output("reg-opt-step-store","data"), Input("reg-opt-step-input","value"), prevent_initial_call=True)
clientside_callback("function(v){return v;}",
    Output("reg-opt-step-unit-store","data"), Input("reg-opt-step-unit-select","value"), prevent_initial_call=True)
clientside_callback("function(v){return v;}",
    Output("reg-fill-in-sample-store","data"), Input("reg-fill-in-sample-select","value"), prevent_initial_call=True)
clientside_callback("function(v){return v;}",
    Output("reg-missing-data-store","data"), Input("reg-missing-data-select","value"), prevent_initial_call=True)
clientside_callback("function(v){return v??1.0;}",
    Output("reg-alpha-store","data"), Input("reg-alpha-input","value"), prevent_initial_call=True)
clientside_callback("function(v){return v??0.5;}",
    Output("reg-l1-ratio-store","data"), Input("reg-l1-ratio-input","value"), prevent_initial_call=True)
clientside_callback("function(v){return v;}",
    Output("reg-active-tab-store","data"), Input("reg-tabs","value"), prevent_initial_call=True)


# ===========================================================================
# Server callbacks
# ===========================================================================

@callback(
    Output("reg-arima-garch-panel", "style"),
    Output("reg-alpha-container", "style"),
    Output("reg-l1-ratio-container", "style"),
    Output("reg-force-zero-intercept-switch", "disabled"),
    Input("reg-model-select", "value"),
    prevent_initial_call=False,
)
def reg_toggle_model_controls(model):
    show = {"display": "block"}
    hide = {"display": "none"}
    arima = show if model in ("ols", "constrained_ols") else hide
    alpha = show if model in ("ridge", "lasso", "elastic_net") else hide
    l1 = show if model == "elastic_net" else hide
    return arima, alpha, l1, (model == "style_analysis")


@callback(
    Output("reg-force-zero-intercept-switch", "checked"),
    Input("reg-model-select", "value"),
    State("reg-force-zero-intercept-switch", "checked"),
    prevent_initial_call=True,
)
def reg_force_zero_for_style(model, current):
    return True if model == "style_analysis" else current


@callback(
    Output("reg-window-size-input", "disabled"),
    Output("reg-opt-step-input", "disabled"),
    Output("reg-opt-step-unit-select", "disabled"),
    Input("reg-window-type-select", "value"),
    prevent_initial_call=False,
)
def reg_toggle_window_controls(window_type):
    is_full = window_type == "full"
    return is_full, is_full, is_full


# ---------------------------------------------------------------------------
# DB add modal (AA Tool indices)
# ---------------------------------------------------------------------------

@callback(
    Output("reg-db-add-modal", "opened", allow_duplicate=True),
    Output("reg-db-add-series-select", "data", allow_duplicate=True),
    Output("reg-db-add-series-select", "value", allow_duplicate=True),
    Input("reg-welcome-add-db-btn", "n_clicks"),
    prevent_initial_call=True,
)
def reg_open_db_add_modal(welcome_clicks):
    return compute_open_db_add_modal(None, welcome_clicks, DB_ENGINE)


@callback(
    Output("reg-db-add-modal", "opened", allow_duplicate=True),
    Output("reg-db-add-series-select", "value", allow_duplicate=True),
    Input("reg-db-add-cancel-button", "n_clicks"),
    prevent_initial_call=True,
)
def reg_close_db_add_modal(n_clicks):
    return compute_close_db_add_modal(n_clicks)


@callback(
    Output("reg-db-add-error-alert", "children"),
    Output("reg-db-add-error-alert", "hide"),
    Output("reg-db-add-ok-button", "disabled"),
    Input("reg-db-add-series-select", "value"),
    Input("dashmat-raw-data-store", "data"),
    Input("reg-db-add-modal", "opened"),
    prevent_initial_call=True,
)
def reg_validate_db_add_selection(selected_benches, raw_data, opened):
    return compute_validate_db_add_selection(selected_benches, raw_data, opened)


@callback(
    Output("dashmat-raw-data-store", "data", allow_duplicate=True),
    Output("dashmat-original-periodicity-store", "data", allow_duplicate=True),
    Output("reg-periodicity-value-store", "data", allow_duplicate=True),
    Output("reg-periodicity-load-sync-dummy", "data", allow_duplicate=True),
    Output("reg-db-add-modal", "opened", allow_duplicate=True),
    Output("reg-db-add-series-select", "value", allow_duplicate=True),
    Output("reg-db-add-error-alert", "children", allow_duplicate=True),
    Output("reg-db-add-error-alert", "hide", allow_duplicate=True),
    Input("reg-db-add-ok-button", "n_clicks"),
    State("reg-db-add-series-select", "value"),
    State("dashmat-raw-data-store", "data"),
    State("dashmat-original-periodicity-store", "data"),
    prevent_initial_call=True,
)
def reg_add_series_from_database(n_clicks, selected_benches, existing_data, existing_periodicity):
    if not n_clicks:
        raise PreventUpdate

    if not selected_benches:
        return no_update, no_update, no_update, no_update, True, no_update, "Select at least one series.", False

    try:
        if existing_data:
            existing_cols = set(json_to_df(existing_data).columns)
            duplicates = [s for s in selected_benches if s in existing_cols]
            if duplicates:
                return (
                    no_update, no_update, no_update, no_update,
                    True, no_update, f"Cannot add duplicate series: {', '.join(duplicates)}", False,
                )

        new_df, _db_meta = load_cma_returns_for_benches_with_meta(DB_ENGINE, selected_benches, MRD_ENGINE)
        if new_df.empty:
            return (
                no_update, no_update, no_update, no_update,
                True, no_update, "No rows returned for selected series.", False,
            )

        merge_result = _shared_merge_uploaded_with_existing(existing_data, existing_periodicity, new_df)
        merged_df = merge_result.merged_df
        merged_periodicity = merge_result.combined_periodicity
        return (
            df_to_json(merged_df),
            merged_periodicity,
            merged_periodicity,
            merged_periodicity,
            False,
            [],
            no_update,
            True,
        )
    except Exception as exc:
        return (
            no_update, no_update, no_update, no_update,
            True, no_update, f"Error loading database series: {exc}", False,
        )


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

@callback(
    Output("reg-sheet-select-sheetnames-store", "data"),
    Output("reg-sheet-select-contents-store", "data"),
    Output("reg-sheet-select-filename-store", "data"),
    Output("reg-sheet-select-modal", "opened"),
    Output("dashmat-raw-data-store", "data", allow_duplicate=True),
    Output("dashmat-original-periodicity-store", "data", allow_duplicate=True),
    Output("reg-periodicity-value-store", "data", allow_duplicate=True),
    Output("reg-periodicity-load-sync-dummy", "data", allow_duplicate=True),
    Input("reg-upload-data", "contents"),
    State("reg-upload-data", "filename"),
    State("dashmat-raw-data-store", "data"),
    State("dashmat-original-periodicity-store", "data"),
    prevent_initial_call=True,
)
def reg_handle_upload(contents, filename, existing_raw, existing_periodicity):
    if not contents:
        raise PreventUpdate
    sheet_names = get_sheet_names(contents, filename)
    if sheet_names and len(sheet_names) > 1:
        return sheet_names, contents, filename, True, no_update, no_update, no_update, no_update
    try:
        new_df = _shared_import_single_upload(contents, filename)
    except Exception:
        raise PreventUpdate
    merge_result = _shared_merge_uploaded_with_existing(existing_raw, existing_periodicity, new_df)
    merged_df = merge_result.merged_df
    merged_periodicity = merge_result.combined_periodicity
    return (no_update, no_update, no_update, False,
            df_to_json(merged_df), merged_periodicity, merged_periodicity, merged_periodicity)


@callback(
    Output("dashmat-raw-data-store", "data", allow_duplicate=True),
    Output("dashmat-original-periodicity-store", "data", allow_duplicate=True),
    Output("reg-periodicity-value-store", "data", allow_duplicate=True),
    Output("reg-periodicity-load-sync-dummy", "data", allow_duplicate=True),
    Output("reg-sheet-select-modal", "opened", allow_duplicate=True),
    Input("reg-sheet-select-ok-button", "n_clicks"),
    State("reg-sheet-select-dropdown", "value"),
    State("reg-sheet-select-contents-store", "data"),
    State("reg-sheet-select-filename-store", "data"),
    State("dashmat-raw-data-store", "data"),
    State("dashmat-original-periodicity-store", "data"),
    prevent_initial_call=True,
)
def reg_handle_sheet_select_ok(n_clicks, selected_sheets, contents, filename, existing_raw, existing_periodicity):
    if not n_clicks or not selected_sheets or not contents:
        raise PreventUpdate
    try:
        new_df, _imported_sheets = _shared_import_selected_workbook_sheets(contents, filename, selected_sheets)
    except Exception:
        raise PreventUpdate
    merge_result = _shared_merge_uploaded_with_existing(existing_raw, existing_periodicity, new_df)
    merged_df = merge_result.merged_df
    merged_periodicity = merge_result.combined_periodicity
    return df_to_json(merged_df), merged_periodicity, merged_periodicity, merged_periodicity, False


# ---------------------------------------------------------------------------
# Welcome / main visibility
# ---------------------------------------------------------------------------

@callback(
    Output("reg-welcome-screen", "style"),
    Output("reg-main-container", "style"),
    Output("reg-periodicity-select", "data"),
    Output("reg-periodicity-select", "value"),
    Input("dashmat-raw-data-store", "data"),
    State("dashmat-original-periodicity-store", "data"),
    State("reg-periodicity-value-store", "data"),
    prevent_initial_call=False,
)
def reg_toggle_welcome(raw_data, original_periodicity, stored_periodicity):
    hide_welcome = {"display": "none"}
    show_welcome = {"display": "flex", "flex": "1", "alignItems": "center", "justifyContent": "center"}
    show_main = {"display": "flex", "flex": "1", "flexDirection": "column", "overflow": "hidden"}
    hide_main = {"display": "none", "flex": "1", "flexDirection": "column", "overflow": "hidden"}
    if not raw_data:
        return show_welcome, hide_main, [{"value": "daily", "label": "Daily"}], "daily"

    period_data = get_available_periodicities(original_periodicity or "daily")
    valid_values = [option["value"] for option in period_data]
    default_value = (
        original_periodicity
        if original_periodicity in valid_values
        else (valid_values[0] if valid_values else "daily")
    )
    period_value = (
        stored_periodicity
        if (stored_periodicity and stored_periodicity in valid_values)
        else default_value
    )
    return hide_welcome, show_main, period_data, period_value


# ---------------------------------------------------------------------------
# Open modal
# ---------------------------------------------------------------------------

@callback(
    Output("reg-series-selection-modal", "opened"),
    Output("reg-temp-series-select", "data", allow_duplicate=True),
    Output("reg-temp-series-order-store", "data", allow_duplicate=True),
    Output("reg-temp-deleted-series-store", "data", allow_duplicate=True),
    Output("reg-temp-benchmark-assignments-store", "data", allow_duplicate=True),
    Output("reg-temp-long-short-store", "data", allow_duplicate=True),
    Output("reg-temp-vol-scaling-assignments-store", "data", allow_duplicate=True),
    Output("reg-temp-dependent-var-store", "data", allow_duplicate=True),
    Output("reg-temp-lag-store", "data", allow_duplicate=True),
    Output("reg-temp-min-beta-store", "data", allow_duplicate=True),
    Output("reg-temp-max-beta-store", "data", allow_duplicate=True),
    Output("reg-temp-enable-constraint-store", "data", allow_duplicate=True),
    Input("reg-open-modal-button", "n_clicks"),
    State("reg-series-select", "data"),
    State("reg-series-order-store", "data"),
    State("reg-benchmark-assignments-store", "data"),
    State("reg-long-short-store", "data"),
    State("reg-vol-scaling-assignments-store", "data"),
    State("reg-dependent-var-store", "data"),
    State("reg-lag-store", "data"),
    State("reg-min-beta-store", "data"),
    State("reg-max-beta-store", "data"),
    State("reg-enable-constraint-store", "data"),
    prevent_initial_call=True,
)
def reg_open_modal(n_clicks, sel, order, bench, ls, vol_scale, dep_var,
                   lag, min_beta, max_beta, enable):
    if not n_clicks:
        raise PreventUpdate
    return (True, sel or [], order or [], [],
            bench or {}, ls or {}, vol_scale or {},
            dep_var, lag or {}, min_beta or {}, max_beta or {}, enable or {})


# ---------------------------------------------------------------------------
# Series grid population
# ---------------------------------------------------------------------------

@callback(
    Output("reg-series-selection-container", "children"),
    Output("reg-temp-series-order-store", "data", allow_duplicate=True),
    Input("dashmat-raw-data-store", "data"),
    Input("reg-temp-series-select", "data"),
    Input("reg-temp-series-order-store", "data"),
    Input("reg-temp-deleted-series-store", "data"),
    Input("reg-series-selection-grid", "cellValueChanged", allow_optional=True),
    Input("reg-temp-benchmark-assignments-store", "data"),
    Input("reg-temp-long-short-store", "data"),
    Input("reg-temp-vol-scaling-assignments-store", "data"),
    Input("reg-temp-dependent-var-store", "data"),
    Input("reg-temp-lag-store", "data"),
    Input("reg-temp-min-beta-store", "data"),
    Input("reg-temp-max-beta-store", "data"),
    Input("reg-temp-enable-constraint-store", "data"),
    prevent_initial_call="initial_duplicate",
)
def reg_update_series_grid(raw_data, selected_x, series_order, deleted_series,
                            _cell, bench_assign, ls_assign, vol_assign,
                            dep_var, lag_assign, min_b_assign, max_b_assign, enable_assign):
    if raw_data is None:
        return [dmc.Text("Upload data to select series", size="sm", c="dimmed")], []
    df = json_to_df(raw_data)
    all_series = list(df.columns)
    if not all_series:
        return [dmc.Text("Upload data to select series", size="sm", c="dimmed")], []
    if not series_order:
        series_order = list(all_series)
    else:
        for s in all_series:
            if s not in series_order:
                series_order.append(s)
        series_order = [s for s in series_order if s in all_series]
    x_set = set(selected_x or [])
    deleted_set = set(deleted_series or [])
    bench_assign = bench_assign or {}
    ls_assign = ls_assign or {}
    vol_assign = vol_assign or {}
    lag_assign = lag_assign or {}
    min_b_assign = min_b_assign or {}
    max_b_assign = max_b_assign or {}
    enable_assign = enable_assign or {}
    benchmark_values = ["None"] + list(all_series)
    row_data = []
    for series in series_order:
        bench_val = bench_assign.get(series, "None")
        if bench_val not in all_series and bench_val != "None":
            bench_val = "None"
        row_data.append({
            "Series": series,
            "Y": bool(series == dep_var),
            "X": bool(series in x_set),
            "Benchmark": bench_val,
            "LongShort": bool(ls_assign.get(series, False)),
            "ScaleVol": bool(vol_assign.get(series, True)),
            "Lag": int(lag_assign.get(series, 0) or 0),
            "MinBeta": float(min_b_assign.get(series, -999.0) or -999.0),
            "MaxBeta": float(max_b_assign.get(series, 999.0) or 999.0),
            "Enable": bool(enable_assign.get(series, False)),
            "Delete": series in deleted_set,
        })
    grid = dag.AgGrid(
        id="reg-series-selection-grid",
        className="ag-theme-alpine dashmat-series-modal-grid",
        getRowId="params.data.Series",
        columnDefs=[
            {"headerName": "", "rowDrag": True, "editable": False, "sortable": False, "filter": False,
             "resizable": False, "width": 36, "pinned": "left", "valueGetter": {"function": "''"},
             "cellClass": "dashmat-series-center-cell"},
            {"field": "Y", "headerName": "Y", "editable": True, "cellRenderer": "agCheckboxCellRenderer",
             "cellEditor": "agCheckboxCellEditor", "width": 54, "pinned": "left",
             "cellClass": "dashmat-series-center-cell", "headerClass": "dashmat-center-header"},
            {"field": "X", "headerName": "X", "editable": True, "cellRenderer": "agCheckboxCellRenderer",
             "cellEditor": "agCheckboxCellEditor", "width": 54, "pinned": "left",
             "cellClass": "dashmat-series-center-cell", "headerClass": "dashmat-center-header"},
            {"field": "Series", "editable": True, "minWidth": 150,
             "cellStyle": {"textAlign": "left", "fontFamily": "monospace"},
             "headerClass": "dashmat-left-header"},
            {"field": "Benchmark", "editable": True, "cellEditor": "agSelectCellEditor",
             "cellEditorParams": {"values": benchmark_values}, "minWidth": 140,
             "cellStyle": {"textAlign": "left"}, "headerClass": "dashmat-left-header"},
            {"field": "LongShort", "headerName": "L/S", "editable": True,
             "cellRenderer": "agCheckboxCellRenderer", "cellEditor": "agCheckboxCellEditor",
             "width": 60, "cellClass": "dashmat-series-center-cell"},
            {"field": "ScaleVol", "headerName": "Scale Vol", "editable": True,
             "cellRenderer": "agCheckboxCellRenderer", "cellEditor": "agCheckboxCellEditor",
             "width": 100, "cellClass": "dashmat-series-center-cell"},
            {"field": "Lag", "headerName": "Lag", "editable": True, "width": 70,
             "valueParser": {"function": "var n=parseInt(params.newValue); return isNaN(n)?0:n;"},
             "cellClass": "dashmat-series-center-cell", "headerClass": "dashmat-center-header"},
            {"field": "MinBeta", "headerName": "Min Beta", "editable": {"function": "params.data.Enable"},
             "width": 100, "valueParser": {"function": "var n=Number(params.newValue); return isFinite(n)?n:-999;"},
             "valueFormatter": {"function": "params.value != null ? params.value.toFixed(2) : ''"},
             "cellClass": "dashmat-series-center-cell", "headerClass": "dashmat-center-header"},
            {"field": "MaxBeta", "headerName": "Max Beta", "editable": {"function": "params.data.Enable"},
             "width": 100, "valueParser": {"function": "var n=Number(params.newValue); return isFinite(n)?n:999;"},
             "valueFormatter": {"function": "params.value != null ? params.value.toFixed(2) : ''"},
             "cellClass": "dashmat-series-center-cell", "headerClass": "dashmat-center-header"},
            {"field": "Enable", "headerName": "Enable", "editable": True,
             "cellRenderer": "agCheckboxCellRenderer", "cellEditor": "agCheckboxCellEditor",
             "width": 80, "cellClass": "dashmat-series-center-cell", "headerClass": "dashmat-center-header"},
            {"field": "Delete", "headerName": "Del", "editable": True,
             "cellRenderer": "agCheckboxCellRenderer", "cellEditor": "agCheckboxCellEditor",
             "width": 60, "cellClass": "dashmat-series-center-cell", "headerClass": "dashmat-center-header"},
        ],
        rowData=row_data,
        defaultColDef={"resizable": True, "sortable": False, "suppressHeaderMenuButton": True},
        dashGridOptions={"rowDragManaged": True, "singleClickEdit": True,
                         "suppressExcelExport": True, "suppressCsvExport": True},
        style={"height": "400px"},
    )
    return [grid], series_order


# ---------------------------------------------------------------------------
# Sync grid cell changes to temp stores
# ---------------------------------------------------------------------------

@callback(
    Output("reg-temp-series-select", "data", allow_duplicate=True),
    Output("reg-temp-dependent-var-store", "data", allow_duplicate=True),
    Output("reg-temp-lag-store", "data", allow_duplicate=True),
    Output("reg-temp-min-beta-store", "data", allow_duplicate=True),
    Output("reg-temp-max-beta-store", "data", allow_duplicate=True),
    Output("reg-temp-enable-constraint-store", "data", allow_duplicate=True),
    Output("reg-temp-benchmark-assignments-store", "data", allow_duplicate=True),
    Output("reg-temp-long-short-store", "data", allow_duplicate=True),
    Output("reg-temp-vol-scaling-assignments-store", "data", allow_duplicate=True),
    Output("reg-temp-deleted-series-store", "data", allow_duplicate=True),
    Output("reg-temp-series-order-store", "data", allow_duplicate=True),
    Input("reg-series-selection-grid", "cellValueChanged"),
    Input("reg-series-selection-grid", "rowData"),
    State("reg-temp-dependent-var-store", "data"),
    prevent_initial_call=True,
)
def reg_sync_grid_to_temp(cell_change, row_data, cur_dep):
    if not row_data:
        raise PreventUpdate
    new_x, new_dep, new_lag, new_min, new_max = [], None, {}, {}, {}
    new_enable, new_bench, new_ls, new_vol, new_deleted, new_order = {}, {}, {}, {}, [], []
    changed_field = (cell_change or {}).get("colId")
    changed_series = (cell_change.get("data", {}).get("Series") if cell_change else None)
    for row in row_data:
        series = row.get("Series", "")
        new_order.append(series)
        y_val = bool(row.get("Y", False))
        # Enforce single-select for Y
        if changed_field == "Y":
            if series == changed_series and y_val:
                new_dep = series
            elif series != changed_series and y_val:
                y_val = False  # clear others (will be re-set on next grid rebuild)
        elif y_val:
            new_dep = series
        if bool(row.get("X", False)):
            new_x.append(series)
        new_bench[series] = row.get("Benchmark") or "None"
        new_ls[series] = bool(row.get("LongShort", False))
        new_vol[series] = bool(row.get("ScaleVol", True))
        try:
            new_lag[series] = int(row.get("Lag", 0) or 0)
        except (ValueError, TypeError):
            new_lag[series] = 0
        try:
            new_min[series] = float(row.get("MinBeta", -999.0) or -999.0)
        except (ValueError, TypeError):
            new_min[series] = -999.0
        try:
            new_max[series] = float(row.get("MaxBeta", 999.0) or 999.0)
        except (ValueError, TypeError):
            new_max[series] = 999.0
        new_enable[series] = bool(row.get("Enable", False))
        if bool(row.get("Delete", False)):
            new_deleted.append(series)
    return (new_x, new_dep, new_lag, new_min, new_max, new_enable,
            new_bench, new_ls, new_vol, new_deleted, new_order)


# ---------------------------------------------------------------------------
# Modal OK
# ---------------------------------------------------------------------------

@callback(
    Output("reg-series-select", "data"),
    Output("reg-benchmark-assignments-store", "data"),
    Output("reg-long-short-store", "data"),
    Output("reg-series-order-store", "data"),
    Output("reg-series-selection-modal", "opened", allow_duplicate=True),
    Output("reg-series-select-value-store", "data"),
    Output("dashmat-raw-data-store", "data", allow_duplicate=True),
    Output("reg-vol-scaling-assignments-store", "data"),
    Output("reg-dependent-var-store", "data"),
    Output("reg-lag-store", "data"),
    Output("reg-min-beta-store", "data"),
    Output("reg-max-beta-store", "data"),
    Output("reg-enable-constraint-store", "data"),
    Input("reg-modal-ok-button", "n_clicks"),
    State("reg-temp-series-select", "data"),
    State("reg-temp-benchmark-assignments-store", "data"),
    State("reg-temp-long-short-store", "data"),
    State("reg-temp-series-order-store", "data"),
    State("reg-temp-deleted-series-store", "data"),
    State("dashmat-raw-data-store", "data"),
    State("reg-temp-vol-scaling-assignments-store", "data"),
    State("reg-temp-dependent-var-store", "data"),
    State("reg-temp-lag-store", "data"),
    State("reg-temp-min-beta-store", "data"),
    State("reg-temp-max-beta-store", "data"),
    State("reg-temp-enable-constraint-store", "data"),
    prevent_initial_call=True,
)
def reg_on_modal_ok(n_clicks, temp_x, temp_bench, temp_ls, temp_order, temp_deleted,
                    raw_data, temp_vol, temp_dep, temp_lag, temp_min, temp_max, temp_enable):
    if not n_clicks:
        raise PreventUpdate
    temp_x = list(temp_x or [])
    if temp_order:
        x_set = set(temp_x)
        temp_x = [s for s in temp_order if s in x_set]
    updated_raw = raw_data
    if temp_deleted and raw_data:
        df = json_to_df(raw_data)
        to_drop = [s for s in temp_deleted if s in df.columns]
        if to_drop:
            df = df.drop(columns=to_drop)
            updated_raw = df_to_json(df)
            drop_set = set(to_drop)
            temp_x = [s for s in temp_x if s not in drop_set]
            if temp_dep in drop_set:
                temp_dep = None
            if temp_order:
                temp_order = [s for s in temp_order if s not in drop_set]
            for store in [temp_bench, temp_ls, temp_vol, temp_lag, temp_min, temp_max, temp_enable]:
                if store:
                    for s in to_drop:
                        store.pop(s, None)
    return (temp_x, temp_bench or {}, temp_ls or {}, temp_order or [],
            False, temp_x, updated_raw, temp_vol or {},
            temp_dep, temp_lag or {}, temp_min or {}, temp_max or {}, temp_enable or {})


@callback(
    Output("reg-series-selection-modal", "opened", allow_duplicate=True),
    Input("reg-modal-cancel-button", "n_clicks"),
    prevent_initial_call=True,
)
def reg_on_modal_cancel(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    return False


# ---------------------------------------------------------------------------
# Date range
# ---------------------------------------------------------------------------

@callback(
    Output("reg-start-date-picker", "value"),
    Output("reg-end-date-picker", "value"),
    Output("reg-date-picker-wrapper", "style"),
    Output("reg-common-range-button", "disabled"),
    Output("reg-maximum-range-button", "disabled"),
    Output("reg-date-range-store", "data", allow_duplicate=True),
    Input("dashmat-raw-data-store", "data"),
    Input("reg-periodicity-select", "value"),
    Input("reg-series-select", "data"),
    Input("reg-dependent-var-store", "data"),
    State("reg-date-range-store", "data"),
    prevent_initial_call="initial_duplicate",
)
def reg_init_date_range(raw_data, periodicity, x_series, dep_var, stored_range):
    disabled_style = {"display": "flex", "opacity": 0.5, "pointerEvents": "none", "alignItems": "flex-start"}
    enabled_style = {"display": "flex", "alignItems": "flex-start"}
    all_series = list(set((x_series or []) + ([dep_var] if dep_var else [])))
    if not raw_data or not all_series:
        return None, None, disabled_style, True, True, None
    try:
        candidates = compute_date_range_candidates(raw_data, periodicity or "daily", tuple(all_series))
        if not candidates.get("available_series"):
            return None, None, disabled_style, True, True, None
        start_date, end_date = resolve_initial_range(candidates, stored_range)
        if not start_date or not end_date:
            return None, None, disabled_style, True, True, None
        new_range = {"start": str(start_date)[:10], "end": str(end_date)[:10]}
        return str(start_date)[:10], str(end_date)[:10], enabled_style, False, False, new_range
    except Exception:
        return None, None, disabled_style, True, True, None


@callback(
    Output("reg-start-date-picker", "value", allow_duplicate=True),
    Output("reg-end-date-picker", "value", allow_duplicate=True),
    Output("reg-date-range-store", "data", allow_duplicate=True),
    Input("reg-common-range-button", "n_clicks"),
    Input("reg-maximum-range-button", "n_clicks"),
    State("dashmat-raw-data-store", "data"),
    State("reg-periodicity-select", "value"),
    State("reg-series-select", "data"),
    State("reg-dependent-var-store", "data"),
    prevent_initial_call=True,
)
def reg_date_range_button(n_common, n_max, raw_data, periodicity, x_series, dep_var):
    all_series = list(set((x_series or []) + ([dep_var] if dep_var else [])))
    if not raw_data or not all_series:
        raise PreventUpdate
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    try:
        candidates = compute_date_range_candidates(raw_data, periodicity or "daily", tuple(all_series))
        start, end, _ = resolve_button_range(candidates, button_id)
        if not start or not end:
            raise PreventUpdate
        return str(start)[:10], str(end)[:10], {"start": str(start)[:10], "end": str(end)[:10]}
    except Exception:
        raise PreventUpdate


@callback(
    Output("reg-date-range-store", "data", allow_duplicate=True),
    Input("reg-start-date-picker", "value"),
    Input("reg-end-date-picker", "value"),
    State("reg-date-range-store", "data"),
    prevent_initial_call=True,
)
def reg_save_date_range(start, end, stored):
    stored = stored or {}
    changed = False
    if start and str(start)[:10] != stored.get("start"):
        stored = {**stored, "start": str(start)[:10]}
        changed = True
    if end and str(end)[:10] != stored.get("end"):
        stored = {**stored, "end": str(end)[:10]}
        changed = True
    if not changed:
        raise PreventUpdate
    return stored


# ---------------------------------------------------------------------------
# Clear series
# ---------------------------------------------------------------------------

@callback(
    Output("dashmat-raw-data-store", "data", allow_duplicate=True),
    Output("reg-series-select", "data", allow_duplicate=True),
    Output("reg-dependent-var-store", "data", allow_duplicate=True),
    Output("reg-series-order-store", "data", allow_duplicate=True),
    Input("reg-menu-clear-series", "n_clicks"),
    prevent_initial_call=True,
)
def reg_clear_series(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    return None, [], None, []


# ---------------------------------------------------------------------------
# Linear Constraints
# ---------------------------------------------------------------------------

@callback(
    Output("reg-linear-constraints-grid", "rowData"),
    Output("reg-linear-constraints-store", "data"),
    Input("reg-add-constraint-btn", "n_clicks"),
    Input("reg-clear-constraints-btn", "n_clicks"),
    Input("reg-linear-constraints-grid", "cellValueChanged"),
    State("reg-linear-constraints-grid", "rowData"),
    prevent_initial_call=True,
)
def reg_update_constraints(n_add, n_clear, cell_change, current_rows):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    trigger = ctx.triggered[0]["prop_id"].split(".")[0]
    current_rows = current_rows or []
    if trigger == "reg-clear-constraints-btn":
        return [], []
    elif trigger == "reg-add-constraint-btn":
        rows = current_rows + [{"Constraint": f"C{len(current_rows)+1}", "Min": None, "Max": None}]
        return rows, rows
    else:
        return current_rows, current_rows


# ---------------------------------------------------------------------------
# Sample data downloads
# ---------------------------------------------------------------------------

@callback(
    Output("reg-download-sample-daily", "data"),
    Input("reg-download-sample-daily-btn", "n_clicks"),
    prevent_initial_call=True,
)
def reg_download_sample_daily(n):
    if not n:
        raise PreventUpdate
    return dcc.send_file(str(get_sample_file_path("daily")))


@callback(
    Output("reg-download-sample-monthly", "data"),
    Input("reg-download-sample-monthly-btn", "n_clicks"),
    prevent_initial_call=True,
)
def reg_download_sample_monthly(n):
    if not n:
        raise PreventUpdate
    return dcc.send_file(str(get_sample_file_path("monthly")))


# ---------------------------------------------------------------------------
# Run Regression
# ---------------------------------------------------------------------------

@callback(
    Output("reg-results-store", "data", allow_duplicate=True),
    Output("reg-result-select", "data", allow_duplicate=True),
    Output("reg-result-select", "value", allow_duplicate=True),
    Output("reg-run-status-text", "children"),
    Input("reg-run-button", "n_clicks"),
    State("dashmat-raw-data-store", "data"),
    State("reg-periodicity-select", "value"),
    State("reg-series-select", "data"),
    State("reg-dependent-var-store", "data"),
    State("reg-benchmark-assignments-store", "data"),
    State("reg-long-short-store", "data"),
    State("reg-date-range-store", "data"),
    State("reg-vol-scaler-value-store", "data"),
    State("reg-vol-scaling-assignments-store", "data"),
    State("reg-lag-store", "data"),
    State("reg-min-beta-store", "data"),
    State("reg-max-beta-store", "data"),
    State("reg-enable-constraint-store", "data"),
    State("reg-model-store", "data"),
    State("reg-regression-name-store", "data"),
    State("reg-force-zero-intercept-store", "data"),
    State("reg-robust-se-store", "data"),
    State("reg-exp-wt-store", "data"),
    State("reg-halflife-store", "data"),
    State("reg-window-type-store", "data"),
    State("reg-window-size-store", "data"),
    State("reg-opt-step-store", "data"),
    State("reg-opt-step-unit-store", "data"),
    State("reg-fill-in-sample-store", "data"),
    State("reg-missing-data-store", "data"),
    State("reg-alpha-store", "data"),
    State("reg-l1-ratio-store", "data"),
    State("reg-arima-p-input", "value"),
    State("reg-arima-d-input", "value"),
    State("reg-arima-q-input", "value"),
    State("reg-garch-p-input", "value"),
    State("reg-garch-q-input", "value"),
    State("reg-linear-constraints-store", "data"),
    State("reg-results-store", "data"),
    prevent_initial_call=True,
)
def reg_run_regression(
    n_clicks, raw_data, periodicity, x_series, dep_var,
    bench_assign, ls_assign, date_range, vol_scaler, vol_scale_assign,
    lag_assign, min_beta_assign, max_beta_assign, enable_assign,
    model, reg_name, force_zero, robust_se, exp_wt, halflife,
    window_type, window_size, opt_step, opt_step_unit, fill_in_sample,
    missing_data, alpha, l1_ratio,
    arima_p, arima_d, arima_q, garch_p, garch_q,
    linear_constraints, current_results,
):
    if not n_clicks:
        raise PreventUpdate
    if not dep_var:
        return no_update, no_update, no_update, "Error: Select a dependent variable (Y)."
    if not x_series:
        return no_update, no_update, no_update, "Error: Select at least one independent variable (X)."
    if not raw_data:
        return no_update, no_update, no_update, "Error: No data loaded."

    all_series = list(dict.fromkeys([dep_var] + [s for s in (x_series or []) if s != dep_var]))
    try:
        df = _reg_get_working_returns(
            raw_data, periodicity, all_series,
            bench_assign, ls_assign, date_range, vol_scaler, vol_scale_assign,
        )
    except Exception as exc:
        return no_update, no_update, no_update, f"Error loading data: {exc}"

    if df.empty or dep_var not in df.columns:
        return no_update, no_update, no_update, "Error: No data for selected series/date range."

    y = df[dep_var]
    x_cols = [c for c in (x_series or []) if c in df.columns and c != dep_var]
    if not x_cols:
        return no_update, no_update, no_update, "Error: No X series data available."

    X = df[x_cols]

    # Build per-variable beta constraints
    per_var_enable = enable_assign or {}
    per_var_min = {c: float((min_beta_assign or {}).get(c, -999.0) or -999.0) for c in x_cols}
    per_var_max = {c: float((max_beta_assign or {}).get(c, 999.0) or 999.0) for c in x_cols}

    config = {
        "model": model or "ols",
        "force_zero_intercept": bool(force_zero),
        "robust_se": bool(robust_se),
        "exp_wt": bool(exp_wt),
        "halflife": float(halflife or 63),
        "window_type": window_type or "full",
        "window_size": int(window_size or 36),
        "opt_step": int(opt_step or 1),
        "opt_step_unit": opt_step_unit or "months",
        "fill_in_sample": fill_in_sample == "on",
        "missing_data": missing_data or "fill_na",
        "alpha": float(alpha or 1.0),
        "l1_ratio": float(l1_ratio or 0.5),
        "min_beta": min(per_var_min.values()) if per_var_min else -999.0,
        "max_beta": max(per_var_max.values()) if per_var_max else 999.0,
        "min_beta_by_var": per_var_min,
        "max_beta_by_var": per_var_max,
        "enable_constraint": per_var_enable,
        "lag_config": lag_assign or {},
        "arima_order": (int(arima_p or 0), int(arima_d or 0), int(arima_q or 0)),
        "garch_order": (int(garch_p or 0), int(garch_q or 0)),
        "linear_constraints": linear_constraints or None,
    }

    try:
        window_results, predicted, residuals, arima_garch_summary = run_regression(y, X, config)
    except Exception as exc:
        return no_update, no_update, no_update, f"Regression error: {exc}"

    if not window_results:
        return no_update, no_update, no_update, "No results — check data length and window settings."

    base_name = str(reg_name or "Regression").strip() or "Regression"
    current_results = current_results or {}
    name = base_name
    counter = 1
    while name in current_results:
        counter += 1
        name = f"{base_name}_{counter}"

    def _clean_val(v):
        if isinstance(v, float) and not np.isfinite(v):
            return None
        return v

    def _clean_dict(d):
        if d is None:
            return None
        return {k: (_clean_dict(v) if isinstance(v, dict) else _clean_val(v)) for k, v in d.items()}

    def _serialize_wr(wr: RegressionWindowResult) -> dict:
        return {
            "est_start": str(wr.est_start)[:10],
            "est_end": str(wr.est_end)[:10],
            "apply_start": str(wr.apply_start)[:10],
            "apply_end": str(wr.apply_end)[:10],
            "coefficients": _clean_dict(wr.coefficients),
            "p_values": _clean_dict(wr.p_values),
            "r_squared": _clean_val(wr.r_squared),
            "adj_r_squared": _clean_val(wr.adj_r_squared),
            "anova_table": _clean_dict(wr.anova_table),
            "diagnostics": _clean_dict(wr.diagnostics),
            "residual_std": _clean_val(wr.residual_std),
            "oos_metrics": _clean_dict(wr.oos_metrics),
            "n_obs": wr.n_obs,
        }

    result_entry = {
        "window_results": [_serialize_wr(wr) for wr in window_results],
        "predicted_json": df_to_json(predicted.to_frame("predicted")),
        "residuals_json": df_to_json(residuals.to_frame("residuals")),
        "dependent_var": dep_var,
        "independent_vars": x_cols,
        "config": config,
        "periodicity": periodicity or "daily",
        "arima_garch_summary": _clean_dict(arima_garch_summary),
    }

    new_results = {**current_results, name: result_entry}
    result_options = [{"value": k, "label": k} for k in new_results]
    status = f"✓ {name}: {len(window_results)} window(s), {len(predicted)} predictions."
    return new_results, result_options, name, status


# ---------------------------------------------------------------------------
# Result selector sync
# ---------------------------------------------------------------------------

@callback(
    Output("reg-result-select", "data"),
    Output("reg-result-select", "value"),
    Output("reg-delete-result-btn", "disabled"),
    Input("reg-results-store", "data"),
    State("reg-result-select", "value"),
    prevent_initial_call=False,
)
def reg_sync_result_options(results, current_val):
    results = results or {}
    options = [{"value": k, "label": k} for k in results]
    if not options:
        return options, None, True
    val = current_val if (current_val and current_val in results) else list(results.keys())[-1]
    return options, val, False


@callback(
    Output("reg-results-store", "data", allow_duplicate=True),
    Input("reg-delete-result-btn", "n_clicks"),
    State("reg-result-select", "value"),
    State("reg-results-store", "data"),
    prevent_initial_call=True,
)
def reg_delete_result(n_clicks, selected, results):
    if not n_clicks or not selected or not results:
        raise PreventUpdate
    return {k: v for k, v in results.items() if k != selected}


# ---------------------------------------------------------------------------
# ANOVA Tab
# ---------------------------------------------------------------------------

@callback(
    Output("reg-anova-content", "children"),
    Input("reg-result-select", "value"),
    Input("reg-results-store", "data"),
    prevent_initial_call=False,
)
def reg_render_anova(selected, results):
    if not selected or not results or selected not in results:
        return dmc.Text("Run a regression to see results.", size="sm", c="dimmed", p="md")

    entry = results[selected]
    wrs = entry.get("window_results", [])
    if not wrs:
        return dmc.Text("No results.", size="sm", c="dimmed")

    wr = wrs[-1]
    coefs = wr.get("coefficients") or {}
    pvals = wr.get("p_values") or {}
    diag = wr.get("diagnostics") or {}
    anova = wr.get("anova_table")
    dep_var = entry.get("dependent_var", "Y")
    config = entry.get("config", {})
    model = config.get("model", "ols")
    n_windows = len(wrs)

    ci_low = (diag.get("ci_low") or {}) if diag else {}
    ci_high = (diag.get("ci_high") or {}) if diag else {}
    std_errs = (diag.get("std_errors") or {}) if diag else {}
    t_stats = (diag.get("t_stats") or {}) if diag else {}

    coef_rows = [
        {
            "Variable": var,
            "Coefficient": _fmt(coefs.get(var)),
            "Std Error": _fmt(std_errs.get(var)),
            "t-stat": _fmt(t_stats.get(var)),
            "p-value": _fmt(pvals.get(var)),
            "CI Low (95%)": _fmt(ci_low.get(var)),
            "CI High (95%)": _fmt(ci_high.get(var)),
        }
        for var in coefs
    ]

    coef_grid = dag.AgGrid(
        className="ag-theme-alpine",
        columnDefs=[{"field": k, "minWidth": 110} for k in
                    ("Variable", "Coefficient", "Std Error", "t-stat", "p-value", "CI Low (95%)", "CI High (95%)")],
        rowData=coef_rows,
        defaultColDef={"resizable": True, "sortable": True},
        style={"height": f"{max(120, 42 + 42 * len(coef_rows))}px"},
        dashGridOptions={"suppressExcelExport": True, "suppressCsvExport": True},
    )

    r2 = wr.get("r_squared")
    adj_r2 = wr.get("adj_r_squared")
    n_obs = wr.get("n_obs", "—")
    resid_std = wr.get("residual_std")

    fit_row = dmc.Group(gap="xl", children=[
        dmc.Stack(gap=2, children=[dmc.Text("R²", size="xs", c="dimmed"), dmc.Text(_fmt(r2), size="sm", fw=600)]),
        dmc.Stack(gap=2, children=[dmc.Text("Adj. R²", size="xs", c="dimmed"), dmc.Text(_fmt(adj_r2), size="sm", fw=600)]),
        dmc.Stack(gap=2, children=[dmc.Text("Observations", size="xs", c="dimmed"), dmc.Text(str(n_obs), size="sm", fw=600)]),
        dmc.Stack(gap=2, children=[dmc.Text("Residual Std", size="xs", c="dimmed"), dmc.Text(_fmt(resid_std), size="sm", fw=600)]),
        dmc.Stack(gap=2, children=[dmc.Text("Model", size="xs", c="dimmed"), dmc.Text(model.replace("_", " ").title(), size="sm", fw=600)]),
        dmc.Stack(gap=2, children=[dmc.Text("Windows", size="xs", c="dimmed"), dmc.Text(str(n_windows), size="sm", fw=600)]),
        dmc.Stack(gap=2, children=[dmc.Text("Dependent", size="xs", c="dimmed"), dmc.Text(dep_var, size="sm", fw=600)]),
    ])

    sections = [
        dmc.Text("Coefficient Table", size="sm", fw=600, mb="xs"),
        coef_grid,
        dmc.Divider(mt="sm", mb="sm"),
        dmc.Text("Model Fit", size="sm", fw=600, mb="xs"),
        fit_row,
    ]

    if anova:
        anova_rows = [
            {"Source": "Model", "df": anova.get("df_model", "—"), "SS": _fmt(anova.get("ss_model")), "MS": _fmt(anova.get("ms_model")), "F": _fmt(anova.get("F_stat")), "p": _fmt(anova.get("F_pvalue"))},
            {"Source": "Residual", "df": anova.get("df_resid", "—"), "SS": _fmt(anova.get("ss_resid")), "MS": _fmt(anova.get("ms_resid")), "F": "—", "p": "—"},
            {"Source": "Total", "df": (anova.get("df_model", 0) or 0) + (anova.get("df_resid", 0) or 0), "SS": _fmt(anova.get("ss_total")), "MS": "—", "F": "—", "p": "—"},
        ]
        sections += [
            dmc.Divider(mt="sm", mb="sm"),
            dmc.Text("ANOVA Table", size="sm", fw=600, mb="xs"),
            dag.AgGrid(
                className="ag-theme-alpine",
                columnDefs=[{"field": k, "minWidth": 90} for k in ("Source", "df", "SS", "MS", "F", "p")],
                rowData=anova_rows,
                defaultColDef={"resizable": True, "sortable": False},
                style={"height": "168px"},
                dashGridOptions={"suppressExcelExport": True, "suppressCsvExport": True},
            ),
        ]

    if diag and not diag.get("note"):
        dw = diag.get("durbin_watson")
        jb_stat = diag.get("jarque_bera_stat")
        jb_p = diag.get("jarque_bera_pvalue")
        aic = diag.get("aic")
        bic = diag.get("bic")
        vif = diag.get("vif") or {}

        diag_row = dmc.Group(gap="xl", children=[
            dmc.Stack(gap=2, children=[dmc.Text("Durbin-Watson", size="xs", c="dimmed"), dmc.Text(_fmt(dw), size="sm", fw=600)]),
            dmc.Stack(gap=2, children=[dmc.Text("Jarque-Bera stat", size="xs", c="dimmed"), dmc.Text(_fmt(jb_stat), size="sm", fw=600)]),
            dmc.Stack(gap=2, children=[dmc.Text("JB p-value", size="xs", c="dimmed"), dmc.Text(_fmt(jb_p), size="sm", fw=600)]),
            dmc.Stack(gap=2, children=[dmc.Text("AIC", size="xs", c="dimmed"), dmc.Text(_fmt(aic), size="sm", fw=600)]),
            dmc.Stack(gap=2, children=[dmc.Text("BIC", size="xs", c="dimmed"), dmc.Text(_fmt(bic), size="sm", fw=600)]),
        ])
        sections += [dmc.Divider(mt="sm", mb="sm"), dmc.Text("Diagnostics", size="sm", fw=600, mb="xs"), diag_row]

        if vif:
            vif_rows = [{"Variable": k, "VIF": _fmt(v)} for k, v in vif.items()]
            sections += [
                dmc.Text("Variance Inflation Factor (VIF)", size="sm", fw=500, mt="sm", mb="xs"),
                dag.AgGrid(
                    className="ag-theme-alpine",
                    columnDefs=[{"field": "Variable", "minWidth": 150}, {"field": "VIF", "minWidth": 100}],
                    rowData=vif_rows,
                    defaultColDef={"resizable": True, "sortable": True},
                    style={"height": f"{max(120, 42 + 42 * len(vif_rows))}px"},
                    dashGridOptions={"suppressExcelExport": True, "suppressCsvExport": True},
                ),
            ]

    arima_garch = entry.get("arima_garch_summary") or (diag or {}).get("arima_garch")
    if arima_garch:
        ag_parts = []
        for key, label in [("arima", "ARIMA"), ("garch", "GARCH")]:
            item = arima_garch.get(key)
            if item and "error" not in item:
                ag_parts.append(dmc.Text(
                    f"{label}{item.get('order','?')}: AIC={_fmt(item.get('aic'))}, BIC={_fmt(item.get('bic'))}",
                    size="sm"
                ))
        if ag_parts:
            sections += [dmc.Divider(mt="sm", mb="sm"), dmc.Text("ARIMA/GARCH Residual Model", size="sm", fw=600, mb="xs")] + ag_parts

    return dmc.Stack(gap="xs", children=sections, p="sm")


# ---------------------------------------------------------------------------
# Rolling Summary Tab
# ---------------------------------------------------------------------------

@callback(
    Output("reg-rolling-content", "children"),
    Input("reg-result-select", "value"),
    Input("reg-results-store", "data"),
    Input("mantine-provider", "forceColorScheme"),
    prevent_initial_call=False,
)
def reg_render_rolling(selected, results, theme):
    if not selected or not results or selected not in results:
        return dmc.Text("Run a regression to see results.", size="sm", c="dimmed", p="md")
    entry = results[selected]
    wrs = entry.get("window_results", [])
    if not wrs:
        return dmc.Text("No results.", size="sm", c="dimmed")
    if len(wrs) == 1:
        return dmc.Alert("Rolling Summary requires rolling or expanding window.", color="blue", title="Info", p="md")

    rows = []
    for wr in wrs:
        row = {
            "Date": wr["apply_start"],
            "R²": wr.get("r_squared"),
            "Adj R²": wr.get("adj_r_squared"),
            "Residual Std": wr.get("residual_std"),
            "N Obs": wr.get("n_obs"),
        }
        for k, v in (wr.get("coefficients") or {}).items():
            row[f"β_{k}"] = v
        oos = wr.get("oos_metrics") or {}
        if oos:
            row.update({"OOS R²": oos.get("oos_r2"), "OOS RMSE": oos.get("oos_rmse"), "OOS MAE": oos.get("oos_mae")})
        rows.append(row)

    df_roll = pd.DataFrame(rows)
    df_roll["Date"] = pd.to_datetime(df_roll["Date"])
    df_roll = df_roll.sort_values("Date")

    fig = go.Figure()
    numeric_cols = [c for c in df_roll.columns if c != "Date"]
    for col in numeric_cols[:8]:
        if df_roll[col].notna().any():
            fig.add_trace(go.Scatter(x=df_roll["Date"], y=df_roll[col], mode="lines", name=col, line={"width": 1.5}))
    fig.update_layout(height=380, margin={"l": 50, "r": 20, "t": 30, "b": 50},
                      legend={"orientation": "h", "yanchor": "bottom", "y": 1.02})
    apply_chart_theme(fig, theme)

    df_display = df_roll.assign(Date=df_roll["Date"].dt.strftime("%Y-%m-%d"))
    return dmc.Stack(gap="sm", p="sm", children=[
        dcc.Graph(figure=fig, config={"displayModeBar": False}),
        dag.AgGrid(
            className="ag-theme-alpine",
            columnDefs=[{"field": c, "minWidth": 110} for c in df_display.columns],
            rowData=df_display.to_dict("records"),
            defaultColDef={"resizable": True, "sortable": True},
            style={"height": "280px"},
            dashGridOptions={"suppressExcelExport": True, "suppressCsvExport": True},
        ),
    ])


# ---------------------------------------------------------------------------
# Weights Tab
# ---------------------------------------------------------------------------

@callback(
    Output("reg-weights-content", "children"),
    Input("reg-result-select", "value"),
    Input("reg-results-store", "data"),
    Input("mantine-provider", "forceColorScheme"),
    prevent_initial_call=False,
)
def reg_render_weights(selected, results, theme):
    if not selected or not results or selected not in results:
        return dmc.Text("Run a regression to see results.", size="sm", c="dimmed", p="md")
    entry = results[selected]
    wrs = entry.get("window_results", [])
    config = entry.get("config", {})
    model = config.get("model", "ols")
    if not wrs:
        return dmc.Text("No results.", size="sm", c="dimmed")

    alert = None
    if model != "style_analysis":
        alert = dmc.Alert("Weights tab is most useful for Style Analysis (coefficients sum to 1).",
                          color="yellow", title="Note", mb="sm")

    dates = [pd.Timestamp(wr["apply_start"]) for wr in wrs]
    coef_keys = list((wrs[0].get("coefficients") or {}).keys())
    fig = go.Figure()
    if len(wrs) > 1 and coef_keys:
        for key in coef_keys:
            vals = [wr.get("coefficients", {}).get(key) for wr in wrs]
            fig.add_trace(go.Scatter(x=dates, y=vals, mode="lines", name=key,
                                     stackgroup="w" if model == "style_analysis" else None,
                                     line={"width": 1.5}))
    elif wrs:
        coefs = wrs[0].get("coefficients") or {}
        fig.add_trace(go.Bar(x=list(coefs.keys()), y=list(coefs.values()), name="Coefficients"))

    fig.update_layout(height=380, margin={"l": 50, "r": 20, "t": 30, "b": 50},
                      yaxis_title="Weight / Coefficient",
                      legend={"orientation": "h", "yanchor": "bottom", "y": 1.02})
    apply_chart_theme(fig, theme)

    children = []
    if alert:
        children.append(alert)
    children.append(dcc.Graph(figure=fig, config={"displayModeBar": False}))
    return dmc.Stack(gap="sm", p="sm", children=children)


# ---------------------------------------------------------------------------
# Returns Tab
# ---------------------------------------------------------------------------

@callback(
    Output("reg-returns-content", "children"),
    Input("reg-result-select", "value"),
    Input("reg-results-store", "data"),
    prevent_initial_call=False,
)
def reg_render_returns(selected, results):
    if not selected or not results or selected not in results:
        return dmc.Text("Run a regression to see results.", size="sm", c="dimmed", p="md")
    entry = results[selected]
    try:
        predicted_df = json_to_df(entry["predicted_json"])
        residuals_df = json_to_df(entry["residuals_json"])
    except Exception:
        return dmc.Text("Could not load series.", size="sm", c="dimmed")

    df = pd.concat([predicted_df, residuals_df], axis=1)
    df.index.name = "Date"
    df_reset = df.reset_index()
    df_reset["Date"] = df_reset["Date"].astype(str).str[:10]

    cols = [{"field": "Date", "pinned": "left", "minWidth": 110}]
    for c in df.columns:
        cols.append({"field": c, "minWidth": 110,
                     "valueFormatter": {"function": "params.value != null ? d3.format('.6f')(params.value) : ''"}})

    return dag.AgGrid(
        className="ag-theme-alpine",
        columnDefs=cols,
        rowData=df_reset.to_dict("records"),
        defaultColDef={"resizable": True, "sortable": True},
        style={"height": "500px"},
        dashGridOptions={"pagination": True, "paginationPageSize": 100,
                         "suppressExcelExport": True, "suppressCsvExport": True},
    )


# ---------------------------------------------------------------------------
# Growth Tab
# ---------------------------------------------------------------------------

@callback(
    Output("reg-growth-content", "children"),
    Input("reg-result-select", "value"),
    Input("reg-results-store", "data"),
    Input("mantine-provider", "forceColorScheme"),
    prevent_initial_call=False,
)
def reg_render_growth(selected, results, theme):
    if not selected or not results or selected not in results:
        return dmc.Text("Run a regression to see results.", size="sm", c="dimmed", p="md")
    entry = results[selected]
    try:
        predicted_df = json_to_df(entry["predicted_json"])
        residuals_df = json_to_df(entry["residuals_json"])
    except Exception:
        return dmc.Text("Could not load series.", size="sm", c="dimmed")

    fig = go.Figure()
    for col_df, label in [(predicted_df, "Predicted"), (residuals_df, "Residuals")]:
        if col_df.empty:
            continue
        s = col_df.iloc[:, 0]
        growth = (1 + s).cumprod()
        fig.add_trace(go.Scatter(x=growth.index, y=growth.values, mode="lines", name=label, line={"width": 1.5}))

    fig.update_layout(height=400, margin={"l": 50, "r": 20, "t": 30, "b": 50},
                      xaxis_title="Date", yaxis_title="Growth of $1",
                      legend={"orientation": "h", "yanchor": "bottom", "y": 1.02})
    apply_chart_theme(fig, theme)
    return dcc.Graph(figure=fig, config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# Statistics Tab
# ---------------------------------------------------------------------------

@callback(
    Output("reg-statistics-content", "children"),
    Input("reg-result-select", "value"),
    Input("reg-results-store", "data"),
    prevent_initial_call=False,
)
def reg_render_statistics(selected, results):
    if not selected or not results or selected not in results:
        return dmc.Text("Run a regression to see results.", size="sm", c="dimmed", p="md")
    entry = results[selected]
    periodicity = entry.get("periodicity", "daily")
    try:
        predicted_df = json_to_df(entry["predicted_json"])
    except Exception:
        return dmc.Text("Could not load predicted series.", size="sm", c="dimmed")

    if predicted_df.empty:
        return dmc.Text("No predicted series available.", size="sm", c="dimmed")

    series_names = tuple(predicted_df.columns)
    try:
        stats = calculate_statistics_cached(
            df_to_json(predicted_df), periodicity, series_names, "{}", "{}"
        )
    except Exception as exc:
        return dmc.Text(f"Statistics error: {exc}", size="sm", c="dimmed")

    if not stats:
        return dmc.Text("No statistics available.", size="sm", c="dimmed")

    rows = []
    for stat_name, series_vals in stats.items():
        row = {"Statistic": stat_name}
        for sname, val in series_vals.items():
            if isinstance(val, float) and not np.isfinite(val):
                row[sname] = "—"
            elif isinstance(val, float):
                row[sname] = f"{val:.4f}"
            else:
                row[sname] = str(val) if val is not None else "—"
        rows.append(row)

    col_names = ["Statistic"] + list(series_names)
    return dag.AgGrid(
        className="ag-theme-alpine",
        columnDefs=[{"field": c, "minWidth": 140} for c in col_names],
        rowData=rows,
        defaultColDef={"resizable": True, "sortable": True},
        style={"height": "600px"},
        dashGridOptions={"suppressExcelExport": True, "suppressCsvExport": True},
    )


# ---------------------------------------------------------------------------
# Scatter Tab
# ---------------------------------------------------------------------------

@callback(
    Output("reg-scatter-content", "children"),
    Input("reg-result-select", "value"),
    Input("reg-results-store", "data"),
    Input("mantine-provider", "forceColorScheme"),
    prevent_initial_call=False,
)
def reg_render_scatter(selected, results, theme):
    if not selected or not results or selected not in results:
        return dmc.Text("Run a regression to see results.", size="sm", c="dimmed", p="md")
    entry = results[selected]
    try:
        predicted_df = json_to_df(entry["predicted_json"])
        residuals_df = json_to_df(entry["residuals_json"])
    except Exception:
        return dmc.Text("Could not load data.", size="sm", c="dimmed")

    if predicted_df.empty:
        return dmc.Text("No predicted data.", size="sm", c="dimmed")

    predicted = predicted_df.iloc[:, 0]
    residuals = residuals_df.iloc[:, 0] if not residuals_df.empty else pd.Series(dtype=float)

    fig = go.Figure()
    if not residuals.empty:
        common_pred, common_resid = predicted.align(residuals, join="inner")
        fig.add_trace(go.Scatter(
            x=common_pred.values, y=common_resid.values,
            mode="markers", name="Residuals vs Fitted",
            marker={"size": 5, "opacity": 0.7},
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)
        title, xlabel, ylabel = "Residuals vs Fitted", "Fitted Values", "Residuals"
    else:
        title, xlabel, ylabel = "Scatter", "Fitted", "Values"

    fig.update_layout(height=400, title=title, margin={"l": 50, "r": 20, "t": 50, "b": 50},
                      xaxis_title=xlabel, yaxis_title=ylabel)
    apply_chart_theme(fig, theme)
    return dcc.Graph(figure=fig, config={"displayModeBar": False})
