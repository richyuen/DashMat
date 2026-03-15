"""Shared welcome-screen, modal builders, and modal callback helpers."""

from __future__ import annotations

from dataclasses import dataclass

import dash_ag_grid as dag
import dash_mantine_components as dmc
from dash import html, no_update
from dash.exceptions import PreventUpdate
from dash_iconify import DashIconify

from utils.add_series_flow import find_duplicate_series
from utils.constants import (
    INDEX_BENCHMARK_TYPE_OPTIONS,
    INDEX_PORTFOLIO_TYPE_OPTIONS,
    OTHER_BENCHMARK_TYPE_OPTIONS,
    OTHER_PORTFOLIO_TYPE_OPTIONS,
    PEER_BENCHMARK_TYPE_OPTIONS,
    PEER_PORTFOLIO_TYPE_OPTIONS,
)
from utils.core_categories import get_core_category_options_cached
from utils.portfolio_series import get_portfolio_options, has_portfolio_benchmark
from utils.raw_data_imports import (
    get_factor_options_cached,
    get_fund_options_cached,
    get_performance_options_cached,
)
from utils.underlying_category_imports import (
    UNDERLYING_CATEGORY_BASE_OPTIONS,
    UNDERLYING_CATEGORY_TYPE_OPTIONS,
)


def _sid(prefix: str, suffix: str) -> str:
    return f"{prefix}-{suffix}"


_ANCHORED_GROWING_MODAL_Y_OFFSET = "28vh"
_ANCHORED_GROWING_MODAL_STYLES = {
    "content": {"maxHeight": "84vh"},
    "body": {"maxHeight": "72vh", "overflowY": "auto"},
}
_UPWARD_MULTISELECT_COMBOBOX_PROPS = {
    "position": "top",
    "middlewares": {"flip": False, "shift": True},
}


@dataclass(frozen=True)
class PagePrefixConfig:
    prefix: str
    page_icon: str
    page_title: str
    page_subtitle: str
    series_modal_size: str
    series_modal_max_width: str
    series_modal_transition_ms: int
    welcome_switch_buttons: tuple[tuple[str, str, str], ...] = ()


def build_welcome_screen(cfg: PagePrefixConfig):
    switch_buttons = [
        dmc.Button(
            label,
            id=_sid(cfg.prefix, suffix),
            variant="light",
            size="sm",
            radius="md",
            leftSection=DashIconify(icon=icon, width=15),
        )
        for suffix, label, icon in (cfg.welcome_switch_buttons or ())
    ]

    header_children = [
        DashIconify(icon=cfg.page_icon, width=54, color="#8b95a1"),
        dmc.Text(cfg.page_title, size="xl", fw=600, mt=2),
        dmc.Text(cfg.page_subtitle, size="sm", c="dimmed"),
    ]
    if switch_buttons:
        header_children.append(
            dmc.Group(
                justify="center",
                gap="xs",
                style={"flexWrap": "wrap", "marginTop": "8px"},
                children=switch_buttons,
            )
        )

    return dmc.Stack(
        align="center",
        justify="center",
        gap="lg",
        style={"width": "100%", "maxWidth": "1160px", "margin": "0 auto", "padding": "4px 8px 12px"},
        children=[
            dmc.Stack(
                align="center",
                gap=2,
                children=header_children,
            ),
            html.Div(
                className="dashmat-welcome-sections-grid",
                children=[
                    dmc.Paper(
                        withBorder=True,
                        radius="md",
                        p="md",
                        className="dashmat-welcome-section-card",
                        children=dmc.Stack(
                            gap="sm",
                            children=[
                                dmc.Group(
                                    gap="xs",
                                    className="dashmat-welcome-section-header",
                                    children=[
                                        dmc.ThemeIcon(
                                            DashIconify(icon="tabler:database"),
                                            size="md",
                                            radius="xl",
                                            variant="light",
                                            color="indigo",
                                        ),
                                        dmc.Stack(
                                            gap=0,
                                            children=[
                                                dmc.Text(
                                                    "Load from Database: Index Data",
                                                    className="dashmat-welcome-section-title",
                                                ),
                                                dmc.Text(
                                                    "Import AA Tool market indices",
                                                    className="dashmat-welcome-section-subtitle",
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                                dmc.Stack(
                                    gap="xs",
                                    className="dashmat-welcome-section-actions",
                                    children=[
                                        dmc.Button(
                                            "AA Tool indices",
                                            leftSection=DashIconify(icon="tabler:database"),
                                            variant="outline",
                                            size="sm",
                                            fullWidth=True,
                                            id=_sid(cfg.prefix, "welcome-add-db-btn"),
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ),
                    dmc.Paper(
                        withBorder=True,
                        radius="md",
                        p="md",
                        className="dashmat-welcome-section-card",
                        children=dmc.Stack(
                            gap="sm",
                            children=[
                                dmc.Group(
                                    gap="xs",
                                    className="dashmat-welcome-section-header",
                                    children=[
                                        dmc.ThemeIcon(
                                            DashIconify(icon="tabler:briefcase"),
                                            size="md",
                                            radius="xl",
                                            variant="light",
                                            color="blue",
                                        ),
                                        dmc.Stack(
                                            gap=0,
                                            children=[
                                                dmc.Text(
                                                    "Load from Database: Portfolio Data",
                                                    className="dashmat-welcome-section-title",
                                                ),
                                                dmc.Text(
                                                    "Import relative portfolio return streams",
                                                    className="dashmat-welcome-section-subtitle",
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                                dmc.Stack(
                                    gap="xs",
                                    className="dashmat-welcome-section-actions",
                                    children=[
                                        dmc.Button(
                                            "Peer-relative",
                                            leftSection=DashIconify(icon="tabler:users"),
                                            variant="outline",
                                            size="sm",
                                            fullWidth=True,
                                            id=_sid(cfg.prefix, "welcome-add-portfolios-peer-btn"),
                                        ),
                                        dmc.Button(
                                            "Index-relative",
                                            leftSection=DashIconify(icon="tabler:chart-line"),
                                            variant="outline",
                                            size="sm",
                                            fullWidth=True,
                                            id=_sid(cfg.prefix, "welcome-add-portfolios-index-btn"),
                                        ),
                                        dmc.Button(
                                            "Alternatives",
                                            leftSection=DashIconify(icon="tabler:stack"),
                                            variant="outline",
                                            size="sm",
                                            fullWidth=True,
                                            id=_sid(cfg.prefix, "welcome-add-portfolios-other-btn"),
                                        ),
                                        dmc.Button(
                                            "Underlying categories",
                                            leftSection=DashIconify(icon="tabler:hierarchy-2"),
                                            variant="outline",
                                            size="sm",
                                            fullWidth=True,
                                            id=_sid(cfg.prefix, "welcome-add-portfolios-underlying-btn"),
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ),
                    dmc.Paper(
                        withBorder=True,
                        radius="md",
                        p="md",
                        className="dashmat-welcome-section-card",
                        children=dmc.Stack(
                            gap="sm",
                            children=[
                                dmc.Group(
                                    gap="xs",
                                    className="dashmat-welcome-section-header",
                                    children=[
                                        dmc.ThemeIcon(
                                            DashIconify(icon="tabler:database-import"),
                                            size="md",
                                            radius="xl",
                                            variant="light",
                                            color="grape",
                                        ),
                                        dmc.Stack(
                                            gap=0,
                                            children=[
                                                dmc.Text(
                                                    "Load from Database: Raw Data",
                                                    className="dashmat-welcome-section-title",
                                                ),
                                                dmc.Text(
                                                    "Import factor, funds, or performance return streams",
                                                    className="dashmat-welcome-section-subtitle",
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                                dmc.Stack(
                                    gap="xs",
                                    className="dashmat-welcome-section-actions",
                                    children=[
                                        dmc.Button(
                                            "Factor data",
                                            leftSection=DashIconify(icon="tabler:chart-dots"),
                                            variant="outline",
                                            size="sm",
                                            fullWidth=True,
                                            id=_sid(cfg.prefix, "welcome-add-raw-factor-btn"),
                                        ),
                                        dmc.Button(
                                            "Funds",
                                            leftSection=DashIconify(icon="tabler:building-bank"),
                                            variant="outline",
                                            size="sm",
                                            fullWidth=True,
                                            id=_sid(cfg.prefix, "welcome-add-raw-funds-btn"),
                                        ),
                                        dmc.Button(
                                            "Performance",
                                            leftSection=DashIconify(icon="tabler:activity-heartbeat"),
                                            variant="outline",
                                            size="sm",
                                            fullWidth=True,
                                            id=_sid(cfg.prefix, "welcome-add-raw-performance-btn"),
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ),
                    dmc.Paper(
                        withBorder=True,
                        radius="md",
                        p="md",
                        className="dashmat-welcome-section-card",
                        children=dmc.Stack(
                            gap="sm",
                            children=[
                                dmc.Group(
                                    gap="xs",
                                    className="dashmat-welcome-section-header",
                                    children=[
                                        dmc.ThemeIcon(
                                            DashIconify(icon="tabler:file-import"),
                                            size="md",
                                            radius="xl",
                                            variant="light",
                                            color="teal",
                                        ),
                                        dmc.Stack(
                                            gap=0,
                                            children=[
                                                dmc.Text(
                                                    "Load from File",
                                                    className="dashmat-welcome-section-title",
                                                ),
                                                dmc.Text(
                                                    "Upload returns or use sample files",
                                                    className="dashmat-welcome-section-subtitle",
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                                dmc.Stack(
                                    gap="xs",
                                    className="dashmat-welcome-section-actions",
                                    children=[
                                        dmc.Button(
                                            "Add series from file",
                                            leftSection=DashIconify(icon="tabler:upload"),
                                            variant="outline",
                                            size="sm",
                                            fullWidth=True,
                                            id=_sid(cfg.prefix, "welcome-add-series-btn"),
                                        ),
                                        dmc.Button(
                                            "Sample daily file",
                                            leftSection=DashIconify(icon="tabler:download"),
                                            id=_sid(cfg.prefix, "download-sample-daily-btn"),
                                            size="sm",
                                            variant="light",
                                            fullWidth=True,
                                        ),
                                        dmc.Button(
                                            "Sample monthly file",
                                            leftSection=DashIconify(icon="tabler:download"),
                                            id=_sid(cfg.prefix, "download-sample-monthly-btn"),
                                            size="sm",
                                            variant="light",
                                            fullWidth=True,
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ),
                    dmc.Paper(
                        withBorder=True,
                        radius="md",
                        p="md",
                        className="dashmat-welcome-section-card",
                        children=dmc.Stack(
                            gap="sm",
                            children=[
                                dmc.Group(
                                    gap="xs",
                                    className="dashmat-welcome-section-header",
                                    children=[
                                        dmc.ThemeIcon(
                                            DashIconify(icon="tabler:list-details"),
                                            size="md",
                                            radius="xl",
                                            variant="light",
                                            color="violet",
                                        ),
                                        dmc.Stack(
                                            gap=0,
                                            children=[
                                                dmc.Text(
                                                    "Load Account List",
                                                    className="dashmat-welcome-section-title",
                                                ),
                                                dmc.Text(
                                                    "Restore saved DB-backed series with latest data",
                                                    className="dashmat-welcome-section-subtitle",
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                                dmc.Stack(
                                    gap="xs",
                                    className="dashmat-welcome-section-actions",
                                    children=[
                                        dmc.Button(
                                            "Load account list",
                                            leftSection=DashIconify(icon="tabler:bookmark-plus"),
                                            variant="outline",
                                            size="sm",
                                            fullWidth=True,
                                            id=_sid(cfg.prefix, "welcome-load-account-list-btn"),
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ),
                ],
            ),
        ],
    )


def build_series_selection_modal(cfg: PagePrefixConfig):
    return dmc.Modal(
        id=_sid(cfg.prefix, "series-selection-modal"),
        title=dmc.Group(
            gap="xs",
            children=[
                dmc.ThemeIcon(DashIconify(icon="tabler:list-check"), color="blue", variant="light", size="sm"),
                dmc.Text("Select Series", fw=600, size="sm"),
            ],
        ),
        size=cfg.series_modal_size,
        styles={"content": {"maxWidth": cfg.series_modal_max_width}},
        centered=True,
        closeOnEscape=False,
        radius="lg",
        className="series-modal-dark dashmat-modal",
        overlayProps={"blur": 2, "opacity": 0.45},
        transitionProps={"transition": "fade", "duration": cfg.series_modal_transition_ms},
        children=[
            dmc.Alert(
                id=_sid(cfg.prefix, "alert-message"),
                title="Info",
                color="blue",
                hide=True,
                mb="md",
                withCloseButton=True,
            ),
            html.Div(
                id=_sid(cfg.prefix, "series-selection-container"),
                children=[dmc.Text("Upload data to select series", size="sm", c="dimmed")],
                style={"maxHeight": "50vh"},
            ),
            dmc.Group(
                mt="md",
                justify="flex-end",
                children=[
                    dmc.Button("Cancel", id=_sid(cfg.prefix, "modal-cancel-button"), variant="outline", color="red"),
                    dmc.Button("OK", id=_sid(cfg.prefix, "modal-ok-button"), color="blue"),
                ],
            ),
        ],
    )


def build_db_add_modal(prefix: str):
    return dmc.Modal(
        id=_sid(prefix, "db-add-modal"),
        title=dmc.Group(
            gap="xs",
            children=[
                dmc.ThemeIcon(DashIconify(icon="tabler:database"), color="indigo", variant="light", size="sm"),
                dmc.Text("AA Tool indices", fw=600, size="sm"),
            ],
        ),
        size="820px",
        centered=False,
        yOffset=_ANCHORED_GROWING_MODAL_Y_OFFSET,
        closeOnClickOutside=True,
        withCloseButton=True,
        radius="lg",
        className="dashmat-modal",
        overlayProps={"blur": 2, "opacity": 0.45},
        transitionProps={"transition": "fade", "duration": 180},
        styles=_ANCHORED_GROWING_MODAL_STYLES,
        children=[
            dmc.Alert(
                id=_sid(prefix, "db-add-error-alert"),
                title="Cannot add series",
                color="red",
                hide=True,
                mb="sm",
            ),
            dmc.MultiSelect(
                id=_sid(prefix, "db-add-series-select"),
                label="Select Series",
                data=[],
                value=[],
                searchable=True,
                clearSearchOnChange=False,
                placeholder="Select one or more series",
                nothingFoundMessage="No categories found",
                maxDropdownHeight=380,
                comboboxProps=_UPWARD_MULTISELECT_COMBOBOX_PROPS,
                w="100%",
            ),
            dmc.Group(
                mt="md",
                justify="flex-end",
                children=[
                    dmc.Button("Cancel", id=_sid(prefix, "db-add-cancel-button"), variant="outline", color="red"),
                    dmc.Button("OK", id=_sid(prefix, "db-add-ok-button"), color="blue", disabled=True),
                ],
            ),
        ],
    )


def build_portfolio_add_modal(prefix: str, ag_grid_license_key: str):
    return dmc.Modal(
        id=_sid(prefix, "portfolio-add-modal"),
        title=dmc.Group(
            gap="xs",
            children=[
                dmc.ThemeIcon(DashIconify(icon="tabler:briefcase"), color="indigo", variant="light", size="sm"),
                dmc.Text("Add portfolios", fw=600, size="sm"),
            ],
        ),
        size="860px",
        centered=True,
        closeOnClickOutside=True,
        withCloseButton=True,
        radius="lg",
        className="dashmat-modal",
        overlayProps={"blur": 2, "opacity": 0.45},
        transitionProps={"transition": "fade", "duration": 180},
        children=[
            dmc.Alert(
                id=_sid(prefix, "portfolio-add-error-alert"),
                title="Cannot stage import",
                color="red",
                hide=True,
                mb="sm",
            ),
            dmc.Stack(
                gap="sm",
                children=[
                    dmc.Select(
                        id=_sid(prefix, "portfolio-add-series-select"),
                        label="Series",
                        data=[],
                        value=None,
                        searchable=True,
                        clearable=True,
                        nothingFoundMessage="No portfolios found",
                        maxDropdownHeight=480,
                    ),
                    dmc.Group(
                        gap="sm",
                        children=[
                            dmc.Select(
                                id=_sid(prefix, "portfolio-add-type-select"),
                                label="Type",
                                data=[],
                                value=None,
                                clearable=False,
                                searchable=True,
                                maxDropdownHeight=480,
                                w=220,
                            ),
                            dmc.Checkbox(
                                id=_sid(prefix, "portfolio-add-include-benchmark"),
                                label="Include Benchmark",
                                checked=False,
                                disabled=True,
                                mt=24,
                            ),
                            dmc.Select(
                                id=_sid(prefix, "portfolio-add-benchmark-type-select"),
                                label="Benchmark Type",
                                data=[],
                                value=None,
                                clearable=False,
                                searchable=True,
                                maxDropdownHeight=480,
                                w=220,
                                disabled=True,
                            ),
                        ],
                    ),
                    dmc.Group(
                        gap="xs",
                        children=[
                            dmc.Button(
                                "Add Series",
                                id=_sid(prefix, "portfolio-add-row-btn"),
                                variant="outline",
                                size="xs",
                                leftSection=DashIconify(icon="tabler:plus"),
                            ),
                            dmc.Button(
                                "Delete One",
                                id=_sid(prefix, "portfolio-delete-row-btn"),
                                variant="outline",
                                size="xs",
                                color="red",
                                leftSection=DashIconify(icon="tabler:row-remove"),
                            ),
                            dmc.Button(
                                "Clear All",
                                id=_sid(prefix, "portfolio-clear-rows-btn"),
                                variant="outline",
                                size="xs",
                                color="red",
                                leftSection=DashIconify(icon="tabler:trash"),
                            ),
                        ],
                    ),
                    dag.AgGrid(
                        id=_sid(prefix, "portfolio-add-grid"),
                        className="ag-theme-alpine",
                        enableEnterpriseModules=True,
                        licenseKey=ag_grid_license_key,
                        columnDefs=[
                            {"field": "Portfolio", "headerName": "Portfolio", "width": 220, "headerClass": "dashmat-center-header"},
                            {"field": "Type", "headerName": "Type", "width": 160, "headerClass": "dashmat-center-header"},
                            {"field": "Include Benchmark", "headerName": "Include Benchmark", "width": 180, "headerClass": "dashmat-center-header"},
                            {"field": "Benchmark Type", "headerName": "Benchmark Type", "width": 180, "headerClass": "dashmat-center-header"},
                        ],
                        rowData=[],
                        defaultColDef={
                            "resizable": True,
                            "sortable": False,
                            "suppressHeaderMenuButton": True,
                            "cellStyle": {"textAlign": "center"},
                            "headerClass": "dashmat-center-header",
                        },
                        style={"height": "230px"},
                        dashGridOptions={
                            "rowSelection": "single",
                            "suppressRowClickSelection": False,
                            "animateRows": True,
                            "suppressExcelExport": True,
                            "suppressCsvExport": True,
                        },
                    ),
                    dmc.Group(
                        mt="sm",
                        justify="flex-end",
                        children=[
                            dmc.Button("Cancel", id=_sid(prefix, "portfolio-add-cancel-button"), variant="outline", color="red"),
                            dmc.Button("OK", id=_sid(prefix, "portfolio-add-ok-button"), color="blue", disabled=True),
                        ],
                    ),
                ],
            ),
        ],
    )


def build_underlying_add_modal(prefix: str, ag_grid_license_key: str):
    return dmc.Modal(
        id=_sid(prefix, "underlying-add-modal"),
        title=dmc.Group(
            gap="xs",
            children=[
                dmc.ThemeIcon(DashIconify(icon="tabler:hierarchy-2"), color="indigo", variant="light", size="sm"),
                dmc.Text("Add underlying categories", fw=600, size="sm"),
            ],
        ),
        size="900px",
        centered=False,
        yOffset=_ANCHORED_GROWING_MODAL_Y_OFFSET,
        closeOnClickOutside=True,
        withCloseButton=True,
        radius="lg",
        className="dashmat-modal",
        overlayProps={"blur": 2, "opacity": 0.45},
        transitionProps={"transition": "fade", "duration": 180},
        styles=_ANCHORED_GROWING_MODAL_STYLES,
        children=[
            dmc.Alert(
                id=_sid(prefix, "underlying-add-error-alert"),
                title="Cannot stage import",
                color="red",
                hide=True,
                mb="sm",
            ),
            dmc.Stack(
                gap="sm",
                children=[
                    dmc.Group(
                        gap="sm",
                        align="flex-start",
                        children=[
                            dmc.Select(
                                id=_sid(prefix, "underlying-add-base-select"),
                                label="Base",
                                data=list(UNDERLYING_CATEGORY_BASE_OPTIONS),
                                value=None,
                                clearable=False,
                                searchable=False,
                                maxDropdownHeight=240,
                                w=180,
                                placeholder="Select base",
                            ),
                            dmc.MultiSelect(
                                id=_sid(prefix, "underlying-add-type-multiselect"),
                                label="Type",
                                data=list(UNDERLYING_CATEGORY_TYPE_OPTIONS),
                                value=[],
                                searchable=False,
                                clearable=True,
                                maxDropdownHeight=240,
                                comboboxProps=_UPWARD_MULTISELECT_COMBOBOX_PROPS,
                                w=280,
                                placeholder="Select one or more types",
                            ),
                        ],
                    ),
                    dmc.MultiSelect(
                        id=_sid(prefix, "underlying-add-desc-multiselect"),
                        label="Desc",
                        data=[],
                        value=[],
                        searchable=True,
                        clearable=True,
                        disabled=True,
                        nothingFoundMessage="No underlying categories found",
                        maxDropdownHeight=480,
                        comboboxProps=_UPWARD_MULTISELECT_COMBOBOX_PROPS,
                        w="100%",
                        placeholder="Select one or more underlying categories",
                    ),
                    dmc.Group(
                        gap="xs",
                        children=[
                            dmc.Button(
                                "Add Series",
                                id=_sid(prefix, "underlying-add-row-btn"),
                                variant="outline",
                                size="xs",
                                leftSection=DashIconify(icon="tabler:plus"),
                            ),
                            dmc.Button(
                                "Delete One",
                                id=_sid(prefix, "underlying-delete-row-btn"),
                                variant="outline",
                                size="xs",
                                color="red",
                                leftSection=DashIconify(icon="tabler:row-remove"),
                            ),
                            dmc.Button(
                                "Clear All",
                                id=_sid(prefix, "underlying-clear-rows-btn"),
                                variant="outline",
                                size="xs",
                                color="red",
                                leftSection=DashIconify(icon="tabler:trash"),
                            ),
                        ],
                    ),
                    dag.AgGrid(
                        id=_sid(prefix, "underlying-add-grid"),
                        className="ag-theme-alpine",
                        enableEnterpriseModules=True,
                        licenseKey=ag_grid_license_key,
                        columnDefs=[
                            {"field": "Series", "headerName": "Series", "minWidth": 280, "flex": 2, "headerClass": "dashmat-center-header"},
                            {"field": "Portfolio", "headerName": "Portfolio", "width": 180, "headerClass": "dashmat-center-header"},
                            {"field": "Desc", "headerName": "Desc", "minWidth": 220, "flex": 1, "headerClass": "dashmat-center-header"},
                        ],
                        rowData=[],
                        defaultColDef={
                            "resizable": True,
                            "sortable": False,
                            "suppressHeaderMenuButton": True,
                            "cellStyle": {"textAlign": "center"},
                            "headerClass": "dashmat-center-header",
                        },
                        style={"height": "230px"},
                        dashGridOptions={
                            "rowSelection": "single",
                            "suppressRowClickSelection": False,
                            "animateRows": True,
                            "suppressExcelExport": True,
                            "suppressCsvExport": True,
                        },
                    ),
                    dmc.Group(
                        mt="sm",
                        justify="flex-end",
                        children=[
                            dmc.Button("Cancel", id=_sid(prefix, "underlying-add-cancel-button"), variant="outline", color="red"),
                            dmc.Button("OK", id=_sid(prefix, "underlying-add-ok-button"), color="blue", disabled=True),
                        ],
                    ),
                ],
            ),
        ],
    )


def build_sheet_select_modal(prefix: str):
    return dmc.Modal(
        id=_sid(prefix, "sheet-select-modal"),
        title=dmc.Group(
            gap="xs",
            children=[
                dmc.ThemeIcon(DashIconify(icon="tabler:table"), color="teal", variant="light", size="sm"),
                dmc.Text("Select Sheets", fw=600, size="sm"),
            ],
        ),
        size="lg",
        centered=False,
        yOffset=_ANCHORED_GROWING_MODAL_Y_OFFSET,
        closeOnClickOutside=False,
        radius="lg",
        className="dashmat-modal",
        overlayProps={"blur": 2, "opacity": 0.45},
        transitionProps={"transition": "fade", "duration": 180},
        styles=_ANCHORED_GROWING_MODAL_STYLES,
        children=[
            dmc.Text("This file contains multiple sheets. Select one or more sheets to import:", size="sm", mb="md"),
            dmc.MultiSelect(
                id=_sid(prefix, "sheet-select-dropdown"),
                data=[],
                value=[],
                w="100%",
                size="sm",
                placeholder="Select sheet(s)",
                maxDropdownHeight=480,
                comboboxProps=_UPWARD_MULTISELECT_COMBOBOX_PROPS,
            ),
            dmc.Group(
                mt="md",
                justify="flex-end",
                style={"flexWrap": "nowrap"},
                children=[
                    dmc.Button("Cancel", id=_sid(prefix, "sheet-select-cancel-button"), variant="outline", color="red"),
                    dmc.Button("Import All Sheets", id=_sid(prefix, "sheet-select-import-all-button"), variant="light"),
                    dmc.Button("Import Selected", id=_sid(prefix, "sheet-select-ok-button"), color="blue"),
                ],
            ),
        ],
    )


def build_raw_db_add_modal(prefix: str, ag_grid_license_key: str):
    return dmc.Modal(
        id=_sid(prefix, "raw-db-add-modal"),
        title=dmc.Group(
            gap="xs",
            children=[
                dmc.ThemeIcon(DashIconify(icon="tabler:database-import"), color="grape", variant="light", size="sm"),
                dmc.Text("Add raw database series", fw=600, size="sm"),
            ],
        ),
        size="980px",
        centered=True,
        closeOnClickOutside=True,
        withCloseButton=True,
        radius="lg",
        className="dashmat-modal",
        overlayProps={"blur": 2, "opacity": 0.45},
        transitionProps={"transition": "fade", "duration": 180},
        children=[
            dmc.Alert(
                id=_sid(prefix, "raw-db-add-error-alert"),
                title="Cannot stage import",
                color="red",
                hide=True,
                mb="sm",
            ),
            dmc.Stack(
                gap="sm",
                children=[
                    dmc.Select(
                        id=_sid(prefix, "raw-db-add-series-select"),
                        label="Series",
                        data=[],
                        value=None,
                        searchable=True,
                        clearable=True,
                        nothingFoundMessage="No series found",
                        maxDropdownHeight=480,
                    ),
                    dmc.Group(
                        gap="sm",
                        children=[
                            dmc.Select(
                                id=_sid(prefix, "raw-db-add-table-select"),
                                label="Table",
                                data=[
                                    {"value": "daily", "label": "Daily"},
                                    {"value": "monthly", "label": "Monthly"},
                                ],
                                value="daily",
                                clearable=False,
                                searchable=False,
                                maxDropdownHeight=480,
                                w=200,
                            ),
                            dmc.Select(
                                id=_sid(prefix, "raw-db-add-fee-select"),
                                label="Gross/Net",
                                data=[
                                    {"value": "gross", "label": "Gross"},
                                    {"value": "net", "label": "Net"},
                                ],
                                value="net",
                                clearable=False,
                                searchable=False,
                                maxDropdownHeight=480,
                                w=200,
                            ),
                            dmc.Checkbox(
                                id=_sid(prefix, "raw-db-add-include-benchmark"),
                                label="Include Benchmark",
                                checked=False,
                                mt=24,
                            ),
                        ],
                    ),
                    dmc.Group(
                        id=_sid(prefix, "raw-db-factor-controls"),
                        gap="sm",
                        children=[
                            dmc.Checkbox(
                                id=_sid(prefix, "raw-db-add-convert-returns"),
                                label="Convert to returns",
                                checked=False,
                                mt=5,
                            ),
                            dmc.NumberInput(
                                id=_sid(prefix, "raw-db-add-divide-by"),
                                label="Divide by",
                                value=100,
                                min=0,
                                step=1,
                                w=180,
                                disabled=True,
                            ),
                        ],
                    ),
                    dmc.Group(
                        gap="xs",
                        children=[
                            dmc.Button(
                                "Add Series",
                                id=_sid(prefix, "raw-db-add-row-btn"),
                                variant="outline",
                                size="xs",
                                leftSection=DashIconify(icon="tabler:plus"),
                            ),
                            dmc.Button(
                                "Delete One",
                                id=_sid(prefix, "raw-db-delete-row-btn"),
                                variant="outline",
                                size="xs",
                                color="red",
                                leftSection=DashIconify(icon="tabler:row-remove"),
                            ),
                            dmc.Button(
                                "Clear All",
                                id=_sid(prefix, "raw-db-clear-rows-btn"),
                                variant="outline",
                                size="xs",
                                color="red",
                                leftSection=DashIconify(icon="tabler:trash"),
                            ),
                        ],
                    ),
                    dag.AgGrid(
                        id=_sid(prefix, "raw-db-add-grid"),
                        className="ag-theme-alpine",
                        enableEnterpriseModules=True,
                        licenseKey=ag_grid_license_key,
                        columnDefs=[
                            {"field": "Series", "headerName": "Series", "minWidth": 260, "flex": 2, "headerClass": "dashmat-center-header"},
                            {"field": "Table", "headerName": "Table", "width": 110, "headerClass": "dashmat-center-header"},
                            {"field": "Fee", "headerName": "Fee", "width": 110, "headerClass": "dashmat-center-header"},
                            {"field": "Include Benchmark", "headerName": "Include Benchmark", "width": 170, "headerClass": "dashmat-center-header"},
                            {"field": "Convert to Returns", "headerName": "Convert", "width": 110, "headerClass": "dashmat-center-header"},
                            {"field": "Divide By", "headerName": "Divide By", "width": 110, "headerClass": "dashmat-center-header"},
                        ],
                        rowData=[],
                        defaultColDef={
                            "resizable": True,
                            "sortable": False,
                            "suppressHeaderMenuButton": True,
                            "cellStyle": {"textAlign": "center"},
                            "headerClass": "dashmat-center-header",
                        },
                        style={"height": "220px"},
                        dashGridOptions={
                            "rowSelection": "single",
                            "suppressRowClickSelection": False,
                            "animateRows": True,
                            "suppressExcelExport": True,
                            "suppressCsvExport": True,
                        },
                    ),
                    dmc.Stack(
                        gap=4,
                        children=[
                            dmc.Text("Preview (first 6 rows, using selected options)", fw=500, size="sm"),
                            dmc.ScrollArea(
                                h=140,
                                offsetScrollbars=True,
                                children=dmc.Code(
                                    id=_sid(prefix, "raw-db-preview-lines"),
                                    block=True,
                                    children="Select a series to preview option-adjusted results (first 6 rows).",
                                    style={
                                        "display": "block",
                                        "padding": "8px 10px",
                                        "fontFamily": "Consolas, monospace",
                                        "fontSize": "12px",
                                        "lineHeight": "1.4",
                                        "whiteSpace": "pre-wrap",
                                        "color": "var(--mantine-color-text)",
                                        "backgroundColor": "var(--mantine-color-body)",
                                        "border": "1px solid var(--mantine-color-default-border)",
                                    },
                                ),
                            ),
                        ],
                    ),
                    dmc.Group(
                        mt="sm",
                        justify="flex-end",
                        children=[
                            dmc.Button("Cancel", id=_sid(prefix, "raw-db-add-cancel-button"), variant="outline", color="red"),
                            dmc.Button("OK", id=_sid(prefix, "raw-db-add-ok-button"), color="blue", disabled=True),
                        ],
                    ),
                ],
            ),
        ],
    )


def _db_options(options: list[dict]) -> list[dict]:
    return [{"value": str(o["db_value"]), "label": str(o["label"])} for o in options if "db_value" in o and "label" in o]


def portfolio_type_options(mode: str) -> tuple[list[dict], list[dict]]:
    if mode == "index":
        return _db_options(INDEX_PORTFOLIO_TYPE_OPTIONS), _db_options(INDEX_BENCHMARK_TYPE_OPTIONS)
    if mode == "other":
        return _db_options(OTHER_PORTFOLIO_TYPE_OPTIONS), _db_options(OTHER_BENCHMARK_TYPE_OPTIONS)
    return _db_options(PEER_PORTFOLIO_TYPE_OPTIONS), _db_options(PEER_BENCHMARK_TYPE_OPTIONS)


def compute_open_db_add_modal(menu_clicks, welcome_clicks, db_engine):
    if not menu_clicks and not welcome_clicks:
        raise PreventUpdate
    options = get_core_category_options_cached(db_engine)
    return True, options, []


def compute_close_db_add_modal(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    return False, []


def compute_validate_db_add_selection(selected_benches, raw_data, opened):
    if not opened:
        raise PreventUpdate

    if not selected_benches:
        return no_update, True, True

    duplicates = find_duplicate_series(selected_benches, raw_data)
    if duplicates:
        return f"Cannot add duplicate series: {', '.join(duplicates)}", False, True
    return no_update, True, False


def compute_open_portfolio_add_modal(
    prefix: str,
    triggered_id,
    peer_clicks,
    index_clicks,
    other_clicks,
    welcome_peer_clicks,
    welcome_index_clicks,
    welcome_other_clicks,
    db_engine,
):
    if (
        not peer_clicks
        and not index_clicks
        and not other_clicks
        and not welcome_peer_clicks
        and not welcome_index_clicks
        and not welcome_other_clicks
    ):
        raise PreventUpdate

    if triggered_id in {_sid(prefix, "menu-add-portfolios-index"), _sid(prefix, "welcome-add-portfolios-index-btn")}:
        mode = "index"
    elif triggered_id in {_sid(prefix, "menu-add-portfolios-other"), _sid(prefix, "welcome-add-portfolios-other-btn")}:
        mode = "other"
    else:
        mode = "peer"

    mode_title_map = {
        "peer": "Add peer-relative portfolios",
        "index": "Add index-relative portfolios",
        "other": "Add alternative portfolios",
    }
    modal_title = mode_title_map.get(mode, "Add peer-relative portfolios")
    series_options = get_portfolio_options(db_engine, mode)
    type_options, bm_type_options = portfolio_type_options(mode)
    type_value = type_options[0]["value"] if type_options else None
    bm_value = bm_type_options[0]["value"] if bm_type_options else None

    return (
        True,
        modal_title,
        mode,
        series_options,
        None,
        type_options,
        type_value,
        bm_type_options,
        bm_value,
        False,
        True,
        [],
        [],
        True,
    )


def compute_sync_include_benchmark_enabled(mode, selected_portfolio, current_checked, db_engine):
    if mode not in {"peer", "index", "other"} or not selected_portfolio:
        return True, False
    has_bm = has_portfolio_benchmark(db_engine, mode, selected_portfolio)
    if not has_bm:
        return True, False
    return False, bool(current_checked)


def compute_close_portfolio_add_modal(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    return False, [], []


def compute_open_underlying_add_modal(menu_clicks, welcome_clicks):
    if not menu_clicks and not welcome_clicks:
        raise PreventUpdate

    return (
        True,
        "Add underlying categories",
        None,
        [],
        [],
        [],
        True,
        [],
        [],
        True,
    )


def compute_close_underlying_add_modal(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    return False, None, [], [], [], True, [], []


def compute_open_raw_db_add_modal(
    prefix: str,
    triggered_id,
    factor_clicks,
    funds_clicks,
    performance_clicks,
    welcome_factor_clicks,
    welcome_funds_clicks,
    welcome_performance_clicks,
    mrd_engine,
    perf_engine,
):
    if (
        not factor_clicks
        and not funds_clicks
        and not performance_clicks
        and not welcome_factor_clicks
        and not welcome_funds_clicks
        and not welcome_performance_clicks
    ):
        raise PreventUpdate

    factor_ids = {_sid(prefix, "menu-add-raw-factor"), _sid(prefix, "welcome-add-raw-factor-btn")}
    funds_ids = {_sid(prefix, "menu-add-raw-funds"), _sid(prefix, "welcome-add-raw-funds-btn")}
    performance_ids = {_sid(prefix, "menu-add-raw-performance"), _sid(prefix, "welcome-add-raw-performance-btn")}

    if triggered_id in factor_ids:
        mode = "factor"
        title = "Add raw factor data"
        options = get_factor_options_cached(mrd_engine)
    elif triggered_id in funds_ids:
        mode = "funds"
        title = "Add fund return series"
        options = get_fund_options_cached(mrd_engine)
    else:
        mode = "performance"
        title = "Add performance return series"
        options = get_performance_options_cached(perf_engine)

    return (
        True,
        title,
        mode,
        options,
        None,
        "daily",
        "net",
        False,
        False,
        100,
        True,
        [],
        [],
        "Select a series to preview option-adjusted results (first 6 rows).",
        True,
    )


def compute_close_raw_db_add_modal(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    return False, [], [], "Select a series to preview option-adjusted results (first 6 rows)."


def js_trigger_upload_with_cancel(prefix: str) -> str:
    return f"""
    function(n_clicks) {{
        if (n_clicks) {{
            setTimeout(function() {{
                var uploadDiv = document.getElementById('{prefix}-upload-data');
                if (uploadDiv) {{
                    var input = uploadDiv.querySelector('input[type="file"]');
                    if (input) {{
                        var onFocus = function() {{
                            window.removeEventListener('focus', onFocus);
                            setTimeout(function() {{
                                if (!input.files || input.files.length === 0) {{
                                    var store = document.getElementById('{prefix}-ui-blocker-store');
                                    if (store && store._dashprivate_setValue) {{
                                        store._dashprivate_setValue(false);
                                    }}
                                    window.dash_clientside.set_props('{prefix}-ui-blocker-store', {{data: false}});
                                }}
                            }}, 500);
                        }};
                        window.addEventListener('focus', onFocus);
                        input.click();
                    }}
                }}
            }}, 100);
            return true;
        }}
        return window.dash_clientside.no_update;
    }}
    """


def js_portfolio_benchmark_toggle() -> str:
    return """
    function(checked, options, currentValue) {
        if (!checked) {
            return [true, null];
        }
        var values = (options || [])
            .filter(function(opt) { return opt && opt.value !== null && opt.value !== undefined; })
            .map(function(opt) { return opt.value; });
        if (values.indexOf(currentValue) >= 0) {
            return [false, currentValue];
        }
        return [false, values.length ? values[0] : null];
    }
    """


def js_portfolio_add_row() -> str:
    return """
    function(nAdd, stagedRows, selectedPortfolio, selectedType, includeBenchmark, benchmarkType) {
        var noUpdate = window.dash_clientside.no_update;
        if (!nAdd) {
            return [noUpdate, noUpdate, noUpdate, noUpdate];
        }

        var rows = Array.isArray(stagedRows) ? stagedRows.slice() : [];
        var portfolio = String(selectedPortfolio || "").trim();
        var retType = String(selectedType || "").trim();
        var bmType = String(benchmarkType || "").trim();
        var includeBm = !!includeBenchmark;

        if (!portfolio) {
            return [rows, rows, "Select a portfolio series.", false];
        }
        if (!retType) {
            return [rows, rows, "Select a portfolio type.", false];
        }
        if (includeBm && !bmType) {
            return [rows, rows, "Select a benchmark type when benchmark is included.", false];
        }

        var exists = rows.some(function(r) {
            var key = "";
            var existingType = "";
            if (r && r.portfolio !== undefined && r.portfolio !== null) {
                key = String(r.portfolio).trim();
            } else if (r && r.Portfolio !== undefined && r.Portfolio !== null) {
                key = String(r.Portfolio).trim();
            }
            if (r && r.type !== undefined && r.type !== null) {
                existingType = String(r.type).trim();
            } else if (r && r.Type !== undefined && r.Type !== null) {
                existingType = String(r.Type).trim();
            }
            return key === portfolio && existingType === retType;
        });
        if (exists) {
            return [rows, rows, "Portfolio `" + portfolio + "` with type `" + retType + "` is already staged.", false];
        }

        rows.push({
            "Portfolio": portfolio,
            "Type": retType,
            "Include Benchmark": includeBm ? "Yes" : "No",
            "Benchmark Type": includeBm ? bmType : "",
            "portfolio": portfolio,
            "type": retType,
            "include_benchmark": includeBm,
            "benchmark_type": includeBm ? bmType : ""
        });
        return [rows, rows, noUpdate, true];
    }
    """


def js_portfolio_delete_row() -> str:
    return """
    function(nDelete, stagedRows, selectedRows) {
        var noUpdate = window.dash_clientside.no_update;
        if (!nDelete) {
            return [noUpdate, noUpdate, noUpdate, noUpdate];
        }
        var rows = Array.isArray(stagedRows) ? stagedRows.slice() : [];
        if (!selectedRows || !selectedRows.length) {
            return [rows, rows, "Select one staged row to delete.", false];
        }
        var selectedKey = String((selectedRows[0] || {}).Portfolio || "").trim();
        var kept = rows.filter(function(r) {
            return String((r && r.Portfolio) || "").trim() !== selectedKey;
        });
        return [kept, kept, noUpdate, true];
    }
    """


def js_underlying_delete_row() -> str:
    return """
    function(nDelete, stagedRows, selectedRows) {
        var noUpdate = window.dash_clientside.no_update;
        if (!nDelete) {
            return [noUpdate, noUpdate, noUpdate, noUpdate];
        }
        var rows = Array.isArray(stagedRows) ? stagedRows.slice() : [];
        if (!selectedRows || !selectedRows.length) {
            return [rows, rows, "Select one staged row to delete.", false];
        }
        var selectedSeries = String((selectedRows[0] || {}).Series || "").trim();
        var kept = rows.filter(function(r) {
            return String((r && r.Series) || "").trim() !== selectedSeries;
        });
        return [kept, kept, noUpdate, true];
    }
    """


def js_portfolio_clear_rows() -> str:
    return """
    function(nClear) {
        var noUpdate = window.dash_clientside.no_update;
        if (!nClear) {
            return [noUpdate, noUpdate, noUpdate, noUpdate];
        }
        return [[], [], noUpdate, true];
    }
    """


def js_portfolio_ok_disabled() -> str:
    return """
    function(rows, opened) {
        if (!opened) {
            return true;
        }
        return !(rows && rows.length);
    }
    """


def js_set_ui_blocker_true() -> str:
    return """
    function(n_clicks) {
        if (n_clicks) {
            return true;
        }
        return window.dash_clientside.no_update;
    }
    """


def js_release_ui_blocker_on_modal_state() -> str:
    return """
    function(opened, errorHidden) {
        if (opened === false || errorHidden === false) {
            return false;
        }
        return window.dash_clientside.no_update;
    }
    """
