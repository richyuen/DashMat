"""Shared landing page for DashMat imports and workspace handoff."""

from __future__ import annotations

from urllib.parse import parse_qs

import dash_mantine_components as dmc
from dash import Input, Output, State, callback, callback_context, clientside_callback, dcc, html, no_update, register_page
from dash.exceptions import PreventUpdate

from utils.add_series_flow import import_selected_disabled
from utils.dashmat_welcome_modal import (
    PagePrefixConfig,
    build_sheet_select_modal,
    build_welcome_screen,
    js_trigger_upload_with_cancel,
)
from utils.page_paths import LANDING_PATH, module_to_label, module_to_workspace_path, normalize_landing_module
from utils.route_intent import (
    ACTION_CONFIGURE_AFTER_IMPORT,
    ACTION_OPEN_IMPORT_MODAL,
    FLOW_DB,
    FLOW_PORTFOLIO,
    FLOW_RAW,
    FLOW_UNDERLYING,
    build_route_intent,
)
from utils.sample_data import get_sample_file_path
from utils.ui_tooltips import apply_tooltips_to_layout
from utils.upload_flow import (
    import_selected_workbook_sheets,
    import_single_upload,
    merge_uploaded_with_existing,
)
from utils.returns import df_to_json
from utils.parsing import get_sheet_names


register_page(__name__, path=LANDING_PATH, name="DashMat", title="DashMat")


DM_CONFIG = PagePrefixConfig(
    prefix="dm",
    page_icon="tabler:layout-grid",
    page_title="Welcome to DashMat",
    page_subtitle="Load data, then open Analytics, Portfolio Optimization, or Regression.",
    series_modal_size="80vw",
    series_modal_max_width="1200px",
    series_modal_transition_ms=180,
)

_MODULE_OPTIONS = [
    {"label": "Analytics Tool", "value": "analyticstool"},
    {"label": "Portfolio Optimization", "value": "portopt"},
    {"label": "Regression", "value": "regression"},
]

_IMPORT_TRIGGER_MAP = {
    "dm-welcome-add-db-btn": {"flow": FLOW_DB},
    "dm-welcome-add-portfolios-peer-btn": {"flow": FLOW_PORTFOLIO, "mode": "peer"},
    "dm-welcome-add-portfolios-index-btn": {"flow": FLOW_PORTFOLIO, "mode": "index"},
    "dm-welcome-add-portfolios-other-btn": {"flow": FLOW_PORTFOLIO, "mode": "other"},
    "dm-welcome-add-portfolios-underlying-btn": {"flow": FLOW_UNDERLYING},
    "dm-welcome-add-raw-factor-btn": {"flow": FLOW_RAW, "mode": "factor"},
    "dm-welcome-add-raw-funds-btn": {"flow": FLOW_RAW, "mode": "funds"},
    "dm-welcome-add-raw-performance-btn": {"flow": FLOW_RAW, "mode": "performance"},
}


layout = dmc.Container(
    size="xl",
    py="xl",
    children=[
        dmc.Stack(
            gap="lg",
            children=[
                dmc.Paper(
                    withBorder=True,
                    radius="lg",
                    p="lg",
                    children=dmc.Stack(
                        gap="md",
                        children=[
                            dmc.Group(
                                justify="space-between",
                                align="end",
                                style={"gap": "16px", "flexWrap": "wrap"},
                                children=[
                                    dmc.Stack(
                                        gap=4,
                                        children=[
                                            dmc.Title("Welcome to DashMat", order=2),
                                            dmc.Text(
                                                "Choose the initial workspace, then load or append returns data.",
                                                c="dimmed",
                                            ),
                                        ],
                                    ),
                                    dmc.Stack(
                                        gap=6,
                                        style={"minWidth": "320px", "flex": "1"},
                                        children=[
                                            dmc.Text("Initial module", size="sm", fw=500),
                                            dmc.SegmentedControl(
                                                id="dm-module-select",
                                                data=_MODULE_OPTIONS,
                                                value="analyticstool",
                                                fullWidth=True,
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                            dmc.Group(
                                justify="space-between",
                                align="center",
                                style={"gap": "12px", "flexWrap": "wrap"},
                                children=[
                                    dmc.Text(
                                        id="dm-module-subtitle",
                                        c="dimmed",
                                        children="Imports below will open in Analytics Tool.",
                                    ),
                                    dmc.Anchor(
                                        id="dm-open-workspace-link",
                                        href=module_to_workspace_path("analyticstool"),
                                        children=dmc.Button(
                                            "Open Analytics Tool",
                                            id="dm-open-workspace-button",
                                            disabled=True,
                                        ),
                                    ),
                                ],
                            ),
                            dmc.Alert(
                                id="dm-alert-message",
                                title="Status",
                                color="red",
                                hide=True,
                            ),
                        ],
                    ),
                ),
                dmc.Box(
                    pos="relative",
                    children=[
                        dmc.LoadingOverlay(
                            id="dm-ui-blocker-overlay",
                            visible=False,
                            zIndex=20,
                            overlayProps={"radius": "sm", "blur": 2},
                            loaderProps={"variant": "bars"},
                        ),
                        build_welcome_screen(DM_CONFIG),
                    ],
                ),
            ],
        ),
        dcc.Store(id="dm-ui-blocker-store", data=False),
        dcc.Store(id="dm-pending-nav-target-store", data=None),
        dcc.Store(id="dm-nav-effect-dummy", data=None),
        dcc.Store(id="dm-query-sync-dummy", data=None),
        dcc.Store(id="dm-sheet-select-contents-store", data=None),
        dcc.Store(id="dm-sheet-select-filename-store", data=None),
        dcc.Store(id="dm-sheet-select-sheetnames-store", data=None),
        html.Div(
            dcc.Upload(
                id="dm-upload-data",
                children=html.Div(),
                multiple=False,
            ),
            style={"display": "none"},
        ),
        dcc.Download(id="dm-download-sample-daily"),
        dcc.Download(id="dm-download-sample-monthly"),
        build_sheet_select_modal("dm"),
    ],
)

layout = apply_tooltips_to_layout(layout, page_key="dashmat")


def _normalized_query_module(search: str | None) -> str:
    params = parse_qs((search or "").lstrip("?"))
    return normalize_landing_module((params.get("module") or [None])[0])


def dm_search_for_module_selection(module_value: str | None, current_search: str | None) -> str | None:
    next_module = normalize_landing_module(module_value)
    current_search_value = str(current_search or "")
    if next_module == "analyticstool":
        return "" if current_search_value else None
    next_search = f"?module={next_module}"
    if current_search_value == next_search:
        return None
    return next_search


clientside_callback(
    js_trigger_upload_with_cancel("dm"),
    Output("dm-ui-blocker-store", "data", allow_duplicate=True),
    Input("dm-welcome-add-series-btn", "n_clicks"),
    prevent_initial_call=True,
)


clientside_callback(
    """
    function(isLoading) {
        return !!isLoading;
    }
    """,
    Output("dm-ui-blocker-overlay", "visible"),
    Input("dm-ui-blocker-store", "data"),
)


clientside_callback(
    """
    function(targetHref) {
        if (!targetHref) {
            return window.dash_clientside.no_update;
        }
        window.location.assign(targetHref);
        return targetHref;
    }
    """,
    Output("dm-nav-effect-dummy", "data"),
    Input("dm-pending-nav-target-store", "data"),
    prevent_initial_call=True,
)


@callback(
    Output("dm-download-sample-daily", "data"),
    Input("dm-download-sample-daily-btn", "n_clicks"),
    prevent_initial_call=True,
)
def dm_download_sample_daily(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    return dcc.send_file(str(get_sample_file_path("daily")))


@callback(
    Output("dm-download-sample-monthly", "data"),
    Input("dm-download-sample-monthly-btn", "n_clicks"),
    prevent_initial_call=True,
)
def dm_download_sample_monthly(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    return dcc.send_file(str(get_sample_file_path("monthly")))


@callback(
    Output("dm-module-select", "value"),
    Input("_pages_location", "search"),
    prevent_initial_call=False,
)
def dm_sync_module_from_query(search):
    return _normalized_query_module(search)


clientside_callback(
    """
    function(moduleValue) {
        const currentSearch = window.location.search || '';
        const normalized = String(moduleValue || '').trim().toLowerCase();
        const nextModule = ['analyticstool', 'portopt', 'regression'].includes(normalized)
            ? normalized
            : 'analyticstool';
        const nextSearch = nextModule === 'analyticstool' ? '' : `?module=${nextModule}`;
        if (currentSearch === nextSearch) {
            return window.dash_clientside.no_update;
        }
        const nextUrl = `${window.location.pathname}${nextSearch}${window.location.hash || ''}`;
        window.history.replaceState(window.history.state, '', nextUrl);
        return nextSearch;
    }
    """,
    Output("dm-query-sync-dummy", "data"),
    Input("dm-module-select", "value"),
    prevent_initial_call=True,
)


@callback(
    Output("dm-module-subtitle", "children"),
    Output("dm-open-workspace-link", "href"),
    Output("dm-open-workspace-button", "children"),
    Output("dm-open-workspace-button", "disabled"),
    Input("dm-module-select", "value"),
    Input("dashmat-raw-data-store", "data"),
    prevent_initial_call=False,
)
def dm_update_workspace_cta(module_value, raw_data):
    module_name = normalize_landing_module(module_value)
    label = module_to_label(module_name)
    return (
        f"Imports below will open in {label}.",
        module_to_workspace_path(module_name),
        f"Open {label}",
        not bool(raw_data),
    )


@callback(
    Output("dashmat-route-intent-store", "data", allow_duplicate=True),
    Output("dm-pending-nav-target-store", "data", allow_duplicate=True),
    Input("dm-welcome-add-db-btn", "n_clicks"),
    Input("dm-welcome-add-portfolios-peer-btn", "n_clicks"),
    Input("dm-welcome-add-portfolios-index-btn", "n_clicks"),
    Input("dm-welcome-add-portfolios-other-btn", "n_clicks"),
    Input("dm-welcome-add-portfolios-underlying-btn", "n_clicks"),
    Input("dm-welcome-add-raw-factor-btn", "n_clicks"),
    Input("dm-welcome-add-raw-funds-btn", "n_clicks"),
    Input("dm-welcome-add-raw-performance-btn", "n_clicks"),
    State("dm-module-select", "value"),
    prevent_initial_call=True,
)
def dm_route_non_file_imports(
    add_db_clicks,
    peer_clicks,
    index_clicks,
    other_clicks,
    underlying_clicks,
    raw_factor_clicks,
    raw_funds_clicks,
    raw_perf_clicks,
    module_value,
):
    triggered_id = callback_context.triggered_id
    trigger_meta = _IMPORT_TRIGGER_MAP.get(triggered_id)
    triggered_clicks = {
        "dm-welcome-add-db-btn": add_db_clicks,
        "dm-welcome-add-portfolios-peer-btn": peer_clicks,
        "dm-welcome-add-portfolios-index-btn": index_clicks,
        "dm-welcome-add-portfolios-other-btn": other_clicks,
        "dm-welcome-add-portfolios-underlying-btn": underlying_clicks,
        "dm-welcome-add-raw-factor-btn": raw_factor_clicks,
        "dm-welcome-add-raw-funds-btn": raw_funds_clicks,
        "dm-welcome-add-raw-performance-btn": raw_perf_clicks,
    }
    if not trigger_meta or not triggered_clicks.get(triggered_id):
        raise PreventUpdate

    module_name = normalize_landing_module(module_value)
    intent = build_route_intent(
        module_name,
        ACTION_OPEN_IMPORT_MODAL,
        flow=trigger_meta["flow"],
        mode=trigger_meta.get("mode"),
    )
    return intent, module_to_workspace_path(module_name)


def _dm_upload_success_outputs(
    merged_df,
    combined_periodicity,
    target_module,
):
    module_name = normalize_landing_module(target_module)
    return (
        df_to_json(merged_df),
        combined_periodicity,
        build_route_intent(module_name, ACTION_CONFIGURE_AFTER_IMPORT),
        module_to_workspace_path(module_name),
        "",
        "blue",
        True,
        False,
        False,
        no_update,
        no_update,
        no_update,
        no_update,
        no_update,
        no_update,
    )


@callback(
    Output("dashmat-raw-data-store", "data", allow_duplicate=True),
    Output("dashmat-original-periodicity-store", "data", allow_duplicate=True),
    Output("dashmat-route-intent-store", "data", allow_duplicate=True),
    Output("dm-pending-nav-target-store", "data", allow_duplicate=True),
    Output("dm-alert-message", "children", allow_duplicate=True),
    Output("dm-alert-message", "color", allow_duplicate=True),
    Output("dm-alert-message", "hide", allow_duplicate=True),
    Output("dm-ui-blocker-store", "data", allow_duplicate=True),
    Output("dm-sheet-select-modal", "opened", allow_duplicate=True),
    Output("dm-sheet-select-dropdown", "data", allow_duplicate=True),
    Output("dm-sheet-select-dropdown", "value", allow_duplicate=True),
    Output("dm-sheet-select-contents-store", "data", allow_duplicate=True),
    Output("dm-sheet-select-filename-store", "data", allow_duplicate=True),
    Output("dm-sheet-select-sheetnames-store", "data", allow_duplicate=True),
    Output("dm-upload-data", "contents", allow_duplicate=True),
    Input("dm-upload-data", "contents"),
    State("dm-upload-data", "filename"),
    State("dashmat-raw-data-store", "data"),
    State("dashmat-original-periodicity-store", "data"),
    State("dm-module-select", "value"),
    prevent_initial_call=True,
)
def dm_handle_upload(contents, filename, existing_data, existing_periodicity, module_value):
    if not contents:
        raise PreventUpdate

    try:
        sheet_names = get_sheet_names(contents, filename)
        if len(sheet_names) > 1:
            dropdown_data = [{"value": name, "label": name} for name in sheet_names]
            return (
                no_update,
                no_update,
                no_update,
                no_update,
                "",
                "blue",
                True,
                True,
                True,
                dropdown_data,
                [sheet_names[0]],
                contents,
                filename,
                sheet_names,
                no_update,
            )

        new_df = import_single_upload(contents, filename)
        merge_result = merge_uploaded_with_existing(existing_data, existing_periodicity, new_df)
        return _dm_upload_success_outputs(
            merge_result.merged_df,
            merge_result.combined_periodicity,
            module_value,
        )
    except Exception as exc:
        return (
            no_update,
            no_update,
            no_update,
            no_update,
            f"Error loading file: {exc}",
            "red",
            False,
            False,
            False,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            None,
        )


@callback(
    Output("dashmat-raw-data-store", "data", allow_duplicate=True),
    Output("dashmat-original-periodicity-store", "data", allow_duplicate=True),
    Output("dashmat-route-intent-store", "data", allow_duplicate=True),
    Output("dm-pending-nav-target-store", "data", allow_duplicate=True),
    Output("dm-alert-message", "children", allow_duplicate=True),
    Output("dm-alert-message", "color", allow_duplicate=True),
    Output("dm-alert-message", "hide", allow_duplicate=True),
    Output("dm-ui-blocker-store", "data", allow_duplicate=True),
    Output("dm-sheet-select-modal", "opened", allow_duplicate=True),
    Output("dm-sheet-select-contents-store", "data", allow_duplicate=True),
    Output("dm-sheet-select-filename-store", "data", allow_duplicate=True),
    Output("dm-sheet-select-sheetnames-store", "data", allow_duplicate=True),
    Output("dm-upload-data", "contents", allow_duplicate=True),
    Input("dm-sheet-select-ok-button", "n_clicks"),
    Input("dm-sheet-select-import-all-button", "n_clicks"),
    State("dm-sheet-select-dropdown", "value"),
    State("dm-sheet-select-contents-store", "data"),
    State("dm-sheet-select-filename-store", "data"),
    State("dm-sheet-select-sheetnames-store", "data"),
    State("dashmat-raw-data-store", "data"),
    State("dashmat-original-periodicity-store", "data"),
    State("dm-module-select", "value"),
    prevent_initial_call=True,
)
def dm_on_sheet_select_ok(
    selected_clicks,
    import_all_clicks,
    selected_sheets,
    stashed_contents,
    stashed_filename,
    stashed_sheet_names,
    existing_data,
    existing_periodicity,
    module_value,
):
    _ = (selected_clicks, import_all_clicks)
    if not stashed_contents:
        raise PreventUpdate

    triggered_id = callback_context.triggered_id
    if triggered_id not in {"dm-sheet-select-ok-button", "dm-sheet-select-import-all-button"}:
        raise PreventUpdate

    workbook_sheets = stashed_sheet_names or get_sheet_names(stashed_contents, stashed_filename)
    target_sheets = workbook_sheets if triggered_id == "dm-sheet-select-import-all-button" else (selected_sheets or [])
    if not target_sheets:
        return (
            no_update,
            no_update,
            no_update,
            no_update,
            "Select at least one sheet to import.",
            "red",
            False,
            False,
            True,
            stashed_contents,
            stashed_filename,
            workbook_sheets,
            no_update,
        )

    try:
        new_df, _imported_sheets = import_selected_workbook_sheets(
            stashed_contents,
            stashed_filename,
            target_sheets,
            workbook_sheets=workbook_sheets,
        )
        merge_result = merge_uploaded_with_existing(existing_data, existing_periodicity, new_df)
        return (
            *_dm_upload_success_outputs(
                merge_result.merged_df,
                merge_result.combined_periodicity,
                module_value,
            )[:8],
            False,
            None,
            None,
            None,
            None,
        )
    except Exception as exc:
        return (
            no_update,
            no_update,
            no_update,
            no_update,
            f"Error loading file: {exc}",
            "red",
            False,
            False,
            False,
            None,
            None,
            None,
            None,
        )


@callback(
    Output("dm-sheet-select-ok-button", "disabled"),
    Input("dm-sheet-select-dropdown", "value"),
)
def dm_toggle_sheet_select_import_selected_disabled(selected_sheets):
    return import_selected_disabled(selected_sheets)


clientside_callback(
    """
    function(n_clicks) {
        if (!n_clicks) {
            return window.dash_clientside.no_update;
        }
        return true;
    }
    """,
    Output("dm-ui-blocker-store", "data", allow_duplicate=True),
    Input("dm-sheet-select-ok-button", "n_clicks"),
    prevent_initial_call=True,
)


clientside_callback(
    """
    function(n_clicks) {
        if (!n_clicks) {
            return window.dash_clientside.no_update;
        }
        return true;
    }
    """,
    Output("dm-ui-blocker-store", "data", allow_duplicate=True),
    Input("dm-sheet-select-import-all-button", "n_clicks"),
    prevent_initial_call=True,
)


@callback(
    Output("dm-sheet-select-modal", "opened", allow_duplicate=True),
    Output("dm-sheet-select-contents-store", "data", allow_duplicate=True),
    Output("dm-sheet-select-filename-store", "data", allow_duplicate=True),
    Output("dm-sheet-select-sheetnames-store", "data", allow_duplicate=True),
    Output("dm-upload-data", "contents", allow_duplicate=True),
    Output("dm-ui-blocker-store", "data", allow_duplicate=True),
    Input("dm-sheet-select-cancel-button", "n_clicks"),
    prevent_initial_call=True,
)
def dm_on_sheet_select_cancel(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    return False, None, None, None, None, False


clientside_callback(
    """
    function(opened) {
        if (!opened) {
            var el = document.getElementById('dm-upload-data');
            if (el) {
                var input = el.querySelector('input[type="file"]');
                if (input) {
                    input.value = '';
                }
            }
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("dm-sheet-select-modal", "title", allow_duplicate=True),
    Input("dm-sheet-select-modal", "opened"),
    prevent_initial_call=True,
)
