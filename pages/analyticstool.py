"""Analytics tool page - Market Returns Time Series Dashboard."""

from dataclasses import dataclass
import hashlib
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
from utils.parsing import get_sheet_names
from utils.add_series_flow import import_selected_disabled
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
from utils.dashmat_welcome_modal import (
    PagePrefixConfig,
    build_db_add_modal,
    build_portfolio_add_modal,
    build_raw_db_add_modal,
    build_series_selection_modal,
    build_sheet_select_modal,
    build_welcome_screen as build_shared_welcome_screen,
    compute_close_db_add_modal,
    compute_close_portfolio_add_modal,
    compute_close_raw_db_add_modal,
    compute_open_db_add_modal,
    compute_open_portfolio_add_modal,
    compute_open_raw_db_add_modal,
    compute_sync_include_benchmark_enabled,
    compute_validate_db_add_selection,
    js_portfolio_add_row,
    js_portfolio_benchmark_toggle,
    js_portfolio_clear_rows,
    js_portfolio_delete_row,
    js_portfolio_ok_disabled,
    js_release_ui_blocker_on_modal_state,
    js_set_ui_blocker_true,
    js_trigger_upload_with_cancel,
)
from dbengine import (
    AG_GRID_LICENSE_KEY,
    engine as DB_ENGINE,
    engine_MRD as MRD_ENGINE,
    engine_PERFORMANCE as PERF_ENGINE,
)
from utils.core_categories import (
    clear_dropdown_caches,
    load_cma_returns_for_benches,
    load_cma_returns_for_benches_with_meta,
)
from utils.portfolio_series import load_portfolio_series
from utils.raw_data_imports import (
    build_preview_row_from_controls,
    factor_defaults_to_returns,
    get_factor_option_meta_cached,
    get_fund_option_meta_cached,
    get_performance_option_meta_cached,
    get_preview_lines_for_row,
    load_factor_series,
    load_fund_series,
    load_performance_series,
)

register_page(__name__, path="/analyticstool", name="Analytics Tool", title="Analytics Tool")

# Performance optimization constants

SAVED_SERIES_CONFIG = {
    RISK_FREE_SERIES: {},
    MARKET_BETA_SERIES: {"start_date": "1988-01-04"},
}

AT_WELCOME_MODAL_CONFIG = PagePrefixConfig(
    prefix="at",
    page_icon="tabler:chart-line",
    page_title="Welcome to the Analytics Tool",
    page_subtitle="Choose a source to load data and get started.",
    series_modal_size="80vw",
    series_modal_max_width="1250px",
    series_modal_transition_ms=180,
)


def _mapping_payload(value) -> str:
    return mapping_payload_for_cache(value)


def _date_range_payload(value) -> str:
    return date_range_payload_for_cache(value)


def _has_complete_date_range(value) -> bool:
    return (
        isinstance(value, dict)
        and bool(value.get("start"))
        and bool(value.get("end"))
    )


def _correlogram_request_key(
    raw_data,
    periodicity,
    selected_series,
    returns_type,
    benchmark_assignments,
    long_short_assignments,
    date_range,
    vol_scaler,
    vol_scaling_assignments,
    correlation_view,
    block_width,
):
    payload = "|".join(
        [
            hashlib.md5((raw_data or "").encode("utf-8")).hexdigest(),
            str(periodicity or "daily"),
            ",".join(selected_series or ()),
            str(returns_type or "total"),
            _mapping_payload(benchmark_assignments),
            _mapping_payload(long_short_assignments),
            _date_range_payload(date_range),
            str(vol_scaler or 0),
            _mapping_payload(vol_scaling_assignments),
            str(correlation_view or "correlogram"),
            str(block_width if block_width is not None else ""),
        ]
    )
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


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


def _import_selected_workbook_sheets(contents, filename, selected_sheets, workbook_sheets=None):
    return _shared_import_selected_workbook_sheets(
        contents,
        filename,
        selected_sheets,
        workbook_sheets=workbook_sheets,
    )


def build_welcome_screen():
    return build_shared_welcome_screen(AT_WELCOME_MODAL_CONFIG)

# Clientside callback to trigger upload from welcome button
clientside_callback(
    js_trigger_upload_with_cancel("at"),
    Output("at-ui-blocker-store", "data", allow_duplicate=True),
    Input("at-welcome-add-series-btn", "n_clicks"),
    prevent_initial_call=True,
)

clientside_callback(
    "function(n) { return true; }",
    Output("at-help-modal", "opened"),
    Input("at-menu-help-guide", "n_clicks"),
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
    Output("at-save-session-dummy", "data"),
    Input("at-menu-save-session", "n_clicks"),
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
    Output("at-load-session-dummy", "data"),
    Input("at-menu-load-session", "n_clicks"),
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
    Output("at-load-session-dummy", "data", allow_duplicate=True),    Input("at-load-session-upload", "contents"),
    prevent_initial_call=True,
)


@callback(
    Output("at-menu-save-session", "disabled"),
    Input("at-welcome-screen-container", "style"),
)
def at_toggle_save_session(welcome_style):
    if not welcome_style:
        return True
    return welcome_style.get("display") != "none"


@callback(
    Output("at-db-add-modal", "opened", allow_duplicate=True),
    Output("at-db-add-series-select", "data", allow_duplicate=True),
    Output("at-db-add-series-select", "value", allow_duplicate=True),
    Input("at-menu-add-from-db", "n_clicks"),
    Input("at-welcome-add-db-btn", "n_clicks"),
    prevent_initial_call=True,
)
def open_db_add_modal(menu_clicks, welcome_clicks):
    return compute_open_db_add_modal(menu_clicks, welcome_clicks, DB_ENGINE)


@callback(
    Output("at-db-add-modal", "opened", allow_duplicate=True),
    Output("at-db-add-series-select", "value", allow_duplicate=True),
    Input("at-db-add-cancel-button", "n_clicks"),
    prevent_initial_call=True,
)
def close_db_add_modal(n_clicks):
    return compute_close_db_add_modal(n_clicks)


@callback(
    Output("at-raw-db-add-modal", "opened", allow_duplicate=True),
    Output("at-raw-db-add-modal", "title", allow_duplicate=True),
    Output("at-raw-db-add-mode-store", "data", allow_duplicate=True),
    Output("at-raw-db-add-series-select", "data", allow_duplicate=True),
    Output("at-raw-db-add-series-select", "value", allow_duplicate=True),
    Output("at-raw-db-add-table-select", "value", allow_duplicate=True),
    Output("at-raw-db-add-fee-select", "value", allow_duplicate=True),
    Output("at-raw-db-add-include-benchmark", "checked", allow_duplicate=True),
    Output("at-raw-db-add-convert-returns", "checked", allow_duplicate=True),
    Output("at-raw-db-add-divide-by", "value", allow_duplicate=True),
    Output("at-raw-db-add-error-alert", "hide", allow_duplicate=True),
    Output("at-raw-db-add-rows-store", "data", allow_duplicate=True),
    Output("at-raw-db-add-grid", "rowData", allow_duplicate=True),
    Output("at-raw-db-preview-lines", "children", allow_duplicate=True),
    Output("at-raw-db-add-ok-button", "disabled", allow_duplicate=True),
    Input("at-menu-add-raw-factor", "n_clicks"),
    Input("at-menu-add-raw-funds", "n_clicks"),
    Input("at-menu-add-raw-performance", "n_clicks"),
    Input("at-welcome-add-raw-factor-btn", "n_clicks"),
    Input("at-welcome-add-raw-funds-btn", "n_clicks"),
    Input("at-welcome-add-raw-performance-btn", "n_clicks"),
    prevent_initial_call=True,
)
def at_open_raw_db_add_modal(
    factor_clicks,
    funds_clicks,
    performance_clicks,
    welcome_factor_clicks,
    welcome_funds_clicks,
    welcome_performance_clicks,
):
    return compute_open_raw_db_add_modal(
        prefix="at",
        triggered_id=callback_context.triggered_id,
        factor_clicks=factor_clicks,
        funds_clicks=funds_clicks,
        performance_clicks=performance_clicks,
        welcome_factor_clicks=welcome_factor_clicks,
        welcome_funds_clicks=welcome_funds_clicks,
        welcome_performance_clicks=welcome_performance_clicks,
        mrd_engine=MRD_ENGINE,
        perf_engine=PERF_ENGINE,
    )


@callback(
    Output("at-raw-db-add-modal", "opened", allow_duplicate=True),
    Output("at-raw-db-add-rows-store", "data", allow_duplicate=True),
    Output("at-raw-db-add-grid", "rowData", allow_duplicate=True),
    Output("at-raw-db-preview-lines", "children", allow_duplicate=True),
    Output("at-raw-db-add-error-alert", "hide", allow_duplicate=True),
    Output("at-raw-db-add-ok-button", "disabled", allow_duplicate=True),
    Output("at-raw-db-add-series-select", "value", allow_duplicate=True),
    Input("at-raw-db-add-cancel-button", "n_clicks"),
    prevent_initial_call=True,
)
def at_close_raw_db_add_modal(n_clicks):
    opened, rows, grid_rows, preview = compute_close_raw_db_add_modal(n_clicks)
    return opened, rows, grid_rows, preview, True, True, None


@callback(
    Output("at-raw-db-add-table-select", "disabled"),
    Output("at-raw-db-add-fee-select", "data"),
    Output("at-raw-db-add-fee-select", "value"),
    Output("at-raw-db-add-fee-select", "disabled"),
    Output("at-raw-db-add-include-benchmark", "disabled"),
    Output("at-raw-db-add-include-benchmark", "checked", allow_duplicate=True),
    Output("at-raw-db-factor-controls", "style"),
    Output("at-raw-db-add-convert-returns", "checked", allow_duplicate=True),
    Input("at-raw-db-add-mode-store", "data"),
    Input("at-raw-db-add-series-select", "value"),
    Input("at-raw-db-add-modal", "opened"),
    State("at-raw-db-add-fee-select", "value"),
    State("at-raw-db-add-include-benchmark", "checked"),
    State("at-raw-db-add-convert-returns", "checked"),
    prevent_initial_call=True,
)
def at_sync_raw_modal_controls(mode, series_key, opened, current_fee, current_include_benchmark, current_convert):
    if not opened:
        raise PreventUpdate

    triggered_id = callback_context.triggered_id
    preserve_series_selection_state = triggered_id == "at-raw-db-add-series-select"
    mode_key = str(mode or "").strip().lower()
    if mode_key == "factor":
        default_convert = False
        if series_key:
            meta = get_factor_option_meta_cached(MRD_ENGINE).get(str(series_key), {})
            default_convert = factor_defaults_to_returns(meta.get("factor_name"))
        # Factor series selection should always apply its default conversion rule.
        convert_value = default_convert
        fee_options = [
            {"value": "gross", "label": "Gross"},
            {"value": "net", "label": "Net"},
        ]
        fee_values = {str(opt["value"]) for opt in fee_options}
        fee_value = str(current_fee) if preserve_series_selection_state and str(current_fee) in fee_values else "net"
        return (
            True,
            fee_options,
            fee_value,
            True,
            True,
            False,
            {},
            convert_value,
        )

    if mode_key == "funds":
        fee_options = [
            {"value": "gross", "label": "Gross"},
            {"value": "net", "label": "Net"},
        ]
        fee_values = {str(opt["value"]) for opt in fee_options}
        fee_value = str(current_fee) if str(current_fee) in fee_values else "net"
        return (
            False,
            fee_options,
            fee_value,
            False,
            True,
            False,
            {"display": "none"},
            False,
        )

    fee_options = [
        {"value": "G", "label": "Gross"},
        {"value": "N", "label": "Net"},
    ]
    fee_values = {str(opt["value"]) for opt in fee_options}
    fee_value = str(current_fee) if str(current_fee) in fee_values else "N"
    include_value = bool(current_include_benchmark) if current_include_benchmark is not None else False
    return (
        False,
        fee_options,
        fee_value,
        False,
        False,
        include_value,
        {"display": "none"},
        False,
    )


@callback(
    Output("at-raw-db-add-divide-by", "disabled"),
    Input("at-raw-db-add-mode-store", "data"),
    Input("at-raw-db-add-convert-returns", "checked"),
    Input("at-raw-db-add-modal", "opened"),
    prevent_initial_call=True,
)
def at_toggle_raw_divide_by(mode, convert_to_returns, opened):
    if not opened:
        raise PreventUpdate
    return not (str(mode or "").strip().lower() == "factor" and not bool(convert_to_returns))


@callback(
    Output("at-raw-db-add-rows-store", "data", allow_duplicate=True),
    Output("at-raw-db-add-grid", "rowData", allow_duplicate=True),
    Output("at-raw-db-add-error-alert", "children", allow_duplicate=True),
    Output("at-raw-db-add-error-alert", "hide", allow_duplicate=True),
    Input("at-raw-db-add-row-btn", "n_clicks"),
    State("at-raw-db-add-rows-store", "data"),
    State("at-raw-db-add-mode-store", "data"),
    State("at-raw-db-add-series-select", "value"),
    State("at-raw-db-add-table-select", "value"),
    State("at-raw-db-add-fee-select", "value"),
    State("at-raw-db-add-include-benchmark", "checked"),
    State("at-raw-db-add-convert-returns", "checked"),
    State("at-raw-db-add-divide-by", "value"),
    prevent_initial_call=True,
)
def at_stage_raw_db_row(
    n_add,
    staged_rows,
    mode,
    series_key,
    table_choice,
    fee_choice,
    include_benchmark,
    convert_to_returns,
    divide_by,
):
    n_no = no_update
    if not n_add:
        raise PreventUpdate

    mode_key = str(mode or "").strip().lower()
    rows = [dict(r) for r in (staged_rows or []) if isinstance(r, dict)]
    key = str(series_key or "").strip()
    if mode_key not in {"factor", "funds", "performance"}:
        return rows, rows, "Select a raw import type first.", False
    if not key:
        return rows, rows, "Select a series to add.", False

    if mode_key == "factor":
        meta = get_factor_option_meta_cached(MRD_ENGINE).get(key)
        if not meta:
            return rows, rows, "Selected factor series is unavailable.", False
        import_name = str(meta.get("import_name", "")).strip()
        if any(str(r.get("import_name", "")).strip() == import_name for r in rows):
            return rows, rows, f"Series `{import_name}` is already staged.", False
        convert = bool(convert_to_returns)
        div_value = pd.to_numeric(pd.Series([divide_by]), errors="coerce").iloc[0]
        if not convert and (pd.isna(div_value) or float(div_value) == 0.0):
            return rows, rows, "Divide by must be a non-zero number when convert-to-returns is unchecked.", False
        row_id = f"factor:{key}"
        row = {
            "row_id": row_id,
            "mode": "factor",
            "series_key": key,
            "series_label": str(meta.get("label", import_name)),
            "import_name": import_name,
            "convert_to_returns": convert,
            "divide_by": float(div_value) if not convert else 100.0,
            "Series": str(meta.get("label", import_name)),
            "Table": "",
            "Fee": "",
            "Include Benchmark": "",
            "Convert to Returns": "Yes" if convert else "No",
            "Divide By": "" if convert else float(div_value),
        }
        rows.append(row)
        return rows, rows, n_no, True

    if mode_key == "funds":
        meta = get_fund_option_meta_cached(MRD_ENGINE).get(key)
        if not meta:
            return rows, rows, "Selected fund series is unavailable.", False
        base_name = str(meta.get("import_name", "")).strip()
        table_key = "monthly" if str(table_choice or "").lower() == "monthly" else "daily"
        fee_key = "net" if str(fee_choice or "").lower().startswith("n") else "gross"
        if table_key == "daily" and fee_key == "net":
            import_name = base_name
        elif table_key == "monthly" and fee_key == "net":
            import_name = f"{base_name}_M"
        elif table_key == "daily" and fee_key == "gross":
            import_name = f"{base_name}_G"
        else:
            import_name = f"{base_name}_GM"
        if any(str(r.get("import_name", "")).strip() == import_name for r in rows):
            return rows, rows, f"Series `{import_name}` is already staged.", False
        row_id = f"funds:{key}:{table_key}:{fee_key}"
        row = {
            "row_id": row_id,
            "mode": "funds",
            "series_key": key,
            "series_label": str(meta.get("label", base_name)),
            "import_name": import_name,
            "table_choice": table_key,
            "fee_choice": fee_key,
            "Series": import_name,
            "Table": "Monthly" if table_key == "monthly" else "Daily",
            "Fee": "Net" if fee_key == "net" else "Gross",
            "Include Benchmark": "",
            "Convert to Returns": "",
            "Divide By": "",
        }
        rows.append(row)
        return rows, rows, n_no, True

    meta = get_performance_option_meta_cached(PERF_ENGINE).get(key)
    if not meta:
        return rows, rows, "Selected performance series is unavailable.", False
    base_name = str(meta.get("import_name", "")).strip()
    table_key = "monthly" if str(table_choice or "").lower() == "monthly" else "daily"
    fee_key = "N" if str(fee_choice or "").upper().startswith("N") else "G"
    if table_key == "daily" and fee_key == "N":
        import_name = base_name
    elif table_key == "monthly" and fee_key == "N":
        import_name = f"{base_name}_M"
    elif table_key == "daily" and fee_key == "G":
        import_name = f"{base_name}_G"
    else:
        import_name = f"{base_name}_GM"
    if any(str(r.get("import_name", "")).strip() == import_name for r in rows):
        return rows, rows, f"Series `{import_name}` is already staged.", False
    include_bm = bool(include_benchmark)
    row_id = f"performance:{key}:{table_key}:{fee_key}:{1 if include_bm else 0}"
    row = {
        "row_id": row_id,
        "mode": "performance",
        "series_key": key,
        "series_label": str(meta.get("label", base_name)),
        "import_name": import_name,
        "table_choice": table_key,
        "fee_choice": fee_key,
        "include_benchmark": include_bm,
        "Series": import_name,
        "Table": "Monthly" if table_key == "monthly" else "Daily",
        "Fee": "Net" if fee_key == "N" else "Gross",
        "Include Benchmark": "Yes" if include_bm else "No",
        "Convert to Returns": "",
        "Divide By": "",
    }
    rows.append(row)
    return rows, rows, n_no, True


@callback(
    Output("at-raw-db-add-rows-store", "data", allow_duplicate=True),
    Output("at-raw-db-add-grid", "rowData", allow_duplicate=True),
    Output("at-raw-db-add-error-alert", "children", allow_duplicate=True),
    Output("at-raw-db-add-error-alert", "hide", allow_duplicate=True),
    Input("at-raw-db-delete-row-btn", "n_clicks"),
    State("at-raw-db-add-rows-store", "data"),
    State("at-raw-db-add-grid", "selectedRows"),
    prevent_initial_call=True,
)
def at_delete_raw_db_row(n_delete, staged_rows, selected_rows):
    n_no = no_update
    if not n_delete:
        raise PreventUpdate
    rows = [dict(r) for r in (staged_rows or []) if isinstance(r, dict)]
    if not selected_rows:
        return rows, rows, "Select one staged row to delete.", False
    selected_id = str((selected_rows[0] or {}).get("row_id", "")).strip()
    if not selected_id:
        return rows, rows, "Select one staged row to delete.", False
    kept = [r for r in rows if str(r.get("row_id", "")).strip() != selected_id]
    return kept, kept, n_no, True


@callback(
    Output("at-raw-db-add-rows-store", "data", allow_duplicate=True),
    Output("at-raw-db-add-grid", "rowData", allow_duplicate=True),
    Output("at-raw-db-add-error-alert", "children", allow_duplicate=True),
    Output("at-raw-db-add-error-alert", "hide", allow_duplicate=True),
    Input("at-raw-db-clear-rows-btn", "n_clicks"),
    prevent_initial_call=True,
)
def at_clear_raw_db_rows(n_clear):
    if not n_clear:
        raise PreventUpdate
    return [], [], no_update, True


clientside_callback(
    js_portfolio_ok_disabled(),
    Output("at-raw-db-add-ok-button", "disabled", allow_duplicate=True),
    Input("at-raw-db-add-rows-store", "data"),
    Input("at-raw-db-add-modal", "opened"),
    prevent_initial_call=True,
)


@callback(
    Output("at-raw-db-preview-lines", "children", allow_duplicate=True),
    Input("at-raw-db-add-modal", "opened"),
    Input("at-raw-db-add-mode-store", "data"),
    Input("at-raw-db-add-series-select", "value"),
    Input("at-raw-db-add-table-select", "value"),
    Input("at-raw-db-add-fee-select", "value"),
    Input("at-raw-db-add-include-benchmark", "checked"),
    Input("at-raw-db-add-convert-returns", "checked"),
    Input("at-raw-db-add-divide-by", "value"),
    prevent_initial_call=True,
)
def at_update_raw_db_preview(
    opened,
    mode,
    series_key,
    table_choice,
    fee_choice,
    include_benchmark,
    convert_to_returns,
    divide_by,
):
    if not opened:
        raise PreventUpdate
    preview_row = build_preview_row_from_controls(
        mode=mode,
        series_key=series_key,
        table_choice=table_choice,
        fee_choice=fee_choice,
        include_benchmark=include_benchmark,
        convert_to_returns=convert_to_returns,
        divide_by=divide_by,
    )
    if not preview_row:
        return "Select a series to preview option-adjusted results (first 6 rows)."

    lines = get_preview_lines_for_row(preview_row, MRD_ENGINE, PERF_ENGINE)
    if not lines:
        return "No rows returned for the selected options."
    return "\n".join(lines)


@callback(
    Output("dashmat-saved-series-cache-store", "data"),
    Input("dashmat-raw-data-store", "data"),
    State("dashmat-saved-series-cache-store", "data"),
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
    Output("at-db-add-error-alert", "children"),
    Output("at-db-add-error-alert", "hide"),
    Output("at-db-add-ok-button", "disabled"),
    Input("at-db-add-series-select", "value"),
    Input("dashmat-raw-data-store", "data"),
    Input("at-db-add-modal", "opened"),
    prevent_initial_call=True,
)
def validate_db_add_selection(selected_benches, raw_data, opened):
    return compute_validate_db_add_selection(selected_benches, raw_data, opened)


@callback(
    Output("at-portfolio-add-modal", "opened", allow_duplicate=True),
    Output("at-portfolio-add-modal", "title", allow_duplicate=True),
    Output("at-portfolio-add-mode-store", "data", allow_duplicate=True),
    Output("at-portfolio-add-series-select", "data", allow_duplicate=True),
    Output("at-portfolio-add-series-select", "value", allow_duplicate=True),
    Output("at-portfolio-add-type-select", "data", allow_duplicate=True),
    Output("at-portfolio-add-type-select", "value", allow_duplicate=True),
    Output("at-portfolio-add-benchmark-type-select", "data", allow_duplicate=True),
    Output("at-portfolio-add-benchmark-type-select", "value", allow_duplicate=True),
    Output("at-portfolio-add-include-benchmark", "checked", allow_duplicate=True),
    Output("at-portfolio-add-benchmark-type-select", "disabled", allow_duplicate=True),
    Output("at-portfolio-add-rows-store", "data", allow_duplicate=True),
    Output("at-portfolio-add-grid", "rowData", allow_duplicate=True),
    Output("at-portfolio-add-error-alert", "hide", allow_duplicate=True),
    Input("at-menu-add-portfolios-peer", "n_clicks"),
    Input("at-menu-add-portfolios-index", "n_clicks"),
    Input("at-menu-add-portfolios-other", "n_clicks"),
    Input("at-welcome-add-portfolios-peer-btn", "n_clicks"),
    Input("at-welcome-add-portfolios-index-btn", "n_clicks"),
    Input("at-welcome-add-portfolios-other-btn", "n_clicks"),
    prevent_initial_call=True,
)
def at_open_portfolio_add_modal(
    peer_clicks,
    index_clicks,
    other_clicks,
    welcome_peer_clicks,
    welcome_index_clicks,
    welcome_other_clicks,
):
    return compute_open_portfolio_add_modal(
        prefix="at",
        triggered_id=callback_context.triggered_id,
        peer_clicks=peer_clicks,
        index_clicks=index_clicks,
        other_clicks=other_clicks,
        welcome_peer_clicks=welcome_peer_clicks,
        welcome_index_clicks=welcome_index_clicks,
        welcome_other_clicks=welcome_other_clicks,
        db_engine=DB_ENGINE,
    )


clientside_callback(
    js_portfolio_benchmark_toggle(),
    Output("at-portfolio-add-benchmark-type-select", "disabled", allow_duplicate=True),
    Output("at-portfolio-add-benchmark-type-select", "value", allow_duplicate=True),
    Input("at-portfolio-add-include-benchmark", "checked"),
    State("at-portfolio-add-benchmark-type-select", "data"),
    State("at-portfolio-add-benchmark-type-select", "value"),
    prevent_initial_call=True,
)


@callback(
    Output("at-portfolio-add-include-benchmark", "disabled"),
    Output("at-portfolio-add-include-benchmark", "checked", allow_duplicate=True),
    Input("at-portfolio-add-mode-store", "data"),
    Input("at-portfolio-add-series-select", "value"),
    State("at-portfolio-add-include-benchmark", "checked"),
    prevent_initial_call=True,
)
def at_sync_include_benchmark_enabled(mode, selected_portfolio, current_checked):
    return compute_sync_include_benchmark_enabled(mode, selected_portfolio, current_checked, DB_ENGINE)


clientside_callback(
    js_portfolio_add_row(),
    Output("at-portfolio-add-rows-store", "data", allow_duplicate=True),
    Output("at-portfolio-add-grid", "rowData", allow_duplicate=True),
    Output("at-portfolio-add-error-alert", "children", allow_duplicate=True),
    Output("at-portfolio-add-error-alert", "hide", allow_duplicate=True),
    Input("at-portfolio-add-row-btn", "n_clicks"),
    State("at-portfolio-add-rows-store", "data"),
    State("at-portfolio-add-series-select", "value"),
    State("at-portfolio-add-type-select", "value"),
    State("at-portfolio-add-include-benchmark", "checked"),
    State("at-portfolio-add-benchmark-type-select", "value"),
    prevent_initial_call=True,
)

clientside_callback(
    js_portfolio_delete_row(),
    Output("at-portfolio-add-rows-store", "data", allow_duplicate=True),
    Output("at-portfolio-add-grid", "rowData", allow_duplicate=True),
    Output("at-portfolio-add-error-alert", "children", allow_duplicate=True),
    Output("at-portfolio-add-error-alert", "hide", allow_duplicate=True),
    Input("at-portfolio-delete-row-btn", "n_clicks"),
    State("at-portfolio-add-rows-store", "data"),
    State("at-portfolio-add-grid", "selectedRows"),
    prevent_initial_call=True,
)

clientside_callback(
    js_portfolio_clear_rows(),
    Output("at-portfolio-add-rows-store", "data", allow_duplicate=True),
    Output("at-portfolio-add-grid", "rowData", allow_duplicate=True),
    Output("at-portfolio-add-error-alert", "children", allow_duplicate=True),
    Output("at-portfolio-add-error-alert", "hide", allow_duplicate=True),
    Input("at-portfolio-clear-rows-btn", "n_clicks"),
    prevent_initial_call=True,
)


@callback(
    Output("at-portfolio-add-modal", "opened", allow_duplicate=True),
    Output("at-portfolio-add-rows-store", "data", allow_duplicate=True),
    Output("at-portfolio-add-grid", "rowData", allow_duplicate=True),
    Input("at-portfolio-add-cancel-button", "n_clicks"),
    prevent_initial_call=True,
)
def at_close_portfolio_add_modal(n_clicks):
    return compute_close_portfolio_add_modal(n_clicks)


clientside_callback(
    js_portfolio_ok_disabled(),
    Output("at-portfolio-add-ok-button", "disabled"),
    Input("at-portfolio-add-rows-store", "data"),
    Input("at-portfolio-add-modal", "opened"),
)

clientside_callback(
    js_set_ui_blocker_true(),
    Output("at-ui-blocker-store", "data", allow_duplicate=True),
    Input("at-portfolio-add-ok-button", "n_clicks"),
    prevent_initial_call=True,
)

clientside_callback(
    js_release_ui_blocker_on_modal_state(),
    Output("at-ui-blocker-store", "data", allow_duplicate=True),
    Input("at-portfolio-add-modal", "opened"),
    Input("at-portfolio-add-error-alert", "hide"),
    prevent_initial_call=True,
)

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
                                                id="at-open-series-modal-button",
                                                variant="light",
                                                size="sm",
                                                w=200,
                                            ),
                                        ]),
                                        dmc.Select(
                                            id="at-periodicity-select",
                                            label="Periodicity",
                                            data=periodicity_options,
                                            value=periodicity_value,
                                            w=200,
                                            disabled=False,
                                        ),
                                        html.Div([
                                            dmc.Text("Returns Type", size="sm", mb=3, fw=500),
                                            dmc.SegmentedControl(
                                                id="at-returns-type-select",
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
                                                    id="at-vol-scaler-input",
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
                                        id="at-date-picker-wrapper",
                                        children=[
                                            html.Div([
                                                dmc.DateInput(
                                                    id="at-start-date-picker",
                                                    label="Start Date",
                                                    value=None,
                                                    w=200,
                                                    valueFormat="YYYY-MM-DD",
                                                ),
                                            ], style={"marginRight": "15px"}),
                                            html.Div([
                                                dmc.DateInput(
                                                    id="at-end-date-picker",
                                                    label="End Date",
                                                    value=None,
                                                    w=200,
                                                    valueFormat="YYYY-MM-DD",
                                                ),
                                            ], style={"marginRight": "15px"}),
                                            html.Div([
                                                dmc.Button(
                                                    "Common Range",
                                                    id="at-common-range-button",
                                                    size="xs",
                                                    variant="outline",
                                                    disabled=True,
                                                    w=120,
                                                ),
                                            ], style={"marginRight": "10px", "alignSelf": "flex-end", "marginBottom": "2px"}),
                                            html.Div([
                                                dmc.Button(
                                                    "Common Daily",
                                                    id="at-common-daily-button",
                                                    size="xs",
                                                    variant="outline",
                                                    disabled=True,
                                                    w=120,
                                                ),
                                            ], style={"marginRight": "10px", "alignSelf": "flex-end", "marginBottom": "2px"}),
                                            html.Div([
                                                dmc.Button(
                                                    "Max Range",
                                                    id="at-maximum-range-button",
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
            id="at-main-tabs",
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
                            id="at-loading-returns",
                            type="default",
                            delay_show=300,
                            delay_hide=150,
                            style={"flex": "1", "display": "flex", "flexDirection": "column", "overflow": "hidden"},
                            parent_style={"flex": "1", "display": "flex", "flexDirection": "column", "overflow": "hidden"},
                            children=[
                                dag.AgGrid(
                                    enableEnterpriseModules=True,
                                    licenseKey=AG_GRID_LICENSE_KEY,
                                    id="at-returns-grid",
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
                                    id="at-rolling-metric-select",
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
                                    id="at-rolling-window-select",
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
                                    id="at-rolling-return-type-select",
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
                                    id="at-rolling-chart-switch",
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
                            id="at-rolling-grid-container",
                            style=rolling_grid_style,
                            children=[
                                dag.AgGrid(
                                    enableEnterpriseModules=True,
                                    licenseKey=AG_GRID_LICENSE_KEY,
                                    id="at-rolling-grid",
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
                            id="at-rolling-chart-container",
                            style=rolling_chart_style,
                            children=[
                                html.Div(id="at-rolling-chart-wrapper", style={"height": "100%", "width": "100%"}),
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
                            id="at-loading-statistics",
                            type="default",
                            delay_show=300,
                            delay_hide=150,
                            style={"flex": "1", "display": "flex", "flexDirection": "column", "overflow": "hidden"},
                            parent_style={"flex": "1", "display": "flex", "flexDirection": "column", "overflow": "hidden"},
                            children=[
                                dag.AgGrid(
                                    enableEnterpriseModules=True,
                                    licenseKey=AG_GRID_LICENSE_KEY,
                                    id="at-statistics-grid",
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
                                    id="at-monthly-view-checkbox",
                                    data=[
                                        {"value": "annual", "label": "Annual"},
                                        {"value": "monthly", "label": "Monthly"},
                                    ],
                                    value=monthly_view,
                                    size="sm",
                                ),
                                dmc.Select(
                                    id="at-monthly-series-select",
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
                            id="at-calendar-grid",
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
                                    id="at-correlation-view-switch",
                                    data=[
                                        {"value": "correlation", "label": "Correlation"},
                                        {"value": "correlogram", "label": "Correlogram"},
                                    ],
                                    value="correlogram",
                                    size="sm",
                                ),
                                dmc.NumberInput(
                                    id="at-correlogram-block-width",
                                    label=None,
                                    value=None,
                                    min=50,
                                    step=50,
                                    suffix="px",
                                    w=100,
                                    size="sm",
                                ),
                            ],
                        ),
                        dcc.Loading(
                            id="at-loading-correlogram",
                            type="default",
                            delay_show=0,
                            delay_hide=150,
                            style={"flex": "1", "display": "flex", "flexDirection": "column", "overflow": "auto"},
                            parent_style={"flex": "1", "display": "flex", "flexDirection": "column", "overflow": "auto"},
                            children=[
                                html.Div(
                                    id="at-correlogram-container",
                                    style={"flex": "1", "minHeight": "520px", "overflow": "auto"},
                                ),
                            ],
                        ),
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
                                    id="at-growth-chart-switch",
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
                            id="at-growth-chart-container",
                            style=growth_chart_style,
                            children=[
                                html.Div(id="at-growth-charts-container", style={"height": "100%", "width": "100%"}),
                            ],
                        ),
                        html.Div(
                            id="at-growth-grid-container",
                            style=growth_grid_style,
                            children=[
                                dag.AgGrid(
                                    enableEnterpriseModules=True,
                                    licenseKey=AG_GRID_LICENSE_KEY,
                                    id="at-growth-grid",
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
                                    id="at-drawdown-chart-switch",
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
                            id="at-drawdown-chart-container",
                            style=drawdown_chart_style,
                            children=[
                                html.Div(id="at-drawdown-charts", style={"height": "100%", "width": "100%"}),
                            ],
                        ),
                        html.Div(
                            id="at-drawdown-grid-container",
                            style=drawdown_grid_style,
                            children=[
                                dag.AgGrid(
                                    enableEnterpriseModules=True,
                                    licenseKey=AG_GRID_LICENSE_KEY,
                                    id="at-drawdown-grid",
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
    style={"height": "calc(100vh - 55px)", "display": "flex", "flexDirection": "column", "overflow": "visible"}, # 45px for header + 10px bottom margin
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
                            trigger="click",
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
                                            "New session",
                                            id="at-menu-clear-local-storage",
                                            leftSection=DashIconify(icon="tabler:trash", width=14),
                                        ),
                                        dmc.MenuItem(
                                            "Load session",
                                            id="at-menu-load-session",
                                            leftSection=DashIconify(icon="tabler:folder-open", width=14),
                                        ),
                                        dmc.MenuItem(
                                            "Save session",
                                            id="at-menu-save-session",
                                            disabled=True,
                                            leftSection=DashIconify(icon="tabler:device-floppy", width=14),
                                        ),
                                        dmc.MenuDivider(),
                                        dmc.MenuItem(
                                            "Download Excel",
                                            id="at-menu-download-excel",
                                            disabled=True,
                                            leftSection=DashIconify(icon="tabler:file-spreadsheet", width=14),
                                        ),
                                        dmc.MenuDivider(),
                                        dmc.MenuItem(
                                            "Exit",
                                            id="at-menu-exit",
                                            color="red",
                                            leftSection=DashIconify(icon="tabler:door-exit", width=14),
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        # Edit Menu (left)
                        dmc.Menu(
                            trigger="click",
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
                                            "Add AA Tool indices...",
                                            id="at-menu-add-from-db",
                                            leftSection=DashIconify(icon="tabler:database", width=14),
                                        ),
                                        dmc.MenuDivider(),
                                        dmc.MenuItem(
                                            "Add peer-relative portfolios...",
                                            id="at-menu-add-portfolios-peer",
                                            leftSection=DashIconify(icon="tabler:users", width=14),
                                        ),
                                        dmc.MenuItem(
                                            "Add index-relative portfolios...",
                                            id="at-menu-add-portfolios-index",
                                            leftSection=DashIconify(icon="tabler:chart-line", width=14),
                                        ),
                                        dmc.MenuItem(
                                            "Add alternative portfolios...",
                                            id="at-menu-add-portfolios-other",
                                            leftSection=DashIconify(icon="tabler:stack", width=14),
                                        ),
                                        dmc.MenuDivider(),
                                        dmc.MenuItem(
                                            "Add raw factor data...",
                                            id="at-menu-add-raw-factor",
                                            leftSection=DashIconify(icon="tabler:chart-dots", width=14),
                                        ),
                                        dmc.MenuItem(
                                            "Add raw funds...",
                                            id="at-menu-add-raw-funds",
                                            leftSection=DashIconify(icon="tabler:building-bank", width=14),
                                        ),
                                        dmc.MenuItem(
                                            "Add raw performance...",
                                            id="at-menu-add-raw-performance",
                                            leftSection=DashIconify(icon="tabler:activity-heartbeat", width=14),
                                        ),
                                        dmc.MenuDivider(),
                                        dmc.MenuItem(
                                            "Add series from file...",
                                            id="at-menu-add-series",
                                            leftSection=DashIconify(icon="tabler:upload", width=14),
                                        ),
                                        dmc.MenuDivider(),
                                        dmc.MenuItem(
                                            "Clear server cache",
                                            id="at-menu-clear-server-cache",
                                            leftSection=DashIconify(icon="tabler:server-off", width=14),
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        # Page switch button
                        dmc.Button(
                            "Switch to Optimization",
                            id="at-menu-view-portfolio",
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
                            id="at-menu-help-guide",
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
                id="at-upload-data",
                children=html.Div(id="at-upload-trigger"),
                multiple=False,
                accept=".csv,.xlsx,.xls",
            ),
            style={"display": "none"},
        ),

        build_series_selection_modal(AT_WELCOME_MODAL_CONFIG),
        build_db_add_modal("at"),
        build_portfolio_add_modal("at", AG_GRID_LICENSE_KEY),
        build_raw_db_add_modal("at", AG_GRID_LICENSE_KEY),
        build_sheet_select_modal("at"),

        # Help Modal
        dmc.Modal(
            id="at-help-modal",
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
            id="at-welcome-screen-container",
            children=build_welcome_screen(),
            style={"display": "block"}
        ),

        # Main App Container (Initially Hidden)
        html.Div(
            id="at-main-app-container",
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
        # dashmat-raw-data-store and dashmat-original-periodicity-store are defined in app.py (shared across pages)
        dcc.Store(id="at-benchmark-assignments-store", data={}, storage_type="session"),
        dcc.Store(id="at-long-short-store", data={}, storage_type="session"),
        dcc.Store(id="at-periodicity-value-store", data="daily_trading", storage_type="session"),
        dcc.Store(id="at-periodicity-load-sync-dummy", data=None),
        dcc.Store(id="at-returns-type-value-store", data="total", storage_type="session"),
        dcc.Store(id="at-series-select-value-store", data=[], storage_type="session"),
        dcc.Store(id="at-series-order-store", data=[], storage_type="session"),
        dcc.Store(id="at-active-tab-store", data="statistics", storage_type="session"),
        dcc.Store(id="at-rolling-window-store", data="1y", storage_type="session"),
        dcc.Store(id="at-rolling-metric-store", data="total_return", storage_type="session"),
        dcc.Store(id="at-rolling-return-type-store", data="annualized", storage_type="session"),
        dcc.Store(id="at-rolling-chart-switch-store", data="chart", storage_type="session"),
        dcc.Store(id="at-drawdown-chart-switch-store", data="chart", storage_type="session"),
        dcc.Store(id="at-growth-chart-switch-store", data="chart", storage_type="session"),
        dcc.Store(id="at-monthly-view-store", data="annual", storage_type="session"),
        dcc.Store(id="at-monthly-series-store", data=None, storage_type="session"),
        dcc.Store(id="at-date-range-store", data=None, storage_type="session"),
        dcc.Store(id="at-state-ready-store", data=False, storage_type="session"),
        dcc.Store(id="at-statistics-loaded-store", data=False, storage_type="session"),
        dcc.Store(id="at-vol-scaler-value-store", data=0, storage_type="session"),
        dcc.Store(id="at-vol-scaling-assignments-store", data={}, storage_type="session"),
        dcc.Store(id="at-download-enabled-store", data=False),
        dcc.Store(id="at-first-load-store", data=False, storage_type="session"),
        # Temporary stores for modal state
        dcc.Store(id="at-temp-series-select", data=[]),
        dcc.Store(id="at-temp-benchmark-assignments-store", data={}),
        dcc.Store(id="at-temp-long-short-store", data={}),
        dcc.Store(id="at-temp-vol-scaling-assignments-store", data={}),
        dcc.Store(id="at-temp-series-order-store", data=[]),
        dcc.Store(id="at-temp-deleted-series-store", data=[]),
        dcc.Store(id="at-portfolio-add-mode-store", data=None),
        dcc.Store(id="at-portfolio-add-rows-store", data=[]),
        dcc.Store(id="at-raw-db-add-mode-store", data=None),
        dcc.Store(id="at-raw-db-add-rows-store", data=[]),
        # Temp stores for sheet selection (stash upload while user picks a tab)
        dcc.Store(id="at-sheet-select-contents-store", data=None),
        dcc.Store(id="at-sheet-select-filename-store", data=None),
        dcc.Store(id="at-sheet-select-sheetnames-store", data=None),
        dcc.Download(id="at-download-excel"),
        dcc.Download(id="at-download-sample-daily"),
        dcc.Download(id="at-download-sample-monthly"),
        # Save/Load session
        dcc.Store(id="at-save-session-dummy", data=None, storage_type="memory"),
        dcc.Store(id="at-load-session-dummy", data=None, storage_type="memory"),
        dcc.Store(id="at-server-cache-clear-result", data=None, storage_type="memory"),
        html.Div(
            dcc.Upload(
                id="at-load-session-upload",
                children=html.Div(),
                multiple=False,
                accept=".json",
            ),
            style={"display": "none"},
        ),
        dcc.Location(id="at-url-location", refresh=False),
        # Moved series-select and edit-mode to global scope
        dcc.Store(id="at-series-select", data=[], storage_type="session"),
        dcc.Store(id="at-series-edit-mode", data=None),

        # Store to trigger clientside focus on edit input
        dcc.Store(id="at-edit-box-focus-trigger", data=None),
        # Dummy div for clientside callback output
        html.Div(id="at-dummy-focus-output"),
        
        # Correlogram metadata for client-side sizing
        dcc.Store(id="at-correlogram-meta-store", data={}),
        dcc.Store(id="at-correlogram-target-key-store", data=None),
        dcc.Store(id="at-correlogram-rendered-key-store", data=None),

        # UI Blocker for file dialog (Overlay)
        dcc.Store(id="at-ui-blocker-store", data=False),
        dmc.LoadingOverlay(
            id="at-ui-blocker-overlay",
            visible=False,
            zIndex=2000,
            overlayProps={"radius": "sm", "blur": 2},
            loaderProps={"variant": "bars"},
        ),

        # One-shot interval to trigger visibility check after session-storage hydration
        dcc.Interval(id="at-page-load-trigger", interval=50, max_intervals=1, n_intervals=0),
    ],
)


# Toggle welcome/main visibility based on dashmat-raw-data-store.
# Uses a one-shot Interval to guarantee session-storage has hydrated on
# cross-page navigation, plus dashmat-raw-data-store Input for same-page uploads.
clientside_callback(
    """
    function(n_intervals, data) {
        if (data) {
            return [{display: "none"}, {display: "flex", flexDirection: "column", flex: "1", overflow: "hidden"}];
        }
        return [{display: "block"}, {display: "none"}];
    }
    """,
    Output("at-welcome-screen-container", "style"),
    Output("at-main-app-container", "style"),
    Input("at-page-load-trigger", "n_intervals"),
    Input("dashmat-raw-data-store", "data"),
)


@callback(
    Output("at-periodicity-select", "data", allow_duplicate=True),
    Output("at-periodicity-select", "value", allow_duplicate=True),
    Output("at-returns-type-select", "value"),
    Output("at-vol-scaler-input", "value"),
    Output("at-main-tabs", "value"),
    Output("at-rolling-window-select", "value"),
    Output("at-rolling-metric-select", "value"),
    Output("at-rolling-return-type-select", "value"),
    Output("at-rolling-return-type-select", "disabled", allow_duplicate=True),
    Output("at-rolling-return-type-select", "style", allow_duplicate=True),
    Output("at-rolling-chart-switch", "value"),
    Output("at-drawdown-chart-switch", "value"),
    Output("at-growth-chart-switch", "value"),
    Output("at-monthly-view-checkbox", "value"),
    Output("at-series-select", "data"),
    Output("at-state-ready-store", "data", allow_duplicate=True),
    Input("at-page-load-trigger", "n_intervals"),
    Input("dashmat-raw-data-store", "data"),
    State("dashmat-original-periodicity-store", "data"),
    State("at-periodicity-value-store", "data"),
    State("at-series-select-value-store", "data"),
    State("at-returns-type-value-store", "data"),
    State("at-vol-scaler-value-store", "data"),
    State("at-active-tab-store", "data"),
    State("at-rolling-window-store", "data"),
    State("at-rolling-metric-store", "data"),
    State("at-rolling-return-type-store", "data"),
    State("at-rolling-chart-switch-store", "data"),
    State("at-drawdown-chart-switch-store", "data"),
    State("at-growth-chart-switch-store", "data"),
    State("at-monthly-view-store", "data"),
    State("at-monthly-series-store", "data"),
    State("dashmat-pending-new-series-store", "data"),
    prevent_initial_call="initial_duplicate",
)
def restore_application_state(n_intervals, raw_data, orig_periodicity, stored_periodicity, stored_series, stored_returns, stored_vol, stored_tab, stored_roll_win, stored_roll_metric, stored_roll_type, stored_roll_chart, stored_dd_chart, stored_gr_chart, stored_monthly_view, stored_monthly_series, pending_series):
    if not raw_data:
        # Reset defaults (visibility handled by clientside callback)
        return (
            [{"value": "daily_trading", "label": "Daily (Trading)"}], "daily_trading", "total", 0, "statistics",
            "1y", "total_return", "annualized", False, {}, "chart", "chart", "chart",
            "annual", [], False
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
            monthly_view, valid_selection, False
        )

    except Exception:
        # Fallback to defaults on error (visibility handled by clientside callback)
        return (
            [{"value": "daily_trading", "label": "Daily (Trading)"}], "daily_trading", "total", 0, "statistics",
            "1y", "total_return", "annualized", False, {}, "chart", "chart", "chart",
            "annual", [], False
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
    Output("at-url-location", "pathname"),
    Input("at-menu-exit", "n_clicks"),
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
    Output("at-url-location", "pathname", allow_duplicate=True),
    Input("at-menu-view-portfolio", "n_clicks"),
    prevent_initial_call=True,
)


# Clientside callback to clear session storage and refresh page
clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks) {
            // Clear all sessionStorage keys for both pages
            const keysToRemove = [
                'dashmat-raw-data-store',
                'dashmat-original-periodicity-store',
                'dashmat-pending-new-series-store',
                'dashmat-saved-series-cache-store',
                'bctbill13-cache-store',
                'at-series-select',
                'at-benchmark-assignments-store',
                'at-long-short-store',
                'at-periodicity-value-store',
                'at-returns-type-value-store',
                'at-series-select-value-store',
                'at-series-order-store',
                'at-active-tab-store',
                'at-rolling-window-store',
                'at-rolling-return-type-store',
                'at-rolling-chart-switch-store',
                'at-drawdown-chart-switch-store',
                'at-growth-chart-switch-store',
                'at-monthly-view-store',
                'at-monthly-series-store',
                'at-date-range-store',
                'at-vol-scaler-value-store',
                'at-vol-scaling-assignments-store',
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
    Output("at-url-location", "pathname", allow_duplicate=True),
    Input("at-menu-clear-local-storage", "n_clicks"),
    prevent_initial_call=True,
)


@callback(
    Output("at-server-cache-clear-result", "data"),
    Input("at-menu-clear-server-cache", "n_clicks"),
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
    js_trigger_upload_with_cancel("at"),
    Output("at-ui-blocker-store", "data", allow_duplicate=True),
    Input("at-menu-add-series", "n_clicks"),
    prevent_initial_call=True,
)


clientside_callback(
    """
    function(is_loading) {
        return is_loading || false;
    }
    """,
    Output("at-ui-blocker-overlay", "visible"),
    Input("at-ui-blocker-store", "data"),
)


@callback(
    Output("at-series-selection-modal", "opened", allow_duplicate=True),
    Output("at-temp-series-select", "data", allow_duplicate=True),
    Output("at-temp-benchmark-assignments-store", "data", allow_duplicate=True),
    Output("at-temp-long-short-store", "data", allow_duplicate=True),
    Output("at-temp-series-order-store", "data", allow_duplicate=True),
    Output("at-temp-deleted-series-store", "data", allow_duplicate=True),
    Output("at-temp-vol-scaling-assignments-store", "data", allow_duplicate=True),
    Input("at-open-series-modal-button", "n_clicks"),
    State("at-series-select", "data"),
    State("at-benchmark-assignments-store", "data"),
    State("at-long-short-store", "data"),
    State("at-series-order-store", "data"),
    State("at-vol-scaling-assignments-store", "data"),
    prevent_initial_call=True,
)
def open_modal(n_clicks, current_select, current_bench, current_ls, current_order, current_vol_scaling):
    if not n_clicks:
        raise PreventUpdate
    return True, current_select, current_bench, current_ls, current_order, [], current_vol_scaling


@callback(
    Output("at-series-select", "data", allow_duplicate=True),
    Output("at-benchmark-assignments-store", "data", allow_duplicate=True),
    Output("at-long-short-store", "data", allow_duplicate=True),
    Output("at-series-order-store", "data", allow_duplicate=True),
    Output("at-series-selection-modal", "opened", allow_duplicate=True),
    Output("at-series-select-value-store", "data", allow_duplicate=True), # Sync persistence
    Output("dashmat-raw-data-store", "data", allow_duplicate=True),
    Output("at-vol-scaling-assignments-store", "data", allow_duplicate=True),
    Input("at-modal-ok-button", "n_clicks"),
    State("at-temp-series-select", "data"),
    State("at-temp-benchmark-assignments-store", "data"),
    State("at-temp-long-short-store", "data"),
    State("at-temp-series-order-store", "data"),
    State("at-temp-deleted-series-store", "data"),
    State("dashmat-raw-data-store", "data"),
    State("at-temp-vol-scaling-assignments-store", "data"),
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

    raw_data_output = updated_raw_data if updated_raw_data != raw_data else no_update
    return temp_select, temp_bench, temp_ls, temp_order, False, temp_select, raw_data_output, temp_vol_scaling


@callback(
    Output("at-series-selection-modal", "opened", allow_duplicate=True),
    Input("at-modal-cancel-button", "n_clicks"),
    prevent_initial_call=True,
)
def on_modal_cancel(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    return False


@callback(
    Output("at-temp-series-order-store", "data", allow_duplicate=True),
    Output("at-temp-series-select", "data", allow_duplicate=True),
    Input("at-series-selection-grid", "virtualRowData", allow_optional=True),
    Input("at-series-selection-grid", "selectedRows", allow_optional=True),
    State("at-temp-series-order-store", "data"),
    State("at-temp-series-select", "data"),
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
    Output("at-periodicity-value-store", "data"),
    Input("at-periodicity-select", "value"),
    prevent_initial_call=True,
)


# Sync periodicity to PortOpt only on raw-data load/update events.
clientside_callback(
    """
    function(rawData, periodicityValue) {
        const ctx = window.dash_clientside.callback_context;
        const triggered = (ctx && ctx.triggered) ? ctx.triggered : [];
        const rawTriggered = triggered.some(
            t => t && t.prop_id && t.prop_id.indexOf("dashmat-raw-data-store.") === 0
        );
        if (!rawTriggered || !rawData || !periodicityValue) {
            return window.dash_clientside.no_update;
        }
        sessionStorage.setItem("po-periodicity-value-store", JSON.stringify(periodicityValue));
        return periodicityValue;
    }
    """,
    Output("at-periodicity-load-sync-dummy", "data"),
    Input("dashmat-raw-data-store", "data"),
    Input("at-periodicity-value-store", "data"),
    prevent_initial_call=True,
)


# Clientside callback for returns type selection storage
clientside_callback(
    "function(value) { return value; }",
    Output("at-returns-type-value-store", "data"),
    Input("at-returns-type-select", "value"),
    prevent_initial_call=True,
)


# Clientside callback for vol scaler value storage
clientside_callback(
    "function(value) { return value; }",
    Output("at-vol-scaler-value-store", "data"),
    Input("at-vol-scaler-input", "value"),
    prevent_initial_call=True,
)


# Clientside callback for series selection storage
clientside_callback(
    "function(value) { return value || []; }",
    Output("at-series-select-value-store", "data"),
    Input("at-series-select", "data"),
    prevent_initial_call=True,
)


# Clientside callback for active tab storage
clientside_callback(
    "function(value) { return value || 'statistics'; }",
    Output("at-active-tab-store", "data"),
    Input("at-main-tabs", "value"),
    prevent_initial_call=True,
)





# Clientside callback for rolling window selection storage
clientside_callback(
    "function(value) { return value || '1y'; }",
    Output("at-rolling-window-store", "data"),
    Input("at-rolling-window-select", "value"),
    prevent_initial_call=True,
)


# Clientside callback for rolling metric selection storage
clientside_callback(
    "function(value) { return value || 'total_return'; }",
    Output("at-rolling-metric-store", "data"),
    Input("at-rolling-metric-select", "value"),
    prevent_initial_call=True,
)


# Clientside callback for rolling return type storage
clientside_callback(
    "function(value) { return value || 'annualized'; }",
    Output("at-rolling-return-type-store", "data"),
    Input("at-rolling-return-type-select", "value"),
    prevent_initial_call=True,
)


@callback(
    Output("at-rolling-return-type-select", "disabled"),
    Output("at-rolling-return-type-select", "style"),
    Input("at-rolling-metric-select", "value"),
)
def update_rolling_controls_state(metric):
    """Enable/disable return type select based on metric."""
    if metric in ["total_return", "excess_return"]:
        return False, {}
    return True, {"opacity": 0.5, "pointerEvents": "none"}





# Clientside callback for rolling chart switch storage
clientside_callback(
    "function(value) { return value !== null && value !== undefined ? value : 'chart'; }",
    Output("at-rolling-chart-switch-store", "data"),
    Input("at-rolling-chart-switch", "value"),
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
    Output("at-rolling-grid-container", "style"),
    Output("at-rolling-chart-container", "style"),
    Input("at-rolling-chart-switch", "value"),
    prevent_initial_call=True,
)


# Clientside callback for drawdown chart switch storage
clientside_callback(
    "function(value) { return value !== null && value !== undefined ? value : 'chart'; }",
    Output("at-drawdown-chart-switch-store", "data"),
    Input("at-drawdown-chart-switch", "value"),
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
    Output("at-drawdown-grid-container", "style"),
    Output("at-drawdown-chart-container", "style"),
    Input("at-drawdown-chart-switch", "value"),
    prevent_initial_call=True,
)


# Clientside callback for growth chart switch storage
clientside_callback(
    "function(value) { return value !== null && value !== undefined ? value : 'chart'; }",
    Output("at-growth-chart-switch-store", "data"),
    Input("at-growth-chart-switch", "value"),
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
    Output("at-growth-grid-container", "style"),
    Output("at-growth-chart-container", "style"),
    Input("at-growth-chart-switch", "value"),
    prevent_initial_call=True,
)


# Clientside callback for monthly view storage
clientside_callback(
    "function(value) { return value !== null && value !== undefined ? value : 'annual'; }",
    Output("at-monthly-view-store", "data"),
    Input("at-monthly-view-checkbox", "value"),
    prevent_initial_call=True,
)


# Clientside callback for monthly series selection storage
clientside_callback(
    "function(value) { return value; }",
    Output("at-monthly-series-store", "data"),
    Input("at-monthly-series-select", "value"),
    prevent_initial_call=True,
)





@callback(
    Output("dashmat-raw-data-store", "data", allow_duplicate=True),
    Output("dashmat-original-periodicity-store", "data", allow_duplicate=True),
    Output("at-periodicity-select", "data", allow_duplicate=True),
    Output("at-periodicity-select", "value", allow_duplicate=True),
    Output("at-periodicity-select", "disabled", allow_duplicate=True),
    Output("at-temp-series-select", "data", allow_duplicate=True),
    Output("at-alert-message", "children", allow_duplicate=True),
    Output("at-alert-message", "color", allow_duplicate=True),
    Output("at-alert-message", "hide", allow_duplicate=True),
    Output("at-periodicity-value-store", "data", allow_duplicate=True),
    Output("at-series-selection-modal", "opened", allow_duplicate=True),
    Output("at-temp-benchmark-assignments-store", "data", allow_duplicate=True),
    Output("at-temp-long-short-store", "data", allow_duplicate=True),
    Output("at-temp-series-order-store", "data", allow_duplicate=True),
    Output("at-first-load-store", "data", allow_duplicate=True),
    Output("at-temp-deleted-series-store", "data", allow_duplicate=True),
    Output("at-temp-vol-scaling-assignments-store", "data", allow_duplicate=True),
    Output("at-db-add-modal", "opened", allow_duplicate=True),
    Output("at-db-add-series-select", "value", allow_duplicate=True),
    Input("at-db-add-ok-button", "n_clicks"),
    State("at-db-add-series-select", "value"),
    State("dashmat-raw-data-store", "data"),
    State("dashmat-original-periodicity-store", "data"),
    State("at-series-select", "data"),
    State("at-benchmark-assignments-store", "data"),
    State("at-long-short-store", "data"),
    State("at-series-order-store", "data"),
    State("at-first-load-store", "data"),
    State("at-vol-scaling-assignments-store", "data"),
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
    Output("dashmat-raw-data-store", "data", allow_duplicate=True),
    Output("dashmat-original-periodicity-store", "data", allow_duplicate=True),
    Output("at-periodicity-select", "data", allow_duplicate=True),
    Output("at-periodicity-select", "value", allow_duplicate=True),
    Output("at-periodicity-select", "disabled", allow_duplicate=True),
    Output("at-temp-series-select", "data", allow_duplicate=True),
    Output("at-alert-message", "children", allow_duplicate=True),
    Output("at-alert-message", "color", allow_duplicate=True),
    Output("at-alert-message", "hide", allow_duplicate=True),
    Output("at-periodicity-value-store", "data", allow_duplicate=True),
    Output("at-series-selection-modal", "opened", allow_duplicate=True),
    Output("at-temp-benchmark-assignments-store", "data", allow_duplicate=True),
    Output("at-temp-long-short-store", "data", allow_duplicate=True),
    Output("at-temp-series-order-store", "data", allow_duplicate=True),
    Output("at-first-load-store", "data", allow_duplicate=True),
    Output("at-temp-deleted-series-store", "data", allow_duplicate=True),
    Output("at-temp-vol-scaling-assignments-store", "data", allow_duplicate=True),
    Output("at-raw-db-add-modal", "opened", allow_duplicate=True),
    Output("at-raw-db-add-rows-store", "data", allow_duplicate=True),
    Output("at-raw-db-add-grid", "rowData", allow_duplicate=True),
    Output("at-raw-db-add-error-alert", "children", allow_duplicate=True),
    Output("at-raw-db-add-error-alert", "hide", allow_duplicate=True),
    Output("at-raw-db-preview-lines", "children", allow_duplicate=True),
    Input("at-raw-db-add-ok-button", "n_clicks"),
    State("at-raw-db-add-mode-store", "data"),
    State("at-raw-db-add-rows-store", "data"),
    State("dashmat-raw-data-store", "data"),
    State("dashmat-original-periodicity-store", "data"),
    State("at-series-select", "data"),
    State("at-benchmark-assignments-store", "data"),
    State("at-long-short-store", "data"),
    State("at-series-order-store", "data"),
    State("at-first-load-store", "data"),
    State("at-vol-scaling-assignments-store", "data"),
    prevent_initial_call=True,
)
def at_add_raw_series_from_database(
    n_clicks,
    mode,
    staged_rows,
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
    rows = [dict(r) for r in (staged_rows or []) if isinstance(r, dict)]
    mode_key = str(mode or "").strip().lower()
    if mode_key not in {"factor", "funds", "performance"} or not rows:
        return (
            n_no, n_no, n_no, n_no, n_no,
            n_no, n_no, n_no, n_no,
            n_no, n_no, n_no, n_no, n_no,
            n_no, n_no, n_no,
            True,
            rows,
            rows,
            "Stage at least one row before importing.",
            False,
            "Select a series to preview option-adjusted results (first 6 rows).",
        )

    try:
        if mode_key == "factor":
            load_result = load_factor_series(MRD_ENGINE, rows)
        elif mode_key == "funds":
            load_result = load_fund_series(MRD_ENGINE, rows)
        else:
            load_result = load_performance_series(PERF_ENGINE, rows)
        new_df = load_result.returns_df
        if new_df.empty:
            raise ValueError("No rows returned for staged raw-data requests.")

        if existing_data:
            existing_cols = set(json_to_df(existing_data).columns)
            duplicates = [s for s in new_df.columns if s in existing_cols]
            if duplicates:
                return (
                    n_no, n_no, n_no, n_no, n_no,
                    n_no, n_no, n_no, n_no,
                    n_no, n_no, n_no, n_no, n_no,
                    n_no, n_no, n_no,
                    True,
                    rows,
                    rows,
                    f"Cannot add duplicate series: {', '.join(duplicates)}",
                    False,
                    n_no,
                )

        new_periodicity = load_result.periodicity
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

        updated_bench = dict(current_bench or {})
        updated_bench.update(load_result.benchmark_assignments or {})

        return (
            df_to_json(merged_df),
            combined_periodicity,
            periodicity_options,
            default_periodicity,
            False,
            updated_selection,
            f"Loaded {len(new_df.columns)} series with {len(new_df)} rows from raw database import",
            "green",
            False,
            default_periodicity,
            True,
            updated_bench,
            current_ls or {},
            current_order or [],
            True if first_load is not None else True,
            [],
            current_vol_scaling or {},
            False,
            [],
            [],
            no_update,
            True,
            "Select a series to preview option-adjusted results (first 6 rows).",
        )
    except Exception as e:
        return (
            n_no, n_no, n_no, n_no, n_no,
            n_no, n_no, n_no, n_no,
            n_no, n_no, n_no, n_no, n_no,
            n_no, n_no, n_no,
            True,
            rows,
            rows,
            f"Error loading raw database series: {str(e)}",
            False,
            n_no,
        )


@callback(
    Output("dashmat-raw-data-store", "data", allow_duplicate=True),
    Output("dashmat-original-periodicity-store", "data", allow_duplicate=True),
    Output("at-periodicity-select", "data", allow_duplicate=True),
    Output("at-periodicity-select", "value", allow_duplicate=True),
    Output("at-periodicity-select", "disabled", allow_duplicate=True),
    Output("at-temp-series-select", "data", allow_duplicate=True),
    Output("at-alert-message", "children", allow_duplicate=True),
    Output("at-alert-message", "color", allow_duplicate=True),
    Output("at-alert-message", "hide", allow_duplicate=True),
    Output("at-periodicity-value-store", "data", allow_duplicate=True),
    Output("at-series-selection-modal", "opened", allow_duplicate=True),
    Output("at-temp-benchmark-assignments-store", "data", allow_duplicate=True),
    Output("at-temp-long-short-store", "data", allow_duplicate=True),
    Output("at-temp-series-order-store", "data", allow_duplicate=True),
    Output("at-first-load-store", "data", allow_duplicate=True),
    Output("at-temp-deleted-series-store", "data", allow_duplicate=True),
    Output("at-temp-vol-scaling-assignments-store", "data", allow_duplicate=True),
    Output("at-portfolio-add-modal", "opened", allow_duplicate=True),
    Output("at-portfolio-add-rows-store", "data", allow_duplicate=True),
    Output("at-portfolio-add-grid", "rowData", allow_duplicate=True),
    Output("at-portfolio-add-error-alert", "children", allow_duplicate=True),
    Output("at-portfolio-add-error-alert", "hide", allow_duplicate=True),
    Input("at-portfolio-add-ok-button", "n_clicks"),
    State("at-portfolio-add-mode-store", "data"),
    State("at-portfolio-add-rows-store", "data"),
    State("dashmat-raw-data-store", "data"),
    State("dashmat-original-periodicity-store", "data"),
    State("at-series-select", "data"),
    State("at-benchmark-assignments-store", "data"),
    State("at-long-short-store", "data"),
    State("at-series-order-store", "data"),
    State("at-first-load-store", "data"),
    State("at-vol-scaling-assignments-store", "data"),
    prevent_initial_call=True,
)
def at_add_portfolios_from_database(
    n_clicks,
    mode,
    staged_rows,
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
    rows = [r for r in (staged_rows or []) if isinstance(r, dict)]
    if mode not in {"peer", "index", "other"} or not rows:
        return (
            n_no, n_no, n_no, n_no, n_no,
            n_no,
            "Stage at least one portfolio row before importing.",
            "orange",
            False,
            n_no, n_no, n_no, n_no, n_no,
            n_no, n_no, n_no,
            True,
            rows,
            rows,
            "Stage at least one portfolio row before importing.",
            False,
        )

    try:
        load_result = load_portfolio_series(
            DB_ENGINE,
            mode,
            rows,
            performance_engine=PERF_ENGINE,
        )
        new_df = load_result.returns_df
        if new_df.empty:
            raise ValueError("No rows returned for staged portfolio requests.")

        if existing_data:
            existing_cols = set(json_to_df(existing_data).columns)
            duplicates = [s for s in new_df.columns if s in existing_cols]
            if duplicates:
                return (
                    n_no, n_no, n_no, n_no, n_no,
                    n_no,
                    f"Cannot add duplicate series: {', '.join(duplicates)}",
                    "red",
                    False,
                    n_no, n_no, n_no, n_no, n_no,
                    n_no, n_no, n_no,
                    True,
                    rows,
                    rows,
                    f"Cannot add duplicate series: {', '.join(duplicates)}",
                    False,
                )

        new_periodicity = load_result.periodicity or "monthly"
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
            merged_df = _normalize_monthly_df_if_needed(new_df, new_periodicity)
            combined_periodicity = new_periodicity

        periodicity_options = get_available_periodicities(combined_periodicity)
        default_periodicity = "daily_trading" if combined_periodicity == "daily" else combined_periodicity
        new_series = [col for col in new_df.columns if col not in (current_selection or [])]
        updated_selection = (current_selection or []) + new_series

        updated_bench = dict(current_bench or {})
        updated_bench.update(load_result.benchmark_assignments or {})

        return (
            df_to_json(merged_df),
            combined_periodicity,
            periodicity_options,
            default_periodicity,
            False,
            updated_selection,
            f"Loaded {len(new_df.columns)} series with {len(new_df)} rows from {mode} portfolios.",
            "green",
            False,
            default_periodicity,
            True,
            updated_bench,
            current_ls or {},
            current_order or [],
            True,
            [],
            current_vol_scaling or {},
            False,
            [],
            [],
            no_update,
            True,
        )
    except Exception as e:
        return (
            n_no, n_no, n_no, n_no, n_no,
            n_no,
            f"Error loading portfolio series: {str(e)}",
            "red",
            False,
            n_no, n_no, n_no, n_no, n_no,
            n_no, n_no, n_no,
            True,
            rows,
            rows,
            f"Error loading portfolio series: {str(e)}",
            False,
        )


@callback(
    Output("dashmat-raw-data-store", "data"),
    Output("dashmat-original-periodicity-store", "data"),
    Output("at-periodicity-select", "data"),
    Output("at-periodicity-select", "value"),
    Output("at-periodicity-select", "disabled"),
    Output("at-temp-series-select", "data", allow_duplicate=True),
    Output("at-alert-message", "children"),
    Output("at-alert-message", "color"),
    Output("at-alert-message", "hide"),
    Output("at-periodicity-value-store", "data", allow_duplicate=True),
    Output("at-series-selection-modal", "opened", allow_duplicate=True),
    Output("at-temp-benchmark-assignments-store", "data", allow_duplicate=True),
    Output("at-temp-long-short-store", "data", allow_duplicate=True),
    Output("at-temp-series-order-store", "data", allow_duplicate=True),
    Output("at-first-load-store", "data"),
    Output("at-temp-deleted-series-store", "data", allow_duplicate=True),
    Output("at-temp-vol-scaling-assignments-store", "data", allow_duplicate=True),
    Output("at-ui-blocker-store", "data", allow_duplicate=True),
    # Sheet-select modal outputs
    Output("at-sheet-select-modal", "opened", allow_duplicate=True),
    Output("at-sheet-select-dropdown", "data", allow_duplicate=True),
    Output("at-sheet-select-dropdown", "value", allow_duplicate=True),
    Output("at-sheet-select-contents-store", "data", allow_duplicate=True),
    Output("at-sheet-select-filename-store", "data", allow_duplicate=True),
    Output("at-sheet-select-sheetnames-store", "data", allow_duplicate=True),
    Input("at-upload-data", "contents"),
    State("at-upload-data", "filename"),
    State("dashmat-raw-data-store", "data"),
    State("dashmat-original-periodicity-store", "data"),
    State("at-series-select", "data"),
    State("at-benchmark-assignments-store", "data"),
    State("at-long-short-store", "data"),
    State("at-series-order-store", "data"),
    State("at-first-load-store", "data"),
    State("at-vol-scaling-assignments-store", "data"),
    prevent_initial_call=True,
)
def handle_upload(contents, filename, existing_data, existing_periodicity, current_selection, current_bench, current_ls, current_order, first_load, current_vol_scaling):
    """Handle file upload, parse data, and update stores."""
    if contents is None:
        raise PreventUpdate

    n_no = no_update
    # Sheet-select outputs default to no_update
    sheet_no = (n_no, n_no, n_no, n_no, n_no, n_no)

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
                False,  # hide blocker
                True, dropdown_data, [sheet_names[0]], contents, filename, sheet_names,  # open sheet modal
            )

        # Parse and merge upload
        new_df = _shared_import_single_upload(contents, filename)
        merge_result = _shared_merge_uploaded_with_existing(existing_data, existing_periodicity, new_df)
        merged_df = merge_result.merged_df
        combined_periodicity = merge_result.combined_periodicity
        periodicity_options = merge_result.periodicity_options
        default_periodicity = merge_result.default_periodicity
        imported_df = merge_result.imported_df

        # Keep current selection and add new series
        new_series = [col for col in imported_df.columns if col not in (current_selection or [])]
        updated_selection = (current_selection or []) + new_series

        # Determine alert state
        if not first_load:
            alert_msg = f"Loaded {len(imported_df.columns)} series with {len(imported_df)} rows from {filename}"
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
            False, # Hide blocker
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
            False, # Hide blocker
            *sheet_no,
        )


# ---------------------------------------------------------------------------
# Sheet selection modal: confirm
# ---------------------------------------------------------------------------
@callback(
    Output("dashmat-raw-data-store", "data", allow_duplicate=True),
    Output("dashmat-original-periodicity-store", "data", allow_duplicate=True),
    Output("at-periodicity-select", "data", allow_duplicate=True),
    Output("at-periodicity-select", "value", allow_duplicate=True),
    Output("at-periodicity-select", "disabled", allow_duplicate=True),
    Output("at-temp-series-select", "data", allow_duplicate=True),
    Output("at-alert-message", "children", allow_duplicate=True),
    Output("at-alert-message", "color", allow_duplicate=True),
    Output("at-alert-message", "hide", allow_duplicate=True),
    Output("at-periodicity-value-store", "data", allow_duplicate=True),
    Output("at-series-selection-modal", "opened", allow_duplicate=True),
    Output("at-temp-benchmark-assignments-store", "data", allow_duplicate=True),
    Output("at-temp-long-short-store", "data", allow_duplicate=True),
    Output("at-temp-series-order-store", "data", allow_duplicate=True),
    Output("at-first-load-store", "data", allow_duplicate=True),
    Output("at-temp-deleted-series-store", "data", allow_duplicate=True),
    Output("at-temp-vol-scaling-assignments-store", "data", allow_duplicate=True),
    Output("at-ui-blocker-store", "data", allow_duplicate=True),
    Output("at-sheet-select-modal", "opened", allow_duplicate=True),
    Output("at-sheet-select-contents-store", "data", allow_duplicate=True),
    Output("at-sheet-select-filename-store", "data", allow_duplicate=True),
    Output("at-sheet-select-sheetnames-store", "data", allow_duplicate=True),
    Output("at-upload-data", "contents", allow_duplicate=True),
    Input("at-sheet-select-ok-button", "n_clicks"),
    Input("at-sheet-select-import-all-button", "n_clicks"),
    State("at-sheet-select-dropdown", "value"),
    State("at-sheet-select-contents-store", "data"),
    State("at-sheet-select-filename-store", "data"),
    State("at-sheet-select-sheetnames-store", "data"),
    State("dashmat-raw-data-store", "data"),
    State("dashmat-original-periodicity-store", "data"),
    State("at-series-select", "data"),
    State("at-benchmark-assignments-store", "data"),
    State("at-long-short-store", "data"),
    State("at-series-order-store", "data"),
    State("at-first-load-store", "data"),
    State("at-vol-scaling-assignments-store", "data"),
    prevent_initial_call=True,
)
def on_sheet_select_ok(n_clicks_selected, n_clicks_all, selected_sheets, stashed_contents, stashed_filename, stashed_sheet_names,
                       existing_data, existing_periodicity, current_selection,
                       current_bench, current_ls, current_order, first_load, current_vol_scaling):
    """Parse selected sheet(s) and complete the import."""
    if not stashed_contents:
        raise PreventUpdate

    n_no = no_update
    triggered_id = callback_context.triggered_id
    if triggered_id not in {"at-sheet-select-ok-button", "at-sheet-select-import-all-button"}:
        raise PreventUpdate

    try:
        workbook_sheets = stashed_sheet_names or get_sheet_names(stashed_contents, stashed_filename)
        if triggered_id == "at-sheet-select-import-all-button":
            target_sheets = workbook_sheets
        else:
            target_sheets = selected_sheets or []

        if not target_sheets:
            return (
                n_no, n_no, n_no, n_no, n_no,
                n_no,
                "Select at least one sheet to import.",
                "red",
                False,
                n_no, n_no, n_no, n_no, n_no,
                n_no, n_no, n_no,
                False,  # Hide blocker
                True, stashed_contents, stashed_filename, workbook_sheets, n_no,  # keep modal open and stash
            )

        new_df, imported_sheets = _import_selected_workbook_sheets(
            stashed_contents,
            stashed_filename,
            target_sheets,
            workbook_sheets=workbook_sheets,
        )
        merge_result = _shared_merge_uploaded_with_existing(existing_data, existing_periodicity, new_df)
        merged_df = merge_result.merged_df
        combined_periodicity = merge_result.combined_periodicity
        periodicity_options = merge_result.periodicity_options
        default_periodicity = merge_result.default_periodicity
        imported_df = merge_result.imported_df
        filename = stashed_filename

        new_series = [col for col in imported_df.columns if col not in (current_selection or [])]
        updated_selection = (current_selection or []) + new_series

        if not first_load:
            if len(imported_sheets) == 1:
                sheet_msg = f"sheet: {imported_sheets[0]}"
            else:
                sheet_msg = f"{len(imported_sheets)} sheets"
            alert_msg = (
                f"Loaded {len(imported_df.columns)} series with {len(imported_df)} rows "
                f"from {filename} ({sheet_msg})"
            )
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
            False,  # Hide blocker
            False, None, None, None, None,  # Close sheet modal, clear stash, reset upload
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
            False,  # Hide blocker
            False, None, None, None, None,  # Close sheet modal, clear stash, reset upload
        )


@callback(
    Output("at-sheet-select-ok-button", "disabled"),
    Input("at-sheet-select-dropdown", "value"),
)
def toggle_sheet_select_import_selected_disabled(selected_sheets):
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
    Output("at-ui-blocker-store", "data", allow_duplicate=True),
    Input("at-sheet-select-ok-button", "n_clicks"),
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
    Output("at-ui-blocker-store", "data", allow_duplicate=True),
    Input("at-sheet-select-import-all-button", "n_clicks"),
    prevent_initial_call=True,
)


# ---------------------------------------------------------------------------
# Sheet selection modal: cancel
# ---------------------------------------------------------------------------
@callback(
    Output("at-sheet-select-modal", "opened", allow_duplicate=True),
    Output("at-sheet-select-contents-store", "data", allow_duplicate=True),
    Output("at-sheet-select-filename-store", "data", allow_duplicate=True),
    Output("at-sheet-select-sheetnames-store", "data", allow_duplicate=True),
    Output("at-upload-data", "contents", allow_duplicate=True),
    Output("at-ui-blocker-store", "data", allow_duplicate=True),
    Input("at-sheet-select-cancel-button", "n_clicks"),
    prevent_initial_call=True,
)
def on_sheet_select_cancel(n_clicks):
    """Cancel sheet selection and clear stashed data."""
    if not n_clicks:
        raise PreventUpdate
    return False, None, None, None, None, False


# Clear the file input so the same file can be re-uploaded
clientside_callback(
    """
    function(opened) {
        if (!opened) {
            var el = document.getElementById('at-upload-data');
            if (el) {
                var inp = el.querySelector('input[type="file"]');
                if (inp) inp.value = '';
            }
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("at-sheet-select-modal", "title", allow_duplicate=True),
    Input("at-sheet-select-modal", "opened"),
    prevent_initial_call=True,
)


@callback(
    Output("at-series-selection-container", "children"),
    Output("at-temp-series-order-store", "data", allow_duplicate=True),
    Input("dashmat-raw-data-store", "data"),
    Input("at-temp-series-select", "data"),
    Input("at-temp-series-order-store", "data"),
    Input("at-temp-deleted-series-store", "data"),
    Input("at-series-selection-grid", "cellValueChanged", allow_optional=True),
    Input("at-temp-benchmark-assignments-store", "data"),
    Input("at-temp-long-short-store", "data"),
    Input("at-temp-vol-scaling-assignments-store", "data"),
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
        id="at-series-selection-grid",
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
    Output("at-temp-deleted-series-store", "data", allow_duplicate=True),
    Input("at-series-selection-grid", "cellValueChanged", allow_optional=True),
    State("at-series-selection-grid", "rowData", allow_optional=True),
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
    Output("dashmat-raw-data-store", "data", allow_duplicate=True),
    Output("at-temp-benchmark-assignments-store", "data", allow_duplicate=True),
    Output("at-temp-long-short-store", "data", allow_duplicate=True),
    Output("at-temp-vol-scaling-assignments-store", "data", allow_duplicate=True),
    Output("at-temp-series-select", "data", allow_duplicate=True),
    Output("at-temp-series-order-store", "data", allow_duplicate=True),
    Output("at-series-edit-mode", "data", allow_duplicate=True),
    Output("at-series-select-value-store", "data", allow_duplicate=True),
    Output("at-edit-box-focus-trigger", "data", allow_duplicate=True),
    Input("at-series-selection-grid", "cellValueChanged", allow_optional=True),
    State("dashmat-raw-data-store", "data"),
    State("at-temp-benchmark-assignments-store", "data"),
    State("at-temp-long-short-store", "data"),
    State("at-temp-vol-scaling-assignments-store", "data"),
    State("at-temp-series-select", "data"),
    State("at-temp-series-order-store", "data"),
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
    Output("at-temp-benchmark-assignments-store", "data"),
    Input("at-series-selection-grid", "cellValueChanged", allow_optional=True),
    State("at-series-selection-grid", "rowData", allow_optional=True),
    State("dashmat-raw-data-store", "data"),
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
    Output("at-temp-long-short-store", "data"),
    Input("at-series-selection-grid", "cellValueChanged", allow_optional=True),
    State("at-series-selection-grid", "rowData", allow_optional=True),
    State("dashmat-raw-data-store", "data"),
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
    Output("at-temp-vol-scaling-assignments-store", "data"),
    Input("at-series-selection-grid", "cellValueChanged", allow_optional=True),
    State("at-series-selection-grid", "rowData", allow_optional=True),
    State("dashmat-raw-data-store", "data"),
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
    Output("at-start-date-picker", "value"),
    Output("at-end-date-picker", "value"),
    Output("at-date-picker-wrapper", "style"),
    Output("at-common-range-button", "disabled"),
    Output("at-common-daily-button", "disabled"),
    Output("at-maximum-range-button", "disabled"),
    Output("at-date-range-store", "data", allow_duplicate=True),
    Output("at-state-ready-store", "data", allow_duplicate=True),
    Input("dashmat-raw-data-store", "data"),
    Input("at-periodicity-select", "value"),
    Input("at-series-select", "data"),
    State("at-date-range-store", "data"),
    State("at-start-date-picker", "value"),
    State("at-end-date-picker", "value"),
    prevent_initial_call="initial_duplicate",
)
def initialize_date_range(raw_data, periodicity, selected_series, stored_range, current_start_date, current_end_date):
    """Initialize date range to maximum range when data is loaded."""
    disabled_style = {"display": "flex", "opacity": 0.5, "pointerEvents": "none", "alignItems": "flex-start"}
    enabled_style = {"display": "flex", "alignItems": "flex-start"}

    if raw_data is None or not selected_series:
        return None, None, disabled_style, True, True, True, None, False

    try:
        candidates = compute_date_range_candidates(
            raw_data,
            periodicity or "daily",
            tuple(selected_series or ()),
        )
        if not candidates.get("available_series"):
            return None, None, disabled_style, True, True, True, None, False

        start_date, end_date = resolve_initial_range(candidates, stored_range)
        if not start_date or not end_date:
            return None, None, disabled_style, True, True, True, None, False

        has_common_daily = bool(candidates.get("common_daily_start") and candidates.get("common_daily_end"))
        next_range = {"start": start_date, "end": end_date}
        start_output = start_date
        end_output = end_date
        if current_start_date == start_date:
            start_output = no_update
        if current_end_date == end_date:
            end_output = no_update
        range_output = (
            no_update
            if _has_complete_date_range(stored_range)
            and stored_range.get("start") == start_date
            and stored_range.get("end") == end_date
            else next_range
        )
        return (
            start_output,
            end_output,
            enabled_style,
            False,
            not has_common_daily,
            False,
            range_output,
            True,
        )

    except Exception:
        return None, None, disabled_style, True, True, True, None, False


@callback(
    Output("at-start-date-picker", "value", allow_duplicate=True),
    Output("at-end-date-picker", "value", allow_duplicate=True),
    Output("at-date-range-store", "data"),
    Output("at-periodicity-select", "value", allow_duplicate=True),
    Output("at-periodicity-value-store", "data", allow_duplicate=True),
    Input("at-common-range-button", "n_clicks"),
    Input("at-common-daily-button", "n_clicks"),
    Input("at-maximum-range-button", "n_clicks"),
    State("dashmat-raw-data-store", "data"),
    State("at-periodicity-select", "value"),
    State("at-series-select", "data"),
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
        candidates = compute_date_range_candidates(
            raw_data,
            periodicity or "daily",
            tuple(selected_series or ()),
        )
        if not candidates.get("available_series"):
            raise PreventUpdate

        start_date, end_date, force_daily = resolve_button_range(candidates, button_id)
        if not start_date or not end_date:
            raise PreventUpdate

        periodicity_value = "daily_trading" if force_daily else no_update

        date_range = {"start": start_date, "end": end_date}
        return start_date, end_date, date_range, periodicity_value, periodicity_value

    except Exception:
        raise PreventUpdate


@callback(
    Output("at-date-range-store", "data", allow_duplicate=True),
    Input("at-start-date-picker", "value"),
    Input("at-end-date-picker", "value"),
    State("at-date-range-store", "data"),
    prevent_initial_call=True,
)
def update_date_range_store(start_date, end_date, existing_range):
    """Store date range when user manually changes dates."""
    if start_date and end_date:
        next_range = {"start": start_date, "end": end_date}
        if _has_complete_date_range(existing_range):
            if existing_range.get("start") == start_date and existing_range.get("end") == end_date:
                return no_update
        return next_range
    return no_update


@callback(
    Output("at-returns-grid", "columnDefs"),
    Output("at-returns-grid", "rowData"),
    Input("dashmat-raw-data-store", "data"),
    Input("at-periodicity-select", "value"),
    Input("at-series-select", "data"),
    Input("at-returns-type-select", "value"),
    Input("at-benchmark-assignments-store", "data"),
    Input("at-long-short-store", "data"),
    Input("at-date-range-store", "data"),
    Input("at-state-ready-store", "data"),
    Input("at-vol-scaler-value-store", "data"),
    Input("at-vol-scaling-assignments-store", "data"),
    prevent_initial_call=True,
)
def update_grid(raw_data, periodicity, selected_series, returns_type, benchmark_assignments, long_short_assignments, date_range, state_ready, vol_scaler, vol_scaling_assignments):
    """Update the AG Grid based on selections (optimized with caching)."""
    if not state_ready or not _has_complete_date_range(date_range):
        raise PreventUpdate

    if raw_data is None or not selected_series:
        return [], []

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

        return column_defs, row_data

    except Exception:
        return [], []


@callback(
    Output("at-menu-download-excel", "disabled"),
    Input("dashmat-raw-data-store", "data"),
    Input("at-series-select", "data"),
    Input("at-date-range-store", "data"),
    Input("at-state-ready-store", "data"),
)
def update_download_excel_disabled(raw_data, selected_series, date_range, state_ready):
    if not raw_data:
        return True
    if not selected_series:
        return True
    if not state_ready:
        return True
    return not _has_complete_date_range(date_range)


@callback(
    Output("at-statistics-loaded-store", "data"),
    Input("at-state-ready-store", "data"),
    prevent_initial_call=True,
)
def reset_statistics_loaded_on_hydration(state_ready):
    if state_ready:
        raise PreventUpdate
    return False


@callback(
    Output("at-loading-statistics", "display"),
    Input("at-main-tabs", "value"),
    Input("at-state-ready-store", "data"),
    Input("at-statistics-loaded-store", "data"),
)
def control_statistics_loading_display(active_tab, state_ready, statistics_loaded):
    if active_tab == "statistics" and (not state_ready or not statistics_loaded):
        return "show"
    return "auto"


@callback(
    Output("at-rolling-grid", "columnDefs"),
    Output("at-rolling-grid", "rowData"),
    Input("at-main-tabs", "value"),
    Input("at-rolling-chart-switch", "value"),
    Input("dashmat-raw-data-store", "data"),
    Input("at-periodicity-select", "value"),
    Input("at-series-select", "data"),
    Input("at-rolling-window-select", "value"),
    Input("at-rolling-return-type-select", "value"),
    Input("at-rolling-metric-select", "value"),
    Input("at-benchmark-assignments-store", "data"),
    Input("at-long-short-store", "data"),
    Input("at-date-range-store", "data"),
    Input("at-state-ready-store", "data"),
    Input("at-vol-scaler-value-store", "data"),
    Input("at-vol-scaling-assignments-store", "data"),
    prevent_initial_call=True,
)
def update_rolling_grid(active_tab, chart_checked, raw_data, periodicity, selected_series, rolling_window, rolling_return_type, rolling_metric, benchmark_assignments, long_short_assignments, date_range, state_ready, vol_scaler, vol_scaling_assignments):
    """Update the Rolling Returns grid with rolling window calculations."""
    # Lazy loading: only calculate when rolling tab/table view is active and ready.
    if active_tab != "rolling" or chart_checked != "table" or not state_ready or not _has_complete_date_range(date_range):
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
    Output("at-rolling-chart-wrapper", "children"),
    Input("at-main-tabs", "value"),
    Input("at-rolling-chart-switch", "value"),
    Input("dashmat-raw-data-store", "data"),
    Input("at-periodicity-select", "value"),
    Input("at-series-select", "data"),
    Input("at-rolling-window-select", "value"),
    Input("at-rolling-return-type-select", "value"),
    Input("at-rolling-metric-select", "value"),
    Input("at-benchmark-assignments-store", "data"),
    Input("at-long-short-store", "data"),
    Input("at-date-range-store", "data"),
    Input("at-state-ready-store", "data"),
    Input("at-vol-scaler-value-store", "data"),
    Input("at-vol-scaling-assignments-store", "data"),
    State("global-color-scheme-toggle", "computedColorScheme"),
    prevent_initial_call=True,
)
def update_rolling_chart(active_tab, chart_checked, raw_data, periodicity, selected_series, rolling_window, rolling_return_type, rolling_metric, benchmark_assignments, long_short_assignments, date_range, state_ready, vol_scaler, vol_scaling_assignments, theme):
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

    # Lazy loading: only calculate when rolling tab/chart view is active and ready.
    if active_tab != "rolling" or chart_checked != "chart" or not state_ready or not _has_complete_date_range(date_range):
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
    Output("at-monthly-series-select", "disabled"),
    Output("at-monthly-series-select", "data"),
    Output("at-monthly-series-select", "value", allow_duplicate=True),
    Input("at-monthly-view-checkbox", "value"),
    Input("at-series-select", "data"),
    State("at-monthly-series-store", "data"),
    State("at-monthly-series-select", "value"),
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
    if triggered_id == "at-monthly-view-checkbox":
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
    Output("at-calendar-grid", "columnDefs"),
    Output("at-calendar-grid", "rowData"),
    Input("at-main-tabs", "value"),
    Input("dashmat-raw-data-store", "data"),
    Input("dashmat-original-periodicity-store", "data"),
    Input("at-periodicity-select", "value"),
    Input("at-series-select", "data"),
    Input("at-returns-type-select", "value"),
    Input("at-benchmark-assignments-store", "data"),
    Input("at-long-short-store", "data"),
    Input("at-date-range-store", "data"),
    Input("at-state-ready-store", "data"),
    Input("at-monthly-view-checkbox", "value"),
    Input("at-monthly-series-select", "value"),
    Input("at-vol-scaler-value-store", "data"),
    Input("at-vol-scaling-assignments-store", "data"),
    prevent_initial_call=True,
)
def update_calendar_grid(active_tab, raw_data, original_periodicity, selected_periodicity, selected_series, returns_type, benchmark_assignments, long_short_assignments, date_range, state_ready, monthly_view, monthly_series, vol_scaler, vol_scaling_assignments):
    """Update the Calendar Year Returns grid (lazy loaded)."""
    # Lazy loading: only calculate when calendar tab is active
    if active_tab != "calendar" or not state_ready or not _has_complete_date_range(date_range):
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
    Output("at-statistics-grid", "columnDefs"),
    Output("at-statistics-grid", "rowData"),
    Output("at-statistics-loaded-store", "data", allow_duplicate=True),
    Input("dashmat-raw-data-store", "data"),
    Input("at-periodicity-select", "value"),
    Input("at-series-select", "data"),
    Input("at-benchmark-assignments-store", "data"),
    Input("at-long-short-store", "data"),
    Input("at-date-range-store", "data"),
    Input("at-state-ready-store", "data"),
    Input("at-vol-scaler-value-store", "data"),
    Input("at-vol-scaling-assignments-store", "data"),
    Input("dashmat-saved-series-cache-store", "data"),
    prevent_initial_call=True,
)
def update_statistics(raw_data, periodicity, selected_series, benchmark_assignments, long_short_assignments, date_range, state_ready, vol_scaler, vol_scaling_assignments, saved_series_store):
    """Update the Statistics grid with transposed data (optimized with caching)."""
    if not state_ready or not _has_complete_date_range(date_range):
        raise PreventUpdate

    if raw_data is None or not selected_series:
        return [], [], True

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
            return [], [], True

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
            
        return column_defs, row_data, True

    except Exception:
        return [], [], True


@callback(
    Output("at-correlogram-meta-store", "data"),
    Input("at-series-select", "data"),
    Input("at-main-tabs", "value"),
)
def update_correlogram_meta(selected_series, active_tab):
    """Update correlogram metadata (num_series) when tab is active."""
    if active_tab != "correlogram" or not selected_series:
        return no_update
    return {"num_series": len(selected_series)}


@callback(
    Output("at-correlogram-target-key-store", "data"),
    Input("at-main-tabs", "value"),
    Input("dashmat-raw-data-store", "data"),
    Input("at-periodicity-select", "value"),
    Input("at-series-select", "data"),
    Input("at-returns-type-select", "value"),
    Input("at-benchmark-assignments-store", "data"),
    Input("at-long-short-store", "data"),
    Input("at-date-range-store", "data"),
    Input("at-state-ready-store", "data"),
    Input("at-vol-scaler-value-store", "data"),
    Input("at-vol-scaling-assignments-store", "data"),
    Input("at-correlation-view-switch", "value"),
    Input("at-correlogram-block-width", "value"),
    State("at-correlogram-target-key-store", "data"),
    prevent_initial_call=True,
)
def update_correlogram_target_key(
    active_tab,
    raw_data,
    periodicity,
    selected_series,
    returns_type,
    benchmark_assignments,
    long_short_assignments,
    date_range,
    state_ready,
    vol_scaler,
    vol_scaling_assignments,
    correlation_view,
    block_width,
    current_target_key,
):
    if active_tab != "correlogram":
        return no_update
    if not state_ready:
        return no_update

    effective_date_range = date_range
    try:
        candidates = compute_date_range_candidates(
            raw_data,
            periodicity or "daily",
            tuple(selected_series or ()),
        )
        start_date, end_date = resolve_initial_range(candidates, date_range)
        if start_date and end_date:
            effective_date_range = {"start": start_date, "end": end_date}
    except Exception:
        effective_date_range = date_range

    if not _has_complete_date_range(effective_date_range):
        return no_update

    next_key = _correlogram_request_key(
        raw_data,
        periodicity,
        tuple(selected_series or ()),
        returns_type,
        benchmark_assignments,
        long_short_assignments,
        effective_date_range,
        vol_scaler,
        vol_scaling_assignments,
        correlation_view,
        block_width,
    )
    if next_key == current_target_key:
        return no_update
    return next_key


@callback(
    Output("at-loading-correlogram", "display"),
    Input("at-main-tabs", "value"),
    Input("at-correlogram-target-key-store", "data"),
    Input("at-correlogram-rendered-key-store", "data"),
)
def control_correlogram_loading_display(active_tab, target_key, rendered_key):
    if active_tab != "correlogram":
        return "auto"
    if target_key and target_key != rendered_key:
        return "show"
    return "auto"


clientside_callback(
    """
    function(meta, currentValue) {
        if (currentValue !== null && currentValue !== undefined && currentValue !== "") {
            return dash_clientside.no_update;
        }
        if (!meta || !meta.num_series || meta.num_series <= 1) {
            return dash_clientside.no_update;
        }

        var container = document.getElementById('at-correlogram-container');
        var container_width = container ? container.clientWidth : 0;
        if (!container_width) {
            // Fallback for first render timing when container width is not measured yet.
            container_width = Math.max((window.innerWidth || 1200) - 260, 400);
        }

        // Default strategy: Clamp between 100 and 200, based on (Container - Buffer) / N
        // This ensures we fill the window if possible, but respect min 100px and max 200px defaults.
        var available_width = Math.max(container_width - 40, 200);
        var default_width = Math.floor(available_width / meta.num_series);

        if (default_width < 100) {
            default_width = 100;
        } else if (default_width > 200) {
            default_width = 200;
        }
        
        return default_width;
    }
    """,
    Output("at-correlogram-block-width", "value"),
    Input("at-correlogram-meta-store", "data"),
    State("at-correlogram-block-width", "value"),
)


@callback(
    Output("at-correlogram-container", "children"),
    Output("at-correlogram-rendered-key-store", "data", allow_duplicate=True),
    Input("at-correlogram-target-key-store", "data"),
    State("at-main-tabs", "value"),
    State("dashmat-raw-data-store", "data"),
    State("at-periodicity-select", "value"),
    State("at-series-select", "data"),
    State("at-returns-type-select", "value"),
    State("at-benchmark-assignments-store", "data"),
    State("at-long-short-store", "data"),
    State("at-date-range-store", "data"),
    State("at-state-ready-store", "data"),
    State("at-vol-scaler-value-store", "data"),
    State("at-vol-scaling-assignments-store", "data"),
    State("at-correlation-view-switch", "value"),
    State("at-correlogram-block-width", "value"),
    State("global-color-scheme-toggle", "computedColorScheme"),
    prevent_initial_call=True,
)
def update_correlogram(target_key, active_tab, raw_data, periodicity, selected_series, returns_type, benchmark_assignments, long_short_assignments, date_range, state_ready, vol_scaler, vol_scaling_assignments, correlation_view, block_width, theme):
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

    # Only generate when there is a fresh target key and correlogram is active/ready.
    if (
        not target_key
        or active_tab != "correlogram"
        or not state_ready
        or not _has_complete_date_range(date_range)
    ):
        raise PreventUpdate

    request_key = target_key

    if raw_data is None or not selected_series or len(selected_series) < 2:
        return empty_graph, request_key

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
            return empty_graph, request_key

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

            return dcc.Graph(figure=heatmap_fig, style={"height": "100%"}), request_key

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
            return dcc.Graph(figure=fig, style=graph_style), request_key

    except Exception:
        return empty_graph, request_key


@callback(
    Output("at-growth-charts-container", "children"),
    Input("at-main-tabs", "value"),
    Input("at-growth-chart-switch", "value"),
    Input("dashmat-raw-data-store", "data"),
    Input("at-periodicity-select", "value"),
    Input("at-series-select", "data"),
    Input("at-benchmark-assignments-store", "data"),
    Input("at-long-short-store", "data"),
    Input("at-date-range-store", "data"),
    Input("at-state-ready-store", "data"),
    Input("at-vol-scaler-value-store", "data"),
    Input("at-vol-scaling-assignments-store", "data"),
    State("global-color-scheme-toggle", "computedColorScheme"),
    prevent_initial_call=True,
)
def update_growth_charts(active_tab, chart_checked, raw_data, periodicity, selected_series, benchmark_assignments, long_short_assignments, date_range, state_ready, vol_scaler, vol_scaling_assignments, theme):
    """Update Growth of $1 charts (lazy loaded)."""
    # Lazy loading: only generate when growth tab is active and chart view is selected
    if (
        active_tab != "growth"
        or chart_checked != "chart"
        or not state_ready
        or not _has_complete_date_range(date_range)
    ):
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
    Output("at-growth-grid", "columnDefs"),
    Output("at-growth-grid", "rowData"),
    Input("at-main-tabs", "value"),
    Input("at-growth-chart-switch", "value"),
    Input("dashmat-raw-data-store", "data"),
    Input("at-periodicity-select", "value"),
    Input("at-series-select", "data"),
    Input("at-benchmark-assignments-store", "data"),
    Input("at-long-short-store", "data"),
    Input("at-date-range-store", "data"),
    Input("at-state-ready-store", "data"),
    Input("at-vol-scaler-value-store", "data"),
    Input("at-vol-scaling-assignments-store", "data"),
    prevent_initial_call=True,
)
def update_growth_grid(active_tab, chart_checked, raw_data, periodicity, selected_series, benchmark_assignments, long_short_assignments, date_range, state_ready, vol_scaler, vol_scaling_assignments):
    """Update Growth of $1 grid (lazy loaded)."""
    # Lazy loading: only generate when growth tab is active and table view is selected
    if (
        active_tab != "growth"
        or chart_checked != "table"
        or not state_ready
        or not _has_complete_date_range(date_range)
    ):
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
    Output("at-drawdown-charts", "children"),
    Input("at-main-tabs", "value"),
    Input("at-drawdown-chart-switch", "value"),
    Input("dashmat-raw-data-store", "data"),
    Input("at-periodicity-select", "value"),
    Input("at-series-select", "data"),
    Input("at-returns-type-select", "value"),
    Input("at-benchmark-assignments-store", "data"),
    Input("at-long-short-store", "data"),
    Input("at-date-range-store", "data"),
    Input("at-state-ready-store", "data"),
    Input("at-vol-scaler-value-store", "data"),
    Input("at-vol-scaling-assignments-store", "data"),
    State("global-color-scheme-toggle", "computedColorScheme"),
    prevent_initial_call=True,
)
def update_drawdown_charts(active_tab, chart_checked, raw_data, periodicity, selected_series, returns_type, benchmark_assignments, long_short_assignments, date_range, state_ready, vol_scaler, vol_scaling_assignments, theme):
    """Update Drawdown charts (lazy loaded)."""
    # Lazy loading: only generate when drawdown tab is active and chart view is selected
    if (
        active_tab != "drawdown"
        or chart_checked != "chart"
        or not state_ready
        or not _has_complete_date_range(date_range)
    ):
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
    Output("at-drawdown-grid", "columnDefs"),
    Output("at-drawdown-grid", "rowData"),
    Input("at-main-tabs", "value"),
    Input("at-drawdown-chart-switch", "value"),
    Input("dashmat-raw-data-store", "data"),
    Input("at-periodicity-select", "value"),
    Input("at-series-select", "data"),
    Input("at-returns-type-select", "value"),
    Input("at-benchmark-assignments-store", "data"),
    Input("at-long-short-store", "data"),
    Input("at-date-range-store", "data"),
    Input("at-state-ready-store", "data"),
    Input("at-vol-scaler-value-store", "data"),
    Input("at-vol-scaling-assignments-store", "data"),
    prevent_initial_call=True,
)
def update_drawdown_grid(active_tab, chart_checked, raw_data, periodicity, selected_series, returns_type, benchmark_assignments, long_short_assignments, date_range, state_ready, vol_scaler, vol_scaling_assignments):
    """Update Drawdown grid (lazy loaded)."""
    # Lazy loading: only generate when drawdown tab is active and table view is selected
    if (
        active_tab != "drawdown"
        or chart_checked != "table"
        or not state_ready
        or not _has_complete_date_range(date_range)
    ):
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
    Output("at-download-excel", "data"),
    Input("at-menu-download-excel", "n_clicks"),
    State("dashmat-raw-data-store", "data"),
    State("dashmat-original-periodicity-store", "data"),
    State("at-periodicity-select", "value"),
    State("at-series-select", "data"),
    State("at-returns-type-select", "value"),
    State("at-benchmark-assignments-store", "data"),
    State("at-long-short-store", "data"),
    State("at-date-range-store", "data"),
    State("at-rolling-window-store", "data"),
    State("at-rolling-return-type-store", "data"),
    State("at-monthly-view-store", "data"),
    State("at-monthly-series-store", "data"),
    State("at-vol-scaler-value-store", "data"),
    State("at-vol-scaling-assignments-store", "data"),
    State("dashmat-saved-series-cache-store", "data"),
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
    Output("at-download-sample-daily", "data"),
    Input("at-download-sample-daily-btn", "n_clicks"),
    prevent_initial_call=True,
)
def download_sample_daily(n_clicks):
    """Download stored sample daily returns file."""
    if n_clicks is None:
        raise PreventUpdate

    return dcc.send_file(str(get_sample_file_path("daily")))


@callback(
    Output("at-download-sample-monthly", "data"),
    Input("at-download-sample-monthly-btn", "n_clicks"),
    prevent_initial_call=True,
)
def download_sample_monthly(n_clicks):
    """Download stored sample monthly returns file."""
    if n_clicks is None:
        raise PreventUpdate

    return dcc.send_file(str(get_sample_file_path("monthly")))
