"""Portfolio Optimization page for DashMat."""

from dataclasses import dataclass
from io import BytesIO, StringIO
import json

import dash_ag_grid as dag
import dash_mantine_components as dmc
from dash_iconify import DashIconify
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sqlalchemy import text
from dash import (
    ClientsideFunction, Input, Output, State, callback, dcc, html, no_update,
    register_page, ALL, clientside_callback, callback_context,
)
from dash.exceptions import PreventUpdate

import cache_config
from utils.parsing import get_sheet_names
from utils.add_series_flow import import_selected_disabled
from utils.date_range_flow import (
    compute_common_daily_candidates,
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
    align_monthly_series_to_month_end,
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
    annualization_factor,
    is_daily,
)
from utils.covariance import (
    covariance_to_correlation,
    estimate_covariance_matrix,
    estimate_mean_vector,
    format_cov_shrinkage_spec_label,
    format_cov_shrinkage_target_label,
    resolve_cov_shrinkage_spec,
    VALID_COV_SHRINKAGE,
    VALID_COV_SHRINKAGE_TARGET,
)
from utils.exponential_weighting import decay_input_mode, normalize_decay_input, resolve_ewm_params
from utils.excel_export import format_excel_dates, format_mdy_date, write_excel_with_autofit
from utils.optimization import run_portfolio_optimization, compute_risk_contributions, compute_efficient_frontier
from utils.perf_timing import timed_block
from utils.serialization import canonical_json_dumps, date_range_payload_for_cache, mapping_payload_for_cache
from utils.shared_metrics import (
    STATS_CONFIG,
    risk_free_json_from_store as _risk_free_json_from_store,
    spx_json_from_store as _spx_json_from_store,
)
from utils.saved_series import save_series_to_raw_data, saved_series_store_names
from utils.statistics import (
    calculate_drawdown,
    calculate_statistics_cached,
    annualized_return,
    annualized_return_calendar_days,
)
from utils.charting import apply_chart_theme
from utils.help_links import PORTOPT_HELP_URL
from utils.core_categories import (
    clear_dropdown_caches,
    get_cma_versions_cached,
    get_cmabench_map_for_fofbench,
    get_unique_cmabench_values_cached,
    load_cma_returns_for_benches_with_meta,
)
from utils.dashmat_welcome_modal import (
    PagePrefixConfig,
    build_db_add_modal,
    build_portfolio_add_modal,
    build_raw_db_add_modal,
    build_series_selection_modal,
    build_sheet_select_modal,
    build_underlying_add_modal,
    build_welcome_screen as build_shared_welcome_screen,
    compute_close_db_add_modal,
    compute_close_underlying_add_modal,
    compute_close_portfolio_add_modal,
    compute_close_raw_db_add_modal,
    compute_open_db_add_modal,
    compute_open_underlying_add_modal,
    compute_open_portfolio_add_modal,
    compute_open_raw_db_add_modal,
    compute_sync_include_benchmark_enabled,
    compute_validate_db_add_selection,
    js_portfolio_add_row,
    js_portfolio_benchmark_toggle,
    js_portfolio_clear_rows,
    js_portfolio_delete_row,
    js_portfolio_ok_disabled,
    js_underlying_delete_row,
)
from utils.portfolio_series import load_portfolio_series
from utils.underlying_category_imports import (
    expand_underlying_category_rows,
    get_underlying_category_desc_options,
    load_underlying_category_series,
)
from dbengine import (
    AG_GRID_LICENSE_KEY,
    engine as DB_ENGINE,
    engine_MRD as MRD_ENGINE,
    engine_PERFORMANCE as PERF_ENGINE,
)
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

register_page(__name__, path="/portopt", name="Portfolio Optimization", title="Portfolio Optimization")

PO_WELCOME_MODAL_CONFIG = PagePrefixConfig(
    prefix="po",
    page_icon="grommet-icons:optimize",
    page_title="Portfolio Optimization",
    page_subtitle="Choose a source to load data and start optimization.",
    series_modal_size="84vw",
    series_modal_max_width="1450px",
    series_modal_transition_ms=200,
    welcome_switch_buttons=(
        ("welcome-view-analytics", "Switch to Analytics", "tabler:chart-line"),
        ("welcome-view-regression", "Switch to Regression", "tabler:chart-dots-3"),
    ),
)

_PO_MODEL_DEFAULT_NAME = {
    "risk_parity": "RP",
    "factor_risk_parity": "FRP",
    "hierarchical_risk_parity": "HRP",
    "hrp": "HRP",
    "maximize_sharpe": "MSR",
    "minimize_variance": "MinVar",
    "minimize_cvar": "MinCVaR",
    "equal_weight": "EW",
    "ex_ante_mv": "ExAnteMV",
    "black_litterman": "BL",
}

PO_TAB_SPECS = (
    {"value": "weight", "label": "Weights", "export_index": False},
    {"value": "attribution", "label": "Attribution", "export_index": False},
    {"value": "risk", "label": "Risk", "export_index": False},
    {"value": "turnover", "label": "Turnover", "export_index": False},
    {"value": "frontier", "label": "Frontier", "export_index": False},
    {"value": "statistics", "label": "Statistics", "export_index": False},
    {"value": "returns", "label": "Returns", "export_index": False},
    {"value": "rolling", "label": "Rolling", "export_index": True},
    {"value": "calendar", "label": "Calendar Year", "export_index": True},
    {"value": "growth", "label": "Growth of $1", "export_index": False},
    {"value": "drawdown", "label": "Drawdown", "export_index": True},
)


def _po_default_name_for_model(model: str) -> str:
    return _PO_MODEL_DEFAULT_NAME.get(model, "Port")


def _po_build_result_grid(
    component_id: str,
    column_defs: list[dict] | None,
    row_data: list[dict] | None,
    *,
    pagination: bool = False,
) -> dag.AgGrid:
    return dag.AgGrid(
        enableEnterpriseModules=True,
        licenseKey=AG_GRID_LICENSE_KEY,
        id=component_id,
        className="ag-theme-alpine",
        columnDefs=column_defs or [],
        rowData=row_data or [],
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
            "pagination": pagination,
            "suppressExcelExport": True,
            "enableRangeSelection": True,
            "suppressCsvExport": True,
        },
    )


def _po_build_help_control() -> dmc.Anchor | dmc.Button:
    help_button = dmc.Button(
        "Help",
        id="po-menu-help-guide",
        variant="gradient",
        gradient={"from": "teal", "to": "cyan", "deg": 90},
        size="sm",
        radius="xl",
        className="dashmat-menu-trigger",
        leftSection=DashIconify(icon="tabler:help-circle", width=14),
        disabled=not PORTOPT_HELP_URL.strip(),
    )
    if not PORTOPT_HELP_URL.strip():
        return help_button
    return dmc.Anchor(
        help_button,
        href=PORTOPT_HELP_URL.strip(),
        target="_blank",
        rel="noopener noreferrer",
        style={"textDecoration": "none"},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mapping_payload(value) -> str:
    return mapping_payload_for_cache(value)


def _date_range_payload(value) -> str:
    return date_range_payload_for_cache(value)


@dataclass(frozen=True)
class _PoWorkingReturnsBundle:
    raw_data: str
    periodicity: str
    benchmark_payload: str
    long_short_payload: str
    date_range_payload: str
    vol_scaler: float
    vol_scaling_payload: str


def _build_po_working_bundle(
    raw_data,
    periodicity,
    benchmark_assignments,
    long_short_assignments,
    date_range,
    vol_scaler,
    vol_scaling_assignments,
) -> _PoWorkingReturnsBundle:
    """Build canonicalized working-return inputs once per callback."""
    return _PoWorkingReturnsBundle(
        raw_data=raw_data,
        periodicity=periodicity or "daily",
        benchmark_payload=_mapping_payload(benchmark_assignments),
        long_short_payload=_mapping_payload(long_short_assignments),
        date_range_payload=_date_range_payload(date_range),
        vol_scaler=vol_scaler or 0,
        vol_scaling_payload=_mapping_payload(vol_scaling_assignments),
    )


def _po_get_working_returns(bundle: _PoWorkingReturnsBundle, selected_series) -> pd.DataFrame:
    series_tuple = tuple(selected_series or ())
    if not series_tuple or not bundle.raw_data:
        return pd.DataFrame()
    return get_working_returns(
        bundle.raw_data,
        bundle.periodicity,
        series_tuple,
        bundle.benchmark_payload,
        bundle.long_short_payload,
        bundle.date_range_payload,
        bundle.vol_scaler,
        bundle.vol_scaling_payload,
    )


def _po_collect_portfolio_returns(results, selected_portfolios=None) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    show = list(selected_portfolios or list(results.keys()))
    all_returns = {}
    for pname in show:
        pdata = (results or {}).get(pname)
        if not pdata:
            continue
        returns_json = pdata.get("returns_json")
        if not returns_json:
            continue
        try:
            s = pd.read_json(StringIO(returns_json), typ="series")
        except Exception:
            continue
        s.index = pd.to_datetime(s.index)
        all_returns[pname] = s
    if not all_returns:
        return pd.DataFrame()
    combined_df = pd.DataFrame(all_returns).sort_index()
    combined_df.index.name = "Date"
    return combined_df


def _po_single_portfolio_return(results, portfolio_name: str) -> pd.Series:
    if not results or not portfolio_name:
        return pd.Series(dtype=float)
    pdata = (results or {}).get(portfolio_name) or {}
    returns_json = pdata.get("returns_json")
    if not returns_json:
        return pd.Series(dtype=float)
    try:
        s = pd.read_json(StringIO(returns_json), typ="series")
    except Exception:
        return pd.Series(dtype=float)
    s.index = pd.to_datetime(s.index)
    return s.dropna().rename(portfolio_name)


@cache_config.cache.memoize(timeout=0)
def _po_build_display_series_cached(
    selected_portfolio,
    returns_json,
    config_payload,
    raw_data,
    periodicity,
    benchmark_payload,
    long_short_payload,
    date_range_payload,
    vol_scaler,
    vol_scaling_payload,
):
    if not selected_portfolio or not returns_json:
        return None, []

    series_map = {}
    try:
        portfolio_series = pd.read_json(StringIO(returns_json), typ="series")
        portfolio_series.index = pd.to_datetime(portfolio_series.index)
        portfolio_series = portfolio_series.dropna().rename(selected_portfolio)
    except Exception:
        portfolio_series = pd.Series(dtype=float)
    if not portfolio_series.empty:
        series_map[selected_portfolio] = portfolio_series

    try:
        config = json.loads(config_payload) if config_payload else {}
    except Exception:
        config = {}
    source_series = list(dict.fromkeys((config or {}).get("selected_series") or []))
    if raw_data and source_series:
        working_bundle = _PoWorkingReturnsBundle(
            raw_data=raw_data,
            periodicity=periodicity or "daily",
            benchmark_payload=benchmark_payload,
            long_short_payload=long_short_payload,
            date_range_payload=date_range_payload,
            vol_scaler=vol_scaler or 0,
            vol_scaling_payload=vol_scaling_payload,
        )
        try:
            working_df = _po_get_working_returns(working_bundle, source_series)
        except Exception:
            working_df = pd.DataFrame()
        for name in source_series:
            if not name or name == selected_portfolio or name not in working_df.columns:
                continue
            s = working_df[name].dropna()
            if not s.empty:
                series_map[name] = s.rename(name)

    if not series_map:
        return None, []

    display_df = pd.concat(series_map, axis=1).sort_index()
    ordered_cols = [c for c in series_map.keys() if c in display_df.columns]
    if not ordered_cols:
        return None, []
    display_df = display_df[ordered_cols]
    display_df.index.name = "Date"
    return df_to_json(display_df), ordered_cols


def _po_build_display_series(
    results,
    selected_portfolio,
    raw_data,
    periodicity,
    benchmark_assignments,
    long_short_assignments,
    date_range,
    vol_scaler,
    vol_scaling_assignments,
) -> tuple[pd.DataFrame, list[str]]:
    if not selected_portfolio or not results or selected_portfolio not in results:
        return pd.DataFrame(), []

    entry = (results or {}).get(selected_portfolio) or {}
    bundle = _build_po_working_bundle(
        raw_data,
        periodicity,
        benchmark_assignments,
        long_short_assignments,
        date_range,
        vol_scaler,
        vol_scaling_assignments,
    )
    display_json, ordered_cols = _po_build_display_series_cached(
        selected_portfolio,
        entry.get("returns_json"),
        canonical_json_dumps(entry.get("config") or {}),
        bundle.raw_data,
        bundle.periodicity,
        bundle.benchmark_payload,
        bundle.long_short_payload,
        bundle.date_range_payload,
        bundle.vol_scaler,
        bundle.vol_scaling_payload,
    )
    if not display_json or not ordered_cols:
        return pd.DataFrame(), []
    try:
        display_df = json_to_df(display_json)
    except Exception:
        return pd.DataFrame(), []
    return display_df, ordered_cols


def _po_missing_source_series(results, selected_portfolio, raw_data) -> list[str]:
    if not selected_portfolio or not results or selected_portfolio not in results:
        return []

    config = ((results or {}).get(selected_portfolio) or {}).get("config", {}) or {}
    source_series = [str(name) for name in (config.get("selected_series") or []) if str(name)]
    if not source_series:
        return []
    if not raw_data:
        return source_series

    try:
        columns = set(json_to_df(raw_data).columns)
    except Exception:
        return source_series
    return [name for name in source_series if name not in columns]


def _po_rolling_metric_label(metric: str) -> str:
    labels = {
        "total_return": "Total Return",
        "volatility": "Volatility",
        "sharpe_ratio": "Sharpe Ratio",
        "sortino_ratio": "Sortino Ratio",
    }
    return labels.get(metric or "total_return", "Total Return")


def _po_rolling_metric_tickformat(metric: str) -> str:
    return ".2%" if metric in {"total_return", "volatility"} else ".2f"


def _po_tab_render_ready(active_tab, expected_tab: str, initial_tab_ready) -> bool:
    return active_tab == expected_tab and bool(initial_tab_ready)


def _po_lazy_tab_render_ready(active_tab, expected_tab: str, tab_loaded) -> bool:
    return active_tab == expected_tab and bool(tab_loaded)


def _build_apply_weight_matrix(
    index: pd.DatetimeIndex,
    series_tuple: tuple[str, ...],
    window_weights,
) -> np.ndarray:
    """Build a dense per-period weight matrix using fast index slicing."""
    n_rows = len(index)
    n_cols = len(series_tuple)
    weight_values = np.zeros((n_rows, n_cols), dtype=float)
    if n_rows == 0 or n_cols == 0:
        return weight_values

    for ww in window_weights or ():
        weights = ww.get("weights", {})
        if not isinstance(weights, dict):
            continue

        start = pd.Timestamp(ww["apply_start"])
        end = pd.Timestamp(ww["apply_end"])
        start_idx = int(index.searchsorted(start, side="left"))
        end_idx = int(index.searchsorted(end, side="right"))
        if start_idx >= end_idx:
            continue

        row_weights = np.fromiter(
            (float(weights.get(name, 0.0) or 0.0) for name in series_tuple),
            dtype=float,
            count=n_cols,
        )
        weight_values[start_idx:end_idx, :] = row_weights

    return weight_values


def _compute_monthly_attribution(
    working_df: pd.DataFrame,
    selected_series,
    window_weights,
) -> pd.DataFrame:
    """Compute monthly attribution from per-window weights and component returns."""
    series_tuple = tuple(selected_series or ())
    if working_df.empty or not series_tuple or not window_weights:
        return pd.DataFrame()

    working_subset = working_df.loc[:, list(series_tuple)].fillna(0.0)
    weight_values = _build_apply_weight_matrix(working_subset.index, series_tuple, window_weights)
    has_weights = weight_values.sum(axis=1) > 0
    if not np.any(has_weights):
        return pd.DataFrame()

    attribution_values = weight_values[has_weights] * working_subset.to_numpy(copy=False)[has_weights]
    attribution_df = pd.DataFrame(
        attribution_values,
        index=working_subset.index[has_weights],
        columns=list(series_tuple),
    )
    return attribution_df.resample("ME").sum().dropna(how="all")


def _compute_window_risk_contributions(
    working_df: pd.DataFrame,
    selected_series,
    window_weights,
    config=None,
):
    """Compute risk-contribution rows for each optimization window."""
    series_tuple = tuple(selected_series or ())
    if working_df.empty or not series_tuple or not window_weights:
        return []

    working_subset = working_df.loc[:, list(series_tuple)]
    index = working_subset.index
    cfg = config or {}
    exp_wt_cov = bool(cfg.get("exp_wt_cov", False))
    decay_value = normalize_decay_input(cfg.get("halflife", 63), 63.0)
    cov_shrinkage, cov_shrinkage_target = resolve_cov_shrinkage_spec(
        cfg.get("cov_shrinkage", "none"),
        cfg.get("cov_shrinkage_target", "scaled_identity"),
        exp_weighted=exp_wt_cov,
    )
    rows = []
    for ww in window_weights:
        weights = ww.get("weights", {})
        if not isinstance(weights, dict):
            continue

        apply_start = pd.Timestamp(ww["apply_start"])
        apply_end = pd.Timestamp(ww["apply_end"])
        est_start = pd.Timestamp(ww.get("est_start", ww["apply_start"]))
        est_end = pd.Timestamp(ww.get("est_end", ww["apply_end"]))

        active_assets = [
            name for name in series_tuple
            if abs(float(weights.get(name, 0) or 0)) > 1e-12
        ]
        if not active_assets:
            continue

        start_idx = int(index.searchsorted(est_start, side="left"))
        end_idx = int(index.searchsorted(est_end, side="right"))
        if start_idx >= end_idx:
            continue

        window_returns = working_subset.iloc[start_idx:end_idx][active_assets].dropna(how="all")
        if window_returns.empty:
            continue

        valid_assets = [name for name in active_assets if window_returns[name].notna().any()]
        if not valid_assets:
            continue

        window_returns = window_returns[valid_assets].fillna(0)
        custom_cov = None
        if exp_wt_cov or cov_shrinkage != "none":
            custom_cov = estimate_covariance_matrix(
                window_returns,
                asset_order=valid_assets,
                exp_weighted=exp_wt_cov,
                decay_value=decay_value,
                shrinkage=cov_shrinkage,
                shrinkage_target=cov_shrinkage_target,
            )
        rc = compute_risk_contributions(
            {name: float(weights.get(name, 0) or 0) for name in valid_assets},
            window_returns,
            custom_cov=custom_cov,
        )
        rows.append(
            {
                "apply_start": apply_start,
                "apply_end": apply_end,
                "risk_contributions": rc,
            }
        )

    return rows


@cache_config.cache.memoize(timeout=0)
def _po_compute_monthly_attribution_cached(
    raw_data: str,
    periodicity: str,
    selected_series: tuple[str, ...],
    benchmark_payload: str,
    long_short_payload: str,
    date_range_payload: str,
    vol_scaler: float,
    vol_scaling_payload: str,
    window_weights_payload: str,
) -> str | None:
    if not raw_data or not selected_series or not window_weights_payload:
        return None
    try:
        window_weights = json.loads(window_weights_payload)
    except Exception:
        return None
    working_df = get_working_returns(
        raw_data,
        periodicity or "daily",
        tuple(selected_series),
        benchmark_payload,
        long_short_payload,
        date_range_payload,
        vol_scaler,
        vol_scaling_payload,
    )
    attribution_df = _compute_monthly_attribution(working_df, selected_series, window_weights)
    if attribution_df.empty:
        return None
    return df_to_json(attribution_df)


def _po_get_monthly_attribution(
    bundle: _PoWorkingReturnsBundle,
    selected_series,
    window_weights,
) -> pd.DataFrame:
    series_tuple = tuple(selected_series or ())
    payload = _po_compute_monthly_attribution_cached(
        bundle.raw_data,
        bundle.periodicity,
        series_tuple,
        bundle.benchmark_payload,
        bundle.long_short_payload,
        bundle.date_range_payload,
        bundle.vol_scaler,
        bundle.vol_scaling_payload,
        canonical_json_dumps(window_weights or []),
    )
    if not payload:
        return pd.DataFrame()
    try:
        return json_to_df(payload)
    except Exception:
        return pd.DataFrame()


def _serialize_risk_rows(rows) -> str:
    serialized = []
    for row in rows or ():
        serialized.append(
            {
                "apply_start": pd.Timestamp(row["apply_start"]).strftime("%Y-%m-%d"),
                "apply_end": pd.Timestamp(row["apply_end"]).strftime("%Y-%m-%d"),
                "risk_contributions": row.get("risk_contributions", {}) or {},
            }
        )
    return canonical_json_dumps(serialized)


def _deserialize_risk_rows(payload: str):
    if not payload:
        return []
    try:
        rows = json.loads(payload)
    except Exception:
        return []
    hydrated = []
    for row in rows or ():
        hydrated.append(
            {
                "apply_start": pd.Timestamp(row.get("apply_start")),
                "apply_end": pd.Timestamp(row.get("apply_end")),
                "risk_contributions": row.get("risk_contributions", {}) or {},
            }
        )
    return hydrated


@cache_config.cache.memoize(timeout=0)
def _po_compute_window_risk_contributions_cached(
    raw_data: str,
    periodicity: str,
    selected_series: tuple[str, ...],
    benchmark_payload: str,
    long_short_payload: str,
    date_range_payload: str,
    vol_scaler: float,
    vol_scaling_payload: str,
    window_weights_payload: str,
    config_payload: str,
) -> str:
    if not raw_data or not selected_series or not window_weights_payload:
        return "[]"
    try:
        window_weights = json.loads(window_weights_payload)
        config = json.loads(config_payload) if config_payload else {}
    except Exception:
        return "[]"
    working_df = get_working_returns(
        raw_data,
        periodicity or "daily",
        tuple(selected_series),
        benchmark_payload,
        long_short_payload,
        date_range_payload,
        vol_scaler,
        vol_scaling_payload,
    )
    rows = _compute_window_risk_contributions(working_df, selected_series, window_weights, config)
    return _serialize_risk_rows(rows)


def _po_get_window_risk_contributions(
    bundle: _PoWorkingReturnsBundle,
    selected_series,
    window_weights,
    config=None,
):
    payload = _po_compute_window_risk_contributions_cached(
        bundle.raw_data,
        bundle.periodicity,
        tuple(selected_series or ()),
        bundle.benchmark_payload,
        bundle.long_short_payload,
        bundle.date_range_payload,
        bundle.vol_scaler,
        bundle.vol_scaling_payload,
        canonical_json_dumps(window_weights or []),
        canonical_json_dumps(config or {}),
    )
    return _deserialize_risk_rows(payload)


def _periodicity_defaults(periodicity):
    """Return (window_size, opt_step_periods, opt_step_months, halflife) defaults."""
    if periodicity and periodicity.startswith("weekly"):
        return 52, 4, 1, 13
    if periodicity == "monthly":
        return 12, 1, 1, 6
    # daily, daily_trading, or any other
    return 252, 21, 1, 63


def _coerce_float(value):
    """Convert value to finite float; return None when invalid."""
    try:
        fval = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(fval):
        return None
    return fval


def _annualization_for_periodicity(periodicity) -> int:
    """Return annualization factor with safe fallback."""
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


def _normalize_monthly_df_if_needed(df: pd.DataFrame, periodicity: str) -> pd.DataFrame:
    """Canonicalize monthly indexes only when the workflow is monthly."""
    if periodicity == "monthly":
        return align_monthly_index_to_month_end(df)
    return df


def _po_import_selected_workbook_sheets(contents, filename, selected_sheets, workbook_sheets=None):
    return _shared_import_selected_workbook_sheets(
        contents,
        filename,
        selected_sheets,
        workbook_sheets=workbook_sheets,
    )


RF_CANONICAL_NAMES = {
    "bctbill13",
    "bctbill13trindex",
}
RF_KEYWORD_TOKENS = (
    "sofr",
    "tbill",
    "t bill",
    "treasury bill",
    "3m cash",
    "3 month cash",
    "3m treasury",
    "3 month treasury",
)


def _normalize_label_for_match(value: str | None) -> str:
    if not value:
        return ""
    text = str(value).strip().lower().replace("-", " ").replace("_", " ")
    text = " ".join(text.split())
    return text


def _looks_like_risk_free_label(value: str | None) -> bool:
    label = _normalize_label_for_match(value)
    if not label:
        return False
    compact = label.replace(" ", "")
    if compact in RF_CANONICAL_NAMES:
        return True
    return any(tok in label for tok in RF_KEYWORD_TOKENS)


def _resolve_risk_free_asset_name(asset_names, cmabench_assignments):
    """Find a risk-free asset among selected assets via name/CMABench mapping."""
    bench_map = cmabench_assignments or {}
    for asset in asset_names or ():
        if _looks_like_risk_free_label(asset):
            return str(asset)
        mapped_bench = bench_map.get(asset)
        if _looks_like_risk_free_label(mapped_bench):
            return str(asset)
    return None


@cache_config.cache.memoize(timeout=3600)
def _get_latest_bctbill13_hmm_mean_cached():
    """Return latest 10-year (hmm) CMA mean for BCTBill13, if available."""
    try:
        with DB_ENGINE.connect() as conn:
            latest_version = conn.execute(
                text("SELECT MAX(Version) FROM CMAStats WHERE Type = 'hmm'")
            ).scalar()
            if latest_version is None:
                return {"version": None, "mean": None}
            mean_value = conn.execute(
                text(
                    "SELECT Value FROM CMAStats "
                    "WHERE Version = :v AND Type = 'hmm' "
                    "AND Bench = 'BCTBill13' AND Item = 'Mean'"
                ),
                {"v": int(latest_version)},
            ).scalar()
            if mean_value is None:
                return {"version": int(latest_version), "mean": None}
            return {"version": int(latest_version), "mean": float(mean_value)}
    except Exception:
        return {"version": None, "mean": None}


def _annualized_return_for_periodicity_local(returns: pd.Series, periodicity: str) -> float:
    """Match statistics annualization behavior for risk-free return streams."""
    if returns is None:
        return np.nan
    clean = returns.dropna()
    if clean.empty:
        return np.nan
    p = periodicity or "daily"
    periods_per_year = _annualization_for_periodicity(p)
    if is_daily(p) or p.startswith("weekly_"):
        return annualized_return_calendar_days(clean, p)
    return annualized_return(clean, periods_per_year)


def _risk_free_series_for_periodicity(saved_series_store, periodicity):
    """Get cached risk-free series (BCTBill13_TRIndex) for selected periodicity."""
    rf_json = _risk_free_json_from_store(saved_series_store)
    if not rf_json:
        return None
    try:
        rf_df = resample_returns_cached(rf_json, periodicity or "daily")
    except Exception:
        return None
    if rf_df is None or rf_df.empty:
        return None
    rf_col = rf_df.columns[0]
    rf_series = rf_df[rf_col]
    if rf_series.empty:
        return None
    rf_series.index = pd.to_datetime(rf_series.index)
    return rf_series.sort_index().dropna()


def _resolve_risk_free_context(
    model,
    asset_order,
    periodicity,
    expected_mu_annual,
    reference_index,
    saved_series_store,
    cmabench_assignments,
):
    """Resolve annual risk-free rate for optimization/frontier Sharpe logic."""
    if model in {"ex_ante_mv", "black_litterman"}:
        rf_asset = _resolve_risk_free_asset_name(asset_order, cmabench_assignments)
        if rf_asset and expected_mu_annual:
            rf_value = _coerce_float(expected_mu_annual.get(rf_asset))
            if rf_value is not None:
                return {
                    "rf_annual": float(rf_value),
                    "rf_source": "asset_expected",
                    "rf_warning": None,
                    "rf_asset": rf_asset,
                }
        cma_payload = _get_latest_bctbill13_hmm_mean_cached() or {}
        cma_mean = _coerce_float(cma_payload.get("mean"))
        if cma_mean is not None:
            return {
                "rf_annual": float(cma_mean),
                "rf_source": "cma_bctbill13_hmm_latest",
                "rf_warning": None,
                "rf_asset": "BCTBill13",
            }
        return {
            "rf_annual": 0.0,
            "rf_source": "fallback_0",
            "rf_warning": "Risk-free rate unavailable; using rf=0 fallback.",
            "rf_asset": None,
        }

    rf_series = _risk_free_series_for_periodicity(saved_series_store, periodicity)
    if rf_series is not None:
        if reference_index is not None:
            aligned = rf_series.reindex(pd.DatetimeIndex(reference_index)).dropna()
        else:
            aligned = rf_series.dropna()
        if not aligned.empty:
            ann_rf = _annualized_return_for_periodicity_local(aligned, periodicity or "daily")
            ann_rf = _coerce_float(ann_rf)
            if ann_rf is not None:
                return {
                    "rf_annual": float(ann_rf),
                    "rf_source": "stats_cached_bctbill13",
                    "rf_warning": None,
                    "rf_asset": "BCTBill13_TRIndex",
                }
    return {
        "rf_annual": 0.0,
        "rf_source": "fallback_0",
        "rf_warning": "Risk-free rate unavailable; using rf=0 fallback.",
        "rf_asset": None,
    }


def _validate_ex_ante_expected_inputs(
    selected_series,
    ex_ante_mode,
    ex_ante_returns,
    ex_ante_cov,
    ex_ante_vol,
    ex_ante_corr,
):
    """Validate completeness of expected-return/covariance-style inputs."""
    assets = [str(s) for s in (selected_series or [])]
    if not assets:
        return "Select at least one series."

    mode = ex_ante_mode or "ret_cov"
    returns_map = ex_ante_returns or {}
    cov_map = ex_ante_cov or {}
    vol_map = ex_ante_vol or {}
    corr_map = ex_ante_corr or {}

    missing_returns = [a for a in assets if _coerce_float(returns_map.get(a)) is None]
    if missing_returns:
        return f"Missing expected return for: {', '.join(missing_returns)}."

    if mode == "ret_vol_corr":
        missing_vols = [a for a in assets if _coerce_float(vol_map.get(a)) is None]
        if missing_vols:
            return f"Missing expected volatility for: {', '.join(missing_vols)}."

        for r in assets:
            row = corr_map.get(r, {}) if isinstance(corr_map, dict) else {}
            if not isinstance(row, dict):
                return f"Correlation row for '{r}' is invalid."
            for c in assets:
                corr_val = _coerce_float(row.get(c))
                if corr_val is None:
                    return f"Missing correlation value for ({r}, {c})."
                if corr_val < -1 or corr_val > 1:
                    return f"Correlation ({r}, {c}) must be between -1 and 1."
        return None

    for r in assets:
        row = cov_map.get(r, {}) if isinstance(cov_map, dict) else {}
        if not isinstance(row, dict):
            return f"Covariance row for '{r}' is invalid."
        for c in assets:
            cov_val = _coerce_float(row.get(c))
            if cov_val is None:
                return f"Missing covariance value for ({r}, {c})."
    return None


def _validate_black_litterman_inputs(selected_series, bl_views, bl_tau):
    """Validate Black-Litterman-specific inputs."""
    tau = _coerce_float(bl_tau)
    if tau is None or tau <= 0:
        return "BL tau must be greater than 0."

    assets = {str(s) for s in (selected_series or [])}
    for i, view in enumerate(bl_views or [], start=1):
        if not isinstance(view, dict):
            return f"BL view #{i} is invalid."
        v_type = str(view.get("type", "absolute")).strip().lower()
        if v_type not in {"absolute", "relative"}:
            return f"BL view #{i} type must be absolute or relative."

        v_return = _coerce_float(view.get("return"))
        if v_return is None:
            return f"BL view #{i} return is invalid."
        confidence = _coerce_float(view.get("confidence", 1.0))
        if confidence is None or confidence <= 0:
            return f"BL view #{i} confidence must be greater than 0."

        asset = str(view.get("asset", "")).strip()
        if v_type == "absolute":
            if not asset or asset not in assets:
                return f"BL view #{i} asset must be one of the selected series."
            continue

        asset_to = str(view.get("asset_to", "")).strip()
        if not asset or not asset_to:
            return f"BL view #{i} relative pair is incomplete."
        if asset not in assets or asset_to not in assets:
            return f"BL view #{i} relative assets must be selected series."
        if asset == asset_to:
            return f"BL view #{i} relative assets must be different."

    return None


def _validate_linear_constraints_inputs(linear_constraints, selected_series):
    """Validate linear constraints from UI before optimization."""
    assets = [str(s) for s in (selected_series or [])]
    for idx, row in enumerate(linear_constraints or [], start=1):
        if not isinstance(row, dict):
            return f"Linear constraint row #{idx} is invalid."

        coeff_count = 0
        for asset in assets:
            val = row.get(asset)
            if val in (None, ""):
                continue
            fval = _coerce_float(val)
            if fval is None:
                return f"Linear constraint row #{idx} has invalid coefficient for {asset}."
            if abs(fval) > 1e-12:
                coeff_count += 1

        min_raw = row.get("Min")
        max_raw = row.get("Max")
        min_val = None if min_raw in (None, "") else _coerce_float(min_raw)
        max_val = None if max_raw in (None, "") else _coerce_float(max_raw)
        if min_raw not in (None, "") and min_val is None:
            return f"Linear constraint row #{idx} has invalid Min value."
        if max_raw not in (None, "") and max_val is None:
            return f"Linear constraint row #{idx} has invalid Max value."
        if min_val is not None and max_val is not None and min_val > max_val:
            return f"Linear constraint row #{idx} has Min greater than Max."
        if coeff_count == 0 and (min_val is not None or max_val is not None):
            return f"Linear constraint row #{idx} needs at least one non-zero coefficient."

    return None


def _validate_optimization_inputs(
    portfolio_name,
    selected_series,
    opt_model,
    opt_window,
    window_size,
    opt_step,
    opt_step_unit,
    exp_wt_cov,
    halflife,
    cov_shrinkage,
    cov_shrinkage_target,
    min_wt,
    max_wt,
    force_max,
    linear_constraints,
    ex_ante_mode,
    ex_ante_returns,
    ex_ante_cov,
    ex_ante_vol,
    ex_ante_corr,
    bl_views,
    bl_tau,
):
    """Return first validation error message, or None when valid."""
    if not portfolio_name or not str(portfolio_name).strip():
        return "Enter a portfolio name."
    if not selected_series:
        return "Select at least one series."
    if len(selected_series) < 2:
        return "Select at least two series."

    valid_models = {
        "risk_parity",
        "factor_risk_parity",
        "equal_weight",
        "hrp",
        "maximize_sharpe",
        "minimize_cvar",
        "minimize_variance",
        "ex_ante_mv",
        "black_litterman",
    }
    if opt_model not in valid_models:
        return "Select a valid optimization model."

    if opt_model not in {"ex_ante_mv", "black_litterman"}:
        if opt_window not in {"full", "rolling", "expanding"}:
            return "Select a valid optimization window."
        if opt_window != "full":
            ws = _coerce_float(window_size)
            if ws is None or ws < 2 or int(ws) != ws:
                return "Window size must be an integer >= 2."
            step = _coerce_float(opt_step)
            if step is None or step < 1 or int(step) != step:
                return "Optimization step must be an integer >= 1."
            if opt_step_unit not in {"periods", "months"}:
                return "Optimization step unit must be periods or months."

    if exp_wt_cov:
        hl = _coerce_float(halflife)
        if hl is None or hl <= 0:
            return "Decay input must be greater than 0 when exponential weighting is enabled."

    raw_shrinkage = "none" if cov_shrinkage in (None, "") else str(cov_shrinkage).strip().lower()
    if raw_shrinkage not in VALID_COV_SHRINKAGE:
        return "Select a valid covariance shrinkage option."
    raw_target = (
        "scaled_identity"
        if cov_shrinkage_target in (None, "")
        else str(cov_shrinkage_target).strip().lower()
    )
    if raw_target not in VALID_COV_SHRINKAGE_TARGET:
        return "Select a valid covariance shrinkage target."

    min_map = min_wt or {}
    max_map = max_wt or {}
    force_map = force_max or {}
    for asset in selected_series:
        mn = _coerce_float(min_map.get(asset, 0))
        mx = _coerce_float(max_map.get(asset, 100))
        if mn is None or mx is None:
            return f"Invalid min/max bound for {asset}."
        if mn < 0 or mx > 100:
            return f"Bounds for {asset} must stay within 0-100%."
        if mn > mx:
            return f"Min bound cannot exceed max bound for {asset}."
        if force_map.get(asset, False) and mx <= 0:
            return f"Force Max requires a positive max bound for {asset}."

    lc_error = _validate_linear_constraints_inputs(linear_constraints, selected_series)
    if lc_error:
        return lc_error

    if opt_model == "ex_ante_mv":
        ex_error = _validate_ex_ante_expected_inputs(
            selected_series,
            ex_ante_mode,
            ex_ante_returns,
            ex_ante_cov,
            ex_ante_vol,
            ex_ante_corr,
        )
        if ex_error:
            return ex_error

    if opt_model == "black_litterman":
        bl_error = _validate_black_litterman_inputs(selected_series, bl_views, bl_tau)
        if bl_error:
            return bl_error

    return None


def _resolve_frontier_window(window_weights, window_idx):
    """Resolve selected frontier window index safely."""
    if not window_weights:
        raise ValueError("No optimization windows available.")
    if window_idx is None:
        idx = len(window_weights) - 1
    else:
        try:
            idx = int(window_idx)
        except (TypeError, ValueError):
            idx = len(window_weights) - 1
    idx = max(0, min(idx, len(window_weights) - 1))
    return idx, window_weights[idx]


def _prepare_frontier_estimation_data(working_df, opt_series, window_weight, missing_data_method):
    """Build frontier estimation frame using the selected optimization window."""
    est_start = pd.Timestamp(window_weight.get("est_start", window_weight["apply_start"]))
    est_end = pd.Timestamp(window_weight.get("est_end", window_weight["apply_end"]))
    mask = (working_df.index >= est_start) & (working_df.index <= est_end)
    est_data = working_df.loc[mask, list(opt_series)].copy()

    if missing_data_method == "fill_0":
        est_data = est_data.fillna(0)
    else:
        valid_cols = [c for c in opt_series if c in est_data.columns and not est_data[c].isna().any()]
        if valid_cols:
            est_data = est_data[valid_cols]
        else:
            est_data = est_data.fillna(0)

    return est_data, est_start, est_end


def _build_ex_ante_mu_cov(config, asset_cols, ann):
    """Build per-period mu/cov for ex-ante optimization/frontier use."""
    mode = config.get("ex_ante_mode", "ret_cov")
    ex_ante_returns = config.get("ex_ante_returns", {}) or {}
    ex_ante_cov = config.get("ex_ante_cov", {}) or {}
    ex_ante_vol = config.get("ex_ante_vol", {}) or {}
    ex_ante_corr = config.get("ex_ante_corr", {}) or {}

    validation_error = _validate_ex_ante_expected_inputs(
        asset_cols,
        mode,
        ex_ante_returns,
        ex_ante_cov,
        ex_ante_vol,
        ex_ante_corr,
    )
    if validation_error:
        return None, None, validation_error

    mu_annual = np.array([float(ex_ante_returns[a]) for a in asset_cols], dtype=float)
    custom_mu = pd.DataFrame([mu_annual / ann], columns=asset_cols)

    if mode == "ret_vol_corr":
        vol_vec = np.array([float(ex_ante_vol[a]) for a in asset_cols], dtype=float)
        corr = np.zeros((len(asset_cols), len(asset_cols)), dtype=float)
        for i, r in enumerate(asset_cols):
            row = ex_ante_corr.get(r, {})
            for j, c in enumerate(asset_cols):
                corr[i, j] = float(row[c])
        cov_ann = np.outer(vol_vec, vol_vec) * corr
    else:
        cov_ann = np.zeros((len(asset_cols), len(asset_cols)), dtype=float)
        for i, r in enumerate(asset_cols):
            row = ex_ante_cov.get(r, {})
            for j, c in enumerate(asset_cols):
                cov_ann[i, j] = float(row[c])

    cov_ann = (cov_ann + cov_ann.T) / 2.0
    custom_cov = pd.DataFrame(cov_ann / ann, index=asset_cols, columns=asset_cols)
    return custom_mu, custom_cov, None


def _build_black_litterman_mu_cov(est_data, config, asset_cols):
    """Build per-period posterior mu/cov from BL inputs."""
    import riskfolio as rp

    if est_data.empty:
        return None, None, "No data available for Black-Litterman expected estimates."

    port = rp.Portfolio(returns=est_data.copy())
    port.assets_stats(method_mu="hist", method_cov="hist")
    exp_wt_cov = bool(config.get("exp_wt_cov", False))
    cov_shrinkage, cov_shrinkage_target = resolve_cov_shrinkage_spec(
        config.get("cov_shrinkage", "none"),
        config.get("cov_shrinkage_target", "scaled_identity"),
        exp_weighted=exp_wt_cov,
    )
    if exp_wt_cov or cov_shrinkage != "none":
        port.cov = estimate_covariance_matrix(
            est_data,
            asset_order=asset_cols,
            exp_weighted=exp_wt_cov,
            decay_value=normalize_decay_input(config.get("halflife", 63), 63.0),
            shrinkage=cov_shrinkage,
            shrinkage_target=cov_shrinkage_target,
        )

    bl_views = config.get("bl_views", []) or []
    n_assets = len(asset_cols)
    if bl_views:
        p_rows = []
        q_rows = []
        asset_idx = {name: i for i, name in enumerate(asset_cols)}
        for view in bl_views:
            v_type = str(view.get("type", "absolute")).strip().lower()
            q_val = _coerce_float(view.get("return"))
            if q_val is None:
                continue
            coeffs = np.zeros(n_assets, dtype=float)
            if v_type == "relative":
                asset = str(view.get("asset", "")).strip()
                asset_to = str(view.get("asset_to", "")).strip()
                if asset in asset_idx:
                    coeffs[asset_idx[asset]] = 1.0
                if asset_to in asset_idx:
                    coeffs[asset_idx[asset_to]] -= 1.0
            else:
                asset = str(view.get("asset", "")).strip()
                if asset in asset_idx:
                    coeffs[asset_idx[asset]] = 1.0
            if np.count_nonzero(coeffs) == 0:
                continue
            p_rows.append(coeffs)
            q_rows.append([q_val])
        if p_rows:
            P = pd.DataFrame(np.array(p_rows), columns=asset_cols)
            Q = pd.DataFrame(np.array(q_rows), columns=["views"])
        else:
            P = pd.DataFrame(np.zeros((1, n_assets)), columns=asset_cols)
            Q = pd.DataFrame(np.zeros((1, 1)), columns=["views"])
    else:
        P = pd.DataFrame(np.zeros((1, n_assets)), columns=asset_cols)
        Q = pd.DataFrame(np.zeros((1, 1)), columns=["views"])

    try:
        port.blacklitterman_stats(P=P, Q=Q, delta=None, rf=0, eq=True)
        mu = getattr(port, "mu_bl", None)
        cov = getattr(port, "cov_bl", None)
    except Exception:
        mu = None
        cov = None

    if mu is None:
        mu = getattr(port, "mu", None)
    if cov is None:
        cov = getattr(port, "cov", None)
    if mu is None or cov is None:
        return None, None, "Unable to compute Black-Litterman expected return/covariance."

    if isinstance(mu, pd.Series):
        mu = mu.to_frame().T
    elif not isinstance(mu, pd.DataFrame):
        mu_arr = np.asarray(mu, dtype=float).reshape(1, -1)
        mu = pd.DataFrame(mu_arr, columns=asset_cols)
    if mu.shape[0] != 1 and mu.shape[1] == 1:
        mu = mu.T
    mu = mu.reindex(columns=asset_cols)

    if not isinstance(cov, pd.DataFrame):
        cov = pd.DataFrame(np.asarray(cov, dtype=float), index=asset_cols, columns=asset_cols)
    cov = cov.reindex(index=asset_cols, columns=asset_cols)
    cov = (cov + cov.T) / 2.0

    if mu.isna().any().any() or cov.isna().any().any():
        return None, None, "Black-Litterman expected return/covariance has missing values."

    return mu.astype(float), cov.astype(float), None


def _normalize_weight_vector(weight_map, asset_cols):
    """Return normalized weight dict/vector aligned to asset columns."""
    w_arr = np.array([float((weight_map or {}).get(c, 0.0) or 0.0) for c in asset_cols], dtype=float)
    w_sum = float(w_arr.sum())
    if abs(w_sum) > 1e-12:
        w_arr = w_arr / w_sum
    return {c: float(w_arr[i]) for i, c in enumerate(asset_cols)}, w_arr


def _build_frontier_snapshot(
    selected_portfolio,
    portfolio_data,
    raw_data,
    periodicity,
    bench,
    ls,
    vol_scaler,
    vol_scaling,
    window_idx,
    rm,
    linear_constraints,
    saved_series_store=None,
    cmabench_assignments=None,
):
    """Compute frontier snapshot for chart/table/export with optional custom moments."""
    window_weights = portfolio_data.get("window_weights", []) or []
    config = portfolio_data.get("config", {}) or {}
    opt_series = config.get("selected_series", []) or []
    if not window_weights or not opt_series or not raw_data:
        raise ValueError("No frontier data available.")
    missing_sources = _po_missing_source_series({selected_portfolio: portfolio_data}, selected_portfolio, raw_data)
    if missing_sources:
        missing_text = ", ".join(missing_sources)
        raise ValueError(f"Missing source series for frontier: {missing_text}")

    frontier_bundle = _build_po_working_bundle(
        raw_data,
        periodicity,
        bench,
        ls,
        None,  # Frontier uses estimation windows directly, not date-range filter.
        vol_scaler,
        vol_scaling,
    )
    working_df = _po_get_working_returns(frontier_bundle, opt_series)
    if working_df.empty:
        raise ValueError("No working returns available for frontier.")

    idx, ww = _resolve_frontier_window(window_weights, window_idx)
    est_data, est_start, est_end = _prepare_frontier_estimation_data(
        working_df,
        opt_series,
        ww,
        config.get("missing_data", "fill_na"),
    )
    if est_data.empty or len(est_data) < 3:
        raise ValueError("Insufficient data for efficient frontier in this window.")

    actual_cols = list(est_data.columns)
    ann = _annualization_for_periodicity(periodicity)
    model = config.get("model", "")

    risk_measure = rm or "MV"
    if model in {"ex_ante_mv", "black_litterman"} and risk_measure == "CVaR":
        risk_measure = "MV"

    custom_mu = None
    custom_cov = None
    exp_wt_cov = bool(config.get("exp_wt_cov", False))
    decay_value = normalize_decay_input(config.get("halflife", 63), 63.0)
    cov_shrinkage, cov_shrinkage_target = resolve_cov_shrinkage_spec(
        config.get("cov_shrinkage", "none"),
        config.get("cov_shrinkage_target", "scaled_identity"),
        exp_weighted=exp_wt_cov,
    )
    if model == "ex_ante_mv":
        custom_mu, custom_cov, error_msg = _build_ex_ante_mu_cov(config, actual_cols, ann)
        if error_msg:
            raise ValueError(error_msg)
    elif model == "black_litterman":
        custom_mu, custom_cov, error_msg = _build_black_litterman_mu_cov(est_data, config, actual_cols)
        if error_msg:
            raise ValueError(error_msg)
    elif exp_wt_cov or cov_shrinkage != "none":
        custom_mu = estimate_mean_vector(
            est_data,
            asset_order=actual_cols,
            exp_weighted=exp_wt_cov,
            decay_value=decay_value,
        )
        custom_cov = estimate_covariance_matrix(
            est_data,
            asset_order=actual_cols,
            exp_weighted=exp_wt_cov,
            decay_value=decay_value,
            shrinkage=cov_shrinkage,
            shrinkage_target=cov_shrinkage_target,
            annualization_factor=ann,
        )

    frontier_pts, asset_pts, frontier_portfolios = compute_efficient_frontier(
        returns_df=est_data,
        ann_factor=ann,
        rm=risk_measure,
        custom_mu=custom_mu,
        custom_cov=custom_cov,
        linear_constraints=linear_constraints,
        return_weights=True,
    )
    if not frontier_pts:
        raise ValueError("Unable to compute efficient frontier points.")

    portfolio_weights, w_arr = _normalize_weight_vector(ww.get("weights", {}), actual_cols)
    if custom_mu is not None and custom_cov is not None:
        mu_vec = custom_mu.values.flatten()
        cov_mat = custom_cov.values
    else:
        mu_vec = est_data.mean().values
        cov_mat = est_data.cov().values

    expected_mu_annual = {c: float(mu_vec[i] * ann) for i, c in enumerate(actual_cols)}
    rf_context = _resolve_risk_free_context(
        model=model,
        asset_order=actual_cols,
        periodicity=periodicity,
        expected_mu_annual=expected_mu_annual,
        reference_index=est_data.index,
        saved_series_store=saved_series_store,
        cmabench_assignments=cmabench_assignments,
    )
    rf_annual = float(rf_context.get("rf_annual", 0.0) or 0.0)

    port_ret = float((w_arr @ mu_vec) * ann)
    if risk_measure == "CVaR":
        port_returns = est_data.values @ w_arr
        sorted_r = np.sort(port_returns)
        cutoff = max(1, int(np.ceil(len(sorted_r) * 0.05)))
        port_risk = float(-sorted_r[:cutoff].mean() * np.sqrt(ann))
    else:
        port_risk = float(np.sqrt(w_arr @ cov_mat @ w_arr) * np.sqrt(ann))

    assets_clean = [
        {
            "name": str(item["name"]),
            "return": float(item["return"]),
            "risk": float(item["risk"]),
        }
        for item in asset_pts
    ]
    frontier_points_clean = [
        {"return": float(item["return"]), "risk": float(item["risk"])}
        for item in frontier_pts
    ]
    frontier_portfolios_clean = []
    for fp in frontier_portfolios:
        frontier_portfolios_clean.append(
            {
                "point_index": int(fp["point_index"]),
                "return": float(fp["return"]),
                "risk": float(fp["risk"]),
                "weights": {k: float(v) for k, v in (fp.get("weights", {}) or {}).items()},
            }
        )

    snapshot = {
        "model": model,
        "risk_measure": risk_measure,
        "window_index": int(idx),
        "window_est_start": est_start.strftime("%Y-%m-%d"),
        "window_est_end": est_end.strftime("%Y-%m-%d"),
        "asset_order": actual_cols,
        "rf_annual": rf_annual,
        "rf_source": rf_context.get("rf_source"),
        "rf_warning": rf_context.get("rf_warning"),
        "portfolio": {
            "name": str(selected_portfolio),
            "return": port_ret,
            "risk": port_risk,
            "weights": portfolio_weights,
        },
        "assets": assets_clean,
        "frontier_points": frontier_points_clean,
        "frontier_portfolios": frontier_portfolios_clean,
    }
    return snapshot


def _normalize_frontier_risk_measure(model: str | None, rm: str | None) -> str:
    risk_measure = rm or "MV"
    if (model or "") in {"ex_ante_mv", "black_litterman"} and risk_measure == "CVaR":
        return "MV"
    return risk_measure


@cache_config.cache.memoize(timeout=0)
def _po_compute_frontier_snapshot_cached(
    selected_portfolio: str,
    raw_data: str,
    periodicity: str,
    benchmark_payload: str,
    long_short_payload: str,
    vol_scaler: float,
    vol_scaling_payload: str,
    window_weights_payload: str,
    config_payload: str,
    window_idx: int,
    risk_measure: str,
    linear_constraints_payload: str,
    saved_series_payload: str,
    cmabench_payload: str,
) -> str:
    if not raw_data or not window_weights_payload or not config_payload:
        raise ValueError("No frontier data available.")

    portfolio_data = {
        "window_weights": json.loads(window_weights_payload),
        "config": json.loads(config_payload),
    }
    snapshot = _build_frontier_snapshot(
        selected_portfolio=selected_portfolio,
        portfolio_data=portfolio_data,
        raw_data=raw_data,
        periodicity=periodicity or "daily",
        bench=json.loads(benchmark_payload) if benchmark_payload else {},
        ls=json.loads(long_short_payload) if long_short_payload else {},
        vol_scaler=float(vol_scaler or 0.0),
        vol_scaling=json.loads(vol_scaling_payload) if vol_scaling_payload else {},
        window_idx=window_idx,
        rm=risk_measure,
        linear_constraints=json.loads(linear_constraints_payload) if linear_constraints_payload else [],
        saved_series_store=json.loads(saved_series_payload) if saved_series_payload else None,
        cmabench_assignments=json.loads(cmabench_payload) if cmabench_payload else None,
    )
    return canonical_json_dumps(snapshot)


def _po_resolve_frontier_snapshot(
    *,
    selected_portfolio,
    portfolio_data,
    raw_data,
    periodicity,
    bench,
    ls,
    vol_scaler,
    vol_scaling,
    window_idx,
    rm,
    linear_constraints,
    saved_series_store=None,
    cmabench_assignments=None,
    persist_cache=False,
):
    window_weights = (portfolio_data or {}).get("window_weights", []) or []
    config = (portfolio_data or {}).get("config", {}) or {}
    if not window_weights or not config:
        raise ValueError("No frontier data available.")

    resolved_idx, _ = _resolve_frontier_window(window_weights, window_idx)
    risk_measure = _normalize_frontier_risk_measure(config.get("model", ""), rm)
    cached = _get_cached_frontier_snapshot(portfolio_data, resolved_idx, risk_measure)
    if cached is not None:
        return cached

    payload = _po_compute_frontier_snapshot_cached(
        str(selected_portfolio),
        raw_data,
        periodicity or "daily",
        _mapping_payload(bench),
        _mapping_payload(ls),
        float(vol_scaler or 0.0),
        _mapping_payload(vol_scaling),
        canonical_json_dumps(window_weights),
        canonical_json_dumps(config),
        int(resolved_idx),
        risk_measure,
        canonical_json_dumps(linear_constraints or []),
        canonical_json_dumps(saved_series_store or {}),
        _mapping_payload(cmabench_assignments),
    )
    snapshot = json.loads(payload) if payload else None
    if snapshot is None:
        raise ValueError("No frontier data available.")
    if persist_cache:
        _cache_frontier_snapshot(portfolio_data, snapshot)
    return snapshot


def _build_frontier_table_rows(snapshot):
    """Build row records for frontier table/export."""
    if not snapshot:
        return []

    asset_order = snapshot.get("asset_order", []) or []
    rf_annual = _coerce_float(snapshot.get("rf_annual"))
    if rf_annual is None:
        rf_annual = 0.0
    rows = []

    def _sharpe(ret_value, risk_value):
        ret_f = _coerce_float(ret_value)
        risk_f = _coerce_float(risk_value)
        if ret_f is None or risk_f is None or abs(risk_f) <= 1e-12:
            return None
        return float((ret_f - rf_annual) / risk_f)

    portfolio = snapshot.get("portfolio", {}) or {}
    prow = {
        "Type": "Optimized Portfolio",
        "Name": portfolio.get("name", ""),
        "Return": portfolio.get("return"),
        "Risk": portfolio.get("risk"),
        "Sharpe Ratio": _sharpe(portfolio.get("return"), portfolio.get("risk")),
    }
    for asset in asset_order:
        prow[f"Wt_{asset}"] = (portfolio.get("weights", {}) or {}).get(asset, 0.0)
    rows.append(prow)

    for asset_point in snapshot.get("assets", []) or []:
        name = asset_point.get("name", "")
        row = {
            "Type": "Asset",
            "Name": name,
            "Return": asset_point.get("return"),
            "Risk": asset_point.get("risk"),
            "Sharpe Ratio": _sharpe(asset_point.get("return"), asset_point.get("risk")),
        }
        for asset in asset_order:
            row[f"Wt_{asset}"] = 1.0 if asset == name else 0.0
        rows.append(row)

    for fp in snapshot.get("frontier_portfolios", []) or []:
        row = {
            "Type": "Frontier Point",
            "Name": f"Frontier {int(fp.get('point_index', 0)) + 1}",
            "Return": fp.get("return"),
            "Risk": fp.get("risk"),
            "Sharpe Ratio": _sharpe(fp.get("return"), fp.get("risk")),
        }
        for asset in asset_order:
            row[f"Wt_{asset}"] = (fp.get("weights", {}) or {}).get(asset, 0.0)
        rows.append(row)

    return rows


def _build_frontier_column_defs(snapshot):
    """Build AG Grid column definitions for frontier table."""
    if not snapshot:
        return []
    asset_order = snapshot.get("asset_order", []) or []
    cols = [
        {"field": "Type", "pinned": "left", "width": 160},
        {"field": "Name", "pinned": "left", "width": 170},
        {
            "field": "Return",
            "headerName": "Annual Return",
            "valueFormatter": {"function": "params.value != null ? d3.format('.2%')(params.value) : ''"},
            "width": 130,
        },
        {
            "field": "Risk",
            "headerName": "Annual Risk",
            "valueFormatter": {"function": "params.value != null ? d3.format('.2%')(params.value) : ''"},
            "width": 130,
        },
        {
            "field": "Sharpe Ratio",
            "headerName": "Sharpe Ratio",
            "valueFormatter": {"function": "params.value != null ? d3.format('.2f')(params.value) : ''"},
            "width": 120,
        },
    ]
    for asset in asset_order:
        cols.append(
            {
                "field": f"Wt_{asset}",
                "headerName": f"Wt {asset}",
                "valueFormatter": {"function": "params.value != null ? d3.format('.2%')(params.value) : ''"},
                "width": 120,
            }
        )
    return cols


def _cache_frontier_snapshot(portfolio_entry, snapshot):
    """Insert a computed snapshot into result-level frontier cache."""
    if not isinstance(portfolio_entry, dict) or not snapshot:
        return
    cache = portfolio_entry.get("frontier_cache") or {}
    idx_key = str(snapshot.get("window_index", 0))
    rm_key = str(snapshot.get("risk_measure", "MV"))
    by_window = cache.get(idx_key) or {}
    by_window[rm_key] = snapshot
    cache[idx_key] = by_window
    portfolio_entry["frontier_cache"] = cache


def _get_cached_frontier_snapshot(portfolio_entry, window_idx, rm):
    """Read cached snapshot if present."""
    if not isinstance(portfolio_entry, dict):
        return None
    cache = portfolio_entry.get("frontier_cache") or {}
    idx_key = str(window_idx)
    rm_key = str(rm)
    return (cache.get(idx_key) or {}).get(rm_key)


def _get_cma_stats_map(version: int, cma_type: str) -> dict[str, dict[str, float]]:
    data: dict[str, dict[str, float]] = {}
    with DB_ENGINE.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT Bench, Item, Value "
                "FROM CMAStats "
                "WHERE Version = :v AND Type = :t"
            ),
            {"v": int(version), "t": cma_type},
        ).fetchall()
    for bench, item, value in rows:
        data.setdefault(str(bench), {})[str(item)] = float(value)
    return data


def _get_cma_corr_map(version: int, cma_type: str) -> dict[str, dict[str, float]]:
    data: dict[str, dict[str, float]] = {}
    with DB_ENGINE.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT Bench1, Bench2, Value "
                "FROM CMACorrelation "
                "WHERE Version = :v AND Type = :t"
            ),
            {"v": int(version), "t": cma_type},
        ).fetchall()
    for b1, b2, value in rows:
        data.setdefault(str(b1), {})[str(b2)] = float(value)
    return data


def _get_cma_corr_value(corr_map: dict[str, dict[str, float]], bench1: str, bench2: str):
    """Read correlation treating CMACorrelation as triangular/symmetric storage."""
    v = corr_map.get(bench1, {}).get(bench2)
    if v is not None:
        return v
    v = corr_map.get(bench2, {}).get(bench1)
    if v is not None:
        return v
    return np.nan


def _compute_cma_missing(
    selected_series: list[str] | None,
    target: str | None,
    mode: str | None,
    stats_map: dict[str, dict[str, float]],
    corr_map: dict[str, dict[str, float]],
) -> list[str]:
    missing: list[str] = []
    series = selected_series or []
    mode = mode or "ret_cov"
    target = target or "returns"

    for s in series:
        bench_stats = stats_map.get(s, {})
        has_mean = "Mean" in bench_stats
        has_sd = "SD" in bench_stats
        if target == "returns":
            if mode == "ret_vol_corr":
                if not (has_mean and has_sd):
                    missing.append(s)
            elif not has_mean:
                missing.append(s)
        else:
            if not has_sd:
                missing.append(s)
                continue
            for c in series:
                if "SD" not in stats_map.get(c, {}):
                    continue
                if pd.isna(_get_cma_corr_value(corr_map, s, c)):
                    missing.append(s)
                    break

    # preserve order, unique
    return list(dict.fromkeys(missing))


def _cma_missing_message(target: str | None, missing: list[str]) -> str:
    if not missing:
        return ""
    if (target or "returns") == "returns":
        return f"Missing series in DB: {', '.join(missing)}. They will be loaded as 0."
    return f"Missing series in DB: {', '.join(missing)}. They will be loaded as NaN."


def _resolve_cma_bench(series_name: str, cmabench_assignments: dict | None) -> str:
    if not cmabench_assignments:
        return series_name
    bench = cmabench_assignments.get(series_name)
    if isinstance(bench, str):
        bench = bench.strip()
    return bench if bench else series_name


def _selected_cma_benches(selected_series: list[str] | None, cmabench_assignments: dict | None) -> list[str]:
    return [_resolve_cma_bench(s, cmabench_assignments) for s in (selected_series or [])]


def _effective_cmabench_assignments(selected_series: list[str] | None, cmabench_assignments: dict | None) -> dict[str, str]:
    series = selected_series or []
    effective = {}
    if cmabench_assignments:
        for s, v in cmabench_assignments.items():
            if isinstance(v, str) and v.strip():
                effective[s] = v.strip()
    missing = [s for s in series if not effective.get(s)]
    if missing:
        effective.update(get_cmabench_map_for_fofbench(DB_ENGINE, missing))
    return effective


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def build_po_welcome_screen():
    return build_shared_welcome_screen(PO_WELCOME_MODAL_CONFIG)


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
                                                    "Common Daily",
                                                    id="po-common-daily-button",
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

            # Linear Constraints Accordion
            dmc.Accordion(
                value=None,
                mb="xs",
                variant="contained",
                children=[
                    # Linear Constraints Accordion Item
                    dmc.AccordionItem(
                        value="linear-constraints",
                        children=[
                            dmc.AccordionControl("Linear Constraints"),
                            dmc.AccordionPanel(
                                children=[
                                    dmc.Text(
                                        "Define linear portfolio constraints with asset coefficients and Min/Max bounds. "
                                        "Each row enforces Min <= sum(coef_i * weight_i) <= Max (example: Equity + Credit between 0.40 and 0.70).",
                                        size="xs", c="dimmed", mb="xs",
                                    ),
                                    dmc.Group(
                                        gap="xs",
                                        mb="sm",
                                        children=[
                                            dmc.Button(
                                                "Add Constraint",
                                                id="po-add-constraint-btn",
                                                variant="outline",
                                                size="xs",
                                                leftSection=DashIconify(icon="tabler:plus"),
                                            ),
                                            dmc.Button(
                                                "Clear Constraints",
                                                id="po-clear-constraints-btn",
                                                variant="outline",
                                                size="xs",
                                                color="red",
                                                leftSection=DashIconify(icon="tabler:trash"),
                                            ),
                                        ],
                                    ),
                                    dag.AgGrid(
                                        enableEnterpriseModules=True,
                                        licenseKey=AG_GRID_LICENSE_KEY,
                                        id="po-linear-constraints-grid",
                                        className='ag-theme-alpine',
                                        columnDefs=[
                                            {"field": "Constraint", "editable": True, "width": 120, "headerClass": "dashmat-center-header"},
                                            {"field": "Min", "editable": True, "width": 90, "type": "numericColumn", 
                                             "valueFormatter": {"function": "d3.format('.4f')(params.value)"}, "headerClass": "dashmat-center-header"},
                                            {"field": "Max", "editable": True, "width": 90, "type": "numericColumn", 
                                             "valueFormatter": {"function": "d3.format('.4f')(params.value)"}, "headerClass": "dashmat-center-header"},
                                        ],
                                        rowData=[],
                                        defaultColDef={"resizable": True, "sortable": False, "suppressHeaderMenuButton": True, "cellStyle": {"textAlign": "center"}, "headerClass": "dashmat-center-header"},
                                        style={"height": "200px"},
                                        dashGridOptions={"singleClickEdit": True, "suppressExcelExport": True, "enableRangeSelection": True, "suppressCsvExport": True, "enterNavigatesVertically": True, "enterNavigatesVerticallyAfterEdit": True},
                                    ),
                                ],
                            ),
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
                                    gap="md",
                                    align="flex-end",
                                    mb="sm",
                                    children=[
                                        # Row 1: Portfolio Name, Model, Exp Wt Cov, Half-Life
                                        dmc.TextInput(
                                            id="po-portfolio-name-input",
                                            label="Portfolio Name",
                                            value=_po_default_name_for_model("risk_parity"),
                                            w=120,
                                            size="sm",
                                        ),
                                        html.Div([
                                            dmc.Text("Model", size="sm", fw=500, mb=3),
                                            dmc.Select(
                                                id="po-opt-model-select",
                                                data=[
                                                    {"value": "risk_parity", "label": "Risk Parity"},
                                                    {"value": "factor_risk_parity", "label": "Factor Risk Parity"},
                                                    {"value": "hierarchical_risk_parity", "label": "Hierarchical RP"},
                                                    {"value": "maximize_sharpe", "label": "Maximize Sharpe Ratio"},
                                                    {"value": "minimize_variance", "label": "Minimize Variance"},
                                                    {"value": "minimize_cvar", "label": "Minimize CVaR"},
                                                    {"value": "equal_weight", "label": "Equal Weight"},
                                                    {"value": "ex_ante_mv", "label": "Ex Ante Mean-Variance"},
                                                    {"value": "black_litterman", "label": "Black-Litterman"},
                                                ],
                                                value="risk_parity",
                                                w=210,
                                                size="sm",
                                                clearable=False,
                                                allowDeselect=False,
                                                maxDropdownHeight=420,
                                            ),
                                        ]),
                                        html.Div(
                                            id="po-objective-container",
                                            children=[
                                                dmc.Text("Objective", size="sm", fw=500, mb=3),
                                                dmc.Select(
                                                    id="po-objective-select",
                                                    data=[
                                                        {"label": "Maximize Sharpe", "value": "maximize_sharpe"},
                                                        {"label": "Minimize Variance", "value": "minimize_variance"},
                                                        {"label": "Maximize Return", "value": "maximize_return"},
                                                    ],
                                                    value="maximize_sharpe",
                                                    searchable=False,
                                                    clearable=False,
                                                    w=190,
                                                    size="sm",
                                                ),
                                            ],
                                            style={"display": "none"},
                                        ),
                                        html.Div(
                                            id="po-ex-ante-mode-container",
                                            children=[
                                                dmc.Text("Risk Input Mode", size="sm", fw=500, mb=3),
                                                html.Div(
                                                    dmc.SegmentedControl(
                                                        id="po-ex-ante-mode-select",
                                                        data=[
                                                            {"label": "Covariance", "value": "ret_cov"},
                                                            {"label": "Vol / Correlation", "value": "ret_vol_corr"},
                                                        ],
                                                        value="ret_vol_corr",
                                                        size="xs",
                                                    ),
                                                    style={"height": "36px", "display": "flex", "alignItems": "center"},
                                                ),
                                            ],
                                            style={"display": "none"},
                                        ),
                                        html.Div([
                                            dmc.Text("Exp Wt", size="sm", fw=500, mb=3),
                                            html.Div(
                                                dmc.Switch(
                                                    id="po-exp-wt-cov-switch",
                                                    checked=False,
                                                    size="sm",
                                                ),
                                                style={"height": "36px", "display": "flex", "alignItems": "center"},
                                            ),
                                        ]),
                                        html.Div([
                                            dmc.Text("Half-Life", size="sm", fw=500, mb=3),
                                            dmc.Tooltip(
                                                label="If value is < 1, it is interpreted as lambda. If value is >= 1, it is interpreted as half-life in periods.",
                                                multiline=True,
                                                w=300,
                                                withArrow=True,
                                                children=dmc.NumberInput(
                                                    id="po-halflife-input",
                                                    value=63,
                                                    min=0.001,
                                                    step=0.01,
                                                    w=90,
                                                    size="sm",
                                                    disabled=True,
                                                    style={"whiteSpace": "nowrap"},
                                                ),
                                            ),
                                        ]),
                                        html.Div([
                                            dmc.Text("Cov Shrinkage", size="sm", fw=500, mb=3),
                                            dmc.Select(
                                                id="po-cov-shrinkage-select",
                                                data=[
                                                    {"value": "none", "label": "None"},
                                                    {"value": "ledoit_wolf", "label": "Ledoit-Wolf"},
                                                    {"value": "oas", "label": "OAS"},
                                                ],
                                                value="none",
                                                searchable=False,
                                                clearable=False,
                                                w=130,
                                                size="sm",
                                            ),
                                        ]),
                                        html.Div([
                                            dmc.Text("Target", size="sm", fw=500, mb=3),
                                            dmc.Select(
                                                id="po-cov-shrinkage-target-select",
                                                data=[
                                                    {"value": "scaled_identity", "label": "Scaled Identity"},
                                                    {"value": "constant_correlation", "label": "Constant Correlation"},
                                                ],
                                                value="scaled_identity",
                                                searchable=False,
                                                clearable=False,
                                                w=180,
                                                size="sm",
                                                disabled=True,
                                            ),
                                        ]),
                                    ],
                                ),
                                dmc.Group(
                                    gap="md",
                                    align="flex-end",
                                    mb="sm",
                                    children=[
                                        # Row 2: Window, Fill In-Sample, Window Size, Opt Step, Missing Data
                                        html.Div([
                                            dmc.Text("Window", size="sm", mb=3, fw=500),
                                            dmc.SegmentedControl(
                                                id="po-opt-window-select",
                                                data=[
                                                    {"value": "expanding", "label": "Expanding"},
                                                    {"value": "rolling", "label": "Rolling"},
                                                    {"value": "full", "label": "Full"},
                                                ],
                                                value="rolling",
                                                size="sm",
                                            ),
                                        ]),
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
                                                disabled=False,
                                            ),
                                        ]),
                                        dmc.NumberInput(
                                            id="po-window-size-input",
                                            label="Window Size (Periods)",
                                            value=252,
                                            min=2,
                                            step=1,
                                            w=150,
                                            size="sm",
                                            disabled=False,
                                        ),
                                        html.Div([
                                            dmc.Text("Opt Step", size="sm", mb=4, fw=500),
                                            dmc.Group(
                                                gap="xs",
                                                wrap="nowrap",
                                                children=[
                                                    dmc.NumberInput(
                                                        id="po-opt-step-input",
                                                        value=1,
                                                        min=1,
                                                        step=1,
                                                        w=90,
                                                        size="sm",
                                                        disabled=False,
                                                    ),
                                                    dmc.Select(
                                                        id="po-opt-step-unit-select",
                                                        data=[
                                                            {"value": "months", "label": "Months"},
                                                            {"value": "periods", "label": "Periods"},
                                                        ],
                                                        value="months",
                                                        w=100,
                                                        size="sm",
                                                        clearable=False,
                                                    ),
                                                ],
                                            ),
                                        ]),
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
                                    ],
                                ),
                                        # Ex Ante Input Panel (hidden by default)
                                html.Div(
                                    id="po-ex-ante-panel",
                                    style={"display": "none"},
                                    children=[
                                        dmc.Divider(label="Ex Ante Inputs", labelPosition="center", mb="sm", mt="sm"),
                                        
                                        # Expected Returns & Volatility
                                        dmc.Text(
                                            "Expected Returns",
                                            id="po-ex-ante-returns-title",
                                            size="sm", fw=600, mb="xs"
                                        ),
                                        dmc.Text(
                                            "Enter percentages as whole numbers (5 = 5%). Upload CSV with columns: Asset, Return "
                                            "and optionally Volatility (used in Vol / Correlation mode).",
                                            size="xs", c="dimmed", mb="xs",
                                        ),
                                        dmc.Group(
                                            gap="xs",
                                            mb="sm",
                                            children=[
                                                dmc.Button(
                                                    "Load from DB",
                                                    id="po-load-db-returns-btn",
                                                    variant="outline",
                                                    size="xs",
                                                    leftSection=DashIconify(icon="tabler:database"),
                                                ),
                                                dmc.Button(
                                                    "Estimate from Data",
                                                    id="po-estimate-returns-btn",
                                                    variant="outline",
                                                    size="xs",
                                                    leftSection=DashIconify(icon="tabler:calculator"),
                                                ),
                                                dcc.Upload(
                                                    id="po-ex-ante-returns-upload",
                                                    children=dmc.Button(
                                                        "Upload Returns CSV",
                                                        variant="outline",
                                                        size="xs",
                                                        leftSection=DashIconify(icon="tabler:upload"),
                                                    ),
                                                    multiple=False,
                                                    accept=".csv",
                                                ),
                                                dmc.Button(
                                                    "Clear All",
                                                    id="po-ex-ante-returns-clear",
                                                    variant="outline",
                                                    color="red",
                                                    size="xs",
                                                    leftSection=DashIconify(icon="tabler:trash"),
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            id="po-ex-ante-returns-grid-container",
                                            children=[
                                                dag.AgGrid(
                                                    enableEnterpriseModules=True,
                                                    licenseKey=AG_GRID_LICENSE_KEY,
                                                    id="po-ex-ante-returns-grid",
                                                    className='ag-theme-alpine',
                                                    columnDefs=[
                                                        {"field": "Asset", "editable": False, "width": 140},
                                                        {"field": "Return", "editable": True, "width": 110,
                                                         "type": "numericColumn",
                                                         "valueFormatter": {"function": "d3.format('.2%')(params.value)"},
                                                         "valueParser": {"function": "var v=params.newValue; if (v===null || v===undefined || v==='') return null; var n=Number(v); if (!isFinite(n)) return null; return Math.abs(n) > 1 ? n/100 : n;"}},
                                                        {"field": "Volatility", "editable": True, "width": 110,
                                                         "type": "numericColumn",
                                                         "valueFormatter": {"function": "d3.format('.2%')(params.value)"},
                                                         "valueParser": {"function": "var v=params.newValue; if (v===null || v===undefined || v==='') return null; var n=Number(v); if (!isFinite(n)) return null; return Math.abs(n) > 1 ? n/100 : n;"},
                                                         "hide": True}, # Hidden by default
                                                    ],
                                                    rowData=[],
                                                    defaultColDef={"resizable": True, "sortable": False, "suppressHeaderMenuButton": True, "cellStyle": {"textAlign": "center"}, "headerClass": "dashmat-center-header"},
                                                    style={"height": "200px"},
                                                    dashGridOptions={"singleClickEdit": True, "suppressExcelExport": True, "enableRangeSelection": True, "suppressCsvExport": True, "enterNavigatesVertically": True, "enterNavigatesVerticallyAfterEdit": True},
                                                ),
                                            ],
                                            style={"marginBottom": "12px"},
                                        ),
                                        
                                        # Matrix Input (Covariance or Correlation)
                                        dmc.Text("Covariance Matrix", id="po-ex-ante-matrix-title", size="sm", fw=600, mb="xs"),
                                        dmc.Text(
                                            "Use a square CSV: first column Asset, remaining columns named by asset. "
                                            "Strongly recommended: use Estimate from Data or upload a prepared matrix.",
                                            size="xs", c="dimmed", mb="xs",
                                        ),
                                        dmc.Group(
                                            gap="xs",
                                            mb="sm",
                                            children=[
                                                dmc.Button(
                                                    "Load from DB",
                                                    id="po-load-db-matrix-btn",
                                                    variant="outline",
                                                    size="xs",
                                                    leftSection=DashIconify(icon="tabler:database"),
                                                ),
                                                dmc.Button(
                                                    "Estimate from Data",
                                                    id="po-estimate-matrix-btn",
                                                    variant="outline",
                                                    size="xs",
                                                    leftSection=DashIconify(icon="tabler:calculator"),
                                                ),
                                                dcc.Upload(
                                                    id="po-ex-ante-matrix-upload",
                                                    children=dmc.Button(
                                                        "Upload Cov CSV",
                                                        id="po-ex-ante-matrix-upload-btn",
                                                        variant="outline",
                                                        size="xs",
                                                        leftSection=DashIconify(icon="tabler:upload"),
                                                    ),
                                                    multiple=False,
                                                    accept=".csv",
                                                ),
                                                dmc.Button(
                                                    "Clear All",
                                                    id="po-ex-ante-matrix-clear",
                                                    variant="outline",
                                                    color="red",
                                                    size="xs",
                                                    leftSection=DashIconify(icon="tabler:trash"),
                                                ),
                                            ],
                                        ),
                                        
                                        html.Div(
                                            id="po-ex-ante-matrix-grid-container",
                                            children=[
                                                dag.AgGrid(
                                                    enableEnterpriseModules=True,
                                                    licenseKey=AG_GRID_LICENSE_KEY,
                                                    id="po-ex-ante-matrix-grid",
                                                    className='ag-theme-alpine',
                                                    columnDefs=[], # Populated dynamically
                                                    rowData=[],
                                                    defaultColDef={"resizable": True, "sortable": False, "editable": True, "width": 100, "suppressHeaderMenuButton": True,
                                                    "valueFormatter": {"function": "params.value !== null && params.value !== undefined && params.value !== '' && isFinite(Number(params.value)) ? d3.format('.4f')(Number(params.value)) : ''"}, "cellStyle": {"textAlign": "center"}, "headerClass": "dashmat-center-header"},
                                                    style={"height": "300px"},
                                                    dashGridOptions={"singleClickEdit": True, "stopEditingWhenCellsLoseFocus": True, "suppressExcelExport": True, "enableRangeSelection": True, "suppressCsvExport": True, "enterNavigatesVertically": True, "enterNavigatesVerticallyAfterEdit": True},
                                                ),
                                            ],
                                            style={"marginBottom": "12px"},
                                        ),
                                        # Black-Litterman Views (shown only for BL)
                                        html.Div(
                                            id="po-bl-views-panel",
                                            style={"display": "none"},
                                            children=[
                                                dmc.Divider(mb="sm"),
                                                dmc.Text("Black-Litterman Views", size="sm", fw=600, mb="xs"),
                                                dmc.Text(
                                                    "Add absolute or relative views. Relative: 'Asset outperforms Asset_To by Return'.",
                                                    size="xs", c="dimmed", mb="xs",
                                                ),
                                                dmc.Text(
                                                    "For Type=absolute, leave 'vs Asset (rel)' blank. It is only used for Type=relative.",
                                                    size="xs", c="dimmed", mb="xs",
                                                ),
                                                dmc.Group(
                                                    gap="xs",
                                                    mb="sm",
                                                    children=[
                                                        dmc.Button(
                                                            "Add View",
                                                            id="po-bl-add-view",
                                                            variant="outline",
                                                            size="xs",
                                                            leftSection=DashIconify(icon="tabler:plus"),
                                                        ),
                                                        dmc.Button(
                                                            "Clear Views",
                                                            id="po-bl-clear-views",
                                                            variant="outline",
                                                            size="xs",
                                                            color="red",
                                                            leftSection=DashIconify(icon="tabler:trash"),
                                                        ),
                                                    ],
                                                ),
                                                dag.AgGrid(
                                                    enableEnterpriseModules=True,
                                                    licenseKey=AG_GRID_LICENSE_KEY,
                                                    id="po-bl-views-grid",
                                                    className='ag-theme-alpine',
                                                    columnDefs=[
                                                        {"field": "Type", "editable": True, "width": 100,
                                                         "cellEditor": "agSelectCellEditor",
                                                         "cellEditorParams": {"values": ["absolute", "relative"]},
                                                         "headerClass": "dashmat-center-header"},
                                                        {"field": "Asset", "editable": True, "width": 150,
                                                         "headerClass": "dashmat-center-header"},
                                                        {"field": "Asset_To", "editable": True, "width": 150,
                                                         "headerName": "vs Asset (rel)",
                                                         "headerClass": "dashmat-center-header"},
                                                        {"field": "Return", "editable": True, "width": 100,
                                                         "type": "numericColumn",
                                                         "valueFormatter": {"function": "d3.format('.2f')(params.value) + '%'"},
                                                         "valueParser": {"function": "Number(params.newValue)"},
                                                         "headerClass": "dashmat-center-header"},
                                                        {"field": "Confidence", "editable": True, "width": 100,
                                                         "type": "numericColumn",
                                                         "valueFormatter": {"function": "d3.format('.2f')(params.value)"},
                                                         "headerClass": "dashmat-center-header"},
                                                    ],
                                                    rowData=[],
                                                    defaultColDef={"resizable": True, "sortable": False, "suppressHeaderMenuButton": True, "cellStyle": {"textAlign": "center"}, "headerClass": "dashmat-center-header"},
                                                    style={"height": "200px"},
                                                    dashGridOptions={"singleClickEdit": True, "suppressExcelExport": True, "enableRangeSelection": True, "suppressCsvExport": True, "enterNavigatesVertically": True, "enterNavigatesVerticallyAfterEdit": True},
                                                ),
                                                dmc.NumberInput(
                                                    id="po-bl-tau-input",
                                                    label="Tau (uncertainty)",
                                                    value=0.05,
                                                    min=0.001,
                                                    max=1.0,
                                                    step=0.01,
                                                    w=120,
                                                    size="sm",
                                                    mt="sm",
                                                    decimalScale=3,
                                                    mb="md",
                                                ),
                                            ],
                                        ),

                                        dmc.Divider(mb="md", mt="md"),
                                    ],
                                ),
                                # Row 3: Run button
                                dmc.Tooltip(
                                    id="po-run-button-tooltip",
                                    label="Load data and complete required inputs.",
                                    withArrow=True,
                                    position="top-start",
                                    disabled=False,
                                    children=html.Div(
                                        style={"display": "inline-block"},
                                        children=[
                                            dmc.Button(
                                                "Run",
                                                id="po-run-button",
                                                color="blue",
                                                size="sm",
                                                leftSection=DashIconify(icon="tabler:player-play"),
                                                disabled=True,
                                            ),
                                        ],
                                    ),
                                ),
                            ]),
                        ],
                    ),
                ],
            ),

            # Portfolio selector and visualization tabs
            dmc.Group(
                mb="xs",
                gap="xs",
                align="flex-end",
                children=[
                    dmc.Select(
                        id="po-weight-portfolio-select",
                        label="Portfolio",
                        data=[],
                        value=None,
                        w=250,
                        size="sm",
                        clearable=False,
                    ),
                    dmc.Button(
                        "Save Series",
                        id="po-save-series-button",
                        variant="light",
                        color="blue",
                        size="sm",
                        disabled=True,
                        leftSection=DashIconify(icon="tabler:device-floppy"),
                    ),
                    dmc.Button(
                        "Delete",
                        id="po-delete-portfolio-button",
                        variant="outline",
                        color="red",
                        size="sm",
                        disabled=True,
                        leftSection=DashIconify(icon="tabler:trash"),
                    ),
                    html.Div(
                        id="po-growth-multiselect-wrapper",
                        style={"display": "none", "flex": "1"},
                        children=[
                            dmc.MultiSelect(
                                id="po-growth-portfolio-multiselect",
                                label="Compare",
                                data=[],
                                value=[],
                                size="sm",
                            ),
                        ],
                    ),
                    dmc.Text(id="po-save-series-status-text", size="sm", c="dimmed"),
                ],
            ),

            dmc.Tabs(
                id="po-vis-tabs",
                value="weight",
                style={"height": "600px", "display": "flex", "flexDirection": "column", "overflow": "hidden"},
                children=[
                    dmc.TabsList(
                        children=[
                            dmc.TabsTab(spec["label"], value=spec["value"])
                            for spec in PO_TAB_SPECS
                        ]
                    ),
                    dmc.TabsPanel(
                        value="weight",
                        pt="md",
                        style={"flex": "1", "display": "flex", "flexDirection": "column", "overflow": "hidden"},
                        children=[
                            dmc.Group(mb="md", children=[
                                dmc.SegmentedControl(
                                    id="po-weight-chart-switch",
                                    data=[
                                        {"value": "table", "label": "Table"},
                                        {"value": "chart", "label": "Chart"},
                                    ],
                                    value="chart",
                                    size="sm",
                                ),
                            ]),
                            html.Div(
                                id="po-weight-chart-container",
                                style={"display": "flex", "flexDirection": "column", "flex": "1", "overflow": "hidden"},
                                children=[html.Div(id="po-weight-chart-content")],
                            ),
                            html.Div(
                                id="po-weight-grid-container",
                                style={"display": "none"},
                                children=[
                                    html.Div(id="po-weight-grid-content", style={"height": "100%", "width": "100%"}),
                                ],
                            ),
                        ],
                    ),
                    dmc.TabsPanel(
                        value="attribution",
                        pt="md",
                        style={"flex": "1", "display": "flex", "flexDirection": "column", "overflow": "hidden"},
                        children=[
                            dmc.Group(mb="md", children=[
                                dmc.SegmentedControl(
                                    id="po-attribution-chart-switch",
                                    data=[
                                        {"value": "table", "label": "Table"},
                                        {"value": "chart", "label": "Chart"},
                                    ],
                                    value="chart",
                                    size="sm",
                                ),
                            ]),
                            html.Div(
                                id="po-attribution-chart-container",
                                style={"display": "flex", "flexDirection": "column", "flex": "1", "overflow": "hidden"},
                                children=[],
                            ),
                            html.Div(
                                id="po-attribution-grid-container",
                                style={"display": "none"},
                                children=[],
                            ),
                        ],
                    ),
                    dmc.TabsPanel(
                        value="risk",
                        pt="md",
                        style={"flex": "1", "display": "flex", "flexDirection": "column", "overflow": "hidden"},
                        children=[
                            dmc.Group(mb="md", children=[
                                dmc.SegmentedControl(
                                    id="po-risk-chart-switch",
                                    data=[
                                        {"value": "table", "label": "Table"},
                                        {"value": "chart", "label": "Chart"},
                                    ],
                                    value="chart",
                                    size="sm",
                                ),
                            ]),
                            html.Div(
                                id="po-risk-chart-container",
                                style={"display": "flex", "flexDirection": "column", "flex": "1", "overflow": "hidden"},
                                children=[],
                            ),
                            html.Div(
                                id="po-risk-grid-container",
                                style={"display": "none"},
                                children=[],
                            ),
                        ],
                    ),
                    dmc.TabsPanel(
                        value="turnover",
                        pt="md",
                        style={"flex": "1", "display": "flex", "flexDirection": "column", "overflow": "hidden"},
                        children=[
                            dmc.Group(mb="md", children=[
                                dmc.SegmentedControl(
                                    id="po-turnover-chart-switch",
                                    data=[
                                        {"value": "table", "label": "Table"},
                                        {"value": "chart", "label": "Chart"},
                                    ],
                                    value="chart",
                                    size="sm",
                                ),
                            ]),
                            html.Div(
                                id="po-turnover-chart-container",
                                style={"display": "flex", "flexDirection": "column", "flex": "1", "overflow": "hidden"},
                                children=[],
                            ),
                            html.Div(
                                id="po-turnover-grid-container",
                                style={"display": "none"},
                                children=[],
                            ),
                        ],
                    ),
                    dmc.TabsPanel(
                        value="frontier",
                        pt="md",
                        style={"flex": "1", "display": "flex", "flexDirection": "column", "overflow": "hidden"},
                        children=[
                            dmc.Group(mb="md", children=[
                                dmc.Select(
                                    id="po-frontier-window-select",
                                    label="Window",
                                    data=[],
                                    value=None,
                                    w=250,
                                    size="sm",
                                    clearable=False,
                                ),
                                dmc.Select(
                                    id="po-frontier-rm-select",
                                    label="Risk Measure",
                                    data=[
                                        {"value": "MV", "label": "Volatility"},
                                        {"value": "CVaR", "label": "CVaR"},
                                    ],
                                    value="MV",
                                    w=150,
                                    size="sm",
                                    clearable=False,
                                ),
                                dmc.SegmentedControl(
                                    id="po-frontier-chart-switch",
                                    data=[
                                        {"value": "table", "label": "Table"},
                                        {"value": "chart", "label": "Chart"},
                                    ],
                                    value="chart",
                                    size="sm",
                                    style={"marginTop": "24px"},
                                ),
                            ]),
                            html.Div(
                                id="po-frontier-rf-warning",
                                style={"display": "none"},
                            ),
                            html.Div(
                                id="po-frontier-chart-container",
                                style={"display": "flex", "flexDirection": "column", "flex": "1", "overflow": "hidden"},
                                children=[],
                            ),
                            html.Div(
                                id="po-frontier-grid-container",
                                style={"display": "none"},
                                children=[],
                            ),
                        ],
                    ),
                    dmc.TabsPanel(
                        value="statistics",
                        pt="md",
                        style={"flex": "1", "display": "flex", "flexDirection": "column", "overflow": "hidden"},
                        children=[
                            html.Div(id="po-statistics-grid-content", style={"height": "100%", "width": "100%"}),
                        ],
                    ),
                    dmc.TabsPanel(
                        value="returns",
                        pt="md",
                        style={"flex": "1", "display": "flex", "flexDirection": "column", "overflow": "hidden"},
                        children=[
                            html.Div(id="po-returns-grid-content", style={"height": "100%", "width": "100%"}),
                        ],
                    ),
                    dmc.TabsPanel(
                        value="growth",
                        pt="md",
                        style={"flex": "1", "display": "flex", "flexDirection": "column", "overflow": "auto"},
                        children=[
                            dmc.Group(
                                mb="md",
                                children=[
                                    dmc.SegmentedControl(
                                        id="po-growth-chart-switch",
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
                                type="default",
                                children=[html.Div(id="po-growth-chart-container")],
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
                                gap="md",
                                children=[
                                    dmc.Select(
                                        id="po-rolling-metric-select",
                                        data=[
                                            {"value": "total_return", "label": "Total Return"},
                                            {"value": "volatility", "label": "Volatility"},
                                            {"value": "sharpe_ratio", "label": "Sharpe Ratio"},
                                            {"value": "sortino_ratio", "label": "Sortino Ratio"},
                                        ],
                                        value="total_return",
                                        w=180,
                                        size="sm",
                                        clearable=False,
                                    ),
                                    dmc.Select(
                                        id="po-rolling-window-select",
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
                                        clearable=False,
                                    ),
                                    dmc.SegmentedControl(
                                        id="po-rolling-return-type-select",
                                        data=[
                                            {"value": "cumulative", "label": "Cumulative"},
                                            {"value": "annualized", "label": "Annualized"},
                                        ],
                                        value="annualized",
                                        size="sm",
                                    ),
                                    dmc.SegmentedControl(
                                        id="po-rolling-chart-switch",
                                        data=[
                                            {"value": "table", "label": "Table"},
                                            {"value": "chart", "label": "Chart"},
                                        ],
                                        value="chart",
                                        size="sm",
                                    ),
                                ],
                            ),
                            html.Div(id="po-rolling-content", style={"height": "100%", "overflow": "auto"}),
                        ],
                    ),
                    dmc.TabsPanel(
                        value="calendar",
                        pt="md",
                        style={"flex": "1", "display": "flex", "flexDirection": "column", "overflow": "hidden"},
                        children=[
                            dmc.Group(
                                mb="md",
                                gap="md",
                                children=[
                                    dmc.SegmentedControl(
                                        id="po-calendar-view-select",
                                        data=[
                                            {"value": "annual", "label": "Annual"},
                                            {"value": "monthly", "label": "Monthly"},
                                        ],
                                        value="annual",
                                        size="sm",
                                    ),
                                    dmc.Select(
                                        id="po-calendar-series-select",
                                        data=[],
                                        value=None,
                                        w=220,
                                        size="sm",
                                        clearable=False,
                                        disabled=True,
                                        placeholder="Series (Monthly view)",
                                    ),
                                ],
                            ),
                            html.Div(id="po-calendar-content", style={"height": "100%", "overflow": "auto"}),
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
                                        id="po-drawdown-chart-switch",
                                        data=[
                                            {"value": "table", "label": "Table"},
                                            {"value": "chart", "label": "Chart"},
                                        ],
                                        value="chart",
                                        size="sm",
                                    ),
                                ],
                            ),
                            html.Div(id="po-drawdown-content", style={"height": "100%", "overflow": "auto"}),
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
    style={"minHeight": "calc(100vh - 55px)", "display": "flex", "flexDirection": "column", "overflow": "visible"},
    className='dashmat-page-container',
    children=[
        # Menu bar
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
                                    )
                                ),
                                dmc.MenuDropdown(className="dashmat-menu-dropdown", children=[
                                    dmc.MenuItem(
                                        "New session",
                                        id="po-menu-clear-local-storage",
                                        leftSection=DashIconify(icon="tabler:trash", width=14),
                                    ),
                                    dmc.MenuItem(
                                        "Load session",
                                        id="po-menu-load-session",
                                        leftSection=DashIconify(icon="tabler:folder-open", width=14),
                                    ),
                                    dmc.MenuItem(
                                        "Save session",
                                        id="po-menu-save-session",
                                        leftSection=DashIconify(icon="tabler:device-floppy", width=14),
                                    ),
                                    dmc.MenuDivider(),
                                    dmc.MenuItem(
                                        "Download Excel",
                                        id="po-menu-download-excel",
                                        leftSection=DashIconify(icon="tabler:file-spreadsheet", width=14),
                                    ),
                                    dmc.MenuDivider(),
                                    dmc.MenuItem(
                                        "Exit",
                                        id="po-menu-exit",
                                        color="red",
                                        leftSection=DashIconify(icon="tabler:door-exit", width=14),
                                    ),
                                ]),
                            ],
                        ),
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
                                        "Add",
                                        variant="subtle",
                                        color="gray",
                                        size="sm",
                                        radius="sm",
                                    )
                                ),
                                dmc.MenuDropdown(className="dashmat-menu-dropdown", children=[
                                    dmc.MenuItem(
                                        "Add AA Tool indices...",
                                        id="po-menu-add-from-db",
                                        leftSection=DashIconify(icon="tabler:database", width=14),
                                    ),
                                    dmc.MenuDivider(),
                                    dmc.MenuItem(
                                        "Add peer-relative portfolios...",
                                        id="po-menu-add-portfolios-peer",
                                        leftSection=DashIconify(icon="tabler:users", width=14),
                                    ),
                                    dmc.MenuItem(
                                        "Add index-relative portfolios...",
                                        id="po-menu-add-portfolios-index",
                                        leftSection=DashIconify(icon="tabler:chart-line", width=14),
                                    ),
                                    dmc.MenuItem(
                                        "Add alternative portfolios...",
                                        id="po-menu-add-portfolios-other",
                                        leftSection=DashIconify(icon="tabler:stack", width=14),
                                    ),
                                    dmc.MenuItem(
                                        "Add underlying categories...",
                                        id="po-menu-add-portfolios-underlying",
                                        leftSection=DashIconify(icon="tabler:hierarchy-2", width=14),
                                    ),
                                    dmc.MenuDivider(),
                                    dmc.MenuItem(
                                        "Add raw factor data...",
                                        id="po-menu-add-raw-factor",
                                        leftSection=DashIconify(icon="tabler:chart-dots", width=14),
                                    ),
                                    dmc.MenuItem(
                                        "Add raw funds...",
                                        id="po-menu-add-raw-funds",
                                        leftSection=DashIconify(icon="tabler:building-bank", width=14),
                                    ),
                                    dmc.MenuItem(
                                        "Add raw performance...",
                                        id="po-menu-add-raw-performance",
                                        leftSection=DashIconify(icon="tabler:activity-heartbeat", width=14),
                                    ),
                                    dmc.MenuDivider(),
                                    dmc.MenuItem(
                                        "Add series from file...",
                                        id="po-menu-add-series",
                                        leftSection=DashIconify(icon="tabler:upload", width=14),
                                    ),
                                ]),
                            ],
                        ),
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
                                    )
                                ),
                                dmc.MenuDropdown(className="dashmat-menu-dropdown", children=[
                                    dmc.MenuItem(
                                        "Clear server cache",
                                        id="po-menu-clear-server-cache",
                                        leftSection=DashIconify(icon="tabler:server-off", width=14),
                                    ),
                                ]),
                            ],
                        ),
                        dmc.Button(
                            "Switch to Analytics",
                            id="po-menu-view-analytics",
                            size="sm",
                            radius="md",
                            variant="gradient",
                            gradient={"from": "orange", "to": "red", "deg": 90},
                            leftSection=DashIconify(icon="tabler:chart-line", width=16),
                        ),
                        dmc.Button(
                            "Switch to Regression",
                            id="po-menu-view-regression",
                            size="sm",
                            radius="md",
                            variant="gradient",
                            gradient={"from": "grape", "to": "indigo", "deg": 90},
                            leftSection=DashIconify(icon="tabler:chart-dots-3", width=16),
                        ),
                        dmc.Box(style={"flexGrow": 1}),
                        _po_build_help_control(),
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

        build_series_selection_modal(PO_WELCOME_MODAL_CONFIG),

        build_sheet_select_modal("po"),

        # Optimization status modal (progress -> completion in one modal)
        dmc.Modal(
            id="po-progress-modal",
            opened=False,
            closeOnClickOutside=False,
            withCloseButton=False,
            size="xs",
            centered=True,
            radius="lg",
            className="dashmat-modal",
            overlayProps={"blur": 2, "opacity": 0.45},
            transitionProps={"transition": "fade", "duration": 180},
            styles={"body": {"padding": "0"}},
            children=[
                # Running state
                html.Div(
                    id="po-running-indicator",
                    children=[
                        dmc.Stack(align="center", gap="md", py="xl",
                                  justify="center", style={"minHeight": "200px"},
                                  children=[
                            dmc.Loader(type="dots", size="xl"),
                            dmc.Text("Running optimization...", size="sm", c="dimmed"),
                        ]),
                    ],
                ),
                # Completion state (hidden initially)
                html.Div(
                    id="po-completion-indicator",
                    style={"display": "none"},
                    children=[
                        dmc.Stack(align="center", gap="md", py="xl",
                                  justify="center", style={"minHeight": "200px"},
                                  children=[
                            DashIconify(id="po-completion-icon", icon="tabler:check", width=48, color="green"),
                            dmc.Text(id="po-completion-text", children="", size="sm", c="dimmed"),
                            dmc.Button("Close", id="po-close-completion-button", size="sm", variant="light"),
                        ]),
                    ],
                ),
            ],
        ),

        build_db_add_modal("po"),
        build_portfolio_add_modal("po", AG_GRID_LICENSE_KEY),
        build_underlying_add_modal("po", AG_GRID_LICENSE_KEY),
        build_raw_db_add_modal("po", AG_GRID_LICENSE_KEY),

        # CMA Load Modal
        dmc.Modal(
            id="po-cma-load-modal",
            title=dmc.Group(
                gap="xs",
                children=[
                    dmc.ThemeIcon(DashIconify(icon="tabler:database"), color="indigo", variant="light", size="sm"),
                    dmc.Text("Load CMA Data from Database", fw=600, size="sm"),
                ],
            ),
            size="sm",
            centered=True,
            closeOnClickOutside=True,
            radius="lg",
            className="dashmat-modal",
            overlayProps={"blur": 2, "opacity": 0.45},
            transitionProps={"transition": "fade", "duration": 180},
            children=[
                dmc.Stack(
                    gap="sm",
                    children=[
                        dmc.Select(
                            id="po-cma-version-select",
                            label="Version",
                            data=[],
                            value=None,
                            clearable=False,
                        ),
                        dmc.Select(
                            id="po-cma-type-select",
                            label="Type",
                            data=[
                                {"value": "hmm", "label": "10-Year"},
                                {"value": "equilibrium.gp", "label": "Equilibrium"},
                            ],
                            value="hmm",
                            clearable=False,
                        ),
                        dmc.Text(id="po-cma-load-missing-text", c="red", size="xs"),
                        dmc.Group(
                            justify="flex-end",
                            children=[
                                dmc.Button("Cancel", id="po-cma-load-cancel", variant="outline", color="red"),
                                dmc.Button("Load", id="po-cma-load-confirm", color="blue"),
                            ],
                        ),
                    ],
                ),
            ],
        ),


        # Welcome screen (Hydration gates visibility)
        html.Div(
            id="po-welcome-screen",
            children=build_po_welcome_screen(),
            style={"display": "none"},
        ),

        # Main container (Hydration gates visibility)
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
        dcc.Store(id="po-cmabench-assignments-store", data={}, storage_type="session"),
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
        dcc.Store(id="po-temp-cmabench-assignments-store", data={}),
        dcc.Store(id="po-temp-long-short-store", data={}),
        dcc.Store(id="po-temp-vol-scaling-assignments-store", data={}),
        dcc.Store(id="po-temp-min-wt-store", data={}),
        dcc.Store(id="po-temp-max-wt-store", data={}),
        dcc.Store(id="po-temp-force-max-store", data={}),
        dcc.Store(id="po-portfolio-add-mode-store", data=None),
        dcc.Store(id="po-portfolio-add-rows-store", data=[]),
        dcc.Store(id="po-underlying-add-rows-store", data=[]),
        dcc.Store(id="po-raw-db-add-mode-store", data=None),
        dcc.Store(id="po-raw-db-add-rows-store", data=[]),
        # Temp stores for sheet selection (stash upload while user picks a tab)
        dcc.Store(id="po-sheet-select-contents-store", data=None),
        dcc.Store(id="po-sheet-select-filename-store", data=None),
        dcc.Store(id="po-sheet-select-sheetnames-store", data=None),
        # Controls stores
        dcc.Store(id="po-periodicity-value-store", data="daily_trading", storage_type="session"),
        dcc.Store(id="po-periodicity-load-sync-dummy", data=None),
        dcc.Store(id="po-vol-scaler-value-store", data=0, storage_type="session"),
        dcc.Store(id="po-date-range-store", data=None, storage_type="session"),
        dcc.Store(id="po-range-candidates-store", data=None, storage_type="memory"),
        dcc.Store(id="po-common-daily-candidates-store", data=None, storage_type="memory"),
        dcc.Store(id="po-series-select-value-store", data=[], storage_type="session"),
        # Optimization stores
        dcc.Store(id="po-opt-window-store", data="rolling", storage_type="session"),
        dcc.Store(id="po-window-size-store", data=252, storage_type="session"),
        dcc.Store(id="po-opt-step-store", data=1, storage_type="session"),
        dcc.Store(id="po-opt-step-unit-store", data="months", storage_type="session"),
        dcc.Store(id="po-opt-model-store", data="risk_parity", storage_type="session"),
        dcc.Store(id="po-portfolio-name-store", data=_po_default_name_for_model("risk_parity"), storage_type="session"),
        dcc.Store(id="po-exp-wt-cov-store", data=False, storage_type="session"),
        dcc.Store(id="po-halflife-store", data=63, storage_type="session"),
        dcc.Store(id="po-cov-shrinkage-store", data="none", storage_type="session"),
        dcc.Store(id="po-cov-shrinkage-target-store", data="scaled_identity", storage_type="session"),
        dcc.Store(id="po-missing-data-store", data="fill_na", storage_type="session"),
        dcc.Store(id="po-fill-in-sample-store", data="off", storage_type="session"),
        # Ex ante stores
        dcc.Store(id="po-ex-ante-returns-store", data={}, storage_type="session"),
        dcc.Store(id="po-ex-ante-cov-store", data={}, storage_type="session"),
        dcc.Store(id="po-ex-ante-vol-store", data={}, storage_type="session"),
        dcc.Store(id="po-ex-ante-corr-store", data={}, storage_type="session"),
        dcc.Store(id="po-ex-ante-mode-store", data="ret_vol_corr", storage_type="session"),
        dcc.Store(id="po-bl-views-store", data=[], storage_type="session"),
        dcc.Store(id="po-linear-constraints-store", data=[], storage_type="session"),
        dcc.Store(id="po-bl-tau-store", data=0.05, storage_type="session"),
        dcc.Store(id="po-objective-store", data="maximize_sharpe", storage_type="session"),
        dcc.Store(id="po-cma-load-target-store", data=None),
        # Results stores
        dcc.Store(id="po-results-store", data={}, storage_type="session"),
        dcc.Store(id="po-opt-status-store", data=None, storage_type="memory"),
        dcc.Store(id="po-active-tab-store", data="weight", storage_type="session"),
        dcc.Store(id="po-initial-tab-render-ready-store", data=False, storage_type="memory"),
        dcc.Store(id="po-secondary-restore-ready-store", data=False, storage_type="memory"),
        dcc.Store(id="po-restore-complete-store", data=False, storage_type="memory"),
        dcc.Store(id="po-attribution-tab-loaded-store", data=False, storage_type="memory"),
        dcc.Store(id="po-risk-tab-loaded-store", data=False, storage_type="memory"),
        dcc.Store(id="po-frontier-tab-loaded-store", data=False, storage_type="memory"),
        # Chart/table switch stores
        dcc.Store(id="po-weight-chart-switch-store", data="chart", storage_type="session"),
        dcc.Store(id="po-attribution-chart-switch-store", data="chart", storage_type="session"),
        dcc.Store(id="po-risk-chart-switch-store", data="chart", storage_type="session"),
        dcc.Store(id="po-turnover-chart-switch-store", data="chart", storage_type="session"),
        dcc.Store(id="po-frontier-chart-switch-store", data="chart", storage_type="session"),
        dcc.Store(id="po-page-visited-store", data=False, storage_type="session"),
        # Save/Load session
        dcc.Store(id="po-save-session-dummy", data=None, storage_type="memory"),
        dcc.Store(id="po-load-session-dummy", data=None, storage_type="memory"),
        dcc.Store(id="po-server-cache-clear-result", data=None, storage_type="memory"),
        html.Div(
            dcc.Upload(
                id="po-load-session-upload",
                children=html.Div(),
                multiple=False,
                accept=".json",
            ),
            style={"display": "none"},
        ),
        # Excel download
        dcc.Download(id="po-download-excel"),
        # Navigation
        dcc.Location(id="po-url-location", refresh=False),
        # One-shot interval to trigger visibility check after session-storage hydration
        dcc.Interval(id="po-page-load-trigger", interval=50, max_intervals=1, n_intervals=0),

        # UI Blocker for file dialog (Overlay)
        dcc.Store(id="po-ui-blocker-store", data=False),
        dmc.LoadingOverlay(
            id="po-ui-blocker-overlay",
            visible=False,
            zIndex=2000,
            overlayProps={"radius": "sm", "blur": 2},
            loaderProps={"variant": "bars"},
        ),
    ],
)


# ===========================================================================
# Clientside callbacks
# ===========================================================================


# Shared clientside navigation
clientside_callback(
    ClientsideFunction(namespace="dashmat_callbacks", function_name="navigatePortopt"),
    Output("po-url-location", "pathname"),
    Input("po-menu-exit", "n_clicks"),
    Input("po-menu-view-analytics", "n_clicks"),
    Input("po-menu-view-regression", "n_clicks"),
    Input("po-welcome-view-analytics", "n_clicks"),
    Input("po-welcome-view-regression", "n_clicks"),
    prevent_initial_call=True,
)


clientside_callback(
    ClientsideFunction(namespace="dashmat_callbacks", function_name="clearWorkspaceSession"),
    Output("po-url-location", "pathname", allow_duplicate=True),
    Input("po-menu-clear-local-storage", "n_clicks"),
    prevent_initial_call=True,
)


@callback(
    Output("po-server-cache-clear-result", "data"),
    Input("po-menu-clear-server-cache", "n_clicks"),
    prevent_initial_call=True,
)
def po_clear_server_cache(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    cache_config.cache.clear()
    clear_dropdown_caches()
    return {"cleared": True, "timestamp": pd.Timestamp.utcnow().isoformat()}


# Open add-from-database modal and refresh options
@callback(
    Output("po-db-add-modal", "opened", allow_duplicate=True),
    Output("po-db-add-series-select", "data", allow_duplicate=True),
    Output("po-db-add-series-select", "value", allow_duplicate=True),
    Output("po-ui-blocker-store", "data", allow_duplicate=True),
    Input("po-menu-add-from-db", "n_clicks"),
    Input("po-welcome-add-db-btn", "n_clicks"),
    prevent_initial_call=True,
)
def po_open_db_add_modal(menu_clicks, welcome_clicks):
    return compute_open_db_add_modal(menu_clicks, welcome_clicks, DB_ENGINE)


@callback(
    Output("po-db-add-modal", "opened", allow_duplicate=True),
    Output("po-db-add-series-select", "value", allow_duplicate=True),
    Input("po-db-add-cancel-button", "n_clicks"),
    prevent_initial_call=True,
)
def po_close_db_add_modal(n_clicks):
    return compute_close_db_add_modal(n_clicks)


@callback(
    Output("po-raw-db-add-modal", "opened", allow_duplicate=True),
    Output("po-raw-db-add-modal", "title", allow_duplicate=True),
    Output("po-raw-db-add-mode-store", "data", allow_duplicate=True),
    Output("po-raw-db-add-series-select", "data", allow_duplicate=True),
    Output("po-raw-db-add-series-select", "value", allow_duplicate=True),
    Output("po-raw-db-add-table-select", "value", allow_duplicate=True),
    Output("po-raw-db-add-fee-select", "value", allow_duplicate=True),
    Output("po-raw-db-add-include-benchmark", "checked", allow_duplicate=True),
    Output("po-raw-db-add-convert-returns", "checked", allow_duplicate=True),
    Output("po-raw-db-add-divide-by", "value", allow_duplicate=True),
    Output("po-raw-db-add-error-alert", "hide", allow_duplicate=True),
    Output("po-raw-db-add-rows-store", "data", allow_duplicate=True),
    Output("po-raw-db-add-grid", "rowData", allow_duplicate=True),
    Output("po-raw-db-preview-lines", "children", allow_duplicate=True),
    Output("po-raw-db-add-ok-button", "disabled", allow_duplicate=True),
    Output("po-ui-blocker-store", "data", allow_duplicate=True),
    Input("po-menu-add-raw-factor", "n_clicks"),
    Input("po-menu-add-raw-funds", "n_clicks"),
    Input("po-menu-add-raw-performance", "n_clicks"),
    Input("po-welcome-add-raw-factor-btn", "n_clicks"),
    Input("po-welcome-add-raw-funds-btn", "n_clicks"),
    Input("po-welcome-add-raw-performance-btn", "n_clicks"),
    prevent_initial_call=True,
)
def po_open_raw_db_add_modal(
    factor_clicks,
    funds_clicks,
    performance_clicks,
    welcome_factor_clicks,
    welcome_funds_clicks,
    welcome_performance_clicks,
):
    return compute_open_raw_db_add_modal(
        prefix="po",
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
    Output("po-raw-db-add-modal", "opened", allow_duplicate=True),
    Output("po-raw-db-add-rows-store", "data", allow_duplicate=True),
    Output("po-raw-db-add-grid", "rowData", allow_duplicate=True),
    Output("po-raw-db-preview-lines", "children", allow_duplicate=True),
    Output("po-raw-db-add-error-alert", "hide", allow_duplicate=True),
    Output("po-raw-db-add-ok-button", "disabled", allow_duplicate=True),
    Output("po-raw-db-add-series-select", "value", allow_duplicate=True),
    Input("po-raw-db-add-cancel-button", "n_clicks"),
    prevent_initial_call=True,
)
def po_close_raw_db_add_modal(n_clicks):
    opened, rows, grid_rows, preview = compute_close_raw_db_add_modal(n_clicks)
    return opened, rows, grid_rows, preview, True, True, None


@callback(
    Output("po-raw-db-add-table-select", "disabled"),
    Output("po-raw-db-add-fee-select", "data"),
    Output("po-raw-db-add-fee-select", "value"),
    Output("po-raw-db-add-fee-select", "disabled"),
    Output("po-raw-db-add-include-benchmark", "disabled"),
    Output("po-raw-db-add-include-benchmark", "checked", allow_duplicate=True),
    Output("po-raw-db-factor-controls", "style"),
    Output("po-raw-db-add-convert-returns", "checked", allow_duplicate=True),
    Input("po-raw-db-add-mode-store", "data"),
    Input("po-raw-db-add-series-select", "value"),
    Input("po-raw-db-add-modal", "opened"),
    State("po-raw-db-add-fee-select", "value"),
    State("po-raw-db-add-include-benchmark", "checked"),
    State("po-raw-db-add-convert-returns", "checked"),
    prevent_initial_call=True,
)
def po_sync_raw_modal_controls(mode, series_key, opened, current_fee, current_include_benchmark, current_convert):
    if not opened:
        raise PreventUpdate

    triggered_id = callback_context.triggered_id
    preserve_series_selection_state = triggered_id == "po-raw-db-add-series-select"
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
    Output("po-raw-db-add-divide-by", "disabled"),
    Input("po-raw-db-add-mode-store", "data"),
    Input("po-raw-db-add-convert-returns", "checked"),
    Input("po-raw-db-add-modal", "opened"),
    prevent_initial_call=True,
)
def po_toggle_raw_divide_by(mode, convert_to_returns, opened):
    if not opened:
        raise PreventUpdate
    return not (str(mode or "").strip().lower() == "factor" and not bool(convert_to_returns))


@callback(
    Output("po-raw-db-add-rows-store", "data", allow_duplicate=True),
    Output("po-raw-db-add-grid", "rowData", allow_duplicate=True),
    Output("po-raw-db-add-error-alert", "children", allow_duplicate=True),
    Output("po-raw-db-add-error-alert", "hide", allow_duplicate=True),
    Input("po-raw-db-add-row-btn", "n_clicks"),
    State("po-raw-db-add-rows-store", "data"),
    State("po-raw-db-add-mode-store", "data"),
    State("po-raw-db-add-series-select", "value"),
    State("po-raw-db-add-table-select", "value"),
    State("po-raw-db-add-fee-select", "value"),
    State("po-raw-db-add-include-benchmark", "checked"),
    State("po-raw-db-add-convert-returns", "checked"),
    State("po-raw-db-add-divide-by", "value"),
    prevent_initial_call=True,
)
def po_stage_raw_db_row(
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
    Output("po-raw-db-add-rows-store", "data", allow_duplicate=True),
    Output("po-raw-db-add-grid", "rowData", allow_duplicate=True),
    Output("po-raw-db-add-error-alert", "children", allow_duplicate=True),
    Output("po-raw-db-add-error-alert", "hide", allow_duplicate=True),
    Input("po-raw-db-delete-row-btn", "n_clicks"),
    State("po-raw-db-add-rows-store", "data"),
    State("po-raw-db-add-grid", "selectedRows"),
    prevent_initial_call=True,
)
def po_delete_raw_db_row(n_delete, staged_rows, selected_rows):
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
    Output("po-raw-db-add-rows-store", "data", allow_duplicate=True),
    Output("po-raw-db-add-grid", "rowData", allow_duplicate=True),
    Output("po-raw-db-add-error-alert", "children", allow_duplicate=True),
    Output("po-raw-db-add-error-alert", "hide", allow_duplicate=True),
    Input("po-raw-db-clear-rows-btn", "n_clicks"),
    prevent_initial_call=True,
)
def po_clear_raw_db_rows(n_clear):
    if not n_clear:
        raise PreventUpdate
    return [], [], no_update, True


clientside_callback(
    js_portfolio_ok_disabled(),
    Output("po-raw-db-add-ok-button", "disabled", allow_duplicate=True),
    Input("po-raw-db-add-rows-store", "data"),
    Input("po-raw-db-add-modal", "opened"),
    prevent_initial_call=True,
)


@callback(
    Output("po-raw-db-preview-lines", "children", allow_duplicate=True),
    Input("po-raw-db-add-modal", "opened"),
    Input("po-raw-db-add-mode-store", "data"),
    Input("po-raw-db-add-series-select", "value"),
    Input("po-raw-db-add-table-select", "value"),
    Input("po-raw-db-add-fee-select", "value"),
    Input("po-raw-db-add-include-benchmark", "checked"),
    Input("po-raw-db-add-convert-returns", "checked"),
    Input("po-raw-db-add-divide-by", "value"),
    prevent_initial_call=True,
)
def po_update_raw_db_preview(
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
    Output("po-db-add-error-alert", "children"),
    Output("po-db-add-error-alert", "hide"),
    Output("po-db-add-ok-button", "disabled"),
    Input("po-db-add-series-select", "value"),
    Input("dashmat-raw-data-store", "data"),
    Input("po-db-add-modal", "opened"),
    prevent_initial_call=True,
)
def po_validate_db_add_selection(selected_benches, raw_data, opened):
    return compute_validate_db_add_selection(selected_benches, raw_data, opened)


@callback(
    Output("po-portfolio-add-modal", "opened", allow_duplicate=True),
    Output("po-portfolio-add-modal", "title", allow_duplicate=True),
    Output("po-portfolio-add-mode-store", "data", allow_duplicate=True),
    Output("po-portfolio-add-series-select", "data", allow_duplicate=True),
    Output("po-portfolio-add-series-select", "value", allow_duplicate=True),
    Output("po-portfolio-add-type-select", "data", allow_duplicate=True),
    Output("po-portfolio-add-type-select", "value", allow_duplicate=True),
    Output("po-portfolio-add-benchmark-type-select", "data", allow_duplicate=True),
    Output("po-portfolio-add-benchmark-type-select", "value", allow_duplicate=True),
    Output("po-portfolio-add-include-benchmark", "checked", allow_duplicate=True),
    Output("po-portfolio-add-benchmark-type-select", "disabled", allow_duplicate=True),
    Output("po-portfolio-add-rows-store", "data", allow_duplicate=True),
    Output("po-portfolio-add-grid", "rowData", allow_duplicate=True),
    Output("po-portfolio-add-error-alert", "hide", allow_duplicate=True),
    Output("po-ui-blocker-store", "data", allow_duplicate=True),
    Input("po-menu-add-portfolios-peer", "n_clicks"),
    Input("po-menu-add-portfolios-index", "n_clicks"),
    Input("po-menu-add-portfolios-other", "n_clicks"),
    Input("po-welcome-add-portfolios-peer-btn", "n_clicks"),
    Input("po-welcome-add-portfolios-index-btn", "n_clicks"),
    Input("po-welcome-add-portfolios-other-btn", "n_clicks"),
    prevent_initial_call=True,
)
def po_open_portfolio_add_modal(
    peer_clicks,
    index_clicks,
    other_clicks,
    welcome_peer_clicks,
    welcome_index_clicks,
    welcome_other_clicks,
):
    return compute_open_portfolio_add_modal(
        prefix="po",
        triggered_id=callback_context.triggered_id,
        peer_clicks=peer_clicks,
        index_clicks=index_clicks,
        other_clicks=other_clicks,
        welcome_peer_clicks=welcome_peer_clicks,
        welcome_index_clicks=welcome_index_clicks,
        welcome_other_clicks=welcome_other_clicks,
        db_engine=DB_ENGINE,
    )


@callback(
    Output("po-underlying-add-modal", "opened", allow_duplicate=True),
    Output("po-underlying-add-modal", "title", allow_duplicate=True),
    Output("po-underlying-add-base-select", "value", allow_duplicate=True),
    Output("po-underlying-add-type-multiselect", "value", allow_duplicate=True),
    Output("po-underlying-add-desc-multiselect", "data", allow_duplicate=True),
    Output("po-underlying-add-desc-multiselect", "value", allow_duplicate=True),
    Output("po-underlying-add-desc-multiselect", "disabled", allow_duplicate=True),
    Output("po-underlying-add-rows-store", "data", allow_duplicate=True),
    Output("po-underlying-add-grid", "rowData", allow_duplicate=True),
    Output("po-underlying-add-error-alert", "hide", allow_duplicate=True),
    Output("po-ui-blocker-store", "data", allow_duplicate=True),
    Input("po-menu-add-portfolios-underlying", "n_clicks"),
    Input("po-welcome-add-portfolios-underlying-btn", "n_clicks"),
    prevent_initial_call=True,
)
def po_open_underlying_add_modal(menu_clicks, welcome_clicks):
    return compute_open_underlying_add_modal(menu_clicks, welcome_clicks)


@callback(
    Output("po-underlying-add-desc-multiselect", "data"),
    Output("po-underlying-add-desc-multiselect", "value"),
    Output("po-underlying-add-desc-multiselect", "disabled"),
    Input("po-underlying-add-base-select", "value"),
    Input("po-underlying-add-type-multiselect", "value"),
    Input("po-underlying-add-modal", "opened"),
    State("po-underlying-add-desc-multiselect", "value"),
    prevent_initial_call=True,
)
def po_sync_underlying_desc_options(base_value, type_values, opened, current_values):
    if not opened:
        raise PreventUpdate

    if not base_value or not type_values:
        return [], [], True

    options = get_underlying_category_desc_options(DB_ENGINE, base_value, type_values)
    valid_values = {str(option.get("value", "")).strip() for option in options}
    selected = [
        value
        for value in (current_values or [])
        if str(value or "").strip() in valid_values
    ]
    return options, selected, False


@callback(
    Output("po-underlying-add-rows-store", "data", allow_duplicate=True),
    Output("po-underlying-add-grid", "rowData", allow_duplicate=True),
    Output("po-underlying-add-error-alert", "children", allow_duplicate=True),
    Output("po-underlying-add-error-alert", "hide", allow_duplicate=True),
    Input("po-underlying-add-row-btn", "n_clicks"),
    State("po-underlying-add-rows-store", "data"),
    State("po-underlying-add-base-select", "value"),
    State("po-underlying-add-type-multiselect", "value"),
    State("po-underlying-add-desc-multiselect", "value"),
    prevent_initial_call=True,
)
def po_stage_underlying_rows(n_clicks, staged_rows, base_value, type_values, desc_values):
    if not n_clicks:
        raise PreventUpdate

    rows = [dict(row) for row in (staged_rows or []) if isinstance(row, dict)]
    if not str(base_value or "").strip():
        return rows, rows, "Select Core or Base before staging underlying categories.", False
    if not type_values:
        return rows, rows, "Select at least one type before staging underlying categories.", False
    if not desc_values:
        return rows, rows, "Select at least one underlying category description.", False

    requested_rows = expand_underlying_category_rows(DB_ENGINE, base_value, type_values, desc_values)
    if not requested_rows:
        return rows, rows, "No matching underlying category rows were found for the selected Base, Type, and Desc values.", False

    existing_pairs = {
        (
            str(row.get("portfolio") or row.get("Portfolio") or "").strip(),
            str(row.get("desc") or row.get("Desc") or "").strip(),
        )
        for row in rows
    }
    new_rows = [
        row for row in requested_rows
        if (
            str(row.get("portfolio") or "").strip(),
            str(row.get("desc") or "").strip(),
        ) not in existing_pairs
    ]
    if not new_rows:
        return rows, rows, "All selected underlying category rows are already staged.", False

    updated_rows = rows + new_rows
    return updated_rows, updated_rows, no_update, True


@callback(
    Output("po-portfolio-add-include-benchmark", "disabled"),
    Output("po-portfolio-add-include-benchmark", "checked", allow_duplicate=True),
    Input("po-portfolio-add-mode-store", "data"),
    Input("po-portfolio-add-series-select", "value"),
    State("po-portfolio-add-include-benchmark", "checked"),
    prevent_initial_call=True,
)
def po_sync_include_benchmark_enabled(mode, selected_portfolio, current_checked):
    return compute_sync_include_benchmark_enabled(mode, selected_portfolio, current_checked, DB_ENGINE)


clientside_callback(
    js_portfolio_benchmark_toggle(),
    Output("po-portfolio-add-benchmark-type-select", "disabled", allow_duplicate=True),
    Output("po-portfolio-add-benchmark-type-select", "value", allow_duplicate=True),
    Input("po-portfolio-add-include-benchmark", "checked"),
    State("po-portfolio-add-benchmark-type-select", "data"),
    State("po-portfolio-add-benchmark-type-select", "value"),
    prevent_initial_call=True,
)


clientside_callback(
    js_portfolio_add_row(),
    Output("po-portfolio-add-rows-store", "data", allow_duplicate=True),
    Output("po-portfolio-add-grid", "rowData", allow_duplicate=True),
    Output("po-portfolio-add-error-alert", "children", allow_duplicate=True),
    Output("po-portfolio-add-error-alert", "hide", allow_duplicate=True),
    Input("po-portfolio-add-row-btn", "n_clicks"),
    State("po-portfolio-add-rows-store", "data"),
    State("po-portfolio-add-series-select", "value"),
    State("po-portfolio-add-type-select", "value"),
    State("po-portfolio-add-include-benchmark", "checked"),
    State("po-portfolio-add-benchmark-type-select", "value"),
    prevent_initial_call=True,
)

clientside_callback(
    js_portfolio_delete_row(),
    Output("po-portfolio-add-rows-store", "data", allow_duplicate=True),
    Output("po-portfolio-add-grid", "rowData", allow_duplicate=True),
    Output("po-portfolio-add-error-alert", "children", allow_duplicate=True),
    Output("po-portfolio-add-error-alert", "hide", allow_duplicate=True),
    Input("po-portfolio-delete-row-btn", "n_clicks"),
    State("po-portfolio-add-rows-store", "data"),
    State("po-portfolio-add-grid", "selectedRows"),
    prevent_initial_call=True,
)

clientside_callback(
    js_portfolio_clear_rows(),
    Output("po-portfolio-add-rows-store", "data", allow_duplicate=True),
    Output("po-portfolio-add-grid", "rowData", allow_duplicate=True),
    Output("po-portfolio-add-error-alert", "children", allow_duplicate=True),
    Output("po-portfolio-add-error-alert", "hide", allow_duplicate=True),
    Input("po-portfolio-clear-rows-btn", "n_clicks"),
    prevent_initial_call=True,
)


clientside_callback(
    js_underlying_delete_row(),
    Output("po-underlying-add-rows-store", "data", allow_duplicate=True),
    Output("po-underlying-add-grid", "rowData", allow_duplicate=True),
    Output("po-underlying-add-error-alert", "children", allow_duplicate=True),
    Output("po-underlying-add-error-alert", "hide", allow_duplicate=True),
    Input("po-underlying-delete-row-btn", "n_clicks"),
    State("po-underlying-add-rows-store", "data"),
    State("po-underlying-add-grid", "selectedRows"),
    prevent_initial_call=True,
)

clientside_callback(
    js_portfolio_clear_rows(),
    Output("po-underlying-add-rows-store", "data", allow_duplicate=True),
    Output("po-underlying-add-grid", "rowData", allow_duplicate=True),
    Output("po-underlying-add-error-alert", "children", allow_duplicate=True),
    Output("po-underlying-add-error-alert", "hide", allow_duplicate=True),
    Input("po-underlying-clear-rows-btn", "n_clicks"),
    prevent_initial_call=True,
)


@callback(
    Output("po-portfolio-add-modal", "opened", allow_duplicate=True),
    Output("po-portfolio-add-rows-store", "data", allow_duplicate=True),
    Output("po-portfolio-add-grid", "rowData", allow_duplicate=True),
    Input("po-portfolio-add-cancel-button", "n_clicks"),
    prevent_initial_call=True,
)
def po_close_portfolio_add_modal(n_clicks):
    return compute_close_portfolio_add_modal(n_clicks)


@callback(
    Output("po-underlying-add-modal", "opened", allow_duplicate=True),
    Output("po-underlying-add-base-select", "value", allow_duplicate=True),
    Output("po-underlying-add-type-multiselect", "value", allow_duplicate=True),
    Output("po-underlying-add-desc-multiselect", "data", allow_duplicate=True),
    Output("po-underlying-add-desc-multiselect", "value", allow_duplicate=True),
    Output("po-underlying-add-desc-multiselect", "disabled", allow_duplicate=True),
    Output("po-underlying-add-rows-store", "data", allow_duplicate=True),
    Output("po-underlying-add-grid", "rowData", allow_duplicate=True),
    Input("po-underlying-add-cancel-button", "n_clicks"),
    prevent_initial_call=True,
)
def po_close_underlying_add_modal(n_clicks):
    return compute_close_underlying_add_modal(n_clicks)


clientside_callback(
    js_portfolio_ok_disabled(),
    Output("po-portfolio-add-ok-button", "disabled"),
    Input("po-portfolio-add-rows-store", "data"),
    Input("po-portfolio-add-modal", "opened"),
)

clientside_callback(
    js_portfolio_ok_disabled(),
    Output("po-underlying-add-ok-button", "disabled"),
    Input("po-underlying-add-rows-store", "data"),
    Input("po-underlying-add-modal", "opened"),
)

clientside_callback(
    ClientsideFunction(namespace="dashmat_callbacks", function_name="uiBlockerEnable"),
    Output("po-ui-blocker-store", "data", allow_duplicate=True),
    Input("po-menu-add-from-db", "n_clicks"),
    Input("po-welcome-add-db-btn", "n_clicks"),
    Input("po-menu-add-raw-factor", "n_clicks"),
    Input("po-menu-add-raw-funds", "n_clicks"),
    Input("po-menu-add-raw-performance", "n_clicks"),
    Input("po-welcome-add-raw-factor-btn", "n_clicks"),
    Input("po-welcome-add-raw-funds-btn", "n_clicks"),
    Input("po-welcome-add-raw-performance-btn", "n_clicks"),
    Input("po-menu-add-portfolios-peer", "n_clicks"),
    Input("po-menu-add-portfolios-index", "n_clicks"),
    Input("po-menu-add-portfolios-other", "n_clicks"),
    Input("po-welcome-add-portfolios-peer-btn", "n_clicks"),
    Input("po-welcome-add-portfolios-index-btn", "n_clicks"),
    Input("po-welcome-add-portfolios-other-btn", "n_clicks"),
    Input("po-menu-add-portfolios-underlying", "n_clicks"),
    Input("po-welcome-add-portfolios-underlying-btn", "n_clicks"),
    Input("po-open-modal-button", "n_clicks"),
    prevent_initial_call=True,
)

clientside_callback(
    ClientsideFunction(namespace="dashmat_callbacks", function_name="uiBlockerEnable"),
    Output("po-ui-blocker-store", "data", allow_duplicate=True),
    Input("po-db-add-ok-button", "n_clicks"),
    Input("po-raw-db-add-ok-button", "n_clicks"),
    Input("po-portfolio-add-ok-button", "n_clicks"),
    Input("po-underlying-add-ok-button", "n_clicks"),
    Input("po-modal-ok-button", "n_clicks"),
    prevent_initial_call=True,
)

clientside_callback(
    ClientsideFunction(namespace="dashmat_callbacks", function_name="uiBlockerRelease"),
    Output("po-ui-blocker-store", "data", allow_duplicate=True),
    Input("po-db-add-error-alert", "hide"),
    Input("po-raw-db-add-error-alert", "hide"),
    Input("po-portfolio-add-error-alert", "hide"),
    Input("po-underlying-add-error-alert", "hide"),
    Input("po-series-selection-modal", "opened"),
    prevent_initial_call=True,
)

# Trigger upload from menu or welcome button
clientside_callback(
    ClientsideFunction(namespace="dashmat_callbacks", function_name="triggerPortoptUpload"),
    Output("po-ui-blocker-store", "data", allow_duplicate=True),
    Input("po-menu-add-series", "n_clicks"),
    Input("po-welcome-add-series-btn", "n_clicks"),
    prevent_initial_call=True,
)

# UI Blocker: sync overlay visibility
clientside_callback(
    """
    function(is_loading) {
        return is_loading || false;
    }
    """,
    Output("po-ui-blocker-overlay", "visible"),
    Input("po-ui-blocker-store", "data"),
)

clientside_callback(
    ClientsideFunction(namespace="dashmat_callbacks", function_name="portoptInitialSeriesBlocker"),
    Output("po-ui-blocker-store", "data", allow_duplicate=True),
    Input("po-url-location", "pathname"),
    Input("dashmat-raw-data-meta-store", "data"),
    Input("po-page-visited-store", "data"),
    Input("po-series-select", "data"),
    prevent_initial_call=True,
)

clientside_callback(
    ClientsideFunction(namespace="dashmat_callbacks", function_name="releaseBlockerOnSeriesGridReady"),
    Output("po-ui-blocker-store", "data", allow_duplicate=True),
    Input("po-series-selection-grid", "virtualRowData", allow_optional=True),
    State("po-series-selection-modal", "opened"),
    prevent_initial_call=True,
)

# Store sync: top-level controls
clientside_callback(
    ClientsideFunction(namespace="dashmat_callbacks", function_name="portoptControlSync"),
    Output("po-periodicity-value-store", "data"),
    Output("po-vol-scaler-value-store", "data"),
    Output("po-active-tab-store", "data"),
    Output("po-series-select-value-store", "data"),
    Output("po-fill-in-sample-store", "data"),
    Output("po-opt-step-unit-store", "data"),
    Output("po-opt-window-store", "data"),
    Output("po-window-size-store", "data"),
    Output("po-opt-step-store", "data"),
    Output("po-opt-model-store", "data"),
    Output("po-portfolio-name-store", "data"),
    Output("po-exp-wt-cov-store", "data"),
    Output("po-halflife-store", "data"),
    Output("po-cov-shrinkage-store", "data"),
    Output("po-cov-shrinkage-target-store", "data"),
    Output("po-missing-data-store", "data"),
    Output("po-objective-store", "data"),
    Output("po-bl-tau-store", "data"),
    Output("po-ex-ante-mode-store", "data"),
    Input("po-periodicity-select", "value"),
    Input("po-vol-scaler-input", "value"),
    Input("po-vis-tabs", "value"),
    Input("po-series-select", "data"),
    Input("po-fill-in-sample-select", "value"),
    Input("po-opt-step-unit-select", "value"),
    Input("po-opt-window-select", "value"),
    Input("po-window-size-input", "value"),
    Input("po-opt-step-input", "value"),
    Input("po-opt-model-select", "value"),
    Input("po-portfolio-name-input", "value"),
    Input("po-exp-wt-cov-switch", "checked"),
    Input("po-halflife-input", "value"),
    Input("po-cov-shrinkage-select", "value"),
    Input("po-cov-shrinkage-target-select", "value"),
    Input("po-missing-data-select", "value"),
    Input("po-objective-select", "value"),
    Input("po-bl-tau-input", "value"),
    Input("po-ex-ante-mode-select", "value"),
    prevent_initial_call=True,
)

# Sync periodicity to Analytics only on raw-data load/update events.
clientside_callback(
    ClientsideFunction(namespace="dashmat_callbacks", function_name="syncPortoptPeriodicity"),
    Output("po-periodicity-load-sync-dummy", "data"),
    Input("dashmat-raw-data-store", "data"),
    Input("po-periodicity-value-store", "data"),
    prevent_initial_call=True,
)

# Toggle halflife disabled based on exp wt cov switch
clientside_callback(
    """
    function(checked, shrinkage) {
        var expWeighted = !!checked;
        var useTarget = !expWeighted && (shrinkage === "ledoit_wolf");
        return [!expWeighted, expWeighted, !useTarget];
    }
    """,
    Output("po-halflife-input", "disabled"),
    Output("po-cov-shrinkage-select", "disabled"),
    Output("po-cov-shrinkage-target-select", "disabled"),
    Input("po-exp-wt-cov-switch", "checked"),
    Input("po-cov-shrinkage-select", "value"),
    prevent_initial_call=True,
)


@callback(
    Output("po-portfolio-name-input", "value", allow_duplicate=True),
    Input("po-opt-model-select", "value"),
    prevent_initial_call=True,
)
def po_sync_name_with_model(model):
    return _po_default_name_for_model(model)


# Toggle window params based on model AND window type
clientside_callback(
    """
    function(model, windowType) {
        var isExAnte = (model === "ex_ante_mv" || model === "black_litterman");
        if (isExAnte) {
            // Disable all windowing controls for ex ante models
            return [true, true, true, true, true];
        }
        // For standard models, disable size/step/fill if window type is 'full'
        var isFull = (windowType === "full");
        return [false, isFull, isFull, isFull, isFull];
    }
    """,
    Output("po-opt-window-select", "disabled"),
    Output("po-window-size-input", "disabled"),
    Output("po-opt-step-input", "disabled"),
    Output("po-fill-in-sample-select", "disabled"),
    Output("po-opt-step-unit-select", "disabled"),
    Input("po-opt-model-select", "value"),
    Input("po-opt-window-select", "value"),
    prevent_initial_call=False,
)

# ---------------------------------------------------------------------------
# Ex ante panel visibility: show/hide based on model selection
# ---------------------------------------------------------------------------
clientside_callback(
    """
    function(model) {
        var isExAnte = (model === "ex_ante_mv" || model === "black_litterman");
        var isBL = (model === "black_litterman");
        var styleBlock = {"display": "block"};
        var styleNone = {"display": "none"};
        return [
            isExAnte ? styleBlock : styleNone,  // po-ex-ante-panel
            isExAnte ? styleBlock : styleNone,  // po-objective-container
            isExAnte ? styleBlock : styleNone,  // po-ex-ante-mode-container
            isBL ? styleBlock : styleNone,      // po-bl-views-panel
        ];
    }
    """,
    Output("po-ex-ante-panel", "style"),
    Output("po-objective-container", "style"),
    Output("po-ex-ante-mode-container", "style"),
    Output("po-bl-views-panel", "style"),
    Input("po-opt-model-select", "value"),
    prevent_initial_call=True,
)

# Populate expected returns grid when selected series changes (ex ante models)
@callback(
    Output("po-ex-ante-returns-grid", "rowData", allow_duplicate=True),
    Output("po-ex-ante-returns-grid", "columnDefs"),
    Input("po-series-select", "data"),
    Input("po-ex-ante-mode-store", "data"),
    Input("po-ex-ante-returns-store", "data"),
    Input("po-ex-ante-vol-store", "data"),
    prevent_initial_call=True,
)
def po_populate_returns_grid(selected_series, mode, existing_returns, existing_vol):
    """Populate the expected returns grid with selected series names."""
    if not selected_series:
        return [], []
    
    existing_returns = existing_returns or {}
    existing_vol = existing_vol or {}
    mode = mode or "ret_cov"
    
    # Hide volatility column unless in Vol/Corr mode
    hide_vol = (mode != "ret_vol_corr")
    
    column_defs = [
        {"field": "Asset", "editable": False, "width": 140, "headerClass": "dashmat-center-header"},
        {"field": "Return", "editable": True, "width": 110,
         "type": "numericColumn",
         "valueFormatter": {"function": "d3.format('.2%')(params.value)"},
         "valueParser": {"function": "var v=params.newValue; if (v===null || v===undefined || v==='') return null; var n=Number(v); if (!isFinite(n)) return null; return Math.abs(n) > 1 ? n/100 : n;"},
         "headerClass": "dashmat-center-header"},
        {"field": "Volatility", "editable": True, "width": 110,
         "type": "numericColumn",
         "valueFormatter": {"function": "d3.format('.2%')(params.value)"},
         "valueParser": {"function": "var v=params.newValue; if (v===null || v===undefined || v==='') return null; var n=Number(v); if (!isFinite(n)) return null; return Math.abs(n) > 1 ? n/100 : n;"},
         "hide": hide_vol,
         "headerClass": "dashmat-center-header"},
    ]

    rows = []
    for s in selected_series:
        rows.append({
            "Asset": s,
            "Return": existing_returns.get(s, 0.0),
            "Volatility": existing_vol.get(s, 0.0),
        })
    return rows, column_defs


# Sync returns grid edits to store
@callback(
    Output("po-ex-ante-returns-store", "data"),
    Output("po-ex-ante-vol-store", "data"),
    Input("po-ex-ante-returns-grid", "cellValueChanged"),
    State("po-ex-ante-returns-grid", "rowData"),
    State("po-ex-ante-returns-store", "data"),
    State("po-ex-ante-vol-store", "data"),
    prevent_initial_call=True,
)
def po_sync_returns_grid_to_store(cell_change, row_data, existing_returns, existing_vols):
    """Save grid edits to session store, merging with existing data."""
    if not cell_change or not row_data:
        raise PreventUpdate
    
    def _normalize_percent_input(value):
        """Accept either whole-percent (5) or decimal (0.05) inputs."""
        n = float(value)
        return (n / 100.0) if abs(n) > 1 else n

    returns = existing_returns or {}
    vols = existing_vols or {}
    
    for row in row_data:
        asset = row.get("Asset", "")
        if not asset:
            continue
            
        ret = row.get("Return", 0.0)
        vol = row.get("Volatility", 0.0)
        
        try:
            returns[asset] = _normalize_percent_input(ret)
        except (ValueError, TypeError):
            returns[asset] = 0.0
            
        try:
            vols[asset] = _normalize_percent_input(vol)
        except (ValueError, TypeError):
            vols[asset] = 0.0
    
    return returns, vols


# Upload returns CSV
@callback(
    Output("po-ex-ante-returns-grid", "rowData", allow_duplicate=True),
    Output("po-ex-ante-returns-store", "data", allow_duplicate=True),
    Input("po-ex-ante-returns-upload", "contents"),
    State("po-ex-ante-returns-upload", "filename"),
    prevent_initial_call=True,
)
def po_upload_returns_csv(contents, filename):
    """Parse uploaded CSV into returns grid."""
    if contents is None:
        raise PreventUpdate
    import base64
    _, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    csv_df = pd.read_csv(StringIO(decoded.decode("utf-8")))
    # Expect columns: Asset, Return (or first two columns)
    if len(csv_df.columns) >= 2:
        csv_df.columns = ["Asset", "Return"] + list(csv_df.columns[2:])
    rows = []
    store = {}
    for _, row in csv_df.iterrows():
        asset = str(row.iloc[0])
        try:
            ret = float(row.iloc[1])
        except (ValueError, TypeError):
            ret = 0.0
        rows.append({"Asset": asset, "Return": ret})
        store[asset] = ret
    return rows, store


# Clear returns
@callback(
    Output("po-ex-ante-returns-grid", "rowData", allow_duplicate=True),
    Output("po-ex-ante-returns-store", "data", allow_duplicate=True),
    Output("po-ex-ante-vol-store", "data", allow_duplicate=True),
    Input("po-ex-ante-returns-clear", "n_clicks"),
    State("po-series-select", "data"),
    prevent_initial_call=True,
)
def po_clear_returns(n_clicks, selected_series):
    """Reset returns grid to zeros."""
    if not n_clicks:
        raise PreventUpdate
    rows = [{"Asset": s, "Return": 0.0, "Volatility": 0.0} for s in (selected_series or [])]
    return rows, {}, {}


@callback(
    Output("po-cma-load-modal", "opened"),
    Output("po-cma-load-target-store", "data"),
    Output("po-cma-version-select", "data"),
    Output("po-cma-version-select", "value"),
    Output("po-cma-type-select", "value"),
    Output("po-cma-load-missing-text", "children"),
    Input("po-load-db-returns-btn", "n_clicks"),
    Input("po-load-db-matrix-btn", "n_clicks"),
    State("po-series-select", "data"),
    State("po-cmabench-assignments-store", "data"),
    State("po-ex-ante-mode-store", "data"),
    prevent_initial_call=True,
)
def po_open_cma_load_modal(n_returns, n_matrix, selected_series, cmabench_assignments, mode):
    if not n_returns and not n_matrix:
        raise PreventUpdate

    triggered = callback_context.triggered_id
    target = "matrix" if triggered == "po-load-db-matrix-btn" else "returns"
    versions = get_cma_versions_cached(DB_ENGINE)
    if not versions:
        return True, target, [], None, "hmm", "No CMA data found in local database."

    version_options = [{"value": str(v), "label": str(v)} for v in versions]
    default_version = str(max(versions))
    default_type = "hmm"
    stats_map = _get_cma_stats_map(default_version, default_type)
    corr_map = _get_cma_corr_map(default_version, default_type)
    effective_cmabench = _effective_cmabench_assignments(selected_series, cmabench_assignments)
    selected_cma = _selected_cma_benches(selected_series, effective_cmabench)
    missing = _compute_cma_missing(selected_cma, target, mode, stats_map, corr_map)
    missing_msg = _cma_missing_message(target, missing)
    return True, target, version_options, default_version, default_type, missing_msg


@callback(
    Output("po-cma-load-modal", "opened", allow_duplicate=True),
    Input("po-cma-load-cancel", "n_clicks"),
    prevent_initial_call=True,
)
def po_close_cma_load_modal(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    return False


@callback(
    Output("po-cma-load-missing-text", "children", allow_duplicate=True),
    Input("po-cma-version-select", "value"),
    Input("po-cma-type-select", "value"),
    Input("po-cma-load-target-store", "data"),
    State("po-series-select", "data"),
    State("po-cmabench-assignments-store", "data"),
    State("po-ex-ante-mode-store", "data"),
    prevent_initial_call=True,
)
def po_update_cma_missing_warning(version, cma_type, target, selected_series, cmabench_assignments, mode):
    if version is None or not cma_type:
        return "Select a valid Version and Type."
    try:
        stats_map = _get_cma_stats_map(int(version), cma_type)
        corr_map = _get_cma_corr_map(int(version), cma_type)
        effective_cmabench = _effective_cmabench_assignments(selected_series, cmabench_assignments)
        selected_cma = _selected_cma_benches(selected_series, effective_cmabench)
        missing = _compute_cma_missing(selected_cma, target, mode, stats_map, corr_map)
        return _cma_missing_message(target, missing)
    except Exception:
        return "Unable to query CMA tables. Check database connection/configuration."


@callback(
    Output("po-ex-ante-returns-store", "data", allow_duplicate=True),
    Output("po-ex-ante-vol-store", "data", allow_duplicate=True),
    Output("po-ex-ante-returns-grid", "rowData", allow_duplicate=True),
    Output("po-ex-ante-cov-store", "data", allow_duplicate=True),
    Output("po-ex-ante-corr-store", "data", allow_duplicate=True),
    Output("po-ex-ante-matrix-grid", "rowData", allow_duplicate=True),
    Output("po-cma-load-modal", "opened", allow_duplicate=True),
    Input("po-cma-load-confirm", "n_clicks"),
    State("po-cma-version-select", "value"),
    State("po-cma-type-select", "value"),
    State("po-cma-load-target-store", "data"),
    State("po-series-select", "data"),
    State("po-cmabench-assignments-store", "data"),
    State("po-ex-ante-mode-store", "data"),
    prevent_initial_call=True,
)
def po_load_cma_from_db(n_clicks, version, cma_type, target, selected_series, cmabench_assignments, mode):
    if not n_clicks:
        raise PreventUpdate
    if version is None or not cma_type or not selected_series:
        raise PreventUpdate

    mode = mode or "ret_cov"
    target = target or "returns"

    try:
        stats_map = _get_cma_stats_map(int(version), cma_type)
        corr_map = _get_cma_corr_map(int(version), cma_type)
    except Exception:
        raise PreventUpdate

    effective_cmabench = _effective_cmabench_assignments(selected_series, cmabench_assignments)
    cma_lookup = {s: _resolve_cma_bench(s, effective_cmabench) for s in selected_series}

    if target == "returns":
        returns_dict = {}
        vols_dict = {}
        rows = []
        for s in selected_series:
            stat = stats_map.get(cma_lookup[s], {})
            mean_val = stat.get("Mean", 0.0)
            sd_val = stat.get("SD", 0.0)
            returns_dict[s] = mean_val
            if mode == "ret_vol_corr":
                vols_dict[s] = sd_val
            rows.append({"Asset": s, "Return": mean_val, "Volatility": sd_val})
        return returns_dict, vols_dict, rows, no_update, no_update, no_update, False

    # target == matrix: compute covariance from SD and correlation
    cov_matrix = {}
    corr_matrix = {}
    cov_rows = []
    corr_rows = []
    for r in selected_series:
        cov_matrix[r] = {}
        corr_matrix[r] = {}
        cov_row = {"Asset": r}
        corr_row = {"Asset": r}
        cma_r = cma_lookup[r]
        sd_r = stats_map.get(cma_r, {}).get("SD", np.nan)
        for c in selected_series:
            cma_c = cma_lookup[c]
            sd_c = stats_map.get(cma_c, {}).get("SD", np.nan)
            corr_val = _get_cma_corr_value(corr_map, cma_r, cma_c)
            if pd.isna(sd_r) or pd.isna(sd_c) or pd.isna(corr_val):
                cov_val = np.nan
            else:
                cov_val = float(sd_r) * float(sd_c) * float(corr_val)
            cov_matrix[r][c] = cov_val
            cov_row[c] = cov_val

            if pd.isna(cov_val) or pd.isna(sd_r) or pd.isna(sd_c) or float(sd_r) == 0 or float(sd_c) == 0:
                corr_from_cov = np.nan
            else:
                corr_from_cov = cov_val / (float(sd_r) * float(sd_c))
            corr_matrix[r][c] = corr_from_cov
            corr_row[c] = corr_from_cov
        cov_rows.append(cov_row)
        corr_rows.append(corr_row)

    if mode == "ret_vol_corr":
        return no_update, no_update, no_update, no_update, corr_matrix, corr_rows, False
    return no_update, no_update, no_update, cov_matrix, no_update, cov_rows, False


# Update matrix UI (title, upload button) based on mode
@callback(
    Output("po-ex-ante-matrix-title", "children"),
    Output("po-ex-ante-matrix-upload-btn", "children"),
    Output("po-ex-ante-returns-title", "children"),
    Input("po-ex-ante-mode-store", "data"),
)
def po_update_matrix_ui(mode):
    mode = mode or "ret_cov"
    if mode == "ret_vol_corr":
        return "Correlation Matrix", "Upload Corr CSV", "Expected Returns and Volatility"
    return "Covariance Matrix", "Upload Cov CSV", "Expected Returns"


# Populate matrix grid
@callback(
    Output("po-ex-ante-matrix-grid", "rowData", allow_duplicate=True),
    Output("po-ex-ante-matrix-grid", "columnDefs"),
    Input("po-series-select", "data"),
    Input("po-ex-ante-mode-store", "data"),
    Input("po-ex-ante-cov-store", "data"),
    Input("po-ex-ante-corr-store", "data"),
    prevent_initial_call="initial_duplicate",
)
def po_populate_matrix_grid(selected_series, mode, cov_store, corr_store):
    """Populate the matrix grid structure. Does NOT auto-estimate - use Estimate from Data button."""
    if not selected_series:
        return [], []
    
    mode = mode or "ret_cov"
    is_corr = (mode == "ret_vol_corr")
    
    existing_matrix = corr_store if is_corr else cov_store
    existing_matrix = existing_matrix or {}

    matrix_defs = [{"field": "Asset", "editable": False, "width": 140, "pinned": "left",
                    "valueFormatter": {"function": "params.value"}, "headerClass": "dashmat-center-header"}]
    for s in selected_series:
        matrix_defs.append({
            "field": s,
            "editable": True, 
            "width": 110,
            "type": "numericColumn",
            "valueFormatter": {"function": "params.value !== null && params.value !== undefined && params.value !== '' && isFinite(Number(params.value)) ? d3.format(',.4f')(Number(params.value)) : ''"},
            "headerClass": "dashmat-center-header",
        })

    rows = []
    for r_name in selected_series:
        r_name_str = str(r_name)
        row = {"Asset": r_name_str}
        
        row_vals = existing_matrix.get(r_name_str, {})
        if not row_vals and r_name in existing_matrix:
             row_vals = existing_matrix.get(r_name, {})

        for c_name in selected_series:
            val = row_vals.get(c_name)
            if val is None:
                val = np.nan
            row[c_name] = val
        rows.append(row)
    
    return rows, matrix_defs

# Estimate matrix from data button
@callback(
    Output("po-ex-ante-cov-store", "data", allow_duplicate=True),
    Output("po-ex-ante-corr-store", "data", allow_duplicate=True),
    Output("po-ex-ante-matrix-grid", "rowData", allow_duplicate=True),
    Input("po-estimate-matrix-btn", "n_clicks"),
    State("dashmat-raw-data-store", "data"),
    State("po-series-select", "data"),
    State("po-ex-ante-mode-store", "data"),
    State("po-periodicity-select", "value"),
    State("po-exp-wt-cov-switch", "checked"),
    State("po-halflife-input", "value"),
    State("po-cov-shrinkage-select", "value"),
    State("po-cov-shrinkage-target-select", "value"),
    prevent_initial_call=True,
)
def po_estimate_matrix_store(
    n_clicks,
    data,
    selected_series,
    mode,
    periodicity,
    exp_wt_cov,
    halflife,
    cov_shrinkage,
    cov_shrinkage_target,
):
    if not n_clicks or not data or not selected_series:
        raise PreventUpdate
        
    mode = mode or "ret_cov"
    is_corr = (mode == "ret_vol_corr")
    
    try:
        df = resample_returns_cached(data, periodicity or "daily")
        
        # Calculate Returns
        valid_series = [s for s in selected_series if s in df.columns]
        if not valid_series:
            raise PreventUpdate
            
        sub_df = df[valid_series].dropna()
        
        effective_shrinkage, effective_target = resolve_cov_shrinkage_spec(
            cov_shrinkage,
            cov_shrinkage_target,
            exp_weighted=bool(exp_wt_cov),
        )

        if is_corr:
            cov_df = estimate_covariance_matrix(
                sub_df,
                asset_order=valid_series,
                exp_weighted=bool(exp_wt_cov),
                decay_value=normalize_decay_input(halflife, 63.0),
                shrinkage=effective_shrinkage,
                shrinkage_target=effective_target,
            )
            est_df = (
                covariance_to_correlation(cov_df)
                if exp_wt_cov or effective_shrinkage != "none"
                else sub_df.corr()
            )
        else:
            ann = _annualization_for_periodicity(periodicity)
            est_df = estimate_covariance_matrix(
                sub_df,
                asset_order=valid_series,
                exp_weighted=bool(exp_wt_cov),
                decay_value=normalize_decay_input(halflife, 63.0),
                shrinkage=effective_shrinkage,
                shrinkage_target=effective_target,
                annualization_factor=ann,
            )
            
        # Convert to dict
        matrix = {}
        rows = []
        for r in valid_series:
            matrix[r] = {}
            row = {"Asset": r}
            for c in valid_series:
                val = float(est_df.loc[r, c])
                matrix[r][c] = val
                row[c] = val
            rows.append(row)
            
        if is_corr:
            return no_update, matrix, rows
        else:
            return matrix, no_update, rows
            
    except Exception as e:

        raise PreventUpdate


# Estimate expected returns & vols from data button
@callback(
    Output("po-ex-ante-returns-store", "data", allow_duplicate=True),
    Output("po-ex-ante-vol-store", "data", allow_duplicate=True),
    Output("po-ex-ante-returns-grid", "rowData", allow_duplicate=True),
    Input("po-estimate-returns-btn", "n_clicks"),
    State("dashmat-raw-data-store", "data"),
    State("po-series-select", "data"),
    State("po-periodicity-select", "value"),
    State("po-exp-wt-cov-switch", "checked"),
    State("po-halflife-input", "value"),
    State("po-ex-ante-mode-store", "data"),
    prevent_initial_call=True,
)
def po_estimate_returns_from_data(n_clicks, data, selected_series, periodicity, exp_wt, halflife, mode):
    if not n_clicks or not data or not selected_series:
        raise PreventUpdate

    try:
        df = resample_returns_cached(data, periodicity or "daily")
        valid_series = [s for s in selected_series if s in df.columns]
        if not valid_series:
            raise PreventUpdate

        sub_df = df[valid_series].dropna()

        # Annualization factor
        p = periodicity or "daily"
        if p.startswith("weekly"):
            ann = 52
        elif p == "monthly":
            ann = 12
        else:
            ann = 252

        if exp_wt:
            decay_value = normalize_decay_input(halflife, 63.0)
            mean_returns = sub_df.ewm(**resolve_ewm_params(decay_value)).mean().iloc[-1] * ann
        else:
            mean_returns = sub_df.mean() * ann

        vols = sub_df.std() * (ann ** 0.5)

        returns_dict = {s: float(mean_returns[s]) for s in valid_series}
        vol_dict = {s: float(vols[s]) for s in valid_series}

        mode = mode or "ret_cov"
        rows = []
        for s in valid_series:
            rows.append({
                "Asset": s,
                "Return": returns_dict[s],
                "Volatility": vol_dict[s],
            })

        return returns_dict, vol_dict, rows

    except Exception as e:

        raise PreventUpdate


# Sync matrix grid edits to store
@callback(
    Output("po-ex-ante-cov-store", "data", allow_duplicate=True),
    Output("po-ex-ante-corr-store", "data", allow_duplicate=True),
    Input("po-ex-ante-matrix-grid", "cellValueChanged"),
    State("po-ex-ante-matrix-grid", "rowData"),
    State("po-ex-ante-mode-store", "data"),
    State("po-ex-ante-cov-store", "data"),
    State("po-ex-ante-corr-store", "data"),
    prevent_initial_call=True,
)
def po_sync_matrix_grid(cell_change, row_data, mode, existing_cov, existing_corr):
    if not cell_change or not row_data:
        raise PreventUpdate
    
    mode = mode or "ret_cov"
    is_corr = (mode == "ret_vol_corr")
    
    # Initialize with existing store data
    matrix = (existing_corr if is_corr else existing_cov) or {}
    
    # Update with grid data
    for row in row_data:
        r_name = row.get("Asset")
        if not r_name:
            continue
            
        # Ensure row dict exists in store
        if r_name not in matrix:
            matrix[r_name] = {}
        elif not isinstance(matrix[r_name], dict):
             # Handle legacy/malformed data
             matrix[r_name] = {}

        for k, v in row.items():
            if k == "Asset":
                continue
            try:
                matrix[r_name][k] = float(v)
            except (ValueError, TypeError):
                matrix[r_name][k] = np.nan

    if is_corr:
        return no_update, matrix
    else:
        return matrix, no_update


# Upload covariance/correlation CSV
@callback(
    Output("po-ex-ante-cov-store", "data", allow_duplicate=True),
    Output("po-ex-ante-corr-store", "data", allow_duplicate=True),
    Output("po-ex-ante-matrix-grid", "rowData", allow_duplicate=True),
    Input("po-ex-ante-matrix-upload", "contents"),
    State("po-ex-ante-matrix-upload", "filename"),
    State("po-ex-ante-mode-store", "data"),
    State("po-series-select", "data"),
    prevent_initial_call=True,
)
def po_upload_matrix_csv(contents, filename, mode, selected_series):
    """Parse uploaded matrix CSV into store and update grid."""
    if contents is None:
        raise PreventUpdate
        
    mode = mode or "ret_cov"
    is_corr = (mode == "ret_vol_corr")
    
    import base64
    _, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    try:
        csv_df = pd.read_csv(StringIO(decoded.decode("utf-8")), index_col=0)
    except Exception:
        # Better error handling needed but suppressing for now
        raise PreventUpdate

    # Convert to nested dict {row_name: {col_name: value}}
    matrix = {}
    # Use selected series to filter/fill if possible, or just take CSV as is
    # Ideally we intersect with selected series
    series = selected_series or list(csv_df.index)
    
    for r in series:
        if r not in matrix:
            matrix[r] = {}
        for c in series:
            if r in csv_df.index and c in csv_df.columns:
                try:
                    val = float(csv_df.loc[r, c])
                except:
                    val = 0.0
            else:
                val = 1.0 if (is_corr and r == c) else 0.0
            matrix[r][c] = val
            
    # Also create rowData for grid update
    rows = []
    for r in series:
        row = {"Asset": r}
        for c in series:
            row[c] = matrix[r][c]
        rows.append(row)

    if is_corr:
        return no_update, matrix, rows
    else:
        return matrix, no_update, rows


# Clear matrix
@callback(
    Output("po-ex-ante-cov-store", "data", allow_duplicate=True),
    Output("po-ex-ante-corr-store", "data", allow_duplicate=True),
    Output("po-ex-ante-matrix-grid", "rowData", allow_duplicate=True),
    Input("po-ex-ante-matrix-clear", "n_clicks"),
    State("po-ex-ante-mode-store", "data"),
    State("po-series-select", "data"),
    prevent_initial_call=True,
)
def po_clear_matrix(n_clicks, mode, selected_series):
    if not n_clicks:
        raise PreventUpdate
    
    mode = mode or "ret_cov"
    is_corr = (mode == "ret_vol_corr")
    series = selected_series or []
    
    matrix = {}
    rows = []
    
    for r in series:
        matrix[r] = {}
        row = {"Asset": r}
        for c in series:
            val = 1.0 if (is_corr and r == c) else 0.0
            matrix[r][c] = val
            row[c] = val
        rows.append(row)
        
    if is_corr:
        return no_update, matrix, rows
    else:
        return matrix, no_update, rows


# Add BL view row
@callback(
    Output("po-bl-views-grid", "rowData", allow_duplicate=True),
    Output("po-bl-views-store", "data", allow_duplicate=True),
    Input("po-bl-add-view", "n_clicks"),
    State("po-bl-views-grid", "rowData"),
    prevent_initial_call=True,
)
def po_add_bl_view(n_clicks, current_rows):
    if not n_clicks:
        raise PreventUpdate
    current_rows = current_rows or []
    new_row = {
        "Type": "absolute",
        "Asset": "",
        "Asset_To": "",
        "Return": 0.0,
        "Confidence": 1.0,
    }
    current_rows.append(new_row)
    
    # Also update store to persist draft row
    # Store format: lowercase keys
    store_data = []
    for row in current_rows:
        store_data.append({
            "type": row.get("Type", "absolute"),
            "asset": row.get("Asset", ""),
            "asset_to": row.get("Asset_To", ""),
            "return": float(row.get("Return", 0.0) or 0.0),
            "confidence": float(row.get("Confidence", 1.0) or 1.0),
        })
        
    return current_rows, store_data


# Clear BL views
@callback(
    Output("po-bl-views-grid", "rowData", allow_duplicate=True),
    Output("po-bl-views-store", "data", allow_duplicate=True),
    Input("po-bl-clear-views", "n_clicks"),
    prevent_initial_call=True,
)
def po_clear_bl_views(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    return [], []

# Sync Tau to store
@callback(
    Output("po-bl-tau-store", "data", allow_duplicate=True),
    Input("po-bl-tau-input", "value"),
    prevent_initial_call=True,
)
def po_sync_tau(value):
    return value or 0.05

# Init Tau from store
@callback(
    Output("po-bl-tau-input", "value", allow_duplicate=True),
    Input("po-bl-tau-store", "data"),
    prevent_initial_call="initial_duplicate",
)
def po_init_tau(store_value):
    if store_value is None:
        raise PreventUpdate
    return store_value


# Sync BL views grid to store (on edit)
@callback(
    Output("po-bl-views-store", "data", allow_duplicate=True),
    Input("po-bl-views-grid", "cellValueChanged"),
    State("po-bl-views-grid", "rowData"),
    prevent_initial_call=True,
)
def po_sync_bl_views_to_store(cell_change, row_data):
    if not row_data:
        return []
    
    # Save ALL rows, including incomplete ones (drafts)
    views = []
    for row in row_data:
        view = {
            "type": row.get("Type", "absolute"),
            "asset": row.get("Asset", ""),
            "asset_to": row.get("Asset_To", ""),
            "return": float(row.get("Return", 0.0) or 0.0),
            "confidence": float(row.get("Confidence", 1.0) or 1.0),
        }
        views.append(view)
    return views


# Initialize BL views grid from store (on load or external update)
@callback(
    Output("po-bl-views-grid", "rowData", allow_duplicate=True),
    Input("po-bl-views-store", "data"),
    State("po-bl-views-grid", "rowData"),
    prevent_initial_call="initial_duplicate",
)
def po_init_bl_views_grid(store_data, current_rows):
    if store_data is None:
        raise PreventUpdate
    
    # Check if data actually changed to avoid loop with sync-to-store callback
    # Simple check: store_data is list of dicts with lowercase keys (type, asset, etc.)
    # Grid expects Title Case keys (Type, Asset, etc.)
    
    grid_data = []
    for item in store_data:
        grid_data.append({
            "Type": item.get("type", "absolute"),
            "Asset": item.get("asset", ""),
            "Asset_To": item.get("asset_to", ""),
            "Return": item.get("return", 0.0),
            "Confidence": item.get("confidence", 1.0),
        })
        
    if current_rows == grid_data:
        raise PreventUpdate
        
    return grid_data


# ---------------------------------------------------------------------------
# Linear Constraints Logic
# ---------------------------------------------------------------------------

# Populate Linear Constraints Grid Columns
@callback(
    Output("po-linear-constraints-grid", "columnDefs"),
    Input("po-series-select", "data"),
    prevent_initial_call=True,
)
def po_populate_linear_constraints_columns(selected_series):
    if not selected_series:
        return []
    
    cols = [
        {"field": "Constraint", "editable": True, "width": 120, "headerClass": "dashmat-center-header"},
        {"field": "Min", "editable": True, "width": 90, "type": "numericColumn", 
         "valueFormatter": {"function": "d3.format('.4f')(params.value)"}, "headerClass": "dashmat-center-header"},
        {"field": "Max", "editable": True, "width": 90, "type": "numericColumn", 
         "valueFormatter": {"function": "d3.format('.4f')(params.value)"}, "headerClass": "dashmat-center-header"},
    ]
    
    for s in selected_series:
        cols.append({
            "field": s,
            "editable": True,
            "width": 100,
            "type": "numericColumn",
            "valueFormatter": {"function": "d3.format('.4f')(params.value)"},
            "headerClass": "dashmat-center-header",
        })
        
    return cols


# Add Linear Constraint Row
@callback(
    Output("po-linear-constraints-grid", "rowData", allow_duplicate=True),
    Output("po-linear-constraints-store", "data", allow_duplicate=True),
    Input("po-add-constraint-btn", "n_clicks"),
    State("po-linear-constraints-grid", "rowData"),
    State("po-series-select", "data"),
    prevent_initial_call=True,
)
def po_add_linear_constraint(n_clicks, current_rows, selected_series):
    if not n_clicks:
        raise PreventUpdate
    current_rows = current_rows or []
    
    # Create new row with defaults
    new_row = {
        "Constraint": f"C{len(current_rows)+1}",
        "Min": 0.0,
        "Max": 1.0,
    }
    
    # Set 0.0 for all selected assets
    if selected_series:
        for s in selected_series:
            new_row[s] = 0.0
            
    current_rows.append(new_row)
    return current_rows, current_rows


# Clear Linear Constraints
@callback(
    Output("po-linear-constraints-grid", "rowData", allow_duplicate=True),
    Output("po-linear-constraints-store", "data", allow_duplicate=True),
    Input("po-clear-constraints-btn", "n_clicks"),
    prevent_initial_call=True,
)
def po_clear_linear_constraints(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    return [], []


# Sync Linear Constraints to Store (on edit)
@callback(
    Output("po-linear-constraints-store", "data", allow_duplicate=True),
    Input("po-linear-constraints-grid", "cellValueChanged"),
    State("po-linear-constraints-grid", "rowData"),
    prevent_initial_call=True,
)
def po_sync_linear_constraints_to_store(cell_change, row_data):
    if row_data is None:
        return []
    # Just store raw row data. We'll parse it in optimization.
    return row_data


# Init Linear Constraints from Store (Persistence)
@callback(
    Output("po-linear-constraints-grid", "rowData", allow_duplicate=True),
    Input("po-linear-constraints-store", "data"),
    State("po-linear-constraints-grid", "rowData"),
    prevent_initial_call="initial_duplicate",
)
def po_init_linear_constraints_grid(store_data, current_rows):
    if store_data is None:
        raise PreventUpdate
    if current_rows == store_data:
        raise PreventUpdate
    return store_data

# Toggle portfolio selector visibility based on active tab
clientside_callback(
    """
    function(tab) {
        return [{}, {}, {display: "none"}];
    }
    """,
    Output("po-weight-portfolio-select", "style"),
    Output("po-delete-portfolio-button", "style"),
    Output("po-growth-multiselect-wrapper", "style"),
    Input("po-vis-tabs", "value"),
    prevent_initial_call=True,
)

# Open progress modal instantly when Run is clicked
clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks) {
            return true;
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("po-progress-modal", "opened"),
    Input("po-run-button", "n_clicks"),
    prevent_initial_call=True,
)

clientside_callback(
    ClientsideFunction(namespace="dashmat_callbacks", function_name="portoptViewSync"),
    Output("po-weight-chart-switch-store", "data"),
    Output("po-weight-grid-container", "style"),
    Output("po-weight-chart-container", "style"),
    Output("po-attribution-chart-switch-store", "data"),
    Output("po-attribution-grid-container", "style"),
    Output("po-attribution-chart-container", "style"),
    Output("po-risk-chart-switch-store", "data"),
    Output("po-risk-grid-container", "style"),
    Output("po-risk-chart-container", "style"),
    Output("po-turnover-chart-switch-store", "data"),
    Output("po-turnover-grid-container", "style"),
    Output("po-turnover-chart-container", "style"),
    Output("po-frontier-chart-switch-store", "data"),
    Output("po-frontier-grid-container", "style"),
    Output("po-frontier-chart-container", "style"),
    Input("po-weight-chart-switch", "value"),
    Input("po-attribution-chart-switch", "value"),
    Input("po-risk-chart-switch", "value"),
    Input("po-turnover-chart-switch", "value"),
    Input("po-frontier-chart-switch", "value"),
    prevent_initial_call=True,
)

clientside_callback(
    ClientsideFunction(namespace="dashmat_callbacks", function_name="saveWorkspaceSession"),
    Output("po-save-session-dummy", "data"),
    Input("po-menu-save-session", "n_clicks"),
    prevent_initial_call=True,
)

clientside_callback(
    ClientsideFunction(namespace="dashmat_callbacks", function_name="loadWorkspaceSessionDialog"),
    Output("po-load-session-dummy", "data"),
    Input("po-load-session-upload", "id"),
    Input("po-menu-load-session", "n_clicks"),
    prevent_initial_call=True,
)

clientside_callback(
    ClientsideFunction(namespace="dashmat_callbacks", function_name="loadWorkspaceSession"),
    Output("po-load-session-dummy", "data", allow_duplicate=True),
    Input("po-load-session-upload", "contents"),
    prevent_initial_call=True,
)


# ===========================================================================
# Server-side callbacks
# ===========================================================================

# ---------------------------------------------------------------------------
# Toggle welcome/main visibility.
# Uses a one-shot Interval to guarantee session-storage has hydrated on
# cross-page navigation, plus dashmat-raw-data-store Input for same-page uploads.
# ---------------------------------------------------------------------------

clientside_callback(
    """
    function(data) {
        if (data) {
            return [{display: "none"}, {display: "flex", flexDirection: "column", flex: "1", overflow: "hidden"}];
        }
        return [{display: "block"}, {display: "none"}];
    }
    """,
    Output("po-welcome-screen", "style"),
    Output("po-main-container", "style"),
    Input("dashmat-raw-data-store", "data"),
)

clientside_callback(
    """
    function(n_intervals) {
        const ready = !!(n_intervals && n_intervals >= 1);
        return [ready, ready];
    }
    """,
    Output("po-initial-tab-render-ready-store", "data"),
    Output("po-secondary-restore-ready-store", "data"),
    Input("po-page-load-trigger", "n_intervals"),
)

# ---------------------------------------------------------------------------
# Restore application state when raw data loads
# ---------------------------------------------------------------------------

@callback(
    Output("po-periodicity-select", "data", allow_duplicate=True),
    Output("po-periodicity-select", "value", allow_duplicate=True),
    Output("po-vol-scaler-input", "value"),
    Output("po-series-select", "data"),
    Input("dashmat-raw-data-meta-store", "data"),
    State("po-periodicity-value-store", "data"),
    State("po-series-select-value-store", "data"),
    State("po-vol-scaler-value-store", "data"),
    prevent_initial_call="initial_duplicate",
)
def po_restore_state(raw_meta, stored_periodicity, stored_series, stored_vol):
    if not isinstance(raw_meta, dict) or not raw_meta.get("has_data"):
        raise PreventUpdate
    try:
        columns = raw_meta.get("columns") or []
        if not columns:
            raise PreventUpdate

        periodicity_options = raw_meta.get("periodicity_options") or [{"value": "daily_trading", "label": "Daily (Trading)"}]
        orig_periodicity = raw_meta.get("original_periodicity") or "daily"
        
        # Validate stored values
        valid_periodicity = stored_periodicity
        if valid_periodicity not in [p["value"] for p in periodicity_options]:
            valid_periodicity = "daily_trading" if orig_periodicity == "daily" else (orig_periodicity or "daily_trading")
            
        valid_vol = stored_vol if stored_vol is not None else 0
        
        # Validate series
        current_selection = stored_series or []
        valid_selection = [s for s in current_selection if s in columns]

        return (
            periodicity_options,
            valid_periodicity,
            valid_vol,
            valid_selection,
        )
    except Exception as e:

        # Critical: Do not return defaults on error, as it wipes persistence.
        # Preserve whatever session state exists.
        raise PreventUpdate


clientside_callback(
    """
    function(n, storedTab) {
        if (!n) {
            return window.dash_clientside.no_update;
        }
        const allowed = ["weight", "attribution", "risk", "turnover", "frontier", "statistics", "returns", "rolling", "calendar", "growth", "drawdown"];
        return allowed.includes(storedTab) ? storedTab : 'weight';
    }
    """,
    Output("po-vis-tabs", "value"),
    Input("po-page-load-trigger", "n_intervals"),
    State("po-active-tab-store", "data"),
    prevent_initial_call=True,
)

clientside_callback(
    """
    function(activeTab, attributionLoaded, riskLoaded, frontierLoaded) {
        return [
            !!attributionLoaded || activeTab === "attribution",
            !!riskLoaded || activeTab === "risk",
            !!frontierLoaded || activeTab === "frontier",
        ];
    }
    """,
    Output("po-attribution-tab-loaded-store", "data"),
    Output("po-risk-tab-loaded-store", "data"),
    Output("po-frontier-tab-loaded-store", "data"),
    Input("po-vis-tabs", "value"),
    State("po-attribution-tab-loaded-store", "data"),
    State("po-risk-tab-loaded-store", "data"),
    State("po-frontier-tab-loaded-store", "data"),
    prevent_initial_call=False,
)


clientside_callback(
    """
    function(n, activeTab, weightView, attributionView, riskView, turnoverView) {
        const nu = window.dash_clientside.no_update;
        if (!n) {
            return [nu, nu, nu, nu];
        }
        return [
            activeTab === 'weight' ? (weightView || 'chart') : nu,
            activeTab === 'attribution' ? (attributionView || 'chart') : nu,
            activeTab === 'risk' ? (riskView || 'chart') : nu,
            activeTab === 'turnover' ? (turnoverView || 'chart') : nu,
        ];
    }
    """,
    Output("po-weight-chart-switch", "value", allow_duplicate=True),
    Output("po-attribution-chart-switch", "value", allow_duplicate=True),
    Output("po-risk-chart-switch", "value", allow_duplicate=True),
    Output("po-turnover-chart-switch", "value", allow_duplicate=True),
    Input("po-page-load-trigger", "n_intervals"),
    State("po-active-tab-store", "data"),
    State("po-weight-chart-switch-store", "data"),
    State("po-attribution-chart-switch-store", "data"),
    State("po-risk-chart-switch-store", "data"),
    State("po-turnover-chart-switch-store", "data"),
    prevent_initial_call=True,
)

clientside_callback(
    """
    function(
        rawMeta,
        secondaryReady,
        periodicityValue,
        volScalerValue,
        selectedSeries,
        activeTab,
        optWindow,
        windowSize,
        optStep,
        optStepUnit,
        model,
        name,
        expWt,
        halflife,
        covShrinkage,
        covShrinkageTarget,
        missingData,
        fillInSample
    ) {
        if (!secondaryReady || !rawMeta || !rawMeta.has_data) {
            return false;
        }

        const columns = Array.isArray(rawMeta.columns) ? rawMeta.columns : [];
        if (!columns.length) {
            return false;
        }

        const periodicityOptions = Array.isArray(rawMeta.periodicity_options)
            ? rawMeta.periodicity_options
            : [{value: "daily_trading", label: "Daily (Trading)"}];
        const validPeriodicities = periodicityOptions.map((option) => option.value);
        const allowedTabs = ["weight", "attribution", "risk", "turnover", "frontier", "statistics", "returns", "rolling", "calendar", "growth", "drawdown"];
        const validWindowTypes = ["rolling", "expanding", "full"];
        const validStepUnits = ["months", "periods"];
        const validModels = [
            "risk_parity",
            "factor_risk_parity",
            "hierarchical_risk_parity",
            "hrp",
            "maximize_sharpe",
            "minimize_variance",
            "minimize_cvar",
            "equal_weight",
            "ex_ante_mv",
            "black_litterman",
        ];
        const validShrinkage = ["none", "ledoit_wolf", "oas"];
        const validShrinkageTargets = ["scaled_identity", "constant_correlation"];
        const validMissingData = ["fill_na", "fill_0"];
        const validFillInSample = ["off", "on"];
        const selected = Array.isArray(selectedSeries) ? selectedSeries : [];
        const allSeriesAvailable = selected.every((series) => columns.includes(series));
        const resolvedActiveTab = activeTab || "weight";
        const resolvedWindowType = optWindow || "rolling";
        const resolvedWindowSize = windowSize !== null && windowSize !== undefined ? windowSize : 252;
        const resolvedOptStep = optStep !== null && optStep !== undefined ? optStep : 1;
        const resolvedStepUnit = optStepUnit || "months";
        const resolvedModel = model || "risk_parity";
        const resolvedName = (name && String(name).trim()) ? String(name).trim() : "RP";
        const resolvedExpWt = !!expWt;
        const resolvedHalflife = halflife !== null && halflife !== undefined ? halflife : 63;
        const resolvedCovShrinkage = covShrinkage || "none";
        const resolvedCovShrinkageTarget = covShrinkageTarget || "scaled_identity";
        const resolvedMissingData = missingData || "fill_na";
        const resolvedFillInSample = fillInSample || "off";
        const resolvedVolScaler = volScalerValue !== null && volScalerValue !== undefined ? volScalerValue : 0;
        const numericWindowSize = Number(resolvedWindowSize);
        const numericOptStep = Number(resolvedOptStep);
        const numericHalflife = Number(resolvedHalflife);

        return (
            validPeriodicities.includes(periodicityValue) &&
            Number.isFinite(Number(resolvedVolScaler)) &&
            allSeriesAvailable &&
            allowedTabs.includes(resolvedActiveTab) &&
            validWindowTypes.includes(resolvedWindowType) &&
            Number.isFinite(numericWindowSize) &&
            numericWindowSize >= 2 &&
            Number.isFinite(numericOptStep) &&
            numericOptStep >= 1 &&
            validStepUnits.includes(resolvedStepUnit) &&
            validModels.includes(resolvedModel) &&
            resolvedName.length > 0 &&
            typeof resolvedExpWt === "boolean" &&
            Number.isFinite(numericHalflife) &&
            numericHalflife > 0 &&
            validShrinkage.includes(resolvedCovShrinkage) &&
            validShrinkageTargets.includes(resolvedCovShrinkageTarget) &&
            validMissingData.includes(resolvedMissingData) &&
            validFillInSample.includes(resolvedFillInSample)
        );
    }
    """,
    Output("po-restore-complete-store", "data"),
    Input("dashmat-raw-data-meta-store", "data"),
    Input("po-secondary-restore-ready-store", "data"),
    Input("po-periodicity-select", "value"),
    Input("po-vol-scaler-input", "value"),
    Input("po-series-select", "data"),
    Input("po-vis-tabs", "value"),
    Input("po-opt-window-select", "value"),
    Input("po-window-size-input", "value"),
    Input("po-opt-step-input", "value"),
    Input("po-opt-step-unit-select", "value"),
    Input("po-opt-model-select", "value"),
    Input("po-portfolio-name-input", "value"),
    Input("po-exp-wt-cov-switch", "checked"),
    Input("po-halflife-input", "value"),
    Input("po-cov-shrinkage-select", "value"),
    Input("po-cov-shrinkage-target-select", "value"),
    Input("po-missing-data-select", "value"),
    Input("po-fill-in-sample-select", "value"),
    prevent_initial_call=False,
)


# ---------------------------------------------------------------------------
# Restore optimization controls from stores on page load
# ---------------------------------------------------------------------------

clientside_callback(
    """
    function(ready, optWindow, windowSize, optStep, optStepUnit, model, name, expWt, halflife, covShrinkage, covShrinkageTarget, missing, fillIS) {
        const nu = window.dash_clientside.no_update;
        if (!ready) {
            return [nu, nu, nu, nu, nu, nu, nu, nu, nu, nu, nu, nu, nu, nu];
        }
        var safeModel = model || "risk_parity";
        var defaults = {
            "risk_parity": "RP",
            "factor_risk_parity": "FRP",
            "hierarchical_risk_parity": "HRP",
            "hrp": "HRP",
            "maximize_sharpe": "MSR",
            "minimize_variance": "MinVar",
            "minimize_cvar": "MinCVaR",
            "equal_weight": "EW",
            "ex_ante_mv": "ExAnteMV",
            "black_litterman": "BL",
        };
        var defaultName = defaults[safeModel] || "Port";
        var shrinkage = covShrinkage || "none";
        var shrinkageTarget = covShrinkageTarget || "scaled_identity";
        var expWeighted = !!expWt;
        var targetDisabled = expWeighted || shrinkage !== "ledoit_wolf";
        return [
            optWindow,
            windowSize,
            optStep,
            optStepUnit,
            safeModel,
            name || defaultName,
            expWeighted,
            halflife,
            shrinkage,
            shrinkageTarget,
            !expWeighted,
            targetDisabled,
            missing,
            fillIS
        ];
    }
    """,
    Output("po-opt-window-select", "value"),
    Output("po-window-size-input", "value", allow_duplicate=True),
    Output("po-opt-step-input", "value", allow_duplicate=True),
    Output("po-opt-step-unit-select", "value"),
    Output("po-opt-model-select", "value"),
    Output("po-portfolio-name-input", "value"),
    Output("po-exp-wt-cov-switch", "checked"),
    Output("po-halflife-input", "value", allow_duplicate=True),
    Output("po-cov-shrinkage-select", "value"),
    Output("po-cov-shrinkage-target-select", "value"),
    Output("po-halflife-input", "disabled", allow_duplicate=True),
    Output("po-cov-shrinkage-target-select", "disabled", allow_duplicate=True),
    Output("po-missing-data-select", "value"),
    Output("po-fill-in-sample-select", "value"),
    Input("po-secondary-restore-ready-store", "data"),
    State("po-opt-window-store", "data"),
    State("po-window-size-store", "data"),
    State("po-opt-step-store", "data"),
    State("po-opt-step-unit-store", "data"),
    State("po-opt-model-store", "data"),
    State("po-portfolio-name-store", "data"),
    State("po-exp-wt-cov-store", "data"),
    State("po-halflife-store", "data"),
    State("po-cov-shrinkage-store", "data"),
    State("po-cov-shrinkage-target-store", "data"),
    State("po-missing-data-store", "data"),
    State("po-fill-in-sample-store", "data"),
    prevent_initial_call=True,
)


# ---------------------------------------------------------------------------
# Restore ex ante controls from stores on page load
# ---------------------------------------------------------------------------

clientside_callback(
    """
    function(ready, mode, objective) {
        if (!ready) {
            return [window.dash_clientside.no_update, window.dash_clientside.no_update];
        }
        return [mode || "ret_cov", objective || "maximize_sharpe"];
    }
    """,
    Output("po-ex-ante-mode-select", "value"),
    Output("po-objective-select", "value"),
    Input("po-secondary-restore-ready-store", "data"),
    State("po-ex-ante-mode-store", "data"),
    State("po-objective-store", "data"),
    prevent_initial_call=True,
)


# ---------------------------------------------------------------------------
# Run button enable/disable
# ---------------------------------------------------------------------------

@callback(
    Output("po-run-button", "disabled"),
    Output("po-run-button-tooltip", "label"),
    Output("po-run-button-tooltip", "disabled"),
    Output("po-menu-save-session", "disabled"),
    Output("po-menu-download-excel", "disabled"),
    Input("po-restore-complete-store", "data"),
    Input("po-portfolio-name-input", "value"),
    Input("po-series-select", "data"),
    Input("po-opt-model-select", "value"),
    Input("po-opt-window-select", "value"),
    Input("po-window-size-input", "value"),
    Input("po-opt-step-input", "value"),
    Input("po-opt-step-unit-select", "value"),
    Input("po-exp-wt-cov-switch", "checked"),
    Input("po-halflife-input", "value"),
    Input("po-cov-shrinkage-select", "value"),
    Input("po-cov-shrinkage-target-select", "value"),
    Input("po-min-wt-store", "data"),
    Input("po-max-wt-store", "data"),
    Input("po-force-max-store", "data"),
    Input("po-linear-constraints-store", "data"),
    Input("po-ex-ante-mode-store", "data"),
    Input("po-ex-ante-returns-store", "data"),
    Input("po-ex-ante-cov-store", "data"),
    Input("po-ex-ante-vol-store", "data"),
    Input("po-ex-ante-corr-store", "data"),
    Input("po-bl-views-store", "data"),
    Input("po-bl-tau-input", "value"),
    Input("po-welcome-screen", "style"),
    Input("po-results-store", "data"),
)
def po_toggle_ui_elements(
    restore_complete,
    name,
    selected,
    opt_model,
    opt_window,
    window_size,
    opt_step,
    opt_step_unit,
    exp_wt_cov,
    halflife,
    cov_shrinkage,
    cov_shrinkage_target,
    min_wt,
    max_wt,
    force_max,
    linear_constraints,
    ex_ante_mode,
    ex_ante_returns,
    ex_ante_cov,
    ex_ante_vol,
    ex_ante_corr,
    bl_views,
    bl_tau,
    welcome_style,
    results_data,
):
    save_disabled = not (welcome_style and welcome_style.get("display") == "none")
    download_disabled = not bool(results_data and len(results_data) > 0)
    if not restore_complete:
        return True, "Loading controls...", False, save_disabled, download_disabled

    validation_error = _validate_optimization_inputs(
        portfolio_name=name,
        selected_series=selected,
        opt_model=opt_model,
        opt_window=opt_window,
        window_size=window_size,
        opt_step=opt_step,
        opt_step_unit=opt_step_unit,
        exp_wt_cov=exp_wt_cov,
        halflife=halflife,
        cov_shrinkage=cov_shrinkage,
        cov_shrinkage_target=cov_shrinkage_target,
        min_wt=min_wt,
        max_wt=max_wt,
        force_max=force_max,
        linear_constraints=linear_constraints,
        ex_ante_mode=ex_ante_mode,
        ex_ante_returns=ex_ante_returns,
        ex_ante_cov=ex_ante_cov,
        ex_ante_vol=ex_ante_vol,
        ex_ante_corr=ex_ante_corr,
        bl_views=bl_views,
        bl_tau=bl_tau,
    )
    run_disabled = validation_error is not None
    tooltip_label = validation_error or "Run optimization."

    return run_disabled, tooltip_label, False, save_disabled, download_disabled


# ---------------------------------------------------------------------------
# Update default window/step/halflife when periodicity changes
# ---------------------------------------------------------------------------

@callback(
    Output("po-window-size-input", "value"),
    Output("po-opt-step-input", "value"),
    Output("po-halflife-input", "value"),
    Input("po-periodicity-select", "value"),
    State("po-opt-step-unit-select", "value"),
    State("po-window-size-store", "data"),
    State("po-opt-step-store", "data"),
    State("po-halflife-store", "data"),
    prevent_initial_call=True,
)
def po_update_periodicity_defaults(periodicity, unit, stored_ws, stored_step, stored_hl):
    # Always apply periodicity defaults when periodicity changes:
    # weekly -> 52, monthly -> 12, daily -> 252.
    ws, step_periods, step_months, hl = _periodicity_defaults(periodicity)
    step = step_months if unit == "months" else step_periods
    return ws, step, hl


@callback(
    Output("po-opt-step-input", "value", allow_duplicate=True),
    Input("po-opt-step-unit-select", "value"),
    State("po-periodicity-select", "value"),
    State("po-opt-step-store", "data"),
    prevent_initial_call=True,
)
def po_update_opt_step_on_unit_change(unit, periodicity, stored_step):
    # On initial restore, preserve stored opt step value
    if stored_step is not None:
        ws, step_p, step_m, hl = _periodicity_defaults(periodicity)
        step_default = step_m if unit == "months" else step_p
        if stored_step != step_default:
            return stored_step
    ws, step_periods, step_months, hl = _periodicity_defaults(periodicity)
    return step_months if unit == "months" else step_periods


# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------

@callback(
    Output("dashmat-raw-data-store", "data", allow_duplicate=True),
    Output("dashmat-original-periodicity-store", "data", allow_duplicate=True),
    Output("po-periodicity-select", "data", allow_duplicate=True),
    Output("po-periodicity-select", "value", allow_duplicate=True),
    Output("po-periodicity-select", "disabled", allow_duplicate=True),
    Output("po-temp-series-select", "data", allow_duplicate=True),
    Output("po-alert-message", "children", allow_duplicate=True),
    Output("po-alert-message", "color", allow_duplicate=True),
    Output("po-alert-message", "hide", allow_duplicate=True),
    Output("po-periodicity-value-store", "data", allow_duplicate=True),
    Output("po-series-selection-modal", "opened", allow_duplicate=True),
    Output("po-temp-benchmark-assignments-store", "data", allow_duplicate=True),
    Output("po-temp-cmabench-assignments-store", "data", allow_duplicate=True),
    Output("po-temp-long-short-store", "data", allow_duplicate=True),
    Output("po-temp-series-order-store", "data", allow_duplicate=True),
    Output("po-temp-deleted-series-store", "data", allow_duplicate=True),
    Output("po-temp-vol-scaling-assignments-store", "data", allow_duplicate=True),
    Output("po-temp-min-wt-store", "data", allow_duplicate=True),
    Output("po-temp-max-wt-store", "data", allow_duplicate=True),
    Output("po-temp-force-max-store", "data", allow_duplicate=True),
    Output("po-db-add-modal", "opened", allow_duplicate=True),
    Output("po-db-add-series-select", "value", allow_duplicate=True),
    Input("po-db-add-ok-button", "n_clicks"),
    State("po-db-add-series-select", "value"),
    State("dashmat-raw-data-store", "data"),
    State("dashmat-original-periodicity-store", "data"),
    State("po-series-select", "data"),
    State("po-benchmark-assignments-store", "data"),
    State("po-cmabench-assignments-store", "data"),
    State("po-long-short-store", "data"),
    State("po-series-order-store", "data"),
    State("po-vol-scaling-assignments-store", "data"),
    State("po-min-wt-store", "data"),
    State("po-max-wt-store", "data"),
    State("po-force-max-store", "data"),
    prevent_initial_call=True,
)
def po_add_series_from_database(
    n_clicks,
    selected_benches,
    existing_data,
    existing_periodicity,
    current_selection,
    current_bench,
    current_cmabench,
    current_ls,
    current_order,
    current_vol_scaling,
    current_min_wt,
    current_max_wt,
    current_force_max,
):
    if not n_clicks:
        raise PreventUpdate

    n_no = no_update
    if not selected_benches:
        return (
            n_no, n_no, n_no, n_no, n_no, n_no,
            "Select at least one series from the database.",
            "orange",
            False,
            n_no, n_no, n_no, n_no, n_no,
            n_no, n_no, n_no, n_no, n_no, n_no,
            True, n_no,
        )

    try:
        if existing_data:
            existing_cols = set(json_to_df(existing_data).columns)
            duplicates = [s for s in selected_benches if s in existing_cols]
            if duplicates:
                return (
                    n_no, n_no, n_no, n_no, n_no, n_no,
                    f"Cannot add duplicate series: {', '.join(duplicates)}",
                    "red",
                    False,
                    n_no, n_no, n_no, n_no, n_no,
                    n_no, n_no, n_no, n_no, n_no, n_no,
                    True, n_no,
                )

        new_df, db_meta = load_cma_returns_for_benches_with_meta(
            DB_ENGINE, selected_benches, MRD_ENGINE
        )
        if new_df.empty:
            raise ValueError("No rows returned for selected FOFBench values.")

        # Treat imports as daily when any selected series has a daily phase.
        # This mirrors raw factor/fund/performance import behavior.
        new_periodicity = "daily"
        any_daily_phase = False
        all_start_daily = True
        daily_transition_notes: list[str] = []
        for series_name in new_df.columns:
            meta = db_meta.get(series_name, {}) if isinstance(db_meta, dict) else {}
            starts_daily = bool(meta.get("starts_daily", True))
            daily_start_date = meta.get("daily_start_date")
            has_daily_phase = bool(daily_start_date) or starts_daily
            any_daily_phase = any_daily_phase or has_daily_phase
            if not starts_daily:
                all_start_daily = False
                if daily_start_date:
                    daily_transition_notes.append(f"{series_name}: {daily_start_date}")
                elif not has_daily_phase:
                    daily_transition_notes.append(f"{series_name}: no daily phase detected")
                else:
                    daily_transition_notes.append(f"{series_name}: daily phase starts after initial history")
        if not any_daily_phase:
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
            # Keep data in daily-capable form, but default selection to monthly
            # when any imported series starts in monthly history.
            default_periodicity = "daily_trading" if all_start_daily else "monthly"
        else:
            default_periodicity = combined_periodicity

        new_series = [col for col in new_df.columns if col not in (current_selection or [])]
        updated_selection = (current_selection or []) + new_series

        alert_msg = f"Loaded {len(new_df.columns)} series with {len(new_df)} rows from database"
        if daily_transition_notes:
            alert_msg = f"{alert_msg}. Series become daily on: {'; '.join(daily_transition_notes)}"

        return (
            df_to_json(merged_df),
            combined_periodicity,
            periodicity_options,
            default_periodicity,
            False,
            updated_selection,
            alert_msg,
            "orange" if daily_transition_notes else "green",
            False,
            default_periodicity,
            True,
            current_bench or {},
            current_cmabench or {},
            current_ls or {},
            current_order or [],
            [],
            current_vol_scaling or {},
            current_min_wt or {},
            current_max_wt or {},
            current_force_max or {},
            False,
            [],
        )
    except Exception as e:
        return (
            n_no, n_no, n_no, n_no, n_no, n_no,
            f"Error loading database series: {str(e)}",
            "red",
            False,
            n_no, n_no, n_no, n_no, n_no,
            n_no, n_no, n_no, n_no, n_no, n_no,
            True, n_no,
        )


@callback(
    Output("dashmat-raw-data-store", "data", allow_duplicate=True),
    Output("dashmat-original-periodicity-store", "data", allow_duplicate=True),
    Output("po-periodicity-select", "data", allow_duplicate=True),
    Output("po-periodicity-select", "value", allow_duplicate=True),
    Output("po-periodicity-select", "disabled", allow_duplicate=True),
    Output("po-temp-series-select", "data", allow_duplicate=True),
    Output("po-alert-message", "children", allow_duplicate=True),
    Output("po-alert-message", "color", allow_duplicate=True),
    Output("po-alert-message", "hide", allow_duplicate=True),
    Output("po-periodicity-value-store", "data", allow_duplicate=True),
    Output("po-series-selection-modal", "opened", allow_duplicate=True),
    Output("po-temp-benchmark-assignments-store", "data", allow_duplicate=True),
    Output("po-temp-cmabench-assignments-store", "data", allow_duplicate=True),
    Output("po-temp-long-short-store", "data", allow_duplicate=True),
    Output("po-temp-series-order-store", "data", allow_duplicate=True),
    Output("po-temp-deleted-series-store", "data", allow_duplicate=True),
    Output("po-temp-vol-scaling-assignments-store", "data", allow_duplicate=True),
    Output("po-temp-min-wt-store", "data", allow_duplicate=True),
    Output("po-temp-max-wt-store", "data", allow_duplicate=True),
    Output("po-temp-force-max-store", "data", allow_duplicate=True),
    Output("po-raw-db-add-modal", "opened", allow_duplicate=True),
    Output("po-raw-db-add-rows-store", "data", allow_duplicate=True),
    Output("po-raw-db-add-grid", "rowData", allow_duplicate=True),
    Output("po-raw-db-add-error-alert", "children", allow_duplicate=True),
    Output("po-raw-db-add-error-alert", "hide", allow_duplicate=True),
    Output("po-raw-db-preview-lines", "children", allow_duplicate=True),
    Input("po-raw-db-add-ok-button", "n_clicks"),
    State("po-raw-db-add-mode-store", "data"),
    State("po-raw-db-add-rows-store", "data"),
    State("dashmat-raw-data-store", "data"),
    State("dashmat-original-periodicity-store", "data"),
    State("po-series-select", "data"),
    State("po-benchmark-assignments-store", "data"),
    State("po-cmabench-assignments-store", "data"),
    State("po-long-short-store", "data"),
    State("po-series-order-store", "data"),
    State("po-vol-scaling-assignments-store", "data"),
    State("po-min-wt-store", "data"),
    State("po-max-wt-store", "data"),
    State("po-force-max-store", "data"),
    prevent_initial_call=True,
)
def po_add_raw_series_from_database(
    n_clicks,
    mode,
    staged_rows,
    existing_data,
    existing_periodicity,
    current_selection,
    current_bench,
    current_cmabench,
    current_ls,
    current_order,
    current_vol_scaling,
    current_min_wt,
    current_max_wt,
    current_force_max,
):
    if not n_clicks:
        raise PreventUpdate

    n_no = no_update
    rows = [dict(r) for r in (staged_rows or []) if isinstance(r, dict)]
    mode_key = str(mode or "").strip().lower()
    if mode_key not in {"factor", "funds", "performance"} or not rows:
        return (
            n_no, n_no, n_no, n_no, n_no, n_no,
            n_no, n_no, n_no,
            n_no, n_no, n_no, n_no, n_no, n_no, n_no,
            n_no, n_no, n_no, n_no,
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
                    n_no, n_no, n_no, n_no, n_no, n_no,
                    n_no, n_no, n_no,
                    n_no, n_no, n_no, n_no, n_no, n_no, n_no,
                    n_no, n_no, n_no, n_no,
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
            current_cmabench or {},
            current_ls or {},
            current_order or [],
            [],
            current_vol_scaling or {},
            current_min_wt or {},
            current_max_wt or {},
            current_force_max or {},
            False,
            [],
            [],
            no_update,
            True,
            "Select a series to preview option-adjusted results (first 6 rows).",
        )
    except Exception as e:
        return (
            n_no, n_no, n_no, n_no, n_no, n_no,
            n_no, n_no, n_no,
            n_no, n_no, n_no, n_no, n_no, n_no, n_no,
            n_no, n_no, n_no, n_no,
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
    Output("po-periodicity-select", "data", allow_duplicate=True),
    Output("po-periodicity-select", "value", allow_duplicate=True),
    Output("po-periodicity-select", "disabled", allow_duplicate=True),
    Output("po-temp-series-select", "data", allow_duplicate=True),
    Output("po-alert-message", "children", allow_duplicate=True),
    Output("po-alert-message", "color", allow_duplicate=True),
    Output("po-alert-message", "hide", allow_duplicate=True),
    Output("po-periodicity-value-store", "data", allow_duplicate=True),
    Output("po-series-selection-modal", "opened", allow_duplicate=True),
    Output("po-temp-benchmark-assignments-store", "data", allow_duplicate=True),
    Output("po-temp-cmabench-assignments-store", "data", allow_duplicate=True),
    Output("po-temp-long-short-store", "data", allow_duplicate=True),
    Output("po-temp-series-order-store", "data", allow_duplicate=True),
    Output("po-temp-deleted-series-store", "data", allow_duplicate=True),
    Output("po-temp-vol-scaling-assignments-store", "data", allow_duplicate=True),
    Output("po-temp-min-wt-store", "data", allow_duplicate=True),
    Output("po-temp-max-wt-store", "data", allow_duplicate=True),
    Output("po-temp-force-max-store", "data", allow_duplicate=True),
    Output("po-underlying-add-modal", "opened", allow_duplicate=True),
    Output("po-underlying-add-rows-store", "data", allow_duplicate=True),
    Output("po-underlying-add-grid", "rowData", allow_duplicate=True),
    Output("po-underlying-add-error-alert", "children", allow_duplicate=True),
    Output("po-underlying-add-error-alert", "hide", allow_duplicate=True),
    Input("po-underlying-add-ok-button", "n_clicks"),
    State("po-underlying-add-rows-store", "data"),
    State("dashmat-raw-data-store", "data"),
    State("dashmat-original-periodicity-store", "data"),
    State("po-series-select", "data"),
    State("po-benchmark-assignments-store", "data"),
    State("po-cmabench-assignments-store", "data"),
    State("po-long-short-store", "data"),
    State("po-series-order-store", "data"),
    State("po-vol-scaling-assignments-store", "data"),
    State("po-min-wt-store", "data"),
    State("po-max-wt-store", "data"),
    State("po-force-max-store", "data"),
    prevent_initial_call=True,
)
def po_add_underlying_categories_from_database(
    n_clicks,
    staged_rows,
    existing_data,
    existing_periodicity,
    current_selection,
    current_bench,
    current_cmabench,
    current_ls,
    current_order,
    current_vol_scaling,
    current_min_wt,
    current_max_wt,
    current_force_max,
):
    if not n_clicks:
        raise PreventUpdate

    n_no = no_update
    rows = [dict(row) for row in (staged_rows or []) if isinstance(row, dict)]
    if not rows:
        return (
            n_no, n_no, n_no, n_no, n_no, n_no,
            n_no, n_no, n_no,
            n_no, n_no, n_no, n_no, n_no, n_no, n_no,
            n_no, n_no, n_no, n_no,
            True,
            rows,
            rows,
            "Stage at least one underlying category row before importing.",
            False,
        )

    try:
        load_result = load_underlying_category_series(DB_ENGINE, rows)
        new_df = load_result.returns_df
        if new_df.empty:
            raise ValueError("No rows returned for staged underlying category requests.")

        if existing_data:
            existing_cols = set(json_to_df(existing_data).columns)
            duplicates = [series_name for series_name in new_df.columns if series_name in existing_cols]
            if duplicates:
                duplicate_text = f"Cannot add duplicate series: {', '.join(duplicates)}"
                return (
                    n_no, n_no, n_no, n_no, n_no, n_no,
                    n_no, n_no, n_no,
                    n_no, n_no, n_no, n_no, n_no, n_no, n_no,
                    n_no, n_no, n_no, n_no,
                    True,
                    rows,
                    rows,
                    duplicate_text,
                    False,
                )

        merge_result = _shared_merge_uploaded_with_existing(existing_data, existing_periodicity, new_df)
        merged_df = merge_result.merged_df
        combined_periodicity = merge_result.combined_periodicity
        periodicity_options = merge_result.periodicity_options
        default_periodicity = merge_result.default_periodicity
        imported_df = merge_result.imported_df

        new_series = [col for col in imported_df.columns if col not in (current_selection or [])]
        updated_selection = (current_selection or []) + new_series

        return (
            df_to_json(merged_df),
            combined_periodicity,
            periodicity_options,
            default_periodicity,
            False,
            updated_selection,
            f"Loaded {len(imported_df.columns)} series with {len(imported_df)} rows from underlying categories.",
            "green",
            False,
            default_periodicity,
            True,
            current_bench or {},
            current_cmabench or {},
            current_ls or {},
            current_order or [],
            [],
            current_vol_scaling or {},
            current_min_wt or {},
            current_max_wt or {},
            current_force_max or {},
            False,
            [],
            [],
            no_update,
            True,
        )
    except Exception as exc:
        error_text = f"Error loading underlying category series: {exc}"
        return (
            n_no, n_no, n_no, n_no, n_no, n_no,
            n_no, n_no, n_no,
            n_no, n_no, n_no, n_no, n_no, n_no, n_no,
            n_no, n_no, n_no, n_no,
            True,
            rows,
            rows,
            error_text,
            False,
        )


@callback(
    Output("dashmat-raw-data-store", "data", allow_duplicate=True),
    Output("dashmat-original-periodicity-store", "data", allow_duplicate=True),
    Output("po-periodicity-select", "data", allow_duplicate=True),
    Output("po-periodicity-select", "value", allow_duplicate=True),
    Output("po-periodicity-select", "disabled", allow_duplicate=True),
    Output("po-temp-series-select", "data", allow_duplicate=True),
    Output("po-alert-message", "children", allow_duplicate=True),
    Output("po-alert-message", "color", allow_duplicate=True),
    Output("po-alert-message", "hide", allow_duplicate=True),
    Output("po-periodicity-value-store", "data", allow_duplicate=True),
    Output("po-series-selection-modal", "opened", allow_duplicate=True),
    Output("po-temp-benchmark-assignments-store", "data", allow_duplicate=True),
    Output("po-temp-cmabench-assignments-store", "data", allow_duplicate=True),
    Output("po-temp-long-short-store", "data", allow_duplicate=True),
    Output("po-temp-series-order-store", "data", allow_duplicate=True),
    Output("po-temp-deleted-series-store", "data", allow_duplicate=True),
    Output("po-temp-vol-scaling-assignments-store", "data", allow_duplicate=True),
    Output("po-temp-min-wt-store", "data", allow_duplicate=True),
    Output("po-temp-max-wt-store", "data", allow_duplicate=True),
    Output("po-temp-force-max-store", "data", allow_duplicate=True),
    Output("po-portfolio-add-modal", "opened", allow_duplicate=True),
    Output("po-portfolio-add-rows-store", "data", allow_duplicate=True),
    Output("po-portfolio-add-grid", "rowData", allow_duplicate=True),
    Output("po-portfolio-add-error-alert", "children", allow_duplicate=True),
    Output("po-portfolio-add-error-alert", "hide", allow_duplicate=True),
    Input("po-portfolio-add-ok-button", "n_clicks"),
    State("po-portfolio-add-mode-store", "data"),
    State("po-portfolio-add-rows-store", "data"),
    State("dashmat-raw-data-store", "data"),
    State("dashmat-original-periodicity-store", "data"),
    State("po-series-select", "data"),
    State("po-benchmark-assignments-store", "data"),
    State("po-cmabench-assignments-store", "data"),
    State("po-long-short-store", "data"),
    State("po-series-order-store", "data"),
    State("po-vol-scaling-assignments-store", "data"),
    State("po-min-wt-store", "data"),
    State("po-max-wt-store", "data"),
    State("po-force-max-store", "data"),
    prevent_initial_call=True,
)
def po_add_portfolios_from_database(
    n_clicks,
    mode,
    staged_rows,
    existing_data,
    existing_periodicity,
    current_selection,
    current_bench,
    current_cmabench,
    current_ls,
    current_order,
    current_vol_scaling,
    current_min_wt,
    current_max_wt,
    current_force_max,
):
    if not n_clicks:
        raise PreventUpdate

    n_no = no_update
    rows = [r for r in (staged_rows or []) if isinstance(r, dict)]
    if mode not in {"peer", "index", "other"} or not rows:
        return (
            n_no, n_no, n_no, n_no, n_no, n_no,
            "Stage at least one portfolio row before importing.",
            "orange",
            False,
            n_no, n_no, n_no, n_no, n_no, n_no,
            n_no, n_no, n_no, n_no, n_no,
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
                    n_no, n_no, n_no, n_no, n_no, n_no,
                    f"Cannot add duplicate series: {', '.join(duplicates)}",
                    "red",
                    False,
                    n_no, n_no, n_no, n_no, n_no, n_no,
                    n_no, n_no, n_no, n_no, n_no,
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
            current_cmabench or {},
            current_ls or {},
            current_order or [],
            [],
            current_vol_scaling or {},
            current_min_wt or {},
            current_max_wt or {},
            current_force_max or {},
            False,
            [],
            [],
            no_update,
            True,
        )
    except Exception as e:
        return (
            n_no, n_no, n_no, n_no, n_no, n_no,
            f"Error loading portfolio series: {str(e)}",
            "red",
            False,
            n_no, n_no, n_no, n_no, n_no, n_no,
            n_no, n_no, n_no, n_no, n_no,
            True,
            rows,
            rows,
            f"Error loading portfolio series: {str(e)}",
            False,
        )


@callback(
    Output("dashmat-raw-data-store", "data", allow_duplicate=True),
    Output("dashmat-original-periodicity-store", "data", allow_duplicate=True),
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
    Output("po-temp-cmabench-assignments-store", "data", allow_duplicate=True),
    Output("po-temp-long-short-store", "data", allow_duplicate=True),
    Output("po-temp-series-order-store", "data", allow_duplicate=True),
    Output("po-temp-deleted-series-store", "data", allow_duplicate=True),
    Output("po-temp-vol-scaling-assignments-store", "data", allow_duplicate=True),
    Output("po-temp-min-wt-store", "data", allow_duplicate=True),
    Output("po-temp-max-wt-store", "data", allow_duplicate=True),
    Output("po-temp-force-max-store", "data", allow_duplicate=True),
    # Sheet-select modal outputs
    Output("po-sheet-select-modal", "opened", allow_duplicate=True),
    Output("po-sheet-select-dropdown", "data", allow_duplicate=True),
    Output("po-sheet-select-dropdown", "value", allow_duplicate=True),
    Output("po-sheet-select-contents-store", "data", allow_duplicate=True),
    Output("po-sheet-select-filename-store", "data", allow_duplicate=True),
    Output("po-sheet-select-sheetnames-store", "data", allow_duplicate=True),
    # Blocker outputs
    Output("po-ui-blocker-store", "data", allow_duplicate=True),
    Input("po-upload-data", "contents"),
    State("po-upload-data", "filename"),
    State("dashmat-raw-data-store", "data"),
    State("dashmat-original-periodicity-store", "data"),
    State("po-series-select", "data"),
    State("po-benchmark-assignments-store", "data"),
    State("po-cmabench-assignments-store", "data"),
    State("po-long-short-store", "data"),
    State("po-series-order-store", "data"),
    State("po-vol-scaling-assignments-store", "data"),
    State("po-min-wt-store", "data"),
    State("po-max-wt-store", "data"),
    State("po-force-max-store", "data"),
    prevent_initial_call=True,
)
def po_handle_upload(contents, filename, existing_data, existing_periodicity,
                     current_selection, current_bench, current_cmabench, current_ls, current_order,
                     current_vol_scaling, current_min_wt, current_max_wt, current_force_max):
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
                n_no, n_no, n_no, n_no, n_no, n_no,
                True, dropdown_data, [sheet_names[0]], contents, filename, sheet_names,  # open sheet modal
                False,  # hide blocker
            )

        new_df = _shared_import_single_upload(contents, filename)
        merge_result = _shared_merge_uploaded_with_existing(existing_data, existing_periodicity, new_df)
        merged_df = merge_result.merged_df
        combined_periodicity = merge_result.combined_periodicity
        periodicity_options = merge_result.periodicity_options
        default_periodicity = merge_result.default_periodicity
        imported_df = merge_result.imported_df

        new_series = [col for col in imported_df.columns if col not in (current_selection or [])]
        updated_selection = (current_selection or []) + new_series

        alert_msg = f"Loaded {len(imported_df.columns)} series with {len(imported_df)} rows from {filename}"

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
            current_cmabench or {},
            current_ls or {},
            current_order or [],
            [],
            current_vol_scaling or {},
            current_min_wt or {},
            current_max_wt or {},
            current_force_max or {},
            *sheet_no,
            True,  # keep blocker until series-selection grid renders
        )
    except Exception as e:
        return (
            n_no, n_no, n_no, n_no, n_no, n_no,
            f"Error loading file: {str(e)}", "red", False,
            n_no, n_no, n_no, n_no, n_no, n_no, n_no, n_no, n_no, n_no, n_no,
            *sheet_no,
            False,  # hide blocker
        )


# ---------------------------------------------------------------------------
# Sheet selection modal: confirm
# ---------------------------------------------------------------------------
@callback(
    Output("dashmat-raw-data-store", "data", allow_duplicate=True),
    Output("dashmat-original-periodicity-store", "data", allow_duplicate=True),
    Output("po-periodicity-select", "data", allow_duplicate=True),
    Output("po-periodicity-select", "value", allow_duplicate=True),
    Output("po-periodicity-select", "disabled", allow_duplicate=True),
    Output("po-temp-series-select", "data", allow_duplicate=True),
    Output("po-alert-message", "children", allow_duplicate=True),
    Output("po-alert-message", "color", allow_duplicate=True),
    Output("po-alert-message", "hide", allow_duplicate=True),
    Output("po-periodicity-value-store", "data", allow_duplicate=True),
    Output("po-series-selection-modal", "opened", allow_duplicate=True),
    Output("po-temp-benchmark-assignments-store", "data", allow_duplicate=True),
    Output("po-temp-cmabench-assignments-store", "data", allow_duplicate=True),
    Output("po-temp-long-short-store", "data", allow_duplicate=True),
    Output("po-temp-series-order-store", "data", allow_duplicate=True),
    Output("po-temp-deleted-series-store", "data", allow_duplicate=True),
    Output("po-temp-vol-scaling-assignments-store", "data", allow_duplicate=True),
    Output("po-temp-min-wt-store", "data", allow_duplicate=True),
    Output("po-temp-max-wt-store", "data", allow_duplicate=True),
    Output("po-temp-force-max-store", "data", allow_duplicate=True),
    Output("po-sheet-select-modal", "opened", allow_duplicate=True),
    Output("po-sheet-select-contents-store", "data", allow_duplicate=True),
    Output("po-sheet-select-filename-store", "data", allow_duplicate=True),
    Output("po-sheet-select-sheetnames-store", "data", allow_duplicate=True),
    Output("po-upload-data", "contents", allow_duplicate=True),
    # Blocker outputs
    Output("po-ui-blocker-store", "data", allow_duplicate=True),
    Input("po-sheet-select-ok-button", "n_clicks"),
    Input("po-sheet-select-import-all-button", "n_clicks"),
    State("po-sheet-select-dropdown", "value"),
    State("po-sheet-select-contents-store", "data"),
    State("po-sheet-select-filename-store", "data"),
    State("po-sheet-select-sheetnames-store", "data"),
    State("dashmat-raw-data-store", "data"),
    State("dashmat-original-periodicity-store", "data"),
    State("po-series-select", "data"),
    State("po-benchmark-assignments-store", "data"),
    State("po-cmabench-assignments-store", "data"),
    State("po-long-short-store", "data"),
    State("po-series-order-store", "data"),
    State("po-vol-scaling-assignments-store", "data"),
    State("po-min-wt-store", "data"),
    State("po-max-wt-store", "data"),
    State("po-force-max-store", "data"),
    prevent_initial_call=True,
)
def po_on_sheet_select_ok(n_clicks_selected, n_clicks_all, selected_sheets, stashed_contents, stashed_filename, stashed_sheet_names,
                          existing_data, existing_periodicity, current_selection,
                          current_bench, current_cmabench, current_ls, current_order,
                          current_vol_scaling, current_min_wt, current_max_wt, current_force_max):
    """Parse selected sheet(s) and complete the import."""
    if not stashed_contents:
        raise PreventUpdate

    n_no = no_update
    triggered_id = callback_context.triggered_id
    if triggered_id not in {"po-sheet-select-ok-button", "po-sheet-select-import-all-button"}:
        raise PreventUpdate

    try:
        workbook_sheets = stashed_sheet_names or get_sheet_names(stashed_contents, stashed_filename)
        if triggered_id == "po-sheet-select-import-all-button":
            target_sheets = workbook_sheets
        else:
            target_sheets = selected_sheets or []

        if not target_sheets:
            return (
                n_no, n_no, n_no, n_no, n_no, n_no,
                "Select at least one sheet to import.", "red", False,
                n_no, n_no, n_no, n_no, n_no, n_no, n_no, n_no, n_no, n_no, n_no,
                True, stashed_contents, stashed_filename, workbook_sheets, n_no,  # keep modal open and stash
                False,  # hide blocker
            )

        new_df, imported_sheets = _po_import_selected_workbook_sheets(
            stashed_contents, stashed_filename, target_sheets, workbook_sheets=workbook_sheets
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

        if len(imported_sheets) == 1:
            sheet_msg = f"sheet: {imported_sheets[0]}"
        else:
            sheet_msg = f"{len(imported_sheets)} sheets"
        alert_msg = (
            f"Loaded {len(imported_df.columns)} series with {len(imported_df)} rows "
            f"from {filename} ({sheet_msg})"
        )

        return (
            df_to_json(merged_df),
            combined_periodicity,
            periodicity_options,
            default_periodicity,
            False,
            updated_selection,
            alert_msg, "green", False,
            default_periodicity,
            True,  # open series-selection modal
            current_bench or {},
            current_cmabench or {},
            current_ls or {},
            current_order or [],
            [],
            current_vol_scaling or {},
            current_min_wt or {},
            current_max_wt or {},
            current_force_max or {},
            False, None, None, None, None,  # close sheet modal, clear stash, reset upload
            True,  # keep blocker until series-selection grid renders
        )
    except Exception as e:
        return (
            n_no, n_no, n_no, n_no, n_no, n_no,
            f"Error loading file: {str(e)}", "red", False,
            n_no, n_no, n_no, n_no, n_no, n_no, n_no, n_no, n_no, n_no, n_no,
            False, None, None, None, None,  # close sheet modal, clear stash, reset upload
            False,  # hide blocker
        )


@callback(
    Output("po-sheet-select-ok-button", "disabled"),
    Input("po-sheet-select-dropdown", "value"),
)
def po_toggle_sheet_select_import_selected_disabled(selected_sheets):
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
    Output("po-ui-blocker-store", "data", allow_duplicate=True),
    Input("po-sheet-select-ok-button", "n_clicks"),
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
    Output("po-ui-blocker-store", "data", allow_duplicate=True),
    Input("po-sheet-select-import-all-button", "n_clicks"),
    prevent_initial_call=True,
)


# ---------------------------------------------------------------------------
# Sheet selection modal: cancel
# ---------------------------------------------------------------------------
@callback(
    Output("po-sheet-select-modal", "opened", allow_duplicate=True),
    Output("po-sheet-select-contents-store", "data", allow_duplicate=True),
    Output("po-sheet-select-filename-store", "data", allow_duplicate=True),
    Output("po-sheet-select-sheetnames-store", "data", allow_duplicate=True),
    Output("po-upload-data", "contents", allow_duplicate=True),
    # Blocker outputs
    Output("po-ui-blocker-store", "data", allow_duplicate=True),
    Input("po-sheet-select-cancel-button", "n_clicks"),
    prevent_initial_call=True,
)
def po_on_sheet_select_cancel(n_clicks):
    """Cancel sheet selection and clear stashed data."""
    if not n_clicks:
        raise PreventUpdate
    return False, None, None, None, None, False


# Clear the file input so the same file can be re-uploaded
clientside_callback(
    """
    function(opened) {
        if (!opened) {
            var el = document.getElementById('po-upload-data');
            if (el) {
                var inp = el.querySelector('input[type="file"]');
                if (inp) inp.value = '';
            }
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("po-sheet-select-modal", "title", allow_duplicate=True),
    Input("po-sheet-select-modal", "opened"),
    prevent_initial_call=True,
)


# ---------------------------------------------------------------------------
# Series selection modal: open
# ---------------------------------------------------------------------------

def _po_get_modal_series_state(raw_meta, current_select, current_order, po_origin_series):
    columns = []
    if isinstance(raw_meta, dict):
        maybe_columns = raw_meta.get("columns")
        if isinstance(maybe_columns, list):
            columns = maybe_columns
    elif isinstance(raw_meta, (list, tuple)):
        columns = list(raw_meta)
    if not columns:
        return [], [], []

    selected_valid = [series for series in (current_select or []) if series in columns]
    known_columns = set(series for series in (current_order or []) if series in columns)
    known_columns.update(selected_valid)
    po_origin_set = {series for series in saved_series_store_names(po_origin_series) if series in columns}
    generic_new = [
        series for series in columns
        if series not in known_columns and series not in po_origin_set
    ]
    return columns, selected_valid, generic_new


@callback(
    Output("po-series-selection-modal", "opened", allow_duplicate=True),
    Output("po-temp-series-select", "data", allow_duplicate=True),
    Output("po-temp-benchmark-assignments-store", "data", allow_duplicate=True),
    Output("po-temp-cmabench-assignments-store", "data", allow_duplicate=True),
    Output("po-temp-long-short-store", "data", allow_duplicate=True),
    Output("po-temp-series-order-store", "data", allow_duplicate=True),
    Output("po-temp-deleted-series-store", "data", allow_duplicate=True),
    Output("po-temp-vol-scaling-assignments-store", "data", allow_duplicate=True),
    Output("po-temp-min-wt-store", "data", allow_duplicate=True),
    Output("po-temp-max-wt-store", "data", allow_duplicate=True),
    Output("po-temp-force-max-store", "data", allow_duplicate=True),
    Output("po-page-visited-store", "data", allow_duplicate=True),
    Output("po-ui-blocker-store", "data", allow_duplicate=True),
    Input("po-open-modal-button", "n_clicks"),
    Input("po-page-load-trigger", "n_intervals"),
    State("po-url-location", "pathname"),
    State("dashmat-raw-data-meta-store", "data"),
    State("po-series-select", "data"),
    State("po-benchmark-assignments-store", "data"),
    State("po-cmabench-assignments-store", "data"),
    State("po-long-short-store", "data"),
    State("po-series-order-store", "data"),
    State("po-vol-scaling-assignments-store", "data"),
    State("po-min-wt-store", "data"),
    State("po-max-wt-store", "data"),
    State("po-force-max-store", "data"),
    State("dashmat-pending-new-series-store", "data"),
    State("po-page-visited-store", "data"),
    prevent_initial_call=True,
)
def po_open_modal(
    n_clicks,
    page_load_intervals,
    pathname,
    raw_meta,
    current_select,
    current_bench,
    current_cmabench,
    current_ls,
    current_order,
    current_vol_scaling,
    current_min_wt,
    current_max_wt,
    current_force_max,
    po_origin_series,
    page_visited,
):
    triggered_id = callback_context.triggered_id
    saved_origin_set = set(saved_series_store_names(po_origin_series))

    if triggered_id == "po-open-modal-button":
        if not n_clicks:
            raise PreventUpdate
        return (
            True,
            current_select,
            current_bench,
            current_cmabench,
            current_ls,
            current_order,
            [],
            current_vol_scaling,
            current_min_wt,
            current_max_wt,
            current_force_max,
            no_update,
            True,
        )

    if triggered_id != "po-page-load-trigger" or page_load_intervals is None:
        raise PreventUpdate

    page_path = str(pathname or "").split("?")[0].rstrip("/") or "/"
    if page_path != "/portopt":
        raise PreventUpdate

    columns, selected_valid, generic_new = _po_get_modal_series_state(
        raw_meta,
        current_select,
        current_order,
        po_origin_series,
    )
    if not columns:
        return (
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            True,
            False,
        )

    if not page_visited and not selected_valid:
        temp_select = [series for series in columns if series not in saved_origin_set]
        should_open = bool(temp_select)
    elif generic_new:
        should_open = True
        selected_set = set(selected_valid)
        selected_set.update(generic_new)
        temp_select = [series for series in columns if series in selected_set]
    else:
        should_open = False
        temp_select = no_update

    if not should_open:
        return (
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            True,
            False,
        )

    return (
        True,
        temp_select,
        current_bench,
        current_cmabench,
        current_ls,
        current_order,
        [],
        current_vol_scaling,
        current_min_wt,
        current_max_wt,
        current_force_max,
        True,
        True,
    )


# ---------------------------------------------------------------------------
# Series selection modal: render rows
# ---------------------------------------------------------------------------

@callback(
    Output("po-series-selection-container", "children"),
    Output("po-temp-series-order-store", "data", allow_duplicate=True),
    Output("po-ui-blocker-store", "data", allow_duplicate=True),
    Input("dashmat-raw-data-store", "data"),
    Input("po-temp-series-select", "data"),
    Input("po-temp-series-order-store", "data"),
    Input("po-temp-deleted-series-store", "data"),
    Input("po-temp-benchmark-assignments-store", "data"),
    Input("po-temp-cmabench-assignments-store", "data"),
    Input("po-temp-long-short-store", "data"),
    Input("po-temp-vol-scaling-assignments-store", "data"),
    Input("po-temp-min-wt-store", "data"),
    Input("po-temp-max-wt-store", "data"),
    Input("po-temp-force-max-store", "data"),
    prevent_initial_call="initial_duplicate",
)
def po_update_series_selectors(
    raw_data,
    selected_series,
    series_order,
    deleted_series,
    current_assignments,
    current_cmabench_assignments,
    long_short_assignments,
    vol_scaling_assignments,
    min_wt,
    max_wt,
    force_max,
):
    if raw_data is None:
        return [dmc.Text("Upload data to select series", size="sm", c="dimmed")], [], False

    df = json_to_df(raw_data)
    all_series = list(df.columns)

    if not all_series:
        return [dmc.Text("Upload data to select series", size="sm", c="dimmed")], [], False

    if not series_order:
        series_order = list(all_series)
    else:
        for s in all_series:
            if s not in series_order:
                series_order.append(s)
        series_order = [s for s in series_order if s in all_series]

    selected_set = set(selected_series or [])
    deleted_set = set(deleted_series or [])
    current_assignments = current_assignments or {}
    current_cmabench_assignments = current_cmabench_assignments or {}
    long_short_assignments = long_short_assignments or {}
    vol_scaling_assignments = vol_scaling_assignments or {}
    min_wt = min_wt or {}
    max_wt = max_wt or {}
    force_max = force_max or {}
    missing_cmabench = [s for s in all_series if not str(current_cmabench_assignments.get(s, "")).strip()]
    core_cmabench_defaults = (
        get_cmabench_map_for_fofbench(DB_ENGINE, missing_cmabench)
        if missing_cmabench
        else {}
    )

    benchmark_values = ["None"] + list(all_series)
    cmabench_values = get_unique_cmabench_values_cached(DB_ENGINE)
    row_data = []
    for series in series_order:
        bench_val = current_assignments.get(series, "None")
        if bench_val not in all_series and bench_val != "None":
            bench_val = "None"
        is_ls = long_short_assignments.get(series, False)
        is_scale_vol = vol_scaling_assignments.get(series, True)
        cmabench_val = current_cmabench_assignments.get(series, core_cmabench_defaults.get(series, ""))
        min_wt_val = min_wt.get(series, 0)
        max_wt_val = max_wt.get(series, 100)
        force_max_val = force_max.get(series, False)
        row_data.append(
            {
                "Series": series,
                "Benchmark": bench_val,
                "CMABench": cmabench_val,
                "LongShort": bool(is_ls),
                "ScaleVol": bool(is_scale_vol),
                "MinWt": min_wt_val,
                "MaxWt": max_wt_val,
                "ForceMax": bool(force_max_val),
                "Delete": series in deleted_set,
            }
        )

    selected_rows = [
        row
        for row in row_data
        if row["Series"] in selected_set and not row["Delete"]
    ]

    grid = dag.AgGrid(
        id="po-series-selection-grid",
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
                "field": "CMABench",
                "editable": True,
                "cellEditor": "agRichSelectCellEditor",
                "cellEditorPopup": True,
                "cellEditorParams": {
                    "values": [""] + sorted(
                        set(cmabench_values).union(
                            {
                                str(v).strip()
                                for v in current_cmabench_assignments.values()
                                if isinstance(v, str) and v.strip()
                            }
                        )
                    ),
                    "allowTyping": True,
                    "filterList": True,
                    "highlightMatch": True,
                },
                "minWidth": 130,
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
                "width": 112,
                "cellClass": "dashmat-series-center-cell",
            },
            {
                "field": "MinWt",
                "headerName": "Min Wt",
                "editable": {"function": "!params.data.ForceMax"},
                "width": 98,
                "valueParser": {"function": "var n=Number(params.newValue); if(!isFinite(n)) return 0; return Math.max(0, Math.min(100, n));"},
                "cellClass": "dashmat-series-center-cell",
                "headerClass": "dashmat-center-header",
            },
            {
                "field": "MaxWt",
                "headerName": "Max Wt",
                "editable": True,
                "width": 98,
                "valueParser": {"function": "var n=Number(params.newValue); if(!isFinite(n)) return 100; return Math.max(0, Math.min(100, n));"},
                "cellClass": "dashmat-series-center-cell",
                "headerClass": "dashmat-center-header",
            },
            {
                "field": "ForceMax",
                "headerName": "Force",
                "editable": True,
                "cellRenderer": "agCheckboxCellRenderer",
                "cellEditor": "agCheckboxCellEditor",
                "width": 70,
                "cellClass": "dashmat-series-center-cell",
            },
            {
                "field": "Delete",
                "editable": True,
                "cellRenderer": "agCheckboxCellRenderer",
                "cellEditor": "agCheckboxCellEditor",
                "width": 74,
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
    return [grid], series_order, no_update


def _po_latest_series_grid_change(cell_change):
    """Normalize AG Grid cellValueChanged payload to the latest dict event."""
    change = cell_change
    if isinstance(change, list):
        change = next((item for item in reversed(change) if isinstance(item, dict)), None)
    return change if isinstance(change, dict) else None


# ---------------------------------------------------------------------------
# Modal: collect benchmark assignments
# ---------------------------------------------------------------------------

@callback(
    Output("po-temp-benchmark-assignments-store", "data"),
    Input("po-series-selection-grid", "cellValueChanged", allow_optional=True),
    State("po-series-selection-grid", "rowData", allow_optional=True),
    State("dashmat-raw-data-store", "data"),
    State("dashmat-raw-data-meta-store", "data"),
    prevent_initial_call=True,
)
def po_update_benchmarks(cell_change, row_data, raw_data, raw_meta):
    change = _po_latest_series_grid_change(cell_change)
    if not change:
        raise PreventUpdate
    if change.get("colId") == "Series":
        raise PreventUpdate
    if raw_data is None or not row_data:
        return {}
    valid_series = set((raw_meta or {}).get("columns") or [])
    if not valid_series:
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


# ---------------------------------------------------------------------------
# Modal: collect CMABench assignments
# ---------------------------------------------------------------------------

@callback(
    Output("po-temp-cmabench-assignments-store", "data"),
    Input("po-series-selection-grid", "cellValueChanged", allow_optional=True),
    State("po-series-selection-grid", "rowData", allow_optional=True),
    prevent_initial_call=True,
)
def po_update_cmabench(cell_change, row_data):
    change = _po_latest_series_grid_change(cell_change)
    if not change:
        raise PreventUpdate
    if change.get("colId") == "Series":
        raise PreventUpdate
    if not row_data:
        return {}
    assignments = {}
    for row in row_data:
        if not isinstance(row, dict):
            continue
        series = row.get("Series")
        val = row.get("CMABench")
        if isinstance(val, str):
            val = val.strip()
        if series and val:
            assignments[series] = val
    return assignments


# ---------------------------------------------------------------------------
# Modal: collect long-short assignments
# ---------------------------------------------------------------------------

@callback(
    Output("po-temp-long-short-store", "data"),
    Input("po-series-selection-grid", "cellValueChanged", allow_optional=True),
    State("po-series-selection-grid", "rowData", allow_optional=True),
    State("dashmat-raw-data-store", "data"),
    State("dashmat-raw-data-meta-store", "data"),
    prevent_initial_call=True,
)
def po_update_ls(cell_change, row_data, raw_data, raw_meta):
    change = _po_latest_series_grid_change(cell_change)
    if not change:
        raise PreventUpdate
    if change.get("colId") == "Series":
        raise PreventUpdate
    if raw_data is None or not row_data:
        return {}
    valid_series = set((raw_meta or {}).get("columns") or [])
    if not valid_series:
        valid_series = set(json_to_df(raw_data).columns)
    assignments = {}
    for row in row_data:
        if not isinstance(row, dict):
            continue
        series = row.get("Series")
        if series and series in valid_series:
            assignments[series] = bool(row.get("LongShort", False))
    return assignments


# ---------------------------------------------------------------------------
# Modal: collect vol scaling assignments
# ---------------------------------------------------------------------------

@callback(
    Output("po-temp-vol-scaling-assignments-store", "data"),
    Input("po-series-selection-grid", "cellValueChanged", allow_optional=True),
    State("po-series-selection-grid", "rowData", allow_optional=True),
    State("dashmat-raw-data-store", "data"),
    State("dashmat-raw-data-meta-store", "data"),
    prevent_initial_call=True,
)
def po_update_vol_scaling(cell_change, row_data, raw_data, raw_meta):
    change = _po_latest_series_grid_change(cell_change)
    if not change:
        raise PreventUpdate
    if change.get("colId") == "Series":
        raise PreventUpdate
    if raw_data is None or not row_data:
        return {}
    valid_series = set((raw_meta or {}).get("columns") or [])
    if not valid_series:
        valid_series = set(json_to_df(raw_data).columns)
    assignments = {}
    for row in row_data:
        if not isinstance(row, dict):
            continue
        series = row.get("Series")
        if series and series in valid_series:
            assignments[series] = bool(row.get("ScaleVol", True))
    return assignments


# ---------------------------------------------------------------------------
# Modal: collect min/max/force_max weights
# ---------------------------------------------------------------------------

@callback(
    Output("po-temp-min-wt-store", "data"),
    Input("po-series-selection-grid", "cellValueChanged", allow_optional=True),
    State("po-series-selection-grid", "rowData", allow_optional=True),
    prevent_initial_call=True,
)
def po_update_min_wt(cell_change, row_data):
    change = _po_latest_series_grid_change(cell_change)
    if not change:
        raise PreventUpdate
    if change.get("colId") == "Series":
        raise PreventUpdate
    if not row_data:
        return {}
    assignments = {}
    for row in row_data:
        if not isinstance(row, dict):
            continue
        series = row.get("Series")
        if not series:
            continue
        if bool(row.get("ForceMax", False)):
            assignments[series] = 0
            continue
        try:
            val = float(row.get("MinWt", 0) or 0)
            assignments[series] = max(0.0, min(100.0, val))
        except (TypeError, ValueError):
            assignments[series] = 0
    return assignments


@callback(
    Output("po-temp-max-wt-store", "data"),
    Input("po-series-selection-grid", "cellValueChanged", allow_optional=True),
    State("po-series-selection-grid", "rowData", allow_optional=True),
    prevent_initial_call=True,
)
def po_update_max_wt(cell_change, row_data):
    change = _po_latest_series_grid_change(cell_change)
    if not change:
        raise PreventUpdate
    if change.get("colId") == "Series":
        raise PreventUpdate
    if not row_data:
        return {}
    assignments = {}
    for row in row_data:
        if not isinstance(row, dict):
            continue
        series = row.get("Series")
        if not series:
            continue
        try:
            val = float(row.get("MaxWt", 100) or 100)
            assignments[series] = max(0.0, min(100.0, val))
        except (TypeError, ValueError):
            assignments[series] = 100
    return assignments


@callback(
    Output("po-temp-force-max-store", "data"),
    Input("po-series-selection-grid", "cellValueChanged", allow_optional=True),
    State("po-series-selection-grid", "rowData", allow_optional=True),
    prevent_initial_call=True,
)
def po_update_force_max(cell_change, row_data):
    change = _po_latest_series_grid_change(cell_change)
    if not change:
        raise PreventUpdate
    if change.get("colId") == "Series":
        raise PreventUpdate
    if not row_data:
        return {}
    assignments = {}
    for row in row_data:
        if not isinstance(row, dict):
            continue
        series = row.get("Series")
        if series:
            assignments[series] = bool(row.get("ForceMax", False))
    return assignments


# ---------------------------------------------------------------------------
# Modal: delete series
# ---------------------------------------------------------------------------

@callback(
    Output("po-temp-deleted-series-store", "data", allow_duplicate=True),
    Input("po-series-selection-grid", "cellValueChanged", allow_optional=True),
    State("po-series-selection-grid", "rowData", allow_optional=True),
    prevent_initial_call=True,
)
def po_delete_series(cell_change, row_data):
    change = _po_latest_series_grid_change(cell_change)
    if not isinstance(change, dict) or change.get("colId") != "Delete":
        raise PreventUpdate

    rows = row_data or []
    deleted = [
        row.get("Series")
        for row in rows
        if isinstance(row, dict) and row.get("Series") and bool(row.get("Delete"))
    ]
    return deleted


# ---------------------------------------------------------------------------
# Modal: save in-grid Series rename
# ---------------------------------------------------------------------------

@callback(
    Output("dashmat-raw-data-store", "data", allow_duplicate=True),
    Output("po-temp-benchmark-assignments-store", "data", allow_duplicate=True),
    Output("po-temp-cmabench-assignments-store", "data", allow_duplicate=True),
    Output("po-temp-long-short-store", "data", allow_duplicate=True),
    Output("po-temp-vol-scaling-assignments-store", "data", allow_duplicate=True),
    Output("po-temp-min-wt-store", "data", allow_duplicate=True),
    Output("po-temp-max-wt-store", "data", allow_duplicate=True),
    Output("po-temp-force-max-store", "data", allow_duplicate=True),
    Output("po-temp-series-select", "data", allow_duplicate=True),
    Output("po-temp-series-order-store", "data", allow_duplicate=True),
    Output("po-series-select-value-store", "data", allow_duplicate=True),
    Input("po-series-selection-grid", "cellValueChanged", allow_optional=True),
    State("dashmat-raw-data-store", "data"),
    State("po-temp-benchmark-assignments-store", "data"),
    State("po-temp-cmabench-assignments-store", "data"),
    State("po-temp-long-short-store", "data"),
    State("po-temp-vol-scaling-assignments-store", "data"),
    State("po-temp-min-wt-store", "data"),
    State("po-temp-max-wt-store", "data"),
    State("po-temp-force-max-store", "data"),
    State("po-temp-series-select", "data"),
    State("po-temp-series-order-store", "data"),
    prevent_initial_call=True,
)
def po_save_edit(
    cell_change,
    raw_data,
    benchmark_assignments,
    cmabench_assignments,
    long_short_assignments,
    vol_scaling_assignments,
    min_wt,
    max_wt,
    force_max,
    selected_series,
    series_order,
):
    change = _po_latest_series_grid_change(cell_change)
    if not isinstance(change, dict) or change.get("colId") != "Series":
        raise PreventUpdate

    old_name = str(change.get("oldValue", "")).strip()
    new_name = str(change.get("newValue", "")).strip()
    if not old_name or not new_name or new_name == old_name:
        raise PreventUpdate

    df = json_to_df(raw_data)
    if old_name not in df.columns or new_name in df.columns:
        raise PreventUpdate
    df = df.rename(columns={old_name: new_name})
    new_raw_data = df_to_json(df)

    def _rename_keys(mapping, rename_values=False):
        mapping = mapping or {}
        updated = {}
        for key, value in mapping.items():
            updated_key = new_name if key == old_name else key
            updated_value = new_name if rename_values and value == old_name else value
            updated[updated_key] = updated_value
        return updated

    new_benchmark_assignments = _rename_keys(benchmark_assignments, rename_values=True)
    new_cmabench_assignments = _rename_keys(cmabench_assignments)
    new_long_short_assignments = _rename_keys(long_short_assignments)
    new_vol_scaling_assignments = _rename_keys(vol_scaling_assignments)
    new_min_wt = _rename_keys(min_wt)
    new_max_wt = _rename_keys(max_wt)
    new_force_max = _rename_keys(force_max)

    selected_series = list(selected_series or [])
    new_series_select = [new_name if s == old_name else s for s in selected_series]

    series_order = list(series_order or list(df.columns))
    new_series_order = [new_name if s == old_name else s for s in series_order]

    return (
        new_raw_data,
        new_benchmark_assignments,
        new_cmabench_assignments,
        new_long_short_assignments,
        new_vol_scaling_assignments,
        new_min_wt,
        new_max_wt,
        new_force_max,
        new_series_select,
        new_series_order,
        new_series_select,
    )


# ---------------------------------------------------------------------------
# Modal: reorder series
# ---------------------------------------------------------------------------

@callback(
    Output("po-temp-series-order-store", "data", allow_duplicate=True),
    Output("po-temp-series-select", "data", allow_duplicate=True),
    Input("po-series-selection-grid", "virtualRowData", allow_optional=True),
    Input("po-series-selection-grid", "selectedRows", allow_optional=True),
    State("po-temp-series-order-store", "data"),
    State("po-temp-series-select", "data"),
    prevent_initial_call=True,
)
def po_reorder_series(virtual_rows, selected_rows, current_order, current_selected):
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
        selected_series = [s for s in ordered_series if s in selected_set]
        # Guard against transient empty selectedRows payloads during grid hydration.
        selected_rows_triggered = any(
            prop.startswith("po-series-selection-grid.selectedRows")
            for prop in triggered_props
        )
        if not selected_series and (current_selected or []) and not selected_rows_triggered:
            selected_fallback = set(current_selected or [])
            selected_series = [s for s in ordered_series if s in selected_fallback]
    else:
        selected_fallback = set(current_selected or [])
        selected_series = [s for s in ordered_series if s in selected_fallback]

    if ordered_series == (current_order or []) and selected_series == (current_selected or []):
        raise PreventUpdate
    return ordered_series, selected_series


# ---------------------------------------------------------------------------
# Modal: OK button
# ---------------------------------------------------------------------------

@callback(
    Output("po-series-select", "data", allow_duplicate=True),
    Output("po-benchmark-assignments-store", "data", allow_duplicate=True),
    Output("po-cmabench-assignments-store", "data", allow_duplicate=True),
    Output("po-long-short-store", "data", allow_duplicate=True),
    Output("po-series-order-store", "data", allow_duplicate=True),
    Output("po-series-selection-modal", "opened", allow_duplicate=True),
    Output("po-series-select-value-store", "data", allow_duplicate=True),
    Output("dashmat-raw-data-store", "data", allow_duplicate=True),
    Output("po-vol-scaling-assignments-store", "data", allow_duplicate=True),
    Output("po-min-wt-store", "data"),
    Output("po-max-wt-store", "data"),
    Output("po-force-max-store", "data"),
    Output("po-results-store", "data", allow_duplicate=True),
    Input("po-modal-ok-button", "n_clicks"),
    State("po-temp-series-select", "data"),
    State("po-temp-benchmark-assignments-store", "data"),
    State("po-temp-cmabench-assignments-store", "data"),
    State("po-temp-long-short-store", "data"),
    State("po-temp-series-order-store", "data"),
    State("po-temp-deleted-series-store", "data"),
    State("dashmat-raw-data-store", "data"),
    State("po-temp-vol-scaling-assignments-store", "data"),
    State("po-temp-min-wt-store", "data"),
    State("po-temp-max-wt-store", "data"),
    State("po-temp-force-max-store", "data"),
    State("po-results-store", "data"),
    prevent_initial_call=True,
)
def po_on_modal_ok(
    n_clicks,
    temp_select,
    temp_bench,
    temp_cmabench,
    temp_ls,
    temp_order,
    temp_deleted,
    raw_data,
    temp_vol_scaling,
    temp_min_wt,
    temp_max_wt,
    temp_force_max,
    current_results,
):
    if not n_clicks:
        raise PreventUpdate

    temp_select = list(temp_select or [])
    if temp_order:
        selected_set = set(temp_select)
        temp_select = [s for s in temp_order if s in selected_set]

    temp_cmabench = temp_cmabench or {}
    series_for_defaults = temp_order if temp_order else temp_select
    missing_cmabench = [s for s in series_for_defaults if not str(temp_cmabench.get(s, "")).strip()]
    if missing_cmabench:
        defaults = get_cmabench_map_for_fofbench(DB_ENGINE, missing_cmabench)
        for s in missing_cmabench:
            mapped = defaults.get(s)
            if mapped:
                temp_cmabench[s] = mapped

    updated_raw_data = raw_data
    updated_results = no_update
    if temp_deleted and raw_data:
        df = json_to_df(raw_data)
        to_drop = [s for s in temp_deleted if s in df.columns]
        if to_drop:
            df = df.drop(columns=to_drop)
            updated_raw_data = df_to_json(df)
            if temp_bench:
                temp_bench = {k: v for k, v in temp_bench.items() if k not in to_drop}
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
                temp_ls = {k: v for k, v in temp_ls.items() if k not in to_drop}
            if temp_cmabench:
                temp_cmabench = {k: v for k, v in temp_cmabench.items() if k not in to_drop}
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
            # Remove deleted portfolios from results store
            if current_results:
                deleted_portfolios = [s for s in to_drop if s in current_results]
                if deleted_portfolios:
                    updated_results = {k: v for k, v in current_results.items()
                                       if k not in deleted_portfolios}

    return (temp_select, temp_bench, temp_cmabench, temp_ls, temp_order, False, temp_select,
            updated_raw_data, temp_vol_scaling, temp_min_wt, temp_max_wt, temp_force_max,
            updated_results)


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
# ---------------------------------------------------------------------------
# Date range initialization
# ---------------------------------------------------------------------------

@callback(
    Output("po-range-candidates-store", "data"),
    Input("dashmat-raw-data-store", "data"),
    Input("po-periodicity-select", "value"),
    Input("po-series-select", "data"),
    prevent_initial_call="initial_duplicate",
)
def po_update_range_candidates(raw_data, periodicity, selected_series):
    return compute_date_range_candidates(
        raw_data or "",
        periodicity or "daily",
        tuple(selected_series or ()),
    )


@callback(
    Output("po-common-daily-candidates-store", "data"),
    Input("dashmat-raw-data-store", "data"),
    Input("po-series-select", "data"),
    prevent_initial_call="initial_duplicate",
)
def po_update_common_daily_candidates(raw_data, selected_series):
    return compute_common_daily_candidates(
        raw_data or "",
        tuple(selected_series or ()),
    )


@callback(
    Output("po-start-date-picker", "value"),
    Output("po-end-date-picker", "value"),
    Output("po-date-picker-wrapper", "style"),
    Output("po-common-range-button", "disabled"),
    Output("po-maximum-range-button", "disabled"),
    Output("po-date-range-store", "data", allow_duplicate=True),
    Input("po-range-candidates-store", "data"),
    State("po-date-range-store", "data"),
    State("po-start-date-picker", "value"),
    State("po-end-date-picker", "value"),
    prevent_initial_call="initial_duplicate",
)
def po_init_date_range(candidates, stored_range, current_start_date, current_end_date):
    disabled_style = {"display": "flex", "opacity": 0.5, "pointerEvents": "none", "alignItems": "flex-start"}
    enabled_style = {"display": "flex", "alignItems": "flex-start"}

    if not isinstance(candidates, dict) or not candidates.get("available_series"):
        return None, None, disabled_style, True, True, None

    try:
        start_date, end_date = resolve_initial_range(candidates, stored_range)
        if not start_date or not end_date:
            return None, None, disabled_style, True, True, None
        next_range = {"start": start_date, "end": end_date}
        start_output = no_update if current_start_date == start_date else start_date
        end_output = no_update if current_end_date == end_date else end_date
        range_output = (
            no_update
            if isinstance(stored_range, dict)
            and stored_range.get("start") == start_date
            and stored_range.get("end") == end_date
            else next_range
        )
        return (
            start_output,
            end_output,
            enabled_style,
            False,
            False,
            range_output,
        )
    except Exception:
        return None, None, disabled_style, True, True, None


clientside_callback(
    ClientsideFunction(namespace="dashmat_callbacks", function_name="commonDailyButtonDisabled"),
    Output("po-common-daily-button", "disabled"),
    Input("po-range-candidates-store", "data"),
    Input("po-common-daily-candidates-store", "data"),
    Input("po-periodicity-select", "data"),
    prevent_initial_call=False,
)


# ---------------------------------------------------------------------------
# Date range buttons
# ---------------------------------------------------------------------------

@callback(
    Output("po-start-date-picker", "value", allow_duplicate=True),
    Output("po-end-date-picker", "value", allow_duplicate=True),
    Output("po-date-range-store", "data"),
    Output("po-periodicity-select", "value", allow_duplicate=True),
    Output("po-periodicity-value-store", "data", allow_duplicate=True),
    Input("po-common-range-button", "n_clicks"),
    Input("po-common-daily-button", "n_clicks"),
    Input("po-maximum-range-button", "n_clicks"),
    State("po-range-candidates-store", "data"),
    State("po-common-daily-candidates-store", "data"),
    prevent_initial_call=True,
)
def po_date_range_buttons(common_clicks, common_daily_clicks, max_clicks, candidates, common_daily_candidates):
    if not isinstance(candidates, dict) or not candidates.get("available_series"):
        raise PreventUpdate
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    try:
        start_date, end_date, force_daily = resolve_button_range(
            candidates,
            button_id,
            common_daily_candidates,
        )
        if not start_date or not end_date:
            raise PreventUpdate

        periodicity_value = "daily_trading" if force_daily else no_update
        return start_date, end_date, {"start": start_date, "end": end_date}, periodicity_value, periodicity_value
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
    Output("dashmat-raw-data-store", "data", allow_duplicate=True),
    Output("po-opt-status-store", "data"),
    Output("dashmat-pending-new-series-store", "data", allow_duplicate=True),
    Input("po-run-button", "n_clicks"),
    State("dashmat-raw-data-store", "data"),
    State("dashmat-original-periodicity-store", "data"),
    State("po-periodicity-select", "value"),
    State("po-series-select", "data"),
    State("po-benchmark-assignments-store", "data"),
    State("po-cmabench-assignments-store", "data"),
    State("po-long-short-store", "data"),
    State("po-date-range-store", "data"),
    State("po-vol-scaler-value-store", "data"),
    State("po-vol-scaling-assignments-store", "data"),
    State("po-min-wt-store", "data"),
    State("po-max-wt-store", "data"),
    State("po-force-max-store", "data"),
    State("po-exp-wt-cov-switch", "checked"),
    State("po-halflife-input", "value"),
    State("po-cov-shrinkage-select", "value"),
    State("po-cov-shrinkage-target-select", "value"),
    State("po-portfolio-name-input", "value"),
    State("po-opt-window-select", "value"),
    State("po-window-size-input", "value"),
    State("po-opt-step-input", "value"),
    State("po-opt-step-unit-select", "value"),
    State("po-opt-model-select", "value"),
    State("po-missing-data-select", "value"),
    State("po-fill-in-sample-select", "value"),
    State("po-results-store", "data"),
    State("dashmat-pending-new-series-store", "data"),
    # Ex ante states
    State("po-ex-ante-returns-store", "data"),
    State("po-ex-ante-cov-store", "data"),
    State("po-bl-views-store", "data"),
    State("po-bl-tau-input", "value"),
    State("po-objective-select", "value"),
    State("po-ex-ante-vol-store", "data"),
    State("po-ex-ante-corr-store", "data"),
    State("po-ex-ante-mode-store", "data"),
    State("po-linear-constraints-store", "data"),
    State("dashmat-saved-series-cache-store", "data"),
    prevent_initial_call=True,
)
def po_run_optimization(n_clicks, raw_data, orig_periodicity, periodicity,
                        selected_series, benchmark_assignments, cmabench_assignments, long_short_assignments,
                        date_range, vol_scaler, vol_scaling_assignments,
                        min_wt, max_wt, force_max, exp_wt_cov, halflife, cov_shrinkage, cov_shrinkage_target,
                        portfolio_name, opt_window, window_size, opt_step,
                        opt_step_unit_value,
                        opt_model, missing_data, fill_in_sample_value, current_results,
                        pending_series,
                        ex_ante_returns, ex_ante_cov, bl_views, bl_tau, objective,
                        ex_ante_vol, ex_ante_corr, ex_ante_mode, linear_constraints,
                        saved_series_store):
    if not n_clicks or not raw_data or not selected_series:
        raise PreventUpdate

    timing_ctx = timed_block(
        "portopt.run_optimization",
        series_count=len(selected_series or ()),
        model=opt_model,
        window_type=opt_window,
    )
    timing_ctx.__enter__()
    try:
        validation_error = _validate_optimization_inputs(
            portfolio_name=portfolio_name,
            selected_series=selected_series,
            opt_model=opt_model,
            opt_window=opt_window,
            window_size=window_size,
            opt_step=opt_step,
            opt_step_unit=opt_step_unit_value,
            exp_wt_cov=exp_wt_cov,
            halflife=halflife,
            cov_shrinkage=cov_shrinkage,
            cov_shrinkage_target=cov_shrinkage_target,
            min_wt=min_wt,
            max_wt=max_wt,
            force_max=force_max,
            linear_constraints=linear_constraints,
            ex_ante_mode=ex_ante_mode,
            ex_ante_returns=ex_ante_returns,
            ex_ante_cov=ex_ante_cov,
            ex_ante_vol=ex_ante_vol,
            ex_ante_corr=ex_ante_corr,
            bl_views=bl_views,
            bl_tau=bl_tau,
        )
        if validation_error:
            return (
                no_update,
                no_update,
                {"status": "error", "name": portfolio_name, "message": validation_error},
                no_update,
            )

        # Compute working returns
        working_bundle = _build_po_working_bundle(
            raw_data,
            periodicity,
            benchmark_assignments,
            long_short_assignments,
            date_range,
            vol_scaler,
            vol_scaling_assignments,
        )
        working_df = _po_get_working_returns(working_bundle, selected_series)

        if working_df.empty:
            return (
                no_update, no_update,
                {"status": "error", "name": portfolio_name, "message": "No data available for selected series."},
                no_update,
            )

        # Filter to only selected series (exclude benchmark columns)
        opt_cols = [s for s in selected_series if s in working_df.columns]
        if len(opt_cols) < 2:
            return (
                no_update,
                no_update,
                {
                    "status": "error",
                    "name": portfolio_name,
                    "message": "Selected series are not available in working returns.",
                },
                no_update,
            )
        opt_df = working_df[opt_cols]
        if opt_df.empty or len(opt_df) < 2:
            return (
                no_update,
                no_update,
                {"status": "error", "name": portfolio_name, "message": "Insufficient data after preprocessing."},
                no_update,
            )

        model_value = opt_model or "risk_parity"
        window_value = opt_window or "full"
        if model_value not in {"ex_ante_mv", "black_litterman"} and window_value in {"rolling", "expanding"}:
            ws = int(_coerce_float(window_size) or 0)
            if ws > len(opt_df):
                return (
                    no_update,
                    no_update,
                    {
                        "status": "error",
                        "name": portfolio_name,
                        "message": (
                            f"Window size ({ws}) exceeds available rows ({len(opt_df)}) "
                            "after filtering."
                        ),
                    },
                    no_update,
                )

        # Re-validate against the exact optimized columns.
        validation_error = _validate_optimization_inputs(
            portfolio_name=portfolio_name,
            selected_series=opt_cols,
            opt_model=model_value,
            opt_window=window_value,
            window_size=window_size,
            opt_step=opt_step,
            opt_step_unit=opt_step_unit_value,
            exp_wt_cov=exp_wt_cov,
            halflife=halflife,
            cov_shrinkage=cov_shrinkage,
            cov_shrinkage_target=cov_shrinkage_target,
            min_wt=min_wt,
            max_wt=max_wt,
            force_max=force_max,
            linear_constraints=linear_constraints,
            ex_ante_mode=ex_ante_mode,
            ex_ante_returns=ex_ante_returns,
            ex_ante_cov=ex_ante_cov,
            ex_ante_vol=ex_ante_vol,
            ex_ante_corr=ex_ante_corr,
            bl_views=bl_views,
            bl_tau=bl_tau,
        )
        if validation_error:
            return (
                no_update,
                no_update,
                {"status": "error", "name": portfolio_name, "message": validation_error},
                no_update,
            )

        effective_cov_shrinkage, effective_cov_target = resolve_cov_shrinkage_spec(
            cov_shrinkage,
            cov_shrinkage_target,
            exp_weighted=bool(exp_wt_cov),
        )

        # Build config
        config = {
            "model": model_value,
            "window_type": window_value,
            "window_size": int(_coerce_float(window_size) or 252),
            "opt_step": int(_coerce_float(opt_step) or 252),
            "opt_step_unit": opt_step_unit_value or "months",
            "exp_wt_cov": bool(exp_wt_cov),
            "halflife": normalize_decay_input(halflife, 63.0),
            "cov_shrinkage": effective_cov_shrinkage,
            "cov_shrinkage_target": effective_cov_target,
            "missing_data": missing_data or "fill_na",
            "fill_in_sample": fill_in_sample_value == "on",
            "selected_series": opt_cols,
            "min_wt": min_wt or {},
            "max_wt": max_wt or {},
            "force_max": force_max or {},
            "periodicity": periodicity or "daily",
        }

        bl_mu_frame = None

        # Add ex ante params if applicable
        if model_value in ("ex_ante_mv", "black_litterman"):
            mode = ex_ante_mode or "ret_cov"
            config["ex_ante_returns"] = ex_ante_returns or {}
            config["ex_ante_vol"] = ex_ante_vol or {}
            config["ex_ante_corr"] = ex_ante_corr or {}
            config["ex_ante_mode"] = mode
            config["objective"] = objective or "maximize_sharpe"

            # If in Vol/Corr mode, ensure we don't pass stale covariance data,
            # so the backend calculates it from Vol + Corr.
            if mode == "ret_vol_corr":
                config["ex_ante_cov"] = {}
            else:
                config["ex_ante_cov"] = ex_ante_cov or {}
        if model_value == "black_litterman":
            # Scale views returns (e.g. 5 -> 0.05) if they come from the UI as percentages
            views_list = bl_views or []
            scaled_views = []
            for v in views_list:
                v_copy = v.copy()
                ret_val = _coerce_float(v_copy.get("return", 0.0))
                if ret_val is None:
                    ret_val = 0.0
                # Accept both decimal and whole-percent entry patterns.
                v_copy["return"] = ret_val / 100.0 if abs(ret_val) > 1 else ret_val
                scaled_views.append(v_copy)
            config["bl_views"] = scaled_views
            config["bl_tau"] = float(bl_tau or 0.05)

            # BL pre-flight: ensure posterior moments can be computed for all optimized assets.
            if config.get("missing_data", "fill_na") == "fill_0":
                bl_data = opt_df.fillna(0)
            else:
                valid_cols = [c for c in opt_cols if not opt_df[c].isna().any()]
                if valid_cols:
                    bl_data = opt_df[valid_cols].copy()
                    for c in opt_cols:
                        if c not in bl_data.columns:
                            bl_data[c] = 0.0
                    bl_data = bl_data[opt_cols]
                else:
                    bl_data = opt_df.fillna(0)
            bl_mu_frame, _, bl_error = _build_black_litterman_mu_cov(bl_data[opt_cols], config, opt_cols)
            if bl_error:
                return (
                    no_update,
                    no_update,
                    {"status": "error", "name": portfolio_name, "message": bl_error},
                    no_update,
                )

        # Add linear constraints to config
        config["linear_constraints"] = linear_constraints or []

        sharpe_target = (
            model_value == "maximize_sharpe"
            or (
                model_value in {"ex_ante_mv", "black_litterman"}
                and (config.get("objective", "") == "maximize_sharpe")
            )
        )
        resolved_rf_context = {
            "rf_annual": 0.0,
            "rf_source": "unused",
            "rf_warning": None,
            "rf_asset": None,
        }
        rf_series_runtime = None
        if sharpe_target:
            expected_mu_annual = None
            ann_factor = _annualization_for_periodicity(periodicity)
            if model_value == "ex_ante_mv":
                expected_mu_annual = {
                    c: float((ex_ante_returns or {}).get(c, 0.0) or 0.0)
                    for c in opt_cols
                }
            elif model_value == "black_litterman" and bl_mu_frame is not None:
                expected_mu_annual = {
                    c: float(bl_mu_frame.iloc[0][c] * ann_factor) for c in opt_cols if c in bl_mu_frame.columns
                }

            resolved_rf_context = _resolve_risk_free_context(
                model=model_value,
                asset_order=opt_cols,
                periodicity=periodicity,
                expected_mu_annual=expected_mu_annual,
                reference_index=opt_df.index,
                saved_series_store=saved_series_store,
                cmabench_assignments=cmabench_assignments,
            )
            if model_value == "maximize_sharpe":
                rf_series_runtime = _risk_free_series_for_periodicity(saved_series_store, periodicity)

        config["risk_free_source"] = resolved_rf_context.get("rf_source")
        config["risk_free_annual_default"] = float(resolved_rf_context.get("rf_annual", 0.0) or 0.0)
        config["risk_free_warning"] = resolved_rf_context.get("rf_warning")

        runtime_config = dict(config)
        runtime_config["risk_free_annual"] = float(resolved_rf_context.get("rf_annual", 0.0) or 0.0)
        runtime_config["risk_free_mode"] = "series" if rf_series_runtime is not None else "fixed_annual"
        runtime_config["risk_free_series"] = rf_series_runtime

        # Run optimization
        run_out = run_portfolio_optimization(opt_df, runtime_config)
        if isinstance(run_out, tuple) and len(run_out) == 3:
            window_results, portfolio_returns, optimization_meta = run_out
        else:
            window_results, portfolio_returns = run_out
            optimization_meta = {}

        # Determine unique portfolio name
        current_results = current_results or {}
        final_name = portfolio_name.strip() or _po_default_name_for_model(model_value)
        existing_df = json_to_df(raw_data)

        # Avoid collisions with existing columns and existing results
        base_name = final_name
        counter = 1
        while final_name in existing_df.columns or final_name in current_results:
            final_name = f"{base_name}_{counter}"
            counter += 1

        # Store results
        window_data = []
        for wr in window_results:
            window_data.append({
                "apply_start": wr.apply_start.isoformat(),
                "apply_end": wr.apply_end.isoformat(),
                "est_start": wr.est_start.isoformat(),
                "est_end": wr.est_end.isoformat(),
                "weights": wr.weights,
            })

        result_entry = {
            "window_weights": window_data,
            "returns_json": portfolio_returns.to_json(date_format="iso"),
            "config": config,
            "saved_series_name": None,
            "risk_free_meta": {
                "source": resolved_rf_context.get("rf_source"),
                "annual": float(resolved_rf_context.get("rf_annual", 0.0) or 0.0),
                "warning": resolved_rf_context.get("rf_warning"),
            },
        }
        try:
            frontier_snapshot = _po_resolve_frontier_snapshot(
                selected_portfolio=final_name,
                portfolio_data=result_entry,
                raw_data=raw_data,
                periodicity=periodicity,
                bench=benchmark_assignments,
                ls=long_short_assignments,
                vol_scaler=vol_scaler,
                vol_scaling=vol_scaling_assignments,
                window_idx=len(window_data) - 1,
                rm="MV",
                linear_constraints=linear_constraints,
                saved_series_store=saved_series_store,
                cmabench_assignments=cmabench_assignments,
                persist_cache=True,
            )
        except Exception:
            frontier_snapshot = None

        current_results[final_name] = result_entry

        warning_parts = []
        if resolved_rf_context.get("rf_warning"):
            warning_parts.append(str(resolved_rf_context.get("rf_warning")))
        if isinstance(optimization_meta, dict) and optimization_meta.get("risk_free_warning"):
            warning_parts.append(str(optimization_meta.get("risk_free_warning")))
        warning_text = " ".join(dict.fromkeys([w for w in warning_parts if w])).strip() or None

        return (
            current_results,
            no_update,
            {"status": "complete", "name": final_name, "warning": warning_text},
            no_update,
        )

    except ValueError as e:
        return (
            no_update, no_update,
            {"status": "error", "name": portfolio_name, "message": str(e)},
            no_update,
        )
    except Exception as e:
        return (
            no_update, no_update,
            {"status": "error", "name": portfolio_name, "message": f"Optimization failed: {str(e)}"},
            no_update,
        )
    finally:
        timing_ctx.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# Completion modal
# ---------------------------------------------------------------------------

@callback(
    Output("po-running-indicator", "style"),
    Output("po-completion-indicator", "style"),
    Output("po-completion-text", "children"),
    Output("po-completion-icon", "icon"),
    Output("po-completion-icon", "color"),
    Output("po-progress-modal", "closeOnClickOutside"),
    Input("po-opt-status-store", "data"),
    prevent_initial_call=True,
)
def po_show_completion(status):
    if not status:
        raise PreventUpdate
    hide = {"display": "none"}
    show = {"display": "block"}
    if status.get("status") == "complete":
        warning = status.get("warning")
        message = f"Portfolio '{status['name']}' created successfully."
        if warning:
            message = f"{message}\nWarning: {warning}"
        return (
            hide, show,
            message,
            "tabler:check", "green",
            True,
        )
    elif status.get("status") == "error":
        return (
            hide, show,
            status.get("message", "An error occurred."),
            "tabler:x", "red",
            True,
        )
    raise PreventUpdate


# Close button in completion state
clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks) { return false; }
        return window.dash_clientside.no_update;
    }
    """,
    Output("po-progress-modal", "opened", allow_duplicate=True),
    Input("po-close-completion-button", "n_clicks"),
    prevent_initial_call=True,
)

# Reset modal to running state when it closes (for next invocation)
clientside_callback(
    """
    function(opened) {
        if (!opened) {
            return [{display: "block"}, {display: "none"}, false];
        }
        return [window.dash_clientside.no_update, window.dash_clientside.no_update,
                window.dash_clientside.no_update];
    }
    """,
    Output("po-running-indicator", "style", allow_duplicate=True),
    Output("po-completion-indicator", "style", allow_duplicate=True),
    Output("po-progress-modal", "closeOnClickOutside", allow_duplicate=True),
    Input("po-progress-modal", "opened"),
    prevent_initial_call=True,
)


# ---------------------------------------------------------------------------
# Update portfolio dropdowns when results change
# ---------------------------------------------------------------------------

@callback(
    Output("po-weight-portfolio-select", "data"),
    Output("po-weight-portfolio-select", "value"),
    Output("po-growth-portfolio-multiselect", "data"),
    Output("po-growth-portfolio-multiselect", "value"),
    Output("po-delete-portfolio-button", "disabled"),
    Input("po-results-store", "data"),
    State("po-weight-portfolio-select", "value"),
    State("po-growth-portfolio-multiselect", "value"),
    prevent_initial_call=True,
)
def po_update_portfolio_dropdowns(results, current_select, current_multi):
    if not results:
        return [], None, [], [], True
    names = list(results.keys())
    options = [{"value": n, "label": n} for n in names]
    # Always select the newest portfolio (last added)
    sel = names[-1] if names else None
    multi = [v for v in (current_multi or []) if v in names]
    newest = names[-1] if names else None
    if newest and newest not in multi:
        multi.append(newest)
    return options, sel, options, multi, not bool(sel)


# ---------------------------------------------------------------------------
# Save portfolio return series
# ---------------------------------------------------------------------------

@callback(
    Output("po-save-series-button", "disabled"),
    Output("po-save-series-status-text", "children"),
    Input("po-weight-portfolio-select", "value"),
    Input("po-results-store", "data"),
    prevent_initial_call=False,
)
def po_sync_save_series_ui(selected_portfolio, results):
    if not selected_portfolio or not results or selected_portfolio not in results:
        return True, ""

    saved_name = ((results or {}).get(selected_portfolio) or {}).get("saved_series_name")
    if not saved_name:
        return False, ""
    return False, f"Saved as {saved_name}."


@callback(
    Output("po-results-store", "data", allow_duplicate=True),
    Output("dashmat-raw-data-store", "data", allow_duplicate=True),
    Output("dashmat-pending-new-series-store", "data", allow_duplicate=True),
    Output("po-save-series-status-text", "children", allow_duplicate=True),
    Input("po-save-series-button", "n_clicks"),
    State("po-weight-portfolio-select", "value"),
    State("po-results-store", "data"),
    State("dashmat-raw-data-store", "data"),
    State("po-periodicity-select", "value"),
    State("dashmat-pending-new-series-store", "data"),
    prevent_initial_call=True,
)
def po_save_series_to_shared_data(
    n_clicks,
    selected_portfolio,
    results,
    raw_data,
    periodicity,
    saved_series_store,
):
    if not n_clicks or not selected_portfolio or not results or selected_portfolio not in results:
        raise PreventUpdate

    entry = dict((results or {}).get(selected_portfolio) or {})
    returns_json = entry.get("returns_json")
    if not returns_json:
        return no_update, no_update, no_update, "No portfolio return series available to save."

    try:
        portfolio_series = pd.read_json(StringIO(returns_json), typ="series")
        portfolio_series.index = pd.to_datetime(portfolio_series.index)
    except Exception as exc:
        return no_update, no_update, no_update, f"Error saving series: {exc}"

    try:
        save_out = save_series_to_raw_data(
            raw_data=raw_data,
            periodicity=((entry.get("config") or {}).get("periodicity") or periodicity or "daily"),
            series=portfolio_series.rename(selected_portfolio),
            base_name=selected_portfolio,
            saved_series_store=saved_series_store,
            origin_page="portopt",
            origin_result=selected_portfolio,
            series_type="portfolio",
            prior_saved_name=entry.get("saved_series_name"),
        )
    except Exception as exc:
        return no_update, no_update, no_update, f"Error saving series: {exc}"

    new_results = dict(results or {})
    entry["saved_series_name"] = save_out["saved_name"]
    new_results[selected_portfolio] = entry

    if save_out["action"] == "overwritten":
        status_text = f"Overwrote shared series {save_out['saved_name']}."
    else:
        status_text = f"Saved as {save_out['saved_name']}."

    return (
        new_results,
        save_out["raw_data"],
        save_out["saved_series_store"],
        status_text,
    )


# ---------------------------------------------------------------------------
# Delete portfolio
# ---------------------------------------------------------------------------

@callback(
    Output("po-results-store", "data", allow_duplicate=True),
    Output("dashmat-raw-data-store", "data", allow_duplicate=True),
    Output("po-weight-portfolio-select", "value", allow_duplicate=True),
    Input("po-delete-portfolio-button", "n_clicks"),
    State("po-weight-portfolio-select", "value"),
    State("po-results-store", "data"),
    State("dashmat-raw-data-store", "data"),
    prevent_initial_call=True,
)
def po_delete_portfolio(n_clicks, selected_portfolio, results, raw_data):
    if not n_clicks or not selected_portfolio or not results:
        raise PreventUpdate
    if selected_portfolio not in results:
        raise PreventUpdate

    # Remove from results
    new_results = {k: v for k, v in results.items() if k != selected_portfolio}

    # Pick next selection
    remaining = list(new_results.keys())
    new_sel = remaining[-1] if remaining else None

    return new_results, no_update, new_sel


# ===========================================================================
# Visualization callbacks
# ===========================================================================

# ---------------------------------------------------------------------------
# Weight chart (stacked area)
# ---------------------------------------------------------------------------

@callback(
    Output("po-weight-chart-content", "children"),
    Input("po-weight-portfolio-select", "value"),
    Input("po-results-store", "data"),
    Input("po-vis-tabs", "value"),
    Input("po-weight-chart-switch", "value"),
    State("global-color-scheme-toggle", "computedColorScheme"),
    Input("po-initial-tab-render-ready-store", "data"),
    prevent_initial_call=True,
)
def po_render_weight_chart(selected_portfolio, results, active_tab, switch_value, theme, initial_tab_ready=True):
    if not _po_tab_render_ready(active_tab, "weight", initial_tab_ready) or switch_value != "chart" or not selected_portfolio or not results:
        return html.Div()
    if selected_portfolio not in results:
        return html.Div()

    portfolio_data = results[selected_portfolio]
    window_weights = portfolio_data.get("window_weights", [])

    if not window_weights:
        return dmc.Text("No weight data available.", c="dimmed")

    timing_ctx = timed_block("portopt.render_weight_chart", portfolio=selected_portfolio, window_count=len(window_weights))
    timing_ctx.__enter__()
    try:
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
            height=420,
            legend={"orientation": "h", "yanchor": "bottom", "y": -0.2},
        )
        apply_chart_theme(fig, theme)

        return dcc.Graph(figure=fig, style={"height": "100%", "width": "100%"})
    finally:
        timing_ctx.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# Growth of $1 chart
# ---------------------------------------------------------------------------

@callback(
    Output("po-growth-chart-container", "children"),
    Input("po-weight-portfolio-select", "value"),
    Input("po-results-store", "data"),
    Input("po-vis-tabs", "value"),
    Input("po-growth-chart-switch", "value"),
    State("dashmat-raw-data-store", "data"),
    State("po-periodicity-select", "value"),
    State("po-benchmark-assignments-store", "data"),
    State("po-long-short-store", "data"),
    State("po-date-range-store", "data"),
    State("po-vol-scaler-value-store", "data"),
    State("po-vol-scaling-assignments-store", "data"),
    State("global-color-scheme-toggle", "computedColorScheme"),
    prevent_initial_call=True,
)
def po_render_growth_chart(
    selected_portfolio,
    results,
    active_tab,
    view_mode,
    raw_data,
    periodicity,
    bench,
    ls,
    date_range,
    vol_scaler,
    vol_scaling,
    theme,
):
    if active_tab != "growth" or not selected_portfolio or not results:
        return html.Div()

    display_df, ordered_cols = _po_build_display_series(
        results,
        selected_portfolio,
        raw_data,
        periodicity,
        bench,
        ls,
        date_range,
        vol_scaler,
        vol_scaling,
    )
    if display_df.empty or not ordered_cols:
        return html.Div()

    growth_df = pd.DataFrame(index=display_df.index)
    for name in ordered_cols:
        returns = display_df[name].dropna()
        if returns.empty:
            continue
        growth_df[name] = (1 + returns).cumprod()
    if growth_df.empty:
        return html.Div()

    if (view_mode or "chart") == "table":
        table_df = growth_df.reset_index()
        table_df["Date"] = pd.to_datetime(table_df.iloc[:, 0]).dt.strftime("%Y-%m-%d")
        table_df = table_df.rename(columns={table_df.columns[0]: "Date"})
        cols = [{"field": "Date", "pinned": "left", "width": 112, "minWidth": 106, "maxWidth": 122}]
        for c in ordered_cols:
            if c in table_df.columns:
                cols.append(
                    {
                        "field": c,
                        "width": 120,
                        "minWidth": 110,
                        "valueFormatter": {"function": "params.value != null ? d3.format('.6f')(params.value) : ''"},
                    }
                )
        return dag.AgGrid(
            className="ag-theme-alpine",
            columnDefs=cols,
            rowData=table_df.to_dict("records"),
            defaultColDef={"resizable": True, "sortable": True, "suppressHeaderMenuButton": True},
            style={"height": "460px", "width": "100%"},
            dashGridOptions={"animateRows": True, "pagination": False, "suppressExcelExport": True, "suppressCsvExport": True},
        )

    fig = go.Figure()
    for name in ordered_cols:
        growth = growth_df[name].dropna() if name in growth_df.columns else pd.Series(dtype=float)
        if growth.empty:
            continue
        fig.add_trace(go.Scatter(
            x=growth.index,
            y=growth.values,
            name=name,
            mode="lines",
        ))

    fig.update_layout(
        title=f"Growth of $1: {selected_portfolio}",
        yaxis_title="Value ($)",
        hovermode="x unified",
        margin={"t": 40, "b": 40, "l": 60, "r": 20},
        height=420,
    )
    apply_chart_theme(fig, theme)

    return dcc.Graph(figure=fig, style={"height": "100%", "width": "100%"})


# ---------------------------------------------------------------------------
# Rolling / Calendar / Drawdown tabs
# ---------------------------------------------------------------------------

@callback(
    Output("po-rolling-return-type-select", "disabled"),
    Output("po-rolling-return-type-select", "style"),
    Input("po-rolling-metric-select", "value"),
    prevent_initial_call=False,
)
def po_toggle_rolling_return_type(metric):
    disabled = (metric or "total_return") != "total_return"
    return disabled, ({} if not disabled else {"opacity": 0.5, "pointerEvents": "none"})


@callback(
    Output("po-rolling-content", "children"),
    Input("po-results-store", "data"),
    Input("po-vis-tabs", "value"),
    Input("po-weight-portfolio-select", "value"),
    Input("po-periodicity-select", "value"),
    Input("po-rolling-window-select", "value"),
    Input("po-rolling-return-type-select", "value"),
    Input("po-rolling-metric-select", "value"),
    Input("po-rolling-chart-switch", "value"),
    State("dashmat-raw-data-store", "data"),
    State("po-benchmark-assignments-store", "data"),
    State("po-long-short-store", "data"),
    State("po-date-range-store", "data"),
    State("po-vol-scaler-value-store", "data"),
    State("po-vol-scaling-assignments-store", "data"),
    State("global-color-scheme-toggle", "computedColorScheme"),
    prevent_initial_call=True,
)
def po_render_rolling(
    results,
    active_tab,
    selected_portfolio,
    periodicity,
    rolling_window,
    return_type,
    metric,
    view_mode,
    raw_data,
    bench,
    ls,
    date_range,
    vol_scaler,
    vol_scaling,
    theme,
):
    if active_tab != "rolling" or not results:
        return html.Div()

    display_df, ordered_cols = _po_build_display_series(
        results,
        selected_portfolio,
        raw_data,
        periodicity,
        bench,
        ls,
        date_range,
        vol_scaler,
        vol_scaling,
    )
    if display_df.empty or not ordered_cols:
        return dmc.Text("No rolling data available.", c="dimmed")

    metric = metric or "total_return"
    rolling_df = calculate_rolling_returns(
        df_to_json(display_df[ordered_cols]),
        periodicity or "daily",
        tuple(ordered_cols),
        "total",
        "{}",
        "{}",
        "null",
        rolling_window or "1y",
        return_type or "annualized",
        metric,
        0,
        "{}",
    )
    if rolling_df.empty:
        return dmc.Text("No rolling values available for selected settings.", c="dimmed")

    if (view_mode or "chart") == "table":
        table_df = rolling_df.reset_index()
        table_df["Date"] = pd.to_datetime(table_df.iloc[:, 0]).dt.strftime("%Y-%m-%d")
        table_df = table_df.rename(columns={table_df.columns[0]: "Date"})
        fmt = ".2%" if metric in {"total_return", "volatility"} else ".4f"
        column_defs = [{"field": "Date", "pinned": "left", "width": 112, "minWidth": 106, "maxWidth": 122}]
        for col in ordered_cols:
            if col in table_df.columns:
                column_defs.append(
                    {
                        "field": col,
                        "valueFormatter": {
                            "function": f"params.value != null ? d3.format('{fmt}')(params.value) : ''"
                        },
                        "width": 120,
                    }
                )
        return dag.AgGrid(
            enableEnterpriseModules=True,
            licenseKey=AG_GRID_LICENSE_KEY,
            className="ag-theme-alpine",
            columnDefs=column_defs,
            rowData=table_df.to_dict("records"),
            defaultColDef={
                "sortable": True,
                "resizable": True,
                "suppressHeaderMenuButton": True,
                "cellStyle": {"textAlign": "center"},
                "headerClass": "dashmat-center-header",
            },
            style={"height": "100%", "width": "100%"},
            dashGridOptions={"animateRows": True, "suppressExcelExport": True, "enableRangeSelection": True, "suppressCsvExport": True},
        )

    fig = go.Figure()
    for col in ordered_cols:
        if col not in rolling_df.columns:
            continue
        s = rolling_df[col].dropna()
        if s.empty:
            continue
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=col))

    fig.update_layout(
        title=f"Rolling {_po_rolling_metric_label(metric)}",
        yaxis_title=_po_rolling_metric_label(metric),
        hovermode="x unified",
        margin={"t": 40, "b": 40, "l": 60, "r": 20},
        height=420,
    )
    fig.update_yaxes(tickformat=_po_rolling_metric_tickformat(metric))
    apply_chart_theme(fig, theme)
    return dcc.Graph(figure=fig, style={"height": "100%", "width": "100%"})


@callback(
    Output("po-calendar-series-select", "disabled"),
    Output("po-calendar-series-select", "data"),
    Output("po-calendar-series-select", "value"),
    Input("po-weight-portfolio-select", "value"),
    Input("po-results-store", "data"),
    Input("po-calendar-view-select", "value"),
    State("po-calendar-series-select", "value"),
    prevent_initial_call=False,
)
def po_sync_calendar_series_select(selected_portfolio, results, view_mode, current_value):
    if not selected_portfolio or not results or selected_portfolio not in results:
        return True, [], None

    config = ((results or {}).get(selected_portfolio) or {}).get("config", {}) or {}
    ordered_cols = [selected_portfolio]
    for name in config.get("selected_series") or []:
        if name and name not in ordered_cols:
            ordered_cols.append(name)

    options = [{"value": c, "label": c} for c in ordered_cols]
    if (view_mode or "annual") != "monthly":
        return True, options, None
    if not ordered_cols:
        return True, [], None
    value = current_value if current_value in ordered_cols else ordered_cols[0]
    return False, options, value


@callback(
    Output("po-calendar-content", "children"),
    Input("po-results-store", "data"),
    Input("po-vis-tabs", "value"),
    Input("po-weight-portfolio-select", "value"),
    Input("po-periodicity-select", "value"),
    Input("po-calendar-view-select", "value"),
    Input("po-calendar-series-select", "value"),
    State("dashmat-raw-data-store", "data"),
    State("po-benchmark-assignments-store", "data"),
    State("po-long-short-store", "data"),
    State("po-date-range-store", "data"),
    State("po-vol-scaler-value-store", "data"),
    State("po-vol-scaling-assignments-store", "data"),
    prevent_initial_call=True,
)
def po_render_calendar(
    results,
    active_tab,
    selected_portfolio,
    periodicity,
    view_mode,
    monthly_series,
    raw_data,
    bench,
    ls,
    date_range,
    vol_scaler,
    vol_scaling,
):
    if active_tab != "calendar" or not results:
        return html.Div()

    display_df, ordered_cols = _po_build_display_series(
        results,
        selected_portfolio,
        raw_data,
        periodicity,
        bench,
        ls,
        date_range,
        vol_scaler,
        vol_scaling,
    )
    if display_df.empty or not ordered_cols:
        return dmc.Text("No calendar data available.", c="dimmed")

    if (view_mode or "annual") == "monthly":
        target_series = monthly_series if monthly_series in ordered_cols else ordered_cols[0]
        monthly_col_defs, monthly_rows = create_monthly_view(
            df_to_json(display_df[ordered_cols]),
            target_series,
            periodicity or "daily",
            periodicity or "daily",
            "total",
            {},
            {},
            tuple(ordered_cols),
            None,
            0,
            {},
        )
        if not monthly_rows:
            return dmc.Text("No complete monthly history available.", c="dimmed")
        return dag.AgGrid(
            enableEnterpriseModules=True,
            licenseKey=AG_GRID_LICENSE_KEY,
            className="ag-theme-alpine",
            columnDefs=monthly_col_defs,
            rowData=monthly_rows,
            defaultColDef={
                "sortable": True,
                "resizable": True,
                "suppressHeaderMenuButton": True,
                "cellStyle": {"textAlign": "center"},
                "headerClass": "dashmat-center-header",
            },
            style={"height": "100%", "width": "100%"},
            dashGridOptions={"animateRows": True, "suppressExcelExport": True, "enableRangeSelection": True, "suppressCsvExport": True},
        )

    cal_df = calculate_calendar_year_returns(
        df_to_json(display_df[ordered_cols]),
        periodicity or "daily",
        periodicity or "daily",
        tuple(ordered_cols),
        "total",
        "{}",
        "{}",
        "null",
        0,
        "{}",
    )
    if cal_df.empty:
        return dmc.Text("No complete calendar years available.", c="dimmed")

    table_df = cal_df.reset_index()
    table_df = table_df.rename(columns={table_df.columns[0]: "Year"})
    table_df["Year"] = table_df["Year"].astype(str)

    column_defs = [{"field": "Year", "pinned": "left", "width": 92}]
    for col in ordered_cols:
        if col in table_df.columns:
            column_defs.append(
                {
                    "field": col,
                    "valueFormatter": {"function": "params.value != null ? d3.format('.2%')(params.value) : ''"},
                    "width": 120,
                }
            )

    return dag.AgGrid(
        enableEnterpriseModules=True,
        licenseKey=AG_GRID_LICENSE_KEY,
        className="ag-theme-alpine",
        columnDefs=column_defs,
        rowData=table_df.to_dict("records"),
        defaultColDef={
            "sortable": True,
            "resizable": True,
            "suppressHeaderMenuButton": True,
            "cellStyle": {"textAlign": "center"},
            "headerClass": "dashmat-center-header",
        },
        style={"height": "100%", "width": "100%"},
        dashGridOptions={"animateRows": True, "suppressExcelExport": True, "enableRangeSelection": True, "suppressCsvExport": True},
    )


@callback(
    Output("po-drawdown-content", "children"),
    Input("po-results-store", "data"),
    Input("po-vis-tabs", "value"),
    Input("po-weight-portfolio-select", "value"),
    Input("po-periodicity-select", "value"),
    Input("po-drawdown-chart-switch", "value"),
    State("dashmat-raw-data-store", "data"),
    State("po-benchmark-assignments-store", "data"),
    State("po-long-short-store", "data"),
    State("po-date-range-store", "data"),
    State("po-vol-scaler-value-store", "data"),
    State("po-vol-scaling-assignments-store", "data"),
    State("global-color-scheme-toggle", "computedColorScheme"),
    prevent_initial_call=True,
)
def po_render_drawdown(
    results,
    active_tab,
    selected_portfolio,
    periodicity,
    view_mode,
    raw_data,
    bench,
    ls,
    date_range,
    vol_scaler,
    vol_scaling,
    theme,
):
    if active_tab != "drawdown" or not results:
        return html.Div()

    display_df, ordered_cols = _po_build_display_series(
        results,
        selected_portfolio,
        raw_data,
        periodicity,
        bench,
        ls,
        date_range,
        vol_scaler,
        vol_scaling,
    )
    if display_df.empty or not ordered_cols:
        return dmc.Text("No drawdown data available.", c="dimmed")

    drawdown_df = calculate_drawdown(
        df_to_json(display_df[ordered_cols]),
        periodicity or "daily",
        tuple(ordered_cols),
        "total",
        "{}",
        "{}",
        "null",
        0,
        "{}",
    )
    if drawdown_df.empty:
        return dmc.Text("No drawdown data available.", c="dimmed")

    if (view_mode or "chart") == "table":
        table_df = drawdown_df.reset_index()
        table_df["Date"] = pd.to_datetime(table_df.iloc[:, 0]).dt.strftime("%Y-%m-%d")
        table_df = table_df.rename(columns={table_df.columns[0]: "Date"})
        column_defs = [{"field": "Date", "pinned": "left", "width": 112, "minWidth": 106, "maxWidth": 122}]
        for col in ordered_cols:
            if col in table_df.columns:
                column_defs.append(
                    {
                        "field": col,
                        "valueFormatter": {"function": "params.value != null ? d3.format('.2%')(params.value) : ''"},
                        "width": 120,
                    }
                )
        return dag.AgGrid(
            enableEnterpriseModules=True,
            licenseKey=AG_GRID_LICENSE_KEY,
            className="ag-theme-alpine",
            columnDefs=column_defs,
            rowData=table_df.to_dict("records"),
            defaultColDef={
                "sortable": True,
                "resizable": True,
                "suppressHeaderMenuButton": True,
                "cellStyle": {"textAlign": "center"},
                "headerClass": "dashmat-center-header",
            },
            style={"height": "100%", "width": "100%"},
            dashGridOptions={"animateRows": True, "suppressExcelExport": True, "enableRangeSelection": True, "suppressCsvExport": True},
        )

    fig = go.Figure()
    for col in ordered_cols:
        if col not in drawdown_df.columns:
            continue
        s = drawdown_df[col].dropna()
        if s.empty:
            continue
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=col, fill="tozeroy"))
    fig.update_layout(
        title="Drawdown",
        yaxis_title="Drawdown",
        hovermode="x unified",
        margin={"t": 40, "b": 40, "l": 60, "r": 20},
        height=420,
    )
    fig.update_yaxes(tickformat=".2%")
    apply_chart_theme(fig, theme)
    return dcc.Graph(figure=fig, style={"height": "100%", "width": "100%"})


# ---------------------------------------------------------------------------
# Attribution chart (monthly stacked bar)
# ---------------------------------------------------------------------------

@callback(
    Output("po-attribution-chart-container", "children"),
    Input("po-weight-portfolio-select", "value"),
    Input("po-results-store", "data"),
    Input("po-vis-tabs", "value"),
    Input("po-attribution-tab-loaded-store", "data"),
    Input("po-attribution-chart-switch", "value"),
    State("dashmat-raw-data-store", "data"),
    State("dashmat-original-periodicity-store", "data"),
    State("po-periodicity-select", "value"),
    State("po-benchmark-assignments-store", "data"),
    State("po-long-short-store", "data"),
    State("po-date-range-store", "data"),
    State("po-vol-scaler-value-store", "data"),
    State("po-vol-scaling-assignments-store", "data"),
    State("global-color-scheme-toggle", "computedColorScheme"),
    prevent_initial_call=True,
)
def po_render_attribution_chart(selected_portfolio, results, active_tab, tab_loaded, switch_value,
                                 raw_data, orig_periodicity, periodicity, bench, ls,
                                 date_range, vol_scaler, vol_scaling, theme):
    if not _po_lazy_tab_render_ready(active_tab, "attribution", tab_loaded) or switch_value != "chart" or not selected_portfolio or not results:
        return html.Div()
    if selected_portfolio not in results:
        return html.Div()

    portfolio_data = results[selected_portfolio]
    window_weights = portfolio_data.get("window_weights", [])
    config = portfolio_data.get("config", {})
    opt_series = config.get("selected_series", [])

    if not window_weights or not opt_series or not raw_data:
        return dmc.Text("No attribution data available.", c="dimmed")
    missing_sources = _po_missing_source_series(results, selected_portfolio, raw_data)
    if missing_sources:
        return dmc.Text(f"Missing source series: {', '.join(missing_sources)}", c="dimmed")

    timing_ctx = timed_block(
        "portopt.render_attribution_chart",
        portfolio=selected_portfolio,
        series_count=len(opt_series),
    )
    timing_ctx.__enter__()
    try:
        # Get the working returns for the component series
        working_bundle = _build_po_working_bundle(
            raw_data, periodicity, bench, ls, date_range, vol_scaler, vol_scaling
        )
        attribution_monthly = _po_get_monthly_attribution(working_bundle, opt_series, window_weights)

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
            height=420,
            legend={"orientation": "h", "yanchor": "bottom", "y": -0.2},
        )
        apply_chart_theme(fig, theme)

        return dcc.Graph(figure=fig, style={"height": "100%", "width": "100%"})

    except Exception:
        return dmc.Text("Error computing attribution.", c="dimmed")
    finally:
        timing_ctx.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# Weight table
# ---------------------------------------------------------------------------

@callback(
    Output("po-weight-grid-content", "children"),
    Input("po-weight-portfolio-select", "value"),
    Input("po-results-store", "data"),
    Input("po-vis-tabs", "value"),
    Input("po-weight-chart-switch", "value"),
    Input("po-initial-tab-render-ready-store", "data"),
    prevent_initial_call=True,
)
def po_render_weight_table(selected_portfolio, results, active_tab, switch_value, initial_tab_ready=True):
    if not _po_tab_render_ready(active_tab, "weight", initial_tab_ready) or switch_value != "table" or not selected_portfolio or not results:
        return html.Div()
    if selected_portfolio not in results:
        return html.Div()

    portfolio_data = results[selected_portfolio]
    window_weights = portfolio_data.get("window_weights", [])

    if not window_weights:
        return html.Div()

    asset_names = list(window_weights[0]["weights"].keys())

    column_defs = [
        {"field": "Apply Start", "pinned": "left", "width": 120},
        {"field": "Apply End", "pinned": "left", "width": 120},
    ]
    for a in asset_names:
        column_defs.append({
            "field": a,
            "valueFormatter": {"function": "params.value != null ? d3.format('.2%')(params.value) : ''"},
            "width": 100,
        })

    row_data = []
    for ww in window_weights:
        row = {
            "Apply Start": pd.Timestamp(ww["apply_start"]).strftime("%Y-%m-%d"),
            "Apply End": pd.Timestamp(ww["apply_end"]).strftime("%Y-%m-%d"),
        }
        for a in asset_names:
            row[a] = ww["weights"].get(a, 0)
        row_data.append(row)

    return _po_build_result_grid("po-weight-grid", column_defs, row_data)


# ---------------------------------------------------------------------------
# Attribution table
# ---------------------------------------------------------------------------

@callback(
    Output("po-attribution-grid-container", "children"),
    Input("po-weight-portfolio-select", "value"),
    Input("po-results-store", "data"),
    Input("po-vis-tabs", "value"),
    Input("po-attribution-tab-loaded-store", "data"),
    Input("po-attribution-chart-switch", "value"),
    State("dashmat-raw-data-store", "data"),
    State("po-periodicity-select", "value"),
    State("po-benchmark-assignments-store", "data"),
    State("po-long-short-store", "data"),
    State("po-date-range-store", "data"),
    State("po-vol-scaler-value-store", "data"),
    State("po-vol-scaling-assignments-store", "data"),
    prevent_initial_call=True,
)
def po_render_attribution_table(selected_portfolio, results, active_tab, tab_loaded, switch_value,
                                raw_data, periodicity, bench, ls, date_range,
                                vol_scaler, vol_scaling):
    if not _po_lazy_tab_render_ready(active_tab, "attribution", tab_loaded) or switch_value != "table" or not selected_portfolio or not results:
        return html.Div()
    if selected_portfolio not in results:
        return html.Div()

    portfolio_data = results[selected_portfolio]
    window_weights = portfolio_data.get("window_weights", [])
    config = portfolio_data.get("config", {})
    opt_series = config.get("selected_series", [])

    if not window_weights or not opt_series or not raw_data:
        return html.Div()
    if _po_missing_source_series(results, selected_portfolio, raw_data):
        return html.Div()

    try:
        with timed_block(
            "portopt.render_attribution_table",
            portfolio=selected_portfolio,
            series_count=len(opt_series),
        ):
            working_bundle = _build_po_working_bundle(
                raw_data, periodicity, bench, ls, date_range, vol_scaler, vol_scaling
            )
            attribution_monthly = _po_get_monthly_attribution(working_bundle, opt_series, window_weights)

            if attribution_monthly.empty:
                return html.Div()

            column_defs = [
                {
                    "field": "Date",
                    "pinned": "left",
                    "width": 120,
                },
            ]
            for s in opt_series:
                if s in attribution_monthly.columns:
                    column_defs.append({
                        "field": s,
                        "valueFormatter": {"function": "params.value != null ? d3.format('.2%')(params.value) : ''"},
                        "width": 100,
                    })

            # Add Total column
            attribution_monthly["Total"] = attribution_monthly[opt_series].sum(axis=1)
            column_defs.append({
                "field": "Total",
                "valueFormatter": {"function": "params.value != null ? d3.format('.2%')(params.value) : ''"},
                "width": 100,
            })

            df_reset = attribution_monthly.reset_index()
            df_reset["Date"] = df_reset["Date"].dt.strftime("%Y-%m-%d")
            row_data = df_reset.to_dict("records")

            return _po_build_result_grid("po-attribution-grid", column_defs, row_data)

    except Exception:
        return html.Div()


# ---------------------------------------------------------------------------
# Statistics tab
# ---------------------------------------------------------------------------

@callback(
    Output("po-statistics-grid-content", "children"),
    Input("po-results-store", "data"),
    Input("po-vis-tabs", "value"),
    Input("po-weight-portfolio-select", "value"),
    Input("dashmat-saved-series-cache-store", "data"),
    State("po-periodicity-select", "value"),
    State("dashmat-raw-data-store", "data"),
    State("po-benchmark-assignments-store", "data"),
    State("po-long-short-store", "data"),
    State("po-date-range-store", "data"),
    State("po-vol-scaler-value-store", "data"),
    State("po-vol-scaling-assignments-store", "data"),
    prevent_initial_call=True,
)
def po_render_statistics(
    results,
    active_tab,
    selected_portfolio,
    saved_series_store,
    periodicity=None,
    raw_data=None,
    bench=None,
    ls=None,
    date_range=None,
    vol_scaler=0,
    vol_scaling=None,
):
    if active_tab != "statistics" or not results:
        return html.Div()

    try:
        with timed_block("portopt.render_statistics", portfolio_count=1):
            legacy_compare = isinstance(selected_portfolio, (list, tuple, set))
            if legacy_compare:
                display_df = _po_collect_portfolio_returns(results, list(selected_portfolio))
                ordered_cols = list(display_df.columns)
            else:
                display_df, ordered_cols = _po_build_display_series(
                    results,
                    selected_portfolio,
                    raw_data,
                    periodicity,
                    bench,
                    ls,
                    date_range,
                    vol_scaler,
                    vol_scaling,
                )
            if display_df.empty or not ordered_cols:
                return html.Div()

            raw_json = df_to_json(display_df[ordered_cols])
            series_names = list(ordered_cols)

            stats = calculate_statistics_cached(
                raw_json,
                periodicity or "daily",
                tuple(series_names),
                "{}",
                "{}",
                "null",
                0,
                "{}",
                _risk_free_json_from_store(saved_series_store),
                _spx_json_from_store(saved_series_store),
            )

            if not stats:
                return html.Div()

            column_defs = [
                {"field": "Statistic", "pinned": "left", "width": 200},
            ]
            for series_name in series_names:
                column_defs.append({
                    "field": series_name,
                    "width": 120,
                    "valueFormatter": {
                        "function": "(!params.data._format || params.value == null) ? params.value : d3.format(params.data._format)(params.value)"
                    },
                })

            row_data = []
            for stat_name, fmt in STATS_CONFIG:
                row = {"Statistic": stat_name, "_format": fmt}
                for series_stats in stats:
                    series_name = series_stats.get("Series")
                    if series_name not in series_names:
                        continue
                    value = series_stats.get(stat_name)
                    if value is None or (isinstance(value, float) and pd.isna(value)):
                        row[series_name] = None
                    else:
                        row[series_name] = value
                row_data.append(row)

            return _po_build_result_grid("po-statistics-grid", column_defs, row_data)

    except Exception:
        return html.Div()


# ---------------------------------------------------------------------------
# Returns tab
# ---------------------------------------------------------------------------

@callback(
    Output("po-returns-grid-content", "children"),
    Input("po-results-store", "data"),
    Input("po-vis-tabs", "value"),
    Input("po-weight-portfolio-select", "value"),
    State("dashmat-raw-data-store", "data"),
    State("po-periodicity-select", "value"),
    State("po-benchmark-assignments-store", "data"),
    State("po-long-short-store", "data"),
    State("po-date-range-store", "data"),
    State("po-vol-scaler-value-store", "data"),
    State("po-vol-scaling-assignments-store", "data"),
    prevent_initial_call=True,
)
def po_render_returns(
    results,
    active_tab,
    selected_portfolio,
    raw_data=None,
    periodicity=None,
    bench=None,
    ls=None,
    date_range=None,
    vol_scaler=0,
    vol_scaling=None,
):
    if active_tab != "returns" or not results:
        return html.Div()

    try:
        legacy_compare = isinstance(selected_portfolio, (list, tuple, set))
        if legacy_compare:
            display_df = _po_collect_portfolio_returns(results, list(selected_portfolio))
            ordered_cols = list(display_df.columns)
        else:
            display_df, ordered_cols = _po_build_display_series(
                results,
                selected_portfolio,
                raw_data,
                periodicity,
                bench,
                ls,
                date_range,
                vol_scaler,
                vol_scaling,
            )
        if display_df.empty or not ordered_cols:
            return html.Div()

        column_defs = [
            {
                "field": "Date",
                "pinned": "left",
                "width": 120,
            },
        ]
        for col in ordered_cols:
            column_defs.append({
                "field": col,
                "valueFormatter": {"function": "params.value != null ? d3.format('.2%')(params.value) : ''"},
                "width": 120,
            })

        df_reset = display_df[ordered_cols].reset_index()
        df_reset["Date"] = df_reset["Date"].dt.strftime("%Y-%m-%d")
        row_data = df_reset.to_dict("records")

        return _po_build_result_grid("po-returns-grid", column_defs, row_data)

    except Exception:
        return html.Div()


# ---------------------------------------------------------------------------
# Excel export
# ---------------------------------------------------------------------------

@callback(
    Output("po-download-excel", "data"),
    Input("po-menu-download-excel", "n_clicks"),
    State("po-results-store", "data"),
    State("dashmat-raw-data-store", "data"),
    State("po-periodicity-select", "value"),
    State("po-benchmark-assignments-store", "data"),
    State("po-cmabench-assignments-store", "data"),
    State("po-long-short-store", "data"),
    State("po-date-range-store", "data"),
    State("po-vol-scaler-value-store", "data"),
    State("po-vol-scaling-assignments-store", "data"),
    State("po-rolling-window-select", "value"),
    State("po-rolling-return-type-select", "value"),
    State("po-rolling-metric-select", "value"),
    State("dashmat-saved-series-cache-store", "data"),
    State("po-weight-portfolio-select", "value"),
    prevent_initial_call=True,
)
def po_download_excel(n_clicks, results, raw_data, periodicity, bench, cmabench, ls,
                      date_range, vol_scaler, vol_scaling, rolling_window=None, rolling_return_type=None, rolling_metric=None, saved_series_store=None, selected_portfolio=None):
    if n_clicks is None or not results:
        raise PreventUpdate

    selected_name = selected_portfolio if selected_portfolio in (results or {}) else None
    if selected_name is None and results:
        selected_name = list(results.keys())[-1]
    if not selected_name:
        raise PreventUpdate
    active_results = {selected_name: (results or {}).get(selected_name)}

    timing_ctx = timed_block("portopt.download_excel.total", portfolio_count=len(active_results))
    timing_ctx.__enter__()
    try:
        working_bundle = _build_po_working_bundle(
            raw_data,
            periodicity,
            bench,
            ls,
            date_range,
            vol_scaler,
            vol_scaling,
        )
        working_df_cache: dict[tuple, pd.DataFrame] = {}

        # Build combined returns DataFrame (shared by multiple tabs/sheets)
        with timed_block("portopt.download_excel.build_returns"):
            combined_df = _po_collect_portfolio_returns(active_results, [selected_name])

        if combined_df.empty:
            raise PreventUpdate

        portfolio_names = list(combined_df.columns)

        # ------------------------------------------------------------------
        # Settings tab
        # ------------------------------------------------------------------
        def _safe_json_text(value):
            if value in (None, ""):
                return ""
            try:
                return json.dumps(value, default=str, sort_keys=True)
            except Exception:
                return str(value)

        def _safe_date_text(value):
            if value in (None, ""):
                return ""
            try:
                return format_mdy_date(pd.Timestamp(value))
            except Exception:
                return str(value)

        settings_rows = []
        pdata = (active_results or {}).get(selected_name) or {}
        cfg = pdata.get("config", {}) or {}
        window_weights = pdata.get("window_weights", []) or []
        risk_free_meta = pdata.get("risk_free_meta", {}) or {}
        selected_series = list(cfg.get("selected_series") or [])
        first_apply_start = _safe_date_text(window_weights[0].get("apply_start")) if window_weights else ""
        last_apply_end = _safe_date_text(window_weights[-1].get("apply_end")) if window_weights else ""
        exp_weighted = bool(cfg.get("exp_wt_cov", False))
        decay_value = normalize_decay_input(cfg.get("halflife", 63), 63.0)
        effective_shrinkage, effective_target = resolve_cov_shrinkage_spec(
            cfg.get("cov_shrinkage", "none"),
            cfg.get("cov_shrinkage_target", "scaled_identity"),
            exp_weighted=exp_weighted,
        )

        def _add_setting(parameter, value):
            settings_rows.append({"Parameter": parameter, "Value": value})

        _add_setting("Result Name", selected_name)
        _add_setting("Model", cfg.get("model", ""))
        _add_setting("Objective", cfg.get("objective", ""))
        _add_setting("Periodicity", cfg.get("periodicity", periodicity or "daily"))
        _add_setting("Selected Series Count", len(selected_series))
        _add_setting("Selected Series", ", ".join(selected_series))
        _add_setting("Date Range Start", (date_range or {}).get("start", ""))
        _add_setting("Date Range End", (date_range or {}).get("end", ""))
        _add_setting("Window Type", cfg.get("window_type", ""))
        _add_setting("Window Size", cfg.get("window_size"))
        _add_setting("Opt Step", cfg.get("opt_step"))
        _add_setting("Opt Step Unit", cfg.get("opt_step_unit"))
        _add_setting("Window Count", len(window_weights))
        _add_setting("First Apply Start", first_apply_start)
        _add_setting("Last Apply End", last_apply_end)
        _add_setting("Missing Data", cfg.get("missing_data", ""))
        _add_setting("Fill In-Sample", bool(cfg.get("fill_in_sample", False)))
        _add_setting("Exponential Weighting (Cov)", exp_weighted)
        _add_setting("Decay Input", float(decay_value))
        _add_setting("Decay Mode", decay_input_mode(decay_value, 63.0))
        _add_setting(
            "Covariance Shrinkage",
            format_cov_shrinkage_spec_label(effective_shrinkage, effective_target)
            if effective_shrinkage != "none"
            else "None",
        )
        if effective_shrinkage == "none":
            shrinkage_target_label = "N/A"
        else:
            shrinkage_target_label = format_cov_shrinkage_target_label(effective_target)
        _add_setting("Covariance Shrinkage Target", shrinkage_target_label)
        _add_setting("Vol Scaler", float(vol_scaler or 0))
        _add_setting("Benchmark Assignments", _safe_json_text(bench or {}))
        _add_setting("CMA Benchmark Assignments", _safe_json_text(cmabench or {}))
        _add_setting("Long/Short Assignments", _safe_json_text(ls or {}))
        _add_setting("Vol Scaling Assignments", _safe_json_text(vol_scaling or {}))
        _add_setting("Min Wt Constraints", _safe_json_text(cfg.get("min_wt") or {}))
        _add_setting("Max Wt Constraints", _safe_json_text(cfg.get("max_wt") or {}))
        _add_setting("Force Max Flags", _safe_json_text(cfg.get("force_max") or {}))
        _add_setting("Linear Constraints", _safe_json_text(cfg.get("linear_constraints") or []))
        _add_setting("Ex-Ante Mode", cfg.get("ex_ante_mode", ""))
        _add_setting("Ex-Ante Returns", _safe_json_text(cfg.get("ex_ante_returns") or {}))
        _add_setting("Ex-Ante Covariance", _safe_json_text(cfg.get("ex_ante_cov") or {}))
        _add_setting("Ex-Ante Volatility", _safe_json_text(cfg.get("ex_ante_vol") or {}))
        _add_setting("Ex-Ante Correlation", _safe_json_text(cfg.get("ex_ante_corr") or {}))
        _add_setting("BL Tau", cfg.get("bl_tau"))
        _add_setting("BL Views", _safe_json_text(cfg.get("bl_views") or []))
        _add_setting("Risk-Free Source", risk_free_meta.get("source", cfg.get("risk_free_source", "")))
        _add_setting("Risk-Free Annual", risk_free_meta.get("annual", cfg.get("risk_free_annual_default")))
        _add_setting("Risk-Free Warning", risk_free_meta.get("warning", cfg.get("risk_free_warning", "")))
        _add_setting("Rolling Window (Export)", rolling_window or "1y")
        _add_setting("Rolling Return Type (Export)", rolling_return_type or "annualized")
        _add_setting("Rolling Metric (Export)", rolling_metric or "total_return")

        settings_df = pd.DataFrame(settings_rows)

        # ------------------------------------------------------------------
        # Weights tab
        # ------------------------------------------------------------------
        selected_portfolio_data = (active_results or {}).get(selected_name) or {}
        selected_window_weights = selected_portfolio_data.get("window_weights", []) or []
        if selected_window_weights:
            asset_names = list((selected_window_weights[0].get("weights") or {}).keys())
            weight_rows = []
            for ww in selected_window_weights:
                row = {
                    "Apply Start": pd.Timestamp(ww["apply_start"]).strftime("%Y-%m-%d"),
                    "Apply End": pd.Timestamp(ww["apply_end"]).strftime("%Y-%m-%d"),
                }
                for asset in asset_names:
                    row[asset] = (ww.get("weights") or {}).get(asset, 0)
                weight_rows.append(row)
            weights_df = pd.DataFrame(weight_rows)
            weights_df = weights_df[["Apply Start", "Apply End", *asset_names]]
        else:
            weights_df = pd.DataFrame(columns=["Apply Start", "Apply End"])

        # ------------------------------------------------------------------
        # Turnover tab
        # ------------------------------------------------------------------
        turnover_rows = []
        turnover_delta_cols = []
        for pname, pdata in active_results.items():
            window_weights = pdata.get("window_weights", []) or []
            if len(window_weights) < 2:
                continue
            all_assets = []
            for ww in window_weights:
                for asset in (ww.get("weights", {}) or {}).keys():
                    if asset not in all_assets:
                        all_assets.append(asset)
            for i in range(1, len(window_weights)):
                prev_w = window_weights[i - 1].get("weights", {}) or {}
                curr_w = window_weights[i].get("weights", {}) or {}
                turnover = sum(abs(curr_w.get(a, 0) - prev_w.get(a, 0)) for a in all_assets) / 2
                row = {
                    "Portfolio": pname,
                    "Rebalance Date": format_mdy_date(pd.Timestamp(window_weights[i]["apply_start"])),
                    "Turnover": float(turnover),
                }
                for asset in all_assets:
                    col = f"Delta_{asset}"
                    row[col] = float(curr_w.get(asset, 0) - prev_w.get(asset, 0))
                    if col not in turnover_delta_cols:
                        turnover_delta_cols.append(col)
                turnover_rows.append(row)
        if turnover_rows:
            turnover_df = pd.DataFrame(turnover_rows)
            turnover_df = turnover_df[["Portfolio", "Rebalance Date", "Turnover", *turnover_delta_cols]]
        else:
            turnover_df = pd.DataFrame(columns=["Portfolio", "Rebalance Date", "Turnover"])

        # ------------------------------------------------------------------
        # Statistics tab
        # ------------------------------------------------------------------
        stats_df = pd.DataFrame(columns=["Statistic"])
        try:
            with timed_block("portopt.download_excel.statistics"):
                raw_json = df_to_json(combined_df)
                stats = calculate_statistics_cached(
                    raw_json,
                    periodicity or "daily",
                    tuple(portfolio_names),
                    "{}",
                    "{}",
                    "null",
                    0,
                    "{}",
                    _risk_free_json_from_store(saved_series_store),
                    _spx_json_from_store(saved_series_store),
                )
                if stats:
                    stats_data = {"Statistic": [sn for sn, _ in STATS_CONFIG]}
                    for series_stats in stats:
                        sname = series_stats["Series"]
                        stats_data[sname] = [series_stats.get(sn) for sn, _ in STATS_CONFIG]
                    stats_df = pd.DataFrame(stats_data)
        except Exception:
            pass

        # ------------------------------------------------------------------
        # Returns tab
        # ------------------------------------------------------------------
        returns_df = combined_df.reset_index()
        returns_date_col = returns_df.columns[0]
        returns_df = returns_df.rename(columns={returns_date_col: "Date"})
        returns_df["Date"] = returns_df["Date"].map(format_mdy_date)

        # ------------------------------------------------------------------
        # Growth tab
        # ------------------------------------------------------------------
        growth_data = {pname: (1 + combined_df[pname].dropna()).cumprod() for pname in portfolio_names}
        growth_df = pd.DataFrame(growth_data).sort_index().reset_index()
        growth_date_col = growth_df.columns[0]
        growth_df = growth_df.rename(columns={growth_date_col: "Date"})
        growth_df["Date"] = growth_df["Date"].map(format_mdy_date)

        # ------------------------------------------------------------------
        # Rolling / Calendar / Drawdown tabs
        # ------------------------------------------------------------------
        rolling_df = pd.DataFrame()
        calendar_df = pd.DataFrame()
        drawdown_df = pd.DataFrame()
        try:
            rolling_df = calculate_rolling_returns(
                df_to_json(combined_df),
                periodicity or "daily",
                tuple(portfolio_names),
                "total",
                "{}",
                "{}",
                "null",
                rolling_window or "1y",
                rolling_return_type or "annualized",
                rolling_metric or "total_return",
                0,
                "{}",
            )
        except Exception:
            rolling_df = pd.DataFrame()

        try:
            calendar_df = calculate_calendar_year_returns(
                df_to_json(combined_df),
                periodicity or "daily",
                periodicity or "daily",
                tuple(portfolio_names),
                "total",
                "{}",
                "{}",
                "null",
                0,
                "{}",
            )
        except Exception:
            calendar_df = pd.DataFrame()

        try:
            drawdown_df = calculate_drawdown(
                df_to_json(combined_df),
                periodicity or "daily",
                tuple(portfolio_names),
                "total",
                "{}",
                "{}",
                "null",
                0,
                "{}",
            )
        except Exception:
            drawdown_df = pd.DataFrame()

        # ------------------------------------------------------------------
        # Attribution tab
        # ------------------------------------------------------------------
        attribution_frames = []
        with timed_block("portopt.download_excel.attribution", portfolio_count=len(active_results)):
            for pname, pdata in active_results.items():
                config = pdata.get("config", {}) or {}
                opt_series = config.get("selected_series", []) or []
                window_weights = pdata.get("window_weights", []) or []
                if not window_weights or not opt_series or not raw_data:
                    continue

                series_key = tuple(opt_series)
                working_df = working_df_cache.get(series_key)
                if working_df is None:
                    working_df = _po_get_working_returns(working_bundle, series_key)
                    working_df_cache[series_key] = working_df
                attribution_monthly = _po_get_monthly_attribution(
                    working_bundle,
                    opt_series,
                    window_weights,
                )
                if attribution_monthly.empty:
                    continue

                avail_cols = [c for c in opt_series if c in attribution_monthly.columns]
                attribution_monthly = attribution_monthly.copy()
                attribution_monthly["Total"] = (
                    attribution_monthly[avail_cols].sum(axis=1) if avail_cols else 0.0
                )

                frame = attribution_monthly.reset_index()
                date_col = frame.columns[0]
                frame = frame.rename(columns={date_col: "Date"})
                frame["Date"] = pd.to_datetime(frame["Date"]).map(format_mdy_date)
                frame.insert(0, "Portfolio", pname)
                attribution_frames.append(frame)
        if attribution_frames:
            attribution_df = pd.concat(attribution_frames, axis=0, ignore_index=True)
        else:
            attribution_df = pd.DataFrame(columns=["Portfolio", "Date"])

        # ------------------------------------------------------------------
        # Risk tab
        # ------------------------------------------------------------------
        risk_rows = []
        risk_asset_cols = []
        for pname, pdata in active_results.items():
            config = pdata.get("config", {}) or {}
            opt_series = config.get("selected_series", []) or []
            window_weights = pdata.get("window_weights", []) or []
            if not window_weights or not opt_series or not raw_data:
                continue

            for rr in _po_get_window_risk_contributions(working_bundle, opt_series, window_weights, config):
                row = {
                    "Portfolio": pname,
                    "Window Start": format_mdy_date(rr["apply_start"]),
                    "Window End": format_mdy_date(rr["apply_end"]),
                }
                for asset, value in (rr.get("risk_contributions", {}) or {}).items():
                    row[asset] = float(value)
                    if asset not in risk_asset_cols:
                        risk_asset_cols.append(asset)
                risk_rows.append(row)
        if risk_rows:
            risk_df = pd.DataFrame(risk_rows)
            risk_df = risk_df[["Portfolio", "Window Start", "Window End", *risk_asset_cols]]
        else:
            risk_df = pd.DataFrame(columns=["Portfolio", "Window Start", "Window End"])

        # ------------------------------------------------------------------
        # Frontier tab (most recent window for each portfolio)
        # ------------------------------------------------------------------
        frontier_rows = []
        frontier_weight_cols = []
        with timed_block("portopt.download_excel.frontier", portfolio_count=len(active_results)):
            for pname, pdata in active_results.items():
                config = pdata.get("config", {}) or {}
                window_weights = pdata.get("window_weights", []) or []
                opt_series = config.get("selected_series", []) or []
                if not window_weights or not opt_series or not raw_data:
                    continue

                try:
                    snapshot = _po_resolve_frontier_snapshot(
                        selected_portfolio=pname,
                        portfolio_data=pdata,
                        raw_data=raw_data,
                        periodicity=periodicity,
                        bench=bench,
                        ls=ls,
                        vol_scaler=vol_scaler,
                        vol_scaling=vol_scaling,
                        window_idx=None,
                        rm="MV",
                        linear_constraints=config.get("linear_constraints", []),
                        saved_series_store=saved_series_store,
                        cmabench_assignments=cmabench,
                    )

                    window_start = format_mdy_date(snapshot.get("window_est_start"))
                    window_end = format_mdy_date(snapshot.get("window_est_end"))
                    window_label = f"{window_start} - {window_end}"
                    for row in _build_frontier_table_rows(snapshot):
                        out_row = {"Portfolio": pname, "Window": window_label, **row}
                        frontier_rows.append(out_row)
                        for k in out_row:
                            if k.startswith("Wt_") and k not in frontier_weight_cols:
                                frontier_weight_cols.append(k)
                except Exception:
                    continue
        frontier_base_cols = ["Portfolio", "Window", "Type", "Name", "Return", "Risk", "Sharpe Ratio"]
        if frontier_rows:
            frontier_df = pd.DataFrame(frontier_rows)
            ordered_cols = [c for c in frontier_base_cols if c in frontier_df.columns] + frontier_weight_cols
            frontier_df = frontier_df[ordered_cols]
        else:
            frontier_df = pd.DataFrame(columns=frontier_base_cols)

        output = BytesIO()
        settings_df = format_excel_dates(settings_df)
        turnover_df = format_excel_dates(turnover_df)
        stats_df = format_excel_dates(stats_df)
        returns_df = format_excel_dates(returns_df)
        growth_df = format_excel_dates(growth_df)
        rolling_df = format_excel_dates(rolling_df, format_index=True)
        calendar_df = format_excel_dates(calendar_df, format_index=True)
        drawdown_df = format_excel_dates(drawdown_df, format_index=True)
        attribution_df = format_excel_dates(attribution_df)
        risk_df = format_excel_dates(risk_df)
        frontier_df = format_excel_dates(frontier_df)

        sheet_frames = {
            "weight": weights_df,
            "attribution": attribution_df,
            "risk": risk_df,
            "turnover": turnover_df,
            "frontier": frontier_df,
            "statistics": stats_df,
            "returns": returns_df,
            "rolling": rolling_df,
            "calendar": calendar_df,
            "growth": growth_df,
            "drawdown": drawdown_df,
        }
        with pd.ExcelWriter(output, engine="xlsxwriter", date_format="m/d/yyyy", datetime_format="m/d/yyyy") as writer:
            write_excel_with_autofit(writer, settings_df, "Settings", index=False)
            for spec in PO_TAB_SPECS:
                frame = sheet_frames.get(spec["value"])
                if frame is None:
                    continue
                if spec["export_index"] and frame.empty:
                    continue
                write_excel_with_autofit(
                    writer,
                    frame,
                    spec["label"],
                    index=spec["export_index"],
                )

        output.seek(0)
        return dcc.send_bytes(output.getvalue(), "portfolio_optimization.xlsx")

    except Exception:
        raise PreventUpdate
    finally:
        timing_ctx.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# Risk Contribution chart
# ---------------------------------------------------------------------------

@callback(
    Output("po-risk-chart-container", "children"),
    Input("po-weight-portfolio-select", "value"),
    Input("po-results-store", "data"),
    Input("po-vis-tabs", "value"),
    Input("po-risk-tab-loaded-store", "data"),
    Input("po-risk-chart-switch", "value"),
    State("dashmat-raw-data-store", "data"),
    State("po-periodicity-select", "value"),
    State("po-benchmark-assignments-store", "data"),
    State("po-long-short-store", "data"),
    State("po-date-range-store", "data"),
    State("po-vol-scaler-value-store", "data"),
    State("po-vol-scaling-assignments-store", "data"),
    State("po-series-select", "data"),
    State("global-color-scheme-toggle", "computedColorScheme"),
    prevent_initial_call=True,
)
def po_render_risk_chart(selected_portfolio, results, active_tab, tab_loaded, switch_value,
                         raw_data, periodicity, bench, ls, date_range,
                         vol_scaler, vol_scaling, series_select, theme):
    if not _po_lazy_tab_render_ready(active_tab, "risk", tab_loaded) or switch_value != "chart" or not selected_portfolio or not results:
        return html.Div()
    if selected_portfolio not in results:
        return html.Div()

    portfolio_data = results[selected_portfolio]
    window_weights = portfolio_data.get("window_weights", [])
    config = portfolio_data.get("config", {})
    opt_series = config.get("selected_series", [])

    if not window_weights or not opt_series or not raw_data:
        return dmc.Text("No risk data available.", c="dimmed")
    missing_sources = _po_missing_source_series(results, selected_portfolio, raw_data)
    if missing_sources:
        return dmc.Text(f"Missing source series: {', '.join(missing_sources)}", c="dimmed")

    timing_ctx = timed_block(
        "portopt.render_risk_chart",
        portfolio=selected_portfolio,
        series_count=len(opt_series),
    )
    timing_ctx.__enter__()
    try:
        working_bundle = _build_po_working_bundle(
            raw_data, periodicity, bench, ls, date_range, vol_scaler, vol_scaling
        )
        risk_rows = _po_get_window_risk_contributions(working_bundle, opt_series, window_weights, config)
        if not risk_rows:
            return dmc.Text("No risk data available.", c="dimmed")

        all_dates = [row["apply_end"] for row in risk_rows]
        all_contributions = {s: [] for s in opt_series}
        for row in risk_rows:
            rc = row["risk_contributions"]
            for s in opt_series:
                all_contributions[s].append(rc.get(s, 0) * 100)

        fig = go.Figure()
        for s in opt_series:
            fig.add_trace(go.Bar(
                x=all_dates,
                y=all_contributions[s],
                name=s,
            ))

        fig.update_layout(
            barmode="relative",
            title=f"Risk Contribution: {selected_portfolio}",
            yaxis_title="Contribution (%)",
            hovermode="x unified",
            margin={"t": 40, "b": 40, "l": 60, "r": 20},
            height=420,
            legend={"orientation": "h", "yanchor": "bottom", "y": -0.2},
        )
        apply_chart_theme(fig, theme)

        return dcc.Graph(figure=fig, style={"height": "100%", "width": "100%"})

    except Exception:
        return dmc.Text("Error computing risk contributions.", c="dimmed")
    finally:
        timing_ctx.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# Risk Contribution table
# ---------------------------------------------------------------------------

@callback(
    Output("po-risk-grid-container", "children"),
    Input("po-weight-portfolio-select", "value"),
    Input("po-results-store", "data"),
    Input("po-vis-tabs", "value"),
    Input("po-risk-tab-loaded-store", "data"),
    Input("po-risk-chart-switch", "value"),
    State("dashmat-raw-data-store", "data"),
    State("po-periodicity-select", "value"),
    State("po-benchmark-assignments-store", "data"),
    State("po-long-short-store", "data"),
    State("po-date-range-store", "data"),
    State("po-vol-scaler-value-store", "data"),
    State("po-vol-scaling-assignments-store", "data"),
    State("po-series-select", "data"),
    prevent_initial_call=True,
)
def po_render_risk_table(selected_portfolio, results, active_tab, tab_loaded, switch_value,
                         raw_data, periodicity, bench, ls, date_range,
                         vol_scaler, vol_scaling, series_select):
    if not _po_lazy_tab_render_ready(active_tab, "risk", tab_loaded) or switch_value != "table" or not selected_portfolio or not results:
        return html.Div()
    if selected_portfolio not in results:
        return html.Div()

    portfolio_data = results[selected_portfolio]
    window_weights = portfolio_data.get("window_weights", [])
    config = portfolio_data.get("config", {})
    opt_series = config.get("selected_series", [])

    if not window_weights or not opt_series or not raw_data:
        return html.Div()

    timing_ctx = timed_block(
        "portopt.render_risk_table",
        portfolio=selected_portfolio,
        series_count=len(opt_series),
    )
    timing_ctx.__enter__()
    try:
        working_bundle = _build_po_working_bundle(
            raw_data, periodicity, bench, ls, date_range, vol_scaler, vol_scaling
        )
        risk_rows = _po_get_window_risk_contributions(working_bundle, opt_series, window_weights, config)
        if not risk_rows:
            return html.Div()

        column_defs = [
            {"field": "Window Start", "pinned": "left", "width": 120},
            {"field": "Window End", "pinned": "left", "width": 120},
        ]
        for a in opt_series:
            column_defs.append({
                "field": a,
                "valueFormatter": {"function": "params.value != null ? d3.format('.2%')(params.value) : ''"},
                "width": 100,
            })

        row_data = []
        for row in risk_rows:
            start = row["apply_start"]
            end = row["apply_end"]
            rc = row["risk_contributions"]
            row = {
                "Window Start": start.strftime("%Y-%m-%d"),
                "Window End": end.strftime("%Y-%m-%d"),
            }
            for a in opt_series:
                row[a] = rc.get(a, 0)
            row_data.append(row)

        return _po_build_result_grid("po-risk-grid", column_defs, row_data)

    except Exception:
        return html.Div()
    finally:
        timing_ctx.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# Turnover chart
# ---------------------------------------------------------------------------

@callback(
    Output("po-turnover-chart-container", "children"),
    Input("po-weight-portfolio-select", "value"),
    Input("po-results-store", "data"),
    Input("po-vis-tabs", "value"),
    Input("po-turnover-chart-switch", "value"),
    State("global-color-scheme-toggle", "computedColorScheme"),
    prevent_initial_call=True,
)
def po_render_turnover_chart(selected_portfolio, results, active_tab, switch_value, theme):
    if active_tab != "turnover" or switch_value != "chart" or not selected_portfolio or not results:
        return html.Div()
    if selected_portfolio not in results:
        return html.Div()

    portfolio_data = results[selected_portfolio]
    window_weights = portfolio_data.get("window_weights", [])

    if not window_weights:
        return dmc.Text("No turnover data available.", c="dimmed")
    if len(window_weights) < 2:
        return dmc.Text("Turnover requires multiple rebalance windows (not available for Full window).", c="dimmed")

    dates = []
    turnovers = []
    for i in range(1, len(window_weights)):
        prev_w = window_weights[i - 1]["weights"]
        curr_w = window_weights[i]["weights"]
        all_assets = set(prev_w.keys()) | set(curr_w.keys())
        turnover = sum(abs(curr_w.get(a, 0) - prev_w.get(a, 0)) for a in all_assets) / 2
        dates.append(pd.Timestamp(window_weights[i]["apply_start"]))
        turnovers.append(turnover * 100)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dates,
        y=turnovers,
        text=[f"{v:.1f}%" for v in turnovers],
        textposition="auto",
    ))
    fig.update_layout(
        title=f"Portfolio Turnover: {selected_portfolio}",
        yaxis_title="Turnover (%)",
        xaxis_title="Rebalance Date",
        hovermode="x unified",
        margin={"t": 40, "b": 40, "l": 60, "r": 20},
        height=420,
    )
    apply_chart_theme(fig, theme)

    return dcc.Graph(figure=fig, style={"height": "100%", "width": "100%"})


# ---------------------------------------------------------------------------
# Turnover table
# ---------------------------------------------------------------------------

@callback(
    Output("po-turnover-grid-container", "children"),
    Input("po-weight-portfolio-select", "value"),
    Input("po-results-store", "data"),
    Input("po-vis-tabs", "value"),
    Input("po-turnover-chart-switch", "value"),
    prevent_initial_call=True,
)
def po_render_turnover_table(selected_portfolio, results, active_tab, switch_value):
    if active_tab != "turnover" or switch_value != "table" or not selected_portfolio or not results:
        return html.Div()
    if selected_portfolio not in results:
        return html.Div()

    portfolio_data = results[selected_portfolio]
    window_weights = portfolio_data.get("window_weights", [])

    if not window_weights or len(window_weights) < 2:
        return html.Div()

    # Get all asset names
    all_assets = list(window_weights[0]["weights"].keys())

    column_defs = [
        {"field": "Rebalance Date", "pinned": "left", "width": 130},
        {
            "field": "Turnover",
            "valueFormatter": {"function": "params.value != null ? d3.format('.2%')(params.value) : ''"},
            "width": 100,
        },
    ]
    for a in all_assets:
        column_defs.append({
            "field": a,
            "valueFormatter": {"function": "params.value != null ? d3.format('.2%')(params.value) : ''"},
            "width": 100,
        })

    row_data = []
    for i in range(1, len(window_weights)):
        prev_w = window_weights[i - 1]["weights"]
        curr_w = window_weights[i]["weights"]
        turnover = sum(abs(curr_w.get(a, 0) - prev_w.get(a, 0)) for a in all_assets) / 2
        row = {
            "Rebalance Date": pd.Timestamp(window_weights[i]["apply_start"]).strftime("%Y-%m-%d"),
            "Turnover": turnover,
        }
        for a in all_assets:
            row[a] = curr_w.get(a, 0) - prev_w.get(a, 0)
        row_data.append(row)

    return _po_build_result_grid("po-turnover-grid", column_defs, row_data)


# ---------------------------------------------------------------------------
# Efficient Frontier: populate window dropdown
# ---------------------------------------------------------------------------

@callback(
    Output("po-frontier-rm-select", "data"),
    Output("po-frontier-rm-select", "value"),
    Input("po-weight-portfolio-select", "value"),
    Input("po-results-store", "data"),
    State("po-frontier-rm-select", "value"),
    prevent_initial_call=True,
)
def po_update_frontier_risk_measure_options(selected_portfolio, results, current_rm):
    all_options = [
        {"value": "MV", "label": "Volatility"},
        {"value": "CVaR", "label": "CVaR"},
    ]
    if not selected_portfolio or not results or selected_portfolio not in results:
        return all_options, (current_rm if current_rm in {"MV", "CVaR"} else "MV")

    model = (results.get(selected_portfolio, {}).get("config", {}) or {}).get("model", "")
    if model in {"ex_ante_mv", "black_litterman"}:
        return [{"value": "MV", "label": "Volatility"}], "MV"

    return all_options, (current_rm if current_rm in {"MV", "CVaR"} else "MV")


@callback(
    Output("po-frontier-window-select", "data"),
    Output("po-frontier-window-select", "value"),
    Output("po-frontier-window-select", "disabled"),
    Input("po-weight-portfolio-select", "value"),
    Input("po-results-store", "data"),
    Input("po-vis-tabs", "value"),
    prevent_initial_call=True,
)
def po_populate_frontier_windows(selected_portfolio, results, active_tab):
    if active_tab != "frontier" or not selected_portfolio or not results:
        return [], None, False
    portfolio_data = results.get(selected_portfolio, {})
    window_weights = portfolio_data.get("window_weights", [])
    config = portfolio_data.get("config", {})
    model = config.get("model", "")
    if not window_weights:
        return [], None, False

    # Ex ante / BL are single-period - disable window selection
    is_single_period = model in ("ex_ante_mv", "black_litterman")

    options = []
    for i, ww in enumerate(window_weights):
        # Show estimation window (used for optimization), not apply window
        est_start = pd.Timestamp(ww.get("est_start", ww["apply_start"])).strftime("%Y-%m-%d")
        est_end = pd.Timestamp(ww.get("est_end", ww["apply_end"])).strftime("%Y-%m-%d")
        options.append({"value": str(i), "label": f"{est_start} - {est_end}"})
    # Default to last window
    return options, str(len(window_weights) - 1), is_single_period


# ---------------------------------------------------------------------------
# Efficient Frontier chart
# ---------------------------------------------------------------------------

@callback(
    Output("po-frontier-chart-container", "children"),
    Input("po-weight-portfolio-select", "value"),
    Input("po-results-store", "data"),
    Input("po-vis-tabs", "value"),
    Input("po-frontier-tab-loaded-store", "data"),
    Input("po-frontier-chart-switch", "value"),
    Input("po-frontier-window-select", "value"),
    Input("po-frontier-rm-select", "value"),
    State("dashmat-raw-data-store", "data"),
    State("po-periodicity-select", "value"),
    State("po-benchmark-assignments-store", "data"),
    State("po-long-short-store", "data"),
    State("po-date-range-store", "data"),
    State("po-vol-scaler-value-store", "data"),
    State("po-vol-scaling-assignments-store", "data"),
    State("po-cmabench-assignments-store", "data"),
    State("dashmat-saved-series-cache-store", "data"),
    State("po-series-select", "data"),
    State("global-color-scheme-toggle", "computedColorScheme"),
    State("po-linear-constraints-store", "data"),
    prevent_initial_call=True,
)
def po_render_frontier_chart(selected_portfolio, results, active_tab, tab_loaded, switch_value,
                             window_idx, rm,
                             raw_data, periodicity, bench, ls, date_range,
                             vol_scaler, vol_scaling, cmabench_assignments, saved_series_store, series_select, theme,
                             linear_constraints):
    if not _po_lazy_tab_render_ready(active_tab, "frontier", tab_loaded) or switch_value != "chart" or not selected_portfolio or not results:
        return html.Div()
    if selected_portfolio not in results:
        return html.Div()

    portfolio_data = results[selected_portfolio]
    window_weights = portfolio_data.get("window_weights", [])
    config = portfolio_data.get("config", {})
    opt_series = config.get("selected_series", [])

    if not window_weights or not opt_series or not raw_data:
        return dmc.Text("No frontier data available.", c="dimmed")
    missing_sources = _po_missing_source_series(results, selected_portfolio, raw_data)
    if missing_sources:
        return dmc.Text(f"Missing source series: {', '.join(missing_sources)}", c="dimmed")

    timing_ctx = timed_block(
        "portopt.render_frontier_chart",
        portfolio=selected_portfolio,
        series_count=len(opt_series),
        risk_measure=rm,
    )
    timing_ctx.__enter__()
    try:
        model = config.get("model", "")
        snapshot = _po_resolve_frontier_snapshot(
            selected_portfolio=selected_portfolio,
            portfolio_data=portfolio_data,
            raw_data=raw_data,
            periodicity=periodicity,
            bench=bench,
            ls=ls,
            vol_scaler=vol_scaler,
            vol_scaling=vol_scaling,
            window_idx=window_idx,
            rm=rm,
            linear_constraints=linear_constraints,
            saved_series_store=saved_series_store,
            cmabench_assignments=cmabench_assignments,
        )

        frontier_pts = snapshot.get("frontier_points", []) or []
        asset_pts = snapshot.get("assets", []) or []
        portfolio_marker = snapshot.get("portfolio", {}) or {}
        risk_measure = snapshot.get("risk_measure", _normalize_frontier_risk_measure(model, rm))

        if not frontier_pts:
            return dmc.Text("No frontier points available for the selected window.", c="dimmed")

        fig = go.Figure()

        # Frontier line
        fig.add_trace(go.Scatter(
            x=[pt["risk"] * 100 for pt in frontier_pts],
            y=[pt["return"] * 100 for pt in frontier_pts],
            mode="lines",
            name="Efficient Frontier",
            line={"color": "royalblue", "width": 2},
        ))

        # Selected portfolio marker
        fig.add_trace(go.Scatter(
            x=[portfolio_marker.get("risk", 0) * 100],
            y=[portfolio_marker.get("return", 0) * 100],
            mode="markers+text",
            name=selected_portfolio,
            marker={"size": 14, "color": "red", "symbol": "star"},
            text=[selected_portfolio],
            textposition="top center",
        ))

        # Individual assets
        fig.add_trace(go.Scatter(
            x=[a["risk"] * 100 for a in asset_pts],
            y=[a["return"] * 100 for a in asset_pts],
            mode="markers+text",
            name="Assets",
            marker={"size": 8, "color": "gray"},
            text=[a["name"] for a in asset_pts],
            textposition="top center",
            textfont={"size": 9},
        ))

        # Annotate frontier type in title
        if model == "ex_ante_mv":
            title_suffix = " (Ex Ante)"
        elif model == "black_litterman":
            title_suffix = " (Black-Litterman)"
        else:
            title_suffix = ""

        x_label = "Annualized CVaR (%)" if risk_measure == "CVaR" else "Annualized Volatility (%)"
        fig.update_layout(
            title=f"Efficient Frontier: {selected_portfolio}{title_suffix}",
            xaxis_title=x_label,
            yaxis_title="Annualized Return (%)",
            hovermode="closest",
            margin={"t": 40, "b": 40, "l": 60, "r": 20},
            height=420,
            showlegend=True,
            legend={"orientation": "h", "yanchor": "bottom", "y": -0.2},
        )
        apply_chart_theme(fig, theme)

        return dcc.Loading(
            type="default",
            children=[dcc.Graph(figure=fig, style={"height": "100%", "width": "100%"})],
        )

    except Exception as e:
        return dmc.Text(f"Error computing efficient frontier: {str(e)}", c="dimmed")
    finally:
        timing_ctx.__exit__(None, None, None)


@callback(
    Output("po-frontier-grid-container", "children"),
    Input("po-weight-portfolio-select", "value"),
    Input("po-results-store", "data"),
    Input("po-vis-tabs", "value"),
    Input("po-frontier-tab-loaded-store", "data"),
    Input("po-frontier-chart-switch", "value"),
    Input("po-frontier-window-select", "value"),
    Input("po-frontier-rm-select", "value"),
    State("dashmat-raw-data-store", "data"),
    State("po-periodicity-select", "value"),
    State("po-benchmark-assignments-store", "data"),
    State("po-long-short-store", "data"),
    State("po-vol-scaler-value-store", "data"),
    State("po-vol-scaling-assignments-store", "data"),
    State("po-cmabench-assignments-store", "data"),
    State("dashmat-saved-series-cache-store", "data"),
    State("po-linear-constraints-store", "data"),
    prevent_initial_call=True,
)
def po_render_frontier_table(
    selected_portfolio,
    results,
    active_tab,
    tab_loaded,
    switch_value,
    window_idx,
    rm,
    raw_data,
    periodicity,
    bench,
    ls,
    vol_scaler,
    vol_scaling,
    cmabench_assignments,
    saved_series_store,
    linear_constraints,
):
    if not _po_lazy_tab_render_ready(active_tab, "frontier", tab_loaded) or switch_value != "table" or not selected_portfolio or not results:
        return html.Div()
    if selected_portfolio not in results:
        return html.Div()

    portfolio_data = results[selected_portfolio]
    window_weights = portfolio_data.get("window_weights", []) or []
    config = portfolio_data.get("config", {}) or {}
    opt_series = config.get("selected_series", []) or []
    if not window_weights or not opt_series or not raw_data:
        return html.Div()
    if _po_missing_source_series(results, selected_portfolio, raw_data):
        return html.Div()

    with timed_block(
        "portopt.render_frontier_table",
        portfolio=selected_portfolio,
        series_count=len(opt_series),
        risk_measure=rm,
    ):
        try:
            snapshot = _po_resolve_frontier_snapshot(
                selected_portfolio=selected_portfolio,
                portfolio_data=portfolio_data,
                raw_data=raw_data,
                periodicity=periodicity,
                bench=bench,
                ls=ls,
                vol_scaler=vol_scaler,
                vol_scaling=vol_scaling,
                window_idx=window_idx,
                rm=rm,
                linear_constraints=linear_constraints,
                saved_series_store=saved_series_store,
                cmabench_assignments=cmabench_assignments,
            )
        except Exception:
            return html.Div()

        return _po_build_result_grid(
            "po-frontier-grid",
            _build_frontier_column_defs(snapshot),
            _build_frontier_table_rows(snapshot),
        )


@callback(
    Output("po-frontier-rf-warning", "children"),
    Output("po-frontier-rf-warning", "style"),
    Input("po-weight-portfolio-select", "value"),
    Input("po-results-store", "data"),
    Input("po-vis-tabs", "value"),
    Input("po-frontier-window-select", "value"),
    Input("po-frontier-rm-select", "value"),
    State("po-periodicity-select", "value"),
    State("dashmat-saved-series-cache-store", "data"),
    State("po-cmabench-assignments-store", "data"),
    prevent_initial_call=True,
)
def po_render_frontier_rf_warning(
    selected_portfolio,
    results,
    active_tab,
    window_idx,
    rm,
    periodicity,
    saved_series_store,
    cmabench_assignments,
):
    hidden = {"display": "none"}
    shown = {"display": "block", "marginBottom": "8px"}
    if active_tab != "frontier" or not selected_portfolio or not results:
        return "", hidden
    if selected_portfolio not in results:
        return "", hidden

    portfolio_data = results[selected_portfolio]
    config = portfolio_data.get("config", {}) or {}
    model = config.get("model", "")
    warning = None

    if model in {"ex_ante_mv", "black_litterman"}:
        risk_measure = rm or "MV"
        if risk_measure == "CVaR":
            risk_measure = "MV"
        try:
            resolved_idx, _ = _resolve_frontier_window(portfolio_data.get("window_weights", []) or [], window_idx)
            snapshot = _get_cached_frontier_snapshot(portfolio_data, resolved_idx, risk_measure)
        except Exception:
            snapshot = None
        warning = (
            (snapshot or {}).get("rf_warning")
            or (portfolio_data.get("risk_free_meta", {}) or {}).get("warning")
        )
    else:
        rf_ctx = _resolve_risk_free_context(
            model=model,
            asset_order=config.get("selected_series", []) or [],
            periodicity=periodicity,
            expected_mu_annual=None,
            reference_index=None,
            saved_series_store=saved_series_store,
            cmabench_assignments=cmabench_assignments,
        )
        warning = rf_ctx.get("rf_warning")

    if warning:
        return dmc.Alert(warning, color="orange", variant="light", withCloseButton=False), shown
    return "", hidden
