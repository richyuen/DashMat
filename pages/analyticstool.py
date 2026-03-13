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
from dash import ClientsideFunction, Input, Output, State, callback, dcc, html, no_update, register_page, ALL, clientside_callback, callback_context
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
    annualization_factor,
    calculate_calendar_year_returns,
    calculate_excess_returns,
    calculate_rolling_returns,
    create_monthly_view,
    df_to_json,
    get_available_periodicities,
    get_working_returns,
    get_working_returns_by_key,
    is_daily,
    json_to_df,
    merge_returns,
    resample_returns,
    resample_returns_cached,
)
from utils.help_links import ANALYTICSTOOL_HELP_URL
from utils.statistics import (
    calculate_drawdown,
    calculate_growth_of_dollar,
    calculate_statistics_cached,
    generate_correlogram_cached,
)
from utils.covariance import (
    format_cov_shrinkage_spec_label,
    resolve_cov_shrinkage_spec,
)
from utils.exponential_weighting import normalize_decay_input
from utils.charting import apply_chart_theme
from utils.ag_grid import literal_field_dash_grid_options
from utils.excel_export import format_excel_dates, write_excel_with_autofit
from utils.perf_timing import timed_block
from utils.qq import build_normal_qq_series, build_qq_figure, build_reference_qq_series
from utils.serialization import canonical_json_dumps, date_range_payload_for_cache, mapping_payload_for_cache
from utils.shared_metrics import (
    MARKET_BETA_SERIES,
    RISK_FREE_SERIES,
    STATS_CONFIG,
    risk_free_json_from_store as _risk_free_json_from_store,
    spx_json_from_store as _spx_json_from_store,
)
from utils.saved_series import saved_series_store_names
from utils.account_lists import (
    add_db_import_provenance_entry,
    prune_db_import_provenance,
    remove_db_import_provenance_series,
    rename_db_import_provenance_series,
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
from utils.underlying_category_imports import (
    expand_underlying_category_rows,
    get_underlying_category_desc_options,
    load_underlying_category_series,
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
from utils.raw_dataset import (
    build_raw_data_store_payload,
    get_raw_data_json_from_store,
    get_raw_dataset_df,
    get_raw_dataset_json,
    resolve_dataset_key,
)
from utils.factor_definitions import (
    FACTOR_AGG_TYPE_OPTIONS,
    OUTPUT_TRANSFORM_OPTIONS,
    compute_factor_preview_lines,
    compute_factor_series,
    compute_factor_series_cached,
    delete_factor_definition,
    factor_tables_available,
    get_sec_factor_component_options_cached,
    load_factor_definitions,
    save_factor_definition,
    validate_factor_definition_payload,
)
from utils.regime_analysis import (
    build_regime_detail_frame,
    build_regime_duration_table,
    regime_required_series,
    regime_series_store_names,
    resolve_regime_source_data,
    build_wide_detail_frame,
    build_regime_statistics_table,
    build_regime_timeline_frame,
    build_regime_transition_matrix,
    compute_regime_artifacts,
    compute_regime_assignments,
)
from utils.regime_definitions import (
    REGIME_METHOD_OPTIONS,
    delete_regime_definition,
    load_regime_definitions,
    regime_tables_available,
    save_regime_definition,
    validate_regime_definition_payload,
)

register_page(__name__, path="/analyticstool", name="Analytics Tool", title="Analytics Tool")

# Performance optimization constants

SAVED_SERIES_CONFIG = {
    RISK_FREE_SERIES: {},
    MARKET_BETA_SERIES: {"start_date": "1988-01-04"},
}

CONDITIONAL_VIEW_OPTIONS = [
    {"value": "coincident", "label": "Coincident"},
    {"value": "forward", "label": "Forward"},
]

CONDITIONAL_DISPLAY_MODE_OPTIONS = [
    {"value": "summary", "label": "Summary"},
    {"value": "detail", "label": "Detail"},
]

REGIME_DETAIL_DISPLAY_MODE_OPTIONS = [
    {"value": "summary", "label": "Summary"},
    {"value": "detail", "label": "Raw Detail"},
]

CONDITIONAL_COMPARATOR_OPTIONS = [
    {"value": "le", "label": "<="},
    {"value": "ge", "label": ">="},
]

CONDITIONAL_FACTOR_CONVERSION_OPTIONS = [
    {"value": "compound", "label": "Compound Return"},
    {"value": "end", "label": "End of Period"},
    {"value": "average", "label": "Average"},
    {"value": "sum", "label": "Sum"},
]

ANALYSIS_DETAIL_RENDER_CELL_WARNING_THRESHOLD = 200000

AT_WELCOME_MODAL_CONFIG = PagePrefixConfig(
    prefix="at",
    page_icon="tabler:chart-line",
    page_title="Welcome to the Analytics Tool",
    page_subtitle="Choose a source to load data and get started.",
    series_modal_size="80vw",
    series_modal_max_width="1250px",
    series_modal_transition_ms=180,
    welcome_switch_buttons=(),
)


def _build_help_control() -> dmc.Anchor | dmc.Button:
    help_button = dmc.Button(
        "Help",
        id="at-menu-help-guide",
        variant="gradient",
        gradient={"from": "teal", "to": "cyan", "deg": 90},
        size="sm",
        radius="xl",
        className="dashmat-menu-trigger",
        leftSection=DashIconify(icon="tabler:help-circle", width=14),
        disabled=not ANALYTICSTOOL_HELP_URL.strip(),
    )
    if not ANALYTICSTOOL_HELP_URL.strip():
        return help_button
    return dmc.Anchor(
        help_button,
        href=ANALYTICSTOOL_HELP_URL.strip(),
        target="_blank",
        style={"textDecoration": "none"},
    )


def _mapping_payload(value) -> str:
    return mapping_payload_for_cache(value)


def _date_range_payload(value) -> str:
    return date_range_payload_for_cache(value)


def _dataset_key(raw_data_store) -> str | None:
    return resolve_dataset_key(raw_data_store) if raw_data_store else None


def _raw_json(raw_data_store) -> str | None:
    if not raw_data_store:
        return None
    if isinstance(raw_data_store, str):
        try:
            return get_raw_data_json_from_store(raw_data_store)
        except Exception:
            return raw_data_store
    return get_raw_data_json_from_store(raw_data_store)


def _raw_df(raw_data_store) -> pd.DataFrame:
    dataset_key = _dataset_key(raw_data_store)
    return get_raw_dataset_df(dataset_key) if dataset_key else pd.DataFrame()


def _has_complete_date_range(value) -> bool:
    return (
        isinstance(value, dict)
        and bool(value.get("start"))
        and bool(value.get("end"))
    )


def _coerce_positive_int(value, default: int = 1) -> int:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        return default
    return max(default, int(parsed))


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
    exp_weighted,
    decay_value,
    shrinkage,
    shrinkage_target,
):
    payload = "|".join(
        [
            str(_dataset_key(raw_data) or ""),
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
            str(bool(exp_weighted)),
            str(normalize_decay_input(decay_value, 63.0)),
            str(shrinkage or "none"),
            str(shrinkage_target or "scaled_identity"),
        ]
    )
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _AnalyticsComputeBundle:
    dataset_key: str
    periodicity: str
    selected_series: tuple
    benchmark_payload: str
    long_short_payload: str
    date_range_payload: str
    vol_scaler: float
    vol_scaling_payload: str


@dataclass(frozen=True)
class _FactorArtifacts:
    dependent_df: pd.DataFrame
    factor_raw: pd.Series
    factor_display: pd.Series
    factor_display_name: str


@dataclass(frozen=True)
class _RegimeAnalysisPayload:
    definition: dict
    diagnostics: dict
    unresolved: tuple[str, ...]
    settings_df: pd.DataFrame
    timeline_df: pd.DataFrame
    stats_df: pd.DataFrame
    transition_df: pd.DataFrame
    duration_df: pd.DataFrame
    detail_df: pd.DataFrame
    signal_label: str


@dataclass(frozen=True)
class _RegimeAnalysisBuildResult:
    status: str
    message: str | None = None
    payload: _RegimeAnalysisPayload | None = None


@dataclass(frozen=True)
class _ConditionalReturnsPayload:
    factor_label: str
    factor_display_name: str
    coincident_mean_df: pd.DataFrame
    coincident_count_df: pd.DataFrame
    forward_mean_by_series: dict[str, pd.DataFrame]
    forward_count_by_series: dict[str, pd.DataFrame]
    coincident_detail_df: pd.DataFrame
    forward_detail_df: pd.DataFrame
    coincident_row_count: int
    forward_row_count: int


@dataclass(frozen=True)
class _ConditionalCoreArtifacts:
    factor_label: str
    factor_display_name: str
    window_labels: tuple[str, ...]
    anchor_index: pd.DatetimeIndex
    factor_windows: dict[str, pd.Series]
    qualified_masks: dict[str, pd.Series]
    coincident_series_windows: dict[str, dict[str, pd.Series]]
    forward_series_windows: dict[str, dict[str, pd.Series]]
    coincident_row_count: int
    forward_row_count: int


@dataclass(frozen=True)
class _ExcelSheetSpec:
    name: str
    frame: pd.DataFrame
    write_index: bool = False
    format_index: bool = False


@dataclass(frozen=True)
class _AnalyticsExportArtifacts:
    returns_df: pd.DataFrame
    stats_df: pd.DataFrame
    corr_df: pd.DataFrame
    cov_df: pd.DataFrame


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
        dataset_key=_dataset_key(raw_data) or "",
        periodicity=periodicity or "daily",
        selected_series=tuple(selected_series or ()),
        benchmark_payload=_mapping_payload(benchmark_assignments),
        long_short_payload=_mapping_payload(long_short_assignments),
        date_range_payload=_date_range_payload(date_range),
        vol_scaler=vol_scaler or 0,
        vol_scaling_payload=_mapping_payload(vol_scaling_assignments),
    )


@cache_config.cache.memoize(timeout=0)
def _compute_selected_returns_cached(
    dataset_key: str,
    periodicity: str,
    selected_series: tuple,
    returns_type: str,
    benchmark_payload: str,
    long_short_payload: str,
    date_range_payload: str,
    vol_scaler: float,
    vol_scaling_payload: str,
) -> pd.DataFrame:
    selected_tuple = tuple(selected_series or ())
    if not dataset_key or not selected_tuple:
        return pd.DataFrame()

    periodicity_value = periodicity or "daily"
    returns_basis = returns_type or "total"
    if returns_basis == "excess":
        selected_returns_df = calculate_excess_returns(
            dataset_key,
            periodicity_value,
            selected_tuple,
            benchmark_payload,
            "excess",
            long_short_payload,
            date_range_payload,
            vol_scaler,
            vol_scaling_payload,
        )
    else:
        selected_returns_df = get_working_returns_by_key(
            dataset_key,
            periodicity_value,
            selected_tuple,
            benchmark_payload,
            long_short_payload,
            date_range_payload,
            vol_scaler,
            vol_scaling_payload,
        )
        selected_returns_df = selected_returns_df[[c for c in selected_tuple if c in selected_returns_df.columns]]

    if selected_returns_df.empty:
        return pd.DataFrame()

    ordered_cols = [c for c in selected_tuple if c in selected_returns_df.columns]
    if not ordered_cols:
        return pd.DataFrame(index=selected_returns_df.index)
    return selected_returns_df.reindex(columns=ordered_cols).dropna(how="all")


def _compute_selected_returns(
    raw_data,
    periodicity,
    selected_series,
    returns_type,
    benchmark_assignments,
    long_short_assignments,
    date_range,
    vol_scaler,
    vol_scaling_assignments,
) -> pd.DataFrame:
    return _compute_selected_returns_cached(
        _dataset_key(raw_data) or "",
        periodicity or "daily",
        tuple(selected_series or ()),
        returns_type or "total",
        _mapping_payload(benchmark_assignments),
        _mapping_payload(long_short_assignments),
        _date_range_payload(date_range),
        vol_scaler or 0,
        _mapping_payload(vol_scaling_assignments),
    )


def _build_regime_warning_text(diagnostics, unresolved: tuple[str, ...]) -> str:
    warning_text = str((diagnostics or {}).get("warning") or "").strip()
    if warning_text and unresolved:
        return f"{warning_text}; Missing source series: {', '.join(unresolved)}"
    if unresolved:
        return f"Missing source series: {', '.join(unresolved)}"
    return warning_text


@cache_config.cache.memoize(timeout=0)
def _compute_regime_analysis_outputs_cached(
    raw_data: str,
    periodicity: str,
    selected_series: tuple,
    returns_type: str,
    benchmark_payload: str,
    long_short_payload: str,
    date_range_payload: str,
    vol_scaler: float,
    vol_scaling_payload: str,
    regime_definition_payload: str,
) -> dict[str, object]:
    try:
        definition = json.loads(str(regime_definition_payload or "{}"))
        if not isinstance(definition, dict):
            definition = {}
    except Exception:
        definition = {}
    try:
        date_range = json.loads(str(date_range_payload or "null"))
    except Exception:
        date_range = date_range_payload

    artifacts = compute_regime_artifacts(
        raw_data=raw_data,
        periodicity=periodicity or "daily",
        definition=definition,
        date_range=date_range,
    )
    states = artifacts.get("states", pd.Series(dtype="Int64", name="Regime"))
    diagnostics = dict(artifacts.get("diagnostics", {}) or {})
    signal = artifacts.get("signal", pd.Series(dtype=float))
    signal_label = str(artifacts.get("signal_label") or getattr(signal, "name", None) or "Regime Signal")
    if not isinstance(states, pd.Series) or states.empty:
        return {
            "status": "no_assignments",
            "diagnostics": diagnostics,
            "signal_label": signal_label,
            "settings_df": pd.DataFrame(),
            "timeline_df": pd.DataFrame(),
            "stats_df": pd.DataFrame(),
            "transition_df": pd.DataFrame(),
            "duration_df": pd.DataFrame(),
            "detail_df": pd.DataFrame(),
        }

    selected_tuple = tuple(selected_series or ())
    regime_dataset_key = (
        build_raw_data_store_payload(raw_data).get("dataset_key")
        if raw_data
        else ""
    )
    selected_returns_df = _compute_selected_returns_cached(
        regime_dataset_key,
        periodicity or "daily",
        selected_tuple,
        returns_type or "total",
        benchmark_payload,
        long_short_payload,
        date_range_payload,
        vol_scaler,
        vol_scaling_payload,
    )

    settings_df = pd.DataFrame(
        [
            {
                "RegimeName": definition.get("RegimeName"),
                "MethodType": diagnostics.get("method_type"),
                "Signal Method": diagnostics.get("method"),
                "Signal Label": signal_label,
                "Signal Return Basis": (
                    ((definition.get("Config") or {}) if isinstance(definition, dict) else {}).get("return_basis") or "total"
                ),
                "PC1 Standardized": (
                    ((definition.get("Config") or {}) if isinstance(definition, dict) else {}).get("pca_standardize")
                    if int(diagnostics.get("method_type") or 0) in {1, 2}
                    else None
                ),
                "Series Return Basis": "excess" if returns_type == "excess" else "total",
                "NumRegimes": diagnostics.get("num_regimes"),
                "Observations": diagnostics.get("observations"),
                "Warning": diagnostics.get("warning"),
            }
        ]
    )
    timeline_df = build_regime_timeline_frame(states)
    stats_df = build_regime_statistics_table(
        selected_returns_df,
        states,
        periodicity,
        selected_series=selected_tuple,
        benchmark_assignments=benchmark_payload,
        long_short_assignments=long_short_payload,
    )
    transition_df = build_regime_transition_matrix(states)
    duration_df = build_regime_duration_table(states)
    detail_df = build_regime_detail_frame(selected_returns_df, states, signal, signal_label)
    return {
        "status": "ok",
        "diagnostics": diagnostics,
        "signal_label": signal_label,
        "settings_df": settings_df,
        "timeline_df": timeline_df,
        "stats_df": stats_df,
        "transition_df": transition_df,
        "duration_df": duration_df,
        "detail_df": detail_df,
    }


def _build_regime_analysis_payload(
    raw_data,
    periodicity,
    selected_series,
    returns_type,
    benchmark_assignments,
    long_short_assignments,
    date_range,
    vol_scaler,
    vol_scaling_assignments,
    regime_definition_key,
    regime_definitions_db,
    regime_definitions_local,
    regime_series_store,
) -> _RegimeAnalysisBuildResult:
    if not regime_definition_key:
        return _RegimeAnalysisBuildResult("missing_definition", "Select a regime definition.")

    _regime_prefix, regime_name = _split_regime_select_key(regime_definition_key)
    definition = _lookup_regime_definition(regime_name, regime_definitions_db, regime_definitions_local)
    if not definition:
        return _RegimeAnalysisBuildResult("definition_unavailable", "Selected regime definition is unavailable.")

    required_series = regime_required_series(definition)
    combined_raw_data, _next_regime_series_store, _resolved, unresolved = resolve_regime_source_data(
        raw_data=raw_data,
        regime_series_store=regime_series_store,
        required_series=required_series,
        db_engine=DB_ENGINE,
        mrd_engine=MRD_ENGINE,
    )
    if not combined_raw_data:
        return _RegimeAnalysisBuildResult(
            "no_source_data",
            "Unable to resolve required source series for regime analysis.",
        )

    bundle = _build_analytics_compute_bundle(
        build_raw_data_store_payload(combined_raw_data),
        periodicity,
        selected_series,
        benchmark_assignments,
        long_short_assignments,
        date_range,
        vol_scaler,
        vol_scaling_assignments,
    )
    regime_outputs = _compute_regime_analysis_outputs_cached(
        combined_raw_data or "",
        bundle.periodicity,
        bundle.selected_series,
        returns_type or "total",
        bundle.benchmark_payload,
        bundle.long_short_payload,
        bundle.date_range_payload,
        bundle.vol_scaler,
        bundle.vol_scaling_payload,
        canonical_json_dumps(definition),
    )
    diagnostics = dict(regime_outputs.get("diagnostics", {}) or {})
    unresolved_tuple = tuple(str(item) for item in (unresolved or []) if str(item).strip())
    if regime_outputs.get("status") != "ok":
        warning = str((diagnostics or {}).get("warning") or "No regime assignments were produced.")
        if unresolved_tuple:
            warning = f"{warning} Missing source series: {', '.join(unresolved_tuple)}."
        return _RegimeAnalysisBuildResult("no_assignments", warning)
    detail_df = regime_outputs.get("detail_df", pd.DataFrame())
    if detail_df.empty:
        return _RegimeAnalysisBuildResult("no_selected_returns", "No selected series returns available for current settings.")
    settings_df = regime_outputs.get("settings_df", pd.DataFrame()).copy()
    if settings_df.empty:
        settings_df = pd.DataFrame([{"RegimeName": definition.get("RegimeName")}])
    settings_df["Warning"] = _build_regime_warning_text(diagnostics, unresolved_tuple) or None
    return _RegimeAnalysisBuildResult(
        "ok",
        payload=_RegimeAnalysisPayload(
            definition=dict(definition),
            diagnostics=dict(diagnostics or {}),
            unresolved=unresolved_tuple,
            settings_df=settings_df,
            timeline_df=regime_outputs.get("timeline_df", pd.DataFrame()),
            stats_df=regime_outputs.get("stats_df", pd.DataFrame()),
            transition_df=regime_outputs.get("transition_df", pd.DataFrame()),
            duration_df=regime_outputs.get("duration_df", pd.DataFrame()),
            detail_df=detail_df,
            signal_label=str(regime_outputs.get("signal_label") or "Regime Signal"),
        ),
    )


def _normalize_monthly_df_if_needed(df: pd.DataFrame, periodicity: str) -> pd.DataFrame:
    """Canonicalize monthly indexes only when the workflow is monthly."""
    if periodicity == "monthly":
        return align_monthly_index_to_month_end(df)
    return df


def _factor_user_label(userinfo):
    role = str((userinfo or {}).get("role", "")).strip() or "Unknown"
    os_user = str((userinfo or {}).get("username", "")).strip()
    if not os_user:
        try:
            import getpass

            os_user = str(getpass.getuser() or "").strip()
        except Exception:
            os_user = ""
    if os_user:
        return f"{role}:{os_user}"
    return role


def _source_badge(source: str) -> str:
    return "[DB]" if str(source or "").strip().lower() == "db" else "[Session]"


def _default_factor_draft() -> dict:
    return {
        "selected_key": None,
        "source": "session",
        "DraftMode": "new",
        "sync_origin": "system",
        "original_name": None,
        "selected_update_date": None,
        "FactorName": "",
        "Description": "",
        "LongComponentList": [],
        "ShortComponentList": [],
        "LongAggType": 1,
        "ShortAggType": None,
        "LongLag": 0,
        "OutputTransform": 0,
    }


def _ensure_factor_draft(value):
    if not isinstance(value, dict):
        return _default_factor_draft()
    draft = _default_factor_draft()
    draft.update(value)
    draft["LongComponentList"] = [str(v) for v in draft.get("LongComponentList") or []]
    draft["ShortComponentList"] = [str(v) for v in draft.get("ShortComponentList") or []]
    draft["LongAggType"] = int(pd.to_numeric(pd.Series([draft.get("LongAggType", 1)]), errors="coerce").iloc[0] or 1)
    short_agg_num = pd.to_numeric(pd.Series([draft.get("ShortAggType")]), errors="coerce").iloc[0]
    draft["ShortAggType"] = int(short_agg_num) if not pd.isna(short_agg_num) else None
    draft["LongLag"] = int(pd.to_numeric(pd.Series([draft.get("LongLag", 0)]), errors="coerce").iloc[0] or 0)
    draft["OutputTransform"] = int(
        pd.to_numeric(pd.Series([draft.get("OutputTransform", 0)]), errors="coerce").iloc[0] or 0
    )
    draft_mode = str(draft.get("DraftMode") or "").strip().lower()
    if draft_mode not in {"db", "session", "new"}:
        if str(draft.get("source") or "").strip().lower() == "db":
            draft_mode = "db"
        elif draft.get("selected_key"):
            draft_mode = "session"
        else:
            draft_mode = "new"
    draft["DraftMode"] = draft_mode
    if draft_mode == "db":
        draft["source"] = "db"
    else:
        draft["source"] = "session"
    if draft_mode == "new":
        draft["selected_key"] = None
        draft["original_name"] = None
        draft["selected_update_date"] = None
    return draft


def _factor_select_key(source: str, name: str) -> str:
    return f"{source}::{name}"


def _split_factor_select_key(value: str | None) -> tuple[str | None, str | None]:
    text_val = str(value or "").strip()
    if not text_val:
        return None, None
    if "::" not in text_val:
        return None, text_val
    prefix, name = text_val.split("::", 1)
    return prefix, name


def _index_factor_definitions(definitions):
    out = {}
    for item in (definitions or []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("FactorName", "")).strip()
        if not name:
            continue
        out[name.lower()] = item
    return out


def _lookup_factor_definition(factor_name, db_definitions, local_definitions):
    key = str(factor_name or "").strip().lower()
    if not key:
        return None
    db_idx = _index_factor_definitions(db_definitions)
    if key in db_idx:
        return db_idx[key]
    local_idx = _index_factor_definitions(local_definitions)
    return local_idx.get(key)


def _normalize_factor_value_for_options(value, raw_names, definition_names):
    if not value:
        return None
    val = str(value).strip()
    if not val:
        return None
    if val.startswith("raw::"):
        raw_name = val.split("::", 1)[1]
        return val if raw_name in raw_names else None
    if val.startswith("def::"):
        def_name = val.split("::", 1)[1]
        return val if def_name in definition_names else None
    if val in raw_names:
        return f"raw::{val}"
    if val in definition_names:
        return f"def::{val}"
    return None


def _factor_option_definitions(db_definitions, local_definitions):
    out = []
    seen = set()
    for item in (db_definitions or []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("FactorName", "")).strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append({"name": name, "source": "db"})
    for item in (local_definitions or []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("FactorName", "")).strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append({"name": name, "source": "session"})
    return out


def _definition_payload_for_compute(definition: dict) -> str:
    return canonical_json_dumps(
        {
            "FactorName": definition.get("FactorName"),
            "LongComponent": definition.get("LongComponent"),
            "ShortComponent": definition.get("ShortComponent"),
            "LongAggType": definition.get("LongAggType"),
            "ShortAggType": definition.get("ShortAggType"),
            "LongLag": definition.get("LongLag"),
            "OutputTransform": definition.get("OutputTransform"),
            "UPDATE_DATE": definition.get("UPDATE_DATE"),
        }
    )


def _factor_definition_signature(definition: dict | None) -> str:
    if not isinstance(definition, dict):
        return ""
    normalized, error = validate_factor_definition_payload(
        {
            "FactorName": definition.get("FactorName"),
            "LongComponent": definition.get("LongComponent", definition.get("LongComponentList")),
            "ShortComponent": definition.get("ShortComponent", definition.get("ShortComponentList")),
            "Description": definition.get("Description"),
            "LongAggType": definition.get("LongAggType"),
            "ShortAggType": definition.get("ShortAggType"),
            "LongLag": definition.get("LongLag"),
            "OutputTransform": definition.get("OutputTransform"),
        }
    )
    if error or not normalized:
        return ""
    return canonical_json_dumps(
        {
            "FactorName": normalized.get("FactorName"),
            "LongComponent": normalized.get("LongComponent"),
            "ShortComponent": normalized.get("ShortComponent"),
            "Description": normalized.get("Description"),
            "LongAggType": normalized.get("LongAggType"),
            "ShortAggType": normalized.get("ShortAggType"),
            "LongLag": normalized.get("LongLag"),
            "OutputTransform": normalized.get("OutputTransform"),
        }
    )


def _factor_db_name_exists(name: str, db_definitions) -> bool:
    key = str(name or "").strip().lower()
    if not key:
        return False
    for item in (db_definitions or []):
        if not isinstance(item, dict):
            continue
        if str(item.get("FactorName", "")).strip().lower() == key:
            return True
    return False


def _definition_to_draft(definition: dict, source: str, selected_key: str | None = None) -> dict:
    base = _ensure_factor_draft(definition)
    name = str(base.get("FactorName", "")).strip()
    if selected_key is None and name:
        selected_key = _factor_select_key(source, name)
    base["selected_key"] = selected_key
    base["source"] = source
    base["DraftMode"] = "db" if source == "db" else "session"
    base["sync_origin"] = "system"
    base["original_name"] = name if source == "db" else base.get("original_name") or name
    base["selected_update_date"] = base.get("UPDATE_DATE")
    return base


def _draft_to_definition_payload(draft: dict) -> dict:
    draft_value = _ensure_factor_draft(draft)
    return {
        "FactorName": draft_value.get("FactorName"),
        "Description": draft_value.get("Description"),
        "LongComponent": draft_value.get("LongComponentList"),
        "ShortComponent": draft_value.get("ShortComponentList"),
        "LongAggType": draft_value.get("LongAggType"),
        "ShortAggType": draft_value.get("ShortAggType"),
        "LongLag": draft_value.get("LongLag"),
        "OutputTransform": draft_value.get("OutputTransform"),
        "UPDATE_DATE": draft_value.get("selected_update_date"),
        "UPDATE_BY": draft_value.get("UPDATE_BY"),
    }


def _default_regime_draft() -> dict:
    return {
        "selected_key": None,
        "source": "session",
        "DraftMode": "new",
        "sync_origin": "system",
        "original_name": None,
        "selected_update_date": None,
        "RegimeName": "",
        "Description": "",
        "MethodType": 1,
        "ReturnBasis": "total",
        "NumRegimes": 3,
        "MinObservations": 60,
        "PcaStandardize": True,
        "UniverseSeries": [],
        "SingleSeries": None,
        "VolScaler": 0.0,
        "BenchmarkAssignmentsJson": {},
        "LongShortAssignmentsJson": {},
        "VolScalingAssignmentsJson": {},
    }


def _ensure_regime_draft(value):
    if not isinstance(value, dict):
        return _default_regime_draft()
    draft = _default_regime_draft()
    draft.update(value)
    draft["MethodType"] = int(pd.to_numeric(pd.Series([draft.get("MethodType", 1)]), errors="coerce").iloc[0] or 1)
    # Return basis is temporarily fixed to total for regime definitions.
    draft["ReturnBasis"] = "total"
    draft["NumRegimes"] = int(pd.to_numeric(pd.Series([draft.get("NumRegimes", 3)]), errors="coerce").iloc[0] or 3)
    draft["NumRegimes"] = max(2, min(draft["NumRegimes"], 10))
    draft["MinObservations"] = int(pd.to_numeric(pd.Series([draft.get("MinObservations", 60)]), errors="coerce").iloc[0] or 60)
    draft["MinObservations"] = max(20, draft["MinObservations"])
    draft["PcaStandardize"] = bool(draft.get("PcaStandardize", True))
    draft["UniverseSeries"] = [str(v) for v in (draft.get("UniverseSeries") or []) if str(v).strip()]
    single_series = str(draft.get("SingleSeries") or "").strip()
    draft["SingleSeries"] = single_series or None
    draft["VolScaler"] = float(pd.to_numeric(pd.Series([draft.get("VolScaler", 0.0)]), errors="coerce").iloc[0] or 0.0)
    draft["VolScaler"] = max(0.0, draft["VolScaler"])
    for key in ("BenchmarkAssignmentsJson", "LongShortAssignmentsJson", "VolScalingAssignmentsJson"):
        value_map = draft.get(key, {})
        if not isinstance(value_map, dict):
            value_map = {}
        draft[key] = dict(value_map)
    draft_mode = str(draft.get("DraftMode") or "").strip().lower()
    if draft_mode not in {"db", "session", "new"}:
        if str(draft.get("source") or "").strip().lower() == "db":
            draft_mode = "db"
        elif draft.get("selected_key"):
            draft_mode = "session"
        else:
            draft_mode = "new"
    draft["DraftMode"] = draft_mode
    if draft_mode == "db":
        draft["source"] = "db"
    else:
        draft["source"] = "session"
    if draft_mode == "new":
        draft["selected_key"] = None
        draft["original_name"] = None
        draft["selected_update_date"] = None
    return draft


def _regime_select_key(source: str, name: str) -> str:
    return f"{source}::{name}"


def _split_regime_select_key(value: str | None) -> tuple[str | None, str | None]:
    text_val = str(value or "").strip()
    if not text_val:
        return None, None
    if "::" not in text_val:
        return None, text_val
    prefix, name = text_val.split("::", 1)
    return prefix, name


def _index_regime_definitions(definitions):
    out = {}
    for item in (definitions or []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("RegimeName", "")).strip()
        if not name:
            continue
        out[name.lower()] = item
    return out


def _lookup_regime_definition(regime_name, db_definitions, local_definitions):
    key = str(regime_name or "").strip().lower()
    if not key:
        return None
    db_idx = _index_regime_definitions(db_definitions)
    if key in db_idx:
        return db_idx[key]
    local_idx = _index_regime_definitions(local_definitions)
    return local_idx.get(key)


def _regime_option_definitions(db_definitions, local_definitions):
    out = []
    seen = set()
    for item in (db_definitions or []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("RegimeName", "")).strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append({"name": name, "source": "db"})
    for item in (local_definitions or []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("RegimeName", "")).strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append({"name": name, "source": "session"})
    return out


def _normalize_regime_value_for_options(value, regime_names):
    if not value:
        return None
    val = str(value).strip()
    if not val:
        return None
    if val.startswith("def::"):
        name = val.split("::", 1)[1]
        return val if name in regime_names else None
    if val in regime_names:
        return f"def::{val}"
    return None


def _regime_definition_to_draft(definition: dict, source: str, selected_key: str | None = None) -> dict:
    config = definition.get("Config", {}) if isinstance(definition, dict) else {}
    if not isinstance(config, dict):
        config = {}
    name = str(definition.get("RegimeName", "")).strip()
    if selected_key is None and name:
        selected_key = _regime_select_key(source, name)
    draft = _default_regime_draft()
    draft.update(
        {
            "selected_key": selected_key,
            "source": source,
            "DraftMode": "db" if source == "db" else "session",
            "sync_origin": "system",
            "original_name": name if source == "db" else definition.get("original_name") or name,
            "selected_update_date": definition.get("UPDATE_DATE"),
            "RegimeName": name,
            "Description": definition.get("Description") or "",
            "MethodType": definition.get("MethodType", config.get("method_type", 1)),
            "ReturnBasis": "total",
            "NumRegimes": config.get("num_regimes", 3),
            "MinObservations": config.get("min_observations", 60),
            "PcaStandardize": config.get("pca_standardize", True),
            "UniverseSeries": config.get("universe_series", []),
            "SingleSeries": config.get("single_series"),
            "VolScaler": config.get("vol_scaler", 0.0),
            "BenchmarkAssignmentsJson": config.get("benchmark_assignments", {}),
            "LongShortAssignmentsJson": config.get("long_short_assignments", {}),
            "VolScalingAssignmentsJson": config.get("vol_scaling_assignments", {}),
            "UPDATE_BY": definition.get("UPDATE_BY"),
        }
    )
    return _ensure_regime_draft(draft)


def _regime_draft_to_definition_payload(draft: dict) -> dict:
    draft_value = _ensure_regime_draft(draft)
    payload = {
        "RegimeName": draft_value.get("RegimeName"),
        "Description": draft_value.get("Description"),
        "MethodType": draft_value.get("MethodType"),
        "Config": {
            "num_regimes": draft_value.get("NumRegimes"),
            "return_basis": "total",
            "min_observations": draft_value.get("MinObservations"),
            "pca_standardize": draft_value.get("PcaStandardize"),
            "universe_series": draft_value.get("UniverseSeries"),
            "single_series": draft_value.get("SingleSeries"),
            "vol_scaler": draft_value.get("VolScaler"),
            "benchmark_assignments": draft_value.get("BenchmarkAssignmentsJson", {}),
            "long_short_assignments": draft_value.get("LongShortAssignmentsJson", {}),
            "vol_scaling_assignments": draft_value.get("VolScalingAssignmentsJson", {}),
        },
        "UPDATE_DATE": draft_value.get("selected_update_date"),
        "UPDATE_BY": draft_value.get("UPDATE_BY"),
    }
    return payload


def _regime_definition_signature(definition: dict | None) -> str:
    if not isinstance(definition, dict):
        return ""
    normalized, error = validate_regime_definition_payload(
        {
            "RegimeName": definition.get("RegimeName"),
            "Description": definition.get("Description"),
            "MethodType": definition.get("MethodType"),
            "Config": definition.get("Config"),
            "ConfigJson": definition.get("ConfigJson"),
        }
    )
    if error or not normalized:
        return ""
    return canonical_json_dumps(
        {
            "RegimeName": normalized.get("RegimeName"),
            "Description": normalized.get("Description"),
            "MethodType": normalized.get("MethodType"),
            "Config": normalized.get("Config"),
        }
    )


def _regime_db_name_exists(name: str, db_definitions) -> bool:
    key = str(name or "").strip().lower()
    if not key:
        return False
    for item in (db_definitions or []):
        if not isinstance(item, dict):
            continue
        if str(item.get("RegimeName", "")).strip().lower() == key:
            return True
    return False


def _build_regime_series_options(raw_data, selected_series, regime_series_store, draft_data=None):
    raw_series_order = []
    if raw_data:
        try:
            df = _raw_df(raw_data)
            all_series = list(df.columns)
            selected_order = [s for s in (selected_series or []) if s in all_series]
            remaining = [s for s in all_series if s not in selected_order]
            raw_series_order = selected_order + remaining
        except Exception:
            raw_series_order = []

    cached_regime_series = [
        name for name in regime_series_store_names(regime_series_store)
        if name not in set(raw_series_order)
    ]

    draft = _ensure_regime_draft(draft_data)
    draft_series = []
    seen_draft = set()
    for value in list(draft.get("UniverseSeries") or []) + [draft.get("SingleSeries")]:
        name = str(value or "").strip()
        if not name:
            continue
        key = name.lower()
        if key in seen_draft:
            continue
        seen_draft.add(key)
        draft_series.append(name)

    ordered_series = []
    labels_by_name = {}
    seen = set()

    def _append(name: str, label: str):
        key = name.lower()
        if key in seen:
            return
        seen.add(key)
        ordered_series.append(name)
        labels_by_name[name] = label

    for name in raw_series_order:
        _append(name, name)
    for name in cached_regime_series:
        _append(name, f"[Loaded for Regime] {name}")
    for name in draft_series:
        _append(name, f"[In Definition] {name}")

    options = [{"value": name, "label": labels_by_name[name]} for name in ordered_series]
    return options, ordered_series, raw_series_order


def _prepare_factor_base_frames(
    dataset_key,
    periodicity,
    selected_series,
    factor_series,
    returns_type,
    benchmark_assignments,
    long_short_assignments,
    date_range,
    vol_scaler,
    vol_scaling_assignments,
    factor_definitions_db=None,
    factor_definitions_local=None,
):
    """Prepare dependent-series returns and raw factor values for factor workflows."""
    if not dataset_key or not selected_series or not factor_series:
        return pd.DataFrame(), pd.Series(dtype=float)

    periodicity_value = periodicity or "daily"
    selected_tuple = tuple(selected_series or ())
    bench_payload = _mapping_payload(benchmark_assignments)
    ls_payload = _mapping_payload(long_short_assignments)
    date_payload = _date_range_payload(date_range)
    vol_scaler_value = vol_scaler or 0
    vol_payload = _mapping_payload(vol_scaling_assignments)

    dependent_df = _compute_selected_returns_cached(
        dataset_key,
        periodicity_value,
        selected_tuple,
        returns_type or "total",
        bench_payload,
        ls_payload,
        date_payload,
        vol_scaler_value,
        vol_payload,
    )
    if dependent_df.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    factor_values = pd.Series(dtype=float)
    factor_prefix, factor_name = _split_factor_select_key(factor_series)
    if factor_prefix == "def":
        definition = _lookup_factor_definition(factor_name, factor_definitions_db, factor_definitions_local)
        if not definition:
            return dependent_df, pd.Series(dtype=float)
        factor_values = compute_factor_series(
            MRD_ENGINE,
            {
                "FactorName": definition.get("FactorName"),
                "LongComponent": definition.get("LongComponent"),
                "ShortComponent": definition.get("ShortComponent"),
                "LongAggType": definition.get("LongAggType"),
                "ShortAggType": definition.get("ShortAggType"),
                "LongLag": definition.get("LongLag"),
                "OutputTransform": definition.get("OutputTransform"),
                "UPDATE_DATE": definition.get("UPDATE_DATE"),
            },
            periodicity_value,
            date_range,
        )
        factor_values.name = str(definition.get("FactorName") or factor_name)
    else:
        raw_factor_name = factor_name if factor_prefix == "raw" else str(factor_series or "")
        # Factor always comes from total-basis stream (with optional L/S if configured).
        factor_df = get_working_returns_by_key(
            dataset_key,
            periodicity_value,
            (raw_factor_name,),
            bench_payload,
            ls_payload,
            date_payload,
            vol_scaler_value,
            vol_payload,
        )
        if factor_df.empty or raw_factor_name not in factor_df.columns:
            return dependent_df, pd.Series(dtype=float)
        factor_values = factor_df[raw_factor_name]

    factor_values = factor_values.replace([np.inf, -np.inf], np.nan).dropna()
    if factor_values.empty:
        return dependent_df, pd.Series(dtype=float)

    return dependent_df, factor_values


def _empty_factor_artifacts() -> _FactorArtifacts:
    return _FactorArtifacts(
        dependent_df=pd.DataFrame(),
        factor_raw=pd.Series(dtype=float),
        factor_display=pd.Series(dtype=float),
        factor_display_name="",
    )


@cache_config.cache.memoize(timeout=0)
def _compute_factor_artifacts_cached(
    dataset_key: str,
    periodicity: str,
    selected_series: tuple,
    factor_series: str,
    returns_type: str,
    benchmark_payload: str,
    long_short_payload: str,
    date_range_payload: str,
    vol_scaler: float,
    vol_scaling_payload: str,
    factor_transform: str,
    factor_definition_payload: str,
) -> _FactorArtifacts:
    factor_definitions_local = None
    if factor_definition_payload:
        try:
            factor_definitions_local = [json.loads(factor_definition_payload)]
        except Exception:
            factor_definitions_local = None

    dependent_df, factor_raw = _prepare_factor_base_frames(
        dataset_key,
        periodicity,
        selected_series,
        factor_series,
        returns_type,
        benchmark_payload,
        long_short_payload,
        date_range_payload,
        vol_scaler,
        vol_scaling_payload,
        None,
        factor_definitions_local,
    )
    if dependent_df.empty or factor_raw.empty:
        return _empty_factor_artifacts()

    factor_display = factor_raw.copy()
    if factor_transform == "zscore":
        std = factor_display.std(ddof=0)
        if std and not np.isclose(std, 0.0):
            factor_display = (factor_display - factor_display.mean()) / std
        else:
            factor_display = pd.Series(0.0, index=factor_display.index, name=factor_display.name)

    factor_prefix, factor_name = _split_factor_select_key(factor_series)
    display_factor_name = factor_name if factor_name else str(factor_series or "")
    if factor_prefix == "raw" and not display_factor_name:
        display_factor_name = str(factor_raw.name or "")
    return _FactorArtifacts(
        dependent_df=dependent_df,
        factor_raw=factor_raw,
        factor_display=factor_display,
        factor_display_name=display_factor_name,
    )


def _compute_factor_artifacts(
    raw_data,
    periodicity,
    selected_series,
    factor_series,
    returns_type,
    benchmark_assignments,
    long_short_assignments,
    date_range,
    vol_scaler,
    vol_scaling_assignments,
    factor_transform,
    factor_definitions_db=None,
    factor_definitions_local=None,
) -> _FactorArtifacts:
    definition_payload = ""
    factor_prefix, factor_name = _split_factor_select_key(factor_series)
    if factor_prefix == "def":
        definition = _lookup_factor_definition(factor_name, factor_definitions_db, factor_definitions_local)
        if definition:
            definition_payload = _definition_payload_for_compute(definition)
    return _compute_factor_artifacts_cached(
        _dataset_key(raw_data) or "",
        periodicity or "daily",
        tuple(selected_series or ()),
        str(factor_series or ""),
        returns_type or "total",
        _mapping_payload(benchmark_assignments),
        _mapping_payload(long_short_assignments),
        _date_range_payload(date_range),
        vol_scaler or 0,
        _mapping_payload(vol_scaling_assignments),
        factor_transform if factor_transform in {"raw", "zscore"} else "raw",
        definition_payload,
    )


def _prepare_factor_analysis_frames(
    raw_data,
    periodicity,
    selected_series,
    factor_series,
    returns_type,
    benchmark_assignments,
    long_short_assignments,
    date_range,
    vol_scaler,
    vol_scaling_assignments,
    factor_transform,
    factor_definitions_db=None,
    factor_definitions_local=None,
):
    """Prepare dependent-series returns and factor returns for factor analysis."""
    artifacts = _compute_factor_artifacts(
        raw_data,
        periodicity,
        selected_series,
        factor_series,
        returns_type,
        benchmark_assignments,
        long_short_assignments,
        date_range,
        vol_scaler,
        vol_scaling_assignments,
        factor_transform,
        factor_definitions_db,
        factor_definitions_local,
    )
    return artifacts.dependent_df, artifacts.factor_display


def _build_factor_pair_df(factor_values: pd.Series, dependent_values: pd.Series) -> pd.DataFrame:
    """Align and clean factor/dependent observations for charting and export."""
    if factor_values is None or dependent_values is None:
        return pd.DataFrame(columns=["Factor", "Dependent"])
    paired = pd.concat(
        [
            factor_values.rename("Factor"),
            dependent_values.rename("Dependent"),
        ],
        axis=1,
    )
    paired = paired.replace([np.inf, -np.inf], np.nan).dropna()
    return paired


def _build_factor_detail_frame(artifacts: _FactorArtifacts, selected_series, quantiles: int) -> pd.DataFrame:
    if artifacts.dependent_df.empty or artifacts.factor_display.empty:
        return pd.DataFrame()

    quantile_labels, _ordered_labels = _factor_quantile_labels(artifacts.factor_display, quantiles)
    factor_name = "Factor Value"
    return build_wide_detail_frame(
        artifacts.dependent_df,
        [
            ("Quantile", quantile_labels.rename("Quantile")),
            (factor_name, artifacts.factor_display.rename(factor_name)),
        ],
        index_name="Date",
        value_columns=[col for col in (selected_series or []) if col in artifacts.dependent_df.columns],
        drop_all_missing_values=True,
    )


def _prepare_factor_analysis_selected_df(
    raw_data,
    periodicity,
    selected_series,
    returns_type,
    benchmark_assignments,
    long_short_assignments,
    date_range,
    vol_scaler,
    vol_scaling_assignments,
):
    return _compute_selected_returns(
        raw_data,
        periodicity,
        selected_series,
        returns_type,
        benchmark_assignments,
        long_short_assignments,
        date_range,
        vol_scaler,
        vol_scaling_assignments,
    )


def _is_weekly_periodicity(periodicity: str) -> bool:
    return str(periodicity or "").startswith("weekly_")


def _conditional_window_specs(periodicity: str) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    periodicity_value = periodicity or "daily"
    if is_daily(periodicity_value):
        specs.append({"key": "1w", "label": "1W", "kind": "days", "value": 7})
    elif _is_weekly_periodicity(periodicity_value):
        specs.append({"key": "1w", "label": "1W", "kind": "periods", "value": 1})

    for months in (1, 3, 6, 9, 12):
        specs.append(
            {
                "key": f"{months}m",
                "label": f"{months}M",
                "kind": "periods" if periodicity_value == "monthly" else "months",
                "value": months,
            }
        )
    return specs


def _resolve_conditional_anchor_positions(
    index: pd.DatetimeIndex,
    step_value,
    step_unit,
) -> np.ndarray:
    if len(index) == 0:
        return np.array([], dtype=int)

    step = _coerce_positive_int(step_value, default=1)
    if (step_unit or "months") != "months":
        return np.arange(0, len(index), step, dtype=int)

    anchors: list[int] = []
    current_anchor_date = pd.Timestamp(index[0]) + pd.offsets.MonthEnd(0)
    last_date = pd.Timestamp(index[-1])

    while current_anchor_date <= last_date:
        pos = index.searchsorted(current_anchor_date, side="right") - 1
        if pos >= 0 and (not anchors or pos > anchors[-1]):
            anchors.append(int(pos))
        current_anchor_date = current_anchor_date + pd.DateOffset(months=step)
        current_anchor_date = current_anchor_date + pd.offsets.MonthEnd(0)

    if not anchors:
        anchors.append(len(index) - 1)
    return np.asarray(anchors, dtype=int)


def _shift_index_by_months(index: pd.DatetimeIndex, months: int) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([pd.Timestamp(dt) + pd.DateOffset(months=months) for dt in index])


def _resolve_window_bounds(
    index: pd.DatetimeIndex,
    spec: dict[str, object],
    *,
    forward: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(index)
    positions = np.arange(n, dtype=int)
    kind = str(spec.get("kind") or "periods")
    value = int(spec.get("value") or 1)

    if kind == "periods":
        if forward:
            start_pos = positions + 1
            end_pos = positions + value
            valid = end_pos < n
        else:
            start_pos = positions - value + 1
            end_pos = positions
            valid = start_pos >= 0
        return start_pos, end_pos, valid

    if kind == "days":
        if forward:
            start_pos = positions + 1
            end_dates = index + pd.Timedelta(days=value)
            end_pos = index.searchsorted(end_dates, side="right") - 1
            valid = (start_pos < n) & (end_dates <= index[-1]) & (end_pos >= start_pos)
        else:
            start_dates = index - pd.Timedelta(days=value)
            start_pos = index.searchsorted(start_dates, side="right")
            end_pos = positions
            valid = start_dates >= index[0]
        return start_pos.astype(int), end_pos.astype(int), valid

    if forward:
        start_pos = positions + 1
        end_dates = _shift_index_by_months(index, value)
        end_pos = index.searchsorted(end_dates, side="right") - 1
        valid = (start_pos < n) & (end_dates <= index[-1]) & (end_pos >= start_pos)
    else:
        start_dates = _shift_index_by_months(index, -value)
        start_pos = index.searchsorted(start_dates, side="right")
        end_pos = positions
        valid = start_dates >= index[0]
    return start_pos.astype(int), end_pos.astype(int), valid


def _aggregate_window_values(
    series: pd.Series,
    start_pos: np.ndarray,
    end_pos: np.ndarray,
    valid_mask: np.ndarray,
    method: str,
) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, copy=False)
    n = len(values)
    result = np.full(n, np.nan, dtype=float)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    safe_start = np.clip(np.asarray(start_pos, dtype=int), 0, max(n - 1, 0))
    safe_end = np.clip(np.asarray(end_pos, dtype=int), 0, max(n - 1, 0))
    if n == 0:
        return pd.Series(dtype=float, index=series.index)

    if method == "end":
        ok = valid_mask & np.isfinite(values)
        result[ok] = values[ok]
        return pd.Series(result, index=series.index, dtype=float)

    lengths = (safe_end - safe_start + 1).astype(int)
    finite = np.isfinite(values)
    prefix_valid = np.concatenate(([0], np.cumsum(finite.astype(int))))

    if method in {"average", "sum"}:
        prefix_sum = np.concatenate(([0.0], np.cumsum(np.where(finite, values, 0.0))))
        sums = prefix_sum[safe_end + 1] - prefix_sum[safe_start]
        complete = valid_mask & (prefix_valid[safe_end + 1] - prefix_valid[safe_start] == lengths)
        result[complete] = sums[complete]
        if method == "average":
            result[complete] = result[complete] / lengths[complete]
        return pd.Series(result, index=series.index, dtype=float)

    compound_valid = finite & (values > -1.0)
    prefix_compound_valid = np.concatenate(([0], np.cumsum(compound_valid.astype(int))))
    prefix_log = np.concatenate(([0.0], np.cumsum(np.where(compound_valid, np.log1p(values), 0.0))))
    complete = valid_mask & (prefix_compound_valid[safe_end + 1] - prefix_compound_valid[safe_start] == lengths)
    result[complete] = np.expm1(prefix_log[safe_end + 1] - prefix_log[safe_start])[complete]
    return pd.Series(result, index=series.index, dtype=float)


def _apply_zscore(values: pd.Series) -> pd.Series:
    clean = values.replace([np.inf, -np.inf], np.nan)
    std = clean.std(ddof=0)
    if std and not np.isclose(std, 0.0):
        return (clean - clean.mean()) / std
    if clean.notna().any():
        return pd.Series(0.0, index=clean.index, dtype=float)
    return clean.astype(float)


def _conditional_conversion_tooltip_text() -> str:
    return (
        "Compound Return is usually most natural for return-like factors. "
        "End of Period or Average is often a better fit for level-like factors. "
        "Sum is usually most natural for additive factors."
    )


def _empty_conditional_returns_payload() -> _ConditionalReturnsPayload:
    return _ConditionalReturnsPayload(
        factor_label="",
        factor_display_name="",
        coincident_mean_df=pd.DataFrame(),
        coincident_count_df=pd.DataFrame(),
        forward_mean_by_series={},
        forward_count_by_series={},
        coincident_detail_df=pd.DataFrame(),
        forward_detail_df=pd.DataFrame(),
        coincident_row_count=0,
        forward_row_count=0,
    )


def _empty_conditional_core_artifacts() -> _ConditionalCoreArtifacts:
    return _ConditionalCoreArtifacts(
        factor_label="",
        factor_display_name="",
        window_labels=(),
        anchor_index=pd.DatetimeIndex([]),
        factor_windows={},
        qualified_masks={},
        coincident_series_windows={},
        forward_series_windows={},
        coincident_row_count=0,
        forward_row_count=0,
    )


def _estimate_conditional_detail_row_counts(
    index: pd.DatetimeIndex,
    periodicity: str,
    step_value,
    step_unit,
) -> tuple[int, int]:
    if len(index) == 0:
        return 0, 0
    window_specs = _conditional_window_specs(periodicity or "daily")
    anchor_count = len(_resolve_conditional_anchor_positions(index, step_value, step_unit))
    coincident_rows = anchor_count * len(window_specs)
    forward_rows = coincident_rows * len(window_specs)
    return coincident_rows, forward_rows


def _build_conditional_summary_frames_from_core(
    core: _ConditionalCoreArtifacts,
    series_names: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    window_labels = list(core.window_labels)
    mean_df = pd.DataFrame(index=window_labels, columns=list(series_names), dtype=float)
    count_df = pd.DataFrame(index=window_labels, columns=list(series_names), dtype=float)
    if not window_labels:
        return mean_df, count_df.fillna(0.0)

    for label in window_labels:
        qualified = core.qualified_masks.get(label)
        if qualified is None or qualified.empty:
            continue
        for series_name in series_names:
            series_window = core.coincident_series_windows.get(label, {}).get(series_name)
            if series_window is None or series_window.empty:
                continue
            selected_mask = qualified & series_window.notna()
            mean_df.loc[label, series_name] = series_window[selected_mask].mean() if selected_mask.any() else np.nan
            count_df.loc[label, series_name] = int(selected_mask.sum())

    return mean_df, count_df.fillna(0.0)


def _build_conditional_forward_summary_from_core(
    core: _ConditionalCoreArtifacts,
    series_names: tuple[str, ...],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    window_labels = list(core.window_labels)
    mean_by_series: dict[str, pd.DataFrame] = {}
    count_by_series: dict[str, pd.DataFrame] = {}

    for series_name in series_names:
        mean_df = pd.DataFrame(index=window_labels, columns=window_labels, dtype=float)
        count_df = pd.DataFrame(index=window_labels, columns=window_labels, dtype=float)
        forward_windows = core.forward_series_windows.get(series_name, {})
        for back_label in window_labels:
            qualified = core.qualified_masks.get(back_label)
            if qualified is None or qualified.empty:
                continue
            for horizon_label in window_labels:
                forward_window = forward_windows.get(horizon_label)
                if forward_window is None or forward_window.empty:
                    continue
                selected_mask = qualified & forward_window.notna()
                mean_df.loc[back_label, horizon_label] = (
                    forward_window[selected_mask].mean() if selected_mask.any() else np.nan
                )
                count_df.loc[back_label, horizon_label] = int(selected_mask.sum())
        mean_by_series[series_name] = mean_df
        count_by_series[series_name] = count_df.fillna(0.0)

    return mean_by_series, count_by_series


def _build_conditional_detail_frames_from_core(
    core: _ConditionalCoreArtifacts,
    series_names: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not core.window_labels or core.anchor_index.empty:
        return pd.DataFrame(), pd.DataFrame()

    anchor_index = core.anchor_index
    coincident_detail_frames: list[pd.DataFrame] = []
    forward_detail_frames: list[pd.DataFrame] = []
    lookback_labels = {label: pd.Series(label, index=anchor_index, dtype="object") for label in core.window_labels}
    forward_labels = {label: pd.Series(label, index=anchor_index, dtype="object") for label in core.window_labels}

    for label in core.window_labels:
        factor_window = core.factor_windows.get(label, pd.Series(dtype=float)).reindex(anchor_index)
        qualified = core.qualified_masks.get(label, pd.Series(dtype=bool)).reindex(anchor_index).fillna(False)
        value_frame = pd.DataFrame(index=anchor_index)
        for series_name in series_names:
            series_window = core.coincident_series_windows.get(label, {}).get(series_name)
            if series_window is not None:
                value_frame[series_name] = series_window.reindex(anchor_index).to_numpy(dtype=float, copy=False)
        base_frame = build_wide_detail_frame(
            value_frame,
            [
                ("Lookback", lookback_labels[label]),
                ("Factor Value", factor_window),
                ("Condition Met", qualified),
            ],
            index_name="End Date",
            value_columns=list(series_names),
            drop_all_missing_values=True,
            inputs_aligned=True,
        )
        coincident_detail_frames.append(base_frame)

        for horizon_label in core.window_labels:
            forward_value_frame = value_frame.copy()
            for series_name in series_names:
                forward_window = core.forward_series_windows.get(series_name, {}).get(horizon_label)
                if forward_window is not None:
                    forward_value_frame[series_name] = forward_window.reindex(anchor_index).to_numpy(dtype=float, copy=False)
            forward_frame = build_wide_detail_frame(
                forward_value_frame,
                [
                    ("Lookback", lookback_labels[label]),
                    ("Forward Period", forward_labels[horizon_label]),
                    ("Factor Value", factor_window),
                    ("Condition Met", qualified),
                ],
                index_name="End Date",
                value_columns=list(series_names),
                drop_all_missing_values=True,
                inputs_aligned=True,
            )
            forward_detail_frames.append(forward_frame)

    coincident_detail_df = _order_conditional_detail_frame(
        pd.concat(coincident_detail_frames, ignore_index=True, sort=False) if coincident_detail_frames else pd.DataFrame(),
        list(core.window_labels),
        include_forward=False,
    )
    forward_detail_df = _order_conditional_detail_frame(
        pd.concat(forward_detail_frames, ignore_index=True, sort=False) if forward_detail_frames else pd.DataFrame(),
        list(core.window_labels),
        include_forward=True,
    )
    return coincident_detail_df, forward_detail_df


def _order_conditional_detail_frame(
    detail_df: pd.DataFrame,
    window_labels: list[str],
    *,
    include_forward: bool,
) -> pd.DataFrame:
    if detail_df is None or detail_df.empty:
        return pd.DataFrame()

    frame = detail_df.copy()
    frame["Lookback"] = pd.Categorical(frame["Lookback"], categories=window_labels, ordered=True)
    ordered_columns = ["Lookback"]
    sort_cols = ["Lookback"]
    if include_forward:
        frame["Forward Period"] = pd.Categorical(frame["Forward Period"], categories=window_labels, ordered=True)
        ordered_columns.append("Forward Period")
        sort_cols.append("Forward Period")
    ordered_columns.extend(["End Date", "Factor Value", "Condition Met"])
    series_columns = [
        col for col in frame.columns
        if col not in {"Lookback", "Forward Period", "End Date", "Factor Value", "Condition Met"}
    ]
    ordered_columns.extend(series_columns)
    sort_cols.append("End Date")
    frame = frame.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    frame["Lookback"] = frame["Lookback"].astype(str)
    if include_forward:
        frame["Forward Period"] = frame["Forward Period"].astype(str)
    return frame.loc[:, [col for col in ordered_columns if col in frame.columns]]


@cache_config.cache.memoize(timeout=0)
def _compute_conditional_core_cached(
    dataset_key: str,
    periodicity: str,
    selected_series: tuple,
    returns_type: str,
    benchmark_payload: str,
    long_short_payload: str,
    date_range_payload: str,
    vol_scaler: float,
    vol_scaling_payload: str,
    factor_series: str,
    factor_transform: str,
    factor_definition_payload: str,
    comparator: str,
    threshold: float,
    window_conversion: str,
    step_value: int,
    step_unit: str,
) -> _ConditionalCoreArtifacts:
    factor_definitions_local = None
    if factor_definition_payload:
        try:
            factor_definitions_local = [json.loads(factor_definition_payload)]
        except Exception:
            factor_definitions_local = None

    dependent_df, factor_values = _prepare_factor_base_frames(
        dataset_key,
        periodicity,
        selected_series,
        factor_series,
        returns_type,
        benchmark_payload,
        long_short_payload,
        date_range_payload,
        vol_scaler,
        vol_scaling_payload,
        None,
        factor_definitions_local,
    )

    factor_values = factor_values.replace([np.inf, -np.inf], np.nan).dropna()
    if dependent_df.empty or factor_values.empty:
        return _empty_conditional_core_artifacts()

    _factor_prefix, factor_name = _split_factor_select_key(factor_series)
    display_factor_name = factor_name if factor_name else str(factor_series or "")
    factor_label = f"{display_factor_name} (Z-Score)" if factor_transform == "zscore" else display_factor_name

    master_index = pd.DatetimeIndex(dependent_df.index.union(factor_values.index).sort_values().unique())
    dependent_aligned = dependent_df.reindex(master_index)
    factor_aligned = factor_values.reindex(master_index)

    window_specs = _conditional_window_specs(periodicity or "daily")
    window_labels = tuple(str(spec["label"]) for spec in window_specs)
    anchor_positions = _resolve_conditional_anchor_positions(master_index, step_value, step_unit)
    anchor_index = master_index[anchor_positions]
    threshold_value = float(pd.to_numeric(pd.Series([threshold]), errors="coerce").iloc[0] or 0.0)

    qualified_masks: dict[str, pd.Series] = {}
    factor_windows: dict[str, pd.Series] = {}
    coincident_series_windows: dict[str, dict[str, pd.Series]] = {}
    forward_series_windows: dict[str, dict[str, pd.Series]] = {}
    eval_mask = np.zeros(len(master_index), dtype=bool)
    eval_mask[anchor_positions] = True

    for spec in window_specs:
        label = str(spec["label"])
        start_pos, end_pos, valid_mask = _resolve_window_bounds(master_index, spec, forward=False)
        factor_window = _aggregate_window_values(factor_aligned, start_pos, end_pos, valid_mask, window_conversion)
        if factor_transform == "zscore":
            factor_window = _apply_zscore(factor_window)
        factor_windows[label] = factor_window

        qualified = pd.Series(False, index=master_index, dtype=bool)
        if comparator == "ge":
            qualified.loc[eval_mask] = factor_window.loc[eval_mask] >= threshold_value
        else:
            qualified.loc[eval_mask] = factor_window.loc[eval_mask] <= threshold_value
        qualified_masks[label] = qualified
        coincident_series_windows[label] = {}

        for series_name in selected_series:
            if series_name not in dependent_aligned.columns:
                continue
            series_values = dependent_aligned[series_name]
            series_window = _aggregate_window_values(series_values, start_pos, end_pos, valid_mask, "compound")
            coincident_series_windows[label][series_name] = series_window

    for series_name in selected_series:
        if series_name not in dependent_aligned.columns:
            continue
        series_values = dependent_aligned[series_name]

        for horizon_spec in window_specs:
            start_pos, end_pos, valid_mask = _resolve_window_bounds(master_index, horizon_spec, forward=True)
            series_window = _aggregate_window_values(series_values, start_pos, end_pos, valid_mask, "compound")
            forward_series_windows.setdefault(series_name, {})[str(horizon_spec["label"])] = series_window

    coincident_rows, forward_rows = _estimate_conditional_detail_row_counts(
        master_index,
        periodicity,
        step_value,
        step_unit,
    )

    return _ConditionalCoreArtifacts(
        factor_label=factor_label,
        factor_display_name=display_factor_name,
        window_labels=window_labels,
        anchor_index=anchor_index,
        factor_windows=factor_windows,
        qualified_masks=qualified_masks,
        coincident_series_windows=coincident_series_windows,
        forward_series_windows=forward_series_windows,
        coincident_row_count=coincident_rows,
        forward_row_count=forward_rows,
    )


@cache_config.cache.memoize(timeout=0)
def _compute_conditional_returns_cached(
    dataset_key: str,
    periodicity: str,
    selected_series: tuple,
    returns_type: str,
    benchmark_payload: str,
    long_short_payload: str,
    date_range_payload: str,
    vol_scaler: float,
    vol_scaling_payload: str,
    factor_series: str,
    factor_transform: str,
    factor_definition_payload: str,
    comparator: str,
    threshold: float,
    window_conversion: str,
    step_value: int,
    step_unit: str,
    include_detail: bool = False,
) -> _ConditionalReturnsPayload:
    core = _compute_conditional_core_cached(
        dataset_key,
        periodicity,
        selected_series,
        returns_type,
        benchmark_payload,
        long_short_payload,
        date_range_payload,
        vol_scaler,
        vol_scaling_payload,
        factor_series,
        factor_transform,
        factor_definition_payload,
        comparator,
        threshold,
        window_conversion,
        step_value,
        step_unit,
    )
    if not core.window_labels:
        return _empty_conditional_returns_payload()

    selected_tuple = tuple(selected_series or ())
    coincident_mean, coincident_count = _build_conditional_summary_frames_from_core(core, selected_tuple)
    forward_mean_by_series, forward_count_by_series = _build_conditional_forward_summary_from_core(core, selected_tuple)

    coincident_detail_df = pd.DataFrame()
    forward_detail_df = pd.DataFrame()
    if include_detail:
        coincident_detail_df, forward_detail_df = _build_conditional_detail_frames_from_core(core, selected_tuple)

    return _ConditionalReturnsPayload(
        factor_label=core.factor_label,
        factor_display_name=core.factor_display_name,
        coincident_mean_df=coincident_mean,
        coincident_count_df=coincident_count,
        forward_mean_by_series=forward_mean_by_series,
        forward_count_by_series=forward_count_by_series,
        coincident_detail_df=coincident_detail_df,
        forward_detail_df=forward_detail_df,
        coincident_row_count=core.coincident_row_count,
        forward_row_count=core.forward_row_count,
    )


def _prepare_at_qq_reference_series(
    raw_data,
    periodicity,
    reference_series,
    returns_type,
    benchmark_assignments,
    long_short_assignments,
    date_range,
    vol_scaler,
    vol_scaling_assignments,
    factor_definitions_db=None,
    factor_definitions_local=None,
):
    if not raw_data or not reference_series:
        return pd.Series(dtype=float)
    reference_prefix, reference_name = _split_factor_select_key(reference_series)
    if reference_prefix == "def":
        definition = _lookup_factor_definition(reference_name, factor_definitions_db, factor_definitions_local)
        if not definition:
            return pd.Series(dtype=float)
        ref_values = compute_factor_series(
            MRD_ENGINE,
            {
                "FactorName": definition.get("FactorName"),
                "LongComponent": definition.get("LongComponent"),
                "ShortComponent": definition.get("ShortComponent"),
                "LongAggType": definition.get("LongAggType"),
                "ShortAggType": definition.get("ShortAggType"),
                "LongLag": definition.get("LongLag"),
                "OutputTransform": definition.get("OutputTransform"),
                "UPDATE_DATE": definition.get("UPDATE_DATE"),
            },
            periodicity or "daily",
            date_range,
        )
        ref_values.name = str(definition.get("FactorName") or reference_name)
        return ref_values.replace([np.inf, -np.inf], np.nan).dropna()

    raw_reference = reference_name if reference_prefix == "raw" else str(reference_series or "")
    ref_df = _compute_selected_returns(
        raw_data,
        periodicity,
        (raw_reference,),
        returns_type,
        benchmark_assignments,
        long_short_assignments,
        date_range,
        vol_scaler,
        vol_scaling_assignments,
    )
    if ref_df.empty or raw_reference not in ref_df.columns:
        return pd.Series(dtype=float)
    return ref_df[raw_reference].replace([np.inf, -np.inf], np.nan).dropna()


def _coerce_factor_quantiles(value, default: int = 5) -> int:
    """Clamp factor quantiles to a practical range."""
    try:
        q = int(value)
    except Exception:
        q = default
    return max(2, min(q, 20))


def _import_selected_workbook_sheets(contents, filename, selected_sheets, workbook_sheets=None):
    return _shared_import_selected_workbook_sheets(
        contents,
        filename,
        selected_sheets,
        workbook_sheets=workbook_sheets,
    )


def build_welcome_screen():
    return build_shared_welcome_screen(AT_WELCOME_MODAL_CONFIG)

clientside_callback(
    ClientsideFunction(namespace="dashmat_callbacks", function_name="saveWorkspaceSession"),
    Output("at-save-session-dummy", "data"),
    Input("at-menu-save-session", "n_clicks"),
    prevent_initial_call=True,
)

clientside_callback(
    ClientsideFunction(namespace="dashmat_callbacks", function_name="loadWorkspaceSessionDialog"),
    Output("at-load-session-dummy", "data"),
    Input("at-load-session-upload", "id"),
    Input("at-menu-load-session", "n_clicks"),
    prevent_initial_call=True,
)

clientside_callback(
    ClientsideFunction(namespace="dashmat_callbacks", function_name="loadWorkspaceSession"),
    Output("at-load-session-dummy", "data", allow_duplicate=True),
    Input("at-load-session-upload", "contents"),
    prevent_initial_call=True,
)


@callback(
    Output("at-menu-save-session", "disabled"),
    Input("dashmat-raw-data-store", "data"),
)
def at_toggle_save_session(raw_data):
    return not bool(raw_data)


@callback(
    Output("at-db-add-modal", "opened", allow_duplicate=True),
    Output("at-db-add-series-select", "data", allow_duplicate=True),
    Output("at-db-add-series-select", "value", allow_duplicate=True),
    Output("at-ui-blocker-store", "data", allow_duplicate=True),
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
    Output("at-ui-blocker-store", "data", allow_duplicate=True),
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
        raw_df = _raw_df(raw_data)
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
    Output("at-ui-blocker-store", "data", allow_duplicate=True),
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


@callback(
    Output("at-underlying-add-modal", "opened", allow_duplicate=True),
    Output("at-underlying-add-modal", "title", allow_duplicate=True),
    Output("at-underlying-add-base-select", "value", allow_duplicate=True),
    Output("at-underlying-add-type-multiselect", "value", allow_duplicate=True),
    Output("at-underlying-add-desc-multiselect", "data", allow_duplicate=True),
    Output("at-underlying-add-desc-multiselect", "value", allow_duplicate=True),
    Output("at-underlying-add-desc-multiselect", "disabled", allow_duplicate=True),
    Output("at-underlying-add-rows-store", "data", allow_duplicate=True),
    Output("at-underlying-add-grid", "rowData", allow_duplicate=True),
    Output("at-underlying-add-error-alert", "hide", allow_duplicate=True),
    Output("at-ui-blocker-store", "data", allow_duplicate=True),
    Input("at-menu-add-portfolios-underlying", "n_clicks"),
    Input("at-welcome-add-portfolios-underlying-btn", "n_clicks"),
    prevent_initial_call=True,
)
def at_open_underlying_add_modal(menu_clicks, welcome_clicks):
    return compute_open_underlying_add_modal(menu_clicks, welcome_clicks)


@callback(
    Output("at-underlying-add-desc-multiselect", "data"),
    Output("at-underlying-add-desc-multiselect", "value"),
    Output("at-underlying-add-desc-multiselect", "disabled"),
    Input("at-underlying-add-base-select", "value"),
    Input("at-underlying-add-type-multiselect", "value"),
    Input("at-underlying-add-modal", "opened"),
    State("at-underlying-add-desc-multiselect", "value"),
    prevent_initial_call=True,
)
def at_sync_underlying_desc_options(base_value, type_values, opened, current_values):
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
    Output("at-underlying-add-rows-store", "data", allow_duplicate=True),
    Output("at-underlying-add-grid", "rowData", allow_duplicate=True),
    Output("at-underlying-add-error-alert", "children", allow_duplicate=True),
    Output("at-underlying-add-error-alert", "hide", allow_duplicate=True),
    Input("at-underlying-add-row-btn", "n_clicks"),
    State("at-underlying-add-rows-store", "data"),
    State("at-underlying-add-base-select", "value"),
    State("at-underlying-add-type-multiselect", "value"),
    State("at-underlying-add-desc-multiselect", "value"),
    prevent_initial_call=True,
)
def at_stage_underlying_rows(n_clicks, staged_rows, base_value, type_values, desc_values):
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


clientside_callback(
    js_underlying_delete_row(),
    Output("at-underlying-add-rows-store", "data", allow_duplicate=True),
    Output("at-underlying-add-grid", "rowData", allow_duplicate=True),
    Output("at-underlying-add-error-alert", "children", allow_duplicate=True),
    Output("at-underlying-add-error-alert", "hide", allow_duplicate=True),
    Input("at-underlying-delete-row-btn", "n_clicks"),
    State("at-underlying-add-rows-store", "data"),
    State("at-underlying-add-grid", "selectedRows"),
    prevent_initial_call=True,
)

clientside_callback(
    js_portfolio_clear_rows(),
    Output("at-underlying-add-rows-store", "data", allow_duplicate=True),
    Output("at-underlying-add-grid", "rowData", allow_duplicate=True),
    Output("at-underlying-add-error-alert", "children", allow_duplicate=True),
    Output("at-underlying-add-error-alert", "hide", allow_duplicate=True),
    Input("at-underlying-clear-rows-btn", "n_clicks"),
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


@callback(
    Output("at-underlying-add-modal", "opened", allow_duplicate=True),
    Output("at-underlying-add-base-select", "value", allow_duplicate=True),
    Output("at-underlying-add-type-multiselect", "value", allow_duplicate=True),
    Output("at-underlying-add-desc-multiselect", "data", allow_duplicate=True),
    Output("at-underlying-add-desc-multiselect", "value", allow_duplicate=True),
    Output("at-underlying-add-desc-multiselect", "disabled", allow_duplicate=True),
    Output("at-underlying-add-rows-store", "data", allow_duplicate=True),
    Output("at-underlying-add-grid", "rowData", allow_duplicate=True),
    Input("at-underlying-add-cancel-button", "n_clicks"),
    prevent_initial_call=True,
)
def at_close_underlying_add_modal(n_clicks):
    return compute_close_underlying_add_modal(n_clicks)


clientside_callback(
    js_portfolio_ok_disabled(),
    Output("at-portfolio-add-ok-button", "disabled"),
    Input("at-portfolio-add-rows-store", "data"),
    Input("at-portfolio-add-modal", "opened"),
)

clientside_callback(
    js_portfolio_ok_disabled(),
    Output("at-underlying-add-ok-button", "disabled"),
    Input("at-underlying-add-rows-store", "data"),
    Input("at-underlying-add-modal", "opened"),
)

clientside_callback(
    ClientsideFunction(namespace="dashmat_callbacks", function_name="uiBlockerEnable"),
    Output("at-ui-blocker-store", "data", allow_duplicate=True),
    Input("at-menu-add-from-db", "n_clicks"),
    Input("at-welcome-add-db-btn", "n_clicks"),
    Input("at-menu-add-raw-factor", "n_clicks"),
    Input("at-menu-add-raw-funds", "n_clicks"),
    Input("at-menu-add-raw-performance", "n_clicks"),
    Input("at-welcome-add-raw-factor-btn", "n_clicks"),
    Input("at-welcome-add-raw-funds-btn", "n_clicks"),
    Input("at-welcome-add-raw-performance-btn", "n_clicks"),
    Input("at-menu-add-portfolios-peer", "n_clicks"),
    Input("at-menu-add-portfolios-index", "n_clicks"),
    Input("at-menu-add-portfolios-other", "n_clicks"),
    Input("at-welcome-add-portfolios-peer-btn", "n_clicks"),
    Input("at-welcome-add-portfolios-index-btn", "n_clicks"),
    Input("at-welcome-add-portfolios-other-btn", "n_clicks"),
    Input("at-menu-add-portfolios-underlying", "n_clicks"),
    Input("at-welcome-add-portfolios-underlying-btn", "n_clicks"),
    Input("at-open-series-modal-button", "n_clicks"),
    prevent_initial_call=True,
)

clientside_callback(
    ClientsideFunction(namespace="dashmat_callbacks", function_name="uiBlockerEnable"),
    Output("at-ui-blocker-store", "data", allow_duplicate=True),
    Input("at-db-add-ok-button", "n_clicks"),
    Input("at-raw-db-add-ok-button", "n_clicks"),
    Input("at-portfolio-add-ok-button", "n_clicks"),
    Input("at-underlying-add-ok-button", "n_clicks"),
    Input("at-modal-ok-button", "n_clicks"),
    prevent_initial_call=True,
)

clientside_callback(
    ClientsideFunction(namespace="dashmat_callbacks", function_name="uiBlockerRelease"),
    Output("at-ui-blocker-store", "data", allow_duplicate=True),
    Input("at-db-add-error-alert", "hide"),
    Input("at-raw-db-add-error-alert", "hide"),
    Input("at-portfolio-add-error-alert", "hide"),
    Input("at-underlying-add-error-alert", "hide"),
    Input("at-series-selection-modal", "opened"),
    prevent_initial_call=True,
)

def _build_at_returns_type_control(control_id, value, show_label=True):
    children = []
    if show_label:
        children.append(dmc.Text("Returns Type", size="sm", mb=3, fw=500))
    children.append(
        dmc.SegmentedControl(
            id=control_id,
            data=[
                {"value": "total", "label": "Total"},
                {"value": "excess", "label": "Excess"},
            ],
            value=value,
            size="sm",
        )
    )
    return html.Div(children)


def build_main_layout(periodicity_options, periodicity_value, returns_type, vol_scaler,
                      active_tab, rolling_window, rolling_metric, rolling_return_type, rolling_chart_switch,
                      drawdown_chart_switch, growth_chart_switch, monthly_view, monthly_series,
                      monthly_series_options, monthly_select_disabled, factor_mode,
                      factor_quantiles, factor_transform, factor_series_options,
                      factor_series_value, factor_qq_reference, conditional_view,
                      conditional_comparator, conditional_threshold, conditional_window_conversion,
                      conditional_step, conditional_step_unit, conditional_display_mode,
                      regime_display_mode):
    
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
    factor_series_options = factor_series_options or []

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
                                        html.Div(
                                            style={"display": "none"},
                                            children=[
                                                dmc.SegmentedControl(
                                                    id="at-returns-type-select",
                                                    data=[
                                                        {"value": "total", "label": "Total"},
                                                        {"value": "excess", "label": "Excess"},
                                                    ],
                                                    value=returns_type,
                                                    w=250,
                                                ),
                                            ],
                                        ),
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
                                        html.Div([
                                            dmc.Text("Sharpe/Sortino RF", size="sm", mb=3, fw=500),
                                            html.Div(
                                                dmc.SegmentedControl(
                                                    id="at-use-risk-free-switch",
                                                    data=[
                                                        {"value": "zero", "label": "Zero"},
                                                        {"value": "tbill", "label": "T-Bill"},
                                                    ],
                                                    value="tbill",
                                                    size="sm",
                                                ),
                                                style={"height": "36px", "display": "flex", "alignItems": "center"},
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
                        dmc.TabsTab("Factor Analysis", value="factor_analysis"),
                        dmc.TabsTab("Conditional Returns", value="conditional_returns"),
                        dmc.TabsTab("Regime Analysis", value="regime_analysis"),
                    ],
                ),
                dmc.TabsPanel(
                    value="returns",
                    pt="md",
                    style={"flex": "1", "display": "flex", "flexDirection": "column", "overflow": "hidden"},
                    children=[
                        dmc.Group(
                            mb="md",
                            children=[_build_at_returns_type_control("at-returns-type-select-returns", returns_type, show_label=False)],
                        ),
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
                                    dashGridOptions=literal_field_dash_grid_options({
                                        "animateRows": True,
                                        "pagination": False,
                                        "suppressExcelExport": True,
                                        "enableRangeSelection": True,
                                    }),
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
                                    dashGridOptions=literal_field_dash_grid_options({
                                        "animateRows": True,
                                        "pagination": False,
                                        "suppressExcelExport": True,
                                        "enableRangeSelection": True,
                                        "suppressCsvExport": True,
                                    }),
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
                                    dashGridOptions=literal_field_dash_grid_options({
                                        "animateRows": True,
                                        "suppressExcelExport": True,
                                        "enableRangeSelection": True,
                                        "suppressCsvExport": True,
                                    }),
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
                                _build_at_returns_type_control("at-returns-type-select-calendar", returns_type, show_label=False),
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
                            dashGridOptions=literal_field_dash_grid_options({
                                "animateRows": True,
                                "suppressExcelExport": True,
                                "enableRangeSelection": True,
                                "suppressCsvExport": True,
                            }),
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
                            gap="md",
                            align="flex-end",
                            children=[
                                _build_at_returns_type_control("at-returns-type-select-correlogram", returns_type),
                                html.Div([
                                    dmc.Text("View", size="sm", fw=500, mb=3),
                                    html.Div(
                                        dmc.SegmentedControl(
                                            id="at-correlation-view-switch",
                                            data=[
                                                {"value": "correlation", "label": "Correlation"},
                                                {"value": "covariance", "label": "Covariance"},
                                                {"value": "correlogram", "label": "Correlogram"},
                                            ],
                                            value="correlogram",
                                            size="sm",
                                        ),
                                        style={"height": "36px", "display": "flex", "alignItems": "center"},
                                    ),
                                ]),
                                html.Div([
                                    dmc.Text("Exp Wt", size="sm", fw=500, mb=3),
                                    html.Div(
                                        dmc.Switch(
                                            id="at-correlation-exp-wt-switch",
                                            checked=False,
                                            size="sm",
                                        ),
                                        style={"height": "36px", "display": "flex", "alignItems": "center"},
                                    ),
                                ]),
                                html.Div([
                                    dmc.Text("Half-Life", size="sm", fw=500, mb=3),
                                    html.Div(
                                        dmc.Tooltip(
                                            label="If value is < 1, it is interpreted as lambda. If value is >= 1, it is interpreted as half-life in periods.",
                                            multiline=True,
                                            w=300,
                                            withArrow=True,
                                            children=dmc.NumberInput(
                                                id="at-correlation-halflife-input",
                                                label=None,
                                                value=63,
                                                min=0.001,
                                                step=0.01,
                                                w=100,
                                                size="sm",
                                                disabled=True,
                                            ),
                                        ),
                                        style={"height": "36px", "display": "flex", "alignItems": "center"},
                                    ),
                                ]),
                                html.Div([
                                    dmc.Text("Cov Shrinkage", size="sm", fw=500, mb=3),
                                    html.Div(
                                        dmc.Select(
                                            id="at-correlation-shrinkage-select",
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
                                        style={"height": "36px", "display": "flex", "alignItems": "center"},
                                    ),
                                ]),
                                html.Div([
                                    dmc.Text("Target", size="sm", fw=500, mb=3),
                                    html.Div(
                                        dmc.Select(
                                            id="at-correlation-shrinkage-target-select",
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
                                        style={"height": "36px", "display": "flex", "alignItems": "center"},
                                    ),
                                ]),
                                html.Div([
                                    dmc.Text("Block Size", size="sm", fw=500, mb=3),
                                    dmc.NumberInput(
                                        id="at-correlogram-block-width",
                                        label=None,
                                        value=None,
                                        min=50,
                                        step=50,
                                        suffix="px",
                                        w=110,
                                        size="sm",
                                    ),
                                ]),
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
                    value="factor_analysis",
                    pt="md",
                    style={"flex": "1", "display": "flex", "flexDirection": "column", "overflow": "hidden"},
                    children=[
                        dmc.Group(
                            mb="md",
                            gap="md",
                            align="flex-end",
                            children=[
                                _build_at_returns_type_control("at-returns-type-select-factor", returns_type),
                                html.Div([
                                    dmc.Text("Mode", size="sm", fw=500, mb=3),
                                    dmc.SegmentedControl(
                                        id="at-factor-mode-select",
                                        data=[
                                            {"value": "box", "label": "Box Plot"},
                                            {"value": "scatter", "label": "Scatter"},
                                            {"value": "detail", "label": "Raw Detail"},
                                            {"value": "qq", "label": "Q-Q Plot"},
                                        ],
                                        value=factor_mode,
                                        size="sm",
                                    ),
                                ]),
                                html.Div(
                                    id="at-factor-qq-reference-wrapper",
                                    style={"display": "none"},
                                    children=[
                                        dmc.Text("Q-Q Reference", size="sm", fw=500, mb=3),
                                        dmc.SegmentedControl(
                                            id="at-factor-qq-reference-select",
                                            data=[
                                                {"value": "normal", "label": "Normal"},
                                                {"value": "reference", "label": "Reference"},
                                            ],
                                            value=factor_qq_reference,
                                            size="sm",
                                        ),
                                    ],
                                ),
                                html.Div(
                                    id="at-factor-series-wrapper",
                                    children=[
                                        dmc.Select(
                                            id="at-factor-series-select",
                                            label="Factor",
                                            data=factor_series_options,
                                            value=factor_series_value,
                                            w=280,
                                            size="sm",
                                            searchable=True,
                                            clearable=False,
                                            placeholder="Select factor series",
                                        ),
                                    ],
                                ),
                                html.Div(
                                    id="at-factor-open-modal-wrapper",
                                    children=[
                                        dmc.Text("Definitions", size="sm", fw=500, mb=3),
                                        dmc.Button(
                                            "Edit factors",
                                            id="at-factor-open-modal-btn",
                                            size="sm",
                                            variant="light",
                                            leftSection=DashIconify(icon="tabler:math-function", width=14),
                                        ),
                                    ],
                                ),
                                html.Div(
                                    id="at-factor-quantiles-wrapper",
                                    style={"display": "block"},
                                    children=[
                                        dmc.NumberInput(
                                            id="at-factor-quantiles-input",
                                            label="Quantiles",
                                            value=factor_quantiles,
                                            min=2,
                                            max=20,
                                            step=1,
                                            w=120,
                                            size="sm",
                                        ),
                                    ],
                                ),
                                html.Div(
                                    id="at-factor-transform-wrapper",
                                    children=[
                                        dmc.Text("Factor Transform", size="sm", fw=500, mb=3),
                                        dmc.SegmentedControl(
                                            id="at-factor-transform-select",
                                            data=[
                                                {"value": "raw", "label": "Raw"},
                                                {"value": "zscore", "label": "Z-Score"},
                                            ],
                                            value=factor_transform,
                                            size="sm",
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        html.Div(id="at-factor-analysis-warning"),
                        html.Div(
                            id="at-factor-analysis-container",
                            style={"flex": "1", "overflow": "auto"},
                        ),
                    ],
                ),
                dmc.TabsPanel(
                    value="conditional_returns",
                    pt="md",
                    style={"flex": "1", "display": "flex", "flexDirection": "column", "overflow": "hidden"},
                    children=[
                        dmc.Group(
                            mb="md",
                            gap="md",
                            align="flex-end",
                            children=[
                                _build_at_returns_type_control("at-returns-type-select-conditional", returns_type),
                                html.Div(
                                    children=[
                                        dmc.Select(
                                            id="at-factor-series-select-conditional",
                                            label="Factor",
                                            data=factor_series_options,
                                            value=factor_series_value,
                                            w=280,
                                            size="sm",
                                            searchable=True,
                                            clearable=False,
                                            placeholder="Select factor series",
                                        ),
                                    ],
                                ),
                                html.Div(
                                    children=[
                                        dmc.Text("Definitions", size="sm", fw=500, mb=3),
                                        dmc.Button(
                                            "Edit factors",
                                            id="at-factor-open-modal-btn-conditional",
                                            size="sm",
                                            variant="light",
                                            leftSection=DashIconify(icon="tabler:math-function", width=14),
                                        ),
                                    ],
                                ),
                                html.Div(
                                    children=[
                                        dmc.Text("Factor Transform", size="sm", fw=500, mb=3),
                                        dmc.SegmentedControl(
                                            id="at-factor-transform-select-conditional",
                                            data=[
                                                {"value": "raw", "label": "Raw"},
                                                {"value": "zscore", "label": "Z-Score"},
                                            ],
                                            value=factor_transform,
                                            size="sm",
                                        ),
                                    ],
                                ),
                                html.Div(
                                    children=[
                                        dmc.Text("Display", size="sm", fw=500, mb=3),
                                        dmc.SegmentedControl(
                                            id="at-conditional-display-mode-select",
                                            data=CONDITIONAL_DISPLAY_MODE_OPTIONS,
                                            value=conditional_display_mode,
                                            size="sm",
                                        ),
                                    ],
                                ),
                                html.Div(
                                    children=[
                                        dmc.Text("View", size="sm", fw=500, mb=3),
                                        dmc.SegmentedControl(
                                            id="at-conditional-view-select",
                                            data=CONDITIONAL_VIEW_OPTIONS,
                                            value=conditional_view,
                                            size="sm",
                                        ),
                                    ],
                                ),
                                html.Div(
                                    children=[
                                        dmc.Text("Comparator", size="sm", fw=500, mb=3),
                                        dmc.SegmentedControl(
                                            id="at-conditional-comparator-select",
                                            data=CONDITIONAL_COMPARATOR_OPTIONS,
                                            value=conditional_comparator,
                                            size="sm",
                                        ),
                                    ],
                                ),
                                dmc.NumberInput(
                                    id="at-conditional-threshold-input",
                                    label="Threshold",
                                    value=conditional_threshold,
                                    step=0.1,
                                    w=120,
                                    size="sm",
                                ),
                                html.Div(
                                    children=[
                                        dmc.Group(
                                            gap=4,
                                            align="center",
                                            mb=3,
                                            children=[
                                                dmc.Text("Factor Window", size="sm", fw=500),
                                                dmc.Tooltip(
                                                    id="at-conditional-window-conversion-tooltip",
                                                    label=_conditional_conversion_tooltip_text(),
                                                    position="top",
                                                    withArrow=True,
                                                    children=html.Span(
                                                        DashIconify(
                                                            icon="tabler:info-circle",
                                                            width=14,
                                                            color="#868e96",
                                                        ),
                                                        id="at-conditional-window-conversion-tooltip-target",
                                                        style={
                                                            "display": "inline-flex",
                                                            "alignItems": "center",
                                                            "cursor": "help",
                                                        },
                                                    ),
                                                ),
                                            ],
                                        ),
                                        dmc.Select(
                                            id="at-conditional-window-conversion-select",
                                            data=CONDITIONAL_FACTOR_CONVERSION_OPTIONS,
                                            value=conditional_window_conversion,
                                            w=170,
                                            size="sm",
                                            clearable=False,
                                        ),
                                    ],
                                ),
                                html.Div(
                                    children=[
                                        dmc.Text("Step", size="sm", fw=500, mb=3),
                                        dmc.Group(
                                            gap="xs",
                                            wrap="nowrap",
                                            children=[
                                                dmc.NumberInput(
                                                    id="at-conditional-step-input",
                                                    value=conditional_step,
                                                    min=1,
                                                    step=1,
                                                    w=90,
                                                    size="sm",
                                                ),
                                                dmc.Select(
                                                    id="at-conditional-step-unit-select",
                                                    data=[
                                                        {"value": "months", "label": "Months"},
                                                        {"value": "periods", "label": "Periods"},
                                                    ],
                                                    value=conditional_step_unit,
                                                    w=100,
                                                    size="sm",
                                                    clearable=False,
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        html.Div(id="at-conditional-returns-warning"),
                        dcc.Loading(
                            id="at-loading-conditional-returns",
                            type="default",
                            delay_show=300,
                            delay_hide=150,
                            style={"flex": "1", "display": "flex", "flexDirection": "column", "overflow": "auto"},
                            parent_style={"flex": "1", "display": "flex", "flexDirection": "column", "overflow": "auto"},
                            children=[
                                html.Div(
                                    id="at-conditional-returns-container",
                                    style={"flex": "1", "overflow": "auto"},
                                ),
                            ],
                        ),
                    ],
                ),
                dmc.TabsPanel(
                    value="regime_analysis",
                    pt="md",
                    style={"flex": "1", "display": "flex", "flexDirection": "column", "overflow": "hidden"},
                    children=[
                        dmc.Group(
                            mb="md",
                            gap="md",
                            align="flex-end",
                            children=[
                                _build_at_returns_type_control("at-returns-type-select-regime", returns_type),
                                dmc.Select(
                                    id="at-regime-definition-select",
                                    label="Regime definition",
                                    data=[],
                                    value=None,
                                    w=340,
                                    size="sm",
                                    searchable=True,
                                    clearable=True,
                                    placeholder="Select regime definition",
                                    nothingFoundMessage="No saved regimes",
                                ),
                                html.Div([
                                    dmc.Text("Definitions", size="sm", fw=500, mb=3),
                                    dmc.Button(
                                        "Edit regimes",
                                        id="at-regime-open-modal-btn",
                                        size="sm",
                                        variant="light",
                                        leftSection=DashIconify(icon="tabler:binary-tree-2", width=14),
                                    ),
                                ]),
                                html.Div([
                                    dmc.Text("View", size="sm", fw=500, mb=3),
                                    dmc.SegmentedControl(
                                        id="at-regime-detail-display-mode-select",
                                        data=REGIME_DETAIL_DISPLAY_MODE_OPTIONS,
                                        value=regime_display_mode,
                                        size="sm",
                                    ),
                                ]),
                            ],
                        ),
                        html.Div(id="at-regime-analysis-warning"),
                        html.Div(
                            id="at-regime-analysis-container",
                            style={"flex": "1", "overflow": "auto"},
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
                                    dashGridOptions=literal_field_dash_grid_options({
                                        "animateRows": True,
                                        "pagination": False,
                                        "suppressExcelExport": True,
                                        "enableRangeSelection": True,
                                        "suppressCsvExport": True,
                                    }),
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
                                _build_at_returns_type_control("at-returns-type-select-drawdown", returns_type, show_label=False),
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
                                    dashGridOptions=literal_field_dash_grid_options({
                                        "animateRows": True,
                                        "pagination": False,
                                        "suppressExcelExport": True,
                                        "enableRangeSelection": True,
                                        "suppressCsvExport": True,
                                    }),
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
                                            "Load account list",
                                            id="at-menu-load-account-list",
                                            leftSection=DashIconify(icon="tabler:list-details", width=14),
                                        ),
                                        dmc.MenuItem(
                                            "Save account list",
                                            id="at-menu-save-account-list",
                                            leftSection=DashIconify(icon="tabler:bookmark-plus", width=14),
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
                        # Add Menu (left)
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
                                        dmc.MenuItem(
                                            "Add underlying categories...",
                                            id="at-menu-add-portfolios-underlying",
                                            leftSection=DashIconify(icon="tabler:hierarchy-2", width=14),
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
                                            "Edit factors...",
                                            id="at-menu-add-factor",
                                            leftSection=DashIconify(icon="tabler:math-function", width=14),
                                        ),
                                        dmc.MenuItem(
                                            "Edit regimes...",
                                            id="at-menu-add-regime",
                                            leftSection=DashIconify(icon="tabler:binary-tree-2", width=14),
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
                        dmc.Button(
                            "Switch to Regression",
                            id="at-menu-view-regression",
                            size="sm",
                            radius="md",
                            variant="gradient",
                            gradient={"from": "grape", "to": "indigo", "deg": 90},
                            leftSection=DashIconify(icon="tabler:chart-dots-3", width=16),
                        ),
                        # Spacer
                        dmc.Box(style={"flexGrow": 1}),
                        _build_help_control(),
                    ],
                ),
            ],
        ),
        html.Div(
            id="dashmat-account-list-notice-container",
            style={"marginTop": "-8px", "marginBottom": "12px"},
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
        build_underlying_add_modal("at", AG_GRID_LICENSE_KEY),
        build_raw_db_add_modal("at", AG_GRID_LICENSE_KEY),
        build_sheet_select_modal("at"),
        dmc.Modal(
            id="at-factor-def-modal",
            title=dmc.Group(
                gap="xs",
                children=[
                    dmc.ThemeIcon(DashIconify(icon="tabler:math-function"), color="violet", variant="light", size="sm"),
                    dmc.Text("Edit factors", fw=600, size="sm"),
                ],
            ),
            size="980px",
            centered=True,
            closeOnClickOutside=False,
            withCloseButton=True,
            radius="lg",
            className="dashmat-modal",
            overlayProps={"blur": 2, "opacity": 0.45},
            transitionProps={"transition": "fade", "duration": 180},
            children=[
                dmc.Alert(
                    id="at-factor-def-status-alert",
                    title="Factor definitions",
                    color="blue",
                    hide=True,
                    mb="sm",
                ),
                dmc.Text(
                    id="at-factor-def-db-available-note",
                    size="xs",
                    c="dimmed",
                    mb="xs",
                ),
                dmc.Stack(
                    gap="sm",
                    children=[
                        dmc.Select(
                            id="at-factor-def-select",
                            label="Database/Session factors",
                            data=[],
                            value=None,
                            clearable=True,
                            searchable=True,
                            placeholder="Select existing factor definition",
                            nothingFoundMessage="No saved factors",
                        ),
                        dmc.TextInput(
                            id="at-factor-def-name-input",
                            label="Factor name",
                            placeholder="Example: Quality Spread",
                        ),
                        dmc.Textarea(
                            id="at-factor-def-description-input",
                            label="Description",
                            minRows=2,
                            maxRows=4,
                        ),
                        dmc.Group(
                            grow=True,
                            children=[
                                dmc.MultiSelect(
                                    id="at-factor-def-long-components",
                                    label="Long components",
                                    data=[],
                                    value=[],
                                    searchable=True,
                                    clearable=True,
                                    nothingFoundMessage="No SEC_FACTOR components",
                                ),
                                dmc.MultiSelect(
                                    id="at-factor-def-short-components",
                                    label="Short components",
                                    data=[],
                                    value=[],
                                    searchable=True,
                                    clearable=True,
                                    nothingFoundMessage="No SEC_FACTOR components",
                                ),
                            ],
                        ),
                        dmc.Group(
                            grow=True,
                            children=[
                                dmc.Select(
                                    id="at-factor-def-long-agg-type",
                                    label="Long aggregation",
                                    data=FACTOR_AGG_TYPE_OPTIONS,
                                    value="1",
                                    clearable=False,
                                ),
                                dmc.Select(
                                    id="at-factor-def-short-agg-type",
                                    label="Short aggregation",
                                    data=FACTOR_AGG_TYPE_OPTIONS,
                                    value=None,
                                    clearable=True,
                                ),
                                dmc.NumberInput(
                                    id="at-factor-def-long-lag",
                                    label="Long lag (periods)",
                                    value=0,
                                    min=0,
                                    step=1,
                                ),
                                dmc.Select(
                                    id="at-factor-def-output-transform",
                                    label="Output transform",
                                    data=OUTPUT_TRANSFORM_OPTIONS,
                                    value="0",
                                    clearable=False,
                                ),
                            ],
                        ),
                        dmc.Stack(
                            gap=4,
                            children=[
                                dmc.Text("Preview (first 6 rows)", fw=500, size="sm"),
                                dmc.ScrollArea(
                                    h=140,
                                    offsetScrollbars=True,
                                    children=dmc.Code(
                                        id="at-factor-def-preview-lines",
                                        block=True,
                                        children="Define a factor to preview values.",
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
                            justify="space-between",
                            children=[
                                dmc.Group(
                                    gap="xs",
                                    children=[
                                        dmc.Button("Save to session", id="at-factor-def-save-local-btn", variant="light"),
                                        dmc.Button("Save to database", id="at-factor-def-save-db-btn", variant="outline"),
                                        dmc.Button("Delete", id="at-factor-def-delete-btn", variant="outline", color="red"),
                                    ],
                                ),
                                dmc.Group(
                                    gap="xs",
                                    children=[
                                        dmc.Button("New factor", id="at-factor-def-new-btn", variant="outline"),
                                        dmc.Button("Use factor", id="at-factor-def-use-btn", color="blue"),
                                        dmc.Button("Close", id="at-factor-def-close-btn", variant="outline", color="gray"),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        dmc.Modal(
            id="at-regime-def-modal",
            title=dmc.Group(
                gap="xs",
                children=[
                    dmc.ThemeIcon(DashIconify(icon="tabler:binary-tree-2"), color="teal", variant="light", size="sm"),
                    dmc.Text("Edit regimes", fw=600, size="sm"),
                ],
            ),
            size="980px",
            centered=True,
            closeOnClickOutside=False,
            withCloseButton=True,
            radius="lg",
            className="dashmat-modal",
            overlayProps={"blur": 2, "opacity": 0.45},
            transitionProps={"transition": "fade", "duration": 180},
            children=[
                dmc.Alert(
                    id="at-regime-def-status-alert",
                    title="Regime definitions",
                    color="blue",
                    hide=True,
                    mb="sm",
                ),
                dmc.Text(
                    id="at-regime-def-db-available-note",
                    size="xs",
                    c="dimmed",
                    mb="xs",
                ),
                dmc.Stack(
                    gap="sm",
                    children=[
                        dmc.Select(
                            id="at-regime-def-select",
                            label="Database/Session regimes",
                            data=[],
                            value=None,
                            clearable=True,
                            searchable=True,
                            placeholder="Select existing regime definition",
                            nothingFoundMessage="No saved regimes",
                        ),
                        dmc.TextInput(
                            id="at-regime-def-name-input",
                            label="Regime name",
                            placeholder="Example: Equity Risk Cycle",
                        ),
                        dmc.Textarea(
                            id="at-regime-def-description-input",
                            label="Description",
                            minRows=2,
                            maxRows=4,
                        ),
                        dmc.Group(
                            grow=True,
                            children=[
                                dmc.Select(
                                    id="at-regime-def-method-type",
                                    label="Method",
                                    data=REGIME_METHOD_OPTIONS,
                                    value="1",
                                    clearable=False,
                                ),
                                dmc.NumberInput(
                                    id="at-regime-def-num-regimes",
                                    label="Regimes",
                                    value=3,
                                    min=2,
                                    max=10,
                                    step=1,
                                ),
                                dmc.NumberInput(
                                    id="at-regime-def-min-observations",
                                    label="Min observations",
                                    value=60,
                                    min=20,
                                    step=5,
                                ),
                            ],
                        ),
                        dmc.Group(
                            grow=True,
                            children=[
                                dmc.Switch(
                                    id="at-regime-def-pca-standardize",
                                    label="PC1 standardize",
                                    checked=True,
                                ),
                                dmc.NumberInput(
                                    id="at-regime-def-vol-scaler",
                                    label="Vol scaler (%)",
                                    value=0,
                                    min=0,
                                    step=1,
                                ),
                            ],
                        ),
                        html.Div(
                            id="at-regime-def-universe-wrapper",
                            children=[
                                dmc.MultiSelect(
                                    id="at-regime-def-universe-series",
                                    label="Universe series (for PC1 methods)",
                                    data=[],
                                    value=[],
                                    searchable=True,
                                    clearable=True,
                                    nothingFoundMessage="No series available",
                                )
                            ],
                        ),
                        html.Div(
                            id="at-regime-def-single-wrapper",
                            style={"display": "none"},
                            children=[
                                dmc.Select(
                                    id="at-regime-def-single-series",
                                    label="Single series (for single-series quantiles)",
                                    data=[],
                                    value=None,
                                    searchable=True,
                                    clearable=True,
                                    nothingFoundMessage="No series available",
                                )
                            ],
                        ),
                        dmc.Stack(
                            gap=4,
                            children=[
                                dmc.Text("Preview (first 8 rows)", fw=500, size="sm"),
                                dmc.ScrollArea(
                                    h=160,
                                    offsetScrollbars=True,
                                    children=dmc.Code(
                                        id="at-regime-def-preview-lines",
                                        block=True,
                                        children="Define a regime to preview assignments.",
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
                            justify="space-between",
                            children=[
                                dmc.Group(
                                    gap="xs",
                                    children=[
                                        dmc.Button("Save to session", id="at-regime-def-save-local-btn", variant="light"),
                                        dmc.Button("Save to database", id="at-regime-def-save-db-btn", variant="outline"),
                                        dmc.Button("Delete", id="at-regime-def-delete-btn", variant="outline", color="red"),
                                    ],
                                ),
                                dmc.Group(
                                    gap="xs",
                                    children=[
                                        dmc.Button("New regime", id="at-regime-def-new-btn", variant="outline"),
                                        dmc.Button("Use regime", id="at-regime-def-use-btn", color="blue"),
                                        dmc.Button("Close", id="at-regime-def-close-btn", variant="outline", color="gray"),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),


        # Welcome Screen (Hydration gates visibility)
        html.Div(
            id="at-welcome-screen-container",
            children=build_welcome_screen(),
            style={"display": "none"}
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
                monthly_select_disabled=True,
                factor_mode="box",
                factor_quantiles=5,
                factor_transform="raw",
                factor_series_options=[],
                factor_series_value=None,
                factor_qq_reference="normal",
                conditional_view="forward",
                conditional_comparator="le",
                conditional_threshold=0,
                conditional_window_conversion="compound",
                conditional_step=1,
                conditional_step_unit="months",
                conditional_display_mode="summary",
                regime_display_mode="summary",
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
        dcc.Store(id="at-use-risk-free-store", data=True, storage_type="session"),
        dcc.Store(id="at-monthly-view-store", data="annual", storage_type="session"),
        dcc.Store(id="at-monthly-series-store", data=None, storage_type="session"),
        dcc.Store(id="at-factor-mode-store", data="box", storage_type="session"),
        dcc.Store(id="at-factor-quantiles-store", data=5, storage_type="session"),
        dcc.Store(id="at-factor-transform-store", data="raw", storage_type="session"),
        dcc.Store(id="at-factor-series-store", data=None, storage_type="session"),
        dcc.Store(id="at-factor-qq-reference-store", data="normal", storage_type="session"),
        dcc.Store(id="at-conditional-view-store", data="forward", storage_type="session"),
        dcc.Store(id="at-conditional-comparator-store", data="le", storage_type="session"),
        dcc.Store(id="at-conditional-threshold-store", data=0, storage_type="session"),
        dcc.Store(id="at-conditional-window-conversion-store", data="compound", storage_type="session"),
        dcc.Store(id="at-conditional-step-store", data=1, storage_type="session"),
        dcc.Store(id="at-conditional-step-unit-store", data="months", storage_type="session"),
        dcc.Store(id="at-conditional-display-mode-store", data="summary", storage_type="session"),
        dcc.Store(id="at-factor-definitions-db-store", data=[], storage_type="session"),
        dcc.Store(id="at-factor-definitions-local-store", data=[], storage_type="session"),
        dcc.Store(id="at-factor-def-modal-draft-store", data=None, storage_type="session"),
        dcc.Store(id="at-factor-def-db-available-store", data=False, storage_type="session"),
        dcc.Store(id="at-factor-def-loaded-store", data=False, storage_type="session"),
        dcc.Store(id="at-regime-definition-store", data=None, storage_type="session"),
        dcc.Store(id="at-regime-detail-display-mode-store", data="summary", storage_type="session"),
        dcc.Store(id="at-regime-definitions-db-store", data=[], storage_type="session"),
        dcc.Store(id="at-regime-definitions-local-store", data=[], storage_type="session"),
        dcc.Store(id="at-regime-def-modal-draft-store", data=None, storage_type="session"),
        dcc.Store(id="at-regime-def-db-available-store", data=False, storage_type="session"),
        dcc.Store(id="at-regime-def-loaded-store", data=False, storage_type="session"),
        dcc.Store(id="at-regime-series-store", data={"series_data": {}}, storage_type="session"),
        dcc.Store(id="at-date-range-store", data=None, storage_type="session"),
        dcc.Store(id="at-range-candidates-store", data=None, storage_type="memory"),
        dcc.Store(id="at-common-daily-candidates-store", data=None, storage_type="memory"),
        dcc.Store(id="at-state-ready-store", data=False, storage_type="session"),
        dcc.Store(id="at-statistics-loaded-store", data=False, storage_type="session"),
        dcc.Store(id="at-initial-tab-render-ready-store", data=False, storage_type="memory"),
        dcc.Store(id="at-secondary-restore-ready-store", data=False, storage_type="memory"),
        dcc.Store(id="at-vol-scaler-value-store", data=0, storage_type="session"),
        dcc.Store(id="at-vol-scaling-assignments-store", data={}, storage_type="session"),
        dcc.Store(id="at-download-enabled-store", data=False),
        dcc.Store(id="at-first-load-store", data=False, storage_type="session"),
        dcc.Store(id="at-page-visited-store", data=False, storage_type="session"),
        # Temporary stores for modal state
        dcc.Store(id="at-temp-series-select", data=[]),
        dcc.Store(id="at-temp-benchmark-assignments-store", data={}),
        dcc.Store(id="at-temp-long-short-store", data={}),
        dcc.Store(id="at-temp-vol-scaling-assignments-store", data={}),
        dcc.Store(id="at-temp-series-order-store", data=[]),
        dcc.Store(id="at-temp-deleted-series-store", data=[]),
        dcc.Store(id="at-series-grid-snapshot-store", data=None),
        dcc.Store(id="at-portfolio-add-mode-store", data=None),
        dcc.Store(id="at-portfolio-add-rows-store", data=[]),
        dcc.Store(id="at-underlying-add-rows-store", data=[]),
        dcc.Store(id="at-raw-db-add-mode-store", data=None),
        dcc.Store(id="at-raw-db-add-rows-store", data=[]),
        # Temp stores for sheet selection (stash upload while user picks a tab)
        dcc.Store(id="at-sheet-select-contents-store", data=None),
        dcc.Store(id="at-sheet-select-filename-store", data=None),
        dcc.Store(id="at-sheet-select-sheetnames-store", data=None),
        dcc.Download(id="at-download-excel"),
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

        # Correlogram metadata for client-side sizing
        dcc.Store(id="at-correlogram-meta-store", data={}),
        dcc.Store(id="at-correlogram-target-key-store", data=None),
        dcc.Store(id="at-correlogram-rendered-key-store", data=None),

        # UI Blocker for file dialog (Overlay)
        dcc.Store(id="at-ui-blocker-store", data=True),
        dmc.LoadingOverlay(
            id="at-ui-blocker-overlay",
            visible=True,
            zIndex=2500,
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
    function(data, n_intervals) {
        if (data) {
            return [{display: "none"}, {display: "flex", flexDirection: "column", flex: "1", overflow: "hidden"}];
        }
        return [{display: "block"}, {display: "none"}];
    }
    """,
    Output("at-welcome-screen-container", "style"),
    Output("at-main-app-container", "style"),
    Input("dashmat-raw-data-store", "data"),
    Input("at-page-load-trigger", "n_intervals"),
)

clientside_callback(
    """
    function(n_intervals) {
        return !!(n_intervals && n_intervals >= 1);
    }
    """,
    Output("at-initial-tab-render-ready-store", "data"),
    Input("at-page-load-trigger", "n_intervals"),
)

clientside_callback(
    """
    function(stateReady) {
        return !!stateReady;
    }
    """,
    Output("at-secondary-restore-ready-store", "data"),
    Input("at-state-ready-store", "data"),
)


def _at_restore_defaults():
    return {
        "periodicity_options": [{"value": "daily_trading", "label": "Daily (Trading)"}],
        "valid_periodicity": "daily_trading",
        "valid_returns": "total",
        "valid_vol": 0,
        "active_tab": "statistics",
        "roll_win": "1y",
        "roll_metric": "total_return",
        "roll_type": "annualized",
        "roll_type_disabled": False,
        "roll_type_style": {},
        "roll_chart": "chart",
        "dd_chart": "chart",
        "gr_chart": "chart",
        "factor_mode": "box",
        "factor_quantiles": 5,
        "factor_transform": "raw",
        "factor_qq_reference": "normal",
        "conditional_view": "forward",
        "conditional_comparator": "le",
        "conditional_threshold": 0,
        "conditional_window_conversion": "compound",
        "conditional_step": 1,
        "conditional_step_unit": "months",
        "conditional_display_mode": "summary",
        "regime_display_mode": "summary",
        "monthly_view": "annual",
        "valid_selection": [],
        "updated_order": [],
    }


def _at_resolve_restore_state(
    raw_meta,
    stored_periodicity,
    stored_series,
    stored_returns,
    stored_vol,
    stored_tab,
    stored_roll_win,
    stored_roll_metric,
    stored_roll_type,
    stored_roll_chart,
    stored_dd_chart,
    stored_gr_chart,
    stored_factor_mode,
    stored_factor_quantiles,
    stored_factor_transform,
    stored_factor_qq_reference,
    stored_conditional_view,
    stored_conditional_comparator,
    stored_conditional_threshold,
    stored_conditional_window_conversion,
    stored_conditional_step,
    stored_conditional_step_unit,
    stored_conditional_display_mode,
    stored_regime_display_mode,
    stored_monthly_view,
    stored_order,
    po_origin_series,
    page_visited,
):
    if not isinstance(raw_meta, dict) or not raw_meta.get("has_data"):
        return _at_restore_defaults()

    resolved = _at_restore_defaults()
    periodicity_options = raw_meta.get("periodicity_options") or resolved["periodicity_options"]
    valid_values = [p["value"] for p in periodicity_options]
    orig_periodicity = raw_meta.get("original_periodicity") or "daily"
    default_periodicity = "daily_trading" if orig_periodicity == "daily" else (orig_periodicity or "daily_trading")
    if default_periodicity not in valid_values:
        default_periodicity = valid_values[0] if valid_values else "daily_trading"
    valid_periodicity = stored_periodicity if stored_periodicity in valid_values else default_periodicity

    active_tab = stored_tab if stored_tab else "statistics"
    roll_metric = stored_roll_metric if stored_roll_metric else "total_return"

    columns, valid_selection, _generic_new, po_new = _at_get_series_page_state(
        raw_meta,
        stored_series,
        stored_order,
        po_origin_series,
    )
    updated_order = [series for series in (stored_order or []) if series in columns]
    for series in valid_selection:
        if series not in updated_order:
            updated_order.append(series)
    if not (not page_visited and not valid_selection):
        for series in po_new:
            if series not in updated_order:
                updated_order.append(series)
        selected_set = set(valid_selection)
        selected_set.update(po_new)
        valid_selection = [series for series in updated_order if series in selected_set]

    resolved.update(
        {
            "periodicity_options": periodicity_options,
            "valid_periodicity": valid_periodicity,
            "valid_returns": stored_returns if stored_returns in ["total", "excess"] else "total",
            "valid_vol": stored_vol if stored_vol is not None else 0,
            "active_tab": active_tab,
            "roll_win": stored_roll_win if stored_roll_win else "1y",
            "roll_metric": roll_metric,
            "roll_type": stored_roll_type if stored_roll_type else "annualized",
            "roll_type_disabled": roll_metric not in ["total_return", "excess_return"],
            "roll_type_style": {} if roll_metric in ["total_return", "excess_return"] else {"opacity": 0.5, "pointerEvents": "none"},
            "roll_chart": stored_roll_chart if stored_roll_chart is not None else "chart",
            "dd_chart": stored_dd_chart if stored_dd_chart is not None else "chart",
            "gr_chart": stored_gr_chart if stored_gr_chart is not None else "chart",
            "factor_mode": stored_factor_mode if stored_factor_mode in {"box", "scatter", "detail", "qq"} else "box",
            "factor_quantiles": _coerce_factor_quantiles(stored_factor_quantiles, default=5),
            "factor_transform": stored_factor_transform if stored_factor_transform in {"raw", "zscore"} else "raw",
            "factor_qq_reference": (
                stored_factor_qq_reference
                if stored_factor_qq_reference in {"normal", "reference"}
                else "normal"
            ),
            "conditional_view": stored_conditional_view if stored_conditional_view in {"coincident", "forward"} else "forward",
            "conditional_comparator": stored_conditional_comparator if stored_conditional_comparator in {"le", "ge"} else "le",
            "conditional_threshold": stored_conditional_threshold if stored_conditional_threshold is not None else 0,
            "conditional_window_conversion": (
                stored_conditional_window_conversion
                if stored_conditional_window_conversion in {"compound", "end", "average", "sum"}
                else "compound"
            ),
            "conditional_step": _coerce_positive_int(stored_conditional_step, default=1),
            "conditional_step_unit": stored_conditional_step_unit if stored_conditional_step_unit in {"periods", "months"} else "months",
            "conditional_display_mode": (
                stored_conditional_display_mode
                if stored_conditional_display_mode in {"summary", "detail"}
                else "summary"
            ),
            "regime_display_mode": (
                stored_regime_display_mode
                if stored_regime_display_mode in {"summary", "detail"}
                else "summary"
            ),
            "monthly_view": stored_monthly_view if stored_monthly_view is not None else "annual",
            "valid_selection": valid_selection,
            "updated_order": updated_order,
        }
    )
    return resolved


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
    Output("at-factor-mode-select", "value"),
    Output("at-factor-quantiles-input", "value"),
    Output("at-factor-transform-select", "value"),
    Output("at-factor-qq-reference-select", "value"),
    Output("at-conditional-view-select", "value"),
    Output("at-conditional-comparator-select", "value"),
    Output("at-conditional-threshold-input", "value"),
    Output("at-conditional-window-conversion-select", "value"),
    Output("at-conditional-step-input", "value"),
    Output("at-conditional-step-unit-select", "value"),
    Output("at-conditional-display-mode-select", "value"),
    Output("at-regime-detail-display-mode-select", "value"),
    Output("at-monthly-view-checkbox", "value"),
    Output("at-series-select", "data"),
    Output("at-series-order-store", "data", allow_duplicate=True),
    Output("at-state-ready-store", "data", allow_duplicate=True),
    Input("at-page-load-trigger", "n_intervals"),
    Input("dashmat-raw-data-meta-store", "data"),
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
    State("at-factor-mode-store", "data"),
    State("at-factor-quantiles-store", "data"),
    State("at-factor-transform-store", "data"),
    State("at-factor-qq-reference-store", "data"),
    State("at-conditional-view-store", "data"),
    State("at-conditional-comparator-store", "data"),
    State("at-conditional-threshold-store", "data"),
    State("at-conditional-window-conversion-store", "data"),
    State("at-conditional-step-store", "data"),
    State("at-conditional-step-unit-store", "data"),
    State("at-conditional-display-mode-store", "data"),
    State("at-regime-detail-display-mode-store", "data"),
    State("at-monthly-view-store", "data"),
    State("at-monthly-series-store", "data"),
    State("at-series-order-store", "data"),
    State("dashmat-pending-new-series-store", "data"),
    State("at-page-visited-store", "data"),
    prevent_initial_call="initial_duplicate",
)
def restore_application_state(
    n_intervals,
    raw_meta,
    stored_periodicity,
    stored_series,
    stored_returns,
    stored_vol,
    stored_tab,
    stored_roll_win,
    stored_roll_metric,
    stored_roll_type,
    stored_roll_chart,
    stored_dd_chart,
    stored_gr_chart,
    stored_factor_mode,
    stored_factor_quantiles,
    stored_factor_transform,
    stored_factor_qq_reference,
    stored_conditional_view,
    stored_conditional_comparator,
    stored_conditional_threshold,
    stored_conditional_window_conversion,
    stored_conditional_step,
    stored_conditional_step_unit,
    stored_conditional_display_mode,
    stored_regime_display_mode,
    stored_monthly_view,
    stored_monthly_series,
    stored_order,
    po_origin_series,
    page_visited,
):
    try:
        resolved = _at_resolve_restore_state(
            raw_meta,
            stored_periodicity,
            stored_series,
            stored_returns,
            stored_vol,
            stored_tab,
            stored_roll_win,
            stored_roll_metric,
            stored_roll_type,
            stored_roll_chart,
            stored_dd_chart,
            stored_gr_chart,
            stored_factor_mode,
            stored_factor_quantiles,
            stored_factor_transform,
            stored_factor_qq_reference,
            stored_conditional_view,
            stored_conditional_comparator,
            stored_conditional_threshold,
            stored_conditional_window_conversion,
            stored_conditional_step,
            stored_conditional_step_unit,
            stored_conditional_display_mode,
            stored_regime_display_mode,
            stored_monthly_view,
            stored_order,
            po_origin_series,
            page_visited,
        )
        active_tab = resolved["active_tab"]
        roll_outputs = (
            resolved["roll_win"],
            resolved["roll_metric"],
            resolved["roll_type"],
            resolved["roll_type_disabled"],
            resolved["roll_type_style"],
            resolved["roll_chart"],
        ) if active_tab == "rolling" else (no_update, no_update, no_update, no_update, no_update, no_update)
        drawdown_output = resolved["dd_chart"] if active_tab == "drawdown" else no_update
        growth_output = resolved["gr_chart"] if active_tab == "growth" else no_update
        factor_outputs = (
            resolved["factor_mode"],
            resolved["factor_quantiles"],
            resolved["factor_transform"],
            resolved["factor_qq_reference"],
        ) if active_tab == "factor_analysis" else (no_update, no_update, no_update, no_update)
        conditional_outputs = (
            resolved["conditional_view"],
            resolved["conditional_comparator"],
            resolved["conditional_threshold"],
            resolved["conditional_window_conversion"],
            resolved["conditional_step"],
            resolved["conditional_step_unit"],
            resolved["conditional_display_mode"],
        ) if active_tab == "conditional_returns" else (no_update, no_update, no_update, no_update, no_update, no_update, no_update)
        regime_output = resolved["regime_display_mode"] if active_tab == "regime_analysis" else no_update
        monthly_output = resolved["monthly_view"] if active_tab == "calendar" else no_update

        return (
            resolved["periodicity_options"],
            resolved["valid_periodicity"],
            resolved["valid_returns"],
            resolved["valid_vol"],
            active_tab,
            *roll_outputs,
            drawdown_output,
            growth_output,
            *factor_outputs,
            *conditional_outputs,
            regime_output,
            monthly_output,
            resolved["valid_selection"],
            resolved["updated_order"],
            False,
        )
    except Exception:
        resolved = _at_restore_defaults()
        return (
            resolved["periodicity_options"], resolved["valid_periodicity"], resolved["valid_returns"], resolved["valid_vol"], resolved["active_tab"],
            no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update,
            no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update,
            no_update,
            resolved["valid_selection"], resolved["updated_order"], False
        )


@callback(
    Output("at-returns-type-select", "value", allow_duplicate=True),
    Input("at-returns-type-select-returns", "value"),
    Input("at-returns-type-select-calendar", "value"),
    Input("at-returns-type-select-drawdown", "value"),
    Input("at-returns-type-select-correlogram", "value"),
    Input("at-returns-type-select-factor", "value"),
    Input("at-returns-type-select-conditional", "value"),
    Input("at-returns-type-select-regime", "value"),
    State("at-returns-type-select", "value"),
    prevent_initial_call=True,
)
def sync_at_returns_type_from_mirrors(
    returns_value,
    calendar_value,
    drawdown_value,
    correlogram_value,
    factor_value,
    conditional_value,
    regime_value,
    current_value,
):
    value_by_trigger = {
        "at-returns-type-select-returns": returns_value,
        "at-returns-type-select-calendar": calendar_value,
        "at-returns-type-select-drawdown": drawdown_value,
        "at-returns-type-select-correlogram": correlogram_value,
        "at-returns-type-select-factor": factor_value,
        "at-returns-type-select-conditional": conditional_value,
        "at-returns-type-select-regime": regime_value,
    }
    next_value = value_by_trigger.get(callback_context.triggered_id)
    if next_value is None:
        return no_update
    normalized = "excess" if next_value == "excess" else "total"
    if normalized == ("excess" if current_value == "excess" else "total"):
        return no_update
    return normalized


@callback(
    Output("at-returns-type-select-returns", "value"),
    Output("at-returns-type-select-calendar", "value"),
    Output("at-returns-type-select-drawdown", "value"),
    Output("at-returns-type-select-correlogram", "value"),
    Output("at-returns-type-select-factor", "value"),
    Output("at-returns-type-select-conditional", "value"),
    Output("at-returns-type-select-regime", "value"),
    Input("at-returns-type-select", "value"),
    State("at-returns-type-select-returns", "value"),
    State("at-returns-type-select-calendar", "value"),
    State("at-returns-type-select-drawdown", "value"),
    State("at-returns-type-select-correlogram", "value"),
    State("at-returns-type-select-factor", "value"),
    State("at-returns-type-select-conditional", "value"),
    State("at-returns-type-select-regime", "value"),
    prevent_initial_call=False,
)
def sync_at_returns_type_mirrors(
    current_value,
    returns_value,
    calendar_value,
    drawdown_value,
    correlogram_value,
    factor_value,
    conditional_value,
    regime_value,
):
    normalized = "excess" if current_value == "excess" else "total"

    def _sync(value):
        return no_update if value == normalized else normalized

    return (
        _sync(returns_value),
        _sync(calendar_value),
        _sync(drawdown_value),
        _sync(correlogram_value),
        _sync(factor_value),
        _sync(conditional_value),
        _sync(regime_value),
    )


@callback(
    Output("at-rolling-window-select", "value", allow_duplicate=True),
    Output("at-rolling-metric-select", "value", allow_duplicate=True),
    Output("at-rolling-return-type-select", "value", allow_duplicate=True),
    Output("at-rolling-return-type-select", "disabled", allow_duplicate=True),
    Output("at-rolling-return-type-select", "style", allow_duplicate=True),
    Output("at-rolling-chart-switch", "value", allow_duplicate=True),
    Output("at-drawdown-chart-switch", "value", allow_duplicate=True),
    Output("at-growth-chart-switch", "value", allow_duplicate=True),
    Output("at-factor-mode-select", "value", allow_duplicate=True),
    Output("at-factor-quantiles-input", "value", allow_duplicate=True),
    Output("at-factor-transform-select", "value", allow_duplicate=True),
    Output("at-factor-qq-reference-select", "value", allow_duplicate=True),
    Output("at-conditional-view-select", "value", allow_duplicate=True),
    Output("at-conditional-comparator-select", "value", allow_duplicate=True),
    Output("at-conditional-threshold-input", "value", allow_duplicate=True),
    Output("at-conditional-window-conversion-select", "value", allow_duplicate=True),
    Output("at-conditional-step-input", "value", allow_duplicate=True),
    Output("at-conditional-step-unit-select", "value", allow_duplicate=True),
    Output("at-conditional-display-mode-select", "value", allow_duplicate=True),
    Output("at-regime-detail-display-mode-select", "value", allow_duplicate=True),
    Output("at-monthly-view-checkbox", "value", allow_duplicate=True),
    Input("at-secondary-restore-ready-store", "data"),
    State("dashmat-raw-data-meta-store", "data"),
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
    State("at-factor-mode-store", "data"),
    State("at-factor-quantiles-store", "data"),
    State("at-factor-transform-store", "data"),
    State("at-factor-qq-reference-store", "data"),
    State("at-conditional-view-store", "data"),
    State("at-conditional-comparator-store", "data"),
    State("at-conditional-threshold-store", "data"),
    State("at-conditional-window-conversion-store", "data"),
    State("at-conditional-step-store", "data"),
    State("at-conditional-step-unit-store", "data"),
    State("at-conditional-display-mode-store", "data"),
    State("at-regime-detail-display-mode-store", "data"),
    State("at-monthly-view-store", "data"),
    State("at-series-order-store", "data"),
    State("dashmat-pending-new-series-store", "data"),
    State("at-page-visited-store", "data"),
    prevent_initial_call=True,
)
def at_restore_secondary_controls(
    secondary_ready,
    raw_meta,
    stored_periodicity,
    stored_series,
    stored_returns,
    stored_vol,
    stored_tab,
    stored_roll_win,
    stored_roll_metric,
    stored_roll_type,
    stored_roll_chart,
    stored_dd_chart,
    stored_gr_chart,
    stored_factor_mode,
    stored_factor_quantiles,
    stored_factor_transform,
    stored_factor_qq_reference,
    stored_conditional_view,
    stored_conditional_comparator,
    stored_conditional_threshold,
    stored_conditional_window_conversion,
    stored_conditional_step,
    stored_conditional_step_unit,
    stored_conditional_display_mode,
    stored_regime_display_mode,
    stored_monthly_view,
    stored_order,
    po_origin_series,
    page_visited,
):
    if not secondary_ready:
        raise PreventUpdate

    resolved = _at_resolve_restore_state(
        raw_meta,
        stored_periodicity,
        stored_series,
        stored_returns,
        stored_vol,
        stored_tab,
        stored_roll_win,
        stored_roll_metric,
        stored_roll_type,
        stored_roll_chart,
        stored_dd_chart,
        stored_gr_chart,
        stored_factor_mode,
        stored_factor_quantiles,
        stored_factor_transform,
        stored_factor_qq_reference,
        stored_conditional_view,
        stored_conditional_comparator,
        stored_conditional_threshold,
        stored_conditional_window_conversion,
        stored_conditional_step,
        stored_conditional_step_unit,
        stored_conditional_display_mode,
        stored_regime_display_mode,
        stored_monthly_view,
        stored_order,
        po_origin_series,
        page_visited,
    )
    active_tab = resolved["active_tab"]
    return (
        no_update if active_tab == "rolling" else resolved["roll_win"],
        no_update if active_tab == "rolling" else resolved["roll_metric"],
        no_update if active_tab == "rolling" else resolved["roll_type"],
        no_update if active_tab == "rolling" else resolved["roll_type_disabled"],
        no_update if active_tab == "rolling" else resolved["roll_type_style"],
        no_update if active_tab == "rolling" else resolved["roll_chart"],
        no_update if active_tab == "drawdown" else resolved["dd_chart"],
        no_update if active_tab == "growth" else resolved["gr_chart"],
        no_update if active_tab == "factor_analysis" else resolved["factor_mode"],
        no_update if active_tab == "factor_analysis" else resolved["factor_quantiles"],
        no_update if active_tab == "factor_analysis" else resolved["factor_transform"],
        no_update if active_tab == "factor_analysis" else resolved["factor_qq_reference"],
        no_update if active_tab == "conditional_returns" else resolved["conditional_view"],
        no_update if active_tab == "conditional_returns" else resolved["conditional_comparator"],
        no_update if active_tab == "conditional_returns" else resolved["conditional_threshold"],
        no_update if active_tab == "conditional_returns" else resolved["conditional_window_conversion"],
        no_update if active_tab == "conditional_returns" else resolved["conditional_step"],
        no_update if active_tab == "conditional_returns" else resolved["conditional_step_unit"],
        no_update if active_tab == "conditional_returns" else resolved["conditional_display_mode"],
        no_update if active_tab == "regime_analysis" else resolved["regime_display_mode"],
        no_update if active_tab == "calendar" else resolved["monthly_view"],
    )


@callback(
    Output("at-factor-def-db-available-store", "data", allow_duplicate=True),
    Output("at-factor-definitions-db-store", "data", allow_duplicate=True),
    Output("at-factor-def-loaded-store", "data", allow_duplicate=True),
    Input("at-main-tabs", "value"),
    State("at-factor-def-loaded-store", "data"),
    prevent_initial_call=True,
)
def at_lazy_load_factor_definitions(active_tab, loaded):
    if active_tab not in {"factor_analysis", "conditional_returns"} or loaded:
        raise PreventUpdate

    factor_available = False
    factor_definitions = []
    try:
        factor_available = bool(factor_tables_available(DB_ENGINE))
        if factor_available:
            factor_definitions = load_factor_definitions(DB_ENGINE)
    except Exception:
        factor_available = False
        factor_definitions = []

    return factor_available, factor_definitions, True


@callback(
    Output("at-regime-def-db-available-store", "data", allow_duplicate=True),
    Output("at-regime-definitions-db-store", "data", allow_duplicate=True),
    Output("at-regime-def-loaded-store", "data", allow_duplicate=True),
    Input("at-main-tabs", "value"),
    State("at-regime-def-loaded-store", "data"),
    prevent_initial_call=True,
)
def at_lazy_load_regime_definitions(active_tab, loaded):
    if active_tab != "regime_analysis" or loaded:
        raise PreventUpdate

    regime_available = False
    regime_definitions = []
    try:
        regime_available = bool(regime_tables_available(DB_ENGINE))
        if regime_available:
            regime_definitions = load_regime_definitions(DB_ENGINE)
    except Exception:
        regime_available = False
        regime_definitions = []

    return regime_available, regime_definitions, True


# Clientside navigation callbacks
clientside_callback(
    ClientsideFunction(namespace="dashmat_callbacks", function_name="navigateAnalytics"),
    Output("at-url-location", "pathname"),
    Input("at-menu-exit", "n_clicks"),
    Input("at-menu-view-portfolio", "n_clicks"),
    Input("at-menu-view-regression", "n_clicks"),
    prevent_initial_call=True,
)


clientside_callback(
    ClientsideFunction(namespace="dashmat_callbacks", function_name="clearWorkspaceSession"),
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


# Clientside callback to trigger upload from menu or welcome button
clientside_callback(
    ClientsideFunction(namespace="dashmat_callbacks", function_name="triggerAnalyticsUpload"),
    Output("at-ui-blocker-store", "data", allow_duplicate=True),
    Input("at-menu-add-series", "n_clicks"),
    Input("at-welcome-add-series-btn", "n_clicks"),
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

clientside_callback(
    ClientsideFunction(namespace="dashmat_callbacks", function_name="analyticsInitialSeriesBlocker"),
    Output("at-ui-blocker-store", "data", allow_duplicate=True),
    Input("at-url-location", "pathname"),
    Input("dashmat-raw-data-meta-store", "data"),
    Input("at-series-select", "data"),
    Input("at-page-load-trigger", "n_intervals"),
    Input("at-series-selection-modal", "opened"),
    Input("at-series-selection-grid", "virtualRowData", allow_optional=True),
    State("at-page-visited-store", "data"),
    State("at-series-order-store", "data"),
    State("dashmat-pending-new-series-store", "data"),
    prevent_initial_call=True,
)

clientside_callback(
    ClientsideFunction(namespace="dashmat_callbacks", function_name="releaseBlockerOnSeriesGridReady"),
    Output("at-ui-blocker-store", "data", allow_duplicate=True),
    Input("at-series-selection-grid", "virtualRowData", allow_optional=True),
    State("at-series-selection-modal", "opened"),
    prevent_initial_call=True,
)


def _at_get_series_page_state(raw_meta, current_select, current_order, po_origin_series):
    """Classify raw columns into page-known, saved-origin, and generic new series."""
    columns = []
    if isinstance(raw_meta, dict):
        maybe_columns = raw_meta.get("columns")
        if isinstance(maybe_columns, list):
            columns = maybe_columns
    elif isinstance(raw_meta, (list, tuple)):
        columns = list(raw_meta)
    if not columns:
        return [], [], [], []

    selected_valid = [series for series in (current_select or []) if series in columns]
    known_columns = set(series for series in (current_order or []) if series in columns)
    known_columns.update(selected_valid)
    po_origin_set = {series for series in saved_series_store_names(po_origin_series) if series in columns}
    generic_new = [
        series for series in columns
        if series not in known_columns and series not in po_origin_set
    ]
    po_new = [
        series for series in columns
        if series not in known_columns and series in po_origin_set
    ]
    return columns, selected_valid, generic_new, po_new


clientside_callback(
    ClientsideFunction(namespace="dashmat_callbacks", function_name="openAnalyticsSeriesModal"),
    Output("at-series-selection-modal", "opened", allow_duplicate=True),
    Output("at-temp-series-select", "data", allow_duplicate=True),
    Output("at-temp-benchmark-assignments-store", "data", allow_duplicate=True),
    Output("at-temp-long-short-store", "data", allow_duplicate=True),
    Output("at-temp-series-order-store", "data", allow_duplicate=True),
    Output("at-temp-deleted-series-store", "data", allow_duplicate=True),
    Output("at-temp-vol-scaling-assignments-store", "data", allow_duplicate=True),
    Output("at-page-visited-store", "data", allow_duplicate=True),
    Output("at-ui-blocker-store", "data", allow_duplicate=True),
    Input("at-open-series-modal-button", "n_clicks"),
    Input("at-page-load-trigger", "n_intervals"),
    State("at-url-location", "pathname"),
    State("dashmat-raw-data-meta-store", "data"),
    State("at-series-select", "data"),
    State("at-benchmark-assignments-store", "data"),
    State("at-long-short-store", "data"),
    State("at-series-order-store", "data"),
    State("at-vol-scaling-assignments-store", "data"),
    State("dashmat-pending-new-series-store", "data"),
    State("at-page-visited-store", "data"),
    prevent_initial_call=True,
)









# Clientside callback for top-level control storage
clientside_callback(
    ClientsideFunction(namespace="dashmat_callbacks", function_name="analyticsControlSync"),
    Output("at-periodicity-value-store", "data"),
    Output("at-returns-type-value-store", "data"),
    Output("at-vol-scaler-value-store", "data"),
    Output("at-series-select-value-store", "data"),
    Output("at-active-tab-store", "data"),
    Output("at-rolling-window-store", "data"),
    Output("at-rolling-metric-store", "data"),
    Output("at-rolling-return-type-store", "data"),
    Output("at-monthly-view-store", "data"),
    Output("at-monthly-series-store", "data"),
    Output("at-use-risk-free-store", "data"),
    Input("at-periodicity-select", "value"),
    Input("at-returns-type-select", "value"),
    Input("at-vol-scaler-input", "value"),
    Input("at-series-select", "data"),
    Input("at-main-tabs", "value"),
    Input("at-rolling-window-select", "value"),
    Input("at-rolling-metric-select", "value"),
    Input("at-rolling-return-type-select", "value"),
    Input("at-monthly-view-checkbox", "value"),
    Input("at-monthly-series-select", "value"),
    Input("at-use-risk-free-switch", "value"),
    prevent_initial_call=True,
)


@callback(
    Output("at-conditional-view-store", "data"),
    Output("at-conditional-comparator-store", "data"),
    Output("at-conditional-threshold-store", "data"),
    Output("at-conditional-window-conversion-store", "data"),
    Output("at-conditional-step-store", "data"),
    Output("at-conditional-step-unit-store", "data"),
    Output("at-conditional-display-mode-store", "data"),
    Input("at-conditional-display-mode-select", "value"),
    Input("at-conditional-view-select", "value"),
    Input("at-conditional-comparator-select", "value"),
    Input("at-conditional-threshold-input", "value"),
    Input("at-conditional-window-conversion-select", "value"),
    Input("at-conditional-step-input", "value"),
    Input("at-conditional-step-unit-select", "value"),
    prevent_initial_call=False,
)
def sync_conditional_returns_control_state(
    display_mode_value,
    view_value,
    comparator_value,
    threshold_value,
    conversion_value,
    step_value,
    step_unit,
):
    normalized_step = _coerce_positive_int(step_value, default=1)
    return (
        view_value if view_value in {"coincident", "forward"} else "forward",
        comparator_value if comparator_value in {"le", "ge"} else "le",
        threshold_value if threshold_value is not None else 0,
        conversion_value if conversion_value in {"compound", "end", "average", "sum"} else "compound",
        normalized_step,
        step_unit if step_unit in {"periods", "months"} else "months",
        display_mode_value if display_mode_value in {"summary", "detail"} else "summary",
    )


@callback(
    Output("at-regime-detail-display-mode-store", "data"),
    Input("at-regime-detail-display-mode-select", "value"),
    prevent_initial_call=False,
)
def sync_regime_detail_display_mode(value):
    return value if value in {"summary", "detail"} else "summary"


# Sync periodicity to PortOpt only on raw-data load/update events.
clientside_callback(
    """
    function(n, storedValue) {
        if (!n) {
            return window.dash_clientside.no_update;
        }
        return storedValue === false ? "zero" : "tbill";
    }
    """,
    Output("at-use-risk-free-switch", "value"),
    Input("at-page-load-trigger", "n_intervals"),
    State("at-use-risk-free-store", "data"),
    prevent_initial_call=True,
)


clientside_callback(
    ClientsideFunction(namespace="dashmat_callbacks", function_name="syncAnalyticsPeriodicity"),
    Output("at-periodicity-load-sync-dummy", "data"),
    Input("dashmat-raw-data-store", "data"),
    Input("at-periodicity-value-store", "data"),
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





clientside_callback(
    ClientsideFunction(namespace="dashmat_callbacks", function_name="analyticsViewSync"),
    Output("at-rolling-chart-switch-store", "data"),
    Output("at-rolling-grid-container", "style"),
    Output("at-rolling-chart-container", "style"),
    Output("at-drawdown-chart-switch-store", "data"),
    Output("at-drawdown-grid-container", "style"),
    Output("at-drawdown-chart-container", "style"),
    Output("at-growth-chart-switch-store", "data"),
    Output("at-growth-grid-container", "style"),
    Output("at-growth-chart-container", "style"),
    Input("at-rolling-chart-switch", "value"),
    Input("at-drawdown-chart-switch", "value"),
    Input("at-growth-chart-switch", "value"),
    prevent_initial_call=True,
)

clientside_callback(
    ClientsideFunction(namespace="dashmat_callbacks", function_name="analyticsFactorRegimeSync"),
    Output("at-factor-mode-store", "data"),
    Output("at-factor-quantiles-store", "data"),
    Output("at-factor-transform-store", "data"),
    Output("at-factor-series-store", "data"),
    Output("at-factor-qq-reference-store", "data"),
    Output("at-factor-series-wrapper", "style"),
    Output("at-factor-open-modal-wrapper", "style"),
    Output("at-factor-quantiles-wrapper", "style"),
    Output("at-factor-transform-wrapper", "style"),
    Output("at-factor-qq-reference-wrapper", "style"),
    Output("at-factor-series-select", "label"),
    Output("at-regime-definition-store", "data"),
    Output("at-regime-def-universe-wrapper", "style"),
    Output("at-regime-def-single-wrapper", "style"),
    Input("at-factor-mode-select", "value"),
    Input("at-factor-quantiles-input", "value"),
    Input("at-factor-transform-select", "value"),
    Input("at-factor-series-select", "value"),
    Input("at-factor-qq-reference-select", "value"),
    Input("at-regime-definition-select", "value"),
    Input("at-regime-def-method-type", "value"),
    prevent_initial_call=False,
)


@callback(
    Output("at-factor-series-select", "data"),
    Output("at-factor-series-select", "value", allow_duplicate=True),
    Output("at-factor-series-select-conditional", "data"),
    Output("at-factor-series-select-conditional", "value", allow_duplicate=True),
    Input("dashmat-raw-data-store", "data"),
    Input("at-series-select", "data"),
    Input("at-factor-definitions-db-store", "data"),
    Input("at-factor-definitions-local-store", "data"),
    State("at-factor-series-store", "data"),
    State("at-factor-series-select", "value"),
    prevent_initial_call="initial_duplicate",
)
def update_factor_series_select(
    raw_data,
    selected_series,
    db_definitions,
    local_definitions,
    stored_factor_series,
    current_factor_series,
):
    """Expose raw and custom factor candidates, with selected raw series first."""
    if raw_data is None:
        return [], None, [], None

    try:
        df = _raw_df(raw_data)
    except Exception:
        return [], None, [], None

    all_series = list(df.columns)
    if not all_series:
        return [], None, [], None

    selected_order = [s for s in (selected_series or []) if s in all_series]
    remaining = [s for s in all_series if s not in selected_order]
    ordered = selected_order + remaining
    options = [{"value": f"raw::{s}", "label": f"[Raw] {s}"} for s in ordered]

    definition_entries = _factor_option_definitions(db_definitions, local_definitions)
    definition_names = [entry["name"] for entry in definition_entries]
    for entry in definition_entries:
        source_label = _source_badge(entry["source"])
        options.append(
            {
                "value": f"def::{entry['name']}",
                "label": f"{source_label} {entry['name']}",
            }
        )

    raw_name_set = set(ordered)
    definition_name_set = set(definition_names)
    candidate_order = [
        current_factor_series,
        stored_factor_series,
        f"raw::{selected_order[0]}" if selected_order else None,
        f"raw::{ordered[0]}" if ordered else None,
    ]
    next_value = None
    for candidate in candidate_order:
        normalized_candidate = _normalize_factor_value_for_options(
            candidate,
            raw_name_set,
            definition_name_set,
        )
        if normalized_candidate:
            next_value = normalized_candidate
            break
    if not next_value and options:
        next_value = options[0]["value"]
    return options, next_value, options, next_value


@callback(
    Output("at-factor-series-select", "value", allow_duplicate=True),
    Output("at-factor-series-select-conditional", "value", allow_duplicate=True),
    Output("at-factor-transform-select", "value", allow_duplicate=True),
    Output("at-factor-transform-select-conditional", "value", allow_duplicate=True),
    Input("at-factor-series-select", "value"),
    Input("at-factor-series-select-conditional", "value"),
    Input("at-factor-transform-select", "value"),
    Input("at-factor-transform-select-conditional", "value"),
    prevent_initial_call=True,
)
def sync_factor_control_mirrors(
    factor_series_value,
    conditional_factor_series_value,
    factor_transform_value,
    conditional_factor_transform_value,
):
    trigger = callback_context.triggered_id
    if trigger == "at-factor-series-select":
        return no_update, factor_series_value, no_update, no_update
    if trigger == "at-factor-series-select-conditional":
        return conditional_factor_series_value, no_update, no_update, no_update
    if trigger == "at-factor-transform-select":
        return no_update, no_update, no_update, factor_transform_value
    if trigger == "at-factor-transform-select-conditional":
        return no_update, no_update, conditional_factor_transform_value, no_update
    raise PreventUpdate


@callback(
    Output("at-factor-def-modal", "opened", allow_duplicate=True),
    Output("at-factor-def-status-alert", "hide", allow_duplicate=True),
    Input("at-menu-add-factor", "n_clicks"),
    Input("at-factor-open-modal-btn", "n_clicks"),
    Input("at-factor-open-modal-btn-conditional", "n_clicks"),
    prevent_initial_call=True,
)
def at_open_factor_definition_modal(menu_clicks, tab_clicks, conditional_tab_clicks):
    if not menu_clicks and not tab_clicks and not conditional_tab_clicks:
        raise PreventUpdate
    return True, True


@callback(
    Output("at-factor-def-db-available-store", "data", allow_duplicate=True),
    Output("at-factor-definitions-db-store", "data", allow_duplicate=True),
    Output("at-factor-def-loaded-store", "data", allow_duplicate=True),
    Output("at-factor-def-long-components", "data"),
    Output("at-factor-def-short-components", "data"),
    Output("at-factor-def-db-available-note", "children"),
    Output("at-factor-def-save-db-btn", "disabled"),
    Input("at-factor-def-modal", "opened"),
    prevent_initial_call=True,
)
def at_load_factor_modal_data(opened):
    if not opened:
        raise PreventUpdate
    component_options = get_sec_factor_component_options_cached(MRD_ENGINE)
    db_available = factor_tables_available(DB_ENGINE)
    db_definitions = load_factor_definitions(DB_ENGINE) if db_available else []
    note = "" if db_available else "Database factor tables are unavailable. Session factors are still supported."
    return db_available, db_definitions, True, component_options, component_options, note, (not db_available)


@callback(
    Output("at-factor-def-select", "data"),
    Input("at-factor-definitions-db-store", "data"),
    Input("at-factor-definitions-local-store", "data"),
)
def at_update_factor_definition_select_options(db_definitions, local_definitions):
    entries = _factor_option_definitions(db_definitions, local_definitions)
    options = []
    for entry in entries:
        source_label = _source_badge(entry["source"])
        options.append(
            {
                "value": _factor_select_key(entry["source"], entry["name"]),
                "label": f"{source_label} {entry['name']}",
            }
        )
    return options


@callback(
    Output("at-factor-def-modal-draft-store", "data", allow_duplicate=True),
    Input("at-factor-def-select", "value"),
    State("at-factor-definitions-db-store", "data"),
    State("at-factor-definitions-local-store", "data"),
    State("at-factor-def-modal-draft-store", "data"),
    prevent_initial_call=True,
)
def at_load_selected_factor_definition(selected_key, db_definitions, local_definitions, current_draft):
    if not selected_key:
        raise PreventUpdate
    current = _ensure_factor_draft(current_draft)
    if current.get("selected_key") == selected_key:
        raise PreventUpdate

    source, name = _split_factor_select_key(selected_key)
    if source == "db":
        definition = _lookup_factor_definition(name, db_definitions, [])
        if not definition:
            raise PreventUpdate
        return _definition_to_draft(definition, "db", selected_key=selected_key)
    if source == "session":
        definition = _lookup_factor_definition(name, [], local_definitions)
        if not definition:
            raise PreventUpdate
        return _definition_to_draft(definition, "session", selected_key=selected_key)
    raise PreventUpdate


@callback(
    Output("at-factor-def-modal-draft-store", "data", allow_duplicate=True),
    Output("at-factor-def-select", "value", allow_duplicate=True),
    Output("at-factor-def-status-alert", "children", allow_duplicate=True),
    Output("at-factor-def-status-alert", "color", allow_duplicate=True),
    Output("at-factor-def-status-alert", "hide", allow_duplicate=True),
    Input("at-factor-def-new-btn", "n_clicks"),
    Input("at-factor-def-select", "value"),
    prevent_initial_call=True,
)
def at_reset_factor_definition_draft(new_clicks, selected_key):
    triggered = callback_context.triggered_id
    if triggered == "at-factor-def-new-btn" and new_clicks:
        return _default_factor_draft(), None, "New session factor draft started.", "blue", False
    if triggered == "at-factor-def-select" and not selected_key:
        return _default_factor_draft(), no_update, "New session factor draft started.", "blue", False
    raise PreventUpdate


@callback(
    Output("at-factor-def-select", "value", allow_duplicate=True),
    Output("at-factor-def-name-input", "value"),
    Output("at-factor-def-description-input", "value"),
    Output("at-factor-def-long-components", "value"),
    Output("at-factor-def-short-components", "value"),
    Output("at-factor-def-long-agg-type", "value"),
    Output("at-factor-def-short-agg-type", "value"),
    Output("at-factor-def-long-lag", "value"),
    Output("at-factor-def-output-transform", "value"),
    Input("at-factor-def-modal-draft-store", "data"),
    prevent_initial_call="initial_duplicate",
)
def at_sync_factor_definition_form(draft_data):
    draft = _ensure_factor_draft(draft_data)
    if draft.get("sync_origin") == "form":
        raise PreventUpdate

    long_agg = str(int(draft.get("LongAggType") or 1))
    short_agg = draft.get("ShortAggType")
    short_agg_value = str(int(short_agg)) if short_agg is not None else None
    output_transform = str(int(draft.get("OutputTransform") or 0))
    return (
        draft.get("selected_key"),
        draft.get("FactorName") or "",
        draft.get("Description") or "",
        draft.get("LongComponentList") or [],
        draft.get("ShortComponentList") or [],
        long_agg,
        short_agg_value,
        int(draft.get("LongLag") or 0),
        output_transform,
    )


@callback(
    Output("at-factor-def-modal-draft-store", "data", allow_duplicate=True),
    Input("at-factor-def-name-input", "value"),
    Input("at-factor-def-description-input", "value"),
    Input("at-factor-def-long-components", "value"),
    Input("at-factor-def-short-components", "value"),
    Input("at-factor-def-long-agg-type", "value"),
    Input("at-factor-def-short-agg-type", "value"),
    Input("at-factor-def-long-lag", "value"),
    Input("at-factor-def-output-transform", "value"),
    State("at-factor-def-modal-draft-store", "data"),
    prevent_initial_call=True,
)
def at_update_factor_definition_draft_from_form(
    factor_name,
    description,
    long_components,
    short_components,
    long_agg_type,
    short_agg_type,
    long_lag,
    output_transform,
    draft_data,
):
    draft = _ensure_factor_draft(draft_data)
    next_draft = dict(draft)
    next_draft["sync_origin"] = "form"
    next_draft["FactorName"] = str(factor_name or "").strip()
    next_draft["Description"] = "" if description is None else str(description)
    next_draft["LongComponentList"] = [str(v) for v in (long_components or [])]
    next_draft["ShortComponentList"] = [str(v) for v in (short_components or [])]
    next_draft["LongAggType"] = int(pd.to_numeric(pd.Series([long_agg_type]), errors="coerce").iloc[0] or 1)
    short_num = pd.to_numeric(pd.Series([short_agg_type]), errors="coerce").iloc[0]
    next_draft["ShortAggType"] = int(short_num) if not pd.isna(short_num) else None
    next_draft["LongLag"] = max(0, int(pd.to_numeric(pd.Series([long_lag]), errors="coerce").iloc[0] or 0))
    next_draft["OutputTransform"] = int(pd.to_numeric(pd.Series([output_transform]), errors="coerce").iloc[0] or 0)
    if next_draft == draft:
        raise PreventUpdate
    return next_draft


@callback(
    Output("at-factor-def-preview-lines", "children"),
    Input("at-factor-def-modal", "opened"),
    Input("at-factor-def-modal-draft-store", "data"),
    Input("at-periodicity-select", "value"),
    Input("at-date-range-store", "data"),
    prevent_initial_call=True,
)
def at_update_factor_definition_preview(opened, draft_data, periodicity, date_range):
    if not opened:
        raise PreventUpdate

    payload = _draft_to_definition_payload(draft_data or {})
    normalized, error = validate_factor_definition_payload(payload)
    if error or not normalized:
        return "Define a valid factor to preview values."

    lines = compute_factor_preview_lines(
        MRD_ENGINE,
        normalized,
        periodicity or "daily",
        date_range,
        max_rows=6,
    )
    if not lines:
        return "No rows returned for the current factor definition and date range."
    return "\n".join(lines)





@callback(
    Output("at-factor-definitions-local-store", "data", allow_duplicate=True),
    Output("at-factor-definitions-db-store", "data", allow_duplicate=True),
    Output("at-factor-def-modal-draft-store", "data", allow_duplicate=True),
    Output("at-factor-def-select", "value", allow_duplicate=True),
    Output("at-factor-series-select", "value", allow_duplicate=True),
    Output("at-factor-def-modal", "opened", allow_duplicate=True),
    Output("at-factor-def-status-alert", "children", allow_duplicate=True),
    Output("at-factor-def-status-alert", "color", allow_duplicate=True),
    Output("at-factor-def-status-alert", "hide", allow_duplicate=True),
    Input("at-factor-def-save-local-btn", "n_clicks"),
    Input("at-factor-def-save-db-btn", "n_clicks"),
    Input("at-factor-def-delete-btn", "n_clicks"),
    Input("at-factor-def-use-btn", "n_clicks"),
    Input("at-factor-def-close-btn", "n_clicks"),
    State("at-factor-def-modal-draft-store", "data"),
    State("at-factor-definitions-local-store", "data"),
    State("at-factor-definitions-db-store", "data"),
    State("at-factor-def-db-available-store", "data"),
    State("userinfo", "data"),
    prevent_initial_call=True,
)
def at_manage_factor_definitions(
    save_local_clicks,
    save_db_clicks,
    delete_clicks,
    use_clicks,
    close_clicks,
    draft_data,
    local_definitions,
    db_definitions,
    db_available,
    userinfo,
):
    triggered = callback_context.triggered_id
    n_no = no_update
    draft = _ensure_factor_draft(draft_data)
    local_list = [dict(item) for item in (local_definitions or []) if isinstance(item, dict)]
    update_by = _factor_user_label(userinfo)

    if triggered == "at-factor-def-close-btn":
        return n_no, n_no, n_no, n_no, n_no, False, n_no, n_no, True

    payload = _draft_to_definition_payload(draft)
    normalized, error = validate_factor_definition_payload(payload)
    if triggered in {"at-factor-def-save-local-btn", "at-factor-def-save-db-btn", "at-factor-def-use-btn"}:
        if error or not normalized:
            return n_no, n_no, n_no, n_no, n_no, n_no, error or "Invalid factor definition.", "red", False
    else:
        normalized = normalized or {}

    if triggered == "at-factor-def-save-local-btn":
        now_str = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        saved_item = dict(normalized)
        saved_item["source"] = "session"
        saved_item["UPDATE_DATE"] = now_str
        saved_item["UPDATE_BY"] = update_by

        target_name = str(saved_item.get("FactorName") or "").strip().lower()
        updated_local = [item for item in local_list if str(item.get("FactorName", "")).strip().lower() != target_name]
        updated_local.append(saved_item)
        updated_local.sort(key=lambda item: str(item.get("FactorName", "")).lower())

        next_draft = _definition_to_draft(saved_item, "session")
        return (
            updated_local,
            n_no,
            next_draft,
            next_draft.get("selected_key"),
            n_no,
            n_no,
            f"Saved session factor `{saved_item['FactorName']}`.",
            "green",
            False,
        )

    if triggered == "at-factor-def-save-db-btn":
        if not db_available:
            return n_no, n_no, n_no, n_no, n_no, n_no, "Database factor tables are unavailable.", "orange", False

        original_name = draft.get("original_name") if draft.get("source") == "db" else None
        expected_update = draft.get("selected_update_date") if draft.get("source") == "db" else None
        success, message, saved_row = save_factor_definition(
            DB_ENGINE,
            payload,
            update_by=update_by,
            original_name=original_name,
            expected_update_date=expected_update,
        )
        if not success or not saved_row:
            return n_no, n_no, n_no, n_no, n_no, n_no, message, "red", False

        updated_db = load_factor_definitions(DB_ENGINE)
        saved_name = str(saved_row.get("FactorName", "")).strip().lower()
        updated_local = [
            item for item in local_list if str(item.get("FactorName", "")).strip().lower() != saved_name
        ]
        next_draft = _definition_to_draft(saved_row, "db")
        return (
            updated_local,
            updated_db,
            next_draft,
            next_draft.get("selected_key"),
            f"def::{saved_row.get('FactorName')}",
            n_no,
            message,
            "green",
            False,
        )

    if triggered == "at-factor-def-delete-btn":
        name = str(draft.get("FactorName", "")).strip()
        if not name:
            return n_no, n_no, n_no, n_no, n_no, n_no, "Select or enter a factor name to delete.", "orange", False

        if draft.get("source") == "db":
            if not db_available:
                return n_no, n_no, n_no, n_no, n_no, n_no, "Database factor tables are unavailable.", "orange", False
            target_name = str(draft.get("original_name") or name).strip()
            success, message = delete_factor_definition(
                DB_ENGINE,
                target_name,
                expected_update_date=draft.get("selected_update_date"),
            )
            if not success:
                return n_no, n_no, n_no, n_no, n_no, n_no, message, "red", False
            updated_db = load_factor_definitions(DB_ENGINE)
            return n_no, updated_db, _default_factor_draft(), None, n_no, n_no, message, "green", False

        target_name = name.lower()
        updated_local = [item for item in local_list if str(item.get("FactorName", "")).strip().lower() != target_name]
        return (
            updated_local,
            n_no,
            _default_factor_draft(),
            None,
            n_no,
            n_no,
            f"Deleted session factor `{name}`.",
            "green",
            False,
        )

    if triggered == "at-factor-def-use-btn":
        factor_name = str(normalized.get("FactorName", "")).strip()
        if not factor_name:
            return n_no, n_no, n_no, n_no, n_no, n_no, "Factor name is required.", "red", False

        draft_mode = str(draft.get("DraftMode") or "").strip().lower()
        if draft_mode == "db":
            baseline_name = str(draft.get("original_name") or factor_name).strip()
            baseline = _lookup_factor_definition(baseline_name, db_definitions, [])
            if not baseline:
                return (
                    n_no,
                    n_no,
                    _definition_to_draft(
                        {
                            **normalized,
                            "UPDATE_DATE": draft.get("selected_update_date"),
                            "UPDATE_BY": draft.get("UPDATE_BY"),
                        },
                        "db",
                        selected_key=_factor_select_key("db", baseline_name or factor_name),
                    ),
                    _factor_select_key("db", baseline_name or factor_name),
                    f"def::{baseline_name or factor_name}",
                    False,
                    "Database factor selected for analysis.",
                    "green",
                    False,
                )

            baseline_name_actual = str(baseline.get("FactorName", "")).strip() or baseline_name
            unchanged_db = (
                str(factor_name).lower() == str(baseline_name_actual).lower()
                and _factor_definition_signature(normalized) == _factor_definition_signature(baseline)
            )
            if unchanged_db:
                next_draft = _definition_to_draft(
                    baseline,
                    "db",
                    selected_key=_factor_select_key("db", baseline_name_actual),
                )
                return (
                    n_no,
                    n_no,
                    next_draft,
                    next_draft.get("selected_key"),
                    f"def::{baseline_name_actual}",
                    False,
                    "Database factor selected for analysis.",
                    "green",
                    False,
                )

            if _factor_db_name_exists(factor_name, db_definitions):
                return (
                    n_no,
                    n_no,
                    n_no,
                    n_no,
                    n_no,
                    n_no,
                    "Rename the factor to create a session copy; that name already exists in Database definitions.",
                    "orange",
                    False,
                )

        now_str = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        session_item = {
            **normalized,
            "source": "session",
            "UPDATE_DATE": now_str,
            "UPDATE_BY": update_by,
        }
        key_name = factor_name.lower()
        next_local = [item for item in local_list if str(item.get("FactorName", "")).strip().lower() != key_name]
        next_local.append(session_item)
        next_local.sort(key=lambda item: str(item.get("FactorName", "")).lower())
        next_draft = _definition_to_draft(session_item, "session")

        return (
            next_local,
            n_no,
            next_draft,
            next_draft.get("selected_key"),
            f"def::{factor_name}",
            False,
            "Session factor selected for analysis.",
            "green",
            False,
        )

    raise PreventUpdate


@callback(
    Output("at-regime-definition-select", "data"),
    Output("at-regime-definition-select", "value", allow_duplicate=True),
    Input("at-regime-definitions-db-store", "data"),
    Input("at-regime-definitions-local-store", "data"),
    State("at-regime-definition-store", "data"),
    State("at-regime-definition-select", "value"),
    prevent_initial_call="initial_duplicate",
)
def at_update_regime_definition_analysis_select_options(
    db_definitions,
    local_definitions,
    stored_selection,
    current_selection,
):
    entries = _regime_option_definitions(db_definitions, local_definitions)
    options = []
    names = []
    for entry in entries:
        source_label = _source_badge(entry["source"])
        options.append(
            {
                "value": f"def::{entry['name']}",
                "label": f"{source_label} {entry['name']}",
            }
        )
        names.append(entry["name"])

    name_set = set(names)
    candidate_order = [
        current_selection,
        stored_selection,
        options[0]["value"] if options else None,
    ]
    next_value = None
    for candidate in candidate_order:
        normalized = _normalize_regime_value_for_options(candidate, name_set)
        if normalized:
            next_value = normalized
            break
    return options, next_value


@callback(
    Output("at-regime-def-modal", "opened", allow_duplicate=True),
    Output("at-regime-def-status-alert", "hide", allow_duplicate=True),
    Input("at-menu-add-regime", "n_clicks"),
    Input("at-regime-open-modal-btn", "n_clicks"),
    prevent_initial_call=True,
)
def at_open_regime_definition_modal(menu_clicks, tab_clicks):
    if not menu_clicks and not tab_clicks:
        raise PreventUpdate
    return True, True


@callback(
    Output("at-regime-def-db-available-store", "data", allow_duplicate=True),
    Output("at-regime-definitions-db-store", "data", allow_duplicate=True),
    Output("at-regime-def-loaded-store", "data", allow_duplicate=True),
    Output("at-regime-def-universe-series", "data"),
    Output("at-regime-def-single-series", "data"),
    Output("at-regime-def-db-available-note", "children"),
    Output("at-regime-def-save-db-btn", "disabled"),
    Output("at-regime-def-modal-draft-store", "data", allow_duplicate=True),
    Input("at-regime-def-modal", "opened"),
    State("dashmat-raw-data-store", "data"),
    State("at-series-select", "data"),
    State("at-regime-series-store", "data"),
    State("at-regime-def-modal-draft-store", "data"),
    State("at-benchmark-assignments-store", "data"),
    State("at-long-short-store", "data"),
    State("at-vol-scaling-assignments-store", "data"),
    State("at-vol-scaler-value-store", "data"),
    prevent_initial_call=True,
)
def at_load_regime_modal_data(
    opened,
    raw_data,
    selected_series,
    regime_series_store,
    current_draft,
    benchmark_assignments,
    long_short_assignments,
    vol_scaling_assignments,
    vol_scaler,
):
    if not opened:
        raise PreventUpdate

    draft = _ensure_regime_draft(current_draft)
    series_options, series_order, raw_series_order = _build_regime_series_options(
        raw_data,
        selected_series,
        regime_series_store,
        draft,
    )

    db_available = regime_tables_available(DB_ENGINE)
    db_definitions = load_regime_definitions(DB_ENGINE) if db_available else []
    note = "" if db_available else "Database regime tables are unavailable. Session regimes are still supported."

    if not isinstance(current_draft, dict):
        draft["ReturnBasis"] = "total"
        draft["BenchmarkAssignmentsJson"] = (
            dict(benchmark_assignments) if isinstance(benchmark_assignments, dict) else {}
        )
        draft["LongShortAssignmentsJson"] = (
            dict(long_short_assignments) if isinstance(long_short_assignments, dict) else {}
        )
        draft["VolScalingAssignmentsJson"] = (
            dict(vol_scaling_assignments) if isinstance(vol_scaling_assignments, dict) else {}
        )
        draft["VolScaler"] = float(pd.to_numeric(pd.Series([vol_scaler]), errors="coerce").iloc[0] or 0.0)
        if raw_series_order:
            draft["UniverseSeries"] = list(series_order[: min(8, len(series_order))])
            draft["SingleSeries"] = raw_series_order[0]

    valid_set = set(series_order)
    draft["UniverseSeries"] = [s for s in draft.get("UniverseSeries", []) if s in valid_set]
    if draft.get("SingleSeries") not in valid_set:
        draft["SingleSeries"] = series_order[0] if series_order else None
    if not draft.get("UniverseSeries") and series_order and int(draft.get("MethodType", 1)) in {1, 2}:
        draft["UniverseSeries"] = list(series_order[: min(8, len(series_order))])
    draft["sync_origin"] = "system"

    return (
        db_available,
        db_definitions,
        True,
        series_options,
        series_options,
        note,
        (not db_available),
        draft,
    )


@callback(
    Output("at-regime-def-select", "data"),
    Input("at-regime-definitions-db-store", "data"),
    Input("at-regime-definitions-local-store", "data"),
)
def at_update_regime_definition_select_options(db_definitions, local_definitions):
    entries = _regime_option_definitions(db_definitions, local_definitions)
    options = []
    for entry in entries:
        source_label = _source_badge(entry["source"])
        options.append(
            {
                "value": _regime_select_key(entry["source"], entry["name"]),
                "label": f"{source_label} {entry['name']}",
            }
        )
    return options


@callback(
    Output("at-regime-def-modal-draft-store", "data", allow_duplicate=True),
    Input("at-regime-def-select", "value"),
    State("at-regime-definitions-db-store", "data"),
    State("at-regime-definitions-local-store", "data"),
    State("at-regime-def-modal-draft-store", "data"),
    prevent_initial_call=True,
)
def at_load_selected_regime_definition(selected_key, db_definitions, local_definitions, current_draft):
    if not selected_key:
        raise PreventUpdate
    current = _ensure_regime_draft(current_draft)
    if current.get("selected_key") == selected_key:
        raise PreventUpdate

    source, name = _split_regime_select_key(selected_key)
    if source == "db":
        definition = _lookup_regime_definition(name, db_definitions, [])
        if not definition:
            raise PreventUpdate
        return _regime_definition_to_draft(definition, "db", selected_key=selected_key)
    if source == "session":
        definition = _lookup_regime_definition(name, [], local_definitions)
        if not definition:
            raise PreventUpdate
        return _regime_definition_to_draft(definition, "session", selected_key=selected_key)
    raise PreventUpdate


@callback(
    Output("at-regime-def-modal-draft-store", "data", allow_duplicate=True),
    Output("at-regime-def-select", "value", allow_duplicate=True),
    Output("at-regime-def-status-alert", "children", allow_duplicate=True),
    Output("at-regime-def-status-alert", "color", allow_duplicate=True),
    Output("at-regime-def-status-alert", "hide", allow_duplicate=True),
    Input("at-regime-def-new-btn", "n_clicks"),
    Input("at-regime-def-select", "value"),
    prevent_initial_call=True,
)
def at_reset_regime_definition_draft(new_clicks, selected_key):
    triggered = callback_context.triggered_id
    if triggered == "at-regime-def-new-btn" and new_clicks:
        return _default_regime_draft(), None, "New session regime draft started.", "blue", False
    if triggered == "at-regime-def-select" and not selected_key:
        return _default_regime_draft(), no_update, "New session regime draft started.", "blue", False
    raise PreventUpdate


@callback(
    Output("at-regime-def-universe-series", "data", allow_duplicate=True),
    Output("at-regime-def-single-series", "data", allow_duplicate=True),
    Input("at-regime-def-select", "value"),
    State("dashmat-raw-data-store", "data"),
    State("at-series-select", "data"),
    State("at-regime-series-store", "data"),
    State("at-regime-definitions-db-store", "data"),
    State("at-regime-definitions-local-store", "data"),
    State("at-regime-def-modal-draft-store", "data"),
    prevent_initial_call=True,
)
def at_refresh_regime_series_options_for_definition(
    selected_key,
    raw_data,
    selected_series,
    regime_series_store,
    db_definitions,
    local_definitions,
    current_draft,
):
    if not selected_key:
        raise PreventUpdate

    source, name = _split_regime_select_key(selected_key)
    definition_draft = _ensure_regime_draft(current_draft)
    if source == "db":
        definition = _lookup_regime_definition(name, db_definitions, [])
        if definition:
            definition_draft = _regime_definition_to_draft(definition, "db", selected_key=selected_key)
    elif source == "session":
        definition = _lookup_regime_definition(name, [], local_definitions)
        if definition:
            definition_draft = _regime_definition_to_draft(definition, "session", selected_key=selected_key)

    series_options, _series_order, _raw_series_order = _build_regime_series_options(
        raw_data,
        selected_series,
        regime_series_store,
        definition_draft,
    )
    return series_options, series_options


@callback(
    Output("at-regime-def-select", "value", allow_duplicate=True),
    Output("at-regime-def-name-input", "value"),
    Output("at-regime-def-description-input", "value"),
    Output("at-regime-def-method-type", "value"),
    Output("at-regime-def-num-regimes", "value"),
    Output("at-regime-def-min-observations", "value"),
    Output("at-regime-def-pca-standardize", "checked"),
    Output("at-regime-def-universe-series", "value"),
    Output("at-regime-def-single-series", "value"),
    Output("at-regime-def-vol-scaler", "value"),
    Input("at-regime-def-modal-draft-store", "data"),
    prevent_initial_call="initial_duplicate",
)
def at_sync_regime_definition_form(draft_data):
    draft = _ensure_regime_draft(draft_data)
    if draft.get("sync_origin") == "form":
        raise PreventUpdate
    method = int(draft.get("MethodType") or 1)
    num_regimes = int(draft.get("NumRegimes") or 3)
    max_regimes = 6 if method == 1 else 10
    return (
        draft.get("selected_key"),
        draft.get("RegimeName") or "",
        draft.get("Description") or "",
        str(method),
        max(2, min(num_regimes, max_regimes)),
        int(draft.get("MinObservations") or 60),
        bool(draft.get("PcaStandardize", True)),
        draft.get("UniverseSeries") or [],
        draft.get("SingleSeries"),
        float(draft.get("VolScaler") or 0.0),
    )


@callback(
    Output("at-regime-def-modal-draft-store", "data", allow_duplicate=True),
    Input("at-regime-def-name-input", "value"),
    Input("at-regime-def-description-input", "value"),
    Input("at-regime-def-method-type", "value"),
    Input("at-regime-def-num-regimes", "value"),
    Input("at-regime-def-min-observations", "value"),
    Input("at-regime-def-pca-standardize", "checked"),
    Input("at-regime-def-universe-series", "value"),
    Input("at-regime-def-single-series", "value"),
    Input("at-regime-def-vol-scaler", "value"),
    State("at-regime-def-modal-draft-store", "data"),
    prevent_initial_call=True,
)
def at_update_regime_definition_draft_from_form(
    regime_name,
    description,
    method_type,
    num_regimes,
    min_observations,
    pca_standardize,
    universe_series,
    single_series,
    vol_scaler_value,
    draft_data,
):
    draft = _ensure_regime_draft(draft_data)
    next_draft = dict(draft)
    next_draft["sync_origin"] = "form"
    next_draft["RegimeName"] = str(regime_name or "").strip()
    next_draft["Description"] = "" if description is None else str(description)
    method_num = int(pd.to_numeric(pd.Series([method_type]), errors="coerce").iloc[0] or 1)
    method_num = 3 if method_num == 3 else (2 if method_num == 2 else 1)
    next_draft["MethodType"] = method_num
    next_draft["ReturnBasis"] = "total"
    max_regimes = 6 if method_num == 1 else 10
    parsed_num_regimes = int(pd.to_numeric(pd.Series([num_regimes]), errors="coerce").iloc[0] or 3)
    next_draft["NumRegimes"] = max(2, min(parsed_num_regimes, max_regimes))
    parsed_min_obs = int(pd.to_numeric(pd.Series([min_observations]), errors="coerce").iloc[0] or 60)
    next_draft["MinObservations"] = max(20, parsed_min_obs)
    next_draft["PcaStandardize"] = bool(pca_standardize)
    next_draft["UniverseSeries"] = [str(v) for v in (universe_series or []) if str(v).strip()]
    single = str(single_series or "").strip()
    next_draft["SingleSeries"] = single or None
    next_draft["VolScaler"] = max(
        0.0,
        float(pd.to_numeric(pd.Series([vol_scaler_value]), errors="coerce").iloc[0] or 0.0),
    )
    if next_draft == draft:
        raise PreventUpdate
    return next_draft


@callback(
    Output("at-regime-def-preview-lines", "children"),
    Output("at-regime-series-store", "data", allow_duplicate=True),
    Input("at-regime-def-modal", "opened"),
    Input("at-regime-def-modal-draft-store", "data"),
    Input("dashmat-raw-data-store", "data"),
    Input("at-periodicity-select", "value"),
    Input("at-date-range-store", "data"),
    State("at-regime-series-store", "data"),
    prevent_initial_call=True,
)
def at_update_regime_definition_preview(
    opened,
    draft_data,
    raw_data,
    periodicity,
    date_range,
    regime_series_store,
):
    if not opened:
        raise PreventUpdate

    payload = _regime_draft_to_definition_payload(draft_data or {})
    normalized, error = validate_regime_definition_payload(payload)
    if error or not normalized:
        return "Define a valid regime to preview assignments.", no_update

    required_series = regime_required_series(normalized)
    combined_raw_data, next_regime_series_store, _resolved, unresolved = resolve_regime_source_data(
        raw_data=raw_data,
        regime_series_store=regime_series_store,
        required_series=required_series,
        db_engine=DB_ENGINE,
        mrd_engine=MRD_ENGINE,
    )
    if not combined_raw_data:
        return "Upload or load return data to preview regime assignments.", next_regime_series_store

    states, diagnostics = compute_regime_assignments(
        raw_data=combined_raw_data,
        periodicity=periodicity or "daily",
        definition=normalized,
        date_range=date_range,
    )
    if states.empty:
        warning = str((diagnostics or {}).get("warning") or "No assignments were produced for the current inputs.")
        if unresolved:
            warning = f"{warning} Missing source series: {', '.join(unresolved)}."
        return f"No assignments returned.\nReason: {warning}", next_regime_series_store

    timeline = build_regime_timeline_frame(states)
    counts = states.value_counts().sort_index()
    lines = [
        f"Regime: {normalized.get('RegimeName')}",
        f"Method: {normalized.get('MethodType')}",
        f"Observations: {len(states)}",
        f"Counts: {', '.join([f'R{int(idx)}={int(val)}' for idx, val in counts.items()])}",
        "Date:Regime",
    ]
    for _, row in timeline.head(8).iterrows():
        lines.append(f"{pd.Timestamp(row['Date']).strftime('%Y-%m-%d')}:{int(row['Regime'])}")
    warning = (diagnostics or {}).get("warning")
    if warning:
        lines.append(f"Warning: {warning}")
    if unresolved:
        lines.append(f"Missing source series: {', '.join(unresolved)}")
    return "\n".join(lines), next_regime_series_store


@callback(
    Output("at-regime-definitions-local-store", "data", allow_duplicate=True),
    Output("at-regime-definitions-db-store", "data", allow_duplicate=True),
    Output("at-regime-def-modal-draft-store", "data", allow_duplicate=True),
    Output("at-regime-def-select", "value", allow_duplicate=True),
    Output("at-regime-definition-select", "value", allow_duplicate=True),
    Output("at-regime-def-modal", "opened", allow_duplicate=True),
    Output("at-regime-def-status-alert", "children", allow_duplicate=True),
    Output("at-regime-def-status-alert", "color", allow_duplicate=True),
    Output("at-regime-def-status-alert", "hide", allow_duplicate=True),
    Input("at-regime-def-save-local-btn", "n_clicks"),
    Input("at-regime-def-save-db-btn", "n_clicks"),
    Input("at-regime-def-delete-btn", "n_clicks"),
    Input("at-regime-def-use-btn", "n_clicks"),
    Input("at-regime-def-close-btn", "n_clicks"),
    State("at-regime-def-modal-draft-store", "data"),
    State("at-regime-definitions-local-store", "data"),
    State("at-regime-definitions-db-store", "data"),
    State("at-regime-def-db-available-store", "data"),
    State("userinfo", "data"),
    prevent_initial_call=True,
)
def at_manage_regime_definitions(
    save_local_clicks,
    save_db_clicks,
    delete_clicks,
    use_clicks,
    close_clicks,
    draft_data,
    local_definitions,
    db_definitions,
    db_available,
    userinfo,
):
    triggered = callback_context.triggered_id
    n_no = no_update
    draft = _ensure_regime_draft(draft_data)
    local_list = [dict(item) for item in (local_definitions or []) if isinstance(item, dict)]
    update_by = _factor_user_label(userinfo)

    if triggered == "at-regime-def-close-btn":
        return n_no, n_no, n_no, n_no, n_no, False, n_no, n_no, True

    payload = _regime_draft_to_definition_payload(draft)
    normalized, error = validate_regime_definition_payload(payload)
    if triggered in {"at-regime-def-save-local-btn", "at-regime-def-save-db-btn", "at-regime-def-use-btn"}:
        if error or not normalized:
            return n_no, n_no, n_no, n_no, n_no, n_no, error or "Invalid regime definition.", "red", False
    else:
        normalized = normalized or {}

    if triggered == "at-regime-def-save-local-btn":
        now_str = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        saved_item = dict(normalized)
        saved_item["source"] = "session"
        saved_item["UPDATE_DATE"] = now_str
        saved_item["UPDATE_BY"] = update_by

        target_name = str(saved_item.get("RegimeName") or "").strip().lower()
        updated_local = [item for item in local_list if str(item.get("RegimeName", "")).strip().lower() != target_name]
        updated_local.append(saved_item)
        updated_local.sort(key=lambda item: str(item.get("RegimeName", "")).lower())

        next_draft = _regime_definition_to_draft(saved_item, "session")
        return (
            updated_local,
            n_no,
            next_draft,
            next_draft.get("selected_key"),
            n_no,
            n_no,
            f"Saved session regime `{saved_item['RegimeName']}`.",
            "green",
            False,
        )

    if triggered == "at-regime-def-save-db-btn":
        if not db_available:
            return n_no, n_no, n_no, n_no, n_no, n_no, "Database regime tables are unavailable.", "orange", False

        original_name = draft.get("original_name") if draft.get("source") == "db" else None
        expected_update = draft.get("selected_update_date") if draft.get("source") == "db" else None
        success, message, saved_row = save_regime_definition(
            DB_ENGINE,
            payload,
            update_by=update_by,
            original_name=original_name,
            expected_update_date=expected_update,
        )
        if not success or not saved_row:
            return n_no, n_no, n_no, n_no, n_no, n_no, message, "red", False

        updated_db = load_regime_definitions(DB_ENGINE)
        saved_name = str(saved_row.get("RegimeName", "")).strip().lower()
        updated_local = [
            item for item in local_list if str(item.get("RegimeName", "")).strip().lower() != saved_name
        ]
        next_draft = _regime_definition_to_draft(saved_row, "db")
        return (
            updated_local,
            updated_db,
            next_draft,
            next_draft.get("selected_key"),
            f"def::{saved_row.get('RegimeName')}",
            n_no,
            message,
            "green",
            False,
        )

    if triggered == "at-regime-def-delete-btn":
        name = str(draft.get("RegimeName", "")).strip()
        if not name:
            return n_no, n_no, n_no, n_no, n_no, n_no, "Select or enter a regime name to delete.", "orange", False

        if draft.get("source") == "db":
            if not db_available:
                return n_no, n_no, n_no, n_no, n_no, n_no, "Database regime tables are unavailable.", "orange", False
            target_name = str(draft.get("original_name") or name).strip()
            success, message = delete_regime_definition(
                DB_ENGINE,
                target_name,
                expected_update_date=draft.get("selected_update_date"),
            )
            if not success:
                return n_no, n_no, n_no, n_no, n_no, n_no, message, "red", False
            updated_db = load_regime_definitions(DB_ENGINE)
            return n_no, updated_db, _default_regime_draft(), None, n_no, n_no, message, "green", False

        target_name = name.lower()
        updated_local = [item for item in local_list if str(item.get("RegimeName", "")).strip().lower() != target_name]
        return (
            updated_local,
            n_no,
            _default_regime_draft(),
            None,
            n_no,
            n_no,
            f"Deleted session regime `{name}`.",
            "green",
            False,
        )

    if triggered == "at-regime-def-use-btn":
        regime_name = str(normalized.get("RegimeName", "")).strip()
        if not regime_name:
            return n_no, n_no, n_no, n_no, n_no, n_no, "Regime name is required.", "red", False

        draft_mode = str(draft.get("DraftMode") or "").strip().lower()
        if draft_mode == "db":
            baseline_name = str(draft.get("original_name") or regime_name).strip()
            baseline = _lookup_regime_definition(baseline_name, db_definitions, [])
            if not baseline:
                return (
                    n_no,
                    n_no,
                    _regime_definition_to_draft(
                        {
                            **normalized,
                            "UPDATE_DATE": draft.get("selected_update_date"),
                            "UPDATE_BY": draft.get("UPDATE_BY"),
                        },
                        "db",
                        selected_key=_regime_select_key("db", baseline_name or regime_name),
                    ),
                    _regime_select_key("db", baseline_name or regime_name),
                    f"def::{baseline_name or regime_name}",
                    False,
                    "Database regime selected for analysis.",
                    "green",
                    False,
                )

            baseline_name_actual = str(baseline.get("RegimeName", "")).strip() or baseline_name
            unchanged_db = (
                str(regime_name).lower() == str(baseline_name_actual).lower()
                and _regime_definition_signature(normalized) == _regime_definition_signature(baseline)
            )
            if unchanged_db:
                next_draft = _regime_definition_to_draft(
                    baseline,
                    "db",
                    selected_key=_regime_select_key("db", baseline_name_actual),
                )
                return (
                    n_no,
                    n_no,
                    next_draft,
                    next_draft.get("selected_key"),
                    f"def::{baseline_name_actual}",
                    False,
                    "Database regime selected for analysis.",
                    "green",
                    False,
                )

            if _regime_db_name_exists(regime_name, db_definitions):
                return (
                    n_no,
                    n_no,
                    n_no,
                    n_no,
                    n_no,
                    n_no,
                    "Rename the regime to create a session copy; that name already exists in Database definitions.",
                    "orange",
                    False,
                )

        now_str = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        session_item = {
            **normalized,
            "source": "session",
            "UPDATE_DATE": now_str,
            "UPDATE_BY": update_by,
        }
        key_name = regime_name.lower()
        next_local = [item for item in local_list if str(item.get("RegimeName", "")).strip().lower() != key_name]
        next_local.append(session_item)
        next_local.sort(key=lambda item: str(item.get("RegimeName", "")).lower())
        next_draft = _regime_definition_to_draft(session_item, "session")

        return (
            next_local,
            n_no,
            next_draft,
            next_draft.get("selected_key"),
            f"def::{regime_name}",
            False,
            "Session regime selected for analysis.",
            "green",
            False,
        )

    raise PreventUpdate


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
    Output("dashmat-db-import-provenance-store", "data", allow_duplicate=True),
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
    State("dashmat-db-import-provenance-store", "data"),
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
    current_provenance,
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
            True, n_no, n_no,
        )

    try:
        if existing_data:
            existing_cols = set(_raw_df(existing_data).columns)
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
                    True, n_no, n_no,
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
            existing_df = _raw_df(existing_data)
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

        alert_msg = (
            f"Loaded {len(new_df.columns)} series with {len(new_df)} rows from database"
        )
        if daily_transition_notes:
            alert_msg = f"{alert_msg}. Series become daily on: {'; '.join(daily_transition_notes)}"
        alert_color = "orange" if daily_transition_notes else "green"
        alert_hide = False
        new_first_load = True

        updated_provenance = add_db_import_provenance_entry(
            current_provenance,
            loader_type="cma_bench",
            loader_args={"selected_benches": list(selected_benches or [])},
            emitted_series=list(new_df.columns),
            primary_series=list(new_df.columns)[0] if list(new_df.columns) else None,
        )
        return (
            build_raw_data_store_payload(merged_df),
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
            updated_provenance,
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
            True, n_no, n_no,
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
    Output("dashmat-db-import-provenance-store", "data", allow_duplicate=True),
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
    State("dashmat-db-import-provenance-store", "data"),
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
    current_provenance,
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
            n_no,
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
            existing_cols = set(_raw_df(existing_data).columns)
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
                    n_no,
                )

        new_periodicity = load_result.periodicity
        if existing_data is not None:
            existing_df = _raw_df(existing_data)
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

        updated_provenance = add_db_import_provenance_entry(
            current_provenance,
            loader_type=f"raw_{mode_key}",
            loader_args={"rows": rows},
            emitted_series=list(new_df.columns),
            primary_series=list(new_df.columns)[0] if list(new_df.columns) else None,
        )
        return (
            build_raw_data_store_payload(merged_df),
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
            updated_provenance,
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
    Output("at-underlying-add-modal", "opened", allow_duplicate=True),
    Output("at-underlying-add-rows-store", "data", allow_duplicate=True),
    Output("at-underlying-add-grid", "rowData", allow_duplicate=True),
    Output("at-underlying-add-error-alert", "children", allow_duplicate=True),
    Output("at-underlying-add-error-alert", "hide", allow_duplicate=True),
    Output("dashmat-db-import-provenance-store", "data", allow_duplicate=True),
    Input("at-underlying-add-ok-button", "n_clicks"),
    State("at-underlying-add-rows-store", "data"),
    State("dashmat-raw-data-store", "data"),
    State("dashmat-original-periodicity-store", "data"),
    State("at-series-select", "data"),
    State("at-benchmark-assignments-store", "data"),
    State("at-long-short-store", "data"),
    State("at-series-order-store", "data"),
    State("at-first-load-store", "data"),
    State("at-vol-scaling-assignments-store", "data"),
    State("dashmat-db-import-provenance-store", "data"),
    prevent_initial_call=True,
)
def at_add_underlying_categories_from_database(
    n_clicks,
    staged_rows,
    existing_data,
    existing_periodicity,
    current_selection,
    current_bench,
    current_ls,
    current_order,
    first_load,
    current_vol_scaling,
    current_provenance,
):
    if not n_clicks:
        raise PreventUpdate

    n_no = no_update
    rows = [dict(row) for row in (staged_rows or []) if isinstance(row, dict)]
    if not rows:
        return (
            n_no, n_no, n_no, n_no, n_no,
            n_no,
            "Stage at least one underlying category row before importing.",
            "orange",
            False,
            n_no, n_no, n_no, n_no, n_no,
            n_no, n_no, n_no,
            True,
            rows,
            rows,
            "Stage at least one underlying category row before importing.",
            False,
            n_no,
        )

    try:
        load_result = load_underlying_category_series(DB_ENGINE, rows)
        new_df = load_result.returns_df
        if new_df.empty:
            raise ValueError("No rows returned for staged underlying category requests.")

        if existing_data:
            existing_cols = set(_raw_df(existing_data).columns)
            duplicates = [series_name for series_name in new_df.columns if series_name in existing_cols]
            if duplicates:
                duplicate_text = f"Cannot add duplicate series: {', '.join(duplicates)}"
                return (
                    n_no, n_no, n_no, n_no, n_no,
                    n_no,
                    duplicate_text,
                    "red",
                    False,
                    n_no, n_no, n_no, n_no, n_no,
                    n_no, n_no, n_no,
                    True,
                    rows,
                    rows,
                    duplicate_text,
                    False,
                    n_no,
                )

        merge_result = _shared_merge_uploaded_with_existing(existing_data, existing_periodicity, new_df)
        merged_df = merge_result.merged_df
        combined_periodicity = merge_result.combined_periodicity
        periodicity_options = merge_result.periodicity_options
        default_periodicity = merge_result.default_periodicity
        imported_df = merge_result.imported_df

        new_series = [col for col in imported_df.columns if col not in (current_selection or [])]
        updated_selection = (current_selection or []) + new_series

        updated_provenance = add_db_import_provenance_entry(
            current_provenance,
            loader_type="underlying_category",
            loader_args={"rows": rows},
            emitted_series=list(imported_df.columns),
            primary_series=list(imported_df.columns)[0] if list(imported_df.columns) else None,
        )
        return (
            build_raw_data_store_payload(merged_df),
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
            updated_provenance,
        )
    except Exception as exc:
        error_text = f"Error loading underlying category series: {exc}"
        return (
            n_no, n_no, n_no, n_no, n_no,
            n_no,
            error_text,
            "red",
            False,
            n_no, n_no, n_no, n_no, n_no,
            n_no, n_no, n_no,
            True,
            rows,
            rows,
            error_text,
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
    Output("dashmat-db-import-provenance-store", "data", allow_duplicate=True),
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
    State("dashmat-db-import-provenance-store", "data"),
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
    current_provenance,
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
            n_no,
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
            existing_cols = set(_raw_df(existing_data).columns)
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
                    n_no,
                )

        new_periodicity = load_result.periodicity or "monthly"
        if existing_data is not None:
            existing_df = _raw_df(existing_data)
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

        updated_provenance = add_db_import_provenance_entry(
            current_provenance,
            loader_type=f"portfolio_{mode}",
            loader_args={"rows": rows},
            emitted_series=list(new_df.columns),
            primary_series=list(new_df.columns)[0] if list(new_df.columns) else None,
        )
        return (
            build_raw_data_store_payload(merged_df),
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
            updated_provenance,
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
            n_no,
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
            build_raw_data_store_payload(merged_df),
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
            True, # Keep blocker until series-selection grid renders
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
            build_raw_data_store_payload(merged_df),
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
            True,  # Keep blocker until series-selection grid renders
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
    Output("at-ui-blocker-store", "data", allow_duplicate=True),
    Input("dashmat-raw-data-store", "data"),
    Input("dashmat-raw-data-meta-store", "data"),
    Input("at-temp-series-select", "data"),
    Input("at-temp-series-order-store", "data"),
    Input("at-temp-deleted-series-store", "data"),
    Input("at-temp-benchmark-assignments-store", "data"),
    Input("at-temp-long-short-store", "data"),
    Input("at-temp-vol-scaling-assignments-store", "data"),
    prevent_initial_call="initial_duplicate",
)
def update_series_selectors(
    raw_data,
    raw_meta,
    selected_series,
    series_order,
    deleted_series,
    current_assignments,
    long_short_assignments,
    vol_scaling_assignments,
):
    """Render Select Series as a single AG Grid with in-grid controls."""
    if raw_data is None:
        return [dmc.Text("Upload data to select series", size="sm", c="dimmed")], [], False

    all_series = list((raw_meta or {}).get("columns") or [])
    if not all_series:
        df = _raw_df(raw_data)
        all_series = list(df.columns)
    if not all_series:
        return [dmc.Text("Upload data to select series", size="sm", c="dimmed")], [], False

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
                "__row_key": series,
                "Selected": series in selected_set and series not in deleted_set,
                "Series": series,
                "Benchmark": benchmark_value,
                "LongShort": bool(long_short_assignments.get(series, False)),
                "ScaleVol": bool(vol_scaling_assignments.get(series, True)),
                "Delete": series in deleted_set,
            }
        )

    grid = dag.AgGrid(
        id="at-series-selection-grid",
        className="ag-theme-alpine dashmat-series-modal-grid",
        getRowId="params.data.__row_key",
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
                "field": "Selected",
                "headerName": "Use",
                "editable": True,
                "cellRenderer": "agCheckboxCellRenderer",
                "cellEditor": "agCheckboxCellEditor",
                "width": 72,
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
                "cellEditor": "agSelectCellEditor",
                "cellEditorParams": {"values": benchmark_values},
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
            "suppressMovableColumns": True,
            "rowDragManaged": True,
            "animateRows": False,
            "singleClickEdit": True,
            "stopEditingWhenCellsLoseFocus": True,
            "suppressExcelExport": True,
            "suppressCsvExport": True,
        },
        enableEnterpriseModules=True,
        licenseKey=AG_GRID_LICENSE_KEY,
    )
    return [grid], series_order, no_update


clientside_callback(
    ClientsideFunction(namespace="dashmat_callbacks", function_name="captureAnalyticsSeriesSnapshot"),
    Output("at-series-grid-snapshot-store", "data"),
    Input("at-modal-ok-button", "n_clicks"),
    State("at-series-selection-modal", "opened"),
    prevent_initial_call=True,
)


@callback(
    Output("at-series-select", "data", allow_duplicate=True),
    Output("at-benchmark-assignments-store", "data", allow_duplicate=True),
    Output("at-long-short-store", "data", allow_duplicate=True),
    Output("at-series-order-store", "data", allow_duplicate=True),
    Output("at-series-selection-modal", "opened", allow_duplicate=True),
    Output("at-series-select-value-store", "data", allow_duplicate=True),
    Output("dashmat-raw-data-store", "data", allow_duplicate=True),
    Output("at-vol-scaling-assignments-store", "data", allow_duplicate=True),
    Output("dashmat-db-import-provenance-store", "data", allow_duplicate=True),
    Input("at-series-grid-snapshot-store", "data"),
    State("dashmat-raw-data-store", "data"),
    State("at-series-select", "data"),
    State("at-benchmark-assignments-store", "data"),
    State("at-long-short-store", "data"),
    State("at-series-order-store", "data"),
    State("at-vol-scaling-assignments-store", "data"),
    State("dashmat-db-import-provenance-store", "data"),
    prevent_initial_call=True,
)
def on_modal_ok(
    snapshot_data,
    raw_data,
    current_select,
    current_bench,
    current_ls,
    current_order,
    current_vol_scaling,
    current_provenance,
):
    rows = []
    if isinstance(snapshot_data, dict) and isinstance(snapshot_data.get("rows"), list):
        rows = [dict(row) for row in snapshot_data["rows"] if isinstance(row, dict)]
    if not rows or not raw_data:
        raise PreventUpdate

    df = _raw_df(raw_data)
    existing_cols = list(df.columns)
    existing_set = set(existing_cols)

    active_rows = []
    final_names = []
    rename_map = {}
    for row in rows:
        original = str(row.get("__row_key") or "").strip()
        if not original or original not in existing_set:
            continue
        final_name = str(row.get("Series") or "").strip()
        if not final_name or final_name in final_names:
            raise PreventUpdate
        final_names.append(final_name)
        if final_name != original:
            rename_map[original] = final_name
        active_rows.append((original, final_name, row))

    if not active_rows:
        raise PreventUpdate

    if rename_map:
        df = df.rename(columns=rename_map)

    deleted_names = [
        final_name
        for original, final_name, row in active_rows
        if bool(row.get("Delete", False))
    ]
    if deleted_names:
        df = df.drop(columns=[name for name in deleted_names if name in df.columns], errors="ignore")

    remaining_cols = set(df.columns)
    next_select = []
    next_bench = {}
    next_ls = {}
    next_order = []
    next_vol_scaling = {}
    for original, final_name, row in active_rows:
        if final_name not in remaining_cols:
            continue
        next_order.append(final_name)
        if bool(row.get("Selected", False)):
            next_select.append(final_name)
        benchmark_value = str(row.get("Benchmark") or "None").strip() or "None"
        benchmark_value = rename_map.get(benchmark_value, benchmark_value)
        if benchmark_value != "None" and benchmark_value not in remaining_cols:
            benchmark_value = "None"
        next_bench[final_name] = benchmark_value
        next_ls[final_name] = bool(row.get("LongShort", False))
        next_vol_scaling[final_name] = bool(row.get("ScaleVol", True))

    current_select = list(current_select or [])
    current_bench = dict(current_bench or {})
    current_ls = dict(current_ls or {})
    current_order = list(current_order or [])
    current_vol_scaling = dict(current_vol_scaling or {})

    next_series_select = no_update if next_select == current_select else next_select
    next_bench_output = no_update if next_bench == current_bench else next_bench
    next_ls_output = no_update if next_ls == current_ls else next_ls
    next_order_output = no_update if next_order == current_order else next_order
    next_series_value = no_update if next_select == current_select else next_select
    next_vol_output = no_update if next_vol_scaling == current_vol_scaling else next_vol_scaling
    updated_raw_data = no_update if list(df.columns) == existing_cols else build_raw_data_store_payload(df)
    updated_provenance = rename_db_import_provenance_series(current_provenance, rename_map) if rename_map else current_provenance
    updated_provenance = remove_db_import_provenance_series(updated_provenance, deleted_names) if deleted_names else updated_provenance
    updated_provenance = prune_db_import_provenance(updated_provenance, list(df.columns))
    next_provenance_output = no_update if updated_provenance == (current_provenance or {}) else updated_provenance

    return (
        next_series_select,
        next_bench_output,
        next_ls_output,
        next_order_output,
        False,
        next_series_value,
        updated_raw_data,
        next_vol_output,
        next_provenance_output,
    )


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
    Output("at-range-candidates-store", "data"),
    Input("dashmat-raw-data-store", "data"),
    Input("at-periodicity-select", "value"),
    Input("at-series-select", "data"),
    prevent_initial_call="initial_duplicate",
)
def update_at_range_candidates(raw_data, periodicity, selected_series):
    return compute_date_range_candidates(
        _dataset_key(raw_data),
        periodicity or "daily",
        tuple(selected_series or ()),
    )


@callback(
    Output("at-common-daily-candidates-store", "data"),
    Input("dashmat-raw-data-store", "data"),
    Input("at-series-select", "data"),
    prevent_initial_call="initial_duplicate",
)
def update_at_common_daily_candidates(raw_data, selected_series):
    return compute_common_daily_candidates(
        _dataset_key(raw_data),
        tuple(selected_series or ()),
    )


@callback(
    Output("at-start-date-picker", "value"),
    Output("at-end-date-picker", "value"),
    Output("at-date-picker-wrapper", "style"),
    Output("at-common-range-button", "disabled"),
    Output("at-maximum-range-button", "disabled"),
    Output("at-date-range-store", "data", allow_duplicate=True),
    Output("at-state-ready-store", "data", allow_duplicate=True),
    Input("at-range-candidates-store", "data"),
    State("at-date-range-store", "data"),
    State("at-start-date-picker", "value"),
    State("at-end-date-picker", "value"),
    State("at-state-ready-store", "data"),
    prevent_initial_call="initial_duplicate",
)
def initialize_date_range(
    candidates,
    stored_range,
    current_start_date,
    current_end_date,
    current_state_ready,
):
    """Initialize date range to maximum range when data is loaded."""
    disabled_style = {"display": "flex", "opacity": 0.5, "pointerEvents": "none", "alignItems": "flex-start"}
    enabled_style = {"display": "flex", "alignItems": "flex-start"}

    if not isinstance(candidates, dict) or not candidates.get("available_series"):
        ready_output = no_update if current_state_ready is False else False
        return None, None, disabled_style, True, True, None, ready_output

    try:
        start_date, end_date = resolve_initial_range(candidates, stored_range)
        if not start_date or not end_date:
            ready_output = no_update if current_state_ready is False else False
            return None, None, disabled_style, True, True, None, ready_output
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
        ready_output = no_update if current_state_ready is True else True
        return (
            start_output,
            end_output,
            enabled_style,
            False,
            False,
            range_output,
            ready_output,
        )

    except Exception:
        ready_output = no_update if current_state_ready is False else False
        return None, None, disabled_style, True, True, None, ready_output


clientside_callback(
    ClientsideFunction(namespace="dashmat_callbacks", function_name="commonDailyButtonDisabled"),
    Output("at-common-daily-button", "disabled"),
    Input("at-range-candidates-store", "data"),
    Input("at-common-daily-candidates-store", "data"),
    Input("at-periodicity-select", "data"),
    prevent_initial_call=False,
)


@callback(
    Output("at-start-date-picker", "value", allow_duplicate=True),
    Output("at-end-date-picker", "value", allow_duplicate=True),
    Output("at-date-range-store", "data"),
    Output("at-periodicity-select", "value", allow_duplicate=True),
    Output("at-periodicity-value-store", "data", allow_duplicate=True),
    Input("at-common-range-button", "n_clicks"),
    Input("at-common-daily-button", "n_clicks"),
    Input("at-maximum-range-button", "n_clicks"),
    State("at-range-candidates-store", "data"),
    State("at-common-daily-candidates-store", "data"),
    prevent_initial_call=True,
)
def update_date_range_buttons(common_clicks, common_daily_clicks, max_clicks, candidates, common_daily_candidates):
    """Update date range based on button clicks."""
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
    Input("at-main-tabs", "value"),
    Input("at-initial-tab-render-ready-store", "data"),
    prevent_initial_call=True,
)
def update_grid(raw_data=None, periodicity=None, selected_series=None, returns_type="total", benchmark_assignments=None, long_short_assignments=None, date_range=None, state_ready=False, vol_scaler=0, vol_scaling_assignments=None, active_tab="returns", initial_tab_ready=True):
    """Update the AG Grid based on selections (optimized with caching)."""
    if active_tab != "returns" or not initial_tab_ready or not state_ready or not _has_complete_date_range(date_range):
        raise PreventUpdate

    if raw_data is None or not selected_series:
        return [], []

    try:
        with timed_block("analyticstool.render_returns_grid", series_count=len(selected_series)):
            display_df = _compute_selected_returns(
                raw_data,
                periodicity,
                selected_series,
                returns_type,
                benchmark_assignments,
                long_short_assignments,
                date_range,
                vol_scaler,
                vol_scaling_assignments,
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
    Input("at-initial-tab-render-ready-store", "data"),
)
def control_statistics_loading_display(active_tab, state_ready, statistics_loaded, initial_tab_ready=True):
    if active_tab == "statistics" and (not initial_tab_ready or not state_ready or not statistics_loaded):
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
    Input("at-use-risk-free-store", "data"),
    Input("dashmat-saved-series-cache-store", "data"),
    prevent_initial_call=True,
)
def update_rolling_grid(active_tab, chart_checked, raw_data, periodicity, selected_series, rolling_window, rolling_return_type, rolling_metric, benchmark_assignments, long_short_assignments, date_range, state_ready, vol_scaler, vol_scaling_assignments, use_risk_free, saved_series_store):
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
            _dataset_key(raw_data),
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
            _mapping_payload(vol_scaling_assignments),
            _risk_free_json_from_store(saved_series_store),
            bool(use_risk_free),
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
    Input("at-use-risk-free-store", "data"),
    Input("dashmat-saved-series-cache-store", "data"),
    State("global-color-scheme-toggle", "computedColorScheme"),
    prevent_initial_call=True,
)
def update_rolling_chart(active_tab, chart_checked, raw_data, periodicity, selected_series, rolling_window, rolling_return_type, rolling_metric, benchmark_assignments, long_short_assignments, date_range, state_ready, vol_scaler, vol_scaling_assignments, use_risk_free, saved_series_store, theme):
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
            _dataset_key(raw_data),
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
            _mapping_payload(vol_scaling_assignments),
            _risk_free_json_from_store(saved_series_store),
            bool(use_risk_free),
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
                _dataset_key(raw_data),
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
                _dataset_key(raw_data),
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
    Input("at-use-risk-free-store", "data"),
    Input("dashmat-saved-series-cache-store", "data"),
    Input("at-main-tabs", "value"),
    Input("at-initial-tab-render-ready-store", "data"),
    prevent_initial_call=True,
)
def update_statistics(raw_data=None, periodicity=None, selected_series=None, benchmark_assignments=None, long_short_assignments=None, date_range=None, state_ready=False, vol_scaler=0, vol_scaling_assignments=None, use_risk_free=True, saved_series_store=None, active_tab="statistics", initial_tab_ready=True):
    """Update the Statistics grid with transposed data (optimized with caching)."""
    if active_tab != "statistics" or not initial_tab_ready or not state_ready or not _has_complete_date_range(date_range):
        raise PreventUpdate

    if raw_data is None or not selected_series:
        return [], [], True

    try:
        with timed_block("analyticstool.render_statistics_grid", series_count=len(selected_series)):
            # Use cached function to avoid repeated computation
            stats = calculate_statistics_cached(
                _dataset_key(raw_data),
                periodicity or "daily",
                tuple(selected_series),
                _mapping_payload(benchmark_assignments),
                _mapping_payload(long_short_assignments),
                _date_range_payload(date_range),
                vol_scaler or 0,
                _mapping_payload(vol_scaling_assignments),
                _risk_free_json_from_store(saved_series_store),
                _spx_json_from_store(saved_series_store),
                bool(use_risk_free),
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


clientside_callback(
    """
    function(checked, shrinkage) {
        var expWeighted = !!checked;
        var useTarget = !expWeighted && (shrinkage === "ledoit_wolf");
        return [!expWeighted, expWeighted, !useTarget];
    }
    """,
    Output("at-correlation-halflife-input", "disabled"),
    Output("at-correlation-shrinkage-select", "disabled"),
    Output("at-correlation-shrinkage-target-select", "disabled"),
    Input("at-correlation-exp-wt-switch", "checked"),
    Input("at-correlation-shrinkage-select", "value"),
    prevent_initial_call=False,
)


clientside_callback(
    "function(view) { return view !== 'correlogram'; }",
    Output("at-correlogram-block-width", "disabled"),
    Input("at-correlation-view-switch", "value"),
    prevent_initial_call=False,
)


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
    Input("at-range-candidates-store", "data"),
    Input("at-correlation-view-switch", "value"),
    Input("at-correlation-exp-wt-switch", "checked"),
    Input("at-correlation-halflife-input", "value"),
    Input("at-correlation-shrinkage-select", "value"),
    Input("at-correlation-shrinkage-target-select", "value"),
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
    range_candidates,
    correlation_view,
    exp_weighted,
    decay_value,
    shrinkage,
    shrinkage_target,
    block_width,
    current_target_key,
):
    if active_tab != "correlogram":
        return no_update
    if not state_ready:
        return no_update

    effective_date_range = date_range
    try:
        start_date, end_date = resolve_initial_range(range_candidates or {}, date_range)
        if start_date and end_date:
            effective_date_range = {"start": start_date, "end": end_date}
    except Exception:
        effective_date_range = date_range

    if not _has_complete_date_range(effective_date_range):
        return no_update

    use_weighted_matrix = bool(
        exp_weighted and correlation_view in {"correlation", "covariance"}
    )
    if correlation_view in {"correlation", "covariance"}:
        effective_shrinkage, effective_target = resolve_cov_shrinkage_spec(
            shrinkage,
            shrinkage_target,
            exp_weighted=use_weighted_matrix,
        )
    else:
        effective_shrinkage, effective_target = "none", "scaled_identity"
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
        use_weighted_matrix,
        decay_value if use_weighted_matrix else 63.0,
        effective_shrinkage,
        effective_target if effective_shrinkage == "ledoit_wolf" else "scaled_identity",
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
    State("at-correlation-exp-wt-switch", "checked"),
    State("at-correlation-halflife-input", "value"),
    State("at-correlation-shrinkage-select", "value"),
    State("at-correlation-shrinkage-target-select", "value"),
    State("at-correlation-view-switch", "value"),
    State("at-correlogram-block-width", "value"),
    State("global-color-scheme-toggle", "computedColorScheme"),
    prevent_initial_call=True,
)
def update_correlogram(target_key, active_tab, raw_data, periodicity, selected_series, returns_type, benchmark_assignments, long_short_assignments, date_range, state_ready, vol_scaler, vol_scaling_assignments, exp_weighted, decay_value, shrinkage, shrinkage_target, correlation_view, block_width, theme):
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

    use_weighted_matrix = bool(
        exp_weighted and correlation_view in {"correlation", "covariance"}
    )
    if correlation_view in {"correlation", "covariance"}:
        effective_shrinkage, effective_target = resolve_cov_shrinkage_spec(
            shrinkage,
            shrinkage_target,
            exp_weighted=use_weighted_matrix,
        )
    else:
        effective_shrinkage, effective_target = "none", "scaled_identity"
    try:
        result = generate_correlogram_cached(
            _dataset_key(raw_data),
            periodicity or "daily",
            tuple(selected_series),
            returns_type,
            _mapping_payload(benchmark_assignments),
            _mapping_payload(long_short_assignments),
            _date_range_payload(date_range),
            vol_scaler or 0,
            _mapping_payload(vol_scaling_assignments),
            use_weighted_matrix,
            normalize_decay_input(decay_value, 63.0),
            effective_shrinkage,
            effective_target,
        )

        if result is None:
            return empty_graph, request_key

        available_series = result['available_series']
        corr_matrix = result['corr_matrix']
        cov_matrix = result['cov_matrix']

        # 1. Correlation/Covariance Matrix (Heatmap)
        if correlation_view in {"correlation", "covariance"}:
            is_covariance = correlation_view == "covariance"
            matrix_df = cov_matrix if is_covariance else corr_matrix
            matrix_label = "Covariance" if is_covariance else "Correlation"
            weighted_suffix = ""
            if use_weighted_matrix:
                weighted_suffix = " (Exp Weighted)"
            elif effective_shrinkage != "none":
                weighted_suffix = (
                    f" ({format_cov_shrinkage_spec_label(effective_shrinkage, effective_target)})"
                )

            heatmap_kwargs = {
                "z": matrix_df.values,
                "x": available_series,
                "y": available_series,
                "colorscale": "RdBu_r",
                "zmid": 0,
                "text": matrix_df.values.round(4 if is_covariance else 2),
                "texttemplate": "%{text}",
                "textfont": {"size": 10},
                "hovertemplate": (
                    f"%{{x}} vs %{{y}}<br>{matrix_label}: %{{z:.6f}}<extra></extra>"
                    if is_covariance
                    else "%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>"
                ),
            }
            if not is_covariance:
                heatmap_kwargs["zmin"] = -1
                heatmap_kwargs["zmax"] = 1

            heatmap_fig = go.Figure(data=go.Heatmap(**heatmap_kwargs))

            heatmap_fig.update_layout(
                title=f"{matrix_label} Matrix{weighted_suffix} ({returns_type.title()} Returns)",
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
                return empty_graph, request_key

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


def _factor_quantile_labels(factor_values: pd.Series, quantiles: int):
    """Build aligned quantile labels for factor buckets."""
    if factor_values is None:
        return pd.Series(dtype="object", name="Quantile"), []

    factor_work = pd.Series(factor_values).replace([np.inf, -np.inf], np.nan)
    labels_out = pd.Series(index=factor_work.index, dtype="object", name="Quantile")
    valid = factor_work.dropna()
    if valid.empty:
        return labels_out, []

    q = _coerce_factor_quantiles(quantiles)
    try:
        buckets = pd.qcut(valid, q=q, duplicates="drop")
    except Exception:
        return labels_out, []

    categories = list(getattr(buckets.cat, "categories", []))
    if not categories:
        return labels_out, []

    label_map = {cat: f"Q{idx + 1}" for idx, cat in enumerate(categories)}
    ordered_labels = [label_map[cat] for cat in categories]
    labels_out.loc[buckets.index] = buckets.map(label_map).astype("object")
    return labels_out, ordered_labels


def _build_factor_box_summary_rows(selected_series, dependent_df, factor_series_name, factor_values, quantiles):
    """Return per-quantile summary rows used by Excel export."""
    rows = []
    for series_name in (selected_series or []):
        if series_name not in dependent_df.columns:
            continue
        paired = _build_factor_pair_df(factor_values, dependent_df[series_name])
        if len(paired) < 2:
            continue

        labels, ordered_labels = _factor_quantile_labels(paired["Factor"], quantiles)
        if not ordered_labels:
            continue
        paired = paired.assign(Quantile=labels.fillna("").astype(str))
        for quantile_label in ordered_labels:
            bucket = paired[paired["Quantile"] == quantile_label]
            if bucket.empty:
                continue

            y_vals = bucket["Dependent"]
            q1 = y_vals.quantile(0.25)
            q3 = y_vals.quantile(0.75)
            iqr = q3 - q1
            lower_fence = q1 - 1.5 * iqr
            upper_fence = q3 + 1.5 * iqr
            outliers = int(((y_vals < lower_fence) | (y_vals > upper_fence)).sum())

            rows.append(
                {
                    "Factor": factor_series_name,
                    "Series": series_name,
                    "Quantile": quantile_label,
                    "Observations": int(len(bucket)),
                    "Factor Min": bucket["Factor"].min(),
                    "Factor Max": bucket["Factor"].max(),
                    "Factor Mean": bucket["Factor"].mean(),
                    "Series Mean": y_vals.mean(),
                    "Series Median": y_vals.median(),
                    "Series Q1": q1,
                    "Series Q3": q3,
                    "Series IQR": iqr,
                    "Lower Fence": lower_fence,
                    "Upper Fence": upper_fence,
                    "Outlier Count": outliers,
                }
            )
    return rows


def _build_factor_scatter_summary_rows(selected_series, dependent_df, factor_series_name, factor_values):
    """Return per-series scatter summary rows used by Excel export."""
    rows = []
    for series_name in (selected_series or []):
        if series_name not in dependent_df.columns:
            continue
        paired = _build_factor_pair_df(factor_values, dependent_df[series_name])
        if len(paired) < 2:
            continue

        x_vals = paired["Factor"].to_numpy()
        y_vals = paired["Dependent"].to_numpy()
        corr = paired["Factor"].corr(paired["Dependent"])

        slope = np.nan
        intercept = np.nan
        if len(paired) >= 2 and pd.Series(x_vals).nunique() > 1:
            slope, intercept = np.polyfit(x_vals, y_vals, 1)

        rows.append(
            {
                "Factor": factor_series_name,
                "Series": series_name,
                "Observations": int(len(paired)),
                "Slope": slope,
                "Intercept": intercept,
                "Correlation": corr,
                "R-Squared": (corr ** 2) if pd.notna(corr) else np.nan,
                "Factor Mean": paired["Factor"].mean(),
                "Factor Std": paired["Factor"].std(ddof=0),
                "Series Mean": paired["Dependent"].mean(),
                "Series Std": paired["Dependent"].std(ddof=0),
            }
        )
    return rows


@callback(
    Output("at-factor-analysis-warning", "children"),
    Output("at-factor-analysis-container", "children"),
    Input("at-main-tabs", "value"),
    Input("at-factor-mode-select", "value"),
    Input("at-factor-qq-reference-select", "value"),
    Input("at-factor-series-select", "value"),
    Input("at-factor-quantiles-input", "value"),
    Input("at-factor-transform-select", "value"),
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
    State("at-factor-definitions-db-store", "data"),
    State("at-factor-definitions-local-store", "data"),
    prevent_initial_call=True,
)
def update_factor_analysis(
    active_tab,
    factor_mode,
    factor_qq_reference,
    factor_series,
    factor_quantiles,
    factor_transform,
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
    theme,
    factor_definitions_db=None,
    factor_definitions_local=None,
):
    """Render Factor Analysis charts for selected series."""
    if (
        active_tab != "factor_analysis"
        or not state_ready
        or not _has_complete_date_range(date_range)
    ):
        raise PreventUpdate

    if raw_data is None or not selected_series:
        return None, dmc.Text("Select series to view factor analysis.", size="sm", c="dimmed")
    mode = factor_mode if factor_mode in {"box", "scatter", "detail", "qq"} else "box"
    qq_reference = factor_qq_reference if factor_qq_reference in {"normal", "reference"} else "normal"
    if mode != "qq" and not factor_series:
        return None, dmc.Text("Select a factor series.", size="sm", c="dimmed")

    if mode == "qq":
        dependent_df = _prepare_factor_analysis_selected_df(
            raw_data,
            periodicity,
            selected_series,
            returns_type,
            benchmark_assignments,
            long_short_assignments,
            date_range,
            vol_scaler,
            vol_scaling_assignments,
        )
        factor_values = pd.Series(dtype=float)
    else:
        factor_artifacts = _compute_factor_artifacts(
            raw_data,
            periodicity,
            selected_series,
            factor_series,
            returns_type,
            benchmark_assignments,
            long_short_assignments,
            date_range,
            vol_scaler,
            vol_scaling_assignments,
            factor_transform,
            factor_definitions_db,
            factor_definitions_local,
        )
        dependent_df = factor_artifacts.dependent_df
        factor_values = factor_artifacts.factor_display
    if dependent_df.empty:
        return None, dmc.Text("No dependent-series data available for current settings.", size="sm", c="dimmed")
    if mode != "qq" and factor_values.empty:
        return None, dmc.Text("No factor data available for current settings.", size="sm", c="dimmed")

    quantiles = _coerce_factor_quantiles(factor_quantiles, default=5)
    _factor_prefix, _factor_name = _split_factor_select_key(factor_series)
    display_factor_name = (
        factor_artifacts.factor_display_name
        if mode != "qq"
        else (_factor_name if _factor_name else str(factor_series))
    )
    factor_label = (
        f"{display_factor_name} (Z-Score)"
        if (factor_transform or "raw") == "zscore"
        else display_factor_name
    )
    if mode == "detail":
        detail_df = _build_factor_detail_frame(factor_artifacts, selected_series, quantiles)
        if detail_df.empty:
            return None, dmc.Text("No overlapping data available for factor detail.", size="sm", c="dimmed")
        warning_children = _detail_render_warning(detail_df)
        detail_grid = _build_regime_grid_component(
            "Factor Detail",
            detail_df,
            theme,
            percent_cols={str(col) for col in selected_series},
            pinned_cols={"Date"},
            max_height=560,
        )
        return warning_children, detail_grid
    qq_reference_series = pd.Series(dtype=float)
    qq_reference_name = None
    if mode == "qq" and qq_reference == "reference":
        if not factor_series:
            return None, dmc.Text("Select a reference series to view this Q-Q plot.", size="sm", c="dimmed")
        qq_reference_series = _prepare_at_qq_reference_series(
            raw_data,
            periodicity,
            factor_series,
            returns_type,
            benchmark_assignments,
            long_short_assignments,
            date_range,
            vol_scaler,
            vol_scaling_assignments,
            factor_definitions_db,
            factor_definitions_local,
        )
        if qq_reference_series.empty:
            return None, dmc.Text("No reference data available for current settings.", size="sm", c="dimmed")
        qq_reference_name = qq_reference_series.name or display_factor_name

    charts = []
    total_points = 0
    for series_name in selected_series:
        if series_name not in dependent_df.columns:
            continue
        if mode == "qq":
            if qq_reference == "reference":
                reference_prefix, reference_name = _split_factor_select_key(factor_series)
                if reference_prefix == "raw" and reference_name == series_name:
                    continue
                qq_data = build_reference_qq_series(
                    dependent_df[series_name],
                    qq_reference_series,
                    standardize=True,
                )
                if qq_data is None:
                    continue
                fig = build_qq_figure(
                    qq_data,
                    title=f"Q-Q Plot: {series_name} vs {qq_reference_name}",
                    xlabel=f"{qq_reference_name} Quantiles (Z-Score)",
                    ylabel=f"{series_name} Quantiles (Z-Score)",
                    theme=theme,
                    height=420,
                )
            else:
                qq_data = build_normal_qq_series(dependent_df[series_name])
                if qq_data is None:
                    continue
                fig = build_qq_figure(
                    qq_data,
                    title=f"Q-Q Plot: {series_name} vs Normal",
                    xlabel="Theoretical Quantiles",
                    ylabel=f"{series_name} Quantiles",
                    theme=theme,
                    height=420,
                )
        else:
            paired = _build_factor_pair_df(factor_values, dependent_df[series_name])
            if len(paired) < 2:
                continue
            total_points += len(paired)

            fig = go.Figure()
            if mode == "scatter":
                fig.add_trace(
                    go.Scattergl(
                        x=paired["Factor"],
                        y=paired["Dependent"],
                        mode="markers",
                        name=series_name,
                        marker={"size": 5, "opacity": 0.65},
                    )
                )

                if paired["Factor"].nunique() > 1:
                    slope, intercept = np.polyfit(paired["Factor"].to_numpy(), paired["Dependent"].to_numpy(), 1)
                    x_line = np.linspace(float(paired["Factor"].min()), float(paired["Factor"].max()), 100)
                    y_line = slope * x_line + intercept
                    fig.add_trace(
                        go.Scatter(
                            x=x_line,
                            y=y_line,
                            mode="lines",
                            name="Trend Line",
                            line={"width": 2},
                        )
                    )

                fig.update_layout(
                    title=f"Factor Scatter: {series_name} vs {display_factor_name}",
                    xaxis_title=factor_label,
                    yaxis_title=series_name,
                    yaxis_tickformat=".2%",
                    hovermode="closest",
                    height=420,
                )
            else:
                labels, ordered_labels = _factor_quantile_labels(paired["Factor"], quantiles)
                if not ordered_labels:
                    continue
                paired = paired.assign(Quantile=labels.fillna("").astype(str))
                fig.add_trace(
                    go.Box(
                        x=paired["Quantile"],
                        y=paired["Dependent"],
                        name=series_name,
                        boxpoints="outliers",
                        jitter=0.3,
                        pointpos=0.0,
                        marker={"size": 4, "opacity": 0.6},
                        showlegend=False,
                    )
                )
                fig.update_xaxes(
                    title_text=f"{factor_label} Quantile",
                    categoryorder="array",
                    categoryarray=ordered_labels,
                )
                fig.update_layout(
                    title=f"Factor Box Plot: {series_name} by {display_factor_name} Quantiles",
                    yaxis_title=series_name,
                    yaxis_tickformat=".2%",
                    hovermode="closest",
                    height=420,
                )

            apply_chart_theme(fig, theme)
        charts.append(dcc.Graph(figure=fig, style={"marginBottom": "1.5rem"}))

    if not charts:
        if mode == "qq" and qq_reference == "reference":
            return None, dmc.Text("No overlapping data available to render Q-Q reference analysis.", size="sm", c="dimmed")
        if mode == "qq":
            return None, dmc.Text("No data available to render Q-Q analysis.", size="sm", c="dimmed")
        return None, dmc.Text("No overlapping data available to render factor analysis.", size="sm", c="dimmed")

    warning_children = None
    if mode != "qq" and (len(charts) > 12 or total_points > 50000):
        warning_children = dmc.Alert(
            "Large factor analysis render. Consider narrowing date range or series selection for faster interaction.",
            color="yellow",
            variant="light",
            mb="sm",
        )

    return warning_children, html.Div(charts, style={"height": "100%"})


def _build_signed_gradient_cell_style(max_abs: float) -> dict:
    if not np.isfinite(max_abs) or max_abs <= 0:
        return {"textAlign": "center"}

    style_conditions = []
    n_bins = 10
    for i in range(n_bins):
        lo = max_abs * i / n_bins
        hi = max_abs * (i + 1) / n_bins
        alpha = round(0.1 + 0.6 * (i + 1) / n_bins, 2)
        text_color = "#fff" if alpha > 0.4 else "inherit"
        if i == n_bins - 1:
            style_conditions.append(
                {
                    "condition": f"params.value >= {lo}",
                    "style": {"backgroundColor": f"rgba(34, 139, 34, {alpha})", "color": text_color, "textAlign": "center"},
                }
            )
            style_conditions.append(
                {
                    "condition": f"params.value <= {-lo}",
                    "style": {"backgroundColor": f"rgba(220, 38, 38, {alpha})", "color": text_color, "textAlign": "center"},
                }
            )
        else:
            style_conditions.append(
                {
                    "condition": f"params.value >= {lo} && params.value < {hi}",
                    "style": {"backgroundColor": f"rgba(34, 139, 34, {alpha})", "color": text_color, "textAlign": "center"},
                }
            )
            style_conditions.append(
                {
                    "condition": f"params.value <= {-lo} && params.value > {-hi}",
                    "style": {"backgroundColor": f"rgba(220, 38, 38, {alpha})", "color": text_color, "textAlign": "center"},
                }
            )
    return {"styleConditions": style_conditions, "defaultStyle": {"textAlign": "center"}}


def _build_conditional_returns_grid_component(
    title: str,
    mean_df: pd.DataFrame,
    count_df: pd.DataFrame,
    *,
    row_label: str,
    max_height: int = 320,
):
    if mean_df is None or mean_df.empty:
        return dmc.Paper(
            withBorder=True,
            radius="md",
            p="sm",
            children=[
                dmc.Text(title, fw=600, size="sm", mb=4),
                dmc.Text("No qualifying observations.", size="sm", c="dimmed"),
            ],
        )

    mean_frame = mean_df.copy()
    mean_frame.index = [str(idx) for idx in mean_frame.index]
    mean_frame.columns = [str(col) for col in mean_frame.columns]
    count_frame = count_df.reindex(mean_frame.index, columns=mean_frame.columns).fillna(0)

    values = mean_frame.to_numpy(dtype=float, copy=False)
    finite_values = values[np.isfinite(values)]
    max_abs = float(np.max(np.abs(finite_values))) if finite_values.size else 0.0
    cell_style = _build_signed_gradient_cell_style(max_abs)

    row_data = []
    for idx_label in mean_frame.index:
        row = {row_label: idx_label}
        for col in mean_frame.columns:
            value = mean_frame.loc[idx_label, col]
            count = int(count_frame.loc[idx_label, col]) if pd.notna(count_frame.loc[idx_label, col]) else 0
            row[col] = None if pd.isna(value) else float(value)
            row[f"__tooltip_{col}"] = f"N: {count}"
        row_data.append(row)

    column_defs = [
        {
            "field": row_label,
            "headerName": row_label,
            "pinned": "left",
            "width": 110,
            "cellStyle": {"textAlign": "center"},
            "headerClass": "dashmat-center-header",
        }
    ]
    for col in mean_frame.columns:
        column_defs.append(
            {
                "field": col,
                "headerName": col,
                "width": 110,
                "valueFormatter": {"function": "params.value != null ? d3.format('.2%')(params.value) : ''"},
                "tooltipValueGetter": {
                    "function": "params.data && params.colDef && params.colDef.field ? params.data['__tooltip_' + params.colDef.field] : ''"
                },
                "cellStyle": cell_style,
                "headerClass": "dashmat-center-header",
            }
        )

    grid_height = min(max_height, max(140, 52 + (len(row_data) + 1) * 28))
    return dmc.Paper(
        withBorder=True,
        radius="md",
        p="sm",
        children=[
            dmc.Text(title, fw=600, size="sm", mb=4),
            dag.AgGrid(
                enableEnterpriseModules=True,
                licenseKey=AG_GRID_LICENSE_KEY,
                id=f"at-conditional-grid-{hashlib.md5(title.encode('utf-8')).hexdigest()[:10]}",
                className="ag-theme-alpine",
                columnDefs=column_defs,
                rowData=row_data,
                defaultColDef={
                    "sortable": True,
                    "resizable": True,
                    "suppressHeaderMenuButton": True,
                    "cellStyle": {"textAlign": "center"},
                    "headerClass": "dashmat-center-header",
                },
                style={"height": f"{grid_height}px", "width": "100%"},
                dashGridOptions=literal_field_dash_grid_options(
                    {
                        "animateRows": False,
                        "pagination": False,
                        "suppressExcelExport": True,
                        "enableRangeSelection": True,
                        "suppressCsvExport": True,
                        "tooltipShowDelay": 100,
                    }
                ),
            ),
        ],
    )


def _build_conditional_detail_grid_component(
    title: str,
    detail_df: pd.DataFrame,
    *,
    series_names: tuple[str, ...],
    include_forward: bool,
    max_height: int = 560,
):
    if detail_df is None or detail_df.empty:
        return dmc.Paper(
            withBorder=True,
            radius="md",
            p="sm",
            children=[
                dmc.Text(title, fw=600, size="sm", mb=4),
                dmc.Text("No evaluated windows available.", size="sm", c="dimmed"),
            ],
        )

    frame = detail_df.copy()
    for col in frame.columns:
        if pd.api.types.is_datetime64_any_dtype(frame[col]):
            frame[col] = pd.to_datetime(frame[col], errors="coerce").dt.strftime("%Y-%m-%d")

    column_defs = [
        {
            "field": "Lookback",
            "headerName": "Lookback",
            "pinned": "left",
            "width": 100,
            "headerClass": "dashmat-center-header",
        },
    ]
    if include_forward:
        column_defs.append(
            {
                "field": "Forward Period",
                "headerName": "Forward Period",
                "pinned": "left",
                "width": 120,
                "headerClass": "dashmat-center-header",
            }
        )
    column_defs.append(
        {
            "field": "End Date",
            "headerName": "End Date",
            "pinned": "left",
            "width": 130,
            "headerClass": "dashmat-center-header",
        }
    )
    column_defs.extend(
        [
            {
                "field": "Factor Value",
                "headerName": "Factor Value",
                "width": 120,
                "valueFormatter": {"function": "params.value != null ? d3.format('.4f')(params.value) : ''"},
                "headerClass": "dashmat-center-header",
            },
            {
                "field": "Condition Met",
                "headerName": "Condition Met",
                "width": 120,
                "filter": "agSetColumnFilter",
                "valueFormatter": {"function": "params.value === true ? 'True' : (params.value === false ? 'False' : '')"},
                "headerClass": "dashmat-center-header",
            },
        ]
    )
    for series_name in series_names:
        if series_name not in frame.columns:
            continue
        column_defs.append(
            {
                "field": series_name,
                "headerName": series_name,
                "width": 120,
                "valueFormatter": {"function": "params.value != null ? d3.format('.2%')(params.value) : ''"},
                "headerClass": "dashmat-center-header",
            }
        )

    return dmc.Paper(
        withBorder=True,
        radius="md",
        p="sm",
        children=[
            dmc.Text(title, fw=600, size="sm", mb=4),
            dag.AgGrid(
                enableEnterpriseModules=True,
                licenseKey=AG_GRID_LICENSE_KEY,
                id=f"at-conditional-detail-grid-{hashlib.md5(title.encode('utf-8')).hexdigest()[:10]}",
                className="ag-theme-alpine",
                columnDefs=column_defs,
                rowData=frame.to_dict("records"),
                defaultColDef={
                    "sortable": True,
                    "resizable": True,
                    "filter": True,
                    "suppressHeaderMenuButton": True,
                    "cellStyle": {"textAlign": "center"},
                    "headerClass": "dashmat-center-header",
                },
                style={"height": f"{max_height}px", "width": "100%"},
                dashGridOptions=literal_field_dash_grid_options(
                    {
                        "animateRows": False,
                        "pagination": False,
                        "suppressExcelExport": True,
                        "enableRangeSelection": True,
                        "suppressCsvExport": True,
                    }
                ),
            ),
        ],
    )


def _conditional_export_block(title: str, df: pd.DataFrame, row_label: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame([{row_label: title}, {row_label: "No qualifying observations."}, {row_label: ""}])

    block = df.copy()
    block.index = [str(idx) for idx in block.index]
    block.columns = [str(col) for col in block.columns]
    block.insert(0, row_label, block.index)
    block = block.reset_index(drop=True)
    title_row = pd.DataFrame([{row_label: title}])
    spacer_row = pd.DataFrame([{row_label: ""}])
    return pd.concat([title_row, block, spacer_row], ignore_index=True, sort=False)


def _build_conditional_export_frame(
    payload: _ConditionalReturnsPayload,
    view: str,
    selected_series: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    blocks = []
    if view == "coincident":
        blocks.append(_conditional_export_block("Mean Returns", payload.coincident_mean_df, "Window"))
        blocks.append(_conditional_export_block("Observation Counts", payload.coincident_count_df, "Window"))
    else:
        for series_name in selected_series:
            blocks.append(
                _conditional_export_block(
                    f"{series_name} - Mean Returns",
                    payload.forward_mean_by_series.get(series_name, pd.DataFrame()),
                    "Lookback",
                )
            )
            blocks.append(
                _conditional_export_block(
                    f"{series_name} - Observation Counts",
                    payload.forward_count_by_series.get(series_name, pd.DataFrame()),
                    "Lookback",
                )
            )
    if not blocks:
        return pd.DataFrame([{"Note": "No qualifying observations."}])
    return pd.concat(blocks, ignore_index=True, sort=False)


def _build_conditional_detail_export_frame(detail_df: pd.DataFrame, include_forward: bool) -> pd.DataFrame:
    if detail_df is None or detail_df.empty:
        note = "No evaluated windows available."
        if include_forward:
            return pd.DataFrame([{"End Date": note}])
        return pd.DataFrame([{"End Date": note}])
    return detail_df.copy()


def _build_statistics_export_frame(stats: list[dict]) -> pd.DataFrame:
    stats_data = {"Statistic": [stat_name for stat_name, _ in STATS_CONFIG]}
    for series_stats in stats:
        series_name = series_stats["Series"]
        stats_data[series_name] = [series_stats.get(stat_name) for stat_name, _ in STATS_CONFIG]
    return pd.DataFrame(stats_data)


def _compute_analytics_export_artifacts(
    bundle: _AnalyticsComputeBundle,
    returns_type,
    use_risk_free,
    saved_series_store,
    correlation_exp_wt,
    correlation_halflife,
    correlation_shrinkage,
    correlation_shrinkage_target,
) -> _AnalyticsExportArtifacts:
    with timed_block("analyticstool.download_excel.returns"):
        returns_df = _compute_selected_returns_cached(
            bundle.dataset_key,
            bundle.periodicity,
            bundle.selected_series,
            returns_type or "total",
            bundle.benchmark_payload,
            bundle.long_short_payload,
            bundle.date_range_payload,
            bundle.vol_scaler,
            bundle.vol_scaling_payload,
        )

    if returns_df.empty:
        return _AnalyticsExportArtifacts(
            returns_df=pd.DataFrame(),
            stats_df=pd.DataFrame(),
            corr_df=pd.DataFrame(),
            cov_df=pd.DataFrame(),
        )

    with timed_block("analyticstool.download_excel.statistics"):
        stats = calculate_statistics_cached(
            bundle.dataset_key,
            bundle.periodicity,
            bundle.selected_series,
            bundle.benchmark_payload,
            bundle.long_short_payload,
            bundle.date_range_payload,
            bundle.vol_scaler,
            bundle.vol_scaling_payload,
            _risk_free_json_from_store(saved_series_store),
            _spx_json_from_store(saved_series_store),
            bool(use_risk_free),
        )
    stats_df = _build_statistics_export_frame(stats)

    effective_corr_shrinkage, effective_corr_target = resolve_cov_shrinkage_spec(
        correlation_shrinkage,
        correlation_shrinkage_target,
        exp_weighted=bool(correlation_exp_wt),
    )
    try:
        matrix_result = generate_correlogram_cached(
            bundle.dataset_key,
            bundle.periodicity,
            bundle.selected_series,
            returns_type,
            bundle.benchmark_payload,
            bundle.long_short_payload,
            bundle.date_range_payload,
            bundle.vol_scaler,
            bundle.vol_scaling_payload,
            bool(correlation_exp_wt),
            normalize_decay_input(correlation_halflife, 63.0),
            effective_corr_shrinkage,
            effective_corr_target,
        )
    except ValueError:
        matrix_result = None

    if matrix_result is not None:
        corr_df = matrix_result["corr_matrix"]
        cov_df = matrix_result["cov_matrix"]
    else:
        corr_df = returns_df.corr()
        cov_df = returns_df.cov()
    corr_df.index.name = "Series"
    cov_df.index.name = "Series"

    return _AnalyticsExportArtifacts(
        returns_df=returns_df,
        stats_df=stats_df,
        corr_df=corr_df,
        cov_df=cov_df,
    )


def _build_rolling_export_sheet(
    bundle: _AnalyticsComputeBundle,
    returns_type,
    rolling_window,
    rolling_return_type,
) -> _ExcelSheetSpec | None:
    try:
        with timed_block("analyticstool.download_excel.rolling"):
            window = rolling_window if rolling_window else "1y"
            return_type = rolling_return_type if rolling_return_type else "annualized"

            rolling_df = calculate_rolling_returns(
                bundle.dataset_key,
                bundle.periodicity,
                bundle.selected_series,
                returns_type,
                bundle.benchmark_payload,
                bundle.long_short_payload,
                bundle.date_range_payload,
                window,
                return_type,
                "total_return",
                bundle.vol_scaler,
                bundle.vol_scaling_payload,
            )
            if rolling_df.empty:
                return None

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
            return _ExcelSheetSpec(
                name=sheet_name,
                frame=rolling_df,
                write_index=True,
                format_index=True,
            )
    except Exception:
        return None


def _build_calendar_export_sheet(
    bundle: _AnalyticsComputeBundle,
    original_periodicity,
    returns_type,
    monthly_view,
    monthly_series,
) -> _ExcelSheetSpec | None:
    if original_periodicity not in {"daily", "monthly"}:
        return None

    try:
        with timed_block("analyticstool.download_excel.calendar"):
            if monthly_view == "monthly" and monthly_series and monthly_series in bundle.selected_series:
                _, row_data = create_monthly_view(
                    bundle.dataset_key,
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

                if not row_data:
                    return None

                calendar_df = pd.DataFrame(row_data).set_index("Year_Label")
                calendar_df.index.name = "Year"
            else:
                calendar_df = calculate_calendar_year_returns(
                    bundle.dataset_key,
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
                if calendar_df.empty:
                    return None

            return _ExcelSheetSpec(
                name="Calendar Year",
                frame=calendar_df,
                write_index=True,
                format_index=True,
            )
    except Exception:
        return None


def _build_growth_export_sheet(bundle: _AnalyticsComputeBundle) -> _ExcelSheetSpec | None:
    try:
        with timed_block("analyticstool.download_excel.growth"):
            growth_df = calculate_growth_of_dollar(
                bundle.dataset_key,
                bundle.periodicity,
                bundle.selected_series,
                bundle.benchmark_payload,
                bundle.long_short_payload,
                bundle.date_range_payload,
                bundle.vol_scaler,
                bundle.vol_scaling_payload,
            )
            if growth_df.empty:
                return None
            return _ExcelSheetSpec(
                name="Growth of $1",
                frame=growth_df,
                write_index=True,
                format_index=True,
            )
    except Exception:
        return None


def _build_drawdown_export_sheet(bundle: _AnalyticsComputeBundle, returns_type) -> _ExcelSheetSpec | None:
    try:
        with timed_block("analyticstool.download_excel.drawdown"):
            drawdown_df = calculate_drawdown(
                bundle.dataset_key,
                bundle.periodicity,
                bundle.selected_series,
                returns_type,
                bundle.benchmark_payload,
                bundle.long_short_payload,
                bundle.date_range_payload,
                bundle.vol_scaler,
                bundle.vol_scaling_payload,
            )
            if drawdown_df.empty:
                return None
            return _ExcelSheetSpec(
                name="Drawdown",
                frame=drawdown_df,
                write_index=True,
                format_index=True,
            )
    except Exception:
        return None


def _build_core_export_sheets(
    bundle: _AnalyticsComputeBundle,
    original_periodicity,
    returns_type,
    rolling_window,
    rolling_return_type,
    monthly_view,
    monthly_series,
    use_risk_free,
    correlation_exp_wt,
    correlation_halflife,
    correlation_shrinkage,
    correlation_shrinkage_target,
    saved_series_store,
) -> list[_ExcelSheetSpec]:
    artifacts = _compute_analytics_export_artifacts(
        bundle,
        returns_type,
        use_risk_free,
        saved_series_store,
        correlation_exp_wt,
        correlation_halflife,
        correlation_shrinkage,
        correlation_shrinkage_target,
    )
    if artifacts.returns_df.empty:
        return []

    sheets = [
        _ExcelSheetSpec(name="Statistics", frame=artifacts.stats_df, write_index=False, format_index=False),
        _ExcelSheetSpec(name="Returns", frame=artifacts.returns_df, write_index=True, format_index=True),
    ]

    for optional_sheet in (
        _build_rolling_export_sheet(bundle, returns_type, rolling_window, rolling_return_type),
        _build_calendar_export_sheet(bundle, original_periodicity, returns_type, monthly_view, monthly_series),
        _build_growth_export_sheet(bundle),
        _build_drawdown_export_sheet(bundle, returns_type),
    ):
        if optional_sheet is not None:
            sheets.append(optional_sheet)

    sheets.extend(
        [
            _ExcelSheetSpec(name="Correlation", frame=artifacts.corr_df, write_index=True, format_index=True),
            _ExcelSheetSpec(name="Covariance", frame=artifacts.cov_df, write_index=True, format_index=True),
        ]
    )
    return sheets


def _build_factor_export_sheets(
    bundle: _AnalyticsComputeBundle,
    returns_type,
    benchmark_assignments,
    long_short_assignments,
    date_range,
    vol_scaling_assignments,
    factor_series,
    factor_quantiles,
    factor_transform,
    factor_definitions_db,
    factor_definitions_local,
) -> list[_ExcelSheetSpec]:
    if not factor_series:
        return []

    factor_transform_value = factor_transform if factor_transform in {"raw", "zscore"} else "raw"
    quantiles = _coerce_factor_quantiles(factor_quantiles, default=5)
    factor_artifacts = _compute_factor_artifacts(
        bundle.dataset_key,
        bundle.periodicity,
        bundle.selected_series,
        factor_series,
        returns_type,
        benchmark_assignments,
        long_short_assignments,
        date_range,
        bundle.vol_scaler,
        vol_scaling_assignments,
        factor_transform_value,
        factor_definitions_db,
        factor_definitions_local,
    )
    dependent_df = factor_artifacts.dependent_df
    factor_values = factor_artifacts.factor_display
    display_factor_name = factor_artifacts.factor_display_name or str(factor_series)

    box_rows = _build_factor_box_summary_rows(
        bundle.selected_series,
        dependent_df,
        display_factor_name,
        factor_values,
        quantiles,
    )
    box_df = pd.DataFrame(box_rows)
    if box_df.empty:
        box_df = pd.DataFrame([{"Note": "No overlapping observations for factor box analysis."}])
    else:
        box_df.insert(1, "Transform", "Z-Score" if factor_transform_value == "zscore" else "Raw")
        box_df.insert(2, "Quantiles", quantiles)

    scatter_rows = _build_factor_scatter_summary_rows(
        bundle.selected_series,
        dependent_df,
        display_factor_name,
        factor_values,
    )
    scatter_df = pd.DataFrame(scatter_rows)
    if scatter_df.empty:
        scatter_df = pd.DataFrame([{"Note": "No overlapping observations for factor scatter analysis."}])
    else:
        scatter_df.insert(1, "Transform", "Z-Score" if factor_transform_value == "zscore" else "Raw")

    detail_df = _build_factor_detail_frame(
        factor_artifacts,
        bundle.selected_series,
        quantiles,
    )
    if detail_df.empty:
        detail_df = pd.DataFrame([{"Note": "No overlapping observations for factor detail."}])

    return [
        _ExcelSheetSpec(name="Factor Analysis - Box", frame=box_df, write_index=False, format_index=False),
        _ExcelSheetSpec(name="Factor Analysis - Scatter", frame=scatter_df, write_index=False, format_index=False),
        _ExcelSheetSpec(name="Factor Analysis - Detail", frame=detail_df, write_index=False, format_index=False),
    ]


def _build_conditional_export_sheets(
    bundle: _AnalyticsComputeBundle,
    returns_type,
    factor_series,
    factor_transform,
    conditional_comparator,
    conditional_threshold,
    conditional_window_conversion,
    conditional_step,
    conditional_step_unit,
    factor_definitions_db,
    factor_definitions_local,
) -> list[_ExcelSheetSpec]:
    if not factor_series:
        return []

    conditional_definition_payload = ""
    factor_prefix, factor_name = _split_factor_select_key(factor_series)
    if factor_prefix == "def":
        definition = _lookup_factor_definition(
            factor_name,
            factor_definitions_db,
            factor_definitions_local,
        )
        if definition:
            conditional_definition_payload = _definition_payload_for_compute(definition)

    conditional_payload = _compute_conditional_returns_cached(
        bundle.dataset_key,
        bundle.periodicity,
        bundle.selected_series,
        returns_type or "total",
        bundle.benchmark_payload,
        bundle.long_short_payload,
        bundle.date_range_payload,
        bundle.vol_scaler,
        bundle.vol_scaling_payload,
        factor_series,
        factor_transform if factor_transform in {"raw", "zscore"} else "raw",
        conditional_definition_payload,
        conditional_comparator if conditional_comparator in {"le", "ge"} else "le",
        float(pd.to_numeric(pd.Series([conditional_threshold]), errors="coerce").iloc[0] or 0.0),
        conditional_window_conversion if conditional_window_conversion in {"compound", "end", "average", "sum"} else "compound",
        _coerce_positive_int(conditional_step, default=1),
        conditional_step_unit if conditional_step_unit in {"periods", "months"} else "months",
        True,
    )

    coincident_export_df = _build_conditional_export_frame(
        conditional_payload,
        "coincident",
        bundle.selected_series,
    )
    forward_export_df = _build_conditional_export_frame(
        conditional_payload,
        "forward",
        bundle.selected_series,
    )
    coincident_detail_export_df = _build_conditional_detail_export_frame(
        conditional_payload.coincident_detail_df,
        False,
    )
    forward_detail_export_df = _build_conditional_detail_export_frame(
        conditional_payload.forward_detail_df,
        True,
    )

    return [
        _ExcelSheetSpec(name="Conditional Coincident", frame=coincident_export_df, write_index=False, format_index=False),
        _ExcelSheetSpec(name="Conditional Forward", frame=forward_export_df, write_index=False, format_index=False),
        _ExcelSheetSpec(name="Cond Coincident Detail", frame=coincident_detail_export_df, write_index=False, format_index=False),
        _ExcelSheetSpec(name="Cond Forward Detail", frame=forward_detail_export_df, write_index=False, format_index=False),
    ]


def _build_regime_export_sheets(
    bundle: _AnalyticsComputeBundle,
    returns_type,
    benchmark_assignments,
    long_short_assignments,
    date_range,
    vol_scaling_assignments,
    regime_definition_key,
    regime_definitions_db,
    regime_definitions_local,
    regime_series_store,
) -> list[_ExcelSheetSpec]:
    if not regime_definition_key:
        return []

    try:
        regime_raw_data = get_raw_dataset_json(bundle.dataset_key)
    except Exception:
        regime_raw_data = str(bundle.dataset_key or "")

    regime_result = _build_regime_analysis_payload(
        regime_raw_data,
        bundle.periodicity,
        bundle.selected_series,
        returns_type,
        benchmark_assignments,
        long_short_assignments,
        date_range,
        bundle.vol_scaler,
        vol_scaling_assignments,
        regime_definition_key,
        regime_definitions_db,
        regime_definitions_local,
        regime_series_store,
    )
    if regime_result.status != "ok" or regime_result.payload is None:
        return []

    regime_payload = regime_result.payload
    stats_df_regime = regime_payload.stats_df
    if stats_df_regime.empty:
        stats_df_regime = pd.DataFrame(
            [{"Note": "No overlapping observations for regime statistics."}]
        )

    detail_out = regime_payload.detail_df
    if detail_out.empty:
        detail_out = pd.DataFrame([{"Note": "No regime raw detail is available."}])

    transition_out = (
        regime_payload.transition_df.reset_index()
        if regime_payload.transition_df is not None and not regime_payload.transition_df.empty
        else pd.DataFrame()
    )
    if transition_out.empty:
        transition_out = pd.DataFrame(
            [{"Note": "No transition matrix available (requires at least two observations)."}]
        )

    duration_out = regime_payload.duration_df
    if duration_out.empty:
        duration_out = pd.DataFrame(
            [{"Note": "No duration summary available."}]
        )

    return [
        _ExcelSheetSpec(name="Regime - Settings", frame=regime_payload.settings_df, write_index=False, format_index=False),
        _ExcelSheetSpec(name="Regime - Statistics", frame=stats_df_regime, write_index=False, format_index=False),
        _ExcelSheetSpec(name="Regime - Detail", frame=detail_out, write_index=False, format_index=False),
        _ExcelSheetSpec(name="Regime - Transition", frame=transition_out, write_index=False, format_index=False),
        _ExcelSheetSpec(name="Regime - Duration", frame=duration_out, write_index=False, format_index=False),
    ]


def _resolve_export_sheet_specs(
    bundle: _AnalyticsComputeBundle,
    original_periodicity,
    returns_type,
    benchmark_assignments,
    long_short_assignments,
    date_range,
    rolling_window,
    rolling_return_type,
    monthly_view,
    monthly_series,
    use_risk_free,
    correlation_exp_wt,
    correlation_halflife,
    correlation_shrinkage,
    correlation_shrinkage_target,
    factor_series,
    factor_quantiles,
    factor_transform,
    conditional_comparator,
    conditional_threshold,
    conditional_window_conversion,
    conditional_step,
    conditional_step_unit,
    factor_definitions_db,
    factor_definitions_local,
    regime_definition_key,
    regime_definitions_db,
    regime_definitions_local,
    regime_series_store,
    vol_scaling_assignments,
    saved_series_store,
) -> list[_ExcelSheetSpec]:
    sheet_specs = _build_core_export_sheets(
        bundle,
        original_periodicity,
        returns_type,
        rolling_window,
        rolling_return_type,
        monthly_view,
        monthly_series,
        use_risk_free,
        correlation_exp_wt,
        correlation_halflife,
        correlation_shrinkage,
        correlation_shrinkage_target,
        saved_series_store,
    )
    if not sheet_specs:
        return []

    try:
        with timed_block("analyticstool.download_excel.factor_analysis"):
            sheet_specs.extend(
                _build_factor_export_sheets(
                    bundle,
                    returns_type,
                    benchmark_assignments,
                    long_short_assignments,
                    date_range,
                    vol_scaling_assignments,
                    factor_series,
                    factor_quantiles,
                    factor_transform,
                    factor_definitions_db,
                    factor_definitions_local,
                )
            )
    except Exception:
        pass

    try:
        with timed_block("analyticstool.download_excel.conditional_returns"):
            sheet_specs.extend(
                _build_conditional_export_sheets(
                    bundle,
                    returns_type,
                    factor_series,
                    factor_transform,
                    conditional_comparator,
                    conditional_threshold,
                    conditional_window_conversion,
                    conditional_step,
                    conditional_step_unit,
                    factor_definitions_db,
                    factor_definitions_local,
                )
            )
    except Exception:
        pass

    try:
        with timed_block("analyticstool.download_excel.regime_analysis"):
            sheet_specs.extend(
                _build_regime_export_sheets(
                    bundle,
                    returns_type,
                    benchmark_assignments,
                    long_short_assignments,
                    date_range,
                    vol_scaling_assignments,
                    regime_definition_key,
                    regime_definitions_db,
                    regime_definitions_local,
                    regime_series_store,
                )
            )
    except Exception:
        pass

    return sheet_specs


def _write_export_sheet_specs(writer, sheet_specs: list[_ExcelSheetSpec]) -> None:
    for sheet in sheet_specs:
        write_excel_with_autofit(
            writer,
            format_excel_dates(sheet.frame, format_index=sheet.format_index),
            sheet.name,
            index=sheet.write_index,
        )


@callback(
    Output("at-conditional-returns-warning", "children"),
    Output("at-conditional-returns-container", "children"),
    Input("at-main-tabs", "value"),
    Input("at-conditional-display-mode-select", "value"),
    Input("at-conditional-view-select", "value"),
    Input("at-conditional-comparator-select", "value"),
    Input("at-conditional-threshold-input", "value"),
    Input("at-conditional-window-conversion-select", "value"),
    Input("at-conditional-step-input", "value"),
    Input("at-conditional-step-unit-select", "value"),
    Input("at-factor-series-select-conditional", "value"),
    Input("at-factor-transform-select-conditional", "value"),
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
    State("at-factor-definitions-db-store", "data"),
    State("at-factor-definitions-local-store", "data"),
    prevent_initial_call=True,
)
def update_conditional_returns(
    active_tab,
    conditional_display_mode,
    conditional_view,
    conditional_comparator,
    conditional_threshold,
    conditional_window_conversion,
    conditional_step,
    conditional_step_unit,
    factor_series,
    factor_transform,
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
    factor_definitions_db=None,
    factor_definitions_local=None,
):
    if (
        active_tab != "conditional_returns"
        or not state_ready
        or not _has_complete_date_range(date_range)
    ):
        raise PreventUpdate

    if raw_data is None or not selected_series:
        return None, dmc.Text("Select series to view conditional returns.", size="sm", c="dimmed")
    if not factor_series:
        return None, dmc.Text("Select a factor series.", size="sm", c="dimmed")

    definition_payload = ""
    factor_prefix, factor_name = _split_factor_select_key(factor_series)
    if factor_prefix == "def":
        definition = _lookup_factor_definition(factor_name, factor_definitions_db, factor_definitions_local)
        if not definition:
            return None, dmc.Text("Selected factor definition is unavailable.", size="sm", c="dimmed")
        definition_payload = _definition_payload_for_compute(definition)

    display_mode = conditional_display_mode if conditional_display_mode in {"summary", "detail"} else "summary"
    normalized_periodicity = periodicity or "daily"
    normalized_returns_type = returns_type or "total"
    benchmark_payload = _mapping_payload(benchmark_assignments)
    long_short_payload = _mapping_payload(long_short_assignments)
    date_payload = _date_range_payload(date_range)
    normalized_vol_scaler = vol_scaler or 0
    vol_scaling_payload = _mapping_payload(vol_scaling_assignments)
    normalized_transform = factor_transform if factor_transform in {"raw", "zscore"} else "raw"
    normalized_comparator = conditional_comparator if conditional_comparator in {"le", "ge"} else "le"
    normalized_threshold = float(pd.to_numeric(pd.Series([conditional_threshold]), errors="coerce").iloc[0] or 0.0)
    normalized_conversion = conditional_window_conversion if conditional_window_conversion in {"compound", "end", "average", "sum"} else "compound"
    normalized_step = _coerce_positive_int(conditional_step, default=1)
    normalized_step_unit = conditional_step_unit if conditional_step_unit in {"periods", "months"} else "months"
    selected_series_tuple = tuple(selected_series or ())

    warning_children = None

    payload = _compute_conditional_returns_cached(
        _dataset_key(raw_data) or "",
        normalized_periodicity,
        selected_series_tuple,
        normalized_returns_type,
        benchmark_payload,
        long_short_payload,
        date_payload,
        normalized_vol_scaler,
        vol_scaling_payload,
        factor_series,
        normalized_transform,
        definition_payload,
        normalized_comparator,
        normalized_threshold,
        normalized_conversion,
        normalized_step,
        normalized_step_unit,
        display_mode == "detail",
    )

    if display_mode == "detail":
        detail_frame = payload.coincident_detail_df if (conditional_view or "forward") == "coincident" else payload.forward_detail_df
        if detail_frame.empty:
            return None, dmc.Text("No evaluated windows available for current settings.", size="sm", c="dimmed")
        if (conditional_view or "forward") == "coincident":
            grid = _build_conditional_detail_grid_component(
                f"Coincident Conditional Returns Detail vs {payload.factor_label}",
                payload.coincident_detail_df,
                series_names=selected_series_tuple,
                include_forward=False,
            )
            return warning_children, grid

        grid = _build_conditional_detail_grid_component(
            f"Forward Conditional Returns Detail vs {payload.factor_label}",
            payload.forward_detail_df,
            series_names=selected_series_tuple,
            include_forward=True,
        )
        return warning_children, grid

    if payload.coincident_mean_df.empty and not payload.forward_mean_by_series:
        return None, dmc.Text("No qualifying data available for current settings.", size="sm", c="dimmed")

    if (conditional_view or "forward") == "forward" and len(payload.forward_mean_by_series) > 10:
        warning_children = dmc.Alert(
            "Large Conditional Returns render. Consider narrowing the selected series for faster interaction.",
            color="yellow",
            variant="light",
            mb="sm",
        )

    if (conditional_view or "forward") == "coincident":
        grid = _build_conditional_returns_grid_component(
            f"Coincident Conditional Returns vs {payload.factor_label}",
            payload.coincident_mean_df,
            payload.coincident_count_df,
            row_label="Window",
        )
        return warning_children, grid

    stack_children = []
    for series_name in selected_series:
        if series_name not in payload.forward_mean_by_series:
            continue
        stack_children.append(
            _build_conditional_returns_grid_component(
                f"{series_name} Forward Conditional Returns vs {payload.factor_label}",
                payload.forward_mean_by_series[series_name],
                payload.forward_count_by_series[series_name],
                row_label="Lookback",
            )
        )

    if not stack_children:
        return warning_children, dmc.Text("No qualifying forward observations available.", size="sm", c="dimmed")

    return warning_children, dmc.Stack(gap="sm", children=stack_children)


def _build_regime_grid_component(
    title: str,
    df: pd.DataFrame,
    theme: str,
    percent_cols: set[str] | None = None,
    integer_cols: set[str] | None = None,
    pinned_cols: set[str] | None = None,
    max_height: int = 320,
):
    if df is None or df.empty:
        return dmc.Paper(
            withBorder=True,
            radius="md",
            p="sm",
            children=[
                dmc.Text(title, fw=600, size="sm", mb=4),
                dmc.Text("No data available.", size="sm", c="dimmed"),
            ],
        )

    percent_cols = {str(c) for c in (percent_cols or set())}
    integer_cols = {str(c) for c in (integer_cols or set())}
    pinned_cols = {str(c) for c in (pinned_cols or set())}
    frame = df.copy()
    frame.columns = [str(c) for c in frame.columns]
    for col in frame.columns:
        if pd.api.types.is_datetime64_any_dtype(frame[col]):
            frame[col] = pd.to_datetime(frame[col], errors="coerce").dt.strftime("%Y-%m-%d")

    column_defs = []
    for idx, col in enumerate(frame.columns):
        col_def = {"field": col, "suppressHeaderMenuButton": True, "resizable": True}
        if col in percent_cols:
            col_def["valueFormatter"] = {"function": "params.value != null ? d3.format('.2%')(params.value) : ''"}
        elif col in integer_cols:
            col_def["valueFormatter"] = {"function": "params.value != null ? d3.format(',d')(params.value) : ''"}
        elif pd.api.types.is_numeric_dtype(frame[col]):
            col_def["valueFormatter"] = {"function": "params.value != null ? d3.format('.4f')(params.value) : ''"}
        if col in pinned_cols or (idx == 0 and not pinned_cols):
            col_def["pinned"] = "left"
        if "date" in col.lower():
            col_def["width"] = 130
        elif col.lower().startswith("series"):
            col_def["width"] = 150
        else:
            col_def["width"] = 120
        column_defs.append(col_def)

    row_data = frame.to_dict("records")
    grid_height = min(max_height, max(120, 52 + (len(row_data) + 1) * 26))
    return dmc.Paper(
        withBorder=True,
        radius="md",
        p="sm",
        children=[
            dmc.Text(title, fw=600, size="sm", mb=4),
            dag.AgGrid(
                enableEnterpriseModules=True,
                licenseKey=AG_GRID_LICENSE_KEY,
                id=f"at-regime-grid-{title.lower().replace(' ', '-')}",
                className="ag-theme-alpine",
                columnDefs=column_defs,
                rowData=row_data,
                defaultColDef={
                    "sortable": True,
                    "resizable": True,
                    "suppressHeaderMenuButton": True,
                    "cellStyle": {"textAlign": "center"},
                    "headerClass": "dashmat-center-header",
                },
                style={"height": f"{grid_height}px", "width": "100%"},
                dashGridOptions=literal_field_dash_grid_options({
                    "animateRows": True,
                    "pagination": False,
                    "suppressExcelExport": True,
                    "enableRangeSelection": True,
                    "suppressCsvExport": True,
                }),
            ),
        ],
    )


def _detail_render_warning(df: pd.DataFrame) -> dmc.Alert | None:
    if df is None or df.empty:
        return None
    cell_count = int(len(df.index) * max(1, len(df.columns)))
    if cell_count <= ANALYSIS_DETAIL_RENDER_CELL_WARNING_THRESHOLD:
        return None
    return dmc.Alert(
        "Large raw detail view. Excel export includes the same full detail if browser rendering feels slow.",
        color="yellow",
        variant="light",
        mb="sm",
    )


def _build_regime_settings_text_component(payload: _RegimeAnalysisPayload):
    diagnostics = payload.diagnostics or {}
    regime_name = str(payload.definition.get("RegimeName") or "").strip() or "—"
    method_type = diagnostics.get("method_type")
    num_regimes = diagnostics.get("num_regimes")
    observations = diagnostics.get("observations")
    warning_text = _build_regime_warning_text(diagnostics, payload.unresolved)
    settings_row = payload.settings_df.iloc[0].to_dict() if payload.settings_df is not None and not payload.settings_df.empty else {}

    lines = [
        dmc.Text(f"Regime: {regime_name}", size="sm"),
        dmc.Text(f"Method type: {method_type if method_type is not None else '—'}", size="sm"),
        dmc.Text(f"Regimes: {num_regimes if num_regimes is not None else '—'}", size="sm"),
        dmc.Text(f"Observations: {observations if observations is not None else '—'}", size="sm"),
        dmc.Text(f"Signal: {settings_row.get('Signal Label') or payload.signal_label or '—'}", size="sm"),
        dmc.Text(f"Signal basis: {settings_row.get('Signal Return Basis') or '—'}", size="sm"),
    ]
    if settings_row.get("PC1 Standardized") is not None:
        lines.append(
            dmc.Text(
                f"PC1 Standardized: {'Yes' if bool(settings_row.get('PC1 Standardized')) else 'No'}",
                size="sm",
            )
        )
    if warning_text:
        lines.append(dmc.Text(f"Warning: {warning_text}", size="sm", c="orange"))

    return dmc.Paper(
        withBorder=True,
        radius="md",
        p="sm",
        children=[
            dmc.Text("Regime Settings", fw=600, size="sm", mb=4),
            dmc.Stack(gap=2, children=lines),
        ],
    )


@callback(
    Output("at-regime-analysis-warning", "children"),
    Output("at-regime-analysis-container", "children"),
    Input("at-main-tabs", "value"),
    Input("at-regime-definition-select", "value"),
    Input("at-regime-detail-display-mode-select", "value"),
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
    State("at-regime-definitions-db-store", "data"),
    State("at-regime-definitions-local-store", "data"),
    State("at-regime-series-store", "data"),
    prevent_initial_call=True,
)
def update_regime_analysis(
    active_tab,
    regime_definition_key,
    regime_display_mode,
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
    theme,
    regime_definitions_db=None,
    regime_definitions_local=None,
    regime_series_store=None,
):
    if (
        active_tab != "regime_analysis"
        or not state_ready
        or not _has_complete_date_range(date_range)
    ):
        raise PreventUpdate

    if not raw_data:
        return None, dmc.Text("Load return data to run regime analysis.", size="sm", c="dimmed")
    if not selected_series:
        return None, dmc.Text("Select one or more series for regime statistics.", size="sm", c="dimmed")
    if not regime_definition_key:
        return None, dmc.Text("Select a regime definition.", size="sm", c="dimmed")

    build_result = _build_regime_analysis_payload(
        _raw_json(raw_data),
        periodicity,
        selected_series,
        returns_type,
        benchmark_assignments,
        long_short_assignments,
        date_range,
        vol_scaler,
        vol_scaling_assignments,
        regime_definition_key,
        regime_definitions_db,
        regime_definitions_local,
        regime_series_store,
    )
    if build_result.status != "ok" or build_result.payload is None:
        if build_result.status in {"no_source_data", "no_assignments"}:
            return (
                dmc.Alert(build_result.message or "No regime assignments were produced.", color="orange", variant="light"),
                dmc.Text("No regime assignments.", size="sm", c="dimmed"),
            )
        return None, dmc.Text(build_result.message or "Unable to build regime analysis.", size="sm", c="dimmed")

    payload = build_result.payload
    warning_children = []
    warning_text = str((payload.diagnostics or {}).get("warning") or "").strip()
    if warning_text:
        warning_children.append(dmc.Alert(warning_text, color="orange", variant="light", mb="sm"))
    if payload.unresolved:
        warning_children.append(
            dmc.Alert(
                f"Missing source series (not resolved from DB): {', '.join(payload.unresolved)}",
                color="orange",
                variant="light",
                mb="sm",
            )
        )

    display_mode = regime_display_mode if regime_display_mode in {"summary", "detail"} else "summary"
    if display_mode == "detail":
        detail_warning = _detail_render_warning(payload.detail_df)
        if payload.detail_df.empty:
            return None, dmc.Text("No regime raw detail is available for current settings.", size="sm", c="dimmed")
        if detail_warning is not None:
            warning_children.append(detail_warning)
        return (
            html.Div(warning_children),
            _build_regime_grid_component(
                "Regime Raw Detail",
                payload.detail_df,
                theme,
                percent_cols={str(col) for col in selected_series},
                integer_cols={"Regime"},
                pinned_cols={"Date", "Regime"},
                max_height=560,
            ),
        )
    timeline = payload.timeline_df
    if timeline.empty:
        return None, dmc.Text("No regime timeline available for current settings.", size="sm", c="dimmed")

    timeline_fig = go.Figure()
    timeline_fig.add_trace(
        go.Scatter(
            x=timeline["Date"],
            y=timeline["Regime"],
            mode="lines+markers",
            name="Regime",
            line={"shape": "hv", "width": 2},
            marker={"size": 4},
        )
    )
    timeline_fig.update_layout(
        title=f"Regime Timeline: {payload.definition.get('RegimeName')}",
        xaxis_title="Date",
        yaxis_title="Regime",
        yaxis={"dtick": 1},
        height=360,
        legend={"orientation": "v", "x": 1.02, "y": 1, "xanchor": "left", "yanchor": "top"},
    )
    apply_chart_theme(timeline_fig, theme)

    transition_display = (
        payload.transition_df.reset_index()
        if payload.transition_df is not None and not payload.transition_df.empty
        else pd.DataFrame()
    )
    transition_percent_cols = set(transition_display.columns[1:]) if not transition_display.empty else set()

    stack_children = [
        _build_regime_settings_text_component(payload),
        _build_regime_grid_component(
            "Regime Statistics",
            payload.stats_df,
            theme,
            percent_cols={
                "Mean Return",
                "Volatility",
                "Annualized Excess Return",
                "Annualized Tracking Error",
                "Min Return",
                "Max Return",
                "Hit Rate",
                "Hit Rate (vs Benchmark)",
                "Max Drawdown",
            },
            integer_cols={"Regime", "Observations"},
            pinned_cols={"Regime", "Series"},
            max_height=360,
        ),
        dcc.Graph(figure=timeline_fig, style={"height": "360px"}),
        _build_regime_grid_component(
            "Transition Matrix",
            transition_display,
            theme,
            percent_cols=transition_percent_cols,
            integer_cols={"From Regime"},
            max_height=260,
        ),
        _build_regime_grid_component(
            "Run Durations",
            payload.duration_df,
            theme,
            integer_cols={"Regime", "Runs", "Current Run Length"},
            max_height=260,
        ),
    ]

    return html.Div(warning_children), dmc.Stack(gap="sm", children=stack_children)


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
        df = get_working_returns_by_key(
            _dataset_key(raw_data) or "", periodicity or "daily", tuple(selected_series),
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
            _dataset_key(raw_data),
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
            _dataset_key(raw_data),
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
            _dataset_key(raw_data),
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

        fig = go.Figure()
        for series in drawdown_df.columns:
            drawdown = drawdown_df[series].dropna()
            if drawdown.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=drawdown.index,
                    y=drawdown,
                    mode="lines",
                    name=series,
                    fill="tozeroy",
                )
            )

        if not fig.data:
            return dmc.Text("No data available for selected series", size="sm", c="dimmed")

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
            _dataset_key(raw_data),
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
    State("at-use-risk-free-store", "data"),
    State("at-correlation-exp-wt-switch", "checked"),
    State("at-correlation-halflife-input", "value"),
    State("at-correlation-shrinkage-select", "value"),
    State("at-correlation-shrinkage-target-select", "value"),
    State("at-factor-series-select", "value"),
    State("at-factor-quantiles-input", "value"),
    State("at-factor-transform-select", "value"),
    State("at-conditional-view-store", "data"),
    State("at-conditional-comparator-store", "data"),
    State("at-conditional-threshold-store", "data"),
    State("at-conditional-window-conversion-store", "data"),
    State("at-conditional-step-store", "data"),
    State("at-conditional-step-unit-store", "data"),
    State("at-factor-definitions-db-store", "data"),
    State("at-factor-definitions-local-store", "data"),
    State("at-regime-definition-select", "value"),
    State("at-regime-definitions-db-store", "data"),
    State("at-regime-definitions-local-store", "data"),
    State("at-regime-series-store", "data"),
    State("dashmat-saved-series-cache-store", "data"),
    prevent_initial_call=True,
)
def download_excel(
    n_clicks,
    raw_data,
    original_periodicity,
    selected_periodicity,
    selected_series,
    returns_type,
    benchmark_assignments,
    long_short_assignments,
    date_range,
    rolling_window,
    rolling_return_type,
    monthly_view,
    monthly_series,
    vol_scaler,
    vol_scaling_assignments,
    use_risk_free,
    correlation_exp_wt,
    correlation_halflife,
    correlation_shrinkage,
    correlation_shrinkage_target,
    factor_series,
    factor_quantiles,
    factor_transform,
    conditional_view,
    conditional_comparator,
    conditional_threshold,
    conditional_window_conversion,
    conditional_step,
    conditional_step_unit,
    factor_definitions_db=None,
    factor_definitions_local=None,
    regime_definition_key=None,
    regime_definitions_db=None,
    regime_definitions_local=None,
    regime_series_store=None,
    saved_series_store=None,
):
    """Generate Excel file with core analytics sheets plus correlation/covariance matrices."""
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

        sheet_specs = _resolve_export_sheet_specs(
            bundle,
            original_periodicity,
            returns_type,
            benchmark_assignments,
            long_short_assignments,
            date_range,
            rolling_window,
            rolling_return_type,
            monthly_view,
            monthly_series,
            use_risk_free,
            correlation_exp_wt,
            correlation_halflife,
            correlation_shrinkage,
            correlation_shrinkage_target,
            factor_series,
            factor_quantiles,
            factor_transform,
            conditional_comparator,
            conditional_threshold,
            conditional_window_conversion,
            conditional_step,
            conditional_step_unit,
            factor_definitions_db,
            factor_definitions_local,
            regime_definition_key,
            regime_definitions_db,
            regime_definitions_local,
            regime_series_store,
            vol_scaling_assignments,
            saved_series_store,
        )
        if not sheet_specs:
            raise PreventUpdate

        output = BytesIO()
        with timed_block("analyticstool.download_excel.workbook"):
            with pd.ExcelWriter(output, engine="xlsxwriter", date_format="m/d/yyyy", datetime_format="m/d/yyyy") as writer:
                _write_export_sheet_specs(writer, sheet_specs)

        output.seek(0)

        # Generate filename
        periodicity_suffix = selected_periodicity.replace("_", "-") if selected_periodicity else "returns"
        returns_suffix = "excess" if returns_type == "excess" else "total"
        filename = f"dashmat_{periodicity_suffix}_{returns_suffix}.xlsx"

        return dcc.send_bytes(output.getvalue(), filename)


