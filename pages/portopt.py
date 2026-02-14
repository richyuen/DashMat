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
    Input, Output, State, callback, dcc, html, no_update,
    register_page, ALL, clientside_callback, callback_context,
)
from dash.exceptions import PreventUpdate

import cache_config
from utils.parsing import detect_periodicity, get_sheet_names, parse_uploaded_file
from utils.returns import (
    calculate_excess_returns,
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
from utils.optimization import run_portfolio_optimization, compute_risk_contributions, compute_efficient_frontier
from utils.perf_timing import timed_block
from utils.serialization import date_range_payload_for_cache, mapping_payload_for_cache
from utils.shared_metrics import (
    STATS_CONFIG,
    risk_free_json_from_store as _risk_free_json_from_store,
    spx_json_from_store as _spx_json_from_store,
)
from utils.statistics import (
    calculate_statistics_cached,
    annualized_return,
    annualized_return_calendar_days,
)
from utils.charting import apply_chart_theme
from utils.sample_data import get_sample_file_path
from utils.core_categories import (
    clear_dropdown_caches,
    get_cma_versions_cached,
    get_common_daily_range,
    get_cmabench_map_for_fofbench,
    get_core_category_options_cached,
    get_unique_cmabench_values_cached,
    load_cma_returns_for_benches_with_meta,
)
from dbengine import AG_GRID_LICENSE_KEY, engine as DB_ENGINE, engine_MRD as MRD_ENGINE

register_page(__name__, path="/portopt", name="Portfolio Optimization", title="Portfolio Optimization")

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
):
    """Compute risk-contribution rows for each optimization window."""
    series_tuple = tuple(selected_series or ())
    if working_df.empty or not series_tuple or not window_weights:
        return []

    working_subset = working_df.loc[:, list(series_tuple)]
    index = working_subset.index
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
        rc = compute_risk_contributions(
            {name: float(weights.get(name, 0) or 0) for name in valid_assets},
            window_returns,
        )
        rows.append(
            {
                "apply_start": apply_start,
                "apply_end": apply_end,
                "risk_contributions": rc,
            }
        )

    return rows


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
            return "Halflife must be greater than 0 when exponential weighting is enabled."

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

    if config.get("exp_wt_cov", False):
        hl = int(config.get("halflife", 63) or 63)
        ewm_cov = est_data.ewm(halflife=hl).cov().iloc[-len(asset_cols):]
        if isinstance(ewm_cov.index, pd.MultiIndex):
            ewm_cov.index = ewm_cov.index.get_level_values(-1)
        ewm_cov = ewm_cov.reindex(index=asset_cols, columns=asset_cols)
        port.cov = ewm_cov

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
    if model == "ex_ante_mv":
        custom_mu, custom_cov, error_msg = _build_ex_ante_mu_cov(config, actual_cols, ann)
        if error_msg:
            raise ValueError(error_msg)
    elif model == "black_litterman":
        custom_mu, custom_cov, error_msg = _build_black_litterman_mu_cov(est_data, config, actual_cols)
        if error_msg:
            raise ValueError(error_msg)

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
    return dmc.Stack(
        align="center",
        justify="center",
        h=400,
        children=[
            DashIconify(icon="tabler:chart-pie", width=60, color="#adb5bd"),
            dmc.Text("Portfolio Optimization", size="xl", fw=500, c="dimmed", mt="md"),
            dmc.Text("Add series from file to begin", size="sm", c="dimmed"),
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
                        id="po-welcome-add-db-btn",
                    ),
                    dmc.Button(
                        "Add series from file",
                        leftSection=DashIconify(icon="tabler:upload"),
                        variant="outline",
                        size="sm",
                        w=210,
                        id="po-welcome-add-series-btn",
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
                        id="po-download-sample-daily-btn",
                        size="sm",
                        variant="light",
                        w=210,
                    ),
                    dmc.Button(
                        "Sample Monthly File",
                        leftSection=DashIconify(icon="tabler:download"),
                        id="po-download-sample-monthly-btn",
                        size="sm",
                        variant="light",
                        w=210,
                    ),
                ],
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
                                            {"field": "Constraint", "editable": True, "width": 120, "headerClass": "center-header"},
                                            {"field": "Min", "editable": True, "width": 90, "type": "numericColumn", 
                                             "valueFormatter": {"function": "d3.format('.4f')(params.value)"}, "headerClass": "center-header"},
                                            {"field": "Max", "editable": True, "width": 90, "type": "numericColumn", 
                                             "valueFormatter": {"function": "d3.format('.4f')(params.value)"}, "headerClass": "center-header"},
                                        ],
                                        rowData=[],
                                        defaultColDef={"resizable": True, "sortable": False, "suppressHeaderMenuButton": True, "cellStyle": {"textAlign": "center"}, "headerClass": "center-header"},
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
                                            value="OptResult",
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
                                            dmc.NumberInput(
                                                id="po-halflife-input",
                                                value=63,
                                                min=1,
                                                step=1,
                                                w=90,
                                                size="sm",
                                                disabled=True,
                                                style={"whiteSpace": "nowrap"},
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
                                                    defaultColDef={"resizable": True, "sortable": False, "suppressHeaderMenuButton": True, "cellStyle": {"textAlign": "center"}, "headerClass": "center-header"},
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
                                                    "valueFormatter": {"function": "params.value !== null && params.value !== undefined && params.value !== '' && isFinite(Number(params.value)) ? d3.format('.4f')(Number(params.value)) : ''"}, "cellStyle": {"textAlign": "center"}, "headerClass": "center-header"},
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
                                                         "headerClass": "center-header"},
                                                        {"field": "Asset", "editable": True, "width": 150,
                                                         "headerClass": "center-header"},
                                                        {"field": "Asset_To", "editable": True, "width": 150,
                                                         "headerName": "vs Asset (rel)",
                                                         "headerClass": "center-header"},
                                                        {"field": "Return", "editable": True, "width": 100,
                                                         "type": "numericColumn",
                                                         "valueFormatter": {"function": "d3.format('.2f')(params.value) + '%'"},
                                                         "valueParser": {"function": "Number(params.newValue)"},
                                                         "headerClass": "center-header"},
                                                        {"field": "Confidence", "editable": True, "width": 100,
                                                         "type": "numericColumn",
                                                         "valueFormatter": {"function": "d3.format('.2f')(params.value)"},
                                                         "headerClass": "center-header"},
                                                    ],
                                                    rowData=[],
                                                    defaultColDef={"resizable": True, "sortable": False, "suppressHeaderMenuButton": True, "cellStyle": {"textAlign": "center"}, "headerClass": "center-header"},
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
                        size="sm",
                        style={"alignSelf": "flex-end", "marginBottom": "8px"},
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
                ],
            ),

            dmc.Tabs(
                id="po-vis-tabs",
                value="weight",
                style={"height": "600px", "display": "flex", "flexDirection": "column", "overflow": "hidden"},
                children=[
                    dmc.TabsList(children=[
                        dmc.TabsTab("Weights", value="weight"),
                        dmc.TabsTab("Turnover", value="turnover"),
                        dmc.TabsTab("Statistics", value="statistics"),
                        dmc.TabsTab("Returns", value="returns"),
                        dmc.TabsTab("Growth of $1", value="growth"),
                        dmc.TabsTab("Attribution", value="attribution"),
                        dmc.TabsTab("Risk", value="risk"),
                        dmc.TabsTab("Frontier", value="frontier"),
                    ]),
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
                                    dag.AgGrid(
                                        enableEnterpriseModules=True,
                                        licenseKey=AG_GRID_LICENSE_KEY,
                                        id="po-weight-grid",
                                        className='ag-theme-alpine',
                                        columnDefs=[],
                                        rowData=[],
                                        defaultColDef={"sortable": True, "resizable": True, "suppressHeaderMenuButton": True, "cellStyle": {"textAlign": "center"}, "headerClass": "center-header"},
                                        style={"height": "100%", "width": "100%"},
                                        dashGridOptions={"animateRows": True, "pagination": False, "suppressExcelExport": True, "enableRangeSelection": True, "suppressCsvExport": True},
                                    ),
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
                                children=[html.Div(id="po-attribution-chart-content")],
                            ),
                            html.Div(
                                id="po-attribution-grid-container",
                                style={"display": "none"},
                                children=[
                                    dag.AgGrid(
                                        enableEnterpriseModules=True,
                                        licenseKey=AG_GRID_LICENSE_KEY,
                                        id="po-attribution-grid",
                                        className='ag-theme-alpine',
                                        columnDefs=[],
                                        rowData=[],
                                        defaultColDef={"sortable": True, "resizable": True, "suppressHeaderMenuButton": True, "cellStyle": {"textAlign": "center"}, "headerClass": "center-header"},
                                        style={"height": "100%", "width": "100%"},
                                        dashGridOptions={"animateRows": True, "pagination": False, "suppressExcelExport": True, "enableRangeSelection": True, "suppressCsvExport": True},
                                    ),
                                ],
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
                                children=[html.Div(id="po-risk-chart-content")],
                            ),
                            html.Div(
                                id="po-risk-grid-container",
                                style={"display": "none"},
                                children=[
                                    dag.AgGrid(
                                        enableEnterpriseModules=True,
                                        licenseKey=AG_GRID_LICENSE_KEY,
                                        id="po-risk-grid",
                                        className='ag-theme-alpine',
                                        columnDefs=[],
                                        rowData=[],
                                        defaultColDef={"sortable": True, "resizable": True, "suppressHeaderMenuButton": True, "cellStyle": {"textAlign": "center"}, "headerClass": "center-header"},
                                        style={"height": "100%", "width": "100%"},
                                        dashGridOptions={"animateRows": True, "pagination": False, "suppressExcelExport": True, "enableRangeSelection": True, "suppressCsvExport": True},
                                    ),
                                ],
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
                                children=[html.Div(id="po-turnover-chart-content")],
                            ),
                            html.Div(
                                id="po-turnover-grid-container",
                                style={"display": "none"},
                                children=[
                                    dag.AgGrid(
                                        enableEnterpriseModules=True,
                                        licenseKey=AG_GRID_LICENSE_KEY,
                                        id="po-turnover-grid",
                                        className='ag-theme-alpine',
                                        columnDefs=[],
                                        rowData=[],
                                        defaultColDef={"sortable": True, "resizable": True, "suppressHeaderMenuButton": True, "cellStyle": {"textAlign": "center"}, "headerClass": "center-header"},
                                        style={"height": "100%", "width": "100%"},
                                        dashGridOptions={"animateRows": True, "pagination": False, "suppressExcelExport": True, "enableRangeSelection": True, "suppressCsvExport": True},
                                    ),
                                ],
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
                                children=[
                                    dcc.Loading(
                                        type="default",
                                        children=[html.Div(id="po-frontier-chart-content")],
                                    ),
                                ],
                            ),
                            html.Div(
                                id="po-frontier-grid-container",
                                style={"display": "none"},
                                children=[
                                    dag.AgGrid(
                                        enableEnterpriseModules=True,
                                        licenseKey=AG_GRID_LICENSE_KEY,
                                        id="po-frontier-grid",
                                        className='ag-theme-alpine',
                                        columnDefs=[],
                                        rowData=[],
                                        defaultColDef={"sortable": True, "resizable": True, "suppressHeaderMenuButton": True, "cellStyle": {"textAlign": "center"}, "headerClass": "center-header"},
                                        style={"height": "100%", "width": "100%"},
                                        dashGridOptions={"animateRows": True, "pagination": False, "suppressExcelExport": True, "enableRangeSelection": True, "suppressCsvExport": True},
                                    ),
                                ],
                            ),
                        ],
                    ),
                    dmc.TabsPanel(
                        value="statistics",
                        pt="md",
                        style={"flex": "1", "display": "flex", "flexDirection": "column", "overflow": "hidden"},
                        children=[
                            dag.AgGrid(
                                enableEnterpriseModules=True,
                                licenseKey=AG_GRID_LICENSE_KEY,
                                id="po-statistics-grid",
                                className='ag-theme-alpine',
                                columnDefs=[],
                                rowData=[],
                                defaultColDef={"sortable": True, "resizable": True, "suppressHeaderMenuButton": True, "cellStyle": {"textAlign": "center"}, "headerClass": "center-header"},
                                style={"height": "100%", "width": "100%"},
                                dashGridOptions={"animateRows": True, "suppressExcelExport": True, "enableRangeSelection": True, "suppressCsvExport": True},
                            ),
                        ],
                    ),
                    dmc.TabsPanel(
                        value="returns",
                        pt="md",
                        style={"flex": "1", "display": "flex", "flexDirection": "column", "overflow": "hidden"},
                        children=[
                            dag.AgGrid(
                                enableEnterpriseModules=True,
                                licenseKey=AG_GRID_LICENSE_KEY,
                                id="po-returns-grid",
                                className='ag-theme-alpine',
                                columnDefs=[],
                                rowData=[],
                                defaultColDef={"sortable": True, "resizable": True, "suppressHeaderMenuButton": True, "cellStyle": {"textAlign": "center"}, "headerClass": "center-header"},
                                style={"height": "100%", "width": "100%"},
                                dashGridOptions={"animateRows": True, "pagination": False, "suppressExcelExport": True, "enableRangeSelection": True, "suppressCsvExport": True},
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
                ],
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Page Layout
# ---------------------------------------------------------------------------

layout = dmc.Container(
    fluid=True,
    style={"minHeight": "calc(100vh - 55px)", "display": "flex", "flexDirection": "column", "overflow": "auto"},
    className='page-container',
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
                                    )
                                ),
                                dmc.MenuDropdown(className="dashmat-menu-dropdown", children=[
                                    dmc.MenuItem(
                                        "Add series from database...",
                                        id="po-menu-add-from-db",
                                        leftSection=DashIconify(icon="tabler:database", width=14),
                                    ),
                                    dmc.MenuItem(
                                        "Add series from file...",
                                        id="po-menu-add-series",
                                        leftSection=DashIconify(icon="tabler:upload", width=14),
                                    ),
                                    dmc.MenuDivider(),
                                    dmc.MenuItem(
                                        "Save Session",
                                        id="po-menu-save-session",
                                        leftSection=DashIconify(icon="tabler:device-floppy", width=14),
                                    ),
                                    dmc.MenuItem(
                                        "Load Session",
                                        id="po-menu-load-session",
                                        leftSection=DashIconify(icon="tabler:folder-open", width=14),
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
                                    )
                                ),
                                dmc.MenuDropdown(className="dashmat-menu-dropdown", children=[
                                    dmc.MenuItem(
                                        "Clear session storage and refresh",
                                        id="po-menu-clear-local-storage",
                                        leftSection=DashIconify(icon="tabler:trash", width=14),
                                    ),
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
                        dmc.Box(style={"flexGrow": 1}),
                        # Help button (opens User Guide)
                        dmc.Button(
                            "Help",
                            id="po-menu-help-guide",
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
            title=dmc.Group(
                gap="xs",
                children=[
                    dmc.ThemeIcon(DashIconify(icon="tabler:list-check"), color="blue", variant="light", size="sm"),
                    dmc.Text("Select Series", fw=600, size="sm"),
                ],
            ),
            size="84vw",
            styles={"content": {"maxWidth": "1450px"}},
            centered=True,
            closeOnEscape=False,
            radius="lg",
            transitionProps={"transition": "fade", "duration": 200},
            className='series-modal-dark dashmat-modal',
            overlayProps={"blur": 2, "opacity": 0.45},
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
                    style={"maxHeight": "50vh"},
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

        # Sheet Selection Modal (for multi-tab Excel files)
        dmc.Modal(
            id="po-sheet-select-modal",
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
                    id="po-sheet-select-dropdown",
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
                        dmc.Button("Cancel", id="po-sheet-select-cancel-button", variant="outline", color="red"),
                        dmc.Button("OK", id="po-sheet-select-ok-button", color="blue"),
                    ],
                ),
            ],
        ),

        # Optimization status modal (progress → completion in one modal)
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

        # Add-from-database modal
        dmc.Modal(
            id="po-db-add-modal",
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
                    id="po-db-add-error-alert",
                    title="Cannot add series",
                    color="red",
                    hide=True,
                    mb="sm",
                ),
                dmc.MultiSelect(
                    id="po-db-add-series-select",
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
                        dmc.Button("Cancel", id="po-db-add-cancel-button", variant="outline", color="red"),
                        dmc.Button("OK", id="po-db-add-ok-button", color="blue", disabled=True),
                    ],
                ),
            ],
        ),

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

        # Help Modal
        dmc.Modal(
            id="po-help-modal",
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
                dmc.Stack(
                    gap="md",
                    children=[
                        dmc.Paper(
                            withBorder=True,
                            radius="md",
                            p="sm",
                            bg="var(--mantine-color-body)",
                            children=dmc.Group(
                                justify="flex-start",
                                align="center",
                                children=[
                                    dmc.Group(
                                        gap="xs",
                                        children=[
                                            dmc.ThemeIcon(DashIconify(icon="tabler:book"), variant="light", color="blue", size="md"),
                                            dmc.Stack(
                                                gap=0,
                                                children=[
                                                    dmc.Text("Portfolio Optimization Guide", fw=600, size="sm"),
                                                    dmc.Text("Use Basic for workflow setup and Advanced for model/constraint details.", size="xs", c="dimmed"),
                                                ],
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                        ),
                        dmc.Tabs(
                            value="basic",
                            variant="outline",
                            color="blue",
                            children=[
                                dmc.TabsList(
                                    children=[
                                        dmc.TabsTab([DashIconify(icon="tabler:compass", width=14), "Basic Guide"], value="basic"),
                                        dmc.TabsTab([DashIconify(icon="tabler:settings-cog", width=14), "Advanced Guide"], value="advanced"),
                                    ],
                                ),
                                dmc.TabsPanel(
                                    value="basic",
                                    pt="sm",
                                    children=dmc.Accordion(
                                        variant="separated",
                                        children=[
                                            dmc.AccordionItem(
                                                value="basic-quick-start",
                                                children=[
                                                    dmc.AccordionControl("Quick Start"),
                                                    dmc.AccordionPanel(dmc.Stack(gap="xs", children=[
                                                        dmc.Text("1) File > Add series from file, then choose the series used as portfolio assets.", size="sm"),
                                                        dmc.Text("2) Open Select Series and configure include/exclude, order, benchmark, long-short, vol scaling, min/max weight, and force max.", size="sm"),
                                                        dmc.Text("3) Set Periodicity, Vol Scaler, and Date Range so estimation and backtest use the intended sample.", size="sm"),
                                                        dmc.Text("4) Choose Model and controls (window, step, missing-data handling, and optional exponential weighting).", size="sm"),
                                                        dmc.Text("5) Click Run to create a named portfolio. Run additional scenarios using different names.", size="sm"),
                                                        dmc.Text("6) Review Weights, Turnover, Statistics, Returns, Growth, Attribution, Risk, and Frontier tabs.", size="sm"),
                                                    ])),
                                                ],
                                            ),
                                            dmc.AccordionItem(
                                                value="basic-data",
                                                children=[
                                                    dmc.AccordionControl("Data Requirements"),
                                                    dmc.AccordionPanel(dmc.Stack(gap="xs", children=[
                                                        dmc.Text("Input files should be date-indexed return series with one column per asset.", size="sm"),
                                                        dmc.Text("Supported uploads: CSV, XLS, XLSX. If a workbook has multiple sheets, pick one sheet before import.", size="sm"),
                                                        dmc.Text("Values may be decimals or percent-style values; parsing normalizes input.", size="sm"),
                                                        dmc.Text("Periodicity is auto-detected. Daily data can be resampled to weekly/monthly; monthly is not upsampled.", size="sm"),
                                                        dmc.Text("Date overlap across selected series affects the usable sample and output availability.", size="sm"),
                                                    ])),
                                                ],
                                            ),
                                            dmc.AccordionItem(
                                                value="basic-series-selection",
                                                children=[
                                                    dmc.AccordionControl("Series Selection Modal"),
                                                    dmc.AccordionPanel(dmc.Stack(gap="xs", children=[
                                                        dmc.Text("Include checkbox controls whether a series participates in optimization.", size="sm"),
                                                        dmc.Text("Drag and drop rows to set display order in series lists and output tables.", size="sm"),
                                                        dmc.Text("Benchmark sets comparison series used by excess-return and long-short behavior.", size="sm"),
                                                        dmc.Text("L/S (Long-Short) transforms selected series to (series - benchmark).", size="sm"),
                                                        dmc.Text("Scale Vol toggles per-series use of the global Vol Scaler target.", size="sm"),
                                                        dmc.Text("Min Wt / Max Wt are per-asset hard bounds used by the optimizer.", size="sm"),
                                                        dmc.Text("Force Max pins an asset to Max Wt and can make optimization infeasible.", size="sm"),
                                                        dmc.Text("Delete removes a series from the working dataset and can invalidate prior results.", size="sm"),
                                                    ])),
                                                ],
                                            ),
                                            dmc.AccordionItem(
                                                value="basic-controls",
                                                children=[
                                                    dmc.AccordionControl("Controls and Preprocessing"),
                                                    dmc.AccordionPanel(dmc.Stack(gap="xs", children=[
                                                        dmc.Text("Periodicity converts returns to analysis frequency used in optimization and tabs.", size="sm"),
                                                        dmc.Text("Vol Scaler applies target annualized volatility scaling. Set 0 to disable.", size="sm"),
                                                        dmc.Text("Date Range limits sample prior to optimization window logic.", size="sm"),
                                                        dmc.Text("Common Range uses only dates where all selected series overlap.", size="sm"),
                                                        dmc.Text("Common Daily jumps to overlap where all selected series are in daily phase and sets periodicity to Daily (Trading).", size="sm"),
                                                        dmc.Text("Max Range uses earliest start to latest end across selected series.", size="sm"),
                                                    ])),
                                                ],
                                            ),
                                            dmc.AccordionItem(
                                                value="basic-optimization",
                                                children=[
                                                    dmc.AccordionControl("Optimization Controls"),
                                                    dmc.AccordionPanel(dmc.Stack(gap="xs", children=[
                                                        dmc.Text("Portfolio Name is the key used to store and select results.", size="sm"),
                                                        dmc.Text("Model chooses risk-based, mean-variance, ex-ante, or Black-Litterman optimization.", size="sm"),
                                                        dmc.Text("Exp Wt enables exponential weighting for historical parameter estimates.", size="sm"),
                                                        dmc.Text("Half-Life controls recency emphasis when Exp Wt is on (smaller means faster decay).", size="sm"),
                                                        dmc.Text("Window options: Expanding, Rolling, or Full.", size="sm"),
                                                        dmc.Text("Window Size sets lookback periods used for each optimization step.", size="sm"),
                                                        dmc.Text("Opt Step + Unit sets rebalance frequency; Months aligns to month-end, Periods uses row counts.", size="sm"),
                                                        dmc.Text("Missing Data: Fill NA forward-fills; Fill 0 treats missing returns as zero.", size="sm"),
                                                        dmc.Text("Run executes optimization and stores results without deleting prior portfolios.", size="sm"),
                                                    ])),
                                                ],
                                            ),
                                            dmc.AccordionItem(
                                                value="basic-tabs",
                                                children=[
                                                    dmc.AccordionControl("Reading the Tabs"),
                                                    dmc.AccordionPanel(dmc.Stack(gap="xs", children=[
                                                        dmc.Text("Weights: allocation by asset over time, chart or table.", size="sm"),
                                                        dmc.Text("Turnover: absolute allocation changes per rebalance window.", size="sm"),
                                                        dmc.Text("Statistics: portfolio-level performance and risk metrics.", size="sm"),
                                                        dmc.Text("Returns: return time series grid.", size="sm"),
                                                        dmc.Text("Growth of $1: compounded path from initial value 1.", size="sm"),
                                                        dmc.Text("Attribution: asset-level return contribution.", size="sm"),
                                                        dmc.Text("Risk: asset-level risk contribution across windows.", size="sm"),
                                                        dmc.Text("Frontier: efficient frontier with active-portfolio marker.", size="sm"),
                                                    ])),
                                                ],
                                            ),
                                        ],
                                    ),
                                ),
                                dmc.TabsPanel(
                                    value="advanced",
                                    pt="sm",
                                    children=dmc.Accordion(
                                        variant="separated",
                                        children=[
                                            dmc.AccordionItem(
                                                value="adv-models",
                                                children=[
                                                    dmc.AccordionControl("Model Details"),
                                                    dmc.AccordionPanel(dmc.Stack(gap="xs", children=[
                                                        dmc.Text([dmc.Text("Risk Parity", fw=700, span=True), " - balances total risk contribution across assets."], size="sm"),
                                                        dmc.Text([dmc.Text("Factor Risk Parity", fw=700, span=True), " - balances risk through factor structure rather than only pairwise covariance."], size="sm"),
                                                        dmc.Text([dmc.Text("Hierarchical Risk Parity", fw=700, span=True), " - clusters assets and allocates top-down for stability under noisy covariance estimates."], size="sm"),
                                                        dmc.Text([dmc.Text("Maximize Sharpe Ratio", fw=700, span=True), " - seeks highest expected return per unit volatility."], size="sm"),
                                                        dmc.Text([dmc.Text("Minimize Variance", fw=700, span=True), " - seeks lowest portfolio volatility under constraints."], size="sm"),
                                                        dmc.Text([dmc.Text("Minimize CVaR", fw=700, span=True), " - emphasizes downside tail-risk control."], size="sm"),
                                                        dmc.Text([dmc.Text("Equal Weight", fw=700, span=True), " - 1/N baseline with no estimation-driven optimizer."], size="sm"),
                                                        dmc.Text([dmc.Text("Ex Ante Mean-Variance", fw=700, span=True), " - optimizes from user-provided forward-looking assumptions."], size="sm"),
                                                        dmc.Text([dmc.Text("Black-Litterman", fw=700, span=True), " - blends prior assumptions with user views and confidence."], size="sm"),
                                                    ])),
                                                ],
                                            ),
                                            dmc.AccordionItem(
                                                value="adv-linear-constraints",
                                                children=[
                                                    dmc.AccordionControl("Linear Constraints"),
                                                    dmc.AccordionPanel(dmc.Stack(gap="xs", children=[
                                                        dmc.Text("Add Constraint creates a row with Min/Max bounds and one coefficient column per selected asset.", size="sm"),
                                                        dmc.Text("Each row enforces Min <= sum(coef_i * weight_i) <= Max.", size="sm"),
                                                        dmc.Text("Example: Equity=1, Credit=1, others=0, Min=0.40, Max=0.70 constrains combined exposure.", size="sm"),
                                                        dmc.Text("Constraint name is informational; coefficients and Min/Max drive behavior.", size="sm"),
                                                    ])),
                                                ],
                                            ),
                                            dmc.AccordionItem(
                                                value="adv-ex-ante",
                                                children=[
                                                    dmc.AccordionControl("Ex Ante Inputs"),
                                                    dmc.AccordionPanel(dmc.Stack(gap="xs", children=[
                                                        dmc.Text("Available for Ex Ante Mean-Variance and Black-Litterman.", size="sm"),
                                                        dmc.Text("Objective applies to Ex Ante Mean-Variance (maximize Sharpe, minimize variance, maximize return).", size="sm"),
                                                        dmc.Text("Input Mode toggles ex-ante inputs between return/covariance and return/volatility/correlation.", size="sm"),
                                                        dmc.Text("Expected Returns grid accepts per-asset assumptions; volatility is editable in ret_vol_corr mode.", size="sm"),
                                                        dmc.Text("Matrix grid captures covariance (ret_cov) or correlation (ret_vol_corr).", size="sm"),
                                                        dmc.Text("Use upload and estimate helpers to seed assumptions; verify magnitudes and symmetry before run.", size="sm"),
                                                    ])),
                                                ],
                                            ),
                                            dmc.AccordionItem(
                                                value="adv-black-litterman",
                                                children=[
                                                    dmc.AccordionControl("Black-Litterman Views"),
                                                    dmc.AccordionPanel(dmc.Stack(gap="xs", children=[
                                                        dmc.Text("Use absolute views (asset expected return) or relative views (asset expected to outperform another).", size="sm"),
                                                        dmc.Text("Absolute example: Asset=SPY, Return=3 means expected return is 3%.", size="sm"),
                                                        dmc.Text("Relative example: Asset=QQQ, Asset_To=SPY, Return=2 means QQQ expected to beat SPY by 2%.", size="sm"),
                                                        dmc.Text("Confidence controls view strength. Tau controls prior uncertainty weighting.", size="sm"),
                                                        dmc.Text("Add View and Clear Views manage rows quickly before running optimization.", size="sm"),
                                                    ])),
                                                ],
                                            ),
                                            dmc.AccordionItem(
                                                value="adv-feasibility",
                                                children=[
                                                    dmc.AccordionControl("Constraints and Feasibility"),
                                                    dmc.AccordionPanel(dmc.Stack(gap="xs", children=[
                                                        dmc.Text("Per-asset Min/Max bounds are hard optimizer limits.", size="sm"),
                                                        dmc.Text("Force Max pins assets and reduces feasible region.", size="sm"),
                                                        dmc.Text("Linear constraints further restrict feasible portfolios.", size="sm"),
                                                        dmc.Text("Infeasibility is common when mins are too high, maxes too low, or too many constraints are combined.", size="sm"),
                                                        dmc.Text("To recover feasibility: relax mins/maxes, remove forced weights, and simplify linear constraints.", size="sm"),
                                                    ])),
                                                ],
                                            ),
                                            dmc.AccordionItem(
                                                value="adv-frontier",
                                                children=[
                                                    dmc.AccordionControl("Frontier Details"),
                                                    dmc.AccordionPanel(dmc.Stack(gap="xs", children=[
                                                        dmc.Text("Frontier window selector lets you inspect different estimation windows.", size="sm"),
                                                        dmc.Text("Risk measure selector controls frontier x-axis risk metric.", size="sm"),
                                                        dmc.Text("For Ex Ante and Black-Litterman models, frontier uses configured assumptions where available.", size="sm"),
                                                        dmc.Text("If data or assumptions are insufficient for a selected window, the tab will show a fallback message.", size="sm"),
                                                    ])),
                                                ],
                                            ),
                                            dmc.AccordionItem(
                                                value="adv-results",
                                                children=[
                                                    dmc.AccordionControl("Results, Session, and Troubleshooting"),
                                                    dmc.AccordionPanel(dmc.Stack(gap="xs", children=[
                                                        dmc.Text("Portfolio dropdown selects the active stored result; compare control overlays portfolios where supported.", size="sm"),
                                                        dmc.Text("Delete icon removes selected result. Use unique names to keep scenario history.", size="sm"),
                                                        dmc.Text("File > Save Session exports JSON state; Load Session restores it.", size="sm"),
                                                        dmc.Text("File > Download Excel exports portfolio outputs for external analysis.", size="sm"),
                                                        dmc.Text("Run disabled: load data and select at least one series.", size="sm"),
                                                        dmc.Text("Optimization fails: review bounds, force-max flags, and linear constraints first.", size="sm"),
                                                        dmc.Text("Turnover missing: Full window mode does not generate multiple rebalance events.", size="sm"),
                                                    ])),
                                                ],
                                            ),
                                        ],
                                    ),
                                ),
                            ],
                        ),
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
        # Temp stores for sheet selection (stash upload while user picks a tab)
        dcc.Store(id="po-sheet-select-contents-store", data=None),
        dcc.Store(id="po-sheet-select-filename-store", data=None),
        # Controls stores
        dcc.Store(id="po-periodicity-value-store", data="daily_trading", storage_type="session"),
        dcc.Store(id="po-periodicity-load-sync-dummy", data=None),
        dcc.Store(id="po-vol-scaler-value-store", data=0, storage_type="session"),
        dcc.Store(id="po-date-range-store", data=None, storage_type="session"),
        dcc.Store(id="po-series-select-value-store", data=[], storage_type="session"),
        # Optimization stores
        dcc.Store(id="po-opt-window-store", data="rolling", storage_type="session"),
        dcc.Store(id="po-window-size-store", data=252, storage_type="session"),
        dcc.Store(id="po-opt-step-store", data=1, storage_type="session"),
        dcc.Store(id="po-opt-step-unit-store", data="months", storage_type="session"),
        dcc.Store(id="po-opt-model-store", data="risk_parity", storage_type="session"),
        dcc.Store(id="po-portfolio-name-store", data="OptResult", storage_type="session"),
        dcc.Store(id="po-exp-wt-cov-store", data=False, storage_type="session"),
        dcc.Store(id="po-halflife-store", data=63, storage_type="session"),
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
        # Chart/table switch stores
        dcc.Store(id="po-weight-chart-switch-store", data="chart", storage_type="session"),
        dcc.Store(id="po-attribution-chart-switch-store", data="chart", storage_type="session"),
        dcc.Store(id="po-risk-chart-switch-store", data="chart", storage_type="session"),
        dcc.Store(id="po-turnover-chart-switch-store", data="chart", storage_type="session"),
        dcc.Store(id="po-frontier-chart-switch-store", data="chart", storage_type="session"),
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
        dcc.Download(id="po-download-sample-daily"),
        dcc.Download(id="po-download-sample-monthly"),
        # Navigation
        dcc.Location(id="po-url-location", refresh=False),
        # One-shot interval to trigger visibility check after session-storage hydration
        dcc.Interval(id="po-page-load-trigger", interval=50, max_intervals=1, n_intervals=0),

        # UI Blocker for file dialog (Overlay)
        dcc.Store(id="po-ui-blocker-store", data=False),
        dcc.Interval(id="po-ui-blocker-timeout", interval=15000, disabled=True),  # 15 second timeout
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

# Open Help modal
clientside_callback(
    "function(n) { return true; }",
    Output("po-help-modal", "opened"),
    Input("po-menu-help-guide", "n_clicks"),
    prevent_initial_call=True,
)


@callback(
    Output("po-download-sample-daily", "data"),
    Input("po-download-sample-daily-btn", "n_clicks"),
    prevent_initial_call=True,
)
def po_download_sample_daily(n_clicks):
    """Download stored sample daily returns file."""
    if n_clicks is None:
        raise PreventUpdate
    return dcc.send_file(str(get_sample_file_path("daily")))


@callback(
    Output("po-download-sample-monthly", "data"),
    Input("po-download-sample-monthly-btn", "n_clicks"),
    prevent_initial_call=True,
)
def po_download_sample_monthly(n_clicks):
    """Download stored sample monthly returns file."""
    if n_clicks is None:
        raise PreventUpdate
    return dcc.send_file(str(get_sample_file_path("monthly")))

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
                'po-cmabench-assignments-store',
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
                'po-opt-step-unit-store',
                'po-opt-model-store',
                'po-portfolio-name-store',
                'po-exp-wt-cov-store',
                'po-halflife-store',
                'po-missing-data-store',
                'po-fill-in-sample-store',
                'po-results-store',
                'po-active-tab-store',
                'po-weight-chart-switch-store',
                'po-attribution-chart-switch-store',
                'po-risk-chart-switch-store',
                'po-turnover-chart-switch-store'
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
    Input("po-menu-add-from-db", "n_clicks"),
    Input("po-welcome-add-db-btn", "n_clicks"),
    prevent_initial_call=True,
)
def po_open_db_add_modal(menu_clicks, welcome_clicks):
    if not menu_clicks and not welcome_clicks:
        raise PreventUpdate
    options = get_core_category_options_cached(DB_ENGINE)
    return True, options, []


@callback(
    Output("po-db-add-modal", "opened", allow_duplicate=True),
    Output("po-db-add-series-select", "value", allow_duplicate=True),
    Input("po-db-add-cancel-button", "n_clicks"),
    prevent_initial_call=True,
)
def po_close_db_add_modal(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    return False, []


@callback(
    Output("po-db-add-error-alert", "children"),
    Output("po-db-add-error-alert", "hide"),
    Output("po-db-add-ok-button", "disabled"),
    Input("po-db-add-series-select", "value"),
    Input("analyticstool-raw-data-store", "data"),
    Input("po-db-add-modal", "opened"),
    prevent_initial_call=True,
)
def po_validate_db_add_selection(selected_benches, raw_data, opened):
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

# Trigger upload from menu
clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks) {
            setTimeout(function() {
                var uploadDiv = document.getElementById('po-upload-data');
                if (uploadDiv) {
                    var input = uploadDiv.querySelector('input[type="file"]');
                    if (input) {
                        // Listen for window focus to detect cancel
                        var onFocus = function() {
                            window.removeEventListener('focus', onFocus);
                            setTimeout(function() {
                                if (!input.files || input.files.length === 0) {
                                    window.dash_clientside.set_props('po-ui-blocker-store', {data: false});
                                    window.dash_clientside.set_props('po-ui-blocker-timeout', {disabled: true});
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
    Output("po-ui-blocker-store", "data", allow_duplicate=True),
    Output("po-ui-blocker-timeout", "disabled", allow_duplicate=True),
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
                    if (input) {
                        var onFocus = function() {
                            window.removeEventListener('focus', onFocus);
                            setTimeout(function() {
                                if (!input.files || input.files.length === 0) {
                                    window.dash_clientside.set_props('po-ui-blocker-store', {data: false});
                                    window.dash_clientside.set_props('po-ui-blocker-timeout', {disabled: true});
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
    Output("po-ui-blocker-store", "data", allow_duplicate=True),
    Output("po-ui-blocker-timeout", "disabled", allow_duplicate=True),
    Input("po-welcome-add-series-btn", "n_clicks"),
    prevent_initial_call=True,
)

# UI Blocker: timeout fallback
clientside_callback(
    """
    function(n) {
        // Hide Blocker (False), Disable Timeout (True)
        return [false, true];
    }
    """,
    Output("po-ui-blocker-store", "data", allow_duplicate=True),
    Output("po-ui-blocker-timeout", "disabled", allow_duplicate=True),
    Input("po-ui-blocker-timeout", "n_intervals"),
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

# Store sync: periodicity
clientside_callback(
    "function(value) { return value; }",
    Output("po-periodicity-value-store", "data"),
    Input("po-periodicity-select", "value"),
    prevent_initial_call=True,
)

# Sync periodicity to Analytics only on raw-data load/update events.
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
        sessionStorage.setItem("periodicity-value-store", JSON.stringify(periodicityValue));
        return periodicityValue;
    }
    """,
    Output("po-periodicity-load-sync-dummy", "data"),
    Input("analyticstool-raw-data-store", "data"),
    Input("po-periodicity-value-store", "data"),
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

# Toggle halflife disabled based on exp wt cov switch
clientside_callback(
    "function(checked) { return !checked; }",
    Output("po-halflife-input", "disabled"),
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

# Store sync: opt step unit
clientside_callback(
    "function(value) { return value; }",
    Output("po-opt-step-unit-store", "data"),
    Input("po-opt-step-unit-select", "value"),
    prevent_initial_call=True,
)

# Store sync: opt window
clientside_callback(
    "function(value) { return value; }",
    Output("po-opt-window-store", "data"),
    Input("po-opt-window-select", "value"),
    prevent_initial_call=True,
)

# Store sync: window size
clientside_callback(
    "function(value) { return value; }",
    Output("po-window-size-store", "data"),
    Input("po-window-size-input", "value"),
    prevent_initial_call=True,
)

# Store sync: opt step
clientside_callback(
    "function(value) { return value; }",
    Output("po-opt-step-store", "data"),
    Input("po-opt-step-input", "value"),
    prevent_initial_call=True,
)

# Store sync: opt model
clientside_callback(
    "function(value) { return value; }",
    Output("po-opt-model-store", "data"),
    Input("po-opt-model-select", "value"),
    prevent_initial_call=True,
)

# Store sync: portfolio name
clientside_callback(
    "function(value) { return value; }",
    Output("po-portfolio-name-store", "data"),
    Input("po-portfolio-name-input", "value"),
    prevent_initial_call=True,
)

# Store sync: exp wt cov
clientside_callback(
    "function(checked) { return checked; }",
    Output("po-exp-wt-cov-store", "data"),
    Input("po-exp-wt-cov-switch", "checked"),
    prevent_initial_call=True,
)

# Store sync: halflife
clientside_callback(
    "function(value) { return value; }",
    Output("po-halflife-store", "data"),
    Input("po-halflife-input", "value"),
    prevent_initial_call=True,
)

# Store sync: missing data
clientside_callback(
    "function(value) { return value; }",
    Output("po-missing-data-store", "data"),
    Input("po-missing-data-select", "value"),
    prevent_initial_call=True,
)

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

# Store sync: objective
clientside_callback(
    "function(value) { return value; }",
    Output("po-objective-store", "data"),
    Input("po-objective-select", "value"),
    prevent_initial_call=True,
)

# Store sync: BL tau
clientside_callback(
    "function(value) { return value; }",
    Output("po-bl-tau-store", "data"),
    Input("po-bl-tau-input", "value"),
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
        {"field": "Asset", "editable": False, "width": 140, "headerClass": "center-header"},
        {"field": "Return", "editable": True, "width": 110,
         "type": "numericColumn",
         "valueFormatter": {"function": "d3.format('.2%')(params.value)"},
         "valueParser": {"function": "var v=params.newValue; if (v===null || v===undefined || v==='') return null; var n=Number(v); if (!isFinite(n)) return null; return Math.abs(n) > 1 ? n/100 : n;"},
         "headerClass": "center-header"},
        {"field": "Volatility", "editable": True, "width": 110,
         "type": "numericColumn",
         "valueFormatter": {"function": "d3.format('.2%')(params.value)"},
         "valueParser": {"function": "var v=params.newValue; if (v===null || v===undefined || v==='') return null; var n=Number(v); if (!isFinite(n)) return null; return Math.abs(n) > 1 ? n/100 : n;"},
         "hide": hide_vol,
         "headerClass": "center-header"},
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


# Update ex ante mode store
@callback(
    Output("po-ex-ante-mode-store", "data"),
    Input("po-ex-ante-mode-select", "value"),
)
def po_update_ex_ante_mode_store(value):
    return value or "ret_cov"


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
    """Populate the matrix grid structure. Does NOT auto-estimate — use Estimate from Data button."""
    if not selected_series:
        return [], []
    
    mode = mode or "ret_cov"
    is_corr = (mode == "ret_vol_corr")
    
    existing_matrix = corr_store if is_corr else cov_store
    existing_matrix = existing_matrix or {}

    matrix_defs = [{"field": "Asset", "editable": False, "width": 140, "pinned": "left",
                    "valueFormatter": {"function": "params.value"}, "headerClass": "center-header"}]
    for s in selected_series:
        matrix_defs.append({
            "field": s,
            "editable": True, 
            "width": 110,
            "type": "numericColumn",
            "valueFormatter": {"function": "params.value !== null && params.value !== undefined && params.value !== '' && isFinite(Number(params.value)) ? d3.format(',.4f')(Number(params.value)) : ''"},
            "headerClass": "center-header",
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
    State("analyticstool-raw-data-store", "data"),
    State("po-series-select", "data"),
    State("po-ex-ante-mode-store", "data"),
    State("po-periodicity-select", "value"),
    State("po-exp-wt-cov-switch", "checked"),
    State("po-halflife-input", "value"),
    prevent_initial_call=True,
)
def po_estimate_matrix_store(n_clicks, data, selected_series, mode, periodicity, exp_wt_cov, halflife):
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
        
        if is_corr:
            est_df = sub_df.corr()
        else:
            # Annualize covariance based on selected periodicity
            p = periodicity or "daily"
            if p.startswith("weekly"):
                ann = 52
            elif p == "monthly":
                ann = 12
            else:
                ann = 252
            
            if exp_wt_cov:
                n_assets = len(valid_series)
                hl = halflife or 63
                est_df = sub_df.ewm(halflife=hl).cov().iloc[-n_assets:]
                est_df.index = valid_series  # Reset MultiIndex to simple asset names
                est_df = est_df * ann
            else:
                est_df = sub_df.cov() * ann
            
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
    State("analyticstool-raw-data-store", "data"),
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
            hl = halflife or 63
            mean_returns = sub_df.ewm(halflife=hl).mean().iloc[-1] * ann
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
        {"field": "Constraint", "editable": True, "width": 120, "headerClass": "center-header"},
        {"field": "Min", "editable": True, "width": 90, "type": "numericColumn", 
         "valueFormatter": {"function": "d3.format('.4f')(params.value)"}, "headerClass": "center-header"},
        {"field": "Max", "editable": True, "width": 90, "type": "numericColumn", 
         "valueFormatter": {"function": "d3.format('.4f')(params.value)"}, "headerClass": "center-header"},
    ]
    
    for s in selected_series:
        cols.append({
            "field": s,
            "editable": True,
            "width": 100,
            "type": "numericColumn",
            "valueFormatter": {"function": "d3.format('.4f')(params.value)"},
            "headerClass": "center-header",
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
        if (tab === "growth" || tab === "statistics" || tab === "returns") {
            return [{display: "none"}, {display: "none"}, {display: "block"}];
        }
        if (tab === "frontier") {
            return [{display: "block"}, {display: "none"}, {display: "none"}];
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

# Clientside callback for weight chart switch storage
clientside_callback(
    "function(value) { return value !== null && value !== undefined ? value : 'chart'; }",
    Output("po-weight-chart-switch-store", "data"),
    Input("po-weight-chart-switch", "value"),
    prevent_initial_call=True,
)

# Clientside callback for weight view toggle
clientside_callback(
    """
    function(view_type) {
        var flex_style = {display: "flex", flexDirection: "column", flex: "1", overflow: "hidden"};
        var flex_scroll_style = {display: "flex", flexDirection: "column", flex: "1", overflow: "auto"};
        if (view_type === "chart") {
            return [{display: "none"}, flex_scroll_style];
        } else {
            return [flex_style, {display: "none"}];
        }
    }
    """,
    Output("po-weight-grid-container", "style"),
    Output("po-weight-chart-container", "style"),
    Input("po-weight-chart-switch", "value"),
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

# Clientside callback for attribution chart switch storage
clientside_callback(
    "function(value) { return value !== null && value !== undefined ? value : 'chart'; }",
    Output("po-attribution-chart-switch-store", "data"),
    Input("po-attribution-chart-switch", "value"),
    prevent_initial_call=True,
)

# Clientside callback for attribution view toggle
clientside_callback(
    """
    function(view_type) {
        var flex_style = {display: "flex", flexDirection: "column", flex: "1", overflow: "hidden"};
        var flex_scroll_style = {display: "flex", flexDirection: "column", flex: "1", overflow: "auto"};
        if (view_type === "chart") {
            return [{display: "none"}, flex_scroll_style];
        } else {
            return [flex_style, {display: "none"}];
        }
    }
    """,
    Output("po-attribution-grid-container", "style"),
    Output("po-attribution-chart-container", "style"),
    Input("po-attribution-chart-switch", "value"),
    prevent_initial_call=True,
)

# Clientside callback for risk chart switch storage
clientside_callback(
    "function(value) { return value !== null && value !== undefined ? value : 'chart'; }",
    Output("po-risk-chart-switch-store", "data"),
    Input("po-risk-chart-switch", "value"),
    prevent_initial_call=True,
)

# Clientside callback for risk view toggle
clientside_callback(
    """
    function(view_type) {
        var flex_style = {display: "flex", flexDirection: "column", flex: "1", overflow: "hidden"};
        var flex_scroll_style = {display: "flex", flexDirection: "column", flex: "1", overflow: "auto"};
        if (view_type === "chart") {
            return [{display: "none"}, flex_scroll_style];
        } else {
            return [flex_style, {display: "none"}];
        }
    }
    """,
    Output("po-risk-grid-container", "style"),
    Output("po-risk-chart-container", "style"),
    Input("po-risk-chart-switch", "value"),
    prevent_initial_call=True,
)

# Clientside callback for turnover chart switch storage
clientside_callback(
    "function(value) { return value !== null && value !== undefined ? value : 'chart'; }",
    Output("po-turnover-chart-switch-store", "data"),
    Input("po-turnover-chart-switch", "value"),
    prevent_initial_call=True,
)

# Clientside callback for turnover view toggle
clientside_callback(
    """
    function(view_type) {
        var flex_style = {display: "flex", flexDirection: "column", flex: "1", overflow: "hidden"};
        var flex_scroll_style = {display: "flex", flexDirection: "column", flex: "1", overflow: "auto"};
        if (view_type === "chart") {
            return [{display: "none"}, flex_scroll_style];
        } else {
            return [flex_style, {display: "none"}];
        }
    }
    """,
    Output("po-turnover-grid-container", "style"),
    Output("po-turnover-chart-container", "style"),
    Input("po-turnover-chart-switch", "value"),
    prevent_initial_call=True,
)

# Clientside callback for frontier chart switch storage
clientside_callback(
    "function(value) { return value !== null && value !== undefined ? value : 'chart'; }",
    Output("po-frontier-chart-switch-store", "data"),
    Input("po-frontier-chart-switch", "value"),
    prevent_initial_call=True,
)

# Clientside callback for frontier view toggle
clientside_callback(
    """
    function(view_type) {
        var flex_style = {display: "flex", flexDirection: "column", flex: "1", overflow: "hidden"};
        var flex_scroll_style = {display: "flex", flexDirection: "column", flex: "1", overflow: "auto"};
        if (view_type === "chart") {
            return [{display: "none"}, flex_scroll_style];
        } else {
            return [flex_style, {display: "none"}];
        }
    }
    """,
    Output("po-frontier-grid-container", "style"),
    Output("po-frontier-chart-container", "style"),
    Input("po-frontier-chart-switch", "value"),
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
    Output("po-save-session-dummy", "data"),
    Input("po-menu-save-session", "n_clicks"),
    prevent_initial_call=True,
)

# Load session: trigger hidden upload file dialog
clientside_callback(
    """
    function(n_clicks) {
        if (!n_clicks) return window.dash_clientside.no_update;
        setTimeout(function() {
            var el = document.querySelector('#po-load-session-upload input[type="file"]');
            if (el) el.click();
        }, 100);
        return window.dash_clientside.no_update;
    }
    """,
    Output("po-load-session-dummy", "data"),
    Input("po-menu-load-session", "n_clicks"),
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
# cross-page navigation, plus analyticstool-raw-data-store Input for same-page uploads.
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
    Input("analyticstool-raw-data-store", "data"),
)

# ---------------------------------------------------------------------------
# Restore application state when raw data loads
# ---------------------------------------------------------------------------

@callback(
    Output("po-periodicity-select", "data", allow_duplicate=True),
    Output("po-periodicity-select", "value", allow_duplicate=True),
    Output("po-vol-scaler-input", "value"),
    Output("po-series-select", "data"),
    Input("analyticstool-raw-data-store", "data"),
    State("analyticstool-original-periodicity-store", "data"),
    State("po-periodicity-value-store", "data"),
    State("po-series-select-value-store", "data"),
    State("po-vol-scaler-value-store", "data"),
    prevent_initial_call="initial_duplicate",
)
def po_restore_state(raw_data, orig_periodicity, stored_periodicity, stored_series, stored_vol):
    if not raw_data:
        raise PreventUpdate
    try:
        df = json_to_df(raw_data)
        if df is None or df.empty:
            raise PreventUpdate
            
        periodicity_options = get_available_periodicities(orig_periodicity or "daily")
        
        # Validate stored values
        valid_periodicity = stored_periodicity
        if valid_periodicity not in [p["value"] for p in periodicity_options]:
            valid_periodicity = "daily_trading" if orig_periodicity == "daily" else (orig_periodicity or "daily_trading")
            
        valid_vol = stored_vol if stored_vol is not None else 0
        
        # Validate series
        current_selection = stored_series or []
        valid_selection = [s for s in current_selection if s in df.columns]
        
        # If no stored selection matches, select all (default behavior)
        if not valid_selection:
            valid_selection = list(df.columns)
            
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


# ---------------------------------------------------------------------------
# Restore optimization controls from stores on page load
# ---------------------------------------------------------------------------

clientside_callback(
    """
    function(n, optWindow, windowSize, optStep, optStepUnit, model, name, expWt, halflife, missing, fillIS) {
        return [optWindow, windowSize, optStep, optStepUnit, model, name || "OptResult",
                expWt || false, halflife, !expWt, missing, fillIS];
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
    Output("po-halflife-input", "disabled", allow_duplicate=True),
    Output("po-missing-data-select", "value"),
    Output("po-fill-in-sample-select", "value"),
    Input("po-page-load-trigger", "n_intervals"),
    State("po-opt-window-store", "data"),
    State("po-window-size-store", "data"),
    State("po-opt-step-store", "data"),
    State("po-opt-step-unit-store", "data"),
    State("po-opt-model-store", "data"),
    State("po-portfolio-name-store", "data"),
    State("po-exp-wt-cov-store", "data"),
    State("po-halflife-store", "data"),
    State("po-missing-data-store", "data"),
    State("po-fill-in-sample-store", "data"),
    prevent_initial_call=True,
)


# ---------------------------------------------------------------------------
# Restore ex ante controls from stores on page load
# ---------------------------------------------------------------------------

clientside_callback(
    """
    function(n, mode, objective) {
        return [mode || "ret_cov", objective || "maximize_sharpe"];
    }
    """,
    Output("po-ex-ante-mode-select", "value"),
    Output("po-objective-select", "value"),
    Input("po-page-load-trigger", "n_intervals"),
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
    Input("po-portfolio-name-input", "value"),
    Input("po-series-select", "data"),
    Input("po-opt-model-select", "value"),
    Input("po-opt-window-select", "value"),
    Input("po-window-size-input", "value"),
    Input("po-opt-step-input", "value"),
    Input("po-opt-step-unit-select", "value"),
    Input("po-exp-wt-cov-switch", "checked"),
    Input("po-halflife-input", "value"),
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
    name,
    selected,
    opt_model,
    opt_window,
    window_size,
    opt_step,
    opt_step_unit,
    exp_wt_cov,
    halflife,
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

    # Save Session Button
    save_disabled = True
    if welcome_style and welcome_style.get("display") == "none":
        save_disabled = False
        
    # Download Excel Button
    download_disabled = True
    if results_data and len(results_data) > 0:
        download_disabled = False

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
    Output("analyticstool-raw-data-store", "data", allow_duplicate=True),
    Output("analyticstool-original-periodicity-store", "data", allow_duplicate=True),
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
    State("analyticstool-raw-data-store", "data"),
    State("analyticstool-original-periodicity-store", "data"),
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

        # Database import is treated as daily by design.
        new_periodicity = "daily"
        all_start_daily = True
        daily_transition_notes: list[str] = []
        for series_name in new_df.columns:
            meta = db_meta.get(series_name, {}) if isinstance(db_meta, dict) else {}
            starts_daily = bool(meta.get("starts_daily", True))
            if not starts_daily:
                all_start_daily = False
                daily_start_date = meta.get("daily_start_date")
                if daily_start_date:
                    daily_transition_notes.append(f"{series_name}: {daily_start_date}")
                else:
                    daily_transition_notes.append(f"{series_name}: no daily phase detected")

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
        if combined_periodicity == "daily":
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
    Output("analyticstool-raw-data-store", "data", allow_duplicate=True),
    Output("analyticstool-original-periodicity-store", "data", allow_duplicate=True),
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
    # Blocker outputs
    Output("po-ui-blocker-store", "data", allow_duplicate=True),
    Output("po-ui-blocker-timeout", "disabled", allow_duplicate=True),
    Input("po-upload-data", "contents"),
    State("po-upload-data", "filename"),
    State("analyticstool-raw-data-store", "data"),
    State("analyticstool-original-periodicity-store", "data"),
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
                n_no, n_no, n_no, n_no, n_no, n_no,
                True, dropdown_data, sheet_names[0], contents, filename,  # open sheet modal
                False, True,  # hide blocker
            )

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
        default_periodicity = "daily_trading" if combined_periodicity == "daily" else combined_periodicity

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
            current_cmabench or {},
            current_ls or {},
            current_order or [],
            [],
            current_vol_scaling or {},
            current_min_wt or {},
            current_max_wt or {},
            current_force_max or {},
            *sheet_no,
            False, True,  # hide blocker
        )
    except Exception as e:
        return (
            n_no, n_no, n_no, n_no, n_no, n_no,
            f"Error loading file: {str(e)}", "red", False,
            n_no, n_no, n_no, n_no, n_no, n_no, n_no, n_no, n_no, n_no, n_no,
            *sheet_no,
            False, True,  # hide blocker
        )


# ---------------------------------------------------------------------------
# Sheet selection modal: confirm
# ---------------------------------------------------------------------------
@callback(
    Output("analyticstool-raw-data-store", "data", allow_duplicate=True),
    Output("analyticstool-original-periodicity-store", "data", allow_duplicate=True),
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
    Output("po-upload-data", "contents", allow_duplicate=True),
    # Blocker outputs
    Output("po-ui-blocker-store", "data", allow_duplicate=True),
    Output("po-ui-blocker-timeout", "disabled", allow_duplicate=True),
    Input("po-sheet-select-ok-button", "n_clicks"),
    State("po-sheet-select-dropdown", "value"),
    State("po-sheet-select-contents-store", "data"),
    State("po-sheet-select-filename-store", "data"),
    State("analyticstool-raw-data-store", "data"),
    State("analyticstool-original-periodicity-store", "data"),
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
def po_on_sheet_select_ok(n_clicks, selected_sheet, stashed_contents, stashed_filename,
                          existing_data, existing_periodicity, current_selection,
                          current_bench, current_cmabench, current_ls, current_order,
                          current_vol_scaling, current_min_wt, current_max_wt, current_force_max):
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
            merged_df = merge_returns(existing_df, new_df)
        else:
            merged_df = new_df
            combined_periodicity = new_periodicity

        periodicity_options = get_available_periodicities(combined_periodicity)
        default_periodicity = "daily_trading" if combined_periodicity == "daily" else combined_periodicity

        new_series = [col for col in new_df.columns if col not in (current_selection or [])]
        updated_selection = (current_selection or []) + new_series

        alert_msg = f"Loaded {len(new_df.columns)} series with {len(new_df)} rows from {filename} (sheet: {selected_sheet})"

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
            False, None, None, None,  # close sheet modal, clear stash, reset upload
            False, True,  # hide blocker
        )
    except Exception as e:
        return (
            n_no, n_no, n_no, n_no, n_no, n_no,
            f"Error loading file: {str(e)}", "red", False,
            n_no, n_no, n_no, n_no, n_no, n_no, n_no, n_no, n_no, n_no, n_no,
            False, None, None, None,  # close sheet modal, clear stash, reset upload
            False, True,  # hide blocker
        )


# ---------------------------------------------------------------------------
# Sheet selection modal: cancel
# ---------------------------------------------------------------------------
@callback(
    Output("po-sheet-select-modal", "opened", allow_duplicate=True),
    Output("po-sheet-select-contents-store", "data", allow_duplicate=True),
    Output("po-sheet-select-filename-store", "data", allow_duplicate=True),
    Output("po-upload-data", "contents", allow_duplicate=True),
    # Blocker outputs
    Output("po-ui-blocker-store", "data", allow_duplicate=True),
    Output("po-ui-blocker-timeout", "disabled", allow_duplicate=True),
    Input("po-sheet-select-cancel-button", "n_clicks"),
    prevent_initial_call=True,
)
def po_on_sheet_select_cancel(n_clicks):
    """Cancel sheet selection and clear stashed data."""
    if not n_clicks:
        raise PreventUpdate
    return False, None, None, None, False, True


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
    Input("po-open-modal-button", "n_clicks"),
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
def po_open_modal(n_clicks, current_select, current_bench, current_cmabench, current_ls, current_order,
                  current_vol_scaling, current_min_wt, current_max_wt, current_force_max):
    if not n_clicks:
        raise PreventUpdate
    return (True, current_select, current_bench, current_cmabench, current_ls, current_order, [],
            current_vol_scaling, current_min_wt, current_max_wt, current_force_max)


# ---------------------------------------------------------------------------
# Series selection modal: render rows
# ---------------------------------------------------------------------------

@callback(
    Output("po-series-selection-container", "children"),
    Output("po-temp-series-order-store", "data", allow_duplicate=True),
    Input("analyticstool-raw-data-store", "data"),
    Input("po-temp-series-select", "data"),
    Input("po-temp-series-order-store", "data"),
    Input("po-temp-deleted-series-store", "data"),
    Input("po-series-selection-grid", "cellValueChanged", allow_optional=True),
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
    _cell_change,
    current_assignments,
    current_cmabench_assignments,
    long_short_assignments,
    vol_scaling_assignments,
    min_wt,
    max_wt,
    force_max,
):
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
        className="ag-theme-alpine series-modal-grid",
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
                "cellClass": "series-center-cell",
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
                "cellClass": "series-center-cell",
            },
            {
                "field": "Series",
                "editable": True,
                "minWidth": 150,
                "cellStyle": {"textAlign": "left", "fontFamily": "monospace"},
                "headerClass": "left-header",
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
                "headerClass": "left-header",
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
                "headerClass": "left-header",
            },
            {
                "field": "LongShort",
                "headerName": "L/S",
                "editable": True,
                "cellRenderer": "agCheckboxCellRenderer",
                "cellEditor": "agCheckboxCellEditor",
                "width": 72,
                "cellClass": "series-center-cell",
            },
            {
                "field": "ScaleVol",
                "headerName": "Scale Vol",
                "editable": True,
                "cellRenderer": "agCheckboxCellRenderer",
                "cellEditor": "agCheckboxCellEditor",
                "width": 112,
                "cellClass": "series-center-cell",
            },
            {
                "field": "MinWt",
                "headerName": "Min Wt",
                "editable": {"function": "!params.data.ForceMax"},
                "width": 98,
                "valueParser": {"function": "var n=Number(params.newValue); if(!isFinite(n)) return 0; return Math.max(0, Math.min(100, n));"},
                "cellClass": "series-center-cell",
                "headerClass": "center-header",
            },
            {
                "field": "MaxWt",
                "headerName": "Max Wt",
                "editable": True,
                "width": 98,
                "valueParser": {"function": "var n=Number(params.newValue); if(!isFinite(n)) return 100; return Math.max(0, Math.min(100, n));"},
                "cellClass": "series-center-cell",
                "headerClass": "center-header",
            },
            {
                "field": "ForceMax",
                "headerName": "Force",
                "editable": True,
                "cellRenderer": "agCheckboxCellRenderer",
                "cellEditor": "agCheckboxCellEditor",
                "width": 70,
                "cellClass": "series-center-cell",
            },
            {
                "field": "Delete",
                "editable": True,
                "cellRenderer": "agCheckboxCellRenderer",
                "cellEditor": "agCheckboxCellEditor",
                "width": 74,
                "cellClass": "series-center-cell",
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
            "headerClass": "center-header",
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
    State("analyticstool-raw-data-store", "data"),
    prevent_initial_call=True,
)
def po_update_benchmarks(cell_change, row_data, raw_data):
    change = _po_latest_series_grid_change(cell_change)
    if not change:
        raise PreventUpdate
    if change.get("colId") == "Series":
        raise PreventUpdate
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
    State("analyticstool-raw-data-store", "data"),
    prevent_initial_call=True,
)
def po_update_ls(cell_change, row_data, raw_data):
    change = _po_latest_series_grid_change(cell_change)
    if not change:
        raise PreventUpdate
    if change.get("colId") == "Series":
        raise PreventUpdate
    if raw_data is None or not row_data:
        return {}
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
    State("analyticstool-raw-data-store", "data"),
    prevent_initial_call=True,
)
def po_update_vol_scaling(cell_change, row_data, raw_data):
    change = _po_latest_series_grid_change(cell_change)
    if not change:
        raise PreventUpdate
    if change.get("colId") == "Series":
        raise PreventUpdate
    if raw_data is None or not row_data:
        return {}
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
    Output("analyticstool-raw-data-store", "data", allow_duplicate=True),
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
    State("analyticstool-raw-data-store", "data"),
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
    Output("analyticstool-raw-data-store", "data", allow_duplicate=True),
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
    State("analyticstool-raw-data-store", "data"),
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
    Output("po-start-date-picker", "value"),
    Output("po-end-date-picker", "value"),
    Output("po-date-picker-wrapper", "style"),
    Output("po-common-range-button", "disabled"),
    Output("po-common-daily-button", "disabled"),
    Output("po-maximum-range-button", "disabled"),
    Output("po-date-range-store", "data", allow_duplicate=True),
    Input("analyticstool-raw-data-store", "data"),
    Input("po-periodicity-select", "value"),
    Input("po-series-select", "data"),
    State("po-date-range-store", "data"),
    prevent_initial_call="initial_duplicate",
)
def po_init_date_range(raw_data, periodicity, selected_series, stored_range):
    disabled_style = {"display": "flex", "opacity": 0.5, "pointerEvents": "none", "alignItems": "flex-start"}
    enabled_style = {"display": "flex", "alignItems": "flex-start"}

    if raw_data is None or not selected_series:
        return None, None, disabled_style, True, True, True, None

    try:
        df = resample_returns_cached(raw_data, periodicity or "daily")
        available = [s for s in selected_series if s in df.columns]
        if not available:
            return None, None, disabled_style, True, True, True, None

        daily_df = resample_returns_cached(raw_data, "daily_trading")
        daily_available = [s for s in selected_series if s in daily_df.columns]
        has_common_daily = bool(get_common_daily_range(daily_df, daily_available)) if daily_available else False

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
    State("analyticstool-raw-data-store", "data"),
    State("po-periodicity-select", "value"),
    State("po-series-select", "data"),
    prevent_initial_call=True,
)
def po_date_range_buttons(common_clicks, common_daily_clicks, max_clicks, raw_data, periodicity, selected_series):
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
            periodicity_value = no_update
        elif button_id == "po-common-daily-button":
            daily_df = resample_returns_cached(raw_data, "daily_trading")
            daily_available = [s for s in selected_series if s in daily_df.columns]
            common_daily = get_common_daily_range(daily_df, daily_available)
            if not common_daily:
                raise PreventUpdate
            start_date = common_daily[0].strftime("%Y-%m-%d")
            end_date = common_daily[1].strftime("%Y-%m-%d")
            periodicity_value = "daily_trading"
        else:
            start_date = df.index.min().strftime("%Y-%m-%d")
            end_date = df.index.max().strftime("%Y-%m-%d")
            periodicity_value = no_update
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
    Output("analyticstool-raw-data-store", "data", allow_duplicate=True),
    Output("po-opt-status-store", "data"),
    Output("analyticstool-pending-new-series-store", "data", allow_duplicate=True),
    Input("po-run-button", "n_clicks"),
    State("analyticstool-raw-data-store", "data"),
    State("analyticstool-original-periodicity-store", "data"),
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
    State("po-portfolio-name-input", "value"),
    State("po-opt-window-select", "value"),
    State("po-window-size-input", "value"),
    State("po-opt-step-input", "value"),
    State("po-opt-step-unit-select", "value"),
    State("po-opt-model-select", "value"),
    State("po-missing-data-select", "value"),
    State("po-fill-in-sample-select", "value"),
    State("po-results-store", "data"),
    State("analyticstool-pending-new-series-store", "data"),
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
    State("analyticstool-saved-series-cache-store", "data"),
    prevent_initial_call=True,
)
def po_run_optimization(n_clicks, raw_data, orig_periodicity, periodicity,
                        selected_series, benchmark_assignments, cmabench_assignments, long_short_assignments,
                        date_range, vol_scaler, vol_scaling_assignments,
                        min_wt, max_wt, force_max, exp_wt_cov, halflife,
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

        # Build config
        config = {
            "model": model_value,
            "window_type": window_value,
            "window_size": int(_coerce_float(window_size) or 252),
            "opt_step": int(_coerce_float(opt_step) or 252),
            "opt_step_unit": opt_step_unit_value or "months",
            "exp_wt_cov": bool(exp_wt_cov),
            "halflife": int(_coerce_float(halflife) or 63),
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
                "est_start": wr.est_start.isoformat(),
                "est_end": wr.est_end.isoformat(),
                "weights": wr.weights,
            })

        result_entry = {
            "window_weights": window_data,
            "returns_json": portfolio_returns.to_json(date_format="iso"),
            "config": config,
            "risk_free_meta": {
                "source": resolved_rf_context.get("rf_source"),
                "annual": float(resolved_rf_context.get("rf_annual", 0.0) or 0.0),
                "warning": resolved_rf_context.get("rf_warning"),
            },
        }
        if model_value in {"ex_ante_mv", "black_litterman"}:
            frontier_snapshot = _build_frontier_snapshot(
                selected_portfolio=final_name,
                portfolio_data=result_entry,
                raw_data=raw_data,
                periodicity=periodicity,
                bench=benchmark_assignments,
                ls=long_short_assignments,
                vol_scaler=vol_scaler,
                vol_scaling=vol_scaling_assignments,
                window_idx=0,
                rm="MV",
                linear_constraints=linear_constraints,
                saved_series_store=saved_series_store,
                cmabench_assignments=cmabench_assignments,
            )
            _cache_frontier_snapshot(result_entry, frontier_snapshot)

        current_results[final_name] = result_entry

        # Add to pending list so Analytics Tool auto-selects this series
        updated_pending = list(pending_series or []) + [final_name]
        warning_parts = []
        if resolved_rf_context.get("rf_warning"):
            warning_parts.append(str(resolved_rf_context.get("rf_warning")))
        if isinstance(optimization_meta, dict) and optimization_meta.get("risk_free_warning"):
            warning_parts.append(str(optimization_meta.get("risk_free_warning")))
        warning_text = " ".join(dict.fromkeys([w for w in warning_parts if w])).strip() or None

        return (
            current_results,
            new_raw_data,
            {"status": "complete", "name": final_name, "warning": warning_text},
            updated_pending,
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
    newest = names[-1] if names else None
    if newest and newest not in multi:
        multi.append(newest)
    return options, sel, options, multi


# ---------------------------------------------------------------------------
# Sync results store when raw data changes (e.g. series deleted in Analytics Tool)
# ---------------------------------------------------------------------------

@callback(
    Output("po-results-store", "data", allow_duplicate=True),
    Input("analyticstool-raw-data-store", "data"),
    Input("po-page-load-trigger", "n_intervals"),
    State("po-results-store", "data"),
    prevent_initial_call=True,
)
def po_sync_results_with_raw_data(raw_data, _n, results):
    if not results:
        raise PreventUpdate
    if not raw_data:
        return {}
    df = json_to_df(raw_data)
    pruned = {k: v for k, v in results.items() if k in df.columns}
    if len(pruned) == len(results):
        raise PreventUpdate
    return pruned


# ---------------------------------------------------------------------------
# Delete portfolio
# ---------------------------------------------------------------------------

@callback(
    Output("po-results-store", "data", allow_duplicate=True),
    Output("analyticstool-raw-data-store", "data", allow_duplicate=True),
    Output("po-weight-portfolio-select", "value", allow_duplicate=True),
    Input("po-delete-portfolio-button", "n_clicks"),
    State("po-weight-portfolio-select", "value"),
    State("po-results-store", "data"),
    State("analyticstool-raw-data-store", "data"),
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
    Output("po-weight-chart-content", "children"),
    Input("po-weight-portfolio-select", "value"),
    Input("po-results-store", "data"),
    Input("po-vis-tabs", "value"),
    Input("po-weight-chart-switch", "value"),
    State("theme-store", "data"),
    prevent_initial_call=True,
)
def po_render_weight_chart(selected_portfolio, results, active_tab, switch_value, theme):
    if active_tab != "weight" or switch_value != "chart" or not selected_portfolio or not results:
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
        height=420,
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.2},
    )
    apply_chart_theme(fig, theme)

    return dcc.Graph(figure=fig, style={"height": "100%", "width": "100%"})


# ---------------------------------------------------------------------------
# Growth of $1 chart
# ---------------------------------------------------------------------------

@callback(
    Output("po-growth-chart-container", "children"),
    Input("po-growth-portfolio-multiselect", "value"),
    Input("po-results-store", "data"),
    Input("po-vis-tabs", "value"),
    State("theme-store", "data"),
    prevent_initial_call=True,
)
def po_render_growth_chart(selected_portfolios, results, active_tab, theme):
    if active_tab != "growth" or not selected_portfolios or not results:
        return html.Div()

    fig = go.Figure()
    for pname in selected_portfolios:
        if pname not in results:
            continue
        returns_json = results[pname].get("returns_json")
        if not returns_json:
            continue
        returns = pd.read_json(StringIO(returns_json), typ="series")
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
        height=420,
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.2},
    )
    apply_chart_theme(fig, theme)

    return dcc.Graph(figure=fig, style={"height": "100%", "width": "100%"})


# ---------------------------------------------------------------------------
# Attribution chart (monthly stacked bar)
# ---------------------------------------------------------------------------

@callback(
    Output("po-attribution-chart-content", "children"),
    Input("po-weight-portfolio-select", "value"),
    Input("po-results-store", "data"),
    Input("po-vis-tabs", "value"),
    Input("po-attribution-chart-switch", "value"),
    State("analyticstool-raw-data-store", "data"),
    State("analyticstool-original-periodicity-store", "data"),
    State("po-periodicity-select", "value"),
    State("po-benchmark-assignments-store", "data"),
    State("po-long-short-store", "data"),
    State("po-date-range-store", "data"),
    State("po-vol-scaler-value-store", "data"),
    State("po-vol-scaling-assignments-store", "data"),
    State("theme-store", "data"),
    prevent_initial_call=True,
)
def po_render_attribution_chart(selected_portfolio, results, active_tab, switch_value,
                                 raw_data, orig_periodicity, periodicity, bench, ls,
                                 date_range, vol_scaler, vol_scaling, theme):
    if active_tab != "attribution" or switch_value != "chart" or not selected_portfolio or not results:
        return html.Div()
    if selected_portfolio not in results:
        return html.Div()

    portfolio_data = results[selected_portfolio]
    window_weights = portfolio_data.get("window_weights", [])
    config = portfolio_data.get("config", {})
    opt_series = config.get("selected_series", [])

    if not window_weights or not opt_series or not raw_data:
        return dmc.Text("No attribution data available.", c="dimmed")

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
        working_df = _po_get_working_returns(working_bundle, opt_series)
        attribution_monthly = _compute_monthly_attribution(working_df, opt_series, window_weights)

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
    Output("po-weight-grid", "columnDefs"),
    Output("po-weight-grid", "rowData"),
    Input("po-weight-portfolio-select", "value"),
    Input("po-results-store", "data"),
    Input("po-vis-tabs", "value"),
    Input("po-weight-chart-switch", "value"),
    prevent_initial_call=True,
)
def po_render_weight_table(selected_portfolio, results, active_tab, switch_value):
    if active_tab != "weight" or switch_value != "table" or not selected_portfolio or not results:
        return [], []
    if selected_portfolio not in results:
        return [], []

    portfolio_data = results[selected_portfolio]
    window_weights = portfolio_data.get("window_weights", [])

    if not window_weights:
        return [], []

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

    return column_defs, row_data


# ---------------------------------------------------------------------------
# Attribution table
# ---------------------------------------------------------------------------

@callback(
    Output("po-attribution-grid", "columnDefs"),
    Output("po-attribution-grid", "rowData"),
    Input("po-weight-portfolio-select", "value"),
    Input("po-results-store", "data"),
    Input("po-vis-tabs", "value"),
    Input("po-attribution-chart-switch", "value"),
    State("analyticstool-raw-data-store", "data"),
    State("po-periodicity-select", "value"),
    State("po-benchmark-assignments-store", "data"),
    State("po-long-short-store", "data"),
    State("po-date-range-store", "data"),
    State("po-vol-scaler-value-store", "data"),
    State("po-vol-scaling-assignments-store", "data"),
    prevent_initial_call=True,
)
def po_render_attribution_table(selected_portfolio, results, active_tab, switch_value,
                                raw_data, periodicity, bench, ls, date_range,
                                vol_scaler, vol_scaling):
    if active_tab != "attribution" or switch_value != "table" or not selected_portfolio or not results:
        return [], []
    if selected_portfolio not in results:
        return [], []

    portfolio_data = results[selected_portfolio]
    window_weights = portfolio_data.get("window_weights", [])
    config = portfolio_data.get("config", {})
    opt_series = config.get("selected_series", [])

    if not window_weights or not opt_series or not raw_data:
        return [], []

    try:
        with timed_block(
            "portopt.render_attribution_table",
            portfolio=selected_portfolio,
            series_count=len(opt_series),
        ):
            working_bundle = _build_po_working_bundle(
                raw_data, periodicity, bench, ls, date_range, vol_scaler, vol_scaling
            )
            working_df = _po_get_working_returns(working_bundle, opt_series)
            attribution_monthly = _compute_monthly_attribution(working_df, opt_series, window_weights)

            if attribution_monthly.empty:
                return [], []

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

            return column_defs, row_data

    except Exception:
        return [], []


# ---------------------------------------------------------------------------
# Statistics tab
# ---------------------------------------------------------------------------

@callback(
    Output("po-statistics-grid", "columnDefs"),
    Output("po-statistics-grid", "rowData"),
    Input("po-results-store", "data"),
    Input("po-vis-tabs", "value"),
    Input("po-growth-portfolio-multiselect", "value"),
    Input("analyticstool-saved-series-cache-store", "data"),
    State("po-periodicity-select", "value"),
    prevent_initial_call=True,
)
def po_render_statistics(results, active_tab, selected_portfolios, saved_series_store, periodicity):
    if active_tab != "statistics" or not results:
        return [], []

    show = selected_portfolios or list(results.keys())

    try:
        with timed_block("portopt.render_statistics", portfolio_count=len(show)):
            # Build a combined returns DataFrame from selected portfolio results
            all_returns = {}
            for pname in show:
                pdata = results.get(pname)
                if not pdata:
                    continue
                returns_json = pdata.get("returns_json")
                if returns_json:
                    s = pd.read_json(StringIO(returns_json), typ="series")
                    s.index = pd.to_datetime(s.index)
                    all_returns[pname] = s

            if not all_returns:
                return [], []

            combined_df = pd.DataFrame(all_returns)
            combined_df = combined_df.sort_index()
            combined_df.index.name = "Date"

            # Convert to raw-data JSON format for calculate_statistics_cached
            raw_json = df_to_json(combined_df)
            portfolio_names = list(all_returns.keys())

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

            if not stats:
                return [], []

            column_defs = [
                {"field": "Statistic", "pinned": "left", "width": 200},
            ]
            for series_stats in stats:
                series_name = series_stats["Series"]
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
                    series_name = series_stats["Series"]
                    value = series_stats.get(stat_name)
                    if value is None or (isinstance(value, float) and pd.isna(value)):
                        row[series_name] = None
                    else:
                        row[series_name] = value
                row_data.append(row)

            return column_defs, row_data

    except Exception:
        return [], []


# ---------------------------------------------------------------------------
# Returns tab
# ---------------------------------------------------------------------------

@callback(
    Output("po-returns-grid", "columnDefs"),
    Output("po-returns-grid", "rowData"),
    Input("po-results-store", "data"),
    Input("po-vis-tabs", "value"),
    Input("po-growth-portfolio-multiselect", "value"),
    prevent_initial_call=True,
)
def po_render_returns(results, active_tab, selected_portfolios):
    if active_tab != "returns" or not results:
        return [], []

    show = selected_portfolios or list(results.keys())

    try:
        all_returns = {}
        for pname in show:
            pdata = results.get(pname)
            if not pdata:
                continue
            returns_json = pdata.get("returns_json")
            if returns_json:
                s = pd.read_json(StringIO(returns_json), typ="series")
                s.index = pd.to_datetime(s.index)
                all_returns[pname] = s

        if not all_returns:
            return [], []

        combined_df = pd.DataFrame(all_returns)
        combined_df = combined_df.sort_index()
        combined_df.index.name = "Date"

        column_defs = [
            {
                "field": "Date",
                "pinned": "left",
                "width": 120,
            },
        ]
        for col in combined_df.columns:
            column_defs.append({
                "field": col,
                "valueFormatter": {"function": "params.value != null ? d3.format('.2%')(params.value) : ''"},
                "width": 120,
            })

        df_reset = combined_df.reset_index()
        df_reset["Date"] = df_reset["Date"].dt.strftime("%Y-%m-%d")
        row_data = df_reset.to_dict("records")

        return column_defs, row_data

    except Exception:
        return [], []


# ---------------------------------------------------------------------------
# Excel export
# ---------------------------------------------------------------------------

@callback(
    Output("po-download-excel", "data"),
    Input("po-menu-download-excel", "n_clicks"),
    State("po-results-store", "data"),
    State("analyticstool-raw-data-store", "data"),
    State("po-periodicity-select", "value"),
    State("po-benchmark-assignments-store", "data"),
    State("po-cmabench-assignments-store", "data"),
    State("po-long-short-store", "data"),
    State("po-date-range-store", "data"),
    State("po-vol-scaler-value-store", "data"),
    State("po-vol-scaling-assignments-store", "data"),
    State("analyticstool-saved-series-cache-store", "data"),
    prevent_initial_call=True,
)
def po_download_excel(n_clicks, results, raw_data, periodicity, bench, cmabench, ls,
                      date_range, vol_scaler, vol_scaling, saved_series_store):
    if n_clicks is None or not results:
        raise PreventUpdate

    timing_ctx = timed_block("portopt.download_excel.total", portfolio_count=len(results))
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
            all_returns = {}
            for pname, pdata in results.items():
                returns_json = pdata.get("returns_json")
                if returns_json:
                    s = pd.read_json(StringIO(returns_json), typ="series")
                    s.index = pd.to_datetime(s.index)
                    all_returns[pname] = s

        if not all_returns:
            raise PreventUpdate

        combined_df = pd.DataFrame(all_returns).sort_index()
        combined_df.index.name = "Date"
        portfolio_names = list(all_returns.keys())

        # ------------------------------------------------------------------
        # Weights tab
        # ------------------------------------------------------------------
        weight_rows = []
        weight_cols = []
        for pname, pdata in results.items():
            for ww in pdata.get("window_weights", []) or []:
                row = {
                    "Portfolio": pname,
                    "Apply Start": pd.Timestamp(ww["apply_start"]).strftime("%Y-%m-%d"),
                    "Apply End": pd.Timestamp(ww["apply_end"]).strftime("%Y-%m-%d"),
                }
                for asset, weight in (ww.get("weights", {}) or {}).items():
                    col = f"Wt_{asset}"
                    row[col] = float(weight)
                    if col not in weight_cols:
                        weight_cols.append(col)
                weight_rows.append(row)
        if weight_rows:
            weights_df = pd.DataFrame(weight_rows)
            weights_df = weights_df[["Portfolio", "Apply Start", "Apply End", *weight_cols]]
        else:
            weights_df = pd.DataFrame(columns=["Portfolio", "Apply Start", "Apply End"])

        # ------------------------------------------------------------------
        # Turnover tab
        # ------------------------------------------------------------------
        turnover_rows = []
        turnover_delta_cols = []
        for pname, pdata in results.items():
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
                    "Rebalance Date": pd.Timestamp(window_weights[i]["apply_start"]).strftime("%Y-%m-%d"),
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
        returns_df["Date"] = returns_df["Date"].dt.strftime("%Y-%m-%d")

        # ------------------------------------------------------------------
        # Growth tab
        # ------------------------------------------------------------------
        growth_data = {pname: (1 + all_returns[pname]).cumprod() for pname in portfolio_names}
        growth_df = pd.DataFrame(growth_data).sort_index().reset_index()
        growth_date_col = growth_df.columns[0]
        growth_df = growth_df.rename(columns={growth_date_col: "Date"})
        growth_df["Date"] = growth_df["Date"].dt.strftime("%Y-%m-%d")

        # ------------------------------------------------------------------
        # Attribution tab
        # ------------------------------------------------------------------
        attribution_frames = []
        with timed_block("portopt.download_excel.attribution", portfolio_count=len(results)):
            for pname, pdata in results.items():
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
                if working_df.empty:
                    continue

                attribution_monthly = _compute_monthly_attribution(
                    working_df,
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
                frame["Date"] = pd.to_datetime(frame["Date"]).dt.strftime("%Y-%m-%d")
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
        for pname, pdata in results.items():
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
            if working_df.empty:
                continue

            for rr in _compute_window_risk_contributions(working_df, opt_series, window_weights):
                row = {
                    "Portfolio": pname,
                    "Window Start": rr["apply_start"].strftime("%Y-%m-%d"),
                    "Window End": rr["apply_end"].strftime("%Y-%m-%d"),
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
        with timed_block("portopt.download_excel.frontier", portfolio_count=len(results)):
            for pname, pdata in results.items():
                config = pdata.get("config", {}) or {}
                window_weights = pdata.get("window_weights", []) or []
                opt_series = config.get("selected_series", []) or []
                if not window_weights or not opt_series or not raw_data:
                    continue

                try:
                    latest_idx, _ = _resolve_frontier_window(window_weights, None)
                    model = config.get("model", "")
                    risk_measure = "MV"
                    snapshot = None
                    if model in {"ex_ante_mv", "black_litterman"}:
                        snapshot = _get_cached_frontier_snapshot(pdata, latest_idx, risk_measure)
                    if snapshot is None:
                        snapshot = _build_frontier_snapshot(
                            selected_portfolio=pname,
                            portfolio_data=pdata,
                            raw_data=raw_data,
                            periodicity=periodicity,
                            bench=bench,
                            ls=ls,
                            vol_scaler=vol_scaler,
                            vol_scaling=vol_scaling,
                            window_idx=latest_idx,
                            rm=risk_measure,
                            linear_constraints=config.get("linear_constraints", []),
                            saved_series_store=saved_series_store,
                            cmabench_assignments=cmabench,
                        )

                    window_label = f"{snapshot.get('window_est_start')} - {snapshot.get('window_est_end')}"
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
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            # Keep exact tab order: Weights, Turnover, Statistics, Returns,
            # Growth, Attribution, Risk, Frontier.
            weights_df.to_excel(writer, sheet_name="Weights", index=False)
            turnover_df.to_excel(writer, sheet_name="Turnover", index=False)
            stats_df.to_excel(writer, sheet_name="Statistics", index=False)
            returns_df.to_excel(writer, sheet_name="Returns", index=False)
            growth_df.to_excel(writer, sheet_name="Growth of $1", index=False)
            attribution_df.to_excel(writer, sheet_name="Attribution", index=False)
            risk_df.to_excel(writer, sheet_name="Risk", index=False)
            frontier_df.to_excel(writer, sheet_name="Frontier", index=False)

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
    Output("po-risk-chart-content", "children"),
    Input("po-weight-portfolio-select", "value"),
    Input("po-results-store", "data"),
    Input("po-vis-tabs", "value"),
    Input("po-risk-chart-switch", "value"),
    State("analyticstool-raw-data-store", "data"),
    State("po-periodicity-select", "value"),
    State("po-benchmark-assignments-store", "data"),
    State("po-long-short-store", "data"),
    State("po-date-range-store", "data"),
    State("po-vol-scaler-value-store", "data"),
    State("po-vol-scaling-assignments-store", "data"),
    State("po-series-select", "data"),
    State("theme-store", "data"),
    prevent_initial_call=True,
)
def po_render_risk_chart(selected_portfolio, results, active_tab, switch_value,
                         raw_data, periodicity, bench, ls, date_range,
                         vol_scaler, vol_scaling, series_select, theme):
    if active_tab != "risk" or switch_value != "chart" or not selected_portfolio or not results:
        return html.Div()
    if selected_portfolio not in results:
        return html.Div()

    portfolio_data = results[selected_portfolio]
    window_weights = portfolio_data.get("window_weights", [])
    config = portfolio_data.get("config", {})
    opt_series = config.get("selected_series", [])

    if not window_weights or not opt_series or not raw_data:
        return dmc.Text("No risk data available.", c="dimmed")

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
        working_df = _po_get_working_returns(working_bundle, opt_series)

        risk_rows = _compute_window_risk_contributions(working_df, opt_series, window_weights)
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
    Output("po-risk-grid", "columnDefs"),
    Output("po-risk-grid", "rowData"),
    Input("po-weight-portfolio-select", "value"),
    Input("po-results-store", "data"),
    Input("po-vis-tabs", "value"),
    Input("po-risk-chart-switch", "value"),
    State("analyticstool-raw-data-store", "data"),
    State("po-periodicity-select", "value"),
    State("po-benchmark-assignments-store", "data"),
    State("po-long-short-store", "data"),
    State("po-date-range-store", "data"),
    State("po-vol-scaler-value-store", "data"),
    State("po-vol-scaling-assignments-store", "data"),
    State("po-series-select", "data"),
    prevent_initial_call=True,
)
def po_render_risk_table(selected_portfolio, results, active_tab, switch_value,
                         raw_data, periodicity, bench, ls, date_range,
                         vol_scaler, vol_scaling, series_select):
    if active_tab != "risk" or switch_value != "table" or not selected_portfolio or not results:
        return [], []
    if selected_portfolio not in results:
        return [], []

    portfolio_data = results[selected_portfolio]
    window_weights = portfolio_data.get("window_weights", [])
    config = portfolio_data.get("config", {})
    opt_series = config.get("selected_series", [])

    if not window_weights or not opt_series or not raw_data:
        return [], []

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
        working_df = _po_get_working_returns(working_bundle, opt_series)
        risk_rows = _compute_window_risk_contributions(working_df, opt_series, window_weights)
        if not risk_rows:
            return [], []

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

        return column_defs, row_data

    except Exception:
        return [], []
    finally:
        timing_ctx.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# Turnover chart
# ---------------------------------------------------------------------------

@callback(
    Output("po-turnover-chart-content", "children"),
    Input("po-weight-portfolio-select", "value"),
    Input("po-results-store", "data"),
    Input("po-vis-tabs", "value"),
    Input("po-turnover-chart-switch", "value"),
    State("theme-store", "data"),
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
    Output("po-turnover-grid", "columnDefs"),
    Output("po-turnover-grid", "rowData"),
    Input("po-weight-portfolio-select", "value"),
    Input("po-results-store", "data"),
    Input("po-vis-tabs", "value"),
    Input("po-turnover-chart-switch", "value"),
    prevent_initial_call=True,
)
def po_render_turnover_table(selected_portfolio, results, active_tab, switch_value):
    if active_tab != "turnover" or switch_value != "table" or not selected_portfolio or not results:
        return [], []
    if selected_portfolio not in results:
        return [], []

    portfolio_data = results[selected_portfolio]
    window_weights = portfolio_data.get("window_weights", [])

    if not window_weights or len(window_weights) < 2:
        return [], []

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

    return column_defs, row_data


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

    # Ex ante / BL are single-period — disable window selection
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
    Output("po-frontier-chart-content", "children"),
    Input("po-weight-portfolio-select", "value"),
    Input("po-results-store", "data"),
    Input("po-vis-tabs", "value"),
    Input("po-frontier-chart-switch", "value"),
    Input("po-frontier-window-select", "value"),
    Input("po-frontier-rm-select", "value"),
    State("analyticstool-raw-data-store", "data"),
    State("po-periodicity-select", "value"),
    State("po-benchmark-assignments-store", "data"),
    State("po-long-short-store", "data"),
    State("po-date-range-store", "data"),
    State("po-vol-scaler-value-store", "data"),
    State("po-vol-scaling-assignments-store", "data"),
    State("po-cmabench-assignments-store", "data"),
    State("analyticstool-saved-series-cache-store", "data"),
    State("po-series-select", "data"),
    State("theme-store", "data"),
    State("po-linear-constraints-store", "data"),
    prevent_initial_call=True,
)
def po_render_frontier_chart(selected_portfolio, results, active_tab, switch_value,
                             window_idx, rm,
                             raw_data, periodicity, bench, ls, date_range,
                             vol_scaler, vol_scaling, cmabench_assignments, saved_series_store, series_select, theme,
                             linear_constraints):
    if active_tab != "frontier" or switch_value != "chart" or not selected_portfolio or not results:
        return html.Div()
    if selected_portfolio not in results:
        return html.Div()

    portfolio_data = results[selected_portfolio]
    window_weights = portfolio_data.get("window_weights", [])
    config = portfolio_data.get("config", {})
    opt_series = config.get("selected_series", [])

    if not window_weights or not opt_series or not raw_data:
        return dmc.Text("No frontier data available.", c="dimmed")

    timing_ctx = timed_block(
        "portopt.render_frontier_chart",
        portfolio=selected_portfolio,
        series_count=len(opt_series),
        risk_measure=rm,
    )
    timing_ctx.__enter__()
    try:
        model = config.get("model", "")
        resolved_idx, _ = _resolve_frontier_window(window_weights, window_idx)
        risk_measure = rm or "MV"
        if model in {"ex_ante_mv", "black_litterman"} and risk_measure == "CVaR":
            risk_measure = "MV"

        snapshot = None
        if model in {"ex_ante_mv", "black_litterman"}:
            snapshot = _get_cached_frontier_snapshot(portfolio_data, resolved_idx, risk_measure)

        if snapshot is None:
            snapshot = _build_frontier_snapshot(
                selected_portfolio=selected_portfolio,
                portfolio_data=portfolio_data,
                raw_data=raw_data,
                periodicity=periodicity,
                bench=bench,
                ls=ls,
                vol_scaler=vol_scaler,
                vol_scaling=vol_scaling,
                window_idx=resolved_idx,
                rm=risk_measure,
                linear_constraints=linear_constraints,
                saved_series_store=saved_series_store,
                cmabench_assignments=cmabench_assignments,
            )

        frontier_pts = snapshot.get("frontier_points", []) or []
        asset_pts = snapshot.get("assets", []) or []
        portfolio_marker = snapshot.get("portfolio", {}) or {}
        risk_measure = snapshot.get("risk_measure", risk_measure)

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

        return dcc.Graph(figure=fig, style={"height": "100%", "width": "100%"})

    except Exception as e:
        return dmc.Text(f"Error computing efficient frontier: {str(e)}", c="dimmed")
    finally:
        timing_ctx.__exit__(None, None, None)


@callback(
    Output("po-frontier-grid", "columnDefs"),
    Output("po-frontier-grid", "rowData"),
    Input("po-weight-portfolio-select", "value"),
    Input("po-results-store", "data"),
    Input("po-vis-tabs", "value"),
    Input("po-frontier-chart-switch", "value"),
    Input("po-frontier-window-select", "value"),
    Input("po-frontier-rm-select", "value"),
    State("analyticstool-raw-data-store", "data"),
    State("po-periodicity-select", "value"),
    State("po-benchmark-assignments-store", "data"),
    State("po-long-short-store", "data"),
    State("po-vol-scaler-value-store", "data"),
    State("po-vol-scaling-assignments-store", "data"),
    State("po-cmabench-assignments-store", "data"),
    State("analyticstool-saved-series-cache-store", "data"),
    State("po-linear-constraints-store", "data"),
    prevent_initial_call=True,
)
def po_render_frontier_table(
    selected_portfolio,
    results,
    active_tab,
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
    if active_tab != "frontier" or switch_value != "table" or not selected_portfolio or not results:
        return [], []
    if selected_portfolio not in results:
        return [], []

    portfolio_data = results[selected_portfolio]
    window_weights = portfolio_data.get("window_weights", []) or []
    config = portfolio_data.get("config", {}) or {}
    opt_series = config.get("selected_series", []) or []
    if not window_weights or not opt_series or not raw_data:
        return [], []

    with timed_block(
        "portopt.render_frontier_table",
        portfolio=selected_portfolio,
        series_count=len(opt_series),
        risk_measure=rm,
    ):
        model = config.get("model", "")
        resolved_idx, _ = _resolve_frontier_window(window_weights, window_idx)
        risk_measure = rm or "MV"
        if model in {"ex_ante_mv", "black_litterman"} and risk_measure == "CVaR":
            risk_measure = "MV"

        snapshot = None
        if model in {"ex_ante_mv", "black_litterman"}:
            snapshot = _get_cached_frontier_snapshot(portfolio_data, resolved_idx, risk_measure)

        if snapshot is None:
            try:
                snapshot = _build_frontier_snapshot(
                    selected_portfolio=selected_portfolio,
                    portfolio_data=portfolio_data,
                    raw_data=raw_data,
                    periodicity=periodicity,
                    bench=bench,
                    ls=ls,
                    vol_scaler=vol_scaler,
                    vol_scaling=vol_scaling,
                    window_idx=resolved_idx,
                    rm=risk_measure,
                    linear_constraints=linear_constraints,
                    saved_series_store=saved_series_store,
                    cmabench_assignments=cmabench_assignments,
                )
            except Exception:
                return [], []

        return _build_frontier_column_defs(snapshot), _build_frontier_table_rows(snapshot)


@callback(
    Output("po-frontier-rf-warning", "children"),
    Output("po-frontier-rf-warning", "style"),
    Input("po-weight-portfolio-select", "value"),
    Input("po-results-store", "data"),
    Input("po-vis-tabs", "value"),
    Input("po-frontier-window-select", "value"),
    Input("po-frontier-rm-select", "value"),
    State("po-periodicity-select", "value"),
    State("analyticstool-saved-series-cache-store", "data"),
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


