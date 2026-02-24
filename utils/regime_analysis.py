"""Regime analysis helpers for AnalyticsTool."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

import cache_config
from utils.regime_definitions import validate_regime_definition_payload
from utils.returns import annualization_factor, calculate_excess_returns, get_working_returns
from utils.serialization import canonical_json_dumps, date_range_payload_for_cache, mapping_payload_for_cache
from utils.statistics import maximum_drawdown, sharpe_ratio, sortino_ratio


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _as_tuple(series: Any) -> tuple:
    if isinstance(series, (list, tuple)):
        return tuple(str(v) for v in series if str(v).strip())
    if series is None:
        return tuple()
    text_val = str(series).strip()
    if not text_val:
        return tuple()
    return (text_val,)


def prepare_regime_input_frame(
    raw_data: str,
    periodicity: str,
    selected_series: list[str] | tuple[str, ...],
    return_basis: str,
    benchmark_assignments: dict | str | None,
    long_short_assignments: dict | str | None,
    date_range: dict | str | None,
    vol_scaler: float = 0,
    vol_scaling_assignments: dict | str | None = None,
) -> pd.DataFrame:
    """Build the input return frame used for regime construction/analysis."""
    series_tuple = _as_tuple(selected_series)
    if not raw_data or not series_tuple:
        return pd.DataFrame()

    benchmark_payload = mapping_payload_for_cache(benchmark_assignments)
    long_short_payload = mapping_payload_for_cache(long_short_assignments)
    date_payload = date_range_payload_for_cache(date_range)
    vol_scaling_payload = mapping_payload_for_cache(vol_scaling_assignments)
    periodicity_value = periodicity or "daily"

    if str(return_basis or "total").lower() == "excess":
        frame = calculate_excess_returns(
            raw_data,
            periodicity_value,
            series_tuple,
            benchmark_payload,
            "excess",
            long_short_payload,
            date_payload,
            _safe_float(vol_scaler, 0.0),
            vol_scaling_payload,
        )
    else:
        frame = get_working_returns(
            raw_data,
            periodicity_value,
            series_tuple,
            benchmark_payload,
            long_short_payload,
            date_payload,
            _safe_float(vol_scaler, 0.0),
            vol_scaling_payload,
        )
        frame = frame[[c for c in series_tuple if c in frame.columns]]

    if frame is None or frame.empty:
        return pd.DataFrame()
    frame = frame.replace([np.inf, -np.inf], np.nan).sort_index()
    return frame.dropna(how="all")


def _pc1_series(frame: pd.DataFrame, standardize: bool) -> tuple[pd.Series, str | None]:
    data = frame.dropna(how="any")
    if data.empty:
        return pd.Series(dtype=float), "No complete observations available for PC1."

    work = data.astype(float).copy()
    if standardize:
        std = work.std(ddof=0)
        nonzero_cols = [c for c in work.columns if pd.notna(std[c]) and not np.isclose(std[c], 0.0)]
        if not nonzero_cols:
            return pd.Series(dtype=float), "PC1 requires at least one non-constant series."
        work = (work[nonzero_cols] - work[nonzero_cols].mean()) / std[nonzero_cols]
    else:
        nonzero_cols = [c for c in work.columns if pd.notna(work[c].std(ddof=0)) and not np.isclose(work[c].std(ddof=0), 0.0)]
        if nonzero_cols:
            work = work[nonzero_cols]

    if work.empty:
        return pd.Series(dtype=float), "PC1 requires at least one valid series."

    if work.shape[1] == 1:
        out = pd.to_numeric(work.iloc[:, 0], errors="coerce").dropna()
        out.name = "PC1"
        return out, None

    pca = PCA(n_components=1)
    scores = pca.fit_transform(work.values).reshape(-1)
    out = pd.Series(scores, index=work.index, name="PC1", dtype=float)
    return out.dropna(), None


def _relabel_states_by_signal(states_zero: pd.Series, signal: pd.Series) -> pd.Series:
    grouped = pd.concat([states_zero.rename("state"), signal.rename("signal")], axis=1).dropna()
    if grouped.empty:
        return pd.Series(dtype=int)
    means = grouped.groupby("state")["signal"].mean().sort_values()
    order = [int(v) for v in means.index.tolist()]
    mapping = {old: idx + 1 for idx, old in enumerate(order)}
    relabeled = grouped["state"].astype(int).map(mapping).astype("Int64")
    relabeled.index = grouped.index
    relabeled.name = "Regime"
    return relabeled


def _compute_hmm_pc1_states(pc1: pd.Series, num_regimes: int) -> tuple[pd.Series, dict[str, Any]]:
    diagnostics: dict[str, Any] = {
        "method": "hmm_pc1",
        "converged": False,
        "aic": None,
        "bic": None,
        "log_likelihood": None,
        "warning": None,
    }
    if pc1.empty:
        diagnostics["warning"] = "No PC1 values are available."
        return pd.Series(dtype=int), diagnostics
    if len(pc1) < max(30, int(num_regimes) * 10):
        diagnostics["warning"] = f"HMM requires more observations (found {len(pc1)})."
        return pd.Series(dtype=int), diagnostics

    try:
        model = MarkovRegression(
            pc1.astype(float),
            k_regimes=int(num_regimes),
            trend="c",
            switching_variance=True,
        )
        result = model.fit(disp=False, maxiter=300)
        diagnostics["converged"] = bool(getattr(result, "mle_retvals", {}).get("converged", True))
        diagnostics["aic"] = _safe_float(getattr(result, "aic", np.nan), np.nan)
        diagnostics["bic"] = _safe_float(getattr(result, "bic", np.nan), np.nan)
        diagnostics["log_likelihood"] = _safe_float(getattr(result, "llf", np.nan), np.nan)

        probs = result.smoothed_marginal_probabilities
        if isinstance(probs, pd.DataFrame):
            probs_df = probs.copy()
        else:
            probs_df = pd.DataFrame(np.asarray(probs), index=pc1.index)
        if probs_df.empty:
            diagnostics["warning"] = "Unable to derive HMM probabilities."
            return pd.Series(dtype=int), diagnostics

        states_zero = probs_df.to_numpy().argmax(axis=1)
        states_zero_series = pd.Series(states_zero, index=pc1.index, name="state", dtype=int)
        relabeled = _relabel_states_by_signal(states_zero_series, pc1)
        if relabeled.empty:
            diagnostics["warning"] = "Unable to label HMM states."
            return pd.Series(dtype=int), diagnostics

        occupancy = relabeled.value_counts().sort_index()
        diagnostics["occupancy"] = {int(k): int(v) for k, v in occupancy.items()}
        if len(occupancy) < int(num_regimes) or occupancy.min() < 3:
            diagnostics["warning"] = "One or more regimes have very low occupancy."
        if not diagnostics["converged"]:
            diagnostics["warning"] = "HMM did not converge."
        return relabeled, diagnostics
    except Exception as exc:
        diagnostics["warning"] = f"HMM fit failed: {exc}"
        return pd.Series(dtype=int), diagnostics


def _compute_quantile_states(signal: pd.Series, num_regimes: int, label: str) -> tuple[pd.Series, dict[str, Any]]:
    diagnostics: dict[str, Any] = {"method": label, "warning": None}
    clean = pd.to_numeric(signal, errors="coerce").dropna()
    if clean.empty:
        diagnostics["warning"] = "No values are available for quantile assignment."
        return pd.Series(dtype=int), diagnostics
    if clean.nunique() < 2:
        diagnostics["warning"] = "Quantile assignment requires at least two unique values."
        return pd.Series(dtype=int), diagnostics
    if len(clean) < int(num_regimes):
        diagnostics["warning"] = f"Need at least {num_regimes} observations for {num_regimes} quantiles."
        return pd.Series(dtype=int), diagnostics

    ranks = clean.rank(method="first")
    try:
        labels = pd.qcut(ranks, q=int(num_regimes), labels=False, duplicates="drop")
    except Exception as exc:
        diagnostics["warning"] = f"Quantile assignment failed: {exc}"
        return pd.Series(dtype=int), diagnostics
    if labels is None:
        diagnostics["warning"] = "Quantile assignment failed."
        return pd.Series(dtype=int), diagnostics

    states = labels.astype(int) + 1
    states.index = clean.index
    states.name = "Regime"
    occupancy = states.value_counts().sort_index()
    diagnostics["occupancy"] = {int(k): int(v) for k, v in occupancy.items()}
    if len(occupancy) < int(num_regimes):
        diagnostics["warning"] = "Dropped quantile bins due to duplicate edges."
    return states.astype("Int64"), diagnostics


def compute_regime_assignments_core(
    raw_data: str,
    periodicity: str,
    definition: dict[str, Any],
    date_range: dict | str | None = None,
) -> dict[str, Any]:
    normalized, error = validate_regime_definition_payload(definition or {})
    if error or not normalized:
        return {
            "states": pd.Series(dtype=int),
            "diagnostics": {"warning": error or "Invalid regime definition."},
            "analysis_df": pd.DataFrame(),
            "signal": pd.Series(dtype=float),
        }

    config = normalized["Config"]
    method_type = int(normalized["MethodType"])
    return_basis = str(config.get("return_basis", "total"))
    vol_scaler = _safe_float(config.get("vol_scaler"), 0.0)
    min_obs = int(config.get("min_observations", 60))

    if method_type in {1, 2}:
        universe = [str(v) for v in config.get("universe_series", []) if str(v).strip()]
    else:
        single = str(config.get("single_series", "")).strip()
        universe = [single] if single else []

    analysis_df = prepare_regime_input_frame(
        raw_data=raw_data,
        periodicity=periodicity or "daily",
        selected_series=universe,
        return_basis=return_basis,
        benchmark_assignments=config.get("benchmark_assignments"),
        long_short_assignments=config.get("long_short_assignments"),
        date_range=date_range,
        vol_scaler=vol_scaler,
        vol_scaling_assignments=config.get("vol_scaling_assignments"),
    )
    if analysis_df.empty:
        return {
            "states": pd.Series(dtype=int),
            "diagnostics": {"warning": "No analysis data available for the current definition."},
            "analysis_df": analysis_df,
            "signal": pd.Series(dtype=float),
        }

    num_regimes = int(config.get("num_regimes", 3))
    if method_type in {1, 2}:
        signal, signal_error = _pc1_series(
            analysis_df,
            bool(config.get("pca_standardize", True)),
        )
        if signal_error:
            return {
                "states": pd.Series(dtype=int),
                "diagnostics": {"warning": signal_error},
                "analysis_df": analysis_df,
                "signal": pd.Series(dtype=float),
            }
        if len(signal) < min_obs:
            return {
                "states": pd.Series(dtype=int),
                "diagnostics": {"warning": f"Need at least {min_obs} observations; found {len(signal)}."},
                "analysis_df": analysis_df,
                "signal": signal,
            }
        if method_type == 1:
            states, diagnostics = _compute_hmm_pc1_states(signal, num_regimes)
        else:
            states, diagnostics = _compute_quantile_states(signal, num_regimes, "quantile_pc1")
    else:
        series_name = universe[0] if universe else ""
        if not series_name or series_name not in analysis_df.columns:
            return {
                "states": pd.Series(dtype=int),
                "diagnostics": {"warning": "Single-series quantile requires a valid series."},
                "analysis_df": analysis_df,
                "signal": pd.Series(dtype=float),
            }
        signal = pd.to_numeric(analysis_df[series_name], errors="coerce").dropna()
        if len(signal) < min_obs:
            return {
                "states": pd.Series(dtype=int),
                "diagnostics": {"warning": f"Need at least {min_obs} observations; found {len(signal)}."},
                "analysis_df": analysis_df,
                "signal": signal,
            }
        states, diagnostics = _compute_quantile_states(signal, num_regimes, "quantile_single_series")

    states = states.dropna().astype("Int64")
    states.name = "Regime"
    diagnostics["method_type"] = method_type
    diagnostics["num_regimes"] = num_regimes
    diagnostics["observations"] = int(len(states))
    return {
        "states": states,
        "diagnostics": diagnostics,
        "analysis_df": analysis_df,
        "signal": signal,
    }


@cache_config.cache.memoize(timeout=0)
def compute_regime_assignments_cached(
    raw_data: str,
    periodicity: str,
    definition_payload_json: str,
    date_range_payload_json: str,
) -> dict[str, Any]:
    try:
        definition = json.loads(str(definition_payload_json or "{}"))
        if not isinstance(definition, dict):
            definition = {}
    except Exception:
        definition = {}
    try:
        date_range = json.loads(str(date_range_payload_json or "null"))
    except Exception:
        date_range = date_range_payload_json

    result = compute_regime_assignments_core(
        raw_data=raw_data,
        periodicity=periodicity,
        definition=definition,
        date_range=date_range,
    )
    states = result.get("states", pd.Series(dtype=int))
    diagnostics = result.get("diagnostics", {}) or {}
    if states is None or states.empty:
        states_json = None
    else:
        state_frame = pd.DataFrame(
            {
                "Date": pd.to_datetime(pd.Index(states.index), errors="coerce"),
                "Regime": pd.to_numeric(states, errors="coerce").astype("Int64").to_numpy(),
            }
        )
        state_frame["Date"] = pd.to_datetime(state_frame["Date"], errors="coerce")
        state_frame = state_frame.dropna(subset=["Date"])
        state_frame = state_frame.sort_values("Date")
        state_frame["Date"] = state_frame["Date"].dt.strftime("%Y-%m-%d")
        states_json = canonical_json_dumps(state_frame.to_dict("records"))
    return {
        "states_json": states_json,
        "diagnostics": diagnostics,
    }


def compute_regime_assignments(
    raw_data: str,
    periodicity: str,
    definition: dict[str, Any],
    date_range: dict | str | None,
) -> tuple[pd.Series, dict[str, Any]]:
    payload = canonical_json_dumps(definition or {})
    date_payload = date_range_payload_for_cache(date_range)
    result = compute_regime_assignments_cached(
        raw_data=raw_data,
        periodicity=periodicity or "daily",
        definition_payload_json=payload,
        date_range_payload_json=date_payload,
    )
    diagnostics = result.get("diagnostics", {}) if isinstance(result, dict) else {}
    states_json = result.get("states_json") if isinstance(result, dict) else None
    if not states_json:
        return pd.Series(dtype="Int64", name="Regime"), diagnostics

    try:
        records = json.loads(states_json)
    except Exception:
        return pd.Series(dtype="Int64", name="Regime"), diagnostics
    if not isinstance(records, list) or not records:
        return pd.Series(dtype="Int64", name="Regime"), diagnostics
    frame = pd.DataFrame(records)
    if frame.empty or "Date" not in frame.columns or "Regime" not in frame.columns:
        return pd.Series(dtype="Int64", name="Regime"), diagnostics
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame["Regime"] = pd.to_numeric(frame["Regime"], errors="coerce").astype("Int64")
    frame = frame.dropna(subset=["Date", "Regime"]).sort_values("Date")
    states = pd.Series(frame["Regime"].to_numpy(), index=pd.DatetimeIndex(frame["Date"]), name="Regime", dtype="Int64")
    return states, diagnostics


def build_regime_statistics_table(returns_df: pd.DataFrame, states: pd.Series, periodicity: str) -> pd.DataFrame:
    if returns_df is None or returns_df.empty or states is None or states.empty:
        return pd.DataFrame()
    aligned = returns_df.copy()
    aligned.index = pd.to_datetime(aligned.index, errors="coerce")
    states_aligned = pd.Series(states.copy())
    states_aligned.index = pd.to_datetime(states_aligned.index, errors="coerce")
    merged = aligned.join(states_aligned.rename("Regime"), how="inner")
    merged = merged.dropna(subset=["Regime"])
    if merged.empty:
        return pd.DataFrame()

    periods_per_year = annualization_factor(periodicity or "daily")
    rows: list[dict[str, Any]] = []
    regimes = sorted(int(v) for v in merged["Regime"].dropna().astype(int).unique())
    for regime in regimes:
        subset = merged[merged["Regime"].astype(int) == regime]
        for col in [c for c in subset.columns if c != "Regime"]:
            series = pd.to_numeric(subset[col], errors="coerce").dropna()
            if series.empty:
                continue
            rows.append(
                {
                    "Regime": regime,
                    "Series": col,
                    "Observations": int(len(series)),
                    "Mean Return": float(series.mean()),
                    "Volatility": float(series.std(ddof=1) * np.sqrt(periods_per_year)) if len(series) > 1 else np.nan,
                    "Sharpe": float(sharpe_ratio(series, periods_per_year, rf=0.0)),
                    "Sortino": float(sortino_ratio(series, periods_per_year, rf=0.0)),
                    "Min Return": float(series.min()),
                    "Max Return": float(series.max()),
                    "Hit Rate": float((series > 0).mean()),
                    "Max Drawdown": float(maximum_drawdown(series)),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["Regime", "Series"]).reset_index(drop=True)


def build_regime_transition_matrix(states: pd.Series) -> pd.DataFrame:
    if states is None or states.empty or len(states) < 2:
        return pd.DataFrame()
    s = pd.to_numeric(states, errors="coerce").dropna().astype(int)
    if len(s) < 2:
        return pd.DataFrame()
    current = s.iloc[:-1]
    nxt = s.iloc[1:]
    matrix = pd.crosstab(current, nxt, normalize="index")
    regimes = sorted(set(s.tolist()))
    matrix = matrix.reindex(index=regimes, columns=regimes, fill_value=0.0)
    matrix.index.name = "From Regime"
    matrix.columns.name = "To Regime"
    return matrix


def build_regime_duration_table(states: pd.Series) -> pd.DataFrame:
    if states is None or states.empty:
        return pd.DataFrame()
    s = pd.to_numeric(states, errors="coerce").dropna().astype(int)
    if s.empty:
        return pd.DataFrame()

    frame = pd.DataFrame({"Regime": s})
    frame["Date"] = pd.to_datetime(frame.index, errors="coerce")
    frame = frame.dropna(subset=["Date"])
    if frame.empty:
        return pd.DataFrame()
    frame["RunId"] = frame["Regime"].ne(frame["Regime"].shift()).cumsum()

    runs = frame.groupby("RunId").agg(
        Regime=("Regime", "first"),
        StartDate=("Date", "first"),
        EndDate=("Date", "last"),
        Length=("Date", "size"),
    )
    if runs.empty:
        return pd.DataFrame()

    summary = runs.groupby("Regime").agg(
        **{
            "Runs": ("Length", "size"),
            "Mean Run Length": ("Length", "mean"),
            "Median Run Length": ("Length", "median"),
            "Max Run Length": ("Length", "max"),
        }
    )
    current_map: dict[int, int] = {}
    for regime, sub in runs.groupby("Regime"):
        current_map[int(regime)] = int(sub.iloc[-1]["Length"])
    summary["Current Run Length"] = [current_map.get(int(idx), np.nan) for idx in summary.index]
    summary = summary.reset_index().rename(columns={"Regime": "Regime"})
    return summary.sort_values("Regime").reset_index(drop=True)


def build_regime_timeline_frame(states: pd.Series) -> pd.DataFrame:
    if states is None or states.empty:
        return pd.DataFrame()
    out = pd.DataFrame({"Date": pd.to_datetime(states.index, errors="coerce"), "Regime": states.values})
    out["Regime"] = pd.to_numeric(out["Regime"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["Date", "Regime"]).sort_values("Date")
    return out.reset_index(drop=True)


def build_regime_conditioned_summary(returns_df: pd.DataFrame, states: pd.Series) -> pd.DataFrame:
    if returns_df is None or returns_df.empty or states is None or states.empty:
        return pd.DataFrame()
    aligned = returns_df.copy()
    aligned.index = pd.to_datetime(aligned.index, errors="coerce")
    states_aligned = pd.Series(states.copy())
    states_aligned.index = pd.to_datetime(states_aligned.index, errors="coerce")
    merged = aligned.join(states_aligned.rename("Regime"), how="inner").dropna(subset=["Regime"])
    if merged.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    regimes = sorted(int(v) for v in merged["Regime"].dropna().astype(int).unique())
    for regime in regimes:
        subset = merged[merged["Regime"].astype(int) == regime]
        for col in [c for c in subset.columns if c != "Regime"]:
            series = pd.to_numeric(subset[col], errors="coerce").dropna()
            if series.empty:
                continue
            growth = (1.0 + series).cumprod()
            rows.append(
                {
                    "Regime": regime,
                    "Series": col,
                    "Observations": int(len(series)),
                    "Growth End": float(growth.iloc[-1]) if not growth.empty else np.nan,
                    "Max Drawdown": float(maximum_drawdown(series)),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["Regime", "Series"]).reset_index(drop=True)
