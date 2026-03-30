from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from statistics import median

import pandas as pd
from playwright.sync_api import sync_playwright

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.playwright import warm_switch_harness as warm
from utils.raw_dataset import build_raw_data_store_payload
from utils.returns import df_to_json


DEFAULT_DB_SERIES = warm.DEFAULT_DB_SERIES
QUIET_WINDOW_MS = 250
QUIET_WINDOW_TIMEOUT_MS = 10000
TIMEOUT_SCALE = 1.0

NETWORK_PROFILES = {
    "none": None,
    "office-wan": {
        "latencyMs": 40,
        "downloadKbps": 10000,
        "uploadKbps": 5000,
        "connectionType": "cellular4g",
    },
    "slow4g": {
        "latencyMs": 150,
        "downloadKbps": 4000,
        "uploadKbps": 3000,
        "connectionType": "cellular4g",
    },
    "fast3g": {
        "latencyMs": 150,
        "downloadKbps": 1600,
        "uploadKbps": 750,
        "connectionType": "cellular3g",
    },
}

NETWORK_TIMEOUT_MULTIPLIERS = {
    "none": 1.0,
    "office-wan": 3.0,
    "slow4g": 4.0,
    "fast3g": 5.0,
}


def _kbps_to_bytes_per_second(kbps: int) -> int:
    return max(int(kbps * 1000 / 8), 1)


def _scaled_timeout(timeout_ms: int) -> int:
    return max(int(round(timeout_ms * TIMEOUT_SCALE)), timeout_ms)


def _apply_network_profile(page, profile_name: str) -> dict[str, object] | None:
    profile = NETWORK_PROFILES.get(profile_name)
    if not profile:
        return None
    session = page.context.new_cdp_session(page)
    session.send("Network.enable")
    session.send(
        "Network.emulateNetworkConditions",
        {
            "offline": False,
            "latency": int(profile["latencyMs"]),
            "downloadThroughput": _kbps_to_bytes_per_second(int(profile["downloadKbps"])),
            "uploadThroughput": _kbps_to_bytes_per_second(int(profile["uploadKbps"])),
            "connectionType": str(profile["connectionType"]),
        },
    )
    return {"name": profile_name, **profile}


def build_synthetic_raw_dataset(series_names: list[str]) -> tuple[dict[str, object], dict[str, object]]:
    df = build_synthetic_raw_frame(series_names)
    return build_raw_data_store_payload(df), {"columns": list(df.columns)}


def build_synthetic_raw_frame(series_names: list[str]) -> pd.DataFrame:
    resolved_series = list(series_names or DEFAULT_DB_SERIES)
    index = pd.bdate_range("2020-01-01", periods=320)
    data = {}
    for idx, series in enumerate(resolved_series):
        offset = (idx + 1) * 0.00015
        values = [offset + ((((day + idx) % 13) - 6) / 10000.0) for day in range(len(index))]
        data[series] = values
    df = pd.DataFrame(data, index=index)
    df.index.name = "Date"
    return df


def _timestamp_token() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def summarize_run_group(run_results: list[dict[str, object]]) -> dict[str, object]:
    flow_values = [int(run["flowMs"]) for run in run_results]
    targeted_values = [int(run["targetedDashUpdateRequestCount"]) for run in run_results]
    request_values = [int(run["dashUpdateRequestCount"]) for run in run_results]
    request_bytes = [int(run["dashUpdateRequestBytes"]) for run in run_results]
    response_bytes = [int(run["dashUpdateResponseBytes"]) for run in run_results]
    server_values = [int(run["dashUpdateSummedServerMs"]) for run in run_results]
    return {
        "runs": len(run_results),
        "perfTarget": any(bool(run.get("perfTarget", True)) for run in run_results),
        "scenarioClass": str(run_results[0].get("scenarioClass", "ui_only")) if run_results else "ui_only",
        "flowMedian": round(median(flow_values)) if flow_values else 0,
        "targetedDashUpdateRequestCountMedian": round(median(targeted_values)) if targeted_values else 0,
        "dashUpdateRequestCountMedian": round(median(request_values)) if request_values else 0,
        "dashUpdateRequestBytesMedian": round(median(request_bytes)) if request_bytes else 0,
        "dashUpdateResponseBytesMedian": round(median(response_bytes)) if response_bytes else 0,
        "dashUpdateSummedServerMsMedian": round(median(server_values)) if server_values else 0,
    }


def filter_targeted_requests(summary: dict[str, object], targeted_outputs: list[str]) -> dict[str, object]:
    targeted = []
    targeted_set = set(targeted_outputs)
    for record in summary.get("dashUpdateRequests", []):
        outputs = [str(output) for output in record.get("outputs", [])]
        if any(output in targeted_set for output in outputs):
            targeted.append(record)

    callbacks: list[str] = []
    total_duration = 0
    total_server_ms = 0.0
    total_request_bytes = 0
    total_response_bytes = 0
    for record in targeted:
        total_duration += int(record.get("durationMs", 0) or 0)
        server_ms = record.get("serverMs")
        if server_ms is not None:
            total_server_ms += float(server_ms)
        total_request_bytes += int(record.get("requestBytes", 0) or 0)
        total_response_bytes += int(record.get("responseBytes", 0) or 0)
        for output in record.get("outputs", []):
            if output in targeted_set and output not in callbacks:
                callbacks.append(output)

    return {
        "targetedOutputs": list(targeted_outputs),
        "targetedDashUpdateRequestCount": len(targeted),
        "targetedDashUpdateTotalMs": total_duration,
        "targetedDashUpdateSummedServerMs": round(total_server_ms),
        "targetedDashUpdateRequestBytes": total_request_bytes,
        "targetedDashUpdateResponseBytes": total_response_bytes,
        "targetedDashUpdateCallbacks": callbacks,
        "targetedDashUpdateRequests": targeted,
    }


def wait_for_input_disabled(page, selector: str, expected: bool, timeout: int = 10000) -> None:
    page.wait_for_function(
        """
        ([sel, wantDisabled]) => {
          const el = document.querySelector(sel);
          if (!el) return false;
          const target = el.matches("input,button,textarea,[role='textbox']") ? el : el.querySelector("input,button,textarea,[role='textbox']");
          if (!target) return false;
          const disabled = !!target.disabled || target.getAttribute("disabled") !== null || target.getAttribute("aria-disabled") === "true";
          return disabled === !!wantDisabled;
        }
        """,
        arg=[selector, expected],
        timeout=_scaled_timeout(timeout),
    )


def wait_for_input_value(page, selector: str, expected: str, timeout: int = 10000) -> None:
    page.wait_for_function(
        """
        ([sel, wantValue]) => {
          const el = document.querySelector(sel);
          if (!el) return false;
          const target = el.matches("input,textarea,[role='textbox']") ? el : el.querySelector("input,textarea,[role='textbox']");
          if (!target) return false;
          return String(target.value || "").trim() === String(wantValue || "").trim();
        }
        """,
        arg=[selector, expected],
        timeout=_scaled_timeout(timeout),
    )


def wait_for_text_content(page, selector: str, expected: str, timeout: int = 10000) -> None:
    page.wait_for_function(
        """
        ([sel, wantText]) => {
          const el = document.querySelector(sel);
          if (!el) return false;
          const text = (el.innerText || el.textContent || "").trim();
          return text === String(wantText || "").trim();
        }
        """,
        arg=[selector, expected],
        timeout=_scaled_timeout(timeout),
    )


def wait_for_js_condition(page, function_body: str, timeout: int = 10000) -> None:
    page.wait_for_function(function_body, timeout=_scaled_timeout(timeout))


def wait_for_style_display(page, selector: str, expected: str, timeout: int = 10000) -> None:
    page.wait_for_function(
        """
        ([sel, wantDisplay]) => {
          const el = document.querySelector(sel);
          if (!el) return false;
          return window.getComputedStyle(el).display === wantDisplay;
        }
        """,
        arg=[selector, expected],
        timeout=_scaled_timeout(timeout),
    )


def wait_for_component_prop(page, component_id: str, prop_name: str, expected, timeout: int = 10000) -> None:
    page.wait_for_function(
        """
        ([componentId, propName, wantValue]) => {
          const root = (((window.store || {}).getState || (() => ({})))().layout || {}).components;
          if (!root) return false;
          const stack = [root];
          while (stack.length) {
            const node = stack.pop();
            if (!node || typeof node !== "object") continue;
            const props = node.props || {};
            if (props.id === componentId) {
              return JSON.stringify(props[propName]) === JSON.stringify(wantValue);
            }
            const children = props.children;
            if (Array.isArray(children)) {
              for (const child of children) stack.push(child);
            } else if (children && typeof children === "object") {
              stack.push(children);
            }
          }
          return false;
        }
        """,
        arg=[component_id, prop_name, expected],
        timeout=_scaled_timeout(timeout),
    )


def wait_for_plotly_graph_ready(
    page,
    graph_selector: str,
    empty_selector: str,
    legacy_container_selector: str | None = None,
    timeout: int = 10000,
) -> None:
    page.wait_for_function(
        """
        ([graphSel, emptySel, legacySel]) => {
          const extractPlot = (root) => {
            if (!root) return null;
            if (root.matches && root.matches(".js-plotly-plot")) return root;
            return root.querySelector ? root.querySelector(".js-plotly-plot") : null;
          };
          const hasPlotData = (plot) => {
            const plotData = plot ? (plot.data || plot._fullData || []) : [];
            return Array.isArray(plotData) && plotData.length > 0;
          };
          const graphRoot = document.querySelector(graphSel);
          const emptyRoot = document.querySelector(emptySel);
          if (graphRoot && emptyRoot) {
            const graphVisible = window.getComputedStyle(graphRoot).display !== "none";
            const emptyHidden = window.getComputedStyle(emptyRoot).display === "none";
            if (graphVisible && emptyHidden && hasPlotData(extractPlot(graphRoot))) {
              return true;
            }
          }
          if (!legacySel) return false;
          const legacyRoot = document.querySelector(legacySel);
          if (!legacyRoot) return false;
          return hasPlotData(extractPlot(legacyRoot));
        }
        """,
        arg=[graph_selector, empty_selector, legacy_container_selector],
        timeout=_scaled_timeout(timeout),
    )


def wait_for_quiet_window(
    page,
    tracker: warm.DashUpdateRequestTracker,
    quiet_ms: int = QUIET_WINDOW_MS,
    timeout_ms: int = QUIET_WINDOW_TIMEOUT_MS,
) -> None:
    deadline = time.perf_counter() + (_scaled_timeout(timeout_ms) / 1000.0)
    stable_started_at: float | None = None
    last_record_count = len(tracker.records)
    while time.perf_counter() < deadline:
        active = bool(tracker.active_requests)
        record_count = len(tracker.records)
        now = time.perf_counter()
        if active or record_count != last_record_count:
            stable_started_at = None
            last_record_count = record_count
        else:
            if stable_started_at is None:
                stable_started_at = now
            elif ((now - stable_started_at) * 1000) >= quiet_ms:
                return
        page.wait_for_timeout(25)
    raise TimeoutError(f"Timed out waiting for a {quiet_ms}ms quiet window")


def wait_for_dash_update_requests(
    page,
    tracker: warm.DashUpdateRequestTracker,
    minimum_count: int = 1,
    timeout: int = 10000,
) -> None:
    deadline = time.perf_counter() + (_scaled_timeout(timeout) / 1000.0)
    while time.perf_counter() < deadline:
        if len(tracker.records) >= minimum_count:
            return
        page.wait_for_timeout(25)
    raise TimeoutError(f"Expected at least {minimum_count} Dash update request(s) before timeout")


def wait_content_ready(page, selector: str, timeout: int = 10000) -> None:
    warm.wait_content_ready(page, selector, timeout=_scaled_timeout(timeout))


def wait_analytics_tab_ready(page, active_tab: str = "statistics", timeout: int = 10000) -> None:
    warm.wait_analytics_tab_ready(page, active_tab=active_tab, timeout=_scaled_timeout(timeout))


def wait_analytics_rolling_ready(page, timeout: int = 10000) -> None:
    page.wait_for_function(
        """
        () => {
          const title = (document.title || "").trim();
          if (!title || title === "Updating...") return false;
          const switchRoot = document.querySelector("#at-rolling-chart-switch");
          const chartWrapper = document.querySelector("#at-rolling-chart-wrapper");
          const grid = document.querySelector("#at-rolling-grid");
          const isVisible = (el) => {
            if (!el) return false;
            const style = window.getComputedStyle(el);
            const rect = el.getBoundingClientRect();
            return style.display !== "none" && style.visibility !== "hidden" && rect.width > 0 && rect.height > 0;
          };
          const switchValue = switchRoot && switchRoot.value ? String(switchRoot.value) : "chart";
          if (switchValue === "table") {
            return isVisible(grid);
          }
          const plot = chartWrapper && chartWrapper.querySelector ? chartWrapper.querySelector(".js-plotly-plot") : null;
          const plotData = plot ? (plot.data || plot._fullData || []) : [];
          return isVisible(chartWrapper) && Array.isArray(plotData) && plotData.length > 0;
        }
        """,
        timeout=_scaled_timeout(timeout),
    )


def wait_persisted_store_value(page, store_id: str, expected, timeout: int = 10000) -> None:
    warm.wait_for_persisted_store_value(page, store_id, expected, timeout=_scaled_timeout(timeout))


def wait_dash_hydrated(page, timeout: int = 30000) -> None:
    warm.wait_dash_hydrated(page, timeout=_scaled_timeout(timeout))


def wait_visible(page, selector: str, timeout: int = 30000) -> None:
    warm.wait_visible(page, selector, timeout=_scaled_timeout(timeout))


def wait_ready(page, selector: str, timeout: int = 30000) -> None:
    warm.wait_ready(page, selector, timeout=_scaled_timeout(timeout))


def wait_hidden_or_absent(page, selector: str, timeout: int = 30000) -> None:
    warm.wait_hidden_or_absent(page, selector, timeout=_scaled_timeout(timeout))


def measure_scenario(
    *,
    page,
    tracker: warm.DashUpdateRequestTracker,
    scenario_name: str,
    targeted_outputs: list[str],
    prepare: Callable[[], None] | None,
    action: Callable[[], None],
    wait_for_ready: Callable[[], None] | None,
    perf_target: bool = True,
    scenario_class: str = "ui_only",
) -> dict[str, object]:
    if prepare is not None:
        prepare()
    tracker.wait_for_settle()
    wait_for_quiet_window(page, tracker)

    start = time.perf_counter()
    tracker.start_window()
    action()
    if wait_for_ready is not None:
        wait_for_ready()
    tracker.wait_for_settle()
    wait_for_quiet_window(page, tracker)
    tracker.stop_window()
    flow_ms = round((time.perf_counter() - start) * 1000)
    summary = tracker.summary()
    targeted = filter_targeted_requests(summary, targeted_outputs)
    return {
        "scenario": scenario_name,
        "flowMs": flow_ms,
        "perfTarget": perf_target,
        "scenarioClass": scenario_class,
        **summary,
        **targeted,
    }


def build_seeded_regression_results(series_names: list[str]) -> dict[str, object]:
    dep_var, x_series = warm.resolve_regression_series(series_names)
    series_order = [dep_var] + x_series
    raw_df = build_synthetic_raw_frame(series_order)
    result_index = raw_df.index[60:220]

    def _result_entry(result_id: int, prediction_scale: float, coefficient_scale: float) -> dict[str, object]:
        actual = raw_df.loc[result_index, dep_var]
        predicted = actual * prediction_scale
        residuals = actual - predicted
        display_df = pd.concat(
            [
                predicted.rename("Predicted"),
                actual.rename("Actual (Y)"),
                raw_df.loc[result_index, x_series],
                residuals.rename("Residual"),
            ],
            axis=1,
        )
        coefficients = {
            x_name: round(coefficient_scale / max(len(x_series), 1), 4)
            for x_name in x_series
        }
        window_result = {
            "est_start": str(result_index[0])[:10],
            "est_end": str(result_index[-1])[:10],
            "apply_start": str(result_index[0])[:10],
            "apply_end": str(result_index[-1])[:10],
            "coefficients": coefficients,
            "p_values": {x_name: 0.05 for x_name in x_series},
            "r_squared": round(0.72 - (result_id * 0.04), 4),
            "adj_r_squared": round(0.68 - (result_id * 0.04), 4),
            "anova_table": {
                "df_model": len(x_series),
                "df_resid": max(len(result_index) - len(x_series) - 1, 1),
                "ss_model": 0.45,
                "ms_model": 0.45,
                "F_stat": 9.0,
                "F_pvalue": 0.05,
                "ss_resid": 0.15,
                "ms_resid": 0.05,
                "ss_total": 0.60,
            },
            "diagnostics": {
                "std_errors": {x_name: 0.1 for x_name in x_series},
                "t_stats": {x_name: 2.0 for x_name in x_series},
                "ci_low": {x_name: coefficients[x_name] - 0.1 for x_name in x_series},
                "ci_high": {x_name: coefficients[x_name] + 0.1 for x_name in x_series},
                "durbin_watson": 2.1,
                "aic": 12.3,
                "bic": 14.2,
                "vif": {x_name: 1.1 for x_name in x_series},
            },
            "residual_std": 0.014,
            "oos_metrics": {"oos_r2": 0.61, "oos_rmse": 0.02, "oos_mae": 0.01},
            "n_obs": len(result_index),
        }
        return {
            "periodicity": "daily",
            "dependent_var": dep_var,
            "independent_vars": list(x_series),
            "independent_vars_internal": list(x_series),
            "config": {
                "model": "ols",
                "window_type": "rolling",
                "window_size": 24,
                "opt_step": 1,
                "opt_step_unit": "months",
                "fill_in_sample": True,
                "missing_data": "fill_na",
                "force_zero_intercept": False,
                "robust_se": True,
                "exp_wt": False,
                "halflife": 63,
                "alpha": 1.0,
                "l1_ratio": 0.5,
            },
            "window_results": [window_result],
            "date_range": {"start": str(result_index[0])[:10], "end": str(result_index[-1])[:10]},
            "effective_date_range": {"start": str(result_index[0])[:10], "end": str(result_index[-1])[:10]},
            "vol_scaler": 0,
            "benchmark_assignments": {},
            "long_short_assignments": {},
            "vol_scaling_assignments": {},
            "predicted_json": df_to_json(predicted.to_frame("predicted")),
            "residuals_json": df_to_json(residuals.to_frame("residuals")),
            "display_json": df_to_json(display_df),
            "display_columns": list(display_df.columns),
            "saved_series_name": None,
        }

    return {
        "Harness Result 1": _result_entry(1, 0.92, 0.8),
        "Harness Result 2": _result_entry(2, 0.85, 0.55),
    }


def build_seeded_portopt_results(series_names: list[str]) -> dict[str, object]:
    opt_series = warm.resolve_portopt_series(series_names)
    raw_df = build_synthetic_raw_frame(opt_series)
    result_index = raw_df.index[40:260]
    component_df = raw_df.loc[result_index, opt_series]
    window_midpoint = len(result_index) // 2

    def _normalized_weights(weight_prefix: list[float]) -> dict[str, float]:
        padded = list(weight_prefix[: len(opt_series)])
        if len(padded) < len(opt_series):
            padded.extend([0.0] * (len(opt_series) - len(padded)))
        total = sum(padded)
        if total <= 0:
            padded = [1.0 / max(len(opt_series), 1)] * len(opt_series)
        else:
            padded = [value / total for value in padded]
        return {
            series_name: round(padded[idx], 4)
            for idx, series_name in enumerate(opt_series)
        }

    def _result_entry(name: str, weight_prefix: list[float], benchmark_name: str, risk_free_enabled: bool) -> dict[str, object]:
        portfolio_weights = _normalized_weights(weight_prefix)
        portfolio_series = (component_df * pd.Series(portfolio_weights)).sum(axis=1).rename(name)
        benchmark_series = component_df[benchmark_name].rename(f"__bm__{name}")
        window_weights = [
            {
                "apply_start": str(result_index[0])[:10],
                "apply_end": str(result_index[window_midpoint - 1])[:10],
                "est_start": str(result_index[0])[:10],
                "est_end": str(result_index[window_midpoint - 1])[:10],
                "weights": dict(portfolio_weights),
            },
            {
                "apply_start": str(result_index[window_midpoint])[:10],
                "apply_end": str(result_index[-1])[:10],
                "est_start": str(result_index[window_midpoint])[:10],
                "est_end": str(result_index[-1])[:10],
                "weights": dict(portfolio_weights),
            },
        ]
        run_inputs = {
            "selected_series": list(opt_series),
            "benchmark_assignments": {},
            "cmabench_assignments": {},
            "long_short_assignments": {},
            "date_range": {"start": str(result_index[0])[:10], "end": str(result_index[-1])[:10]},
            "vol_scaler": 0,
            "vol_scaling_assignments": {},
            "periodicity": "daily",
        }
        return {
            "config": {
                "model": "risk_parity",
                "selected_series": list(opt_series),
                "periodicity": "daily",
            },
            "run_inputs": run_inputs,
            "reporting_basis": "match_optimization",
            "window_weights": window_weights,
            "reporting_returns_json": portfolio_series.to_json(date_format="iso"),
            "optimization_returns_json": portfolio_series.to_json(date_format="iso"),
            "returns_json": portfolio_series.to_json(date_format="iso"),
            "benchmark_returns_json": benchmark_series.to_json(date_format="iso"),
            "risk_free_meta": {"enabled": risk_free_enabled},
        }

    benchmark_two = opt_series[1] if len(opt_series) > 1 else opt_series[0]
    return {
        "Harness Portfolio 1": _result_entry("Harness Portfolio 1", [0.55, 0.45, 0.0], opt_series[0], True),
        "Harness Portfolio 2": _result_entry("Harness Portfolio 2", [0.2, 0.8, 0.0], benchmark_two, False),
    }


def _seed_regression_result_state(page, db_series: list[str], tab_value: str, selected_result: str = "Harness Result 1") -> None:
    results = build_seeded_regression_results(db_series)
    options = [{"value": name, "label": name} for name in results]
    warm.set_component_props(page, "reg-results-store", {"data": results})
    warm.set_component_props(page, "reg-result-select", {"data": options, "value": selected_result})
    warm.seed_regression_restore_tab(page, tab_value)


def _seed_portopt_result_state(page, db_series: list[str], tab_value: str = "weight", selected_portfolio: str = "Harness Portfolio 1") -> None:
    results = build_seeded_portopt_results(db_series)
    options = [{"value": name, "label": name} for name in results]
    warm.set_component_props(page, "po-results-store", {"data": results})
    warm.set_component_props(page, "po-growth-portfolio-multiselect", {"data": options, "value": [selected_portfolio]})
    warm.set_component_props(page, "po-active-tab-store", {"data": tab_value})
    warm.set_component_props(page, "po-vis-tabs", {"value": tab_value})
    if tab_value == "rolling":
        warm.set_component_props(page, "po-rolling-chart-switch", {"value": "chart"})
    # Let result-store driven selector syncs settle, then force the intended starting portfolio.
    page.wait_for_timeout(600)
    warm.set_component_props(page, "po-weight-portfolio-select", {"data": options, "value": selected_portfolio})
    warm.set_component_props(page, "po-growth-portfolio-multiselect", {"data": options, "value": [selected_portfolio]})
    page.wait_for_timeout(200)


def _seed_portopt_raw_db_modal(page, rows: list[dict[str, object]] | None = None) -> None:
    warm.set_component_props(page, "po-raw-db-add-modal", {"opened": True})
    warm.set_component_props(page, "po-raw-db-add-rows-store", {"data": rows or []})
    warm.set_component_props(page, "po-raw-db-add-grid", {"rowData": rows or []})
    page.wait_for_timeout(200)


def _seed_regression_raw_db_modal(page, rows: list[dict[str, object]] | None = None, selected_rows: list[dict[str, object]] | None = None) -> None:
    seeded_rows = rows or []
    warm.set_component_props(page, "reg-raw-db-add-modal", {"opened": True})
    warm.set_component_props(page, "reg-raw-db-add-rows-store", {"data": seeded_rows})
    warm.set_component_props(page, "reg-raw-db-add-grid", {"rowData": seeded_rows, "selectedRows": selected_rows or []})
    page.wait_for_timeout(200)


def run_portopt_scenarios(page, tracker: warm.DashUpdateRequestTracker, db_series: list[str] | None = None) -> list[dict[str, object]]:
    sample_rows = [{"Series": "SPX_TRIndex", "Table": "G", "Fee": "N", "Include Benchmark": False}]
    sample_returns = [{"Asset": "SPX_TRIndex", "Return": 1.5, "Volatility": 3.0}]
    resolved_db_series = list(db_series or DEFAULT_DB_SERIES)
    results: list[dict[str, object]] = []

    results.append(
        measure_scenario(
            page=page,
            tracker=tracker,
            scenario_name="portopt_model_name_sync",
            targeted_outputs=["po-portfolio-name-input.value"],
            prepare=lambda: warm.set_component_value(page, "po-opt-model-select", "risk_parity"),
            action=lambda: warm.set_component_value(page, "po-opt-model-select", "black_litterman"),
            wait_for_ready=lambda: wait_for_input_value(page, "#po-portfolio-name-input", "BL"),
        )
    )

    results.append(
        measure_scenario(
            page=page,
            tracker=tracker,
            scenario_name="portopt_raw_db_toggle_divide_by",
            targeted_outputs=["po-raw-db-add-divide-by.disabled"],
            prepare=lambda: (
                _seed_portopt_raw_db_modal(page),
                warm.set_component_props(page, "po-raw-db-add-mode-store", {"data": "factor"}),
                warm.set_component_props(page, "po-raw-db-add-convert-returns", {"checked": False}),
            ),
            action=lambda: warm.set_component_props(page, "po-raw-db-add-convert-returns", {"checked": True}),
            wait_for_ready=None,
        )
    )

    results.append(
        measure_scenario(
            page=page,
            tracker=tracker,
            scenario_name="portopt_raw_db_clear_rows",
            targeted_outputs=[
                "po-raw-db-add-rows-store.data",
                "po-raw-db-add-grid.rowData",
                "po-raw-db-add-error-alert.children",
                "po-raw-db-add-error-alert.hide",
            ],
            prepare=lambda: _seed_portopt_raw_db_modal(page, sample_rows),
            action=lambda: warm.set_component_props(page, "po-raw-db-clear-rows-btn", {"n_clicks": 1}),
            wait_for_ready=None,
        )
    )

    results.append(
        measure_scenario(
            page=page,
            tracker=tracker,
            scenario_name="portopt_clear_returns",
            targeted_outputs=[
                "po-ex-ante-returns-grid.rowData",
                "po-ex-ante-returns-store.data",
                "po-ex-ante-vol-store.data",
            ],
            prepare=lambda: (
                warm.set_component_props(page, "po-series-select", {"data": ["SPX_TRIndex"]}),
                warm.set_component_props(page, "po-ex-ante-returns-grid", {"rowData": sample_returns}),
            ),
            action=lambda: page.locator("#po-ex-ante-returns-clear").click(force=True),
            wait_for_ready=lambda: wait_for_js_condition(
                page,
                "() => document.querySelectorAll('#po-ex-ante-returns-grid .ag-center-cols-container [row-index]').length <= 1",
                timeout=10000,
            ),
        )
    )

    results.append(
        measure_scenario(
            page=page,
            tracker=tracker,
            scenario_name="portopt_tau_sync",
            targeted_outputs=["po-bl-tau-store.data"],
            prepare=lambda: warm.set_component_value(page, "po-bl-tau-input", 0.05),
            action=lambda: warm.set_component_value(page, "po-bl-tau-input", 0.12),
            wait_for_ready=lambda: wait_persisted_store_value(page, "po-bl-tau-store", 0.12, timeout=10000),
        )
    )

    results.append(
        measure_scenario(
            page=page,
            tracker=tracker,
            scenario_name="portopt_opt_step_unit_change",
            targeted_outputs=["po-opt-step-input.value"],
            prepare=lambda: (
                warm.set_component_value(page, "po-periodicity-select", "monthly"),
                warm.set_component_props(page, "po-opt-step-store", {"data": 1}),
                warm.set_component_value(page, "po-opt-step-input", 6),
                warm.set_component_value(page, "po-opt-step-unit-select", "months"),
            ),
            action=lambda: warm.set_component_value(page, "po-opt-step-unit-select", "periods"),
            wait_for_ready=None,
        )
    )

    results.append(
        measure_scenario(
            page=page,
            tracker=tracker,
            scenario_name="portopt_date_range_store",
            targeted_outputs=["po-date-range-store.data"],
            prepare=lambda: None,
            action=lambda: (
                warm.set_component_props(page, "po-start-date-picker", {"value": "2024-01-02"}),
                warm.set_component_props(page, "po-end-date-picker", {"value": "2024-12-31"}),
            ),
            wait_for_ready=lambda: wait_persisted_store_value(
                page,
                "po-date-range-store",
                {"start": "2024-01-02", "end": "2024-12-31"},
                timeout=10000,
            ),
        )
    )

    results.append(
        measure_scenario(
            page=page,
            tracker=tracker,
            scenario_name="portopt_rolling_return_type",
            targeted_outputs=[
                "po-rolling-return-type-select.disabled",
                "po-rolling-return-type-select.style",
            ],
            prepare=lambda: warm.set_component_value(page, "po-rolling-metric-select", "total_return"),
            action=lambda: warm.set_component_value(page, "po-rolling-metric-select", "volatility"),
            wait_for_ready=lambda: wait_for_input_disabled(page, "#po-rolling-return-type-select", True),
        )
    )

    results.append(
        measure_scenario(
            page=page,
            tracker=tracker,
            scenario_name="portopt_statistics_portfolio_switch_visible",
            targeted_outputs=["po-statistics-grid-content.children"],
            prepare=lambda: _seed_portopt_result_state(
                page,
                resolved_db_series,
                "statistics",
                "Harness Portfolio 1",
            ),
            action=lambda: warm.set_component_value(page, "po-weight-portfolio-select", "Harness Portfolio 2"),
            wait_for_ready=lambda: wait_content_ready(page, "#po-statistics-grid-content", timeout=10000),
            scenario_class="visible_result_tab",
        )
    )

    results.append(
        measure_scenario(
            page=page,
            tracker=tracker,
            scenario_name="portopt_weight_portfolio_switch_visible",
            targeted_outputs=["po-weight-chart-graph.figure", "po-weight-chart-content.children"],
            prepare=lambda: _seed_portopt_result_state(
                page,
                resolved_db_series,
                "weight",
                "Harness Portfolio 1",
            ),
            action=lambda: warm.set_component_value(page, "po-weight-portfolio-select", "Harness Portfolio 2"),
            wait_for_ready=lambda: wait_for_plotly_graph_ready(
                page,
                "#po-weight-chart-graph",
                "#po-weight-chart-empty",
                "#po-weight-chart-content",
                timeout=10000,
            ),
            scenario_class="visible_result_tab",
        )
    )

    results.append(
        measure_scenario(
            page=page,
            tracker=tracker,
            scenario_name="portopt_weight_table_portfolio_switch_visible",
            targeted_outputs=["po-weight-grid-content.children"],
            prepare=lambda: (
                _seed_portopt_result_state(page, resolved_db_series, "weight", "Harness Portfolio 1"),
                warm.set_component_value(page, "po-weight-chart-switch", "table"),
            ),
            action=lambda: warm.set_component_value(page, "po-weight-portfolio-select", "Harness Portfolio 2"),
            wait_for_ready=lambda: wait_content_ready(page, "#po-weight-grid-content", timeout=10000),
            scenario_class="visible_result_tab",
        )
    )

    results.append(
        measure_scenario(
            page=page,
            tracker=tracker,
            scenario_name="portopt_growth_visible",
            targeted_outputs=["po-growth-chart-container.children"],
            prepare=lambda: _seed_portopt_result_state(page, resolved_db_series, "weight"),
            action=lambda: warm.set_component_value(page, "po-vis-tabs", "growth"),
            wait_for_ready=lambda: wait_content_ready(page, "#po-growth-chart-container", timeout=10000),
            scenario_class="visible_result_tab",
        )
    )

    results.append(
        measure_scenario(
            page=page,
            tracker=tracker,
            scenario_name="portopt_growth_portfolio_switch_visible",
            targeted_outputs=["po-growth-chart-container.children"],
            prepare=lambda: _seed_portopt_result_state(
                page,
                resolved_db_series,
                "growth",
                "Harness Portfolio 1",
            ),
            action=lambda: warm.set_component_value(page, "po-weight-portfolio-select", "Harness Portfolio 2"),
            wait_for_ready=lambda: wait_content_ready(page, "#po-growth-chart-container", timeout=10000),
            scenario_class="visible_result_tab",
        )
    )

    results.append(
        measure_scenario(
            page=page,
            tracker=tracker,
            scenario_name="portopt_turnover_portfolio_switch_visible",
            targeted_outputs=["po-turnover-chart-container.children"],
            prepare=lambda: _seed_portopt_result_state(
                page,
                resolved_db_series,
                "turnover",
                "Harness Portfolio 1",
            ),
            action=lambda: warm.set_component_value(page, "po-weight-portfolio-select", "Harness Portfolio 2"),
            wait_for_ready=lambda: wait_content_ready(page, "#po-turnover-chart-container", timeout=10000),
            scenario_class="visible_result_tab",
        )
    )

    results.append(
        measure_scenario(
            page=page,
            tracker=tracker,
            scenario_name="portopt_turnover_table_portfolio_switch_visible",
            targeted_outputs=["po-turnover-grid-container.children"],
            prepare=lambda: (
                _seed_portopt_result_state(page, resolved_db_series, "turnover", "Harness Portfolio 1"),
                warm.set_component_value(page, "po-turnover-chart-switch", "table"),
            ),
            action=lambda: warm.set_component_value(page, "po-weight-portfolio-select", "Harness Portfolio 2"),
            wait_for_ready=lambda: wait_content_ready(page, "#po-turnover-grid-container", timeout=10000),
            scenario_class="visible_result_tab",
        )
    )

    results.append(
        measure_scenario(
            page=page,
            tracker=tracker,
            scenario_name="portopt_frontier_portfolio_switch_visible",
            targeted_outputs=["po-frontier-chart-graph.figure", "po-frontier-chart-container.children"],
            prepare=lambda: _seed_portopt_result_state(
                page,
                resolved_db_series,
                "frontier",
                "Harness Portfolio 1",
            ),
            action=lambda: warm.set_component_value(page, "po-weight-portfolio-select", "Harness Portfolio 2"),
            wait_for_ready=lambda: (
                wait_for_dash_update_requests(page, tracker, minimum_count=1, timeout=10000),
                wait_for_plotly_graph_ready(
                    page,
                    "#po-frontier-chart-graph",
                    "#po-frontier-chart-empty",
                    "#po-frontier-chart-container",
                    timeout=10000,
                ),
            ),
            scenario_class="visible_result_tab",
        )
    )

    results.append(
        measure_scenario(
            page=page,
            tracker=tracker,
            scenario_name="portopt_frontier_table_portfolio_switch_visible",
            targeted_outputs=["po-frontier-grid-container.children"],
            prepare=lambda: (
                _seed_portopt_result_state(page, resolved_db_series, "frontier", "Harness Portfolio 1"),
                warm.set_component_value(page, "po-frontier-chart-switch", "table"),
            ),
            action=lambda: warm.set_component_value(page, "po-weight-portfolio-select", "Harness Portfolio 2"),
            wait_for_ready=lambda: (
                wait_for_dash_update_requests(page, tracker, minimum_count=1, timeout=10000),
                wait_content_ready(page, "#po-frontier-grid-container", timeout=10000),
            ),
            scenario_class="visible_result_tab",
        )
    )

    results.append(
        measure_scenario(
            page=page,
            tracker=tracker,
            scenario_name="portopt_risk_portfolio_switch_visible",
            targeted_outputs=["po-risk-chart-container.children"],
            prepare=lambda: _seed_portopt_result_state(
                page,
                resolved_db_series,
                "risk",
                "Harness Portfolio 1",
            ),
            action=lambda: warm.set_component_value(page, "po-weight-portfolio-select", "Harness Portfolio 2"),
            wait_for_ready=lambda: wait_content_ready(page, "#po-risk-chart-container", timeout=10000),
            scenario_class="visible_result_tab",
        )
    )

    results.append(
        measure_scenario(
            page=page,
            tracker=tracker,
            scenario_name="portopt_risk_table_portfolio_switch_visible",
            targeted_outputs=["po-risk-grid-container.children"],
            prepare=lambda: (
                _seed_portopt_result_state(page, resolved_db_series, "risk", "Harness Portfolio 1"),
                warm.set_component_value(page, "po-risk-chart-switch", "table"),
            ),
            action=lambda: warm.set_component_value(page, "po-weight-portfolio-select", "Harness Portfolio 2"),
            wait_for_ready=lambda: wait_content_ready(page, "#po-risk-grid-container", timeout=10000),
            scenario_class="visible_result_tab",
        )
    )

    results.append(
        measure_scenario(
            page=page,
            tracker=tracker,
            scenario_name="portopt_attribution_portfolio_switch_visible",
            targeted_outputs=["po-attribution-chart-container.children"],
            prepare=lambda: _seed_portopt_result_state(
                page,
                resolved_db_series,
                "attribution",
                "Harness Portfolio 1",
            ),
            action=lambda: warm.set_component_value(page, "po-weight-portfolio-select", "Harness Portfolio 2"),
            wait_for_ready=lambda: wait_content_ready(page, "#po-attribution-chart-container", timeout=10000),
            scenario_class="visible_result_tab",
        )
    )

    results.append(
        measure_scenario(
            page=page,
            tracker=tracker,
            scenario_name="portopt_attribution_table_portfolio_switch_visible",
            targeted_outputs=["po-attribution-grid-container.children"],
            prepare=lambda: (
                _seed_portopt_result_state(page, resolved_db_series, "attribution", "Harness Portfolio 1"),
                warm.set_component_value(page, "po-attribution-chart-switch", "table"),
            ),
            action=lambda: warm.set_component_value(page, "po-weight-portfolio-select", "Harness Portfolio 2"),
            wait_for_ready=lambda: wait_content_ready(page, "#po-attribution-grid-container", timeout=10000),
            scenario_class="visible_result_tab",
        )
    )

    results.append(
        measure_scenario(
            page=page,
            tracker=tracker,
            scenario_name="portopt_statistics_visible",
            targeted_outputs=["po-statistics-grid-content.children"],
            prepare=lambda: _seed_portopt_result_state(page, resolved_db_series, "weight"),
            action=lambda: warm.set_component_value(page, "po-vis-tabs", "statistics"),
            wait_for_ready=lambda: wait_content_ready(page, "#po-statistics-grid-content", timeout=10000),
            scenario_class="visible_result_tab",
        )
    )

    results.append(
        measure_scenario(
            page=page,
            tracker=tracker,
            scenario_name="portopt_rolling_metric_visible",
            targeted_outputs=["po-rolling-content.children"],
            prepare=lambda: (
                _seed_portopt_result_state(page, resolved_db_series, "rolling"),
                warm.set_component_value(page, "po-rolling-window-select", "3m"),
                warm.set_component_value(page, "po-rolling-return-type-select", "annualized"),
                warm.set_component_value(page, "po-rolling-metric-select", "total_return"),
            ),
            action=lambda: warm.set_component_value(page, "po-rolling-metric-select", "volatility"),
            wait_for_ready=lambda: wait_content_ready(page, "#po-rolling-content", timeout=10000),
            scenario_class="visible_result_tab",
        )
    )

    results.append(
        measure_scenario(
            page=page,
            tracker=tracker,
            scenario_name="portopt_rolling_window_visible",
            targeted_outputs=["po-rolling-content.children"],
            prepare=lambda: (
                _seed_portopt_result_state(page, resolved_db_series, "rolling"),
                warm.set_component_value(page, "po-rolling-window-select", "3m"),
                warm.set_component_value(page, "po-rolling-metric-select", "total_return"),
            ),
            action=lambda: warm.set_component_value(page, "po-rolling-window-select", "6m"),
            wait_for_ready=lambda: wait_content_ready(page, "#po-rolling-content", timeout=10000),
            scenario_class="visible_result_tab",
        )
    )

    results.append(
        measure_scenario(
            page=page,
            tracker=tracker,
            scenario_name="portopt_rolling_return_type_visible",
            targeted_outputs=["po-rolling-content.children"],
            prepare=lambda: (
                _seed_portopt_result_state(page, resolved_db_series, "rolling"),
                warm.set_component_value(page, "po-rolling-metric-select", "total_return"),
                warm.set_component_value(page, "po-rolling-return-type-select", "annualized"),
            ),
            action=lambda: warm.set_component_value(page, "po-rolling-return-type-select", "cumulative"),
            wait_for_ready=lambda: wait_content_ready(page, "#po-rolling-content", timeout=10000),
            scenario_class="visible_result_tab",
        )
    )

    results.append(
        measure_scenario(
            page=page,
            tracker=tracker,
            scenario_name="portopt_returns_visible",
            targeted_outputs=["po-returns-grid-content.children"],
            prepare=lambda: _seed_portopt_result_state(page, resolved_db_series, "weight"),
            action=lambda: warm.set_component_value(page, "po-vis-tabs", "returns"),
            wait_for_ready=lambda: wait_content_ready(page, "#po-returns-grid-content", timeout=10000),
            scenario_class="visible_result_tab",
        )
    )

    results.append(
        measure_scenario(
            page=page,
            tracker=tracker,
            scenario_name="portopt_returns_portfolio_switch_visible",
            targeted_outputs=["po-returns-grid-content.children"],
            prepare=lambda: _seed_portopt_result_state(page, resolved_db_series, "returns", "Harness Portfolio 1"),
            action=lambda: warm.set_component_value(page, "po-weight-portfolio-select", "Harness Portfolio 2"),
            wait_for_ready=lambda: wait_content_ready(page, "#po-returns-grid-content", timeout=10000),
            scenario_class="visible_result_tab",
        )
    )

    results.append(
        measure_scenario(
            page=page,
            tracker=tracker,
            scenario_name="portopt_calendar_visible",
            targeted_outputs=["po-calendar-content.children"],
            prepare=lambda: _seed_portopt_result_state(page, resolved_db_series, "weight"),
            action=lambda: warm.set_component_value(page, "po-vis-tabs", "calendar"),
            wait_for_ready=lambda: wait_content_ready(page, "#po-calendar-content", timeout=10000),
            scenario_class="visible_result_tab",
        )
    )

    results.append(
        measure_scenario(
            page=page,
            tracker=tracker,
            scenario_name="portopt_calendar_portfolio_switch_visible",
            targeted_outputs=["po-calendar-content.children"],
            prepare=lambda: _seed_portopt_result_state(page, resolved_db_series, "calendar", "Harness Portfolio 1"),
            action=lambda: warm.set_component_value(page, "po-weight-portfolio-select", "Harness Portfolio 2"),
            wait_for_ready=lambda: wait_content_ready(page, "#po-calendar-content", timeout=10000),
            scenario_class="visible_result_tab",
        )
    )

    results.append(
        measure_scenario(
            page=page,
            tracker=tracker,
            scenario_name="portopt_calendar_monthly_portfolio_switch_visible",
            targeted_outputs=["po-calendar-content.children"],
            prepare=lambda: (
                _seed_portopt_result_state(page, resolved_db_series, "calendar", "Harness Portfolio 1"),
                warm.set_component_value(page, "po-calendar-view-select", "monthly"),
            ),
            action=lambda: warm.set_component_value(page, "po-weight-portfolio-select", "Harness Portfolio 2"),
            wait_for_ready=lambda: wait_content_ready(page, "#po-calendar-content", timeout=10000),
            scenario_class="visible_result_tab",
        )
    )

    results.append(
        measure_scenario(
            page=page,
            tracker=tracker,
            scenario_name="portopt_drawdown_visible",
            targeted_outputs=["po-drawdown-content.children"],
            prepare=lambda: _seed_portopt_result_state(page, resolved_db_series, "weight"),
            action=lambda: warm.set_component_value(page, "po-vis-tabs", "drawdown"),
            wait_for_ready=lambda: wait_content_ready(page, "#po-drawdown-content", timeout=10000),
            scenario_class="visible_result_tab",
        )
    )

    results.append(
        measure_scenario(
            page=page,
            tracker=tracker,
            scenario_name="portopt_drawdown_portfolio_switch_visible",
            targeted_outputs=["po-drawdown-content.children"],
            prepare=lambda: _seed_portopt_result_state(page, resolved_db_series, "drawdown", "Harness Portfolio 1"),
            action=lambda: warm.set_component_value(page, "po-weight-portfolio-select", "Harness Portfolio 2"),
            wait_for_ready=lambda: wait_content_ready(page, "#po-drawdown-content", timeout=10000),
            scenario_class="visible_result_tab",
        )
    )

    return results


def seed_portopt_page(page, base_url: str, db_series: list[str]) -> None:
    opt_series = warm.resolve_portopt_series(db_series)
    raw_data_payload, raw_meta_payload = build_synthetic_raw_dataset(opt_series)
    page.goto(base_url + "/portopt", wait_until="domcontentloaded")
    wait_dash_hydrated(page, timeout=30000)
    warm.try_set_component_props(page, "po-page-visited-store", {"data": True})
    warm.try_set_component_props(page, "dashmat-raw-data-store", {"data": raw_data_payload})
    warm.try_set_component_props(page, "dashmat-raw-data-meta-store", {"data": raw_meta_payload})
    warm.try_set_component_props(page, "po-series-select", {"data": opt_series})
    warm.try_set_component_props(page, "po-series-select-value-store", {"data": opt_series})
    warm.try_set_component_props(page, "po-series-order-store", {"data": opt_series})
    warm.try_set_component_props(page, "po-cmabench-defaults-store", {"data": {}})
    wait_visible(page, "#po-main-container", timeout=30000)
    wait_ready(page, "#po-opt-model-select", timeout=30000)
    wait_hidden_or_absent(page, "#po-ui-blocker-overlay", timeout=30000)


def seed_analytics_page(page, base_url: str, db_series: list[str]) -> None:
    warm.warm_analytics_db(page, base_url, db_series)
    warm.wait_for_analytics_state_ready(page, timeout=30000)
    warm.ensure_analytics_selection(page, db_series, active_tab="statistics", timeout=30000)
    wait_visible(page, "#at-main-app-container", timeout=30000)
    wait_ready(page, "#at-periodicity-select", timeout=30000)


def run_analytics_scenarios(page, tracker: warm.DashUpdateRequestTracker, db_series: list[str]) -> list[dict[str, object]]:
    resolved_db_series = list(db_series or DEFAULT_DB_SERIES)
    if not resolved_db_series:
        resolved_db_series = list(DEFAULT_DB_SERIES)
    alternate_series = resolved_db_series[1] if len(resolved_db_series) > 1 else resolved_db_series[0]
    results: list[dict[str, object]] = []

    results.append(
        measure_scenario(
            page=page,
            tracker=tracker,
            scenario_name="analytics_calendar_visible",
            targeted_outputs=["at-calendar-grid.columnDefs", "at-calendar-grid.rowData"],
            prepare=lambda: warm.ensure_analytics_selection(page, resolved_db_series, active_tab="statistics", timeout=30000),
            action=lambda: warm.set_component_value(page, "at-main-tabs", "calendar"),
            wait_for_ready=lambda: wait_analytics_tab_ready(page, "calendar", timeout=10000),
            scenario_class="visible_result_tab",
        )
    )

    results.append(
        measure_scenario(
            page=page,
            tracker=tracker,
            scenario_name="analytics_calendar_monthly_visible",
            targeted_outputs=["at-calendar-grid.columnDefs", "at-calendar-grid.rowData"],
            prepare=lambda: (
                warm.ensure_analytics_selection(page, resolved_db_series, active_tab="calendar", timeout=30000),
                warm.set_component_value(page, "at-monthly-view-checkbox", "annual"),
            ),
            action=lambda: warm.set_component_value(page, "at-monthly-view-checkbox", "monthly"),
            wait_for_ready=lambda: wait_analytics_tab_ready(page, "calendar", timeout=10000),
            scenario_class="visible_result_tab",
        )
    )

    results.append(
        measure_scenario(
            page=page,
            tracker=tracker,
            scenario_name="analytics_calendar_series_switch_visible",
            targeted_outputs=["at-calendar-grid.columnDefs", "at-calendar-grid.rowData"],
            prepare=lambda: (
                warm.ensure_analytics_selection(page, resolved_db_series, active_tab="calendar", timeout=30000),
                warm.set_component_value(page, "at-monthly-view-checkbox", "monthly"),
                warm.set_component_value(page, "at-monthly-series-select", resolved_db_series[0]),
            ),
            action=lambda: warm.set_component_value(page, "at-monthly-series-select", alternate_series),
            wait_for_ready=lambda: (
                wait_for_input_value(page, "#at-monthly-series-select", alternate_series),
                wait_analytics_tab_ready(page, "calendar", timeout=10000),
            ),
            scenario_class="visible_result_tab",
        )
    )

    results.append(
        measure_scenario(
            page=page,
            tracker=tracker,
            scenario_name="analytics_rolling_metric_visible",
            targeted_outputs=["at-rolling-chart-wrapper.children", "at-rolling-grid.columnDefs", "at-rolling-grid.rowData"],
            prepare=lambda: (
                warm.ensure_analytics_selection(page, resolved_db_series, active_tab="statistics", timeout=30000),
                warm.set_component_value(page, "at-main-tabs", "rolling"),
                warm.set_component_value(page, "at-rolling-chart-switch", "chart"),
                warm.set_component_value(page, "at-rolling-metric-select", "total_return"),
                warm.set_component_value(page, "at-rolling-return-type-select", "annualized"),
                wait_analytics_rolling_ready(page, timeout=10000),
            ),
            action=lambda: warm.set_component_value(page, "at-rolling-metric-select", "volatility"),
            wait_for_ready=lambda: wait_analytics_rolling_ready(page, timeout=10000),
            scenario_class="visible_result_tab",
        )
    )

    results.append(
        measure_scenario(
            page=page,
            tracker=tracker,
            scenario_name="analytics_rolling_window_visible",
            targeted_outputs=["at-rolling-chart-wrapper.children", "at-rolling-grid.columnDefs", "at-rolling-grid.rowData"],
            prepare=lambda: (
                warm.ensure_analytics_selection(page, resolved_db_series, active_tab="statistics", timeout=30000),
                warm.set_component_value(page, "at-main-tabs", "rolling"),
                warm.set_component_value(page, "at-rolling-chart-switch", "chart"),
                warm.set_component_value(page, "at-rolling-window-select", "3m"),
                warm.set_component_value(page, "at-rolling-metric-select", "total_return"),
                wait_analytics_rolling_ready(page, timeout=10000),
            ),
            action=lambda: warm.set_component_value(page, "at-rolling-window-select", "6m"),
            wait_for_ready=lambda: wait_analytics_rolling_ready(page, timeout=10000),
            scenario_class="visible_result_tab",
        )
    )

    return results


def run_regression_scenarios(page, tracker: warm.DashUpdateRequestTracker, db_series: list[str]) -> list[dict[str, object]]:
    sample_rows = [{"row_id": "alpha", "Series": "SPX_TRIndex"}]
    populated_results = {"OLS": {"status": "ok"}}
    results: list[dict[str, object]] = []

    results.append(
        measure_scenario(
            page=page,
            tracker=tracker,
            scenario_name="regression_idle_baseline",
            targeted_outputs=[],
            prepare=None,
            action=lambda: page.wait_for_timeout(150),
            wait_for_ready=None,
            scenario_class="ui_only",
        )
    )

    results.append(
        measure_scenario(
            page=page,
            tracker=tracker,
            scenario_name="regression_model_select_sync",
            targeted_outputs=[
                "reg-arima-garch-panel.style",
                "reg-alpha-container.style",
                "reg-l1-ratio-container.style",
                "reg-force-zero-intercept-switch.disabled",
                "reg-force-zero-intercept-switch.checked",
                "reg-regression-name-input.value",
            ],
            prepare=lambda: warm.set_component_value(page, "reg-model-select", "ols"),
            action=lambda: warm.set_component_value(page, "reg-model-select", "style_analysis"),
            wait_for_ready=lambda: (
                wait_for_input_value(page, "#reg-regression-name-input", "Style Analysis"),
                wait_for_input_disabled(page, "#reg-force-zero-intercept-switch", True),
                wait_for_style_display(page, "#reg-alpha-container", "none"),
            ),
            scenario_class="ui_only",
        )
    )

    results.append(
        measure_scenario(
            page=page,
            tracker=tracker,
            scenario_name="regression_window_controls",
            targeted_outputs=[
                "reg-window-size-input.disabled",
                "reg-opt-step-input.disabled",
                "reg-opt-step-unit-select.disabled",
            ],
            prepare=lambda: warm.set_component_value(page, "reg-window-type-select", "rolling"),
            action=lambda: warm.set_component_value(page, "reg-window-type-select", "full"),
            wait_for_ready=lambda: (
                wait_for_input_disabled(page, "#reg-window-size-input", True),
                wait_for_input_disabled(page, "#reg-opt-step-input", True),
                wait_for_input_disabled(page, "#reg-opt-step-unit-select", True),
            ),
            scenario_class="ui_only",
        )
    )

    results.append(
        measure_scenario(
            page=page,
            tracker=tracker,
            scenario_name="regression_rolling_return_type",
            targeted_outputs=[
                "reg-rolling-return-type-select.disabled",
                "reg-rolling-return-type-select.style",
            ],
            prepare=lambda: warm.set_component_value(page, "reg-rolling-metric-select", "total_return"),
            action=lambda: warm.set_component_value(page, "reg-rolling-metric-select", "volatility"),
            wait_for_ready=lambda: wait_for_input_disabled(page, "#reg-rolling-return-type-select", True),
            scenario_class="ui_only",
        )
    )

    results.append(
        measure_scenario(
            page=page,
            tracker=tracker,
            scenario_name="regression_raw_db_delete_row",
            targeted_outputs=[
                "reg-raw-db-add-rows-store.data",
                "reg-raw-db-add-grid.rowData",
                "reg-raw-db-add-error-alert.children",
                "reg-raw-db-add-error-alert.hide",
            ],
            prepare=lambda: _seed_regression_raw_db_modal(page, sample_rows, sample_rows),
            action=lambda: warm.set_component_props(page, "reg-raw-db-delete-row-btn", {"n_clicks": 1}),
            wait_for_ready=None,
            scenario_class="ui_only",
        )
    )

    results.append(
        measure_scenario(
            page=page,
            tracker=tracker,
            scenario_name="regression_raw_db_clear_rows",
            targeted_outputs=[
                "reg-raw-db-add-rows-store.data",
                "reg-raw-db-add-grid.rowData",
                "reg-raw-db-add-error-alert.children",
                "reg-raw-db-add-error-alert.hide",
            ],
            prepare=lambda: _seed_regression_raw_db_modal(page, sample_rows),
            action=lambda: warm.set_component_props(page, "reg-raw-db-clear-rows-btn", {"n_clicks": 1}),
            wait_for_ready=None,
            scenario_class="ui_only",
        )
    )

    results.append(
        measure_scenario(
            page=page,
            tracker=tracker,
            scenario_name="regression_sheet_select_disabled",
            targeted_outputs=["reg-sheet-select-ok-button.disabled"],
            prepare=lambda: (
                warm.set_component_props(page, "reg-sheet-select-modal", {"opened": True}),
                warm.set_component_props(
                    page,
                    "reg-sheet-select-dropdown",
                    {"data": [{"value": "Sheet1", "label": "Sheet1"}], "value": []},
                ),
                warm.set_component_props(page, "reg-sheet-select-dropdown", {"value": []}),
                wait_for_input_disabled(page, "#reg-sheet-select-ok-button", True),
            ),
            action=lambda: warm.set_component_props(page, "reg-sheet-select-dropdown", {"value": ["Sheet1"]}),
            wait_for_ready=lambda: wait_for_input_disabled(page, "#reg-sheet-select-ok-button", False),
            scenario_class="ui_only",
        )
    )

    results.append(
        measure_scenario(
            page=page,
            tracker=tracker,
            scenario_name="regression_statistics_use_risk_free_visible",
            targeted_outputs=["reg-statistics-content.children"],
            prepare=lambda: (
                _seed_regression_result_state(page, db_series, "statistics", "Harness Result 1"),
                warm.set_component_value(page, "reg-use-risk-free-switch", "tbill"),
            ),
            action=lambda: warm.set_component_value(page, "reg-use-risk-free-switch", "zero"),
            wait_for_ready=lambda: wait_content_ready(page, "#reg-statistics-content", timeout=10000),
            scenario_class="visible_result_tab",
        )
    )

    results.append(
        measure_scenario(
            page=page,
            tracker=tracker,
            scenario_name="regression_rolling_metric_visible",
            targeted_outputs=["reg-rolling-returns-content.children"],
            prepare=lambda: (
                _seed_regression_result_state(page, db_series, "rolling_returns", "Harness Result 1"),
                warm.set_component_value(page, "reg-rolling-metric-select", "total_return"),
                warm.set_component_value(page, "reg-rolling-return-type-select", "annualized"),
            ),
            action=lambda: warm.set_component_value(page, "reg-rolling-metric-select", "volatility"),
            wait_for_ready=lambda: wait_content_ready(page, "#reg-rolling-returns-content", timeout=10000),
            scenario_class="visible_result_tab",
        )
    )

    results.append(
        measure_scenario(
            page=page,
            tracker=tracker,
            scenario_name="regression_rolling_window_visible",
            targeted_outputs=["reg-rolling-returns-content.children"],
            prepare=lambda: (
                _seed_regression_result_state(page, db_series, "rolling_returns", "Harness Result 1"),
                warm.set_component_value(page, "reg-rolling-window-select", "1y"),
            ),
            action=lambda: warm.set_component_value(page, "reg-rolling-window-select", "3y"),
            wait_for_ready=lambda: wait_content_ready(page, "#reg-rolling-returns-content", timeout=10000),
            scenario_class="visible_result_tab",
        )
    )

    results.append(
        measure_scenario(
            page=page,
            tracker=tracker,
            scenario_name="regression_rolling_return_type_visible",
            targeted_outputs=["reg-rolling-returns-content.children"],
            prepare=lambda: (
                _seed_regression_result_state(page, db_series, "rolling_returns", "Harness Result 1"),
                warm.set_component_value(page, "reg-rolling-metric-select", "total_return"),
                warm.set_component_value(page, "reg-rolling-return-type-select", "annualized"),
            ),
            action=lambda: warm.set_component_value(page, "reg-rolling-return-type-select", "cumulative"),
            wait_for_ready=lambda: wait_content_ready(page, "#reg-rolling-returns-content", timeout=10000),
            scenario_class="visible_result_tab",
        )
    )

    results.append(
        measure_scenario(
            page=page,
            tracker=tracker,
            scenario_name="regression_calendar_visible",
            targeted_outputs=["reg-calendar-content.children"],
            prepare=lambda: _seed_regression_result_state(page, db_series, "statistics", "Harness Result 1"),
            action=lambda: warm.set_component_value(page, "reg-tabs", "calendar"),
            wait_for_ready=lambda: wait_content_ready(page, "#reg-calendar-content", timeout=10000),
            scenario_class="visible_result_tab",
        )
    )

    results.append(
        measure_scenario(
            page=page,
            tracker=tracker,
            scenario_name="regression_calendar_monthly_visible",
            targeted_outputs=["reg-calendar-content.children"],
            prepare=lambda: (
                _seed_regression_result_state(page, db_series, "calendar", "Harness Result 1"),
                warm.set_component_value(page, "reg-calendar-view-select", "annual"),
            ),
            action=lambda: warm.set_component_value(page, "reg-calendar-view-select", "monthly"),
            wait_for_ready=lambda: wait_content_ready(page, "#reg-calendar-content", timeout=10000),
            scenario_class="visible_result_tab",
        )
    )

    results.append(
        measure_scenario(
            page=page,
            tracker=tracker,
            scenario_name="regression_result_select_visible_calendar",
            targeted_outputs=["reg-calendar-content.children"],
            prepare=lambda: _seed_regression_result_state(page, db_series, "calendar", "Harness Result 1"),
            action=lambda: warm.set_component_value(page, "reg-result-select", "Harness Result 2"),
            wait_for_ready=lambda: (
                wait_for_input_value(page, "#reg-result-select", "Harness Result 2"),
                wait_content_ready(page, "#reg-calendar-content", timeout=10000),
            ),
            scenario_class="visible_result_tab",
        )
    )

    results.append(
        measure_scenario(
            page=page,
            tracker=tracker,
            scenario_name="regression_anova_visible",
            targeted_outputs=["reg-anova-content.children"],
            prepare=lambda: _seed_regression_result_state(page, db_series, "statistics", "Harness Result 1"),
            action=lambda: warm.set_component_value(page, "reg-tabs", "anova"),
            wait_for_ready=lambda: wait_content_ready(page, "#reg-anova-content", timeout=10000),
            scenario_class="visible_result_tab",
        )
    )

    results.append(
        measure_scenario(
            page=page,
            tracker=tracker,
            scenario_name="regression_result_select_visible_anova",
            targeted_outputs=["reg-anova-content.children"],
            prepare=lambda: _seed_regression_result_state(page, db_series, "anova", "Harness Result 1"),
            action=lambda: warm.set_component_value(page, "reg-result-select", "Harness Result 2"),
            wait_for_ready=lambda: (
                wait_for_input_value(page, "#reg-result-select", "Harness Result 2"),
                wait_content_ready(page, "#reg-anova-content", timeout=10000),
            ),
            scenario_class="visible_result_tab",
        )
    )

    results.append(
        measure_scenario(
            page=page,
            tracker=tracker,
            scenario_name="regression_rolling_visible",
            targeted_outputs=["reg-rolling-content.children"],
            prepare=lambda: _seed_regression_result_state(page, db_series, "statistics", "Harness Result 1"),
            action=lambda: warm.set_component_value(page, "reg-tabs", "rolling"),
            wait_for_ready=lambda: wait_content_ready(page, "#reg-rolling-content", timeout=10000),
            scenario_class="visible_result_tab",
        )
    )

    results.append(
        measure_scenario(
            page=page,
            tracker=tracker,
            scenario_name="regression_result_select_visible_rolling",
            targeted_outputs=["reg-rolling-content.children"],
            prepare=lambda: _seed_regression_result_state(page, db_series, "rolling", "Harness Result 1"),
            action=lambda: warm.set_component_value(page, "reg-result-select", "Harness Result 2"),
            wait_for_ready=lambda: (
                wait_for_input_value(page, "#reg-result-select", "Harness Result 2"),
                wait_content_ready(page, "#reg-rolling-content", timeout=10000),
            ),
            scenario_class="visible_result_tab",
        )
    )

    results.append(
        measure_scenario(
            page=page,
            tracker=tracker,
            scenario_name="regression_result_select_visible_statistics",
            targeted_outputs=["reg-statistics-content.children"],
            prepare=lambda: _seed_regression_result_state(page, db_series, "statistics", "Harness Result 1"),
            action=lambda: warm.set_component_value(page, "reg-result-select", "Harness Result 2"),
            wait_for_ready=lambda: (
                wait_for_input_value(page, "#reg-result-select", "Harness Result 2"),
                wait_content_ready(page, "#reg-statistics-content", timeout=10000),
            ),
            scenario_class="visible_result_tab",
        )
    )

    results.append(
        measure_scenario(
            page=page,
            tracker=tracker,
            scenario_name="regression_weights_visible",
            targeted_outputs=["reg-weights-content.children"],
            prepare=lambda: _seed_regression_result_state(page, db_series, "statistics", "Harness Result 1"),
            action=lambda: warm.set_component_value(page, "reg-tabs", "weights"),
            wait_for_ready=lambda: wait_content_ready(page, "#reg-weights-content", timeout=10000),
            scenario_class="visible_result_tab",
        )
    )

    results.append(
        measure_scenario(
            page=page,
            tracker=tracker,
            scenario_name="regression_result_select_visible_weights",
            targeted_outputs=["reg-weights-content.children"],
            prepare=lambda: _seed_regression_result_state(page, db_series, "weights", "Harness Result 1"),
            action=lambda: warm.set_component_value(page, "reg-result-select", "Harness Result 2"),
            wait_for_ready=lambda: (
                wait_for_input_value(page, "#reg-result-select", "Harness Result 2"),
                wait_content_ready(page, "#reg-weights-content", timeout=10000),
            ),
            scenario_class="visible_result_tab",
        )
    )

    results.append(
        measure_scenario(
            page=page,
            tracker=tracker,
            scenario_name="regression_result_select_visible_scatter",
            targeted_outputs=["reg-scatter-content.children"],
            prepare=lambda: _seed_regression_result_state(page, db_series, "scatter", "Harness Result 1"),
            action=lambda: warm.set_component_value(page, "reg-result-select", "Harness Result 2"),
            wait_for_ready=lambda: (
                wait_for_input_value(page, "#reg-result-select", "Harness Result 2"),
                wait_content_ready(page, "#reg-scatter-content", timeout=10000),
            ),
            scenario_class="visible_result_tab",
        )
    )

    results.append(
        measure_scenario(
            page=page,
            tracker=tracker,
            scenario_name="regression_result_select_visible_rolling_returns",
            targeted_outputs=["reg-rolling-returns-content.children"],
            prepare=lambda: _seed_regression_result_state(page, db_series, "rolling_returns", "Harness Result 1"),
            action=lambda: warm.set_component_value(page, "reg-result-select", "Harness Result 2"),
            wait_for_ready=lambda: (
                wait_for_input_value(page, "#reg-result-select", "Harness Result 2"),
                wait_content_ready(page, "#reg-rolling-returns-content", timeout=10000),
            ),
            scenario_class="visible_result_tab",
        )
    )

    results.append(
        measure_scenario(
            page=page,
            tracker=tracker,
            scenario_name="regression_file_menu_actions",
            targeted_outputs=[
                "reg-menu-save-session.disabled",
                "reg-menu-download-excel.disabled",
            ],
            prepare=lambda: (
                warm.set_component_props(page, "dashmat-raw-data-store", {"data": None}),
                warm.set_component_props(page, "reg-results-store", {"data": {}}),
            ),
            action=lambda: (
                warm.set_component_props(page, "dashmat-raw-data-store", {"data": "raw-json"}),
                warm.set_component_props(page, "reg-results-store", {"data": populated_results}),
            ),
            wait_for_ready=None,
            perf_target=False,
            scenario_class="functional_non_perf",
        )
    )

    return results


def seed_regression_page(page, base_url: str, db_series: list[str]) -> None:
    dep_var, x_series = warm.resolve_regression_series(db_series)
    series_order = [dep_var] + x_series
    raw_data_payload, raw_meta_payload = build_synthetic_raw_dataset(series_order)
    page.goto(base_url + "/regression", wait_until="domcontentloaded")
    wait_dash_hydrated(page, timeout=30000)
    warm.try_set_component_props(page, "reg-page-visited-store", {"data": True})
    warm.try_set_component_props(page, "dashmat-raw-data-store", {"data": raw_data_payload})
    warm.try_set_component_props(page, "dashmat-raw-data-meta-store", {"data": raw_meta_payload})
    warm.try_set_component_props(page, "reg-series-select", {"data": x_series})
    warm.try_set_component_props(page, "reg-series-select-value-store", {"data": x_series})
    warm.try_set_component_props(page, "reg-series-order-store", {"data": series_order})
    warm.try_set_component_props(page, "reg-dependent-var-store", {"data": dep_var})
    wait_visible(page, "#reg-main-container", timeout=30000)
    wait_ready(page, "#reg-model-select", timeout=30000)
    wait_hidden_or_absent(page, "#reg-ui-blocker-overlay", timeout=30000)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DashMat UI callback interaction harness")
    parser.add_argument("--base-url", default="http://127.0.0.1:8050")
    parser.add_argument("--pages", choices=["analytics", "portopt", "regression", "both", "all"], default="both")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--label", default="ui-callback-harness")
    parser.add_argument("--db-series", nargs="*", default=DEFAULT_DB_SERIES)
    parser.add_argument("--network-profile", choices=list(NETWORK_PROFILES.keys()), default="none")
    return parser


def run_page_suite(
    page_name: str,
    base_url: str,
    db_series: list[str],
    headless: bool,
    runs: int,
    network_profile: str,
) -> dict[str, object]:
    run_results: list[dict[str, object]] = []
    applied_network_profile: dict[str, object] | None = None
    for _ in range(max(runs, 1)):
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=headless)
            context = browser.new_context()
            page = context.new_page()
            tracker = warm.DashUpdateRequestTracker(page)
            if page_name == "analytics":
                seed_analytics_page(page, base_url, db_series)
                applied_network_profile = _apply_network_profile(page, network_profile)
                run_results.extend(run_analytics_scenarios(page, tracker, db_series))
            elif page_name == "portopt":
                seed_portopt_page(page, base_url, db_series)
                applied_network_profile = _apply_network_profile(page, network_profile)
                run_results.extend(run_portopt_scenarios(page, tracker, db_series))
            else:
                seed_regression_page(page, base_url, db_series)
                applied_network_profile = _apply_network_profile(page, network_profile)
                run_results.extend(run_regression_scenarios(page, tracker, db_series))
            context.close()
            browser.close()

    grouped: dict[str, list[dict[str, object]]] = {}
    for result in run_results:
        grouped.setdefault(str(result["scenario"]), []).append(result)
    return {
        "page": page_name,
        "runs": max(runs, 1),
        "networkProfile": applied_network_profile or {"name": "none"},
        "scenarios": {name: {"summary": summarize_run_group(values), "runs": values} for name, values in grouped.items()},
    }


def main(argv: list[str] | None = None) -> int:
    global TIMEOUT_SCALE
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.pages == "both":
        pages = ["portopt", "regression"]
    elif args.pages == "all":
        pages = ["analytics", "portopt", "regression"]
    else:
        pages = [args.pages]
    TIMEOUT_SCALE = NETWORK_TIMEOUT_MULTIPLIERS.get(args.network_profile, 1.0)

    out_dir = REPO_ROOT / "output" / "playwright"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{_timestamp_token()}_{args.label.replace(' ', '_')}"
    output_path = out_dir / f"{stem}.json"

    try:
        result = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "label": args.label,
            "baseUrl": args.base_url,
            "networkProfile": args.network_profile,
            "pages": {
                page_name: run_page_suite(
                    page_name,
                    args.base_url,
                    list(args.db_series or DEFAULT_DB_SERIES),
                    args.headless,
                    args.runs,
                    args.network_profile,
                )
                for page_name in pages
            },
        }
    except Exception as exc:
        failure_path = out_dir / f"{stem}_failure.txt"
        failure_path.write_text(traceback.format_exc(), encoding="utf-8")
        print(f"FAIL: {exc}")
        print(f"TRACEBACK_PATH={failure_path}")
        return 1

    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"RESULT_PATH={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
