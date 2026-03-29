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
        timeout=timeout,
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
        timeout=timeout,
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
        timeout=timeout,
    )


def wait_for_js_condition(page, function_body: str, timeout: int = 10000) -> None:
    page.wait_for_function(function_body, timeout=timeout)


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
        timeout=timeout,
    )


def wait_for_quiet_window(
    page,
    tracker: warm.DashUpdateRequestTracker,
    quiet_ms: int = QUIET_WINDOW_MS,
    timeout_ms: int = QUIET_WINDOW_TIMEOUT_MS,
) -> None:
    deadline = time.perf_counter() + (timeout_ms / 1000.0)
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


def _seed_regression_result_state(page, db_series: list[str], tab_value: str, selected_result: str = "Harness Result 1") -> None:
    results = build_seeded_regression_results(db_series)
    options = [{"value": name, "label": name} for name in results]
    warm.set_component_props(page, "reg-results-store", {"data": results})
    warm.set_component_props(page, "reg-result-select", {"data": options, "value": selected_result})
    warm.seed_regression_restore_tab(page, tab_value)


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


def run_portopt_scenarios(page, tracker: warm.DashUpdateRequestTracker) -> list[dict[str, object]]:
    sample_rows = [{"Series": "SPX_TRIndex", "Table": "G", "Fee": "N", "Include Benchmark": False}]
    sample_returns = [{"Asset": "SPX_TRIndex", "Return": 1.5, "Volatility": 3.0}]
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
            wait_for_ready=lambda: warm.wait_for_persisted_store_value(page, "po-bl-tau-store", 0.12, timeout=10000),
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
            wait_for_ready=lambda: warm.wait_for_persisted_store_value(
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

    return results


def seed_portopt_page(page, base_url: str, db_series: list[str]) -> None:
    opt_series = warm.resolve_portopt_series(db_series)
    raw_data_payload, raw_meta_payload = build_synthetic_raw_dataset(opt_series)
    page.goto(base_url + "/portopt", wait_until="domcontentloaded")
    warm.wait_dash_hydrated(page, timeout=30000)
    warm.try_set_component_props(page, "po-page-visited-store", {"data": True})
    warm.try_set_component_props(page, "dashmat-raw-data-store", {"data": raw_data_payload})
    warm.try_set_component_props(page, "dashmat-raw-data-meta-store", {"data": raw_meta_payload})
    warm.try_set_component_props(page, "po-series-select", {"data": opt_series})
    warm.try_set_component_props(page, "po-series-select-value-store", {"data": opt_series})
    warm.try_set_component_props(page, "po-series-order-store", {"data": opt_series})
    warm.try_set_component_props(page, "po-cmabench-defaults-store", {"data": {}})
    warm.wait_visible(page, "#po-main-container", timeout=30000)
    warm.wait_ready(page, "#po-opt-model-select", timeout=30000)
    warm.wait_hidden_or_absent(page, "#po-ui-blocker-overlay", timeout=30000)


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
            wait_for_ready=lambda: warm.wait_content_ready(page, "#reg-statistics-content", timeout=10000),
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
            wait_for_ready=lambda: warm.wait_content_ready(page, "#reg-rolling-returns-content", timeout=10000),
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
            wait_for_ready=lambda: warm.wait_content_ready(page, "#reg-rolling-returns-content", timeout=10000),
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
            wait_for_ready=lambda: warm.wait_content_ready(page, "#reg-rolling-returns-content", timeout=10000),
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
                warm.wait_content_ready(page, "#reg-anova-content", timeout=10000),
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
                warm.wait_content_ready(page, "#reg-statistics-content", timeout=10000),
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
                warm.wait_content_ready(page, "#reg-scatter-content", timeout=10000),
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
                warm.wait_content_ready(page, "#reg-rolling-returns-content", timeout=10000),
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
    warm.wait_dash_hydrated(page, timeout=30000)
    warm.try_set_component_props(page, "reg-page-visited-store", {"data": True})
    warm.try_set_component_props(page, "dashmat-raw-data-store", {"data": raw_data_payload})
    warm.try_set_component_props(page, "dashmat-raw-data-meta-store", {"data": raw_meta_payload})
    warm.try_set_component_props(page, "reg-series-select", {"data": x_series})
    warm.try_set_component_props(page, "reg-series-select-value-store", {"data": x_series})
    warm.try_set_component_props(page, "reg-series-order-store", {"data": series_order})
    warm.try_set_component_props(page, "reg-dependent-var-store", {"data": dep_var})
    warm.wait_visible(page, "#reg-main-container", timeout=30000)
    warm.wait_ready(page, "#reg-model-select", timeout=30000)
    warm.wait_hidden_or_absent(page, "#reg-ui-blocker-overlay", timeout=30000)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DashMat UI callback interaction harness")
    parser.add_argument("--base-url", default="http://127.0.0.1:8050")
    parser.add_argument("--pages", choices=["portopt", "regression", "both"], default="both")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--label", default="ui-callback-harness")
    parser.add_argument("--db-series", nargs="*", default=DEFAULT_DB_SERIES)
    return parser


def run_page_suite(page_name: str, base_url: str, db_series: list[str], headless: bool, runs: int) -> dict[str, object]:
    run_results: list[dict[str, object]] = []
    for _ in range(max(runs, 1)):
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=headless)
            context = browser.new_context()
            page = context.new_page()
            tracker = warm.DashUpdateRequestTracker(page)
            if page_name == "portopt":
                seed_portopt_page(page, base_url, db_series)
                run_results.extend(run_portopt_scenarios(page, tracker))
            else:
                seed_regression_page(page, base_url, db_series)
                run_results.extend(run_regression_scenarios(page, tracker, db_series))
            context.close()
            browser.close()

    grouped: dict[str, list[dict[str, object]]] = {}
    for result in run_results:
        grouped.setdefault(str(result["scenario"]), []).append(result)
    return {
        "page": page_name,
        "runs": max(runs, 1),
        "scenarios": {name: {"summary": summarize_run_group(values), "runs": values} for name, values in grouped.items()},
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    pages = ["portopt", "regression"] if args.pages == "both" else [args.pages]

    out_dir = REPO_ROOT / "output" / "playwright"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{_timestamp_token()}_{args.label.replace(' ', '_')}"
    output_path = out_dir / f"{stem}.json"

    try:
        result = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "label": args.label,
            "baseUrl": args.base_url,
            "pages": {
                page_name: run_page_suite(page_name, args.base_url, list(args.db_series or DEFAULT_DB_SERIES), args.headless, args.runs)
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
