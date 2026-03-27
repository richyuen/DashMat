from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import median

import pandas as pd
from playwright.sync_api import sync_playwright

import warm_switch_harness as warm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.raw_dataset import build_raw_data_store_payload


ACTIVE_VIS_TAB_LABELS = {
    "weight": "Weights",
    "returns": "Returns",
    "rolling": "Rolling",
    "statistics": "Statistics",
    "calendar": "Calendar Year",
    "drawdown": "Drawdown",
}


def extract_dash_output_ids(post_data: str | None) -> list[str]:
    if not post_data:
        return []

    try:
        payload = json.loads(post_data)
    except json.JSONDecodeError:
        return []

    outputs: list[str] = []

    def _append_output(value) -> None:
        if isinstance(value, str):
            outputs.append(value)
            return
        if isinstance(value, dict):
            output_id = value.get("id")
            output_prop = value.get("property")
            if isinstance(output_id, dict):
                output_id = json.dumps(output_id, sort_keys=True, separators=(",", ":"))
            if output_id is not None and output_prop:
                outputs.append(f"{output_id}.{output_prop}")
            return
        if isinstance(value, list):
            for item in value:
                _append_output(item)

    _append_output(payload.get("output"))
    _append_output(payload.get("outputs"))
    return sorted(set(outputs))


class DashUpdateRequestTracker:
    def __init__(self, page):
        self.page = page
        self.active_requests: dict[int, dict[str, object]] = {}
        self.records: list[dict[str, object]] = []
        self.window_start_at: float | None = None
        self.window_end_at: float | None = None
        page.on("request", self._on_request)
        page.on("requestfinished", self._on_request_finished)
        page.on("requestfailed", self._on_request_failed)

    def _on_request(self, request) -> None:
        if "/_dash-update-component" not in request.url:
            return
        post_data = request.post_data or ""
        self.active_requests[id(request)] = {
            "started_at": time.perf_counter(),
            "requestBytes": len(post_data.encode("utf-8")) if post_data else 0,
            "outputs": extract_dash_output_ids(post_data),
        }

    def _finalize_request(self, request) -> None:
        record = self.active_requests.pop(id(request), None)
        if record is None:
            return
        started_at = float(record.get("started_at", 0) or 0)
        if self.window_start_at is None or started_at < self.window_start_at:
            return
        if self.window_end_at is not None and started_at > self.window_end_at:
            return

        response_bytes = 0
        server_ms = None
        response = request.response()
        if response is not None:
            try:
                content_length = response.header_value("content-length")
                if content_length:
                    response_bytes = int(content_length)
                else:
                    response_bytes = len(response.body())
            except Exception:
                response_bytes = 0
            try:
                server_ms = warm.parse_server_timing_duration(response.header_value("server-timing"))
            except Exception:
                server_ms = None

        finished_at = time.perf_counter()
        duration_ms = round((finished_at - record["started_at"]) * 1000)
        window_start_at = self.window_start_at or started_at
        self.records.append(
            {
                "_started_at": started_at,
                "_finished_at": finished_at,
                "durationMs": duration_ms,
                "serverMs": round(server_ms, 2) if server_ms is not None else None,
                "startedOffsetMs": round((started_at - window_start_at) * 1000),
                "finishedOffsetMs": round((finished_at - window_start_at) * 1000),
                "requestBytes": record["requestBytes"],
                "responseBytes": response_bytes,
                "outputs": record["outputs"],
            }
        )

    def _on_request_finished(self, request) -> None:
        self._finalize_request(request)

    def _on_request_failed(self, request) -> None:
        self._finalize_request(request)

    def start_window(self) -> None:
        self.records = []
        self.window_start_at = time.perf_counter()
        self.window_end_at = None

    def stop_window(self) -> None:
        self.window_end_at = time.perf_counter()

    def wait_for_settle(self, timeout_ms: int = 5000) -> None:
        deadline = time.perf_counter() + (timeout_ms / 1000.0)
        while self.active_requests and time.perf_counter() < deadline:
            self.page.wait_for_timeout(50)

    def _public_record(self, record: dict[str, object]) -> dict[str, object]:
        return {
            "durationMs": int(record.get("durationMs", 0) or 0),
            "serverMs": record.get("serverMs"),
            "startedOffsetMs": int(record.get("startedOffsetMs", 0) or 0),
            "finishedOffsetMs": int(record.get("finishedOffsetMs", 0) or 0),
            "requestBytes": int(record.get("requestBytes", 0) or 0),
            "responseBytes": int(record.get("responseBytes", 0) or 0),
            "outputs": list(record.get("outputs", [])),
        }

    def summary(self) -> dict[str, object]:
        callback_ids: list[str] = []
        total_duration = 0
        total_server_ms = 0.0
        missing_server_timing = 0
        total_request_bytes = 0
        total_response_bytes = 0
        last_finished_offset_ms = None
        for record in self.records:
            total_duration += int(record.get("durationMs", 0) or 0)
            server_ms = record.get("serverMs")
            if server_ms is None:
                missing_server_timing += 1
            else:
                total_server_ms += float(server_ms)
            total_request_bytes += int(record.get("requestBytes", 0) or 0)
            total_response_bytes += int(record.get("responseBytes", 0) or 0)
            finished_offset_ms = int(record.get("finishedOffsetMs", 0) or 0)
            if last_finished_offset_ms is None or finished_offset_ms > last_finished_offset_ms:
                last_finished_offset_ms = finished_offset_ms
            for output_id in record.get("outputs", []):
                if output_id not in callback_ids:
                    callback_ids.append(output_id)
        return {
            "dashUpdateRequestCount": len(self.records),
            "dashUpdateTotalMs": total_duration,
            "dashUpdateSummedServerMs": round(total_server_ms),
            "dashUpdateServerTimingMissingCount": missing_server_timing,
            "dashUpdateRequestBytes": total_request_bytes,
            "dashUpdateResponseBytes": total_response_bytes,
            "dashUpdateLastFinishedOffsetMs": last_finished_offset_ms,
            "dashUpdateCallbacks": callback_ids,
            "dashUpdateRequests": [self._public_record(record) for record in self.records],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="")
    parser.add_argument("--base-url", default="http://127.0.0.1:8050")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--label", default="")
    parser.add_argument("--git-ref", default="")
    parser.add_argument("--startup-timeout", type=int, default=30)
    parser.add_argument("--skip-db-build", action="store_true")
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--mode", choices=("synthetic", "db"), default="synthetic")
    parser.add_argument(
        "--scenario",
        choices=("noop", "selection", "order", "metadata", "rename", "delete"),
        default="selection",
    )
    parser.add_argument(
        "--active-tab",
        choices=tuple(ACTIVE_VIS_TAB_LABELS.keys()),
        default="weight",
    )
    parser.add_argument("--db-series", nargs="+", default=warm.DEFAULT_DB_SERIES)
    parser.add_argument("--server-log", default="")
    return parser.parse_args()


def build_artifact_stem(label: str, git_ref: str, base_url: str, timestamp: str, mode: str, scenario: str, active_tab: str) -> str:
    git_token = warm.sanitize_token((git_ref or "unknown")[:8], "unknown")
    label_token = warm.sanitize_token(label, "run")
    mode_token = warm.sanitize_token(mode, "synthetic")
    scenario_token = warm.sanitize_token(scenario, "selection")
    active_tab_token = warm.sanitize_token(active_tab, "weight")
    port = urllib.parse.urlparse(base_url).port
    port_token = f"p{port}" if port else "punknown"
    return f"portopt_series_modal_{timestamp}_{label_token}_{mode_token}_{scenario_token}_{active_tab_token}_{git_token}_{port_token}"


def build_synthetic_raw_dataset(series_names: list[str]) -> tuple[dict[str, object], dict[str, object]]:
    resolved_series = list(series_names or warm.DEFAULT_DB_SERIES)
    index = pd.bdate_range("2020-01-01", periods=320)
    data = {}
    for idx, series in enumerate(resolved_series):
        offset = (idx + 1) * 0.00015
        values = [offset + ((((day + idx) % 13) - 6) / 10000.0) for day in range(len(index))]
        data[series] = values
    df = pd.DataFrame(data, index=index)
    df.index.name = "Date"
    return build_raw_data_store_payload(df), {"columns": resolved_series}


def wait_modal_grid_ready(page, expected_rows: int, timeout: int = 30000) -> None:
    warm.wait_visible(page, "#po-modal-ok-button", timeout=timeout)
    warm.wait_ready(page, "#po-modal-ok-button", timeout=timeout)
    page.wait_for_function(
        """
        async (expected) => {
          const grid = document.querySelector("#po-series-selection-grid");
          if (!grid) return false;
          const style = window.getComputedStyle(grid);
          const rect = grid.getBoundingClientRect();
          if (style.display === "none" || style.visibility === "hidden" || rect.width <= 0 || rect.height <= 0) {
            return false;
          }
          if (!window.dash_ag_grid || !window.dash_ag_grid.getApiAsync) {
            return false;
          }
          try {
            const api = await window.dash_ag_grid.getApiAsync("po-series-selection-grid");
            if (!api) return false;
            let rowCount = 0;
            api.forEachNodeAfterFilterAndSort(function () {
              rowCount += 1;
            });
            return rowCount >= expected;
          } catch (_err) {
            return false;
          }
        }
        """,
        arg=max(int(expected_rows or 0), 1),
        timeout=timeout,
    )


def grid_rows(page):
    return page.evaluate(
        """
        async () => {
          if (!window.dash_ag_grid || !window.dash_ag_grid.getApiAsync) {
            return [];
          }
          try {
            const api = await window.dash_ag_grid.getApiAsync("po-series-selection-grid");
            if (!api) {
              return [];
            }
            const rows = [];
            api.forEachNodeAfterFilterAndSort(function (node) {
              if (node && node.data) {
                rows.push(Object.assign({}, node.data));
              }
            });
            return rows;
          } catch (_err) {
            return [];
          }
        }
        """
    )


def wait_grid_rows(page, expected_rows, timeout: int = 15000) -> None:
    page.wait_for_function(
        """
        async (expectedRows) => {
          if (!window.dash_ag_grid || !window.dash_ag_grid.getApiAsync) {
            return false;
          }
          try {
            const api = await window.dash_ag_grid.getApiAsync("po-series-selection-grid");
            if (!api) {
              return false;
            }
            const rows = [];
            api.forEachNodeAfterFilterAndSort(function (node) {
              if (node && node.data) {
                rows.push(Object.assign({}, node.data));
              }
            });
            return JSON.stringify(rows) === JSON.stringify(expectedRows);
          } catch (_err) {
            return false;
          }
        }
        """,
        arg=expected_rows,
        timeout=timeout,
    )


def wait_selected_state(page, expected: bool, expected_rows: int, timeout: int = 15000) -> None:
    page.wait_for_function(
        """
        async ([expected, minimumRows]) => {
          if (!window.dash_ag_grid || !window.dash_ag_grid.getApiAsync) {
            return false;
          }
          try {
            const api = await window.dash_ag_grid.getApiAsync("po-series-selection-grid");
            if (!api) {
              return false;
            }
            let rowCount = 0;
            let matched = 0;
            api.forEachNodeAfterFilterAndSort(function (node) {
              if (!node || !node.data || node.data.Delete) {
                return;
              }
              rowCount += 1;
              if (!!node.data.Selected === expected) {
                matched += 1;
              }
            });
            return rowCount >= minimumRows && rowCount === matched;
          } catch (_err) {
            return false;
          }
        }
        """,
        arg=[bool(expected), max(int(expected_rows or 0), 1)],
        timeout=timeout,
    )


def set_first_row_selected(page, value: bool):
    return page.evaluate(
        """
        async (nextValue) => {
          if (!window.dash_ag_grid || !window.dash_ag_grid.getApiAsync) {
            return null;
          }
          try {
            const api = await window.dash_ag_grid.getApiAsync("po-series-selection-grid");
            if (!api) {
              return null;
            }
            let firstNode = null;
            api.forEachNodeAfterFilterAndSort(function (node) {
              if (!firstNode && node && node.data && !node.data.Delete) {
                firstNode = node;
              }
            });
            if (!firstNode || !firstNode.data) {
              return null;
            }
            firstNode.setData(Object.assign({}, firstNode.data, { Selected: !!nextValue }));
            try {
              api.refreshCells({ columns: ["Selected"], force: true });
            } catch (_err) {
            }
            return firstNode.data.Series || firstNode.data.__row_key || null;
          } catch (_err) {
            return null;
          }
        }
        """,
        bool(value),
    )


def set_grid_rows(page, rows: list[dict]) -> None:
    applied = page.evaluate(
        """
        async (nextRows) => {
          if (!window.dash_ag_grid || !window.dash_ag_grid.getApiAsync) {
            return false;
          }
          try {
            const api = await window.dash_ag_grid.getApiAsync("po-series-selection-grid");
            if (!api) {
              return false;
            }
            if (typeof api.setGridOption === "function") {
              api.setGridOption("rowData", nextRows);
            } else if (typeof api.setRowData === "function") {
              api.setRowData(nextRows);
            } else {
              return false;
            }
            return true;
          } catch (_err) {
            return false;
          }
        }
        """,
        rows,
    )
    if not applied and not warm.try_set_component_props(page, "po-series-selection-grid", {"rowData": rows}):
        raise RuntimeError("PortOpt modal harness could not set po-series-selection-grid rowData.")
    wait_grid_rows(page, rows, timeout=15000)


def install_ok_timing_probe(page) -> None:
    page.wait_for_function(
        """
        () => !!(
          window.dash_clientside &&
          window.dash_clientside.dashmat_callbacks &&
          typeof window.dash_clientside.dashmat_callbacks.capturePortoptSeriesSnapshot === "function"
        )
        """,
        timeout=30000,
    )
    page.evaluate(
        """
        () => {
          window.__poModalOkPerf = window.__poModalOkPerf || { clickStartedAt: null, snapshotSetAt: null };
          if (window.__poModalOkPerfInstalled) {
            return true;
          }
          const namespace = window.dash_clientside.dashmat_callbacks;
          const original = namespace.capturePortoptSeriesSnapshot.bind(namespace);
          namespace.capturePortoptSeriesSnapshot = async function (...args) {
            const result = await original(...args);
            window.__poModalOkPerf.snapshotSetAt = performance.now();
            return result;
          };
          window.__poModalOkPerfInstalled = true;
          return true;
        }
        """
    )


def arm_ok_timing_probe(page) -> None:
    page.evaluate(
        """
        () => {
          window.__poModalOkPerf = {
            clickStartedAt: performance.now(),
            snapshotSetAt: null,
          };
        }
        """
    )


def wait_for_snapshot_capture(page, timeout: int = 15000) -> None:
    page.wait_for_function(
        """
        () => {
          const state = window.__poModalOkPerf;
          return !!(
            state &&
            typeof state.clickStartedAt === "number" &&
            typeof state.snapshotSetAt === "number" &&
            state.snapshotSetAt >= state.clickStartedAt
          );
        }
        """,
        timeout=timeout,
    )


def apply_ok_scenario(page, scenario: str) -> tuple[list[dict], list[dict]]:
    baseline_rows = [dict(row) for row in grid_rows(page)]
    if not baseline_rows:
        raise RuntimeError("PortOpt modal harness could not read baseline modal rows.")

    next_rows = [dict(row) for row in baseline_rows]
    live_indexes = [idx for idx, row in enumerate(next_rows) if not row.get("Delete")]
    if not live_indexes:
        raise RuntimeError("PortOpt modal harness found no live rows for OK scenario.")

    first_idx = live_indexes[0]
    if scenario == "noop":
        return baseline_rows, next_rows
    if scenario == "selection":
        next_rows[first_idx]["Selected"] = not bool(next_rows[first_idx].get("Selected", False))
    elif scenario == "order":
        if len(live_indexes) < 2:
            raise RuntimeError("PortOpt modal harness needs at least two live rows for order scenario.")
        ordered = [next_rows[idx] for idx in live_indexes]
        moved = ordered.pop(0)
        ordered.append(moved)
        for row_idx, row in zip(live_indexes, ordered, strict=False):
            next_rows[row_idx] = row
    elif scenario == "metadata":
        current_max = next_rows[first_idx].get("MaxWt")
        next_rows[first_idx]["MaxWt"] = 55 if str(current_max) != "55" else 65
    elif scenario == "rename":
        next_rows[first_idx]["Series"] = f"{next_rows[first_idx]['Series']}_Renamed"
    elif scenario == "delete":
        next_rows[first_idx]["Delete"] = True
        next_rows[first_idx]["Selected"] = False
    else:
        raise RuntimeError(f"Unsupported PortOpt modal OK scenario: {scenario}")

    set_grid_rows(page, next_rows)
    return baseline_rows, next_rows


def wait_session_store(page, component_id: str, expected_value, timeout: int = 15000) -> None:
    page.wait_for_function(
        """
        ([componentId, expected]) => {
          try {
            const raw = window.sessionStorage.getItem(componentId);
            if (raw === null) {
              return false;
            }
            return JSON.stringify(JSON.parse(raw)) === JSON.stringify(expected);
          } catch (_err) {
            return false;
          }
        }
        """,
        arg=[component_id, expected_value],
        timeout=timeout,
    )


def wait_session_store_keys(page, component_id: str, expected_keys: list[str], timeout: int = 15000) -> None:
    page.wait_for_function(
        """
        ([componentId, expectedKeys]) => {
          try {
            const raw = window.sessionStorage.getItem(componentId);
            if (raw === null) {
              return false;
            }
            const parsed = JSON.parse(raw);
            if (!parsed || typeof parsed !== "object") {
              return false;
            }
            return expectedKeys.every(function (key) {
              return typeof parsed[key] === "string" && parsed[key].trim().length > 0;
            });
          } catch (_err) {
            return false;
          }
        }
        """,
        arg=[component_id, list(expected_keys or [])],
        timeout=timeout,
    )


def reset_modal_seed_state(page, opt_series: list[str]) -> None:
    warm.try_set_component_props(page, "po-page-visited-store", {"data": True})
    warm.try_set_component_props(page, "po-series-select", {"data": opt_series})
    warm.try_set_component_props(page, "po-series-select-value-store", {"data": opt_series})
    warm.try_set_component_props(page, "po-series-order-store", {"data": opt_series})
    warm.try_set_component_props(page, "po-temp-series-select", {"data": []})
    warm.try_set_component_props(page, "po-temp-benchmark-assignments-store", {"data": {}})
    warm.try_set_component_props(page, "po-temp-cmabench-assignments-store", {"data": {}})
    warm.try_set_component_props(page, "po-temp-long-short-store", {"data": {}})
    warm.try_set_component_props(page, "po-temp-series-order-store", {"data": []})
    warm.try_set_component_props(page, "po-temp-deleted-series-store", {"data": []})
    warm.try_set_component_props(page, "po-temp-vol-scaling-assignments-store", {"data": {}})
    warm.try_set_component_props(page, "po-temp-min-wt-store", {"data": {}})
    warm.try_set_component_props(page, "po-temp-max-wt-store", {"data": {}})
    warm.try_set_component_props(page, "po-temp-force-max-store", {"data": {}})
    warm.try_set_component_props(page, "po-series-selection-modal", {"opened": False})
    warm.try_set_component_props(page, "po-series-selection-grid", {"rowData": []})
    warm.try_set_component_props(page, "po-series-grid-snapshot-store", {"data": None})
    warm.try_set_component_props(page, "po-ui-blocker-store", {"data": False})
    warm.wait_hidden_or_absent(page, "#po-ui-blocker-overlay", timeout=30000)
    page.wait_for_selector("#po-modal-ok-button", state="hidden", timeout=30000)


def set_active_vis_tab(page, active_tab: str) -> None:
    target_tab = active_tab or "weight"
    if target_tab == "weight":
        return
    tab_label = ACTIVE_VIS_TAB_LABELS.get(target_tab)
    if not tab_label:
        raise RuntimeError(f"Unsupported PortOpt active tab '{target_tab}'.")
    tab = page.get_by_role("tab", name=tab_label, exact=True)
    tab.click(force=True)
    page.wait_for_function(
        """
        (expectedLabel) => {
          const active = document.querySelector('[role="tab"][aria-selected="true"]');
          return !!active && ((active.textContent || '').trim() === expectedLabel);
        }
        """,
        arg=tab_label,
        timeout=30000,
    )
    warm.wait_hidden_or_absent(page, "#po-ui-blocker-overlay", timeout=30000)


def seed_portopt_page_synthetic(page, base_url: str, opt_series: list[str], active_tab: str) -> None:
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
    warm.wait_ready(page, "#po-open-modal-button", timeout=30000)
    warm.wait_hidden_or_absent(page, "#po-ui-blocker-overlay", timeout=30000)

    page.locator("#po-open-modal-button").click(force=True)
    wait_modal_grid_ready(page, expected_rows=len(opt_series), timeout=30000)
    page.locator("#po-modal-cancel-button").click()
    page.wait_for_selector("#po-modal-ok-button", state="hidden", timeout=30000)
    install_ok_timing_probe(page)
    set_active_vis_tab(page, active_tab)


def seed_portopt_page_db(page, base_url: str, opt_series: list[str], active_tab: str) -> None:
    page.goto(base_url + "/portopt", wait_until="domcontentloaded")
    warm.wait_dash_hydrated(page, timeout=30000)
    warm.try_set_component_props(page, "po-page-visited-store", {"data": True})

    if not warm.try_set_component_props(page, "po-db-add-series-select", {"value": opt_series}):
        raise RuntimeError("PortOpt modal harness could not seed po-db-add-series-select for database mode.")
    if not warm.try_set_component_props(page, "po-db-add-ok-button", {"n_clicks": 1}):
        raise RuntimeError("PortOpt modal harness could not trigger po-db-add-ok-button for database mode.")

    wait_modal_grid_ready(page, expected_rows=len(opt_series), timeout=60000)
    wait_session_store_keys(page, "po-cmabench-defaults-store", opt_series, timeout=60000)

    page.locator("#po-modal-ok-button").click()
    page.wait_for_selector("#po-modal-ok-button", state="hidden", timeout=30000)
    warm.wait_hidden_or_absent(page, "#po-ui-blocker-overlay", timeout=30000)
    wait_session_store(page, "po-series-select", opt_series, timeout=30000)
    install_ok_timing_probe(page)
    set_active_vis_tab(page, active_tab)


def measure_modal_run(page, opt_series: list[str], scenario: str, request_tracker: DashUpdateRequestTracker) -> dict[str, object]:
    reset_modal_seed_state(page, opt_series)

    open_start = time.perf_counter()
    page.locator("#po-open-modal-button").click(force=True)
    wait_modal_grid_ready(page, expected_rows=len(opt_series), timeout=30000)
    modal_open_ms = round((time.perf_counter() - open_start) * 1000)
    baseline_rows = [dict(row) for row in grid_rows(page)]
    if not baseline_rows:
        raise RuntimeError("PortOpt modal harness could not capture modal baseline rows.")

    first_series = set_first_row_selected(page, False)
    if not first_series:
        raise RuntimeError("PortOpt modal harness could not identify a live grid row.")

    select_start = time.perf_counter()
    page.locator("#po-select-all-button").click()
    wait_selected_state(page, True, expected_rows=len(opt_series), timeout=15000)
    select_all_ms = round((time.perf_counter() - select_start) * 1000)

    unselect_start = time.perf_counter()
    page.locator("#po-unselect-all-button").click()
    wait_selected_state(page, False, expected_rows=len(opt_series), timeout=15000)
    unselect_all_ms = round((time.perf_counter() - unselect_start) * 1000)

    page.locator("#po-modal-cancel-button").click(force=True)
    page.wait_for_selector("#po-modal-ok-button", state="hidden", timeout=30000)
    warm.wait_hidden_or_absent(page, "#po-ui-blocker-overlay", timeout=30000)

    reset_modal_seed_state(page, opt_series)
    page.locator("#po-open-modal-button").click(force=True)
    wait_modal_grid_ready(page, expected_rows=len(opt_series), timeout=30000)
    _, scenario_rows = apply_ok_scenario(page, scenario)

    request_tracker.wait_for_settle(timeout_ms=5000)
    request_tracker.start_window()
    arm_ok_timing_probe(page)
    ok_start = time.perf_counter()
    page.locator("#po-modal-ok-button").click()
    wait_for_snapshot_capture(page, timeout=30000)
    snapshot_ms = round((time.perf_counter() - ok_start) * 1000)
    page.wait_for_selector("#po-modal-ok-button", state="hidden", timeout=30000)
    warm.wait_hidden_or_absent(page, "#po-ui-blocker-overlay", timeout=30000)
    ok_confirm_ms = round((time.perf_counter() - ok_start) * 1000)
    request_tracker.stop_window()
    request_tracker.wait_for_settle(timeout_ms=5000)
    apply_ms = max(ok_confirm_ms - snapshot_ms, 0)
    request_summary = request_tracker.summary()

    return {
        "scenario": scenario,
        "modalOpenMs": modal_open_ms,
        "selectAllMs": select_all_ms,
        "unselectAllMs": unselect_all_ms,
        "snapshotMs": snapshot_ms,
        "applyMs": apply_ms,
        "okConfirmMs": ok_confirm_ms,
        "selectedAfterOk": [row["Series"] for row in scenario_rows if row.get("Selected") and not row.get("Delete")],
        "gridRowCount": len(baseline_rows),
        **request_summary,
    }


def run_harness(args: argparse.Namespace, resolved_git_ref: str) -> dict[str, object]:
    server_log_path = Path(args.server_log).resolve() if args.server_log else None
    console_messages: list[dict[str, str]] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not args.headed)
        opt_series = warm.resolve_portopt_series(args.db_series)
        timing_start_offset = warm.current_log_offset(server_log_path)

        run_results = []
        for run_index in range(1, args.runs + 1):
            page = browser.new_page(viewport={"width": 1440, "height": 960})
            request_tracker = DashUpdateRequestTracker(page)
            page.on(
                "console",
                lambda msg: console_messages.append({"type": msg.type, "text": msg.text})
                if len(console_messages) < 120 and msg.type in {"error", "warning"}
                else None,
            )
            page.on(
                "pageerror",
                lambda err: console_messages.append({"type": "pageerror", "text": str(err)})
                if len(console_messages) < 120
                else None,
            )
            if args.mode == "db":
                seed_portopt_page_db(page, args.base_url, opt_series, args.active_tab)
            else:
                seed_portopt_page_synthetic(page, args.base_url, opt_series, args.active_tab)
            warm.wait_hidden_or_absent(page, "#po-ui-blocker-overlay", timeout=30000)
            result = measure_modal_run(page, opt_series, args.scenario, request_tracker)
            result["run"] = run_index
            run_results.append(result)
            page.close()

        browser.close()

    modal_open_values = [run["modalOpenMs"] for run in run_results]
    select_all_values = [run["selectAllMs"] for run in run_results]
    unselect_all_values = [run["unselectAllMs"] for run in run_results]
    snapshot_values = [run["snapshotMs"] for run in run_results]
    apply_values = [run["applyMs"] for run in run_results]
    ok_confirm_values = [run["okConfirmMs"] for run in run_results]
    dash_request_counts = [run["dashUpdateRequestCount"] for run in run_results]
    dash_request_durations = [run["dashUpdateTotalMs"] for run in run_results]
    dash_server_durations = [run.get("dashUpdateSummedServerMs", 0) for run in run_results]
    dash_server_missing = [run.get("dashUpdateServerTimingMissingCount", 0) for run in run_results]
    dash_request_bytes = [run["dashUpdateRequestBytes"] for run in run_results]
    dash_response_bytes = [run["dashUpdateResponseBytes"] for run in run_results]
    callback_frequency: Counter[str] = Counter()
    for run in run_results:
        for request in run.get("dashUpdateRequests", []):
            for output_id in request.get("outputs", []):
                callback_frequency[str(output_id)] += 1

    return {
        "timestamp": datetime.now().astimezone().isoformat(),
        "label": args.label or "portopt-series-modal",
        "gitRef": resolved_git_ref,
        "baseUrl": args.base_url,
        "mode": args.mode,
        "scenario": args.scenario,
        "activeTab": args.active_tab,
        "dbSeries": args.db_series,
        "selectedSeries": opt_series,
        "runs": run_results,
        "summary": {
            "modalOpenMedian": round(median(modal_open_values)),
            "selectAllMedian": round(median(select_all_values)),
            "unselectAllMedian": round(median(unselect_all_values)),
            "snapshotMedian": round(median(snapshot_values)),
            "applyMedian": round(median(apply_values)),
            "okConfirmMedian": round(median(ok_confirm_values)),
            "dashUpdateRequestCountMedian": round(median(dash_request_counts)),
            "dashUpdateTotalMsMedian": round(median(dash_request_durations)),
            "dashUpdateSummedServerMsMedian": round(median(dash_server_durations)),
            "dashUpdateServerTimingMissingCountMedian": round(median(dash_server_missing)),
            "dashUpdateRequestBytesMedian": round(median(dash_request_bytes)),
            "dashUpdateResponseBytesMedian": round(median(dash_response_bytes)),
            "topDashUpdateCallbacksByFrequency": [
                {"id": callback_id, "count": count}
                for callback_id, count in callback_frequency.most_common(10)
            ],
        },
        "consoleMessages": console_messages,
        "timingStartOffset": timing_start_offset,
    }


def main() -> int:
    args = parse_args()
    root = warm.resolve_repo_root(args.repo_root)
    resolved_git_ref = warm.resolve_git_ref(root, args.git_ref)

    db_rebuilt = False
    db_rebuild_reasons: list[str] = []
    if not args.skip_db_build:
        db_rebuilt, db_rebuild_reasons = warm.ensure_local_seed_databases(root)

    warm.wait_for_app(args.base_url, args.startup_timeout)

    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    stem = build_artifact_stem(
        args.label or "portopt-series-modal",
        resolved_git_ref,
        args.base_url,
        timestamp,
        args.mode,
        args.scenario,
        args.active_tab,
    )
    out_dir = root / "output" / "playwright"
    fail_dir = out_dir / "failures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fail_dir.mkdir(parents=True, exist_ok=True)

    server_log_path = Path(args.server_log).resolve() if args.server_log else None

    try:
        result = run_harness(args, resolved_git_ref)
    except Exception as exc:
        page_state = {"url": args.base_url + "/portopt"}
        raw_path = warm.write_failure_artifacts(
            out_dir=out_dir,
            fail_dir=fail_dir,
            stem=stem,
            repo_root=root,
            base_url=args.base_url,
            git_ref=resolved_git_ref,
            label=args.label or "portopt-series-modal",
            db_series=args.db_series,
            startup_timeout=args.startup_timeout,
            console_messages=[],
            exc=exc,
            page_state=page_state,
        )
        print(f"RAW_PATH={raw_path}")
        return 1

    timing_summary = warm.parse_timing_log(server_log_path, start_offset=result["timingStartOffset"])
    timing_summary["copiedPath"] = warm.copy_server_log(server_log_path, out_dir, stem)

    payload = {
        "timestamp": result["timestamp"],
        "label": result["label"],
        "gitRef": resolved_git_ref,
        "baseUrl": args.base_url,
        "repoRoot": str(root),
        "mode": args.mode,
        "scenario": args.scenario,
        "dbSeries": args.db_series,
        "dbRebuilt": db_rebuilt,
        "dbRebuildReasons": db_rebuild_reasons,
        "selectedSeries": result["selectedSeries"],
        "runs": result["runs"],
        "summary": result["summary"],
        "consoleMessages": result["consoleMessages"],
        "timingSummary": timing_summary,
    }
    out_path = out_dir / f"{stem}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"OUT_PATH={out_path}")
    print("TIMING=" + json.dumps(timing_summary, separators=(",", ":")))
    print("PORTOPT_MODAL=" + json.dumps({"summary": result["summary"], "runs": result["runs"]}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
