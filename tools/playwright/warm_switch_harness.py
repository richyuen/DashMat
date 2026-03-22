from __future__ import annotations

import argparse
from collections import Counter
import json
import re
import sqlite3
import subprocess
import shutil
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from statistics import median

from playwright.sync_api import sync_playwright

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.date_range_flow import compute_date_range_candidates
from utils.raw_dataset import resolve_dataset_key
from utils.account_lists import (
    ACCOUNT_LIST_CAPTURE_STORE_IDS,
    build_account_list_payload,
    list_account_lists,
    load_account_list_by_id,
    save_account_list,
)
from dbengine import engine as DB_ENGINE


DEFAULT_DB_SERIES = [
    "SPX_TRIndex",
    "R2000_TRIndex",
    "EAFE_TRIndex",
    "BCTBill13_TRIndex",
]
PAGE_ORDER = ["analytics", "portopt", "regression"]

TIMING_EVENT_NAMES = (
    "analyticstool.render_statistics_grid",
    "analyticstool.render_returns_grid",
    "analyticstool.download_excel.total",
    "portopt.performance_frames",
    "portopt.project_results",
    "portopt.render_weight_chart",
    "portopt.render_attribution_chart",
    "portopt.render_statistics",
    "portopt.render_risk_chart",
    "portopt.render_frontier_chart",
    "portopt.render_attribution_table",
    "portopt.render_risk_table",
    "portopt.render_frontier_table",
    "regression.sync_result_options",
    "regression.sync_save_series_ui",
    "regression.sync_anova_window_options",
    "regression.display_series",
    "regression.render_anova",
    "regression.render_rolling",
    "regression.render_rolling_returns",
    "regression.render_weights",
    "regression.render_returns",
    "regression.render_growth",
    "regression.render_calendar",
    "regression.render_drawdown",
    "regression.render_statistics",
    "regression.render_scatter",
)

REGRESSION_EMPTY_STATE_TEXTS = (
    "Run a regression to see results.",
    "No results.",
)


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
        response = request.response()
        if response is not None:
            try:
                content_length = response.headers.get("content-length")
                if content_length:
                    response_bytes = int(content_length)
                else:
                    response_bytes = len(response.body())
            except Exception:
                response_bytes = 0

        duration_ms = round((time.perf_counter() - record["started_at"]) * 1000)
        self.records.append(
            {
                "durationMs": duration_ms,
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

    def summary(self) -> dict[str, object]:
        callback_ids: list[str] = []
        total_duration = 0
        total_request_bytes = 0
        total_response_bytes = 0
        for record in self.records:
            total_duration += int(record.get("durationMs", 0) or 0)
            total_request_bytes += int(record.get("requestBytes", 0) or 0)
            total_response_bytes += int(record.get("responseBytes", 0) or 0)
            for output_id in record.get("outputs", []):
                if output_id not in callback_ids:
                    callback_ids.append(output_id)
        return {
            "dashUpdateRequestCount": len(self.records),
            "dashUpdateTotalMs": total_duration,
            "dashUpdateRequestBytes": total_request_bytes,
            "dashUpdateResponseBytes": total_response_bytes,
            "dashUpdateCallbacks": callback_ids,
            "dashUpdateRequests": self.records,
        }


def summarize_dash_update_runs(run_results: list[dict[str, object]]) -> dict[str, object]:
    dash_request_counts = [run["dashUpdateRequestCount"] for run in run_results]
    dash_request_durations = [run["dashUpdateTotalMs"] for run in run_results]
    dash_request_bytes = [run["dashUpdateRequestBytes"] for run in run_results]
    dash_response_bytes = [run["dashUpdateResponseBytes"] for run in run_results]
    flow_durations = [run["flowMs"] for run in run_results]
    callback_counter: Counter[str] = Counter()
    for run in run_results:
        for request in run.get("dashUpdateRequests", []):
            for output_id in request.get("outputs", []):
                callback_counter[output_id] += 1

    return {
        "runs": len(run_results),
        "flowMs": flow_durations,
        "flowMedian": round(median(flow_durations)) if flow_durations else 0,
        "dashUpdateRequestCountMedian": round(median(dash_request_counts)) if dash_request_counts else 0,
        "dashUpdateTotalMsMedian": round(median(dash_request_durations)) if dash_request_durations else 0,
        "dashUpdateRequestBytesMedian": round(median(dash_request_bytes)) if dash_request_bytes else 0,
        "dashUpdateResponseBytesMedian": round(median(dash_response_bytes)) if dash_response_bytes else 0,
        "topDashUpdateCallbacksByFrequency": [
            {"callback": callback_id, "count": count}
            for callback_id, count in callback_counter.most_common(12)
        ],
        "runResults": run_results,
    }


def summarize_account_list_runs(run_results: list[dict[str, object]]) -> dict[str, object]:
    click_to_reload = [int(run["clickToReloadStartMs"]) for run in run_results]
    reload_to_ready = [int(run["reloadStartToReadyMs"]) for run in run_results]
    total_click_to_ready = [int(run["totalClickToReadyMs"]) for run in run_results]
    dash_request_counts = [int(run["dashUpdateRequestCount"]) for run in run_results]
    dash_request_durations = [int(run["dashUpdateTotalMs"]) for run in run_results]
    dash_request_bytes = [int(run["dashUpdateRequestBytes"]) for run in run_results]
    dash_response_bytes = [int(run["dashUpdateResponseBytes"]) for run in run_results]
    callback_counter: Counter[str] = Counter()
    for run in run_results:
        for request in run.get("dashUpdateRequests", []):
            for output_id in request.get("outputs", []):
                callback_counter[output_id] += 1

    return {
        "runs": len(run_results),
        "clickToReloadStartMs": click_to_reload,
        "reloadStartToReadyMs": reload_to_ready,
        "totalClickToReadyMs": total_click_to_ready,
        "clickToReloadStartMedian": round(median(click_to_reload)) if click_to_reload else 0,
        "reloadStartToReadyMedian": round(median(reload_to_ready)) if reload_to_ready else 0,
        "totalClickToReadyMedian": round(median(total_click_to_ready)) if total_click_to_ready else 0,
        "dashUpdateRequestCountMedian": round(median(dash_request_counts)) if dash_request_counts else 0,
        "dashUpdateTotalMsMedian": round(median(dash_request_durations)) if dash_request_durations else 0,
        "dashUpdateRequestBytesMedian": round(median(dash_request_bytes)) if dash_request_bytes else 0,
        "dashUpdateResponseBytesMedian": round(median(dash_response_bytes)) if dash_response_bytes else 0,
        "topDashUpdateCallbacksByFrequency": [
            {"callback": callback_id, "count": count}
            for callback_id, count in callback_counter.most_common(12)
        ],
        "runResults": run_results,
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
    parser.add_argument("--db-series", nargs="+", default=DEFAULT_DB_SERIES)
    parser.add_argument("--portopt-restore-tab", default="weight")
    parser.add_argument("--portopt-entry-only", action="store_true")
    parser.add_argument("--measure-account-list-load", action="store_true")
    parser.add_argument("--server-log", default="")
    parser.add_argument("--pages", nargs="+", default=PAGE_ORDER)
    return parser.parse_args()


def normalize_pages(values: list[str] | tuple[str, ...] | None) -> list[str]:
    requested = values or PAGE_ORDER
    normalized: list[str] = []
    for raw_value in requested:
        for token in str(raw_value or "").split(","):
            page = token.strip().lower()
            if not page:
                continue
            if page not in PAGE_ORDER:
                raise ValueError(f"Unsupported page selection: {page}")
            if page not in normalized:
                normalized.append(page)
    return normalized or PAGE_ORDER


def resolve_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_repo_root(repo_root: str) -> Path:
    return Path(repo_root).resolve() if repo_root else resolve_root()


def resolve_git_ref(root: Path, git_ref: str) -> str:
    if git_ref:
        return git_ref
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _sqlite_has_table(path: Path, table_name: str) -> bool:
    try:
        with sqlite3.connect(path) as conn:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name = ?",
                (table_name,),
            ).fetchone()
        return bool(row)
    except sqlite3.DatabaseError:
        return False


def ensure_local_seed_databases(repo_root: Path) -> tuple[bool, list[str]]:
    data_dir = repo_root / "data"
    checks = [
        (data_dir / "dashmat_local.db", "CoreCategories"),
        (data_dir / "MRD.db", "CORE_DATA.ACCOUNT"),
        (data_dir / "Performance.db", "ACCOUNT"),
    ]
    reasons: list[str] = []
    for path, required_table in checks:
        if not path.exists():
            reasons.append(f"missing {path.name}")
            continue
        if path.stat().st_size == 0:
            reasons.append(f"empty {path.name}")
            continue
        if not _sqlite_has_table(path, required_table):
            reasons.append(f"missing table {required_table} in {path.name}")
    if not reasons:
        return False, []
    script_path = repo_root / "tools" / "db" / "init_local_cma_db.py"
    print(f"DB_BUILD_REASON={'; '.join(reasons)}", flush=True)
    print(f"DB_BUILD=running {script_path}", flush=True)
    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=repo_root,
        check=True,
    )
    print("DB_BUILD=completed", flush=True)
    print("DB_BUILD_WARNING=restart or start the app for this repo root after rebuild before trusting timing results", flush=True)
    return True, reasons


def wait_for_app(base_url: str, startup_timeout: int) -> None:
    deadline = time.time() + max(startup_timeout, 1)
    last_error = "unknown"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(base_url, timeout=5) as response:
                if 200 <= response.status < 500:
                    return
                last_error = f"unexpected status {response.status}"
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = str(exc)
        time.sleep(1)
    raise RuntimeError(f"App did not become reachable at {base_url} within {startup_timeout}s: {last_error}")


def sanitize_token(value: str, fallback: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "-" for ch in str(value or "").strip().lower())
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    return cleaned or fallback


def build_artifact_stem(label: str, git_ref: str, base_url: str, timestamp: str) -> str:
    git_token = sanitize_token((git_ref or "unknown")[:8], "unknown")
    label_token = sanitize_token(label, "run")
    port = urllib.parse.urlparse(base_url).port
    port_token = f"p{port}" if port else "punknown"
    return f"warm_switch_{timestamp}_{label_token}_{git_token}_{port_token}"


def parse_timing_log(server_log: Path | None, start_offset: int = 0) -> dict[str, object]:
    summary = {
        "sourcePath": str(server_log) if server_log else None,
        "copiedPath": None,
        "startOffset": max(int(start_offset or 0), 0),
        "warning": None,
        "eventsPresent": {name: False for name in TIMING_EVENT_NAMES},
        "eventCounts": {name: 0 for name in TIMING_EVENT_NAMES},
        "matchedLines": [],
    }
    if not server_log or not server_log.exists():
        if server_log:
            summary["warning"] = (
                "Server log path was provided but does not exist. "
                "Launch the app with unbuffered stdout and a real file path if you want timing correlation."
            )
        return summary

    line_re = re.compile(r"timing name=(?P<name>[^ ]+)")
    matched_lines: list[str] = []
    raw_text = server_log.read_text(encoding="utf-8", errors="replace")
    if summary["startOffset"]:
        raw_text = raw_text[summary["startOffset"]:]
    for line in raw_text.splitlines():
        match = line_re.search(line)
        if not match:
            continue
        name = match.group("name")
        if name not in summary["eventsPresent"]:
            continue
        summary["eventsPresent"][name] = True
        summary["eventCounts"][name] += 1
        if len(matched_lines) < 50:
            matched_lines.append(line)
    summary["matchedLines"] = matched_lines
    if not matched_lines:
        summary["warning"] = (
            "No timing events were found in the provided server log during the measured window. "
            "Use tools/playwright/start_timed_server.ps1 so DASHMAT_TIMING_ENABLED=1 and "
            "`conda run --no-capture-output -n dashmat python -u ...` are applied consistently."
        )
    return summary


def current_log_offset(server_log: Path | None) -> int:
    if not server_log or not server_log.exists():
        return 0
    return server_log.stat().st_size


def wait_for_timing_event(server_log: Path | None, start_offset: int, name_prefix: str, timeout_ms: int = 15000) -> None:
    if not server_log or not server_log.exists():
        raise RuntimeError("Regression timing validation requires a live --server-log path.")

    needle = f"timing name={name_prefix}"
    deadline = time.time() + max(timeout_ms, 1000) / 1000.0
    while time.time() < deadline:
        raw_text = server_log.read_text(encoding="utf-8", errors="replace")
        if start_offset:
            raw_text = raw_text[start_offset:]
        if needle in raw_text:
            return
        time.sleep(0.2)

    raise RuntimeError(
        f"Expected timing events with prefix '{name_prefix}' during the measured window, "
        "but none were found in the server log."
    )


def parse_timing_fields(line: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in str(line or "").split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        fields[key] = value
    return fields


def is_regression_restored_timing_line(line: str) -> bool:
    fields = parse_timing_fields(line)
    name = str(fields.get("name", "")).strip()
    if not name.startswith("regression."):
        return False
    if name == "regression.sync_result_options":
        try:
            return int(fields.get("result_count", "0")) > 0
        except ValueError:
            return False
    result = str(fields.get("result", "")).strip()
    return bool(result and result.lower() not in {"none", "null"})


def has_regression_restored_timing_event(server_log: Path | None, start_offset: int = 0) -> bool:
    if not server_log or not server_log.exists():
        return False
    raw_text = server_log.read_text(encoding="utf-8", errors="replace")
    if start_offset:
        raw_text = raw_text[start_offset:]
    return any(is_regression_restored_timing_line(line) for line in raw_text.splitlines())


def wait_for_regression_restored_timing_event(
    server_log: Path | None,
    start_offset: int,
    timeout_ms: int = 3000,
) -> bool:
    deadline = time.time() + max(timeout_ms, 1000) / 1000.0
    while time.time() < deadline:
        if has_regression_restored_timing_event(server_log, start_offset):
            return True
        time.sleep(0.2)
    return has_regression_restored_timing_event(server_log, start_offset)


def get_regression_restore_state(page) -> dict[str, object]:
    return page.evaluate(
        """
        (emptyTexts) => {
          const storages = [window.sessionStorage, window.localStorage];
          const readPersisted = (componentId) => {
            for (const storage of storages) {
              const raw = storage.getItem(componentId);
              if (raw === null) {
                continue;
              }
              try {
                return JSON.parse(raw);
              } catch (err) {
                return raw;
              }
            }
            return null;
          };
          const normalizeText = (el) => (el ? ((el.innerText || el.textContent || "").trim()) : "");
          const results = readPersisted("reg-results-store");
          const resultKeys = results && typeof results === "object" && !Array.isArray(results)
            ? Object.keys(results)
            : [];
          const activeTab = String(readPersisted("reg-active-tab-store") || "anova");
          const contentSelectorMap = {
            anova: "#reg-anova-content",
            rolling: "#reg-rolling-content",
            scatter: "#reg-scatter-content",
            weights: "#reg-weights-content",
            statistics: "#reg-statistics-content",
            returns: "#reg-returns-content",
            rolling_returns: "#reg-rolling-returns-content",
            calendar: "#reg-calendar-content",
            growth: "#reg-growth-content",
            drawdown: "#reg-drawdown-content",
          };
          const resultInput = document.querySelector("#reg-result-select input");
          const windowInput = document.querySelector("#reg-anova-window-select input");
          const contentRoot = document.querySelector(contentSelectorMap[activeTab] || "#reg-anova-content");
          const selectedResult = resultInput ? (resultInput.value || "").trim() : "";
          const anovaWindowValue = windowInput ? (windowInput.value || "").trim() : "";
          const activeContentText = normalizeText(contentRoot);
          return {
            resultCount: resultKeys.length,
            resultKeys,
            activeTab,
            selectedResult,
            selectedInStore: !!selectedResult && resultKeys.includes(selectedResult),
            anovaWindowValue,
            anovaWindowDisabled: !!(windowInput && windowInput.disabled),
            activeContentText,
            activeContentEmptyState: emptyTexts.includes(activeContentText),
          };
        }
        """,
        list(REGRESSION_EMPTY_STATE_TEXTS),
    )


def regression_restore_state_ready(state: dict[str, object] | None) -> bool:
    state = state or {}
    result_count = int(state.get("resultCount") or 0)
    selected_in_store = bool(state.get("selectedInStore"))
    return bool(
        result_count > 0
        and (selected_in_store or result_count == 1)
        and str(state.get("activeContentText") or "").strip()
        and not bool(state.get("activeContentEmptyState"))
    )


def wait_for_regression_restore_state(page, timeout: int = 120000) -> dict[str, object]:
    deadline = time.time() + max(timeout, 1000) / 1000.0
    last_state: dict[str, object] = {}
    while time.time() < deadline:
        last_state = get_regression_restore_state(page)
        if regression_restore_state_ready(last_state):
            return last_state
        time.sleep(0.2)
    raise RuntimeError(f"Regression restore state did not become ready: {last_state}")


def ensure_regression_timing_event(page, server_log: Path | None) -> bool:
    persisted_results = get_persisted_store_value(page, "reg-results-store")
    if isinstance(persisted_results, dict) and persisted_results:
        store_offset = current_log_offset(server_log)
        set_component_props(page, "reg-results-store", {"data": {}})
        set_component_props(page, "reg-results-store", {"data": persisted_results})
        try:
            wait_for_timing_event(server_log, store_offset, "regression.sync_result_options", timeout_ms=1500)
            return True
        except RuntimeError:
            pass

    current_result = ""
    try:
        current_result = page.locator("#reg-result-select input").input_value(timeout=5000).strip()
    except Exception:
        current_result = ""

    if current_result:
        result_offset = current_log_offset(server_log)
        set_component_value(page, "reg-result-select", current_result)
        try:
            wait_for_timing_event(server_log, result_offset, "regression.", timeout_ms=1500)
            return True
        except RuntimeError:
            pass

    tab_offset = current_log_offset(server_log)
    try:
        set_component_value(page, "reg-tabs", "returns")
        wait_for_persisted_store_value(page, "reg-active-tab-store", "returns")
        wait_for_timing_event(server_log, tab_offset, "regression.", timeout_ms=3000)
        set_component_value(page, "reg-tabs", "anova")
        wait_for_persisted_store_value(page, "reg-active-tab-store", "anova")
        wait_content_ready(page, REGRESSION_TAB_CONFIG["anova"]["content"])
        return True
    except RuntimeError:
        return False


def copy_server_log(server_log: Path | None, out_dir: Path, stem: str) -> str | None:
    if not server_log or not server_log.exists():
        return None
    copied_path = out_dir / f"{stem}_server.log"
    shutil.copyfile(server_log, copied_path)
    return str(copied_path)


def write_failure_artifacts(
    *,
    out_dir: Path,
    fail_dir: Path,
    stem: str,
    repo_root: Path,
    base_url: str,
    git_ref: str,
    label: str,
    db_series: list[str],
    startup_timeout: int,
    console_messages: list[dict[str, str]] | None,
    exc: Exception,
    page_state: dict | None = None,
) -> Path:
    raw_path = out_dir / f"{stem}_traceback.txt"
    raw_path.write_text(traceback.format_exc(), encoding="utf-8")
    status = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "label": label,
        "gitRef": git_ref,
        "baseUrl": base_url,
        "repoRoot": str(repo_root),
        "dbSeries": db_series,
        "startupTimeout": startup_timeout,
        "error": str(exc),
        "consoleMessages": console_messages or [],
        "page": page_state or {},
        "tracebackPath": str(raw_path),
    }
    status_path = fail_dir / f"{stem}_status.json"
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    return raw_path


def wait_visible(page, selector: str, timeout: int = 30000) -> None:
    page.wait_for_function(
        """
        (sel) => {
          const el = document.querySelector(sel);
          if (!el) return false;
          const style = window.getComputedStyle(el);
          const rect = el.getBoundingClientRect();
          return style.display !== "none" && style.visibility !== "hidden" && rect.width > 0 && rect.height > 0;
        }
        """,
        arg=selector,
        timeout=timeout,
    )


def wait_dash_hydrated(page, timeout: int = 30000) -> None:
    page.wait_for_function(
        """
        () => {
          const title = (document.title || "").trim();
          if (!title || title === "Updating...") {
            return false;
          }
          return !document.querySelector('[data-dash-is-loading="true"]');
        }
        """,
        timeout=timeout,
    )


def wait_ready(page, selector: str, timeout: int = 30000) -> None:
    page.wait_for_function(
        """
        (sel) => {
          const el = document.querySelector(sel);
          if (!el) return false;
          const style = window.getComputedStyle(el);
          const rect = el.getBoundingClientRect();
          const visible = style.display !== "none" && style.visibility !== "hidden" && rect.width > 0 && rect.height > 0;
          if (!visible) return false;
          const selfDisabled = !!el.disabled || el.getAttribute("disabled") !== null || el.getAttribute("aria-disabled") === "true";
          const ancestorDisabled = !!el.closest('[aria-disabled="true"], [disabled]');
          return !(selfDisabled || ancestorDisabled);
        }
        """,
        arg=selector,
        timeout=timeout,
    )


def wait_analytics_db_modal_ready(page, timeout: int = 30000) -> None:
    page.wait_for_function(
        """
        () => {
          const visible = (el) => {
            if (!el) return false;
            const style = window.getComputedStyle(el);
            const rect = el.getBoundingClientRect();
            return style.display !== "none" && style.visibility !== "hidden" && rect.width > 0 && rect.height > 0;
          };
          const select = document.querySelector("#at-db-add-series-select");
          const ok = document.querySelector("#at-db-add-ok-button");
          const cancel = document.querySelector("#at-db-add-cancel-button");
          const title = (document.title || "").trim();
          return (
            !!title &&
            title !== "Updating..." &&
            visible(select) &&
            visible(ok) &&
            visible(cancel)
          );
        }
        """,
        timeout=timeout,
    )


def wait_content_ready(page, selector: str, timeout: int = 30000) -> None:
    page.wait_for_function(
        """
        (sel) => {
          const root = document.querySelector(sel);
          if (!root) return false;
          const visible = (el) => {
            if (!el) return false;
            const style = window.getComputedStyle(el);
            const rect = el.getBoundingClientRect();
            return style.display !== "none" && style.visibility !== "hidden" && rect.width > 0 && rect.height > 0;
          };
          if (root.matches(".js-plotly-plot, .ag-root-wrapper, .mantine-Text-root") && visible(root)) {
            return true;
          }
          const meaningful = root.querySelector(".js-plotly-plot, .ag-root-wrapper, .mantine-Text-root");
          if (meaningful && visible(meaningful)) {
            return true;
          }
          return Array.from(root.children || []).some((child) => visible(child));
        }
        """,
        arg=selector,
        timeout=timeout,
    )


def detect_renderer_mode(page) -> str:
    return str(
        page.evaluate(
            """
            () => {
              const scripts = Array.from(document.querySelectorAll("script[src]"));
              const src = scripts.map((el) => el.getAttribute("src") || "").find((x) => x.includes("dash_renderer")) || "";
              if (src.includes(".dev.js")) return "dev";
              if (src.includes(".min.js")) return "min";
              return "unknown";
            }
            """
        )
    )


def warm_analytics_db(page, base_url: str, db_series: list[str]) -> str:
    analytics_path = "/analyticstool"
    page.goto(base_url + analytics_path, wait_until="domcontentloaded")
    wait_dash_hydrated(page)
    renderer_mode = detect_renderer_mode(page)
    wait_visible(page, "#at-welcome-add-db-btn")
    # The welcome flow can keep a modal overlay mounted during idle states.
    modal_ready = False
    for _ in range(3):
        page.locator("#at-welcome-add-db-btn").click(force=True)
        try:
            wait_analytics_db_modal_ready(page, timeout=5000)
            modal_ready = True
            break
        except Exception:
            wait_dash_hydrated(page, timeout=10000)
    if not modal_ready:
        raise RuntimeError("AnalyticsTool DB import modal did not become ready during harness warmup.")
    page.evaluate(
        """
        (series) => {
          window.dash_clientside.set_props("at-db-add-series-select", { value: series });
        }
        """,
        db_series,
    )
    wait_ready(page, "#at-db-add-ok-button")
    page.locator("#at-db-add-ok-button").click(force=True)
    page.wait_for_selector("#at-modal-ok-button", state="visible", timeout=30000)
    page.locator("#at-modal-ok-button").click(force=True)
    wait_visible(page, "#at-main-app-container")
    wait_ready(page, "#at-periodicity-select")
    return renderer_mode


def wait_analytics_statistics_idle(page, timeout: int = 30000) -> None:
    page.wait_for_function(
        """
        () => {
          const overlay = document.querySelector("#at-loading-statistics");
          const grid = document.querySelector("#at-statistics-grid");
          const title = (document.title || "").trim();
          const visible = (el) => {
            if (!el) return false;
            const style = window.getComputedStyle(el);
            const rect = el.getBoundingClientRect();
            return style.display !== "none" && style.visibility !== "hidden" && rect.width > 0 && rect.height > 0;
          };
          const overlayHidden = !overlay || overlay.getAttribute("data-show") === "false" || !visible(overlay);
          return !!title && title !== "Updating..." && overlayHidden && visible(grid);
        }
        """,
        timeout=timeout,
    )


def wait_analytics_tab_ready(page, active_tab: str = "statistics", timeout: int = 30000) -> None:
    normalized_tab = str(active_tab or "statistics")
    if normalized_tab == "statistics":
        wait_analytics_statistics_idle(page, timeout=timeout)
        return
    selector_map = {
        "returns": "#at-returns-grid",
        "rolling": "#at-rolling-grid",
        "calendar": "#at-calendar-grid",
        "growth": "#at-growth-grid",
        "drawdown": "#at-drawdown-grid",
        "factor_analysis": "#at-factor-analysis-container",
        "regime_analysis": "#at-regime-analysis-container",
        "conditional_returns": "#at-conditional-returns-container",
        "correlogram": "#at-correlogram-container",
    }
    selector = selector_map.get(normalized_tab, "#at-main-app-container")
    wait_visible(page, selector, timeout=timeout)
    page.wait_for_function(
        """
        (sel) => {
          const el = document.querySelector(sel);
          if (!el) return false;
          const style = window.getComputedStyle(el);
          const rect = el.getBoundingClientRect();
          const title = (document.title || "").trim();
          return !!title && title !== "Updating..." && style.display !== "none" && style.visibility !== "hidden" && rect.width > 0 && rect.height > 0;
        }
        """,
        arg=selector,
        timeout=timeout,
    )


def wait_for_analytics_state_ready(page, timeout: int = 30000) -> None:
    deadline = time.time() + max(timeout, 1000) / 1000.0
    while time.time() < deadline:
        if get_persisted_store_value(page, "at-state-ready-store") is True:
            wait_analytics_statistics_idle(page, timeout=timeout)
            return
        time.sleep(0.1)
    raise RuntimeError("Timed out waiting for AnalyticsTool state-ready store.")


def ensure_analytics_selection(page, selected_series: list[str], active_tab: str = "statistics", timeout: int = 30000) -> None:
    target_series = list(selected_series or [])
    normalized_tab = str(active_tab or "statistics")
    set_component_value_if_needed(page, "at-main-tabs", normalized_tab, store_id="at-active-tab-store")
    wait_for_persisted_store_value(page, "at-active-tab-store", normalized_tab, timeout=timeout)
    set_component_props(page, "at-series-select", {"data": target_series})
    wait_for_persisted_store_value(page, "at-series-select-value-store", target_series, timeout=timeout)
    deadline = time.time() + max(timeout, 1000) / 1000.0
    while time.time() < deadline:
        if get_persisted_store_value(page, "at-state-ready-store") is True:
            wait_analytics_tab_ready(page, normalized_tab, timeout=timeout)
            return
        time.sleep(0.1)
    raise RuntimeError("Timed out waiting for AnalyticsTool state-ready store.")


def _analytics_target_selection(db_series: list[str]) -> list[str]:
    resolved = list(db_series or DEFAULT_DB_SERIES)
    if len(resolved) >= 3:
        return resolved[:3]
    if len(resolved) == 2:
        return resolved[:1]
    return resolved


def _analytics_narrow_date_range(page, selected_series: list[str]) -> dict[str, str]:
    raw_store = get_persisted_store_value(page, "dashmat-raw-data-store")
    dataset_key = resolve_dataset_key(raw_store)
    periodicity = get_persisted_store_value(page, "at-periodicity-value-store") or "daily"
    candidates = compute_date_range_candidates(dataset_key, periodicity, tuple(selected_series or ()))
    max_start = candidates.get("max_start")
    max_end = candidates.get("max_end")
    if not max_start or not max_end:
        raise RuntimeError("AnalyticsTool date-range candidates were unavailable during harness setup.")
    if max_start != max_end:
        return {"start": max_start, "end": max_start}
    return {"start": max_start, "end": max_end}


def set_analytics_date_range(page, date_range: dict[str, str], timeout: int = 10000) -> None:
    start_value = date_range.get("start")
    end_value = date_range.get("end")
    page.evaluate(
        """
        ([startValue, endValue]) => {
          window.dash_clientside.set_props("at-start-date-picker", { value: startValue });
          window.dash_clientside.set_props("at-end-date-picker", { value: endValue });
        }
        """,
        [start_value, end_value],
    )
    wait_for_persisted_store_value(page, "at-date-range-store", {"start": start_value, "end": end_value}, timeout=timeout)


def measure_analytics_selection_flow(page, request_tracker: DashUpdateRequestTracker, db_series: list[str], active_tab: str = "statistics") -> dict[str, object]:
    baseline_series = list(db_series or DEFAULT_DB_SERIES)
    target_series = _analytics_target_selection(baseline_series)
    normalized_tab = str(active_tab or "statistics")
    ensure_analytics_selection(page, baseline_series, active_tab=normalized_tab)
    request_tracker.wait_for_settle()
    start = time.perf_counter()
    request_tracker.start_window()
    set_component_props(page, "at-series-select", {"data": target_series})
    wait_for_persisted_store_value(page, "at-series-select-value-store", target_series)
    deadline = time.time() + 30
    while time.time() < deadline:
        if get_persisted_store_value(page, "at-state-ready-store") is True:
            wait_analytics_tab_ready(page, normalized_tab, timeout=30000)
            break
        time.sleep(0.1)
    else:
        raise RuntimeError("Timed out waiting for AnalyticsTool state-ready store during selection flow.")
    request_tracker.wait_for_settle()
    request_tracker.stop_window()
    summary = request_tracker.summary()
    flow_ms = round((time.perf_counter() - start) * 1000)
    ensure_analytics_selection(page, baseline_series, active_tab=normalized_tab)
    request_tracker.wait_for_settle()
    return {"flowMs": flow_ms, "activeTab": normalized_tab, **summary}


def measure_analytics_date_range_flow(page, request_tracker: DashUpdateRequestTracker, db_series: list[str]) -> dict[str, object]:
    baseline_series = list(db_series or DEFAULT_DB_SERIES)
    ensure_analytics_selection(page, baseline_series)
    narrow_range = _analytics_narrow_date_range(page, baseline_series)
    set_analytics_date_range(page, narrow_range)
    wait_for_analytics_state_ready(page)
    request_tracker.wait_for_settle()
    candidates = compute_date_range_candidates(
        resolve_dataset_key(get_persisted_store_value(page, "dashmat-raw-data-store")),
        get_persisted_store_value(page, "at-periodicity-value-store") or "daily",
        tuple(baseline_series),
    )
    expected_range = {
        "start": candidates.get("max_start"),
        "end": candidates.get("max_end"),
    }
    start = time.perf_counter()
    request_tracker.start_window()
    page.locator("#at-maximum-range-button").click(force=True)
    wait_for_persisted_store_value(page, "at-date-range-store", expected_range)
    wait_for_analytics_state_ready(page)
    request_tracker.wait_for_settle()
    request_tracker.stop_window()
    summary = request_tracker.summary()
    flow_ms = round((time.perf_counter() - start) * 1000)
    return {"flowMs": flow_ms, **summary}


def measure(page, cfg: dict[str, str]) -> dict[str, int]:
    start = time.perf_counter()
    page.evaluate("(path) => { window.location.pathname = path; }", cfg["path"])
    page.wait_for_function(
        "(path) => window.location.pathname === path",
        arg=cfg["path"],
        timeout=30000,
    )
    wait_visible(page, cfg["shell"])
    shell_ms = round((time.perf_counter() - start) * 1000)
    wait_ready(page, cfg["ready"])
    ready_ms = round((time.perf_counter() - start) * 1000)
    return {"shellMs": shell_ms, "readyMs": ready_ms}


def set_component_value(page, component_id: str, value) -> None:
    set_component_props(page, component_id, {"value": value})


def set_component_props(page, component_id: str, props: dict) -> None:
    page.evaluate(
        """
        ([componentId, nextProps]) => {
          window.dash_clientside.set_props(componentId, nextProps);
        }
        """,
        [component_id, props],
    )


def wait_hidden_or_absent(page, selector: str, timeout: int = 30000) -> None:
    page.wait_for_function(
        """
        (sel) => {
          const el = document.querySelector(sel);
          if (!el) return true;
          const style = window.getComputedStyle(el);
          const rect = el.getBoundingClientRect();
          return style.display === "none" || style.visibility === "hidden" || rect.width === 0 || rect.height === 0;
        }
        """,
        arg=selector,
        timeout=timeout,
    )


def try_set_component_props(page, component_id: str, props: dict) -> bool:
    return bool(
        page.evaluate(
            """
            ([componentId, nextProps]) => {
              try {
                window.dash_clientside.set_props(componentId, nextProps);
                return true;
              } catch (err) {
                return false;
              }
            }
            """,
            [component_id, props],
        )
    )


def replay_store_data(page, component_id: str) -> bool:
    return bool(
        page.evaluate(
            """
            (componentId) => {
              const storages = [window.sessionStorage, window.localStorage];
              for (const storage of storages) {
                const raw = storage.getItem(componentId);
                if (raw === null) {
                  continue;
                }
                try {
                  window.dash_clientside.set_props(componentId, { data: JSON.parse(raw) });
                  return true;
                } catch (err) {
                  return false;
                }
              }
              return false;
            }
            """,
            component_id,
        )
    )


def fire_component_click(page, component_id: str) -> bool:
    return bool(
        page.evaluate(
            """
            (componentId) => {
              try {
                window.dash_clientside.set_props(componentId, { n_clicks: Date.now() });
                return true;
              } catch (err) {
                return false;
              }
            }
            """,
            component_id,
        )
    )


def wait_plotly_content(page, container_selector: str, timeout: int = 30000) -> None:
    wait_visible(page, container_selector, timeout=timeout)
    page.wait_for_function(
        """
        (sel) => {
          const root = document.querySelector(sel);
          if (!root) return false;
          const plot = root.matches(".js-plotly-plot") ? root : root.querySelector(".js-plotly-plot");
          if (!plot) return false;
          const style = window.getComputedStyle(plot);
          const rect = plot.getBoundingClientRect();
          return style.display !== "none" && style.visibility !== "hidden" && rect.width > 0 && rect.height > 0;
        }
        """,
        arg=container_selector,
        timeout=timeout,
    )


def resolve_portopt_series(db_series: list[str]) -> list[str]:
    preferred = [series for series in db_series if "BCTBill13" not in series]
    selected = preferred[:3]
    if len(selected) >= 2:
        return selected
    return db_series[: min(len(db_series), 3)]


def resolve_regression_series(db_series: list[str]) -> tuple[str, list[str]]:
    preferred = [series for series in db_series if "BCTBill13" not in series]
    selected = preferred[:3] if len(preferred) >= 2 else db_series[: min(len(db_series), 3)]
    if len(selected) < 2:
        raise RuntimeError(f"Need at least 2 series for Regression harness solve, got: {selected}")
    return selected[0], selected[1:]


PORTOPT_RESTORE_TAB_CONFIG = {
    "weight": {
        "content": "#po-weight-chart-content",
        "switch": "po-weight-chart-switch",
        "switch_store": "po-weight-chart-switch-store",
    },
    "frontier": {
        "content": "#po-frontier-chart-container",
        "switch": "po-frontier-chart-switch",
        "switch_store": "po-frontier-chart-switch-store",
    },
    "risk": {
        "content": "#po-risk-chart-container",
        "switch": "po-risk-chart-switch",
        "switch_store": "po-risk-chart-switch-store",
    },
    "attribution": {
        "content": "#po-attribution-chart-container",
        "switch": "po-attribution-chart-switch",
        "switch_store": "po-attribution-chart-switch-store",
    },
}


def normalize_portopt_restore_tab(value: str) -> str:
    normalized = str(value or "weight").strip().lower()
    return normalized if normalized in PORTOPT_RESTORE_TAB_CONFIG else "weight"


def get_persisted_store_value(page, component_id: str):
    return page.evaluate(
        """
        (componentId) => {
          const storages = [window.sessionStorage, window.localStorage];
          for (const storage of storages) {
            const raw = storage.getItem(componentId);
            if (raw === null) {
              continue;
            }
            try {
              return JSON.parse(raw);
            } catch (err) {
              return raw;
            }
          }
          return null;
        }
        """,
        component_id,
    )


def wait_for_persisted_store_value(page, component_id: str, expected, timeout: int = 10000) -> None:
    deadline = time.time() + max(timeout, 1000) / 1000.0
    while time.time() < deadline:
        if get_persisted_store_value(page, component_id) == expected:
            return
        time.sleep(0.1)
    raise RuntimeError(f"Timed out waiting for {component_id} to become {expected!r}")


def set_persisted_store_value(page, component_id: str, value) -> None:
    page.evaluate(
        """
        ([componentId, nextValue]) => {
          window.sessionStorage.setItem(componentId, JSON.stringify(nextValue));
        }
        """,
        [component_id, value],
    )


def set_component_value_if_needed(page, component_id: str, value, *, store_id: str | None = None) -> bool:
    current_value = get_persisted_store_value(page, store_id or component_id)
    if current_value == value:
        return False
    set_component_value(page, component_id, value)
    return True


def seed_portopt_restore_tab(page, restore_tab: str) -> None:
    resolved_restore_tab = normalize_portopt_restore_tab(restore_tab)
    cfg = PORTOPT_RESTORE_TAB_CONFIG[resolved_restore_tab]
    current_active_tab = get_persisted_store_value(page, "po-active-tab-store")
    if current_active_tab != resolved_restore_tab:
        set_persisted_store_value(page, "po-active-tab-store", resolved_restore_tab)
        wait_for_persisted_store_value(page, "po-active-tab-store", resolved_restore_tab, timeout=30000)

    current_switch = get_persisted_store_value(page, cfg["switch_store"])
    if current_switch != "chart":
        set_persisted_store_value(page, cfg["switch_store"], "chart")
        wait_for_persisted_store_value(page, cfg["switch_store"], "chart", timeout=30000)


def seed_regression_restore_tab(page, tab_value: str) -> None:
    cfg = REGRESSION_TAB_CONFIG[tab_value]
    set_component_value(page, "reg-tabs", tab_value)
    set_component_props(page, "reg-active-tab-store", {"data": tab_value})
    wait_for_persisted_store_value(page, "reg-active-tab-store", tab_value, timeout=30000)
    if cfg.get("switch"):
        set_component_value(page, cfg["switch"], cfg["switch_value"])
    wait_content_ready(page, cfg["content"], timeout=60000)


def warm_portopt_results(page, base_url: str, db_series: list[str], restore_tab: str) -> None:
    page.goto(base_url + "/portopt", wait_until="domcontentloaded")
    wait_dash_hydrated(page, timeout=120000)
    wait_visible(page, "#po-main-container", timeout=120000)
    wait_ready(page, "#po-periodicity-select", timeout=120000)
    opt_series = resolve_portopt_series(db_series)
    if len(opt_series) < 2:
        raise RuntimeError(f"Need at least 2 series for PortOpt harness solve, got: {opt_series}")

    modal_ok = page.locator("#po-modal-ok-button")
    deadline = time.perf_counter() + 5
    while time.perf_counter() < deadline and not modal_ok.is_visible():
        page.wait_for_timeout(200)

    modal_seeded = False
    if modal_ok.is_visible():
        if not try_set_component_props(page, "po-temp-series-select", {"data": opt_series}):
            raise RuntimeError("PortOpt harness could not seed po-temp-series-select during modal flow.")
        wait_ready(page, "#po-modal-ok-button")
        modal_ok.click(force=True)
        page.wait_for_selector("#po-modal-ok-button", state="hidden", timeout=30000)
        modal_seeded = True

    seeded = try_set_component_props(page, "po-series-select", {"data": opt_series})
    mirrored = try_set_component_props(page, "po-series-select-value-store", {"data": opt_series})
    if not seeded and mirrored:
        replay_store_data(page, "dashmat-raw-data-store")
    if not modal_seeded and not seeded and not mirrored:
        raise RuntimeError(
            "PortOpt harness could not seed selected series. Expected modal flow or selected-series store."
        )
    wait_ready(page, "#po-run-button")
    page.locator("#po-run-button").click()
    page.wait_for_selector("#po-close-completion-button", state="visible", timeout=120000)
    completion_text = page.locator("#po-completion-text").inner_text(timeout=5000)
    if "created successfully" not in completion_text.lower():
        raise RuntimeError(f"PortOpt harness solve failed: {completion_text}")
    page.locator("#po-close-completion-button").click()
    page.wait_for_selector("#po-close-completion-button", state="hidden", timeout=30000)
    seed_portopt_restore_tab(page, restore_tab)


def select_portopt_tab_and_measure(page, tab_value: str, content_selector: str, switch_id: str | None = None) -> int:
    start = time.perf_counter()
    set_component_value_if_needed(page, "po-vis-tabs", tab_value, store_id="po-active-tab-store")
    if switch_id:
        switch_store_id = None
        for cfg in PORTOPT_RESTORE_TAB_CONFIG.values():
            if cfg["switch"] == switch_id:
                switch_store_id = cfg["switch_store"]
                break
        set_component_value_if_needed(page, switch_id, "chart", store_id=switch_store_id)
    wait_plotly_content(page, content_selector)
    return round((time.perf_counter() - start) * 1000)


REGRESSION_TAB_CONFIG = {
    "anova": {"content": "#reg-anova-content", "label": "ANOVA"},
    "returns": {"content": "#reg-returns-content", "label": "Returns"},
    "rolling": {"content": "#reg-rolling-content", "label": "Rolling Summary", "switch": "reg-rolling-summary-chart-switch", "switch_value": "chart"},
    "growth": {"content": "#reg-growth-content", "label": "Growth of $1", "switch": "reg-growth-chart-switch", "switch_value": "chart"},
    "drawdown": {"content": "#reg-drawdown-content", "label": "Drawdown", "switch": "reg-drawdown-chart-switch", "switch_value": "chart"},
}


def warm_regression_results(page, base_url: str, db_series: list[str]) -> None:
    page.goto(base_url + "/regression", wait_until="domcontentloaded")
    wait_dash_hydrated(page, timeout=120000)
    wait_visible(page, "#reg-main-container", timeout=120000)
    wait_ready(page, "#reg-periodicity-select", timeout=120000)

    dep_var, x_series = resolve_regression_series(db_series)
    series_order = [dep_var] + x_series

    try_set_component_props(page, "reg-page-visited-store", {"data": True})
    try_set_component_props(page, "reg-series-select", {"data": x_series})
    try_set_component_props(page, "reg-series-select-value-store", {"data": x_series})
    try_set_component_props(page, "reg-series-order-store", {"data": series_order})
    try_set_component_props(page, "reg-dependent-var-store", {"data": dep_var})
    try_set_component_props(page, "reg-active-tab-store", {"data": "anova"})
    try_set_component_props(page, "reg-series-selection-modal", {"opened": False})
    try_set_component_props(page, "reg-ui-blocker-store", {"data": False})

    wait_hidden_or_absent(page, "#reg-ui-blocker-overlay", timeout=60000)
    wait_visible(page, "#reg-run-button")
    set_component_props(page, "reg-run-button", {"n_clicks": 1})
    wait_ready(page, "#reg-result-select", timeout=120000)
    wait_for_regression_restore_state(page, timeout=120000)
    wait_content_ready(page, "#reg-anova-content", timeout=120000)


def click_regression_tab(page, tab_value: str) -> None:
    cfg = REGRESSION_TAB_CONFIG[tab_value]
    page.get_by_role("tab", name=cfg["label"], exact=True).click(force=True)


def select_regression_tab_and_measure(page, tab_value: str) -> int:
    cfg = REGRESSION_TAB_CONFIG[tab_value]
    start = time.perf_counter()
    set_component_value(page, "reg-tabs", tab_value)
    wait_for_persisted_store_value(page, "reg-active-tab-store", tab_value)
    if cfg.get("switch"):
        set_component_value(page, cfg["switch"], cfg["switch_value"])
    wait_content_ready(page, cfg["content"])
    return round((time.perf_counter() - start) * 1000)


def measure_regression(page, cfg: dict[str, str], server_log: Path | None = None) -> dict[str, object]:
    start = time.perf_counter()
    nav_log_offset = current_log_offset(server_log)
    page.evaluate("(path) => { window.location.pathname = path; }", cfg["path"])
    page.wait_for_function(
        "(path) => window.location.pathname === path",
        arg=cfg["path"],
        timeout=30000,
    )
    wait_visible(page, cfg["shell"])
    shell_ms = round((time.perf_counter() - start) * 1000)
    wait_ready(page, cfg["ready"])
    ready_ms = round((time.perf_counter() - start) * 1000)
    wait_ready(page, "#reg-result-select")
    restore_state = wait_for_regression_restore_state(page, timeout=120000)
    restore_state_ready_ms = round((time.perf_counter() - start) * 1000)
    timing_validated = wait_for_regression_restored_timing_event(server_log, nav_log_offset, timeout_ms=3000)
    diagnostic_timing_validated = False

    returns_open_ms = select_regression_tab_and_measure(page, "returns")
    rolling_open_ms = select_regression_tab_and_measure(page, "rolling")
    growth_open_ms = select_regression_tab_and_measure(page, "growth")
    drawdown_open_ms = select_regression_tab_and_measure(page, "drawdown")
    set_component_value(page, "reg-tabs", "anova")
    set_component_props(page, "reg-active-tab-store", {"data": "anova"})
    if not timing_validated:
        diagnostic_timing_validated = ensure_regression_timing_event(page, server_log)

    return {
        "shellMs": shell_ms,
        "readyMs": ready_ms,
        "restoreStateReadyMs": restore_state_ready_ms,
        "resultReadyMs": restore_state_ready_ms,
        "timingValidated": timing_validated,
        "timingDiagnosticValidated": diagnostic_timing_validated,
        "restoreState": restore_state,
        "returnsOpenMs": returns_open_ms,
        "rollingOpenMs": rolling_open_ms,
        "growthOpenMs": growth_open_ms,
        "drawdownOpenMs": drawdown_open_ms,
    }


def validate_regression_timing_preflight(metrics: dict[str, object], server_log: Path | None) -> None:
    if not server_log:
        return
    if metrics.get("timingValidated"):
        return
    raise RuntimeError(
        "Regression timing preflight failed: no restored-result regression timing events were found. "
        "Start the app with tools/playwright/start_timed_server.ps1 and pass its STDOUT log path to "
        "--server-log."
    )


def measure_portopt(page, cfg: dict[str, str], restore_tab: str, entry_only: bool) -> dict[str, int]:
    resolved_restore_tab = normalize_portopt_restore_tab(restore_tab)
    restore_cfg = PORTOPT_RESTORE_TAB_CONFIG[resolved_restore_tab]
    start = time.perf_counter()
    page.evaluate("(path) => { window.location.pathname = path; }", cfg["path"])
    page.wait_for_function(
        "(path) => window.location.pathname === path",
        arg=cfg["path"],
        timeout=30000,
    )
    wait_visible(page, cfg["shell"])
    shell_ms = round((time.perf_counter() - start) * 1000)
    wait_ready(page, cfg["ready"])
    ready_ms = round((time.perf_counter() - start) * 1000)

    wait_plotly_content(page, restore_cfg["content"])
    restored_tab_ready_ms = round((time.perf_counter() - start) * 1000)

    if entry_only:
        return {
            "shellMs": shell_ms,
            "readyMs": ready_ms,
            "restoredTabReadyMs": restored_tab_ready_ms,
        }

    set_component_value_if_needed(page, "po-vis-tabs", "weight", store_id="po-active-tab-store")
    set_component_value_if_needed(page, "po-weight-chart-switch", "chart", store_id="po-weight-chart-switch-store")
    wait_plotly_content(page, "#po-weight-chart-content")
    weights_ready_ms = round((time.perf_counter() - start) * 1000)

    frontier_open_ms = select_portopt_tab_and_measure(
        page,
        "frontier",
        "#po-frontier-chart-container",
        "po-frontier-chart-switch",
    )
    risk_open_ms = select_portopt_tab_and_measure(
        page,
        "risk",
        "#po-risk-chart-container",
        "po-risk-chart-switch",
    )
    attribution_open_ms = select_portopt_tab_and_measure(
        page,
        "attribution",
        "#po-attribution-chart-container",
        "po-attribution-chart-switch",
    )
    seed_portopt_restore_tab(page, resolved_restore_tab)

    return {
        "shellMs": shell_ms,
        "readyMs": ready_ms,
        "restoredTabReadyMs": restored_tab_ready_ms,
        "weightsReadyMs": weights_ready_ms,
        "frontierOpenMs": frontier_open_ms,
        "riskOpenMs": risk_open_ms,
        "attributionOpenMs": attribution_open_ms,
    }


def _get_session_storage_json(page, key: str):
    try:
        return page.evaluate(
            """
            (storageKey) => {
              try {
                const raw = window.sessionStorage.getItem(storageKey);
                return raw ? JSON.parse(raw) : null;
              } catch (err) {
                return null;
              }
            }
            """,
            key,
        )
    except Exception:
        return None


def _wait_for_account_list_row(page, list_name: str, timeout: int = 30000) -> dict[str, object]:
    deadline = time.time() + max(timeout, 1000) / 1000.0
    while time.time() < deadline:
        rows = get_persisted_store_value(page, "dashmat-account-list-rows-store") or []
        if isinstance(rows, list):
            for row in rows:
                if str((row or {}).get("ListName") or "") == list_name:
                    return row
        time.sleep(0.1)
    raise RuntimeError(f"Timed out waiting for account list row '{list_name}'.")


def _capture_account_list_session_snapshot(page) -> dict[str, object]:
    return page.evaluate(
        """
        (keys) => {
          const out = {};
          for (const key of keys) {
            const raw = window.sessionStorage.getItem(key);
            if (raw == null) {
              continue;
            }
            try {
              out[key] = JSON.parse(raw);
            } catch (err) {
              out[key] = null;
            }
          }
          return out;
        }
        """,
        ACCOUNT_LIST_CAPTURE_STORE_IDS,
    )


def _save_account_list_fixture(page, list_name: str) -> None:
    session_snapshot = _capture_account_list_session_snapshot(page)
    provenance_store = get_persisted_store_value(page, "dashmat-db-import-provenance-store")
    raw_data_store = get_persisted_store_value(page, "dashmat-raw-data-store")
    userinfo = get_persisted_store_value(page, "userinfo") or {}
    username = str((userinfo or {}).get("username") or "").strip()
    if not username:
        raise RuntimeError("Harness could not resolve account-list username from userinfo store.")

    payload = build_account_list_payload(
        provenance_store,
        session_snapshot,
        raw_data_store,
    )
    ok, message, _saved = save_account_list(
        DB_ENGINE,
        username=username,
        update_by=username,
        list_name=list_name,
        payload=payload,
    )
    if not ok:
        raise RuntimeError(f"Unable to save account-list fixture '{list_name}': {message}")


def _account_list_username_from_page(page) -> str:
    userinfo = get_persisted_store_value(page, "userinfo") or {}
    username = str((userinfo or {}).get("username") or "").strip()
    if not username:
        raise RuntimeError("Harness could not resolve account-list username from userinfo store.")
    return username


def _open_account_list_modal(page, trigger_id: str) -> None:
    wait_dash_hydrated(page, timeout=120000)
    trigger = page.locator(f"#{trigger_id}")
    if "-menu-" in trigger_id:
        try:
            page.get_by_role("button", name="File", exact=True).click(force=True)
            page.wait_for_timeout(250)
        except Exception:
            pass
    try:
        if trigger.count() and trigger.is_visible(timeout=500):
            trigger.click(force=True)
        elif not fire_component_click(page, trigger_id):
            try:
                page.get_by_role("button", name="File", exact=True).click(force=True)
            except Exception:
                page.get_by_role("button", name="Menu").click(force=True)
            page.locator(f"#{trigger_id}").click(force=True)
    except Exception:
        if not fire_component_click(page, trigger_id):
            raise
    wait_visible(page, "#dashmat-account-list-modal")
    wait_ready(page, "#dashmat-account-list-close-button")


def _close_account_list_modal_if_open(page) -> None:
    try:
        if page.locator("#dashmat-account-list-close-button").is_visible(timeout=500):
            page.locator("#dashmat-account-list-close-button").click(force=True)
            page.wait_for_selector("#dashmat-account-list-modal", state="hidden", timeout=10000)
    except Exception:
        pass


def _save_current_account_list(page, trigger_id: str, list_name: str) -> None:
    _open_account_list_modal(page, trigger_id)
    deadline = time.time() + 10
    while time.time() < deadline:
        snapshot = get_persisted_store_value(page, "dashmat-account-list-session-snapshot-store")
        if isinstance(snapshot, dict) and snapshot:
            break
        time.sleep(0.1)
    else:
        raise RuntimeError("Timed out waiting for account-list save snapshot capture.")
    set_component_props(page, "dashmat-account-list-name-input", {"value": list_name})
    wait_ready(page, "#dashmat-account-list-save-button")
    page.locator("#dashmat-account-list-save-button").click(force=True)
    page.wait_for_selector("#dashmat-account-list-modal", state="hidden", timeout=30000)


def _prepare_account_list_load(page, trigger_id: str, list_name: str) -> dict[str, object]:
    username = _account_list_username_from_page(page)
    rows = list_account_lists(DB_ENGINE, username)
    row = next(
        (next_row for next_row in rows if str((next_row or {}).get("ListName") or "") == list_name),
        None,
    )
    if not isinstance(row, dict):
        raise RuntimeError(f"Account list '{list_name}' not found for username '{username}'.")
    set_component_props(page, "dashmat-account-list-modal-mode-store", {"data": "load"})
    set_component_props(page, "dashmat-account-list-rows-store", {"data": rows})
    selected_id = row.get("AccountListID")
    if selected_id is None:
        raise RuntimeError(f"Account list '{list_name}' is missing AccountListID.")
    detail = load_account_list_by_id(DB_ENGINE, selected_id, username)
    if not isinstance(detail, dict):
        raise RuntimeError(f"Timed out resolving selected detail for account list '{list_name}'.")
    set_component_props(page, "dashmat-account-list-selected-id-store", {"data": selected_id})
    set_component_props(page, "dashmat-account-list-selected-detail-store", {"data": detail})
    return detail


def _wait_for_account_list_reload_start(page, timeout: int = 30000) -> dict[str, object] | None:
    deadline = time.time() + max(timeout, 1000) / 1000.0
    while time.time() < deadline:
        payload = _get_session_storage_json(page, "dashmat-account-list-load-timing")
        if isinstance(payload, dict) and payload.get("reloadStartEpochMs"):
            return payload
        time.sleep(0.1)
    return None


def _extract_click_to_ready_from_console(console_lines: list[str], page_path: str) -> int | None:
    for line in reversed(console_lines):
        if "timing name=account_list.click_to_ready" not in line:
            continue
        fields = parse_timing_fields(line)
        if str(fields.get("page", "")).strip() != page_path:
            continue
        for key in ("click_to_reload_start_ms", "click_to_live_apply_commit_ms"):
            if key not in fields:
                continue
            try:
                return int(fields[key])
            except (TypeError, ValueError):
                continue
    return None


def _perturb_session_for_account_list_load(page, page_name: str) -> None:
    if page_name == "portopt":
        set_persisted_store_value(page, "po-active-tab-store", "weight")
        wait_for_persisted_store_value(page, "po-active-tab-store", "weight", timeout=30000)
        set_persisted_store_value(page, "po-weight-chart-switch-store", "chart")
        wait_for_persisted_store_value(page, "po-weight-chart-switch-store", "chart", timeout=30000)
        return
    if page_name == "regression":
        seed_regression_restore_tab(page, "anova")
        return
    raise RuntimeError(f"Unsupported account-list perturbation page: {page_name}")


def _wait_for_portopt_account_list_ready(page, restore_tab: str) -> None:
    resolved_tab = normalize_portopt_restore_tab(restore_tab)
    wait_dash_hydrated(page, timeout=120000)
    wait_visible(page, "#po-main-container", timeout=120000)
    wait_ready(page, "#po-periodicity-select", timeout=120000)
    wait_for_persisted_store_value(page, "po-active-tab-store", resolved_tab, timeout=120000)
    wait_plotly_content(page, PORTOPT_RESTORE_TAB_CONFIG[resolved_tab]["content"], timeout=120000)


def _wait_for_regression_account_list_ready(page, restore_tab: str) -> None:
    wait_dash_hydrated(page, timeout=120000)
    wait_visible(page, "#reg-main-container", timeout=120000)
    wait_ready(page, "#reg-periodicity-select", timeout=120000)
    wait_for_persisted_store_value(page, "reg-active-tab-store", restore_tab, timeout=120000)
    wait_for_regression_restore_state(page, timeout=120000)
    wait_content_ready(page, REGRESSION_TAB_CONFIG[restore_tab]["content"], timeout=120000)


def _measure_account_list_load(
    page,
    request_tracker: DashUpdateRequestTracker,
    *,
    page_name: str,
    page_path: str,
    load_trigger_id: str,
    list_name: str,
    restore_tab: str,
    timing_messages: list[str],
) -> dict[str, object]:
    _perturb_session_for_account_list_load(page, page_name)
    request_tracker.wait_for_settle()
    page.evaluate("() => { window.sessionStorage.removeItem('dashmat-account-list-load-timing'); }")
    timing_start = len(timing_messages)
    _prepare_account_list_load(page, load_trigger_id, list_name)
    start = time.perf_counter()
    request_tracker.start_window()
    click_to_reload_start = None
    triggered_load = False
    reload_start_measure = time.perf_counter()
    try:
        with page.expect_navigation(wait_until="commit", timeout=30000):
            triggered_load = fire_component_click(page, "dashmat-account-list-load-button")
        if triggered_load:
            click_to_reload_start = round((time.perf_counter() - reload_start_measure) * 1000)
    except Exception:
        if not triggered_load:
            triggered_load = fire_component_click(page, "dashmat-account-list-load-button")
    if not triggered_load:
        raise RuntimeError("Harness could not trigger dashmat-account-list-load-button.")
    timing_payload = None if click_to_reload_start is not None else _wait_for_account_list_reload_start(page, timeout=30000)
    if page_name == "portopt":
        _wait_for_portopt_account_list_ready(page, restore_tab)
    elif page_name == "regression":
        _wait_for_regression_account_list_ready(page, restore_tab)
    else:
        raise RuntimeError(f"Unsupported account-list measurement page: {page_name}")
    request_tracker.wait_for_settle()
    request_tracker.stop_window()
    total_click_to_ready = round((time.perf_counter() - start) * 1000)
    if click_to_reload_start is None and isinstance(timing_payload, dict):
        try:
            click_to_reload_start = int(
                int(timing_payload.get("reloadStartEpochMs")) - int(timing_payload.get("clickStartEpochMs"))
            )
        except (TypeError, ValueError):
            click_to_reload_start = None
    if click_to_reload_start is None:
        click_to_reload_start = _extract_click_to_ready_from_console(timing_messages[timing_start:], page_path)
    if click_to_reload_start is None:
        raise RuntimeError(f"Timed out waiting for account-list reload start timing on {page_path}.")
    reload_to_ready = max(total_click_to_ready - click_to_reload_start, 0)
    summary = request_tracker.summary()
    _close_account_list_modal_if_open(page)
    return {
        "page": page_name,
        "restoredTab": restore_tab,
        "clickToReloadStartMs": click_to_reload_start,
        "reloadStartToReadyMs": reload_to_ready,
        "totalClickToReadyMs": total_click_to_ready,
        **summary,
    }


def run_harness(
    base_url: str,
    runs: int,
    label: str,
    db_series: list[str],
    headed: bool,
    restore_tab: str,
    entry_only: bool,
    measure_account_list_load: bool,
    server_log: Path | None,
    selected_pages: list[str],
) -> dict:
    pages = {
        "analytics": {"path": "/analyticstool", "shell": "#at-main-app-container", "ready": "#at-periodicity-select"},
        "portopt": {"path": "/portopt", "shell": "#po-main-container", "ready": "#po-periodicity-select"},
        "regression": {"path": "/regression", "shell": "#reg-main-container", "ready": "#reg-periodicity-select"},
    }
    console_messages: list[dict[str, str]] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not headed)
        page = browser.new_page(viewport={"width": 1440, "height": 960})
        request_tracker = DashUpdateRequestTracker(page)
        timing_messages: list[str] = []

        def on_console(msg) -> None:
            if "timing name=account_list.click_to_ready" in msg.text:
                timing_messages.append(msg.text)
            if len(console_messages) >= 120:
                return
            if msg.type in {"error", "warning"}:
                console_messages.append({"type": msg.type, "text": msg.text})

        def on_page_error(err) -> None:
            if len(console_messages) >= 120:
                return
            console_messages.append({"type": "pageerror", "text": str(err)})

        page.on("console", on_console)
        page.on("pageerror", on_page_error)

        renderer_mode = None
        account_list_fixtures: dict[str, dict[str, str]] = {}
        if "analytics" in selected_pages:
            renderer_mode = warm_analytics_db(page, base_url, db_series)
        elif measure_account_list_load and any(page_name in selected_pages for page_name in ("portopt", "regression")):
            renderer_mode = warm_analytics_db(page, base_url, db_series)
        if "portopt" in selected_pages:
            warm_portopt_results(page, base_url, db_series, restore_tab)
            if measure_account_list_load:
                portopt_list_name = f"codex-phase16-portopt-frontier-{int(time.time())}"
                wait_dash_hydrated(page, timeout=120000)
                wait_visible(page, "#po-main-container", timeout=120000)
                wait_ready(page, "#po-periodicity-select", timeout=120000)
                seed_portopt_restore_tab(page, "frontier")
                _save_account_list_fixture(page, portopt_list_name)
                seed_portopt_restore_tab(page, restore_tab)
                account_list_fixtures["portopt"] = {
                    "triggerId": "po-welcome-load-account-list-btn",
                    "listName": portopt_list_name,
                    "restoreTab": "frontier",
                }
            measure_portopt(page, pages["portopt"], restore_tab, entry_only)
        if "regression" in selected_pages:
            warm_regression_results(page, base_url, db_series)
            if measure_account_list_load:
                regression_list_name = f"codex-phase16-regression-returns-{int(time.time())}"
                wait_dash_hydrated(page, timeout=120000)
                wait_visible(page, "#reg-main-container", timeout=120000)
                wait_ready(page, "#reg-periodicity-select", timeout=120000)
                seed_regression_restore_tab(page, "returns")
                _save_account_list_fixture(page, regression_list_name)
                seed_regression_restore_tab(page, "anova")
                account_list_fixtures["regression"] = {
                    "triggerId": "reg-menu-load-account-list",
                    "listName": regression_list_name,
                    "restoreTab": "returns",
                }
            regression_preflight = measure_regression(page, pages["regression"], server_log)
            if not measure_account_list_load:
                validate_regression_timing_preflight(regression_preflight, server_log)
        timing_start_offset = server_log.stat().st_size if server_log and server_log.exists() else 0

        results = {
            name: {"runs": 0, "shellMs": [], "readyMs": []}
            for name in pages
        }
        results["analytics"].update(
            {
                "selectionFlowRuns": [],
                "dateRangeFlowRuns": [],
                "returnsSelectionFlowRuns": [],
            }
        )
        results["portopt"].update(
            {
                "restoredTab": normalize_portopt_restore_tab(restore_tab),
                "restoredTabReadyMs": [],
            }
        )
        if measure_account_list_load and "portopt" in selected_pages:
            results["portopt"]["accountListLoadRuns"] = []
        if not entry_only:
            results["portopt"].update(
                {
                    "weightsReadyMs": [],
                    "frontierOpenMs": [],
                    "riskOpenMs": [],
                    "attributionOpenMs": [],
                }
            )
        results["regression"].update(
            {
                "restoreStateReadyMs": [],
                "resultReadyMs": [],
                "timingValidated": [],
                "timingDiagnosticValidated": [],
                "returnsOpenMs": [],
                "rollingOpenMs": [],
                "growthOpenMs": [],
                "drawdownOpenMs": [],
            }
        )
        if measure_account_list_load and "regression" in selected_pages:
            results["regression"]["accountListLoadRuns"] = []
        order = [name for name in PAGE_ORDER if name in selected_pages]
        for _ in range(runs):
            for name in order:
                if name == "portopt":
                    metrics = measure_portopt(page, pages[name], restore_tab, entry_only)
                elif name == "regression":
                    metrics = measure_regression(page, pages[name], server_log)
                else:
                    metrics = measure(page, pages[name])
                results[name]["runs"] += 1
                results[name]["shellMs"].append(metrics["shellMs"])
                results[name]["readyMs"].append(metrics["readyMs"])
                if name == "portopt":
                    results[name]["restoredTabReadyMs"].append(metrics["restoredTabReadyMs"])
                    if not entry_only:
                        results[name]["weightsReadyMs"].append(metrics["weightsReadyMs"])
                        results[name]["frontierOpenMs"].append(metrics["frontierOpenMs"])
                        results[name]["riskOpenMs"].append(metrics["riskOpenMs"])
                        results[name]["attributionOpenMs"].append(metrics["attributionOpenMs"])
                elif name == "regression":
                    results[name]["restoreStateReadyMs"].append(metrics["restoreStateReadyMs"])
                    results[name]["resultReadyMs"].append(metrics["resultReadyMs"])
                    results[name]["timingValidated"].append(metrics["timingValidated"])
                    results[name]["timingDiagnosticValidated"].append(metrics["timingDiagnosticValidated"])
                    results[name]["returnsOpenMs"].append(metrics["returnsOpenMs"])
                    results[name]["rollingOpenMs"].append(metrics["rollingOpenMs"])
                    results[name]["growthOpenMs"].append(metrics["growthOpenMs"])
                    results[name]["drawdownOpenMs"].append(metrics["drawdownOpenMs"])
                if measure_account_list_load and name in account_list_fixtures:
                    fixture = account_list_fixtures[name]
                    results[name]["accountListLoadRuns"].append(
                        _measure_account_list_load(
                            page,
                            request_tracker,
                            page_name=name,
                            page_path=pages[name]["path"],
                            load_trigger_id=fixture["triggerId"],
                            list_name=fixture["listName"],
                            restore_tab=fixture["restoreTab"],
                            timing_messages=timing_messages,
                        )
                    )
            if "analytics" in selected_pages:
                measure(page, pages["analytics"])
                results["analytics"]["selectionFlowRuns"].append(
                    measure_analytics_selection_flow(page, request_tracker, db_series)
                )
                results["analytics"]["dateRangeFlowRuns"].append(
                    measure_analytics_date_range_flow(page, request_tracker, db_series)
                )
                results["analytics"]["returnsSelectionFlowRuns"].append(
                    measure_analytics_selection_flow(page, request_tracker, db_series, active_tab="returns")
                )

        for name in selected_pages:
            data = results[name]
            data["shellMedian"] = round(median(data["shellMs"]))
            data["readyMedian"] = round(median(data["readyMs"]))
        if "analytics" in selected_pages:
            results["analytics"]["selectionFlow"] = summarize_dash_update_runs(results["analytics"]["selectionFlowRuns"])
            results["analytics"]["dateRangeFlow"] = summarize_dash_update_runs(results["analytics"]["dateRangeFlowRuns"])
            results["analytics"]["returnsSelectionFlow"] = summarize_dash_update_runs(results["analytics"]["returnsSelectionFlowRuns"])
        if "portopt" in selected_pages:
            results["portopt"]["restoredTabReadyMedian"] = round(median(results["portopt"]["restoredTabReadyMs"]))
            if measure_account_list_load and results["portopt"].get("accountListLoadRuns"):
                results["portopt"]["accountListLoad"] = summarize_account_list_runs(results["portopt"]["accountListLoadRuns"])
            if not entry_only:
                results["portopt"]["weightsReadyMedian"] = round(median(results["portopt"]["weightsReadyMs"]))
                results["portopt"]["frontierOpenMedian"] = round(median(results["portopt"]["frontierOpenMs"]))
                results["portopt"]["riskOpenMedian"] = round(median(results["portopt"]["riskOpenMs"]))
                results["portopt"]["attributionOpenMedian"] = round(median(results["portopt"]["attributionOpenMs"]))
        if "regression" in selected_pages:
            results["regression"]["restoreStateReadyMedian"] = round(median(results["regression"]["restoreStateReadyMs"]))
            results["regression"]["resultReadyMedian"] = round(median(results["regression"]["resultReadyMs"]))
            results["regression"]["timingValidatedCount"] = sum(1 for ok in results["regression"]["timingValidated"] if ok)
            results["regression"]["timingValidatedAll"] = all(results["regression"]["timingValidated"])
            results["regression"]["timingDiagnosticValidatedCount"] = sum(
                1 for ok in results["regression"]["timingDiagnosticValidated"] if ok
            )
            results["regression"]["timingDiagnosticValidatedAll"] = all(results["regression"]["timingDiagnosticValidated"])
            results["regression"]["returnsOpenMedian"] = round(median(results["regression"]["returnsOpenMs"]))
            results["regression"]["rollingOpenMedian"] = round(median(results["regression"]["rollingOpenMs"]))
            results["regression"]["growthOpenMedian"] = round(median(results["regression"]["growthOpenMs"]))
            results["regression"]["drawdownOpenMedian"] = round(median(results["regression"]["drawdownOpenMs"]))
            if measure_account_list_load and results["regression"].get("accountListLoadRuns"):
                results["regression"]["accountListLoad"] = summarize_account_list_runs(results["regression"]["accountListLoadRuns"])

        browser.close()

    warmup_segments: list[str] = []
    if "analytics" in selected_pages:
        warmup_segments.append("analyticstool-aa-db-import+series-selection-confirm")
    if "portopt" in selected_pages:
        warmup_segments.append("portopt-risk-parity-solve")
    if "regression" in selected_pages:
        warmup_segments.append("regression-ols-solve")

    return {
        "ok": True,
        "label": label,
        "baseUrl": base_url,
        "dbSeries": db_series,
        "portoptRestoreTab": normalize_portopt_restore_tab(restore_tab),
        "portoptEntryOnly": bool(entry_only),
        "measureAccountListLoad": bool(measure_account_list_load),
        "selectedPages": selected_pages,
        "runs": runs,
        "warmupFlow": "+".join(warmup_segments),
        "rendererMode": renderer_mode,
        "results": results,
        "consoleMessages": console_messages,
        "timingStartOffset": timing_start_offset,
    }


def main() -> int:
    args = parse_args()
    root = resolve_repo_root(args.repo_root)
    selected_pages = normalize_pages(args.pages)
    out_dir = root / "output" / "playwright"
    fail_dir = out_dir / "failures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fail_dir.mkdir(parents=True, exist_ok=True)
    resolved_git_ref = resolve_git_ref(root, args.git_ref)
    if args.skip_db_build:
        db_rebuilt = False
        db_rebuild_reasons: list[str] = []
    else:
        db_rebuilt, db_rebuild_reasons = ensure_local_seed_databases(root)

    stem = build_artifact_stem(
        label=args.label,
        git_ref=resolved_git_ref,
        base_url=args.base_url,
        timestamp=datetime.now().strftime("%Y-%m-%dT%H-%M-%S"),
    )
    server_log_path = Path(args.server_log).resolve() if args.server_log else None

    try:
        wait_for_app(args.base_url, args.startup_timeout)
        result = run_harness(
            base_url=args.base_url,
            runs=args.runs,
            label=args.label,
            db_series=args.db_series,
            headed=args.headed,
            restore_tab=args.portopt_restore_tab,
            entry_only=args.portopt_entry_only,
            measure_account_list_load=args.measure_account_list_load,
            server_log=server_log_path,
            selected_pages=selected_pages,
        )
    except Exception as exc:
        page_state: dict | None = None
        console_messages: list[dict[str, str]] = []
        try:
            with sync_playwright() as pw:
                browser = pw.chromium.launch(headless=True)
                page = browser.new_page(viewport={"width": 1440, "height": 960})
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
                failure_path = {
                    "analytics": "/analyticstool",
                    "portopt": "/portopt",
                    "regression": "/regression",
                }[selected_pages[0]]
                page.goto(args.base_url + failure_path, wait_until="domcontentloaded", timeout=10000)
                screenshot_path = fail_dir / f"{stem}.png"
                page.screenshot(path=str(screenshot_path), full_page=True)
                page_state = {
                    "url": page.url,
                    "title": page.title(),
                    "screenshotPath": str(screenshot_path),
                }
                browser.close()
        except Exception:
            page_state = page_state or {}
        raw_path = write_failure_artifacts(
            out_dir=out_dir,
            fail_dir=fail_dir,
            stem=stem,
            repo_root=root,
            base_url=args.base_url,
            git_ref=resolved_git_ref,
            label=args.label,
            db_series=args.db_series,
            startup_timeout=args.startup_timeout,
            console_messages=console_messages,
            exc=exc,
            page_state=page_state,
        )
        print(f"RAW_PATH={raw_path}")
        return 1

    timing_summary = parse_timing_log(server_log_path, start_offset=result.get("timingStartOffset", 0))
    timing_summary["copiedPath"] = copy_server_log(server_log_path, out_dir, stem)
    out_path = out_dir / f"{stem}.json"
    payload = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "label": result["label"],
        "gitRef": resolved_git_ref,
        "rendererMode": result["rendererMode"],
        "baseUrl": result["baseUrl"],
        "repoRoot": str(root),
        "dbSeries": result["dbSeries"],
        "portoptRestoreTab": result["portoptRestoreTab"],
        "portoptEntryOnly": result["portoptEntryOnly"],
        "measureAccountListLoad": result["measureAccountListLoad"],
        "selectedPages": result["selectedPages"],
        "warmupFlow": result["warmupFlow"],
        "runs": result["runs"],
        "dbRebuilt": db_rebuilt,
        "dbRebuildReasons": db_rebuild_reasons,
        "analytics": result["results"]["analytics"],
        "portopt": result["results"]["portopt"],
        "regression": result["results"]["regression"],
        "consoleMessages": result["consoleMessages"],
        "timingSummary": timing_summary,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"OUT_PATH={out_path}")
    print("TIMING=" + json.dumps(timing_summary, separators=(",", ":")))
    print("ANALYTICS=" + json.dumps(result["results"]["analytics"], separators=(",", ":")))
    print("PORTOPT=" + json.dumps(result["results"]["portopt"], separators=(",", ":")))
    print("REGRESSION=" + json.dumps(result["results"]["regression"], separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
