from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import median

from playwright.sync_api import sync_playwright

import warm_switch_harness as warm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class PortfolioImportFixture:
    mode: str
    portfolio: str
    type_value: str
    include_benchmark: bool
    benchmark_type: str | None = None


@dataclass(frozen=True)
class PortoptImportRunSpec:
    spec_index: int
    db_series: tuple[str, ...]
    peer: PortfolioImportFixture
    index: PortfolioImportFixture


def _option_label(option) -> str:
    if isinstance(option, dict):
        return str(option.get("label") or option.get("value") or "").strip()
    return str(option or "").strip()


def _option_value(option) -> str:
    if isinstance(option, dict):
        return str(option.get("value") or option.get("label") or "").strip()
    return str(option or "").strip()


def _sorted_option_entries(options) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for option in options or []:
        value = _option_value(option)
        label = _option_label(option)
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append({"value": value, "label": label or value})
    return sorted(normalized, key=lambda item: (item["label"].casefold(), item["value"].casefold()))


def _build_artifact_stem(label: str, git_ref: str, base_url: str, timestamp: str) -> str:
    git_token = warm.sanitize_token((git_ref or "unknown")[:8], "unknown")
    label_token = warm.sanitize_token(label, "run")
    port = urllib.parse.urlparse(base_url).port
    port_token = f"p{port}" if port else "punknown"
    return f"portopt_import_{timestamp}_{label_token}_{git_token}_{port_token}"


def _get_component_prop(page, component_id: str, prop_name: str):
    return page.evaluate(
        """
        ([componentId, propName]) => {
          const root = (((window.store || {}).getState || (() => ({})))().layout || {}).components;
          if (!root) return null;
          const stack = [root];
          while (stack.length) {
            const node = stack.pop();
            if (!node || typeof node !== "object") continue;
            const props = node.props || {};
            if (props.id === componentId) {
              const value = props[propName];
              return value === undefined ? null : value;
            }
            const children = props.children;
            if (Array.isArray(children)) {
              for (const child of children) stack.push(child);
            } else if (children && typeof children === "object") {
              stack.push(children);
            }
          }
          return null;
        }
        """,
        [component_id, prop_name],
    )


def _wait_for_component_prop(page, component_id: str, prop_name: str, expected, timeout: int = 15000) -> None:
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
        timeout=timeout,
    )


def _wait_for_component_prop_nonempty(page, component_id: str, prop_name: str, timeout: int = 15000) -> None:
    page.wait_for_function(
        """
        ([componentId, propName]) => {
          const root = (((window.store || {}).getState || (() => ({})))().layout || {}).components;
          if (!root) return false;
          const stack = [root];
          while (stack.length) {
            const node = stack.pop();
            if (!node || typeof node !== "object") continue;
            const props = node.props || {};
            if (props.id === componentId) {
              const value = props[propName];
              if (Array.isArray(value)) return value.length > 0;
              if (value && typeof value === "object") return Object.keys(value).length > 0;
              return !!value;
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
        arg=[component_id, prop_name],
        timeout=timeout,
    )


def _wait_for_store_keys(page, component_id: str, expected_keys: list[str], timeout: int = 15000) -> None:
    page.wait_for_function(
        """
        ([componentId, expectedKeys]) => {
          try {
            const raw = window.sessionStorage.getItem(componentId);
            if (raw === null) return false;
            const parsed = JSON.parse(raw);
            if (!parsed || typeof parsed !== "object") return false;
            return expectedKeys.every((key) => typeof parsed[key] === "string" && parsed[key].trim().length > 0);
          } catch (_err) {
            return false;
          }
        }
        """,
        arg=[component_id, expected_keys],
        timeout=timeout,
    )


def _get_grid_rows(page, grid_id: str) -> list[dict]:
    return page.evaluate(
        """
        async (gridId) => {
          if (!window.dash_ag_grid || !window.dash_ag_grid.getApiAsync) {
            return [];
          }
          try {
            const api = await window.dash_ag_grid.getApiAsync(gridId);
            if (!api) return [];
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
        """,
        grid_id,
    )


def _clear_session_and_reload(page, base_url: str) -> dict[str, float]:
    page.evaluate("() => { window.sessionStorage.clear(); }")
    reload_start_ts = time.perf_counter()
    page.goto(base_url + "/portopt", wait_until="domcontentloaded", timeout=30000)
    page.wait_for_function(
        """
        () => {
          const welcome = document.querySelector("#po-welcome-screen");
          if (!welcome) return false;
          const title = (document.title || "").trim();
          if (!title || title === "Updating...") return false;
          const visible = (selector) => {
            const el = document.querySelector(selector);
            if (!el) return false;
            const style = window.getComputedStyle(el);
            if (style.display === "none" || style.visibility === "hidden") return false;
            const rect = el.getBoundingClientRect();
            return rect.width > 0 && rect.height > 0;
          };
          return visible("#po-welcome-screen")
            && visible("#po-welcome-add-db-btn")
            && visible("#po-welcome-add-portfolios-peer-btn");
        }
        """,
        timeout=30000,
    )
    hydrated_ts = time.perf_counter()
    warm.wait_visible(page, "#po-welcome-screen", timeout=30000)
    welcome_visible_ts = time.perf_counter()
    return {
        "reloadStartTs": reload_start_ts,
        "hydratedTs": hydrated_ts,
        "welcomeVisibleTs": welcome_visible_ts,
    }

def _open_db_add_modal(page) -> None:
    warm.fire_component_click(page, "po-welcome-add-db-btn")
    _wait_for_component_prop(page, "po-db-add-modal", "opened", True, timeout=30000)
    _wait_for_component_prop_nonempty(page, "po-db-add-series-select", "data", timeout=30000)
    warm.wait_visible(page, "#po-db-add-ok-button", timeout=30000)


def _open_portfolio_add_modal(page, mode: str) -> None:
    button_map = {
        "peer": "po-welcome-add-portfolios-peer-btn",
        "index": "po-welcome-add-portfolios-index-btn",
    }
    warm.fire_component_click(page, button_map[mode])
    _wait_for_component_prop(page, "po-portfolio-add-modal", "opened", True, timeout=30000)
    _wait_for_component_prop(page, "po-portfolio-add-mode-store", "data", mode, timeout=30000)
    _wait_for_component_prop_nonempty(page, "po-portfolio-add-series-select", "data", timeout=30000)
    _wait_for_component_prop_nonempty(page, "po-portfolio-add-type-select", "data", timeout=30000)
    warm.wait_visible(page, "#po-portfolio-add-row-btn", timeout=30000)


def _discover_db_fixture(page, series_count: int = 4) -> list[str]:
    _open_db_add_modal(page)
    options = _sorted_option_entries(_get_component_prop(page, "po-db-add-series-select", "data"))
    selected_values = [entry["value"] for entry in options[:series_count]]
    page.locator("#po-db-add-cancel-button").click(force=True)
    page.wait_for_selector("#po-db-add-ok-button", state="hidden", timeout=30000)
    if len(selected_values) < series_count:
        raise RuntimeError(
            f"Need at least {series_count} DB add options for PortOpt import harness, got {len(selected_values)}."
        )
    return selected_values


def _discover_portfolio_fixtures(page, mode: str, count: int = 5) -> list[PortfolioImportFixture]:
    _open_portfolio_add_modal(page, mode)
    series_options = _sorted_option_entries(_get_component_prop(page, "po-portfolio-add-series-select", "data"))
    type_options = _sorted_option_entries(_get_component_prop(page, "po-portfolio-add-type-select", "data"))
    if not series_options or not type_options:
        raise RuntimeError(f"PortOpt {mode} modal did not expose the series/type options needed for harness discovery.")

    type_value = type_options[0]["value"]
    fixtures: list[PortfolioImportFixture] = []
    for option in series_options:
        portfolio_value = option["value"]
        warm.set_component_props(page, "po-portfolio-add-series-select", {"value": portfolio_value})
        page.wait_for_timeout(250)
        warm.set_component_props(page, "po-portfolio-add-type-select", {"value": type_value})
        page.wait_for_timeout(150)

        include_benchmark = False
        benchmark_type = None
        warm.set_component_props(page, "po-portfolio-add-include-benchmark", {"checked": False})

        fixtures.append(
            PortfolioImportFixture(
                mode=mode,
                portfolio=portfolio_value,
                type_value=type_value,
                include_benchmark=include_benchmark,
                benchmark_type=benchmark_type,
            )
        )
        if len(fixtures) >= count:
            break

    page.locator("#po-portfolio-add-cancel-button").click(force=True)
    page.wait_for_selector("#po-portfolio-add-ok-button", state="hidden", timeout=30000)
    if len(fixtures) < count:
        raise RuntimeError(f"Need at least {count} deterministic {mode} portfolio fixtures, got {len(fixtures)}.")
    return fixtures


def build_run_specs(page) -> list[PortoptImportRunSpec]:
    db_series = tuple(_discover_db_fixture(page))
    peer_fixtures = _discover_portfolio_fixtures(page, "peer")
    index_fixtures = _discover_portfolio_fixtures(page, "index")
    return [
        PortoptImportRunSpec(
            spec_index=index,
            db_series=db_series,
            peer=peer_fixtures[index],
            index=index_fixtures[index],
        )
        for index in range(min(len(peer_fixtures), len(index_fixtures)))
    ]


def _wait_for_series_selection_modal(page, timeout: int = 30000) -> None:
    warm.wait_visible(page, "#po-modal-ok-button", timeout=timeout)
    warm.wait_ready(page, "#po-modal-ok-button", timeout=timeout)


def _wait_for_portopt_data_ready(
    page,
    expected_series: list[str] | None = None,
    min_series_count: int | None = None,
    expect_db_defaults: list[str] | None = None,
) -> None:
    warm.wait_visible(page, "#po-main-container", timeout=30000)
    warm.wait_hidden_or_absent(page, "#po-ui-blocker-overlay", timeout=30000)
    warm.wait_ready(page, "#po-open-modal-button", timeout=30000)

    deadline = time.time() + 30
    while time.time() < deadline:
        selected = warm.get_persisted_store_value(page, "po-series-select")
        if isinstance(selected, list):
            if expected_series is not None:
                expected_set = set(expected_series)
                if expected_set.issubset({str(value) for value in selected}):
                    break
            elif min_series_count is not None and len(selected) >= min_series_count:
                break
        page.wait_for_timeout(100)
    else:
        if expected_series is not None:
            raise RuntimeError(f"Timed out waiting for po-series-select to include {expected_series!r}.")
        raise RuntimeError(f"Timed out waiting for po-series-select count to reach {min_series_count}.")

    raw_data_store = warm.get_persisted_store_value(page, "dashmat-raw-data-store")
    if not isinstance(raw_data_store, dict) or not raw_data_store.get("dataset_key"):
        raise RuntimeError("PortOpt import harness did not observe a populated dashmat-raw-data-store.")

    original_periodicity = warm.get_persisted_store_value(page, "dashmat-original-periodicity-store")
    if not isinstance(original_periodicity, str) or not original_periodicity.strip():
        raise RuntimeError("PortOpt import harness did not observe a populated dashmat-original-periodicity-store.")

    period_value = _get_component_prop(page, "po-periodicity-select", "value")
    if not isinstance(period_value, str) or not period_value.strip():
        raise RuntimeError("PortOpt import harness did not observe a stable po-periodicity-select value.")

    if expect_db_defaults:
        _wait_for_store_keys(page, "po-cmabench-defaults-store", expect_db_defaults, timeout=30000)


def _wait_for_dash_quiet(
    page,
    request_tracker: warm.DashUpdateRequestTracker,
    *,
    timeout_ms: int = 5000,
    quiet_ms: int = 350,
) -> None:
    """Wait for active and shortly-queued Dash requests to settle.

    The shared request tracker only waits for requests that have already
    started.  PortOpt import phases have tight UI boundaries, so follow-up
    callbacks can start just after the modal-ready condition and leak into the
    next timing window.  A short quiet period makes phase attribution stable.
    """
    deadline = time.perf_counter() + (timeout_ms / 1000)
    quiet_started_at: float | None = None
    quiet_seconds = quiet_ms / 1000

    while time.perf_counter() < deadline:
        if request_tracker.active_requests:
            quiet_started_at = None
            request_tracker.wait_for_settle(timeout_ms=min(timeout_ms, 500))
            continue
        if quiet_started_at is None:
            quiet_started_at = time.perf_counter()
        if time.perf_counter() - quiet_started_at >= quiet_seconds:
            return
        page.wait_for_timeout(50)


def _stage_portfolio_row(page, fixture: PortfolioImportFixture) -> list[dict]:
    warm.set_component_props(page, "po-portfolio-add-series-select", {"value": fixture.portfolio})
    page.wait_for_timeout(200)
    warm.set_component_props(page, "po-portfolio-add-type-select", {"value": fixture.type_value})
    page.wait_for_timeout(150)
    if fixture.include_benchmark:
        warm.set_component_props(page, "po-portfolio-add-include-benchmark", {"checked": True})
        page.wait_for_timeout(150)
        if fixture.benchmark_type:
            warm.set_component_props(page, "po-portfolio-add-benchmark-type-select", {"value": fixture.benchmark_type})
    else:
        warm.set_component_props(page, "po-portfolio-add-include-benchmark", {"checked": False})

    warm.wait_ready(page, "#po-portfolio-add-row-btn", timeout=30000)
    page.locator("#po-portfolio-add-row-btn").click(force=True)

    deadline = time.time() + 15
    while time.time() < deadline:
        rows = _get_grid_rows(page, "po-portfolio-add-grid")
        if isinstance(rows, list) and rows:
            return rows
        page.wait_for_timeout(100)
    raise RuntimeError("PortOpt import harness could not stage a portfolio row.")

def _measure_db_add_flow(page, spec: PortoptImportRunSpec, request_tracker: warm.DashUpdateRequestTracker) -> dict[str, object]:
    _open_db_add_modal(page)
    warm.set_component_props(page, "po-db-add-series-select", {"value": list(spec.db_series)})
    page.wait_for_timeout(200)
    warm.wait_ready(page, "#po-db-add-ok-button", timeout=30000)

    _wait_for_dash_quiet(page, request_tracker, timeout_ms=5000)
    request_tracker.start_window()
    import_start = time.perf_counter()
    page.locator("#po-db-add-ok-button").click(force=True)
    _wait_for_series_selection_modal(page, timeout=60000)
    db_import_to_modal_ms = round((time.perf_counter() - import_start) * 1000)
    _wait_for_dash_quiet(page, request_tracker, timeout_ms=5000)
    request_tracker.stop_window()
    import_window = request_tracker.summary()

    _wait_for_dash_quiet(page, request_tracker, timeout_ms=5000)
    request_tracker.start_window()
    confirm_start = time.perf_counter()
    page.locator("#po-modal-ok-button").click(force=True)
    page.wait_for_selector("#po-modal-ok-button", state="hidden", timeout=30000)
    _wait_for_portopt_data_ready(page, list(spec.db_series), expect_db_defaults=list(spec.db_series))
    db_confirm_to_ready_ms = round((time.perf_counter() - confirm_start) * 1000)
    _wait_for_dash_quiet(page, request_tracker, timeout_ms=5000)
    request_tracker.stop_window()
    confirm_window = request_tracker.summary()

    return {
        "dbImportToSeriesModalMs": db_import_to_modal_ms,
        "dbSeriesConfirmToReadyMs": db_confirm_to_ready_ms,
        "dbImportWindow": import_window,
        "dbConfirmWindow": confirm_window,
        "selectedSeriesAfterDbAdd": list(spec.db_series),
    }


def _measure_portfolio_add_flow(page, fixture: PortfolioImportFixture, current_series: list[str], request_tracker: warm.DashUpdateRequestTracker) -> tuple[dict[str, object], list[str]]:
    trigger_map = {
        "peer": "po-menu-add-portfolios-peer",
        "index": "po-menu-add-portfolios-index",
    }
    warm.fire_component_click(page, trigger_map[fixture.mode])
    _wait_for_component_prop(page, "po-portfolio-add-modal", "opened", True, timeout=30000)
    _stage_portfolio_row(page, fixture)
    warm.wait_ready(page, "#po-portfolio-add-ok-button", timeout=30000)

    _wait_for_dash_quiet(page, request_tracker, timeout_ms=5000)
    request_tracker.start_window()
    import_start = time.perf_counter()
    page.locator("#po-portfolio-add-ok-button").click(force=True)
    _wait_for_series_selection_modal(page, timeout=60000)
    import_to_modal_ms = round((time.perf_counter() - import_start) * 1000)
    _wait_for_dash_quiet(page, request_tracker, timeout_ms=5000)
    request_tracker.stop_window()
    import_window = request_tracker.summary()

    _wait_for_dash_quiet(page, request_tracker, timeout_ms=5000)
    expected_count = len(current_series) + 1
    request_tracker.start_window()
    confirm_start = time.perf_counter()
    page.locator("#po-modal-ok-button").click(force=True)
    page.wait_for_selector("#po-modal-ok-button", state="hidden", timeout=30000)
    _wait_for_portopt_data_ready(page, min_series_count=expected_count)
    confirm_to_ready_ms = round((time.perf_counter() - confirm_start) * 1000)
    _wait_for_dash_quiet(page, request_tracker, timeout_ms=5000)
    request_tracker.stop_window()
    confirm_window = request_tracker.summary()
    selected_after = warm.get_persisted_store_value(page, "po-series-select") or []
    imported_series = [
        str(series)
        for series in selected_after
        if str(series) not in {str(existing) for existing in current_series}
    ]
    expected_series = current_series + [series for series in imported_series if series not in current_series]

    prefix = "peerPortfolio" if fixture.mode == "peer" else "indexPortfolio"
    return (
        {
            f"{prefix}ImportToSeriesModalMs": import_to_modal_ms,
            f"{prefix}SeriesConfirmToReadyMs": confirm_to_ready_ms,
            f"{prefix}ImportWindow": import_window,
            f"{prefix}ConfirmWindow": confirm_window,
            f"{prefix}ImportedSeries": imported_series,
        },
        expected_series,
    )


def measure_run(page, spec: PortoptImportRunSpec, request_tracker: warm.DashUpdateRequestTracker, base_url: str) -> dict[str, object]:
    reset_times = _clear_session_and_reload(page, base_url)
    reset_to_welcome_ms = round((reset_times["welcomeVisibleTs"] - reset_times["reloadStartTs"]) * 1000)
    reset_reload_start_to_hydrated_ms = round((reset_times["hydratedTs"] - reset_times["reloadStartTs"]) * 1000)
    reset_hydrated_to_welcome_visible_ms = round((reset_times["welcomeVisibleTs"] - reset_times["hydratedTs"]) * 1000)

    total_start = time.perf_counter()
    db_metrics = _measure_db_add_flow(page, spec, request_tracker)
    current_series = list(db_metrics["selectedSeriesAfterDbAdd"])

    peer_metrics, current_series = _measure_portfolio_add_flow(page, spec.peer, current_series, request_tracker)
    index_metrics, current_series = _measure_portfolio_add_flow(page, spec.index, current_series, request_tracker)
    total_run_ms = round((time.perf_counter() - total_start) * 1000)

    return {
        "specIndex": spec.spec_index,
        "dbSeries": list(spec.db_series),
        "peerFixture": asdict(spec.peer),
        "indexFixture": asdict(spec.index),
        "resetToWelcomeMs": reset_to_welcome_ms,
        "resetReloadStartToHydratedMs": reset_reload_start_to_hydrated_ms,
        "resetHydratedToWelcomeVisibleMs": reset_hydrated_to_welcome_visible_ms,
        "totalRunMs": total_run_ms,
        **db_metrics,
        **peer_metrics,
        **index_metrics,
    }


def _build_summary(run_results: list[dict[str, object]]) -> dict[str, object]:
    metrics = [
        "resetToWelcomeMs",
        "resetReloadStartToHydratedMs",
        "resetHydratedToWelcomeVisibleMs",
        "dbImportToSeriesModalMs",
        "dbSeriesConfirmToReadyMs",
        "peerPortfolioImportToSeriesModalMs",
        "peerPortfolioSeriesConfirmToReadyMs",
        "indexPortfolioImportToSeriesModalMs",
        "indexPortfolioSeriesConfirmToReadyMs",
        "totalRunMs",
    ]
    summary: dict[str, object] = {"runs": len(run_results)}
    for metric in metrics:
        values = [int(run[metric]) for run in run_results]
        summary[metric] = values
        summary[f"{metric.removesuffix('Ms')}Median"] = round(median(values)) if values else 0

    callback_frequency: Counter[str] = Counter()
    request_counts: list[int] = []
    request_durations: list[int] = []
    request_bytes: list[int] = []
    response_bytes: list[int] = []
    for run in run_results:
        combined_requests = []
        for window_key in (
            "dbImportWindow",
            "dbConfirmWindow",
            "peerPortfolioImportWindow",
            "peerPortfolioConfirmWindow",
            "indexPortfolioImportWindow",
            "indexPortfolioConfirmWindow",
        ):
            window = run.get(window_key, {})
            if not isinstance(window, dict):
                continue
            combined_requests.extend(window.get("dashUpdateRequests", []))
            for request in window.get("dashUpdateRequests", []):
                for output_id in request.get("outputs", []):
                    callback_frequency[str(output_id)] += 1
        request_counts.append(len(combined_requests))
        request_durations.append(sum(int(req.get("durationMs", 0) or 0) for req in combined_requests))
        request_bytes.append(sum(int(req.get("requestBytes", 0) or 0) for req in combined_requests))
        response_bytes.append(sum(int(req.get("responseBytes", 0) or 0) for req in combined_requests))

    summary["dashUpdateRequestCountMedian"] = round(median(request_counts)) if request_counts else 0
    summary["dashUpdateTotalMsMedian"] = round(median(request_durations)) if request_durations else 0
    summary["dashUpdateRequestBytesMedian"] = round(median(request_bytes)) if request_bytes else 0
    summary["dashUpdateResponseBytesMedian"] = round(median(response_bytes)) if response_bytes else 0
    summary["topCallbacksByFrequency"] = [
        {"id": callback_id, "count": count}
        for callback_id, count in callback_frequency.most_common(10)
    ]
    return summary

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PortOpt import-flow harness")
    parser.add_argument("--repo-root", default="")
    parser.add_argument("--base-url", default="http://127.0.0.1:8050")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--label", default="")
    parser.add_argument("--git-ref", default="")
    parser.add_argument("--startup-timeout", type=int, default=30)
    parser.add_argument("--skip-db-build", action="store_true")
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--server-log", default="")
    return parser.parse_args()


def run_harness(args: argparse.Namespace, resolved_git_ref: str) -> dict[str, object]:
    server_log_path = Path(args.server_log).resolve() if args.server_log else None
    console_messages: list[dict[str, str]] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not args.headed)
        page = browser.new_page(viewport={"width": 1440, "height": 960})
        page.set_default_timeout(30000)
        page.set_default_navigation_timeout(30000)
        request_tracker = warm.DashUpdateRequestTracker(page)
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

        warm.wait_for_app(args.base_url, args.startup_timeout)
        timing_start_offset = warm.current_log_offset(server_log_path)

        page.goto(args.base_url + "/portopt", wait_until="domcontentloaded")
        warm.wait_dash_hydrated(page, timeout=30000)
        run_specs = build_run_specs(page)

        measure_run(page, run_specs[0], request_tracker, args.base_url)

        run_results: list[dict[str, object]] = []
        for run_index in range(1, args.runs + 1):
            spec = run_specs[(run_index - 1) % len(run_specs)]
            result = measure_run(page, spec, request_tracker, args.base_url)
            result["run"] = run_index
            run_results.append(result)

        browser.close()

    return {
        "timestamp": datetime.now().astimezone().isoformat(),
        "label": args.label or "portopt-import-harness",
        "gitRef": resolved_git_ref,
        "baseUrl": args.base_url,
        "runSpecs": [
            {
                "specIndex": spec.spec_index,
                "dbSeries": list(spec.db_series),
                "peer": asdict(spec.peer),
                "index": asdict(spec.index),
            }
            for spec in run_specs
        ],
        "runs": run_results,
        "summary": _build_summary(run_results),
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

    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    stem = _build_artifact_stem(args.label or "portopt-import-harness", resolved_git_ref, args.base_url, timestamp)
    out_dir = root / "output" / "playwright"
    fail_dir = out_dir / "failures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fail_dir.mkdir(parents=True, exist_ok=True)
    server_log_path = Path(args.server_log).resolve() if args.server_log else None

    try:
        result = run_harness(args, resolved_git_ref)
    except Exception as exc:
        raw_path = warm.write_failure_artifacts(
            out_dir=out_dir,
            fail_dir=fail_dir,
            stem=stem,
            repo_root=root,
            base_url=args.base_url,
            git_ref=resolved_git_ref,
            label=args.label or "portopt-import-harness",
            db_series=[],
            startup_timeout=args.startup_timeout,
            console_messages=[],
            exc=exc,
            page_state={"url": args.base_url + "/portopt"},
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
        "dbRebuilt": db_rebuilt,
        "dbRebuildReasons": db_rebuild_reasons,
        "runSpecs": result["runSpecs"],
        "runs": result["runs"],
        "summary": result["summary"],
        "consoleMessages": result["consoleMessages"],
        "timingSummary": timing_summary,
    }
    out_path = out_dir / f"{stem}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"OUT_PATH={out_path}")
    print(f"SUMMARY={json.dumps(payload['summary'], separators=(',', ':'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
