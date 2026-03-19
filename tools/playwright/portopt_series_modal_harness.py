from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
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
    parser.add_argument("--db-series", nargs="+", default=warm.DEFAULT_DB_SERIES)
    parser.add_argument("--server-log", default="")
    return parser.parse_args()


def build_artifact_stem(label: str, git_ref: str, base_url: str, timestamp: str) -> str:
    git_token = warm.sanitize_token((git_ref or "unknown")[:8], "unknown")
    label_token = warm.sanitize_token(label, "run")
    port = urllib.parse.urlparse(base_url).port
    port_token = f"p{port}" if port else "punknown"
    return f"portopt_series_modal_{timestamp}_{label_token}_{git_token}_{port_token}"


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
    warm.try_set_component_props(page, "po-ui-blocker-store", {"data": False})
    warm.wait_hidden_or_absent(page, "#po-ui-blocker-overlay", timeout=30000)
    page.wait_for_selector("#po-modal-ok-button", state="hidden", timeout=30000)


def seed_portopt_page(page, base_url: str, opt_series: list[str]) -> None:
    raw_data_payload, raw_meta_payload = build_synthetic_raw_dataset(opt_series)
    page.goto(base_url + "/portopt", wait_until="domcontentloaded")
    warm.wait_dash_hydrated(page, timeout=30000)
    warm.try_set_component_props(page, "po-page-visited-store", {"data": True})
    warm.try_set_component_props(page, "dashmat-raw-data-store", {"data": raw_data_payload})
    warm.try_set_component_props(page, "dashmat-raw-data-meta-store", {"data": raw_meta_payload})
    warm.try_set_component_props(page, "po-series-select", {"data": opt_series})
    warm.try_set_component_props(page, "po-series-select-value-store", {"data": opt_series})
    warm.try_set_component_props(page, "po-series-order-store", {"data": opt_series})
    warm.wait_visible(page, "#po-main-container", timeout=30000)
    warm.wait_ready(page, "#po-open-modal-button", timeout=30000)
    warm.wait_hidden_or_absent(page, "#po-ui-blocker-overlay", timeout=30000)

    page.locator("#po-open-modal-button").click()
    wait_modal_grid_ready(page, expected_rows=len(opt_series), timeout=30000)
    page.locator("#po-modal-cancel-button").click()
    page.wait_for_selector("#po-modal-ok-button", state="hidden", timeout=30000)


def measure_modal_run(page, opt_series: list[str]) -> dict[str, object]:
    reset_modal_seed_state(page, opt_series)

    open_start = time.perf_counter()
    page.locator("#po-open-modal-button").click()
    wait_modal_grid_ready(page, expected_rows=len(opt_series), timeout=30000)
    modal_open_ms = round((time.perf_counter() - open_start) * 1000)

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

    first_series = set_first_row_selected(page, True)
    if not first_series:
        raise RuntimeError("PortOpt modal harness could not reseed a selected row before OK.")

    ok_start = time.perf_counter()
    page.locator("#po-modal-ok-button").click()
    page.wait_for_selector("#po-modal-ok-button", state="hidden", timeout=30000)
    warm.wait_hidden_or_absent(page, "#po-ui-blocker-overlay", timeout=30000)
    ok_confirm_ms = round((time.perf_counter() - ok_start) * 1000)

    return {
        "modalOpenMs": modal_open_ms,
        "selectAllMs": select_all_ms,
        "unselectAllMs": unselect_all_ms,
        "okConfirmMs": ok_confirm_ms,
        "selectedAfterOk": [first_series],
        "gridRowCount": len(grid_rows(page)),
    }


def run_harness(args: argparse.Namespace, resolved_git_ref: str) -> dict[str, object]:
    server_log_path = Path(args.server_log).resolve() if args.server_log else None
    console_messages: list[dict[str, str]] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not args.headed)
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

        opt_series = warm.resolve_portopt_series(args.db_series)
        seed_portopt_page(page, args.base_url, opt_series)
        timing_start_offset = warm.current_log_offset(server_log_path)
        warm.wait_hidden_or_absent(page, "#po-ui-blocker-overlay", timeout=30000)

        run_results = []
        for run_index in range(1, args.runs + 1):
            result = measure_modal_run(page, opt_series)
            result["run"] = run_index
            run_results.append(result)

        browser.close()

    modal_open_values = [run["modalOpenMs"] for run in run_results]
    select_all_values = [run["selectAllMs"] for run in run_results]
    unselect_all_values = [run["unselectAllMs"] for run in run_results]
    ok_confirm_values = [run["okConfirmMs"] for run in run_results]

    return {
        "timestamp": datetime.now().astimezone().isoformat(),
        "label": args.label or "portopt-series-modal",
        "gitRef": resolved_git_ref,
        "baseUrl": args.base_url,
        "dbSeries": args.db_series,
        "selectedSeries": opt_series,
        "runs": run_results,
        "summary": {
            "modalOpenMedian": round(median(modal_open_values)),
            "selectAllMedian": round(median(select_all_values)),
            "unselectAllMedian": round(median(unselect_all_values)),
            "okConfirmMedian": round(median(ok_confirm_values)),
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
    stem = build_artifact_stem(args.label or "portopt-series-modal", resolved_git_ref, args.base_url, timestamp)
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
