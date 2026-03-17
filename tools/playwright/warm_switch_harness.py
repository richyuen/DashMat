from __future__ import annotations

import argparse
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


DEFAULT_DB_SERIES = [
    "SPX_TRIndex",
    "R2000_TRIndex",
    "EAFE_TRIndex",
    "BCTBill13_TRIndex",
]

TIMING_EVENT_NAMES = (
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
)


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
    parser.add_argument("--server-log", default="")
    return parser.parse_args()


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
            "If the app was launched with `conda run`, prefer `conda run --no-capture-output -n dashmat "
            "python -u ...` so stdout reaches the log file while the server is still running."
        )
    return summary


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


def set_component_value_if_needed(page, component_id: str, value, *, store_id: str | None = None) -> bool:
    current_value = get_persisted_store_value(page, store_id or component_id)
    if current_value == value:
        return False
    set_component_value(page, component_id, value)
    return True


def seed_portopt_restore_tab(page, restore_tab: str) -> None:
    resolved_restore_tab = normalize_portopt_restore_tab(restore_tab)
    cfg = PORTOPT_RESTORE_TAB_CONFIG[resolved_restore_tab]
    set_component_value_if_needed(page, "po-vis-tabs", resolved_restore_tab, store_id="po-active-tab-store")
    set_component_value_if_needed(page, cfg["switch"], "chart", store_id=cfg["switch_store"])
    wait_plotly_content(page, cfg["content"], timeout=60000)


def warm_portopt_results(page, base_url: str, db_series: list[str], restore_tab: str) -> None:
    page.goto(base_url + "/portopt", wait_until="domcontentloaded")
    wait_visible(page, "#po-main-container")
    wait_ready(page, "#po-periodicity-select")
    wait_dash_hydrated(page)
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


def run_harness(
    base_url: str,
    runs: int,
    label: str,
    db_series: list[str],
    headed: bool,
    restore_tab: str,
    entry_only: bool,
    server_log: Path | None,
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

        def on_console(msg) -> None:
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

        renderer_mode = warm_analytics_db(page, base_url, db_series)
        warm_portopt_results(page, base_url, db_series, restore_tab)
        measure(page, pages["portopt"])
        measure(page, pages["regression"])
        timing_start_offset = server_log.stat().st_size if server_log and server_log.exists() else 0

        results = {
            name: {"runs": 0, "shellMs": [], "readyMs": []}
            for name in pages
        }
        results["portopt"].update(
            {
                "restoredTab": normalize_portopt_restore_tab(restore_tab),
                "restoredTabReadyMs": [],
            }
        )
        if not entry_only:
            results["portopt"].update(
                {
                    "weightsReadyMs": [],
                    "frontierOpenMs": [],
                    "riskOpenMs": [],
                    "attributionOpenMs": [],
                }
            )
        order = ["analytics", "portopt", "regression"]
        for _ in range(runs):
            for name in order:
                metrics = measure_portopt(page, pages[name], restore_tab, entry_only) if name == "portopt" else measure(page, pages[name])
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

        for data in results.values():
            data["shellMedian"] = round(median(data["shellMs"]))
            data["readyMedian"] = round(median(data["readyMs"]))
        results["portopt"]["restoredTabReadyMedian"] = round(median(results["portopt"]["restoredTabReadyMs"]))
        if not entry_only:
            results["portopt"]["weightsReadyMedian"] = round(median(results["portopt"]["weightsReadyMs"]))
            results["portopt"]["frontierOpenMedian"] = round(median(results["portopt"]["frontierOpenMs"]))
            results["portopt"]["riskOpenMedian"] = round(median(results["portopt"]["riskOpenMs"]))
            results["portopt"]["attributionOpenMedian"] = round(median(results["portopt"]["attributionOpenMs"]))

        browser.close()

    return {
        "ok": True,
        "label": label,
        "baseUrl": base_url,
        "dbSeries": db_series,
        "portoptRestoreTab": normalize_portopt_restore_tab(restore_tab),
        "portoptEntryOnly": bool(entry_only),
        "runs": runs,
        "warmupFlow": "analyticstool-aa-db-import+series-selection-confirm+portopt-risk-parity-solve",
        "rendererMode": renderer_mode,
        "results": results,
        "consoleMessages": console_messages,
        "timingStartOffset": timing_start_offset,
    }


def main() -> int:
    args = parse_args()
    root = resolve_repo_root(args.repo_root)
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
            server_log=server_log_path,
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
                page.goto(args.base_url + "/analyticstool", wait_until="domcontentloaded", timeout=10000)
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
