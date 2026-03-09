from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
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
    renderer_mode = detect_renderer_mode(page)
    page.locator("#at-welcome-add-db-btn").click()
    page.wait_for_selector("#at-db-add-series-select", state="visible", timeout=30000)
    page.evaluate(
        """
        (series) => {
          window.dash_clientside.set_props("at-db-add-series-select", { value: series });
        }
        """,
        db_series,
    )
    page.wait_for_timeout(300)
    page.locator("#at-db-add-ok-button").click()
    page.wait_for_selector("#at-modal-ok-button", state="visible", timeout=30000)
    page.locator("#at-modal-ok-button").click()
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


def run_harness(base_url: str, runs: int, label: str, db_series: list[str], headed: bool) -> dict:
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
        measure(page, pages["portopt"])
        measure(page, pages["regression"])

        results = {
            name: {"runs": 0, "shellMs": [], "readyMs": []}
            for name in pages
        }
        order = ["analytics", "portopt", "regression"]
        for _ in range(runs):
            for name in order:
                metrics = measure(page, pages[name])
                results[name]["runs"] += 1
                results[name]["shellMs"].append(metrics["shellMs"])
                results[name]["readyMs"].append(metrics["readyMs"])

        for data in results.values():
            data["shellMedian"] = round(median(data["shellMs"]))
            data["readyMedian"] = round(median(data["readyMs"]))

        browser.close()

    return {
        "ok": True,
        "label": label,
        "baseUrl": base_url,
        "dbSeries": db_series,
        "runs": runs,
        "warmupFlow": "analyticstool-aa-db-import+series-selection-confirm",
        "rendererMode": renderer_mode,
        "results": results,
        "consoleMessages": console_messages,
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

    try:
        wait_for_app(args.base_url, args.startup_timeout)
        result = run_harness(
            base_url=args.base_url,
            runs=args.runs,
            label=args.label,
            db_series=args.db_series,
            headed=args.headed,
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

    out_path = out_dir / f"{stem}.json"
    payload = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "label": result["label"],
        "gitRef": resolved_git_ref,
        "rendererMode": result["rendererMode"],
        "baseUrl": result["baseUrl"],
        "repoRoot": str(root),
        "dbSeries": result["dbSeries"],
        "warmupFlow": result["warmupFlow"],
        "runs": result["runs"],
        "dbRebuilt": db_rebuilt,
        "dbRebuildReasons": db_rebuild_reasons,
        "analytics": result["results"]["analytics"],
        "portopt": result["results"]["portopt"],
        "regression": result["results"]["regression"],
        "consoleMessages": result["consoleMessages"],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"OUT_PATH={out_path}")
    print("ANALYTICS=" + json.dumps(result["results"]["analytics"], separators=(",", ":")))
    print("PORTOPT=" + json.dumps(result["results"]["portopt"], separators=(",", ":")))
    print("REGRESSION=" + json.dumps(result["results"]["regression"], separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
