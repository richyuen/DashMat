from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from playwright.sync_api import sync_playwright

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.playwright.result_flow_harness import (  # noqa: E402
    build_seed_payloads,
    collect_artifact_store_metrics,
    collect_browser_storage_metrics,
    ensure_modal_hidden,
    force_modal_closed,
    make_portopt_storage_seed,
    make_regression_storage_seed,
    navigate_path,
    push_live_store_seed,
    seed_session_storage,
    set_component_props,
    set_component_value,
    wait_content_ready,
    wait_for_app,
    wait_ready,
    wait_visible,
)


WORKFLOW_CHOICES = ("analyticstool", "portopt", "regression", "combined")
MODE_CHOICES = ("same_machine_restore", "portable_restore", "tampered_restore", "repeat_cycle")
TAMPER_CHOICES = ("missing_required_artifact", "invalid_version", "malformed_json", "missing_workspace_session")
AT_SELECTED_SERIES = ["SPX_TRIndex", "R2000_TRIndex", "EAFE_TRIndex"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="")
    parser.add_argument("--base-url", default="http://127.0.0.1:8050")
    parser.add_argument("--port", type=int, default=8066)
    parser.add_argument("--launch-app", action="store_true")
    parser.add_argument("--artifact-root", default="")
    parser.add_argument("--startup-timeout", type=int, default=30)
    parser.add_argument("--workflow", choices=WORKFLOW_CHOICES, default="combined")
    parser.add_argument("--mode", choices=MODE_CHOICES, default="same_machine_restore")
    parser.add_argument("--tamper-kind", choices=TAMPER_CHOICES, default="missing_required_artifact")
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--portable-every", type=int, default=5)
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--label", default="")
    parser.add_argument("--git-ref", default="")
    return parser.parse_args()


def resolve_repo_root(repo_root: str) -> Path:
    return Path(repo_root).resolve() if repo_root else REPO_ROOT


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


def sanitize_token(value: str, fallback: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "-" for ch in str(value or "").strip().lower())
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    return cleaned or fallback


def build_artifact_stem(label: str, workflow: str, mode: str, git_ref: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    return (
        f"save_load_session_{sanitize_token(workflow, 'workflow')}_"
        f"{sanitize_token(mode, 'mode')}_{ts}_"
        f"{sanitize_token(label, 'run')}_{sanitize_token((git_ref or 'unknown')[:8], 'unknown')}"
    )


def resolve_artifact_root(repo_root: Path, artifact_root: str) -> Path:
    if artifact_root:
        path = Path(artifact_root)
        return path if path.is_absolute() else (repo_root / path).resolve()
    env_root = os.environ.get("DASHMAT_ARTIFACT_ROOT")
    if env_root:
        path = Path(env_root)
        return path if path.is_absolute() else (repo_root / path).resolve()
    return (repo_root / ".cache" / "dashmat_artifacts").resolve()


def launch_app_process(repo_root: Path, port: int, artifact_root: Path, log_path: Path) -> tuple[subprocess.Popen, Any]:
    env = os.environ.copy()
    env["DASHMAT_ARTIFACT_ROOT"] = str(artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    log_handle = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-c", f"import app; app.app.run(port={port})"],
        cwd=repo_root,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    return proc, log_handle


def build_analyticstool_storage_seed(seed_payload: dict[str, Any]) -> dict[str, Any]:
    dataset = seed_payload["dataset_meta"]
    return {
        "dashmat-raw-data-store": seed_payload["raw_json"],
        "dashmat-raw-data-meta-store": seed_payload["raw_meta"],
        "dashmat-original-periodicity-store": "daily",
        "dashmat-pending-new-series-store": {},
        "dashmat-session-id-store": seed_payload["session_id"],
        "at-page-visited-store": True,
        "at-first-load-store": True,
        "at-periodicity-value-store": "daily_trading",
        "at-series-select-value-store": list(AT_SELECTED_SERIES),
        "at-series-select": list(AT_SELECTED_SERIES),
        "at-series-order-store": list(AT_SELECTED_SERIES),
        "at-benchmark-assignments-store": {},
        "at-long-short-store": {},
        "at-returns-type-value-store": "total",
        "at-active-tab-store": "statistics",
        "at-rolling-window-store": "1y",
        "at-rolling-metric-store": "total_return",
        "at-rolling-return-type-store": "annualized",
        "at-rolling-chart-switch-store": "chart",
        "at-drawdown-chart-switch-store": "chart",
        "at-growth-chart-switch-store": "chart",
        "at-factor-mode-store": "box",
        "at-factor-quantiles-store": 5,
        "at-factor-transform-store": "raw",
        "at-monthly-view-store": "annual",
        "at-monthly-series-store": None,
        "at-vol-scaler-value-store": 0,
        "at-vol-scaling-assignments-store": {},
        "at-date-range-store": {"start": dataset["start"], "end": dataset["end"]},
        "at-state-ready-store": True,
    }


def build_storage_seed(workflow: str, seed_payload: dict[str, Any]) -> dict[str, Any]:
    if workflow == "analyticstool":
        return build_analyticstool_storage_seed(seed_payload)
    if workflow == "portopt":
        return make_portopt_storage_seed(seed_payload, include_results=True)
    if workflow == "regression":
        return make_regression_storage_seed(seed_payload, include_results=True)
    if workflow == "combined":
        payload = build_analyticstool_storage_seed(seed_payload)
        payload.update(make_portopt_storage_seed(seed_payload, include_results=True))
        payload.update(make_regression_storage_seed(seed_payload, include_results=True))
        return payload
    raise ValueError(f"Unsupported workflow: {workflow}")


def bundle_expectations_for_workflow(workflow: str) -> dict[str, Any]:
    if workflow == "analyticstool":
        return {"min_artifacts": 0, "required_any_groups": []}
    if workflow == "portopt":
        return {"min_artifacts": 1, "required_any_groups": [["po_portfolio_returns", "po_returns_series"]]}
    if workflow == "regression":
        return {"min_artifacts": 2, "required_any_groups": [["reg_predicted_series"], ["reg_residuals_series"]]}
    if workflow == "combined":
        return {
            "min_artifacts": 3,
            "required_any_groups": [
                ["po_portfolio_returns", "po_returns_series"],
                ["reg_predicted_series"],
                ["reg_residuals_series"],
            ],
        }
    raise ValueError(f"Unsupported workflow: {workflow}")


def summarize_cycles(cycle_records: list[dict[str, Any]]) -> dict[str, Any]:
    durations = [int(record.get("durationMs", 0)) for record in cycle_records]
    bundle_sizes = [int(record.get("bundleBytes", 0)) for record in cycle_records]
    artifact_counts = [int(record.get("bundleArtifactCount", 0)) for record in cycle_records]
    failures = [record for record in cycle_records if not record.get("ok")]
    summary = {
        "cycles": len(cycle_records),
        "passed": len(cycle_records) - len(failures),
        "failed": len(failures),
        "failureModes": {},
    }
    if durations:
        ordered = sorted(durations)
        summary["durationMedianMs"] = ordered[len(ordered) // 2]
        summary["durationP95Ms"] = ordered[min(len(ordered) - 1, max(0, round(len(ordered) * 0.95) - 1))]
    if bundle_sizes:
        summary["bundleBytesMedian"] = sorted(bundle_sizes)[len(bundle_sizes) // 2]
        summary["bundleBytesMax"] = max(bundle_sizes)
    if artifact_counts:
        summary["bundleArtifactCountMedian"] = sorted(artifact_counts)[len(artifact_counts) // 2]
        summary["bundleArtifactCountMax"] = max(artifact_counts)
    for record in failures:
        key = str(record.get("failureMode") or "unknown")
        summary["failureModes"][key] = int(summary["failureModes"].get(key, 0)) + 1
    return summary


def tamper_bundle_for_mode(bundle: dict[str, Any], tamper_kind: str, workflow: str) -> dict[str, Any] | str:
    tampered = json.loads(json.dumps(bundle))
    if tamper_kind == "invalid_version":
        tampered["version"] = -1
        return tampered
    if tamper_kind == "missing_workspace_session":
        tampered.pop("workspace_session", None)
        return tampered
    if tamper_kind == "malformed_json":
        return '{"version": 2, "workspace_session": '
    if tamper_kind == "missing_required_artifact":
        artifacts = tampered.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            return tampered
        if workflow == "portopt":
            tampered["artifacts"] = artifacts[1:]
            return tampered
        if workflow == "regression":
            filtered = [record for record in artifacts if record.get("artifact_type") != "reg_predicted_series"]
            tampered["artifacts"] = filtered if filtered else artifacts[:-1]
            return tampered
        if workflow == "combined":
            filtered = [
                record for record in artifacts
                if record.get("artifact_type") not in {"po_portfolio_returns", "po_returns_series"}
            ]
            tampered["artifacts"] = filtered if filtered else artifacts[:-1]
            return tampered
        raise ValueError("missing_required_artifact tamper is only valid for artifact-backed workflows.")
    raise ValueError(f"Unsupported tamper kind: {tamper_kind}")


def validate_bundle_structure(bundle: dict[str, Any], workflow: str) -> dict[str, Any]:
    if not isinstance(bundle, dict):
        raise RuntimeError("Downloaded bundle is not a JSON object.")
    if "version" not in bundle:
        raise RuntimeError("Bundle missing version.")
    workspace_session = bundle.get("workspace_session")
    if not isinstance(workspace_session, dict):
        raise RuntimeError("Bundle missing workspace_session.")
    if not isinstance(bundle.get("artifact_refs"), list):
        raise RuntimeError("Bundle missing artifact_refs.")
    if not isinstance(bundle.get("artifacts"), list):
        raise RuntimeError("Bundle missing artifacts.")
    if "dashmat-saved-series-cache-store" in workspace_session:
        raise RuntimeError("Bundle should not export dashmat-saved-series-cache-store as runtime workspace state.")
    if "dashmat-raw-data-artifact-store" in workspace_session:
        raise RuntimeError("Bundle should not export dashmat-raw-data-artifact-store as runtime workspace state.")
    expectations = bundle_expectations_for_workflow(workflow)
    artifact_types = [str(record.get("artifact_type")) for record in bundle.get("artifacts", []) if isinstance(record, dict)]
    if len(artifact_types) < int(expectations["min_artifacts"]):
        raise RuntimeError(f"Expected at least {expectations['min_artifacts']} artifacts, got {len(artifact_types)}.")
    for group in expectations["required_any_groups"]:
        if not any(artifact_type in artifact_types for artifact_type in group):
            raise RuntimeError(f"Bundle missing required artifact types for {workflow}: {group}")
    return {
        "artifactCount": len(artifact_types),
        "artifactTypes": artifact_types,
        "warningCount": len(bundle.get("export_warnings") or []),
        "workspaceKeyCount": len(workspace_session),
    }


def clear_browser_session(page) -> None:
    page.goto("about:blank")
    page.evaluate(
        """
        () => {
          try { window.sessionStorage.clear(); } catch (_) {}
          try { window.localStorage.clear(); } catch (_) {}
        }
        """
    )


def collect_artifact_root_metrics(artifact_root: Path) -> dict[str, Any]:
    if not artifact_root.exists():
        return {"exists": False, "path": str(artifact_root), "fileCount": 0, "totalBytes": 0}
    file_count = 0
    total_bytes = 0
    for path in artifact_root.rglob("*"):
        if not path.is_file():
            continue
        file_count += 1
        total_bytes += path.stat().st_size
    return {"exists": True, "path": str(artifact_root), "fileCount": file_count, "totalBytes": total_bytes}


def delete_artifacts(artifact_root: Path) -> dict[str, Any]:
    before = collect_artifact_root_metrics(artifact_root)
    shutil.rmtree(artifact_root, ignore_errors=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    after = collect_artifact_root_metrics(artifact_root)
    return {"before": before, "after": after}


def reset_artifacts_for_cycle(artifact_root: Path) -> None:
    shutil.rmtree(artifact_root, ignore_errors=True)
    artifact_root.mkdir(parents=True, exist_ok=True)


def write_bundle_payload(path: Path, payload: dict[str, Any] | str) -> Path:
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def live_seed_for_workflow(storage_seed: dict[str, Any], workflow: str) -> dict[str, Any]:
    prefixes = ["dashmat-"]
    if workflow == "analyticstool":
        prefixes.append("at-")
    elif workflow == "portopt":
        prefixes.append("po-")
    elif workflow in {"regression", "combined"}:
        prefixes.append("reg-")
    return {
        key: value
        for key, value in storage_seed.items()
        if any(str(key).startswith(prefix) for prefix in prefixes)
    }


def configure_analyticstool_workspace(page, storage_seed: dict[str, Any]) -> None:
    push_live_store_seed(page, live_seed_for_workflow(storage_seed, "analyticstool"))
    try:
        ensure_modal_hidden(page, "#at-series-selection-modal", timeout=1000)
    except Exception:
        force_modal_closed(page, "at-series-selection-modal", "#at-series-selection-modal")
    set_component_value(page, "at-periodicity-select", "daily_trading")
    set_component_value(page, "at-main-tabs", "statistics")
    set_component_props(page, "at-state-ready-store", {"data": True})
    wait_content_ready(page, "#at-statistics-grid", timeout=60000)


def configure_portopt_workspace(page, storage_seed: dict[str, Any]) -> None:
    push_live_store_seed(page, live_seed_for_workflow(storage_seed, "portopt"))
    try:
        ensure_modal_hidden(page, "#po-series-selection-modal", timeout=1000)
    except Exception:
        force_modal_closed(page, "po-series-selection-modal", "#po-series-selection-modal")
    set_component_value(page, "po-vis-tabs", "returns")
    wait_content_ready(page, "#po-returns-grid-content", timeout=60000)


def configure_regression_workspace(page, storage_seed: dict[str, Any]) -> None:
    push_live_store_seed(page, live_seed_for_workflow(storage_seed, "regression"))
    try:
        ensure_modal_hidden(page, "#reg-series-selection-modal", timeout=1000)
    except Exception:
        force_modal_closed(page, "reg-series-selection-modal", "#reg-series-selection-modal")
    set_component_value(page, "reg-tabs", "anova")
    wait_content_ready(page, "#reg-anova-content", timeout=60000)


def load_workflow_page(page, base_url: str, workflow: str, storage_seed: dict[str, Any]) -> None:
    target_path = {
        "analyticstool": "/analyticstool",
        "portopt": "/portopt",
        "regression": "/regression",
        "combined": "/regression",
    }[workflow]
    page.goto(base_url, wait_until="domcontentloaded")
    seed_session_storage(page, storage_seed)
    page.goto(base_url + target_path, wait_until="domcontentloaded")
    shell_selector = {
        "analyticstool": "#at-main-app-container",
        "portopt": "#po-main-container",
        "regression": "#reg-main-container",
        "combined": "#reg-main-container",
    }[workflow]
    ready_selector = {
        "analyticstool": "#at-periodicity-select",
        "portopt": "#po-periodicity-select",
        "regression": "#reg-periodicity-select",
        "combined": "#reg-periodicity-select",
    }[workflow]
    wait_visible(page, shell_selector, timeout=60000)
    wait_ready(page, ready_selector, timeout=60000)
    if workflow == "analyticstool":
        configure_analyticstool_workspace(page, storage_seed)
    elif workflow == "portopt":
        configure_portopt_workspace(page, storage_seed)
    elif workflow == "regression":
        configure_regression_workspace(page, storage_seed)
    else:
        configure_regression_workspace(page, storage_seed)


def assert_analyticstool_restored(page, *, lightweight: bool = False) -> dict[str, Any]:
    wait_visible(page, "#at-main-app-container", timeout=60000)
    wait_ready(page, "#at-periodicity-select", timeout=60000)
    set_component_value(page, "at-main-tabs", "statistics")
    wait_content_ready(page, "#at-statistics-grid", timeout=60000)
    return {"page": "analyticstool"}


def assert_portopt_restored(page, *, lightweight: bool = False) -> dict[str, Any]:
    wait_visible(page, "#po-main-container", timeout=60000)
    wait_ready(page, "#po-periodicity-select", timeout=60000)
    metrics = {}
    set_component_value(page, "po-vis-tabs", "returns")
    wait_content_ready(page, "#po-returns-grid-content", timeout=60000)
    metrics["returnsReady"] = True
    if lightweight:
        return metrics
    set_component_value(page, "po-vis-tabs", "statistics")
    wait_content_ready(page, "#po-statistics-grid-content", timeout=60000)
    metrics["statisticsReady"] = True
    set_component_value(page, "po-vis-tabs", "growth")
    wait_content_ready(page, "#po-growth-chart-container", timeout=60000)
    metrics["growthReady"] = True
    set_component_value(page, "po-vis-tabs", "frontier")
    set_component_value(page, "po-frontier-chart-switch", "chart")
    wait_content_ready(page, "#po-frontier-chart-container", timeout=60000)
    metrics["frontierReady"] = True
    return metrics


def assert_regression_restored(page, *, lightweight: bool = False) -> dict[str, Any]:
    wait_visible(page, "#reg-main-container", timeout=60000)
    wait_ready(page, "#reg-periodicity-select", timeout=60000)
    metrics = {}
    set_component_value(page, "reg-tabs", "anova")
    wait_content_ready(page, "#reg-anova-content", timeout=60000)
    metrics["anovaReady"] = True
    if lightweight:
        return metrics
    set_component_value(page, "reg-tabs", "returns")
    wait_content_ready(page, "#reg-returns-content", timeout=60000)
    metrics["returnsReady"] = True
    set_component_value(page, "reg-tabs", "statistics")
    wait_content_ready(page, "#reg-statistics-content", timeout=60000)
    metrics["statisticsReady"] = True
    set_component_value(page, "reg-tabs", "scatter")
    wait_content_ready(page, "#reg-scatter-content", timeout=60000)
    metrics["scatterReady"] = True
    return metrics


def assert_raw_data_artifact_regenerated(page) -> None:
    page.wait_for_function(
        """
        () => {
          const value = window.sessionStorage.getItem('dashmat-raw-data-artifact-store');
          if (!value) return false;
          try {
            const parsed = JSON.parse(value);
            return !!(parsed && parsed.raw_data_key);
          } catch (error) {
            return false;
          }
        }
        """,
        timeout=60000,
    )


def decode_downloaded_bundle(raw_bytes: bytes) -> tuple[dict[str, Any], str]:
    text = raw_bytes.decode("utf-8")
    decoder = json.JSONDecoder()
    bundle, end = decoder.raw_decode(text)
    remainder = text[end:].strip()
    if remainder:
        # Some Windows runs leave stale trailing bytes at the destination path.
        # Parse the first complete JSON object and persist the normalized bundle text.
        text = text[:end]
    if not isinstance(bundle, dict):
        raise RuntimeError("Downloaded bundle is not a JSON object.")
    return bundle, text


def capture_session_bundle(
    page,
    trigger_component_id: str,
    bundle_path: Path,
    *,
    attempts: int = 2,
) -> tuple[dict[str, Any], int]:
    last_error: Exception | None = None
    for _ in range(max(attempts, 1)):
        if bundle_path.exists():
            bundle_path.unlink()
        try:
            with page.expect_download(timeout=60000) as download_info:
                set_component_props(page, trigger_component_id, {"n_clicks": int(time.time() * 1000)})
            download = download_info.value
            source_path = download.path()
            if not source_path:
                raise RuntimeError("Download path was not available.")
            raw_bytes = Path(source_path).read_bytes()
            bundle, bundle_text = decode_downloaded_bundle(raw_bytes)
            bundle_path.write_text(bundle_text, encoding="utf-8")
            return bundle, len(bundle_text.encode("utf-8"))
        except Exception as exc:
            last_error = exc
            page.wait_for_timeout(500)
    if last_error is not None:
        raise last_error
    raise RuntimeError("Bundle download failed.")


def upload_session_bundle(page, upload_component_id: str, bundle_path: Path) -> None:
    page.locator(f"#{upload_component_id} input[type=file]").set_input_files(str(bundle_path))


def safe_collect_browser_storage_metrics(page, attempts: int = 3) -> dict[str, Any]:
    last_error: Exception | None = None
    for _ in range(max(attempts, 1)):
        try:
            page.wait_for_load_state("domcontentloaded", timeout=10000)
            page.wait_for_timeout(300)
            return collect_browser_storage_metrics(page)
        except Exception as exc:
            last_error = exc
            if "Execution context was destroyed" not in str(exc):
                raise
            page.wait_for_timeout(400)
    if last_error is not None:
        raise last_error
    return {}


def apply_restore(
    page,
    base_url: str,
    workflow: str,
    bundle_path: Path,
    expect_alert: str | None = None,
    expect_pageerror: str | None = None,
    lightweight_assertions: bool = False,
) -> dict[str, Any]:
    target_path = {
        "analyticstool": "/analyticstool",
        "portopt": "/portopt",
        "regression": "/regression",
        "combined": "/regression",
    }[workflow]
    page.goto(base_url + target_path, wait_until="domcontentloaded")
    if expect_alert:
        with page.expect_event("dialog", timeout=60000) as dialog_info:
            upload_session_bundle(page, {
                "analyticstool": "at-load-session-upload",
                "portopt": "po-load-session-upload",
                "regression": "reg-load-session-upload",
                "combined": "reg-load-session-upload",
            }[workflow], bundle_path)
        dialog = dialog_info.value
        message = dialog.message
        dialog.accept()
        if expect_alert not in message:
            raise RuntimeError(f"Expected alert containing {expect_alert!r}, got {message!r}")
        return {"alert": message}

    if expect_pageerror:
        page_errors: list[str] = []
        page.on("pageerror", lambda err: page_errors.append(str(err)) if len(page_errors) < 20 else None)
        upload_session_bundle(page, {
            "analyticstool": "at-load-session-upload",
            "portopt": "po-load-session-upload",
            "regression": "reg-load-session-upload",
            "combined": "reg-load-session-upload",
        }[workflow], bundle_path)
        page.wait_for_timeout(1500)
        if not any(expect_pageerror in message for message in page_errors):
            raise RuntimeError(f"Expected pageerror containing {expect_pageerror!r}, got {page_errors!r}")
        return {"pageerror": page_errors[-1] if page_errors else ""}

    upload_session_bundle(page, {
        "analyticstool": "at-load-session-upload",
        "portopt": "po-load-session-upload",
        "regression": "reg-load-session-upload",
        "combined": "reg-load-session-upload",
    }[workflow], bundle_path)

    if workflow == "analyticstool":
        metrics = assert_analyticstool_restored(page, lightweight=lightweight_assertions)
    elif workflow == "portopt":
        metrics = assert_portopt_restored(page, lightweight=lightweight_assertions)
    elif workflow == "regression":
        metrics = assert_regression_restored(page, lightweight=lightweight_assertions)
    else:
        metrics = assert_regression_restored(page, lightweight=lightweight_assertions)
        page.goto(base_url + "/portopt", wait_until="domcontentloaded")
        metrics["portopt"] = assert_portopt_restored(page, lightweight=lightweight_assertions)
    assert_raw_data_artifact_regenerated(page)
    return metrics


def run_cycle(
    *,
    browser,
    base_url: str,
    workflow: str,
    effective_mode: str,
    use_lightweight_assertions: bool,
    cycle_index: int,
    artifact_root: Path,
    out_dir: Path,
    repo_root: Path,
    tamper_kind: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    console_messages: list[dict[str, str]] = []
    reset_artifacts_for_cycle(artifact_root)
    seed_payload = build_seed_payloads(repo_root, f"session-{cycle_index}-{int(time.time())}")
    storage_seed = build_storage_seed(workflow, seed_payload)
    context = browser.new_context(viewport={"width": 1440, "height": 960}, accept_downloads=True)
    page = context.new_page()
    page.on(
        "console",
        lambda msg: console_messages.append({"type": msg.type, "text": msg.text})
        if len(console_messages) < 200 and msg.type in {"warning", "error"}
        else None,
    )
    page.on(
        "pageerror",
        lambda err: console_messages.append({"type": "pageerror", "text": str(err)})
        if len(console_messages) < 200
        else None,
    )
    bundle_path = out_dir / f"save_load_bundle_cycle{cycle_index:03d}.json"
    artifact_delete_metrics = None
    try:
        load_workflow_page(page, base_url, workflow, storage_seed)
        trigger_id = {
            "analyticstool": "at-menu-save-session",
            "portopt": "po-menu-save-session",
            "regression": "reg-menu-save-session",
            "combined": "reg-menu-save-session",
        }[workflow]
        bundle, bundle_bytes = capture_session_bundle(page, trigger_id, bundle_path)
        bundle_validation_error = None
        if use_lightweight_assertions:
            try:
                bundle_meta = validate_bundle_structure(bundle, workflow)
            except Exception as exc:
                bundle_meta = {
                    "artifactCount": len(bundle.get("artifacts") or []) if isinstance(bundle, dict) else 0,
                    "artifactTypes": sorted(
                        {
                            str(record.get("artifact_type"))
                            for record in (bundle.get("artifacts") or [])
                            if isinstance(record, dict) and record.get("artifact_type")
                        }
                    ) if isinstance(bundle, dict) else [],
                    "warningCount": len(bundle.get("export_warnings") or []) if isinstance(bundle, dict) else 0,
                }
                bundle_validation_error = str(exc)
        else:
            bundle_meta = validate_bundle_structure(bundle, workflow)
        clear_browser_session(page)
        tampered_bundle_path = bundle_path
        expected_alert = None
        expected_pageerror = None
        if effective_mode == "portable_restore":
            artifact_delete_metrics = delete_artifacts(artifact_root)
        elif effective_mode == "tampered_restore":
            tampered = tamper_bundle_for_mode(bundle, tamper_kind, workflow)
            tampered_bundle_path = out_dir / f"save_load_bundle_cycle{cycle_index:03d}_tampered.json"
            write_bundle_payload(tampered_bundle_path, tampered)
            if tamper_kind == "invalid_version":
                expected_alert = "Unsupported session bundle version."
            elif tamper_kind == "missing_workspace_session":
                expected_alert = "Malformed session bundle."
            elif tamper_kind == "malformed_json":
                expected_pageerror = "Unexpected end of JSON input"
        restore_metrics = apply_restore(
            page,
            base_url,
            workflow,
            tampered_bundle_path,
            expect_alert=expected_alert,
            expect_pageerror=expected_pageerror,
            lightweight_assertions=use_lightweight_assertions,
        )
        browser_storage = safe_collect_browser_storage_metrics(page)
        result = {
            "cycle": cycle_index,
            "workflow": workflow,
            "mode": effective_mode,
            "ok": True,
            "durationMs": round((time.perf_counter() - started) * 1000),
            "bundleBytes": bundle_bytes,
            "bundleArtifactCount": bundle_meta["artifactCount"],
            "bundleArtifactTypes": bundle_meta["artifactTypes"],
            "bundleWarningCount": bundle_meta["warningCount"],
            "restoreMetrics": restore_metrics,
            "browserStorage": browser_storage,
            "storageBytes": ((browser_storage or {}).get("sessionStorage") or {}).get("itemBytes") or {},
            "artifactStore": collect_artifact_store_metrics(repo_root),
            "artifactDelete": artifact_delete_metrics,
            "consoleMessages": console_messages,
        }
        if bundle_validation_error:
            result["bundleValidationError"] = bundle_validation_error
        if effective_mode == "tampered_restore" and tamper_kind == "missing_required_artifact":
            result["degraded"] = True
        return result
    except Exception as exc:
        screenshot_path = out_dir / f"save_load_cycle{cycle_index:03d}_failure.png"
        try:
            page.screenshot(path=str(screenshot_path), full_page=True)
        except Exception:
            screenshot_path = None
        return {
            "cycle": cycle_index,
            "workflow": workflow,
            "mode": effective_mode,
            "ok": False,
            "durationMs": round((time.perf_counter() - started) * 1000),
            "failureMode": str(exc),
            "traceback": traceback.format_exc(),
            "screenshotPath": str(screenshot_path) if screenshot_path else None,
            "consoleMessages": console_messages,
            "artifactStore": collect_artifact_store_metrics(repo_root),
            "artifactDelete": artifact_delete_metrics,
        }
    finally:
        context.close()


def resolve_effective_mode(args: argparse.Namespace, cycle_index: int) -> str:
    if args.mode != "repeat_cycle":
        return args.mode
    if args.portable_every > 0 and cycle_index % args.portable_every == 0:
        return "portable_restore"
    return "same_machine_restore"


def run_harness(args: argparse.Namespace, repo_root: Path, artifact_root: Path) -> dict[str, Any]:
    cycle_records: list[dict[str, Any]] = []
    os.environ["DASHMAT_ARTIFACT_ROOT"] = str(artifact_root)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not args.headed)
        for cycle_index in range(1, max(args.cycles, 1) + 1):
            effective_mode = resolve_effective_mode(args, cycle_index)
            cycle_records.append(
                run_cycle(
                    browser=browser,
                    base_url=args.base_url,
                    workflow=args.workflow,
                    effective_mode=effective_mode,
                    use_lightweight_assertions=(args.mode == "repeat_cycle"),
                    cycle_index=cycle_index,
                    artifact_root=artifact_root,
                    out_dir=repo_root / "output" / "playwright",
                    repo_root=repo_root,
                    tamper_kind=args.tamper_kind,
                )
            )
        browser.close()
    return {
        "workflow": args.workflow,
        "mode": args.mode,
        "cycles": cycle_records,
        "summary": summarize_cycles(cycle_records),
        "artifactStore": collect_artifact_store_metrics(repo_root),
    }


def main() -> int:
    args = parse_args()
    repo_root = resolve_repo_root(args.repo_root)
    resolved_git_ref = resolve_git_ref(repo_root, args.git_ref)
    if args.launch_app:
        args.base_url = f"http://127.0.0.1:{args.port}"
    artifact_root = resolve_artifact_root(repo_root, args.artifact_root)
    out_dir = repo_root / "output" / "playwright"
    fail_dir = out_dir / "failures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fail_dir.mkdir(parents=True, exist_ok=True)
    stem = build_artifact_stem(args.label, args.workflow, args.mode, resolved_git_ref)

    proc = None
    log_handle = None
    try:
        if args.launch_app:
            log_path = out_dir / f"{stem}_app.log"
            proc, log_handle = launch_app_process(repo_root, args.port, artifact_root, log_path)
            time.sleep(2)
        wait_for_app(args.base_url, args.startup_timeout)
        result = run_harness(args, repo_root, artifact_root)
    except Exception as exc:
        failure_payload = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "label": args.label,
            "workflow": args.workflow,
            "mode": args.mode,
            "baseUrl": args.base_url,
            "repoRoot": str(repo_root),
            "artifactRoot": str(artifact_root),
            "gitRef": resolved_git_ref,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        fail_path = fail_dir / f"{stem}.json"
        fail_path.write_text(json.dumps(failure_payload, indent=2), encoding="utf-8")
        print(f"OUT_PATH={fail_path}")
        return 1
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        if log_handle is not None:
            log_handle.close()

    out_path = out_dir / f"{stem}.json"
    payload = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "label": args.label,
        "workflow": args.workflow,
        "mode": args.mode,
        "gitRef": resolved_git_ref,
        "baseUrl": args.base_url,
        "repoRoot": str(repo_root),
        "artifactRoot": str(artifact_root),
        "cyclesRequested": args.cycles,
        "portableEvery": args.portable_every,
        "tamperKind": args.tamper_kind,
        "result": result,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"OUT_PATH={out_path}")
    print("SUMMARY=" + json.dumps(result["summary"], separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
