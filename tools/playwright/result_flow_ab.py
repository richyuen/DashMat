from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-repo-root", required=True)
    parser.add_argument("--variant-repo-root", required=True)
    parser.add_argument("--baseline-port", type=int, default=8051)
    parser.add_argument("--variant-port", type=int, default=8052)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--label", default="")
    parser.add_argument("--mode", choices=("consume_only", "run_and_consume"), default="consume_only")
    parser.add_argument("--startup-timeout", type=int, default=30)
    parser.add_argument("--headed", action="store_true")
    return parser.parse_args()


def sanitize_token(value: str, fallback: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "-" for ch in str(value or "").strip().lower())
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    return cleaned or fallback


def resolve_git_ref(root: Path) -> str:
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


def build_compare_stem(label: str, mode: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    return f"result_flow_ab_{sanitize_token(mode, 'mode')}_{ts}_{sanitize_token(label, 'run')}"


def launch_app(repo_root: Path, port: int, artifact_root: Path, log_path: Path) -> tuple[subprocess.Popen, Any]:
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


def run_harness(repo_root: Path, port: int, runs: int, label: str, mode: str, startup_timeout: int, headed: bool, artifact_root: Path) -> tuple[dict[str, Any], str]:
    script_path = Path(__file__).resolve().with_name("result_flow_harness.py")
    env = os.environ.copy()
    env["DASHMAT_ARTIFACT_ROOT"] = str(artifact_root)
    cmd = [
        sys.executable,
        str(script_path),
        "--repo-root",
        str(repo_root),
        "--base-url",
        f"http://127.0.0.1:{port}",
        "--runs",
        str(runs),
        "--label",
        label,
        "--mode",
        mode,
        "--startup-timeout",
        str(startup_timeout),
    ]
    if headed:
        cmd.append("--headed")
    proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[2], env=env, capture_output=True, text=True, check=True)
    out_path = None
    for line in proc.stdout.splitlines():
        if line.startswith("OUT_PATH="):
            out_path = line.split("=", 1)[1].strip()
            break
    if not out_path:
        raise RuntimeError(f"Harness did not report OUT_PATH.\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    payload = json.loads(Path(out_path).read_text(encoding="utf-8"))
    return payload, proc.stdout


def _delta(baseline: int | float, variant: int | float) -> dict[str, Any]:
    absolute = float(variant) - float(baseline)
    pct = None if float(baseline) == 0 else round((absolute / float(baseline)) * 100.0, 2)
    return {"baseline": baseline, "variant": variant, "absoluteDelta": round(absolute, 2), "percentDelta": pct}


def compare_metric_maps(baseline: dict[str, Any], variant: dict[str, Any], metric_names: list[str]) -> dict[str, Any]:
    return {name: _delta(baseline.get(name, 0), variant.get(name, 0)) for name in metric_names}


def is_confounded(baseline_payload: dict[str, Any], variant_payload: dict[str, Any]) -> bool:
    broad_regression = (
        variant_payload["portopt"].get("pageReadyMedian", 0) > baseline_payload["portopt"].get("pageReadyMedian", 0) + 75
        and variant_payload["regression"].get("pageReadyMedian", 0) > baseline_payload["regression"].get("pageReadyMedian", 0) + 75
        and variant_payload["portopt"].get("revisitReadyMedian", 0) > baseline_payload["portopt"].get("revisitReadyMedian", 0) + 75
        and variant_payload["regression"].get("revisitReadyMedian", 0) > baseline_payload["regression"].get("revisitReadyMedian", 0) + 75
    )
    baseline_saved = (baseline_payload["portopt"].get("storageBytesMedian") or {}).get("dashmat-saved-series-cache-store", 0)
    variant_saved = (variant_payload["portopt"].get("storageBytesMedian") or {}).get("dashmat-saved-series-cache-store", 0)
    return broad_regression and abs(int(variant_saved) - int(baseline_saved)) > 65536


def build_comparison_payload(baseline_payload: dict[str, Any], variant_payload: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    portopt_metrics = [
        "pageReadyMedian",
        "initialDefaultContentReadyMedian",
        "returnsOpenMedian",
        "statisticsOpenMedian",
        "rollingOpenMedian",
        "calendarOpenMedian",
        "growthOpenMedian",
        "drawdownOpenMedian",
        "frontierOpenMedian",
        "revisitReadyMedian",
        "defaultContentReadyMedian",
    ]
    regression_metrics = [
        "pageReadyMedian",
        "initialDefaultContentReadyMedian",
        "returnsOpenMedian",
        "statisticsOpenMedian",
        "rollingReturnsOpenMedian",
        "calendarOpenMedian",
        "growthOpenMedian",
        "drawdownOpenMedian",
        "scatterOpenMedian",
        "anovaOpenMedian",
        "revisitReadyMedian",
        "defaultContentReadyMedian",
    ]
    storage_keys = [
        "po-results-store",
        "reg-results-store",
        "dashmat-raw-data-store",
        "dashmat-raw-data-artifact-store",
        "dashmat-saved-series-cache-store",
    ]
    return {
        "timestamp": datetime.now().astimezone().isoformat(),
        "label": args.label,
        "mode": args.mode,
        "runs": args.runs,
        "baseline": baseline_payload,
        "variant": variant_payload,
        "portopt": compare_metric_maps(baseline_payload["portopt"], variant_payload["portopt"], portopt_metrics),
        "regression": compare_metric_maps(baseline_payload["regression"], variant_payload["regression"], regression_metrics),
        "storage": {
            key: _delta(
                (baseline_payload["portopt"].get("storageBytesMedian") or {}).get(key, 0)
                or (baseline_payload["regression"].get("storageBytesMedian") or {}).get(key, 0),
                (variant_payload["portopt"].get("storageBytesMedian") or {}).get(key, 0)
                or (variant_payload["regression"].get("storageBytesMedian") or {}).get(key, 0),
            )
            for key in storage_keys
        },
        "confounded": is_confounded(baseline_payload, variant_payload),
    }


def main() -> int:
    args = parse_args()
    baseline_root = Path(args.baseline_repo_root).resolve()
    variant_root = Path(args.variant_repo_root).resolve()
    out_dir = variant_root / "output" / "playwright"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = build_compare_stem(args.label, args.mode)
    baseline_artifact_root = variant_root / ".cache" / f"{stem}_baseline_artifacts"
    variant_artifact_root = variant_root / ".cache" / f"{stem}_variant_artifacts"
    baseline_log = out_dir / f"{stem}_baseline_app.log"
    variant_log = out_dir / f"{stem}_variant_app.log"
    baseline_proc, baseline_log_handle = launch_app(baseline_root, args.baseline_port, baseline_artifact_root, baseline_log)
    variant_proc, variant_log_handle = launch_app(variant_root, args.variant_port, variant_artifact_root, variant_log)
    try:
        time.sleep(2)
        baseline_payload, _baseline_stdout = run_harness(
            baseline_root,
            args.baseline_port,
            args.runs,
            f"{args.label}-baseline" if args.label else "baseline",
            args.mode,
            args.startup_timeout,
            args.headed,
            baseline_artifact_root,
        )
        variant_payload, _variant_stdout = run_harness(
            variant_root,
            args.variant_port,
            args.runs,
            f"{args.label}-variant" if args.label else "variant",
            args.mode,
            args.startup_timeout,
            args.headed,
            variant_artifact_root,
        )
    finally:
        for proc in (baseline_proc, variant_proc):
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        baseline_log_handle.close()
        variant_log_handle.close()
    payload = build_comparison_payload(baseline_payload, variant_payload, args)
    out_path = out_dir / f"{stem}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"OUT_PATH={out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
