from __future__ import annotations

import argparse
import json
import os
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
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from playwright.sync_api import sync_playwright

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.returns import build_raw_data_metadata, df_to_json

try:
    from utils.artifact_store import ArtifactStore, write_raw_data_frame
except Exception:  # pragma: no cover - baseline repos may not support artifacts
    ArtifactStore = None  # type: ignore[assignment]
    write_raw_data_frame = None  # type: ignore[assignment]


DEFAULT_SERIES = [
    "SPX_TRIndex",
    "R2000_TRIndex",
    "EAFE_TRIndex",
    "BCTBill13_TRIndex",
]

PORTOPT_TAB_SPECS = [
    {"metric": "returnsOpenMs", "tab": "returns", "container": "#po-returns-grid-content"},
    {"metric": "statisticsOpenMs", "tab": "statistics", "container": "#po-statistics-grid-content"},
    {"metric": "rollingOpenMs", "tab": "rolling", "container": "#po-rolling-content"},
    {"metric": "calendarOpenMs", "tab": "calendar", "container": "#po-calendar-content"},
    {"metric": "growthOpenMs", "tab": "growth", "container": "#po-growth-chart-container"},
    {"metric": "drawdownOpenMs", "tab": "drawdown", "container": "#po-drawdown-content"},
    {
        "metric": "frontierOpenMs",
        "tab": "frontier",
        "container": "#po-frontier-chart-container",
        "switch_id": "po-frontier-chart-switch",
        "switch_value": "chart",
    },
]

REGRESSION_TAB_SPECS = [
    {"metric": "returnsOpenMs", "tab": "returns", "container": "#reg-returns-content"},
    {"metric": "statisticsOpenMs", "tab": "statistics", "container": "#reg-statistics-content"},
    {"metric": "rollingReturnsOpenMs", "tab": "rolling_returns", "container": "#reg-rolling-returns-content"},
    {"metric": "calendarOpenMs", "tab": "calendar", "container": "#reg-calendar-content"},
    {"metric": "growthOpenMs", "tab": "growth", "container": "#reg-growth-content"},
    {"metric": "drawdownOpenMs", "tab": "drawdown", "container": "#reg-drawdown-content"},
    {"metric": "scatterOpenMs", "tab": "scatter", "container": "#reg-scatter-content"},
    {"metric": "anovaOpenMs", "tab": "anova", "container": "#reg-anova-content"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="")
    parser.add_argument("--base-url", default="http://127.0.0.1:8050")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--label", default="")
    parser.add_argument("--git-ref", default="")
    parser.add_argument("--startup-timeout", type=int, default=30)
    parser.add_argument("--mode", choices=("consume_only", "run_and_consume"), default="consume_only")
    parser.add_argument("--headed", action="store_true")
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


def build_artifact_stem(prefix: str, label: str, git_ref: str, base_url: str, timestamp: str) -> str:
    git_token = sanitize_token((git_ref or "unknown")[:8], "unknown")
    label_token = sanitize_token(label, "run")
    mode_token = sanitize_token(prefix, "result-flow")
    port = urllib.parse.urlparse(base_url).port
    port_token = f"p{port}" if port else "punknown"
    return f"{mode_token}_{timestamp}_{label_token}_{git_token}_{port_token}"


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


def set_component_props(page, component_id: str, props: dict[str, Any]) -> None:
    page.evaluate(
        """
        ([componentId, nextProps]) => {
          window.dash_clientside.set_props(componentId, nextProps);
        }
        """,
        [component_id, props],
    )


def set_component_value(page, component_id: str, value: Any) -> None:
    set_component_props(page, component_id, {"value": value})


def navigate_path(page, path: str, timeout: int = 30000) -> None:
    page.evaluate("(nextPath) => { window.location.pathname = nextPath; }", path)
    page.wait_for_function("(nextPath) => window.location.pathname === nextPath", arg=path, timeout=timeout)


def ensure_modal_hidden(page, selector: str, timeout: int = 2000) -> None:
    page.wait_for_function(
        """
        (sel) => {
          const el = document.querySelector(sel);
          if (!el) return true;
          const style = window.getComputedStyle(el);
          const rect = el.getBoundingClientRect();
          const visible = style.display !== "none" && style.visibility !== "hidden" && rect.width > 0 && rect.height > 0;
          return !visible;
        }
        """,
        arg=selector,
        timeout=timeout,
    )


def force_modal_closed(page, component_id: str, selector: str) -> None:
    set_component_props(page, component_id, {"opened": False})
    ensure_modal_hidden(page, selector, timeout=5000)


def collect_browser_storage_metrics(page) -> dict[str, Any]:
    return page.evaluate(
        """
        () => {
          const encoder = new TextEncoder();
          const summarize = (store) => {
            const keys = [];
            let totalBytes = 0;
            const itemBytes = {};
            for (let i = 0; i < store.length; i += 1) {
              const key = store.key(i);
              if (!key) continue;
              const value = store.getItem(key) || "";
              const size = encoder.encode(key).length + encoder.encode(value).length;
              keys.push(key);
              itemBytes[key] = size;
              totalBytes += size;
            }
            keys.sort();
            return { count: keys.length, totalBytes, itemBytes, keys };
          };
          return {
            sessionStorage: summarize(window.sessionStorage),
            localStorage: summarize(window.localStorage),
          };
        }
        """
    )


def collect_artifact_store_metrics(repo_root: Path) -> dict[str, Any]:
    configured = Path(os.environ.get("DASHMAT_ARTIFACT_ROOT", str(repo_root / ".cache" / "dashmat_artifacts")))
    root = configured if configured.is_absolute() else (repo_root / configured).resolve()
    if not root.exists():
        return {"exists": False, "path": str(root), "fileCount": 0, "totalBytes": 0}
    file_count = 0
    total_bytes = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        file_count += 1
        total_bytes += path.stat().st_size
    return {
        "exists": True,
        "path": str(root),
        "fileCount": file_count,
        "totalBytes": total_bytes,
    }


def build_seed_dataset(periods: int = 756) -> pd.DataFrame:
    idx = pd.bdate_range("2021-01-04", periods=periods)
    t = np.arange(periods, dtype=float)
    spx = 0.00035 + 0.0065 * np.sin(t / 17.0) / 10.0 + 0.0022 * np.cos(t / 41.0) / 10.0
    r2000 = 0.00042 + 0.0072 * np.sin((t + 5.0) / 15.0) / 10.0 + 0.0028 * np.cos(t / 29.0) / 10.0
    eafe = 0.00028 + 0.0058 * np.sin((t + 11.0) / 21.0) / 10.0 + 0.0021 * np.cos(t / 35.0) / 10.0
    bills = np.full(periods, 0.00008)
    df = pd.DataFrame(
        {
            "SPX_TRIndex": spx,
            "R2000_TRIndex": r2000,
            "EAFE_TRIndex": eafe,
            "BCTBill13_TRIndex": bills,
        },
        index=idx,
    )
    df.index.name = "Date"
    return df


def get_seed_dataset_metadata(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(df.shape[0]),
        "columns": list(df.columns),
        "start": str(df.index.min())[:10] if not df.empty else None,
        "end": str(df.index.max())[:10] if not df.empty else None,
        "periodicity": "daily",
    }


def detect_repo_features(repo_root: Path) -> dict[str, bool]:
    app_text = (repo_root / "app.py").read_text(encoding="utf-8")
    artifact_text = (repo_root / "utils" / "artifact_store.py").read_text(encoding="utf-8") if (repo_root / "utils" / "artifact_store.py").exists() else ""
    return {
        "session_id_store": "dashmat-session-id-store" in app_text,
        "raw_data_descriptor_store": "RAW_DATA_DESCRIPTOR_VERSION" in artifact_text,
        "artifact_module": (repo_root / "utils" / "artifact_store.py").exists(),
    }


def maybe_store_dataframe_artifact(
    *,
    session_id: str,
    artifact_type: str,
    payload: dict[str, Any],
    frame: pd.DataFrame,
) -> str | None:
    if ArtifactStore is None or frame.empty:
        return None
    store = ArtifactStore()
    descriptor = store.put_dataframe(
        df=frame,
        artifact_type=artifact_type,
        session_id=session_id,
        payload=payload,
        metadata={},
    )
    return descriptor.key


def build_portopt_seed_result(
    raw_df: pd.DataFrame,
    *,
    session_id: str | None,
    use_artifacts: bool,
    portfolio_name: str = "Seeded Risk Parity",
) -> tuple[str, dict[str, Any], list[str]]:
    selected_series = ["SPX_TRIndex", "R2000_TRIndex", "EAFE_TRIndex"]
    weights = {"SPX_TRIndex": 0.5, "R2000_TRIndex": 0.3, "EAFE_TRIndex": 0.2}
    portfolio_returns = sum(raw_df[col] * weight for col, weight in weights.items()).rename(portfolio_name)
    entry: dict[str, Any] = {
        "saved_series_name": None,
        "config": {
            "selected_series": selected_series,
            "periodicity": "daily",
            "model": "risk_parity",
            "missing_data": "fill_na",
            "objective": "risk_parity",
        },
        "window_weights": [
            {
                "est_start": str(raw_df.index[0])[:10],
                "est_end": str(raw_df.index[-1])[:10],
                "apply_start": str(raw_df.index[0])[:10],
                "apply_end": str(raw_df.index[-1])[:10],
                "weights": dict(weights),
            }
        ],
        "frontier_cache": {
            "0": {
                "MV": {
                    "window_index": 0,
                    "risk_measure": "MV",
                    "frontier_points": [
                        {"risk": 0.085, "return": 0.055},
                        {"risk": 0.105, "return": 0.071},
                        {"risk": 0.125, "return": 0.083},
                    ],
                    "assets": [
                        {"name": "SPX_TRIndex", "risk": 0.16, "return": 0.09},
                        {"name": "R2000_TRIndex", "risk": 0.21, "return": 0.11},
                        {"name": "EAFE_TRIndex", "risk": 0.14, "return": 0.075},
                    ],
                    "portfolio": {"name": portfolio_name, "risk": 0.098, "return": 0.068},
                    "frontier_portfolios": [
                        {
                            "point_index": 1,
                            "name": portfolio_name,
                            "return": 0.068,
                            "risk": 0.098,
                            "sharpe": 0.69,
                            "weights": dict(weights),
                        }
                    ],
                }
            }
        },
    }
    if use_artifacts and session_id:
        returns_key = maybe_store_dataframe_artifact(
            session_id=session_id,
            artifact_type="po_returns_series",
            payload={"portfolio": portfolio_name},
            frame=portfolio_returns.to_frame(portfolio_name),
        )
        if returns_key:
            entry["returns_key"] = returns_key
        else:
            entry["returns_json"] = portfolio_returns.to_json(date_format="iso")
    else:
        entry["returns_json"] = portfolio_returns.to_json(date_format="iso")
    return portfolio_name, entry, selected_series


def build_regression_seed_result(
    raw_df: pd.DataFrame,
    *,
    session_id: str | None,
    use_artifacts: bool,
    result_name: str = "Seeded OLS",
) -> tuple[str, dict[str, Any], str, list[str]]:
    dependent_var = "SPX_TRIndex"
    independent_vars = ["R2000_TRIndex", "EAFE_TRIndex"]
    predicted = (
        0.55 * raw_df["SPX_TRIndex"]
        + 0.30 * raw_df["R2000_TRIndex"]
        - 0.10 * raw_df["EAFE_TRIndex"]
        + 0.00005
    ).rename("predicted")
    residuals = (raw_df[dependent_var] - predicted).rename("residuals")
    feature_label_map = {name: name for name in independent_vars}
    entry: dict[str, Any] = {
        "window_results": [
            {
                "est_start": str(raw_df.index[0])[:10],
                "est_end": str(raw_df.index[-1])[:10],
                "apply_start": str(raw_df.index[0])[:10],
                "apply_end": str(raw_df.index[-1])[:10],
                "coefficients": {"Intercept": 0.00005, "R2000_TRIndex": 0.30, "EAFE_TRIndex": -0.10},
                "p_values": {"Intercept": 0.04, "R2000_TRIndex": 0.01, "EAFE_TRIndex": 0.03},
                "r_squared": 0.72,
                "adj_r_squared": 0.71,
                "anova_table": {
                    "df_model": 2,
                    "df_resid": int(max(len(raw_df) - 3, 1)),
                    "ss_model": 0.85,
                    "ss_resid": 0.33,
                    "ss_total": 1.18,
                    "ms_model": 0.425,
                    "ms_resid": 0.0005,
                    "F_stat": 33.2,
                    "F_pvalue": 0.0001,
                },
                "diagnostics": {
                    "vif": {"R2000_TRIndex": 1.2, "EAFE_TRIndex": 1.1},
                    "std_errors": {"Intercept": 0.00002, "R2000_TRIndex": 0.05, "EAFE_TRIndex": 0.04},
                    "t_stats": {"Intercept": 2.5, "R2000_TRIndex": 6.0, "EAFE_TRIndex": -2.5},
                },
                "arima_garch": {},
                "residual_std": 0.009,
                "oos_metrics": {"rmse": 0.0092, "mae": 0.0075},
                "n_obs": int(len(raw_df)),
            }
        ],
        "saved_series_name": None,
        "dependent_var": dependent_var,
        "independent_vars": list(independent_vars),
        "independent_vars_internal": list(independent_vars),
        "benchmark_assignments": {},
        "long_short_assignments": {},
        "date_range": None,
        "effective_date_range": {"start": str(raw_df.index[0])[:10], "end": str(raw_df.index[-1])[:10]},
        "vol_scaler": 0,
        "vol_scaling_assignments": {},
        "config": {
            "model": "ols",
            "force_zero_intercept": False,
            "robust_se": False,
            "exp_wt": False,
            "halflife": 63,
            "window_type": "full",
            "window_size": 36,
            "opt_step": 1,
            "opt_step_unit": "months",
            "fill_in_sample": False,
            "missing_data": "fill_na",
            "alpha": 1.0,
            "l1_ratio": 0.5,
            "lag_config": {name: 0 for name in independent_vars},
            "lag_config_display": {name: 0 for name in independent_vars},
            "feature_label_map": feature_label_map,
            "independent_vars_internal": list(independent_vars),
            "independent_vars_display": list(independent_vars),
        },
        "periodicity": "daily",
        "arima_garch_summary": {},
    }
    if use_artifacts and session_id:
        predicted_key = maybe_store_dataframe_artifact(
            session_id=session_id,
            artifact_type="reg_predicted_series",
            payload={"result": result_name, "kind": "predicted"},
            frame=predicted.to_frame("predicted"),
        )
        residuals_key = maybe_store_dataframe_artifact(
            session_id=session_id,
            artifact_type="reg_residuals_series",
            payload={"result": result_name, "kind": "residuals"},
            frame=residuals.to_frame("residuals"),
        )
        if predicted_key:
            entry["predicted_key"] = predicted_key
        else:
            entry["predicted_json"] = df_to_json(predicted.to_frame("predicted"))
        if residuals_key:
            entry["residuals_key"] = residuals_key
        else:
            entry["residuals_json"] = df_to_json(residuals.to_frame("residuals"))
    else:
        entry["predicted_json"] = df_to_json(predicted.to_frame("predicted"))
        entry["residuals_json"] = df_to_json(residuals.to_frame("residuals"))
    return result_name, entry, dependent_var, independent_vars


def build_seed_payloads(repo_root: Path, session_id: str) -> dict[str, Any]:
    features = detect_repo_features(repo_root)
    raw_df = build_seed_dataset()
    raw_json = df_to_json(raw_df)
    raw_store = raw_json
    raw_meta = build_raw_data_metadata(raw_json, "daily")
    use_artifacts = features["artifact_module"] and features["session_id_store"]
    if features["raw_data_descriptor_store"] and write_raw_data_frame is not None:
        raw_store, raw_meta = write_raw_data_frame(
            df=raw_df,
            session_id=session_id,
            original_periodicity="daily",
        )
        raw_store = raw_store or ""
    po_name, po_entry, po_series = build_portopt_seed_result(
        raw_df,
        session_id=session_id if use_artifacts else None,
        use_artifacts=use_artifacts,
    )
    reg_name, reg_entry, dep_var, reg_series = build_regression_seed_result(
        raw_df,
        session_id=session_id if use_artifacts else None,
        use_artifacts=use_artifacts,
    )
    return {
        "features": features,
        "session_id": session_id,
        "raw_df": raw_df,
        "raw_json": raw_json,
        "raw_store": raw_store,
        "raw_meta": raw_meta,
        "dataset_meta": get_seed_dataset_metadata(raw_df),
        "portopt": {
            "results": {po_name: po_entry},
            "portfolio_name": po_name,
            "selected_series": po_series,
        },
        "regression": {
            "results": {reg_name: reg_entry},
            "result_name": reg_name,
            "dependent_var": dep_var,
            "selected_series": list(reg_series),
            "all_series": [dep_var, *reg_series],
            "independent_vars": reg_series,
        },
    }


def make_portopt_storage_seed(seed_payload: dict[str, Any], *, include_results: bool = True) -> dict[str, Any]:
    payload = {
        "dashmat-raw-data-store": seed_payload["raw_store"],
        "dashmat-raw-data-meta-store": seed_payload["raw_meta"],
        "dashmat-original-periodicity-store": "daily",
        "dashmat-pending-new-series-store": {},
        "po-page-visited-store": True,
        "po-restore-complete-store": True,
        "po-periodicity-value-store": "daily",
        "po-series-select": seed_payload["portopt"]["selected_series"],
        "po-series-select-value-store": seed_payload["portopt"]["selected_series"],
        "po-series-order-store": seed_payload["portopt"]["selected_series"],
        "po-active-tab-store": "weight",
        "po-weight-chart-switch-store": "chart",
        "po-frontier-chart-switch-store": "chart",
    }
    if include_results:
        payload["po-results-store"] = seed_payload["portopt"]["results"]
    else:
        payload["po-results-store"] = {}
    if seed_payload["features"]["session_id_store"]:
        payload["dashmat-session-id-store"] = seed_payload["session_id"]
    return payload


def make_regression_storage_seed(seed_payload: dict[str, Any], *, include_results: bool = True) -> dict[str, Any]:
    reg = seed_payload["regression"]
    payload = {
        "dashmat-raw-data-store": seed_payload["raw_store"],
        "dashmat-raw-data-meta-store": seed_payload["raw_meta"],
        "dashmat-original-periodicity-store": "daily",
        "reg-page-visited-store": True,
        "reg-periodicity-value-store": "daily",
        "reg-series-select": reg["selected_series"],
        "reg-series-select-value-store": reg["selected_series"],
        "reg-series-order-store": reg["selected_series"],
        "reg-dependent-var-store": reg["dependent_var"],
        "reg-active-tab-store": "anova",
        "reg-model-store": "ols",
        "reg-regression-name-store": "Benchmark OLS",
        "reg-window-type-store": "full",
    }
    if include_results:
        payload["reg-results-store"] = reg["results"]
    else:
        payload["reg-results-store"] = {}
    if seed_payload["features"]["session_id_store"]:
        payload["dashmat-session-id-store"] = seed_payload["session_id"]
    return payload


def seed_session_storage(page, storage_seed: dict[str, Any]) -> None:
    page.evaluate(
        """
        (payload) => {
          Object.entries(payload).forEach(([key, value]) => {
            window.sessionStorage.setItem(key, JSON.stringify(value));
          });
        }
        """,
        storage_seed,
    )


def push_live_store_seed(page, storage_seed: dict[str, Any]) -> None:
    for key, value in storage_seed.items():
        set_component_props(page, key, {"data": value})


def configure_portopt_run_inputs(page) -> None:
    set_component_value(page, "po-periodicity-select", "daily")
    set_component_value(page, "po-opt-window-select", "full")
    set_component_value(page, "po-opt-model-select", "minimize_variance")
    set_component_props(page, "po-portfolio-name-input", {"value": "Benchmark MinVar"})
    set_component_props(page, "po-exp-wt-cov-switch", {"checked": False})
    set_component_props(page, "po-halflife-input", {"value": 63})
    set_component_value(page, "po-cov-shrinkage-select", "none")
    set_component_value(page, "po-cov-shrinkage-target-select", "scaled_identity")
    set_component_value(page, "po-missing-data-select", "fill_na")
    set_component_value(page, "po-fill-in-sample-select", "off")
    set_component_value(page, "po-objective-select", "maximize_sharpe")


def configure_regression_run_inputs(page) -> None:
    set_component_value(page, "reg-periodicity-select", "daily")
    set_component_props(page, "reg-regression-name-input", {"value": "Benchmark OLS"})


def _normalize_text(value: str | None) -> str:
    return str(value or "").strip()


def wait_for_non_empty_text(page, selector: str, timeout: int = 120000) -> str:
    page.wait_for_function(
        """
        (sel) => {
          const el = document.querySelector(sel);
          return !!el && !!String(el.textContent || "").trim();
        }
        """,
        arg=selector,
        timeout=timeout,
    )
    return _normalize_text(page.locator(selector).text_content(timeout=5000))


def _wait_for_portopt_run_outcome(page, timeout: int = 120000) -> str:
    return wait_for_non_empty_text(page, "#po-completion-text", timeout=timeout)


def _wait_for_regression_run_outcome(page, timeout: int = 120000) -> str:
    return wait_for_non_empty_text(page, "#reg-run-status-text", timeout=timeout)


def measure_tab_open(page, *, tab_component: str, tab_value: str, container: str, switch_id: str | None = None, switch_value: str = "chart") -> int:
    start = time.perf_counter()
    set_component_value(page, tab_component, tab_value)
    if switch_id:
        set_component_value(page, switch_id, switch_value)
    wait_content_ready(page, container, timeout=60000)
    return round((time.perf_counter() - start) * 1000)


def summarize_numeric_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"runs": len(runs)}
    if not runs:
        return summary
    numeric_keys = sorted(
        {
            key
            for run in runs
            for key, value in run.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    for key in numeric_keys:
        values = [int(run[key]) for run in runs if key in run]
        summary[key] = values
        summary[f"{key[:-2] if key.endswith('Ms') else key}Median"] = round(median(values))
    storage_runs = [run.get("storageBytes", {}) for run in runs]
    if storage_runs:
        median_bytes: dict[str, int] = {}
        for store_key in sorted({k for item in storage_runs for k in item.keys()}):
            values = [int(item.get(store_key, 0)) for item in storage_runs]
            median_bytes[store_key] = round(median(values))
        summary["storageBytesPerRun"] = storage_runs
        summary["storageBytesMedian"] = median_bytes
    return summary


def _store_bytes_subset(browser_storage: dict[str, Any]) -> dict[str, int]:
    item_bytes = ((browser_storage or {}).get("sessionStorage") or {}).get("itemBytes") or {}
    keys = [
        "po-results-store",
        "reg-results-store",
        "dashmat-raw-data-store",
        "dashmat-saved-series-cache-store",
    ]
    return {key: int(item_bytes.get(key, 0)) for key in keys}


def measure_portopt_context(page, base_url: str, storage_seed: dict[str, Any], mode: str) -> dict[str, Any]:
    page.goto(base_url, wait_until="domcontentloaded")
    seed_session_storage(page, storage_seed)
    start = time.perf_counter()
    page.goto(base_url + "/portopt", wait_until="domcontentloaded")
    wait_visible(page, "#po-main-container")
    wait_ready(page, "#po-periodicity-select", timeout=60000)
    page_ready_ms = round((time.perf_counter() - start) * 1000)
    try:
        ensure_modal_hidden(page, "#po-series-selection-modal")
    except Exception:
        if mode != "run_and_consume":
            raise
        force_modal_closed(page, "po-series-selection-modal", "#po-series-selection-modal")
    push_live_store_seed(page, storage_seed)
    if mode == "run_and_consume":
        configure_portopt_run_inputs(page)
        page.wait_for_timeout(500)
        wait_ready(page, "#po-run-button", timeout=60000)
        set_component_props(page, "po-run-button", {"n_clicks": 1})
        completion_text = _wait_for_portopt_run_outcome(page, timeout=120000)
        close_button = page.locator("#po-close-completion-button")
        if close_button.is_visible(timeout=1000):
            close_button.click(force=True)
        else:
            set_component_props(page, "po-progress-modal", {"opened": False})
        if "created successfully" not in completion_text.lower():
            raise RuntimeError(f"PortOpt run failed: {completion_text}")
    wait_content_ready(page, "#po-weight-chart-content", timeout=60000)
    metrics = {
        "pageReadyMs": page_ready_ms,
        "initialDefaultContentReadyMs": round((time.perf_counter() - start) * 1000),
    }
    for spec in PORTOPT_TAB_SPECS:
        metrics[spec["metric"]] = measure_tab_open(
            page,
            tab_component="po-vis-tabs",
            tab_value=spec["tab"],
            container=spec["container"],
            switch_id=spec.get("switch_id"),
            switch_value=spec.get("switch_value", "chart"),
        )
    set_component_value(page, "po-vis-tabs", "returns")
    set_component_props(page, "po-active-tab-store", {"data": "returns"})
    wait_content_ready(page, "#po-returns-grid-content", timeout=60000)
    navigate_path(page, "/")
    revisit_start = time.perf_counter()
    navigate_path(page, "/portopt")
    wait_visible(page, "#po-main-container")
    wait_ready(page, "#po-periodicity-select", timeout=60000)
    metrics["revisitReadyMs"] = round((time.perf_counter() - revisit_start) * 1000)
    wait_content_ready(page, "#po-returns-grid-content", timeout=60000)
    metrics["defaultContentReadyMs"] = round((time.perf_counter() - revisit_start) * 1000)
    browser_storage = collect_browser_storage_metrics(page)
    metrics["storageBytes"] = _store_bytes_subset(browser_storage)
    metrics["sessionStorageTotalBytes"] = int(((browser_storage or {}).get("sessionStorage") or {}).get("totalBytes") or 0)
    return metrics


def measure_regression_context(page, base_url: str, storage_seed: dict[str, Any], mode: str) -> dict[str, Any]:
    page.goto(base_url, wait_until="domcontentloaded")
    seed_session_storage(page, storage_seed)
    start = time.perf_counter()
    page.goto(base_url + "/regression", wait_until="domcontentloaded")
    wait_visible(page, "#reg-main-container")
    wait_ready(page, "#reg-periodicity-select", timeout=60000)
    page_ready_ms = round((time.perf_counter() - start) * 1000)
    try:
        ensure_modal_hidden(page, "#reg-series-selection-modal")
    except Exception:
        if mode != "run_and_consume":
            raise
        force_modal_closed(page, "reg-series-selection-modal", "#reg-series-selection-modal")
    push_live_store_seed(page, storage_seed)
    if mode == "run_and_consume":
        configure_regression_run_inputs(page)
        page.wait_for_timeout(300)
        wait_ready(page, "#reg-run-button", timeout=60000)
        set_component_props(page, "reg-run-button", {"n_clicks": 1})
        run_status = _wait_for_regression_run_outcome(page, timeout=120000)
        if not run_status.startswith("✓"):
            raise RuntimeError(f"Regression run failed: {run_status}")
    wait_content_ready(page, "#reg-anova-content", timeout=60000)
    metrics = {
        "pageReadyMs": page_ready_ms,
        "initialDefaultContentReadyMs": round((time.perf_counter() - start) * 1000),
    }
    for spec in REGRESSION_TAB_SPECS:
        metrics[spec["metric"]] = measure_tab_open(
            page,
            tab_component="reg-tabs",
            tab_value=spec["tab"],
            container=spec["container"],
        )
    set_component_value(page, "reg-tabs", "returns")
    set_component_props(page, "reg-active-tab-store", {"data": "returns"})
    wait_content_ready(page, "#reg-returns-content", timeout=60000)
    navigate_path(page, "/")
    revisit_start = time.perf_counter()
    navigate_path(page, "/regression")
    wait_visible(page, "#reg-main-container")
    wait_ready(page, "#reg-periodicity-select", timeout=60000)
    metrics["revisitReadyMs"] = round((time.perf_counter() - revisit_start) * 1000)
    wait_content_ready(page, "#reg-returns-content", timeout=60000)
    metrics["defaultContentReadyMs"] = round((time.perf_counter() - revisit_start) * 1000)
    browser_storage = collect_browser_storage_metrics(page)
    metrics["storageBytes"] = _store_bytes_subset(browser_storage)
    metrics["sessionStorageTotalBytes"] = int(((browser_storage or {}).get("sessionStorage") or {}).get("totalBytes") or 0)
    return metrics


def run_harness(base_url: str, runs: int, label: str, mode: str, headed: bool, repo_root: Path) -> dict[str, Any]:
    console_messages: list[dict[str, str]] = []
    seed_payload = build_seed_payloads(repo_root, str(uuid4()))
    portopt_runs: list[dict[str, Any]] = []
    regression_runs: list[dict[str, Any]] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not headed)

        def register_page_handlers(page) -> None:
            def on_console(msg) -> None:
                if len(console_messages) >= 200:
                    return
                if msg.type in {"error", "warning"}:
                    console_messages.append({"type": msg.type, "text": msg.text})

            def on_page_error(err) -> None:
                if len(console_messages) >= 200:
                    return
                console_messages.append({"type": "pageerror", "text": str(err)})

            page.on("console", on_console)
            page.on("pageerror", on_page_error)

        for _ in range(runs):
            port_context = browser.new_context(viewport={"width": 1440, "height": 960})
            port_page = port_context.new_page()
            register_page_handlers(port_page)
            run_seed = build_seed_payloads(repo_root, str(uuid4()))
            portopt_runs.append(
                measure_portopt_context(
                    port_page,
                    base_url,
                    make_portopt_storage_seed(run_seed, include_results=(mode == "consume_only")),
                    mode,
                )
            )
            port_context.close()

            reg_context = browser.new_context(viewport={"width": 1440, "height": 960})
            reg_page = reg_context.new_page()
            register_page_handlers(reg_page)
            reg_seed = build_seed_payloads(repo_root, str(uuid4()))
            regression_runs.append(
                measure_regression_context(
                    reg_page,
                    base_url,
                    make_regression_storage_seed(reg_seed, include_results=(mode == "consume_only")),
                    mode,
                )
            )
            reg_context.close()

        browser.close()

    return {
        "ok": True,
        "label": label,
        "mode": mode,
        "baseUrl": base_url,
        "dataset": seed_payload["dataset_meta"],
        "portopt": summarize_numeric_runs(portopt_runs),
        "regression": summarize_numeric_runs(regression_runs),
        "artifactStore": collect_artifact_store_metrics(repo_root),
        "consoleMessages": console_messages,
        "repoFeatures": seed_payload["features"],
    }


def write_failure_artifacts(
    *,
    out_dir: Path,
    fail_dir: Path,
    stem: str,
    repo_root: Path,
    base_url: str,
    git_ref: str,
    label: str,
    mode: str,
    startup_timeout: int,
    console_messages: list[dict[str, str]] | None,
    exc: Exception,
) -> Path:
    raw_path = out_dir / f"{stem}_traceback.txt"
    raw_path.write_text(traceback.format_exc(), encoding="utf-8")
    status = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "label": label,
        "gitRef": git_ref,
        "mode": mode,
        "baseUrl": base_url,
        "repoRoot": str(repo_root),
        "startupTimeout": startup_timeout,
        "error": str(exc),
        "consoleMessages": console_messages or [],
        "tracebackPath": str(raw_path),
    }
    status_path = fail_dir / f"{stem}_status.json"
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    return raw_path


def main() -> int:
    args = parse_args()
    root = resolve_repo_root(args.repo_root)
    out_dir = root / "output" / "playwright"
    fail_dir = out_dir / "failures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fail_dir.mkdir(parents=True, exist_ok=True)
    resolved_git_ref = resolve_git_ref(root, args.git_ref)
    stem = build_artifact_stem(
        prefix=f"result_flow_{args.mode}",
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
            mode=args.mode,
            headed=args.headed,
            repo_root=root,
        )
    except Exception as exc:
        raw_path = write_failure_artifacts(
            out_dir=out_dir,
            fail_dir=fail_dir,
            stem=stem,
            repo_root=root,
            base_url=args.base_url,
            git_ref=resolved_git_ref,
            label=args.label,
            mode=args.mode,
            startup_timeout=args.startup_timeout,
            console_messages=[],
            exc=exc,
        )
        print(f"RAW_PATH={raw_path}")
        return 1

    out_path = out_dir / f"{stem}.json"
    payload = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "label": result["label"],
        "mode": result["mode"],
        "gitRef": resolved_git_ref,
        "baseUrl": result["baseUrl"],
        "repoRoot": str(root),
        "runs": args.runs,
        "dataset": result["dataset"],
        "repoFeatures": result["repoFeatures"],
        "portopt": result["portopt"],
        "regression": result["regression"],
        "artifactStore": result["artifactStore"],
        "consoleMessages": result["consoleMessages"],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"OUT_PATH={out_path}")
    print("PORTOPT=" + json.dumps(result["portopt"], separators=(",", ":")))
    print("REGRESSION=" + json.dumps(result["regression"], separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
