"""AT Statistics Harness — measures AT Statistics-ready timing via imports or account-list loads.

Two measurement modes against 5 deterministic peer/index run specs:
  - imports:       clear session, re-import each peer/index pair through the AT UI
  - account-list:  clear session, load a pre-seeded account list containing each pair

Reuses warm-switch helpers for AA DB warmup, AT state-ready waiting, statistics-idle
waiting, Dash request/bytes/callback attribution, and failure artifacts.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import median

from playwright.sync_api import sync_playwright

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.account_lists import (
    add_db_import_provenance_entry,
    build_account_list_payload,
    delete_account_list,
    list_account_lists,
    save_account_list,
)
from utils.portfolio_series import get_portfolio_options, has_portfolio_benchmark, load_portfolio_series
from dbengine import engine as DB_ENGINE

from tools.playwright.warm_switch_harness import (
    DashUpdateRequestTracker,
    build_artifact_stem,
    copy_server_log,
    current_log_offset,
    ensure_local_seed_databases,
    parse_timing_log,
    resolve_git_ref,
    resolve_repo_root,
    sanitize_token,
    summarize_dash_update_runs,
    wait_analytics_statistics_idle,
    wait_dash_hydrated,
    wait_for_analytics_state_ready,
    wait_for_app,
    wait_for_persisted_store_value,
    wait_ready,
    wait_visible,
    warm_analytics_db,
    write_failure_artifacts,
    get_persisted_store_value,
    set_component_props,
    set_persisted_store_value,
    fire_component_click,
    TIMING_EVENT_NAMES,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HARNESS_PREFIX = "at-stats-harness"
REQUIRED_PEER_COUNT = 5
REQUIRED_INDEX_COUNT = 5

DEFAULT_DB_SERIES = [
    "SPX_TRIndex",
    "R2000_TRIndex",
    "EAFE_TRIndex",
    "BCTBill13_TRIndex",
]

NETWORK_PROFILES = {
    "none": None,
    "office-wan": {
        "latencyMs": 40,
        "downloadKbps": 10000,
        "uploadKbps": 5000,
        "connectionType": "cellular4g",
    },
    "slow4g": {
        "latencyMs": 150,
        "downloadKbps": 4000,
        "uploadKbps": 3000,
        "connectionType": "cellular4g",
    },
    "fast3g": {
        "latencyMs": 150,
        "downloadKbps": 1600,
        "uploadKbps": 750,
        "connectionType": "cellular3g",
    },
}

NETWORK_TIMEOUT_MULTIPLIERS = {
    "none": 1.0,
    "office-wan": 3.0,
    "slow4g": 4.0,
    "fast3g": 5.0,
}

TIMEOUT_SCALE = 1.0
ACCOUNT_LIST_CLICK_TIMING_RE = re.compile(
    r"timing name=account_list\.click_to_ready "
    r"(?:(?:click_to_reload_start_ms=(?P<click_to_reload>\d+))|(?:click_to_live_apply_commit_ms=(?P<click_to_live_apply>\d+))) "
    r"(?:(?:reload_start_to_ready_ms=(?P<reload_to_ready>\d+))|(?:live_apply_commit_to_ready_ms=(?P<live_apply_to_ready>\d+))) "
    r"total_click_to_ready_ms=(?P<total>\d+)"
)


# ---------------------------------------------------------------------------
# Run-spec builder
# ---------------------------------------------------------------------------

def _resolve_eligible_peer_portfolios(engine) -> list[dict]:
    """Return peer portfolios that have Actual returns AND an Estimated peer benchmark.

    The harness hardcodes ``benchmark_type="Estimated"``, so we must verify
    that the specific ``MeanRet|Estimated`` row exists for each candidate's
    vintage — not just any ``MeanRet`` row.
    """
    from sqlalchemy import text as sa_text

    options = get_portfolio_options(engine, "peer")
    eligible: list[dict] = []
    for opt in options:
        portfolio = str(opt.get("value", "")).strip()
        if not portfolio:
            continue
        if not has_portfolio_benchmark(engine, "peer", portfolio):
            continue
        # Verify the specific Estimated benchmark exists for this vintage.
        with engine.connect() as conn:
            row = conn.execute(
                sa_text(
                    "SELECT PeerVintage FROM Portfolios WHERE Portfolio = :portfolio"
                ),
                {"portfolio": portfolio},
            ).first()
        if not row:
            continue
        vintage = str(row[0] or "").strip()
        if not vintage:
            continue
        with engine.connect() as conn:
            count = conn.execute(
                sa_text(
                    "SELECT COUNT(1) FROM PeerTS "
                    "WHERE Item = 'MeanRet' AND [Desc] = 'Estimated' "
                    "AND Portfolio = :vintage"
                ),
                {"vintage": vintage},
            ).scalar()
        if not count or int(count) == 0:
            continue
        eligible.append({"portfolio": portfolio, "label": opt.get("label", portfolio)})
    return sorted(eligible, key=lambda x: x["portfolio"])


def _resolve_eligible_index_portfolios(engine) -> list[dict]:
    """Return index portfolios that have Actual returns AND a ``Benchmark`` desc.

    The harness hardcodes ``benchmark_type="Benchmark"``, so we must verify
    that the specific ``PortRet|Benchmark`` row exists — not just any desc
    in ``INDEX_BENCHMARK_TYPE_OPTIONS`` (which also includes ``Calculated``).
    """
    from sqlalchemy import text as sa_text

    options = get_portfolio_options(engine, "index")
    eligible: list[dict] = []
    for opt in options:
        portfolio = str(opt.get("value", "")).strip()
        if not portfolio:
            continue
        with engine.connect() as conn:
            count = conn.execute(
                sa_text(
                    "SELECT COUNT(1) FROM IndexTS "
                    "WHERE Portfolio = :portfolio "
                    "AND Item = 'PortRet' AND [Desc] = 'Benchmark'"
                ),
                {"portfolio": portfolio},
            ).scalar()
        if not count or int(count) == 0:
            continue
        eligible.append({"portfolio": portfolio, "label": opt.get("label", portfolio)})
    return sorted(eligible, key=lambda x: x["portfolio"])


def build_run_specs(engine) -> list[dict]:
    """Build 5 deterministic run specs, each pairing one peer + one index portfolio.

    Peer portfolios are imported as Actual + Estimated (benchmark).
    Index portfolios are imported as Actual + Benchmark.
    """
    peers = _resolve_eligible_peer_portfolios(engine)
    indices = _resolve_eligible_index_portfolios(engine)

    if len(peers) < REQUIRED_PEER_COUNT:
        raise RuntimeError(
            f"Need at least {REQUIRED_PEER_COUNT} eligible peer portfolios, got {len(peers)}: "
            f"{[p['portfolio'] for p in peers]}"
        )
    if len(indices) < REQUIRED_INDEX_COUNT:
        raise RuntimeError(
            f"Need at least {REQUIRED_INDEX_COUNT} eligible index portfolios, got {len(indices)}: "
            f"{[i['portfolio'] for i in indices]}"
        )

    specs: list[dict] = []
    for i in range(5):
        peer = peers[i]
        index = indices[i]
        specs.append({
            "specIndex": i,
            "peer": {
                "portfolio": peer["portfolio"],
                "type": "Actual",
                "include_benchmark": True,
                "benchmark_type": "Estimated",
            },
            "index": {
                "portfolio": index["portfolio"],
                "type": "Actual",
                "include_benchmark": True,
                "benchmark_type": "Benchmark",
            },
        })
    return specs


def _staged_row(portfolio: str, ret_type: str, include_benchmark: bool, benchmark_type: str) -> dict:
    """Build a staged row dict matching the portfolio import modal format."""
    return {
        "Portfolio": portfolio,
        "Type": ret_type,
        "Include Benchmark": "Yes" if include_benchmark else "No",
        "Benchmark Type": benchmark_type if include_benchmark else "",
        "portfolio": portfolio,
        "type": ret_type,
        "include_benchmark": include_benchmark,
        "benchmark_type": benchmark_type if include_benchmark else "",
    }


def _kbps_to_bytes_per_second(kbps: int) -> int:
    return max(int(kbps * 1000 / 8), 1)


def _scaled_timeout(timeout_ms: int) -> int:
    return max(int(round(timeout_ms * TIMEOUT_SCALE)), timeout_ms)


def _apply_network_profile(page, profile_name: str) -> dict | None:
    profile = NETWORK_PROFILES.get(profile_name)
    if not profile:
        return None
    session = page.context.new_cdp_session(page)
    session.send("Network.enable")
    session.send(
        "Network.emulateNetworkConditions",
        {
            "offline": False,
            "latency": int(profile["latencyMs"]),
            "downloadThroughput": _kbps_to_bytes_per_second(int(profile["downloadKbps"])),
            "uploadThroughput": _kbps_to_bytes_per_second(int(profile["uploadKbps"])),
            "connectionType": str(profile["connectionType"]),
        },
    )
    return {"name": profile_name, **profile}


# ---------------------------------------------------------------------------
# Account-list fixture builder
# ---------------------------------------------------------------------------

def _build_account_list_fixture_payload(spec: dict, engine) -> dict:
    """Build a complete account-list payload for one run spec's peer/index pair.

    Uses ``load_portfolio_series`` to resolve the real emitted column names
    (including the correct peer vintage benchmark) so the fixture exactly
    matches what the import-mode UI would produce.
    """
    peer = spec["peer"]
    index_port = spec["index"]

    peer_portfolio = peer["portfolio"]
    peer_type = peer["type"]
    peer_bm_type = peer["benchmark_type"]
    index_portfolio = index_port["portfolio"]
    index_type = index_port["type"]
    index_bm_type = index_port["benchmark_type"]

    # Resolve real emitted series via load_portfolio_series
    peer_row = _staged_row(peer_portfolio, peer_type, True, peer_bm_type)
    peer_result = load_portfolio_series(engine, "peer", [peer_row])
    peer_cols = list(peer_result.returns_df.columns)
    peer_benchmarks = peer_result.benchmark_assignments

    index_row = _staged_row(index_portfolio, index_type, True, index_bm_type)
    index_result = load_portfolio_series(engine, "index", [index_row])
    index_cols = list(index_result.returns_df.columns)
    index_benchmarks = index_result.benchmark_assignments

    if not peer_cols:
        raise RuntimeError(f"load_portfolio_series returned no columns for peer {peer_portfolio}")
    if not index_cols:
        raise RuntimeError(f"load_portfolio_series returned no columns for index {index_portfolio}")

    peer_primary = peer_cols[0]

    # Build provenance entries using the real emitted series
    provenance: dict = {}
    provenance = add_db_import_provenance_entry(
        provenance,
        loader_type="portfolio_peer",
        loader_args={"rows": [peer_row]},
        emitted_series=peer_cols,
        primary_series=peer_primary,
    )

    index_primary = index_cols[0]
    provenance = add_db_import_provenance_entry(
        provenance,
        loader_type="portfolio_index",
        loader_args={"rows": [index_row]},
        emitted_series=index_cols,
        primary_series=index_primary,
    )

    all_series = peer_cols + index_cols
    benchmark_map = {**peer_benchmarks, **index_benchmarks}

    session_snapshot = {
        "at-series-select": list(all_series),
        "at-series-order-store": list(all_series),
        "at-benchmark-assignments-store": benchmark_map,
        "at-long-short-store": {},
        "at-vol-scaling-assignments-store": {name: True for name in all_series},
        "at-periodicity-value-store": "daily",
        "at-returns-type-value-store": "total",
        "at-active-tab-store": "statistics",
        "at-date-range-store": None,
        "at-vol-scaler-value-store": 0,
    }

    return build_account_list_payload(provenance, session_snapshot)


def create_account_list_fixtures(engine, username: str, specs: list[dict]) -> list[dict]:
    """Create account-list fixtures for all run specs. Returns fixture metadata."""
    # Delete existing harness-prefixed lists for this user
    existing = list_account_lists(engine, username)
    for row in existing:
        list_name = str(row.get("ListName") or "")
        if list_name.startswith(HARNESS_PREFIX):
            delete_account_list(
                engine,
                account_list_id=row["AccountListID"],
                username=username,
            )

    fixtures: list[dict] = []
    for spec in specs:
        list_name = f"{HARNESS_PREFIX}-spec{spec['specIndex']}"
        payload = _build_account_list_fixture_payload(spec, engine)
        ok, message, saved = save_account_list(
            engine,
            username=username,
            update_by=username,
            list_name=list_name,
            payload=payload,
        )
        if not ok:
            raise RuntimeError(f"Failed to save account-list fixture '{list_name}': {message}")
        fixtures.append({
            "specIndex": spec["specIndex"],
            "listName": list_name,
            "accountListId": saved.get("AccountListID") if saved else None,
        })

    return fixtures


# ---------------------------------------------------------------------------
# AT UI interaction helpers
# ---------------------------------------------------------------------------

def _clear_session_and_reload(page, base_url: str) -> dict[str, float]:
    """Clear sessionStorage, reload /analyticstool, and wait for welcome screen."""
    page.evaluate("() => { window.sessionStorage.clear(); }")
    reload_start_ts = time.perf_counter()
    page.goto(base_url + "/analyticstool", wait_until="domcontentloaded", timeout=_scaled_timeout(30000))
    page.wait_for_function(
        """
        () => {
          const welcome = document.querySelector("#at-welcome-screen-container");
          if (!welcome) {
            return false;
          }
          const title = (document.title || "").trim();
          if (!title || title === "Updating...") {
            return false;
          }
          const visible = (selector) => {
            const element = document.querySelector(selector);
            if (!element) {
              return false;
            }
            const style = window.getComputedStyle(element);
            if (style.display === "none" || style.visibility === "hidden") {
              return false;
            }
            const rect = element.getBoundingClientRect();
            return rect.width > 0 && rect.height > 0;
          };
          const cards = document.querySelectorAll(
            "#at-welcome-screen-container .dashmat-welcome-section-card"
          );
          return (
            visible("#at-welcome-screen-container")
            && cards.length >= 5
            && visible("#at-welcome-add-db-btn")
            && visible("#at-welcome-add-portfolios-peer-btn")
            && visible("#at-welcome-add-series-btn")
            && visible("#at-welcome-load-account-list-btn")
          );
        }
        """,
        timeout=_scaled_timeout(30000),
    )
    hydrated_ts = time.perf_counter()
    wait_visible(page, "#at-welcome-screen-container", timeout=_scaled_timeout(30000))
    welcome_visible_ts = time.perf_counter()
    return {
        "reloadStartTs": reload_start_ts,
        "hydratedTs": hydrated_ts,
        "welcomeVisibleTs": welcome_visible_ts,
    }



def _stage_portfolio_row(page, row: dict) -> None:
    """Stage one portfolio row in the portfolio import modal."""
    portfolio = row["portfolio"]
    ret_type = row["type"]
    include_benchmark = row["include_benchmark"]
    benchmark_type = row.get("benchmark_type", "")

    set_component_props(page, "at-portfolio-add-series-select", {"value": portfolio})
    page.wait_for_timeout(300)
    set_component_props(page, "at-portfolio-add-type-select", {"value": ret_type})
    if include_benchmark:
        set_component_props(page, "at-portfolio-add-include-benchmark", {"checked": True})
        page.wait_for_timeout(200)
        set_component_props(page, "at-portfolio-add-benchmark-type-select", {"value": benchmark_type})
    else:
        set_component_props(page, "at-portfolio-add-include-benchmark", {"checked": False})

    page.wait_for_timeout(200)
    wait_ready(page, "#at-portfolio-add-row-btn", timeout=_scaled_timeout(10000))
    fire_component_click(page, "at-portfolio-add-row-btn")
    # Wait for the row to appear in the staged-rows store and OK to become enabled
    page.wait_for_timeout(500)
    wait_ready(page, "#at-portfolio-add-ok-button", timeout=_scaled_timeout(10000))



def _wait_statistics_ready(page, timeout: int = 60000) -> None:
    """Wait for AT state-ready and statistics grid idle."""
    wait_for_analytics_state_ready(page, timeout=_scaled_timeout(timeout))


def _open_portfolio_import_modal(page, mode: str) -> None:
    """Open the portfolio import modal with retry logic.

    Uses ``fire_component_click`` (``set_props(n_clicks=…)``) to trigger the
    Dash callback reliably, retrying up to 3 times with Dash-hydrated waits
    between attempts.
    """
    if mode == "peer":
        welcome_id = "at-welcome-add-portfolios-peer-btn"
        menu_id = "at-menu-add-portfolios-peer"
    else:
        welcome_id = "at-welcome-add-portfolios-index-btn"
        menu_id = "at-menu-add-portfolios-index"

    modal_ready = False
    for _attempt in range(3):
        # Use fire_component_click which works regardless of element visibility;
        # try welcome button first, then fall back to the menu trigger.
        welcome_btn = page.locator(f"#{welcome_id}")
        triggered = False
        try:
            if welcome_btn.count() > 0 and welcome_btn.is_visible(timeout=_scaled_timeout(1000)):
                triggered = fire_component_click(page, welcome_id)
        except Exception:
            pass
        if not triggered:
            triggered = fire_component_click(page, menu_id)

        try:
            # Mantine Modals render via a portal; the root #at-portfolio-add-modal
            # may not appear as a queryable element. Wait for the OK button instead.
            # The OK button starts disabled (no rows staged), so only check visibility.
            wait_visible(page, "#at-portfolio-add-ok-button", timeout=_scaled_timeout(10000))
            modal_ready = True
            break
        except Exception:
            wait_dash_hydrated(page, timeout=_scaled_timeout(10000))

    if not modal_ready:
        raise RuntimeError(
            f"Portfolio import modal ({mode}) did not become ready after 3 attempts."
        )


def _import_portfolio(page, mode: str, row: dict) -> dict:
    """Import a single portfolio via the AT UI (open modal, stage, confirm).

    Returns a dict of checkpoint timestamps (perf_counter values) so the
    caller can compute granular timing windows::

        modalOpenTs:       after modal is visible and ready for staging
        seriesModalTs:     after portfolio OK is clicked and series modal appears
        seriesConfirmedTs: after series modal OK is clicked
    """
    _open_portfolio_import_modal(page, mode)
    _stage_portfolio_row(page, row)
    modal_open_ts = time.perf_counter()

    # Click portfolio OK → wait for series selection modal
    wait_ready(page, "#at-portfolio-add-ok-button", timeout=_scaled_timeout(10000))
    page.locator("#at-portfolio-add-ok-button").click(force=True)
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            if not page.locator("#at-portfolio-add-ok-button").is_visible(timeout=_scaled_timeout(500)):
                break
        except Exception:
            break
        time.sleep(0.2)
    wait_visible(page, "#at-modal-ok-button", timeout=_scaled_timeout(30000))
    series_modal_ts = time.perf_counter()

    # Confirm series selection
    wait_ready(page, "#at-modal-ok-button", timeout=_scaled_timeout(10000))
    page.locator("#at-modal-ok-button").click(force=True)
    series_confirmed_ts = time.perf_counter()

    return {
        "modalOpenTs": modal_open_ts,
        "seriesModalTs": series_modal_ts,
        "seriesConfirmedTs": series_confirmed_ts,
    }


def _last_dash_response_to_ready_ms(
    window_summary: dict[str, object],
    window_start_at: float | None,
    ready_ts: float,
    fallback_start_ts: float,
) -> int:
    if window_start_at is None:
        return max(0, round((ready_ts - fallback_start_ts) * 1000))

    last_finished_offset_ms = window_summary.get("dashUpdateLastFinishedOffsetMs")
    if last_finished_offset_ms is None:
        return max(0, round((ready_ts - fallback_start_ts) * 1000))

    last_finished_ts = window_start_at + (float(last_finished_offset_ms) / 1000.0)
    return max(0, round((ready_ts - last_finished_ts) * 1000))


# ---------------------------------------------------------------------------
# Account-list load helpers
# ---------------------------------------------------------------------------

def _open_welcome_account_list_load(page) -> None:
    """Open the account-list modal in load mode from the AT welcome screen."""
    wait_visible(page, "#at-welcome-load-account-list-btn", timeout=_scaled_timeout(10000))
    modal_ready = False
    for _attempt in range(3):
        triggered = fire_component_click(page, "at-welcome-load-account-list-btn")
        if not triggered:
            page.locator("#at-welcome-load-account-list-btn").click(force=True)
        try:
            wait_visible(page, "#dashmat-account-list-modal-root", timeout=_scaled_timeout(10000))
            modal_ready = True
            break
        except Exception:
            wait_dash_hydrated(page, timeout=_scaled_timeout(10000))
    if not modal_ready:
        raise RuntimeError("Could not open account-list modal from AT welcome screen.")


def _select_and_load_account_list(page, fixture: dict) -> dict[str, float | None]:
    """Select and load an account list through the modal's real callback flow.

    Waits for the modal's row-fetch callback to populate the grid, then
    selects the target row in the AG Grid (which triggers the detail-load
    callback), waits for the detail to resolve, and clicks Load.
    """
    account_list_id = fixture["accountListId"]

    # 1. Wait for the modal's _refresh_account_list_rows callback to populate
    #    the grid via the real callback chain.  The AG Grid eventually renders
    #    rows from dashmat-account-list-grid-rows-store.  We detect that by
    #    waiting for the grid to contain at least one DOM row element.
    page.wait_for_function(
        """
        () => {
          const grid = document.querySelector('#dashmat-account-list-grid');
          if (!grid) return false;
          return grid.querySelectorAll('.ag-row').length > 0;
        }
        """,
        timeout=_scaled_timeout(15000),
    )

    # 2. Select the target row by setting the AG Grid selectedRows property.
    #    This triggers _sync_account_list_selected_id → selected-id-store →
    #    _refresh_account_list_selected_detail → selected-detail-store.
    #    We read the current rows from the grid's rowData to find our target.
    rows = page.evaluate(
        """
        () => {
          const grid = document.querySelector('#dashmat-account-list-grid');
          if (!grid || !grid.props) return [];
          return grid.props.rowData || [];
        }
        """
    )
    if not isinstance(rows, list) or not rows:
        # Fallback: read from Dash store
        rows = list_account_lists(DB_ENGINE, _get_username(page))
    target_row = next(
        (r for r in rows if r.get("AccountListID") == account_list_id),
        None,
    )
    if target_row is None:
        raise RuntimeError(
            f"Account list ID {account_list_id} not found in grid rows. "
            f"Available: {[r.get('AccountListID') for r in rows]}"
        )
    set_component_props(page, "dashmat-account-list-grid", {"selectedRows": [target_row]})

    # 3. Wait for the Load button to become enabled (signals that both the
    #    selected-id and selected-detail callbacks have completed).
    wait_ready(page, "#dashmat-account-list-load-button", timeout=_scaled_timeout(15000))

    # 4. Click Load through the real callback. AnalyticsTool uses same-page
    #    live apply, so waiting for a navigation event here can add a false
    #    30s timeout to the measured path.
    load_clicked_ts = time.perf_counter()
    triggered = fire_component_click(page, "dashmat-account-list-load-button")
    if not triggered:
        page.locator("#dashmat-account-list-load-button").click(force=True)
    return {
        "loadClickedTs": load_clicked_ts,
        "reloadStartTs": None,
    }


def _get_username(page) -> str:
    """Resolve username from the page's userinfo store."""
    userinfo = get_persisted_store_value(page, "userinfo") or {}
    username = str((userinfo or {}).get("username") or "").strip()
    if not username:
        raise RuntimeError("Could not resolve username from userinfo store.")
    return username


def _parse_account_list_click_timing(message: str) -> dict[str, int | str] | None:
    match = ACCOUNT_LIST_CLICK_TIMING_RE.search(str(message or "").strip())
    if not match:
        return None
    click_to_reload = match.group("click_to_reload")
    reload_to_ready = match.group("reload_to_ready")
    click_to_live_apply = match.group("click_to_live_apply")
    live_apply_to_ready = match.group("live_apply_to_ready")
    payload: dict[str, int | str] = {
        "totalClickToReadyMs": int(match.group("total")),
    }
    if click_to_reload is not None and reload_to_ready is not None:
        payload["mode"] = "reload"
        payload["clickToReloadStartMs"] = int(click_to_reload)
        payload["reloadStartToReadyMs"] = int(reload_to_ready)
        return payload
    if click_to_live_apply is not None and live_apply_to_ready is not None:
        payload["mode"] = "live_apply"
        payload["clickToLiveApplyCommitMs"] = int(click_to_live_apply)
        payload["liveApplyCommitToReadyMs"] = int(live_apply_to_ready)
        return payload
    return None


# ---------------------------------------------------------------------------
# Measurement flows
# ---------------------------------------------------------------------------

def _measure_import_run(
    page,
    request_tracker: DashUpdateRequestTracker,
    base_url: str,
    spec: dict,
    server_log: Path | None,
) -> dict:
    """Measure a single import-mode run for one spec."""
    # Reset to welcome
    t0 = time.perf_counter()
    reset_timing_start = current_log_offset(server_log)
    request_tracker.start_window()
    reset_window_start_at = request_tracker.window_start_at
    reset_ts = _clear_session_and_reload(page, base_url)
    request_tracker.wait_for_settle()
    request_tracker.stop_window()
    reset_timing_end = current_log_offset(server_log)
    reset_ms = round((time.perf_counter() - t0) * 1000)
    reset_reload_start_to_hydrated_ms = round((reset_ts["hydratedTs"] - reset_ts["reloadStartTs"]) * 1000)
    reset_hydrated_to_welcome_visible_ms = round((reset_ts["welcomeVisibleTs"] - reset_ts["hydratedTs"]) * 1000)
    reset_summary = request_tracker.summary()
    reset_summary["timingSummary"] = parse_timing_log(
        server_log,
        start_offset=reset_timing_start,
        end_offset=reset_timing_end,
    )
    reset_last_response_to_welcome_ms = _last_dash_response_to_ready_ms(
        reset_summary,
        reset_window_start_at,
        reset_ts["welcomeVisibleTs"],
        reset_ts["reloadStartTs"],
    )

    # Import peer
    peer_timing_start = current_log_offset(server_log)
    t_peer_start = time.perf_counter()
    request_tracker.start_window()
    peer_window_start_at = request_tracker.window_start_at
    peer_ts = _import_portfolio(page, "peer", spec["peer"])
    # modalOpenTs → seriesModalTs: staging + OK click → series modal appears
    peer_import_to_series_modal_ms = round((peer_ts["seriesModalTs"] - t_peer_start) * 1000)
    # seriesConfirmedTs onward: series OK clicked → statistics ready
    peer_series_confirm_ts = peer_ts["seriesConfirmedTs"]

    # Wait for statistics ready after peer
    _wait_statistics_ready(page)
    peer_ready_ts = time.perf_counter()
    request_tracker.wait_for_settle()
    request_tracker.stop_window()
    peer_timing_end = current_log_offset(server_log)
    peer_confirm_to_ready_ms = round((peer_ready_ts - peer_series_confirm_ts) * 1000)
    peer_summary = request_tracker.summary()
    peer_summary["timingSummary"] = parse_timing_log(
        server_log,
        start_offset=peer_timing_start,
        end_offset=peer_timing_end,
    )
    peer_confirm_phase_summary = request_tracker.summary(start_at=peer_series_confirm_ts)
    peer_last_response_to_ready_ms = _last_dash_response_to_ready_ms(
        peer_confirm_phase_summary,
        peer_window_start_at,
        peer_ready_ts,
        peer_series_confirm_ts,
    )

    # Import index
    index_timing_start = current_log_offset(server_log)
    t_index_start = time.perf_counter()
    request_tracker.start_window()
    index_window_start_at = request_tracker.window_start_at
    index_ts = _import_portfolio(page, "index", spec["index"])
    index_import_to_series_modal_ms = round((index_ts["seriesModalTs"] - t_index_start) * 1000)
    index_series_confirm_ts = index_ts["seriesConfirmedTs"]

    # Wait for statistics ready after index
    _wait_statistics_ready(page)
    index_ready_ts = time.perf_counter()
    request_tracker.wait_for_settle()
    request_tracker.stop_window()
    index_timing_end = current_log_offset(server_log)
    index_confirm_to_ready_ms = round((index_ready_ts - index_series_confirm_ts) * 1000)
    index_summary = request_tracker.summary()
    index_summary["timingSummary"] = parse_timing_log(
        server_log,
        start_offset=index_timing_start,
        end_offset=index_timing_end,
    )
    index_confirm_phase_summary = request_tracker.summary(start_at=index_series_confirm_ts)
    index_last_response_to_ready_ms = _last_dash_response_to_ready_ms(
        index_confirm_phase_summary,
        index_window_start_at,
        index_ready_ts,
        index_series_confirm_ts,
    )

    total_ms = round((time.perf_counter() - t0) * 1000)

    return {
        "specIndex": spec["specIndex"],
        "peerPortfolio": spec["peer"]["portfolio"],
        "indexPortfolio": spec["index"]["portfolio"],
        "resetToWelcomeMs": reset_ms,
        "resetReloadStartToHydratedMs": reset_reload_start_to_hydrated_ms,
        "resetHydratedToWelcomeVisibleMs": reset_hydrated_to_welcome_visible_ms,
        "resetLastDashResponseToWelcomeVisibleMs": reset_last_response_to_welcome_ms,
        "peerImportToSeriesModalMs": peer_import_to_series_modal_ms,
        "peerSeriesConfirmToStatisticsReadyMs": peer_confirm_to_ready_ms,
        "peerLastDashResponseToReadyMs": peer_last_response_to_ready_ms,
        "indexImportToSeriesModalMs": index_import_to_series_modal_ms,
        "indexSeriesConfirmToStatisticsReadyMs": index_confirm_to_ready_ms,
        "indexLastDashResponseToReadyMs": index_last_response_to_ready_ms,
        "totalRunMs": total_ms,
        "resetWindow": reset_summary,
        "peerWindow": peer_summary,
        "peerConfirmPhaseWindow": peer_confirm_phase_summary,
        "indexWindow": index_summary,
        "indexConfirmPhaseWindow": index_confirm_phase_summary,
    }


def _measure_account_list_run(
    page,
    request_tracker: DashUpdateRequestTracker,
    base_url: str,
    spec: dict,
    fixture: dict,
    server_log: Path | None,
    browser_timing_messages: list[str],
) -> dict:
    """Measure a single account-list-mode run for one spec."""
    # Reset to welcome
    t0 = time.perf_counter()
    reset_timing_start = current_log_offset(server_log)
    request_tracker.start_window()
    reset_window_start_at = request_tracker.window_start_at
    reset_ts = _clear_session_and_reload(page, base_url)
    request_tracker.wait_for_settle()
    request_tracker.stop_window()
    reset_timing_end = current_log_offset(server_log)
    reset_ms = round((time.perf_counter() - t0) * 1000)
    reset_reload_start_to_hydrated_ms = round((reset_ts["hydratedTs"] - reset_ts["reloadStartTs"]) * 1000)
    reset_hydrated_to_welcome_visible_ms = round((reset_ts["welcomeVisibleTs"] - reset_ts["hydratedTs"]) * 1000)
    reset_summary = request_tracker.summary()
    reset_summary["timingSummary"] = parse_timing_log(
        server_log,
        start_offset=reset_timing_start,
        end_offset=reset_timing_end,
    )
    reset_last_response_to_welcome_ms = _last_dash_response_to_ready_ms(
        reset_summary,
        reset_window_start_at,
        reset_ts["welcomeVisibleTs"],
        reset_ts["reloadStartTs"],
    )

    # Open account-list load from welcome, select and trigger load
    load_timing_start = current_log_offset(server_log)
    t_load_start = time.perf_counter()
    timing_message_start = len(browser_timing_messages)
    request_tracker.start_window()
    load_window_start_at = request_tracker.window_start_at
    _open_welcome_account_list_load(page)
    load_ts = _select_and_load_account_list(page, fixture)

    # Wait for statistics ready
    try:
        # Account list load triggers a full page reload
        wait_dash_hydrated(page, timeout=_scaled_timeout(60000))
        hydrated_ts = time.perf_counter()
        wait_visible(page, "#at-main-app-container", timeout=_scaled_timeout(60000))
        _wait_statistics_ready(page, timeout=60000)
    except Exception:
        # Fallback: the load may navigate; re-wait
        wait_dash_hydrated(page, timeout=_scaled_timeout(60000))
        hydrated_ts = time.perf_counter()
        _wait_statistics_ready(page, timeout=60000)

    ready_ts = time.perf_counter()
    request_tracker.wait_for_settle()
    request_tracker.stop_window()
    load_timing_end = current_log_offset(server_log)
    load_to_ready_ms = round((ready_ts - t_load_start) * 1000)
    load_summary = request_tracker.summary()
    load_summary["timingSummary"] = parse_timing_log(
        server_log,
        start_offset=load_timing_start,
        end_offset=load_timing_end,
    )
    load_last_response_to_ready_ms = _last_dash_response_to_ready_ms(
        load_summary,
        load_window_start_at,
        ready_ts,
        t_load_start,
    )
    total_ms = round((time.perf_counter() - t0) * 1000)
    browser_click_timing: dict[str, int | str] | None = None
    for message in reversed(browser_timing_messages[timing_message_start:]):
        browser_click_timing = _parse_account_list_click_timing(message)
        if browser_click_timing:
            break
    account_list_hydrated_to_statistics_ready_ms = max(
        0,
        round((ready_ts - hydrated_ts) * 1000),
    )
    load_click_to_reload_start_ms = int(browser_click_timing.get("clickToReloadStartMs", 0)) if browser_click_timing else 0
    account_list_reload_start_to_hydrated_ms = max(
        0,
        int(browser_click_timing.get("reloadStartToReadyMs", 0)) - account_list_hydrated_to_statistics_ready_ms,
    ) if browser_click_timing and browser_click_timing.get("mode") == "reload" else 0
    account_list_load_click_to_live_apply_commit_ms = (
        int(browser_click_timing.get("clickToLiveApplyCommitMs", 0))
        if browser_click_timing and browser_click_timing.get("mode") == "live_apply"
        else 0
    )
    account_list_live_apply_commit_to_ready_ms = (
        int(browser_click_timing.get("liveApplyCommitToReadyMs", 0))
        if browser_click_timing and browser_click_timing.get("mode") == "live_apply"
        else 0
    )

    return {
        "specIndex": spec["specIndex"],
        "peerPortfolio": spec["peer"]["portfolio"],
        "indexPortfolio": spec["index"]["portfolio"],
        "accountListName": fixture["listName"],
        "accountListId": fixture["accountListId"],
        "resetToWelcomeMs": reset_ms,
        "resetReloadStartToHydratedMs": reset_reload_start_to_hydrated_ms,
        "resetHydratedToWelcomeVisibleMs": reset_hydrated_to_welcome_visible_ms,
        "resetLastDashResponseToWelcomeVisibleMs": reset_last_response_to_welcome_ms,
        "accountListOpenToReadyMs": load_to_ready_ms,
        "accountListLoadClickToReloadStartMs": load_click_to_reload_start_ms,
        "accountListReloadStartToHydratedMs": account_list_reload_start_to_hydrated_ms,
        "accountListLoadClickToLiveApplyCommitMs": account_list_load_click_to_live_apply_commit_ms,
        "accountListLiveApplyCommitToStatisticsReadyMs": account_list_live_apply_commit_to_ready_ms,
        "accountListHydratedToStatisticsReadyMs": account_list_hydrated_to_statistics_ready_ms,
        "accountListLastDashResponseToReadyMs": load_last_response_to_ready_ms,
        "totalRunMs": total_ms,
        "resetWindow": reset_summary,
        "accountListWindow": load_summary,
    }


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------

def _summarize_import_runs(run_results: list[dict]) -> dict:
    """Summarize import-mode timing results."""
    if not run_results:
        return {"runs": 0}

    reset_ms = [r["resetToWelcomeMs"] for r in run_results]
    reset_reload_ms = [r["resetReloadStartToHydratedMs"] for r in run_results]
    reset_welcome_ms = [r["resetHydratedToWelcomeVisibleMs"] for r in run_results]
    reset_tail_ms = [r["resetLastDashResponseToWelcomeVisibleMs"] for r in run_results]
    peer_import_ms = [r["peerImportToSeriesModalMs"] for r in run_results]
    peer_ready_ms = [r["peerSeriesConfirmToStatisticsReadyMs"] for r in run_results]
    peer_tail_ms = [r["peerLastDashResponseToReadyMs"] for r in run_results]
    index_import_ms = [r["indexImportToSeriesModalMs"] for r in run_results]
    index_ready_ms = [r["indexSeriesConfirmToStatisticsReadyMs"] for r in run_results]
    index_tail_ms = [r["indexLastDashResponseToReadyMs"] for r in run_results]
    total_ms = [r["totalRunMs"] for r in run_results]

    callback_counter: Counter[str] = Counter()
    for run in run_results:
        for window_key in ("resetWindow", "peerWindow", "indexWindow"):
            window = run.get(window_key, {})
            for request in window.get("dashUpdateRequests", []):
                for output_id in request.get("outputs", []):
                    callback_counter[output_id] += 1

    return {
        "runs": len(run_results),
        "resetToWelcomeMs": reset_ms,
        "resetToWelcomeMedian": round(median(reset_ms)),
        "resetReloadStartToHydratedMs": reset_reload_ms,
        "resetReloadStartToHydratedMedian": round(median(reset_reload_ms)),
        "resetHydratedToWelcomeVisibleMs": reset_welcome_ms,
        "resetHydratedToWelcomeVisibleMedian": round(median(reset_welcome_ms)),
        "resetLastDashResponseToWelcomeVisibleMs": reset_tail_ms,
        "resetLastDashResponseToWelcomeVisibleMedian": round(median(reset_tail_ms)),
        "peerImportToSeriesModalMs": peer_import_ms,
        "peerImportToSeriesModalMedian": round(median(peer_import_ms)),
        "peerSeriesConfirmToStatisticsReadyMs": peer_ready_ms,
        "peerSeriesConfirmToStatisticsReadyMedian": round(median(peer_ready_ms)),
        "peerLastDashResponseToReadyMs": peer_tail_ms,
        "peerLastDashResponseToReadyMedian": round(median(peer_tail_ms)),
        "indexImportToSeriesModalMs": index_import_ms,
        "indexImportToSeriesModalMedian": round(median(index_import_ms)),
        "indexSeriesConfirmToStatisticsReadyMs": index_ready_ms,
        "indexSeriesConfirmToStatisticsReadyMedian": round(median(index_ready_ms)),
        "indexLastDashResponseToReadyMs": index_tail_ms,
        "indexLastDashResponseToReadyMedian": round(median(index_tail_ms)),
        "totalRunMs": total_ms,
        "totalRunMedian": round(median(total_ms)),
        "topCallbacksByFrequency": [
            {"callback": cb, "count": count}
            for cb, count in callback_counter.most_common(12)
        ],
        "runResults": run_results,
    }


def _summarize_account_list_runs(run_results: list[dict]) -> dict:
    """Summarize account-list-mode timing results."""
    if not run_results:
        return {"runs": 0}

    reset_ms = [r["resetToWelcomeMs"] for r in run_results]
    reset_reload_ms = [r["resetReloadStartToHydratedMs"] for r in run_results]
    reset_welcome_ms = [r["resetHydratedToWelcomeVisibleMs"] for r in run_results]
    reset_tail_ms = [r["resetLastDashResponseToWelcomeVisibleMs"] for r in run_results]
    load_ms = [r["accountListOpenToReadyMs"] for r in run_results]
    load_click_ms = [r["accountListLoadClickToReloadStartMs"] for r in run_results]
    load_reload_ms = [r["accountListReloadStartToHydratedMs"] for r in run_results]
    load_live_apply_click_ms = [r.get("accountListLoadClickToLiveApplyCommitMs", 0) for r in run_results]
    load_live_apply_ready_ms = [r.get("accountListLiveApplyCommitToStatisticsReadyMs", 0) for r in run_results]
    load_ready_ms = [r["accountListHydratedToStatisticsReadyMs"] for r in run_results]
    tail_ms = [r["accountListLastDashResponseToReadyMs"] for r in run_results]
    total_ms = [r["totalRunMs"] for r in run_results]

    callback_counter: Counter[str] = Counter()
    for run in run_results:
        for window_key in ("resetWindow", "accountListWindow"):
            window = run.get(window_key, {})
            for request in window.get("dashUpdateRequests", []):
                for output_id in request.get("outputs", []):
                    callback_counter[output_id] += 1

    return {
        "runs": len(run_results),
        "resetToWelcomeMs": reset_ms,
        "resetToWelcomeMedian": round(median(reset_ms)),
        "resetReloadStartToHydratedMs": reset_reload_ms,
        "resetReloadStartToHydratedMedian": round(median(reset_reload_ms)),
        "resetHydratedToWelcomeVisibleMs": reset_welcome_ms,
        "resetHydratedToWelcomeVisibleMedian": round(median(reset_welcome_ms)),
        "resetLastDashResponseToWelcomeVisibleMs": reset_tail_ms,
        "resetLastDashResponseToWelcomeVisibleMedian": round(median(reset_tail_ms)),
        "accountListOpenToReadyMs": load_ms,
        "accountListOpenToReadyMedian": round(median(load_ms)),
        "accountListLoadClickToReloadStartMs": load_click_ms,
        "accountListLoadClickToReloadStartMedian": round(median(load_click_ms)),
        "accountListReloadStartToHydratedMs": load_reload_ms,
        "accountListReloadStartToHydratedMedian": round(median(load_reload_ms)),
        "accountListLoadClickToLiveApplyCommitMs": load_live_apply_click_ms,
        "accountListLoadClickToLiveApplyCommitMedian": round(median(load_live_apply_click_ms)),
        "accountListLiveApplyCommitToStatisticsReadyMs": load_live_apply_ready_ms,
        "accountListLiveApplyCommitToStatisticsReadyMedian": round(median(load_live_apply_ready_ms)),
        "accountListHydratedToStatisticsReadyMs": load_ready_ms,
        "accountListHydratedToStatisticsReadyMedian": round(median(load_ready_ms)),
        "accountListLastDashResponseToReadyMs": tail_ms,
        "accountListLastDashResponseToReadyMedian": round(median(tail_ms)),
        "totalRunMs": total_ms,
        "totalRunMedian": round(median(total_ms)),
        "topCallbacksByFrequency": [
            {"callback": cb, "count": count}
            for cb, count in callback_counter.most_common(12)
        ],
        "runResults": run_results,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AT Statistics harness")
    parser.add_argument("--repo-root", default="")
    parser.add_argument("--base-url", default="http://127.0.0.1:8050")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--label", default="")
    parser.add_argument("--git-ref", default="")
    parser.add_argument("--startup-timeout", type=int, default=30)
    parser.add_argument("--skip-db-build", action="store_true")
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--server-log", default="")
    parser.add_argument("--mode", choices=["imports", "account-list"], default="imports")
    parser.add_argument("--network-profile", choices=list(NETWORK_PROFILES.keys()), default="none")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main harness
# ---------------------------------------------------------------------------

def run_harness(
    base_url: str,
    runs: int,
    label: str,
    headed: bool,
    server_log: Path | None,
    mode: str,
    network_profile: str,
) -> dict:
    global TIMEOUT_SCALE
    console_messages: list[dict[str, str]] = []
    browser_timing_messages: list[str] = []
    TIMEOUT_SCALE = NETWORK_TIMEOUT_MULTIPLIERS.get(network_profile, 1.0)

    # Build run specs
    specs = build_run_specs(DB_ENGINE)
    print(f"RUN_SPECS={json.dumps([{'peer': s['peer']['portfolio'], 'index': s['index']['portfolio']} for s in specs], separators=(',', ':'))}", flush=True)
    print(f"NETWORK_PROFILE={network_profile}", flush=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not headed)
        page = browser.new_page(viewport={"width": 1440, "height": 960})
        page.set_default_timeout(_scaled_timeout(30000))
        page.set_default_navigation_timeout(_scaled_timeout(30000))
        applied_network_profile = None
        request_tracker = DashUpdateRequestTracker(page)

        def on_console(msg) -> None:
            if "timing name=account_list." in msg.text:
                if len(browser_timing_messages) < 240:
                    browser_timing_messages.append(msg.text)
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

        # Warmup: open /analyticstool and run AA DB retrieval once
        renderer_mode = warm_analytics_db(page, base_url, DEFAULT_DB_SERIES)
        _wait_statistics_ready(page)
        print("WARMUP=aa_db_complete", flush=True)

        # Create account-list fixtures if needed
        account_list_fixtures: list[dict] | None = None
        if mode == "account-list":
            username = _get_username(page)
            account_list_fixtures = create_account_list_fixtures(DB_ENGINE, username, specs)
            print(f"FIXTURES={json.dumps(account_list_fixtures, separators=(',', ':'))}", flush=True)

        # Rehearsal: one unmeasured run with the first spec
        print("REHEARSAL=starting", flush=True)
        if mode == "imports":
            _measure_import_run(page, request_tracker, base_url, specs[0], server_log)
        else:
            _measure_account_list_run(
                page,
                request_tracker,
                base_url,
                specs[0],
                account_list_fixtures[0],
                server_log,
                browser_timing_messages,
            )
        print("REHEARSAL=complete", flush=True)

        applied_network_profile = _apply_network_profile(page, network_profile)

        # Measured runs
        timing_start_offset = current_log_offset(server_log)
        run_results: list[dict] = []
        for i in range(runs):
            spec = specs[i % len(specs)]
            print(f"RUN={i + 1}/{runs} spec={spec['specIndex']} peer={spec['peer']['portfolio']} index={spec['index']['portfolio']}", flush=True)

            if mode == "imports":
                result = _measure_import_run(page, request_tracker, base_url, spec, server_log)
            else:
                fixture = account_list_fixtures[spec["specIndex"]]
                result = _measure_account_list_run(
                    page,
                    request_tracker,
                    base_url,
                    spec,
                    fixture,
                    server_log,
                    browser_timing_messages,
                )

            run_results.append(result)
            print(f"RUN={i + 1}/{runs} totalMs={result['totalRunMs']}", flush=True)

        browser.close()

    # Summarize
    if mode == "imports":
        summary = _summarize_import_runs(run_results)
    else:
        summary = _summarize_account_list_runs(run_results)

    warmup_flow = "analyticstool-aa-db-import+statistics-ready"
    return {
        "ok": True,
        "label": label,
        "baseUrl": base_url,
        "mode": mode,
        "networkProfile": applied_network_profile or {"name": "none"},
        "warmupFlow": warmup_flow,
        "rendererMode": renderer_mode,
        "runs": runs,
        "runSpecs": [
            {
                "specIndex": s["specIndex"],
                "peerPortfolio": s["peer"]["portfolio"],
                "peerType": s["peer"]["type"],
                "peerBenchmarkType": s["peer"]["benchmark_type"],
                "indexPortfolio": s["index"]["portfolio"],
                "indexType": s["index"]["type"],
                "indexBenchmarkType": s["index"]["benchmark_type"],
            }
            for s in specs
        ],
        "accountListFixtures": account_list_fixtures,
        "summary": summary,
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

    timestamp_str = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    mode_token = sanitize_token(args.mode, "imports")
    stem = f"at_statistics_{timestamp_str}_{sanitize_token(args.label, 'run')}_{mode_token}_{sanitize_token((resolved_git_ref or 'unknown')[:8], 'unknown')}"
    server_log_path = Path(args.server_log).resolve() if args.server_log else None

    try:
        wait_for_app(args.base_url, args.startup_timeout)
        result = run_harness(
            base_url=args.base_url,
            runs=args.runs,
            label=args.label,
            headed=args.headed,
            server_log=server_log_path,
            mode=args.mode,
            network_profile=args.network_profile,
        )
    except Exception as exc:
        console_messages: list[dict[str, str]] = []
        raw_path = write_failure_artifacts(
            out_dir=out_dir,
            fail_dir=fail_dir,
            stem=stem,
            repo_root=root,
            base_url=args.base_url,
            git_ref=resolved_git_ref,
            label=args.label,
            db_series=DEFAULT_DB_SERIES,
            startup_timeout=args.startup_timeout,
            console_messages=console_messages,
            exc=exc,
        )
        print(f"RAW_PATH={raw_path}")
        print(traceback.format_exc())
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
        "mode": result["mode"],
        "warmupFlow": result["warmupFlow"],
        "runs": result["runs"],
        "dbRebuilt": db_rebuilt,
        "dbRebuildReasons": db_rebuild_reasons,
        "runSpecs": result["runSpecs"],
        "accountListFixtures": result["accountListFixtures"],
        "summary": result["summary"],
        "consoleMessages": result["consoleMessages"],
        "timingSummary": timing_summary,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"OUT_PATH={out_path}")
    print("SUMMARY=" + json.dumps(result["summary"], separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
