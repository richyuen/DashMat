from __future__ import annotations

from io import BytesIO
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from dash import no_update
from dash.exceptions import PreventUpdate

from utils.raw_dataset import get_raw_dataset_df, resolve_dataset_key
from utils.returns import build_raw_data_metadata, df_to_json


def _sample_window_weights() -> list[dict]:
    return [
        {
            "apply_start": "2024-01-01",
            "apply_end": "2024-01-31",
            "est_start": "2023-12-01",
            "est_end": "2023-12-31",
            "weights": {"Asset_A": 0.6, "Asset_B": 0.4},
        },
        {
            "apply_start": "2024-02-01",
            "apply_end": "2024-02-29",
            "est_start": "2024-01-01",
            "est_end": "2024-01-31",
            "weights": {"Asset_A": 0.5, "Asset_B": 0.5},
        },
    ]


def _collect_component_text(node):
    if node is None:
        return []
    if isinstance(node, str):
        return [node]
    if isinstance(node, (int, float, bool)):
        return [str(node)]
    if isinstance(node, (list, tuple, set)):
        out = []
        for item in node:
            out.extend(_collect_component_text(item))
        return out
    if isinstance(node, dict):
        out = []
        for value in node.values():
            out.extend(_collect_component_text(value))
        return out

    out = []
    children = getattr(node, "children", None)
    out.extend(_collect_component_text(children))
    props = getattr(node, "props", None)
    if isinstance(props, dict):
        for value in props.values():
            out.extend(_collect_component_text(value))
    return out


def _raw_meta(raw_json: str, original_periodicity: str = "daily") -> dict:
    return build_raw_data_metadata(raw_json, original_periodicity)


def _raw_json_value(value):
    if isinstance(value, dict):
        return value.get("raw_data_json", "")
    return value


def _series_snapshot(rows: list[dict]) -> dict:
    return {"rows": rows, "capturedAt": 1}


def _find_component_by_id(node, target_id):
    if node is None:
        return None
    node_id = getattr(node, "id", None)
    if node_id == target_id:
        return node

    children = getattr(node, "children", None)
    if isinstance(children, (list, tuple)):
        for child in children:
            found = _find_component_by_id(child, target_id)
            if found is not None:
                return found
    else:
        found = _find_component_by_id(children, target_id)
        if found is not None:
            return found

    props = getattr(node, "props", None)
    if isinstance(props, dict):
        for value in props.values():
            found = _find_component_by_id(value, target_id)
            if found is not None:
                return found
    return None


def test_build_po_working_bundle_normalizes_inputs(page_modules, raw_json):
    _, portopt = page_modules

    bundle = portopt._build_po_working_bundle(
        raw_json,
        None,
        {"Asset_A": "Asset_B"},
        {"Asset_A": True},
        {"start": "2024-01-01", "end": "2024-12-31"},
        None,
        {"Asset_A": False},
    )

    assert bundle.periodicity == "daily"
    assert bundle.vol_scaler == 0
    assert bundle.benchmark_payload == '{"Asset_A":"Asset_B"}'


def test_po_get_result_basis_bundle_uses_dataset_key(monkeypatch, page_modules, raw_json):
    _, portopt = page_modules
    captured = {}

    def _fake_cached(dataset_key, run_inputs_payload):
        captured["dataset_key"] = dataset_key
        captured["run_inputs_payload"] = run_inputs_payload
        return portopt.canonical_json_dumps({})

    monkeypatch.setattr(portopt, "_po_build_result_basis_bundle_cached", _fake_cached)

    portopt._po_get_result_basis_bundle(
        {"run_inputs": {"selected_series": ["Asset_A"], "periodicity": "daily"}},
        raw_json,
    )

    assert captured["dataset_key"] == resolve_dataset_key(raw_json)


def test_po_get_performance_frames_uses_dataset_key(monkeypatch, page_modules, raw_json):
    _, portopt = page_modules
    captured = {}

    def _fake_cached(selected_portfolio, reporting_returns_json, benchmark_returns_json, run_inputs_payload, dataset_key):
        captured["selected_portfolio"] = selected_portfolio
        captured["dataset_key"] = dataset_key
        captured["run_inputs_payload"] = run_inputs_payload
        return portopt.canonical_json_dumps({})

    monkeypatch.setattr(portopt, "_po_build_performance_source_cached", _fake_cached)

    frames = portopt._po_get_performance_frames(
        {
            "Port1": {
                "reporting_returns_json": "",
                "benchmark_returns_json": "",
                "run_inputs": {"selected_series": ["Asset_A"], "periodicity": "daily"},
            }
        },
        "Port1",
        raw_json,
        "daily",
        {},
        {},
        None,
        0,
        {},
    )

    assert captured["selected_portfolio"] == "Port1"
    assert captured["dataset_key"] == resolve_dataset_key(raw_json)
    assert frames["display_cols"] == []


def test_po_build_display_series_cached_uses_dataset_key(monkeypatch, page_modules, raw_json):
    _, portopt = page_modules
    captured = {}
    working_df = pd.read_json(StringIO(raw_json), orient="split")[["Asset_A", "Asset_B"]]
    working_df.index = pd.to_datetime(working_df.index)

    def _fake_cached(dataset_key, run_inputs_payload):
        captured["dataset_key"] = dataset_key
        captured["run_inputs_payload"] = run_inputs_payload
        return portopt.canonical_json_dumps({"reporting_df": df_to_json(working_df)})

    monkeypatch.setattr(portopt, "_po_build_result_basis_bundle_cached", _fake_cached)

    payload, ordered_cols = portopt._po_build_display_series_cached(
        "Port1",
        working_df["Asset_A"].rename("Port1").to_json(date_format="iso"),
        portopt.canonical_json_dumps({"selected_series": ["Asset_A", "Asset_B"]}),
        resolve_dataset_key(raw_json),
    )

    assert captured["dataset_key"] == resolve_dataset_key(raw_json)
    assert payload
    assert ordered_cols == ["Port1", "Asset_A", "Asset_B"]


def test_po_bootstrap_helpers_default_to_idle_state(page_modules):
    _, portopt = page_modules

    state = portopt._po_bootstrap_state(None)

    assert state == {
        "phase": "idle",
        "loadedTabs": {
            "weight": False,
            "attribution": False,
            "risk": False,
            "frontier": False,
        },
    }
    assert portopt._po_bootstrap_ready(None) is False


def test_po_results_meta_helper_stays_lightweight(page_modules):
    _, portopt = page_modules

    assert portopt._po_results_meta(None) == {"has_results": False, "count": 0}
    assert portopt._po_results_meta({"RP": {"reporting_returns_json": "large"}}) == {
        "has_results": True,
        "count": 1,
    }


def test_po_bootstrap_tab_render_ready_requires_matching_loaded_tab(page_modules):
    _, portopt = page_modules

    bootstrap_state = {
        "phase": "ready",
        "loadedTabs": {
            "weight": False,
            "attribution": False,
            "risk": True,
            "frontier": False,
        },
    }

    assert portopt._po_bootstrap_tab_render_ready("risk", "risk", bootstrap_state) is True
    assert portopt._po_bootstrap_tab_render_ready("weight", "weight", bootstrap_state) is False
    assert portopt._po_bootstrap_tab_render_ready("frontier", "risk", bootstrap_state) is False


def test_sync_po_returns_basis_from_mirrors_updates_canonical(monkeypatch, page_modules):
    _, portopt = page_modules
    monkeypatch.setattr(
        portopt,
        "callback_context",
        type("Ctx", (), {"triggered_id": "po-returns-basis-control-calendar"})(),
    )

    result = portopt.sync_po_returns_basis_from_mirrors(
        "total",
        "excess",
        "total",
        "total",
    )

    assert result == "excess"


def test_sync_po_returns_basis_mirrors_only_updates_mismatched(page_modules):
    _, portopt = page_modules

    result = portopt.sync_po_returns_basis_mirrors(
        "excess",
        "excess",
        "total",
        "excess",
    )

    assert result[0] is no_update
    assert result[1] == "excess"
    assert result[2] is no_update


def test_po_open_modal_uses_clientside_open_seed_callback():
    page_text = Path("pages/portopt.py").read_text(encoding="utf-8")
    assert 'ClientsideFunction(namespace="dashmat_callbacks", function_name="openPortoptSeriesModal")' in page_text
    assert 'Input("po-open-modal-button", "n_clicks")' in page_text
    assert 'Input("po-url-location", "pathname")' in page_text
    assert 'Input("po-page-load-trigger", "n_intervals")' in page_text
    assert 'Input("dashmat-raw-data-meta-store", "data")' in page_text
    open_block = page_text.split('ClientsideFunction(namespace="dashmat_callbacks", function_name="openPortoptSeriesModal")', 1)[1]
    open_callback = open_block.split("# ---------------------------------------------------------------------------\n# Series selection modal: render rows", 1)[0]
    assert open_callback.count('Input("po-page-load-trigger", "n_intervals")') == 1


def test_po_open_modal_js_preserves_first_visit_and_generic_new_behavior():
    js_text = Path("assets/dashmat_callbacks.js").read_text(encoding="utf-8")
    assert "function openPortoptSeriesModal(" in js_text
    assert 'const selected = resolveStoredList(currentSelect, "po-series-select");' in js_text
    assert '(selectedValid.length ? resolveStoredList(currentOrder, "po-series-order-store") : []).filter(function (series) {' in js_text
    assert 'if (!resolveStoredBool(pageVisited, "po-page-visited-store") && !selectedValid.length) {' in js_text
    assert "genericNew.length" in js_text
    assert 'const poOriginSet = new Set(resolveStoredNames(poOriginSeries, "dashmat-pending-new-series-store").filter(function (series) {' in js_text
    assert 'return !knownColumns.has(series) && !poOriginSet.has(series);' in js_text
    assert 'if (trigger === "dashmat-raw-data-meta-store") {' in js_text
    assert 'if (trigger === "po-url-location") {' in js_text
    assert 'if (trigger === "po-page-load-trigger"' in js_text


def test_po_open_modal_js_ignores_seeded_order_when_no_series_selected():
    js_text = Path("assets/dashmat_callbacks.js").read_text(encoding="utf-8")
    assert js_text.count('(selectedValid.length ? resolveStoredList(currentOrder, "po-series-order-store") : []).filter(function (series) {') >= 2
    assert "selectedValid.forEach(function (series) {" in js_text


def test_po_open_modal_js_keeps_manual_open_and_blocker_seed():
    js_text = Path("assets/dashmat_callbacks.js").read_text(encoding="utf-8")
    assert 'if (trigger === "po-open-modal-button") {' in js_text
    assert "currentBench" in js_text
    assert "currentCmabench" in js_text
    assert "currentForceMax" in js_text
    assert "true" in js_text


def test_po_series_modal_bulk_actions_use_shared_clientside_helper():
    page_text = Path("pages/portopt.py").read_text(encoding="utf-8")
    js_text = Path("assets/dashmat_callbacks.js").read_text(encoding="utf-8")

    assert 'ClientsideFunction(namespace="dashmat_callbacks", function_name="bulkUpdateSeriesSelection")' in page_text
    assert 'Output("po-series-bulk-action-dummy", "data")' in page_text
    assert 'Input("po-select-all-button", "n_clicks")' in page_text
    assert 'Input("po-unselect-all-button", "n_clicks")' in page_text
    assert 'State("po-series-selection-modal", "opened")' in page_text
    assert 'gridId = "po-series-selection-grid";' in js_text
    assert 'targetField = "Selected";' in js_text


def test_po_layout_starts_with_welcome_and_main_hidden(page_modules):
    _, portopt = page_modules

    welcome = _find_component_by_id(portopt.layout, "po-welcome-screen")
    main = _find_component_by_id(portopt.layout, "po-main-container")
    blocker_store = _find_component_by_id(portopt.layout, "po-ui-blocker-store")
    blocker_overlay = _find_component_by_id(portopt.layout, "po-ui-blocker-overlay")
    bootstrap_store = _find_component_by_id(portopt.layout, "po-bootstrap-store")
    results_meta_store = _find_component_by_id(portopt.layout, "po-results-meta-store")
    cmabench_defaults_store = _find_component_by_id(portopt.layout, "po-cmabench-defaults-store")
    series_grid = _find_component_by_id(portopt.layout, "po-series-selection-grid")

    assert getattr(welcome, "style", {})["display"] == "none"
    assert getattr(main, "style", {})["display"] == "none"
    assert getattr(blocker_store, "data", None) is False
    assert getattr(blocker_overlay, "visible", None) is False
    assert getattr(blocker_overlay, "zIndex", None) == 2500
    assert getattr(results_meta_store, "data", None) == {"has_results": False, "count": 0}
    assert getattr(cmabench_defaults_store, "data", None) == {}
    assert series_grid is not None
    assert getattr(series_grid, "rowData", None) == []
    assert getattr(bootstrap_store, "data", None) == {
        "phase": "idle",
        "loadedTabs": {
            "weight": False,
            "attribution": False,
            "risk": False,
            "frontier": False,
        },
    }


def test_po_bootstrap_keeps_single_page_load_interval_and_no_dead_results_sync():
    page_text = Path("pages/portopt.py").read_text(encoding="utf-8")
    js_text = Path("assets/dashmat_callbacks.js").read_text(encoding="utf-8")
    assert page_text.count('dcc.Interval(id="po-page-load-trigger"') == 1
    assert "def po_sync_results_with_raw_data" not in page_text
    assert 'id="po-bootstrap-store"' in page_text
    assert 'ClientsideFunction(namespace="dashmat_callbacks", function_name="portoptBootstrapRestore")' in page_text
    assert 'ClientsideFunction(namespace="dashmat_callbacks", function_name="portoptMarkVisitedTabLoaded")' in page_text
    assert "function portoptBootstrapRestore(" in js_text
    assert "function portoptMarkVisitedTabLoaded(" in js_text
    assert 'Output("po-vis-tabs", "value")' in page_text
    assert 'State("po-active-tab-store", "data")' in page_text
    assert 'po-restore-complete-store' not in page_text
    assert 'po-secondary-restore-ready-store' not in page_text
    assert 'po-initial-tab-render-ready-store' not in page_text
    assert 'po-attribution-tab-loaded-store' not in page_text
    assert 'po-risk-tab-loaded-store' not in page_text
    assert 'po-frontier-tab-loaded-store' not in page_text
    assert 'Output("po-attribution-chart-container", "children")' in page_text
    assert 'Output("po-attribution-grid-container", "children")' in page_text
    assert 'po-attribution-chart-content' not in page_text
    assert 'po-attribution-grid-content' not in page_text
    assert 'Output("po-frontier-chart-container", "children")' in page_text
    assert 'Output("po-frontier-grid-container", "children")' in page_text
    assert 'po-frontier-chart-content' not in page_text
    assert 'po-frontier-grid-content' not in page_text
    assert 'Output("po-risk-chart-container", "children")' in page_text
    assert 'Output("po-risk-grid-container", "children")' in page_text
    assert 'po-risk-chart-content' not in page_text
    assert 'po-risk-grid-content' not in page_text
    assert 'Output("po-turnover-chart-container", "children")' in page_text
    assert 'Output("po-turnover-grid-container", "children")' in page_text
    assert 'po-turnover-chart-content' not in page_text
    assert 'po-turnover-grid-content' not in page_text
    assert page_text.count('Input("po-bootstrap-store", "data")') >= 9
    assert page_text.count('State("po-bootstrap-store", "data")') >= 1


def test_po_shell_visibility_uses_raw_data_presence_and_page_load_tick():
    page_text = Path("pages/portopt.py").read_text(encoding="utf-8")
    visibility_block = page_text.split('Output("po-welcome-screen", "style")', 1)[1].split(
        'ClientsideFunction(namespace="dashmat_callbacks", function_name="portoptBootstrapRestore")',
        1,
    )[0]
    assert 'Output("po-main-container", "style")' in visibility_block
    assert 'Input("dashmat-raw-data-store", "data")' in visibility_block
    assert 'Input("po-page-load-trigger", "n_intervals")' in visibility_block


def test_po_toggle_ui_elements_uses_bootstrap_store():
    page_text = Path("pages/portopt.py").read_text(encoding="utf-8")
    toggle_block = page_text.split("def po_toggle_ui_elements", 1)[0]
    toggle_callback = toggle_block.rsplit("@callback(", 1)[-1]
    assert 'Input("po-bootstrap-store", "data")' in toggle_callback
    assert 'Input("po-results-meta-store", "data")' in toggle_callback
    assert 'Input("po-results-store", "data")' not in toggle_callback
    assert 'po-restore-complete-store' not in toggle_callback


def test_po_bootstrap_reducer_reads_stored_controls_and_marks_loaded_tabs():
    page_text = Path("pages/portopt.py").read_text(encoding="utf-8")
    js_text = Path("assets/dashmat_callbacks.js").read_text(encoding="utf-8")
    bootstrap_block = page_text.split('ClientsideFunction(namespace="dashmat_callbacks", function_name="portoptBootstrapRestore")', 1)[1]
    bootstrap_callback = bootstrap_block.split('ClientsideFunction(namespace="dashmat_callbacks", function_name="portoptMarkVisitedTabLoaded")', 1)[0]
    assert 'Input("po-page-load-trigger", "n_intervals")' in bootstrap_callback
    assert 'Input("dashmat-raw-data-meta-store", "data")' in bootstrap_callback
    assert 'State("po-active-tab-store", "data")' in bootstrap_callback
    assert 'State("po-frontier-chart-switch-store", "data")' in bootstrap_callback
    assert 'State("po-periodicity-select", "data")' in bootstrap_callback
    assert 'State("po-series-select", "data")' in bootstrap_callback
    assert 'State("po-opt-window-select", "value")' in bootstrap_callback
    assert 'State("po-returns-basis-control", "value")' in bootstrap_callback
    assert 'State("po-vis-tabs", "value")' in bootstrap_callback
    assert 'State("po-weight-chart-switch", "value")' in bootstrap_callback
    assert 'Output("po-bootstrap-store", "data")' in bootstrap_callback
    assert "defaultPortoptLoadedTabs" in js_text
    assert "function resolvedOutput(nextValue, currentValue)" in js_text
    assert "function sameValue(left, right)" in js_text
    assert 'phase: "ready"' in js_text


def test_po_init_date_range_no_longer_depends_on_common_daily_store():
    page_text = Path("pages/portopt.py").read_text(encoding="utf-8")
    init_block = page_text.split("def po_init_date_range", 1)[0]
    init_callback = init_block.rsplit("@callback(", 1)[-1]
    assert 'Input("po-range-candidates-store", "data")' in init_callback
    assert 'Input("po-common-daily-candidates-store", "data")' not in init_callback


def test_warm_switch_harness_tracks_portopt_performance_frames():
    harness_text = Path("tools/playwright/warm_switch_harness.py").read_text(encoding="utf-8")
    assert '"portopt.performance_frames"' in harness_text


def test_warm_switch_wrapper_forwards_restore_tab_and_entry_only():
    wrapper_text = Path("tools/playwright/warm_switch_harness.ps1").read_text(encoding="utf-8")
    assert '[string]$PortoptRestoreTab = \'weight\'' in wrapper_text
    assert '[switch]$PortoptEntryOnly' in wrapper_text
    assert "'--portopt-restore-tab', $PortoptRestoreTab" in wrapper_text
    assert "$args += '--portopt-entry-only'" in wrapper_text


def test_po_common_daily_button_uses_shared_clientside_helper():
    page_text = Path("pages/portopt.py").read_text(encoding="utf-8")
    js_text = Path("assets/dashmat_callbacks.js").read_text(encoding="utf-8")
    assert 'ClientsideFunction(namespace="dashmat_callbacks", function_name="commonDailyButtonDisabled")' in page_text
    assert 'Output("po-common-daily-button", "disabled")' in page_text
    assert "function commonDailyButtonDisabled(candidates, commonDailyCandidates, periodicityOptions)" in js_text


def test_po_active_vis_trigger_store_exists(page_modules):
    _, portopt = page_modules
    assert _find_component_by_id(portopt.layout, "po-active-vis-trigger-store") is not None


def test_po_require_active_vis_trigger_rejects_mismatched_tab(page_modules):
    _, portopt = page_modules

    assert portopt._po_require_active_vis_trigger({"tab": "returns"}, "returns") is None

    with pytest.raises(PreventUpdate):
        portopt._po_require_active_vis_trigger(None, "returns")

    with pytest.raises(PreventUpdate):
        portopt._po_require_active_vis_trigger({"tab": "weight"}, "returns")


def test_po_selection_date_candidates_callback_returns_both_payloads(monkeypatch, page_modules):
    _, portopt = page_modules

    monkeypatch.setattr(portopt, "_dataset_key", lambda raw_data: f"dataset:{raw_data}")
    monkeypatch.setattr(
        portopt,
        "compute_date_range_candidates",
        lambda dataset_key, periodicity, selected_series: {
            "kind": "range",
            "dataset": dataset_key,
            "periodicity": periodicity,
            "selected": list(selected_series),
        },
    )
    monkeypatch.setattr(
        portopt,
        "compute_common_daily_candidates",
        lambda dataset_key, selected_series: {
            "kind": "common_daily",
            "dataset": dataset_key,
            "selected": list(selected_series),
        },
    )

    range_candidates, common_daily_candidates = portopt.po_update_selection_date_candidates(
        "raw-payload",
        "monthly",
        ["Asset_A", "Asset_B"],
    )

    assert range_candidates == {
        "kind": "range",
        "dataset": "dataset:raw-payload",
        "periodicity": "monthly",
        "selected": ["Asset_A", "Asset_B"],
    }
    assert common_daily_candidates == {
        "kind": "common_daily",
        "dataset": "dataset:raw-payload",
        "selected": ["Asset_A", "Asset_B"],
    }


def test_po_linear_constraints_columns_use_clientside_builder():
    page_text = Path("pages/portopt.py").read_text(encoding="utf-8")
    js_text = Path("assets/dashmat_callbacks.js").read_text(encoding="utf-8")

    assert 'ClientsideFunction(namespace="dashmat_callbacks", function_name="portoptLinearConstraintColumnDefs")' in page_text
    assert 'def po_populate_linear_constraints_columns' not in page_text
    assert "function portoptLinearConstraintColumnDefs(selectedSeries)" in js_text
    assert 'field: "Constraint"' in js_text
    assert 'field: "Min"' in js_text
    assert 'field: "Max"' in js_text


def test_po_returns_grid_uses_clientside_builder():
    page_text = Path("pages/portopt.py").read_text(encoding="utf-8")
    js_text = Path("assets/dashmat_callbacks.js").read_text(encoding="utf-8")

    assert 'ClientsideFunction(namespace="dashmat_callbacks", function_name="portoptReturnsGridData")' in page_text
    assert 'def po_populate_returns_grid' not in page_text
    assert "function portoptReturnsGridData(selectedSeries, mode, existingReturns, existingVol)" in js_text
    assert 'field: "Return"' in js_text
    assert 'field: "Volatility"' in js_text


def test_po_matrix_grid_uses_clientside_builder():
    page_text = Path("pages/portopt.py").read_text(encoding="utf-8")
    js_text = Path("assets/dashmat_callbacks.js").read_text(encoding="utf-8")

    assert 'ClientsideFunction(namespace="dashmat_callbacks", function_name="portoptMatrixGridData")' in page_text
    assert 'def po_populate_matrix_grid' not in page_text
    assert "function portoptMatrixGridData(selectedSeries, mode, covStore, corrStore)" in js_text
    assert 'field: "Asset"' in js_text
    assert "d3.format(',.4f')" in js_text


def test_po_hidden_vis_tabs_use_shared_trigger_store():
    page_text = Path("pages/portopt.py").read_text(encoding="utf-8")
    js_text = Path("assets/dashmat_callbacks.js").read_text(encoding="utf-8")

    assert 'ClientsideFunction(namespace="dashmat_callbacks", function_name="portoptActiveVisTrigger")' in page_text
    assert 'Output("po-active-vis-trigger-store", "data")' in page_text
    assert "function portoptActiveVisTrigger(activeTab" in js_text
    for tab_name in ("returns", "rolling", "statistics", "calendar", "drawdown"):
        assert f'"{tab_name}"' in js_text
    for tab_name in ("returns", "rolling", "statistics", "calendar", "drawdown"):
        assert f'_po_require_active_vis_trigger(trigger_payload, "{tab_name}")' in page_text


def test_portopt_modal_harness_tracks_dash_update_attribution():
    harness_text = Path("tools/playwright/portopt_series_modal_harness.py").read_text(encoding="utf-8")
    assert '"/_dash-update-component"' in harness_text
    assert '"dashUpdateRequestCount"' in harness_text
    assert '"dashUpdateTotalMs"' in harness_text
    assert '"dashUpdateCallbacks"' in harness_text
    assert '"dashUpdateRequests"' in harness_text
    assert '"dashUpdateRequestCountMedian"' in harness_text


def test_portopt_modal_harness_supports_active_tab():
    harness_text = Path("tools/playwright/portopt_series_modal_harness.py").read_text(encoding="utf-8")
    wrapper_text = Path("tools/playwright/portopt_series_modal_harness.ps1").read_text(encoding="utf-8")

    assert '"--active-tab"' in harness_text
    assert '"activeTab": args.active_tab' in harness_text
    assert "def set_active_vis_tab(page, active_tab: str) -> None:" in harness_text
    assert "[ValidateSet('weight', 'returns', 'rolling', 'statistics', 'calendar', 'drawdown')][string]$ActiveTab = 'weight'" in wrapper_text
    assert "'--active-tab', $ActiveTab" in wrapper_text


def test_po_init_date_range_is_idempotent_when_range_is_current(monkeypatch, page_modules):
    _, portopt = page_modules

    monkeypatch.setattr(
        portopt,
        "resolve_initial_range",
        lambda *_args, **_kwargs: ("2024-01-01", "2024-12-31"),
    )

    start, end, _style, _common_disabled, _max_disabled, range_store = portopt.po_init_date_range(
        {
            "available_series": ["Asset_A"],
        },
        {"start": "2024-01-01", "end": "2024-12-31"},
        "2024-01-01",
        "2024-12-31",
    )

    assert start is no_update
    assert end is no_update
    assert range_store is no_update


def test_po_layout_uses_construction_first_tab_order(page_modules):
    _, portopt = page_modules

    tabs = _find_component_by_id(portopt.layout, "po-vis-tabs")
    tabs_list = getattr(tabs, "children", [])[0]
    labels = [getattr(tab, "children", None) for tab in getattr(tabs_list, "children", [])]

    assert labels == [
        "Weights",
        "Attribution",
        "Risk",
        "Turnover",
        "Frontier",
        "Statistics",
        "Returns",
        "Rolling",
        "Calendar Year",
        "Growth of $1",
        "Drawdown",
    ]


def test_build_apply_weight_matrix_assigns_weights_by_windows(page_modules):
    _, portopt = page_modules
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    weights = [
        {"apply_start": "2024-01-01", "apply_end": "2024-01-03", "weights": {"A": 0.7, "B": 0.3}},
        {"apply_start": "2024-01-04", "apply_end": "2024-01-05", "weights": {"A": 0.2, "B": 0.8}},
    ]
    mat = portopt._build_apply_weight_matrix(idx, ("A", "B"), weights)

    assert mat.shape == (5, 2)
    assert mat[0, 0] == pytest.approx(0.7)
    assert mat[4, 1] == pytest.approx(0.8)


def test_compute_monthly_attribution_matches_expected(page_modules):
    _, portopt = page_modules
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    working_df = pd.DataFrame({"A": 0.01, "B": 0.02}, index=idx)
    working_df.index.name = "Date"

    monthly = portopt._compute_monthly_attribution(
        working_df,
        ["A", "B"],
        [
            {"apply_start": "2024-01-01", "apply_end": "2024-01-05", "weights": {"A": 0.6, "B": 0.4}},
            {"apply_start": "2024-01-06", "apply_end": "2024-01-10", "weights": {"A": 0.3, "B": 0.7}},
        ],
    )

    # 5 days * (0.6*1% + 0.4*2%) + 5 days * (0.3*1% + 0.7*2%)
    expected_total = (5 * (0.006 + 0.008)) + (5 * (0.003 + 0.014))
    assert monthly.sum(axis=1).iloc[0] == pytest.approx(expected_total)


def test_po_get_monthly_attribution_uses_cached_working_return_path(monkeypatch, page_modules, raw_json):
    _, portopt = page_modules
    working_df = pd.read_json(StringIO(raw_json), orient="split")[["Asset_A", "Asset_B"]]
    working_df.index = pd.to_datetime(working_df.index)

    bundle = portopt._build_po_working_bundle(raw_json, "daily", {}, {}, None, 0, {})
    monkeypatch.setattr(portopt, "get_working_returns_by_key", lambda *_args, **_kwargs: working_df)

    monthly = portopt._po_get_monthly_attribution(
        bundle,
        ["Asset_A", "Asset_B"],
        _sample_window_weights(),
    )

    assert list(monthly.columns) == ["Asset_A", "Asset_B"]
    assert not monthly.empty


def test_po_render_attribution_table_returns_grid_data(monkeypatch, page_modules):
    _, portopt = page_modules
    idx = pd.date_range("2024-01-01", periods=60, freq="D")
    working_df = pd.DataFrame({"Asset_A": 0.01, "Asset_B": 0.02}, index=idx)
    working_df.index.name = "Date"
    monkeypatch.setattr(portopt, "_po_get_working_returns", lambda *_args, **_kwargs: working_df)
    raw_json = df_to_json(working_df)

    results = {
        "P1": {
            "config": {"selected_series": ["Asset_A", "Asset_B"]},
            "window_weights": _sample_window_weights(),
        }
    }

    grid = portopt.po_render_attribution_table(
        "P1",
        results,
        "attribution",
        "table",
        {"phase": "ready", "loadedTabs": {"attribution": True}},
        raw_json,
        "daily",
        {},
        {},
        None,
        0,
        {},
    )

    assert getattr(grid, "columnDefs", [])[0]["field"] == "Date"
    assert any(c["field"] == "Total" for c in getattr(grid, "columnDefs", []))
    assert len(getattr(grid, "rowData", [])) > 0


def test_po_render_statistics_transposes_stats(monkeypatch, page_modules):
    _, portopt = page_modules

    def _fake_stats(*_args, **_kwargs):
        return [
            {"Series": "P1", "Cumulative Return": 0.1},
            {"Series": "P2", "Cumulative Return": 0.2},
        ]

    monkeypatch.setattr(portopt, "calculate_statistics_cached", _fake_stats)
    idx = pd.to_datetime(["2024-01-01", "2024-01-02"])
    monkeypatch.setattr(
        portopt,
        "_po_get_performance_frames",
        lambda *_args, **_kwargs: {
            "source_df": pd.DataFrame({"P1": [0.01, 0.02], "P2": [0.0, 0.01]}, index=idx),
            "total_df": pd.DataFrame({"P1": [0.01, 0.02], "P2": [0.0, 0.01]}, index=idx),
            "excess_df": pd.DataFrame({"P1": [0.01, 0.02], "P2": [0.0, 0.01]}, index=idx),
            "display_cols": ["P1", "P2"],
            "benchmark_map": {},
            "periodicity": "daily",
        },
    )

    results = {"P1": {"risk_free_meta": {"enabled": True}}}

    grid = portopt.po_render_statistics(results, "statistics", "P1", None, True, "daily")

    assert getattr(grid, "columnDefs", [])[0]["field"] == "Statistic"
    assert {c["field"] for c in getattr(grid, "columnDefs", [])[1:]} == {"P1", "P2"}
    row = next(r for r in getattr(grid, "rowData", []) if r["Statistic"] == "Cumulative Return")
    assert row["P1"] == pytest.approx(0.1)
    assert row["P2"] == pytest.approx(0.2)


def test_po_render_statistics_uses_result_rf_setting_not_live_toggle(monkeypatch, page_modules):
    _, portopt = page_modules
    captured = {}

    def _fake_stats(*args, **_kwargs):
        captured["use_risk_free"] = args[-1]
        return [{"Series": "P1", "Cumulative Return": 0.1}]

    monkeypatch.setattr(portopt, "calculate_statistics_cached", _fake_stats)

    s1 = pd.Series([0.01, 0.02], index=pd.to_datetime(["2024-01-01", "2024-01-02"]))
    results = {
        "P1": {
            "returns_json": s1.to_json(date_format="iso"),
            "risk_free_meta": {"enabled": False},
            "config": {"model": "risk_parity"},
        }
    }

    portopt.po_render_statistics(results, "statistics", "P1", None, True, "daily")

    assert captured["use_risk_free"] is False


def test_po_render_returns_builds_returns_grid(page_modules):
    _, portopt = page_modules

    s1 = pd.Series([0.01, 0.02], index=pd.to_datetime(["2024-01-01", "2024-01-02"]))
    results = {"P1": {"returns_json": s1.to_json(date_format="iso")}}

    grid = portopt.po_render_returns(results, "returns", "P1")
    assert getattr(grid, "columnDefs", [])[0]["field"] == "Date"
    assert getattr(grid, "columnDefs", [])[1]["field"] == "P1"
    assert getattr(grid, "rowData", [])[0]["Date"] == "2024-01-01"


def test_po_render_returns_uses_excess_basis_frame(monkeypatch, page_modules):
    _, portopt = page_modules
    idx = pd.to_datetime(["2024-01-01", "2024-01-02"])
    monkeypatch.setattr(
        portopt,
        "_po_get_performance_frames",
        lambda *_args, **_kwargs: {
            "source_df": pd.DataFrame({"P1": [0.01, 0.02]}, index=idx),
            "total_df": pd.DataFrame({"P1": [0.01, 0.02]}, index=idx),
            "excess_df": pd.DataFrame({"P1": [0.005, 0.01]}, index=idx),
            "display_cols": ["P1"],
            "benchmark_map": {"P1": "__bm__P1"},
            "periodicity": "daily",
        },
    )

    grid = portopt.po_render_returns(
        {"P1": {"run_inputs": {"selected_series": []}}},
        "returns",
        "P1",
        "excess",
        "raw-json",
        "daily",
        {},
        {},
        None,
        0,
        {},
    )

    assert getattr(grid, "rowData", [])[0]["P1"] == pytest.approx(0.005)


def test_po_render_returns_preserves_dotted_series_fields(monkeypatch, page_modules):
    _, portopt = page_modules
    idx = pd.to_datetime(["2024-01-01", "2024-01-02"])
    dotted_name = "T. Rowe Fund"
    monkeypatch.setattr(
        portopt,
        "_po_get_performance_frames",
        lambda *_args, **_kwargs: {
            "source_df": pd.DataFrame({dotted_name: [0.01, 0.02]}, index=idx),
            "total_df": pd.DataFrame({dotted_name: [0.01, 0.02]}, index=idx),
            "excess_df": pd.DataFrame({dotted_name: [0.01, 0.02]}, index=idx),
            "display_cols": [dotted_name],
            "benchmark_map": {},
            "periodicity": "daily",
        },
    )

    grid = portopt.po_render_returns(
        {"Portfolio": {"run_inputs": {"selected_series": []}}},
        "returns",
        "Portfolio",
        "total",
        "raw-json",
        "daily",
        {},
        {},
        None,
        0,
        {},
    )

    assert getattr(grid, "columnDefs", [])[1]["field"] == dotted_name
    assert getattr(grid, "dashGridOptions", {})["suppressFieldDotNotation"] is True
    assert getattr(grid, "dashGridOptions", {})["processCellForClipboard"] == {
        "function": "dashmatProcessCellForClipboard(params)"
    }
    assert getattr(grid, "rowData", [])[0][dotted_name] == pytest.approx(0.01)


def test_po_render_statistics_requires_active_tab(page_modules):
    _, portopt = page_modules

    grid = portopt.po_render_statistics(
        {"P1": {"returns_json": pd.Series([0.01], index=pd.to_datetime(["2024-01-01"])).to_json(date_format="iso")}},
        "weight",
        "P1",
        None,
        True,
        "daily",
        None,
        {},
        {},
        None,
        0,
        {},
    )

    assert getattr(grid, "children", None) is None


def test_po_update_portfolio_dropdowns_sets_delete_disabled_state(page_modules):
    _, portopt = page_modules

    empty = portopt.po_update_portfolio_dropdowns(None, None, None, None)
    assert empty == ([], None, [], [], True)

    results = {"P1": {"x": 1}, "P2": {"x": 2}}
    options, selected, multi_options, multi_value, delete_disabled = portopt.po_update_portfolio_dropdowns(
        results,
        "P1",
        ["P1"],
        None,
    )

    assert [o["value"] for o in options] == ["P1", "P2"]
    assert selected == "P1"
    assert [o["value"] for o in multi_options] == ["P1", "P2"]
    assert multi_value == ["P1"]
    assert delete_disabled is False


def test_po_update_portfolio_dropdowns_selects_new_result_when_optimization_completes(page_modules):
    _, portopt = page_modules

    results = {"P1": {"x": 1}, "P2": {"x": 2}}
    options, selected, multi_options, multi_value, delete_disabled = portopt.po_update_portfolio_dropdowns(
        results,
        "P1",
        ["P1"],
        {"status": "complete", "name": "P2"},
    )

    assert [o["value"] for o in options] == ["P1", "P2"]
    assert selected == "P2"
    assert [o["value"] for o in multi_options] == ["P1", "P2"]
    assert multi_value == ["P1", "P2"]
    assert delete_disabled is False


def test_po_default_name_for_model_uses_short_aliases(page_modules):
    _, portopt = page_modules

    assert portopt._po_default_name_for_model("risk_parity") == "RP"
    assert portopt._po_default_name_for_model("factor_risk_parity") == "FRP"
    assert portopt._po_default_name_for_model("hierarchical_risk_parity") == "HRP"
    assert portopt._po_default_name_for_model("maximize_sharpe") == "MSR"
    assert portopt._po_default_name_for_model("minimize_variance") == "MinVar"
    assert portopt._po_default_name_for_model("minimize_cvar") == "MinCVaR"
    assert portopt._po_default_name_for_model("equal_weight") == "EW"
    assert portopt._po_default_name_for_model("ex_ante_mv") == "ExAnteMV"
    assert portopt._po_default_name_for_model("black_litterman") == "BL"
    assert portopt._po_default_name_for_model("unknown_model") == "Port"


def test_po_sync_name_with_model_uses_aliases(page_modules):
    _, portopt = page_modules

    assert portopt.po_sync_name_with_model("risk_parity") == "RP"
    assert portopt.po_sync_name_with_model("black_litterman") == "BL"
    assert portopt.po_sync_name_with_model("minimize_variance") == "MinVar"


def test_po_render_growth_chart_table_mode_returns_grid_with_wide_date_column(monkeypatch, page_modules):
    _, portopt = page_modules
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    display_df = pd.DataFrame(
        {
            "P1": [0.01, -0.005, 0.002],
            "Asset_A": [0.008, -0.004, 0.001],
        },
        index=idx,
    )
    monkeypatch.setattr(
        portopt,
        "_po_build_display_series",
        lambda *_args, **_kwargs: (display_df, ["P1", "Asset_A"]),
    )

    grid = portopt.po_render_growth_chart(
        "P1",
        {"P1": {"window_weights": _sample_window_weights()}},
        "growth",
        "table",
        "raw-json",
        "daily",
        {},
        {},
        None,
        0,
        {},
        "light",
    )

    assert getattr(grid, "columnDefs", [])[0]["field"] == "Date"
    assert getattr(grid, "columnDefs", [])[0]["width"] == 112
    assert getattr(grid, "columnDefs", [])[0]["headerClass"] == "dashmat-center-header"
    assert getattr(grid, "columnDefs", [])[0]["cellStyle"] == {"textAlign": "center"}
    assert getattr(grid, "dashGridOptions", {})["enableRangeSelection"] is True
    assert getattr(grid, "defaultColDef", {})["headerClass"] == "dashmat-center-header"
    assert getattr(grid, "defaultColDef", {})["cellStyle"] == {"textAlign": "center"}
    assert len(getattr(grid, "rowData", [])) == 3


def test_po_render_rolling_table_mode_returns_grid_with_wide_date_column(monkeypatch, page_modules):
    _, portopt = page_modules
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    source_df = pd.DataFrame({"P1": [0.01, -0.005, 0.002, 0.003]}, index=idx)
    monkeypatch.setattr(
        portopt,
        "_po_get_performance_frames",
        lambda *_args, **_kwargs: {
            "source_df": source_df,
            "total_df": source_df,
            "excess_df": source_df,
            "display_cols": ["P1"],
            "benchmark_map": {},
            "periodicity": "daily",
        },
    )
    monkeypatch.setattr(
        portopt,
        "calculate_rolling_returns",
        lambda *_args, **_kwargs: pd.DataFrame({"P1": [0.08]}, index=[pd.Timestamp("2024-01-31")]),
    )

    grid = portopt.po_render_rolling(
        {"P1": {}},
        "rolling",
        "P1",
        "daily",
        "1y",
        "annualized",
        "total_return",
        "table",
        None,
        True,
        "raw-json",
        {},
        {},
        None,
        0,
        {},
        "light",
    )

    assert getattr(grid, "columnDefs", [])[0]["field"] == "Date"
    assert getattr(grid, "columnDefs", [])[0]["width"] == 112


def test_po_render_rolling_uses_result_rf_setting_not_live_toggle(monkeypatch, page_modules):
    _, portopt = page_modules
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    source_df = pd.DataFrame({"P1": [0.01, -0.005, 0.002, 0.003]}, index=idx)
    captured = {}

    monkeypatch.setattr(
        portopt,
        "_po_get_performance_frames",
        lambda *_args, **_kwargs: {
            "source_df": source_df,
            "total_df": source_df,
            "excess_df": source_df,
            "display_cols": ["P1"],
            "benchmark_map": {},
            "periodicity": "daily",
        },
    )

    def _fake_rolling(*args, **_kwargs):
        captured["use_risk_free"] = args[-1]
        return pd.DataFrame({"P1": [0.08]}, index=[pd.Timestamp("2024-01-31")])

    monkeypatch.setattr(portopt, "calculate_rolling_returns", _fake_rolling)

    portopt.po_render_rolling(
        {"P1": {"risk_free_meta": {"enabled": False}}},
        "rolling",
        "P1",
        "daily",
        "1y",
        "annualized",
        "sharpe_ratio",
        "table",
        None,
        True,
        "raw-json",
        {},
        {},
        None,
        0,
        {},
        "light",
    )

    assert captured["use_risk_free"] is False


def test_po_render_drawdown_table_mode_returns_grid_with_wide_date_column(monkeypatch, page_modules):
    _, portopt = page_modules
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    source_df = pd.DataFrame({"P1": [0.01, -0.005, 0.002, 0.003]}, index=idx)
    monkeypatch.setattr(
        portopt,
        "_po_get_performance_frames",
        lambda *_args, **_kwargs: {
            "source_df": source_df,
            "total_df": source_df,
            "excess_df": source_df,
            "display_cols": ["P1"],
            "benchmark_map": {},
            "periodicity": "daily",
        },
    )
    monkeypatch.setattr(
        portopt,
        "calculate_drawdown",
        lambda *_args, **_kwargs: pd.DataFrame({"P1": [0.0, -0.02]}, index=[pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-31")]),
    )

    grid = portopt.po_render_drawdown(
        {"P1": {}},
        "drawdown",
        "P1",
        "daily",
        "table",
        "total",
        "raw-json",
        {},
        {},
        None,
        0,
        {},
        "light",
    )

    assert getattr(grid, "columnDefs", [])[0]["field"] == "Date"
    assert getattr(grid, "columnDefs", [])[0]["width"] == 112


def test_po_render_calendar_passes_returns_basis_to_shared_helper(monkeypatch, page_modules):
    _, portopt = page_modules
    idx = pd.to_datetime(["2024-01-01", "2024-01-02"])
    captured = {}

    monkeypatch.setattr(
        portopt,
        "_po_get_performance_frames",
        lambda *_args, **_kwargs: {
            "source_df": pd.DataFrame({"P1": [0.01, 0.02], "__bm__P1": [0.005, 0.01]}, index=idx),
            "total_df": pd.DataFrame({"P1": [0.01, 0.02]}, index=idx),
            "excess_df": pd.DataFrame({"P1": [0.005, 0.01]}, index=idx),
            "display_cols": ["P1"],
            "benchmark_map": {"P1": "__bm__P1"},
            "periodicity": "daily",
        },
    )

    def _fake_calendar_year_returns(raw_data, original_periodicity, selected_periodicity, selected_series, returns_type, benchmark_assignments, *_args, **_kwargs):
        captured["returns_type"] = returns_type
        captured["benchmark_assignments"] = benchmark_assignments
        return pd.DataFrame({"P1": [0.1]}, index=[2024])

    monkeypatch.setattr(portopt, "calculate_calendar_year_returns", _fake_calendar_year_returns)

    grid = portopt.po_render_calendar(
        {"P1": {"run_inputs": {"selected_series": []}}},
        "calendar",
        "P1",
        "daily",
        "annual",
        None,
        "excess",
        "raw-json",
        {},
        {},
        None,
        0,
        {},
    )

    assert getattr(grid, "columnDefs", [])[0]["field"] == "Year"
    assert captured["returns_type"] == "excess"
    assert captured["benchmark_assignments"] == '{"P1":"__bm__P1"}'


def test_po_render_drawdown_passes_returns_basis_to_shared_helper(monkeypatch, page_modules):
    _, portopt = page_modules
    idx = pd.to_datetime(["2024-01-01", "2024-01-02"])
    captured = {}

    monkeypatch.setattr(
        portopt,
        "_po_get_performance_frames",
        lambda *_args, **_kwargs: {
            "source_df": pd.DataFrame({"P1": [0.01, 0.02], "__bm__P1": [0.005, 0.01]}, index=idx),
            "total_df": pd.DataFrame({"P1": [0.01, 0.02]}, index=idx),
            "excess_df": pd.DataFrame({"P1": [0.005, 0.01]}, index=idx),
            "display_cols": ["P1"],
            "benchmark_map": {"P1": "__bm__P1"},
            "periodicity": "daily",
        },
    )

    def _fake_drawdown(raw_data, periodicity, selected_series, returns_type, benchmark_assignments, *_args, **_kwargs):
        captured["returns_type"] = returns_type
        captured["benchmark_assignments"] = benchmark_assignments
        return pd.DataFrame({"P1": [0.0, -0.02]}, index=[pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-31")])

    monkeypatch.setattr(portopt, "calculate_drawdown", _fake_drawdown)

    grid = portopt.po_render_drawdown(
        {"P1": {"run_inputs": {"selected_series": []}}},
        "drawdown",
        "P1",
        "daily",
        "table",
        "excess",
        "raw-json",
        {},
        {},
        None,
        0,
        {},
        "light",
    )

    assert getattr(grid, "columnDefs", [])[0]["field"] == "Date"
    assert captured["returns_type"] == "excess"
    assert captured["benchmark_assignments"] == '{"P1":"__bm__P1"}'


def test_po_populate_frontier_windows_disables_for_ex_ante_model(page_modules):
    _, portopt = page_modules
    results = {
        "P1": {
            "config": {"model": "ex_ante_mv"},
            "window_weights": _sample_window_weights(),
        }
    }

    options, value, disabled = portopt.po_populate_frontier_windows("P1", results, "frontier")
    assert len(options) == 2
    assert value == "1"
    assert disabled is True


def test_po_render_turnover_table_computes_turnover(page_modules):
    _, portopt = page_modules
    results = {
        "P1": {
            "window_weights": _sample_window_weights(),
        }
    }

    grid = portopt.po_render_turnover_table("P1", results, "turnover", "table")
    assert getattr(grid, "columnDefs", [])[0]["field"] == "Rebalance Date"
    assert getattr(grid, "rowData", [])[0]["Turnover"] == pytest.approx(0.1)


def test_po_delete_portfolio_removes_result_but_keeps_saved_series(page_modules, raw_json):
    _, portopt = page_modules
    df = pd.read_json(StringIO(raw_json), orient="split")
    df["P1"] = 0.0
    raw_with_portfolio = df_to_json(df)
    results = {"P1": {"x": 1}, "P2": {"x": 2}}

    new_results, new_raw, new_sel = portopt.po_delete_portfolio(1, "P1", results, raw_with_portfolio)

    assert "P1" not in new_results
    assert new_raw is no_update
    assert new_sel == "P2"


def test_po_run_optimization_returns_error_when_working_df_empty(monkeypatch, page_modules):
    _, portopt = page_modules
    monkeypatch.setattr(portopt, "_po_get_working_returns", lambda *_args, **_kwargs: pd.DataFrame())

    with pytest.raises(PreventUpdate):
        # Guard path first: no click should PreventUpdate.
        portopt.po_run_optimization(
            0, "raw", "daily", "daily", ["Asset_A", "Asset_B"], {}, {}, {}, None, 0, {},
            {}, {}, {}, False, 63, "none", "scaled_identity", "MyPortfolio", "full", 252, 21, "periods",
            "risk_parity", "fill_na", "off", {}, [],
            {}, {}, [], 0.05, "maximize_sharpe",
            {}, {}, "ret_cov", [], True, None,
        )

    # Now force callback path and verify returned error payload.
    result = portopt.po_run_optimization(
        1, "raw", "daily", "daily", ["Asset_A", "Asset_B"], {}, {}, {}, None, 0, {},
        {}, {}, {}, False, 63, "none", "scaled_identity", "MyPortfolio", "full", 252, 21, "periods",
        "risk_parity", "fill_na", "off", {}, [],
        {}, {}, [], 0.05, "maximize_sharpe",
        {}, {}, "ret_cov", [], True, None,
    )
    status = result[2]
    assert status["status"] == "error"
    assert "No data available" in status["message"]


def test_po_run_optimization_blocks_split_reporting_without_full_benchmark_coverage(monkeypatch, page_modules, raw_json):
    _, portopt = page_modules
    df = pd.read_json(StringIO(raw_json), orient="split")[["Asset_A", "Asset_B"]]
    df.index = pd.to_datetime(df.index)
    monkeypatch.setattr(portopt, "_po_get_working_returns", lambda *_args, **_kwargs: df.copy())

    _results, _new_raw, status, _pending = portopt.po_run_optimization(
        1,
        raw_json,
        "daily",
        "daily",
        ["Asset_A", "Asset_B"],
        {"Asset_A": "None", "Asset_B": "Asset_B"},
        {},
        {"Asset_A": True},
        None,
        0,
        {},
        {},
        {},
        {},
        False,
        63,
        "none",
        "scaled_identity",
        "MyPort",
        "full",
        252,
        1,
        "months",
        "risk_parity",
        "fill_na",
        "off",
        {},
        [],
        {},
        {},
        [],
        0.05,
        "maximize_sharpe",
        {},
        {},
        "ret_cov",
        [],
        True,
        True,
        None,
    )

    assert status["status"] == "error"
    assert "benchmark assignment" in status["message"].lower()


def test_po_run_optimization_stores_split_reporting_payloads(monkeypatch, page_modules, raw_json):
    _, portopt = page_modules
    opt_df = pd.read_json(StringIO(raw_json), orient="split")[["Asset_A", "Asset_B"]].head(3)
    opt_df.index = pd.to_datetime(opt_df.index)
    reporting_df = opt_df.copy() * pd.Series({"Asset_A": 1.5, "Asset_B": 0.5})
    benchmark_df = pd.DataFrame(
        {
            "Asset_A": [0.001, 0.002, 0.003],
            "Asset_B": [0.0005, 0.001, 0.0015],
        },
        index=opt_df.index,
    )

    monkeypatch.setattr(portopt, "_po_get_working_returns", lambda *_args, **_kwargs: opt_df.copy())
    monkeypatch.setattr(
        portopt,
        "_po_get_result_basis_bundle",
        lambda *_args, **_kwargs: (
            {
                "selected_series": ["Asset_A", "Asset_B"],
                "benchmark_assignments": {"Asset_A": "Bench_A", "Asset_B": "Bench_B"},
                "cmabench_assignments": {"Asset_A": "BCTBill13", "Asset_B": "BCTBill13"},
                "long_short_assignments": {"Asset_A": True},
                "date_range": None,
                "vol_scaler": 0.0,
                "vol_scaling_assignments": {},
                "periodicity": "daily",
            },
            {
                "optimization_df": opt_df.copy(),
                "reporting_df": reporting_df.copy(),
                "benchmark_asset_df": benchmark_df.copy(),
            },
        ),
    )
    monkeypatch.setattr(
        portopt,
        "run_portfolio_optimization",
        lambda *_args, **_kwargs: (
            [
                SimpleNamespace(
                    apply_start=opt_df.index[0],
                    apply_end=opt_df.index[-1],
                    est_start=opt_df.index[0],
                    est_end=opt_df.index[-1],
                    weights={"Asset_A": 0.6, "Asset_B": 0.4},
                )
            ],
            pd.Series([0.02, 0.01, -0.005], index=opt_df.index),
        ),
    )
    monkeypatch.setattr(
        portopt,
        "_po_resolve_frontier_snapshot",
        lambda **_kwargs: {
            "window_index": 0,
            "risk_measure": "MV",
            "asset_order": ["Asset_A", "Asset_B"],
            "portfolio": {"name": "MyPort", "return": 0.1, "risk": 0.2, "weights": {"Asset_A": 0.6, "Asset_B": 0.4}},
            "assets": [],
            "frontier_points": [],
            "frontier_portfolios": [],
            "window_est_start": "2024-01-01",
            "window_est_end": "2024-01-03",
        },
    )

    results_out, new_raw, status, pending = portopt.po_run_optimization(
        1,
        raw_json,
        "daily",
        "daily",
        ["Asset_A", "Asset_B"],
        {"Asset_A": "Bench_A", "Asset_B": "Bench_B"},
        {"Asset_A": "BCTBill13", "Asset_B": "BCTBill13"},
        {"Asset_A": True},
        None,
        0,
        {},
        {},
        {},
        {},
        False,
        63,
        "none",
        "scaled_identity",
        "MyPort",
        "full",
        252,
        1,
        "months",
        "risk_parity",
        "fill_na",
        "off",
        {},
        [],
        {},
        {},
        [],
        0.05,
        "maximize_sharpe",
        {},
        {},
        "ret_cov",
        [],
        True,
        True,
        None,
    )

    assert status["status"] == "complete"
    assert new_raw is no_update
    assert pending is no_update
    entry = results_out["MyPort"]
    assert entry["reporting_basis"] == "long_only_performance"
    assert "reporting_returns_json" in entry
    assert "optimization_returns_json" in entry
    assert "benchmark_returns_json" in entry
    assert entry["run_inputs"]["benchmark_assignments"]["Asset_A"] == "Bench_A"
    reporting_series = pd.read_json(StringIO(entry["reporting_returns_json"]), typ="series")
    optimization_series = pd.read_json(StringIO(entry["optimization_returns_json"]), typ="series")
    benchmark_series = pd.read_json(StringIO(entry["benchmark_returns_json"]), typ="series")
    assert not reporting_series.equals(optimization_series)
    assert len(benchmark_series) == len(reporting_series)


def test_po_toggle_ui_elements_sets_validation_tooltip(page_modules):
    _, portopt = page_modules

    run_disabled, tooltip, tooltip_disabled, save_disabled, download_disabled = (
        portopt.po_toggle_ui_elements(
            {"phase": "idle", "loadedTabs": {}},
            "MyPortfolio",
            ["Asset_A"],
            "risk_parity",
            "full",
            252,
            1,
            "months",
            False,
            63,
            "none",
            "scaled_identity",
            {},
            {},
            {},
            [],
            "ret_cov",
            {},
            {},
            {},
            {},
            [],
            0.05,
            {"display": "none"},
            {"has_results": True, "count": 1},
        )
    )

    assert run_disabled is True
    assert tooltip == "Loading controls..."
    assert tooltip_disabled is False
    assert save_disabled is False
    assert download_disabled is False


def test_po_sync_reporting_basis_control_disables_when_ineligible(page_modules):
    _, portopt = page_modules

    disabled, value, help_text = portopt.po_sync_reporting_basis_control(
        "maximize_sharpe",
        ["Asset_A", "Asset_B"],
        {"Asset_A": True},
        "split",
    )

    assert disabled is True
    assert value == "match"
    assert "supported risk-based models" in help_text


def test_po_toggle_ui_elements_waits_for_restore_completion(page_modules):
    _, portopt = page_modules

    run_disabled, tooltip, tooltip_disabled, save_disabled, download_disabled = (
        portopt.po_toggle_ui_elements(
            {"phase": "ready", "loadedTabs": {"weight": True}},
            "MyPortfolio",
            [],
            "risk_parity",
            "rolling",
            252,
            1,
            "months",
            False,
            63,
            "none",
            "scaled_identity",
            {},
            {},
            {},
            [],
            "ret_cov",
            {},
            {},
            {},
            {},
            [],
            0.05,
            {"display": "none"},
            {},
        )
    )

    assert run_disabled is True
    assert "Select at least one series" in tooltip
    assert tooltip_disabled is False
    assert save_disabled is False
    assert download_disabled is True


def test_po_toggle_ui_elements_ex_ante_requires_complete_expected_inputs(page_modules):
    _, portopt = page_modules

    run_disabled, tooltip, *_rest = portopt.po_toggle_ui_elements(
        {"phase": "ready", "loadedTabs": {"weight": True}},
        "MyPortfolio",
        ["Asset_A", "Asset_B"],
        "ex_ante_mv",
        "full",
        252,
        1,
        "months",
        False,
        63,
        "none",
        "scaled_identity",
        {},
        {},
        {},
        [],
        "ret_cov",
        {"Asset_A": 0.08},
        {"Asset_A": {"Asset_A": 0.04, "Asset_B": 0.01}},
        {},
        {},
        [],
        0.05,
        {"display": "none"},
        {},
    )

    assert run_disabled is True
    assert "Missing expected return" in tooltip


def test_validate_optimization_inputs_accepts_lambda_decay(page_modules):
    _, portopt = page_modules

    err = portopt._validate_optimization_inputs(
        portfolio_name="MyPortfolio",
        selected_series=["Asset_A", "Asset_B"],
        opt_model="risk_parity",
        opt_window="rolling",
        window_size=252,
        opt_step=1,
        opt_step_unit="months",
        exp_wt_cov=True,
        halflife=0.94,
        cov_shrinkage="none",
        cov_shrinkage_target="scaled_identity",
        min_wt={},
        max_wt={},
        force_max={},
        linear_constraints=[],
        ex_ante_mode="ret_cov",
        ex_ante_returns={},
        ex_ante_cov={},
        ex_ante_vol={},
        ex_ante_corr={},
        bl_views=[],
        bl_tau=0.05,
    )

    assert err is None


def test_validate_optimization_inputs_rejects_non_positive_decay(page_modules):
    _, portopt = page_modules

    err = portopt._validate_optimization_inputs(
        portfolio_name="MyPortfolio",
        selected_series=["Asset_A", "Asset_B"],
        opt_model="risk_parity",
        opt_window="rolling",
        window_size=252,
        opt_step=1,
        opt_step_unit="months",
        exp_wt_cov=True,
        halflife=0,
        cov_shrinkage="none",
        cov_shrinkage_target="scaled_identity",
        min_wt={},
        max_wt={},
        force_max={},
        linear_constraints=[],
        ex_ante_mode="ret_cov",
        ex_ante_returns={},
        ex_ante_cov={},
        ex_ante_vol={},
        ex_ante_corr={},
        bl_views=[],
        bl_tau=0.05,
    )

    assert err == "Decay input must be greater than 0 when exponential weighting is enabled."


def test_validate_optimization_inputs_rejects_invalid_cov_shrinkage(page_modules):
    _, portopt = page_modules

    err = portopt._validate_optimization_inputs(
        portfolio_name="MyPortfolio",
        selected_series=["Asset_A", "Asset_B"],
        opt_model="risk_parity",
        opt_window="rolling",
        window_size=252,
        opt_step=1,
        opt_step_unit="months",
        exp_wt_cov=False,
        halflife=63,
        cov_shrinkage="bad_value",
        cov_shrinkage_target="scaled_identity",
        min_wt={},
        max_wt={},
        force_max={},
        linear_constraints=[],
        ex_ante_mode="ret_cov",
        ex_ante_returns={},
        ex_ante_cov={},
        ex_ante_vol={},
        ex_ante_corr={},
        bl_views=[],
        bl_tau=0.05,
    )

    assert err == "Select a valid covariance shrinkage option."


def test_validate_optimization_inputs_rejects_invalid_cov_shrinkage_target(page_modules):
    _, portopt = page_modules

    err = portopt._validate_optimization_inputs(
        portfolio_name="MyPortfolio",
        selected_series=["Asset_A", "Asset_B"],
        opt_model="risk_parity",
        opt_window="rolling",
        window_size=252,
        opt_step=1,
        opt_step_unit="months",
        exp_wt_cov=False,
        halflife=63,
        cov_shrinkage="ledoit_wolf",
        cov_shrinkage_target="bad_target",
        min_wt={},
        max_wt={},
        force_max={},
        linear_constraints=[],
        ex_ante_mode="ret_cov",
        ex_ante_returns={},
        ex_ante_cov={},
        ex_ante_vol={},
        ex_ante_corr={},
        bl_views=[],
        bl_tau=0.05,
    )

    assert err == "Select a valid covariance shrinkage target."


def test_po_estimate_matrix_store_uses_selected_shrinkage(monkeypatch, page_modules, raw_json):
    _, portopt = page_modules
    captured = {}

    def _fake_estimate_covariance_matrix(df, **kwargs):
        captured["kwargs"] = kwargs
        cols = list(df.columns)
        return pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)

    monkeypatch.setattr(portopt, "estimate_covariance_matrix", _fake_estimate_covariance_matrix)

    cov_store, corr_store, rows = portopt.po_estimate_matrix_store(
        1,
        raw_json,
        ["Asset_A", "Asset_B"],
        "ret_cov",
        "daily",
        False,
        63,
        "ledoit_wolf",
        "constant_correlation",
    )

    assert captured["kwargs"]["shrinkage"] == "ledoit_wolf"
    assert captured["kwargs"]["shrinkage_target"] == "constant_correlation"
    assert set(cov_store) == {"Asset_A", "Asset_B"}
    assert corr_store is no_update
    assert len(rows) == 2


def test_po_estimate_matrix_store_ignores_shrinkage_when_exp_weighted(monkeypatch, page_modules, raw_json):
    _, portopt = page_modules
    captured = {}

    def _fake_estimate_covariance_matrix(df, **kwargs):
        captured["kwargs"] = kwargs
        cols = list(df.columns)
        return pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)

    monkeypatch.setattr(portopt, "estimate_covariance_matrix", _fake_estimate_covariance_matrix)

    cov_store, corr_store, rows = portopt.po_estimate_matrix_store(
        1,
        raw_json,
        ["Asset_A", "Asset_B"],
        "ret_cov",
        "daily",
        True,
        0.94,
        "oas",
        "constant_correlation",
    )

    assert captured["kwargs"]["exp_weighted"] is True
    assert captured["kwargs"]["shrinkage"] == "none"
    assert captured["kwargs"]["shrinkage_target"] == "scaled_identity"
    assert set(cov_store) == {"Asset_A", "Asset_B"}
    assert corr_store is no_update
    assert len(rows) == 2


def test_po_update_frontier_risk_measure_options_restricts_ex_ante(page_modules):
    _, portopt = page_modules
    results = {"P1": {"config": {"model": "ex_ante_mv"}}}

    options, value = portopt.po_update_frontier_risk_measure_options("P1", results, "CVaR")
    assert options == [{"value": "MV", "label": "Volatility"}]
    assert value == "MV"


def test_po_resolve_frontier_snapshot_uses_existing_cache_for_standard_model(monkeypatch, page_modules):
    _, portopt = page_modules
    snapshot = {
        "window_index": 1,
        "risk_measure": "MV",
        "asset_order": ["Asset_A", "Asset_B"],
        "portfolio": {"name": "P1", "return": 0.1, "risk": 0.2, "weights": {"Asset_A": 0.6, "Asset_B": 0.4}},
        "assets": [],
        "frontier_points": [{"return": 0.1, "risk": 0.2}],
        "frontier_portfolios": [],
        "window_est_start": "2024-01-01",
        "window_est_end": "2024-01-31",
    }
    portfolio_data = {
        "config": {"model": "risk_parity", "selected_series": ["Asset_A", "Asset_B"]},
        "window_weights": _sample_window_weights(),
        "frontier_cache": {"1": {"MV": snapshot}},
    }
    monkeypatch.setattr(
        portopt,
        "_po_compute_frontier_snapshot_cached",
        lambda *_args, **_kwargs: pytest.fail("memoized frontier builder should not run on cache hit"),
    )

    resolved = portopt._po_resolve_frontier_snapshot(
        selected_portfolio="P1",
        portfolio_data=portfolio_data,
        raw_data="{}",
        periodicity="daily",
        bench={},
        ls={},
        vol_scaler=0,
        vol_scaling={},
        window_idx="1",
        rm="MV",
        linear_constraints=[],
        saved_series_store=None,
        cmabench_assignments=None,
    )

    assert resolved == snapshot


def test_po_render_frontier_table_includes_frontier_points_and_weights(monkeypatch, page_modules):
    _, portopt = page_modules
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    raw_json = df_to_json(pd.DataFrame({"Asset_A": 0.01, "Asset_B": 0.02}, index=idx))
    snapshot = {
        "asset_order": ["Asset_A", "Asset_B"],
        "risk_measure": "MV",
        "portfolio": {"name": "P1", "return": 0.1, "risk": 0.2, "weights": {"Asset_A": 0.6, "Asset_B": 0.4}},
        "assets": [
            {"name": "Asset_A", "return": 0.12, "risk": 0.25},
            {"name": "Asset_B", "return": 0.08, "risk": 0.18},
        ],
        "frontier_portfolios": [
            {"point_index": 0, "return": 0.09, "risk": 0.19, "weights": {"Asset_A": 0.5, "Asset_B": 0.5}},
        ],
    }
    monkeypatch.setattr(portopt, "_po_resolve_frontier_snapshot", lambda **_kwargs: snapshot)

    results = {
        "P1": {
            "config": {"model": "risk_parity", "selected_series": ["Asset_A", "Asset_B"]},
            "window_weights": _sample_window_weights(),
        }
    }

    grid = portopt.po_render_frontier_table(
        "P1",
        results,
        "frontier",
        "table",
        {"phase": "ready", "loadedTabs": {"frontier": True}},
        "1",
        "MV",
        raw_json,
        "daily",
        {},
        {},
        0,
        {},
        {},
        None,
        True,
        [],
    )

    assert any(col["field"] == "Wt_Asset_A" for col in getattr(grid, "columnDefs", []))
    assert any(col["field"] == "Sharpe Ratio" for col in getattr(grid, "columnDefs", []))
    assert any(row["Type"] == "Optimized Portfolio" for row in getattr(grid, "rowData", []))
    assert any(row["Type"] == "Frontier Point" for row in getattr(grid, "rowData", []))


def test_po_render_frontier_chart_uses_shared_snapshot_resolver(monkeypatch, page_modules):
    _, portopt = page_modules
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    raw_json = df_to_json(pd.DataFrame({"Asset_A": 0.01, "Asset_B": 0.02}, index=idx))
    calls = {"count": 0}
    snapshot = {
        "asset_order": ["Asset_A", "Asset_B"],
        "risk_measure": "MV",
        "portfolio": {"name": "P1", "return": 0.1, "risk": 0.2, "weights": {"Asset_A": 0.6, "Asset_B": 0.4}},
        "assets": [{"name": "Asset_A", "return": 0.12, "risk": 0.25}],
        "frontier_points": [{"return": 0.09, "risk": 0.19}],
        "frontier_portfolios": [],
    }

    def _fake_resolve(**_kwargs):
        calls["count"] += 1
        return snapshot

    monkeypatch.setattr(portopt, "_po_resolve_frontier_snapshot", _fake_resolve)

    comp = portopt.po_render_frontier_chart(
        "P1",
        {"P1": {"config": {"model": "risk_parity", "selected_series": ["Asset_A", "Asset_B"]}, "window_weights": _sample_window_weights()}},
        "frontier",
        "chart",
        {"phase": "ready", "loadedTabs": {"frontier": True}},
        "1",
        "MV",
        raw_json,
        "daily",
        {},
        {},
        None,
        0,
        {},
        {},
        None,
        True,
        ["Asset_A", "Asset_B"],
        "light",
        [],
    )

    assert calls["count"] == 1
    assert type(comp).__name__ == "Loading"


def test_po_render_frontier_chart_reports_missing_source_series(page_modules, raw_json):
    _, portopt = page_modules
    raw_df = pd.read_json(StringIO(raw_json), orient="split")[["Asset_A"]]
    raw_df.index = pd.to_datetime(raw_df.index)
    results = {
        "P1": {
            "window_weights": _sample_window_weights(),
            "config": {"selected_series": ["Asset_A", "Asset_B"], "model": "risk_parity"},
        }
    }

    comp = portopt.po_render_frontier_chart(
        "P1",
        results,
        "frontier",
        "chart",
        {"phase": "ready", "loadedTabs": {"frontier": True}},
        "0",
        "MV",
        df_to_json(raw_df),
        "daily",
        {},
        {},
        None,
        0,
        {},
        {},
        None,
        True,
        ["Asset_A", "Asset_B"],
        "light",
        [],
    )

    assert "Missing source series: Asset_B" in " ".join(_collect_component_text(comp))


def test_po_run_optimization_stores_default_frontier_cache_for_standard_model(monkeypatch, page_modules, raw_json):
    _, portopt = page_modules
    df = pd.read_json(StringIO(raw_json), orient="split")[["Asset_A", "Asset_B"]]
    df.index = pd.to_datetime(df.index)

    monkeypatch.setattr(portopt, "_po_get_working_returns", lambda *_args, **_kwargs: df.copy())
    monkeypatch.setattr(
        portopt,
        "run_portfolio_optimization",
        lambda *_args, **_kwargs: (
            [
                SimpleNamespace(
                    apply_start=df.index[0],
                    apply_end=df.index[-1],
                    est_start=df.index[0],
                    est_end=df.index[-1],
                    weights={"Asset_A": 0.5, "Asset_B": 0.5},
                )
            ],
            pd.Series(0.001, index=df.index),
        ),
    )
    def _fake_resolve_frontier_snapshot(**kwargs):
        snapshot = {
            "window_index": 0,
            "risk_measure": "MV",
            "asset_order": ["Asset_A", "Asset_B"],
            "portfolio": {"name": "MyPort", "return": 0.1, "risk": 0.2, "weights": {"Asset_A": 0.5, "Asset_B": 0.5}},
            "assets": [],
            "frontier_points": [],
            "frontier_portfolios": [],
            "window_est_start": "2024-01-01",
            "window_est_end": "2024-01-31",
        }
        if kwargs.get("persist_cache"):
            kwargs["portfolio_data"]["frontier_cache"] = {"0": {"MV": snapshot}}
        return snapshot

    monkeypatch.setattr(portopt, "_po_resolve_frontier_snapshot", _fake_resolve_frontier_snapshot)

    results_out, new_raw, status, pending = portopt.po_run_optimization(
        1,
        raw_json,
        "daily",
        "daily",
        ["Asset_A", "Asset_B"],
        {},
        {},
        {},
        None,
        0,
        {},
        {},
        {},
        {},
        False,
        63,
        "none",
        "scaled_identity",
        "MyPort",
        "full",
        252,
        1,
        "months",
        "risk_parity",
        "fill_na",
        "off",
        {},
        [],
        {},
        {},
        [],
        0.05,
        "maximize_sharpe",
        {},
        {},
        "ret_cov",
        [],
        True,
        None,
    )

    assert status["status"] == "complete"
    assert "frontier_cache" in results_out["MyPort"]
    assert "MV" in results_out["MyPort"]["frontier_cache"]["0"]
    assert new_raw is no_update
    assert pending is no_update


def test_po_render_frontier_rf_warning_non_ex_ante_skips_snapshot_lookup(monkeypatch, page_modules):
    _, portopt = page_modules
    monkeypatch.setattr(
        portopt,
        "_get_cached_frontier_snapshot",
        lambda *_args, **_kwargs: pytest.fail("non-ex-ante RF warning should not read frontier snapshots"),
    )
    monkeypatch.setattr(
        portopt,
        "_resolve_risk_free_context",
        lambda **_kwargs: {"rf_warning": "Warning text"},
    )

    warning, style = portopt.po_render_frontier_rf_warning(
        "P1",
        {"P1": {"config": {"model": "risk_parity", "selected_series": ["Asset_A", "Asset_B"]}, "window_weights": _sample_window_weights()}},
        "frontier",
        "1",
        "MV",
        True,
        "daily",
        None,
        None,
    )

    assert "Warning text" in " ".join(_collect_component_text(warning))
    assert style["display"] == "block"


def test_po_render_statistics_uses_stored_portfolio_benchmark(monkeypatch, page_modules):
    _, portopt = page_modules
    idx = pd.to_datetime(["2024-01-01", "2024-01-02"])
    captured = {}

    def _fake_calculate_statistics_cached(raw_json, periodicity, selected_series, benchmark_assignments, *_args, **_kwargs):
        dataset_key = resolve_dataset_key(raw_json)
        if dataset_key and not str(raw_json).startswith("{"):
            captured["df"] = get_raw_dataset_df(dataset_key)
        else:
            captured["df"] = pd.read_json(StringIO(raw_json), orient="split")
        captured["periodicity"] = periodicity
        captured["selected_series"] = selected_series
        captured["benchmark_assignments"] = benchmark_assignments
        return [{"Series": "P1"}]

    monkeypatch.setattr(portopt, "calculate_statistics_cached", _fake_calculate_statistics_cached)

    results = {
        "P1": {
            "reporting_returns_json": pd.Series([0.01, 0.02], index=idx).to_json(date_format="iso"),
            "benchmark_returns_json": pd.Series([0.005, 0.01], index=idx).to_json(date_format="iso"),
            "run_inputs": {"selected_series": [], "periodicity": "weekly_friday"},
            "risk_free_meta": {"enabled": False},
        }
    }

    component = portopt.po_render_statistics(
        results,
        "statistics",
        "P1",
        None,
        False,
        "daily",
        None,
        {"live": "bench"},
        {"live": True},
        None,
        0,
        {},
    )

    assert component is not None
    assert captured["periodicity"] == "weekly_friday"
    assert captured["selected_series"] == ("P1",)
    assert captured["benchmark_assignments"] == '{"P1":"__bm__P1"}'
    assert "__bm__P1" in captured["df"].columns


def test_po_render_frontier_rf_warning_uses_result_rf_setting_not_live_toggle(monkeypatch, page_modules):
    _, portopt = page_modules
    monkeypatch.setattr(
        portopt,
        "_resolve_risk_free_context",
        lambda **_kwargs: pytest.fail("stored disabled RF should bypass warning resolution"),
    )

    warning, style = portopt.po_render_frontier_rf_warning(
        "P1",
        {
            "P1": {
                "config": {"model": "risk_parity", "selected_series": ["Asset_A", "Asset_B"]},
                "window_weights": _sample_window_weights(),
                "risk_free_meta": {"enabled": False},
            }
        },
        "frontier",
        "1",
        "MV",
        True,
        "daily",
        None,
        None,
    )

    assert warning == ""
    assert style["display"] == "none"


def test_po_resolve_frontier_snapshot_prefers_stored_run_inputs(monkeypatch, page_modules):
    _, portopt = page_modules
    captured = {}

    monkeypatch.setattr(portopt, "_get_cached_frontier_snapshot", lambda *_args, **_kwargs: None)

    def _fake_compute_frontier_snapshot_cached(
        selected_portfolio,
        raw_data,
        periodicity,
        bench_payload,
        ls_payload,
        vol_scaler,
        vol_scaling_payload,
        *_args
    ):
        captured["periodicity"] = periodicity
        captured["bench_payload"] = bench_payload
        captured["ls_payload"] = ls_payload
        captured["vol_scaler"] = vol_scaler
        captured["vol_scaling_payload"] = vol_scaling_payload
        return portopt.canonical_json_dumps(
            {
                "window_index": 0,
                "risk_measure": "MV",
                "asset_order": ["Asset_A", "Asset_B"],
                "portfolio": {"name": "P1", "return": 0.1, "risk": 0.2, "weights": {"Asset_A": 0.6, "Asset_B": 0.4}},
                "assets": [],
                "frontier_points": [{"return": 0.1, "risk": 0.2}],
                "frontier_portfolios": [],
                "window_est_start": "2024-01-01",
                "window_est_end": "2024-01-31",
            }
        )

    monkeypatch.setattr(portopt, "_po_compute_frontier_snapshot_cached", _fake_compute_frontier_snapshot_cached)

    snapshot = portopt._po_resolve_frontier_snapshot(
        selected_portfolio="P1",
        portfolio_data={
            "window_weights": _sample_window_weights(),
            "config": {"model": "risk_parity", "selected_series": ["Asset_A", "Asset_B"]},
            "run_inputs": {
                "selected_series": ["Asset_A", "Asset_B"],
                "benchmark_assignments": {"Asset_A": "Bench_A", "Asset_B": "Bench_B"},
                "cmabench_assignments": {"Asset_A": "BCTBill13", "Asset_B": "BCTBill13"},
                "long_short_assignments": {"Asset_A": True},
                "date_range": {"start": "2024-01-01", "end": "2024-01-31"},
                "vol_scaler": 7.5,
                "vol_scaling_assignments": {"Asset_A": False},
                "periodicity": "weekly_friday",
            },
        },
        raw_data="raw",
        periodicity="daily",
        bench={"live": "bench"},
        ls={"live": True},
        vol_scaler=0,
        vol_scaling={},
        window_idx=None,
        rm="MV",
        linear_constraints=[],
        saved_series_store=None,
        cmabench_assignments={"live": "cma"},
        use_risk_free=False,
    )

    assert snapshot["risk_measure"] == "MV"
    assert captured["periodicity"] == "weekly_friday"
    assert captured["bench_payload"] == '{"Asset_A":"Bench_A","Asset_B":"Bench_B"}'
    assert captured["ls_payload"] == '{"Asset_A":true}'
    assert captured["vol_scaler"] == 7.5
    assert captured["vol_scaling_payload"] == '{"Asset_A":false}'


def test_po_run_optimization_stores_frontier_cache_for_ex_ante(monkeypatch, page_modules, raw_json):
    _, portopt = page_modules
    df = pd.read_json(StringIO(raw_json), orient="split")[["Asset_A", "Asset_B"]]
    df.index = pd.to_datetime(df.index)

    monkeypatch.setattr(portopt, "_po_get_working_returns", lambda *_args, **_kwargs: df.copy())
    monkeypatch.setattr(
        portopt,
        "run_portfolio_optimization",
        lambda *_args, **_kwargs: (
            [
                SimpleNamespace(
                    apply_start=df.index[0],
                    apply_end=df.index[-1],
                    est_start=df.index[0],
                    est_end=df.index[-1],
                    weights={"Asset_A": 0.5, "Asset_B": 0.5},
                )
            ],
            pd.Series(0.001, index=df.index),
        ),
    )
    monkeypatch.setattr(
        portopt,
        "_po_resolve_frontier_snapshot",
        lambda **kwargs: (
            kwargs["portfolio_data"].setdefault(
                "frontier_cache",
                {
                    "0": {
                        "MV": {
                            "window_index": 0,
                            "risk_measure": "MV",
                            "asset_order": ["Asset_A", "Asset_B"],
                            "portfolio": {"name": "MyPort", "return": 0.1, "risk": 0.2, "weights": {"Asset_A": 0.5, "Asset_B": 0.5}},
                            "assets": [],
                            "frontier_points": [],
                            "frontier_portfolios": [],
                            "window_est_start": "2024-01-01",
                            "window_est_end": "2024-01-31",
                        }
                    }
                },
            )
            or kwargs["portfolio_data"]["frontier_cache"]["0"]["MV"]
        ),
    )

    ex_cov = {
        "Asset_A": {"Asset_A": 0.04, "Asset_B": 0.01},
        "Asset_B": {"Asset_A": 0.01, "Asset_B": 0.09},
    }

    results_out, new_raw, status, pending = portopt.po_run_optimization(
        1,
        raw_json,
        "daily",
        "daily",
        ["Asset_A", "Asset_B"],
        {},
        {},
        {},
        None,
        0,
        {},
        {},
        {},
        {},
        False,
        63,
        "none",
        "scaled_identity",
        "MyPort",
        "full",
        252,
        1,
        "months",
        "ex_ante_mv",
        "fill_na",
        "off",
        {},
        [],
        {"Asset_A": 0.08, "Asset_B": 0.06},
        ex_cov,
        [],
        0.05,
        "maximize_sharpe",
        {},
        {},
        "ret_cov",
        [],
        True,
        None,
    )

    assert status["status"] == "complete"
    assert "frontier_cache" in results_out["MyPort"]
    assert "MV" in results_out["MyPort"]["frontier_cache"]["0"]
    assert new_raw is no_update
    assert pending is no_update


def test_po_save_series_aligns_month_end_and_updates_result(page_modules):
    _, portopt = page_modules

    raw_idx = pd.to_datetime(["1976-06-30", "1976-07-30", "1976-08-30", "1976-09-30"])
    raw_df = pd.DataFrame(
        {
            "Asset_A": [0.01, 0.02, 0.03, 0.04],
            "Asset_B": [0.02, 0.01, -0.01, 0.00],
        },
        index=raw_idx,
    )
    raw_df.index.name = "Date"
    raw_json = df_to_json(raw_df)
    results = {
        "MyPort": {
            "returns_json": pd.Series(
                [0.005, 0.006, 0.007, 0.008],
                index=pd.to_datetime(["1976-06-30", "1976-07-31", "1976-08-31", "1976-09-30"]),
            ).to_json(date_format="iso"),
            "config": {"periodicity": "monthly"},
            "saved_series_name": None,
        }
    }

    results_out, new_raw, saved_store, status = portopt.po_save_series_to_shared_data(
        1,
        "MyPort",
        results,
        raw_json,
        "monthly",
        {},
    )

    df_after = pd.read_json(StringIO(_raw_json_value(new_raw)), orient="split")
    df_after.index = pd.to_datetime(df_after.index)
    assert pd.Timestamp("1976-07-30") not in df_after.index
    assert pd.Timestamp("1976-07-31") in df_after.index
    assert pd.Timestamp("1976-08-31") in df_after.index
    assert df_after.index.is_month_end.all()
    assert df_after.loc[pd.Timestamp("1976-07-31"), "MyPort"] == pytest.approx(0.006)
    assert df_after.loc[pd.Timestamp("1976-08-31"), "MyPort"] == pytest.approx(0.007)
    assert results_out["MyPort"]["saved_series_name"] == "MyPort"
    assert saved_store["MyPort"]["origin_page"] == "portopt"
    assert status == "Saved as MyPort."


def test_po_run_optimization_persists_cov_shrinkage_in_config(monkeypatch, page_modules, raw_json):
    _, portopt = page_modules
    df = pd.read_json(StringIO(raw_json), orient="split")[["Asset_A", "Asset_B"]]
    df.index = pd.to_datetime(df.index)
    captured = {}

    monkeypatch.setattr(portopt, "_po_get_working_returns", lambda *_args, **_kwargs: df.copy())

    def _fake_run_portfolio_optimization(_returns_df, config):
        captured["config"] = config
        return (
            [
                SimpleNamespace(
                    apply_start=df.index[0],
                    apply_end=df.index[-1],
                    est_start=df.index[0],
                    est_end=df.index[-1],
                    weights={"Asset_A": 0.5, "Asset_B": 0.5},
                )
            ],
            pd.Series(0.001, index=df.index),
        )

    monkeypatch.setattr(portopt, "run_portfolio_optimization", _fake_run_portfolio_optimization)

    results_out, new_raw, status, pending = portopt.po_run_optimization(
        1,
        raw_json,
        "daily",
        "daily",
        ["Asset_A", "Asset_B"],
        {},
        {},
        {},
        None,
        0,
        {},
        {},
        {},
        {},
        False,
        63,
        "oas",
        "scaled_identity",
        "MyPort",
        "full",
        252,
        1,
        "months",
        "risk_parity",
        "fill_na",
        "off",
        {},
        [],
        {},
        {},
        [],
        0.05,
        "maximize_sharpe",
        {},
        {},
        "ret_cov",
        [],
        True,
        None,
    )

    assert status["status"] == "complete"
    assert captured["config"]["cov_shrinkage"] == "oas"
    assert captured["config"]["cov_shrinkage_target"] == "scaled_identity"
    assert results_out["MyPort"]["config"]["cov_shrinkage"] == "oas"
    assert results_out["MyPort"]["config"]["cov_shrinkage_target"] == "scaled_identity"
    assert new_raw is no_update
    assert pending is no_update


def test_compute_window_risk_contributions_uses_custom_cov_for_shrinkage(monkeypatch, page_modules, raw_json):
    _, portopt = page_modules
    working_df = pd.read_json(StringIO(raw_json), orient="split")[["Asset_A", "Asset_B"]]
    working_df.index = pd.to_datetime(working_df.index)
    captured = {}

    def _fake_estimate_covariance_matrix(df, **kwargs):
        captured["cov_kwargs"] = kwargs
        cols = list(df.columns)
        return pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)

    def _fake_compute_risk_contributions(weights, returns_df, custom_cov=None):
        captured["custom_cov"] = custom_cov
        return {name: float(val) for name, val in weights.items()}

    monkeypatch.setattr(portopt, "estimate_covariance_matrix", _fake_estimate_covariance_matrix)
    monkeypatch.setattr(portopt, "compute_risk_contributions", _fake_compute_risk_contributions)

    rows = portopt._compute_window_risk_contributions(
        working_df,
        ["Asset_A", "Asset_B"],
        _sample_window_weights(),
        {
            "exp_wt_cov": False,
            "halflife": 63,
            "cov_shrinkage": "ledoit_wolf",
            "cov_shrinkage_target": "constant_correlation",
        },
    )

    assert len(rows) == 2
    assert captured["custom_cov"] is not None
    assert captured["cov_kwargs"]["shrinkage"] == "ledoit_wolf"
    assert captured["cov_kwargs"]["shrinkage_target"] == "constant_correlation"


def test_po_get_window_risk_contributions_uses_cached_working_return_path(monkeypatch, page_modules, raw_json):
    _, portopt = page_modules
    working_df = pd.read_json(StringIO(raw_json), orient="split")[["Asset_A", "Asset_B"]]
    working_df.index = pd.to_datetime(working_df.index)

    bundle = portopt._build_po_working_bundle(raw_json, "daily", {}, {}, None, 0, {})
    monkeypatch.setattr(portopt, "get_working_returns_by_key", lambda *_args, **_kwargs: working_df)
    monkeypatch.setattr(
        portopt,
        "_compute_window_risk_contributions",
        lambda *_args, **_kwargs: [
            {
                "apply_start": pd.Timestamp("2024-01-01"),
                "apply_end": pd.Timestamp("2024-01-31"),
                "risk_contributions": {"Asset_A": 0.6, "Asset_B": 0.4},
            }
        ],
    )

    rows = portopt._po_get_window_risk_contributions(
        bundle,
        ["Asset_A", "Asset_B"],
        _sample_window_weights(),
        {"cov_shrinkage": "none"},
    )

    assert rows[0]["apply_start"] == pd.Timestamp("2024-01-01")
    assert rows[0]["risk_contributions"]["Asset_A"] == 0.6


def test_po_download_excel_respects_tab_order_and_frontier_weights(monkeypatch, page_modules):
    _, portopt = page_modules

    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    ret_series = pd.Series([0.01, -0.005, 0.002, 0.003, -0.001, 0.004], index=idx)
    ret_series.index = pd.to_datetime(ret_series.index)
    ret_df = pd.DataFrame({"Asset_A": 0.01, "Asset_B": 0.02}, index=idx)
    ret_df.index.name = "Date"

    results = {
        "P1": {
            "returns_json": ret_series.to_json(date_format="iso"),
            "window_weights": _sample_window_weights(),
            "config": {"selected_series": ["Asset_A", "Asset_B"], "model": "risk_parity", "use_risk_free": False},
            "risk_free_meta": {"enabled": False},
        }
    }

    frontier_calls = {}

    monkeypatch.setattr(
        portopt,
        "calculate_statistics_cached",
        lambda *_args, **_kwargs: [{"Series": "P1", "Cumulative Return": 0.1}],
    )
    monkeypatch.setattr(portopt, "_po_get_working_returns", lambda *_args, **_kwargs: ret_df.copy())
    monkeypatch.setattr(
        portopt,
        "_compute_monthly_attribution",
        lambda *_args, **_kwargs: pd.DataFrame({"Asset_A": [0.01], "Asset_B": [0.02]}, index=[pd.Timestamp("2024-01-31")]),
    )
    monkeypatch.setattr(
        portopt,
        "_compute_window_risk_contributions",
        lambda *_args, **_kwargs: [
            {
                "apply_start": pd.Timestamp("2024-01-01"),
                "apply_end": pd.Timestamp("2024-01-31"),
                "risk_contributions": {"Asset_A": 0.6, "Asset_B": 0.4},
            }
        ],
    )
    monkeypatch.setattr(
        portopt,
        "_po_resolve_frontier_snapshot",
        lambda **kwargs: frontier_calls.setdefault("use_risk_free", kwargs["use_risk_free"]) or {
            "window_index": 0,
            "risk_measure": "MV",
            "asset_order": ["Asset_A", "Asset_B"],
            "portfolio": {"name": "P1", "return": 0.09, "risk": 0.16, "weights": {"Asset_A": 0.55, "Asset_B": 0.45}},
            "assets": [{"name": "Asset_A", "return": 0.1, "risk": 0.2}],
            "frontier_points": [{"return": 0.08, "risk": 0.15}],
            "frontier_portfolios": [{"point_index": 0, "return": 0.08, "risk": 0.15, "weights": {"Asset_A": 0.5, "Asset_B": 0.5}}],
            "window_est_start": "2024-01-01",
            "window_est_end": "2024-01-31",
        },
    )
    monkeypatch.setattr(
        portopt,
        "calculate_rolling_returns",
        lambda *_args, **_kwargs: pd.DataFrame(
            {"P1": [0.12]},
            index=[pd.Timestamp("2024-01-31")],
        ),
    )
    monkeypatch.setattr(
        portopt,
        "calculate_calendar_year_returns",
        lambda *_args, **_kwargs: pd.DataFrame(
            {"P1": [0.10]},
            index=[2024],
        ),
    )
    monkeypatch.setattr(
        portopt,
        "calculate_drawdown",
        lambda *_args, **_kwargs: pd.DataFrame(
            {"P1": [0.0, -0.02]},
            index=[pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-31")],
        ),
    )
    monkeypatch.setattr(portopt.dcc, "send_bytes", lambda b, filename: {"content": b, "filename": filename})

    payload = portopt.po_download_excel(
        1,
        results,
        df_to_json(ret_df),
        "daily",
        {},
        {},
        {},
        None,
        0,
        {},
        None,
    )

    workbook = BytesIO(payload["content"])
    xl = pd.ExcelFile(workbook)
    assert xl.sheet_names == [
        "Settings",
        "Weights",
        "Attribution",
        "Risk",
        "Turnover",
        "Frontier",
        "Statistics",
        "Returns",
        "Rolling",
        "Calendar Year",
        "Growth of $1",
        "Drawdown",
    ]

    settings_df = pd.read_excel(BytesIO(payload["content"]), sheet_name="Settings")
    settings_map = dict(zip(settings_df["Parameter"], settings_df["Value"]))
    assert settings_map["Result Name"] == "P1"
    assert settings_map["Use BCTBill13 for Sharpe/Sortino"] == False
    assert settings_map["Decay Input"] == pytest.approx(63.0)
    assert settings_map["Decay Mode"] == "halflife_periods"
    assert settings_map["Selected Series"] == "Asset_A, Asset_B"
    assert settings_map["Benchmark Assignments"] == "{}"
    assert frontier_calls["use_risk_free"] is False

    weights_df = pd.read_excel(BytesIO(payload["content"]), sheet_name="Weights")
    assert list(weights_df.columns) == ["Apply Start", "Apply End", "Asset_A", "Asset_B"]
    assert weights_df.loc[0, "Apply Start"] == "2024-01-01"
    assert weights_df.loc[0, "Apply End"] == "2024-01-31"
    assert weights_df.loc[0, "Asset_A"] == pytest.approx(0.6)
    assert weights_df.loc[0, "Asset_B"] == pytest.approx(0.4)
    assert "Portfolio" not in set(weights_df.columns)
    assert "Wt_Asset_A" not in set(weights_df.columns)

    frontier_df = pd.read_excel(BytesIO(payload["content"]), sheet_name="Frontier")
    assert "Wt_Asset_A" in frontier_df.columns
    assert "Sharpe Ratio" in frontier_df.columns
    assert "Frontier Point" in set(frontier_df["Type"])


def test_po_help_modal_has_three_guide_sections(page_modules):
    _, portopt = page_modules
    help_control = _find_component_by_id(portopt.layout, "po-menu-help-guide")
    assert help_control is not None

    text_blob = Path("docs/help/portopt.md").read_text(encoding="utf-8").lower()
    assert "portfolio optimization" in text_blob
    assert "typical workflow" in text_blob
    assert "model guide" in text_blob


def test_po_help_modal_model_deep_dive_covers_all_models(page_modules):
    _, portopt = page_modules
    help_control = _find_component_by_id(portopt.layout, "po-menu-help-guide")
    assert help_control is not None

    text_blob = Path("docs/help/portopt.md").read_text(encoding="utf-8").lower()
    required_models = [
        "risk parity",
        "factor risk parity",
        "hierarchical risk parity",
        "maximize sharpe ratio",
        "minimize variance",
        "minimize cvar",
        "equal weight",
        "ex ante mean-variance",
        "black-litterman",
    ]
    for model in required_models:
        assert model in text_blob


def test_po_ui_blocker_release_uses_db_error_alert():
    text_blob = Path("pages/portopt.py").read_text(encoding="utf-8")
    assert 'Input("po-db-add-error-alert", "hide")' in text_blob


def test_portopt_file_menu_includes_account_list_actions():
    page_text = Path("pages/portopt.py").read_text(encoding="utf-8")
    assert 'id="po-menu-load-account-list"' in page_text
    assert 'id="po-menu-save-account-list"' in page_text
    assert "welcome_switch_buttons=()," in page_text
    assert 'id="po-menu-save-session"' in page_text
    assert 'disabled=True' in page_text
    assert page_text.index('id="po-menu-save-session"') < page_text.index('id="po-menu-load-account-list"')


def test_ui_blocker_release_only_clears_on_series_selection_modal_close():
    text_blob = Path("assets/dashmat_callbacks.js").read_text(encoding="utf-8")
    assert 'if (trigger.indexOf("series-selection-modal") !== -1) {' in text_blob
    assert "return seriesSelectionOpened === false ? false : noUpdate();" in text_blob
    assert "function uiBlockerRelease(dbErrorHidden, rawErrorHidden, portfolioErrorHidden, underlyingErrorHidden, seriesSelectionOpened)" in text_blob


def test_po_blocker_wiring_covers_add_modal_entry_and_series_render():
    page_text = Path("pages/portopt.py").read_text(encoding="utf-8")
    js_text = Path("assets/dashmat_callbacks.js").read_text(encoding="utf-8")

    assert 'Input("po-menu-add-from-db", "n_clicks")' in page_text
    assert 'Input("po-open-modal-button", "n_clicks")' in page_text
    assert 'Output("po-ui-blocker-store", "data", allow_duplicate=True)' in page_text
    assert 'Output("po-series-selection-grid", "rowData")' in page_text
    assert 'Output("po-series-selection-grid", "columnDefs")' in page_text
    assert 'ClientsideFunction(namespace="dashmat_callbacks", function_name="syncPortoptSeriesModalGrid")' in page_text
    assert 'ClientsideFunction(namespace="dashmat_callbacks", function_name="releaseBlockerOnSeriesGridReady")' in page_text
    assert 'Input("po-series-selection-grid", "virtualRowData", allow_optional=True)' in page_text
    assert 'ClientsideFunction(namespace="dashmat_callbacks", function_name="portoptInitialSeriesBlocker")' in page_text
    assert 'Input("po-url-location", "pathname")' in page_text
    assert 'Input("po-series-selection-modal", "opened")' in page_text
    assert 'Input("po-series-selection-grid", "virtualRowData", allow_optional=True)' in page_text
    assert 'Input("po-page-load-trigger", "n_intervals")' in page_text
    assert 'State("po-page-visited-store", "data")' in page_text
    assert 'State("po-series-order-store", "data")' in page_text
    assert 'State("dashmat-pending-new-series-store", "data")' in page_text
    assert "function portoptInitialSeriesBlocker(pathname, rawMeta, currentSelect, pageLoadReady, modalOpened, virtualRows, pageVisited, currentOrder, poOriginSeries)" in js_text
    assert "function portoptInitialSeriesModalPending(rawMeta, currentSelect, currentOrder, poOriginSeries, pageVisited)" in js_text
    assert "function syncPortoptSeriesModalGrid(" in js_text


def test_po_series_selection_grid_is_fully_clientside():
    page_text = Path("pages/portopt.py").read_text(encoding="utf-8")
    js_text = Path("assets/dashmat_callbacks.js").read_text(encoding="utf-8")

    assert 'ClientsideFunction(namespace="dashmat_callbacks", function_name="syncPortoptSeriesModalGrid")' in page_text
    assert 'Input("dashmat-raw-data-meta-store", "data")' in page_text
    assert 'Input("po-cmabench-defaults-store", "data")' in page_text
    assert 'Input("po-cmabench-option-values-store", "data")' in page_text
    assert 'State("po-series-selection-grid", "columnDefs")' in page_text
    assert 'State("po-series-selection-modal", "opened")' in page_text
    assert "def po_update_series_selectors" not in page_text
    assert "function syncPortoptSeriesModalGrid(" in js_text
    assert 'const explicitCmabench = explicitCmabenchAssignments[series] || "";' in js_text
    assert 'const defaultCmabench = importedCmabenchDefaults[series] || "";' in js_text
    assert 'CMABench: explicitCmabench || defaultCmabench || ""' in js_text
    assert "portoptSeriesSelectionColumnDefs" in js_text


def test_po_effective_cmabench_assignments_use_imported_defaults_without_autoresolve(page_modules):
    _, portopt = page_modules

    effective = portopt._effective_cmabench_assignments(
        ["Asset_A", "Asset_B", "Asset_C"],
        {"Asset_A": "Explicit_Bench", "Asset_B": ""},
        {"Asset_B": "Imported_Bench"},
    )

    assert effective == {
        "Asset_A": "Explicit_Bench",
        "Asset_B": "Imported_Bench",
    }
    assert portopt._missing_cmabench_assignments(["Asset_A", "Asset_B", "Asset_C"], effective) == ["Asset_C"]


def test_po_update_cmabench_defaults_store_handles_rename_and_delete(page_modules):
    _, portopt = page_modules

    updated = portopt._po_update_cmabench_defaults_store(
        {"Asset_A": "Bench_A", "Asset_B": "Bench_B", "Asset_C": ""},
        {"Asset_A": "Asset_A_Renamed"},
        ["Asset_B"],
        ["Asset_A_Renamed"],
    )

    assert updated == {"Asset_A_Renamed": "Bench_A"}


def test_po_add_series_from_database_persists_imported_cmabench_defaults(monkeypatch, page_modules):
    _, portopt = page_modules
    imported_df = pd.DataFrame({"Asset_A": [0.01, 0.02], "Asset_B": [0.03, 0.01]}, index=pd.date_range("2024-01-01", periods=2, freq="B"))
    imported_df.index.name = "Date"

    monkeypatch.setattr(
        portopt,
        "load_cma_returns_for_benches_with_meta",
        lambda *_args, **_kwargs: (imported_df, {"Asset_A": {}, "Asset_B": {}}),
    )
    monkeypatch.setattr(portopt, "_po_cached_cmabench_defaults", lambda key: {"Asset_A": "Bench_A", "Asset_B": "Bench_B"})
    monkeypatch.setattr(portopt, "add_db_import_provenance_entry", lambda current, **_kwargs: {"updated": True})

    result = portopt.po_add_series_from_database(
        1,
        ["Asset_A", "Asset_B"],
        None,
        None,
        [],
        {},
        {},
        {},
        [],
        {},
        {},
        {},
        {},
        {},
        {},
    )

    assert result[-2] == {"updated": True}
    assert result[-1] == {"Asset_A": "Bench_A", "Asset_B": "Bench_B"}


def test_po_load_cmabench_option_values_is_lazy(monkeypatch, page_modules):
    _, portopt = page_modules
    calls = []
    monkeypatch.setattr(portopt, "get_unique_cmabench_values_cached", lambda *_args: calls.append(True) or ["Bench_1"])

    with pytest.raises(PreventUpdate):
        portopt.po_load_cmabench_option_values(True, ["Bench_1"])
    assert calls == []

    loaded = portopt.po_load_cmabench_option_values(True, None)
    assert loaded == ["Bench_1"]
    assert calls == [True]


def test_po_series_selection_grid_no_longer_fetches_cmabench_values_inline():
    page_text = Path("pages/portopt.py").read_text(encoding="utf-8")
    js_text = Path("assets/dashmat_callbacks.js").read_text(encoding="utf-8")
    assert 'Input("po-cmabench-option-values-store", "data")' in page_text
    assert 'Input("po-cmabench-defaults-store", "data")' in page_text
    assert "function syncPortoptSeriesModalGrid(" in js_text
    assert "get_unique_cmabench_values_cached(DB_ENGINE)" not in js_text
    assert 'def _po_series_selection_column_defs(' in page_text
    assert '"cellEditor": "agSelectCellEditor"' in page_text


def test_po_cma_modal_treats_blank_cmabench_as_missing(monkeypatch, page_modules):
    _, portopt = page_modules
    monkeypatch.setattr(portopt, "callback_context", type("Ctx", (), {"triggered_id": "po-load-db-returns-btn"})())
    monkeypatch.setattr(portopt, "get_cma_versions_cached", lambda *_args: [2025])
    monkeypatch.setattr(portopt, "_get_cma_stats_map", lambda *_args: {"Bench_A": {"Mean": 0.05}})
    monkeypatch.setattr(portopt, "_get_cma_corr_map", lambda *_args: {})

    result = portopt.po_open_cma_load_modal(1, None, ["Asset_A"], {}, {}, "ret_cov")

    assert result[0] is True
    assert "Select CMA Benchmarks for: Asset_A." in result[5]


def test_po_cma_load_modal_accepts_imported_defaults_without_explicit_assignments(monkeypatch, page_modules):
    _, portopt = page_modules
    monkeypatch.setattr(portopt, "_get_cma_stats_map", lambda *_args: {"Bench_A": {"Mean": 0.05, "SD": 0.1}})
    monkeypatch.setattr(portopt, "_get_cma_corr_map", lambda *_args: {"Bench_A": {"Bench_A": 1.0}})

    result = portopt.po_load_cma_from_db(
        1,
        "2025",
        "hmm",
        "returns",
        ["Asset_A"],
        {},
        {"Asset_A": "Bench_A"},
        "ret_cov",
    )

    assert result[0] == {"Asset_A": 0.05}
    assert result[2][0]["Asset"] == "Asset_A"


def test_po_modal_ok_keeps_blank_cmabench_without_backfill(page_modules, raw_json):
    _, portopt = page_modules
    raw_meta = {"columns": ["Asset_A"]}

    result = portopt.po_on_modal_ok(
        _series_snapshot(
            [
                {
                    "__row_key": "Asset_A",
                    "Selected": True,
                    "Series": "Asset_A",
                    "Benchmark": "None",
                    "CMABench": "",
                    "LongShort": False,
                    "ScaleVol": True,
                    "MinWt": 0,
                    "MaxWt": 100,
                    "ForceMax": False,
                    "Delete": False,
                }
            ]
        ),
        raw_json,
        raw_meta,
        {},
        [],
        {},
        {},
        {"Asset_A": "Imported_Bench"},
        {},
        ["Asset_A"],
        {},
        {"Asset_A": 0.0},
        {"Asset_A": 100.0},
        {"Asset_A": False},
        {},
    )

    assert result[2] == {"Asset_A": ""}
    assert result[14] is no_update


def test_po_modal_ok_returns_no_update_for_unchanged_common_path(page_modules, raw_json):
    _, portopt = page_modules
    raw_meta = {"columns": ["Asset_A"]}

    def _fail_raw_df(_raw_data):
        raise AssertionError("noop modal OK path should not parse raw data")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(portopt, "_raw_df", _fail_raw_df)

    try:
        result = portopt.po_on_modal_ok(
            _series_snapshot(
                [
                    {
                        "__row_key": "Asset_A",
                        "Selected": True,
                        "Series": "Asset_A",
                        "Benchmark": "None",
                        "CMABench": "Bench_A",
                        "LongShort": False,
                        "ScaleVol": True,
                        "MinWt": 0,
                        "MaxWt": 100,
                        "ForceMax": False,
                        "Delete": False,
                    }
                ]
            ),
            raw_json,
            raw_meta,
            {},
            ["Asset_A"],
            {"Asset_A": "None"},
            {"Asset_A": "Bench_A"},
            {},
            {"Asset_A": False},
            ["Asset_A"],
            {"Asset_A": True},
            {"Asset_A": 0.0},
            {"Asset_A": 100.0},
            {"Asset_A": False},
            {},
        )
    finally:
        monkeypatch.undo()

    assert result[0] is no_update
    assert result[1] is no_update
    assert result[2] is no_update
    assert result[3] is no_update
    assert result[4] is no_update
    assert result[5] is False
    assert result[6] is no_update
    assert result[7] is no_update
    assert result[8] is no_update
    assert result[9] is no_update
    assert result[10] is no_update
    assert result[11] is no_update
    assert result[12] is no_update
    assert result[13] is no_update
    assert result[14] is no_update


def test_po_modal_ok_requires_a_real_grid_snapshot(page_modules, raw_json):
    _, portopt = page_modules

    with pytest.raises(PreventUpdate):
        portopt.po_on_modal_ok(
            None,
            raw_json,
            {"columns": ["Asset_A", "Asset_B"]},
            {},
            [],
            {},
            {},
            {},
            {},
            ["Asset_A", "Asset_B"],
            {},
            {"Asset_A": 0.0, "Asset_B": 0.0},
            {"Asset_A": 100.0, "Asset_B": 100.0},
            {"Asset_A": False, "Asset_B": False},
            {},
            ["Asset_A"],
            ["Asset_A", "Asset_B"],
            [],
            {"Asset_A": "None", "Asset_B": "None"},
            {"Asset_A": "Bench_A"},
            {"Asset_A": False, "Asset_B": False},
            {"Asset_A": True, "Asset_B": True},
            {"Asset_A": 0.0, "Asset_B": 0.0},
            {"Asset_A": 100.0, "Asset_B": 100.0},
            {"Asset_A": False, "Asset_B": False},
        )


def test_po_modal_ok_delete_path_updates_only_raw_and_results(page_modules):
    _, portopt = page_modules
    raw_df = pd.DataFrame({"Asset_A": [0.01, 0.02], "Port_1": [0.03, 0.04]}, index=pd.date_range("2024-01-01", periods=2, freq="B"))
    raw_df.index.name = "Date"

    result = portopt.po_on_modal_ok(
        _series_snapshot(
            [
                {
                    "__row_key": "Asset_A",
                    "Selected": True,
                    "Series": "Asset_A",
                    "Benchmark": "None",
                    "CMABench": "",
                    "LongShort": False,
                    "ScaleVol": True,
                    "MinWt": 0,
                    "MaxWt": 100,
                    "ForceMax": False,
                    "Delete": False,
                },
                {
                    "__row_key": "Port_1",
                    "Selected": False,
                    "Series": "Port_1",
                    "Benchmark": "None",
                    "CMABench": "",
                    "LongShort": False,
                    "ScaleVol": True,
                    "MinWt": 0,
                    "MaxWt": 100,
                    "ForceMax": False,
                    "Delete": True,
                },
            ]
        ),
        df_to_json(raw_df),
        {"columns": ["Asset_A", "Port_1"]},
        {"Port_1": {"weights": []}},
        ["Asset_A", "Port_1"],
        {},
        {},
        {},
        {},
        ["Asset_A", "Port_1"],
        {},
        {},
        {},
        {},
        {},
    )

    updated_df = pd.read_json(StringIO(_raw_json_value(result[7])), orient="split")
    assert list(updated_df.columns) == ["Asset_A"]
    assert result[12] == {}
    assert result[14] is no_update


def test_po_modal_ok_renames_and_prunes_cmabench_defaults(page_modules, raw_json):
    _, portopt = page_modules

    result = portopt.po_on_modal_ok(
        _series_snapshot(
            [
                {
                    "__row_key": "Asset_A",
                    "Selected": True,
                    "Series": "Asset_Renamed",
                    "Benchmark": "None",
                    "CMABench": "",
                    "LongShort": False,
                    "ScaleVol": True,
                    "MinWt": 0,
                    "MaxWt": 100,
                    "ForceMax": False,
                    "Delete": False,
                },
                {
                    "__row_key": "Asset_B",
                    "Selected": False,
                    "Series": "Asset_B",
                    "Benchmark": "None",
                    "CMABench": "",
                    "LongShort": False,
                    "ScaleVol": True,
                    "MinWt": 0,
                    "MaxWt": 100,
                    "ForceMax": False,
                    "Delete": True,
                },
            ]
        ),
        raw_json,
        {"columns": ["Asset_A", "Asset_B"]},
        {},
        ["Asset_A", "Asset_B"],
        {},
        {},
        {"Asset_A": "Bench_A", "Asset_B": "Bench_B"},
        {},
        ["Asset_A", "Asset_B"],
        {},
        {"Asset_A": 0.0, "Asset_B": 0.0},
        {"Asset_A": 100.0, "Asset_B": 100.0},
        {"Asset_A": False, "Asset_B": False},
        {},
    )

    assert result[14] == {"Asset_Renamed": "Bench_A"}


def test_po_modal_ok_selection_only_updates_only_selection_outputs(page_modules, raw_json, monkeypatch):
    _, portopt = page_modules
    monkeypatch.setattr(portopt, "_raw_df", lambda *_args, **_kwargs: pytest.fail("selection-only path should not parse raw data"))

    result = portopt.po_on_modal_ok(
        _series_snapshot(
            [
                {
                    "__row_key": "Asset_A",
                    "Selected": True,
                    "Series": "Asset_A",
                    "Benchmark": "None",
                    "CMABench": "",
                    "LongShort": False,
                    "ScaleVol": True,
                    "MinWt": 0,
                    "MaxWt": 100,
                    "ForceMax": False,
                    "Delete": False,
                },
                {
                    "__row_key": "Asset_B",
                    "Selected": False,
                    "Series": "Asset_B",
                    "Benchmark": "None",
                    "CMABench": "",
                    "LongShort": False,
                    "ScaleVol": True,
                    "MinWt": 0,
                    "MaxWt": 100,
                    "ForceMax": False,
                    "Delete": False,
                },
            ]
        ),
        raw_json,
        {"columns": ["Asset_A", "Asset_B"]},
        {},
        ["Asset_A", "Asset_B"],
        {"Asset_A": "None", "Asset_B": "None"},
        {"Asset_A": "", "Asset_B": ""},
        {},
        {"Asset_A": False, "Asset_B": False},
        ["Asset_A", "Asset_B"],
        {"Asset_A": True, "Asset_B": True},
        {"Asset_A": 0.0, "Asset_B": 0.0},
        {"Asset_A": 100.0, "Asset_B": 100.0},
        {"Asset_A": False, "Asset_B": False},
        {},
    )

    assert result[0] == ["Asset_A"]
    assert result[6] == ["Asset_A"]
    assert result[1] is no_update
    assert result[4] is no_update
    assert result[7] is no_update
    assert result[12] is no_update
    assert result[13] is no_update
    assert result[14] is no_update


def test_po_modal_ok_order_only_updates_only_order_outputs(page_modules, raw_json, monkeypatch):
    _, portopt = page_modules
    monkeypatch.setattr(portopt, "_raw_df", lambda *_args, **_kwargs: pytest.fail("order-only path should not parse raw data"))

    result = portopt.po_on_modal_ok(
        _series_snapshot(
            [
                {
                    "__row_key": "Asset_B",
                    "Selected": True,
                    "Series": "Asset_B",
                    "Benchmark": "None",
                    "CMABench": "",
                    "LongShort": False,
                    "ScaleVol": True,
                    "MinWt": 0,
                    "MaxWt": 100,
                    "ForceMax": False,
                    "Delete": False,
                },
                {
                    "__row_key": "Asset_A",
                    "Selected": True,
                    "Series": "Asset_A",
                    "Benchmark": "None",
                    "CMABench": "",
                    "LongShort": False,
                    "ScaleVol": True,
                    "MinWt": 0,
                    "MaxWt": 100,
                    "ForceMax": False,
                    "Delete": False,
                },
            ]
        ),
        raw_json,
        {"columns": ["Asset_A", "Asset_B"]},
        {},
        ["Asset_A", "Asset_B"],
        {"Asset_A": "None", "Asset_B": "None"},
        {"Asset_A": "", "Asset_B": ""},
        {},
        {"Asset_A": False, "Asset_B": False},
        ["Asset_A", "Asset_B"],
        {"Asset_A": True, "Asset_B": True},
        {"Asset_A": 0.0, "Asset_B": 0.0},
        {"Asset_A": 100.0, "Asset_B": 100.0},
        {"Asset_A": False, "Asset_B": False},
        {},
    )

    assert result[4] == ["Asset_B", "Asset_A"]
    assert result[0] is no_update
    assert result[6] is no_update
    assert result[7] is no_update
    assert result[12] is no_update
    assert result[13] is no_update
    assert result[14] is no_update


def test_po_modal_ok_metadata_only_updates_only_metadata_outputs(page_modules, raw_json, monkeypatch):
    _, portopt = page_modules
    monkeypatch.setattr(portopt, "_raw_df", lambda *_args, **_kwargs: pytest.fail("metadata-only path should not parse raw data"))

    result = portopt.po_on_modal_ok(
        _series_snapshot(
            [
                {
                    "__row_key": "Asset_A",
                    "Selected": True,
                    "Series": "Asset_A",
                    "Benchmark": "None",
                    "CMABench": "",
                    "LongShort": False,
                    "ScaleVol": True,
                    "MinWt": 0,
                    "MaxWt": 55,
                    "ForceMax": False,
                    "Delete": False,
                },
                {
                    "__row_key": "Asset_B",
                    "Selected": True,
                    "Series": "Asset_B",
                    "Benchmark": "None",
                    "CMABench": "",
                    "LongShort": False,
                    "ScaleVol": True,
                    "MinWt": 0,
                    "MaxWt": 100,
                    "ForceMax": False,
                    "Delete": False,
                },
            ]
        ),
        raw_json,
        {"columns": ["Asset_A", "Asset_B"]},
        {},
        ["Asset_A", "Asset_B"],
        {"Asset_A": "None", "Asset_B": "None"},
        {"Asset_A": "", "Asset_B": ""},
        {},
        {"Asset_A": False, "Asset_B": False},
        ["Asset_A", "Asset_B"],
        {"Asset_A": True, "Asset_B": True},
        {"Asset_A": 0.0, "Asset_B": 0.0},
        {"Asset_A": 100.0, "Asset_B": 100.0},
        {"Asset_A": False, "Asset_B": False},
        {},
    )

    assert result[10] == {"Asset_A": 55.0, "Asset_B": 100.0}
    assert result[0] is no_update
    assert result[4] is no_update
    assert result[7] is no_update
    assert result[12] is no_update
    assert result[13] is no_update
    assert result[14] is no_update


def test_po_session_actions_use_shared_workspace_helpers():
    text_blob = Path("pages/portopt.py").read_text(encoding="utf-8")
    assert 'ClientsideFunction(namespace="dashmat_callbacks", function_name="saveWorkspaceSession")' in text_blob
    assert 'ClientsideFunction(namespace="dashmat_callbacks", function_name="loadWorkspaceSessionDialog")' in text_blob
    assert 'ClientsideFunction(namespace="dashmat_callbacks", function_name="loadWorkspaceSession")' in text_blob
    assert 'ClientsideFunction(namespace="dashmat_callbacks", function_name="clearWorkspaceSession")' in text_blob
    assert "sessionStorage.clear()" not in text_blob
    assert "sessionStorage.length" not in text_blob


def test_workspace_session_helper_scopes_keys_consistently():
    text_blob = Path("assets/dashmat_callbacks.js").read_text(encoding="utf-8")
    assert 'const workspacePrefixes = ["dashmat-", "at-", "po-", "reg-"];' in text_blob
    assert 'const workspaceExtraKeys = ["dashmat-bctbill13-cache-store"];' in text_blob
    assert 'const workspaceExcludedKeys = ["userinfo"];' in text_blob
    assert "function collectWorkspaceSessionData()" in text_blob
    assert "function clearWorkspaceSessionKeys()" in text_blob
