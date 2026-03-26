from __future__ import annotations

import inspect
import json
from io import BytesIO
from io import StringIO
from pathlib import Path
import subprocess

import pandas as pd
import pytest
from dash import no_update
from dash.exceptions import PreventUpdate
from utils.returns import build_raw_data_metadata


def _main_analyticstool_page_text() -> str:
    return Path("pages/analyticstool.py").read_text(encoding="utf-8")


def _advanced_analyticstool_page_text() -> str:
    return Path("utils/analyticstool_advanced_source.py.txt").read_text(encoding="utf-8")


def _combined_analyticstool_page_text() -> str:
    return _main_analyticstool_page_text() + "\n" + _advanced_analyticstool_page_text()


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


def _find_component_by_id(node, target_id):
    if node is None:
        return None
    if getattr(node, "id", None) == target_id:
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


def _raw_meta(raw_json: str, original_periodicity: str = "daily") -> dict:
    return build_raw_data_metadata(raw_json, original_periodicity)


def _run_dashmat_callbacks_js(expression: str):
    repo_root = Path(__file__).resolve().parents[2]
    script = f"""
const path = require("path");
global.window = {{ dash_clientside: {{ no_update: {{ __dash_no_update__: true }} }} }};
require(path.resolve("assets/dashmat_callbacks.js"));
const ns = window.dash_clientside.dashmat_callbacks;
function normalize(value) {{
  if (value && value.__dash_no_update__) {{
    return "__NO_UPDATE__";
  }}
  if (Array.isArray(value)) {{
    return value.map(normalize);
  }}
  if (value && typeof value === "object") {{
    const out = {{}};
    for (const [key, nextValue] of Object.entries(value)) {{
      out[key] = normalize(nextValue);
    }}
    return out;
  }}
  return value;
}}
const result = {expression};
process.stdout.write(JSON.stringify(normalize(result)));
"""
    completed = subprocess.run(
        ["node", "-e", script],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _callback_block(page_text: str, function_name: str) -> str:
    marker = f"def {function_name}("
    idx = page_text.index(marker)
    start = page_text.rfind("@callback(", 0, idx)
    end = page_text.find("\n\n", idx)
    return page_text[start:end]


def _raw_json_value(value):
    if isinstance(value, dict):
        return value.get("raw_data_json", "")
    return value


def _series_snapshot(rows: list[dict]) -> dict:
    return {"rows": rows, "capturedAt": 1}


def _stack_section_titles(stack_component):
    def _graph_title(node):
        fig = getattr(node, "figure", None)
        if fig is None:
            return None
        if isinstance(fig, dict):
            return (((fig.get("layout") or {}).get("title") or {}).get("text"))
        layout = getattr(fig, "layout", None)
        title = getattr(layout, "title", None) if layout is not None else None
        return getattr(title, "text", None) if title is not None else None

    titles = []
    children = getattr(stack_component, "children", None)
    if children is None:
        return titles
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        graph_title = _graph_title(child)
        if graph_title:
            titles.append(str(graph_title))
            continue
        child_children = getattr(child, "children", None)
        if isinstance(child_children, (list, tuple)) and child_children:
            graph_title = _graph_title(child_children[0])
            if graph_title:
                titles.append(str(graph_title))
                continue
            title_text = _collect_component_text(child_children[0])
            if title_text:
                titles.append(str(title_text[0]))
                continue
        text = _collect_component_text(child)
        if text:
            titles.append(str(text[0]))
    return titles


def _db_factor_definition(name="DBFactor", description=None):
    return {
        "FactorName": name,
        "LongComponentList": ["ACC1 TRIndex"],
        "ShortComponentList": [],
        "LongComponent": "ACC1 TRIndex",
        "ShortComponent": None,
        "Description": description,
        "LongAggType": 1,
        "ShortAggType": None,
        "LongLag": 0,
        "OutputTransform": 0,
        "source": "db",
        "UPDATE_DATE": "2026-02-26 00:00:00",
        "UPDATE_BY": "Admin:tester",
    }


def _db_regime_definition(name="DBRegime", description=None):
    return {
        "RegimeName": name,
        "Description": description,
        "MethodType": 3,
        "Config": {
            "schema_version": 1,
            "num_regimes": 3,
            "return_basis": "total",
            "benchmark_assignments": {},
            "long_short_assignments": {},
            "vol_scaling_assignments": {},
            "vol_scaler": 0.0,
            "min_observations": 40,
            "pca_standardize": True,
            "single_series": "Asset_A",
            "quantile_window": "in_sample_full_range",
        },
        "source": "db",
        "UPDATE_DATE": "2026-02-26 00:00:00",
        "UPDATE_BY": "Admin:tester",
    }


def test_build_analytics_compute_bundle_normalizes_inputs(page_modules, raw_json):
    analyticstool, _ = page_modules

    bundle = analyticstool._build_analytics_compute_bundle(
        raw_json,
        None,
        ["Asset_A", "Asset_B"],
        {"Asset_A": "Asset_B"},
        {"Asset_A": True},
        {"start": "2024-01-01", "end": "2024-12-31"},
        None,
        {"Asset_A": False},
    )

    assert bundle.periodicity == "daily"
    assert bundle.selected_series == ("Asset_A", "Asset_B")
    assert bundle.vol_scaler == 0
    assert bundle.benchmark_payload == '{"Asset_A":"Asset_B"}'


def test_update_date_range_store_returns_payload_or_no_update(page_modules):
    analyticstool, _ = page_modules

    assert analyticstool.update_date_range_store("2024-01-01", "2024-12-31", None) == {
        "start": "2024-01-01",
        "end": "2024-12-31",
    }
    assert analyticstool.update_date_range_store("2024-01-01", None, None) is no_update
    assert (
        analyticstool.update_date_range_store(
            "2024-01-01",
            "2024-12-31",
            {"start": "2024-01-01", "end": "2024-12-31"},
        )
        is no_update
    )


def test_initialize_date_range_skips_store_write_when_range_unchanged(monkeypatch, page_modules):
    analyticstool, _ = page_modules

    monkeypatch.setattr(
        analyticstool,
        "resolve_initial_range",
        lambda *_args, **_kwargs: ("2024-01-01", "2024-12-31"),
    )

    start, end, _style, _common_disabled, _max_disabled, range_store, ready = (
        analyticstool.initialize_date_range(
            {
                "available_series": ["Asset_A"],
            },
            {"start": "2024-01-01", "end": "2024-12-31"},
            None,
            None,
            False,
        )
    )

    assert start == "2024-01-01"
    assert end == "2024-12-31"
    assert range_store is no_update
    assert ready is True


def test_initialize_date_range_skips_ready_write_when_already_ready_and_range_unchanged(monkeypatch, page_modules):
    analyticstool, _ = page_modules

    monkeypatch.setattr(
        analyticstool,
        "resolve_initial_range",
        lambda *_args, **_kwargs: ("2024-01-01", "2024-12-31"),
    )

    start, end, _style, _common_disabled, _max_disabled, range_store, ready = (
        analyticstool.initialize_date_range(
            {"available_series": ["Asset_A"]},
            {"start": "2024-01-01", "end": "2024-12-31"},
            "2024-01-01",
            "2024-12-31",
            True,
        )
    )

    assert start is no_update
    assert end is no_update
    assert range_store is no_update
    assert ready is no_update


def test_at_range_candidates_use_raw_data_meta_dataset_key(monkeypatch, page_modules):
    analyticstool, _ = page_modules
    captured = {}

    def _fake_compute(dataset_key, periodicity, selected_series):
        captured["dataset_key"] = dataset_key
        captured["periodicity"] = periodicity
        captured["selected_series"] = selected_series
        return {"available_series": list(selected_series)}

    monkeypatch.setattr(analyticstool, "compute_date_range_candidates", _fake_compute)

    result = analyticstool.update_at_range_candidates(
        "ds-123",
        "monthly",
        ["Asset_A", "Asset_B"],
    )

    assert captured == {
        "dataset_key": "ds-123",
        "periodicity": "monthly",
        "selected_series": ("Asset_A", "Asset_B"),
    }
    assert result == {"available_series": ["Asset_A", "Asset_B"]}


def test_at_common_daily_candidates_use_raw_data_meta_dataset_key(monkeypatch, page_modules):
    analyticstool, _ = page_modules
    captured = {}

    def _fake_compute(dataset_key, selected_series):
        captured["dataset_key"] = dataset_key
        captured["selected_series"] = selected_series
        return {"common_daily_start": "2024-01-01", "common_daily_end": "2024-12-31"}

    monkeypatch.setattr(analyticstool, "compute_common_daily_candidates", _fake_compute)

    result = analyticstool.update_at_common_daily_candidates(
        "ds-123",
        ["Asset_A", "Asset_B"],
    )

    assert captured == {
        "dataset_key": "ds-123",
        "selected_series": ("Asset_A", "Asset_B"),
    }
    assert result == {"common_daily_start": "2024-01-01", "common_daily_end": "2024-12-31"}


def test_at_date_candidate_stores_dedupe_unchanged_outputs(monkeypatch, page_modules):
    analyticstool, _ = page_modules
    candidates = {"available_series": ["Asset_A"], "max_start": "2020-01-31", "max_end": "2025-12-31"}
    common_daily = {"common_daily_start": "2020-01-31", "common_daily_end": "2025-12-31"}
    monkeypatch.setattr(analyticstool, "update_at_range_candidates", lambda *_args: candidates)
    monkeypatch.setattr(analyticstool, "update_at_common_daily_candidates", lambda *_args: common_daily)

    result = analyticstool.update_at_date_candidate_stores(
        {"phase": "bootstrap"},
        "ds-123",
        "monthly",
        ["Asset_A"],
        candidates,
        common_daily,
    )

    assert result == (no_update, no_update)


def test_at_initialize_date_range_no_longer_depends_on_common_daily_store():
    page_text = Path("pages/analyticstool.py").read_text(encoding="utf-8")
    init_callback = page_text.split('ClientsideFunction(namespace="dashmat_callbacks", function_name="analyticsInitDateRange")', 1)[-1]
    init_callback = init_callback.split('ClientsideFunction(namespace="dashmat_callbacks", function_name="commonDailyButtonDisabled")', 1)[0]
    assert 'Input("at-range-candidates-store", "data")' in init_callback
    assert 'Input("at-common-daily-candidates-store", "data")' not in init_callback


def test_at_bootstrap_date_candidate_callback_uses_trigger_store():
    page_text = Path("pages/analyticstool.py").read_text(encoding="utf-8")
    callback_text = page_text.split('def update_at_date_candidate_stores(', 1)[0]
    callback_text = callback_text.rsplit('@callback(', 1)[-1]
    assert 'Input("at-bootstrap-candidate-trigger-store", "data")' in callback_text
    assert 'State("at-dataset-key-store", "data")' in callback_text
    assert 'State("at-periodicity-select", "value")' in callback_text
    assert 'State("at-series-select", "data")' in callback_text
    assert 'Output("at-range-candidates-store", "data", allow_duplicate=True)' in callback_text
    assert 'Output("at-common-daily-candidates-store", "data", allow_duplicate=True)' in callback_text


def test_at_common_daily_button_uses_shared_clientside_helper():
    page_text = Path("pages/analyticstool.py").read_text(encoding="utf-8")
    js_text = Path("assets/dashmat_callbacks.js").read_text(encoding="utf-8")
    assert 'ClientsideFunction(namespace="dashmat_callbacks", function_name="commonDailyButtonDisabled")' in page_text
    assert 'Output("at-common-daily-button", "disabled")' in page_text
    assert "function commonDailyButtonDisabled(candidates, commonDailyCandidates, periodicityOptions)" in js_text


def test_at_initialize_date_range_uses_clientside_helper():
    page_text = Path("pages/analyticstool.py").read_text(encoding="utf-8")
    js_text = Path("assets/dashmat_callbacks.js").read_text(encoding="utf-8")
    assert 'ClientsideFunction(namespace="dashmat_callbacks", function_name="analyticsInitDateRange")' in page_text
    assert "function analyticsInitDateRange(" in js_text


def test_at_date_range_buttons_and_store_use_clientside_helpers():
    page_text = Path("pages/analyticstool.py").read_text(encoding="utf-8")
    js_text = Path("assets/dashmat_callbacks.js").read_text(encoding="utf-8")
    assert 'ClientsideFunction(namespace="dashmat_callbacks", function_name="analyticsDateRangeButtons")' in page_text
    assert 'ClientsideFunction(namespace="dashmat_callbacks", function_name="analyticsDateRangeStoreUpdate")' in page_text
    assert "function analyticsDateRangeButtons(" in js_text
    assert "function analyticsDateRangeStoreUpdate(" in js_text


def test_analytics_resolve_initial_range_clientside_matches_python(page_modules):
    analyticstool, _ = page_modules
    candidates = {
        "available_series": ["Asset_A"],
        "max_start": "2024-01-05",
        "max_end": "2024-12-31",
    }

    assert _run_dashmat_callbacks_js(
        f"ns.analyticsResolveInitialRange({json.dumps(candidates)}, {json.dumps(None)})"
    ) == list(analyticstool.resolve_initial_range(candidates, None))

    stored_max_end = {"start": "2024-02-01", "end": "3999-12-31"}
    assert _run_dashmat_callbacks_js(
        f"ns.analyticsResolveInitialRange({json.dumps(candidates)}, {json.dumps(stored_max_end)})"
    ) == list(analyticstool.resolve_initial_range(candidates, stored_max_end))

    stored_out_of_range = {"start": "2023-01-01", "end": "2023-12-31"}
    assert _run_dashmat_callbacks_js(
        f"ns.analyticsResolveInitialRange({json.dumps(candidates)}, {json.dumps(stored_out_of_range)})"
    ) == list(analyticstool.resolve_initial_range(candidates, stored_out_of_range))


def test_analytics_resolve_button_range_clientside_matches_python(page_modules):
    analyticstool, _ = page_modules
    candidates = {
        "available_series": ["Asset_A"],
        "common_start": "2024-03-01",
        "common_end": "2024-11-30",
        "max_start": "2024-01-01",
        "max_end": "2024-12-31",
    }
    common_daily = {
        "common_daily_start": "2024-04-01",
        "common_daily_end": "2024-10-31",
    }

    for button_id in [
        "at-common-range-button",
        "at-common-daily-button",
        "at-maximum-range-button",
        "at-unknown-button",
    ]:
        assert _run_dashmat_callbacks_js(
            f"ns.analyticsResolveButtonRange({json.dumps(candidates)}, {json.dumps(button_id)}, {json.dumps(common_daily)})"
        ) == list(analyticstool.resolve_button_range(candidates, button_id, common_daily))


def test_analytics_date_range_buttons_clientside_outputs_expected_payloads():
    candidates = {
        "available_series": ["Asset_A"],
        "common_start": "2024-03-01",
        "common_end": "2024-11-30",
        "max_start": "2024-01-01",
        "max_end": "2024-12-31",
    }
    common_daily = {
        "common_daily_start": "2024-04-01",
        "common_daily_end": "2024-10-31",
    }

    common = _run_dashmat_callbacks_js(
        "(window.dash_clientside.callback_context = { triggered: [{ prop_id: 'at-common-range-button.n_clicks' }] }, "
        + f"ns.analyticsDateRangeButtons(1, null, null, {json.dumps(candidates)}, {json.dumps(common_daily)}))"
    )
    assert common == [
        "2024-03-01",
        "2024-11-30",
        {"start": "2024-03-01", "end": "2024-11-30"},
        "__NO_UPDATE__",
        "__NO_UPDATE__",
    ]

    common_daily_result = _run_dashmat_callbacks_js(
        "(window.dash_clientside.callback_context = { triggered: [{ prop_id: 'at-common-daily-button.n_clicks' }] }, "
        + f"ns.analyticsDateRangeButtons(null, 1, null, {json.dumps(candidates)}, {json.dumps(common_daily)}))"
    )
    assert common_daily_result == [
        "2024-04-01",
        "2024-10-31",
        {"start": "2024-04-01", "end": "2024-10-31"},
        "daily_trading",
        "daily_trading",
    ]

    max_result = _run_dashmat_callbacks_js(
        "(window.dash_clientside.callback_context = { triggered: [{ prop_id: 'at-maximum-range-button.n_clicks' }] }, "
        + f"ns.analyticsDateRangeButtons(null, null, 1, {json.dumps(candidates)}, {json.dumps(common_daily)}))"
    )
    assert max_result == [
        "2024-01-01",
        "2024-12-31",
        {"start": "2024-01-01", "end": "2024-12-31"},
        "__NO_UPDATE__",
        "__NO_UPDATE__",
    ]

    invalid = _run_dashmat_callbacks_js(
        "(window.dash_clientside.callback_context = { triggered: [{ prop_id: 'at-common-range-button.n_clicks' }] }, "
        "ns.analyticsDateRangeButtons(1, null, null, null, null))"
    )
    assert invalid == ["__NO_UPDATE__", "__NO_UPDATE__", "__NO_UPDATE__", "__NO_UPDATE__", "__NO_UPDATE__"]


def test_analytics_init_date_range_clientside_idempotent():
    result = _run_dashmat_callbacks_js(
        "ns.analyticsInitDateRange("
        + json.dumps({"available_series": ["Asset_A"], "max_start": "2024-01-01", "max_end": "2024-12-31"})
        + ","
        + json.dumps({"start": "2024-01-01", "end": "2024-12-31"})
        + ","
        + json.dumps("2024-01-01")
        + ","
        + json.dumps("2024-12-31")
        + ","
        + json.dumps({"display": "flex", "alignItems": "flex-start"})
        + ",false,false,true)"
    )

    assert result == [
        "__NO_UPDATE__",
        "__NO_UPDATE__",
        "__NO_UPDATE__",
        "__NO_UPDATE__",
        "__NO_UPDATE__",
        "__NO_UPDATE__",
        "__NO_UPDATE__",
    ]


def test_at_hidden_tab_trigger_stores_exist(page_modules):
    analyticstool, _ = page_modules
    for component_id in [
        "at-bootstrap-candidate-trigger-store",
        "at-statistics-tab-trigger-store",
        "at-returns-tab-trigger-store",
        "at-candidate-refresh-trigger-store",
        "at-rolling-tab-trigger-store",
        "at-calendar-tab-trigger-store",
        "at-growth-tab-trigger-store",
        "at-drawdown-tab-trigger-store",
        "at-factor-tab-trigger-store",
        "at-regime-tab-trigger-store",
        "at-conditional-tab-trigger-store",
        "at-correlogram-tab-trigger-store",
        "at-factor-preview-trigger-store",
        "at-regime-preview-trigger-store",
        "at-dataset-key-store",
        "at-shared-benchmark-stamp-store",
    ]:
        assert _find_component_by_id(analyticstool.layout, component_id) is not None


def test_analytics_tab_trigger_clientside_helper_respects_active_tab():
    matched = _run_dashmat_callbacks_js(
        'ns.analyticsTabTrigger("returns", "returns", true, true)'
    )
    assert matched["tab"] == "returns"
    assert matched["reason"]
    assert isinstance(matched["stamp"], int)

    assert _run_dashmat_callbacks_js(
        'ns.analyticsTabTrigger("returns", "statistics", true, true)'
    ) == "__NO_UPDATE__"


def test_analytics_modal_preview_trigger_clientside_helper():
    matched = _run_dashmat_callbacks_js('ns.analyticsModalPreviewTrigger(true)')
    assert matched["opened"] is True
    assert matched["reason"]
    assert isinstance(matched["stamp"], int)

    assert _run_dashmat_callbacks_js('ns.analyticsModalPreviewTrigger(false)') == "__NO_UPDATE__"
    assert _run_dashmat_callbacks_js(
        'ns.analyticsTabTrigger("returns", "returns", false, true)'
    ) == "__NO_UPDATE__"
    assert _run_dashmat_callbacks_js(
        'ns.analyticsTabTrigger("returns", "returns", true, false)'
    ) == "__NO_UPDATE__"


def test_analytics_candidate_refresh_trigger_clientside_helper():
    bootstrap = _run_dashmat_callbacks_js(
        'ns.analyticsBootstrapCandidateTrigger("dataset-1", "monthly", ["Asset_A"], false)'
    )
    assert bootstrap["phase"] == "bootstrap"
    assert _run_dashmat_callbacks_js(
        'ns.analyticsBootstrapCandidateTrigger("dataset-1", "monthly", ["Asset_A"], true)'
    ) == "__NO_UPDATE__"

    assert _run_dashmat_callbacks_js(
        'ns.analyticsCandidateRefreshTrigger("statistics", false, "dataset-1", ["Asset_A"])'
    ) == "__NO_UPDATE__"

    matched = _run_dashmat_callbacks_js(
        'ns.analyticsCandidateRefreshTrigger("correlogram", true, "dataset-1", ["Asset_A"])'
    )
    assert matched["tab"] == "correlogram"

    assert _run_dashmat_callbacks_js(
        'ns.analyticsCandidateRefreshTrigger("statistics", true, "dataset-1", ["Asset_A"])'
    ) == "__NO_UPDATE__"
    assert _run_dashmat_callbacks_js(
        'ns.analyticsCandidateRefreshTrigger("correlogram", true, null, ["Asset_A"])'
    ) == "__NO_UPDATE__"


def test_at_require_tab_trigger_accepts_match_and_raises_for_mismatch(page_modules):
    analyticstool, _ = page_modules

    payload = {"tab": "returns", "stamp": 1, "reason": "selection"}
    assert analyticstool._at_require_tab_trigger(payload, "returns") == payload

    with pytest.raises(PreventUpdate):
        analyticstool._at_require_tab_trigger({"tab": "rolling"}, "returns")

    with pytest.raises(PreventUpdate):
        analyticstool._at_require_tab_trigger(None, "returns")


def test_hidden_at_callbacks_use_family_trigger_inputs():
    main_page_text = _main_analyticstool_page_text()
    for trigger_id in [
        'Input("at-statistics-tab-trigger-store", "data")',
        'Input("at-returns-tab-trigger-store", "data")',
        'Input("at-rolling-tab-trigger-store", "data")',
        'Input("at-calendar-tab-trigger-store", "data")',
        'Input("at-growth-tab-trigger-store", "data")',
        'Input("at-drawdown-tab-trigger-store", "data")',
        'Input("at-correlogram-tab-trigger-store", "data")',
    ]:
        assert trigger_id in main_page_text

    advanced_page_text = _advanced_analyticstool_page_text()
    for trigger_id in [
        'Input("at-factor-tab-trigger-store", "data")',
        'Input("at-regime-tab-trigger-store", "data")',
        'Input("at-conditional-tab-trigger-store", "data")',
    ]:
        assert trigger_id in advanced_page_text


def test_hidden_at_trigger_emitters_include_restore_ready_guards():
    main_page_text = _main_analyticstool_page_text()
    assert 'Input("at-main-tabs", "value")' in main_page_text
    assert 'Input("at-initial-tab-render-ready-store", "data")' in main_page_text
    assert 'Input("at-state-ready-store", "data")' in main_page_text
    assert 'analyticsTabTrigger("statistics"' in main_page_text
    assert 'analyticsTabTrigger("returns"' in main_page_text
    assert 'analyticsTabTrigger("correlogram"' in main_page_text
    assert 'analyticsBootstrapCandidateTrigger' in main_page_text
    assert 'analyticsCandidateRefreshTrigger' in main_page_text

    advanced_page_text = _advanced_analyticstool_page_text()
    assert 'analyticsModalPreviewTrigger(opened)' in advanced_page_text


def test_at_account_list_live_apply_restores_from_raw_meta_without_secondary_restore_ready():
    page_text = Path("pages/analyticstool.py").read_text(encoding="utf-8")
    assert 'dcc.Store(id="at-account-list-live-apply-trigger-store"' not in page_text
    assert 'Input("at-account-list-live-apply-trigger-store", "data")' not in page_text
    assert 'Input("dashmat-raw-data-meta-store", "data")' in page_text
    assert 'Input("at-secondary-restore-ready-store", "data")' not in page_text


def test_analytics_download_excel_disabled_clientside_helper():
    assert _run_dashmat_callbacks_js(
        'ns.analyticsDownloadExcelDisabled(null, ["Asset_A"], {"start":"2024-01-01","end":"2024-12-31"}, true)'
    ) is True
    assert _run_dashmat_callbacks_js(
        'ns.analyticsDownloadExcelDisabled("raw", ["Asset_A"], {"start":"2024-01-01","end":"2024-12-31"}, true)'
    ) is False


def test_at_series_selection_grid_keeps_blocker_until_virtual_rows(monkeypatch, page_modules, raw_json):
    analyticstool, _ = page_modules

    monkeypatch.setattr(
        analyticstool,
        "get_raw_dataset_df",
        lambda *_args, **_kwargs: pytest.fail("should not fetch full dataset when metadata already has columns"),
    )
    children, _order, blocker = analyticstool.update_series_selectors(
        _raw_meta(raw_json),
        ["Asset_A"],
        ["Asset_A", "Asset_B"],
        [],
        {},
        {},
        {},
    )

    assert blocker is no_update
    assert getattr(children[0], "id", None) == "at-series-selection-grid"


def test_at_series_selection_blocker_release_uses_virtual_rows():
    page_text = Path("pages/analyticstool.py").read_text(encoding="utf-8")
    js_text = Path("assets/dashmat_callbacks.js").read_text(encoding="utf-8")
    assert 'ClientsideFunction(namespace="dashmat_callbacks", function_name="releaseBlockerOnSeriesGridReady")' in page_text
    assert 'Input("at-series-selection-grid", "virtualRowData", allow_optional=True)' in page_text
    assert "function releaseBlockerOnSeriesGridReady(virtualRows, modalOpened)" in js_text


def test_refresh_saved_series_cache_uses_raw_meta_max_date(monkeypatch, page_modules, raw_json):
    analyticstool, _ = page_modules
    raw_meta = _raw_meta(raw_json)
    saved_df = pd.DataFrame(
        {
            analyticstool.RISK_FREE_SERIES: [0.001, 0.0015],
            analyticstool.MARKET_BETA_SERIES: [0.01, -0.005],
        },
        index=pd.to_datetime(["2025-01-31", "2025-02-28"]),
    )
    saved_df.index.name = "Date"

    monkeypatch.setattr(analyticstool, "load_cma_returns_for_benches", lambda *_args, **_kwargs: saved_df)

    result = analyticstool.refresh_saved_series_cache(raw_meta, None)

    assert set(result["series_data"]) == {
        analyticstool.RISK_FREE_SERIES,
        analyticstool.MARKET_BETA_SERIES,
    }
    assert result["series_data"][analyticstool.MARKET_BETA_SERIES]["max_date"] == "2025-02-28"


def test_validate_db_add_selection_uses_raw_metadata(page_modules, raw_json):
    analyticstool, _ = page_modules

    message, hidden, disabled = analyticstool.validate_db_add_selection(
        ["Asset_A"],
        _raw_meta(raw_json),
        True,
    )

    assert message == "Cannot add duplicate series: Asset_A"
    assert hidden is False
    assert disabled is True


def test_restore_application_state_keeps_empty_selection_when_nothing_is_stored(page_modules, raw_json):
    analyticstool, _ = page_modules

    restored = analyticstool.restore_application_state(
        1,
        _raw_meta(raw_json),
        stored_periodicity="daily_trading",
        stored_series=[],
        stored_returns=None,
        stored_vol=None,
        stored_tab=None,
        stored_roll_win=None,
        stored_roll_metric=None,
        stored_roll_type=None,
        stored_roll_chart=None,
        stored_dd_chart=None,
        stored_gr_chart=None,
        stored_monthly_view=None,
        stored_monthly_series=[],
        stored_order=[],
        po_origin_series=[],
        page_visited=False,
    )

    assert restored[14] == []
    assert restored[15] == []
    assert restored[16] is False


def test_restore_application_state_silently_adds_po_series_after_first_visit(page_modules, raw_json):
    analyticstool, _ = page_modules

    restored = analyticstool.restore_application_state(
        1,
        _raw_meta(raw_json),
        stored_periodicity="daily_trading",
        stored_series=["Asset_A"],
        stored_returns=None,
        stored_vol=None,
        stored_tab=None,
        stored_roll_win=None,
        stored_roll_metric=None,
        stored_roll_type=None,
        stored_roll_chart=None,
        stored_dd_chart=None,
        stored_gr_chart=None,
        stored_monthly_view=None,
        stored_monthly_series=None,
        stored_order=["Asset_A", "Asset_B"],
        po_origin_series={"Asset_C": {"origin_page": "portopt", "origin_result": "Asset_C", "series_type": "portfolio"}},
        page_visited=True,
    )

    assert restored[14] == ["Asset_A", "Asset_C"]
    assert restored[15] == ["Asset_A", "Asset_B", "Asset_C"]
    assert restored[16] is False


def test_restore_application_state_defers_non_active_tab_controls(page_modules, raw_json):
    analyticstool, _ = page_modules

    restored = analyticstool.restore_application_state(
        1,
        _raw_meta(raw_json),
        stored_periodicity="daily_trading",
        stored_series=["Asset_A"],
        stored_returns="excess",
        stored_vol=7,
        stored_tab="statistics",
        stored_roll_win="3y",
        stored_roll_metric="volatility",
        stored_roll_type="cumulative",
        stored_roll_chart="table",
        stored_dd_chart="table",
        stored_gr_chart="table",
        stored_monthly_view="monthly",
        stored_monthly_series=None,
        stored_order=["Asset_A"],
        po_origin_series=[],
        page_visited=True,
    )

    assert restored[2] == "excess"
    assert restored[3] == 7
    assert restored[4] == "statistics"
    # Rolling outputs (5-10) deferred when not on rolling tab
    for i in range(5, 11):
        assert restored[i] is no_update
    # Drawdown, growth, monthly deferred when not on those tabs
    assert restored[11] is no_update
    assert restored[12] is no_update
    assert restored[13] is no_update
    assert restored[14] == ["Asset_A"]


def test_at_restore_secondary_controls_restores_only_active_tab_family(page_modules, raw_json):
    analyticstool, _ = page_modules

    restored = analyticstool.at_restore_secondary_controls(
        "rolling",
        True,
        _raw_meta(raw_json),
        stored_periodicity="daily_trading",
        stored_series=["Asset_A"],
        stored_returns="excess",
        stored_vol=7,
        stored_tab="rolling",
        stored_roll_win="3y",
        stored_roll_metric="volatility",
        stored_roll_type="cumulative",
        stored_roll_chart="table",
        stored_dd_chart="table",
        stored_gr_chart="table",
        stored_factor_mode="scatter",
        stored_factor_quantiles=7,
        stored_factor_transform="zscore",
        stored_factor_qq_reference="reference",
        stored_conditional_view=None,
        stored_conditional_comparator=None,
        stored_conditional_threshold=None,
        stored_conditional_window_conversion=None,
        stored_conditional_step=None,
        stored_conditional_step_unit=None,
        stored_conditional_display_mode=None,
        stored_regime_display_mode="detail",
        stored_monthly_view="monthly",
        stored_order=["Asset_A"],
        po_origin_series=[],
        page_visited=True,
        current_roll_win=None,
        current_roll_metric=None,
        current_roll_type=None,
        current_roll_type_disabled=None,
        current_roll_type_style=None,
        current_roll_chart=None,
        current_dd_chart=None,
        current_gr_chart=None,
        current_factor_mode=None,
        current_factor_quantiles=None,
        current_factor_transform=None,
        current_factor_qq_reference=None,
        current_conditional_view=None,
        current_conditional_comparator=None,
        current_conditional_threshold=None,
        current_conditional_window_conversion=None,
        current_conditional_step=None,
        current_conditional_step_unit=None,
        current_conditional_display_mode=None,
        current_regime_display_mode=None,
        current_monthly_view=None,
    )

    assert restored[0] == "3y"
    assert restored[1] == "volatility"
    assert restored[2] == "cumulative"
    assert restored[5] == "table"
    assert restored[6] is no_update
    assert restored[7] is no_update
    assert restored[8] is no_update
    assert restored[9] is no_update
    assert restored[10] is no_update
    assert restored[11] is no_update
    assert restored[12] is no_update
    assert restored[13] is no_update
    assert restored[14] is no_update
    assert restored[15] is no_update
    assert restored[16] is no_update
    assert restored[17] is no_update
    assert restored[18] is no_update
    assert restored[19] is no_update
    assert restored[20] is no_update


def test_at_restore_secondary_controls_skips_when_active_family_is_already_hydrated(page_modules, raw_json):
    analyticstool, _ = page_modules

    with pytest.raises(PreventUpdate):
        analyticstool.at_restore_secondary_controls(
            "rolling",
            True,
            _raw_meta(raw_json),
            stored_periodicity="daily_trading",
            stored_series=["Asset_A"],
            stored_returns="excess",
            stored_vol=7,
            stored_tab="rolling",
            stored_roll_win="3y",
            stored_roll_metric="volatility",
            stored_roll_type="cumulative",
            stored_roll_chart="table",
            stored_dd_chart="table",
            stored_gr_chart="table",
            stored_factor_mode="scatter",
            stored_factor_quantiles=7,
            stored_factor_transform="zscore",
            stored_factor_qq_reference="reference",
            stored_conditional_view=None,
            stored_conditional_comparator=None,
            stored_conditional_threshold=None,
            stored_conditional_window_conversion=None,
            stored_conditional_step=None,
            stored_conditional_step_unit=None,
            stored_conditional_display_mode=None,
            stored_regime_display_mode="detail",
            stored_monthly_view="monthly",
            stored_order=["Asset_A"],
            po_origin_series=[],
            page_visited=True,
            current_roll_win="3y",
            current_roll_metric="volatility",
            current_roll_type="cumulative",
            current_roll_type_disabled=True,
            current_roll_type_style={"opacity": 0.5, "pointerEvents": "none"},
            current_roll_chart="table",
            current_dd_chart=None,
            current_gr_chart=None,
            current_factor_mode=None,
            current_factor_quantiles=None,
            current_factor_transform=None,
            current_factor_qq_reference=None,
            current_conditional_view=None,
            current_conditional_comparator=None,
            current_conditional_threshold=None,
            current_conditional_window_conversion=None,
            current_conditional_step=None,
            current_conditional_step_unit=None,
            current_conditional_display_mode=None,
            current_regime_display_mode=None,
            current_monthly_view=None,
        )


def test_at_series_modal_open_is_clientside():
    page_text = Path("pages/analyticstool.py").read_text(encoding="utf-8")
    js_text = Path("assets/dashmat_callbacks.js").read_text(encoding="utf-8")

    assert 'ClientsideFunction(namespace="dashmat_callbacks", function_name="openAnalyticsSeriesModal")' in page_text
    assert "function openAnalyticsSeriesModal(" in js_text
    assert "def open_modal(" not in page_text


def test_at_series_modal_bulk_actions_use_shared_clientside_helper():
    page_text = Path("pages/analyticstool.py").read_text(encoding="utf-8")
    js_text = Path("assets/dashmat_callbacks.js").read_text(encoding="utf-8")

    assert 'ClientsideFunction(namespace="dashmat_callbacks", function_name="bulkUpdateSeriesSelection")' in page_text
    assert 'Output("at-series-bulk-action-dummy", "data")' in page_text
    assert 'Input("at-select-all-button", "n_clicks")' in page_text
    assert 'Input("at-unselect-all-button", "n_clicks")' in page_text
    assert 'State("at-series-selection-modal", "opened")' in page_text
    assert "function bulkUpdateSeriesSelection(selectAllClicks, unselectAllClicks, modalOpened)" in js_text
    assert 'targetField = "Selected";' in js_text
    assert "if (!node || !node.data || node.data.Delete) {" in js_text


def test_at_blocker_wiring_covers_add_modal_entry_and_series_render():
    page_text = Path("pages/analyticstool.py").read_text(encoding="utf-8")
    js_text = Path("assets/dashmat_callbacks.js").read_text(encoding="utf-8")

    assert 'Input("at-menu-add-from-db", "n_clicks")' in page_text
    assert 'Input("at-open-series-modal-button", "n_clicks")' in page_text
    assert 'Output("at-ui-blocker-store", "data", allow_duplicate=True)' in page_text
    assert 'Output("at-series-selection-container", "children")' in page_text
    assert 'Output("at-ui-blocker-store", "data", allow_duplicate=True),' in page_text
    assert 'ClientsideFunction(namespace="dashmat_callbacks", function_name="analyticsInitialSeriesBlocker")' in page_text
    assert 'Input("at-url-location", "pathname")' in page_text
    assert 'Input("at-series-selection-modal", "opened")' in page_text
    assert 'Input("at-series-selection-grid", "virtualRowData", allow_optional=True)' in page_text
    assert 'Input("at-page-load-trigger", "n_intervals")' in page_text
    assert 'State("at-page-visited-store", "data")' in page_text
    assert 'State("at-series-order-store", "data")' in page_text
    assert 'State("dashmat-pending-new-series-store", "data")' in page_text
    assert "function analyticsInitialSeriesBlocker(pathname, rawMeta, currentSelect, pageLoadReady, modalOpened, virtualRows, pageVisited, currentOrder, poOriginSeries)" in js_text
    assert "function analyticsInitialSeriesModalPending(rawMeta, currentSelect, currentOrder, poOriginSeries, pageVisited)" in js_text


def test_analyticstool_file_menu_includes_account_list_actions():
    page_text = Path("pages/analyticstool.py").read_text(encoding="utf-8")
    assert 'id="at-menu-load-account-list"' in page_text
    assert 'id="at-menu-save-account-list"' in page_text


def test_analyticstool_layout_drops_dead_focus_artifacts():
    page_text = Path("pages/analyticstool.py").read_text(encoding="utf-8")
    assert 'id="at-edit-box-focus-trigger"' not in page_text
    assert 'id="at-dummy-focus-output"' not in page_text
    assert "welcome_switch_buttons=()," in page_text
    assert page_text.index('id="at-menu-save-session"') < page_text.index('id="at-menu-load-account-list"')
    assert 'Input("dashmat-raw-data-meta-store", "data")' in page_text


def test_analyticstool_save_session_disabled_without_raw_data(page_modules):
    analyticstool, _ = page_modules

    assert analyticstool.at_toggle_save_session(None) is True
    assert analyticstool.at_toggle_save_session({}) is True
    assert analyticstool.at_toggle_save_session({"has_data": True}) is False


def test_analyticstool_save_session_disable_is_clientside_and_uses_raw_meta():
    page_text = Path("pages/analyticstool.py").read_text(encoding="utf-8")
    callback_text = page_text.split('Output("at-menu-save-session", "disabled")', 1)[-1]
    callback_text = callback_text.split("@callback(", 1)[0]
    assert 'Input("dashmat-raw-data-meta-store", "data")' in callback_text
    assert 'Input("dashmat-raw-data-store", "data")' not in callback_text


def test_analyticstool_open_advanced_disabled_without_raw_data(page_modules):
    analyticstool, _ = page_modules

    assert analyticstool.at_toggle_open_advanced(None) is True
    assert analyticstool.at_toggle_open_advanced({}) is True
    assert analyticstool.at_toggle_open_advanced({"has_data": True}) is False


def test_analyticstool_open_advanced_disable_is_clientside_and_uses_raw_meta():
    page_text = Path("pages/analyticstool.py").read_text(encoding="utf-8")
    callback_text = page_text.split('Output("at-menu-open-advanced", "disabled")', 1)[-1]
    callback_text = callback_text.split("@callback(", 1)[0]
    assert 'Input("dashmat-raw-data-meta-store", "data")' in callback_text
    assert 'Input("dashmat-raw-data-store", "data")' not in callback_text


def test_initial_series_blocker_holds_while_modal_is_open():
    js_text = Path("assets/dashmat_callbacks.js").read_text(encoding="utf-8")
    assert "function startInitialSeriesModalBlocker(pathname, pageLoadReady, modalOpened, modalStillNeeded, virtualRows, targetPath)" in js_text
    assert 'if (Array.isArray(virtualRows)) {' in js_text
    assert 'if (modalOpened === true) {' in js_text
    assert 'if (modalStillNeeded) {' in js_text


def test_at_layout_starts_with_welcome_and_main_hidden(page_modules):
    analyticstool, _ = page_modules

    welcome = _find_component_by_id(analyticstool.layout, "at-welcome-screen-container")
    main = _find_component_by_id(analyticstool.layout, "at-main-app-container")
    blocker_store = _find_component_by_id(analyticstool.layout, "at-ui-blocker-store")
    blocker_overlay = _find_component_by_id(analyticstool.layout, "at-ui-blocker-overlay")

    assert getattr(welcome, "style", {})["display"] == "none"
    assert getattr(main, "style", {})["display"] == "none"
    assert getattr(blocker_store, "data", None) is False
    assert getattr(blocker_overlay, "visible", None) is False
    assert getattr(blocker_overlay, "zIndex", None) == 2500


def test_at_result_grids_treat_series_fields_as_literal_keys(page_modules):
    analyticstool, _ = page_modules

    grid_ids = [
        "at-returns-grid",
        "at-rolling-grid",
        "at-statistics-grid",
        "at-calendar-grid",
        "at-growth-grid",
        "at-drawdown-grid",
    ]

    for grid_id in grid_ids:
        grid = _find_component_by_id(analyticstool.layout, grid_id)
        assert getattr(grid, "dashGridOptions", {})["suppressFieldDotNotation"] is True
        assert getattr(grid, "dashGridOptions", {})["processCellForClipboard"] == {
            "function": "dashmatProcessCellForClipboard(params)"
        }


def test_conditional_factor_window_uses_tooltip_help_target(page_modules):
    analyticstool, _ = page_modules

    tooltip = _find_component_by_id(analyticstool.layout, "at-conditional-window-conversion-tooltip")
    target = _find_component_by_id(analyticstool.layout, "at-conditional-window-conversion-tooltip-target")
    removed_note = _find_component_by_id(analyticstool.layout, "at-conditional-conversion-note")

    assert tooltip is not None
    assert getattr(tooltip, "position", None) == "top"
    assert getattr(tooltip, "withArrow", None) is True
    assert getattr(tooltip, "disabled", None) in (None, False)
    assert "return-like factors" in str(getattr(tooltip, "label", ""))
    assert "additive factors" in str(getattr(tooltip, "label", ""))
    assert target is not None
    assert removed_note is None


def test_conditional_conversion_tooltip_text_is_static(page_modules):
    analyticstool, _ = page_modules

    label = analyticstool._conditional_conversion_tooltip_text()
    assert "return-like factors" in label
    assert "level-like factors" in label
    assert "additive factors" in label


def test_at_series_modal_has_explicit_zindex():
    modal_text = Path("utils/dashmat_welcome_modal.py").read_text(encoding="utf-8")
    assert "zIndex=1900" in modal_text


def test_at_series_modal_has_bulk_action_controls_and_dummy_sink():
    modal_text = Path("utils/dashmat_welcome_modal.py").read_text(encoding="utf-8")
    assert '"series-bulk-action-dummy"' in modal_text
    assert '"select-all-button"' in modal_text
    assert '"unselect-all-button"' in modal_text


def test_at_bootstrap_uses_only_page_load_interval_without_live_apply_trigger_store():
    page_text = Path("pages/analyticstool.py").read_text(encoding="utf-8")
    assert 'dcc.Interval(id="at-page-load-trigger"' in page_text
    assert 'at-initial-tab-render-trigger' not in page_text
    assert 'at-secondary-restore-trigger' not in page_text
    assert 'Output("at-initial-tab-render-ready-store", "data")' in page_text
    assert 'Input("at-page-load-trigger", "n_intervals")' in page_text
    assert 'dcc.Store(id="at-account-list-live-apply-trigger-store"' not in page_text
    assert 'Input("at-account-list-live-apply-trigger-store", "data")' not in page_text
    assert 'Input("at-state-ready-store", "data")' in page_text
    assert 'Output("at-welcome-screen-container", "style")' in page_text
    assert 'Input("dashmat-raw-data-store", "data")' in page_text
    visibility_block = page_text.split('Output("at-welcome-screen-container", "style")', 1)[1].split(
        'Output("at-initial-tab-render-ready-store", "data")',
        1,
    )[0]
    assert 'Input("at-page-load-trigger", "n_intervals")' in visibility_block


def test_update_statistics_transposes_series_into_columns(monkeypatch, page_modules):
    analyticstool, _ = page_modules
    candidates = {
        "available_series": ["Asset_A", "Asset_B"],
        "max_start": "2024-01-01",
        "max_end": "2024-12-31",
        "common_start": "2024-01-01",
        "common_end": "2024-12-31",
    }
    common_daily = {"common_daily_start": "2024-01-01", "common_daily_end": "2024-12-31"}

    def _fake_stats(*_args, **_kwargs):
        return [
            {"Series": "Asset_A", "Cumulative Return": 0.10},
            {"Series": "Asset_B", "Cumulative Return": 0.20},
        ]

    monkeypatch.setattr(analyticstool, "calculate_statistics_cached", _fake_stats)

    target_key = analyticstool._statistics_tab_signature(
        "unit-test-dataset",
        "daily",
        ["Asset_A", "Asset_B"],
        {},
        {},
        {"start": "2024-01-01", "end": "2024-12-31"},
        0,
        {},
        True,
        {},
    )

    column_defs, row_data, loaded, rendered_key = analyticstool.update_statistics(
        {"tab": "statistics"},
        "statistics",
        "unit-test-dataset",
        "daily",
        ["Asset_A", "Asset_B"],
        {},
        {},
        {"start": "2024-01-01", "end": "2024-12-31"},
        True,
        0,
        {},
        True,
        {},
        True,
        None,
        candidates,
    )

    assert column_defs[0]["field"] == "Statistic"
    assert {c["field"] for c in column_defs[1:]} == {"Asset_A", "Asset_B"}
    cum_row = next(row for row in row_data if row["Statistic"] == "Cumulative Return")
    assert cum_row["Asset_A"] == pytest.approx(0.10)
    assert cum_row["Asset_B"] == pytest.approx(0.20)
    assert loaded is True
    assert rendered_key == target_key


def test_update_download_excel_disabled_uses_ready_state(page_modules):
    analyticstool, _ = page_modules
    assert analyticstool.update_download_excel_disabled(None, ["Asset_A"], None, True) is True
    assert analyticstool.update_download_excel_disabled("raw", ["Asset_A"], None, True) is True
    assert (
        analyticstool.update_download_excel_disabled(
            "raw",
            ["Asset_A"],
            {"start": "2024-01-01", "end": "2024-12-31"},
            True,
        )
        is False
    )
    assert (
        analyticstool.update_download_excel_disabled(
            "raw",
            ["Asset_A"],
            {"start": "2024-01-01", "end": "2024-12-31"},
            True,
            False,
        )
        is no_update
    )


def test_analytics_date_range_store_update_clientside_matches_python(page_modules):
    analyticstool, _ = page_modules

    assert _run_dashmat_callbacks_js(
        'ns.analyticsDateRangeStoreUpdate("2024-01-01", "2024-12-31", null)'
    ) == analyticstool.update_date_range_store("2024-01-01", "2024-12-31", None)
    assert _run_dashmat_callbacks_js(
        'ns.analyticsDateRangeStoreUpdate("2024-01-01", null, null)'
    ) == "__NO_UPDATE__"
    assert _run_dashmat_callbacks_js(
        'ns.analyticsDateRangeStoreUpdate("2024-01-01", "2024-12-31", {"start":"2024-01-01","end":"2024-12-31"})'
    ) == "__NO_UPDATE__"


def test_update_statistics_requires_ready_state(page_modules):
    analyticstool, _ = page_modules

    with pytest.raises(PreventUpdate):
        analyticstool.update_statistics(
            {"tab": "statistics"},
            "statistics",
            "unit-test-dataset",
            "daily",
            ["Asset_A"],
            {},
            {},
            None,
            False,
            0,
            {},
            True,
            {},
            True,
            None,
            None,
        )


def test_update_statistics_requires_selected_tab_and_initial_ready(page_modules):
    analyticstool, _ = page_modules

    with pytest.raises(PreventUpdate):
        analyticstool.update_statistics(
            {"tab": "statistics"},
            "returns",
            "unit-test-dataset",
            "daily",
            ["Asset_A"],
            {},
            {},
            {"start": "2024-01-01", "end": "2024-12-31"},
            True,
            0,
            {},
            True,
            {},
            True,
            None,
            None,
        )

    with pytest.raises(PreventUpdate):
        analyticstool.update_statistics(
            {"tab": "statistics"},
            "statistics",
            "unit-test-dataset",
            "daily",
            ["Asset_A"],
            {},
            {},
            {"start": "2024-01-01", "end": "2024-12-31"},
            True,
            0,
            {},
            True,
            {},
            False,
            None,
            None,
        )


def test_update_returns_grid_skips_unchanged_tab_revisit(monkeypatch, page_modules, raw_json):
    analyticstool, _ = page_modules
    candidates = {"available_series": ["Asset_A"], "max_start": "2024-01-01", "max_end": "2024-12-31"}
    signature = analyticstool._returns_tab_signature(
        raw_json,
        "daily",
        ["Asset_A"],
        "total",
        {},
        {},
        {"start": "2024-01-01", "end": "2024-12-31"},
        0,
        {},
    )
    monkeypatch.setattr(analyticstool, "callback_context", type("Ctx", (), {"triggered_id": "at-main-tabs"})())
    monkeypatch.setattr(analyticstool, "_compute_selected_returns", lambda *_args, **_kwargs: pytest.fail("should skip unchanged revisit"))

    result = analyticstool.update_grid(
        {"tab": "returns"},
        raw_json,
        "daily",
        ["Asset_A"],
        "total",
        {},
        {},
        {"start": "2024-01-01", "end": "2024-12-31"},
        True,
        0,
        {},
        "returns",
        True,
        {"returns": signature},
        candidates,
    )

    assert result == (no_update, no_update, no_update)


def test_update_statistics_skips_when_target_already_rendered(monkeypatch, page_modules, raw_json):
    analyticstool, _ = page_modules
    dataset_key = analyticstool._dataset_key(raw_json)
    candidates = analyticstool.update_at_range_candidates(dataset_key, "daily", ["Asset_A"])
    resolved_start, resolved_end = analyticstool.resolve_initial_range(
        candidates,
        {"start": "2024-01-01", "end": "2024-12-31"},
    )
    signature = analyticstool._statistics_tab_signature(
        dataset_key,
        "daily",
        ["Asset_A"],
        {},
        {},
        {"start": resolved_start, "end": resolved_end},
        0,
        {},
        True,
        {},
    )
    monkeypatch.setattr(analyticstool, "calculate_statistics_cached", lambda *_args, **_kwargs: pytest.fail("should skip unchanged revisit"))

    assert analyticstool.update_statistics(
        {"tab": "statistics"},
        "statistics",
        dataset_key,
        "daily",
        ["Asset_A"],
        {},
        {},
        {"start": "2024-01-01", "end": "2024-12-31"},
        True,
        0,
        {},
        True,
        {},
        True,
        signature,
        candidates,
    ) == (no_update, no_update, no_update, no_update)


def test_update_statistics_updates_candidates_and_uses_resolved_range(monkeypatch, page_modules):
    analyticstool, _ = page_modules
    captured = {}
    candidates = {
        "available_series": ["Asset_A"],
        "max_start": "2024-02-01",
        "max_end": "2024-12-31",
        "common_start": "2024-02-01",
        "common_end": "2024-12-31",
    }
    def _fake_stats(*args):
        captured["date_range_payload"] = args[5]
        return [{"Series": "Asset_A", "Total Return": 0.1}]

    monkeypatch.setattr(analyticstool, "calculate_statistics_cached", _fake_stats)
    monkeypatch.setattr(
        analyticstool,
        "_resolve_shared_benchmark_payload",
        lambda *_args: {"risk_free_json": "", "spx_json": ""},
    )

    result = analyticstool.update_statistics(
        {"tab": "statistics"},
        "statistics",
        "unit-test-dataset",
        "daily",
        ["Asset_A"],
        {},
        {},
        {"start": "2024-01-01", "end": "2024-12-31"},
        True,
        0,
        {},
        True,
        {},
        True,
        None,
        candidates,
    )

    assert json.loads(captured["date_range_payload"]) == {"start": "2024-02-01", "end": "2024-12-31"}


def test_update_returns_grid_updates_candidates_and_uses_resolved_range(monkeypatch, page_modules, raw_json):
    analyticstool, _ = page_modules
    captured = {}
    candidates = {
        "available_series": ["Asset_A"],
        "max_start": "2024-02-01",
        "max_end": "2024-12-31",
        "common_start": "2024-02-01",
        "common_end": "2024-12-31",
    }
    display_df = pd.DataFrame(
        {"Asset_A": [0.1]},
        index=pd.to_datetime(["2024-02-01"]),
    )
    display_df.index.name = "Date"

    def _fake_returns(*args):
        captured["date_range"] = args[6]
        return display_df

    monkeypatch.setattr(analyticstool, "_compute_selected_returns", _fake_returns)

    result = analyticstool.update_grid(
        {"tab": "returns"},
        raw_json,
        "daily",
        ["Asset_A"],
        "total",
        {},
        {},
        {"start": "2024-01-01", "end": "2024-12-31"},
        True,
        0,
        {},
        "returns",
        True,
        {},
        candidates,
    )

    assert captured["date_range"] == {"start": "2024-02-01", "end": "2024-12-31"}


def test_update_at_dataset_key_store_dedupes_unchanged(page_modules):
    analyticstool, _ = page_modules

    assert (
        analyticstool.update_at_dataset_key_store({"dataset_key": "unit-test-dataset"}, None)
        == "unit-test-dataset"
    )

    with pytest.raises(PreventUpdate):
        analyticstool.update_at_dataset_key_store(
            {"dataset_key": "unit-test-dataset"},
            "unit-test-dataset",
        )


def test_shared_benchmark_stamp_store_helpers_round_trip(page_modules):
    analyticstool, _ = page_modules
    shared_store = {
        "series_data": {
            analyticstool.RISK_FREE_SERIES: {
                "max_date": "2024-12-31",
                "returns_json": "rf-json",
            },
            analyticstool.MARKET_BETA_SERIES: {
                "max_date": "2025-01-31",
                "returns_json": "spx-json",
            },
        }
    }

    payload = analyticstool._extract_shared_benchmark_payload(shared_store)
    assert payload == {
        "risk_free_json": "rf-json",
        "spx_json": "spx-json",
        "risk_free_max_date": "2024-12-31",
        "spx_max_date": "2025-01-31",
    }

    stamp = analyticstool._build_shared_benchmark_stamp(payload)
    assert set(stamp.keys()) == {
        "risk_free_max_date",
        "spx_max_date",
        "risk_free_hash",
        "spx_hash",
    }
    assert stamp["risk_free_hash"]
    assert stamp["spx_hash"]

    analyticstool._cache_shared_benchmark_payload(stamp, payload)
    assert analyticstool._resolve_shared_benchmark_payload(stamp) == {
        "risk_free_json": "rf-json",
        "spx_json": "spx-json",
    }
    assert analyticstool._resolve_shared_benchmark_payload(shared_store) == {
        "risk_free_json": "rf-json",
        "spx_json": "spx-json",
    }


def test_update_at_shared_benchmark_stamp_store_dedupes_unchanged(page_modules):
    analyticstool, _ = page_modules
    shared_store = {
        "series_data": {
            analyticstool.RISK_FREE_SERIES: {
                "max_date": "2024-12-31",
                "returns_json": "rf-json",
            },
            analyticstool.MARKET_BETA_SERIES: {
                "max_date": "2025-01-31",
                "returns_json": "spx-json",
            },
        }
    }

    next_stamp = analyticstool.update_at_shared_benchmark_stamp_store(shared_store, None)
    assert set(next_stamp.keys()) == {
        "risk_free_max_date",
        "spx_max_date",
        "risk_free_hash",
        "spx_hash",
    }

    with pytest.raises(PreventUpdate):
        analyticstool.update_at_shared_benchmark_stamp_store(shared_store, next_stamp)


def test_resolve_shared_benchmark_payload_uses_stamp_lookup_when_cache_misses(monkeypatch, page_modules):
    analyticstool, _ = page_modules
    stamp = {
        "risk_free_max_date": "2024-12-31",
        "spx_max_date": "2025-01-31",
        "risk_free_hash": "rf-hash",
        "spx_hash": "spx-hash",
    }

    monkeypatch.setattr(analyticstool.cache_config.cache, "get", lambda _key: None)
    monkeypatch.setattr(
        analyticstool,
        "_load_shared_benchmark_payload_from_stamp",
        lambda *_args: {"risk_free_json": "rf-json", "spx_json": "spx-json"},
    )

    assert analyticstool._resolve_shared_benchmark_payload(stamp) == {
        "risk_free_json": "rf-json",
        "spx_json": "spx-json",
    }


def test_statistics_render_schedules_from_statistics_trigger_store():
    page_text = Path("pages/analyticstool.py").read_text(encoding="utf-8")
    assert 'Input("at-statistics-tab-trigger-store", "data")' in page_text
    assert 'Input("at-statistics-target-key-store", "data")' not in page_text
    assert 'dcc.Store(id="at-statistics-target-key-store"' not in page_text
    assert 'State("at-dataset-key-store", "data")' in page_text
    trigger_callback = page_text.split('Output("at-statistics-tab-trigger-store", "data")', 1)[-1]
    trigger_callback = trigger_callback.split('Output("at-returns-tab-trigger-store", "data")', 1)[0]
    assert 'Input("at-shared-benchmark-stamp-store", "data")' in trigger_callback
    assert 'Input("dashmat-saved-series-cache-store", "data")' not in trigger_callback
    render_callback = page_text.split('Output("at-statistics-grid", "columnDefs")', 1)[-1]
    render_callback = render_callback.split('Output("at-correlation-loaded-store", "data"', 1)[0]
    assert 'State("at-shared-benchmark-stamp-store", "data")' in render_callback
    assert 'State("dashmat-saved-series-cache-store", "data")' not in render_callback
    assert 'State("at-range-candidates-store", "data")' in render_callback
    assert 'State("at-common-daily-candidates-store", "data")' not in render_callback
    assert 'Output("at-range-candidates-store", "data", allow_duplicate=True)' not in render_callback
    assert 'Output("at-common-daily-candidates-store", "data", allow_duplicate=True)' not in render_callback


def test_returns_render_no_longer_writes_candidate_stores():
    page_text = Path("pages/analyticstool.py").read_text(encoding="utf-8")
    render_callback = page_text.split('Output("at-returns-grid", "columnDefs")', 1)[-1]
    render_callback = render_callback.split('Output("at-menu-download-excel", "disabled")', 1)[0]
    assert 'State("at-range-candidates-store", "data")' in render_callback
    assert 'State("at-common-daily-candidates-store", "data")' not in render_callback
    assert 'Output("at-range-candidates-store", "data", allow_duplicate=True)' not in render_callback
    assert 'Output("at-common-daily-candidates-store", "data", allow_duplicate=True)' not in render_callback


def test_analytics_date_candidate_callback_uses_dataset_key_store():
    page_text = Path("pages/analyticstool.py").read_text(encoding="utf-8")
    callback_text = page_text.split('def update_at_date_candidate_stores(', 1)[0]
    callback_text = callback_text.rsplit('@callback(', 1)[-1]
    assert 'State("at-dataset-key-store", "data")' in callback_text
    assert 'State("at-periodicity-select", "value")' in callback_text
    assert 'State("at-series-select", "data")' in callback_text


def test_correlogram_candidate_refresh_callback_uses_trigger_store():
    page_text = Path("pages/analyticstool.py").read_text(encoding="utf-8")
    callback_text = page_text.split('def refresh_correlogram_candidate_stores(', 1)[0]
    callback_text = callback_text.rsplit('@callback(', 1)[-1]
    assert 'Input("at-candidate-refresh-trigger-store", "data")' in callback_text
    assert 'State("at-dataset-key-store", "data")' in callback_text
    assert 'State("at-periodicity-select", "value")' in callback_text
    assert 'State("at-series-select", "data")' in callback_text


def test_candidate_refresh_trigger_callback_does_not_depend_on_candidate_stores():
    page_text = Path("pages/analyticstool.py").read_text(encoding="utf-8")
    trigger_block = page_text.split('Output("at-candidate-refresh-trigger-store", "data")', 1)[-1]
    trigger_block = trigger_block.split('Output("at-statistics-tab-trigger-store", "data")', 1)[0]
    assert 'Input("at-range-candidates-store", "data")' not in trigger_block
    assert 'Input("at-common-daily-candidates-store", "data")' not in trigger_block


def test_download_excel_uses_shared_benchmark_stamp_store():
    page_text = Path("pages/analyticstool.py").read_text(encoding="utf-8")
    download_callback = page_text.split('Output("at-download-excel", "data")', 1)[-1]
    download_callback = download_callback.split('def download_excel(', 1)[0]
    assert 'State("at-shared-benchmark-stamp-store", "data")' in download_callback
    assert 'State("dashmat-saved-series-cache-store", "data")' not in download_callback



def test_statistics_loading_uses_native_loading_without_manual_display_callback():
    page_text = Path("pages/analyticstool.py").read_text(encoding="utf-8")
    js_text = Path("assets/dashmat_callbacks.js").read_text(encoding="utf-8")
    assert 'Output("at-loading-statistics", "display")' not in page_text
    assert "analyticsStatisticsLoadingDisplay" not in js_text


def test_update_growth_grid_requires_growth_table_view(page_modules):
    analyticstool, _ = page_modules
    with pytest.raises(PreventUpdate):
        analyticstool.update_growth_grid(
            {"tab": "growth"},
            "returns",
            "table",
            "raw-json",
            "daily",
            ["Asset_A"],
            {},
            {},
            {"start": "2024-01-01", "end": "2024-12-31"},
            True,
            0,
            {},
        )


def test_update_growth_grid_builds_columns_and_rows(monkeypatch, page_modules):
    analyticstool, _ = page_modules
    growth_df = pd.DataFrame(
        {"Asset_A": [1.0, 1.1]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )
    growth_df.index.name = "Date"
    monkeypatch.setattr(analyticstool, "calculate_growth_of_dollar", lambda *args, **kwargs: growth_df)

    column_defs, row_data = analyticstool.update_growth_grid(
        {"tab": "growth"},
        "growth",
        "table",
        "raw-json",
        "daily",
        ["Asset_A"],
        {},
        {},
        {"start": "2024-01-01", "end": "2024-12-31"},
        True,
        0,
        {},
    )

    assert column_defs[0]["field"] == "Date"
    assert column_defs[1]["field"] == "Asset_A"
    assert row_data[0]["Date"] == "2024-01-01"


def test_update_drawdown_grid_builds_columns_and_rows(monkeypatch, page_modules):
    analyticstool, _ = page_modules
    drawdown_df = pd.DataFrame(
        {"Asset_A": [0.0, -0.03]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )
    drawdown_df.index.name = "Date"
    monkeypatch.setattr(analyticstool, "calculate_drawdown", lambda *args, **kwargs: drawdown_df)

    column_defs, row_data = analyticstool.update_drawdown_grid(
        {"tab": "drawdown"},
        "drawdown",
        "table",
        "raw-json",
        "daily",
        ["Asset_A"],
        "total",
        {},
        {},
        {"start": "2024-01-01", "end": "2024-12-31"},
        True,
        0,
        {},
    )

    assert column_defs[0]["field"] == "Date"
    assert column_defs[1]["field"] == "Asset_A"
    assert row_data[1]["Asset_A"] == pytest.approx(-0.03)


def test_update_drawdown_charts_matches_portopt_style(monkeypatch, page_modules):
    analyticstool, _ = page_modules
    drawdown_df = pd.DataFrame(
        {"Asset_A": [0.0, -0.03], "Asset_B": [0.0, -0.01]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )
    drawdown_df.index.name = "Date"
    monkeypatch.setattr(analyticstool, "calculate_drawdown", lambda *args, **kwargs: drawdown_df)

    graph = analyticstool.update_drawdown_charts(
        {"tab": "drawdown"},
        "drawdown",
        "chart",
        "raw-json",
        "daily",
        ["Asset_A", "Asset_B"],
        "total",
        {},
        {"Asset_B": True},
        {"start": "2024-01-01", "end": "2024-12-31"},
        True,
        0,
        {},
        "light",
    )

    fig = getattr(graph, "figure", None)
    assert fig is not None
    assert [trace.name for trace in fig.data] == ["Asset_A", "Asset_B"]
    assert all(getattr(trace, "fill", None) == "tozeroy" for trace in fig.data)
    assert fig.layout.title.text == "Drawdown"
    assert fig.layout.yaxis.title.text == "Drawdown"
    assert fig.layout.yaxis.tickformat == ".2%"
    assert fig.layout.hovermode == "x unified"
    assert fig.layout.margin.t == 40
    assert fig.layout.margin.b == 40
    assert fig.layout.margin.l == 60
    assert fig.layout.margin.r >= 160


def test_update_correlogram_meta_returns_no_update_when_not_active(page_modules):
    analyticstool, _ = page_modules
    with pytest.raises(PreventUpdate):
        analyticstool.update_correlogram_meta({"tab": "growth"}, ["Asset_A", "Asset_B"])
    assert analyticstool.update_correlogram_meta({"tab": "correlogram"}, ["Asset_A", "Asset_B"]) == {"num_series": 2}


def test_sync_at_returns_type_from_mirrors_updates_canonical(monkeypatch, page_modules):
    analyticstool, _ = page_modules
    monkeypatch.setattr(
        analyticstool,
        "callback_context",
        type("Ctx", (), {"triggered_id": "at-returns-type-select-factor"})(),
    )

    result = analyticstool.sync_at_returns_type_from_mirrors(
        "total",
        "total",
        "total",
        "total",
        "excess",
        "total",
        "total",
        "total",
    )

    assert result == "excess"


def test_sync_at_returns_type_mirrors_only_updates_mismatched(page_modules):
    analyticstool, _ = page_modules

    result = analyticstool.sync_at_returns_type_mirrors(
        "excess",
        "excess",
        "total",
        "excess",
        "total",
        "total",
        "total",
        "excess",
    )

    assert result[0] is no_update
    assert result[1] == "excess"
    assert result[2] is no_update
    assert result[3] == "excess"
    assert result[4] == "excess"
    assert result[5] == "excess"
    assert result[6] is no_update


def test_update_correlogram_target_key_changes_on_exp_weight_inputs(page_modules):
    analyticstool, _ = page_modules
    date_range = {"start": "2024-01-01", "end": "2024-12-31"}

    key_unweighted = analyticstool.update_correlogram_target_key(
        {"tab": "correlogram"},
        None,
        None,
        "daily",
        ["Asset_A", "Asset_B"],
        "total",
        {},
        {},
        date_range,
        True,
        0,
        {},
        {},
        "correlation",
        False,
        63,
        "none",
        "scaled_identity",
        None,
    )
    key_weighted = analyticstool.update_correlogram_target_key(
        {"tab": "correlogram"},
        None,
        None,
        "daily",
        ["Asset_A", "Asset_B"],
        "total",
        {},
        {},
        date_range,
        True,
        0,
        {},
        {},
        "correlation",
        True,
        0.94,
        "none",
        "scaled_identity",
        None,
    )

    assert isinstance(key_unweighted, str)
    assert isinstance(key_weighted, str)
    assert key_unweighted != key_weighted
    assert (
        analyticstool.update_correlogram_target_key(
            {"tab": "correlogram"},
            None,
            None,
            "daily",
            ["Asset_A", "Asset_B"],
            "total",
            {},
            {},
            date_range,
            True,
            0,
            {},
            {},
            "correlation",
            True,
            0.94,
            "none",
            "scaled_identity",
            key_weighted,
        )
        is no_update
    )


def test_update_correlogram_target_key_changes_on_shrinkage_for_matrix_views(page_modules):
    analyticstool, _ = page_modules
    date_range = {"start": "2024-01-01", "end": "2024-12-31"}

    key_none = analyticstool.update_correlogram_target_key(
        {"tab": "correlogram"},
        None,
        None,
        "daily",
        ["Asset_A", "Asset_B"],
        "total",
        {},
        {},
        date_range,
        True,
        0,
        {},
        {},
        "correlation",
        False,
        63,
        "none",
        "scaled_identity",
        None,
    )
    key_shrunk = analyticstool.update_correlogram_target_key(
        {"tab": "correlogram"},
        None,
        None,
        "daily",
        ["Asset_A", "Asset_B"],
        "total",
        {},
        {},
        date_range,
        True,
        0,
        {},
        {},
        "correlation",
        False,
        63,
        "ledoit_wolf",
        "scaled_identity",
        None,
    )

    assert isinstance(key_none, str)
    assert isinstance(key_shrunk, str)
    assert key_none != key_shrunk


def test_update_correlogram_target_key_ignores_shrinkage_for_scatter_view(page_modules):
    analyticstool, _ = page_modules
    date_range = {"start": "2024-01-01", "end": "2024-12-31"}

    key_scatter = analyticstool.update_correlogram_target_key(
        {"tab": "correlogram"},
        None,
        None,
        "daily",
        ["Asset_A", "Asset_B"],
        "total",
        {},
        {},
        date_range,
        True,
        0,
        {},
        {},
        "scatter",
        False,
        63,
        "none",
        "scaled_identity",
        None,
    )

    assert (
        analyticstool.update_correlogram_target_key(
            {"tab": "correlogram"},
            None,
            None,
            "daily",
            ["Asset_A", "Asset_B"],
            "total",
            {},
            {},
            date_range,
            True,
            0,
            {},
            {},
            "scatter",
            False,
            63,
            "oas",
            "constant_correlation",
            key_scatter,
        )
        is no_update
    )


def test_update_correlogram_heatmap_title_includes_shrinkage(monkeypatch, page_modules):
    analyticstool, _ = page_modules
    result = {
        "display_df": pd.DataFrame({"Asset_A": [0.01, 0.02], "Asset_B": [0.00, 0.03]}),
        "corr_matrix": pd.DataFrame([[1.0, 0.5], [0.5, 1.0]], index=["Asset_A", "Asset_B"], columns=["Asset_A", "Asset_B"]),
        "cov_matrix": pd.DataFrame([[0.04, 0.01], [0.01, 0.09]], index=["Asset_A", "Asset_B"], columns=["Asset_A", "Asset_B"]),
        "available_series": ["Asset_A", "Asset_B"],
        "n": 2,
    }
    monkeypatch.setattr(analyticstool, "generate_correlogram_cached", lambda *_args, **_kwargs: result)

    graph, rendered_key = analyticstool.update_correlogram(
        "req-key",
        "correlogram",
        "raw-json",
        "daily",
        ["Asset_A", "Asset_B"],
        "total",
        {},
        {},
        {"start": "2024-01-01", "end": "2024-12-31"},
        True,
        0,
        {},
        False,
        63,
        "ledoit_wolf",
        "scaled_identity",
        "correlation",
        120,
        "light",
    )

    assert rendered_key == "req-key"
    assert "Ledoit-Wolf" in graph.figure.layout.title.text


def test_update_correlogram_heatmap_title_includes_shrinkage_target(monkeypatch, page_modules):
    analyticstool, _ = page_modules
    result = {
        "display_df": pd.DataFrame({"Asset_A": [0.01, 0.02], "Asset_B": [0.00, 0.03]}),
        "corr_matrix": pd.DataFrame([[1.0, 0.5], [0.5, 1.0]], index=["Asset_A", "Asset_B"], columns=["Asset_A", "Asset_B"]),
        "cov_matrix": pd.DataFrame([[0.04, 0.01], [0.01, 0.09]], index=["Asset_A", "Asset_B"], columns=["Asset_A", "Asset_B"]),
        "available_series": ["Asset_A", "Asset_B"],
        "n": 2,
    }
    monkeypatch.setattr(analyticstool, "generate_correlogram_cached", lambda *_args, **_kwargs: result)

    graph, rendered_key = analyticstool.update_correlogram(
        "req-key",
        "correlogram",
        "raw-json",
        "daily",
        ["Asset_A", "Asset_B"],
        "total",
        {},
        {},
        {"start": "2024-01-01", "end": "2024-12-31"},
        True,
        0,
        {},
        False,
        63,
        "ledoit_wolf",
        "constant_correlation",
        "correlation",
        120,
        "light",
    )

    assert rendered_key == "req-key"
    assert "Constant Correlation" in graph.figure.layout.title.text


def test_on_modal_ok_does_not_emit_raw_data_when_unchanged(page_modules, raw_json):
    analyticstool, _ = page_modules

    result = analyticstool.on_modal_ok(
        _series_snapshot(
            [
                {
                    "__row_key": "Asset_A",
                    "Selected": True,
                    "Series": "Asset_A",
                    "Benchmark": "None",
                    "LongShort": False,
                    "ScaleVol": True,
                    "Delete": False,
                }
            ]
        ),
        raw_json,
        ["Asset_A"],
        {},
        {},
        ["Asset_A"],
        {},
        {},
    )

    assert result[6] is no_update
    assert len(result) == 9


def test_on_modal_ok_returns_no_update_for_unchanged_persisted_outputs(page_modules, raw_json):
    analyticstool, _ = page_modules

    result = analyticstool.on_modal_ok(
        _series_snapshot(
            [
                {
                    "__row_key": "Asset_A",
                    "Selected": True,
                    "Series": "Asset_A",
                    "Benchmark": "None",
                    "LongShort": False,
                    "ScaleVol": True,
                    "Delete": False,
                }
            ]
        ),
        raw_json,
        ["Asset_A"],
        {"Asset_A": "None"},
        {"Asset_A": False},
        ["Asset_A"],
        {"Asset_A": True},
        {},
    )

    assert result[0] is no_update
    assert result[1] is no_update
    assert result[2] is no_update
    assert result[3] is no_update
    assert result[5] is no_update
    assert result[6] is no_update
    assert result[7] is no_update
    assert result[8] is no_update


def test_add_series_from_database_monthly_only_normalizes_to_month_end(monkeypatch, page_modules):
    analyticstool, _ = page_modules

    imported = pd.DataFrame(
        {"Test_TRIndex": [0.01, 0.02, 0.03]},
        index=pd.to_datetime(["1976-06-30", "1976-07-30", "1976-08-30"]),
    )
    imported.index.name = "Date"
    meta = {
        "Test_TRIndex": {
            "starts_daily": False,
            "daily_start_date": None,
        }
    }

    monkeypatch.setattr(
        analyticstool,
        "load_cma_returns_for_benches_with_meta",
        lambda *_args, **_kwargs: (imported.copy(), meta),
    )

    result = analyticstool.add_series_from_database(
        1,
        ["Test_TRIndex"],
        None,
        None,
        [],
        {},
        {},
        [],
        False,
        {},
        {},
    )

    out_json = result[0]
    out_periodicity = result[1]
    out_default_periodicity = result[3]

    out_df = pd.read_json(StringIO(_raw_json_value(out_json)), orient="split")
    out_df.index = pd.to_datetime(out_df.index)

    assert out_periodicity == "monthly"
    assert out_default_periodicity == "monthly"
    assert out_df.index.is_month_end.all()
    assert pd.Timestamp("1976-07-30") not in out_df.index
    assert pd.Timestamp("1976-07-31") in out_df.index


def test_update_factor_series_select_includes_unselected_series(page_modules, raw_json):
    analyticstool, _ = page_modules

    options, value, conditional_options, conditional_value = analyticstool.update_factor_series_select(
        {"tab": "factor_analysis"},
        None,
        raw_json,
        ["Asset_C", "Asset_A"],
        [],
        [],
        None,
        None,
        None,
    )

    ordered_values = [opt["value"] for opt in options]
    assert ordered_values[:2] == ["raw::Asset_C", "raw::Asset_A"]
    assert set(ordered_values) == {"raw::Asset_A", "raw::Asset_B", "raw::Asset_C", "raw::Asset_D"}
    assert value == "raw::Asset_C"
    assert conditional_options == options
    assert conditional_value == value


def test_update_factor_series_select_includes_saved_and_session_definitions(page_modules, raw_json):
    analyticstool, _ = page_modules

    options, _value, conditional_options, _conditional_value = analyticstool.update_factor_series_select(
        None,
        {"tab": "conditional_returns"},
        raw_json,
        ["Asset_A"],
        [{"FactorName": "SavedFactor"}],
        [{"FactorName": "SessionFactor"}],
        None,
        None,
        None,
    )

    option_map = {opt["value"]: opt["label"] for opt in options}
    assert "def::SavedFactor" in option_map
    assert "def::SessionFactor" in option_map
    assert conditional_options == options
    assert option_map["def::SavedFactor"].startswith("[DB]")
    assert option_map["def::SessionFactor"].startswith("[Session]")


def test_definition_modal_copy_uses_database_session_language(page_modules):
    analyticstool, _ = page_modules

    factor_modal = _find_component_by_id(analyticstool.layout, "at-factor-def-modal")
    factor_select = _find_component_by_id(factor_modal, "at-factor-def-select")
    factor_save_local = _find_component_by_id(factor_modal, "at-factor-def-save-local-btn")
    factor_save_db = _find_component_by_id(factor_modal, "at-factor-def-save-db-btn")
    assert getattr(factor_select, "label", None) == "Database/Session factors"
    assert "Save to session" in " ".join(_collect_component_text(factor_save_local))
    assert "Save to database" in " ".join(_collect_component_text(factor_save_db))

    regime_modal = _find_component_by_id(analyticstool.layout, "at-regime-def-modal")
    regime_select = _find_component_by_id(regime_modal, "at-regime-def-select")
    regime_save_local = _find_component_by_id(regime_modal, "at-regime-def-save-local-btn")
    regime_save_db = _find_component_by_id(regime_modal, "at-regime-def-save-db-btn")
    assert getattr(regime_select, "label", None) == "Database/Session regimes"
    assert "Save to session" in " ".join(_collect_component_text(regime_save_local))
    assert "Save to database" in " ".join(_collect_component_text(regime_save_db))


def test_reset_factor_draft_from_new_button_and_clear(monkeypatch, page_modules):
    analyticstool, _ = page_modules

    monkeypatch.setattr(analyticstool, "callback_context", type("Ctx", (), {"triggered_id": "at-factor-def-new-btn"})())
    select_data, draft, select_value, msg, color, hide = analyticstool.at_factor_definition_selection(
        [], [], "db::DBFactor", 1, None,
    )
    assert draft["DraftMode"] == "new"
    assert select_value is None
    assert color == "blue"
    assert hide is False
    assert "New session factor draft started." in msg

    monkeypatch.setattr(analyticstool, "callback_context", type("Ctx", (), {"triggered_id": "at-factor-def-select"})())
    select_data2, draft2, select_value2, msg2, color2, hide2 = analyticstool.at_factor_definition_selection(
        [], [], None, 0, None,
    )
    assert draft2["DraftMode"] == "new"
    assert select_value2 is no_update
    assert color2 == "blue"
    assert hide2 is False
    assert "New session factor draft started." in msg2


def test_use_factor_promotes_edited_db_draft_to_session(monkeypatch, page_modules):
    analyticstool, _ = page_modules
    db_def = _db_factor_definition(name="DBFactor")
    draft = analyticstool._definition_to_draft(db_def, "db")
    draft["DraftMode"] = "db"
    draft["FactorName"] = "SessionFactor"
    draft["sync_origin"] = "form"

    monkeypatch.setattr(analyticstool, "callback_context", type("Ctx", (), {"triggered_id": "at-factor-def-use-btn"})())
    out = analyticstool.at_factor_definition_actions(
        None, None, None, None, 1, None,
        True, draft, "daily", None,
        [], [db_def], True,
        {"role": "Admin", "username": "tester"},
    )

    local_rows = out[1]
    assert isinstance(local_rows, list)
    assert any(str(item.get("FactorName")) == "SessionFactor" for item in local_rows)
    assert out[5] == "def::SessionFactor"
    assert "Session factor selected for analysis." in out[7]


def test_use_factor_keeps_db_selection_when_unchanged(monkeypatch, page_modules):
    analyticstool, _ = page_modules
    db_def = _db_factor_definition(name="DBFactor")
    draft = analyticstool._definition_to_draft(db_def, "db")
    draft["DraftMode"] = "db"

    monkeypatch.setattr(analyticstool, "callback_context", type("Ctx", (), {"triggered_id": "at-factor-def-use-btn"})())
    out = analyticstool.at_factor_definition_actions(
        None, None, None, None, 1, None,
        True, draft, "daily", None,
        [], [db_def], True,
        {"role": "Admin", "username": "tester"},
    )

    assert out[1] is no_update
    assert out[5] == "def::DBFactor"
    assert "Database factor selected for analysis." in out[7]


def test_use_factor_blocks_db_name_collision_for_edited_db_draft(monkeypatch, page_modules):
    analyticstool, _ = page_modules
    db_def = _db_factor_definition(name="DBFactor", description="original")
    draft = analyticstool._definition_to_draft(db_def, "db")
    draft["DraftMode"] = "db"
    draft["Description"] = "edited"
    draft["sync_origin"] = "form"

    monkeypatch.setattr(analyticstool, "callback_context", type("Ctx", (), {"triggered_id": "at-factor-def-use-btn"})())
    out = analyticstool.at_factor_definition_actions(
        None, None, None, None, 1, None,
        True, draft, "daily", None,
        [], [db_def], True,
        {"role": "Admin", "username": "tester"},
    )

    assert out[1] is no_update
    assert out[7].startswith("Rename the factor to create a session copy")
    assert out[8] == "orange"


def test_sync_factor_definition_form_ignores_form_origin_updates(monkeypatch, page_modules):
    analyticstool, _ = page_modules
    monkeypatch.setattr(analyticstool, "callback_context", type("Ctx", (), {"triggered_id": "at-factor-def-modal-draft-store"})())
    with pytest.raises(PreventUpdate):
        analyticstool.at_factor_definition_form_sync(
            {
                "sync_origin": "form",
                "FactorName": "MyFactor",
                "Description": "line 1\nline 2",
                "LongComponentList": ["ACC1 TRIndex"],
                "LongAggType": 1,
                "LongLag": 0,
                "OutputTransform": 0,
            },
            "MyFactor", "line 1\nline 2", ["ACC1 TRIndex"], [],
            "1", None, 0, "0",
        )


def test_update_factor_definition_draft_preserves_description_text(monkeypatch, page_modules):
    analyticstool, _ = page_modules

    current = analyticstool._default_factor_draft()
    monkeypatch.setattr(analyticstool, "callback_context", type("Ctx", (), {"triggered_id": "at-factor-def-name-input"})())
    result = analyticstool.at_factor_definition_form_sync(
        current,
        "MyFactor",
        "line 1\n",
        ["ACC1 TRIndex"],
        [],
        "1",
        None,
        0,
        "0",
    )
    updated = result[0]

    assert updated["sync_origin"] == "form"
    assert updated["Description"] == "line 1\n"


def test_prepare_factor_analysis_frames_uses_factor_total_basis(monkeypatch, page_modules):
    analyticstool, _ = page_modules
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    dependent_df = pd.DataFrame({"Asset_A": [0.01, 0.02, -0.01, 0.0]}, index=idx)
    factor_df = pd.DataFrame({"Asset_B": [0.03, -0.01, 0.02, 0.01]}, index=idx)
    captured = {}

    def _fake_excess(*args, **kwargs):
        captured["returns_type"] = args[4]
        return dependent_df

    def _fake_working(*args, **kwargs):
        captured["factor_selected"] = args[2]
        return factor_df

    monkeypatch.setattr(analyticstool, "calculate_excess_returns", _fake_excess)
    monkeypatch.setattr(analyticstool, "get_working_returns", _fake_working)
    monkeypatch.setattr(analyticstool, "get_working_returns_by_key", _fake_working)

    dep_out, factor_out = analyticstool._prepare_factor_analysis_frames(
        "raw-json",
        "daily",
        ["Asset_A"],
        "Asset_B",
        "excess",
        {"Asset_A": "Asset_B"},
        {"Asset_B": True},
        {"start": "2024-01-01", "end": "2024-01-31"},
        0,
        {},
        "raw",
    )

    assert captured["returns_type"] == "excess"
    assert captured["factor_selected"] == ("Asset_B",)
    assert list(dep_out.columns) == ["Asset_A"]
    assert factor_out.name == "Asset_B"


def test_compute_selected_returns_preserves_selected_order_for_total_basis(monkeypatch, page_modules):
    analyticstool, _ = page_modules
    idx = pd.date_range("2024-01-01", periods=3, freq="D")

    monkeypatch.setattr(
        analyticstool,
        "get_working_returns",
        lambda *_args, **_kwargs: pd.DataFrame(
            {
                "Bench_X": [0.0, 0.0, 0.0],
                "Asset_B": [0.03, 0.02, 0.01],
                "Asset_A": [0.01, 0.00, -0.01],
            },
            index=idx,
        ),
    )
    monkeypatch.setattr(
        analyticstool,
        "get_working_returns_by_key",
        lambda *_args, **_kwargs: pd.DataFrame(
            {
                "Bench_X": [0.0, 0.0, 0.0],
                "Asset_B": [0.03, 0.02, 0.01],
                "Asset_A": [0.01, 0.00, -0.01],
            },
            index=idx,
        ),
    )
    monkeypatch.setattr(
        analyticstool,
        "calculate_excess_returns",
        lambda *_args, **_kwargs: pytest.fail("total-basis selected returns should not use excess path"),
    )

    out = analyticstool._compute_selected_returns(
        "raw-json-selected-order",
        "daily",
        ["Asset_A", "Asset_B"],
        "total",
        {},
        {},
        {"start": "2024-01-01", "end": "2024-01-31"},
        0,
        {},
    )

    assert list(out.columns) == ["Asset_A", "Asset_B"]
    assert out.iloc[0].to_dict() == {"Asset_A": pytest.approx(0.01), "Asset_B": pytest.approx(0.03)}


def test_update_factor_analysis_renders_one_scatter_per_selected_series(monkeypatch, page_modules):
    analyticstool, _ = page_modules
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    artifacts = analyticstool._FactorArtifacts(
        dependent_df=pd.DataFrame(
            {
                "Asset_A": [0.01, 0.02, 0.0, -0.01, 0.005, 0.008],
                "Asset_B": [0.015, 0.01, -0.005, 0.0, 0.004, 0.006],
            },
            index=idx,
        ),
        factor_raw=pd.Series([0.2, 0.1, -0.1, 0.0, 0.05, 0.08], index=idx, name="Factor_X"),
        factor_display=pd.Series([0.2, 0.1, -0.1, 0.0, 0.05, 0.08], index=idx, name="Factor_X"),
        factor_display_name="Factor_X",
    )
    monkeypatch.setattr(
        analyticstool,
        "_compute_factor_artifacts",
        lambda *_args, **_kwargs: artifacts,
    )

    warning, content = analyticstool.update_factor_analysis(
        {"tab": "factor_analysis"},
        "factor_analysis",
        "scatter",
        "normal",
        "Factor_X",
        5,
        "raw",
        "raw-json",
        "daily",
        ["Asset_A", "Asset_B"],
        "excess",
        {},
        {},
        {"start": "2024-01-01", "end": "2024-01-31"},
        True,
        0,
        {},
        "light",
    )

    assert warning is None
    graphs = [child for child in (content.children or []) if getattr(child, "figure", None) is not None]
    assert len(graphs) == 2
    assert all("Factor Scatter" in graph.figure.layout.title.text for graph in graphs)


def test_update_factor_analysis_renders_raw_detail_grid(monkeypatch, page_modules):
    analyticstool, _ = page_modules
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    artifacts = analyticstool._FactorArtifacts(
        dependent_df=pd.DataFrame({"Asset_A": [0.01, 0.02, -0.01]}, index=idx),
        factor_raw=pd.Series([0.3, 0.1, -0.2], index=idx, name="Factor_X"),
        factor_display=pd.Series([1.0, 0.0, -1.0], index=idx, name="Factor_X"),
        factor_display_name="Factor_X",
    )
    monkeypatch.setattr(analyticstool, "_compute_factor_artifacts", lambda *_args, **_kwargs: artifacts)

    warning, content = analyticstool.update_factor_analysis(
        {"tab": "factor_analysis"},
        "factor_analysis",
        "detail",
        "normal",
        "raw::Factor_X",
        5,
        "zscore",
        "raw-json",
        "daily",
        ["Asset_A"],
        "total",
        {},
        {},
        {"start": "2024-01-01", "end": "2024-01-31"},
        True,
        0,
        {},
        "light",
    )

    assert warning is None
    grid = content.children[1]
    assert grid.rowData[0]["Factor Value"] == pytest.approx(1.0)
    assert grid.rowData[0]["Quantile"] in {"Q1", "Q2", "Q3", "Q4", "Q5"}
    assert grid.rowData[0]["Asset_A"] == pytest.approx(0.01)


def test_factor_quantile_labels_handles_collapsed_buckets(page_modules):
    analyticstool, _ = page_modules
    idx = pd.date_range("2024-01-01", periods=5, freq="D")

    labels, ordered = analyticstool._factor_quantile_labels(
        pd.Series([1.0, 1.0, 1.0, 2.0, 2.0], index=idx),
        5,
    )

    assert labels.index.equals(idx)
    assert all(label in {"Q1", "Q2", None} or pd.isna(label) for label in labels.tolist())
    assert len(ordered) <= 2


def test_prepare_at_qq_reference_series_uses_current_returns_basis_for_raw_reference(monkeypatch, page_modules):
    analyticstool, _ = page_modules
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    captured = {}

    def _fake_excess(*args, **_kwargs):
        captured["selected"] = args[2]
        captured["returns_type"] = args[4]
        return pd.DataFrame({"Asset_B": [0.02, 0.01, -0.01, 0.0]}, index=idx)

    monkeypatch.setattr(analyticstool, "calculate_excess_returns", _fake_excess)

    out = analyticstool._prepare_at_qq_reference_series(
        "raw-json",
        "daily",
        "raw::Asset_B",
        "excess",
        {"Asset_A": "Asset_B"},
        {},
        {"start": "2024-01-01", "end": "2024-01-31"},
        0,
        {},
    )

    assert captured["selected"] == ("Asset_B",)
    assert captured["returns_type"] == "excess"
    assert out.name == "Asset_B"


def test_update_factor_analysis_renders_qq_normal_without_reference(monkeypatch, page_modules):
    analyticstool, _ = page_modules
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    dependent_df = pd.DataFrame(
        {
            "Asset_A": [0.01, 0.02, 0.0, -0.01, 0.005, 0.008],
            "Asset_B": [0.015, 0.01, -0.005, 0.0, 0.004, 0.006],
        },
        index=idx,
    )
    monkeypatch.setattr(
        analyticstool,
        "_prepare_factor_analysis_selected_df",
        lambda *_args, **_kwargs: dependent_df,
    )

    warning, content = analyticstool.update_factor_analysis(
        {"tab": "factor_analysis"},
        "factor_analysis",
        "qq",
        "normal",
        None,
        5,
        "raw",
        "raw-json",
        "daily",
        ["Asset_A", "Asset_B"],
        "excess",
        {},
        {},
        {"start": "2024-01-01", "end": "2024-01-31"},
        True,
        0,
        {},
        "light",
    )

    assert warning is None
    graphs = [child for child in (content.children or []) if getattr(child, "figure", None) is not None]
    assert len(graphs) == 2
    assert all("vs Normal" in graph.figure.layout.title.text for graph in graphs)


def test_update_factor_analysis_renders_qq_reference_with_zscore_axes(monkeypatch, page_modules):
    analyticstool, _ = page_modules
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    dependent_df = pd.DataFrame({"Asset_A": [0.01, 0.03, -0.02, 0.0, 0.02]}, index=idx)
    reference_series = pd.Series([0.015, 0.01, -0.005, -0.01, 0.03], index=idx, name="Ref_X")
    monkeypatch.setattr(
        analyticstool,
        "_prepare_factor_analysis_selected_df",
        lambda *_args, **_kwargs: dependent_df,
    )
    monkeypatch.setattr(
        analyticstool,
        "_prepare_at_qq_reference_series",
        lambda *_args, **_kwargs: reference_series,
    )

    warning, content = analyticstool.update_factor_analysis(
        {"tab": "factor_analysis"},
        "factor_analysis",
        "qq",
        "reference",
        "raw::Ref_X",
        5,
        "raw",
        "raw-json",
        "daily",
        ["Asset_A"],
        "total",
        {},
        {},
        {"start": "2024-01-01", "end": "2024-01-31"},
        True,
        0,
        {},
        "light",
    )

    assert warning is None
    graphs = [child for child in (content.children or []) if getattr(child, "figure", None) is not None]
    assert len(graphs) == 1
    fig = graphs[0].figure
    assert fig.layout.title.text == "Q-Q Plot: Asset_A vs Ref_X"
    assert fig.layout.xaxis.title.text.endswith("(Z-Score)")
    assert fig.layout.yaxis.title.text.endswith("(Z-Score)")


def test_conditional_window_specs_include_1w_for_daily_and_weekly(page_modules):
    analyticstool, _ = page_modules

    daily_labels = [spec["label"] for spec in analyticstool._conditional_window_specs("daily_trading")]
    weekly_labels = [spec["label"] for spec in analyticstool._conditional_window_specs("weekly_friday")]
    monthly_labels = [spec["label"] for spec in analyticstool._conditional_window_specs("monthly")]

    assert daily_labels[0] == "1W"
    assert weekly_labels[0] == "1W"
    assert "1W" not in monthly_labels


def test_compute_conditional_returns_cached_builds_coincident_and_forward_frames(page_modules, raw_json):
    analyticstool, _ = page_modules

    payload = analyticstool._compute_conditional_returns_cached(
        raw_json,
        "daily_trading",
        ("Asset_B", "Asset_C"),
        "total",
        analyticstool._mapping_payload({}),
        analyticstool._mapping_payload({}),
        analyticstool._date_range_payload({"start": "2023-01-02", "end": "2024-03-31"}),
        0,
        analyticstool._mapping_payload({}),
        "raw::Asset_A",
        "zscore",
        "",
        "le",
        0.0,
        "compound",
        1,
        "months",
    )

    assert list(payload.coincident_mean_df.index) == ["1W", "1M", "3M", "6M", "9M", "12M"]
    assert list(payload.coincident_mean_df.columns) == ["Asset_B", "Asset_C"]
    assert set(payload.forward_mean_by_series) == {"Asset_B", "Asset_C"}
    assert list(payload.forward_mean_by_series["Asset_B"].columns) == ["1W", "1M", "3M", "6M", "9M", "12M"]
    assert payload.factor_label.endswith("(Z-Score)")


def test_compute_conditional_core_cached_builds_window_artifacts(page_modules, raw_json):
    analyticstool, _ = page_modules

    core = analyticstool._compute_conditional_core_cached(
        raw_json,
        "daily_trading",
        ("Asset_B", "Asset_C"),
        "total",
        analyticstool._mapping_payload({}),
        analyticstool._mapping_payload({}),
        analyticstool._date_range_payload({"start": "2023-01-02", "end": "2024-03-31"}),
        0,
        analyticstool._mapping_payload({}),
        "raw::Asset_A",
        "zscore",
        "",
        "le",
        0.0,
        "compound",
        1,
        "months",
    )

    assert core.window_labels == ("1W", "1M", "3M", "6M", "9M", "12M")
    assert core.factor_label.endswith("(Z-Score)")
    assert len(core.anchor_index) > 0
    assert "1W" in core.factor_windows
    assert "1W" in core.qualified_masks
    assert {"Asset_B", "Asset_C"}.issubset(core.coincident_series_windows["1W"])
    assert {"1W", "1M"}.issubset(core.forward_series_windows["Asset_B"])
    assert core.forward_row_count > core.coincident_row_count > 0


def test_compute_conditional_returns_cached_builds_detail_frames_when_requested(page_modules, raw_json):
    analyticstool, _ = page_modules

    payload = analyticstool._compute_conditional_returns_cached(
        raw_json,
        "daily_trading",
        ("Asset_B", "Asset_C"),
        "total",
        analyticstool._mapping_payload({}),
        analyticstool._mapping_payload({}),
        analyticstool._date_range_payload({"start": "2023-01-02", "end": "2024-03-31"}),
        0,
        analyticstool._mapping_payload({}),
        "raw::Asset_A",
        "zscore",
        "",
        "le",
        0.0,
        "compound",
        1,
        "months",
        True,
    )

    assert list(payload.coincident_detail_df.columns[:4]) == ["Lookback", "End Date", "Factor Value", "Condition Met"]
    assert list(payload.forward_detail_df.columns[:5]) == ["Lookback", "Forward Period", "End Date", "Factor Value", "Condition Met"]
    assert {"Asset_B", "Asset_C"}.issubset(payload.coincident_detail_df.columns)
    assert {"Asset_B", "Asset_C"}.issubset(payload.forward_detail_df.columns)
    assert payload.coincident_row_count > 0
    assert payload.forward_row_count > payload.coincident_row_count
    first_forward = payload.forward_detail_df[["Lookback", "Forward Period"]].drop_duplicates().iloc[0].to_dict()
    assert first_forward == {"Lookback": "1W", "Forward Period": "1W"}


def test_update_conditional_returns_skips_when_target_already_rendered(monkeypatch, page_modules, raw_json):
    analyticstool, _ = page_modules
    signature = analyticstool._conditional_tab_signature(
        raw_json,
        "daily_trading",
        ("Asset_B", "Asset_C"),
        "total",
        {},
        {},
        {"start": "2023-01-02", "end": "2024-03-31"},
        0,
        {},
        "summary",
        "forward",
        "le",
        0.0,
        "compound",
        1,
        "months",
        "raw::Asset_A",
        "zscore",
        "",
    )
    monkeypatch.setattr(analyticstool, "_compute_conditional_returns_cached", lambda *_args, **_kwargs: pytest.fail("should skip unchanged revisit"))

    with pytest.raises(PreventUpdate):
        analyticstool.update_conditional_returns(
            {"tab": "conditional_returns"},
            signature,
            "conditional_returns",
            None,
            None,
            "raw::Asset_A",
            "zscore",
            raw_json,
            "daily_trading",
            ["Asset_B", "Asset_C"],
            "total",
            {},
            {},
            {"start": "2023-01-02", "end": "2024-03-31"},
            True,
            0,
            {},
            "summary",
            "forward",
            "le",
            0.0,
            "compound",
            1,
            "months",
            signature,
        )


def test_update_conditional_returns_renders_with_decorator_argument_order(monkeypatch, page_modules, raw_json):
    analyticstool, _ = page_modules
    signature = "conditional-signature"
    mean_df = pd.DataFrame({"Asset_B": [0.12]}, index=["1M"])
    count_df = pd.DataFrame({"Asset_B": [8]}, index=["1M"])

    monkeypatch.setattr(
        analyticstool,
        "_compute_conditional_returns_cached",
        lambda *_args, **_kwargs: analyticstool._ConditionalReturnsPayload(
            factor_label="SPX",
            factor_display_name="SPX",
            coincident_mean_df=mean_df,
            coincident_count_df=count_df,
            forward_mean_by_series={"Asset_B": mean_df},
            forward_count_by_series={"Asset_B": count_df},
            coincident_detail_df=pd.DataFrame(),
            forward_detail_df=pd.DataFrame(),
            coincident_row_count=0,
            forward_row_count=0,
        ),
    )

    warning, container, rendered_key = analyticstool.update_conditional_returns(
        {"tab": "conditional_returns"},
        signature,
        "conditional_returns",
        None,
        None,
        "raw::Asset_A",
        "raw",
        raw_json,
        "daily_trading",
        ["Asset_B"],
        "total",
        {},
        {},
        {"start": "2023-01-02", "end": "2024-03-31"},
        True,
        0,
        {},
        "summary",
        "coincident",
        "le",
        0.0,
        "compound",
        1,
        "months",
        None,
    )

    assert warning is None
    assert container is not None
    assert rendered_key == signature


def test_target_key_render_callbacks_keep_decorator_and_signature_order(page_modules):
    analyticstool, _ = page_modules
    advanced_page_text = _advanced_analyticstool_page_text()
    main_page_text = _main_analyticstool_page_text()

    conditional_block = _callback_block(advanced_page_text, "update_conditional_returns")
    conditional_params = list(inspect.signature(analyticstool.update_conditional_returns).parameters)
    assert 'Input("at-conditional-tab-trigger-store", "data")' in conditional_block
    assert 'Input("at-conditional-returns-target-key-store", "data")' in conditional_block
    assert 'State("at-main-tabs", "value")' in conditional_block
    assert conditional_params[:3] == ["trigger_payload", "target_key", "active_tab"]

    correlogram_block = _callback_block(main_page_text, "update_correlogram")
    correlogram_params = list(inspect.signature(analyticstool.update_correlogram).parameters)
    assert 'Input("at-correlogram-target-key-store", "data")' in correlogram_block
    assert 'State("at-main-tabs", "value")' in correlogram_block
    assert correlogram_params[:2] == ["target_key", "active_tab"]


def test_control_conditional_returns_loading_display(page_modules):
    analyticstool, _ = page_modules
    assert analyticstool.control_conditional_returns_loading_display("conditional_returns", False, False, None, None) == "show"
    assert analyticstool.control_conditional_returns_loading_display("conditional_returns", True, True, "sig", None) == "show"
    assert analyticstool.control_conditional_returns_loading_display("conditional_returns", True, True, "sig", "sig") == "hide"
    assert analyticstool.control_conditional_returns_loading_display("statistics", True, True, "sig", None) == "hide"


def test_download_excel_includes_factor_analysis_sheets(monkeypatch, page_modules):
    analyticstool, _ = page_modules
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    returns_df = pd.DataFrame({"Asset_A": [0.01, 0.0, -0.01, 0.02, 0.005]}, index=idx)
    returns_df.index.name = "Date"

    monkeypatch.setattr(analyticstool, "calculate_excess_returns", lambda *_args, **_kwargs: returns_df.copy())
    monkeypatch.setattr(analyticstool, "get_working_returns", lambda *_args, **_kwargs: returns_df.copy())
    monkeypatch.setattr(analyticstool, "get_working_returns_by_key", lambda *_args, **_kwargs: returns_df.copy())
    monkeypatch.setattr(
        analyticstool,
        "calculate_statistics_cached",
        lambda *_args, **_kwargs: [{"Series": "Asset_A", "Cumulative Return": 0.1}],
    )
    monkeypatch.setattr(analyticstool, "generate_correlogram_cached", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(analyticstool, "calculate_rolling_returns", lambda *_args, **_kwargs: returns_df.copy())
    monkeypatch.setattr(
        analyticstool,
        "calculate_calendar_year_returns",
        lambda *_args, **_kwargs: pd.DataFrame({"Asset_A": [0.1]}, index=[2024]),
    )
    monkeypatch.setattr(analyticstool, "calculate_growth_of_dollar", lambda *_args, **_kwargs: returns_df.copy())
    monkeypatch.setattr(analyticstool, "calculate_drawdown", lambda *_args, **_kwargs: returns_df.copy())
    monkeypatch.setattr(
        analyticstool,
        "_compute_factor_artifacts",
        lambda *_args, **_kwargs: analyticstool._FactorArtifacts(
            dependent_df=returns_df.copy(),
            factor_raw=pd.Series([0.2, 0.1, 0.0, -0.1, 0.05], index=idx, name="Factor_X"),
            factor_display=pd.Series([0.2, 0.1, 0.0, -0.1, 0.05], index=idx, name="Factor_X"),
            factor_display_name="Factor_X",
        ),
    )
    monkeypatch.setattr(
        analyticstool,
        "_build_factor_box_summary_rows",
        lambda *_args, **_kwargs: [{"Factor": "Factor_X", "Series": "Asset_A", "Quantile": "Q1", "Observations": 5}],
    )
    monkeypatch.setattr(
        analyticstool,
        "_build_factor_scatter_summary_rows",
        lambda *_args, **_kwargs: [{"Factor": "Factor_X", "Series": "Asset_A", "Observations": 5, "Slope": 1.1}],
    )
    regime_payload = analyticstool._RegimeAnalysisPayload(
        definition={"RegimeName": "SavedRegime"},
        diagnostics={"method_type": 2, "num_regimes": 3, "observations": 5, "warning": None},
        unresolved=(),
        settings_df=pd.DataFrame([{"RegimeName": "SavedRegime", "Signal Label": "PC1"}]),
        timeline_df=pd.DataFrame({"Date": idx, "Regime": [1, 1, 2, 2, 3]}),
        stats_df=pd.DataFrame(
            [{"Regime": 1, "Series": "Asset_A", "Observations": 2, "Mean Return": 0.01}]
        ),
        transition_df=pd.DataFrame(
            [[0.5, 0.5], [0.2, 0.8]],
            index=pd.Index([1, 2], name="From Regime"),
            columns=[1, 2],
        ),
        duration_df=pd.DataFrame([{"Regime": 1, "Runs": 1, "Current Run Length": 2}]),
        detail_df=pd.DataFrame({"Date": idx, "Regime": [1, 1, 2, 2, 3], "Regime Signal": [0.1, 0.2, 0.0, -0.1, 0.3], "Asset_A": returns_df["Asset_A"].tolist()}),
        signal_label="PC1",
    )
    monkeypatch.setattr(
        analyticstool,
        "_build_regime_analysis_payload",
        lambda *_args, **_kwargs: analyticstool._RegimeAnalysisBuildResult("ok", payload=regime_payload),
    )
    monkeypatch.setattr(analyticstool.dcc, "send_bytes", lambda b, filename: {"content": b, "filename": filename})

    payload = analyticstool.download_excel(
        1,
        "raw-json",
        "daily",
        "daily",
        ["Asset_A"],
        "total",
        {},
        {},
        {"start": "2024-01-01", "end": "2024-01-31"},
        "1y",
        "annualized",
        "annual",
        None,
        0,
        {},
        True,
        False,
        63,
        "none",
        "scaled_identity",
        "Factor_X",
        5,
        "raw",
        "forward",
        "le",
        0,
        "compound",
        1,
        "months",
        None,
        None,
        "def::SavedRegime",
        [{"RegimeName": "SavedRegime", "MethodType": 2, "Config": {"num_regimes": 3}}],
        [],
        None,
    )

    xl = pd.ExcelFile(BytesIO(payload["content"]))
    assert "Factor Analysis - Box" in xl.sheet_names
    assert "Factor Analysis - Scatter" in xl.sheet_names
    assert "Factor Analysis - Detail" in xl.sheet_names
    assert "Conditional Coincident" in xl.sheet_names
    assert "Conditional Forward" in xl.sheet_names
    assert "Cond Coincident Detail" in xl.sheet_names
    assert "Cond Forward Detail" in xl.sheet_names
    assert "Regime - Settings" in xl.sheet_names
    assert "Regime - Statistics" in xl.sheet_names
    assert "Regime - Detail" in xl.sheet_names
    assert "Regime - Transition" in xl.sheet_names
    assert "Regime - Duration" in xl.sheet_names
    assert "Regime - Conditioned" not in xl.sheet_names
    regime_sheet_positions = {name: xl.sheet_names.index(name) for name in xl.sheet_names if name.startswith("Regime - ")}
    assert regime_sheet_positions["Regime - Settings"] < regime_sheet_positions["Regime - Statistics"]
    assert regime_sheet_positions["Regime - Statistics"] < regime_sheet_positions["Regime - Detail"]
    assert regime_sheet_positions["Regime - Detail"] < regime_sheet_positions["Regime - Transition"]
    assert regime_sheet_positions["Regime - Transition"] < regime_sheet_positions["Regime - Duration"]


def test_download_excel_falls_back_to_sample_matrices_on_shrinkage_error(monkeypatch, page_modules):
    analyticstool, _ = page_modules
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    returns_df = pd.DataFrame(
        {
            "Asset_A": [0.01, 0.0, -0.01, 0.02, 0.005],
            "Asset_B": [0.0, 0.01, 0.0, -0.005, 0.002],
        },
        index=idx,
    )
    returns_df.index.name = "Date"

    monkeypatch.setattr(analyticstool, "calculate_excess_returns", lambda *_args, **_kwargs: returns_df.copy())
    monkeypatch.setattr(analyticstool, "get_working_returns", lambda *_args, **_kwargs: returns_df.copy())
    monkeypatch.setattr(analyticstool, "get_working_returns_by_key", lambda *_args, **_kwargs: returns_df.copy())
    monkeypatch.setattr(
        analyticstool,
        "calculate_statistics_cached",
        lambda *_args, **_kwargs: [{"Series": "Asset_A", "Cumulative Return": 0.1}],
    )
    monkeypatch.setattr(analyticstool, "calculate_rolling_returns", lambda *_args, **_kwargs: returns_df.copy())
    monkeypatch.setattr(
        analyticstool,
        "calculate_calendar_year_returns",
        lambda *_args, **_kwargs: pd.DataFrame({"Asset_A": [0.1], "Asset_B": [0.2]}, index=[2024]),
    )
    monkeypatch.setattr(analyticstool, "calculate_growth_of_dollar", lambda *_args, **_kwargs: returns_df.copy())
    monkeypatch.setattr(analyticstool, "calculate_drawdown", lambda *_args, **_kwargs: returns_df.copy())
    monkeypatch.setattr(
        analyticstool,
        "generate_correlogram_cached",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("Insufficient overlapping observations for shrinkage covariance estimate.")
        ),
    )
    monkeypatch.setattr(analyticstool.dcc, "send_bytes", lambda b, filename: {"content": b, "filename": filename})

    payload = analyticstool.download_excel(
        1,
        "raw-json",
        "daily",
        "daily",
        ["Asset_A", "Asset_B"],
        "total",
        {},
        {},
        {"start": "2024-01-01", "end": "2024-01-31"},
        "1y",
        "annualized",
        "annual",
        None,
        0,
        {},
        True,
        False,
        63,
        "ledoit_wolf",
        "constant_correlation",
        None,
        5,
        "raw",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )

    xl = pd.ExcelFile(BytesIO(payload["content"]))
    corr_df = xl.parse("Correlation", index_col=0)
    cov_df = xl.parse("Covariance", index_col=0)

    assert corr_df.loc["Asset_A", "Asset_A"] == pytest.approx(1.0)
    assert cov_df.loc["Asset_A", "Asset_A"] == pytest.approx(returns_df.cov().loc["Asset_A", "Asset_A"])


def test_update_regime_definition_select_includes_saved_and_session(page_modules):
    analyticstool, _ = page_modules

    options, value = analyticstool.at_update_regime_definition_analysis_select_options(
        [{"RegimeName": "SavedRegime"}],
        [{"RegimeName": "SessionRegime"}],
        None,
        None,
    )

    option_map = {opt["value"]: opt["label"] for opt in options}
    assert "def::SavedRegime" in option_map
    assert "def::SessionRegime" in option_map
    assert option_map["def::SavedRegime"].startswith("[DB]")
    assert option_map["def::SessionRegime"].startswith("[Session]")
    assert value == "def::SavedRegime"


def test_reset_regime_draft_from_new_button_and_clear(monkeypatch, page_modules):
    analyticstool, _ = page_modules

    monkeypatch.setattr(analyticstool, "callback_context", type("Ctx", (), {"triggered_id": "at-regime-def-new-btn"})())
    select_data, draft, select_value, msg, color, hide = analyticstool.at_regime_definition_selection(
        [], [], "db::DBRegime", 1, None,
    )
    assert draft["DraftMode"] == "new"
    assert select_value is None
    assert color == "blue"
    assert hide is False
    assert "New session regime draft started." in msg

    monkeypatch.setattr(analyticstool, "callback_context", type("Ctx", (), {"triggered_id": "at-regime-def-select"})())
    select_data2, draft2, select_value2, msg2, color2, hide2 = analyticstool.at_regime_definition_selection(
        [], [], None, 0, None,
    )
    assert draft2["DraftMode"] == "new"
    assert select_value2 is no_update
    assert color2 == "blue"
    assert hide2 is False
    assert "New session regime draft started." in msg2


def test_use_regime_promotes_edited_db_draft_to_session(monkeypatch, page_modules):
    analyticstool, _ = page_modules
    db_def = _db_regime_definition(name="DBRegime")
    draft = analyticstool._regime_definition_to_draft(db_def, "db")
    draft["DraftMode"] = "db"
    draft["RegimeName"] = "SessionRegime"
    draft["sync_origin"] = "form"

    monkeypatch.setattr(analyticstool, "callback_context", type("Ctx", (), {"triggered_id": "at-regime-def-use-btn"})())
    out = analyticstool.at_regime_definition_actions(
        None, None, None, None, 1, None,
        True, draft, None, "daily", None, None,
        [], [db_def], True,
        {"role": "Admin", "username": "tester"},
    )

    local_rows = out[2]
    assert isinstance(local_rows, list)
    assert any(str(item.get("RegimeName")) == "SessionRegime" for item in local_rows)
    assert out[6] == "def::SessionRegime"
    assert "Session regime selected for analysis." in out[8]


def test_use_regime_keeps_db_selection_when_unchanged(monkeypatch, page_modules):
    analyticstool, _ = page_modules
    db_def = _db_regime_definition(name="DBRegime")
    draft = analyticstool._regime_definition_to_draft(db_def, "db")
    draft["DraftMode"] = "db"

    monkeypatch.setattr(analyticstool, "callback_context", type("Ctx", (), {"triggered_id": "at-regime-def-use-btn"})())
    out = analyticstool.at_regime_definition_actions(
        None, None, None, None, 1, None,
        True, draft, None, "daily", None, None,
        [], [db_def], True,
        {"role": "Admin", "username": "tester"},
    )

    assert out[2] is no_update
    assert out[6] == "def::DBRegime"
    assert "Database regime selected for analysis." in out[8]


def test_use_regime_blocks_db_name_collision_for_edited_db_draft(monkeypatch, page_modules):
    analyticstool, _ = page_modules
    db_def = _db_regime_definition(name="DBRegime", description="original")
    draft = analyticstool._regime_definition_to_draft(db_def, "db")
    draft["DraftMode"] = "db"
    draft["Description"] = "edited"
    draft["sync_origin"] = "form"

    monkeypatch.setattr(analyticstool, "callback_context", type("Ctx", (), {"triggered_id": "at-regime-def-use-btn"})())
    out = analyticstool.at_regime_definition_actions(
        None, None, None, None, 1, None,
        True, draft, None, "daily", None, None,
        [], [db_def], True,
        {"role": "Admin", "username": "tester"},
    )

    assert out[2] is no_update
    assert out[8].startswith("Rename the regime to create a session copy")
    assert out[9] == "orange"


def test_lazy_load_factor_and_regime_definitions_on_first_tab_open(monkeypatch, page_modules):
    analyticstool, _ = page_modules
    monkeypatch.setattr(analyticstool, "factor_tables_available", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(analyticstool, "regime_tables_available", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        analyticstool,
        "load_factor_definitions",
        lambda *_args, **_kwargs: [{"FactorName": "SavedFactor"}],
    )
    monkeypatch.setattr(
        analyticstool,
        "load_regime_definitions",
        lambda *_args, **_kwargs: [{"RegimeName": "SavedRegime"}],
    )

    factor_available, factor_rows, factor_loaded = analyticstool.at_lazy_load_factor_definitions(
        "factor_analysis",
        False,
    )
    regime_available, regime_rows, regime_loaded = analyticstool.at_lazy_load_regime_definitions(
        "regime_analysis",
        False,
    )

    assert factor_available is True
    assert regime_available is True
    assert factor_loaded is True
    assert regime_loaded is True
    assert factor_rows == [{"FactorName": "SavedFactor"}]
    assert regime_rows == [{"RegimeName": "SavedRegime"}]


def test_regime_definition_modal_hides_return_basis_control(page_modules):
    analyticstool, _ = page_modules
    modal = _find_component_by_id(analyticstool.layout, "at-regime-def-modal")
    assert modal is not None
    assert _find_component_by_id(modal, "at-regime-def-return-basis") is None


def test_update_regime_analysis_renders_content(monkeypatch, page_modules):
    analyticstool, _ = page_modules
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    returns_df = pd.DataFrame({"Asset_A": [0.01, -0.005, 0.02, 0.0, 0.01, -0.01]}, index=idx)
    regime_payload = analyticstool._RegimeAnalysisPayload(
        definition={"RegimeName": "SavedRegime"},
        diagnostics={"method_type": 3, "num_regimes": 3, "observations": 6, "warning": None},
        unresolved=(),
        settings_df=pd.DataFrame([{"RegimeName": "SavedRegime", "Signal Label": "Asset_A", "Signal Return Basis": "total"}]),
        timeline_df=pd.DataFrame({"Date": idx, "Regime": [1, 1, 2, 2, 3, 3]}),
        stats_df=pd.DataFrame([{"Regime": 1, "Series": "Asset_A", "Observations": 2, "Mean Return": 0.01}]),
        transition_df=pd.DataFrame([[0.5, 0.5], [0.2, 0.8]], index=pd.Index([1, 2], name="From Regime"), columns=[1, 2]),
        duration_df=pd.DataFrame([{"Regime": 1, "Runs": 1, "Current Run Length": 2}]),
        detail_df=pd.DataFrame({"Date": idx, "Regime": [1, 1, 2, 2, 3, 3], "Regime Signal": returns_df["Asset_A"].tolist(), "Asset_A": returns_df["Asset_A"].tolist()}),
        signal_label="Asset_A",
    )
    monkeypatch.setattr(
        analyticstool,
        "_build_regime_analysis_payload",
        lambda *_args, **_kwargs: analyticstool._RegimeAnalysisBuildResult("ok", payload=regime_payload),
    )

    warning, content = analyticstool.update_regime_analysis(
        {"tab": "regime_analysis"},
        "regime_analysis",
        "def::SavedRegime",
        "summary",
        "raw-json",
        "daily",
        ["Asset_A"],
        "total",
        {},
        {},
        {"start": "2024-01-01", "end": "2024-01-31"},
        True,
        0,
        {},
        "light",
        [{"RegimeName": "SavedRegime", "MethodType": 3, "Config": {"num_regimes": 3}}],
        [],
    )

    assert warning is not None
    assert content is not None
    section_titles = _stack_section_titles(content)
    assert section_titles[0] == "Regime Settings"
    assert section_titles[1] == "Regime Statistics"
    assert section_titles[2].startswith("Regime Timeline:")
    assert section_titles[3] == "Transition Matrix"
    assert section_titles[4] == "Run Durations"
    text_blob = " ".join(_collect_component_text(content)).lower()
    assert "regime settings" in text_blob
    assert "regime statistics" in text_blob
    assert "transition matrix" in text_blob


def test_update_regime_analysis_renders_raw_detail_grid(monkeypatch, page_modules):
    analyticstool, _ = page_modules
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    regime_payload = analyticstool._RegimeAnalysisPayload(
        definition={"RegimeName": "SavedRegime"},
        diagnostics={"method_type": 3, "num_regimes": 3, "observations": 3, "warning": None},
        unresolved=(),
        settings_df=pd.DataFrame([{"RegimeName": "SavedRegime", "Signal Label": "Asset_A", "Signal Return Basis": "total"}]),
        timeline_df=pd.DataFrame({"Date": idx, "Regime": [1, 2, 3]}),
        stats_df=pd.DataFrame(),
        transition_df=pd.DataFrame(),
        duration_df=pd.DataFrame(),
        detail_df=pd.DataFrame({"Date": idx, "Regime": [1, 2, 3], "Regime Signal": [0.1, 0.0, -0.1], "Asset_A": [0.01, 0.02, -0.03]}),
        signal_label="Asset_A",
    )
    monkeypatch.setattr(
        analyticstool,
        "_build_regime_analysis_payload",
        lambda *_args, **_kwargs: analyticstool._RegimeAnalysisBuildResult("ok", payload=regime_payload),
    )

    warning, content = analyticstool.update_regime_analysis(
        {"tab": "regime_analysis"},
        "regime_analysis",
        "def::SavedRegime",
        "detail",
        "raw-json",
        "daily",
        ["Asset_A"],
        "total",
        {},
        {},
        {"start": "2024-01-01", "end": "2024-01-31"},
        True,
        0,
        {},
        "light",
        [{"RegimeName": "SavedRegime", "MethodType": 3, "Config": {"num_regimes": 3}}],
        [],
    )

    assert warning is not None
    grid = content.children[1]
    assert grid.rowData[0]["Regime"] == 1
    assert grid.rowData[0]["Regime Signal"] == pytest.approx(0.1)


def test_help_modal_mentions_factor_analysis(page_modules):
    analyticstool, _ = page_modules
    help_control = _find_component_by_id(analyticstool.layout, "at-menu-help-guide")
    assert help_control is not None

    text_blob = Path("docs/help/analyticstool.md").read_text(encoding="utf-8").lower()
    assert "analytics tool" in text_blob
    assert "factor analysis" in text_blob
    assert "regime analysis" in text_blob


def test_analytics_ui_blocker_release_uses_db_error_alert():
    text_blob = Path("pages/analyticstool.py").read_text(encoding="utf-8")
    assert 'Input("at-db-add-error-alert", "hide")' in text_blob


def test_analytics_session_actions_use_shared_workspace_helpers():
    text_blob = Path("pages/analyticstool.py").read_text(encoding="utf-8")
    assert 'ClientsideFunction(namespace="dashmat_callbacks", function_name="saveWorkspaceSession")' in text_blob
    assert 'ClientsideFunction(namespace="dashmat_callbacks", function_name="loadWorkspaceSessionDialog")' in text_blob
    assert 'ClientsideFunction(namespace="dashmat_callbacks", function_name="loadWorkspaceSession")' in text_blob
    assert 'ClientsideFunction(namespace="dashmat_callbacks", function_name="clearWorkspaceSession")' in text_blob
    assert '#load-session-upload input[type="file"]' not in text_blob
    assert "sessionStorage.clear()" not in text_blob
