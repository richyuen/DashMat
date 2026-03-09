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


def test_po_restore_state_keeps_empty_selection_when_nothing_is_stored(page_modules, raw_json):
    _, portopt = page_modules

    restored = portopt.po_restore_state(
        _raw_meta(raw_json),
        "daily_trading",
        [],
        None,
    )

    assert restored[3] == []


def test_po_open_modal_auto_opens_on_page_load_with_no_selection(monkeypatch, page_modules, raw_json, sample_returns_df):
    _, portopt = page_modules
    monkeypatch.setattr(portopt, "callback_context", type("Ctx", (), {"triggered_id": "po-page-load-trigger"})())

    result = portopt.po_open_modal(
        None,
        1,
        "/portopt",
        _raw_meta(raw_json),
        [],
        {},
        {},
        {},
        [],
        {},
        {},
        {},
        {},
        [],
        False,
    )

    assert result[0] is True
    assert result[1] == list(sample_returns_df.columns)
    assert result[11] is True
    assert result[12] is True


def test_po_open_modal_ignores_po_only_series_on_revisit(monkeypatch, page_modules, raw_json):
    _, portopt = page_modules
    monkeypatch.setattr(portopt, "callback_context", type("Ctx", (), {"triggered_id": "po-page-load-trigger"})())

    result = portopt.po_open_modal(
        None,
        1,
        "/portopt",
        _raw_meta(raw_json),
        ["Asset_A"],
        {},
        {},
        {},
        ["Asset_A", "Asset_B", "Asset_D"],
        {},
        {},
        {},
        {},
        {"Asset_C": {"origin_page": "portopt", "origin_result": "Asset_C", "series_type": "portfolio"}},
        True,
    )

    assert result[0] is no_update
    assert result[11] is True
    assert result[12] is False


def test_po_open_modal_auto_adds_only_generic_new_columns_on_page_load(monkeypatch, page_modules, raw_json):
    _, portopt = page_modules
    monkeypatch.setattr(portopt, "callback_context", type("Ctx", (), {"triggered_id": "po-page-load-trigger"})())

    result = portopt.po_open_modal(
        None,
        1,
        "/portopt",
        _raw_meta(raw_json),
        ["Asset_A"],
        {},
        {},
        {},
        ["Asset_A", "Asset_B"],
        {},
        {},
        {},
        {},
        {"Asset_C": {"origin_page": "regression", "origin_result": "Asset_C", "series_type": "predicted"}},
        True,
    )

    assert result[0] is True
    assert result[1] == ["Asset_A", "Asset_D"]
    assert result[11] is True
    assert result[12] is True


def test_po_open_modal_does_not_auto_select_saved_series_on_first_visit(monkeypatch, page_modules, raw_json):
    _, portopt = page_modules
    monkeypatch.setattr(portopt, "callback_context", type("Ctx", (), {"triggered_id": "po-page-load-trigger"})())

    result = portopt.po_open_modal(
        None,
        1,
        "/portopt",
        _raw_meta(raw_json),
        [],
        {},
        {},
        {},
        [],
        {},
        {},
        {},
        {},
        {
            "Asset_C": {"origin_page": "portopt", "origin_result": "Asset_C", "series_type": "portfolio"},
            "Asset_D": {"origin_page": "regression", "origin_result": "Asset_D", "series_type": "predicted"},
        },
        False,
    )

    assert result[0] is True
    assert result[1] == ["Asset_A", "Asset_B"]
    assert result[11] is True
    assert result[12] is True


def test_po_open_modal_skips_auto_open_when_only_saved_series_exist(monkeypatch, page_modules):
    _, portopt = page_modules
    monkeypatch.setattr(portopt, "callback_context", type("Ctx", (), {"triggered_id": "po-page-load-trigger"})())
    raw_df = pd.DataFrame({"SavedPort": [0.01, 0.02]}, index=pd.date_range("2024-01-01", periods=2, freq="B"))
    raw_df.index.name = "Date"

    result = portopt.po_open_modal(
        None,
        1,
        "/portopt",
        _raw_meta(df_to_json(raw_df)),
        [],
        {},
        {},
        {},
        [],
        {},
        {},
        {},
        {},
        {"SavedPort": {"origin_page": "portopt", "origin_result": "SavedPort", "series_type": "portfolio"}},
        False,
    )

    assert result[0] is no_update
    assert result[11] is True
    assert result[12] is False


def test_po_layout_starts_with_welcome_and_main_hidden(page_modules):
    _, portopt = page_modules

    welcome = _find_component_by_id(portopt.layout, "po-welcome-screen")
    main = _find_component_by_id(portopt.layout, "po-main-container")

    assert getattr(welcome, "style", {})["display"] == "none"
    assert getattr(main, "style", {})["display"] == "none"


def test_po_bootstrap_keeps_single_page_load_interval_and_no_dead_results_sync():
    page_text = Path("pages/portopt.py").read_text(encoding="utf-8")
    assert page_text.count('dcc.Interval(id="po-page-load-trigger"') == 1
    assert "def po_sync_results_with_raw_data" not in page_text
    assert 'dcc.Store(id="po-restore-complete-store", data=False, storage_type="memory")' in page_text
    assert 'Output("po-vis-tabs", "value")' in page_text
    assert 'State("po-active-tab-store", "data")' in page_text
    assert 'Output("po-attribution-tab-loaded-store", "data")' in page_text
    assert 'Output("po-risk-tab-loaded-store", "data")' in page_text
    assert 'Output("po-frontier-tab-loaded-store", "data")' in page_text
    assert page_text.count('Input("po-attribution-tab-loaded-store", "data")') == 2
    assert page_text.count('Input("po-risk-tab-loaded-store", "data")') == 2
    assert page_text.count('Input("po-frontier-tab-loaded-store", "data")') == 2
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
    assert page_text.count('Input("po-initial-tab-render-ready-store", "data")') == 2


def test_po_shell_visibility_no_longer_depends_on_page_load_trigger():
    page_text = Path("pages/portopt.py").read_text(encoding="utf-8")
    visibility_block = page_text.split('Output("po-secondary-restore-ready-store", "data")', 1)[0]
    assert 'Output("po-welcome-screen", "style")' in visibility_block
    assert 'Output("po-main-container", "style")' in visibility_block
    assert 'Input("dashmat-raw-data-store", "data")' in visibility_block
    assert 'Input("po-page-load-trigger", "n_intervals")' not in visibility_block


def test_po_toggle_ui_elements_uses_restore_complete_store():
    page_text = Path("pages/portopt.py").read_text(encoding="utf-8")
    toggle_block = page_text.split("def po_toggle_ui_elements", 1)[0]
    toggle_callback = toggle_block.rsplit("@callback(", 1)[-1]
    assert 'Input("po-restore-complete-store", "data")' in toggle_callback
    assert 'Input("po-secondary-restore-ready-store", "data")' not in toggle_callback


def test_po_restore_complete_store_waits_for_secondary_restore_and_valid_controls():
    page_text = Path("pages/portopt.py").read_text(encoding="utf-8")
    assert 'Output("po-restore-complete-store", "data")' in page_text
    restore_block = page_text.split('Output("po-restore-complete-store", "data")', 1)[1]
    restore_callback = restore_block.split("# ---------------------------------------------------------------------------\n# Restore optimization controls from stores on page load", 1)[0]
    assert 'Input("po-secondary-restore-ready-store", "data")' in restore_callback
    assert 'Input("po-periodicity-select", "value")' in restore_callback
    assert 'Input("po-series-select", "data")' in restore_callback


def test_po_init_date_range_no_longer_depends_on_common_daily_store():
    page_text = Path("pages/portopt.py").read_text(encoding="utf-8")
    init_block = page_text.split("def po_init_date_range", 1)[0]
    init_callback = init_block.rsplit("@callback(", 1)[-1]
    assert 'Input("po-range-candidates-store", "data")' in init_callback
    assert 'Input("po-common-daily-candidates-store", "data")' not in init_callback


def test_po_common_daily_button_uses_shared_clientside_helper():
    page_text = Path("pages/portopt.py").read_text(encoding="utf-8")
    js_text = Path("assets/dashmat_callbacks.js").read_text(encoding="utf-8")
    assert 'ClientsideFunction(namespace="dashmat_callbacks", function_name="commonDailyButtonDisabled")' in page_text
    assert 'Output("po-common-daily-button", "disabled")' in page_text
    assert "function commonDailyButtonDisabled(candidates, commonDailyCandidates, periodicityOptions)" in js_text


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
    monkeypatch.setattr(portopt, "get_working_returns", lambda *_args, **_kwargs: working_df)

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
        True,
        "table",
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

    s1 = pd.Series([0.01, 0.02], index=pd.to_datetime(["2024-01-01", "2024-01-02"]))
    s2 = pd.Series([0.00, 0.01], index=pd.to_datetime(["2024-01-01", "2024-01-02"]))
    results = {
        "P1": {"returns_json": s1.to_json(date_format="iso")},
        "P2": {"returns_json": s2.to_json(date_format="iso")},
    }

    grid = portopt.po_render_statistics(results, "statistics", ["P1", "P2"], None, "daily")

    assert getattr(grid, "columnDefs", [])[0]["field"] == "Statistic"
    assert {c["field"] for c in getattr(grid, "columnDefs", [])[1:]} == {"P1", "P2"}
    row = next(r for r in getattr(grid, "rowData", []) if r["Statistic"] == "Cumulative Return")
    assert row["P1"] == pytest.approx(0.1)
    assert row["P2"] == pytest.approx(0.2)


def test_po_render_returns_builds_returns_grid(page_modules):
    _, portopt = page_modules

    s1 = pd.Series([0.01, 0.02], index=pd.to_datetime(["2024-01-01", "2024-01-02"]))
    results = {"P1": {"returns_json": s1.to_json(date_format="iso")}}

    grid = portopt.po_render_returns(results, "returns", ["P1"])
    assert getattr(grid, "columnDefs", [])[0]["field"] == "Date"
    assert getattr(grid, "columnDefs", [])[1]["field"] == "P1"
    assert getattr(grid, "rowData", [])[0]["Date"] == "2024-01-01"


def test_po_render_statistics_requires_active_tab(page_modules):
    _, portopt = page_modules

    grid = portopt.po_render_statistics(
        {"P1": {"returns_json": pd.Series([0.01], index=pd.to_datetime(["2024-01-01"])).to_json(date_format="iso")}},
        "weight",
        "P1",
        None,
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

    empty = portopt.po_update_portfolio_dropdowns(None, None, None)
    assert empty == ([], None, [], [], True)

    results = {"P1": {"x": 1}, "P2": {"x": 2}}
    options, selected, multi_options, multi_value, delete_disabled = portopt.po_update_portfolio_dropdowns(
        results,
        "P1",
        ["P1"],
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
    assert len(getattr(grid, "rowData", [])) == 3


def test_po_render_rolling_table_mode_returns_grid_with_wide_date_column(monkeypatch, page_modules):
    _, portopt = page_modules
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    display_df = pd.DataFrame({"P1": [0.01, -0.005, 0.002, 0.003]}, index=idx)
    monkeypatch.setattr(portopt, "_po_build_display_series", lambda *_args, **_kwargs: (display_df, ["P1"]))
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


def test_po_render_drawdown_table_mode_returns_grid_with_wide_date_column(monkeypatch, page_modules):
    _, portopt = page_modules
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    display_df = pd.DataFrame({"P1": [0.01, -0.005, 0.002, 0.003]}, index=idx)
    monkeypatch.setattr(portopt, "_po_build_display_series", lambda *_args, **_kwargs: (display_df, ["P1"]))
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
            {}, {}, "ret_cov", [], None,
        )

    # Now force callback path and verify returned error payload.
    result = portopt.po_run_optimization(
        1, "raw", "daily", "daily", ["Asset_A", "Asset_B"], {}, {}, {}, None, 0, {},
        {}, {}, {}, False, 63, "none", "scaled_identity", "MyPortfolio", "full", 252, 21, "periods",
        "risk_parity", "fill_na", "off", {}, [],
        {}, {}, [], 0.05, "maximize_sharpe",
        {}, {}, "ret_cov", [], None,
    )
    status = result[2]
    assert status["status"] == "error"
    assert "No data available" in status["message"]


def test_po_toggle_ui_elements_sets_validation_tooltip(page_modules):
    _, portopt = page_modules

    run_disabled, tooltip, tooltip_disabled, save_disabled, download_disabled = (
        portopt.po_toggle_ui_elements(
            False,
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
            {"P1": {}},
        )
    )

    assert run_disabled is True
    assert tooltip == "Loading controls..."
    assert tooltip_disabled is False
    assert save_disabled is False
    assert download_disabled is False


def test_po_toggle_ui_elements_waits_for_restore_completion(page_modules):
    _, portopt = page_modules

    run_disabled, tooltip, tooltip_disabled, save_disabled, download_disabled = (
        portopt.po_toggle_ui_elements(
            True,
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
        True,
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
        True,
        "table",
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
        True,
        "chart",
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
        True,
        "chart",
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
        "daily",
        None,
        None,
    )

    assert "Warning text" in " ".join(_collect_component_text(warning))
    assert style["display"] == "block"


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

    df_after = pd.read_json(StringIO(new_raw), orient="split")
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
    monkeypatch.setattr(portopt, "get_working_returns", lambda *_args, **_kwargs: working_df)
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
            "config": {"selected_series": ["Asset_A", "Asset_B"], "model": "risk_parity"},
        }
    }

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
        lambda **_kwargs: {
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
    assert settings_map["Decay Input"] == pytest.approx(63.0)
    assert settings_map["Decay Mode"] == "halflife_periods"
    assert settings_map["Selected Series"] == "Asset_A, Asset_B"
    assert settings_map["Benchmark Assignments"] == "{}"

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
    assert 'Output("po-series-selection-container", "children")' in page_text
    assert 'ClientsideFunction(namespace="dashmat_callbacks", function_name="releaseBlockerOnSeriesGridReady")' in page_text
    assert 'Input("po-series-selection-grid", "virtualRowData", allow_optional=True)' in page_text
    assert 'ClientsideFunction(namespace="dashmat_callbacks", function_name="portoptInitialSeriesBlocker")' in page_text
    assert 'Input("po-url-location", "pathname")' in page_text
    assert "function portoptInitialSeriesBlocker(pathname, rawMeta, pageVisited, currentSelect)" in js_text


def test_po_series_selection_grid_keeps_blocker_until_virtual_rows(page_modules, raw_json):
    _, portopt = page_modules

    children, _order, blocker = portopt.po_update_series_selectors(
        raw_json,
        ["Asset_A"],
        ["Asset_A", "Asset_B"],
        [],
        {},
        {},
        {},
        {},
        {},
        {},
        {},
    )

    assert blocker is no_update
    assert getattr(children[0], "id", None) == "po-series-selection-grid"


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
