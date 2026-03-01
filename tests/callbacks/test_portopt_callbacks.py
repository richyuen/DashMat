from __future__ import annotations

from io import BytesIO
from io import StringIO
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from dash import no_update
from dash.exceptions import PreventUpdate

from utils.returns import df_to_json
from utils.route_intent import ACTION_OPEN_IMPORT_MODAL, FLOW_DB, build_route_intent


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


def _component_prop(node, prop_name):
    if hasattr(node, prop_name):
        return getattr(node, prop_name)
    to_plotly = getattr(node, "to_plotly_json", None)
    if callable(to_plotly):
        return ((to_plotly().get("props") or {})).get(prop_name)
    props = getattr(node, "props", None)
    if isinstance(props, dict):
        return props.get(prop_name)
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


def test_po_render_attribution_table_returns_grid_data(monkeypatch, page_modules):
    _, portopt = page_modules
    idx = pd.date_range("2024-01-01", periods=60, freq="D")
    working_df = pd.DataFrame({"Asset_A": 0.01, "Asset_B": 0.02}, index=idx)
    working_df.index.name = "Date"
    monkeypatch.setattr(portopt, "_po_get_working_returns", lambda *_args, **_kwargs: working_df)

    results = {
        "P1": {
            "config": {"selected_series": ["Asset_A", "Asset_B"]},
            "window_weights": _sample_window_weights(),
        }
    }

    column_defs, row_data = portopt.po_render_attribution_table(
        "P1",
        results,
        "attribution",
        "table",
        None,
        "raw-json",
        "daily",
        {},
        {},
        None,
        0,
        {},
        ["attribution"],
    )

    assert column_defs[0]["field"] == "Date"
    assert any(c["field"] == "Total" for c in column_defs)
    assert len(row_data) > 0


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

    column_defs, row_data = portopt.po_render_statistics(
        results,
        "statistics",
        ["P1", "P2"],
        None,
        None,
        "daily",
        mounted_tabs=["statistics"],
    )

    assert column_defs[0]["field"] == "Statistic"
    assert {c["field"] for c in column_defs[1:]} == {"P1", "P2"}
    row = next(r for r in row_data if r["Statistic"] == "Cumulative Return")
    assert row["P1"] == pytest.approx(0.1)
    assert row["P2"] == pytest.approx(0.2)


def test_po_render_returns_builds_returns_grid(page_modules):
    _, portopt = page_modules

    s1 = pd.Series([0.01, 0.02], index=pd.to_datetime(["2024-01-01", "2024-01-02"]))
    results = {"P1": {"returns_json": s1.to_json(date_format="iso")}}

    column_defs, row_data = portopt.po_render_returns(results, "returns", ["P1"], None, mounted_tabs=["returns"])
    assert column_defs[0]["field"] == "Date"
    assert column_defs[1]["field"] == "P1"
    assert row_data[0]["Date"] == "2024-01-01"


def test_po_open_db_add_modal_clears_blocker_with_modal_payload(monkeypatch, page_modules):
    _, portopt = page_modules
    expected = (True, [{"value": "IDX_A", "label": "Index A"}], [])
    monkeypatch.setattr(portopt, "compute_open_db_add_modal", lambda *_args, **_kwargs: expected)

    assert portopt.po_open_db_add_modal(1) == (*expected, False, portopt.no_update)


def test_po_resolve_import_modal_request_returns_db_request(page_modules):
    _, portopt = page_modules
    route_intent = build_route_intent("portopt", ACTION_OPEN_IMPORT_MODAL, flow=FLOW_DB)

    request = portopt.po_resolve_import_modal_request(
        "portopt",
        route_intent,
        None,
    )

    assert request == (
        {"flow": FLOW_DB, "token": route_intent["token"]},
        no_update,
        no_update,
        no_update,
    )


def test_po_open_db_add_modal_uses_request_store_token(monkeypatch, page_modules):
    _, portopt = page_modules
    expected = (True, [{"value": "IDX_A", "label": "Index A"}], [])
    monkeypatch.setattr(portopt, "compute_open_db_add_modal", lambda *_args, **_kwargs: expected)
    monkeypatch.setattr(
        portopt,
        "callback_context",
        SimpleNamespace(triggered_id="po-db-add-request-store"),
    )

    assert portopt.po_open_db_add_modal(None, {"flow": FLOW_DB, "token": "tok"}) == (
        *expected,
        False,
        "tok",
    )


def test_po_resolve_import_modal_request_ignores_stale_intent(page_modules):
    _, portopt = page_modules
    route_intent = build_route_intent("portopt", ACTION_OPEN_IMPORT_MODAL, flow=FLOW_DB)
    route_intent["created_at"] = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(seconds=61)).isoformat()

    with pytest.raises(PreventUpdate):
        portopt.po_resolve_import_modal_request(
            "portopt",
            route_intent,
            None,
        )


def test_po_populate_returns_grid_adds_header_tooltips(page_modules):
    _, portopt = page_modules

    rows, cols = portopt.po_populate_returns_grid(
        ["Asset_A"],
        "ret_cov",
        {"Asset_A": 0.03},
        {"Asset_A": 0.15},
    )

    assert rows and rows[0]["Asset"] == "Asset_A"
    col_map = {c["field"]: c for c in cols}
    assert col_map["Asset"].get("headerTooltip")
    assert col_map["Return"].get("headerTooltip")
    assert col_map["Volatility"].get("headerTooltip")


def test_po_populate_matrix_grid_adds_header_tooltips(page_modules):
    _, portopt = page_modules

    cov_store = {
        "Asset_A": {"Asset_A": 0.1, "Asset_B": 0.02},
        "Asset_B": {"Asset_A": 0.02, "Asset_B": 0.3},
    }
    rows, cols = portopt.po_populate_matrix_grid(["Asset_A", "Asset_B"], "ret_cov", cov_store, None)

    assert len(rows) == 2
    col_map = {c["field"]: c for c in cols}
    assert col_map["Asset"].get("headerTooltip")
    assert col_map["Asset_A"].get("headerTooltip")
    assert col_map["Asset_B"].get("headerTooltip")


def test_po_populate_linear_constraints_columns_adds_header_tooltips(page_modules):
    _, portopt = page_modules

    cols = portopt.po_populate_linear_constraints_columns(["Asset_A", "Asset_B"])
    col_map = {c["field"]: c for c in cols}

    assert col_map["Constraint"].get("headerTooltip")
    assert col_map["Min"].get("headerTooltip")
    assert col_map["Max"].get("headerTooltip")
    assert col_map["Asset_A"].get("headerTooltip")
    assert col_map["Asset_B"].get("headerTooltip")


def test_po_ordered_modal_rows_prefers_virtual_row_order(page_modules):
    _, portopt = page_modules

    ordered = portopt._po_ordered_modal_rows(
        [
            {"__orig_series": "Asset_A", "Series": "Asset_A"},
            {"__orig_series": "Asset_B", "Series": "Asset_B"},
            {"__orig_series": "Asset_C", "Series": "Asset_C"},
        ],
        [
            {"__orig_series": "Asset_C"},
            {"__orig_series": "Asset_A"},
            {"__orig_series": "Asset_B"},
        ],
    )

    assert [portopt._po_modal_orig_series(row) for row in ordered] == [
        "Asset_C",
        "Asset_A",
        "Asset_B",
    ]


def test_po_update_series_selectors_adds_header_tooltips_and_grid_tooltip_options(
    monkeypatch, page_modules, raw_json
):
    _, portopt = page_modules
    monkeypatch.setattr(portopt, "get_cmabench_map_for_fofbench", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(portopt, "get_unique_cmabench_values_cached", lambda *_args, **_kwargs: [])

    children, status = portopt.po_update_series_selectors(
        "token",
        raw_json,
        ["Asset_A"],
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
    )

    grid = children[0]
    col_map = {c.get("field"): c for c in getattr(grid, "columnDefs", []) if c.get("field")}
    assert getattr(grid, "getRowId", None) == "params.data.__orig_series"
    assert col_map["Series"].get("headerTooltip")
    assert col_map["Benchmark"].get("headerTooltip")
    assert status["status"] == "rendered"
    assert col_map["CMABench"].get("headerTooltip")
    assert getattr(grid, "dashGridOptions", {}).get("tooltipShowDelay") == 500


def test_po_on_modal_ok_commits_local_series_modal_state(monkeypatch, page_modules, raw_json):
    _, portopt = page_modules
    monkeypatch.setattr(
        portopt,
        "get_cmabench_map_for_fofbench",
        lambda *_args, **_kwargs: {"Asset_A": "DefaultBench"},
    )

    result = portopt.po_on_modal_ok(
        1,
        raw_json,
        {"Asset_B": {"x": 1}, "KeepMe": {"x": 2}},
        [
            {
                "__orig_series": "Asset_A",
                "Series": "Renamed_A",
                "Benchmark": "Asset_B",
                "CMABench": "",
                "LongShort": True,
                "ScaleVol": False,
                "MinWt": 5,
                "MaxWt": 60,
                "ForceMax": False,
                "Delete": False,
            },
            {
                "__orig_series": "Asset_B",
                "Series": "Asset_B",
                "Benchmark": "None",
                "CMABench": "KeepBench",
                "LongShort": False,
                "ScaleVol": True,
                "MinWt": 0,
                "MaxWt": 100,
                "ForceMax": False,
                "Delete": True,
            },
        ],
        [{"__orig_series": "Asset_A", "Series": "Renamed_A"}],
        [{"__orig_series": "Asset_B"}, {"__orig_series": "Asset_A"}],
    )

    assert result[0] == ["Renamed_A"]
    assert result[1] == {"Renamed_A": "None"}
    assert result[2] == {"Renamed_A": "DefaultBench"}
    assert result[3] == {"Renamed_A": True}
    assert result[4] == ["Renamed_A"]
    assert result[10] == {"Renamed_A": False}
    assert result[11] == {"Renamed_A": 5.0}
    assert result[12] == {"Renamed_A": 60.0}
    assert result[13] == {"Renamed_A": False}
    assert result[14] == {"KeepMe": {"x": 2}}
    assert result[15] is False

    updated_df = pd.read_json(StringIO(result[9]), orient="split")
    assert "Renamed_A" in updated_df.columns
    assert "Asset_A" not in updated_df.columns
    assert "Asset_B" not in updated_df.columns


def test_po_on_modal_ok_blocks_duplicate_series_names(page_modules, raw_json):
    _, portopt = page_modules

    result = portopt.po_on_modal_ok(
        1,
        raw_json,
        {},
        [
            {
                "__orig_series": "Asset_A",
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
                "__orig_series": "Asset_B",
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
        ],
        [],
        None,
    )

    assert result[5] is True
    assert result[15] is False
    assert "duplicate" in str(result[18]).lower()
    assert result[20] is False


def test_po_begin_series_selection_request_opens_modal_and_releases_blocker(page_modules):
    _, portopt = page_modules

    assert portopt.po_begin_series_selection_request("token") == (True, False)


def test_po_resolve_series_selection_modal_controls_overlay_and_ok(page_modules):
    _, portopt = page_modules

    assert portopt.po_resolve_series_selection_modal("token", None) == (True, True, "", "blue", True)
    assert portopt.po_resolve_series_selection_modal(
        "token", {"token": "token", "status": "ready", "message": ""}
    ) == (False, False, "", "blue", True)
    assert portopt.po_resolve_series_selection_modal(
        "token", {"token": "token", "status": "error", "message": "bad"}
    ) == (False, True, "bad", "red", False)


def test_portopt_layout_includes_page_ready_stores_and_visible_overlay(page_modules):
    _, portopt = page_modules

    base_store = _find_component_by_id(portopt.layout, "po-base-controls-ready-store")
    ex_ante_store = _find_component_by_id(portopt.layout, "po-ex-ante-controls-ready-store")
    page_store = _find_component_by_id(portopt.layout, "po-page-ready-store")
    overlay = _find_component_by_id(portopt.layout, "po-ui-blocker-overlay")

    assert base_store is not None
    assert ex_ante_store is not None
    assert page_store is not None
    assert _component_prop(base_store, "data") is False
    assert _component_prop(ex_ante_store, "data") is False
    assert _component_prop(page_store, "data") is False
    assert _component_prop(overlay, "visible") is True


def test_po_init_date_range_sets_page_ready(monkeypatch, page_modules, raw_json):
    _, portopt = page_modules

    monkeypatch.setattr(
        portopt,
        "get_periodicity_range_metadata",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        portopt,
        "compute_date_range_candidates_from_metadata",
        lambda *_args, **_kwargs: {
            "available_series": ["Asset_A"],
            "common_daily_start": "2024-01-01",
            "common_daily_end": "2024-12-31",
        },
    )
    monkeypatch.setattr(
        portopt,
        "resolve_initial_range",
        lambda *_args, **_kwargs: ("2024-01-01", "2024-12-31"),
    )

    result = portopt.po_init_date_range(
        {"raw_data_hash": "hash"},
        "daily",
        ["Asset_A"],
        True,
        True,
        raw_json,
        {"start": "2024-01-01", "end": "2024-12-31"},
        False,
    )

    assert result[-2] == {"start": "2024-01-01", "end": "2024-12-31"}
    assert result[-1] is True


def test_po_init_date_range_leaves_page_ready_unchanged_without_data(page_modules):
    _, portopt = page_modules

    result = portopt.po_init_date_range(None, "daily", [], True, True, None, None, False)

    assert result[-2] is None
    assert result[-1] is no_update


def test_po_overlay_visible_uses_restore_and_page_ready(page_modules):
    _, portopt = page_modules

    assert portopt._po_overlay_visible(False, None, False, False, False) is True
    assert portopt._po_overlay_visible(False, None, True, True, False) is False
    assert portopt._po_overlay_visible(False, "raw", True, True, False) is True
    assert portopt._po_overlay_visible(False, "raw", True, True, True) is False
    assert portopt._po_overlay_visible(True, "raw", True, True, True) is True


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
        None,
        "raw-json",
        "daily",
        {},
        {},
        None,
        0,
        {},
        "light",
        ["growth"],
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
        None,
        "raw-json",
        {},
        {},
        None,
        0,
        {},
        "light",
        ["rolling"],
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
        None,
        "raw-json",
        {},
        {},
        None,
        0,
        {},
        "light",
        ["drawdown"],
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

    options, value, disabled = portopt.po_populate_frontier_windows("P1", results, "frontier", None, ["frontier"])
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

    column_defs, row_data = portopt.po_render_turnover_table("P1", results, "turnover", "table", None, ["turnover"])
    assert column_defs[0]["field"] == "Rebalance Date"
    assert row_data[0]["Turnover"] == pytest.approx(0.1)


def test_po_sync_results_with_raw_data_prunes_missing_portfolios(page_modules, raw_json):
    _, portopt = page_modules
    df = pd.read_json(StringIO(raw_json), orient="split")
    df["KeepMe"] = 0.0
    raw_with_portfolio = df_to_json(df)
    results = {"KeepMe": {"x": 1}, "DropMe": {"x": 2}}

    pruned = portopt.po_sync_results_with_raw_data(raw_with_portfolio, results)
    assert pruned == {"KeepMe": {"x": 1}}


def test_po_open_modal_opens_on_portopt_activation_with_summary(monkeypatch, page_modules):
    _, portopt = page_modules
    monkeypatch.setattr(
        portopt,
        "callback_context",
        SimpleNamespace(triggered_id="wb-active-module-store"),
    )

    out = portopt.po_open_modal(
        None,
        1,
        {"columns": ["P1", "P2"]},
        "portopt",
        [],
        None,
        None,
        None,
        None,
    )

    assert isinstance(out[0], str) and out[0]
    assert out[1:] == ("", "blue", True, None)


def test_po_open_modal_ignores_inactive_module(monkeypatch, page_modules):
    _, portopt = page_modules
    monkeypatch.setattr(
        portopt,
        "callback_context",
        SimpleNamespace(triggered_id="wb-active-module-store"),
    )

    with pytest.raises(PreventUpdate):
        portopt.po_open_modal(
            None,
            1,
            {"columns": ["P1", "P2"]},
            "analyticstool",
            [],
            None,
            None,
            None,
            None,
        )


def test_po_delete_portfolio_removes_from_results_and_raw(page_modules, raw_json):
    _, portopt = page_modules
    df = pd.read_json(StringIO(raw_json), orient="split")
    df["P1"] = 0.0
    raw_with_portfolio = df_to_json(df)
    results = {"P1": {"x": 1}, "P2": {"x": 2}}

    new_results, new_raw, new_sel = portopt.po_delete_portfolio(1, "P1", results, raw_with_portfolio)

    assert "P1" not in new_results
    df_after = pd.read_json(StringIO(new_raw), orient="split")
    assert "P1" not in df_after.columns
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

    assert corr_store is no_update
    assert captured["kwargs"]["shrinkage"] == "ledoit_wolf"
    assert captured["kwargs"]["shrinkage_target"] == "constant_correlation"
    assert set(cov_store) == {"Asset_A", "Asset_B"}
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

    assert corr_store is no_update
    assert captured["kwargs"]["exp_weighted"] is True
    assert captured["kwargs"]["shrinkage"] == "none"
    assert captured["kwargs"]["shrinkage_target"] == "scaled_identity"
    assert len(rows) == 2


def test_po_update_frontier_risk_measure_options_restricts_ex_ante(page_modules):
    _, portopt = page_modules
    results = {"P1": {"config": {"model": "ex_ante_mv"}}}

    options, value = portopt.po_update_frontier_risk_measure_options("P1", results, None, "CVaR", ["frontier"])
    assert options == [{"value": "MV", "label": "Volatility"}]
    assert value == "MV"


def test_po_render_frontier_table_includes_frontier_points_and_weights(monkeypatch, page_modules):
    _, portopt = page_modules
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
    monkeypatch.setattr(portopt, "_build_frontier_snapshot", lambda **_kwargs: snapshot)
    monkeypatch.setattr(portopt, "_get_cached_frontier_snapshot", lambda *_args, **_kwargs: None)

    results = {
        "P1": {
            "config": {"model": "risk_parity", "selected_series": ["Asset_A", "Asset_B"]},
            "window_weights": _sample_window_weights(),
        }
    }

    column_defs, row_data = portopt.po_render_frontier_table(
        "P1",
        results,
        "frontier",
        "table",
        "1",
        "MV",
        None,
        "raw-json",
        "daily",
        {},
        {},
        0,
        {},
        {},
        None,
        [],
        ["frontier"],
    )

    assert any(col["field"] == "Wt_Asset_A" for col in column_defs)
    assert any(col["field"] == "Sharpe Ratio" for col in column_defs)
    assert any(row["Type"] == "Optimized Portfolio" for row in row_data)
    assert any(row["Type"] == "Frontier Point" for row in row_data)


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
        "_build_frontier_snapshot",
        lambda **_kwargs: {
            "window_index": 0,
            "risk_measure": "MV",
            "asset_order": ["Asset_A", "Asset_B"],
            "portfolio": {"name": "MyPort", "return": 0.1, "risk": 0.2, "weights": {"Asset_A": 0.5, "Asset_B": 0.5}},
            "assets": [],
            "frontier_points": [],
            "frontier_portfolios": [],
            "window_est_start": "2024-01-01",
            "window_est_end": "2024-01-31",
        },
    )

    ex_cov = {
        "Asset_A": {"Asset_A": 0.04, "Asset_B": 0.01},
        "Asset_B": {"Asset_A": 0.01, "Asset_B": 0.09},
    }

    results_out, _new_raw, status, _pending = portopt.po_run_optimization(
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


def test_po_run_optimization_monthly_writeback_aligns_month_end(monkeypatch, page_modules):
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

    working_df = raw_df.copy()
    working_df.index = pd.to_datetime(["1976-06-30", "1976-07-31", "1976-08-31", "1976-09-30"])
    monkeypatch.setattr(portopt, "_po_get_working_returns", lambda *_args, **_kwargs: working_df.copy())

    monkeypatch.setattr(
        portopt,
        "run_portfolio_optimization",
        lambda *_args, **_kwargs: (
            [
                SimpleNamespace(
                    apply_start=pd.Timestamp("1976-06-30"),
                    apply_end=pd.Timestamp("1976-09-30"),
                    est_start=pd.Timestamp("1976-06-30"),
                    est_end=pd.Timestamp("1976-09-30"),
                    weights={"Asset_A": 0.5, "Asset_B": 0.5},
                )
            ],
            pd.Series(
                [0.005, 0.006, 0.007, 0.008],
                index=pd.to_datetime(["1976-06-30", "1976-07-31", "1976-08-31", "1976-09-30"]),
            ),
            {},
        ),
    )

    results_out, new_raw, status, _pending = portopt.po_run_optimization(
        1,
        raw_json,
        "monthly",
        "monthly",
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
        12,
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
    assert "MyPort" in results_out

    df_after = pd.read_json(StringIO(new_raw), orient="split")
    df_after.index = pd.to_datetime(df_after.index)
    assert pd.Timestamp("1976-07-30") not in df_after.index
    assert pd.Timestamp("1976-07-31") in df_after.index
    assert pd.Timestamp("1976-08-31") in df_after.index
    assert df_after.index.is_month_end.all()
    assert df_after.loc[pd.Timestamp("1976-07-31"), "MyPort"] == pytest.approx(0.006)
    assert df_after.loc[pd.Timestamp("1976-08-31"), "MyPort"] == pytest.approx(0.007)


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
            {},
        )

    monkeypatch.setattr(portopt, "run_portfolio_optimization", _fake_run_portfolio_optimization)

    results_out, _new_raw, status, _pending = portopt.po_run_optimization(
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
        "constant_correlation",
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


def test_build_frontier_snapshot_uses_custom_moments_for_shrinkage(monkeypatch, page_modules, raw_json):
    _, portopt = page_modules
    df = pd.read_json(StringIO(raw_json), orient="split")[["Asset_A", "Asset_B"]]
    df.index = pd.to_datetime(df.index)
    captured = {}

    monkeypatch.setattr(portopt, "_po_get_working_returns", lambda *_args, **_kwargs: df.copy())
    monkeypatch.setattr(
        portopt,
        "estimate_mean_vector",
        lambda *_args, **_kwargs: pd.DataFrame([[0.01, 0.02]], columns=["Asset_A", "Asset_B"]),
    )
    def _fake_estimate_covariance_matrix(*_args, **kwargs):
        captured["cov_kwargs"] = kwargs
        return pd.DataFrame(
            [[0.04, 0.01], [0.01, 0.09]],
            index=["Asset_A", "Asset_B"],
            columns=["Asset_A", "Asset_B"],
        )

    monkeypatch.setattr(portopt, "estimate_covariance_matrix", _fake_estimate_covariance_matrix)
    def _fake_compute_efficient_frontier(**kwargs):
        captured["kwargs"] = kwargs
        return (
            [{"return": 0.1, "risk": 0.2}],
            [{"name": "Asset_A", "return": 0.1, "risk": 0.2}],
            [{"point_index": 0, "return": 0.1, "risk": 0.2, "weights": {"Asset_A": 0.5, "Asset_B": 0.5}}],
        )

    monkeypatch.setattr(portopt, "compute_efficient_frontier", _fake_compute_efficient_frontier)
    monkeypatch.setattr(
        portopt,
        "_resolve_risk_free_context",
        lambda **_kwargs: {"rf_annual": 0.0, "rf_source": "unused", "rf_warning": None},
    )

    snapshot = portopt._build_frontier_snapshot(
        selected_portfolio="P1",
        portfolio_data={
            "window_weights": _sample_window_weights(),
            "config": {
                "selected_series": ["Asset_A", "Asset_B"],
                "model": "risk_parity",
                "missing_data": "fill_na",
                "exp_wt_cov": False,
                "halflife": 63,
                "cov_shrinkage": "ledoit_wolf",
                "cov_shrinkage_target": "constant_correlation",
            },
        },
        raw_data=raw_json,
        periodicity="daily",
        bench={},
        ls={},
        vol_scaler=0,
        vol_scaling={},
        window_idx=0,
        rm="MV",
        linear_constraints=[],
        saved_series_store=None,
        cmabench_assignments={},
    )

    assert snapshot["risk_measure"] == "MV"
    assert captured["kwargs"]["custom_mu"] is not None
    assert captured["kwargs"]["custom_cov"] is not None
    assert captured["cov_kwargs"]["shrinkage_target"] == "constant_correlation"


def test_compute_window_risk_contributions_uses_custom_covariance_estimator(monkeypatch, page_modules):
    _, portopt = page_modules
    idx = pd.date_range("2023-12-01", periods=70, freq="D")
    working_df = pd.DataFrame({"Asset_A": 0.01, "Asset_B": 0.02}, index=idx)
    captured = {}

    def _fake_estimate_covariance_matrix(*_args, **kwargs):
        captured["cov_kwargs"] = kwargs
        return pd.DataFrame(
            [[0.04, 0.01], [0.01, 0.09]],
            index=["Asset_A", "Asset_B"],
            columns=["Asset_A", "Asset_B"],
        )

    monkeypatch.setattr(portopt, "estimate_covariance_matrix", _fake_estimate_covariance_matrix)

    def _fake_compute_risk_contributions(weights_dict, returns_df, custom_cov=None):
        captured["custom_cov"] = custom_cov
        return {"Asset_A": 0.6, "Asset_B": 0.4}

    monkeypatch.setattr(portopt, "compute_risk_contributions", _fake_compute_risk_contributions)

    rows = portopt._compute_window_risk_contributions(
        working_df,
        ["Asset_A", "Asset_B"],
        _sample_window_weights(),
        {"exp_wt_cov": False, "halflife": 63, "cov_shrinkage": "ledoit_wolf", "cov_shrinkage_target": "constant_correlation"},
    )

    assert len(rows) == 2
    assert captured["custom_cov"] is not None
    assert captured["cov_kwargs"]["shrinkage_target"] == "constant_correlation"


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
    monkeypatch.setattr(portopt, "_get_cached_frontier_snapshot", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        portopt,
        "_build_frontier_snapshot",
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
    assert xl.sheet_names[:6] == [
        "Settings",
        "Weights",
        "Turnover",
        "Statistics",
        "Returns",
        "Growth of $1",
    ]
    settings_df = xl.parse("Settings", keep_default_na=False)
    settings_map = dict(zip(settings_df["Parameter"], settings_df["Value"]))
    assert settings_map["Covariance Shrinkage"] == "None"
    assert settings_map["Covariance Shrinkage Target"] == "N/A"
    assert "Drawdown" in xl.sheet_names
    assert xl.sheet_names[-3:] == ["Attribution", "Risk", "Frontier"]

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
    modal = _find_component_by_id(portopt.layout, "po-help-modal")
    assert modal is not None

    text_blob = " ".join(_collect_component_text(modal)).lower()
    assert "basic guide" in text_blob
    assert "advanced guide" in text_blob
    assert "model deep dive" in text_blob


def test_po_help_modal_model_deep_dive_covers_all_models(page_modules):
    _, portopt = page_modules
    modal = _find_component_by_id(portopt.layout, "po-help-modal")
    assert modal is not None

    text_blob = " ".join(_collect_component_text(modal)).lower()
    required_models = [
        "risk parity",
        "factor risk parity",
        "hierarchical rp",
        "maximize sharpe ratio",
        "minimize variance",
        "minimize cvar",
        "equal weight",
        "ex ante mean-variance",
        "black-litterman",
    ]
    for model in required_models:
        assert model in text_blob
