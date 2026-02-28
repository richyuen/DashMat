from __future__ import annotations

from io import BytesIO
from io import StringIO
from types import SimpleNamespace

import pandas as pd
import pytest
from dash import no_update
from dash.exceptions import PreventUpdate
from utils.route_intent import (
    ACTION_CONFIGURE_AFTER_IMPORT,
    ACTION_OPEN_IMPORT_MODAL,
    FLOW_DB,
    build_route_intent,
)
from utils.returns import df_to_json


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
        "compute_date_range_candidates",
        lambda *_args, **_kwargs: {
            "available_series": ["Asset_A"],
            "common_daily_start": "2024-01-01",
            "common_daily_end": "2024-12-31",
        },
    )
    monkeypatch.setattr(
        analyticstool,
        "resolve_initial_range",
        lambda *_args, **_kwargs: ("2024-01-01", "2024-12-31"),
    )

    start, end, _style, _common_disabled, _daily_disabled, _max_disabled, range_store, ready, page_ready = (
        analyticstool.initialize_date_range(
            "raw-json",
            "daily",
            ["Asset_A"],
            1,
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
    assert page_ready is True


def test_analyticstool_layout_includes_page_ready_store_and_visible_overlay(page_modules):
    analyticstool, _ = page_modules

    ready_store = _find_component_by_id(analyticstool.layout, "at-page-ready-store")
    overlay = _find_component_by_id(analyticstool.layout, "at-ui-blocker-overlay")

    assert ready_store is not None
    assert _component_prop(ready_store, "data") is False
    assert overlay is not None
    assert _component_prop(overlay, "visible") is True


def test_restore_application_state_marks_empty_page_ready_after_page_load(page_modules):
    analyticstool, _ = page_modules

    result = analyticstool.restore_application_state(
        1,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        False,
    )

    assert result[-2] is False
    assert result[-1] is True


def test_restore_application_state_keeps_loaded_page_ready_unchanged(page_modules, raw_json):
    analyticstool, _ = page_modules

    result = analyticstool.restore_application_state(
        1,
        raw_json,
        "daily",
        "daily_trading",
        ["Asset_A"],
        "total",
        0,
        "statistics",
        "1y",
        "total_return",
        "annualized",
        "chart",
        "chart",
        "chart",
        "box",
        5,
        "raw",
        "annual",
        None,
        [],
        False,
    )

    assert result[-2] is False
    assert result[-1] is no_update


def test_at_overlay_visible_uses_ready_and_blocker(page_modules):
    analyticstool, _ = page_modules

    assert analyticstool._at_overlay_visible(False, False) is True
    assert analyticstool._at_overlay_visible(True, True) is True
    assert analyticstool._at_overlay_visible(False, True) is False


def test_update_statistics_transposes_series_into_columns(monkeypatch, page_modules):
    analyticstool, _ = page_modules

    def _fake_stats(*_args, **_kwargs):
        return [
            {"Series": "Asset_A", "Cumulative Return": 0.10},
            {"Series": "Asset_B", "Cumulative Return": 0.20},
        ]

    monkeypatch.setattr(analyticstool, "calculate_statistics_cached", _fake_stats)

    column_defs, row_data, loaded = analyticstool.update_statistics(
        "raw-json",
        "daily",
        ["Asset_A", "Asset_B"],
        {},
        {},
        {"start": "2024-01-01", "end": "2024-12-31"},
        True,
        0,
        {},
        None,
    )

    assert column_defs[0]["field"] == "Statistic"
    assert {c["field"] for c in column_defs[1:]} == {"Asset_A", "Asset_B"}
    cum_row = next(row for row in row_data if row["Statistic"] == "Cumulative Return")
    assert cum_row["Asset_A"] == pytest.approx(0.10)
    assert cum_row["Asset_B"] == pytest.approx(0.20)
    assert loaded is True


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


def test_open_db_add_modal_clears_blocker_with_modal_payload(monkeypatch, page_modules):
    analyticstool, _ = page_modules
    expected = (True, [{"value": "IDX_A", "label": "Index A"}], [])
    monkeypatch.setattr(analyticstool, "compute_open_db_add_modal", lambda *_args, **_kwargs: expected)

    assert analyticstool.open_db_add_modal(1, None) == (*expected, False, analyticstool.no_update)


def test_open_db_add_modal_consumes_fresh_page_load_intent(monkeypatch, page_modules):
    analyticstool, _ = page_modules
    expected = (True, [{"value": "IDX_A", "label": "Index A"}], [])
    route_intent = build_route_intent("analyticstool", ACTION_OPEN_IMPORT_MODAL, flow=FLOW_DB)
    monkeypatch.setattr(analyticstool, "callback_context", SimpleNamespace(triggered_id="at-page-load-trigger"))
    monkeypatch.setattr(analyticstool, "compute_open_db_add_modal", lambda *_args, **_kwargs: expected)

    assert analyticstool.open_db_add_modal(None, None, 1, route_intent, None) == (
        *expected,
        False,
        route_intent["token"],
    )


def test_open_db_add_modal_ignores_stale_page_load_intent(monkeypatch, page_modules):
    analyticstool, _ = page_modules
    route_intent = build_route_intent("analyticstool", ACTION_OPEN_IMPORT_MODAL, flow=FLOW_DB)
    route_intent["created_at"] = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(seconds=61)).isoformat()
    monkeypatch.setattr(analyticstool, "callback_context", SimpleNamespace(triggered_id="at-page-load-trigger"))

    with pytest.raises(PreventUpdate):
        analyticstool.open_db_add_modal(None, None, 1, route_intent, None)


def test_update_statistics_requires_ready_state(page_modules):
    analyticstool, _ = page_modules

    with pytest.raises(PreventUpdate):
        analyticstool.update_statistics(
            "raw-json",
            "daily",
            ["Asset_A"],
            {},
            {},
            None,
            False,
            0,
            {},
            None,
        )


def test_control_statistics_loading_display(page_modules):
    analyticstool, _ = page_modules
    assert analyticstool.control_statistics_loading_display("statistics", False, False) == "show"
    assert analyticstool.control_statistics_loading_display("statistics", True, False) == "show"
    assert analyticstool.control_statistics_loading_display("statistics", True, True) == "auto"
    assert analyticstool.control_statistics_loading_display("returns", False, False) == "auto"


def test_update_growth_grid_requires_growth_table_view(page_modules):
    analyticstool, _ = page_modules
    with pytest.raises(PreventUpdate):
        analyticstool.update_growth_grid(
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


def test_at_ordered_modal_rows_prefers_virtual_row_order(page_modules):
    analyticstool, _ = page_modules

    ordered = analyticstool._at_ordered_modal_rows(
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

    assert [analyticstool._at_modal_orig_series(row) for row in ordered] == [
        "Asset_C",
        "Asset_A",
        "Asset_B",
    ]


def test_update_correlogram_meta_returns_no_update_when_not_active(page_modules):
    analyticstool, _ = page_modules
    assert analyticstool.update_correlogram_meta(["Asset_A", "Asset_B"], "growth") is no_update
    assert analyticstool.update_correlogram_meta(["Asset_A", "Asset_B"], "correlogram") == {"num_series": 2}


def test_update_correlogram_target_key_changes_on_exp_weight_inputs(page_modules):
    analyticstool, _ = page_modules
    date_range = {"start": "2024-01-01", "end": "2024-12-31"}

    key_unweighted = analyticstool.update_correlogram_target_key(
        "correlogram",
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
        "correlation",
        False,
        63,
        "none",
        "scaled_identity",
        120,
        None,
    )
    key_weighted = analyticstool.update_correlogram_target_key(
        "correlogram",
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
        "correlation",
        True,
        0.94,
        "none",
        "scaled_identity",
        120,
        None,
    )

    assert isinstance(key_unweighted, str)
    assert isinstance(key_weighted, str)
    assert key_unweighted != key_weighted
    assert (
        analyticstool.update_correlogram_target_key(
            "correlogram",
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
            "correlation",
            True,
            0.94,
            "none",
            "scaled_identity",
            120,
            key_weighted,
        )
        is no_update
    )


def test_update_correlogram_target_key_changes_on_shrinkage_for_matrix_views(page_modules):
    analyticstool, _ = page_modules
    date_range = {"start": "2024-01-01", "end": "2024-12-31"}

    key_none = analyticstool.update_correlogram_target_key(
        "correlogram",
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
        "correlation",
        False,
        63,
        "none",
        "scaled_identity",
        120,
        None,
    )
    key_shrunk = analyticstool.update_correlogram_target_key(
        "correlogram",
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
        "correlation",
        False,
        63,
        "ledoit_wolf",
        "scaled_identity",
        120,
        None,
    )

    assert isinstance(key_none, str)
    assert isinstance(key_shrunk, str)
    assert key_none != key_shrunk


def test_update_correlogram_target_key_changes_on_target_for_ledoit_wolf_matrix_views(page_modules):
    analyticstool, _ = page_modules
    date_range = {"start": "2024-01-01", "end": "2024-12-31"}

    key_scaled = analyticstool.update_correlogram_target_key(
        "correlogram",
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
        "covariance",
        False,
        63,
        "ledoit_wolf",
        "scaled_identity",
        120,
        None,
    )
    key_constant = analyticstool.update_correlogram_target_key(
        "correlogram",
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
        "covariance",
        False,
        63,
        "ledoit_wolf",
        "constant_correlation",
        120,
        None,
    )

    assert isinstance(key_scaled, str)
    assert isinstance(key_constant, str)
    assert key_scaled != key_constant


def test_update_correlogram_target_key_ignores_shrinkage_for_scatter_view(page_modules):
    analyticstool, _ = page_modules
    date_range = {"start": "2024-01-01", "end": "2024-12-31"}

    key_scatter = analyticstool.update_correlogram_target_key(
        "correlogram",
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
        "correlogram",
        False,
        63,
        "none",
        "scaled_identity",
        120,
        None,
    )

    assert (
        analyticstool.update_correlogram_target_key(
            "correlogram",
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
            "correlogram",
            False,
            63,
            "oas",
            "constant_correlation",
            120,
            key_scatter,
        )
        is no_update
    )


def test_update_correlogram_target_key_ignores_target_when_not_effective(page_modules):
    analyticstool, _ = page_modules
    date_range = {"start": "2024-01-01", "end": "2024-12-31"}

    key_oas = analyticstool.update_correlogram_target_key(
        "correlogram",
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
        "correlation",
        False,
        63,
        "oas",
        "scaled_identity",
        120,
        None,
    )

    assert (
        analyticstool.update_correlogram_target_key(
            "correlogram",
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
            "correlation",
            False,
            63,
            "oas",
            "constant_correlation",
            120,
            key_oas,
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


def test_update_correlogram_passes_effective_target(monkeypatch, page_modules):
    analyticstool, _ = page_modules
    captured = {}

    def _fake_generate_correlogram_cached(*args, **kwargs):
        captured["args"] = args
        return {
            "display_df": pd.DataFrame({"Asset_A": [0.01, 0.02], "Asset_B": [0.00, 0.03]}),
            "corr_matrix": pd.DataFrame([[1.0, 0.5], [0.5, 1.0]], index=["Asset_A", "Asset_B"], columns=["Asset_A", "Asset_B"]),
            "cov_matrix": pd.DataFrame([[0.04, 0.01], [0.01, 0.09]], index=["Asset_A", "Asset_B"], columns=["Asset_A", "Asset_B"]),
            "available_series": ["Asset_A", "Asset_B"],
            "n": 2,
        }

    monkeypatch.setattr(analyticstool, "generate_correlogram_cached", _fake_generate_correlogram_cached)

    analyticstool.update_correlogram(
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
        "covariance",
        120,
        "light",
    )

    assert captured["args"][11] == "ledoit_wolf"
    assert captured["args"][12] == "constant_correlation"


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


def test_update_correlogram_shrinkage_error_renders_annotation(monkeypatch, page_modules):
    analyticstool, _ = page_modules
    monkeypatch.setattr(
        analyticstool,
        "generate_correlogram_cached",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("Insufficient overlapping observations for shrinkage covariance estimate.")
        ),
    )

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
        "oas",
        "scaled_identity",
        "correlation",
        120,
        "light",
    )

    assert rendered_key == "req-key"
    assert graph.figure.layout.annotations[0].text == "Insufficient overlapping observations for shrinkage covariance estimate."


def test_on_modal_ok_commits_local_series_modal_state(page_modules, raw_json):
    analyticstool, _ = page_modules

    result = analyticstool.on_modal_ok(
        1,
        raw_json,
        "daily",
        None,
        None,
        [
            {
                "__orig_series": "Asset_A",
                "Series": "Renamed_A",
                "Benchmark": "Asset_B",
                "LongShort": True,
                "ScaleVol": False,
                "Delete": False,
            },
            {
                "__orig_series": "Asset_B",
                "Series": "Asset_B",
                "Benchmark": "None",
                "LongShort": False,
                "ScaleVol": True,
                "Delete": False,
            },
        ],
        [{"__orig_series": "Asset_B", "Series": "Asset_B"}],
        [{"__orig_series": "Asset_B"}, {"__orig_series": "Asset_A"}],
    )

    assert result[0] == ["Asset_B"]
    assert result[1] == {"Renamed_A": "Asset_B", "Asset_B": "None"}
    assert result[2] == {"Renamed_A": True, "Asset_B": False}
    assert result[3] == ["Asset_B", "Renamed_A"]
    assert result[9] == {"Renamed_A": False, "Asset_B": True}
    assert result[10] is False

    updated_df = pd.read_json(StringIO(result[8]), orient="split")
    assert "Renamed_A" in updated_df.columns
    assert "Asset_A" not in updated_df.columns


def test_on_modal_ok_blocks_duplicate_series_names(page_modules, raw_json):
    analyticstool, _ = page_modules

    result = analyticstool.on_modal_ok(
        1,
        raw_json,
        "daily",
        None,
        None,
        [
            {
                "__orig_series": "Asset_A",
                "Series": "Asset_B",
                "Benchmark": "None",
                "LongShort": False,
                "ScaleVol": True,
                "Delete": False,
            },
            {
                "__orig_series": "Asset_B",
                "Series": "Asset_B",
                "Benchmark": "None",
                "LongShort": False,
                "ScaleVol": True,
                "Delete": False,
            },
        ],
        [],
        None,
    )

    assert result[4] is True
    assert result[10] is False
    assert "duplicate" in str(result[13]).lower()
    assert result[15] is False


def test_on_modal_ok_does_not_emit_raw_data_when_unchanged(page_modules, raw_json):
    analyticstool, _ = page_modules

    result = analyticstool.on_modal_ok(
        1,
        raw_json,
        "daily",
        None,
        None,
        [
            {
                "__orig_series": "Asset_A",
                "Series": "Asset_A",
                "Benchmark": "None",
                "LongShort": False,
                "ScaleVol": True,
                "Delete": False,
            }
        ],
        [{"__orig_series": "Asset_A", "Series": "Asset_A"}],
        [{"__orig_series": "Asset_A"}],
    )

    assert result[8] is no_update


def test_on_modal_ok_commits_pending_import_and_clears_staging(page_modules):
    analyticstool, _ = page_modules
    imported = pd.DataFrame(
        {"New_A": [0.01, 0.02], "New_B": [0.0, -0.01]},
        index=pd.to_datetime(["2024-01-31", "2024-02-29"]),
    )
    imported.index.name = "Date"
    pending_payload = analyticstool._at_build_imported_pending_payload(
        imported,
        "monthly",
        "monthly",
        ["New_A", "New_B"],
        ["New_A", "New_B"],
        {"New_A": "None", "New_B": "New_A"},
        {"New_A": False, "New_B": True},
        {"New_A": True, "New_B": False},
        "token-1",
        {"mode": "show", "message": "Loaded pending import", "color": "green"},
    )

    result = analyticstool.on_modal_ok(
        1,
        None,
        "daily",
        "token-1",
        pending_payload,
        [
            {
                "__orig_series": "New_A",
                "Series": "Renamed_A",
                "Benchmark": "None",
                "LongShort": False,
                "ScaleVol": True,
                "Delete": False,
            },
            {
                "__orig_series": "New_B",
                "Series": "New_B",
                "Benchmark": "New_A",
                "LongShort": True,
                "ScaleVol": False,
                "Delete": False,
            },
        ],
        [
            {"__orig_series": "New_A", "Series": "Renamed_A"},
            {"__orig_series": "New_B", "Series": "New_B"},
        ],
        [{"__orig_series": "New_B"}, {"__orig_series": "New_A"}],
    )

    assert result[0] == ["New_B", "Renamed_A"]
    assert result[1] == {"Renamed_A": "None", "New_B": "Renamed_A"}
    assert result[2] == {"Renamed_A": False, "New_B": True}
    assert result[3] == ["New_B", "Renamed_A"]
    assert result[7] == ["New_B", "Renamed_A"]
    assert result[9] == {"Renamed_A": True, "New_B": False}
    assert result[13] == ""
    assert result[14] == "blue"
    assert result[15] is True
    assert result[16] == "monthly"
    assert result[18] == "monthly"
    assert result[19] is False
    assert result[20] == "monthly"
    assert result[21] is True
    assert result[22] == []
    assert result[23] is None

    updated_df = pd.read_json(StringIO(result[8]), orient="split")
    assert list(updated_df.columns) == ["Renamed_A", "New_B"]


def test_on_modal_ok_keeps_pending_import_when_validation_fails(page_modules):
    analyticstool, _ = page_modules
    imported = pd.DataFrame(
        {"New_A": [0.01], "New_B": [0.02]},
        index=pd.to_datetime(["2024-01-31"]),
    )
    imported.index.name = "Date"
    pending_payload = analyticstool._at_build_imported_pending_payload(
        imported,
        "monthly",
        "monthly",
        ["New_A", "New_B"],
        ["New_A", "New_B"],
        {"New_A": "None", "New_B": "None"},
        {"New_A": False, "New_B": False},
        {"New_A": True, "New_B": True},
        "token-2",
        {"mode": "show", "message": "Loaded pending import", "color": "green"},
    )

    result = analyticstool.on_modal_ok(
        1,
        None,
        "daily",
        "token-2",
        pending_payload,
        [
            {
                "__orig_series": "New_A",
                "Series": "Duplicate",
                "Benchmark": "None",
                "LongShort": False,
                "ScaleVol": True,
                "Delete": False,
            },
            {
                "__orig_series": "New_B",
                "Series": "Duplicate",
                "Benchmark": "None",
                "LongShort": False,
                "ScaleVol": True,
                "Delete": False,
            },
        ],
        [],
        None,
    )

    assert result[4] is True
    assert result[8] is no_update
    assert "duplicate" in str(result[13]).lower()
    assert result[15] is False
    assert result[23] is no_update


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
    )

    pending_payload = result[19]
    out_json = pending_payload["raw_data"]
    out_periodicity = pending_payload["original_periodicity"]
    out_default_periodicity = pending_payload["default_periodicity"]

    out_df = pd.read_json(StringIO(out_json), orient="split")
    out_df.index = pd.to_datetime(out_df.index)

    assert result[0] is no_update
    assert result[10]
    assert out_periodicity == "monthly"
    assert out_default_periodicity == "monthly"
    assert out_df.index.is_month_end.all()
    assert pd.Timestamp("1976-07-30") not in out_df.index
    assert pd.Timestamp("1976-07-31") in out_df.index


def test_at_begin_series_selection_request_opens_modal_and_releases_blocker(page_modules):
    analyticstool, _ = page_modules

    assert analyticstool.at_begin_series_selection_request("token") == (True, False)


def test_open_modal_ignores_stale_configure_after_import_route_intent(monkeypatch, page_modules, raw_json):
    analyticstool, _ = page_modules
    route_intent = build_route_intent("analyticstool", ACTION_CONFIGURE_AFTER_IMPORT)
    route_intent["created_at"] = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(seconds=61)).isoformat()
    monkeypatch.setattr(analyticstool, "callback_context", SimpleNamespace(triggered_id="at-page-load-trigger"))

    with pytest.raises(PreventUpdate):
        analyticstool.open_modal(
            None,
            1,
            raw_json,
            "/analyticstool",
            ["Asset_A"],
            None,
            None,
            route_intent,
            None,
        )


def test_at_resolve_series_selection_modal_controls_overlay_and_ok(page_modules):
    analyticstool, _ = page_modules

    assert analyticstool.at_resolve_series_selection_modal("token", None, None) == (True, True, "", "blue", True)
    assert analyticstool.at_resolve_series_selection_modal(
        "token", {"token": "token", "status": "rendered", "message": ""}, None
    ) == (True, True, "", "blue", True)
    assert analyticstool.at_resolve_series_selection_modal(
        "token", {"token": "token", "status": "ready", "message": ""}, None
    ) == (False, False, "", "blue", True)
    assert analyticstool.at_resolve_series_selection_modal(
        "token", {"token": "token", "status": "timeout", "message": "slow"}, None
    ) == (False, True, "slow", "red", False)


def test_at_resolve_series_selection_modal_shows_staged_import_message(page_modules):
    analyticstool, _ = page_modules
    pending_payload = {
        "token": "token",
        "commit_alert": {
            "mode": "show",
            "message": "Loaded 3 series",
            "color": "green",
        },
    }

    assert analyticstool.at_resolve_series_selection_modal("token", None, pending_payload) == (
        True,
        True,
        "Loaded 3 series",
        "green",
        False,
    )
    assert analyticstool.at_resolve_series_selection_modal(
        "token",
        {"token": "token", "status": "ready", "message": ""},
        pending_payload,
    ) == (
        False,
        False,
        "Loaded 3 series",
        "green",
        False,
    )


def test_at_resolve_series_selection_modal_error_overrides_staged_message(page_modules):
    analyticstool, _ = page_modules
    pending_payload = {
        "token": "token",
        "commit_alert": {
            "mode": "show",
            "message": "Loaded 3 series",
            "color": "green",
        },
    }

    assert analyticstool.at_resolve_series_selection_modal(
        "token",
        {"token": "token", "status": "timeout", "message": "slow"},
        pending_payload,
    ) == (
        False,
        True,
        "slow",
        "red",
        False,
    )


def test_add_series_from_database_stages_pending_selection_state(monkeypatch, page_modules):
    analyticstool, _ = page_modules
    existing_idx = pd.date_range("2024-01-01", periods=3, freq="B")
    existing_raw = pd.DataFrame({"Existing": [0.01, 0.0, -0.01]}, index=existing_idx)
    imported = pd.DataFrame({"New_A": [0.02, 0.01, 0.0]}, index=existing_idx)
    imported.index.name = "Date"
    meta = {"New_A": {"starts_daily": True, "daily_start_date": existing_idx[0]}}

    monkeypatch.setattr(
        analyticstool,
        "load_cma_returns_for_benches_with_meta",
        lambda *_args, **_kwargs: (imported.copy(), meta),
    )

    result = analyticstool.add_series_from_database(
        1,
        ["New_A"],
        df_to_json(existing_raw),
        "daily",
        ["Existing"],
        {},
        {},
        ["Existing"],
        True,
        {},
    )

    pending_payload = result[19]
    assert result[5] is no_update
    assert result[13] is no_update
    assert result[10]
    assert pending_payload["selected_series"] == ["Existing", "New_A"]
    assert pending_payload["series_order"] == ["Existing", "New_A"]


def test_update_series_selectors_prefers_matching_pending_payload(page_modules, raw_json):
    analyticstool, _ = page_modules
    imported = pd.DataFrame(
        {"Pending_A": [0.01], "Pending_B": [0.02]},
        index=pd.to_datetime(["2024-01-31"]),
    )
    imported.index.name = "Date"
    pending_payload = analyticstool._at_build_imported_pending_payload(
        imported,
        "monthly",
        "monthly",
        ["Pending_B"],
        ["Pending_B", "Pending_A"],
        {"Pending_B": "Pending_A"},
        {"Pending_B": True},
        {"Pending_A": False, "Pending_B": True},
        "pending-token",
        {"mode": "show", "message": "Loaded", "color": "green"},
    )

    children, status = analyticstool.update_series_selectors(
        "pending-token",
        raw_json,
        ["Asset_A"],
        ["Asset_A", "Asset_B"],
        {"Asset_A": "None"},
        {"Asset_A": False},
        {"Asset_A": True},
        pending_payload,
    )

    grid = children[0]
    row_data = _component_prop(grid, "rowData")
    selected_rows = _component_prop(grid, "selectedRows")

    assert status["status"] == "rendered"
    assert [row["Series"] for row in row_data] == ["Pending_B", "Pending_A"]
    assert row_data[0]["Benchmark"] == "Pending_A"
    assert row_data[0]["LongShort"] is True
    assert row_data[1]["ScaleVol"] is False
    assert [row["Series"] for row in selected_rows] == ["Pending_B"]


def test_update_series_selectors_ignores_stale_pending_payload(page_modules, raw_json):
    analyticstool, _ = page_modules
    imported = pd.DataFrame(
        {"Pending_A": [0.01]},
        index=pd.to_datetime(["2024-01-31"]),
    )
    imported.index.name = "Date"
    pending_payload = analyticstool._at_build_imported_pending_payload(
        imported,
        "monthly",
        "monthly",
        ["Pending_A"],
        ["Pending_A"],
        {"Pending_A": "None"},
        {"Pending_A": False},
        {"Pending_A": True},
        "other-token",
        {"mode": "show", "message": "Loaded", "color": "green"},
    )

    children, status = analyticstool.update_series_selectors(
        "manual-token",
        raw_json,
        ["Asset_A"],
        ["Asset_B", "Asset_A"],
        {"Asset_A": "Asset_B"},
        {"Asset_A": True},
        {"Asset_A": False},
        pending_payload,
    )

    grid = children[0]
    row_data = _component_prop(grid, "rowData")

    assert status["status"] == "rendered"
    assert [row["Series"] for row in row_data[:2]] == ["Asset_B", "Asset_A"]
    assert row_data[1]["Benchmark"] == "Asset_B"


def test_handle_upload_stages_pending_import(monkeypatch, page_modules):
    analyticstool, _ = page_modules
    imported = pd.DataFrame(
        {"Upload_A": [0.01, 0.0]},
        index=pd.to_datetime(["2024-01-31", "2024-02-29"]),
    )
    imported.index.name = "Date"

    monkeypatch.setattr(analyticstool, "get_sheet_names", lambda *_args, **_kwargs: ["Sheet1"])
    monkeypatch.setattr(analyticstool, "_shared_import_single_upload", lambda *_args, **_kwargs: imported.copy())

    result = analyticstool.handle_upload(
        "contents",
        "upload.xlsx",
        None,
        None,
        [],
        {},
        {},
        [],
        False,
        {},
    )

    pending_payload = result[24]
    assert result[0] is no_update
    assert result[10]
    assert pending_payload["selected_series"] == ["Upload_A"]
    assert pending_payload["commit_alert"] == {
        "mode": "show",
        "message": "Loaded 1 series with 2 rows from upload.xlsx",
        "color": "green",
    }


def test_on_sheet_select_ok_stages_pending_import(monkeypatch, page_modules):
    analyticstool, _ = page_modules
    imported = pd.DataFrame(
        {"Sheet_A": [0.01, -0.01]},
        index=pd.to_datetime(["2024-01-31", "2024-02-29"]),
    )
    imported.index.name = "Date"

    monkeypatch.setattr(
        analyticstool,
        "callback_context",
        SimpleNamespace(triggered_id="at-sheet-select-ok-button"),
    )
    monkeypatch.setattr(
        analyticstool,
        "_import_selected_workbook_sheets",
        lambda *_args, **_kwargs: (imported.copy(), ["Sheet1"]),
    )

    result = analyticstool.on_sheet_select_ok(
        1,
        0,
        ["Sheet1"],
        "contents",
        "book.xlsx",
        ["Sheet1"],
        None,
        None,
        [],
        {},
        {},
        [],
        False,
        {},
    )

    pending_payload = result[23]
    assert result[0] is no_update
    assert result[10]
    assert result[18] is False
    assert pending_payload["selected_series"] == ["Sheet_A"]
    assert pending_payload["commit_alert"] == {
        "mode": "show",
        "message": "Loaded 1 series with 2 rows from book.xlsx (sheet: Sheet1)",
        "color": "green",
    }


def test_on_modal_cancel_clears_pending_import(page_modules):
    analyticstool, _ = page_modules

    result = analyticstool.on_modal_cancel(1)

    assert result[:6] == (False, None, None, False, False, True)
    assert result[6] is no_update
    assert result[7] is no_update
    assert result[8] is no_update
    assert result[9] is None


def test_update_factor_series_select_includes_unselected_series(page_modules, raw_json):
    analyticstool, _ = page_modules

    options, value = analyticstool.update_factor_series_select(
        raw_json,
        ["Asset_C", "Asset_A"],
        [],
        [],
        None,
        None,
    )

    ordered_values = [opt["value"] for opt in options]
    assert ordered_values[:2] == ["raw::Asset_C", "raw::Asset_A"]
    assert set(ordered_values) == {"raw::Asset_A", "raw::Asset_B", "raw::Asset_C", "raw::Asset_D"}
    assert value == "raw::Asset_C"


def test_update_factor_series_select_includes_saved_and_session_definitions(page_modules, raw_json):
    analyticstool, _ = page_modules

    options, _value = analyticstool.update_factor_series_select(
        raw_json,
        ["Asset_A"],
        [{"FactorName": "SavedFactor"}],
        [{"FactorName": "SessionFactor"}],
        None,
        None,
    )

    option_map = {opt["value"]: opt["label"] for opt in options}
    assert "def::SavedFactor" in option_map
    assert "def::SessionFactor" in option_map
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
    draft, select_value, msg, color, hide = analyticstool.at_reset_factor_definition_draft(1, "db::DBFactor")
    assert draft["DraftMode"] == "new"
    assert select_value is None
    assert color == "blue"
    assert hide is False
    assert "New session factor draft started." in msg

    monkeypatch.setattr(analyticstool, "callback_context", type("Ctx", (), {"triggered_id": "at-factor-def-select"})())
    draft2, select_value2, msg2, color2, hide2 = analyticstool.at_reset_factor_definition_draft(0, None)
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
    out = analyticstool.at_manage_factor_definitions(
        None,
        None,
        None,
        1,
        None,
        draft,
        [],
        [db_def],
        True,
        {"role": "Admin", "username": "tester"},
    )

    local_rows = out[0]
    assert isinstance(local_rows, list)
    assert any(str(item.get("FactorName")) == "SessionFactor" for item in local_rows)
    assert out[4] == "def::SessionFactor"
    assert "Session factor selected for analysis." in out[6]


def test_use_factor_keeps_db_selection_when_unchanged(monkeypatch, page_modules):
    analyticstool, _ = page_modules
    db_def = _db_factor_definition(name="DBFactor")
    draft = analyticstool._definition_to_draft(db_def, "db")
    draft["DraftMode"] = "db"

    monkeypatch.setattr(analyticstool, "callback_context", type("Ctx", (), {"triggered_id": "at-factor-def-use-btn"})())
    out = analyticstool.at_manage_factor_definitions(
        None,
        None,
        None,
        1,
        None,
        draft,
        [],
        [db_def],
        True,
        {"role": "Admin", "username": "tester"},
    )

    assert out[0] is no_update
    assert out[4] == "def::DBFactor"
    assert "Database factor selected for analysis." in out[6]


def test_use_factor_blocks_db_name_collision_for_edited_db_draft(monkeypatch, page_modules):
    analyticstool, _ = page_modules
    db_def = _db_factor_definition(name="DBFactor", description="original")
    draft = analyticstool._definition_to_draft(db_def, "db")
    draft["DraftMode"] = "db"
    draft["Description"] = "edited"
    draft["sync_origin"] = "form"

    monkeypatch.setattr(analyticstool, "callback_context", type("Ctx", (), {"triggered_id": "at-factor-def-use-btn"})())
    out = analyticstool.at_manage_factor_definitions(
        None,
        None,
        None,
        1,
        None,
        draft,
        [],
        [db_def],
        True,
        {"role": "Admin", "username": "tester"},
    )

    assert out[0] is no_update
    assert out[6].startswith("Rename the factor to create a session copy")
    assert out[7] == "orange"


def test_sync_factor_definition_form_ignores_form_origin_updates(page_modules):
    analyticstool, _ = page_modules
    with pytest.raises(PreventUpdate):
        analyticstool.at_sync_factor_definition_form(
            {
                "sync_origin": "form",
                "FactorName": "MyFactor",
                "Description": "line 1\nline 2",
                "LongComponentList": ["ACC1 TRIndex"],
                "LongAggType": 1,
                "LongLag": 0,
                "OutputTransform": 0,
            }
        )


def test_update_factor_definition_draft_preserves_description_text(page_modules):
    analyticstool, _ = page_modules

    current = analyticstool._default_factor_draft()
    updated = analyticstool.at_update_factor_definition_draft_from_form(
        "MyFactor",
        "line 1\n",
        ["ACC1 TRIndex"],
        [],
        "1",
        None,
        0,
        "0",
        current,
    )

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


def test_update_factor_analysis_renders_one_scatter_per_selected_series(monkeypatch, page_modules):
    analyticstool, _ = page_modules
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    dependent_df = pd.DataFrame(
        {
            "Asset_A": [0.01, 0.02, 0.0, -0.01, 0.005, 0.008],
            "Asset_B": [0.015, 0.01, -0.005, 0.0, 0.004, 0.006],
        },
        index=idx,
    )
    factor_vals = pd.Series([0.2, 0.1, -0.1, 0.0, 0.05, 0.08], index=idx, name="Factor_X")
    monkeypatch.setattr(
        analyticstool,
        "_prepare_factor_analysis_frames",
        lambda *_args, **_kwargs: (dependent_df, factor_vals),
    )

    warning, content = analyticstool.update_factor_analysis(
        "factor_analysis",
        "scatter",
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


def test_download_excel_includes_factor_analysis_sheets(monkeypatch, page_modules):
    analyticstool, _ = page_modules
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    returns_df = pd.DataFrame({"Asset_A": [0.01, 0.0, -0.01, 0.02, 0.005]}, index=idx)
    returns_df.index.name = "Date"

    monkeypatch.setattr(analyticstool, "calculate_excess_returns", lambda *_args, **_kwargs: returns_df.copy())
    monkeypatch.setattr(analyticstool, "get_working_returns", lambda *_args, **_kwargs: returns_df.copy())
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
        "_prepare_factor_analysis_frames",
        lambda *_args, **_kwargs: (
            returns_df.copy(),
            pd.Series([0.2, 0.1, 0.0, -0.1, 0.05], index=idx, name="Factor_X"),
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
    regime_states = pd.Series([1, 1, 2, 2, 3], index=idx, dtype="Int64", name="Regime")
    monkeypatch.setattr(
        analyticstool,
        "compute_regime_assignments",
        lambda *_args, **_kwargs: (
            regime_states,
            {"method_type": 2, "num_regimes": 3, "observations": 5, "warning": None},
        ),
    )
    monkeypatch.setattr(
        analyticstool,
        "build_regime_timeline_frame",
        lambda *_args, **_kwargs: pd.DataFrame({"Date": idx, "Regime": [1, 1, 2, 2, 3]}),
    )
    monkeypatch.setattr(
        analyticstool,
        "build_regime_statistics_table",
        lambda *_args, **_kwargs: pd.DataFrame(
            [
                {
                    "Regime": 1,
                    "Series": "Asset_A",
                    "Observations": 2,
                    "Mean Return": 0.01,
                }
            ]
        ),
    )
    monkeypatch.setattr(
        analyticstool,
        "build_regime_transition_matrix",
        lambda *_args, **_kwargs: pd.DataFrame(
            [[0.5, 0.5], [0.2, 0.8]],
            index=pd.Index([1, 2], name="From Regime"),
            columns=[1, 2],
        ),
    )
    monkeypatch.setattr(
        analyticstool,
        "build_regime_duration_table",
        lambda *_args, **_kwargs: pd.DataFrame([{"Regime": 1, "Runs": 1, "Current Run Length": 2}]),
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
        False,
        63,
        "none",
        "scaled_identity",
        "Factor_X",
        5,
        "raw",
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
    assert "Regime - Settings" in xl.sheet_names
    assert "Regime - Statistics" in xl.sheet_names
    assert "Regime - Timeline" in xl.sheet_names
    assert "Regime - Transition" in xl.sheet_names
    assert "Regime - Duration" in xl.sheet_names
    assert "Regime - Conditioned" not in xl.sheet_names
    regime_sheet_positions = {name: xl.sheet_names.index(name) for name in xl.sheet_names if name.startswith("Regime - ")}
    assert regime_sheet_positions["Regime - Settings"] < regime_sheet_positions["Regime - Statistics"]
    assert regime_sheet_positions["Regime - Statistics"] < regime_sheet_positions["Regime - Timeline"]
    assert regime_sheet_positions["Regime - Timeline"] < regime_sheet_positions["Regime - Transition"]
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
    monkeypatch.setattr(
        analyticstool,
        "calculate_statistics_cached",
        lambda *_args, **_kwargs: [
            {"Series": "Asset_A", "Cumulative Return": 0.1},
            {"Series": "Asset_B", "Cumulative Return": 0.05},
        ],
    )
    monkeypatch.setattr(analyticstool, "calculate_rolling_returns", lambda *_args, **_kwargs: returns_df.copy())
    monkeypatch.setattr(
        analyticstool,
        "calculate_calendar_year_returns",
        lambda *_args, **_kwargs: pd.DataFrame({"Asset_A": [0.1], "Asset_B": [0.05]}, index=[2024]),
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
        False,
        63,
        "ledoit_wolf",
        "scaled_identity",
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

    pd.testing.assert_frame_equal(corr_df, returns_df.corr(), check_names=False)
    pd.testing.assert_frame_equal(cov_df, returns_df.cov(), check_names=False)


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
    draft, select_value, msg, color, hide = analyticstool.at_reset_regime_definition_draft(1, "db::DBRegime")
    assert draft["DraftMode"] == "new"
    assert select_value is None
    assert color == "blue"
    assert hide is False
    assert "New session regime draft started." in msg

    monkeypatch.setattr(analyticstool, "callback_context", type("Ctx", (), {"triggered_id": "at-regime-def-select"})())
    draft2, select_value2, msg2, color2, hide2 = analyticstool.at_reset_regime_definition_draft(0, None)
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
    out = analyticstool.at_manage_regime_definitions(
        None,
        None,
        None,
        1,
        None,
        draft,
        [],
        [db_def],
        True,
        {"role": "Admin", "username": "tester"},
    )

    local_rows = out[0]
    assert isinstance(local_rows, list)
    assert any(str(item.get("RegimeName")) == "SessionRegime" for item in local_rows)
    assert out[4] == "def::SessionRegime"
    assert "Session regime selected for analysis." in out[6]


def test_use_regime_keeps_db_selection_when_unchanged(monkeypatch, page_modules):
    analyticstool, _ = page_modules
    db_def = _db_regime_definition(name="DBRegime")
    draft = analyticstool._regime_definition_to_draft(db_def, "db")
    draft["DraftMode"] = "db"

    monkeypatch.setattr(analyticstool, "callback_context", type("Ctx", (), {"triggered_id": "at-regime-def-use-btn"})())
    out = analyticstool.at_manage_regime_definitions(
        None,
        None,
        None,
        1,
        None,
        draft,
        [],
        [db_def],
        True,
        {"role": "Admin", "username": "tester"},
    )

    assert out[0] is no_update
    assert out[4] == "def::DBRegime"
    assert "Database regime selected for analysis." in out[6]


def test_use_regime_blocks_db_name_collision_for_edited_db_draft(monkeypatch, page_modules):
    analyticstool, _ = page_modules
    db_def = _db_regime_definition(name="DBRegime", description="original")
    draft = analyticstool._regime_definition_to_draft(db_def, "db")
    draft["DraftMode"] = "db"
    draft["Description"] = "edited"
    draft["sync_origin"] = "form"

    monkeypatch.setattr(analyticstool, "callback_context", type("Ctx", (), {"triggered_id": "at-regime-def-use-btn"})())
    out = analyticstool.at_manage_regime_definitions(
        None,
        None,
        None,
        1,
        None,
        draft,
        [],
        [db_def],
        True,
        {"role": "Admin", "username": "tester"},
    )

    assert out[0] is no_update
    assert out[6].startswith("Rename the regime to create a session copy")
    assert out[7] == "orange"


def test_preload_factor_and_regime_definitions_on_page_load(monkeypatch, page_modules):
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

def test_regime_definition_modal_hides_return_basis_control(page_modules):
    analyticstool, _ = page_modules
    modal = _find_component_by_id(analyticstool.layout, "at-regime-def-modal")
    assert modal is not None
    assert _find_component_by_id(modal, "at-regime-def-return-basis") is None


def test_update_regime_analysis_renders_content(monkeypatch, page_modules):
    analyticstool, _ = page_modules
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    states = pd.Series([1, 1, 2, 2, 3, 3], index=idx, dtype="Int64", name="Regime")
    returns_df = pd.DataFrame({"Asset_A": [0.01, -0.005, 0.02, 0.0, 0.01, -0.01]}, index=idx)

    monkeypatch.setattr(
        analyticstool,
        "compute_regime_assignments",
        lambda *_args, **_kwargs: (
            states,
            {"method_type": 3, "num_regimes": 3, "observations": 6, "warning": None},
        ),
    )
    monkeypatch.setattr(analyticstool, "get_working_returns", lambda *_args, **_kwargs: returns_df.copy())

    warning, content = analyticstool.update_regime_analysis(
        "regime_analysis",
        "def::SavedRegime",
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


def test_help_modal_mentions_factor_analysis(page_modules):
    analyticstool, _ = page_modules
    modal = _find_component_by_id(analyticstool.layout, "at-help-modal")
    assert modal is not None

    text_blob = " ".join(_collect_component_text(modal)).lower()
    assert "basic guide" in text_blob
    assert "advanced guide" in text_blob
    assert "factor analysis page" in text_blob
    assert "ignores excess mode" in text_blob
    assert "regime analysis page" in text_blob
