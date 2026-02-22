from __future__ import annotations

from io import StringIO

import pandas as pd
import pytest
from dash import no_update
from dash.exceptions import PreventUpdate


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

    start, end, _style, _common_disabled, _daily_disabled, _max_disabled, range_store, ready = (
        analyticstool.initialize_date_range(
            "raw-json",
            "daily",
            ["Asset_A"],
            {"start": "2024-01-01", "end": "2024-12-31"},
            None,
            None,
        )
    )

    assert start == "2024-01-01"
    assert end == "2024-12-31"
    assert range_store is no_update
    assert ready is True


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
            120,
            key_weighted,
        )
        is no_update
    )


def test_on_modal_ok_does_not_emit_raw_data_when_unchanged(page_modules, raw_json):
    analyticstool, _ = page_modules

    result = analyticstool.on_modal_ok(
        1,
        ["Asset_A"],
        {},
        {},
        ["Asset_A"],
        [],
        raw_json,
        {},
    )

    assert result[6] is no_update


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

    out_json = result[0]
    out_periodicity = result[1]
    out_default_periodicity = result[3]

    out_df = pd.read_json(StringIO(out_json), orient="split")
    out_df.index = pd.to_datetime(out_df.index)

    assert out_periodicity == "monthly"
    assert out_default_periodicity == "monthly"
    assert out_df.index.is_month_end.all()
    assert pd.Timestamp("1976-07-30") not in out_df.index
    assert pd.Timestamp("1976-07-31") in out_df.index
