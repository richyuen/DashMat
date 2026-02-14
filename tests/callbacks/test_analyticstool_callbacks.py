from __future__ import annotations

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

    assert analyticstool.update_date_range_store("2024-01-01", "2024-12-31") == {
        "start": "2024-01-01",
        "end": "2024-12-31",
    }
    assert analyticstool.update_date_range_store("2024-01-01", None) is no_update


def test_update_statistics_transposes_series_into_columns(monkeypatch, page_modules):
    analyticstool, _ = page_modules

    def _fake_stats(*_args, **_kwargs):
        return [
            {"Series": "Asset_A", "Cumulative Return": 0.10},
            {"Series": "Asset_B", "Cumulative Return": 0.20},
        ]

    monkeypatch.setattr(analyticstool, "calculate_statistics_cached", _fake_stats)

    column_defs, row_data = analyticstool.update_statistics(
        "raw-json",
        "daily",
        ["Asset_A", "Asset_B"],
        {},
        {},
        None,
        0,
        {},
        None,
    )

    assert column_defs[0]["field"] == "Statistic"
    assert {c["field"] for c in column_defs[1:]} == {"Asset_A", "Asset_B"}
    cum_row = next(row for row in row_data if row["Statistic"] == "Cumulative Return")
    assert cum_row["Asset_A"] == pytest.approx(0.10)
    assert cum_row["Asset_B"] == pytest.approx(0.20)


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
            None,
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
        None,
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
        None,
        0,
        {},
    )

    assert column_defs[0]["field"] == "Date"
    assert column_defs[1]["field"] == "Asset_A"
    assert row_data[1]["Asset_A"] == pytest.approx(-0.03)


def test_update_correlogram_meta_returns_no_update_when_not_active(page_modules):
    analyticstool, _ = page_modules
    assert analyticstool.update_correlogram_meta(["Asset_A", "Asset_B"], "growth") is no_update
    assert analyticstool.update_correlogram_meta(["Asset_A", "Asset_B"], "correlogram") == {"num_series": 2}
