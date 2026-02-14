from __future__ import annotations

from io import BytesIO
from io import StringIO
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from dash.exceptions import PreventUpdate

from utils.returns import df_to_json


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
        "raw-json",
        "daily",
        {},
        {},
        None,
        0,
        {},
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

    column_defs, row_data = portopt.po_render_statistics(results, "statistics", ["P1", "P2"], None, "daily")

    assert column_defs[0]["field"] == "Statistic"
    assert {c["field"] for c in column_defs[1:]} == {"P1", "P2"}
    row = next(r for r in row_data if r["Statistic"] == "Cumulative Return")
    assert row["P1"] == pytest.approx(0.1)
    assert row["P2"] == pytest.approx(0.2)


def test_po_render_returns_builds_returns_grid(page_modules):
    _, portopt = page_modules

    s1 = pd.Series([0.01, 0.02], index=pd.to_datetime(["2024-01-01", "2024-01-02"]))
    results = {"P1": {"returns_json": s1.to_json(date_format="iso")}}

    column_defs, row_data = portopt.po_render_returns(results, "returns", ["P1"])
    assert column_defs[0]["field"] == "Date"
    assert column_defs[1]["field"] == "P1"
    assert row_data[0]["Date"] == "2024-01-01"


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

    column_defs, row_data = portopt.po_render_turnover_table("P1", results, "turnover", "table")
    assert column_defs[0]["field"] == "Rebalance Date"
    assert row_data[0]["Turnover"] == pytest.approx(0.1)


def test_po_sync_results_with_raw_data_prunes_missing_portfolios(page_modules, raw_json):
    _, portopt = page_modules
    df = pd.read_json(StringIO(raw_json), orient="split")
    df["KeepMe"] = 0.0
    raw_with_portfolio = df_to_json(df)
    results = {"KeepMe": {"x": 1}, "DropMe": {"x": 2}}

    pruned = portopt.po_sync_results_with_raw_data(raw_with_portfolio, 1, results)
    assert pruned == {"KeepMe": {"x": 1}}


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
            {}, {}, {}, False, 63, "MyPortfolio", "full", 252, 21, "periods",
            "risk_parity", "fill_na", "off", {}, [],
            {}, {}, [], 0.05, "maximize_sharpe",
            {}, {}, "ret_cov", [], None,
        )

    # Now force callback path and verify returned error payload.
    result = portopt.po_run_optimization(
        1, "raw", "daily", "daily", ["Asset_A", "Asset_B"], {}, {}, {}, None, 0, {},
        {}, {}, {}, False, 63, "MyPortfolio", "full", 252, 21, "periods",
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


def test_po_update_frontier_risk_measure_options_restricts_ex_ante(page_modules):
    _, portopt = page_modules
    results = {"P1": {"config": {"model": "ex_ante_mv"}}}

    options, value = portopt.po_update_frontier_risk_measure_options("P1", results, "CVaR")
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
        "raw-json",
        "daily",
        {},
        {},
        0,
        {},
        {},
        None,
        [],
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
    assert xl.sheet_names == [
        "Weights",
        "Turnover",
        "Statistics",
        "Returns",
        "Growth of $1",
        "Attribution",
        "Risk",
        "Frontier",
    ]

    frontier_df = pd.read_excel(BytesIO(payload["content"]), sheet_name="Frontier")
    assert "Wt_Asset_A" in frontier_df.columns
    assert "Sharpe Ratio" in frontier_df.columns
    assert "Frontier Point" in set(frontier_df["Type"])
