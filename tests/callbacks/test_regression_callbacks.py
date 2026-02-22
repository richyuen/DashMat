from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd
import pytest
from dash import no_update

from utils.regression import RegressionWindowResult
from utils.returns import df_to_json
from utils.shared_metrics import STATS_CONFIG


@pytest.fixture(scope="module")
def regression_page():
    import app  # noqa: F401
    import pages.regression as regression

    return regression


def _call_reg_run(regression_page, **overrides):
    params = {
        "n_clicks": 1,
        "raw_data": "raw-json",
        "periodicity": "daily",
        "x_series": ["X1"],
        "dep_var": "Y",
        "bench_assign": {},
        "ls_assign": {},
        "date_range": {"start": "2020-01-01", "end": "2020-06-30"},
        "vol_scaler": 0,
        "vol_scale_assign": {},
        "lag_assign": {},
        "min_beta_assign": {},
        "max_beta_assign": {},
        "enable_assign": {},
        "model": "ols",
        "reg_name": "TestRegression",
        "force_zero": False,
        "robust_se": False,
        "exp_wt": False,
        "halflife": 63,
        "window_type": "full",
        "window_size": 36,
        "opt_step": 1,
        "opt_step_unit": "months",
        "fill_in_sample": "off",
        "missing_data": "fill_na",
        "alpha": 1.0,
        "l1_ratio": 0.5,
        "arima_p": 0,
        "arima_d": 0,
        "arima_q": 0,
        "garch_p": 0,
        "garch_q": 0,
        "linear_constraints": None,
        "current_results": {},
    }
    params.update(overrides)
    return regression_page.reg_run_regression(**params)


def test_reg_run_regression_includes_run_level_arima_summary_and_per_var_bounds(monkeypatch, regression_page):
    idx = pd.date_range("2020-01-01", periods=6, freq="B")
    working_df = pd.DataFrame(
        {
            "Y": np.linspace(0.01, 0.06, len(idx)),
            "X1": np.linspace(0.0, 0.05, len(idx)),
            "X2": np.linspace(-0.02, 0.03, len(idx)),
        },
        index=idx,
    )

    monkeypatch.setattr(regression_page, "_reg_get_working_returns", lambda *_args, **_kwargs: working_df)

    wr = RegressionWindowResult(
        est_start=idx[0],
        est_end=idx[-1],
        apply_start=idx[0],
        apply_end=idx[-1],
        coefficients={"intercept": 0.1, "X1": 0.3, "X2": 0.2},
        p_values={"intercept": 0.2, "X1": 0.1, "X2": 0.1},
        diagnostics={"note": "ok"},
        n_obs=len(idx),
    )
    predicted = pd.Series(np.linspace(0.01, 0.06, len(idx)), index=idx, name="predicted")
    residuals = pd.Series(np.zeros(len(idx)), index=idx, name="residuals")

    captured = {}
    expected_summary = {"arima": {"order": (1, 0, 1), "aic": 1.0, "bic": 2.0, "params": {"ar.L1": 0.2}}}

    def _fake_run_regression(_y, _X, config):
        captured["config"] = config
        return [wr], predicted, residuals, expected_summary

    monkeypatch.setattr(regression_page, "run_regression", _fake_run_regression)

    new_results, _options, selected, status = _call_reg_run(
        regression_page,
        x_series=["X1", "X2"],
        model="constrained_ols",
        min_beta_assign={"X1": -0.1, "X2": 0.2},
        max_beta_assign={"X1": 0.4, "X2": 0.6},
        enable_assign={"X1": True, "X2": True},
        arima_p=1,
        arima_q=1,
    )

    cfg = captured["config"]
    assert cfg["min_beta_by_var"] == {"X1": -0.1, "X2": 0.2}
    assert cfg["max_beta_by_var"] == {"X1": 0.4, "X2": 0.6}
    assert cfg["min_beta"] == -0.1
    assert cfg["max_beta"] == 0.6

    entry = new_results[selected]
    assert entry["arima_garch_summary"] == expected_summary
    assert "arima_garch" not in (entry["window_results"][0].get("diagnostics") or {})
    assert "1 window(s)" in status


def test_reg_run_regression_persists_stats_inputs(monkeypatch, regression_page):
    idx = pd.date_range("2020-01-01", periods=6, freq="B")
    working_df = pd.DataFrame(
        {
            "Y": np.linspace(0.01, 0.06, len(idx)),
            "X1": np.linspace(0.0, 0.05, len(idx)),
        },
        index=idx,
    )
    monkeypatch.setattr(regression_page, "_reg_get_working_returns", lambda *_args, **_kwargs: working_df)

    wr = RegressionWindowResult(
        est_start=idx[0],
        est_end=idx[-1],
        apply_start=idx[0],
        apply_end=idx[-1],
        coefficients={"intercept": 0.1, "X1": 0.3},
        p_values={"intercept": 0.2, "X1": 0.1},
        diagnostics={"note": "ok"},
        n_obs=len(idx),
    )
    predicted = pd.Series(np.linspace(0.01, 0.06, len(idx)), index=idx, name="predicted")
    residuals = pd.Series(np.zeros(len(idx)), index=idx, name="residuals")
    monkeypatch.setattr(
        regression_page,
        "run_regression",
        lambda *_args, **_kwargs: ([wr], predicted, residuals, None),
    )

    date_range = {"start": "2020-01-01", "end": "2020-01-08"}
    new_results, _options, selected, _status = _call_reg_run(
        regression_page,
        bench_assign={"Y": "X1"},
        ls_assign={"Y": False},
        date_range=date_range,
        vol_scaler=12,
        vol_scale_assign={"Y": True, "X1": False},
    )

    entry = new_results[selected]
    assert entry["benchmark_assignments"] == {"Y": "X1"}
    assert entry["long_short_assignments"] == {"Y": False}
    assert entry["date_range"] == date_range
    assert entry["vol_scaler"] == 12
    assert entry["vol_scaling_assignments"] == {"Y": True, "X1": False}


def test_reg_run_regression_errors_when_dependent_variable_missing(regression_page):
    out = _call_reg_run(regression_page, dep_var=None)
    assert out[0] is no_update
    assert out[1] is no_update
    assert out[2] is no_update
    assert "dependent variable" in out[3].lower()


def test_reg_run_regression_errors_when_x_series_missing(regression_page):
    out = _call_reg_run(regression_page, x_series=[])
    assert out[0] is no_update
    assert out[1] is no_update
    assert out[2] is no_update
    assert "independent variable" in out[3].lower()


def test_reg_run_regression_errors_when_raw_data_missing(regression_page):
    out = _call_reg_run(regression_page, raw_data=None)
    assert out[0] is no_update
    assert out[1] is no_update
    assert out[2] is no_update
    assert "no data loaded" in out[3].lower()


def test_reg_run_regression_handles_blank_linear_constraints(monkeypatch, regression_page):
    idx = pd.date_range("2020-01-01", periods=80, freq="B")
    x1 = np.linspace(-0.05, 0.05, len(idx))
    x2 = np.linspace(0.03, -0.02, len(idx))
    y = 0.01 + 0.6 * x1 + 0.2 * x2
    working_df = pd.DataFrame({"Y": y, "X1": x1, "X2": x2}, index=idx)
    monkeypatch.setattr(regression_page, "_reg_get_working_returns", lambda *_args, **_kwargs: working_df)

    out = _call_reg_run(
        regression_page,
        model="constrained_ols",
        x_series=["X1", "X2"],
        enable_assign={"X1": True},
        linear_constraints=[{"X1": "", "X2": "", "Min": "", "Max": " "}],
    )

    assert out[0] is not no_update
    assert out[1] is not no_update
    assert out[2] is not no_update
    assert "regression error" not in out[3].lower()


def test_reg_open_db_add_modal_uses_helper(monkeypatch, regression_page):
    expected = (True, [{"value": "IDX_A", "label": "Index A"}], [])
    monkeypatch.setattr(regression_page, "compute_open_db_add_modal", lambda *_args, **_kwargs: expected)
    assert regression_page.reg_open_db_add_modal(1) == expected


def test_reg_add_series_from_database_imports_and_updates_stores(monkeypatch, regression_page):
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    new_df = pd.DataFrame({"IDX_A": [0.01, 0.0, 0.002, -0.003, 0.004]}, index=idx)
    monkeypatch.setattr(
        regression_page,
        "load_cma_returns_for_benches_with_meta",
        lambda *_args, **_kwargs: (new_df, {"IDX_A": {"starts_daily": True}}),
    )

    raw, orig_p, p_value, p_sync, opened, selected, err_text, err_hide = (
        regression_page.reg_add_series_from_database(1, ["IDX_A"], None, None)
    )

    assert isinstance(raw, str)
    assert orig_p == "daily"
    assert p_value == "daily"
    assert p_sync == "daily"
    assert opened is False
    assert selected == []
    assert err_hide is True
    assert err_text is no_update


def test_reg_add_series_from_database_rejects_duplicates(monkeypatch, regression_page):
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    existing_df = pd.DataFrame({"IDX_A": [0.01, 0.0, -0.01]}, index=idx)
    existing_raw = df_to_json(existing_df)
    monkeypatch.setattr(
        regression_page,
        "load_cma_returns_for_benches_with_meta",
        lambda *_args, **_kwargs: (existing_df.copy(), {"IDX_A": {"starts_daily": True}}),
    )

    raw, orig_p, p_value, p_sync, opened, selected, err_text, err_hide = (
        regression_page.reg_add_series_from_database(1, ["IDX_A"], existing_raw, "daily")
    )

    assert raw is no_update
    assert orig_p is no_update
    assert p_value is no_update
    assert p_sync is no_update
    assert opened is True
    assert selected is no_update
    assert "duplicate" in str(err_text).lower()
    assert err_hide is False


def test_reg_toggle_welcome_uses_original_periodicity(monkeypatch, regression_page):
    captured = {}

    def _fake_get_available_periodicities(original_periodicity):
        captured["arg"] = original_periodicity
        return [
            {"value": "daily", "label": "Daily"},
            {"value": "monthly", "label": "Monthly"},
        ]

    monkeypatch.setattr(regression_page, "get_available_periodicities", _fake_get_available_periodicities)
    welcome_style, main_style, options, value = regression_page.reg_toggle_welcome("raw", "daily", "monthly")

    assert captured["arg"] == "daily"
    assert welcome_style["display"] == "none"
    assert main_style["display"] == "flex"
    assert options == [{"value": "daily", "label": "Daily"}, {"value": "monthly", "label": "Monthly"}]
    assert value == "monthly"


def test_reg_sync_grid_to_temp_handles_list_cell_change_payload(regression_page):
    row_data = [
        {"Series": "A", "Y": True, "X": True},
        {"Series": "B", "Y": True, "X": True},
    ]
    cell_change = [{"colId": "Y", "rowIndex": 1}]

    out = regression_page.reg_sync_grid_to_temp(cell_change, None, row_data, None)
    new_x, new_dep = out[0], out[1]

    assert new_dep == "B"
    assert new_x == ["A", "B"]


def test_reg_series_grid_uses_stable_checkbox_interaction_options(regression_page):
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    raw = df_to_json(pd.DataFrame({"A": [0.01, 0.0, -0.01], "B": [0.0, 0.01, 0.02]}, index=idx))

    children, _order = regression_page.reg_update_series_grid(
        raw,
        ["A"],
        ["A", "B"],
        [],
        {},
        {},
        {},
        "A",
        {},
        {},
        {},
        {},
    )

    grid = children[0]
    opts = getattr(grid, "dashGridOptions", {}) or {}
    assert opts.get("suppressMovableColumns") is True
    assert opts.get("stopEditingWhenCellsLoseFocus") is True
    assert opts.get("singleClickEdit") is True

    cols = getattr(grid, "columnDefs", []) or []
    x_col = next((c for c in cols if c.get("field") == "X"), None)
    scale_col = next((c for c in cols if c.get("field") == "ScaleVol"), None)
    assert x_col is not None
    assert scale_col is not None
    assert x_col.get("cellRenderer") == "agCheckboxCellRenderer"
    assert scale_col.get("cellRenderer") == "agCheckboxCellRenderer"


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


def test_reg_toggle_welcome_no_data_shows_top_aligned_welcome(regression_page):
    welcome_style, main_style, options, value = regression_page.reg_toggle_welcome(None, None, None)

    assert welcome_style == {"display": "block"}
    assert main_style["display"] == "none"
    assert options == [{"value": "daily", "label": "Daily"}]
    assert value == "daily"


def test_reg_help_modal_covers_three_sections_and_model_explainers(regression_page):
    modal = regression_page.build_reg_help_modal()
    text_blob = " ".join(_collect_component_text(modal)).lower()

    required_phrases = [
        "basic guide",
        "advanced guide",
        "model deep dive",
        "what it is: baseline linear regression with unconstrained coefficients",
        "what it is: ols with per-variable beta limits and optional linear constraints",
        "what it is: constrained style decomposition where exposures are bounded and sum to one",
        "what it is: l2-regularized regression that shrinks coefficients toward zero",
        "what it is: l1-regularized regression that can zero out coefficients",
        "what it is: combined l1 and l2 regularization",
        "series selection modal",
        "periodicity",
        "vol scaler",
        "date range",
        "common range",
        "max range",
        "fill in-sample",
        "linear constraints",
        "run regression",
        "anova, rolling summary, rolling, weights, statistics, returns, growth of $1, calendar year, drawdown, and scatter",
        "save session",
        "load session",
        "download excel",
        "clear server cache",
        "arima(p,d,q)",
        "garch(p,q)",
        "arima and garch residual overlay",
    ]
    for phrase in required_phrases:
        assert phrase in text_blob


def test_reg_render_statistics_uses_current_stats_signature_and_list_shape(monkeypatch, regression_page):
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    predicted = pd.DataFrame({"Predicted": [0.01, -0.005, 0.003, 0.002]}, index=idx)
    entry = {"periodicity": "daily", "predicted_json": df_to_json(predicted)}
    captured = {}

    def _fake_stats(*args, **kwargs):
        captured["args"] = args
        return [
            {
                "Series": "Predicted",
                "Cumulative Return": 0.0100,
                "Annualized Return": 0.0300,
            }
        ]

    monkeypatch.setattr(regression_page, "calculate_statistics_cached", _fake_stats)
    comp = regression_page.reg_render_statistics("R1", {"R1": entry})

    assert captured["args"][5] == "null"
    assert captured["args"][6] == 0
    assert getattr(comp, "rowData", None)
    assert any(row.get("Statistic") == "Cumulative Return" for row in comp.rowData)


def test_reg_render_statistics_uses_full_stats_config_rows(monkeypatch, regression_page):
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    predicted = pd.DataFrame({"Predicted": [0.01, -0.005, 0.003, 0.002]}, index=idx)
    entry = {"periodicity": "daily", "predicted_json": df_to_json(predicted)}

    monkeypatch.setattr(
        regression_page,
        "calculate_statistics_cached",
        lambda *_args, **_kwargs: [{"Series": "Predicted", "Start Date": "2024-01-01", "End Date": "2024-01-04"}],
    )
    comp = regression_page.reg_render_statistics("R1", {"R1": entry})

    stat_names = [row.get("Statistic") for row in getattr(comp, "rowData", [])]
    expected = [name for name, _fmt in STATS_CONFIG]
    assert stat_names[: len(expected)] == expected


def test_reg_render_statistics_includes_actual_predicted_residual_when_available(monkeypatch, regression_page):
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    predicted = pd.DataFrame({"predicted": [0.01, -0.005, 0.003, 0.002, -0.001]}, index=idx)
    residuals = pd.DataFrame({"residuals": [0.002, -0.001, 0.000, 0.001, -0.002]}, index=idx)
    entry = {
        "periodicity": "daily",
        "predicted_json": df_to_json(predicted),
        "residuals_json": df_to_json(residuals),
    }

    def _fake_stats(_json_str, _periodicity, selected_series, *_args, **_kwargs):
        assert tuple(selected_series) == ("Predicted", "Actual (Y)", "Residual")
        return [
            {"Series": "Actual (Y)", "Start Date": "2024-01-01", "End Date": "2024-01-05"},
            {"Series": "Predicted", "Start Date": "2024-01-01", "End Date": "2024-01-05"},
            {"Series": "Residual", "Start Date": "2024-01-01", "End Date": "2024-01-05"},
        ]

    monkeypatch.setattr(regression_page, "calculate_statistics_cached", _fake_stats)
    comp = regression_page.reg_render_statistics("R1", {"R1": entry})

    col_fields = [c.get("field") for c in getattr(comp, "columnDefs", [])]
    assert "Actual (Y)" in col_fields
    assert "Predicted" in col_fields
    assert "Residual" in col_fields


def test_reg_render_statistics_combines_run_series_and_model_output_stats(monkeypatch, regression_page):
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    raw_df = pd.DataFrame(
        {
            "SPX_TRIndex": [0.01, -0.005, 0.003, 0.002, -0.001],
            "EM_TRIndex": [0.008, -0.004, 0.002, 0.001, -0.002],
            "EAFE_TRIndex": [0.007, -0.003, 0.001, 0.0005, -0.0015],
        },
        index=idx,
    )
    entry = {
        "periodicity": "daily",
        "dependent_var": "SPX_TRIndex",
        "independent_vars": ["EM_TRIndex", "EAFE_TRIndex"],
        "benchmark_assignments": {},
        "long_short_assignments": {},
        "date_range": {"start": "2024-01-01", "end": "2024-01-05"},
        "vol_scaler": 0,
        "vol_scaling_assignments": {},
        "predicted_json": df_to_json(pd.DataFrame({"predicted": raw_df["SPX_TRIndex"]}, index=idx)),
        "residuals_json": df_to_json(pd.DataFrame({"residuals": np.zeros(len(idx))}, index=idx)),
    }
    calls = []

    def _fake_stats(*args, **kwargs):
        calls.append(args)
        selected = tuple(args[2])
        if selected == ("Predicted", "Actual (Y)", "EM_TRIndex", "EAFE_TRIndex", "Residual"):
            return [
                {"Series": "Actual (Y)", "Start Date": "2024-01-01", "End Date": "2024-01-05", "Cumulative Return": 0.009},
                {"Series": "Predicted", "Start Date": "2024-01-01", "End Date": "2024-01-05", "Cumulative Return": 0.007},
                {"Series": "EM_TRIndex", "Start Date": "2024-01-01", "End Date": "2024-01-05", "Cumulative Return": 0.005},
                {"Series": "EAFE_TRIndex", "Start Date": "2024-01-01", "End Date": "2024-01-05", "Cumulative Return": 0.004},
                {"Series": "Residual", "Start Date": "2024-01-01", "End Date": "2024-01-05", "Cumulative Return": 0.002},
            ]
        return []

    monkeypatch.setattr(regression_page, "calculate_statistics_cached", _fake_stats)
    comp = regression_page.reg_render_statistics("R1", {"R1": entry}, df_to_json(raw_df), {})

    selected_payloads = [tuple(call[2]) for call in calls]
    assert ("Predicted", "Actual (Y)", "EM_TRIndex", "EAFE_TRIndex", "Residual") in selected_payloads
    run_call = calls[0]
    assert run_call[5] == "null"

    col_fields = [c.get("field") for c in getattr(comp, "columnDefs", [])]
    assert col_fields[:6] == ["Statistic", "Predicted", "Actual (Y)", "EM_TRIndex", "EAFE_TRIndex", "Residual"]
    assert "SPX_TRIndex" not in col_fields


def test_reg_build_display_series_clips_x_to_model_window_for_rolling(regression_page):
    full_idx = pd.date_range("2024-01-01", periods=8, freq="B")
    model_idx = full_idx[3:]
    raw_df = pd.DataFrame(
        {
            "Y": [0.01, 0.02, -0.01, 0.00, 0.01, -0.02, 0.03, 0.01],
            "X1": [0.02, -0.01, 0.00, 0.01, 0.00, 0.02, -0.01, 0.03],
            "X2": [0.01, 0.00, -0.02, 0.02, 0.01, -0.01, 0.00, 0.01],
        },
        index=full_idx,
    )
    predicted = pd.DataFrame({"predicted": [0.001, 0.002, 0.003, 0.004, 0.005]}, index=model_idx)
    residuals = pd.DataFrame({"residuals": [0.0, -0.001, 0.0, 0.001, -0.001]}, index=model_idx)
    entry = {
        "periodicity": "daily",
        "dependent_var": "Y",
        "independent_vars": ["X1", "X2"],
        "benchmark_assignments": {},
        "long_short_assignments": {},
        "date_range": {"start": "2024-01-01", "end": "2024-01-31"},
        "vol_scaler": 0,
        "vol_scaling_assignments": {},
        "config": {"window_type": "rolling"},
        "predicted_json": df_to_json(predicted),
        "residuals_json": df_to_json(residuals),
    }

    display_df, ordered_cols = regression_page._reg_build_display_series(entry, df_to_json(raw_df))

    assert ordered_cols == ["Predicted", "Actual (Y)", "X1", "X2", "Residual"]
    assert list(display_df.index) == list(model_idx)
    assert list(display_df["X1"].index) == list(model_idx)


def test_reg_sync_name_with_model_uses_model_defaults(regression_page):
    assert regression_page.reg_sync_name_with_model("ols") == "OLS"
    assert regression_page.reg_sync_name_with_model("ridge") == "Ridge"
    assert regression_page.reg_sync_name_with_model("style_analysis") == "Style Analysis"
    assert regression_page.reg_sync_name_with_model("unknown_model") == "Regression"


def test_reg_download_excel_matches_tab_order_and_settings_sheet(monkeypatch, regression_page):
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    predicted = pd.DataFrame({"Predicted": [0.01, -0.002, 0.003, 0.004, -0.001]}, index=idx)
    residuals = pd.DataFrame({"Residual": [0.001, -0.001, 0.0, 0.001, -0.002]}, index=idx)

    window_result = {
        "est_start": idx[0],
        "est_end": idx[-1],
        "apply_start": idx[0],
        "apply_end": idx[-1],
        "r_squared": 0.82,
        "adj_r_squared": 0.76,
        "residual_std": 0.014,
        "n_obs": len(idx),
        "coefficients": {"intercept": 0.001, "X1": 0.95},
        "p_values": {"intercept": 0.21, "X1": 0.01},
        "anova_table": {
            "df_model": 1,
            "df_resid": 3,
            "ss_model": 0.45,
            "ms_model": 0.45,
            "F_stat": 9.0,
            "F_pvalue": 0.05,
            "ss_resid": 0.15,
            "ms_resid": 0.05,
            "ss_total": 0.60,
        },
        "diagnostics": {
            "std_errors": {"intercept": 0.05, "X1": 0.12},
            "t_stats": {"intercept": 2.0, "X1": 7.9},
            "ci_low": {"intercept": -0.09, "X1": 0.70},
            "ci_high": {"intercept": 0.11, "X1": 1.20},
            "durbin_watson": 2.10,
            "aic": 12.3,
            "bic": 14.2,
            "vif": {"X1": 1.1},
        },
        "oos_metrics": {"oos_r2": 0.61, "oos_rmse": 0.02, "oos_mae": 0.01},
    }

    results = {
        "R1": {
            "periodicity": "daily",
            "dependent_var": "Y",
            "independent_vars": ["X1"],
            "config": {
                "model": "ols",
                "window_type": "rolling",
                "window_size": 24,
                "opt_step": 1,
                "opt_step_unit": "months",
                "fill_in_sample": True,
                "missing_data": "fill_na",
                "force_zero_intercept": False,
                "robust_se": True,
                "exp_wt": False,
                "halflife": 63,
                "alpha": 1.0,
                "l1_ratio": 0.5,
            },
            "window_results": [window_result],
            "date_range": {"start": "2024-01-01", "end": "2024-01-31"},
            "vol_scaler": 0,
            "benchmark_assignments": {},
            "long_short_assignments": {},
            "vol_scaling_assignments": {},
            "predicted_json": df_to_json(predicted),
            "residuals_json": df_to_json(residuals),
        }
    }

    monkeypatch.setattr(
        regression_page,
        "calculate_statistics_cached",
        lambda *_args, **_kwargs: [
            {"Series": "Predicted", "Cumulative Return": 0.015, "Annualized Return": 0.20},
            {"Series": "Actual (Y)", "Cumulative Return": 0.014, "Annualized Return": 0.18},
            {"Series": "Residual", "Cumulative Return": -0.001, "Annualized Return": -0.01},
        ],
    )
    monkeypatch.setattr(
        regression_page,
        "calculate_rolling_returns",
        lambda *_args, **_kwargs: pd.DataFrame(
            {
                "Predicted": [0.10],
                "Actual (Y)": [0.08],
                "Residual": [0.02],
            },
            index=[pd.Timestamp("2024-01-31")],
        ),
    )
    monkeypatch.setattr(
        regression_page,
        "create_monthly_view",
        lambda *_args, **_kwargs: ([], [{"Year_Label": "2024", "Jan": 0.01, "YTD": 0.01}]),
    )
    monkeypatch.setattr(
        regression_page,
        "calculate_drawdown",
        lambda *_args, **_kwargs: pd.DataFrame(
            {
                "Predicted": [0.0, -0.02],
                "Actual (Y)": [0.0, -0.03],
                "Residual": [0.0, -0.01],
            },
            index=[pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-31")],
        ),
    )
    monkeypatch.setattr(regression_page.dcc, "send_bytes", lambda b, filename: {"content": b, "filename": filename})

    payload = regression_page.reg_download_excel(
        1,
        results,
        None,
        "R1",
        0,
        "1y",
        "annualized",
        "total_return",
        "monthly",
        "Predicted",
    )

    workbook = BytesIO(payload["content"])
    xl = pd.ExcelFile(workbook)
    assert xl.sheet_names == [
        "Settings",
        "ANOVA",
        "Rolling Summary",
        "Weights",
        "Statistics",
        "Returns",
        "Rolling",
        "Calendar Year",
        "Growth of $1",
        "Drawdown",
    ]

    settings_df = pd.read_excel(BytesIO(payload["content"]), sheet_name="Settings")
    settings_map = dict(zip(settings_df["Parameter"], settings_df["Value"]))
    assert settings_map["Result Name"] == "R1"
    assert settings_map["Model"] == "ols"
