from __future__ import annotations

from io import BytesIO, StringIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from dash import no_update
from dash.exceptions import PreventUpdate

from utils.regression import RegressionWindowResult
from utils.returns import build_raw_data_metadata, df_to_json
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


def test_reg_run_regression_errors_when_x_series_missing_for_model_that_requires_x(regression_page):
    out = _call_reg_run(regression_page, x_series=[], model="style_analysis")
    assert out[0] is no_update
    assert out[1] is no_update
    assert out[2] is no_update
    assert "independent variable" in out[3].lower()


def test_reg_run_regression_allows_ols_intercept_only_when_x_missing(monkeypatch, regression_page):
    idx = pd.date_range("2020-01-01", periods=6, freq="B")
    working_df = pd.DataFrame({"Y": np.linspace(0.01, 0.06, len(idx))}, index=idx)
    monkeypatch.setattr(regression_page, "_reg_get_working_returns", lambda *_args, **_kwargs: working_df)

    wr = RegressionWindowResult(
        est_start=idx[0],
        est_end=idx[-1],
        apply_start=idx[0],
        apply_end=idx[-1],
        coefficients={"intercept": 0.02},
        p_values={"intercept": 0.1},
        diagnostics={"note": "ok"},
        n_obs=len(idx),
    )
    predicted = pd.Series(np.linspace(0.01, 0.06, len(idx)), index=idx, name="predicted")
    residuals = pd.Series(np.zeros(len(idx)), index=idx, name="residuals")

    captured = {}

    def _fake_run_regression(_y, _X, config):
        captured["x_columns"] = list(_X.columns)
        captured["config"] = config
        return [wr], predicted, residuals, {"arima": {"order": (1, 0, 0)}}

    monkeypatch.setattr(regression_page, "run_regression", _fake_run_regression)

    out = _call_reg_run(
        regression_page,
        model="ols",
        x_series=[],
        force_zero=False,
        linear_constraints=[{"X1": 1.0, "Min": 0.0}],
    )

    new_results, _options, selected, status = out
    assert new_results is not no_update
    assert selected in new_results
    assert "window(s)" in status
    assert captured["x_columns"] == []
    assert captured["config"]["linear_constraints"] is None
    assert new_results[selected]["independent_vars"] == []


def test_reg_run_regression_rejects_intercept_only_when_force_zero_enabled(regression_page):
    out = _call_reg_run(regression_page, x_series=[], model="ols", force_zero=True)
    assert out[0] is no_update
    assert out[1] is no_update
    assert out[2] is no_update
    assert "force zero intercept" in out[3].lower()


def test_reg_run_regression_rejects_self_lag_without_positive_lag(monkeypatch, regression_page):
    idx = pd.date_range("2020-01-01", periods=6, freq="B")
    working_df = pd.DataFrame({"Y": np.linspace(0.01, 0.06, len(idx))}, index=idx)
    monkeypatch.setattr(regression_page, "_reg_get_working_returns", lambda *_args, **_kwargs: working_df)

    out = _call_reg_run(
        regression_page,
        model="ols",
        dep_var="Y",
        x_series=["Y"],
        lag_assign={"Y": 0},
    )

    assert out[0] is no_update
    assert out[1] is no_update
    assert out[2] is no_update
    assert "lag to at least 1" in out[3].lower()


def test_reg_run_regression_supports_self_lag_with_display_labels(monkeypatch, regression_page):
    idx = pd.date_range("2020-01-01", periods=6, freq="B")
    working_df = pd.DataFrame({"Y": np.linspace(0.01, 0.06, len(idx))}, index=idx)
    monkeypatch.setattr(regression_page, "_reg_get_working_returns", lambda *_args, **_kwargs: working_df)

    wr = RegressionWindowResult(
        est_start=idx[1],
        est_end=idx[-1],
        apply_start=idx[1],
        apply_end=idx[-1],
        coefficients={"intercept": 0.02, "Y": 0.5},
        p_values={"intercept": 0.1, "Y": 0.05},
        diagnostics={
            "std_errors": {"intercept": 0.01, "Y": 0.2},
            "t_stats": {"intercept": 2.0, "Y": 2.5},
            "ci_low": {"intercept": 0.0, "Y": 0.1},
            "ci_high": {"intercept": 0.03, "Y": 0.9},
            "vif": {"Y": 1.0},
        },
        n_obs=5,
    )
    predicted = pd.Series(np.linspace(0.02, 0.06, len(idx) - 1), index=idx[1:], name="predicted")
    residuals = pd.Series(np.zeros(len(idx) - 1), index=idx[1:], name="residuals")

    captured = {}

    def _fake_run_regression(_y, _X, config):
        captured["x_columns"] = list(_X.columns)
        captured["config"] = config
        return [wr], predicted, residuals, None

    monkeypatch.setattr(regression_page, "run_regression", _fake_run_regression)

    new_results, _options, selected, status = _call_reg_run(
        regression_page,
        model="ols",
        dep_var="Y",
        x_series=["Y"],
        lag_assign={"Y": 1},
    )

    assert "window(s)" in status
    assert captured["x_columns"] == ["Y"]
    assert captured["config"]["lag_config"] == {"Y": 1}
    assert captured["config"]["lag_config_display"] == {"Y (lag 1)": 1}

    entry = new_results[selected]
    assert entry["independent_vars"] == ["Y (lag 1)"]
    assert entry["independent_vars_internal"] == ["Y"]
    assert entry["effective_date_range"] == {"start": "2020-01-02", "end": "2020-01-08"}

    wr_saved = entry["window_results"][0]
    assert "Y (lag 1)" in wr_saved["coefficients"]
    assert "Y (lag 1)" in wr_saved["p_values"]
    assert wr_saved["diagnostics"]["std_errors"]["Y (lag 1)"] == pytest.approx(0.2)


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


def test_reg_save_series_to_shared_data_saves_predicted_and_updates_result(regression_page):
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    raw_df = pd.DataFrame({"Asset_A": [0.01, 0.02, 0.03]}, index=idx)
    results = {
        "R1": {
            "predicted_json": df_to_json(pd.DataFrame({"predicted": [0.001, 0.002, 0.003]}, index=idx)),
            "periodicity": "daily",
            "config": {"periodicity": "daily"},
            "saved_series_name": None,
        }
    }

    new_results, new_raw, saved_store, status = regression_page.reg_save_series_to_shared_data(
        1,
        "R1",
        results,
        df_to_json(raw_df),
        "daily",
        {},
    )

    raw_after = pd.read_json(StringIO(new_raw), orient="split")
    assert "R1" in raw_after.columns
    assert new_results["R1"]["saved_series_name"] == "R1"
    assert saved_store["R1"]["origin_page"] == "regression"
    assert status == "Saved as R1."


def test_reg_save_series_to_shared_data_overwrites_existing_saved_name(regression_page):
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    raw_df = pd.DataFrame({"R1": [0.5, 0.5, 0.5]}, index=idx)
    results = {
        "R1": {
            "predicted_json": df_to_json(pd.DataFrame({"predicted": [0.001, 0.002, 0.003]}, index=idx)),
            "periodicity": "daily",
            "config": {"periodicity": "daily"},
            "saved_series_name": "R1",
        }
    }

    _new_results, new_raw, saved_store, status = regression_page.reg_save_series_to_shared_data(
        1,
        "R1",
        results,
        df_to_json(raw_df),
        "daily",
        {"R1": {"origin_page": "regression", "origin_result": "R1", "series_type": "predicted"}},
    )

    raw_after = pd.read_json(StringIO(new_raw), orient="split")
    assert raw_after["R1"].tolist() == pytest.approx([0.001, 0.002, 0.003])
    assert saved_store["R1"]["series_type"] == "predicted"
    assert status == "Overwrote shared series R1."


def test_reg_layout_starts_with_welcome_and_main_hidden(regression_page):
    welcome = _find_component_by_id(regression_page.layout, "reg-welcome-screen")
    main = _find_component_by_id(regression_page.layout, "reg-main-container")
    blocker_store = _find_component_by_id(regression_page.layout, "reg-ui-blocker-store")
    blocker_overlay = _find_component_by_id(regression_page.layout, "reg-ui-blocker-overlay")

    assert getattr(welcome, "style", {})["display"] == "none"
    assert getattr(main, "style", {})["display"] == "none"
    assert getattr(blocker_store, "data", None) is True
    assert getattr(blocker_overlay, "visible", None) is True
    assert getattr(blocker_overlay, "zIndex", None) == 2500


def test_reg_bootstrap_uses_only_page_load_interval_for_tab_ready():
    page_text = Path("pages/regression.py").read_text(encoding="utf-8")
    assert 'dcc.Interval(id="reg-page-load-trigger"' in page_text
    assert 'reg-initial-tab-render-trigger' not in page_text
    assert 'Output("reg-initial-tab-render-ready-store", "data")' in page_text
    assert 'Input("reg-page-load-trigger", "n_intervals")' in page_text
    assert 'Output("reg-tabs", "value")' in page_text
    assert 'State("reg-active-tab-store", "data")' in page_text
    assert 'Output("reg-welcome-screen", "style")' in page_text
    assert 'Input("dashmat-raw-data-store", "data")' in page_text
    assert 'Input("reg-page-load-trigger", "n_intervals")' in page_text


def test_reg_layout_uses_diagnostics_first_tab_order(regression_page):
    tabs = _find_component_by_id(regression_page.layout, "reg-tabs")
    tabs_list = getattr(tabs, "children", [])[0]
    labels = [getattr(tab, "children", None) for tab in getattr(tabs_list, "children", [])]

    assert labels == [
        "ANOVA",
        "Rolling Summary",
        "Scatter",
        "Weights",
        "Statistics",
        "Returns",
        "Rolling",
        "Calendar Year",
        "Growth of $1",
        "Drawdown",
    ]


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
    options, value = regression_page.reg_toggle_welcome({"has_data": True, "original_periodicity": "daily"}, "monthly")

    assert captured["arg"] == "daily"
    assert options == [{"value": "daily", "label": "Daily"}, {"value": "monthly", "label": "Monthly"}]
    assert value == "monthly"


def test_regression_layout_includes_common_daily_controls():
    page_text = Path("pages/regression.py").read_text(encoding="utf-8")
    assert 'id="reg-common-daily-button"' in page_text
    assert 'dcc.Store(id="reg-common-daily-candidates-store", data=None, storage_type="memory")' in page_text


def test_regression_common_daily_candidates_combine_x_and_dependent_var(monkeypatch, regression_page):
    captured = {}

    def _fake_compute_common_daily_candidates(raw_data, selected_series):
        captured["raw_data"] = raw_data
        captured["selected_series"] = tuple(selected_series)
        return {"common_daily_start": "2020-01-01", "common_daily_end": "2020-12-31"}

    monkeypatch.setattr(regression_page, "compute_common_daily_candidates", _fake_compute_common_daily_candidates)

    result = regression_page.reg_update_common_daily_candidates(
        "raw-json",
        ["X2", "X1"],
        "Y",
    )

    assert result == {"common_daily_start": "2020-01-01", "common_daily_end": "2020-12-31"}
    assert captured["raw_data"] == "raw-json"
    assert captured["selected_series"] == ("X1", "X2", "Y")


def test_reg_init_date_range_does_not_depend_on_common_daily_store():
    page_text = Path("pages/regression.py").read_text(encoding="utf-8")
    assert 'Input("reg-range-candidates-store", "data")' in page_text
    assert 'Input("reg-common-daily-candidates-store", "data")' in page_text
    init_block = page_text.split("def reg_init_date_range", 1)[0]
    init_callback = init_block.rsplit("@callback(", 1)[-1]
    assert 'Input("reg-common-daily-candidates-store", "data")' not in init_callback


def test_regression_common_daily_button_uses_clientside_disabled_toggle():
    page_text = Path("pages/regression.py").read_text(encoding="utf-8")
    js_text = Path("assets/dashmat_callbacks.js").read_text(encoding="utf-8")
    assert 'ClientsideFunction(namespace="dashmat_callbacks", function_name="commonDailyButtonDisabled")' in page_text
    assert 'Output("reg-common-daily-button", "disabled")' in page_text
    assert 'Input("reg-common-daily-candidates-store", "data")' in page_text
    assert "function commonDailyButtonDisabled(candidates, commonDailyCandidates, periodicityOptions)" in js_text
    assert 'Input("reg-periodicity-select", "data")' in page_text


def test_reg_date_range_common_daily_sets_daily_trading(monkeypatch, regression_page):
    monkeypatch.setattr(
        regression_page,
        "callback_context",
        type("Ctx", (), {"triggered": [{"prop_id": "reg-common-daily-button.n_clicks"}]})(),
    )

    result = regression_page.reg_date_range_button(
        None,
        1,
        None,
        {"available_series": ("X1", "Y"), "max_start": "2020-01-01", "max_end": "2020-12-31"},
        {"common_daily_start": "2020-02-01", "common_daily_end": "2020-11-30"},
    )

    assert result == (
        "2020-02-01",
        "2020-11-30",
        {"start": "2020-02-01", "end": "2020-11-30"},
        "daily_trading",
        "daily_trading",
    )


def test_reg_date_range_common_range_preserves_periodicity(monkeypatch, regression_page):
    monkeypatch.setattr(
        regression_page,
        "callback_context",
        type("Ctx", (), {"triggered": [{"prop_id": "reg-common-range-button.n_clicks"}]})(),
    )

    result = regression_page.reg_date_range_button(
        1,
        None,
        None,
        {
            "available_series": ("X1", "Y"),
            "common_start": "2020-03-01",
            "common_end": "2020-10-31",
            "max_start": "2020-01-01",
            "max_end": "2020-12-31",
        },
        {"common_daily_start": "2020-02-01", "common_daily_end": "2020-11-30"},
    )

    assert result[:3] == (
        "2020-03-01",
        "2020-10-31",
        {"start": "2020-03-01", "end": "2020-10-31"},
    )
    assert result[3] is no_update
    assert result[4] is no_update


def test_regression_upload_clientside_trigger_targets_blocker_store():
    page_text = Path("pages/regression.py").read_text(encoding="utf-8")
    assert 'ClientsideFunction(namespace="dashmat_callbacks", function_name="triggerRegressionUpload")' in page_text
    assert 'Output("reg-ui-blocker-store", "data", allow_duplicate=True)' in page_text


def test_regression_upload_trigger_uses_cancel_aware_shared_helper():
    js_text = Path("assets/dashmat_callbacks.js").read_text(encoding="utf-8")
    assert 'return triggerUploadWithCancel("reg-upload-data", "reg-ui-blocker-store");' in js_text
    assert 'if (typeof input.showPicker === "function") {' in js_text
    assert "input.showPicker();" in js_text
    assert "input.click();" in js_text


def test_reg_handle_upload_multi_sheet_opens_modal_and_releases_blocker(monkeypatch, regression_page):
    monkeypatch.setattr(regression_page, "get_sheet_names", lambda *_args, **_kwargs: ["Sheet A", "Sheet B"])

    result = regression_page.reg_handle_upload("contents", "multi.xlsx", None, None)

    assert result[0] is True
    assert result[1] == [{"value": "Sheet A", "label": "Sheet A"}, {"value": "Sheet B", "label": "Sheet B"}]
    assert result[2] == ["Sheet A"]
    assert result[3] == "contents"
    assert result[4] == "multi.xlsx"
    assert result[5] == ["Sheet A", "Sheet B"]
    assert result[6] is no_update
    assert result[10] is False


def test_reg_handle_upload_error_releases_blocker(monkeypatch, regression_page):
    monkeypatch.setattr(regression_page, "get_sheet_names", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        regression_page,
        "_shared_import_single_upload",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("boom")),
    )

    result = regression_page.reg_handle_upload("contents", "bad.xlsx", None, None)

    assert all(value is no_update for value in result[:-1])
    assert result[-1] is False


def test_reg_handle_sheet_select_import_all_releases_blocker_and_clears_stash(monkeypatch, regression_page):
    idx = pd.date_range("2024-01-01", periods=2, freq="B")
    merged_df = pd.DataFrame({"Asset_A": [0.01, 0.02]}, index=idx)
    monkeypatch.setattr(
        regression_page,
        "callback_context",
        type("Ctx", (), {"triggered_id": "reg-sheet-select-import-all-button"})(),
    )
    monkeypatch.setattr(
        regression_page,
        "_shared_import_selected_workbook_sheets",
        lambda *_args, **_kwargs: (merged_df, ["Sheet A", "Sheet B"]),
    )
    monkeypatch.setattr(
        regression_page,
        "_shared_merge_uploaded_with_existing",
        lambda *_args, **_kwargs: SimpleNamespace(merged_df=merged_df, combined_periodicity="daily"),
    )

    result = regression_page.reg_handle_sheet_select_ok(
        None,
        1,
        [],
        "contents",
        "multi.xlsx",
        ["Sheet A", "Sheet B"],
        None,
        None,
    )

    assert isinstance(result[0], str)
    assert result[1] == "daily"
    assert result[4] is False
    assert result[5] is None
    assert result[8] is None
    assert result[9] is True


def test_reg_handle_sheet_select_error_keeps_modal_open_and_releases_blocker(monkeypatch, regression_page):
    monkeypatch.setattr(
        regression_page,
        "callback_context",
        type("Ctx", (), {"triggered_id": "reg-sheet-select-ok-button"})(),
    )
    monkeypatch.setattr(
        regression_page,
        "_shared_import_selected_workbook_sheets",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("boom")),
    )

    result = regression_page.reg_handle_sheet_select_ok(
        1,
        None,
        ["Sheet A"],
        "contents",
        "multi.xlsx",
        ["Sheet A", "Sheet B"],
        None,
        None,
    )

    assert result[0] is no_update
    assert result[4] is True
    assert result[5] == "contents"
    assert result[6] == "multi.xlsx"
    assert result[7] == ["Sheet A", "Sheet B"]
    assert result[8] is no_update
    assert result[9] is False


def test_reg_sheet_select_cancel_clears_stash_and_releases_blocker(regression_page):
    assert regression_page.reg_on_sheet_select_cancel(1) == (False, None, None, None, None, False)


def test_reg_series_modal_open_is_clientside():
    page_text = Path("pages/regression.py").read_text(encoding="utf-8")
    js_text = Path("assets/dashmat_callbacks.js").read_text(encoding="utf-8")

    assert 'ClientsideFunction(namespace="dashmat_callbacks", function_name="openRegressionSeriesModal")' in page_text
    assert "function openRegressionSeriesModal(" in js_text
    assert "def reg_open_modal(" not in page_text


def test_reg_blocker_wiring_covers_add_modal_entry_and_series_render():
    page_text = Path("pages/regression.py").read_text(encoding="utf-8")
    js_text = Path("assets/dashmat_callbacks.js").read_text(encoding="utf-8")

    assert 'Input("reg-menu-add-from-db", "n_clicks")' in page_text
    assert 'Input("reg-open-modal-button", "n_clicks")' in page_text
    assert 'Output("reg-series-selection-container", "children")' in page_text
    assert 'Output("reg-ui-blocker-store", "data", allow_duplicate=True)' in page_text
    assert 'if (trigger.indexOf("series-selection-modal") !== -1) {' in js_text
    assert 'ClientsideFunction(namespace="dashmat_callbacks", function_name="releaseBlockerOnSeriesGridReady")' in page_text
    assert 'Input("reg-series-selection-grid", "virtualRowData", allow_optional=True)' in page_text
    assert "function releaseBlockerOnSeriesGridReady(virtualRows, modalOpened)" in js_text
    assert 'ClientsideFunction(namespace="dashmat_callbacks", function_name="regressionInitialSeriesBlocker")' in page_text
    assert 'Input("reg-url-location", "pathname")' in page_text
    assert 'Input("reg-series-selection-modal", "opened")' in page_text
    assert 'Input("reg-series-selection-grid", "virtualRowData", allow_optional=True)' in page_text
    assert 'Input("reg-page-load-trigger", "n_intervals")' in page_text
    assert 'State("reg-page-visited-store", "data")' in page_text
    assert 'State("reg-series-order-store", "data")' in page_text
    assert 'State("reg-dependent-var-store", "data")' in page_text
    assert 'State("dashmat-pending-new-series-store", "data")' in page_text
    assert "function regressionInitialSeriesBlocker(pathname, rawMeta, currentSelect, pageLoadReady, modalOpened, virtualRows, pageVisited, currentOrder, currentDepVar, poOriginSeries)" in js_text
    assert "function regressionInitialSeriesModalPending(rawMeta, currentSelect, currentOrder, currentDepVar, poOriginSeries, pageVisited)" in js_text


def test_reg_on_modal_ok_returns_no_update_for_unchanged_outputs(regression_page, raw_json):
    result = regression_page.reg_on_modal_ok(
        1,
        ["Asset_A"],
        {"Asset_A": "None"},
        {"Asset_A": False},
        ["Asset_A"],
        [],
        raw_json,
        {"Asset_A": True},
        "Asset_A",
        {"Asset_A": 0},
        {"Asset_A": -999.0},
        {"Asset_A": 999.0},
        {"Asset_A": False},
        ["Asset_A"],
        {"Asset_A": "None"},
        {"Asset_A": False},
        ["Asset_A"],
        {"Asset_A": True},
        "Asset_A",
        {"Asset_A": 0},
        {"Asset_A": -999.0},
        {"Asset_A": 999.0},
        {"Asset_A": False},
    )

    assert result[0] is no_update
    assert result[1] is no_update
    assert result[2] is no_update
    assert result[3] is no_update
    assert result[5] is no_update
    assert result[6] is no_update
    assert result[7] is no_update
    assert result[8] is no_update
    assert result[9] is no_update
    assert result[10] is no_update
    assert result[11] is no_update
    assert result[12] is no_update


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

    children, _order, blocker = regression_page.reg_update_series_grid(
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
    assert blocker is no_update

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


def _collect_ag_grids(node):
    if node is None:
        return []
    if isinstance(node, (str, int, float, bool, dict)):
        return []
    if isinstance(node, (list, tuple, set)):
        out = []
        for item in node:
            out.extend(_collect_ag_grids(item))
        return out

    out = []
    if getattr(node, "columnDefs", None) is not None and getattr(node, "rowData", None) is not None:
        out.append(node)
    children = getattr(node, "children", None)
    out.extend(_collect_ag_grids(children))
    props = getattr(node, "props", None)
    if isinstance(props, dict):
        for value in props.values():
            out.extend(_collect_ag_grids(value))
    return out


def test_reg_toggle_welcome_no_data_shows_top_aligned_welcome(regression_page):
    options, value = regression_page.reg_toggle_welcome(None, None)
    assert options == [{"value": "daily", "label": "Daily"}]
    assert value == "daily"


def test_reg_help_modal_covers_three_sections_and_model_explainers(regression_page):
    help_control = _find_component_by_id(regression_page.layout, "reg-menu-help-guide")
    assert help_control is not None
    text_blob = Path("docs/help/regression.md").read_text(encoding="utf-8").lower()

    required_phrases = [
        "regression analysis",
        "series selection",
        "periodicity",
        "vol scaler",
        "date range",
        "ols",
        "constrained ols",
        "style analysis",
        "ridge",
        "lasso",
        "elastic net",
        "linear constraints",
        "rolling summary",
        "anova",
        "scatter",
        "save session",
        "load session",
        "download excel",
        "result management",
        "arima",
        "garch",
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


def test_reg_session_actions_use_shared_workspace_helpers():
    text_blob = Path("pages/regression.py").read_text(encoding="utf-8")
    assert 'ClientsideFunction(namespace="dashmat_callbacks", function_name="saveWorkspaceSession")' in text_blob
    assert 'ClientsideFunction(namespace="dashmat_callbacks", function_name="loadWorkspaceSessionDialog")' in text_blob
    assert 'ClientsideFunction(namespace="dashmat_callbacks", function_name="loadWorkspaceSession")' in text_blob
    assert 'ClientsideFunction(namespace="dashmat_callbacks", function_name="clearWorkspaceSession")' in text_blob
    assert "sessionStorage.clear()" not in text_blob
    assert "sessionStorage.length" not in text_blob


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


def test_reg_render_statistics_requires_active_tab_and_initial_ready(regression_page):
    entry = {
        "periodicity": "daily",
        "predicted_json": df_to_json(pd.DataFrame({"predicted": [0.01]}, index=pd.to_datetime(["2024-01-01"]))),
    }

    with pytest.raises(PreventUpdate):
        regression_page.reg_render_statistics("R1", {"R1": entry}, None, None, "returns", True)

    with pytest.raises(PreventUpdate):
        regression_page.reg_render_statistics("R1", {"R1": entry}, None, None, "statistics", False)


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


def test_reg_build_display_series_clips_to_effective_window_for_full_lagged_self_x(regression_page):
    full_idx = pd.date_range("2024-01-01", periods=6, freq="B")
    model_idx = full_idx[1:]
    raw_df = pd.DataFrame(
        {"Y": [0.01, 0.02, -0.01, 0.00, 0.01, -0.02]},
        index=full_idx,
    )
    predicted = pd.DataFrame({"predicted": [0.011, 0.018, -0.005, 0.004, -0.018]}, index=model_idx)
    residuals = pd.DataFrame({"residuals": [0.001, 0.002, -0.001, 0.003, -0.002]}, index=model_idx)
    entry = {
        "periodicity": "daily",
        "dependent_var": "Y",
        "independent_vars": ["Y (lag 1)"],
        "independent_vars_internal": ["Y"],
        "benchmark_assignments": {},
        "long_short_assignments": {},
        "date_range": {"start": "2024-01-01", "end": "2024-01-31"},
        "vol_scaler": 0,
        "vol_scaling_assignments": {},
        "config": {
            "window_type": "full",
            "feature_label_map": {"Y": "Y (lag 1)"},
            "lag_config": {"Y": 1},
        },
        "predicted_json": df_to_json(predicted),
        "residuals_json": df_to_json(residuals),
    }

    display_df, ordered_cols = regression_page._reg_build_display_series(entry, df_to_json(raw_df))

    assert ordered_cols == ["Predicted", "Actual (Y)", "Y (lag 1)", "Residual"]
    assert list(display_df.index) == list(model_idx)
    assert display_df.iloc[0]["Y (lag 1)"] == pytest.approx(raw_df.iloc[0]["Y"])


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
        "arima_garch": {
            "arima": {"order": [1, 0, 1], "aic": 11.1, "bic": 12.2, "params": {"const": 0.02, "ar.L1": 0.33}},
            "garch": {"order": [1, 1], "aic": 13.3, "bic": 14.4, "params": {"mu": 0.01, "omega": 0.2}},
        },
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
        "Scatter",
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
    assert settings_map["Effective Sample Start"] == "2024-01-01"
    assert settings_map["Effective Sample End"] == "2024-01-05"

    weights_df = pd.read_excel(BytesIO(payload["content"]), sheet_name="Weights")
    assert list(weights_df.columns) == ["Window", "Date", "intercept", "X1"]
    assert "ARIMA_AIC" not in set(weights_df.columns)
    assert "GARCH_AIC" not in set(weights_df.columns)
    assert weights_df.loc[0, "Date"] == "2024-01-01"
    assert weights_df.loc[0, "intercept"] == pytest.approx(0.001)
    assert weights_df.loc[0, "X1"] == pytest.approx(0.95)

    anova_df = pd.read_excel(BytesIO(payload["content"]), sheet_name="ANOVA")
    assert "Block" in anova_df.columns
    assert "Parameters" in set(anova_df["Block"].dropna())

    scatter_df = pd.read_excel(BytesIO(payload["content"]), sheet_name="Scatter")
    assert "Date" in scatter_df.columns
    assert "Predicted" in scatter_df.columns
    assert "Actual (Y)" in scatter_df.columns
    assert "Residual" in scatter_df.columns
    assert "Overall Fit" in set(anova_df["Block"].dropna())
    assert "ARIMA.const" in set(anova_df.get("Parameter", pd.Series(dtype=str)).dropna())
    assert "GARCH.mu" in set(anova_df.get("Parameter", pd.Series(dtype=str)).dropna())


def test_reg_sync_anova_window_options_defaults_to_latest_on_result_change(monkeypatch, regression_page):
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    results = {
        "R1": {
            "window_results": [
                {"apply_start": idx[0], "apply_end": idx[0]},
                {"apply_start": idx[1], "apply_end": idx[1]},
                {"apply_start": idx[2], "apply_end": idx[2]},
            ]
        }
    }
    monkeypatch.setattr(regression_page, "callback_context", type("Ctx", (), {"triggered_id": "reg-result-select"})())

    options, value, disabled = regression_page.reg_sync_anova_window_options("R1", results, "0")

    assert len(options) == 3
    assert value == "2"
    assert disabled is False


def test_reg_sync_anova_window_options_defaults_to_latest_on_results_refresh(monkeypatch, regression_page):
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    results = {
        "R1": {
            "window_results": [
                {"apply_start": idx[0], "apply_end": idx[0]},
                {"apply_start": idx[1], "apply_end": idx[1]},
                {"apply_start": idx[2], "apply_end": idx[2]},
            ]
        }
    }
    monkeypatch.setattr(regression_page, "callback_context", type("Ctx", (), {"triggered_id": "reg-results-store"})())

    _options, value, _disabled = regression_page.reg_sync_anova_window_options("R1", results, "1")

    assert value == "2"


def test_reg_render_rolling_returns_table_uses_wide_date_column(monkeypatch, regression_page):
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    predicted = pd.DataFrame({"predicted": [0.01, 0.0, 0.002, -0.001]}, index=idx)
    residuals = pd.DataFrame({"residuals": [0.0, 0.0, 0.0, 0.0]}, index=idx)
    results = {
        "R1": {
            "periodicity": "daily",
            "predicted_json": df_to_json(predicted),
            "residuals_json": df_to_json(residuals),
        }
    }

    monkeypatch.setattr(
        regression_page,
        "calculate_rolling_returns",
        lambda *_args, **_kwargs: pd.DataFrame(
            {"Predicted": [0.05], "Actual (Y)": [0.04], "Residual": [0.01]},
            index=[pd.Timestamp("2024-01-31")],
        ),
    )
    grid = regression_page.reg_render_rolling_returns(
        "R1",
        results,
        None,
        "1y",
        "annualized",
        "total_return",
        "table",
        None,
        True,
        "light",
    )

    assert getattr(grid, "columnDefs", [])[0]["field"] == "Date"
    assert getattr(grid, "columnDefs", [])[0]["width"] == 112


def test_reg_render_drawdown_table_uses_wide_date_column(monkeypatch, regression_page):
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    predicted = pd.DataFrame({"predicted": [0.01, 0.0, 0.002, -0.001]}, index=idx)
    residuals = pd.DataFrame({"residuals": [0.0, 0.0, 0.0, 0.0]}, index=idx)
    results = {
        "R1": {
            "periodicity": "daily",
            "predicted_json": df_to_json(predicted),
            "residuals_json": df_to_json(residuals),
        }
    }

    monkeypatch.setattr(
        regression_page,
        "calculate_drawdown",
        lambda *_args, **_kwargs: pd.DataFrame(
            {"Predicted": [0.0, -0.02]},
            index=[pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-31")],
        ),
    )
    grid = regression_page.reg_render_drawdown("R1", results, None, "table", "light")

    assert getattr(grid, "columnDefs", [])[0]["field"] == "Date"
    assert getattr(grid, "columnDefs", [])[0]["width"] == 112


def test_reg_render_growth_table_mode_returns_grid_with_wide_date_column(regression_page):
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    predicted = pd.DataFrame({"predicted": [0.01, 0.0, 0.002, -0.001]}, index=idx)
    residuals = pd.DataFrame({"residuals": [0.0, 0.0, 0.0, 0.0]}, index=idx)
    results = {
        "R1": {
            "periodicity": "daily",
            "predicted_json": df_to_json(predicted),
            "residuals_json": df_to_json(residuals),
        }
    }

    grid = regression_page.reg_render_growth("R1", results, None, "table", "light")

    assert getattr(grid, "columnDefs", [])[0]["field"] == "Date"
    assert getattr(grid, "columnDefs", [])[0]["width"] == 112
    assert len(getattr(grid, "rowData", [])) == 4


def test_reg_render_weights_table_mode_returns_grid_with_wide_date_column(regression_page):
    results = {
        "R1": {
            "config": {"model": "style_analysis"},
            "window_results": [
                {
                    "apply_start": "2024-01-01",
                    "coefficients": {"Asset_A": 0.6, "Asset_B": 0.4},
                },
                {
                    "apply_start": "2024-02-01",
                    "coefficients": {"Asset_A": 0.5, "Asset_B": 0.5},
                },
            ],
        }
    }

    stack = regression_page.reg_render_weights("R1", results, "table", "light")
    children = list(getattr(stack, "children", []) or [])
    grid = children[0]

    date_col = next(c for c in getattr(grid, "columnDefs", []) if c.get("field") == "Date")
    assert date_col["width"] == 112


def test_reg_render_anova_uses_three_block_layout_with_arima_garch_params(regression_page):
    results = {
        "R1": {
            "dependent_var": "Y",
            "config": {"model": "ols"},
            "window_results": [
                {
                    "est_start": "2024-01-01",
                    "est_end": "2024-01-31",
                    "apply_start": "2024-01-01",
                    "apply_end": "2024-01-31",
                    "coefficients": {"intercept": 0.01, "X1": 0.5},
                    "p_values": {"intercept": 0.1, "X1": 0.02},
                    "diagnostics": {
                        "durbin_watson": 1.9,
                        "jarque_bera_stat": 2.1,
                        "jarque_bera_pvalue": 0.3,
                    },
                    "r_squared": 0.8,
                    "adj_r_squared": 0.79,
                    "n_obs": 21,
                    "residual_std": 0.02,
                    "anova_table": {
                        "df_model": 1,
                        "df_resid": 19,
                        "ss_model": 0.45,
                        "ms_model": 0.45,
                        "F_stat": 9.0,
                        "F_pvalue": 0.01,
                        "ss_resid": 0.15,
                        "ms_resid": 0.0079,
                        "ss_total": 0.60,
                    },
                    "arima_garch": {
                        "arima": {"order": [1, 0, 1], "aic": 1.2, "bic": 1.4, "params": {"const": 0.01, "ar.L1": 0.22}},
                        "garch": {"order": [1, 1], "aic": 2.3, "bic": 2.5, "params": {"mu": 0.02, "omega": 0.1}},
                    },
                }
            ],
        }
    }

    comp = regression_page.reg_render_anova("R1", results, "0")
    grids = _collect_ag_grids(comp)
    assert len(grids) >= 2

    anova_grid = next(
        (g for g in grids if {"Source", "df", "SS", "MS", "F", "p-value"}.issubset({c.get("field") for c in (getattr(g, "columnDefs", []) or [])})),
        None,
    )
    assert anova_grid is not None
    assert (getattr(anova_grid, "style", {}) or {}).get("height") == "132px"
    anova_sources = [row.get("Source") for row in (getattr(anova_grid, "rowData", []) or [])]
    assert set(anova_sources) == {"Model", "Residual", "Total"}

    param_grid = next(
        (g for g in grids if {"Parameter", "Coefficient"}.issubset({c.get("field") for c in (getattr(g, "columnDefs", []) or [])})),
        None,
    )
    assert param_grid is not None
    param_names = [row.get("Parameter") for row in (getattr(param_grid, "rowData", []) or [])]
    assert "intercept" in param_names
    assert "X1" in param_names
    assert "ARIMA.const" in param_names
    assert "ARIMA.ar.L1" in param_names
    assert "GARCH.mu" in param_names
    assert "GARCH.omega" in param_names

    text_blob = " ".join(_collect_component_text(comp))
    assert "Overall Fit" in text_blob
    assert "Regression Fit" in text_blob
    assert "R-Squared" in text_blob
    assert "ARIMA Fit" in text_blob
    assert "AIC" in text_blob
    assert "GARCH Fit" in text_blob


def test_reg_render_rolling_table_merges_arima_garch_columns(regression_page):
    results = {
        "R1": {
            "window_results": [
                {
                    "apply_start": "2024-01-01",
                    "r_squared": 0.5,
                    "adj_r_squared": 0.4,
                    "residual_std": 0.02,
                    "n_obs": 20,
                    "coefficients": {"X1": 0.3},
                    "arima_garch": {"arima": {"order": [1, 0, 0], "aic": 1.0, "bic": 1.1, "params": {"ar.L1": 0.2}}},
                },
                {
                    "apply_start": "2024-02-01",
                    "r_squared": 0.6,
                    "adj_r_squared": 0.5,
                    "residual_std": 0.01,
                    "n_obs": 20,
                    "coefficients": {"X1": 0.4},
                    "arima_garch": {"arima": {"order": [1, 0, 0], "aic": 0.9, "bic": 1.0, "params": {"ar.L1": 0.25}}},
                },
            ]
        }
    }

    grid = regression_page.reg_render_rolling("R1", results, "table", "advanced", "light")
    fields = [c.get("field") for c in getattr(grid, "columnDefs", [])]
    assert "ARIMA_AIC" in fields
    assert "ARIMA_ar_L1" in fields

    rows = getattr(grid, "rowData", []) or []
    assert rows[0].get("ARIMA_ar_L1") == 0.2
    assert rows[1].get("ARIMA_ar_L1") == 0.25

    basic_grid = regression_page.reg_render_rolling("R1", results, "table", "basic", "light")
    basic_fields = [c.get("field") for c in getattr(basic_grid, "columnDefs", [])]
    assert basic_fields.index("β_X1") < basic_fields.index("ARIMA_AIC")
    assert "ARIMA_AIC" in basic_fields
    assert "ARIMA_ar_L1" not in basic_fields


def test_reg_render_rolling_chart_respects_basic_advanced_field_scope(regression_page):
    results = {
        "R1": {
            "window_results": [
                {
                    "apply_start": "2024-01-01",
                    "r_squared": 0.5,
                    "adj_r_squared": 0.4,
                    "residual_std": 0.02,
                    "n_obs": 20,
                    "coefficients": {"intercept": 0.1, "X1": 0.3},
                    "arima_garch": {"arima": {"order": [1, 0, 0], "aic": 1.0, "bic": 1.1, "params": {"ar.L1": 0.2}}},
                },
                {
                    "apply_start": "2024-02-01",
                    "r_squared": 0.6,
                    "adj_r_squared": 0.5,
                    "residual_std": 0.01,
                    "n_obs": 20,
                    "coefficients": {"intercept": 0.1, "X1": 0.4},
                    "arima_garch": {"arima": {"order": [1, 0, 0], "aic": 0.9, "bic": 1.0, "params": {"ar.L1": 0.25}}},
                },
            ]
        }
    }

    basic_chart = regression_page.reg_render_rolling("R1", results, "chart", "basic", "light")
    basic_names = [trace.name for trace in getattr(getattr(basic_chart, "figure", None), "data", [])]
    assert "β_intercept" in basic_names
    assert "β_X1" in basic_names
    assert "ARIMA_AIC" in basic_names
    assert "ARIMA_ar_L1" not in basic_names

    advanced_chart = regression_page.reg_render_rolling("R1", results, "chart", "advanced", "light")
    advanced_names = [trace.name for trace in getattr(getattr(advanced_chart, "figure", None), "data", [])]
    assert "ARIMA_ar_L1" in advanced_names


def test_reg_render_weights_table_only_shows_prediction_coefficients(regression_page):
    results = {
        "R1": {
            "config": {"model": "ols"},
            "window_results": [
                {
                    "apply_start": "2024-01-01",
                    "coefficients": {"X1": 0.3},
                    "arima_garch": {"garch": {"order": [1, 1], "aic": 2.0, "bic": 2.1, "params": {"omega": 0.12}}},
                },
                {
                    "apply_start": "2024-02-01",
                    "coefficients": {"X1": 0.4},
                    "arima_garch": {"garch": {"order": [1, 1], "aic": 1.9, "bic": 2.0, "params": {"omega": 0.10}}},
                },
            ]
        }
    }

    comp = regression_page.reg_render_weights("R1", results, "table", "light")
    children = list(getattr(comp, "children", []) or [])
    grid = next(c for c in children if getattr(c, "columnDefs", None) is not None)
    fields = [c.get("field") for c in getattr(grid, "columnDefs", [])]
    assert fields == ["Window", "Date", "X1"]
    row_data = getattr(grid, "rowData", []) or []
    assert row_data[0].get("X1") == 0.3
    assert row_data[1].get("X1") == 0.4
