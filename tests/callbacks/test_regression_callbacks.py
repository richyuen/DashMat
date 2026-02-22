from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from dash import no_update

from utils.regression import RegressionWindowResult
from utils.returns import df_to_json


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
