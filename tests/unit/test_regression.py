from __future__ import annotations

import numpy as np
import pandas as pd

from utils.regression import run_regression


def _sample_regression_inputs(seed: int = 2026, n_obs: int = 260) -> tuple[pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n_obs, freq="B")
    x1 = rng.normal(0.0, 1.0, n_obs)
    x2 = rng.normal(0.0, 1.0, n_obs)
    noise = rng.normal(0.0, 0.05, n_obs)

    X = pd.DataFrame({"x1": x1, "x2": x2}, index=idx)
    y = pd.Series(0.01 + 0.7 * x1 + 0.2 * x2 + noise, index=idx, name="y")
    return y, X


def test_constrained_ols_respects_per_variable_bounds():
    y, X = _sample_regression_inputs()
    config = {
        "model": "constrained_ols",
        "window_type": "full",
        "enable_constraint": {"x1": True, "x2": True},
        "min_beta": -10.0,
        "max_beta": 10.0,
        "min_beta_by_var": {"x1": -0.5, "x2": 0.4},
        "max_beta_by_var": {"x1": 0.5, "x2": 0.6},
    }

    wrs, predicted, residuals, arima_garch_summary = run_regression(y, X, config)

    assert len(wrs) == 1
    coefs = wrs[0].coefficients
    assert -0.5 <= coefs["x1"] <= 0.5
    assert 0.4 <= coefs["x2"] <= 0.6
    assert len(predicted) > 0
    assert len(residuals) > 0
    assert arima_garch_summary is None


def test_constrained_ols_falls_back_to_global_bounds_without_per_variable_maps():
    y, X = _sample_regression_inputs()
    config = {
        "model": "constrained_ols",
        "window_type": "full",
        "enable_constraint": {"x1": True, "x2": True},
        "min_beta": 0.25,
        "max_beta": 0.35,
    }

    wrs, _predicted, _residuals, _arima_garch_summary = run_regression(y, X, config)

    assert len(wrs) == 1
    coefs = wrs[0].coefficients
    assert 0.25 <= coefs["x1"] <= 0.35
    assert 0.25 <= coefs["x2"] <= 0.35


def test_run_level_arima_garch_summary_is_compact_and_not_duplicated_per_window():
    y, X = _sample_regression_inputs()
    config = {
        "model": "ols",
        "window_type": "full",
        "arima_order": (1, 0, 1),
        "garch_order": (1, 1),
    }

    wrs, _predicted, _residuals, arima_garch_summary = run_regression(y, X, config)

    assert len(wrs) == 1
    assert isinstance(arima_garch_summary, dict)
    for wr in wrs:
        assert "arima_garch" not in (wr.diagnostics or {})

    for item in arima_garch_summary.values():
        assert "summary_text" not in item


def test_rolling_ols_attaches_per_window_arima_garch_results():
    y, X = _sample_regression_inputs(n_obs=320)
    config = {
        "model": "ols",
        "window_type": "rolling",
        "window_size": 84,
        "opt_step": 21,
        "opt_step_unit": "periods",
        "fill_in_sample": False,
        "arima_order": (1, 0, 1),
        "garch_order": (1, 1),
    }

    wrs, _predicted, _residuals, arima_garch_summary = run_regression(y, X, config)

    assert len(wrs) > 1
    assert any(isinstance(wr.arima_garch, dict) for wr in wrs)
    assert all(("arima_garch" not in (wr.diagnostics or {})) for wr in wrs)
    assert isinstance(arima_garch_summary, dict)


def test_ols_supports_intercept_only_with_no_x():
    y, _X = _sample_regression_inputs()
    config = {
        "model": "ols",
        "window_type": "full",
        "force_zero_intercept": False,
        "arima_order": (1, 0, 1),
        "garch_order": (1, 1),
    }

    wrs, predicted, residuals, arima_garch_summary = run_regression(y, pd.DataFrame(index=y.index), config)

    assert len(wrs) == 1
    assert "intercept" in wrs[0].coefficients
    assert len(predicted) == len(y)
    assert len(residuals) == len(y)
    assert isinstance(arima_garch_summary, dict)


def test_ols_force_zero_intercept_with_no_x_returns_no_results():
    y, _X = _sample_regression_inputs()
    config = {
        "model": "ols",
        "window_type": "full",
        "force_zero_intercept": True,
    }

    wrs, predicted, residuals, arima_garch_summary = run_regression(y, pd.DataFrame(index=y.index), config)

    assert wrs == []
    assert predicted.empty
    assert residuals.empty
    assert arima_garch_summary is None


def test_constrained_ols_ignores_blank_linear_constraint_bounds_without_crashing():
    y, X = _sample_regression_inputs()
    config = {
        "model": "constrained_ols",
        "window_type": "full",
        "enable_constraint": {"x1": True},
        "min_beta": -1.0,
        "max_beta": 1.0,
        "linear_constraints": [{"x1": "", "x2": "", "Min": "", "Max": " "}],
    }

    wrs, predicted, residuals, _summary = run_regression(y, X, config)

    assert len(wrs) == 1
    assert len(predicted) > 0
    assert len(residuals) > 0


def test_style_analysis_ignores_blank_linear_constraint_bounds_without_crashing():
    y, X = _sample_regression_inputs()
    config = {
        "model": "style_analysis",
        "window_type": "full",
        "linear_constraints": [{"x1": "", "x2": "", "Min": "", "Max": ""}],
    }

    wrs, predicted, residuals, _summary = run_regression(y, X, config)

    assert len(wrs) == 1
    assert len(predicted) > 0
    assert len(residuals) > 0


def test_constrained_ols_returns_no_results_for_infeasible_constraints():
    y, X = _sample_regression_inputs()
    config = {
        "model": "constrained_ols",
        "window_type": "full",
        "enable_constraint": {"x1": True},
        "min_beta": -1.0,
        "max_beta": 1.0,
        "linear_constraints": [{"x1": 1.0, "x2": 0.0, "Min": 2.0}],
    }

    wrs, predicted, residuals, _summary = run_regression(y, X, config)

    assert wrs == []
    assert predicted.empty
    assert residuals.empty


def test_style_analysis_returns_no_results_for_infeasible_constraints():
    y, X = _sample_regression_inputs()
    config = {
        "model": "style_analysis",
        "window_type": "full",
        "linear_constraints": [{"x1": 1.0, "x2": 0.0, "Min": 2.0}],
    }

    wrs, predicted, residuals, _summary = run_regression(y, X, config)

    assert wrs == []
    assert predicted.empty
    assert residuals.empty
