from __future__ import annotations

import numpy as np
import pandas as pd

from utils.optimization import run_portfolio_optimization


def _build_factor_rp_returns() -> pd.DataFrame:
    np.random.seed(42)
    n_obs = 320
    dates = pd.date_range("2021-01-01", periods=n_obs, freq="B")

    factor_1 = np.random.normal(0.00025, 0.0090, n_obs)
    factor_2 = np.random.normal(0.00010, 0.0075, n_obs)
    noise = np.random.normal(0.0, 0.0030, (n_obs, 4))

    data = {
        "Asset_A": 0.9 * factor_1 + 0.2 * factor_2 + noise[:, 0],
        "Asset_B": 0.4 * factor_1 + 0.6 * factor_2 + noise[:, 1],
        "Asset_C": -0.1 * factor_1 + 0.9 * factor_2 + noise[:, 2],
        "Asset_D": 0.2 * factor_1 - 0.3 * factor_2 + noise[:, 3],
    }
    return pd.DataFrame(data, index=dates)


def _build_constraint_returns() -> pd.DataFrame:
    np.random.seed(7)
    dates = pd.date_range("2020-01-01", periods=360, freq="B")
    cov = np.array(
        [
            [0.00010, 0.00004, 0.00002],
            [0.00004, 0.00012, 0.00003],
            [0.00002, 0.00003, 0.00009],
        ]
    )
    mean = np.array([0.00025, 0.00018, 0.00012])
    draws = np.random.multivariate_normal(mean=mean, cov=cov, size=len(dates))
    return pd.DataFrame(draws, index=dates, columns=["Equity", "Credit", "Rates"])


def _build_sharpe_returns() -> pd.DataFrame:
    np.random.seed(17)
    dates = pd.date_range("2021-01-01", periods=420, freq="B")
    cov = np.array(
        [
            [0.00008, 0.00002, 0.00001],
            [0.00002, 0.00013, 0.00002],
            [0.00001, 0.00002, 0.00007],
        ]
    )
    mean = np.array([0.00045, 0.00018, 0.00030])
    draws = np.random.multivariate_normal(mean=mean, cov=cov, size=len(dates))
    return pd.DataFrame(draws, index=dates, columns=["Alpha", "Beta", "Gamma"])


def _assert_linear_constraints(weights: dict[str, float], constraints: list[dict], tol: float = 1e-6) -> None:
    for row in constraints:
        lhs = 0.0
        for asset, weight in weights.items():
            coef = row.get(asset)
            if coef in (None, ""):
                continue
            lhs += float(coef) * float(weight)

        min_v = row.get("Min")
        max_v = row.get("Max")
        if min_v not in (None, ""):
            assert lhs + tol >= float(min_v)
        if max_v not in (None, ""):
            assert lhs - tol <= float(max_v)


def test_factor_risk_parity_optimization_smoke():
    returns_df = _build_factor_rp_returns()
    selected = list(returns_df.columns)
    config = {
        "model": "factor_risk_parity",
        "window_type": "full",
        "window_size": 252,
        "opt_step": 21,
        "selected_series": selected,
        "missing_data": "fill_na",
        "fill_in_sample": True,
        "min_wt": {s: 0.0 for s in selected},
        "max_wt": {s: 100.0 for s in selected},
        "force_max": {s: False for s in selected},
        "linear_constraints": [],
    }

    windows, portfolio_returns, _meta = run_portfolio_optimization(returns_df, config)

    assert len(windows) == 1
    weights = windows[0].weights
    assert set(weights.keys()) == set(selected)
    assert np.isfinite(np.array(list(weights.values()))).all()
    assert abs(sum(weights.values()) - 1.0) <= 1e-6
    assert len(portfolio_returns) > 0
    assert np.isfinite(portfolio_returns.values).all()


def test_linear_constraints_respected_in_rolling_min_variance():
    returns_df = _build_constraint_returns()
    selected = list(returns_df.columns)
    linear_constraints = [
        {"Equity": 1.0, "Credit": 1.0, "Rates": 0.0, "Min": 0.45, "Max": 0.80},
        {"Equity": 0.0, "Credit": 0.0, "Rates": 1.0, "Min": 0.20, "Max": 0.55},
    ]
    config = {
        "model": "minimize_variance",
        "window_type": "rolling",
        "window_size": 126,
        "opt_step": 21,
        "selected_series": selected,
        "missing_data": "fill_na",
        "fill_in_sample": False,
        "min_wt": {s: 0.0 for s in selected},
        "max_wt": {s: 100.0 for s in selected},
        "force_max": {s: False for s in selected},
        "linear_constraints": linear_constraints,
    }

    windows, portfolio_returns, _meta = run_portfolio_optimization(returns_df, config)

    assert len(windows) > 0
    assert len(portfolio_returns) > 0
    for w in windows:
        assert abs(sum(w.weights.values()) - 1.0) <= 1e-6
        _assert_linear_constraints(w.weights, linear_constraints)


def test_maximize_sharpe_with_constraints_and_exp_weighting():
    returns_df = _build_sharpe_returns()
    selected = list(returns_df.columns)
    linear_constraints = [
        {"Alpha": 1.0, "Beta": 0.0, "Gamma": 0.0, "Min": 0.25, "Max": 0.80},
        {"Alpha": 0.0, "Beta": 1.0, "Gamma": 1.0, "Min": 0.20, "Max": 0.75},
    ]
    config = {
        "model": "maximize_sharpe",
        "window_type": "rolling",
        "window_size": 126,
        "opt_step": 21,
        "selected_series": selected,
        "missing_data": "fill_na",
        "fill_in_sample": False,
        "min_wt": {s: 0.0 for s in selected},
        "max_wt": {s: 100.0 for s in selected},
        "force_max": {s: False for s in selected},
        "linear_constraints": linear_constraints,
        "exp_wt_cov": True,
        "halflife": 63,
        "risk_free_annual": 0.02,
        "periodicity": "daily",
    }

    windows, portfolio_returns, _meta = run_portfolio_optimization(returns_df, config)

    assert len(windows) > 0
    assert len(portfolio_returns) > 0
    assert np.isfinite(portfolio_returns.values).all()
    for w in windows:
        assert abs(sum(w.weights.values()) - 1.0) <= 1e-6
        _assert_linear_constraints(w.weights, linear_constraints)
        assert all(0.0 <= value <= 1.0 for value in w.weights.values())
