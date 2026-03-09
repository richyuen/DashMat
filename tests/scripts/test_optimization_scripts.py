from __future__ import annotations

import numpy as np
import pandas as pd

import utils.optimization as optimization
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


def _build_risk_parity_returns() -> pd.DataFrame:
    np.random.seed(19)
    dates = pd.date_range("2018-01-01", periods=520, freq="B")
    factor_1 = np.random.normal(0.00020, 0.0080, len(dates))
    factor_2 = np.random.normal(0.00010, 0.0060, len(dates))
    factor_3 = np.random.normal(0.00005, 0.0050, len(dates))
    noise = np.random.normal(0.0, 0.0025, (len(dates), 5))
    data = {
        "Asset_A": 0.9 * factor_1 + 0.2 * factor_2 + noise[:, 0],
        "Asset_B": 0.4 * factor_1 + 0.6 * factor_2 + noise[:, 1],
        "Asset_C": 0.3 * factor_1 + 0.4 * factor_3 + noise[:, 2],
        "Asset_D": 0.7 * factor_2 + 0.2 * factor_3 + noise[:, 3],
        "Asset_E": -0.1 * factor_1 + 0.8 * factor_3 + noise[:, 4],
    }
    return pd.DataFrame(data, index=dates)


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


def test_risk_parity_native_matches_riskfolio_reference_full_window():
    returns_df = _build_risk_parity_returns()
    selected = list(returns_df.columns)
    lower_bounds = {s: 0.0 for s in selected}
    upper_bounds = {s: 1.0 for s in selected}
    forced_weights = {}
    free_series = selected[:]

    native = optimization._optimize_single_window(
        returns_df,
        "risk_parity",
        selected,
        lower_bounds,
        upper_bounds,
        forced_weights,
        free_series,
    )
    reference = optimization._optimize_risk_parity_riskfolio_reference(
        returns_df,
        selected,
        lower_bounds,
        upper_bounds,
        forced_weights,
    )

    assert reference is not None
    native_arr = np.array([native[s] for s in selected])
    ref_arr = np.array([reference[s] for s in selected])
    assert np.all(np.isfinite(native_arr))
    assert abs(native_arr.sum() - 1.0) <= 1e-6
    assert np.max(np.abs(native_arr - ref_arr)) <= 0.08


def test_risk_parity_native_matches_reference_rolling_box_and_returns():
    returns_df = _build_risk_parity_returns()
    selected = list(returns_df.columns)
    config = {
        "model": "risk_parity",
        "window_type": "rolling",
        "window_size": 126,
        "opt_step": 21,
        "selected_series": selected,
        "missing_data": "fill_na",
        "fill_in_sample": False,
        "min_wt": {s: 0.0 for s in selected},
        "max_wt": {s: 35.0 for s in selected},
        "force_max": {s: False for s in selected},
        "linear_constraints": [],
    }

    native_windows, native_returns, _ = run_portfolio_optimization(returns_df, config)

    windows = optimization._compute_windows(
        returns_df,
        "rolling",
        126,
        21,
        False,
        opt_step_unit="periods",
    )
    lower_bounds = {s: 0.0 for s in selected}
    upper_bounds = {s: 0.35 for s in selected}
    weights_df = pd.DataFrame(0.0, index=returns_df.index, columns=selected)
    reference_windows = []
    for est_start, est_end, apply_start, apply_end in windows:
        est_data = returns_df.iloc[est_start:est_end + 1].copy()
        ref_weights = optimization._optimize_risk_parity_riskfolio_reference(
            est_data,
            selected,
            lower_bounds,
            upper_bounds,
            {},
        )
        assert ref_weights is not None
        reference_windows.append(ref_weights)
        for s in selected:
            weights_df.iloc[apply_start:apply_end + 1, weights_df.columns.get_loc(s)] = ref_weights[s]

    reference_returns = (returns_df.fillna(0) * weights_df).sum(axis=1)
    reference_returns = reference_returns[weights_df.sum(axis=1) > 0]

    assert len(native_windows) == len(reference_windows)
    for native_window, ref_weights in zip(native_windows, reference_windows):
        native_arr = np.array([native_window.weights[s] for s in selected])
        ref_arr = np.array([ref_weights[s] for s in selected])
        assert abs(native_arr.sum() - 1.0) <= 1e-6
        assert np.max(np.abs(native_arr - ref_arr)) <= 0.10

    aligned = pd.concat([native_returns.rename("native"), reference_returns.rename("reference")], axis=1).dropna()
    assert not aligned.empty
    assert np.max(np.abs(aligned["native"] - aligned["reference"])) <= 5e-4


def test_risk_parity_native_respects_linear_constraints():
    returns_df = _build_risk_parity_returns()
    selected = list(returns_df.columns)
    linear_constraints = [
        {"Asset_A": 1.0, "Asset_B": 1.0, "Asset_C": 0.0, "Asset_D": 0.0, "Asset_E": 0.0, "Min": 0.20, "Max": 0.65},
        {"Asset_A": 0.0, "Asset_B": 0.0, "Asset_C": 1.0, "Asset_D": 1.0, "Asset_E": 1.0, "Min": 0.25, "Max": 0.80},
    ]
    config = {
        "model": "risk_parity",
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

    windows, portfolio_returns, _ = run_portfolio_optimization(returns_df, config)
    assert len(windows) > 0
    assert len(portfolio_returns) > 0
    for w in windows:
        assert abs(sum(w.weights.values()) - 1.0) <= 1e-6
        _assert_linear_constraints(w.weights, linear_constraints)

    lower_bounds = {s: 0.0 for s in selected}
    upper_bounds = {s: 1.0 for s in selected}
    sample_window = returns_df.iloc[:126].copy()
    native_or_hybrid = optimization._optimize_single_window(
        sample_window,
        "risk_parity",
        selected,
        lower_bounds,
        upper_bounds,
        {},
        selected[:],
        linear_constraints=linear_constraints,
    )
    reference = optimization._optimize_risk_parity_riskfolio_reference(
        sample_window,
        selected,
        lower_bounds,
        upper_bounds,
        {},
        linear_constraints=linear_constraints,
    )
    assert reference is not None
    native_arr = np.array([native_or_hybrid[s] for s in selected])
    ref_arr = np.array([reference[s] for s in selected])
    assert np.max(np.abs(native_arr - ref_arr)) <= 1e-9
