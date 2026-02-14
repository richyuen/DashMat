"""Smoke test for factor risk parity optimization path.

Run with:
    conda run -n dashmat python test_factor_rp.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.optimization import run_portfolio_optimization


def _build_test_returns() -> pd.DataFrame:
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


def main() -> None:
    returns_df = _build_test_returns()
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

    windows, portfolio_returns = run_portfolio_optimization(returns_df, config)

    assert len(windows) == 1, "Expected a single full-window optimization result."
    weights = windows[0].weights

    assert set(weights.keys()) == set(selected), "Weights should include all selected assets."
    assert np.isfinite(np.array(list(weights.values()))).all(), "Weights must be finite."
    assert abs(sum(weights.values()) - 1.0) <= 1e-6, "Weights must sum to 1."
    assert len(portfolio_returns) > 0, "Portfolio return series should not be empty."
    assert np.isfinite(portfolio_returns.values).all(), "Portfolio returns must be finite."

    print("test_factor_rp.py: PASS")


if __name__ == "__main__":
    main()
