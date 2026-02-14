"""Verify linear constraint handling in optimization.

Run with:
    conda run -n dashmat python verify_linear_constraints.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.optimization import run_portfolio_optimization


def _build_returns() -> pd.DataFrame:
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


def _check_constraints(weights: dict[str, float], constraints: list[dict], tol: float = 1e-6) -> None:
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
            assert lhs + tol >= float(min_v), f"Constraint min violated: lhs={lhs:.6f} min={float(min_v):.6f}"
        if max_v not in (None, ""):
            assert lhs - tol <= float(max_v), f"Constraint max violated: lhs={lhs:.6f} max={float(max_v):.6f}"


def main() -> None:
    returns_df = _build_returns()
    selected = list(returns_df.columns)

    # Constraints are in decimal (0..1) weight space.
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

    windows, portfolio_returns = run_portfolio_optimization(returns_df, config)

    assert len(windows) > 0, "Expected at least one optimization window."
    assert len(portfolio_returns) > 0, "Expected non-empty portfolio returns."

    for w in windows:
        assert abs(sum(w.weights.values()) - 1.0) <= 1e-6, "Weights must sum to 1."
        _check_constraints(w.weights, linear_constraints)

    print("verify_linear_constraints.py: PASS")


if __name__ == "__main__":
    main()
