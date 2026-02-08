"""Test factor risk parity with shifting volatility regimes.

Creates 3 series where volatility regimes shift over time, ensuring
that factor risk parity produces different weights in each rolling window.
"""

import numpy as np
import pandas as pd
from utils.optimization import run_portfolio_optimization


def generate_regime_data(n_periods=756, seed=42):
    """Generate 3 series with distinct volatility regimes.

    Split into 3 equal regimes (252 periods each):
      Regime 1: A is high-vol, B and C are low-vol
      Regime 2: B is high-vol, A and C are low-vol
      Regime 3: C is high-vol, A and B are low-vol

    This guarantees risk parity shifts weight away from the high-vol
    series in each regime.
    """
    rng = np.random.default_rng(seed)
    n_third = n_periods // 3

    dates = pd.bdate_range("2020-01-01", periods=n_periods, freq="B")

    high_vol = 0.03   # ~48% annualized
    low_vol = 0.005   # ~8% annualized

    returns = {}
    for label, high_regime in [("A", 0), ("B", 1), ("C", 2)]:
        vols = np.full(n_periods, low_vol)
        vols[high_regime * n_third : (high_regime + 1) * n_third] = high_vol
        returns[label] = rng.normal(0.0003, vols)

    return pd.DataFrame(returns, index=dates)


def main():
    df = generate_regime_data()
    print(f"Data: {len(df)} periods, {list(df.columns)}")
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}\n")

    config = {
        "model": "factor_risk_parity",
        "window_type": "rolling",
        "window_size": 252,
        "opt_step": 252,
        "opt_step_unit": "periods",
        "fill_in_sample": False,
        "missing_data": "fill_na",
        "selected_series": ["A", "B", "C"],
    }

    window_results, port_returns = run_portfolio_optimization(df, config)

    print(f"Windows: {len(window_results)}")
    print(f"Portfolio returns: {len(port_returns)} periods\n")

    for i, wr in enumerate(window_results):
        w = wr.weights
        print(f"Window {i+1}: {wr.apply_start.date()} to {wr.apply_end.date()}")
        print(f"  A={w['A']:.4f}  B={w['B']:.4f}  C={w['C']:.4f}  sum={sum(w.values()):.4f}")

    # Verify weights actually change
    all_weights = [tuple(round(wr.weights[s], 4) for s in ["A", "B", "C"]) for wr in window_results]
    unique = set(all_weights)
    print(f"\nUnique weight vectors: {len(unique)} (expect {len(window_results)})")
    if len(unique) > 1:
        print("PASS: Weights change between windows")
    else:
        print("FAIL: Weights are identical across all windows")


if __name__ == "__main__":
    main()
