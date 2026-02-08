"""Test that factor risk parity, risk parity, and HRP produce different weights.

Creates 8 series with block correlation structure and varying volatilities
so that each model's approach to risk allocation yields distinct results.
"""

import numpy as np
import pandas as pd
from utils.optimization import run_portfolio_optimization


def generate_block_correlated_data(n_periods=504, seed=42):
    """Generate 8 series with block correlation and different volatilities.

    Structure:
      Block 1 (equity-like):  A, B, C  — high correlation, different vols
      Block 2 (bond-like):    D, E     — moderate correlation, lower vol
      Block 3 (commodity):    F, G, H  — moderate correlation, high vol

    The three-block structure ensures:
      - Risk Parity: allocates by marginal risk contribution (vol-driven)
      - Factor RP: allocates by factor risk contribution (block-driven,
        with fewer factors than assets the decomposition differs)
      - HRP: clusters by correlation tree, allocates within/between clusters
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_periods, freq="B")

    # Three independent block factors
    f_equity = rng.normal(0, 0.012, n_periods)
    f_bond = rng.normal(0, 0.004, n_periods)
    f_commodity = rng.normal(0, 0.010, n_periods)

    # Small cross-block leakage so blocks aren't perfectly orthogonal
    leak = rng.normal(0, 0.002, n_periods)

    returns = {
        # Block 1: equity-like (high factor loading + varied idio vol)
        "Eq_Large":  f_equity * 1.0 + leak * 0.3 + rng.normal(0.0003, 0.004, n_periods),
        "Eq_Mid":    f_equity * 1.3 + leak * 0.2 + rng.normal(0.0002, 0.007, n_periods),
        "Eq_Small":  f_equity * 1.6 + leak * 0.1 + rng.normal(0.0001, 0.012, n_periods),
        # Block 2: bond-like (lower vol, moderate correlation)
        "Bond_Govt": f_bond * 1.0 - f_equity * 0.15 + rng.normal(0.0001, 0.003, n_periods),
        "Bond_Corp": f_bond * 0.8 - f_equity * 0.05 + rng.normal(0.0001, 0.005, n_periods),
        # Block 3: commodity-like (high vol, own factor)
        "Cmdty_Ene": f_commodity * 1.4 + rng.normal(0, 0.014, n_periods),
        "Cmdty_Met": f_commodity * 1.0 + rng.normal(0, 0.010, n_periods),
        "Cmdty_Agr": f_commodity * 0.6 + rng.normal(0, 0.008, n_periods),
    }

    return pd.DataFrame(returns, index=dates)


def run_model(df, model):
    """Run a full-window optimization and return the weight dict."""
    config = {
        "model": model,
        "window_type": "full",
        "window_size": len(df),
        "opt_step": len(df),
        "opt_step_unit": "periods",
        "missing_data": "fill_na",
        "selected_series": list(df.columns),
    }
    window_results, _ = run_portfolio_optimization(df, config)
    return window_results[0].weights


def print_weights(label, weights, series):
    parts = "  ".join(f"{s}={weights[s]:.4f}" for s in series)
    print(f"  {label:25s} {parts}  sum={sum(weights.values()):.4f}")


def main():
    df = generate_block_correlated_data()
    series = list(df.columns)
    print(f"Data: {len(df)} periods, {series}")
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Annualized vols: {', '.join(f'{s}={df[s].std()*np.sqrt(252):.1%}' for s in series)}\n")

    models = [
        ("risk_parity", "Risk Parity"),
        ("factor_risk_parity", "Factor Risk Parity"),
        ("hrp", "HRP"),
    ]

    results = {}
    for model_id, model_name in models:
        w = run_model(df, model_id)
        results[model_id] = w
        print_weights(model_name, w, series)

    # Check all three are different
    print()
    rounded = {m: tuple(round(w[s], 4) for s in series) for m, w in results.items()}
    all_different = len(set(rounded.values())) == len(models)

    for i, (m1, _) in enumerate(models):
        for m2, _ in models[i+1:]:
            diff = sum(abs(results[m1][s] - results[m2][s]) for s in series)
            print(f"  L1 distance {m1} vs {m2}: {diff:.4f}")

    print()
    if all_different:
        print("PASS: All three models produce different weight vectors")
    else:
        print("FAIL: Some models produced identical weights")


if __name__ == "__main__":
    main()
