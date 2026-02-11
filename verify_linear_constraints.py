
import pandas as pd
import numpy as np
import riskfolio as rp
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from utils.optimization import _optimize_ex_ante_mv, _parse_linear_constraints

def test_linear_constraints():
    print("Testing Linear Constraints in Optimization...")
    
    asset_names = ["AssetA", "AssetB", "AssetC", "AssetD"]
    n_assets = len(asset_names)
    
    # Mock returns: AssetA > AssetB > AssetC > AssetD
    ex_ante_returns = {
        "AssetA": 0.10,
        "AssetB": 0.08,
        "AssetC": 0.06,
        "AssetD": 0.04
    }
    
    # Mock Covariance (diagonal for simplicity)
    cov_values = np.eye(n_assets) * 0.01
    ex_ante_cov = {}
    for i, a in enumerate(asset_names):
        ex_ante_cov[a] = {}
        for j, b in enumerate(asset_names):
            ex_ante_cov[a][b] = cov_values[i, j]
            
    # Standard bounds
    lower_bounds = {a: 0.0 for a in asset_names}
    upper_bounds = {a: 1.0 for a in asset_names}
    forced_weights = {}
    free_series = asset_names
    
    # Case 1: No linear constraints
    # Should allocate heavily to AssetA (highest return, equal risk)
    print("\n--- Case 1: Unconstrained (Maximize Sharpe) ---")
    w1 = _optimize_ex_ante_mv(
        asset_names, lower_bounds, upper_bounds, forced_weights, free_series,
        ex_ante_returns, ex_ante_cov, objective="maximize_sharpe"
    )
    print("Weights:", w1)
    
    # Case 2: Constraint: AssetA + AssetB <= 0.3
    # Expect A+B to be capped, and C/D to take up slack
    print("\n--- Case 2: Constraint (AssetA + AssetB <= 0.3) ---")
    linear_constraints = [
        {"AssetA": 1, "AssetB": 1, "Max": 0.3}
    ]
    w2 = _optimize_ex_ante_mv(
        asset_names, lower_bounds, upper_bounds, forced_weights, free_series,
        ex_ante_returns, ex_ante_cov, objective="maximize_sharpe",
        linear_constraints=linear_constraints
    )
    print("Weights:", w2)
    sum_ab = w2.get("AssetA", 0) + w2.get("AssetB", 0)
    print(f"Sum(A+B) = {sum_ab:.4f} (Expected <= 0.3)")
    if sum_ab > 0.3001:
        print("FAIL: Constraint violated")
    else:
        print("PASS: Constraint satisfied")
        
    # Case 3: Constraint: AssetC >= 0.4
    # Expect C to be at least 0.4
    print("\n--- Case 3: Constraint (AssetC >= 0.4) ---")
    linear_constraints = [
        {"AssetC": 1, "Min": 0.4}
    ]
    w3 = _optimize_ex_ante_mv(
        asset_names, lower_bounds, upper_bounds, forced_weights, free_series,
        ex_ante_returns, ex_ante_cov, objective="maximize_sharpe",
        linear_constraints=linear_constraints
    )
    print("Weights:", w3)
    val_c = w3.get("AssetC", 0)
    print(f"AssetC = {val_c:.4f} (Expected >= 0.4)")
    if val_c < 0.3999:
        print("FAIL: Constraint violated")
    else:
        print("PASS: Constraint satisfied")

    # Case 4: complex constraint: A - B <= 0 (B >= A)
    print("\n--- Case 4: Constraint (AssetA - AssetB <= 0 -> AssetB >= AssetA) ---")
    linear_constraints = [
        {"AssetA": 1, "AssetB": -1, "Max": 0}
    ]
    w4 = _optimize_ex_ante_mv(
        asset_names, lower_bounds, upper_bounds, forced_weights, free_series,
        ex_ante_returns, ex_ante_cov, objective="maximize_sharpe",
        linear_constraints=linear_constraints
    )
    print("Weights:", w4)
    diff = w4.get("AssetA", 0) - w4.get("AssetB", 0)
    print(f"A - B = {diff:.4f} (Expected <= 0)")
    if diff > 0.0001:
        print("FAIL: Constraint violated")
    else:
        print("PASS: Constraint satisfied")


if __name__ == "__main__":
    test_linear_constraints()
