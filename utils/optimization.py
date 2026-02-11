"""Portfolio optimization engine using riskfolio-lib."""

from dataclasses import dataclass
import warnings
import numpy as np
import pandas as pd
import riskfolio as rp

# Suppress cvxpy deprecation warning from riskfolio-lib internals
warnings.filterwarnings("ignore", category=UserWarning, module="cvxpy")


@dataclass
class WindowResult:
    """Result of a single optimization window."""
    apply_start: pd.Timestamp
    apply_end: pd.Timestamp
    weights: dict  # {series_name: weight}
    est_start: pd.Timestamp = None  # Estimation window start
    est_end: pd.Timestamp = None    # Estimation window end    


def _compute_windows(df, window_type, window_size, opt_step, fill_in_sample=False,
                     opt_step_unit="periods"):
    """Compute optimization windows as (est_start, est_end, apply_start, apply_end) tuples.

    Args:
        df: DataFrame with DatetimeIndex
        window_type: 'full', 'expanding', or 'rolling'
        window_size: Number of periods for initial window
        opt_step: Number of periods (or months if opt_step_unit='months') to step forward
        fill_in_sample: If True, first window weights apply from period 0 (in-sample).
                        If False, first window weights apply from period window_size (out-of-sample only).
        opt_step_unit: 'periods' for raw count stepping, 'months' for calendar-month-anchored stepping

    Returns:
        List of (est_start_idx, est_end_idx, apply_start_idx, apply_end_idx) tuples
    """
    n = len(df)

    if window_type == "full":
        return [(0, n - 1, 0, n - 1)]

    if window_size > n:
        raise ValueError(
            f"Insufficient data for window size: {window_size} periods required but only {n} available"
        )

    if opt_step_unit == "months":
        return _compute_windows_monthly(df, window_type, window_size, opt_step, fill_in_sample)

    windows = []

    if window_type == "expanding":
        est_end = window_size - 1
        if fill_in_sample:
            apply_start = 0
        else:
            apply_start = window_size
        apply_end = min(window_size + opt_step - 1, n - 1)
        windows.append((0, est_end, apply_start, apply_end))

        while apply_end < n - 1:
            est_end = min(apply_end, n - 1)
            new_apply_start = apply_end + 1
            new_apply_end = min(new_apply_start + opt_step - 1, n - 1)
            windows.append((0, est_end, new_apply_start, new_apply_end))
            apply_end = new_apply_end

    elif window_type == "rolling":
        est_start = 0
        est_end = window_size - 1
        if fill_in_sample:
            apply_start = 0
        else:
            apply_start = window_size
        apply_end = min(window_size + opt_step - 1, n - 1)
        windows.append((est_start, est_end, apply_start, apply_end))

        while apply_end < n - 1:
            est_start = est_start + opt_step
            est_end = est_start + window_size - 1
            if est_end >= n:
                est_end = n - 1
            new_apply_start = apply_end + 1
            new_apply_end = min(new_apply_start + opt_step - 1, n - 1)
            windows.append((est_start, est_end, new_apply_start, new_apply_end))
            apply_end = new_apply_end

    return windows


def _compute_windows_monthly(df, window_type, window_size, opt_step_months, fill_in_sample):
    """Compute optimization windows anchored to calendar month-end dates.

    Rebalance points snap to the last available date in df.index on or before
    each target calendar month-end.

    Args:
        df: DataFrame with DatetimeIndex (must be sorted)
        window_type: 'expanding' or 'rolling'
        window_size: Number of periods for estimation window
        opt_step_months: Number of calendar months between rebalance points
        fill_in_sample: If True, first window applies from period 0

    Returns:
        List of (est_start_idx, est_end_idx, apply_start_idx, apply_end_idx) tuples
    """
    n = len(df)
    idx = df.index

    # Find the first anchor: month-end at or after the end of the first estimation window
    first_est_end_date = idx[window_size - 1]
    # Snap to the calendar month-end on or after this date
    anchor_date = first_est_end_date + pd.offsets.MonthEnd(0)
    # Find the last index date <= anchor_date
    anchor_pos = idx.searchsorted(anchor_date, side="right") - 1
    if anchor_pos < window_size - 1:
        # Month-end fell before our minimum window; move to next month-end
        anchor_date = first_est_end_date + pd.offsets.MonthEnd(1)
        anchor_pos = idx.searchsorted(anchor_date, side="right") - 1

    if anchor_pos < 0 or anchor_pos >= n:
        raise ValueError("Cannot find a valid month-end anchor within the data range.")

    # Build list of anchor positions (index positions where rebalancing occurs)
    anchors = []
    current_anchor_date = anchor_date
    while True:
        pos = idx.searchsorted(current_anchor_date, side="right") - 1
        if pos >= n:
            pos = n - 1
        if pos < 0:
            break
        # Avoid duplicate anchors
        if not anchors or pos > anchors[-1]:
            anchors.append(pos)
        # Step forward by opt_step_months
        current_anchor_date = current_anchor_date + pd.DateOffset(months=opt_step_months)
        # Snap to month-end
        current_anchor_date = current_anchor_date + pd.offsets.MonthEnd(0)
        if pos >= n - 1:
            break

    if not anchors:
        raise ValueError("No valid rebalance points found for the given data and step size.")

    windows = []

    for i, anchor in enumerate(anchors):
        # Estimation window: data up to and including anchor
        if window_type == "expanding":
            est_start = 0
        else:  # rolling
            est_start = max(0, anchor - window_size + 1)
        est_end = anchor

        # Application window: weights apply from anchor+1 until the next anchor
        if i == 0 and fill_in_sample:
            apply_start = 0
        else:
            apply_start = anchor + 1
            if apply_start >= n:
                continue

        if i < len(anchors) - 1:
            apply_end = anchors[i + 1]
        else:
            apply_end = n - 1

        if apply_start <= apply_end:
            windows.append((est_start, est_end, apply_start, apply_end))

    return windows


def _validate_weight_constraints(asset_names, lower_bounds, upper_bounds, forced_weights):
    """Validate that weight constraints are feasible.

    Args:
        asset_names: List of asset names
        lower_bounds: Dict {name: lower_bound} (0-1 scale)
        upper_bounds: Dict {name: upper_bound} (0-1 scale)
        forced_weights: Dict {name: weight} for forced assets (0-1 scale)

    Raises:
        ValueError if constraints are infeasible
    """
    forced_total = sum(forced_weights.values())
    if forced_total > 1.0 + 1e-9:
        raise ValueError(
            f"Forced weights sum to {forced_total*100:.1f}%, which exceeds 100%. "
            f"Reduce forced allocations."
        )

    free_assets = [a for a in asset_names if a not in forced_weights]
    if not free_assets:
        # All assets are forced - check they sum to ~1
        if abs(forced_total - 1.0) > 0.01:
            raise ValueError(
                f"All series use Force Max but weights sum to {forced_total*100:.1f}%, not 100%."
            )
        return

    remaining_budget = 1.0 - forced_total

    free_lower_sum = sum(lower_bounds.get(a, 0) for a in free_assets)
    free_upper_sum = sum(upper_bounds.get(a, 1) for a in free_assets)

    if free_lower_sum > remaining_budget + 1e-9:
        raise ValueError(
            f"Minimum weights for free series sum to {free_lower_sum*100:.1f}%, "
            f"but only {remaining_budget*100:.1f}% budget remains after forced allocations. "
            f"Reduce minimum weights or forced allocations."
        )

    if free_upper_sum < remaining_budget - 1e-9:
        raise ValueError(
            f"Maximum weights for free series sum to {free_upper_sum*100:.1f}%, "
            f"but {remaining_budget*100:.1f}% budget needs to be allocated. "
            f"Increase maximum weights or reduce forced allocations."
        )


def _parse_linear_constraints(linear_constraints, asset_names):
    """Parse linear constraints into A and B matrices for Riskfolio.
    
    Riskfolio constraint form: A @ w <= B
    
    Args:
        linear_constraints: List of dicts from the UI grid.
            Each dict has 'Min', 'Max', and asset keys.
        asset_names: List of asset names (columns of A).
            
    Returns:
        (A, B) tuple of numpy arrays, or (None, None) if no constraints.
    """
    if not linear_constraints:
        return None, None
        
    A_list = []
    B_list = []
    
    asset_idx = {name: i for i, name in enumerate(asset_names)}
    n_assets = len(asset_names)
    
    for row in linear_constraints:
        # Extract coefficients
        coeffs = np.zeros(n_assets)
        has_coeffs = False
        for name, idx in asset_idx.items():
            val = row.get(name)
            if val is not None and val != "":
                try:
                    coeffs[idx] = float(val)
                    if coeffs[idx] != 0:
                        has_coeffs = True
                except ValueError:
                    pass
                    
        if not has_coeffs:
            # If no coefficients, check if user meant "Sum(w) >= Min" (i.e. implied all 1s)? 
            # No, assume explicit coefficients required.
            continue
            
        # Min constraint: Sum(w * c) >= Min  =>  -Sum(w * c) <= -Min
        min_val = row.get("Min")
        if min_val is not None and min_val != "":
            try:
                min_val = float(min_val)
                A_list.append(-coeffs)
                B_list.append(-min_val)
            except ValueError:
                pass
                
        # Max constraint: Sum(w * c) <= Max
        max_val = row.get("Max")
        if max_val is not None and max_val != "":
            try:
                max_val = float(max_val)
                A_list.append(coeffs)
                B_list.append(max_val)
            except ValueError:
                pass
                
    if not A_list:
        return None, None
        
    return np.array(A_list), np.array(B_list).reshape(-1, 1)


def _extract_pca_factors(returns_df, n_factors=None):
    """Extract PCA-based statistical factors from returns.

    Args:
        returns_df: DataFrame of asset returns (clean, no NaN)
        n_factors: Number of PCA factors to extract. Defaults to
                   min(n_assets - 1, n_observations) to ensure the factor
                   model is a true dimensionality reduction (not full rank).

    Returns:
        DataFrame of factor returns
    """
    from sklearn.decomposition import PCA

    n_assets = returns_df.shape[1]
    n_obs = returns_df.shape[0]
    if n_factors is None:
        # Use roughly half the assets — enough to capture the main factor
        # structure while keeping the decomposition a true reduction.
        n_factors = max(1, n_assets // 2)
    n_factors = min(n_factors, n_assets - 1, n_obs)
    if n_factors < 1:
        n_factors = 1

    pca = PCA(n_components=n_factors)
    factor_returns = pca.fit_transform(returns_df.values)
    factor_df = pd.DataFrame(
        factor_returns,
        index=returns_df.index,
        columns=[f"Factor_{i+1}" for i in range(n_factors)],
    )
    return factor_df


def _optimize_ex_ante_mv(asset_names, lower_bounds, upper_bounds,
                         forced_weights, free_series,
                         ex_ante_returns, ex_ante_cov, objective,
        window_data=None, exp_wt_cov=False, halflife=63,
        ex_ante_vol=None, ex_ante_corr=None, linear_constraints=None):
    """Run ex ante mean-variance optimization.

    Args:
        asset_names: List of asset names
        lower_bounds: Dict {name: lower_bound} (0-1 scale)
        upper_bounds: Dict {name: upper_bound} (0-1 scale)
        forced_weights: Dict {name: weight} for forced assets
        free_series: List of free (non-forced) asset names
        ex_ante_returns: Dict {name: expected_annual_return} (decimal, e.g. 0.08 for 8%)
        ex_ante_cov: Nested dict or 2D structure {row_name: {col_name: value}}
        objective: 'maximize_sharpe', 'minimize_variance', or 'maximize_return'
        window_data: Optional DataFrame of returns (used to create Portfolio object)
        exp_wt_cov: Whether to use exponentially weighted covariance (if estimating from data)
        halflife: Halflife for exponentially weighted covariance
        ex_ante_vol: Dict {name: expected_annual_volatility} (decimal, e.g. 0.15 for 15%)
        ex_ante_corr: Nested dict or 2D structure {row_name: {col_name: correlation}}

    Returns:
        Dict of {asset_name: weight}
    """
    if not free_series:
        return dict(forced_weights)

    if len(free_series) == 1:
        result = dict(forced_weights)
        remaining = 1.0 - sum(forced_weights.values())
        result[free_series[0]] = max(0, remaining)
        return result

    n_assets = len(asset_names)

    # Build mu vector (DataFrame with shape 1 x n_assets, row vector — riskfolio expects mu @ w)
    mu_values = [ex_ante_returns.get(name, 0.0) for name in asset_names]
    mu_df = pd.DataFrame([mu_values], columns=asset_names)

    # Build cov matrix — if user supplied one, use it; otherwise estimate from data
    has_custom_cov = bool(ex_ante_cov)
    if has_custom_cov:
        cov_values = np.zeros((n_assets, n_assets))
        for i, row_name in enumerate(asset_names):
            row_vals = ex_ante_cov.get(row_name, {})
            for j, col_name in enumerate(asset_names):
                cov_values[i, j] = row_vals.get(col_name, 0.0)
        cov_df = pd.DataFrame(cov_values, index=asset_names, columns=asset_names)
    elif ex_ante_vol and ex_ante_corr:
        # Construct covariance from volatility and correlation: Cov = D * Corr * D
        # where D is diag(volatility)
        vol_vec = np.array([ex_ante_vol.get(name, 0.0) for name in asset_names])
        corr_mat = np.eye(n_assets)
        for i, row_name in enumerate(asset_names):
            row_vals = ex_ante_corr.get(row_name, {})
            for j, col_name in enumerate(asset_names):
                corr_mat[i, j] = row_vals.get(col_name, 0.0 if i != j else 1.0)
        
        # Ensure symmetry for correlation (sometimes user input might be upper tri)
        # But we assume the grid gives us full matrix or we trust the input loop
        
        # Cov_ij = Vol_i * Vol_j * Corr_ij
        D = np.diag(vol_vec)
        cov_values = D @ corr_mat @ D
        cov_df = pd.DataFrame(cov_values, index=asset_names, columns=asset_names)
    elif window_data is not None:
        # Estimate from historical returns — drop rows with NaN for clean estimation
        clean_data = window_data[asset_names].dropna()
        if len(clean_data) < 2:
            # Not enough clean data — use fillna(0) as fallback
            clean_data = window_data[asset_names].fillna(0)
        
        if exp_wt_cov:
            # Estimate using exponential weighting
            n_assets = len(asset_names)
            cov_df = clean_data.ewm(halflife=halflife).cov().iloc[-n_assets:]
        else:
            cov_df = clean_data.cov()
    else:
        raise ValueError("No covariance matrix provided and no historical data to estimate from.")

    # Create Portfolio object — use clean data (drop NaN rows)
    if window_data is not None:
        clean_returns = window_data[asset_names].dropna()
        if len(clean_returns) < 10:
            clean_returns = window_data[asset_names].fillna(0)
        port = rp.Portfolio(returns=clean_returns.copy())
    else:
        # Create a minimal dummy returns DataFrame
        dummy = pd.DataFrame(
            np.random.randn(100, n_assets) * 0.01,
            columns=asset_names
        )
        port = rp.Portfolio(returns=dummy)

    # First compute default stats so internal structures are initialized
    port.assets_stats(method_mu="hist", method_cov="hist")

    # Override with custom mu and cov
    port.mu = mu_df
    port.cov = cov_df

    # Build bounds
    lower_arr = np.zeros(n_assets)
    upper_arr = np.ones(n_assets)
    for i, name in enumerate(asset_names):
        if name in forced_weights:
            lower_arr[i] = forced_weights[name]
            upper_arr[i] = forced_weights[name]
        else:
            lower_arr[i] = lower_bounds.get(name, 0)
            upper_arr[i] = upper_bounds.get(name, 1)
    port.lowerlng = lower_arr.reshape(-1, 1)
    port.upperlng = upper_arr.reshape(-1, 1)

    # Apply linear constraints if any
    if linear_constraints:
        A, B = _parse_linear_constraints(linear_constraints, asset_names)
        if A is not None:
            port.ainequality = A
            port.binequality = B

    # Map objective to riskfolio params
    obj_map = {
        "maximize_sharpe": ("Sharpe", "MV"),
        "minimize_variance": ("MinRisk", "MV"),
        "maximize_return": ("MaxRet", "MV"),
    }
    obj_str, rm = obj_map.get(objective, ("Sharpe", "MV"))

    # If all expected returns are zero/near-zero, Sharpe is degenerate — fall back to MinRisk
    mu_arr = mu_df.values.flatten()
    if np.allclose(mu_arr, 0, atol=1e-10) and obj_str == "Sharpe":
        obj_str = "MinRisk"

    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w = port.optimization(model="Classic", rm=rm, obj=obj_str, hist=False)
    except Exception:
        w = None

    # If primary objective failed, retry with MinRisk as fallback
    if (w is None or w.empty) and obj_str != "MinRisk":
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                w = port.optimization(model="Classic", rm=rm, obj="MinRisk", hist=False)
        except Exception:
            w = None

    # If all optimization attempts failed, fall back to equal weight
    if w is None or (hasattr(w, 'empty') and w.empty):
        result = dict(forced_weights)
        remaining = 1.0 - sum(forced_weights.values())
        eq = remaining / len(free_series) if free_series else 0
        for s in free_series:
            result[s] = eq
        return result

    result = {}
    for name in asset_names:
        if name in w.index:
            result[name] = float(w.loc[name, "weights"])
        elif name in forced_weights:
            result[name] = forced_weights[name]
        else:
            result[name] = 0.0
    return result


def _optimize_black_litterman(window_data, asset_names, lower_bounds, upper_bounds,
                              forced_weights, free_series,
                              bl_views, bl_tau, objective,
                              exp_wt_cov=False, halflife=63):
    """Run Black-Litterman optimization.

    Uses historical returns for the equilibrium prior, then blends with user views.

    Args:
        window_data: DataFrame of returns for estimation
        asset_names: List of asset names
        lower_bounds: Dict {name: lower_bound} (0-1 scale)
        upper_bounds: Dict {name: upper_bound} (0-1 scale)
        forced_weights: Dict {name: weight} for forced assets
        free_series: List of free (non-forced) asset names
        bl_views: List of view dicts, each with:
            - 'type': 'absolute' or 'relative'
            - 'asset': asset name (for absolute) or asset_from (for relative)
            - 'asset_to': asset name (for relative, the underperformer)
            - 'return': expected return (decimal)
            - 'confidence': confidence level 0-1 (default 1.0)
        bl_tau: Scalar uncertainty parameter (default 0.05)
        objective: 'maximize_sharpe', 'minimize_variance', or 'maximize_return'
        exp_wt_cov: Whether to use exponentially weighted covariance
        halflife: Halflife for exponentially weighted covariance

    Returns:
        Dict of {asset_name: weight}
    """
    if not free_series:
        return dict(forced_weights)

    if len(free_series) == 1:
        result = dict(forced_weights)
        remaining = 1.0 - sum(forced_weights.values())
        result[free_series[0]] = max(0, remaining)
        return result

    n_assets = len(asset_names)
    port_data = window_data[asset_names].copy()

    port = rp.Portfolio(returns=port_data)
    port.assets_stats(method_mu="hist", method_cov="hist")

    if exp_wt_cov:
        port.cov = port_data.ewm(halflife=halflife).cov().iloc[-n_assets:]

    # Build P (views matrix) and Q (views vector)
    n_views = len(bl_views)
    if n_views == 0:
        # No views: fall back to historical MV
        P = np.zeros((1, n_assets))
        Q = np.zeros((1, 1))
    else:
        P = np.zeros((n_views, n_assets))
        Q = np.zeros((n_views, 1))

        asset_idx = {name: i for i, name in enumerate(asset_names)}
        for v_idx, view in enumerate(bl_views):
            view_type = view.get("type", "absolute")
            view_return = view.get("return", 0.0)
            Q[v_idx, 0] = view_return

            if view_type == "absolute":
                asset_name = view.get("asset", "")
                if asset_name in asset_idx:
                    P[v_idx, asset_idx[asset_name]] = 1.0
            elif view_type == "relative":
                asset_from = view.get("asset", "")
                asset_to = view.get("asset_to", "")
                if asset_from in asset_idx:
                    P[v_idx, asset_idx[asset_from]] = 1.0
                if asset_to in asset_idx:
                    P[v_idx, asset_idx[asset_to]] = -1.0

    P_df = pd.DataFrame(P, columns=asset_names)
    Q_df = pd.DataFrame(Q, columns=["views"])

    # Build confidence diagonal (omega)
    # Default omega = tau * diag(P * Sigma * P')
    # Scale by inverse of confidence: higher confidence -> smaller omega entry
    confidences = []
    for view in bl_views:
        conf = view.get("confidence", 1.0)
        confidences.append(max(0.01, min(1.0, conf)))

    try:
        port.blacklitterman_stats(
            P=P_df,
            Q=Q_df,
            delta=None,  # auto-compute market risk aversion
            rf=0,
            eq=True,  # use equilibrium returns as prior
        )

        # Scale omega by confidence (lower confidence = more uncertainty)
        if hasattr(port, "cov_bl") and port.cov_bl is not None:
            sigma = port.cov_bl.values if hasattr(port.cov_bl, 'values') else port.cov_bl
        else:
            sigma = port.cov.values

        omega_diag = bl_tau * np.diag(P @ sigma @ P.T)
        for i, conf in enumerate(confidences):
            omega_diag[i] /= conf

    except Exception:
        pass  # Use whatever BL stats were computed

    # Build bounds
    lower_arr = np.zeros(n_assets)
    upper_arr = np.ones(n_assets)
    for i, name in enumerate(asset_names):
        if name in forced_weights:
            lower_arr[i] = forced_weights[name]
            upper_arr[i] = forced_weights[name]
        else:
            lower_arr[i] = lower_bounds.get(name, 0)
            upper_arr[i] = upper_bounds.get(name, 1)
    port.lowerlng = lower_arr.reshape(-1, 1)
    port.upperlng = upper_arr.reshape(-1, 1)

    # Apply linear constraints if any
    if linear_constraints:
        A, B = _parse_linear_constraints(linear_constraints, asset_names)
        if A is not None:
            port.ainequality = A
            port.binequality = B

    # Map objective
    obj_map = {
        "maximize_sharpe": ("Sharpe", "MV"),
        "minimize_variance": ("MinRisk", "MV"),
        "maximize_return": ("MaxRet", "MV"),
    }
    obj_str, rm = obj_map.get(objective, ("Sharpe", "MV"))

    try:
        w = port.optimization(model="BL", rm=rm, obj=obj_str, hist=False)
    except Exception:
        # Fallback to equal weight
        result = dict(forced_weights)
        remaining = 1.0 - sum(forced_weights.values())
        eq = remaining / len(free_series) if free_series else 0
        for s in free_series:
            result[s] = eq
        return result

    if w is None or w.empty:
        result = dict(forced_weights)
        remaining = 1.0 - sum(forced_weights.values())
        eq = remaining / len(free_series) if free_series else 0
        for s in free_series:
            result[s] = eq
        return result

    result = {}
    for name in asset_names:
        if name in w.index:
            result[name] = float(w.loc[name, "weights"])
        elif name in forced_weights:
            result[name] = forced_weights[name]
        else:
            result[name] = 0.0
    return result


def _optimize_single_window(window_data, model, asset_names, lower_bounds, upper_bounds,
                             forced_weights, free_series, exp_wt_cov=False, halflife=63,
                             ex_ante_returns=None, ex_ante_cov=None,
                             ex_ante_vol=None, ex_ante_corr=None,
                             bl_views=None, bl_tau=0.05, objective="maximize_sharpe",
                             linear_constraints=None):
    """Run optimization for a single window.

    Args:
        window_data: DataFrame of returns for this estimation window
        model: Optimization model name
        asset_names: List of asset names participating in this window
        lower_bounds: Dict {name: lower_bound} (0-1 scale)
        upper_bounds: Dict {name: upper_bound} (0-1 scale)
        forced_weights: Dict {name: weight} for forced assets
        free_series: List of free (non-forced) asset names
        exp_wt_cov: Whether to use exponentially weighted covariance
        halflife: Halflife for exponentially weighted covariance
        ex_ante_returns: Optional dict of ex-ante returns
        ex_ante_cov: Optional dict of ex-ante covariance
        ex_ante_vol: Optional dict of ex-ante volatility
        ex_ante_corr: Optional dict of ex-ante correlation
        bl_views: List of Black-Litterman views
        bl_tau: Black-Litterman uncertainty parameter
        objective: Optimization objective
        linear_constraints: List of dicts defining linear constraints

    Returns:
        Dict of {asset_name: weight}
    """
    if not free_series:
        # All forced - just return forced weights
        return dict(forced_weights)

    if len(free_series) == 1:
        # Only one free series - it gets the remaining budget
        result = dict(forced_weights)
        remaining = 1.0 - sum(forced_weights.values())
        result[free_series[0]] = max(0, remaining)
        return result

    # Equal weight model doesn't need optimization
    if model == "equal_weight":
        result = dict(forced_weights)
        remaining = 1.0 - sum(forced_weights.values())
        equal_wt = remaining / len(free_series)
        for s in free_series:
            result[s] = equal_wt
        return result

    # Ex ante models use separate optimization functions
    if model == "ex_ante_mv":
        return _optimize_ex_ante_mv(
            asset_names, lower_bounds, upper_bounds,
            forced_weights, free_series,
            ex_ante_returns or {}, ex_ante_cov or {},
            objective, window_data,
            exp_wt_cov, halflife,
            ex_ante_vol, ex_ante_corr,
            linear_constraints,
        )

    if model == "black_litterman":
        return _optimize_black_litterman(
            window_data, asset_names, lower_bounds, upper_bounds,
            forced_weights, free_series,
            bl_views or [], bl_tau, objective,
            exp_wt_cov, halflife,
            linear_constraints,
        )

    # Build riskfolio Portfolio
    # Use only the columns that are in asset_names
    port_data = window_data[asset_names].copy()

    port = rp.Portfolio(returns=port_data)

    # Compute asset-level stats (mu, cov) for all models
    port.assets_stats(method_mu="hist", method_cov="hist")

    # For factor risk parity, also compute factor-level stats
    if model == "factor_risk_parity":
        port.factors = _extract_pca_factors(port_data)
        port.factors_stats(method_mu="hist", method_cov="hist")

    # Override covariance if exponentially weighted
    if exp_wt_cov:
        port.cov = port_data.ewm(halflife=halflife).cov().iloc[-len(asset_names):]
        # Also override mu with exponentially weighted mean
        port.mu = port_data.ewm(halflife=halflife).mean().iloc[-1:].T
        port.mu.columns = ["mu"]

    # Build bounds arrays (ordered by asset_names)
    n_assets = len(asset_names)
    lower_arr = np.zeros(n_assets)
    upper_arr = np.ones(n_assets)

    for i, name in enumerate(asset_names):
        if name in forced_weights:
            lower_arr[i] = forced_weights[name]
            upper_arr[i] = forced_weights[name]
        else:
            lower_arr[i] = lower_bounds.get(name, 0)
            upper_arr[i] = upper_bounds.get(name, 1)

    # Set bounds for Classic optimization (uses lowerlng/upperlng)
    # Use numpy arrays (not DataFrames) for cvxpy compatibility in Sharpe formulation
    port.lowerlng = lower_arr.reshape(-1, 1)
    port.upperlng = upper_arr.reshape(-1, 1)

    # For risk parity, convert box constraints to linear inequality constraints
    # ainequality * w <= binequality
    # lower_i <= w_i  =>  -w_i <= -lower_i  (row: -e_i, bound: -lower_i)
    # w_i <= upper_i  =>  w_i <= upper_i     (row: e_i, bound: upper_i)
    has_nontrivial_bounds = any(lower_arr[i] > 0 or upper_arr[i] < 1 for i in range(n_assets))
    if has_nontrivial_bounds:
        A_rows = []
        b_rows = []
        for i, name in enumerate(asset_names): # Iterate over asset_names to ensure correct order
            if lower_arr[i] > 0:
                row = np.zeros(n_assets)
                row[i] = -1.0
                A_rows.append(row)
                b_rows.append(-lower_arr[i])
            if upper_arr[i] < 1:
                row = np.zeros(n_assets)
                row[i] = 1.0
                A_rows.append(row)
                b_rows.append(upper_arr[i])
        if A_rows:
            # Append to existing ainequality/binequality if they exist
            if hasattr(port, 'ainequality') and port.ainequality is not None:
                port.ainequality = pd.concat([port.ainequality, pd.DataFrame(np.array(A_rows), columns=asset_names)])
                port.binequality = pd.concat([port.binequality, pd.DataFrame(np.array(b_rows), columns=["b"])])
            else:
                port.ainequality = pd.DataFrame(np.array(A_rows), columns=asset_names)
                port.binequality = pd.DataFrame(np.array(b_rows), columns=["b"])

    # Apply additional linear constraints if any
    if linear_constraints:
        A_lc, B_lc = _parse_linear_constraints(linear_constraints, asset_names)
        if A_lc is not None:
            if hasattr(port, 'ainequality') and port.ainequality is not None:
                port.ainequality = pd.concat([port.ainequality, A_lc])
                port.binequality = pd.concat([port.binequality, B_lc])
            else:
                port.ainequality = A_lc
                port.binequality = B_lc

    try:
        if model == "risk_parity":
            w = port.rp_optimization(model="Classic", rm="MV", hist=True)
        elif model == "factor_risk_parity":
            w = port.rp_optimization(model="FM", rm="MV", hist=False)
        elif model == "hrp":
            hc_port = rp.HCPortfolio(returns=port_data)
            w = hc_port.optimization(model="HRP", rm="MV", codependence="pearson", leaf_order=True)
        elif model == "maximize_sharpe":
            use_hist = not exp_wt_cov  # Use overridden mu/cov when exp_wt is on
            w = port.optimization(model="Classic", rm="MV", obj="Sharpe", hist=use_hist)
        elif model == "minimize_cvar":
            w = port.optimization(model="Classic", rm="CVaR", obj="MinRisk", hist=True)
        elif model == "minimize_variance":
            w = port.optimization(model="Classic", rm="MV", obj="MinRisk", hist=True)
        else:
            raise ValueError(f"Unknown model: {model}")
    except Exception:
        # Fallback to equal weight on optimization failure
        result = dict(forced_weights)
        remaining = 1.0 - sum(forced_weights.values())
        equal_wt = remaining / len(free_series)
        for s in free_series:
            result[s] = equal_wt
        return result

    if w is None or w.empty:
        # Fallback to equal weight
        result = dict(forced_weights)
        remaining = 1.0 - sum(forced_weights.values())
        equal_wt = remaining / len(free_series)
        for s in free_series:
            result[s] = equal_wt
        return result

    # Convert result to dict
    result = {}
    for name in asset_names:
        if name in w.index:
            result[name] = float(w.loc[name, "weights"])
        elif name in forced_weights:
            result[name] = forced_weights[name]
        else:
            result[name] = 0.0

    return result


def run_portfolio_optimization(returns_df, config, progress_callback=None):
    """Run portfolio optimization.

    Args:
        returns_df: DataFrame of working returns (selected series, aligned, date-filtered)
        config: Dict with optimization parameters
        progress_callback: Dash set_progress function for background callbacks

    Returns:
        Tuple of (list[WindowResult], pd.Series of portfolio returns)

    Raises:
        ValueError: If constraints are infeasible or insufficient data
    """
    model = config.get("model", "risk_parity")
    window_type = config.get("window_type", "full")
    window_size = config.get("window_size", 252)
    opt_step = config.get("opt_step", 252)
    opt_step_unit = config.get("opt_step_unit", "periods")
    exp_wt_cov = config.get("exp_wt_cov", False)
    halflife = config.get("halflife", 63)
    missing_data = config.get("missing_data", "fill_na")
    fill_in_sample = config.get("fill_in_sample", False)
    selected_series = config.get("selected_series", list(returns_df.columns))
    min_wt = config.get("min_wt", {})  # percent 0-100
    max_wt = config.get("max_wt", {})  # percent 0-100
    force_max = config.get("force_max", {})

    # Ex ante config
    ex_ante_returns = config.get("ex_ante_returns", None)
    ex_ante_cov = config.get("ex_ante_cov", None)
    ex_ante_vol = config.get("ex_ante_vol", None)
    ex_ante_corr = config.get("ex_ante_corr", None)
    bl_views = config.get("bl_views", None)
    bl_tau = config.get("bl_tau", 0.05)
    objective = config.get("objective", "maximize_sharpe")
    linear_constraints = config.get("linear_constraints", None)

    # Filter to selected series only
    available_cols = [s for s in selected_series if s in returns_df.columns]
    if not available_cols:
        raise ValueError("No valid series selected for optimization.")

    df = returns_df[available_cols].copy()

    # Convert percent constraints to decimal
    lower_bounds = {s: min_wt.get(s, 0) / 100.0 for s in available_cols}
    upper_bounds = {s: max_wt.get(s, 100) / 100.0 for s in available_cols}

    # Identify forced weights
    forced_weights = {}
    for s in available_cols:
        if force_max.get(s, False):
            forced_weights[s] = upper_bounds[s]

    free_series = [s for s in available_cols if s not in forced_weights]

    # Validate constraints globally
    _validate_weight_constraints(available_cols, lower_bounds, upper_bounds, forced_weights, linear_constraints)

    # Check if all (or all but one) are forced - skip optimization
    if len(free_series) <= 1:
        # Deterministic weights
        weights = dict(forced_weights)
        remaining = 1.0 - sum(forced_weights.values())
        if free_series:
            weights[free_series[0]] = max(0, remaining)

        portfolio_returns = (df * pd.Series(weights)).sum(axis=1)
        window_results = [WindowResult(
            apply_start=df.index[0],
            apply_end=df.index[-1],
            weights=weights,
            est_start=df.index[0],
            est_end=df.index[-1],
        )]
        return window_results, portfolio_returns

    # ----- Ex ante models: single-period optimization (no windowing) -----
    is_ex_ante = model in ("ex_ante_mv", "black_litterman")
    if is_ex_ante:
        if progress_callback is not None:
            progress_callback(
                (
                    [{"value": 50, "color": "blue"}],
                    "Running ex ante optimization...",
                )
            )

        # Handle missing data for ex ante
        if missing_data == "fill_0":
            ex_ante_df = df.fillna(0)
        else:
            # Drop columns with any NaN, or fillna(0) if none survive
            valid_cols = [c for c in available_cols if not df[c].isna().any()]
            if not valid_cols:
                ex_ante_df = df.fillna(0)
            else:
                ex_ante_df = df[valid_cols].copy()
                # Ensure all available_cols are present (fill missing cols with 0)
                for c in available_cols:
                    if c not in ex_ante_df.columns:
                        ex_ante_df[c] = 0.0

        params = {
            "window_data": ex_ante_df,
            "model": model,
            "asset_names": available_cols,
            "lower_bounds": lower_bounds,
            "upper_bounds": upper_bounds,
            "forced_weights": forced_weights,
            "free_series": free_series,
            "exp_wt_cov": exp_wt_cov,
            "halflife": halflife,
        }
        params.update({
            "ex_ante_returns": ex_ante_returns,
            "ex_ante_cov": ex_ante_cov,
            "ex_ante_vol": ex_ante_vol,
            "ex_ante_corr": ex_ante_corr,
            "bl_views": bl_views,
            "bl_tau": bl_tau,
            "objective": objective,
            "linear_constraints": linear_constraints,
        })
            
        # Optimization
        weights = _optimize_single_window(**params)
        
        portfolio_returns = (df.fillna(0) * pd.Series(weights)).sum(axis=1)
        window_results = [WindowResult(
            apply_start=df.index[0],
            apply_end=df.index[-1],
            weights=weights,
            est_start=df.index[0],
            est_end=df.index[-1],
        )]

        if progress_callback is not None:
            progress_callback(
                (
                    [{"value": 100, "color": "blue"}],
                    "Optimization complete.",
                )
            )

        return window_results, portfolio_returns

    # ----- Standard models: windowed optimization -----
    # Compute windows
    windows = _compute_windows(df, window_type, window_size, opt_step, fill_in_sample,
                               opt_step_unit=opt_step_unit)
    total_windows = len(windows)

    window_results = []
    weights_df = pd.DataFrame(0.0, index=df.index, columns=available_cols)

    for i, (est_start, est_end, apply_start, apply_end) in enumerate(windows):
        # Report progress
        if progress_callback is not None:
            pct = int((i / total_windows) * 100)
            progress_callback(
                (
                    [{"value": pct, "color": "blue"}],
                    f"Optimizing window {i+1}/{total_windows}",
                )
            )

        # Extract estimation window data
        est_data = df.iloc[est_start:est_end + 1].copy()

        # Handle missing data
        if missing_data == "fill_0":
            est_data = est_data.fillna(0)
            window_assets = available_cols
        else:
            # fill_na: exclude series with any NaN in this window
            valid_cols = [c for c in available_cols if not est_data[c].isna().any()]
            if not valid_cols:
                # No complete series - fallback to fill_0 for this window
                est_data = est_data.fillna(0)
                window_assets = available_cols
            else:
                est_data = est_data[valid_cols]
                window_assets = valid_cols

        # Compute forced/free for this window
        window_forced = {s: v for s, v in forced_weights.items() if s in window_assets}
        window_free = [s for s in window_assets if s not in window_forced]

        # Per-window constraint adjustment for fill_na excluded series
        window_lower = {s: lower_bounds[s] for s in window_assets}
        window_upper = {s: upper_bounds[s] for s in window_assets}

        # Try per-window validation, fallback to equal weight if infeasible
        try:
            _validate_weight_constraints(window_assets, window_lower, window_upper, window_forced)
        except ValueError:
            # Fallback to equal weight for this window
            w_result = dict(window_forced)
            remaining = 1.0 - sum(window_forced.values())
            if window_free:
                eq = remaining / len(window_free)
                for s in window_free:
                    w_result[s] = eq
            # Add zero for excluded series
            for s in available_cols:
                if s not in w_result:
                    w_result[s] = 0.0
            w_result_final = w_result
        else:
            # Run optimization
            w_result = _optimize_single_window(
                est_data, model, window_assets, window_lower, window_upper,
                window_forced, window_free, exp_wt_cov, halflife,
            )
            # Add zero weight for excluded series
            w_result_final = {s: w_result.get(s, 0.0) for s in available_cols}

        # Record window result
        apply_start_ts = df.index[apply_start]
        apply_end_ts = df.index[apply_end]
        est_start_ts = df.index[est_start]
        est_end_ts = df.index[est_end]
        window_results.append(WindowResult(
            apply_start=apply_start_ts,
            apply_end=apply_end_ts,
            weights=w_result_final,
            est_start=est_start_ts,
            est_end=est_end_ts,
        ))

        # Apply weights to periods
        for s in available_cols:
            weights_df.iloc[apply_start:apply_end + 1, weights_df.columns.get_loc(s)] = w_result_final[s]

    # Final progress
    if progress_callback is not None:
        progress_callback(
            (
                [{"value": 100, "color": "blue"}],
                "Computing portfolio returns...",
            )
        )

    # Calculate portfolio returns
    portfolio_returns = (df.fillna(0) * weights_df).sum(axis=1)

    # Trim to periods where weights were actually applied (non-zero row sum)
    # so that pre-application periods don't appear as zero returns
    has_weights = weights_df.sum(axis=1) > 0
    portfolio_returns = portfolio_returns[has_weights]

    return window_results, portfolio_returns


def compute_risk_contributions(weights_dict, returns_df):
    """Compute percentage risk contribution of each asset.

    Args:
        weights_dict: Dict of {asset_name: weight}
        returns_df: DataFrame of returns for assets in weights_dict

    Returns:
        Dict of {asset_name: pct_contribution} where values sum to 1.0
    """
    cols = [c for c in returns_df.columns if c in weights_dict]
    w = np.array([weights_dict[c] for c in cols])
    cov = returns_df[cols].cov().values
    marginal = cov @ w
    total_var = w @ cov @ w
    if total_var == 0:
        return {c: 1.0 / len(cols) for c in cols}
    rc = w * marginal / total_var
    return dict(zip(cols, rc))


def compute_efficient_frontier(returns_df, ann_factor, rm="MV", n_points=50,
                               custom_mu=None, custom_cov=None, linear_constraints=None):
    """Compute the efficient frontier for a given risk measure.

    Args:
        returns_df: DataFrame of asset returns (clean, no NaN)
        ann_factor: Annualization factor (252 for daily, 52 for weekly, 12 for monthly)
        rm: Risk measure - "MV" for volatility, "CVaR" for Conditional Value-at-Risk
        n_points: Number of frontier points
        custom_mu: Optional DataFrame of expected returns (already annualized, shape 1 x n_assets)
        custom_cov: Optional DataFrame of covariance matrix (already annualized)
        linear_constraints: Optional list of dicts defining linear constraints

    Returns:
        Tuple of (frontier_points, asset_points) where:
        - frontier_points: list of {"return": float, "risk": float}
        - asset_points: list of {"name": str, "return": float, "risk": float}
    """
    port = rp.Portfolio(returns=returns_df)
    port.assets_stats(method_mu="hist", method_cov="hist")

    use_custom = custom_mu is not None and custom_cov is not None
    if use_custom:
        port.mu = custom_mu
        port.cov = custom_cov

    # Apply linear constraints if any
    if linear_constraints:
        A, B = _parse_linear_constraints(linear_constraints, returns_df.columns)
        if A is not None:
            port.ainequality = A
            port.binequality = B

    frontier = port.efficient_frontier(
        model="Classic", rm=rm, points=n_points, rf=0, hist=not use_custom
    )

    mu = port.mu.values.flatten()
    cov = port.cov.values
    returns_arr = returns_df.values
    alpha = 0.05

    def _compute_risk(w):
        """Compute risk for a weight vector using the selected risk measure."""
        if rm == "CVaR":
            port_returns = returns_arr @ w
            sorted_r = np.sort(port_returns)
            cutoff = max(1, int(np.ceil(len(sorted_r) * alpha)))
            return -sorted_r[:cutoff].mean() * np.sqrt(ann_factor)
        return np.sqrt(w @ cov @ w) * np.sqrt(ann_factor)

    results = []
    for col in frontier.columns:
        w = frontier[col].values
        ret = (w @ mu) * ann_factor
        risk = _compute_risk(w)
        results.append({"return": ret, "risk": risk})

    assets = []
    for i, name in enumerate(returns_df.columns):
        # Single-asset weight vector
        w_single = np.zeros(len(returns_df.columns))
        w_single[i] = 1.0
        assets.append({
            "name": name,
            "return": mu[i] * ann_factor,
            "risk": _compute_risk(w_single),
        })

    return results, assets
