"""Regression engine for DashMat — pure computation, no Dash."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from utils.optimization import _compute_windows, _compute_windows_monthly  # noqa: F401


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RegressionWindowResult:
    """Result of a single regression window."""

    est_start: pd.Timestamp
    est_end: pd.Timestamp
    apply_start: pd.Timestamp
    apply_end: pd.Timestamp
    coefficients: dict = field(default_factory=dict)   # {"intercept": float, "X1": float, ...}
    p_values: dict = field(default_factory=dict)        # {"intercept": float, "X1": float, ...}
    r_squared: float = np.nan
    adj_r_squared: float = np.nan
    anova_table: dict | None = None    # F-stat, df, SS, MS, p-value
    diagnostics: dict | None = None    # DW, JB, VIF, AIC, BIC
    arima_garch: dict | None = None    # Optional residual ARIMA/GARCH summary for this window
    residual_std: float = np.nan
    oos_metrics: dict | None = None    # OOS R², RMSE, MAE (for rolling/expanding only)
    n_obs: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_exponential_weights(n_obs: int, halflife: float) -> np.ndarray:
    """Build normalised exponential observation weights (recent = higher)."""
    if halflife <= 0:
        return np.ones(n_obs) / n_obs
    decay = np.exp(-np.log(2) / halflife * np.arange(n_obs - 1, -1, -1))
    return decay / decay.sum()


def _parse_optional_float(value: Any) -> float | None:
    """Parse user-provided numeric input; return None for blank/invalid/non-finite."""
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    return parsed


def _apply_lags(X: pd.DataFrame, lag_config: dict) -> pd.DataFrame:
    """Shift selected X columns by their specified lag, drop resulting NaN rows.

    Args:
        X: DataFrame of independent variables (DatetimeIndex)
        lag_config: Dict {col_name: lag_periods}. Columns not present are lag=0.

    Returns:
        DataFrame with lagged columns; rows with any NaN dropped.
    """
    result = X.copy()
    for col, lag in (lag_config or {}).items():
        if col in result.columns and lag and int(lag) != 0:
            result[col] = result[col].shift(int(lag))
    return result.dropna()


def _strip_intercept_col(X: pd.DataFrame) -> pd.DataFrame:
    """Remove any pre-existing intercept/const column."""
    drop = [c for c in X.columns if c.lower() in ("const", "intercept", "__intercept__")]
    return X.drop(columns=drop) if drop else X


def _add_intercept(X: pd.DataFrame) -> pd.DataFrame:
    """Prepend a constant column named 'intercept'."""
    intercept = pd.Series(1.0, index=X.index, name="intercept")
    return pd.concat([intercept, X], axis=1)


# ---------------------------------------------------------------------------
# Individual model implementations
# ---------------------------------------------------------------------------


def _run_ols(y: pd.Series, X: pd.DataFrame, force_zero_intercept: bool,
             robust_se: bool, sample_weights: np.ndarray | None) -> dict:
    """OLS via statsmodels — returns full ANOVA/diagnostic info."""
    import statsmodels.api as sm
    from statsmodels.stats.stattools import durbin_watson, jarque_bera
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    Xm = _strip_intercept_col(X)
    if not force_zero_intercept:
        Xm = sm.add_constant(Xm, has_constant="add")
        intercept_name = "const"
    else:
        intercept_name = None

    # Align
    combined = pd.concat([y, Xm], axis=1).dropna()
    if len(combined) < max(3, Xm.shape[1] + 1):
        return {}
    y_clean = combined.iloc[:, 0]
    X_clean = combined.iloc[:, 1:]

    if sample_weights is not None and len(sample_weights) == len(y_clean):
        w = sample_weights
    else:
        w = None

    try:
        if w is not None:
            model = sm.WLS(y_clean, X_clean, weights=w).fit()
        elif robust_se:
            model = sm.OLS(y_clean, X_clean).fit(cov_type="HAC",
                                                  cov_kwds={"maxlags": None})
        else:
            model = sm.OLS(y_clean, X_clean).fit()
    except Exception:
        return {}

    # Build coefficient map (rename const -> intercept)
    coefs = {}
    pvals = {}
    for name, coef, pval in zip(model.params.index, model.params.values,
                                 model.pvalues.values):
        display = "intercept" if name == "const" else name
        coefs[display] = float(coef)
        pvals[display] = float(pval)

    # ANOVA table
    anova = None
    if not force_zero_intercept and hasattr(model, "fvalue") and np.isfinite(model.fvalue or np.nan):
        anova = {
            "F_stat": float(model.fvalue),
            "F_pvalue": float(model.f_pvalue),
            "df_model": int(model.df_model),
            "df_resid": int(model.df_resid),
            "ss_model": float(model.ess),
            "ss_resid": float(model.ssr),
            "ss_total": float(model.centered_tss),
            "ms_model": float(model.ess / model.df_model) if model.df_model else np.nan,
            "ms_resid": float(model.ssr / model.df_resid) if model.df_resid else np.nan,
        }

    # Confidence intervals
    try:
        conf = model.conf_int()
        ci_low = {("intercept" if k == "const" else k): float(v)
                  for k, v in conf[0].items()}
        ci_high = {("intercept" if k == "const" else k): float(v)
                   for k, v in conf[1].items()}
    except Exception:
        ci_low = {}
        ci_high = {}

    # Standard errors
    try:
        bse = {("intercept" if k == "const" else k): float(v)
               for k, v in model.bse.items()}
    except Exception:
        bse = {}

    # t-stats
    try:
        tstat = {("intercept" if k == "const" else k): float(v)
                 for k, v in model.tvalues.items()}
    except Exception:
        tstat = {}

    # Diagnostics
    resid = model.resid.values
    dw = float(durbin_watson(resid))
    jb_stat, jb_p, _, _ = jarque_bera(resid)
    vif = {}
    if X_clean.shape[1] > 1:
        X_vif = X_clean.copy()
        if intercept_name and intercept_name in X_vif.columns:
            X_vif = X_vif.drop(columns=[intercept_name])
        try:
            for i, col in enumerate(X_vif.columns):
                vif[str(col)] = float(variance_inflation_factor(X_vif.values, i))
        except Exception:
            pass

    diagnostics = {
        "durbin_watson": dw,
        "jarque_bera_stat": float(jb_stat),
        "jarque_bera_pvalue": float(jb_p),
        "vif": vif,
        "aic": float(model.aic),
        "bic": float(model.bic),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "std_errors": bse,
        "t_stats": tstat,
    }

    return {
        "coefficients": coefs,
        "p_values": pvals,
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "residual_std": float(np.std(resid, ddof=1)) if len(resid) > 1 else np.nan,
        "anova_table": anova,
        "diagnostics": diagnostics,
        "n_obs": int(len(y_clean)),
        "fitted": pd.Series(model.fittedvalues.values, index=y_clean.index),
        "residuals": pd.Series(resid, index=y_clean.index),
    }


def _run_constrained_ols(y: pd.Series, X: pd.DataFrame, min_beta: float,
                         max_beta: float, min_beta_by_var: dict,
                         max_beta_by_var: dict, enable_constraint: dict,
                         force_zero_intercept: bool,
                         linear_constraints: list | None,
                         sample_weights: np.ndarray | None) -> dict:
    """Constrained OLS via scipy SLSQP with per-variable beta bounds."""
    Xm = _strip_intercept_col(X)
    if not force_zero_intercept:
        Xm = _add_intercept(Xm)

    combined = pd.concat([y, Xm], axis=1).dropna()
    if len(combined) < max(3, Xm.shape[1] + 1):
        return {}
    y_clean = combined.iloc[:, 0].values
    X_clean = combined.iloc[:, 1:].values
    col_names = list(combined.columns[1:])

    n = X_clean.shape[1]
    w = sample_weights if (sample_weights is not None and len(sample_weights) == len(y_clean)) else None

    def objective(beta):
        resid = y_clean - X_clean @ beta
        if w is not None:
            return float(np.sum(w * resid ** 2))
        return float(np.sum(resid ** 2))

    # Per-variable bounds
    bounds = []
    min_map = min_beta_by_var or {}
    max_map = max_beta_by_var or {}
    for i, col in enumerate(col_names):
        if col == "intercept" or not enable_constraint.get(col, False):
            bounds.append((-np.inf, np.inf))
        else:
            try:
                lower = float(min_map.get(col, min_beta))
            except (TypeError, ValueError):
                lower = float(min_beta)
            try:
                upper = float(max_map.get(col, max_beta))
            except (TypeError, ValueError):
                upper = float(max_beta)
            if lower > upper:
                lower, upper = upper, lower
            bounds.append((lower, upper))

    # Linear constraints from UI (A @ beta <= B)
    scipy_constraints = []
    if linear_constraints:
        for row in linear_constraints:
            coeffs = np.array(
                [(_parse_optional_float(row.get(c, 0)) or 0.0) for c in col_names],
                dtype=float,
            )
            mn = _parse_optional_float(row.get("Min"))
            mx = _parse_optional_float(row.get("Max"))
            if mn is not None:
                scipy_constraints.append({
                    "type": "ineq",
                    "fun": lambda beta, c=coeffs, b=mn: c @ beta - b,
                })
            if mx is not None:
                scipy_constraints.append({
                    "type": "ineq",
                    "fun": lambda beta, c=coeffs, b=mx: b - c @ beta,
                })

    x0 = np.linalg.lstsq(X_clean, y_clean, rcond=None)[0]
    try:
        res = minimize(objective, x0, method="SLSQP", bounds=bounds,
                       constraints=scipy_constraints,
                       options={"maxiter": 1000, "ftol": 1e-12})
    except Exception:
        return {}
    if not getattr(res, "success", False):
        return {}
    beta_hat = np.asarray(getattr(res, "x", None), dtype=float)
    if beta_hat.size != n or not np.all(np.isfinite(beta_hat)):
        return {}

    fitted = X_clean @ beta_hat
    resid = y_clean - fitted
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y_clean - np.mean(y_clean)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    n_obs = len(y_clean)
    k = n  # includes intercept if present
    adj_r2 = 1 - (1 - r2) * (n_obs - 1) / (n_obs - k) if (n_obs > k and ss_tot > 0) else np.nan

    coefs = {c: float(b) for c, b in zip(col_names, beta_hat)}
    # Return minimal result (no p-values for constrained OLS — use NaN)
    pvals = {c: np.nan for c in col_names}

    residual_std = float(np.std(resid, ddof=1)) if len(resid) > 1 else np.nan

    return {
        "coefficients": coefs,
        "p_values": pvals,
        "r_squared": r2,
        "adj_r_squared": adj_r2,
        "residual_std": residual_std,
        "anova_table": None,
        "diagnostics": {"note": "Constrained OLS — p-values not available"},
        "n_obs": n_obs,
        "fitted": pd.Series(fitted, index=combined.index),
        "residuals": pd.Series(resid, index=combined.index),
    }


def _run_style_analysis(y: pd.Series, X: pd.DataFrame,
                         linear_constraints: list | None,
                         sample_weights: np.ndarray | None) -> dict:
    """Sharpe Style Analysis: weights in [0,1] sum to 1, no intercept, minimise TE."""
    Xm = _strip_intercept_col(X)
    combined = pd.concat([y, Xm], axis=1).dropna()
    if len(combined) < 3:
        return {}
    y_clean = combined.iloc[:, 0].values
    X_clean = combined.iloc[:, 1:].values
    col_names = list(combined.columns[1:])
    n_assets = X_clean.shape[1]
    w = sample_weights if (sample_weights is not None and len(sample_weights) == len(y_clean)) else None

    def objective(beta):
        resid = y_clean - X_clean @ beta
        if w is not None:
            return float(np.sum(w * resid ** 2))
        return float(np.sum(resid ** 2))

    bounds = [(0.0, 1.0)] * n_assets
    constraints = [{"type": "eq", "fun": lambda b: np.sum(b) - 1.0}]

    # Extra linear constraints (applied to style weights)
    if linear_constraints:
        for row in linear_constraints:
            coeffs = np.array(
                [(_parse_optional_float(row.get(c, 0)) or 0.0) for c in col_names],
                dtype=float,
            )
            mn = _parse_optional_float(row.get("Min"))
            mx = _parse_optional_float(row.get("Max"))
            if mn is not None:
                constraints.append({
                    "type": "ineq",
                    "fun": lambda beta, c=coeffs, b=mn: c @ beta - b,
                })
            if mx is not None:
                constraints.append({
                    "type": "ineq",
                    "fun": lambda beta, c=coeffs, b=mx: b - c @ beta,
                })

    x0 = np.full(n_assets, 1.0 / n_assets)
    try:
        res = minimize(objective, x0, method="SLSQP", bounds=bounds,
                       constraints=constraints,
                       options={"maxiter": 2000, "ftol": 1e-13})
    except Exception:
        return {}
    if not getattr(res, "success", False):
        return {}
    beta_hat = np.asarray(getattr(res, "x", None), dtype=float)
    if beta_hat.size != n_assets or not np.all(np.isfinite(beta_hat)):
        return {}

    fitted = X_clean @ beta_hat
    resid = y_clean - fitted
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y_clean - np.mean(y_clean)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    n_obs = len(y_clean)

    coefs = {c: float(b) for c, b in zip(col_names, beta_hat)}
    pvals = {c: np.nan for c in col_names}
    residual_std = float(np.std(resid, ddof=1)) if len(resid) > 1 else np.nan

    return {
        "coefficients": coefs,
        "p_values": pvals,
        "r_squared": r2,
        "adj_r_squared": np.nan,
        "residual_std": residual_std,
        "anova_table": None,
        "diagnostics": {"note": "Style Analysis — weights sum to 1, no intercept"},
        "n_obs": n_obs,
        "fitted": pd.Series(fitted, index=combined.index),
        "residuals": pd.Series(resid, index=combined.index),
    }


def _run_sklearn_regression(y: pd.Series, X: pd.DataFrame, model_name: str,
                             alpha: float, l1_ratio: float,
                             force_zero_intercept: bool,
                             sample_weights: np.ndarray | None) -> dict:
    """Ridge / Lasso / ElasticNet via sklearn."""
    from sklearn.linear_model import Ridge, Lasso, ElasticNet

    Xm = _strip_intercept_col(X)
    combined = pd.concat([y, Xm], axis=1).dropna()
    if len(combined) < 3:
        return {}
    y_clean = combined.iloc[:, 0].values
    X_clean = combined.iloc[:, 1:].values
    col_names = list(combined.columns[1:])

    w = sample_weights if (sample_weights is not None and len(sample_weights) == len(y_clean)) else None
    fit_intercept = not force_zero_intercept

    if model_name == "ridge":
        mdl = Ridge(alpha=alpha, fit_intercept=fit_intercept)
    elif model_name == "lasso":
        mdl = Lasso(alpha=alpha, fit_intercept=fit_intercept, max_iter=10000)
    else:
        mdl = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept, max_iter=10000)

    try:
        mdl.fit(X_clean, y_clean, sample_weight=w)
    except Exception:
        return {}

    fitted = mdl.predict(X_clean)
    resid = y_clean - fitted

    coefs = {c: float(b) for c, b in zip(col_names, mdl.coef_)}
    if fit_intercept:
        coefs = {"intercept": float(mdl.intercept_), **coefs}
    pvals = {c: np.nan for c in coefs}

    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y_clean - np.mean(y_clean)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    n_obs = len(y_clean)
    k = len(coefs)
    adj_r2 = 1 - (1 - r2) * (n_obs - 1) / (n_obs - k) if (n_obs > k and ss_tot > 0) else np.nan

    return {
        "coefficients": coefs,
        "p_values": pvals,
        "r_squared": r2,
        "adj_r_squared": adj_r2,
        "residual_std": float(np.std(resid, ddof=1)) if len(resid) > 1 else np.nan,
        "anova_table": None,
        "diagnostics": {"note": f"{model_name.capitalize()} — p-values not applicable"},
        "n_obs": n_obs,
        "fitted": pd.Series(fitted, index=combined.index),
        "residuals": pd.Series(resid, index=combined.index),
    }


def _run_ridge(y: pd.Series, X: pd.DataFrame, alpha: float,
               force_zero_intercept: bool, sample_weights: np.ndarray | None) -> dict:
    return _run_sklearn_regression(y, X, "ridge", alpha, 0.5,
                                   force_zero_intercept, sample_weights)


def _run_lasso(y: pd.Series, X: pd.DataFrame, alpha: float,
               force_zero_intercept: bool, sample_weights: np.ndarray | None) -> dict:
    return _run_sklearn_regression(y, X, "lasso", alpha, 0.5,
                                   force_zero_intercept, sample_weights)


def _run_elastic_net(y: pd.Series, X: pd.DataFrame, alpha: float, l1_ratio: float,
                     force_zero_intercept: bool, sample_weights: np.ndarray | None) -> dict:
    return _run_sklearn_regression(y, X, "elastic_net", alpha, l1_ratio,
                                   force_zero_intercept, sample_weights)


# ---------------------------------------------------------------------------
# Post-regression: ARIMA/GARCH residual modeling
# ---------------------------------------------------------------------------


def _fit_arima_garch(residuals: pd.Series, arima_order: tuple,
                     garch_order: tuple) -> dict | None:
    """Fit ARIMA(p,d,q) + GARCH(p,q) to OLS residuals.

    Returns None if no model requested (all orders are 0).
    """
    ap, ad, aq = arima_order
    gp, gq = garch_order

    if ap == 0 and ad == 0 and aq == 0 and gp == 0 and gq == 0:
        return None

    clean = residuals.dropna()
    if len(clean) < 20:
        return None

    summary = {}

    if ap > 0 or ad > 0 or aq > 0:
        try:
            from statsmodels.tsa.arima.model import ARIMA
            arima_model = ARIMA(clean.values, order=(ap, ad, aq)).fit()
            summary["arima"] = {
                "order": (ap, ad, aq),
                "aic": float(arima_model.aic),
                "bic": float(arima_model.bic),
                "params": {str(k): float(v) for k, v in
                           zip(arima_model.param_names, arima_model.params)},
            }
            residuals_for_garch = pd.Series(arima_model.resid, index=clean.index)
        except Exception as exc:
            summary["arima"] = {"error": str(exc)}
            residuals_for_garch = clean
    else:
        residuals_for_garch = clean

    if gp > 0 or gq > 0:
        try:
            from arch import arch_model
            garch = arch_model(residuals_for_garch, vol="Garch", p=gp, q=gq,
                               dist="normal")
            garch_fit = garch.fit(disp="off")
            summary["garch"] = {
                "order": (gp, gq),
                "aic": float(garch_fit.aic),
                "bic": float(garch_fit.bic),
                "params": {str(k): float(v) for k, v in garch_fit.params.items()},
            }
        except Exception as exc:
            summary["garch"] = {"error": str(exc)}

    return summary if summary else None


# ---------------------------------------------------------------------------
# OOS metrics
# ---------------------------------------------------------------------------


def _compute_oos_metrics(y_actual: pd.Series, y_predicted: pd.Series) -> dict:
    """Compute OOS R², RMSE, MAE."""
    aligned = pd.concat([y_actual, y_predicted], axis=1).dropna()
    if len(aligned) < 2:
        return {"oos_r2": np.nan, "oos_rmse": np.nan, "oos_mae": np.nan}
    ya = aligned.iloc[:, 0].values
    yp = aligned.iloc[:, 1].values
    ss_res = np.sum((ya - yp) ** 2)
    ss_tot = np.sum((ya - np.mean(ya)) ** 2)
    oos_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(np.mean((ya - yp) ** 2)))
    mae = float(np.mean(np.abs(ya - yp)))
    return {"oos_r2": float(oos_r2), "oos_rmse": rmse, "oos_mae": mae}


# ---------------------------------------------------------------------------
# Rolling summary builder
# ---------------------------------------------------------------------------


def _build_rolling_summary(window_results: list[RegressionWindowResult]) -> pd.DataFrame:
    """Convert list of window results to a DataFrame indexed by apply_start."""
    if not window_results:
        return pd.DataFrame()

    rows = []
    for wr in window_results:
        row: dict[str, Any] = {
            "est_start": wr.est_start,
            "est_end": wr.est_end,
            "apply_start": wr.apply_start,
            "apply_end": wr.apply_end,
            "r_squared": wr.r_squared,
            "adj_r_squared": wr.adj_r_squared,
            "n_obs": wr.n_obs,
            "residual_std": wr.residual_std,
        }
        for k, v in wr.coefficients.items():
            row[f"coef_{k}"] = v
        for k, v in wr.p_values.items():
            row[f"pval_{k}"] = v
        if wr.oos_metrics:
            row.update(wr.oos_metrics)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.index = pd.DatetimeIndex([r.apply_start for r in window_results])
    return df


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def run_regression(
    y: pd.Series,
    X: pd.DataFrame,
    config: dict,
) -> tuple[list[RegressionWindowResult], pd.Series, pd.Series, dict | None]:
    """Run regression analysis.

    Args:
        y: Dependent variable (returns Series with DatetimeIndex)
        X: Independent variables (returns DataFrame with DatetimeIndex)
        config: Dict with regression parameters. Keys:
            model: 'ols', 'constrained_ols', 'style_analysis', 'ridge', 'lasso', 'elastic_net'
            force_zero_intercept: bool
            robust_se: bool (HAC standard errors for OLS)
            exp_wt: bool (exponential observation weights)
            halflife: int (for exp_wt)
            window_type: 'full', 'expanding', 'rolling'
            window_size: int
            opt_step: int
            opt_step_unit: 'periods' or 'months'
            fill_in_sample: bool
            missing_data: 'fill_na' or 'fill_0'
            alpha: float (Ridge/Lasso/ElasticNet regularisation)
            l1_ratio: float (ElasticNet mix)
            min_beta: float (Constrained OLS lower bound, when enabled)
            max_beta: float (Constrained OLS upper bound, when enabled)
            min_beta_by_var: dict {col: float} (per-variable lower bounds)
            max_beta_by_var: dict {col: float} (per-variable upper bounds)
            enable_constraint: dict {col: bool}  (per-variable constraint toggle)
            lag_config: dict {col: lag_periods}
            arima_order: (p, d, q)
            garch_order: (p, q)
            linear_constraints: list of dicts (UI constraint rows)

    Returns:
        (window_results, predicted_series, residuals_series, arima_garch_summary)
    """
    model_name = config.get("model", "ols")
    force_zero_intercept = bool(config.get("force_zero_intercept", False))
    robust_se = bool(config.get("robust_se", False))
    exp_wt = bool(config.get("exp_wt", False))
    halflife = float(config.get("halflife", 63) or 63)
    window_type = config.get("window_type", "full")
    window_size = int(config.get("window_size", 36) or 36)
    opt_step = int(config.get("opt_step", 1) or 1)
    opt_step_unit = config.get("opt_step_unit", "periods")
    fill_in_sample = bool(config.get("fill_in_sample", False))
    missing_data = config.get("missing_data", "fill_na")
    alpha = float(config.get("alpha", 1.0) or 1.0)
    l1_ratio = float(config.get("l1_ratio", 0.5) or 0.5)
    min_beta = float(config.get("min_beta", -999) or -999)
    max_beta = float(config.get("max_beta", 999) or 999)
    min_beta_by_var = config.get("min_beta_by_var", {}) or {}
    max_beta_by_var = config.get("max_beta_by_var", {}) or {}
    enable_constraint = config.get("enable_constraint", {}) or {}
    lag_config = config.get("lag_config", {}) or {}
    arima_order = tuple(config.get("arima_order", (0, 0, 0)) or (0, 0, 0))
    garch_order = tuple(config.get("garch_order", (0, 0)) or (0, 0))
    linear_constraints = config.get("linear_constraints", None)
    run_arima_garch = model_name in ("ols", "constrained_ols")

    supports_intercept_only = model_name in ("ols", "constrained_ols") and not force_zero_intercept
    X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(index=y.index)

    # Apply lags to X
    X_lagged = _apply_lags(X, lag_config)

    # Align y and X on common dates after lagging
    common_idx = y.index.intersection(X_lagged.index)
    y_aligned = y.loc[common_idx].dropna()
    X_aligned = X_lagged.loc[y_aligned.index]

    if y_aligned.empty:
        return [], pd.Series(dtype=float, name="predicted"), pd.Series(dtype=float, name="residuals"), None
    if X_aligned.empty and not supports_intercept_only:
        return [], pd.Series(dtype=float, name="predicted"), pd.Series(dtype=float, name="residuals"), None

    # Handle missing data globally
    if missing_data == "fill_0":
        X_aligned = X_aligned.fillna(0.0)
    else:
        # Drop rows where y or any X is NaN
        combined = pd.concat([y_aligned, X_aligned], axis=1).dropna()
        y_aligned = combined.iloc[:, 0]
        X_aligned = combined.iloc[:, 1:]

    if y_aligned.empty:
        return [], pd.Series(dtype=float, name="predicted"), pd.Series(dtype=float, name="residuals"), None
    if X_aligned.empty and not supports_intercept_only:
        return [], pd.Series(dtype=float, name="predicted"), pd.Series(dtype=float, name="residuals"), None

    if len(y_aligned) < 3:
        return [], pd.Series(dtype=float, name="predicted"), pd.Series(dtype=float, name="residuals"), None

    # Build combined df for window indexing
    combined_df = pd.concat([y_aligned, X_aligned], axis=1)

    # Compute windows
    if window_type == "full":
        windows = [(0, len(combined_df) - 1, 0, len(combined_df) - 1)]
    else:
        try:
            windows = _compute_windows(
                combined_df, window_type, window_size, opt_step,
                fill_in_sample, opt_step_unit=opt_step_unit,
            )
        except ValueError:
            return [], pd.Series(dtype=float, name="predicted"), pd.Series(dtype=float, name="residuals"), None

    window_results: list[RegressionWindowResult] = []
    predicted_all = pd.Series(np.nan, index=combined_df.index, dtype=float)
    residuals_all = pd.Series(np.nan, index=combined_df.index, dtype=float)

    for est_start_i, est_end_i, apply_start_i, apply_end_i in windows:
        est_data = combined_df.iloc[est_start_i:est_end_i + 1]
        y_est = est_data.iloc[:, 0]
        X_est = est_data.iloc[:, 1:]

        # Build exponential observation weights for estimation window
        sample_weights: np.ndarray | None = None
        if exp_wt:
            sample_weights = _build_exponential_weights(len(y_est), halflife)

        # Fit model
        result_dict = _fit_model(
            model_name=model_name,
            y=y_est,
            X=X_est,
            force_zero_intercept=force_zero_intercept,
            robust_se=robust_se,
            min_beta=min_beta,
            max_beta=max_beta,
            min_beta_by_var=min_beta_by_var,
            max_beta_by_var=max_beta_by_var,
            enable_constraint=enable_constraint,
            alpha=alpha,
            l1_ratio=l1_ratio,
            linear_constraints=linear_constraints,
            sample_weights=sample_weights,
        )

        if not result_dict:
            continue

        # Apply model to application window (for OOS prediction)
        apply_data = combined_df.iloc[apply_start_i:apply_end_i + 1]
        y_apply = apply_data.iloc[:, 0]
        X_apply = apply_data.iloc[:, 1:]

        predicted_apply = _predict(result_dict["coefficients"], X_apply, force_zero_intercept,
                                   model_name)
        residuals_apply = y_apply - predicted_apply

        # Per-window residual model diagnostics are most meaningful for rolling/expanding.
        # For full-window runs, keep legacy run-level summary behavior below.
        window_arima_garch = None
        if run_arima_garch and window_type != "full":
            est_residuals = result_dict.get("residuals")
            if isinstance(est_residuals, pd.Series):
                window_arima_garch = _fit_arima_garch(est_residuals, arima_order, garch_order)

        # OOS metrics: only for non-full windows where apply != estimation
        oos_metrics = None
        if window_type != "full" and apply_start_i > est_end_i:
            oos_metrics = _compute_oos_metrics(y_apply, predicted_apply)

        # Fill predicted / residuals for application window
        predicted_all.iloc[apply_start_i:apply_end_i + 1] = predicted_apply.values
        residuals_all.iloc[apply_start_i:apply_end_i + 1] = residuals_apply.values

        window_results.append(RegressionWindowResult(
            est_start=combined_df.index[est_start_i],
            est_end=combined_df.index[est_end_i],
            apply_start=combined_df.index[apply_start_i],
            apply_end=combined_df.index[apply_end_i],
            coefficients=result_dict.get("coefficients", {}),
            p_values=result_dict.get("p_values", {}),
            r_squared=result_dict.get("r_squared", np.nan),
            adj_r_squared=result_dict.get("adj_r_squared", np.nan),
            anova_table=result_dict.get("anova_table"),
            diagnostics=result_dict.get("diagnostics"),
            arima_garch=window_arima_garch,
            residual_std=result_dict.get("residual_std", np.nan),
            oos_metrics=oos_metrics,
            n_obs=result_dict.get("n_obs", 0),
        ))

    # Keep run-level ARIMA/GARCH summary for compatibility. Full-window uses its full
    # residual stream; rolling/expanding uses aggregate applied residuals as fallback.
    arima_garch_summary = None
    if window_results and run_arima_garch:
        resid_series = residuals_all.dropna()
        if len(resid_series) > 20:
            arima_garch_summary = _fit_arima_garch(resid_series, arima_order, garch_order)

    predicted_all.name = "predicted"
    residuals_all.name = "residuals"

    return window_results, predicted_all.dropna(), residuals_all.dropna(), arima_garch_summary


# ---------------------------------------------------------------------------
# Internal dispatch
# ---------------------------------------------------------------------------


def _fit_model(model_name, y, X, force_zero_intercept, robust_se,
               min_beta, max_beta, min_beta_by_var, max_beta_by_var,
               enable_constraint, alpha, l1_ratio,
               linear_constraints, sample_weights) -> dict:
    """Dispatch to individual model function."""
    if model_name == "ols":
        return _run_ols(y, X, force_zero_intercept, robust_se, sample_weights)
    elif model_name == "constrained_ols":
        return _run_constrained_ols(
            y, X, min_beta, max_beta, min_beta_by_var, max_beta_by_var,
            enable_constraint, force_zero_intercept, linear_constraints, sample_weights,
        )
    elif model_name == "style_analysis":
        return _run_style_analysis(y, X, linear_constraints, sample_weights)
    elif model_name == "ridge":
        return _run_ridge(y, X, alpha, force_zero_intercept, sample_weights)
    elif model_name == "lasso":
        return _run_lasso(y, X, alpha, force_zero_intercept, sample_weights)
    elif model_name == "elastic_net":
        return _run_elastic_net(y, X, alpha, l1_ratio, force_zero_intercept, sample_weights)
    else:
        return {}


def _predict(coefficients: dict, X: pd.DataFrame, force_zero_intercept: bool,
             model_name: str) -> pd.Series:
    """Predict y values using fitted coefficients and X features."""
    if not coefficients:
        return pd.Series(np.nan, index=X.index)

    Xm = _strip_intercept_col(X.copy())
    pred = pd.Series(0.0, index=X.index)

    for col, coef in coefficients.items():
        if col == "intercept":
            pred += coef
        elif col in Xm.columns:
            pred += coef * Xm[col].fillna(0.0)

    return pred
