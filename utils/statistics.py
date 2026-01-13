"""Statistics calculations for returns analysis."""

import numpy as np
import pandas as pd
from scipy import stats


def annualization_factor(periodicity: str) -> float:
    """Get annualization factor based on periodicity."""
    factors = {
        "daily": 252,
        "weekly_monday": 52,
        "weekly_tuesday": 52,
        "weekly_wednesday": 52,
        "weekly_thursday": 52,
        "weekly_friday": 52,
        "monthly": 12,
    }
    return factors.get(periodicity, 252)


def cumulative_return(returns: pd.Series) -> float:
    """Calculate cumulative return from a series of returns."""
    return (1 + returns).prod() - 1


def annualized_return(returns: pd.Series, periods_per_year: float) -> float:
    """Calculate annualized return."""
    n_periods = len(returns)
    if n_periods == 0:
        return np.nan
    cum_ret = cumulative_return(returns)
    years = n_periods / periods_per_year
    if years == 0:
        return np.nan
    return (1 + cum_ret) ** (1 / years) - 1


def annualized_volatility(returns: pd.Series, periods_per_year: float) -> float:
    """Calculate annualized volatility (standard deviation)."""
    if len(returns) < 2:
        return np.nan
    return returns.std() * np.sqrt(periods_per_year)


def sharpe_ratio(returns: pd.Series, periods_per_year: float, rf: float = 0.0) -> float:
    """Calculate Sharpe ratio (assuming rf=0 by default)."""
    ann_ret = annualized_return(returns, periods_per_year)
    ann_vol = annualized_volatility(returns, periods_per_year)
    if ann_vol == 0 or np.isnan(ann_vol):
        return np.nan
    return (ann_ret - rf) / ann_vol


def tracking_error(returns: pd.Series, benchmark_returns: pd.Series, periods_per_year: float) -> float:
    """Calculate annualized tracking error."""
    excess = returns - benchmark_returns
    if len(excess) < 2:
        return np.nan
    return excess.std() * np.sqrt(periods_per_year)


def information_ratio(returns: pd.Series, benchmark_returns: pd.Series, periods_per_year: float) -> float:
    """Calculate information ratio."""
    excess = returns - benchmark_returns
    ann_excess = annualized_return(excess, periods_per_year)
    te = tracking_error(returns, benchmark_returns, periods_per_year)
    if te == 0 or np.isnan(te):
        return np.nan
    return ann_excess / te


def hit_rate(returns: pd.Series) -> float:
    """Calculate hit rate (% of positive returns)."""
    if len(returns) == 0:
        return np.nan
    return (returns > 0).sum() / len(returns)


def hit_rate_vs_benchmark(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Calculate hit rate vs benchmark (% of periods outperforming)."""
    excess = returns - benchmark_returns
    if len(excess) == 0:
        return np.nan
    return (excess > 0).sum() / len(excess)


def maximum_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown."""
    if len(returns) == 0:
        return np.nan
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def calculate_statistics(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    periodicity: str,
    series_name: str,
    is_long_short: bool = False,
) -> dict:
    """Calculate all statistics for a single series (optimized for performance)."""
    periods_per_year = annualization_factor(periodicity)

    # Align returns and benchmark more efficiently
    if returns.name == benchmark_returns.name:
        # Same series - avoid unnecessary concatenation
        ret = returns.dropna()
        bench = ret.copy()
    else:
        aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
        if len(aligned) == 0:
            return {"Series": series_name}
        ret = aligned.iloc[:, 0]
        bench = aligned.iloc[:, 1]

    # Calculate excess once for reuse
    excess = ret - bench

    # For long-short mode, calculate returns based on the period-by-period difference
    if is_long_short:
        # Long-short returns are the excess returns (difference)
        ls_returns = excess

        result = {
            "Series": series_name,
            "Start Date": ls_returns.index.min().strftime("%Y-%m-%d") if len(ls_returns) > 0 else "",
            "End Date": ls_returns.index.max().strftime("%Y-%m-%d") if len(ls_returns) > 0 else "",
            "Number of Periods": len(ls_returns),
            "Cumulative Return": cumulative_return(ls_returns),
            "Annualized Return": annualized_return(ls_returns, periods_per_year),
            "Annualized Excess Return": annualized_return(ls_returns, periods_per_year),  # Same as Annualized Return
            "Annualized Volatility": annualized_volatility(ls_returns, periods_per_year),
            "Annualized Tracking Error": annualized_volatility(ls_returns, periods_per_year),  # Same as volatility for long-short
            "Sharpe Ratio": sharpe_ratio(ls_returns, periods_per_year),
            "Information Ratio": sharpe_ratio(ls_returns, periods_per_year),  # Same as Sharpe for long-short
            "Hit Rate": hit_rate(ls_returns),
            "Hit Rate (vs Benchmark)": hit_rate(ls_returns),  # Same as Hit Rate for long-short
            "Best Period Return": ls_returns.max() if len(ls_returns) > 0 else np.nan,
            "Worst Period Return": ls_returns.min() if len(ls_returns) > 0 else np.nan,
            "Maximum Drawdown": maximum_drawdown(ls_returns),
            "Skewness": stats.skew(ls_returns) if len(ls_returns) > 2 else np.nan,
            "Kurtosis": stats.kurtosis(ls_returns) if len(ls_returns) > 3 else np.nan,
        }

        # Calculate trailing period statistics for long-short
        for years, label in [(1, "1Y"), (3, "3Y"), (5, "5Y")]:
            n_periods = int(years * periods_per_year)
            if len(ls_returns) >= n_periods:
                trailing_ls = ls_returns.iloc[-n_periods:]

                result[f"{label} Annualized Return"] = annualized_return(trailing_ls, periods_per_year)
                result[f"{label} Sharpe Ratio"] = sharpe_ratio(trailing_ls, periods_per_year)
                result[f"{label} Excess Return"] = annualized_return(trailing_ls, periods_per_year)  # Same as annualized return
                result[f"{label} Information Ratio"] = sharpe_ratio(trailing_ls, periods_per_year)  # Same as Sharpe
            else:
                result[f"{label} Annualized Return"] = np.nan
                result[f"{label} Sharpe Ratio"] = np.nan
                result[f"{label} Excess Return"] = np.nan
                result[f"{label} Information Ratio"] = np.nan
    else:
        # Normal mode (non-long-short)
        result = {
            "Series": series_name,
            "Start Date": ret.index.min().strftime("%Y-%m-%d") if len(ret) > 0 else "",
            "End Date": ret.index.max().strftime("%Y-%m-%d") if len(ret) > 0 else "",
            "Number of Periods": len(ret),
            "Cumulative Return": cumulative_return(ret),
            "Annualized Return": annualized_return(ret, periods_per_year),
            "Annualized Excess Return": annualized_return(excess, periods_per_year),
            "Annualized Volatility": annualized_volatility(ret, periods_per_year),
            "Annualized Tracking Error": tracking_error(ret, bench, periods_per_year),
            "Sharpe Ratio": sharpe_ratio(ret, periods_per_year),
            "Information Ratio": information_ratio(ret, bench, periods_per_year),
            "Hit Rate": hit_rate(ret),
            "Hit Rate (vs Benchmark)": hit_rate_vs_benchmark(ret, bench),
            "Best Period Return": ret.max() if len(ret) > 0 else np.nan,
            "Worst Period Return": ret.min() if len(ret) > 0 else np.nan,
            "Maximum Drawdown": maximum_drawdown(ret),
            "Skewness": stats.skew(ret) if len(ret) > 2 else np.nan,
            "Kurtosis": stats.kurtosis(ret) if len(ret) > 3 else np.nan,
        }

        # Calculate trailing period statistics
        for years, label in [(1, "1Y"), (3, "3Y"), (5, "5Y")]:
            n_periods = int(years * periods_per_year)
            if len(ret) >= n_periods:
                trailing_ret = ret.iloc[-n_periods:]
                trailing_bench = bench.iloc[-n_periods:]
                trailing_excess = trailing_ret - trailing_bench

                result[f"{label} Annualized Return"] = annualized_return(trailing_ret, periods_per_year)
                result[f"{label} Sharpe Ratio"] = sharpe_ratio(trailing_ret, periods_per_year)
                result[f"{label} Excess Return"] = annualized_return(trailing_excess, periods_per_year)
                result[f"{label} Information Ratio"] = information_ratio(trailing_ret, trailing_bench, periods_per_year)
            else:
                result[f"{label} Annualized Return"] = np.nan
                result[f"{label} Sharpe Ratio"] = np.nan
                result[f"{label} Excess Return"] = np.nan
                result[f"{label} Information Ratio"] = np.nan

    return result


def calculate_all_statistics(
    df: pd.DataFrame,
    selected_series: list[str],
    benchmark_assignments: dict,
    periodicity: str,
    long_short_assignments: dict = None,
) -> list[dict]:
    """Calculate statistics for all selected series."""
    results = []
    long_short_assignments = long_short_assignments or {}

    for series in selected_series:
        if series not in df.columns:
            continue

        benchmark = benchmark_assignments.get(series, selected_series[0]) if benchmark_assignments else selected_series[0]
        if benchmark not in df.columns:
            benchmark = series

        is_long_short = long_short_assignments.get(series, False)

        stats_dict = calculate_statistics(
            df[series],
            df[benchmark],
            periodicity,
            series,
            is_long_short,
        )
        results.append(stats_dict)

    return results
