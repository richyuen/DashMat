"""Statistics calculations for returns analysis."""

import numpy as np
import pandas as pd
from scipy import stats

import cache_config
from utils.returns import resample_returns_cached


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
    """Calculate annualized return. Returns cumulative return if period <= 1 year."""
    n_periods = len(returns)
    if n_periods == 0:
        return np.nan
    cum_ret = cumulative_return(returns)
    years = n_periods / periods_per_year
    if years == 0:
        return np.nan
    # If period is 1 year or less, return cumulative return (don't annualize)
    if years <= 1.0:
        return cum_ret
    return (1 + cum_ret) ** (1 / years) - 1


def annualized_return_calendar_days(returns: pd.Series, periodicity: str) -> float:
    """Calculate annualized return based on calendar days for daily/weekly data.

    For weekly data, the starting day is the first period's date minus 6 days.
    Returns cumulative return if period <= 1 year.
    """
    if len(returns) == 0:
        return np.nan

    cum_ret = cumulative_return(returns)

    # Get start and end dates
    end_date = returns.index.max()
    start_date = returns.index.min()

    # For weekly data, adjust start date to be 6 days earlier
    if periodicity.startswith("weekly_"):
        start_date = start_date - pd.Timedelta(days=6)

    # Calculate calendar days (inclusive)
    calendar_days = (end_date - start_date).days + 1

    if calendar_days <= 0:
        return np.nan

    # Calculate years
    years = calendar_days / 365.25

    # If period is 1 year or less, return cumulative return (don't annualize)
    if years <= 1.0:
        return cum_ret

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


def sortino_ratio(returns: pd.Series, periods_per_year: float, rf: float = 0.0, target_return: float = 0.0) -> float:
    """Calculate Sortino ratio."""
    ann_ret = annualized_return(returns, periods_per_year)

    # Calculate downside deviation
    # We care about returns below target_return
    downside_diff = returns - target_return
    downside_diff[downside_diff > 0] = 0

    # Calculate semi-variance (using N-1 for consistency with sample std dev if len > 1)
    if len(returns) < 2:
        return np.nan

    downside_sq = downside_diff ** 2
    # Use N-1 to align with pandas std() behavior for volatility
    semi_variance = downside_sq.sum() / (len(returns) - 1)
    downside_dev = np.sqrt(semi_variance) * np.sqrt(periods_per_year)

    if downside_dev == 0 or np.isnan(downside_dev):
        return np.nan

    return (ann_ret - rf) / downside_dev


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
    same_series = returns.name == benchmark_returns.name
    if same_series:
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

        # Use calendar-based annualization for daily/weekly data
        use_calendar_days = periodicity == "daily" or periodicity.startswith("weekly_")

        if use_calendar_days:
            ls_ann_ret = annualized_return_calendar_days(ls_returns, periodicity)
        else:
            ls_ann_ret = annualized_return(ls_returns, periods_per_year)

        result = {
            "Series": series_name,
            "Start Date": ls_returns.index.min().strftime("%Y-%m-%d") if len(ls_returns) > 0 else "",
            "End Date": ls_returns.index.max().strftime("%Y-%m-%d") if len(ls_returns) > 0 else "",
            "Number of Periods": len(ls_returns),
            "Cumulative Return": cumulative_return(ls_returns),
            "Annualized Return": ls_ann_ret,
            "Annualized Excess Return": ls_ann_ret,  # Same as Annualized Return
            "Annualized Volatility": annualized_volatility(ls_returns, periods_per_year),
            "Annualized Tracking Error": annualized_volatility(ls_returns, periods_per_year),  # Same as volatility for long-short
            "Sharpe Ratio": sharpe_ratio(ls_returns, periods_per_year),
            "Sortino Ratio": sortino_ratio(ls_returns, periods_per_year),
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

                if use_calendar_days:
                    trailing_ls_ann_ret = annualized_return_calendar_days(trailing_ls, periodicity)
                else:
                    trailing_ls_ann_ret = annualized_return(trailing_ls, periods_per_year)

                result[f"{label} Annualized Return"] = trailing_ls_ann_ret
                result[f"{label} Sharpe Ratio"] = sharpe_ratio(trailing_ls, periods_per_year)
                result[f"{label} Sortino Ratio"] = sortino_ratio(trailing_ls, periods_per_year)
                result[f"{label} Excess Return"] = trailing_ls_ann_ret  # Same as annualized return
                result[f"{label} Information Ratio"] = sharpe_ratio(trailing_ls, periods_per_year)  # Same as Sharpe
            else:
                result[f"{label} Annualized Return"] = np.nan
                result[f"{label} Sharpe Ratio"] = np.nan
                result[f"{label} Sortino Ratio"] = np.nan
                result[f"{label} Excess Return"] = np.nan
                result[f"{label} Information Ratio"] = np.nan
    else:
        # Normal mode (non-long-short)
        # Use calendar-based annualization for daily/weekly data
        use_calendar_days = periodicity == "daily" or periodicity.startswith("weekly_")

        if use_calendar_days:
            ann_ret = annualized_return_calendar_days(ret, periodicity)
            ann_bench = annualized_return_calendar_days(bench, periodicity)
        else:
            ann_ret = annualized_return(ret, periods_per_year)
            ann_bench = annualized_return(bench, periods_per_year)

        result = {
            "Series": series_name,
            "Start Date": ret.index.min().strftime("%Y-%m-%d") if len(ret) > 0 else "",
            "End Date": ret.index.max().strftime("%Y-%m-%d") if len(ret) > 0 else "",
            "Number of Periods": len(ret),
            "Cumulative Return": cumulative_return(ret),
            "Annualized Return": ann_ret,
            "Annualized Excess Return": np.nan if same_series else (ann_ret - ann_bench),
            "Annualized Volatility": annualized_volatility(ret, periods_per_year),
            "Annualized Tracking Error": np.nan if same_series else tracking_error(ret, bench, periods_per_year),
            "Sharpe Ratio": sharpe_ratio(ret, periods_per_year),
            "Sortino Ratio": sortino_ratio(ret, periods_per_year),
            "Information Ratio": np.nan if same_series else information_ratio(ret, bench, periods_per_year),
            "Hit Rate": hit_rate(ret),
            "Hit Rate (vs Benchmark)": np.nan if same_series else hit_rate_vs_benchmark(ret, bench),
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

                if use_calendar_days:
                    trailing_ann_ret = annualized_return_calendar_days(trailing_ret, periodicity)
                    trailing_ann_bench = annualized_return_calendar_days(trailing_bench, periodicity)
                else:
                    trailing_ann_ret = annualized_return(trailing_ret, periods_per_year)
                    trailing_ann_bench = annualized_return(trailing_bench, periods_per_year)

                result[f"{label} Annualized Return"] = trailing_ann_ret
                result[f"{label} Sharpe Ratio"] = sharpe_ratio(trailing_ret, periods_per_year)
                result[f"{label} Sortino Ratio"] = sortino_ratio(trailing_ret, periods_per_year)
                result[f"{label} Excess Return"] = np.nan if same_series else (trailing_ann_ret - trailing_ann_bench)
                result[f"{label} Information Ratio"] = np.nan if same_series else information_ratio(trailing_ret, trailing_bench, periods_per_year)
            else:
                result[f"{label} Annualized Return"] = np.nan
                result[f"{label} Sharpe Ratio"] = np.nan
                result[f"{label} Sortino Ratio"] = np.nan
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

        benchmark = benchmark_assignments.get(series, "None") if benchmark_assignments else "None"

        # Handle "None" benchmark as zero returns
        if benchmark == "None":
            # Create a zero returns series with the same index
            benchmark_returns = pd.Series(0.0, index=df.index, name="None")
        elif benchmark not in df.columns:
            benchmark = series
            benchmark_returns = df[benchmark]
        else:
            benchmark_returns = df[benchmark]

        is_long_short = long_short_assignments.get(series, False)

        stats_dict = calculate_statistics(
            df[series],
            benchmark_returns,
            periodicity,
            series,
            is_long_short,
        )
        results.append(stats_dict)

    return results


# Growth of $1 calculation

def _compute_growth_of_dollar(df, periodicity, available_series, benchmark_dict, long_short_dict):
    """Helper function to compute growth of $1 for all series.

    Returns:
        DataFrame: Growth of $1 with Date as index
    """
    # Determine the period offset based on periodicity
    periodicity_str = periodicity or "daily"
    if periodicity_str == "daily":
        period_offset = pd.DateOffset(days=1)
    elif periodicity_str == "monthly":
        period_offset = pd.tseries.offsets.MonthEnd(1)
    elif periodicity_str.startswith("weekly"):
        period_offset = pd.DateOffset(weeks=1)
    else:
        period_offset = pd.DateOffset(days=1)

    # Calculate growth for each series and collect data
    series_growth_data = {}
    all_dates = set()

    for series in available_series:
        is_long_short = long_short_dict.get(series, False)
        benchmark = benchmark_dict.get(series, available_series[0])

        if is_long_short:
            # For long-short, use the difference
            if benchmark == "None":
                returns = df[series]
            elif benchmark == series:
                # Skip series where benchmark is itself for long-short
                continue
            elif benchmark in df.columns:
                returns = df[series] - df[benchmark]
            else:
                returns = df[series]
        else:
            # For non-long-short, use total returns
            returns = df[series]

        # Drop NaNs before calculation
        returns = returns.dropna()

        if returns.empty:
            continue

        # Calculate cumulative growth
        growth = (1 + returns).cumprod()

        # Prepend 1.0 at start_date
        first_date = growth.index[0]
        start_date = first_date - period_offset

        start_val = pd.Series([1.0], index=[start_date])
        growth_with_start = pd.concat([start_val, growth])

        series_growth_data[series] = growth_with_start
        all_dates.update(growth_with_start.index)

    if not series_growth_data:
        return pd.DataFrame()

    # Build DataFrame with all dates
    sorted_dates = sorted(list(all_dates))
    growth_df = pd.DataFrame(index=sorted_dates)
    growth_df.index.name = "Date"

    for series, growth in series_growth_data.items():
        growth_df[series] = growth

    return growth_df


@cache_config.cache.memoize(timeout=0)
def calculate_growth_of_dollar(raw_data, periodicity, selected_series, benchmark_assignments, long_short_assignments, date_range):
    """Calculate growth of $1 for Excel export with starting value of 1.0."""
    try:
        df = resample_returns_cached(raw_data, periodicity or "daily")

        # Apply date range filter if provided
        date_range_dict = eval(str(date_range)) if date_range and str(date_range) != "None" else None
        if date_range_dict:
            start_date = pd.to_datetime(date_range_dict["start"])
            end_date = pd.to_datetime(date_range_dict["end"])
            df = df[(df.index >= start_date) & (df.index <= end_date)]

        benchmark_dict = eval(str(benchmark_assignments)) if benchmark_assignments else {}
        long_short_dict = eval(str(long_short_assignments)) if long_short_assignments else {}

        # Filter to selected series only
        available_series = [s for s in selected_series if s in df.columns]
        if not available_series:
            return pd.DataFrame()

        # Compute growth using shared helper
        growth_df = _compute_growth_of_dollar(
            df, periodicity, available_series, benchmark_dict, long_short_dict
        )

        return growth_df

    except Exception:
        return pd.DataFrame()


# Drawdown calculation

def _compute_drawdown(df, periodicity, available_series, returns_type, benchmark_dict, long_short_dict):
    """Helper function to compute drawdown for all series.

    Returns:
        DataFrame: Drawdown with Date as index
    """
    # Determine the period offset based on periodicity
    periodicity_str = periodicity or "daily"
    if periodicity_str == "daily":
        period_offset = pd.DateOffset(days=1)
    elif periodicity_str == "monthly":
        period_offset = pd.tseries.offsets.MonthEnd(1)
    elif periodicity_str.startswith("weekly"):
        period_offset = pd.DateOffset(weeks=1)
    else:
        period_offset = pd.DateOffset(days=1)

    # Calculate drawdown for each series and collect data
    series_drawdown_data = {}
    all_dates = set()

    for series in available_series:
        is_long_short = long_short_dict.get(series, False)
        benchmark = benchmark_dict.get(series, available_series[0])

        if is_long_short:
            # For long-short, use the difference
            if benchmark == "None":
                returns = df[series]
            elif benchmark == series:
                # Skip series where benchmark is itself for long-short
                continue
            elif benchmark in df.columns:
                returns = df[series] - df[benchmark]
            else:
                returns = df[series]

            # Drop NaNs before calculation to handle different start dates
            returns = returns.dropna()

            # Calculate cumulative growth
            growth = (1 + returns).cumprod()
        elif returns_type == "excess" and benchmark != "None" and benchmark != series and benchmark in df.columns:
            # For excess returns, calculate drawdown of series relative to benchmark
            # Compound each separately, then calculate relative performance

            # Align data first by dropping NaNs where either series or benchmark is missing
            aligned_df = df[[series, benchmark]].dropna()

            series_growth = (1 + aligned_df[series]).cumprod()
            benchmark_growth = (1 + aligned_df[benchmark]).cumprod()

            # Relative growth (series vs benchmark)
            growth = series_growth / benchmark_growth
        else:
            # For total returns, use series returns directly
            returns = df[series].dropna()
            growth = (1 + returns).cumprod()

        if growth.empty:
            continue

        # Prepend starting value of 1.0 to properly calculate drawdown from initial capital
        # This ensures that a negative first period return counts as a drawdown
        growth_array = np.concatenate([[1.0], growth.values])
        running_max_array = np.maximum.accumulate(growth_array)

        # Calculate drawdown (exclude the prepended 1.0)
        drawdown_array = (growth_array[1:] / running_max_array[1:]) - 1
        drawdown = pd.Series(drawdown_array, index=growth.index)

        # Prepend 0.0 drawdown at one period before this series' first date
        first_date = drawdown.index[0]
        start_date = first_date - period_offset
        start_val = pd.Series([0.0], index=[start_date])
        drawdown_with_start = pd.concat([start_val, drawdown])

        series_drawdown_data[series] = drawdown_with_start
        all_dates.update(drawdown_with_start.index)

    if not series_drawdown_data:
        return pd.DataFrame()

    # Build DataFrame with all dates
    sorted_dates = sorted(list(all_dates))
    drawdown_df = pd.DataFrame(index=sorted_dates)
    drawdown_df.index.name = "Date"

    for series, drawdown in series_drawdown_data.items():
        drawdown_df[series] = drawdown

    return drawdown_df


@cache_config.cache.memoize(timeout=0)
def calculate_drawdown(raw_data, periodicity, selected_series, returns_type, benchmark_assignments, long_short_assignments, date_range):
    """Calculate drawdown for Excel export."""
    try:
        df = resample_returns_cached(raw_data, periodicity or "daily")

        # Apply date range filter if provided
        date_range_dict = eval(str(date_range)) if date_range and str(date_range) != "None" else None
        if date_range_dict:
            start_date = pd.to_datetime(date_range_dict["start"])
            end_date = pd.to_datetime(date_range_dict["end"])
            df = df[(df.index >= start_date) & (df.index <= end_date)]

        benchmark_dict = eval(str(benchmark_assignments)) if benchmark_assignments else {}
        long_short_dict = eval(str(long_short_assignments)) if long_short_assignments else {}

        # Filter to selected series only
        available_series = [s for s in selected_series if s in df.columns]
        if not available_series:
            return pd.DataFrame()

        # Compute drawdown using shared helper
        drawdown_df = _compute_drawdown(
            df, periodicity, available_series, returns_type, benchmark_dict, long_short_dict
        )

        return drawdown_df

    except Exception:
        return pd.DataFrame()
