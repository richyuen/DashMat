"""Statistics calculations for returns analysis."""

import numpy as np
import pandas as pd
from scipy import stats

import cache_config
from utils.returns import resample_returns_cached, get_working_returns, calculate_excess_returns, annualization_factor





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


def correlation(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Calculate correlation with benchmark."""
    if len(returns) < 2 or len(benchmark_returns) < 2:
        return np.nan
    return returns.corr(benchmark_returns)


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
    
    # Check if benchmark is valid (not "None" placeholder)
    has_benchmark = benchmark_returns.name != "None"

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
            "Annualized Volatility": annualized_volatility(ls_returns, periods_per_year),
            "Sharpe Ratio": sharpe_ratio(ls_returns, periods_per_year),
            "Sortino Ratio": sortino_ratio(ls_returns, periods_per_year),
            # For L/S, "Excess Return" is typically just the return itself, but if we follow strict "relative to bench" rule:
            # If bench is None, L/S return is absolute. 
            # If we enforce "no relative stats if no bench", then for L/S:
            # The "Excess Return" field in the table usually means "Active Return".
            # For L/S, the strategy return IS the active return (vs cash/0).
            # But the user asked: "If no benchmark is selected, then don't calculate any value for Excess Return..."
            # This implies they want to see blank if benchmark is None.
            # However, for L/S, the whole point is excess return.
            # But technically, if benchmark is None, `excess` is just `ret`.
            # Let's follow the instruction strictly for the fields labeled "Excess Return", "Tracking Error", "Information Ratio".
            # But wait, earlier code mapped "Annualized Excess Return" to `ls_ann_ret`.
            # If I make it NaN, I lose the main return metric for L/S in that column?
            # No, L/S has "Annualized Return" column too.
            # So I will set these to NaN if `has_benchmark` is False.
            "Annualized Excess Return": ls_ann_ret if has_benchmark else np.nan, 
            "Annualized Tracking Error": annualized_volatility(ls_returns, periods_per_year) if has_benchmark else np.nan,
            "Information Ratio": sharpe_ratio(ls_returns, periods_per_year) if has_benchmark else np.nan,
            "Correlation": np.nan, # L/S correlation to constituents? Or bench? 
                                   # If has_benchmark, we could calculate corr(ls_returns, bench).
                                   # But standard logic was returning NaN. Let's keep it NaN or implement it?
                                   # User said "Correlation seems to be correctly showing blank already".
                                   # So I'll leave Correlation as NaN for L/S or implement if needed. 
                                   # I'll stick to NaN for L/S as it's a derived series.
            "Hit Rate": hit_rate(ls_returns),
            "Hit Rate (vs Benchmark)": hit_rate(ls_returns) if has_benchmark else np.nan,
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
                result[f"{label} Annualized Volatility"] = annualized_volatility(trailing_ls, periods_per_year)
                result[f"{label} Sharpe Ratio"] = sharpe_ratio(trailing_ls, periods_per_year)
                result[f"{label} Sortino Ratio"] = sortino_ratio(trailing_ls, periods_per_year)
                result[f"{label} Excess Return"] = trailing_ls_ann_ret if has_benchmark else np.nan
                result[f"{label} Tracking Error"] = annualized_volatility(trailing_ls, periods_per_year) if has_benchmark else np.nan
                result[f"{label} Information Ratio"] = sharpe_ratio(trailing_ls, periods_per_year) if has_benchmark else np.nan
                result[f"{label} Correlation"] = np.nan
            else:
                result[f"{label} Annualized Return"] = np.nan
                result[f"{label} Annualized Volatility"] = np.nan
                result[f"{label} Sharpe Ratio"] = np.nan
                result[f"{label} Sortino Ratio"] = np.nan
                result[f"{label} Excess Return"] = np.nan
                result[f"{label} Tracking Error"] = np.nan
                result[f"{label} Information Ratio"] = np.nan
                result[f"{label} Correlation"] = np.nan
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
            "Annualized Volatility": annualized_volatility(ret, periods_per_year),
            "Sharpe Ratio": sharpe_ratio(ret, periods_per_year),
            "Sortino Ratio": sortino_ratio(ret, periods_per_year),
            "Annualized Excess Return": (ann_ret - ann_bench) if has_benchmark and not same_series else np.nan,
            "Annualized Tracking Error": tracking_error(ret, bench, periods_per_year) if has_benchmark and not same_series else np.nan,
            "Information Ratio": information_ratio(ret, bench, periods_per_year) if has_benchmark and not same_series else np.nan,
            "Correlation": correlation(ret, bench) if has_benchmark and not same_series else np.nan,
            "Hit Rate": hit_rate(ret),
            "Hit Rate (vs Benchmark)": hit_rate_vs_benchmark(ret, bench) if has_benchmark and not same_series else np.nan,
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
                result[f"{label} Annualized Volatility"] = annualized_volatility(trailing_ret, periods_per_year)
                result[f"{label} Sharpe Ratio"] = sharpe_ratio(trailing_ret, periods_per_year)
                result[f"{label} Sortino Ratio"] = sortino_ratio(trailing_ret, periods_per_year)
                result[f"{label} Excess Return"] = (trailing_ann_ret - trailing_ann_bench) if has_benchmark and not same_series else np.nan
                result[f"{label} Tracking Error"] = tracking_error(trailing_ret, trailing_bench, periods_per_year) if has_benchmark and not same_series else np.nan
                result[f"{label} Information Ratio"] = information_ratio(trailing_ret, trailing_bench, periods_per_year) if has_benchmark and not same_series else np.nan
                result[f"{label} Correlation"] = correlation(trailing_ret, trailing_bench) if has_benchmark and not same_series else np.nan
            else:
                result[f"{label} Annualized Return"] = np.nan
                result[f"{label} Annualized Volatility"] = np.nan
                result[f"{label} Sharpe Ratio"] = np.nan
                result[f"{label} Sortino Ratio"] = np.nan
                result[f"{label} Excess Return"] = np.nan
                result[f"{label} Tracking Error"] = np.nan
                result[f"{label} Information Ratio"] = np.nan
                result[f"{label} Correlation"] = np.nan

    return result


@cache_config.cache.memoize(timeout=0)
def calculate_statistics_cached(
    json_str: str,
    periodicity: str,
    selected_series: tuple,
    benchmark_assignments: str,
    long_short_assignments: str,
    date_range_str: str,
    vol_scaler: float = 0,
    vol_scaling_assignments: str = ""
) -> list:
    """Calculate statistics for all selected series with caching."""
    # Use get_working_returns to get aligned data + benchmarks
    df = get_working_returns(
        json_str, periodicity, selected_series,
        benchmark_assignments, long_short_assignments,
        date_range_str, vol_scaler, vol_scaling_assignments
    )

    if df.empty:
        return []

    benchmark_dict = eval(str(benchmark_assignments)) if benchmark_assignments else {}
    long_short_dict = eval(str(long_short_assignments)) if long_short_assignments else {}
    
    results = []
    # Ensure selected_series is iterable
    series_list = list(selected_series) if selected_series else []

    for series in series_list:
        if series not in df.columns:
            continue

        benchmark = benchmark_dict.get(series, "None")

        # Handle "None" benchmark as zero returns
        if benchmark == "None":
            # Create a zero returns series with the same index
            benchmark_returns = pd.Series(0.0, index=df.index, name="None")
        elif benchmark not in df.columns:
            # If benchmark is specified but not in data, fallback to series itself (excess = 0)
            benchmark = series
            benchmark_returns = df[benchmark]
        else:
            benchmark_returns = df[benchmark]

        is_long_short = long_short_dict.get(series, False)

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

@cache_config.cache.memoize(timeout=0)
def calculate_growth_of_dollar(raw_data, periodicity, selected_series, benchmark_assignments, long_short_assignments, date_range, vol_scaler: float = 0, vol_scaling_assignments: str = ""):
    """Calculate growth of $1 for Excel export with starting value of 1.0."""
    try:
        # Use get_working_returns
        working_df = get_working_returns(
            raw_data, periodicity or "daily", tuple(selected_series),
            str(benchmark_assignments), str(long_short_assignments), str(date_range),
            vol_scaler, str(vol_scaling_assignments)
        )

        if working_df.empty:
            return pd.DataFrame()
            
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

        series_growth_data = {}
        all_dates = set()

        for series in selected_series:
            if series not in working_df.columns:
                continue

            returns = working_df[series].dropna()
            
            if returns.empty:
                continue

            # Total Return growth (standard logic)
            # For L/S, working_df already contains the (L-S) difference stream.
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

    except Exception:
        return pd.DataFrame()


# Drawdown calculation

@cache_config.cache.memoize(timeout=0)
def calculate_drawdown(raw_data, periodicity, selected_series, returns_type, benchmark_assignments, long_short_assignments, date_range, vol_scaler: float = 0, vol_scaling_assignments: str = ""):
    """Calculate drawdown for Excel export."""
    try:
        # Use get_working_returns
        working_df = get_working_returns(
            raw_data, periodicity or "daily", tuple(selected_series),
            str(benchmark_assignments), str(long_short_assignments), str(date_range),
            vol_scaler, str(vol_scaling_assignments)
        )

        if working_df.empty:
            return pd.DataFrame()

        benchmark_dict = eval(str(benchmark_assignments)) if benchmark_assignments else {}
        long_short_dict = eval(str(long_short_assignments)) if long_short_assignments else {}

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

        series_drawdown_data = {}
        all_dates = set()

        for series in selected_series:
            if series not in working_df.columns:
                continue

            returns = working_df[series].dropna()

            if returns.empty:
                continue
                
            is_ls = long_short_dict.get(series, False)
            
            # Check for Excess Return mode (non-L/S)
            if returns_type == "excess" and not is_ls:
                benchmark = benchmark_dict.get(series, "None")
                if benchmark != "None" and benchmark in working_df.columns:
                    # Calculate Geometric Relative Drawdown (GrowthS / GrowthB)
                    
                    # Align benchmark (use working_df which includes date-filtered benchmark)
                    bench_series = working_df[benchmark].reindex(returns.index)
                    
                    # Compute growth indices
                    growth_s = (1 + returns).cumprod()
                    growth_b = (1 + bench_series).cumprod()
                    
                    # Relative Wealth Index
                    # Handle division by zero or NaN?
                    # returns should be aligned/filtered already.
                    rel_wealth = growth_s / growth_b
                    
                    # Prepend 1.0 (Base relative wealth)
                    growth_array = np.concatenate([[1.0], rel_wealth.values])
                    
                else:
                    # Fallback to total return drawdown if no benchmark
                    growth = (1 + returns).cumprod()
                    growth_array = np.concatenate([[1.0], growth.values])
            else:
                # Total Return or L/S (L/S is already an absolute stream)
                growth = (1 + returns).cumprod()
                # Prepend starting value of 1.0
                growth_array = np.concatenate([[1.0], growth.values])

            running_max_array = np.maximum.accumulate(growth_array)

            # Calculate drawdown (exclude the prepended 1.0)
            with np.errstate(divide='ignore', invalid='ignore'):
                drawdown_array = (growth_array[1:] / running_max_array[1:]) - 1
            
            drawdown = pd.Series(drawdown_array, index=returns.index)

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

    except Exception:
        return pd.DataFrame()


@cache_config.cache.memoize(timeout=0)
def generate_correlogram_cached(json_str: str, periodicity: str, selected_series: tuple,
                                returns_type: str, benchmark_assignments: str, long_short_assignments: str,
                                date_range_str: str, vol_scaler: float = 0, vol_scaling_assignments: str = ""):
    """Generate correlogram with caching."""
    display_df = calculate_excess_returns(
        json_str, periodicity, selected_series, benchmark_assignments, returns_type, long_short_assignments, date_range_str,
        vol_scaler, vol_scaling_assignments
    )

    if display_df.empty:
        return None

    available_series = list(display_df.columns)
    n = len(available_series)

    # Calculate correlation matrix
    corr_matrix = display_df.corr()

    return {
        'display_df': display_df,
        'corr_matrix': corr_matrix,
        'available_series': available_series,
        'n': n
    }
