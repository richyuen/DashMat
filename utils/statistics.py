"""Statistics calculations for returns analysis."""

import logging
from typing import Optional
import numpy as np
import pandas as pd
from scipy import stats

import cache_config
from utils.exponential_weighting import resolve_ewm_params
from utils.returns import resample_returns_cached, get_working_returns, calculate_excess_returns, annualization_factor, is_daily
from utils.serialization import (
    date_range_payload_for_cache,
    mapping_payload_for_cache,
    parse_mapping_payload,
)

SPX_DAILY_INCEPTION_DATE = pd.Timestamp("1988-01-04")

logger = logging.getLogger(__name__)




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


def _annualized_return_for_periodicity(
    returns: pd.Series, periodicity: str, periods_per_year: float
) -> float:
    if is_daily(periodicity) or periodicity.startswith("weekly_"):
        return annualized_return_calendar_days(returns, periodicity)
    return annualized_return(returns, periods_per_year)


def sharpe_ratio_with_risk_free(
    returns: pd.Series,
    periodicity: str,
    periods_per_year: float,
    risk_free_returns: Optional[pd.Series] = None,
) -> float:
    """Calculate Sharpe ratio using an annualized risk-free proxy when provided.

    If risk-free returns are provided, computation uses the date intersection
    between the series and risk-free history.
    """
    working_returns = returns.dropna()
    if working_returns.empty:
        return np.nan

    ann_rf = 0.0
    if risk_free_returns is not None and len(risk_free_returns) > 0:
        aligned = pd.concat([working_returns, risk_free_returns], axis=1).dropna()
        if aligned.empty:
            return np.nan
        working_returns = aligned.iloc[:, 0]
        rf_series = aligned.iloc[:, 1]
        ann_rf = _annualized_return_for_periodicity(rf_series, periodicity, periods_per_year)

    ann_ret = _annualized_return_for_periodicity(working_returns, periodicity, periods_per_year)
    ann_vol = annualized_volatility(working_returns, periods_per_year)
    if ann_vol == 0 or np.isnan(ann_vol):
        return np.nan
    return (ann_ret - ann_rf) / ann_vol


def sortino_ratio_with_risk_free(
    returns: pd.Series,
    periodicity: str,
    periods_per_year: float,
    risk_free_returns: Optional[pd.Series] = None,
) -> float:
    """Calculate Sortino ratio using annualized risk-free return when provided."""
    working_returns = returns.dropna()
    if len(working_returns) < 2:
        return np.nan

    ann_rf = 0.0
    if risk_free_returns is not None and len(risk_free_returns) > 0:
        aligned = pd.concat([working_returns, risk_free_returns], axis=1).dropna()
        if aligned.empty:
            return np.nan
        working_returns = aligned.iloc[:, 0]
        rf_series = aligned.iloc[:, 1]
        ann_rf = _annualized_return_for_periodicity(rf_series, periodicity, periods_per_year)

    return sortino_ratio(working_returns, periods_per_year, rf=ann_rf)


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


def beta_to_benchmark(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series],
) -> float:
    """Calculate beta relative to a benchmark return stream."""
    if benchmark_returns is None:
        return np.nan

    aligned = pd.concat([returns.dropna(), benchmark_returns.dropna()], axis=1).dropna()
    if len(aligned) < 2:
        return np.nan

    series_returns = aligned.iloc[:, 0]
    series_benchmark = aligned.iloc[:, 1]
    benchmark_var = series_benchmark.var(ddof=1)
    if benchmark_var == 0 or np.isnan(benchmark_var):
        return np.nan

    covariance = series_returns.cov(series_benchmark)
    if np.isnan(covariance):
        return np.nan
    return covariance / benchmark_var


def beta_to_spx_if_eligible(
    returns: pd.Series,
    spx_returns: Optional[pd.Series],
    beta_allowed: bool,
) -> float:
    """Calculate beta to S&P 500 only when the working stream is eligible."""
    if not beta_allowed:
        return np.nan
    return beta_to_benchmark(returns, spx_returns)


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


def _get_trailing_window(
    returns: pd.Series,
    periodicity: str,
    years: int,
    periods_per_year: float,
) -> Optional[pd.Series]:
    """Get trailing window for statistics.

    Daily/daily_trading windows are calendar-day based; other periodicities
    use period counts.
    """
    series = returns.dropna()
    if series.empty:
        return None

    if is_daily(periodicity):
        end_date = series.index.max()
        start_date = end_date - pd.DateOffset(years=years) + pd.Timedelta(days=1)
        # Require enough history to fully cover trailing window.
        if series.index.min() > start_date:
            return None
        window = series.loc[(series.index >= start_date) & (series.index <= end_date)]
        return window if not window.empty else None

    n_periods = int(years * periods_per_year)
    if len(series) < n_periods:
        return None
    return series.iloc[-n_periods:]


def calculate_statistics(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    periodicity: str,
    series_name: str,
    is_long_short: bool = False,
    risk_free_returns: Optional[pd.Series] = None,
    spx_returns: Optional[pd.Series] = None,
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
    beta_allowed = (
        len(ret) > 0
        and pd.Timestamp(ret.index.min()) >= SPX_DAILY_INCEPTION_DATE
    )

    # For long-short mode, calculate returns based on the period-by-period difference
    if is_long_short:
        # Long-short returns are the excess returns (difference)
        ls_returns = ret

        # Use calendar-based annualization for daily/weekly data
        use_calendar_days = is_daily(periodicity) or periodicity.startswith("weekly_")

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
            "Sharpe Ratio": sharpe_ratio_with_risk_free(
                ls_returns, periodicity, periods_per_year, risk_free_returns
            ),
            "Sortino Ratio": sortino_ratio_with_risk_free(
                ls_returns, periodicity, periods_per_year, risk_free_returns
            ),
            "Beta to S&P 500": beta_to_spx_if_eligible(
                ls_returns, spx_returns, beta_allowed
            ),
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
            trailing_ls = _get_trailing_window(
                ls_returns, periodicity, years, periods_per_year
            )
            if trailing_ls is not None:

                if use_calendar_days:
                    trailing_ls_ann_ret = annualized_return_calendar_days(trailing_ls, periodicity)
                else:
                    trailing_ls_ann_ret = annualized_return(trailing_ls, periods_per_year)

                result[f"{label} Annualized Return"] = trailing_ls_ann_ret
                result[f"{label} Annualized Volatility"] = annualized_volatility(trailing_ls, periods_per_year)
                result[f"{label} Sharpe Ratio"] = sharpe_ratio_with_risk_free(
                    trailing_ls, periodicity, periods_per_year, risk_free_returns
                )
                result[f"{label} Sortino Ratio"] = sortino_ratio_with_risk_free(
                    trailing_ls, periodicity, periods_per_year, risk_free_returns
                )
                result[f"{label} Beta to S&P 500"] = beta_to_spx_if_eligible(
                    trailing_ls, spx_returns, beta_allowed
                )
                result[f"{label} Excess Return"] = trailing_ls_ann_ret if has_benchmark else np.nan
                result[f"{label} Tracking Error"] = annualized_volatility(trailing_ls, periods_per_year) if has_benchmark else np.nan
                result[f"{label} Information Ratio"] = sharpe_ratio(trailing_ls, periods_per_year) if has_benchmark else np.nan
                result[f"{label} Correlation"] = np.nan
            else:
                result[f"{label} Annualized Return"] = np.nan
                result[f"{label} Annualized Volatility"] = np.nan
                result[f"{label} Sharpe Ratio"] = np.nan
                result[f"{label} Sortino Ratio"] = np.nan
                result[f"{label} Beta to S&P 500"] = np.nan
                result[f"{label} Excess Return"] = np.nan
                result[f"{label} Tracking Error"] = np.nan
                result[f"{label} Information Ratio"] = np.nan
                result[f"{label} Correlation"] = np.nan
    else:
        # Normal mode (non-long-short)
        # Use calendar-based annualization for daily/weekly data
        use_calendar_days = is_daily(periodicity) or periodicity.startswith("weekly_")

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
            "Sharpe Ratio": sharpe_ratio_with_risk_free(
                ret, periodicity, periods_per_year, risk_free_returns
            ),
            "Sortino Ratio": sortino_ratio_with_risk_free(
                ret, periodicity, periods_per_year, risk_free_returns
            ),
            "Beta to S&P 500": beta_to_spx_if_eligible(
                ret, spx_returns, beta_allowed
            ),
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
            trailing_ret = _get_trailing_window(
                ret, periodicity, years, periods_per_year
            )
            if trailing_ret is not None:
                trailing_bench = bench.reindex(trailing_ret.index)
                aligned_trailing = pd.concat([trailing_ret, trailing_bench], axis=1).dropna()
                if aligned_trailing.empty:
                    result[f"{label} Annualized Return"] = np.nan
                    result[f"{label} Annualized Volatility"] = np.nan
                    result[f"{label} Sharpe Ratio"] = np.nan
                    result[f"{label} Sortino Ratio"] = np.nan
                    result[f"{label} Beta to S&P 500"] = np.nan
                    result[f"{label} Excess Return"] = np.nan
                    result[f"{label} Tracking Error"] = np.nan
                    result[f"{label} Information Ratio"] = np.nan
                    result[f"{label} Correlation"] = np.nan
                    continue
                trailing_ret = aligned_trailing.iloc[:, 0]
                trailing_bench = aligned_trailing.iloc[:, 1]

                if use_calendar_days:
                    trailing_ann_ret = annualized_return_calendar_days(trailing_ret, periodicity)
                    trailing_ann_bench = annualized_return_calendar_days(trailing_bench, periodicity)
                else:
                    trailing_ann_ret = annualized_return(trailing_ret, periods_per_year)
                    trailing_ann_bench = annualized_return(trailing_bench, periods_per_year)

                result[f"{label} Annualized Return"] = trailing_ann_ret
                result[f"{label} Annualized Volatility"] = annualized_volatility(trailing_ret, periods_per_year)
                result[f"{label} Sharpe Ratio"] = sharpe_ratio_with_risk_free(
                    trailing_ret, periodicity, periods_per_year, risk_free_returns
                )
                result[f"{label} Sortino Ratio"] = sortino_ratio_with_risk_free(
                    trailing_ret, periodicity, periods_per_year, risk_free_returns
                )
                result[f"{label} Beta to S&P 500"] = beta_to_spx_if_eligible(
                    trailing_ret, spx_returns, beta_allowed
                )
                result[f"{label} Excess Return"] = (trailing_ann_ret - trailing_ann_bench) if has_benchmark and not same_series else np.nan
                result[f"{label} Tracking Error"] = tracking_error(trailing_ret, trailing_bench, periods_per_year) if has_benchmark and not same_series else np.nan
                result[f"{label} Information Ratio"] = information_ratio(trailing_ret, trailing_bench, periods_per_year) if has_benchmark and not same_series else np.nan
                result[f"{label} Correlation"] = correlation(trailing_ret, trailing_bench) if has_benchmark and not same_series else np.nan
            else:
                result[f"{label} Annualized Return"] = np.nan
                result[f"{label} Annualized Volatility"] = np.nan
                result[f"{label} Sharpe Ratio"] = np.nan
                result[f"{label} Sortino Ratio"] = np.nan
                result[f"{label} Beta to S&P 500"] = np.nan
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
    vol_scaling_assignments: str = "",
    risk_free_returns_json: str = "",
    spx_returns_json: str = "",
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

    benchmark_dict = parse_mapping_payload(benchmark_assignments)
    long_short_dict = parse_mapping_payload(long_short_assignments)
    risk_free_series: Optional[pd.Series] = None
    if risk_free_returns_json:
        try:
            rf_df = resample_returns_cached(risk_free_returns_json, periodicity)
            if not rf_df.empty:
                rf_col = rf_df.columns[0]
                risk_free_series = rf_df[rf_col].dropna()
        except Exception:
            logger.exception("Risk-free benchmark payload could not be resampled.")
            risk_free_series = None

    spx_series: Optional[pd.Series] = None
    if spx_returns_json:
        try:
            spx_df = resample_returns_cached(spx_returns_json, periodicity)
            if not spx_df.empty:
                spx_col = spx_df.columns[0]
                spx_series = spx_df[spx_col].dropna()
        except Exception:
            logger.exception("S&P benchmark payload could not be resampled.")
            spx_series = None
    
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
            risk_free_series,
            spx_series,
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
            mapping_payload_for_cache(benchmark_assignments),
            mapping_payload_for_cache(long_short_assignments),
            date_range_payload_for_cache(date_range),
            vol_scaler,
            mapping_payload_for_cache(vol_scaling_assignments),
        )

        if working_df.empty:
            return pd.DataFrame()
            
        # Determine the period offset based on periodicity
        periodicity_str = periodicity or "daily"
        if is_daily(periodicity_str):
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
        logger.exception("Growth-of-dollar calculation failed.")
        return pd.DataFrame()


# Drawdown calculation

@cache_config.cache.memoize(timeout=0)
def calculate_drawdown(raw_data, periodicity, selected_series, returns_type, benchmark_assignments, long_short_assignments, date_range, vol_scaler: float = 0, vol_scaling_assignments: str = ""):
    """Calculate drawdown for Excel export."""
    try:
        # Use get_working_returns
        working_df = get_working_returns(
            raw_data, periodicity or "daily", tuple(selected_series),
            mapping_payload_for_cache(benchmark_assignments),
            mapping_payload_for_cache(long_short_assignments),
            date_range_payload_for_cache(date_range),
            vol_scaler,
            mapping_payload_for_cache(vol_scaling_assignments),
        )

        if working_df.empty:
            return pd.DataFrame()

        benchmark_dict = parse_mapping_payload(benchmark_assignments)
        long_short_dict = parse_mapping_payload(long_short_assignments)

        # Determine the period offset based on periodicity
        periodicity_str = periodicity or "daily"
        if is_daily(periodicity_str):
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
        logger.exception("Drawdown calculation failed.")
        return pd.DataFrame()


@cache_config.cache.memoize(timeout=0)
def generate_correlogram_cached(json_str: str, periodicity: str, selected_series: tuple,
                                returns_type: str, benchmark_assignments: str, long_short_assignments: str,
                                date_range_str: str, vol_scaler: float = 0, vol_scaling_assignments: str = "",
                                exp_weighted: bool = False, decay_value: float = 63.0):
    """Generate correlogram with caching."""
    display_df = calculate_excess_returns(
        json_str, periodicity, selected_series, benchmark_assignments, returns_type, long_short_assignments, date_range_str,
        vol_scaler, vol_scaling_assignments
    )

    if display_df.empty:
        return None

    available_series = list(display_df.columns)
    n = len(available_series)

    # Calculate correlation/covariance matrices
    corr_matrix = display_df.corr()
    cov_matrix = display_df.cov()

    if exp_weighted:
        ewm_cov = display_df.ewm(**resolve_ewm_params(decay_value)).cov().iloc[-n:]
        if isinstance(ewm_cov.index, pd.MultiIndex):
            ewm_cov.index = ewm_cov.index.get_level_values(-1)
        cov_matrix = ewm_cov.reindex(index=available_series, columns=available_series)

        cov_values = cov_matrix.to_numpy(dtype=float, copy=False)
        std = np.sqrt(np.clip(np.diag(cov_values), a_min=0.0, a_max=None))
        denom = np.outer(std, std)
        corr_values = np.divide(
            cov_values,
            denom,
            out=np.full_like(cov_values, np.nan, dtype=float),
            where=denom > 0,
        )
        np.fill_diagonal(corr_values, 1.0)
        corr_matrix = pd.DataFrame(corr_values, index=available_series, columns=available_series)

    return {
        'display_df': display_df,
        'corr_matrix': corr_matrix,
        'cov_matrix': cov_matrix,
        'available_series': available_series,
        'n': n
    }
