"""Returns calculation utilities for compounding and resampling."""

import hashlib
import json
from io import StringIO
import numpy as np
import pandas as pd

import cache_config


# Mapping of periodicity options to pandas resample codes
RESAMPLE_CODES = {
    "daily": None,  # No resampling needed
    "weekly_monday": "W-MON",
    "weekly_tuesday": "W-TUE",
    "weekly_wednesday": "W-WED",
    "weekly_thursday": "W-THU",
    "weekly_friday": "W-FRI",
    "monthly": "ME",
}

PERIODICITY_LABELS = {
    "daily": "Daily",
    "weekly_monday": "Weekly (Monday)",
    "weekly_tuesday": "Weekly (Tuesday)",
    "weekly_wednesday": "Weekly (Wednesday)",
    "weekly_thursday": "Weekly (Thursday)",
    "weekly_friday": "Weekly (Friday)",
    "monthly": "Monthly",
}


def compound_returns(returns: pd.Series) -> float:
    """Compound a series of returns into a single return.

    Formula: (1 + r1) * (1 + r2) * ... * (1 + rn) - 1

    Optimized with numpy for better performance.
    """
    if len(returns) == 0:
        return np.nan
    # Use numpy for faster computation
    growth_factors = 1 + returns.values
    return np.prod(growth_factors) - 1


def resample_returns(df: pd.DataFrame, periodicity: str) -> pd.DataFrame:
    """Resample returns to a different periodicity.

    Args:
        df: DataFrame with DatetimeIndex and returns columns
        periodicity: Target periodicity (see RESAMPLE_CODES keys)

    Returns:
        Resampled DataFrame with compounded returns

    Optimized for performance with vectorized operations where possible.
    """
    if periodicity == "daily":
        return df

    resample_code = RESAMPLE_CODES.get(periodicity)
    if resample_code is None:
        raise ValueError(f"Unknown periodicity: {periodicity}")

    # Resample and compound returns for each period
    # Using agg for better performance than apply
    resampled = df.resample(resample_code).agg(compound_returns)

    # Drop rows where all values are NaN (periods with no data)
    resampled = resampled.dropna(how="all")

    # For weekly periodicity, exclude partial weeks at start and end
    if periodicity.startswith("weekly_") and len(resampled) > 0:
        # Count the number of observations in each period
        counts = df.resample(resample_code).count()

        # Align counts with resampled data (in case some periods were dropped)
        counts = counts.reindex(resampled.index, fill_value=0)

        # Get the typical week size (mode of the counts, excluding zeros)
        # Use the first column to determine count
        first_col = counts.columns[0]
        non_zero_counts = counts[first_col][counts[first_col] > 0]
        if len(non_zero_counts) > 0:
            typical_week_size = non_zero_counts.mode().iloc[0] if len(non_zero_counts.mode()) > 0 else 5

            # Identify periods to keep (not partial weeks)
            periods_to_keep = []
            for i, idx in enumerate(resampled.index):
                count = counts.loc[idx, first_col]
                # Keep if it's a full week, or if it's a middle period (not first or last)
                if count >= typical_week_size:
                    periods_to_keep.append(True)
                elif i == 0 or i == len(resampled) - 1:
                    # Drop partial first or last period
                    periods_to_keep.append(False)
                else:
                    # Keep middle periods even if partial (e.g., holidays)
                    periods_to_keep.append(True)

            resampled = resampled[periods_to_keep]

    return resampled


def get_available_periodicities(original_periodicity: str) -> list[dict]:
    """Get list of available periodicity options based on original data frequency.

    Args:
        original_periodicity: 'daily' or 'monthly'

    Returns:
        List of dicts with 'value' and 'label' keys for dropdown
    """
    if original_periodicity == "monthly":
        # Monthly data cannot be upsampled
        return [{"value": "monthly", "label": "Monthly"}]

    # Daily data can be converted to any periodicity
    return [
        {"value": key, "label": label}
        for key, label in PERIODICITY_LABELS.items()
    ]


def merge_returns(existing_df: pd.DataFrame | None, new_df: pd.DataFrame) -> pd.DataFrame:
    """Merge new returns data with existing data.

    Performs an outer join on the date index, appending new columns to the right.

    Args:
        existing_df: Existing returns DataFrame (or None if first upload)
        new_df: New returns DataFrame to append

    Returns:
        Merged DataFrame
    """
    if existing_df is None or existing_df.empty:
        return new_df

    # Handle duplicate column names by adding suffix
    overlap = set(existing_df.columns) & set(new_df.columns)
    if overlap:
        new_df = new_df.rename(
            columns={col: f"{col}_new" for col in overlap}
        )

    # Outer join on index
    merged = existing_df.join(new_df, how="outer")
    merged = merged.sort_index()

    return merged


# JSON/DataFrame conversion with caching

@cache_config.cache.memoize(timeout=0)
def json_to_df_cached(json_str: str) -> pd.DataFrame:
    """Convert JSON string back to DataFrame with caching.

    This is the primary performance bottleneck - caching this operation
    prevents repeated deserialization of the same data.
    """
    df = pd.read_json(StringIO(json_str), orient="split")
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df


def json_to_df(json_str: str) -> pd.DataFrame:
    """Convert JSON string back to DataFrame (cached wrapper)."""
    return json_to_df_cached(json_str)


@cache_config.cache.memoize(timeout=0)
def resample_returns_cached(json_str: str, periodicity: str) -> pd.DataFrame:
    """Resample returns with caching to avoid repeated computation."""
    df = json_to_df(json_str)
    if periodicity == "daily":
        return df
    return resample_returns(df, periodicity)


@cache_config.cache.memoize(timeout=0)
def get_working_returns(json_str: str, periodicity: str, selected_series: tuple, 
                        benchmark_assignments: str, long_short_assignments: str, 
                        date_range_str: str) -> pd.DataFrame:
    """Calculate working returns with date filtering, benchmark intersection, and L/S logic.
    
    Args:
        json_str: Raw data JSON string
        periodicity: Selected periodicity
        selected_series: Tuple of selected series names
        benchmark_assignments: String representation of benchmark dict
        long_short_assignments: String representation of L/S dict
        date_range_str: String representation of date range dict
        
    Returns:
        DataFrame with calculated returns for selected series AND unselected benchmarks.
        For L/S series, returns (Series - Benchmark).
        For standard series, returns Series (aligned to benchmark intersection).
        For unselected benchmarks, returns Series (date filtered only).
        Does NOT calculate excess returns for standard series.
    """
    # 1. Get base data
    df = resample_returns_cached(json_str, periodicity)
    
    # 2. Parse configurations
    bench_dict = eval(str(benchmark_assignments)) if benchmark_assignments else {}
    ls_dict = eval(str(long_short_assignments)) if long_short_assignments else {}
    date_range = eval(str(date_range_str)) if date_range_str and str(date_range_str) != "None" else None
    
    # 3. Global Date Range Filter
    if date_range:
        start_date = pd.to_datetime(date_range["start"])
        end_date = pd.to_datetime(date_range["end"])
        df = df[(df.index >= start_date) & (df.index <= end_date)]

    # 4. Process Series
    result_df = pd.DataFrame(index=df.index)
    
    # Ensure selected_series is iterable
    series_list = list(selected_series) if selected_series else []
    
    # Identify unselected benchmarks
    unselected_benchmarks = set()
    for series in series_list:
        benchmark = bench_dict.get(series, "None")
        if benchmark != "None" and benchmark in df.columns and benchmark not in series_list:
             unselected_benchmarks.add(benchmark)
    
    for series in series_list:
        if series not in df.columns:
            continue
            
        s_data = df[series]
        
        benchmark = bench_dict.get(series, "None")
        is_ls = ls_dict.get(series, False)
        
        # Determine effective benchmark data
        bench_data = None
        if benchmark != "None" and benchmark in df.columns and benchmark != series:
            bench_data = df[benchmark]
        
        # Benchmark Intersection Logic
        if bench_data is not None:
            # "Only dates in common... included"
            # Intersect indices
            common_idx = s_data.dropna().index.intersection(bench_data.dropna().index)
            
            # Align data to common index
            s_aligned = s_data.reindex(common_idx)
            bench_aligned = bench_data.reindex(common_idx)
            
            # Reindex back to result index (introduces NaNs for non-common dates)
            s_data = s_aligned.reindex(df.index)
            bench_data = bench_aligned.reindex(df.index)
            
        # Calculation Logic
        if is_ls and bench_data is not None:
            # L/S: Series - Benchmark
            final_series = s_data - bench_data
        else:
            # Standard: Just the series (aligned)
            final_series = s_data
            
        result_df[series] = final_series

    # Add unselected benchmarks (date filtered only)
    for bench in unselected_benchmarks:
        result_df[bench] = df[bench]
        
    return result_df.dropna(how='all')


@cache_config.cache.memoize(timeout=0)
def calculate_excess_returns(json_str: str, periodicity: str, selected_series: tuple,
                             benchmark_assignments: str, returns_type: str, long_short_assignments: str,
                             date_range_str: str) -> pd.DataFrame:
    """Calculate excess returns with caching."""
    # Get base working returns (Series aligned to Bench, or L/S diff)
    display_df = get_working_returns(
        json_str, periodicity, selected_series,
        benchmark_assignments, long_short_assignments,
        date_range_str
    )
    
    if display_df.empty:
        return display_df

    # If returns_type is "excess", we need to calculate Series - Benchmark
    # for non-L/S series. L/S series are already diffs.
    if returns_type == "excess":
        # We use display_df which now includes benchmarks
        benchmark_dict = eval(str(benchmark_assignments)) if benchmark_assignments else {}
        ls_dict = eval(str(long_short_assignments)) if long_short_assignments else {}
        
        # Iterate over SELECTED series only
        for series in selected_series:
            if series not in display_df.columns:
                continue

            is_ls = ls_dict.get(series, False)
            if not is_ls:
                benchmark = benchmark_dict.get(series, "None")
                if benchmark != "None" and benchmark in display_df.columns:
                    # Align benchmark to display_df (which is already date filtered)
                    # Use the benchmark column directly from display_df
                    bench_series = display_df[benchmark]
                    
                    # Calculate arithmetic excess for the grid
                    display_df[series] = display_df[series] - bench_series

    # Filter to show only selected series (remove benchmark columns if they were added but not selected)
    # Ensure we only return columns that are in selected_series
    final_cols = [col for col in selected_series if col in display_df.columns]
    return display_df[final_cols]


# Rolling returns calculation

@cache_config.cache.memoize(timeout=0)
def calculate_rolling_returns(raw_data, periodicity, selected_series, returns_type, benchmark_assignments, long_short_assignments, date_range, rolling_window="1y", rolling_return_type="annualized", rolling_metric="total_return"):
    """Calculate rolling returns for Excel export - matches the Rolling grid logic."""
    try:
        from utils.statistics import annualization_factor, sharpe_ratio, sortino_ratio

        # Get working returns (forces alignment and filtering)
        # working_df contains Series (aligned) OR (Series - Bench) if L/S
        # NOW also contains unselected benchmarks
        working_df = get_working_returns(
            raw_data, periodicity or "daily", tuple(selected_series),
            str(benchmark_assignments), str(long_short_assignments), str(date_range)
        )
        
        if working_df.empty:
            return pd.DataFrame()

        # Parse assignments
        benchmark_dict = eval(str(benchmark_assignments)) if benchmark_assignments else {}
        long_short_dict = eval(str(long_short_assignments)) if long_short_assignments else {}

        # Calculate periods per year and window size
        periods_per_year = annualization_factor(periodicity or "daily")

        # For daily data, use calendar days; for other periodicities, use number of periods
        use_calendar_days = (periodicity or "daily") == "daily"

        if use_calendar_days:
            # Map rolling window to calendar days
            window_map_days = {
                "3m": "91D",
                "6m": "183D",
                "1y": "365D",
                "3y": "1096D",
                "5y": "1826D",
                "10y": "3652D",
            }
            window_spec = window_map_days.get(rolling_window, "365D")

            # Extract the number of days from the window spec
            window_days_map = {
                "3m": 91,
                "6m": 183,
                "1y": 365,
                "3y": 1096,
                "5y": 1826,
                "10y": 3652,
            }
            min_calendar_days = window_days_map.get(rolling_window, 365)
            window_size = None  # Not used for time-based rolling
        else:
            # Map rolling window to number of periods
            window_map = {
                "3m": int(periods_per_year / 4),
                "6m": int(periods_per_year / 2),
                "1y": int(periods_per_year),
                "3y": int(periods_per_year * 3),
                "5y": int(periods_per_year * 5),
                "10y": int(periods_per_year * 10),
            }
            window_size = window_map.get(rolling_window, int(periods_per_year))
            window_size = max(1, window_size)
            window_spec = window_size

        # Map rolling window to number of years for annualization (only used for returns)
        window_years_map = {
            "3m": 0.25,
            "6m": 0.5,
            "1y": 1.0,
            "3y": 3.0,
            "5y": 5.0,
            "10y": 10.0,
        }
        window_years = window_years_map.get(rolling_window, 1.0)

        # Helper functions for rolling apply
        def calc_rolling_return(window):
            if len(window) == 0:
                return np.nan
            # For count-based windows, check minimum size
            if not use_calendar_days and len(window) < window_size:
                return np.nan
            cum_ret = (1 + window).prod() - 1
            if rolling_return_type == "annualized":
                if window_years <= 1.0:
                    return cum_ret
                return (1 + cum_ret) ** (1 / window_years) - 1
            else:
                return cum_ret

        # Wrapper for statistics functions to handle window requirements
        def apply_rolling_stat(series, func):
            if use_calendar_days:
                return series.rolling(window=window_spec).apply(func, raw=False)
            else:
                return series.rolling(window=window_spec, min_periods=window_size).apply(func, raw=False)

        # Calculate rolling metrics for each series
        rolling_df = pd.DataFrame(index=working_df.index)

        # Iterate over SELECTED series only
        for series in selected_series:
            if series not in working_df.columns:
                continue

            is_long_short = long_short_dict.get(series, False)
            benchmark = benchmark_dict.get(series, "None") # Default to None if not found

            # If relative metric and no benchmark, return NaN
            if rolling_metric in ["excess_return", "tracking_error", "information_ratio", "correlation"] and benchmark == "None":
                 rolling_df[series] = np.nan
                 continue

            # 1. Resolve Series Returns
            series_ret = working_df[series]

            # 2. Resolve Benchmark Returns from working_df
            if benchmark in working_df.columns and benchmark != "None":
                bench_ret = working_df[benchmark]
                # Reindex bench_ret to match series_ret (though logic mostly relies on index alignment)
                # working_df columns share same index
            else:
                bench_ret = None

            # 3. Calculate based on Metric
            if rolling_metric == "total_return":
                if use_calendar_days:
                    res = series_ret.rolling(window=window_spec).apply(calc_rolling_return, raw=False)
                else:
                    res = series_ret.rolling(window=window_spec, min_periods=window_size).apply(calc_rolling_return, raw=False)
                rolling_df[series] = res

            elif rolling_metric == "excess_return":
                # Excess Return: Rolling(Series) - Rolling(Bench)
                if bench_ret is not None and not is_long_short:
                     # Calculate separately
                     if use_calendar_days:
                         roll_s = series_ret.rolling(window=window_spec).apply(calc_rolling_return, raw=False)
                         roll_b = bench_ret.rolling(window=window_spec).apply(calc_rolling_return, raw=False)
                     else:
                         roll_s = series_ret.rolling(window=window_spec, min_periods=window_size).apply(calc_rolling_return, raw=False)
                         roll_b = bench_ret.rolling(window=window_spec, min_periods=window_size).apply(calc_rolling_return, raw=False)
                     res = roll_s - roll_b
                else:
                     # L/S or No Benchmark -> Just calculate rolling of the series (which IS the L/S diff)
                     # or if no benchmark, just total return
                     if use_calendar_days:
                        res = series_ret.rolling(window=window_spec).apply(calc_rolling_return, raw=False)
                     else:
                        res = series_ret.rolling(window=window_spec, min_periods=window_size).apply(calc_rolling_return, raw=False)
                
                rolling_df[series] = res

            elif rolling_metric == "volatility":
                if use_calendar_days:
                    res = series_ret.rolling(window=window_spec).std() * np.sqrt(periods_per_year)
                else:
                    res = series_ret.rolling(window=window_spec, min_periods=window_size).std() * np.sqrt(periods_per_year)
                rolling_df[series] = res

            elif rolling_metric == "tracking_error":
                # For TE, we usually use the arithmetic difference stream's volatility
                if bench_ret is not None and not is_long_short:
                    diff = series_ret - bench_ret
                else:
                    diff = series_ret
                
                if use_calendar_days:
                    res = diff.rolling(window=window_spec).std() * np.sqrt(periods_per_year)
                else:
                    res = diff.rolling(window=window_spec, min_periods=window_size).std() * np.sqrt(periods_per_year)
                rolling_df[series] = res

            elif rolling_metric == "correlation":
                if bench_ret is None:
                    rolling_df[series] = np.nan
                else:
                    if use_calendar_days:
                        res = series_ret.rolling(window=window_spec).corr(bench_ret)
                    else:
                        res = series_ret.rolling(window=window_spec, min_periods=window_size).corr(bench_ret)
                    rolling_df[series] = res

            elif rolling_metric == "sharpe_ratio":
                func = lambda x: sharpe_ratio(x, periods_per_year)
                rolling_df[series] = apply_rolling_stat(series_ret, func)

            elif rolling_metric == "sortino_ratio":
                func = lambda x: sortino_ratio(x, periods_per_year)
                rolling_df[series] = apply_rolling_stat(series_ret, func)

            elif rolling_metric == "information_ratio":
                if bench_ret is not None and not is_long_short:
                    diff = series_ret - bench_ret
                else:
                    diff = series_ret
                
                func = lambda x: sharpe_ratio(x, periods_per_year)
                rolling_df[series] = apply_rolling_stat(diff, func)

        # For calendar-based windows, filter out periods that don't have enough calendar days
        if use_calendar_days and len(rolling_df) > 0:
            first_date = working_df.index.min()
            valid_dates_mask = (rolling_df.index - first_date).days >= min_calendar_days - 1
            rolling_df = rolling_df[valid_dates_mask]

        # Drop rows with all NaN values
        rolling_df = rolling_df.dropna(how='all')

        return rolling_df

    except Exception as e:
        return pd.DataFrame()





@cache_config.cache.memoize(timeout=0)
def calculate_calendar_year_returns(raw_data, original_periodicity, selected_periodicity, selected_series, returns_type, benchmark_assignments, long_short_assignments, date_range):
    """Calculate calendar year returns for Excel export."""
    try:
        # Use get_working_returns for data prep
        working_df = get_working_returns(
            raw_data, selected_periodicity or "daily", tuple(selected_series),
            str(benchmark_assignments), str(long_short_assignments), str(date_range)
        )

        if working_df.empty:
            return pd.DataFrame()
            
        benchmark_dict = eval(str(benchmark_assignments)) if benchmark_assignments else {}
        long_short_dict = eval(str(long_short_assignments)) if long_short_assignments else {}

        calendar_returns = {}

        # Compute annual returns for each SELECTED series
        for series in selected_series:
            if series not in working_df.columns:
                continue

            series_returns = working_df[series].dropna()
            
            if series_returns.empty:
                continue

            # Group by year and compound returns
            series_returns_df = series_returns.to_frame(name='returns')
            series_returns_df['year'] = series_returns.index.year

            # Calculate annual returns
            annual_returns = series_returns_df.groupby('year')['returns'].apply(
                lambda x: (1 + x).prod(min_count=1) - 1
            )
            
            # Filter out partial years (exclude first and last year if partial)
            if len(annual_returns) > 0:
                first_year = annual_returns.index.min()
                last_year = annual_returns.index.max()

                # Check if first year is complete
                first_year_data = series_returns[series_returns.index.year == first_year]
                if len(first_year_data) > 0:
                    current_periodicity = selected_periodicity or "daily"
                    if current_periodicity == "daily":
                        # For daily data, check if it starts in January
                        first_date = first_year_data.index.min()
                        if not first_date.is_year_start:
                            annual_returns = annual_returns.drop(first_year, errors='ignore')
                    elif current_periodicity == "monthly":
                        # For monthly data, check if all 12 months are present
                        if len(first_year_data) < 12:
                            annual_returns = annual_returns.drop(first_year, errors='ignore')

                # Check if last year is complete
                last_year_data = series_returns[series_returns.index.year == last_year]
                if len(last_year_data) > 0:
                    last_date = last_year_data.index.max()
                    if not last_date.is_year_end:
                        annual_returns = annual_returns.drop(last_year, errors='ignore')
            
            # If Excess Return requested (and not L/S), subtract Annual Benchmark Return
            is_ls = long_short_dict.get(series, False)
            if returns_type == "excess" and not is_ls:
                benchmark = benchmark_dict.get(series, "None")
                if benchmark != "None" and benchmark in working_df.columns:
                    # Calculate annual returns for benchmark
                    # Use benchmark from working_df
                    bench_series = working_df[benchmark].dropna()
                    
                    bench_df = bench_series.to_frame(name='returns')
                    bench_df['year'] = bench_series.index.year
                    
                    annual_bench = bench_df.groupby('year')['returns'].apply(
                        lambda x: (1 + x).prod(min_count=1) - 1
                    )
                    
                    # Align to series annual returns (years match)
                    annual_bench = annual_bench.reindex(annual_returns.index)
                    
                    # Subtract
                    annual_returns = annual_returns - annual_bench

            calendar_returns[series] = annual_returns

        if not calendar_returns:
            return pd.DataFrame()

        # Get all years that have data for at least one series
        all_years = sorted(set().union(*[set(cr.index) for cr in calendar_returns.values()]))

        if not all_years:
            return pd.DataFrame()

        # Build DataFrame
        result = pd.DataFrame(index=all_years)
        result.index.name = 'Year'

        for series in selected_series:
            if series in calendar_returns:
                result[series] = calendar_returns[series]

        return result

    except Exception:
        return pd.DataFrame()


# Monthly view creation

def create_monthly_view(raw_data, series_name, original_periodicity, selected_periodicity, returns_type, benchmark_assignments, long_short_assignments, selected_series, date_range):
    """Create monthly view with Jan-Dec columns plus Year column."""
    # Use get_working_returns for data prep
    working_df = get_working_returns(
        raw_data, selected_periodicity or "daily", (series_name,),
        str(benchmark_assignments), str(long_short_assignments), str(date_range)
    )
    
    if series_name not in working_df.columns:
        return [], []
        
    series_returns = working_df[series_name].dropna()

    if series_returns.empty:
        return [], []
        
    # Check configurations
    benchmark_dict = eval(str(benchmark_assignments)) if benchmark_assignments else {}
    long_short_dict = eval(str(long_short_assignments)) if long_short_assignments else {}
    is_ls = long_short_dict.get(series_name, False)
    
    # If Excess requested (and not L/S), we need benchmark data
    calc_excess = (returns_type == "excess" and not is_ls)
    if calc_excess:
        benchmark = benchmark_dict.get(series_name, "None")
        if benchmark != "None" and benchmark in working_df.columns:
             # Use benchmark from working_df
             bench_returns = working_df[benchmark].reindex(series_returns.index)
        else:
             calc_excess = False

    # Helper to aggregate to monthly
    def aggregate_monthly(rets):
        # Convert to DataFrame for processing
        s_data = rets.to_frame(name='returns')

        if selected_periodicity == "daily":
            # For daily data, resample to monthly
            s_data['year'] = s_data.index.year
            s_data['month'] = s_data.index.month

            # Group by year and month, compound returns
            monthly_data = s_data.groupby(['year', 'month'])['returns'].apply(
                lambda x: (1 + x).prod(min_count=1) - 1
            ).reset_index()

        elif selected_periodicity == "monthly":
            # Already monthly, just add year and month columns
            monthly_data = pd.DataFrame({
                'year': s_data.index.year,
                'month': s_data.index.month,
                'returns': s_data['returns']
            }).reset_index(drop=True)
        else:
            return pd.DataFrame()
            
        return monthly_data

    # Calculate monthly data for series
    monthly_data = aggregate_monthly(series_returns)
    
    if monthly_data.empty:
        return [], []

    # If excess, calculate monthly data for benchmark and diff
    if calc_excess:
        # For grid cells: use ALIGNED benchmark (bench_returns)
        # For Annual column: use FULL benchmark (to match Annual Grid)
        
        # 1. Grid Cells (Aligned)
        bench_monthly = aggregate_monthly(bench_returns)
        if not bench_monthly.empty:
            # Merge on year/month
            merged = monthly_data.merge(bench_monthly, on=['year', 'month'], suffixes=('_s', '_b'))
            # Calculate excess (Arithmetic for monthly cells)
            merged['returns'] = merged['returns_s'] - merged['returns_b']
            
            # Keep track of component returns for Annual calc
            monthly_data = merged
        else:
            calc_excess = False # Fallback

    # Pivot to get months as columns
    pivot_data = monthly_data.pivot(index='year', columns='month', values='returns')

    # Rename columns to month names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    pivot_data.columns = [month_names[m-1] if m <= 12 else f'M{m}' for m in pivot_data.columns]

    # Helper for annual calc (requires full year)
    def calc_annual(row):
        # Must have data for all 12 months
        if row.count() < 12: return None
        return (1 + row.dropna()).prod() - 1

    # Calculate Annual column
    if calc_excess:
        # For excess, Annual = Ann(S) - Ann(B)
        # Both S and B must be full years
        
        pivot_s = monthly_data.pivot(index='year', columns='month', values='returns_s')
        ann_s = pivot_s.apply(calc_annual, axis=1)
        
        # Calculate Full Benchmark Annual Returns
        full_bench_series = working_df[benchmark].dropna()
        full_bench_monthly = aggregate_monthly(full_bench_series)
        
        if not full_bench_monthly.empty:
            pivot_b_full = full_bench_monthly.pivot(index='year', columns='month', values='returns')
            ann_b_full = pivot_b_full.apply(calc_annual, axis=1)
            
            # Align B to S years
            ann_b_full = ann_b_full.reindex(ann_s.index)
            
            # Subtract (only if both are non-None)
            pivot_data['Ann'] = ann_s - ann_b_full
        else:
            pivot_data['Ann'] = None
            
    else:
        # Standard compound
        pivot_data['Ann'] = pivot_data.apply(calc_annual, axis=1)
    # Reset index to make year a column
    pivot_data = pivot_data.reset_index()
    pivot_data = pivot_data.rename(columns={'year': 'Year_Label'})

    # Reorder columns: Year_Label, Jan, Feb, ..., Dec, Ann
    month_cols = [m for m in month_names if m in pivot_data.columns]
    col_order = ['Year_Label'] + month_cols + ['Ann']
    pivot_data = pivot_data[col_order]

    # Create column definitions for monthly view
    column_defs = [
        {
            "field": "Year_Label",
            "headerName": "Year",
            "pinned": "left",
            "width": 80,
        }
    ]

    # Add month columns
    for month in month_cols:
        column_defs.append({
            "field": month,
            "valueFormatter": {"function": "params.value != null ? d3.format('.2%')(params.value) : ''"},
            "width": 90,
        })

    # Add Annual column
    column_defs.append({
        "field": "Ann",
        "valueFormatter": {"function": "params.value != null ? d3.format('.2%')(params.value) : ''"},
        "width": 90,
    })

    # Convert to row data
    row_data = pivot_data.to_dict("records")

    return column_defs, row_data