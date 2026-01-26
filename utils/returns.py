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


# Rolling returns calculation

@cache_config.cache.memoize(timeout=0)
def calculate_rolling_returns(raw_data, periodicity, selected_series, returns_type, benchmark_assignments, long_short_assignments, date_range, rolling_window="1y", rolling_return_type="annualized"):
    """Calculate rolling returns for Excel export - matches the Rolling grid logic."""
    try:
        from utils.statistics import annualization_factor

        # Get the resampled data
        df = resample_returns_cached(raw_data, periodicity or "daily")

        # Apply date range filter if provided
        date_range_dict = eval(str(date_range)) if date_range and str(date_range) != "None" else None
        if date_range_dict:
            start_date = pd.to_datetime(date_range_dict["start"])
            end_date = pd.to_datetime(date_range_dict["end"])
            df = df[(df.index >= start_date) & (df.index <= end_date)]

        # Parse assignments
        benchmark_dict = eval(str(benchmark_assignments)) if benchmark_assignments else {}
        long_short_dict = eval(str(long_short_assignments)) if long_short_assignments else {}

        # Filter to selected series only
        available_series = [s for s in selected_series if s in df.columns]
        if not available_series:
            return pd.DataFrame()

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

        # Map rolling window to number of years for annualization
        window_years_map = {
            "3m": 0.25,
            "6m": 0.5,
            "1y": 1.0,
            "3y": 3.0,
            "5y": 5.0,
            "10y": 10.0,
        }
        window_years = window_years_map.get(rolling_window, 1.0)

        # Calculate rolling returns for each series
        rolling_df = pd.DataFrame(index=df.index)

        for series in available_series:
            is_long_short = long_short_dict.get(series, False)
            benchmark = benchmark_dict.get(series, available_series[0])

            # Calculate rolling returns
            def calc_rolling_return(window):
                if len(window) == 0:
                    return np.nan
                # For count-based windows, check minimum size
                if not use_calendar_days and len(window) < window_size:
                    return np.nan
                cum_ret = (1 + window).prod() - 1
                if rolling_return_type == "annualized":
                    # If period is 1 year or less, return cumulative return (don't annualize)
                    if window_years <= 1.0:
                        return cum_ret
                    # Annualize based on the window years, not actual periods
                    return (1 + cum_ret) ** (1 / window_years) - 1
                else:  # cumulative
                    return cum_ret

            if is_long_short:
                # Long-short: always calculate period-by-period difference, then compound
                if benchmark == "None":
                    series_returns = df[series]
                elif benchmark == series:
                    rolling_df[series] = np.nan
                    continue
                elif benchmark in df.columns:
                    series_returns = df[series] - df[benchmark]
                else:
                    series_returns = df[series]

                if use_calendar_days:
                    rolling_returns = series_returns.rolling(window=window_spec).apply(
                        calc_rolling_return, raw=False
                    )
                else:
                    rolling_returns = series_returns.rolling(window=window_spec, min_periods=window_size).apply(
                        calc_rolling_return, raw=False
                    )
                rolling_df[series] = rolling_returns
            else:
                # Non-long-short: apply returns_type logic
                if returns_type == "excess":
                    # For excess returns: compound(series) - compound(benchmark)
                    if benchmark == "None":
                        if use_calendar_days:
                            rolling_returns = df[series].rolling(window=window_spec).apply(
                                calc_rolling_return, raw=False
                            )
                        else:
                            rolling_returns = df[series].rolling(window=window_spec, min_periods=window_size).apply(
                                calc_rolling_return, raw=False
                            )
                        rolling_df[series] = rolling_returns
                    elif benchmark == series:
                        rolling_df[series] = np.nan
                    elif benchmark in df.columns:
                        # Calculate rolling returns for series and benchmark separately
                        if use_calendar_days:
                            rolling_series = df[series].rolling(window=window_spec).apply(
                                calc_rolling_return, raw=False
                            )
                            rolling_bench = df[benchmark].rolling(window=window_spec).apply(
                                calc_rolling_return, raw=False
                            )
                        else:
                            rolling_series = df[series].rolling(window=window_spec, min_periods=window_size).apply(
                                calc_rolling_return, raw=False
                            )
                            rolling_bench = df[benchmark].rolling(window=window_spec, min_periods=window_size).apply(
                                calc_rolling_return, raw=False
                            )
                        rolling_df[series] = rolling_series - rolling_bench
                    else:
                        if use_calendar_days:
                            rolling_returns = df[series].rolling(window=window_spec).apply(
                                calc_rolling_return, raw=False
                            )
                        else:
                            rolling_returns = df[series].rolling(window=window_spec, min_periods=window_size).apply(
                                calc_rolling_return, raw=False
                            )
                        rolling_df[series] = rolling_returns
                else:  # total returns
                    if use_calendar_days:
                        rolling_returns = df[series].rolling(window=window_spec).apply(
                            calc_rolling_return, raw=False
                        )
                    else:
                        rolling_returns = df[series].rolling(window=window_spec, min_periods=window_size).apply(
                            calc_rolling_return, raw=False
                        )
                    rolling_df[series] = rolling_returns

        # For calendar-based windows, filter out periods that don't have enough calendar days
        if use_calendar_days and len(rolling_df) > 0:
            first_date = df.index.min()
            # Create a mask for dates that have at least min_calendar_days from the first date
            valid_dates_mask = (rolling_df.index - first_date).days >= min_calendar_days - 1
            rolling_df = rolling_df[valid_dates_mask]

        # Drop rows with all NaN values
        rolling_df = rolling_df.dropna(how='all')

        return rolling_df

    except Exception:
        return pd.DataFrame()


# Calendar year returns calculation

def _compute_calendar_year_returns(df, original_periodicity, available_series, returns_type, benchmark_dict, long_short_dict):
    """Helper function to compute calendar year returns for all series.

    Returns:
        dict: Dictionary mapping series names to their annual returns Series
    """
    calendar_returns = {}

    for series in available_series:
        is_long_short = long_short_dict.get(series, False)
        benchmark = benchmark_dict.get(series, available_series[0])

        # Get series returns based on returns_type and long-short
        if is_long_short:
            if benchmark == "None":
                series_returns = df[series]
            elif benchmark == series:
                continue  # Skip if benchmark equals series
            elif benchmark in df.columns:
                series_returns = df[series] - df[benchmark]
            else:
                series_returns = df[series]
        else:
            if returns_type == "excess":
                if benchmark == "None":
                    series_returns = df[series]
                elif benchmark == series:
                    continue  # Skip if benchmark equals series
                elif benchmark in df.columns:
                    series_returns = df[series] - df[benchmark]
                else:
                    series_returns = df[series]
            else:  # total returns
                series_returns = df[series]

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
                if original_periodicity == "daily":
                    # For daily data, check if it starts in January
                    first_date = first_year_data.index.min()
                    if not first_date.is_year_start:
                        annual_returns = annual_returns.drop(first_year, errors='ignore')
                else:  # monthly
                    # For monthly data, check if all 12 months are present
                    if len(first_year_data) < 12:
                        annual_returns = annual_returns.drop(first_year, errors='ignore')

            # Check if last year is complete
            last_year_data = series_returns[series_returns.index.year == last_year]
            if len(last_year_data) > 0:
                last_date = last_year_data.index.max()
                if not last_date.is_year_end:
                    annual_returns = annual_returns.drop(last_year, errors='ignore')

        calendar_returns[series] = annual_returns

    return calendar_returns


@cache_config.cache.memoize(timeout=0)
def calculate_calendar_year_returns(raw_data, original_periodicity, selected_periodicity, selected_series, returns_type, benchmark_assignments, long_short_assignments, date_range):
    """Calculate calendar year returns for Excel export."""
    try:
        df = json_to_df(raw_data)

        # Apply date range filter if provided
        date_range_dict = eval(str(date_range)) if date_range and str(date_range) != "None" else None
        if date_range_dict:
            start_date = pd.to_datetime(date_range_dict["start"])
            end_date = pd.to_datetime(date_range_dict["end"])

            if selected_periodicity == "monthly":
                # Fall back to beginning of month (e.g., 1/31 -> 1/1)
                start_date = start_date.replace(day=1)
            elif selected_periodicity and selected_periodicity.startswith("weekly_"):
                # Fall back 6 days (e.g., 1/8 -> 1/2)
                start_date = start_date - pd.Timedelta(days=6)

            df = df[(df.index >= start_date) & (df.index <= end_date)]

        # Parse assignments
        benchmark_dict = eval(str(benchmark_assignments)) if benchmark_assignments else {}
        long_short_dict = eval(str(long_short_assignments)) if long_short_assignments else {}

        # Filter to selected series only
        available_series = [s for s in selected_series if s in df.columns]
        if not available_series:
            return pd.DataFrame()

        # Compute calendar year returns using shared helper
        calendar_returns = _compute_calendar_year_returns(
            df, original_periodicity, available_series, returns_type, benchmark_dict, long_short_dict
        )

        if not calendar_returns:
            return pd.DataFrame()

        # Get all years that have data for at least one series
        all_years = sorted(set().union(*[set(cr.index) for cr in calendar_returns.values()]))

        if not all_years:
            return pd.DataFrame()

        # Build DataFrame
        result = pd.DataFrame(index=all_years)
        result.index.name = 'Year'

        for series in available_series:
            if series in calendar_returns:
                result[series] = calendar_returns[series]

        return result

    except Exception:
        return pd.DataFrame()


# Monthly view creation

def create_monthly_view(raw_data, series_name, original_periodicity, selected_periodicity,returns_type, benchmark_assignments, long_short_assignments, selected_series, date_range):
    """Create monthly view with Jan-Dec columns plus Year column."""
    # Use raw data (original periodicity) regardless of selected periodicity
    df = json_to_df(raw_data)

    # Apply date range filter if provided
    date_range_dict = eval(str(date_range)) if date_range and str(date_range) != "None" else None
    if date_range_dict:
        start_date = pd.to_datetime(date_range_dict["start"])
        end_date = pd.to_datetime(date_range_dict["end"])

        if selected_periodicity == "monthly":
            # Fall back to beginning of month (e.g., 1/31 -> 1/1)
            start_date = start_date.replace(day=1)
        elif selected_periodicity and selected_periodicity.startswith("weekly_"):
            # Fall back 6 days (e.g., 1/8 -> 1/2)
            start_date = start_date - pd.Timedelta(days=6)

        df = df[(df.index >= start_date) & (df.index <= end_date)]

    # Parse assignments
    benchmark_dict = eval(str(benchmark_assignments)) if benchmark_assignments else {}
    long_short_dict = eval(str(long_short_assignments)) if long_short_assignments else {}

    # Determine if series is long-short
    is_long_short = long_short_dict.get(series_name, False)
    benchmark = benchmark_dict.get(series_name, selected_series[0])

    # Get series returns based on returns_type and long-short
    if is_long_short:
        if benchmark == "None":
            series_returns = df[series_name]
        elif benchmark == series_name:
            # Return empty for self-benchmark
            return [], []
        elif benchmark in df.columns:
            series_returns = df[series_name] - df[benchmark]
        else:
            series_returns = df[series_name]
    else:
        if returns_type == "excess":
            if benchmark == "None":
                series_returns = df[series_name]
            elif benchmark == series_name:
                # Return empty for self-benchmark
                return [], []
            elif benchmark in df.columns:
                series_returns = df[series_name] - df[benchmark]
            else:
                series_returns = df[series_name]
        else:  # total returns
            series_returns = df[series_name]

    # Convert to DataFrame for processing
    series_data = series_returns.to_frame(name='returns')

    # Determine if we need to resample to monthly
    if original_periodicity == "daily":
        # For daily data, resample to monthly and only include full months
        # Add year and month columns
        series_data['year'] = series_data.index.year
        series_data['month'] = series_data.index.month

        # Group by year and month, compound returns
        monthly_data = series_data.groupby(['year', 'month'])['returns'].apply(
            lambda x: (1 + x).prod(min_count=1) - 1
        ).reset_index()

        # Count VALID days in each month to filter out partial months
        days_per_month = series_data.groupby(['year', 'month'])['returns'].count().reset_index(name='days')
        monthly_data = monthly_data.merge(days_per_month, on=['year', 'month'])

        # Only keep months with reasonable number of days (assume full month has at least 15 trading days)
        monthly_data = monthly_data[monthly_data['days'] >= 15]
        monthly_data = monthly_data.drop('days', axis=1)

    elif original_periodicity == "monthly":
        # Already monthly, just add year and month columns
        monthly_data = pd.DataFrame({
            'year': series_data.index.year,
            'month': series_data.index.month,
            'returns': series_data['returns']
        }).reset_index(drop=True)
    else:
        # For other periodicities, return empty
        return [], []

    if monthly_data.empty:
        return [], []

    # Pivot to get months as columns
    pivot_data = monthly_data.pivot(index='year', columns='month', values='returns')

    # Rename columns to month names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    pivot_data.columns = [month_names[m-1] if m <= 12 else f'M{m}' for m in pivot_data.columns]

    # Calculate Annual column (compound all months in the row)
    pivot_data['Ann'] = pivot_data.apply(
        lambda row: (1 + row.dropna()).prod() - 1 if row.notna().any() else None,
        axis=1
    )

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
