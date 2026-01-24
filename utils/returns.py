"""Returns calculation utilities for compounding and resampling."""

import numpy as np
import pandas as pd


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
