"""Returns calculation utilities for compounding and resampling."""

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
    """
    growth_factors = 1 + returns
    return growth_factors.prod() - 1


def resample_returns(df: pd.DataFrame, periodicity: str) -> pd.DataFrame:
    """Resample returns to a different periodicity.

    Args:
        df: DataFrame with DatetimeIndex and returns columns
        periodicity: Target periodicity (see RESAMPLE_CODES keys)

    Returns:
        Resampled DataFrame with compounded returns
    """
    if periodicity == "daily":
        return df

    resample_code = RESAMPLE_CODES.get(periodicity)
    if resample_code is None:
        raise ValueError(f"Unknown periodicity: {periodicity}")

    # Resample and compound returns for each period
    resampled = df.resample(resample_code).apply(compound_returns)

    # Drop rows where all values are NaN (periods with no data)
    resampled = resampled.dropna(how="all")

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
