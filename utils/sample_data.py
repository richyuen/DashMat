"""Sample data generation for DashMat."""

from io import BytesIO

import numpy as np
import pandas as pd


# Series names for sample files
SAMPLE_SERIES_NAMES = [
    "US Equity",
    "Intl Equity",
    "US Bonds",
    "Real Estate",
    "Commodities",
    "High Yield",
]


def generate_sample_returns(periodicity: str) -> pd.DataFrame:
    """
    Generate sample returns data for download.

    Args:
        periodicity: 'daily' or 'monthly'

    Returns:
        DataFrame with Date index and 6 return series columns
    """
    # Fixed seed for reproducibility
    rng = np.random.default_rng(42)

    # Define date range
    full_start = pd.Timestamp("2021-01-01")
    full_end = pd.Timestamp("2025-12-31")

    # Create date index based on periodicity
    if periodicity == "daily":
        # Business days only
        master_index = pd.date_range(start=full_start, end=full_end, freq="B")
        periods_per_year = 252
    else:
        # Monthly (month end)
        master_index = pd.date_range(start=full_start, end=full_end, freq="ME")
        periods_per_year = 12

    df = pd.DataFrame(index=master_index)
    df.index.name = "Date"

    # Random parameters for each series
    # Annual means between 0% and 8%
    annual_means = rng.uniform(0.00, 0.08, size=len(SAMPLE_SERIES_NAMES))
    # Annual vols between 6% and 20%
    annual_vols = rng.uniform(0.06, 0.20, size=len(SAMPLE_SERIES_NAMES))

    # Convert to per-period
    period_means = annual_means / periods_per_year
    period_vols = annual_vols / np.sqrt(periods_per_year)

    # Random start dates for series 2-6 (between 2021 and 2022)
    # Random end dates for series 2-6 (between 2024 and 2025)
    start_range = pd.date_range("2021-01-01", "2022-12-31", freq="D")
    end_range = pd.date_range("2024-01-01", "2025-12-31", freq="D")

    for i, series_name in enumerate(SAMPLE_SERIES_NAMES):
        if i == 0:
            # First series has full date range
            series_start = full_start
            series_end = full_end
        else:
            # Other series have random start/end dates
            series_start = rng.choice(start_range)
            series_end = rng.choice(end_range)

        # Create mask for valid dates
        mask = (df.index >= series_start) & (df.index <= series_end)
        n_periods = mask.sum()

        if n_periods > 0:
            # Generate returns with series-specific mean and vol
            returns = rng.normal(
                loc=period_means[i], scale=period_vols[i], size=n_periods
            )

            # Initialize column and set values
            df[series_name] = np.nan
            df.loc[mask, series_name] = returns

    # Drop rows where all columns are NaN
    df.dropna(how="all", inplace=True)

    return df


def create_sample_excel(periodicity: str) -> bytes:
    """
    Create an Excel file with sample returns data.

    Args:
        periodicity: 'daily' or 'monthly'

    Returns:
        Excel file as bytes
    """
    df = generate_sample_returns(periodicity)

    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Returns")
    output.seek(0)

    return output.getvalue()
