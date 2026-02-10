"""Parsing utilities for CSV/Excel files with returns data."""

import base64
import io
import pandas as pd


def get_sheet_names(contents: str, filename: str) -> list[str]:
    """Return the list of sheet names for an Excel file.

    Args:
        contents: Base64 encoded file contents from dcc.Upload
        filename: Original filename to determine file type

    Returns:
        List of sheet names, or empty list for non-Excel files.
    """
    if not filename.endswith((".xlsx", ".xls")):
        return []
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    xls = pd.ExcelFile(io.BytesIO(decoded))
    return xls.sheet_names


def parse_uploaded_file(contents: str, filename: str, sheet_name=0) -> pd.DataFrame:
    """Parse uploaded file contents into a DataFrame.

    Args:
        contents: Base64 encoded file contents from dcc.Upload
        filename: Original filename to determine file type
        sheet_name: Sheet name or index for Excel files (default: 0, first sheet)

    Returns:
        DataFrame with DatetimeIndex and returns as columns
    """
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)

    if filename.endswith(".csv"):
        df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
    elif filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(decoded), sheet_name=sheet_name)
    else:
        raise ValueError(f"Unsupported file type: {filename}")

    # First column is assumed to be dates
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    df.index.name = "Date"

    # Convert percent values to decimals
    df = convert_percents_to_decimals(df)

    # Sort by date
    df = df.sort_index()

    return df


def convert_percents_to_decimals(df: pd.DataFrame) -> pd.DataFrame:
    """Convert any percent-formatted values to decimals.

    Detects values with '%' suffix and divides by 100.
    """
    result = df.copy()

    for col in result.columns:
        if result[col].dtype == object:
            # Convert to string once and cache it
            str_series = result[col].astype(str)
            # Check if any values contain '%'
            mask = str_series.str.contains("%", na=False)
            if mask.any():
                # Convert entire column: strip '%' and divide by 100
                result[col] = (
                    str_series
                    .str.replace("%", "", regex=False)
                    .astype(float)
                    / 100
                )
        # Also handle case where values are already numeric but > 1 (likely percents)
        # We don't auto-convert these as they could be valid large returns

    # Ensure all columns are float
    for col in result.columns:
        result[col] = pd.to_numeric(result[col], errors="coerce")

    return result


def detect_periodicity(df: pd.DataFrame) -> str:
    """Detect if the data is daily or monthly using the first few rows.

    Returns:
        'daily' or 'monthly'
    """
    if len(df) < 2:
        return "daily"

    # Calculate median difference between consecutive dates using first 5 rows
    # for performance on large datasets
    sample_index = df.index[:5]
    date_diffs = pd.Series(sample_index).diff().dropna()
    median_diff = date_diffs.median().days

    # If median difference is > 20 days, assume monthly
    if median_diff > 20:
        return "monthly"
    return "daily"
