"""Helpers for consistent Excel export date formatting."""

from __future__ import annotations

from datetime import date, datetime
import re

import numpy as np
import pandas as pd

_DATE_PREFIX_PATTERN = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}(?:[ T].*)?$|^\d{1,2}/\d{1,2}/\d{4}(?:[ T].*)?$")


def format_mdy_date(value):
    """Format a single date-like value as m/d/yyyy, else return original."""
    ts = None
    if isinstance(value, pd.Timestamp):
        ts = value
    elif isinstance(value, np.datetime64):
        ts = pd.Timestamp(value)
    elif isinstance(value, (datetime, date)):
        ts = pd.Timestamp(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text or not _DATE_PREFIX_PATTERN.match(text):
            return value
        try:
            ts = pd.to_datetime(text, errors="raise")
        except Exception:
            return value
    else:
        return value

    if ts is None or pd.isna(ts):
        return value

    return f"{int(ts.month)}/{int(ts.day)}/{int(ts.year)}"


def format_excel_dates(df: pd.DataFrame, *, format_index: bool = False) -> pd.DataFrame:
    """Return a copy of DataFrame with date-like cells rendered as m/d/yyyy strings."""
    if df is None or df.empty:
        return df

    out = df.copy()

    if format_index:
        if isinstance(out.index, pd.DatetimeIndex):
            out.index = out.index.map(format_mdy_date)
        elif out.index.dtype == "object":
            out.index = out.index.map(format_mdy_date)

    for col in out.columns:
        series = out[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            out[col] = pd.to_datetime(series, errors="coerce").map(format_mdy_date)
        elif pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            out[col] = series.map(format_mdy_date)

    return out


def _display_len(value) -> int:
    """Return display length for Excel cell autosizing."""
    if value is None:
        return 0
    try:
        if pd.isna(value):
            return 0
    except Exception:
        pass
    return len(str(value))


def autofit_excel_columns(
    writer,
    sheet_name: str,
    df: pd.DataFrame,
    *,
    include_index: bool = False,
    index_label: str | None = None,
    min_width: float = 8.0,
    max_width: float = 60.0,
    padding: float = 2.0,
) -> None:
    """Autofit Excel columns for a written worksheet using DataFrame contents."""
    worksheet = writer.sheets.get(sheet_name)
    if worksheet is None or df is None:
        return

    col_offset = 0
    if include_index:
        idx_header = index_label if index_label is not None else (df.index.name or "")
        idx_values = pd.Index(df.index)
        idx_max = _display_len(idx_header)
        if len(idx_values):
            idx_max = max(idx_max, int(idx_values.map(_display_len).max()))
        idx_width = max(min_width, min(max_width, idx_max + padding))
        worksheet.set_column(0, 0, idx_width)
        col_offset = 1

    for i, col in enumerate(df.columns):
        header_len = _display_len(col)
        series = df[col]
        max_len = header_len
        if len(series):
            try:
                series_max = int(series.map(_display_len).max())
            except Exception:
                series_max = 0
            max_len = max(max_len, series_max)
        width = max(min_width, min(max_width, max_len + padding))
        worksheet.set_column(col_offset + i, col_offset + i, width)


def write_excel_with_autofit(
    writer,
    df: pd.DataFrame,
    sheet_name: str,
    *,
    index: bool = False,
    index_label: str | None = None,
    min_width: float = 8.0,
    max_width: float = 60.0,
    padding: float = 2.0,
) -> None:
    """Write DataFrame to Excel and autofit worksheet columns."""
    df.to_excel(writer, sheet_name=sheet_name, index=index, index_label=index_label)
    autofit_excel_columns(
        writer,
        sheet_name,
        df,
        include_index=index,
        index_label=index_label,
        min_width=min_width,
        max_width=max_width,
        padding=padding,
    )
