"""Shared upload/import helpers for Analytics and PortOpt pages."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from utils.parsing import detect_periodicity, get_sheet_names, parse_uploaded_file, parse_uploaded_sheets
from utils.returns import (
    align_monthly_index_to_month_end,
    get_available_periodicities,
    merge_returns,
    resample_returns,
)
from utils.raw_dataset import get_dataset_key, get_raw_dataset_df


def import_selected_workbook_sheets(contents, filename, selected_sheets, workbook_sheets=None):
    """Parse selected workbook sheets in workbook order.

    Later sheets overwrite earlier sheet values where they provide non-null data.
    """
    if workbook_sheets is None:
        workbook_sheets = get_sheet_names(contents, filename)
    else:
        workbook_sheets = list(workbook_sheets)
    if isinstance(selected_sheets, str):
        selected_values = [selected_sheets]
    else:
        selected_values = list(selected_sheets or [])
    selected_set = set(selected_values)
    ordered_sheets = [sheet for sheet in workbook_sheets if sheet in selected_set]
    if not ordered_sheets:
        raise ValueError("Select at least one sheet to import.")

    parsed_by_sheet = parse_uploaded_sheets(
        contents,
        filename,
        ordered_sheets,
        ignore_errors=True,
    )

    combined_df = None
    periodicity_hint = None
    imported_sheets: list[str] = []
    for sheet in ordered_sheets:
        parsed_df = parsed_by_sheet.get(sheet)
        if parsed_df is None or parsed_df.empty:
            continue

        if periodicity_hint is None:
            periodicity_hint = parsed_df.attrs.get("periodicity_hint")
        if combined_df is None:
            combined_df = parsed_df
        else:
            combined_df = parsed_df.combine_first(combined_df)
        imported_sheets.append(sheet)

    if combined_df is None:
        raise ValueError("No importable data rows found in selected sheets.")

    combined_df = combined_df.sort_index()
    if periodicity_hint in {"daily", "monthly"}:
        combined_df.attrs["periodicity_hint"] = periodicity_hint
    return combined_df, imported_sheets


def import_single_upload(contents, filename):
    """Parse a single uploaded file."""
    workbook_sheets = get_sheet_names(contents, filename)
    if len(workbook_sheets) > 1:
        raise ValueError("File contains multiple sheets.")
    if len(workbook_sheets) == 1:
        parsed_by_sheet = parse_uploaded_sheets(contents, filename, [workbook_sheets[0]])
        return parsed_by_sheet[workbook_sheets[0]]
    return parse_uploaded_file(contents, filename)


def _normalize_monthly_df_if_needed(df: pd.DataFrame, periodicity: str) -> pd.DataFrame:
    if periodicity == "monthly":
        return align_monthly_index_to_month_end(df)
    return df


@dataclass(frozen=True)
class UploadMergeResult:
    merged_df: pd.DataFrame
    combined_periodicity: str
    periodicity_options: list[dict]
    default_periodicity: str
    imported_df: pd.DataFrame


def merge_uploaded_with_existing(existing_data, existing_periodicity, new_df) -> UploadMergeResult:
    """Apply periodicity compatibility and merge new upload data."""
    new_periodicity = detect_periodicity(new_df)
    effective_new_df = new_df

    if existing_data is not None:
        dataset_key = get_dataset_key(existing_data)
        existing_df = get_raw_dataset_df(dataset_key) if dataset_key else pd.DataFrame()

        if existing_periodicity == "monthly" and new_periodicity == "daily":
            effective_new_df = resample_returns(new_df, "monthly")
            combined_periodicity = "monthly"
        elif new_periodicity == "monthly" and existing_periodicity == "daily":
            existing_df = resample_returns(existing_df, "monthly")
            combined_periodicity = "monthly"
        else:
            combined_periodicity = existing_periodicity

        existing_df = _normalize_monthly_df_if_needed(existing_df, combined_periodicity)
        effective_new_df = _normalize_monthly_df_if_needed(effective_new_df, combined_periodicity)
        merged_df = merge_returns(existing_df, effective_new_df)
    else:
        combined_periodicity = new_periodicity
        effective_new_df = _normalize_monthly_df_if_needed(new_df, combined_periodicity)
        merged_df = effective_new_df

    periodicity_options = get_available_periodicities(combined_periodicity)
    default_periodicity = "daily_trading" if combined_periodicity == "daily" else combined_periodicity
    return UploadMergeResult(
        merged_df=merged_df,
        combined_periodicity=combined_periodicity,
        periodicity_options=periodicity_options,
        default_periodicity=default_periodicity,
        imported_df=effective_new_df,
    )
