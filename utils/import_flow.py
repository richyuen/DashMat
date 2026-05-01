"""Shared helpers for import/database-add callback business logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import pandas as pd

from utils.add_series_flow import find_duplicate_series
from utils.upload_flow import UploadMergeResult, merge_uploaded_with_existing


@dataclass(frozen=True)
class CmaImportSummary:
    new_periodicity: str
    all_start_daily: bool
    daily_transition_notes: tuple[str, ...]


def find_import_duplicates(
    imported_columns: Iterable[str] | None,
    *,
    existing_data=None,
    existing_dataset_key: str | None = None,
) -> list[str]:
    raw_meta = {"dataset_key": existing_dataset_key} if existing_dataset_key else None
    imported_values = [] if imported_columns is None else list(imported_columns)
    return find_duplicate_series(imported_values, raw_data=existing_data, raw_meta=raw_meta)


def merge_imported_dataset(
    existing_periodicity,
    new_df: pd.DataFrame,
    *,
    existing_data=None,
    existing_dataset_key: str | None = None,
    new_periodicity: str | None = None,
    daily_default_periodicity: str = "daily_trading",
) -> UploadMergeResult:
    return merge_uploaded_with_existing(
        existing_data,
        existing_periodicity,
        new_df,
        dataset_key=existing_dataset_key,
        new_periodicity=new_periodicity,
        daily_default_periodicity=daily_default_periodicity,
    )


def build_cma_import_summary(
    imported_df: pd.DataFrame,
    db_meta,
) -> CmaImportSummary:
    any_daily_phase = False
    all_start_daily = True
    daily_transition_notes: list[str] = []

    for series_name in imported_df.columns:
        meta = db_meta.get(series_name, {}) if isinstance(db_meta, dict) else {}
        starts_daily = bool(meta.get("starts_daily", True)) if isinstance(meta, dict) else True
        daily_start_date = meta.get("daily_start_date") if isinstance(meta, dict) else None
        has_daily_phase = bool(daily_start_date) or starts_daily
        any_daily_phase = any_daily_phase or has_daily_phase
        if not starts_daily:
            all_start_daily = False
            if daily_start_date:
                daily_transition_notes.append(f"{series_name}: {daily_start_date}")
            elif not has_daily_phase:
                daily_transition_notes.append(f"{series_name}: no daily phase detected")
            else:
                daily_transition_notes.append(f"{series_name}: daily phase starts after initial history")

    return CmaImportSummary(
        new_periodicity="daily" if any_daily_phase else "monthly",
        all_start_daily=all_start_daily,
        daily_transition_notes=tuple(daily_transition_notes),
    )


def resolve_cma_default_periodicity(
    combined_periodicity: str,
    cma_summary: CmaImportSummary,
    *,
    daily_default_periodicity: str,
    monthly_fallback_periodicity: str = "monthly",
) -> str:
    if combined_periodicity != "daily":
        return combined_periodicity
    return daily_default_periodicity if cma_summary.all_start_daily else monthly_fallback_periodicity


def extend_selection(current_selection, imported_columns: Iterable[str]) -> list[str]:
    updated = list(current_selection or [])
    seen = set(updated)
    for column in imported_columns:
        if column not in seen:
            updated.append(column)
            seen.add(column)
    return updated


def merge_assignments(
    current_assignments: Mapping[str, str] | None,
    imported_assignments: Mapping[str, str] | None,
) -> dict[str, str]:
    merged = dict(current_assignments or {})
    merged.update(imported_assignments or {})
    return merged
