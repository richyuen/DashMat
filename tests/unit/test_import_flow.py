from __future__ import annotations

import pandas as pd

from utils.import_flow import (
    build_cma_import_summary,
    extend_selection,
    find_import_duplicates,
    merge_assignments,
    merge_imported_dataset,
    resolve_cma_default_periodicity,
)


def test_find_import_duplicates_supports_dataset_key(raw_data_store):
    duplicates = find_import_duplicates(
        ["Asset_A", "Imported"],
        existing_dataset_key=raw_data_store["dataset_key"],
    )

    assert duplicates == ["Asset_A"]


def test_build_cma_import_summary_tracks_transition_notes():
    imported_df = pd.DataFrame({"Daily": [0.01], "Transition": [0.02], "Monthly": [0.03]})
    summary = build_cma_import_summary(
        imported_df,
        {
            "Daily": {"starts_daily": True},
            "Transition": {"starts_daily": False, "daily_start_date": "2020-01-01"},
            "Monthly": {"starts_daily": False, "daily_start_date": None},
        },
    )

    assert summary.new_periodicity == "daily"
    assert summary.all_start_daily is False
    assert summary.daily_transition_notes == (
        "Transition: 2020-01-01",
        "Monthly: no daily phase detected",
    )


def test_resolve_cma_default_periodicity_uses_monthly_fallback_for_non_daily_starts():
    imported_df = pd.DataFrame({"Transition": [0.02]})
    summary = build_cma_import_summary(
        imported_df,
        {"Transition": {"starts_daily": False, "daily_start_date": "2020-01-01"}},
    )

    assert resolve_cma_default_periodicity("daily", summary, daily_default_periodicity="daily_trading") == "monthly"


def test_merge_imported_dataset_respects_periodicity_override(raw_data_store):
    imported_df = pd.DataFrame(
        {"Imported": [0.01, 0.02, 0.03]},
        index=pd.to_datetime(["2024-01-30", "2024-01-31", "2024-02-29"]),
    )
    imported_df.index.name = "Date"

    result = merge_imported_dataset(
        "monthly",
        imported_df,
        existing_data=raw_data_store,
        new_periodicity="daily",
    )

    assert result.combined_periodicity == "monthly"
    assert result.default_periodicity == "monthly"
    assert result.imported_df.index.is_month_end.all()


def test_extend_selection_and_merge_assignments_preserve_existing_order():
    updated_selection = extend_selection(["Asset_A"], ["Asset_A", "Imported"])
    updated_assignments = merge_assignments({"Asset_A": "None"}, {"Imported": "Bench_1"})

    assert updated_selection == ["Asset_A", "Imported"]
    assert updated_assignments == {"Asset_A": "None", "Imported": "Bench_1"}
