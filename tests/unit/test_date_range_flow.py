from __future__ import annotations

import hashlib

import pandas as pd

from utils.date_range_flow import (
    build_raw_data_metadata,
    compute_date_range_candidates,
    compute_date_range_candidates_from_global_metadata,
    resolve_button_range,
    resolve_initial_range,
)
from utils.returns import df_to_json


def _raw_daily_df() -> str:
    idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    df = pd.DataFrame(
        {
            "A": [0.01, 0.02, 0.03],
            "B": [0.01, None, 0.03],
        },
        index=idx,
    )
    df.index.name = "Date"
    return df_to_json(df)


def test_compute_date_range_candidates_produces_bounds():
    candidates = compute_date_range_candidates(_raw_daily_df(), "daily_trading", ("A", "B"))

    assert candidates["max_start"] == "2024-01-02"
    assert candidates["max_end"] == "2024-01-03"
    assert candidates["common_start"] == "2024-01-02"
    assert candidates["common_end"] == "2024-01-03"
    assert candidates["common_daily_start"] == "2024-01-02"
    assert candidates["common_daily_end"] == "2024-01-03"


def test_build_raw_data_metadata_returns_expected_top_level_payload():
    raw_data = _raw_daily_df()

    metadata = build_raw_data_metadata(raw_data, "daily")

    assert metadata["raw_data_hash"] == hashlib.md5(raw_data.encode("utf-8")).hexdigest()
    assert metadata["columns"] == ["A", "B"]
    assert metadata["available_periodicity_values"] == [
        "daily_trading",
        "daily",
        "monthly",
        "weekly_monday",
        "weekly_tuesday",
        "weekly_wednesday",
        "weekly_thursday",
        "weekly_friday",
    ]
    assert metadata["original_periodicity"] == "daily"
    assert metadata["periodicities"]["daily_trading"]["dataset_start"] == "2024-01-02"
    assert metadata["periodicities"]["daily_trading"]["dataset_end"] == "2024-01-03"
    assert metadata["daily_phase_ranges"]["A"]["start"] == "2024-01-02"


def test_compute_date_range_candidates_from_global_metadata_matches_compatibility_helper():
    raw_data = _raw_daily_df()
    metadata = build_raw_data_metadata(raw_data, "daily")

    candidates = compute_date_range_candidates_from_global_metadata(metadata, "daily_trading", ("A", "B"))

    assert candidates == compute_date_range_candidates(raw_data, "daily_trading", ("A", "B"))


def test_resolve_initial_range_prefers_stored_when_in_bounds():
    candidates = {
        "max_start": "2024-01-01",
        "max_end": "2024-12-31",
    }
    stored = {"start": "2024-02-01", "end": "2024-03-01"}
    assert resolve_initial_range(candidates, stored) == ("2024-02-01", "2024-03-01")


def test_resolve_button_range_switches_daily_for_common_daily():
    candidates = {
        "common_start": "2024-01-01",
        "common_end": "2024-01-31",
        "common_daily_start": "2024-01-02",
        "common_daily_end": "2024-01-29",
        "max_start": "2024-01-01",
        "max_end": "2024-02-15",
    }

    assert resolve_button_range(candidates, "at-common-range-button") == ("2024-01-01", "2024-01-31", False)
    assert resolve_button_range(candidates, "po-common-daily-button") == ("2024-01-02", "2024-01-29", True)
    assert resolve_button_range(candidates, "at-maximum-range-button") == ("2024-01-01", "2024-02-15", False)
