from __future__ import annotations

import pandas as pd

from utils.date_range_flow import (
    ACCOUNT_LIST_MAX_END_SENTINEL,
    compute_common_daily_candidates,
    compute_date_range_candidates,
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
    assert "common_daily_start" not in candidates
    assert "common_daily_end" not in candidates


def test_compute_date_range_candidates_ignores_series_order_and_duplicates():
    ordered = compute_date_range_candidates(_raw_daily_df(), "daily_trading", ("A", "B", "A"))
    reordered = compute_date_range_candidates(_raw_daily_df(), "daily_trading", ("B", "A"))

    assert ordered == reordered


def test_compute_common_daily_candidates_ignores_series_order_and_duplicates():
    ordered = compute_common_daily_candidates(_raw_daily_df(), ("A", "B", "A"))
    reordered = compute_common_daily_candidates(_raw_daily_df(), ("B", "A"))

    assert ordered == reordered


def test_compute_common_daily_candidates_produces_bounds():
    candidates = compute_common_daily_candidates(_raw_daily_df(), ("A", "B"))

    assert candidates["common_daily_start"] == "2024-01-02"
    assert candidates["common_daily_end"] == "2024-01-03"


def test_resolve_initial_range_prefers_stored_when_in_bounds():
    candidates = {
        "max_start": "2024-01-01",
        "max_end": "2024-12-31",
    }
    stored = {"start": "2024-02-01", "end": "2024-03-01"}
    assert resolve_initial_range(candidates, stored) == ("2024-02-01", "2024-03-01")


def test_resolve_initial_range_maps_latest_sentinel_to_max_end():
    candidates = {
        "max_start": "2024-01-01",
        "max_end": "2024-12-31",
    }
    stored = {"start": "2024-02-01", "end": ACCOUNT_LIST_MAX_END_SENTINEL}

    assert resolve_initial_range(candidates, stored) == ("2024-02-01", "2024-12-31")


def test_resolve_initial_range_falls_back_when_sentinel_start_is_out_of_bounds():
    candidates = {
        "max_start": "2024-03-01",
        "max_end": "2024-12-31",
    }
    stored = {"start": "2024-02-01", "end": ACCOUNT_LIST_MAX_END_SENTINEL}

    assert resolve_initial_range(candidates, stored) == ("2024-03-01", "2024-12-31")


def test_resolve_button_range_switches_daily_for_common_daily():
    candidates = {
        "common_start": "2024-01-01",
        "common_end": "2024-01-31",
        "max_start": "2024-01-01",
        "max_end": "2024-02-15",
    }
    common_daily = {
        "common_daily_start": "2024-01-02",
        "common_daily_end": "2024-01-29",
    }

    assert resolve_button_range(candidates, "at-common-range-button") == ("2024-01-01", "2024-01-31", False)
    assert resolve_button_range(candidates, "po-common-daily-button", common_daily) == ("2024-01-02", "2024-01-29", True)
    assert resolve_button_range(candidates, "at-maximum-range-button") == ("2024-01-01", "2024-02-15", False)
