from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from utils.returns import (
    annualization_factor,
    calculate_excess_returns,
    df_to_json,
    fill_calendar_gaps,
    get_available_periodicities,
    get_working_returns,
    merge_returns,
    resample_returns,
)
from utils.serialization import date_range_payload_for_cache, mapping_payload_for_cache


def test_fill_calendar_gaps_fills_internal_missing_days_with_zero():
    idx = pd.to_datetime(["2024-01-01", "2024-01-03"])
    df = pd.DataFrame({"A": [0.01, 0.02]}, index=idx)
    df.index.name = "Date"

    filled = fill_calendar_gaps(df)

    assert pd.Timestamp("2024-01-02") in filled.index
    assert filled.loc[pd.Timestamp("2024-01-02"), "A"] == pytest.approx(0.0)


def test_resample_returns_monthly_compounds_returns():
    idx = pd.date_range("2024-01-01", periods=31, freq="D")
    df = pd.DataFrame({"A": np.full(len(idx), 0.001)}, index=idx)
    df.index.name = "Date"

    monthly = resample_returns(df, "monthly")
    expected = (1.001 ** 31) - 1
    assert len(monthly) == 1
    assert monthly["A"].iloc[0] == pytest.approx(expected, rel=1e-6)


def test_get_available_periodicities_for_monthly_input_is_monthly_only():
    assert get_available_periodicities("monthly") == [{"value": "monthly", "label": "Monthly"}]


def test_merge_returns_renames_overlapping_columns():
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    left = pd.DataFrame({"A": [0.01, 0.02]}, index=idx)
    right = pd.DataFrame({"A": [0.03, 0.04], "B": [0.05, 0.06]}, index=idx)

    merged = merge_returns(left, right)

    assert list(merged.columns) == ["A", "A_new", "B"]


def test_get_working_returns_aligns_to_benchmark_and_keeps_unselected_benchmark():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    df = pd.DataFrame(
        {
            "A": [0.01, 0.02, 0.03, 0.04, 0.05],
            "B": [np.nan, 0.01, 0.01, 0.01, 0.01],
            "C": [0.00, 0.00, 0.00, 0.00, 0.00],
        },
        index=idx,
    )
    df.index.name = "Date"
    raw_json = df_to_json(df)

    working = get_working_returns(
        raw_json,
        "daily",
        ("A", "C"),
        mapping_payload_for_cache({"A": "B", "C": "None"}),
        mapping_payload_for_cache({"A": False, "C": False}),
        date_range_payload_for_cache(None),
        0,
        mapping_payload_for_cache({}),
    )

    assert {"A", "B", "C"} <= set(working.columns)
    # A is benchmark-aligned; benchmark is missing on first row so A becomes NaN there.
    assert pd.isna(working.loc[pd.Timestamp("2024-01-01"), "A"])
    assert working.loc[pd.Timestamp("2024-01-01"), "C"] == pytest.approx(0.0)


def test_calculate_excess_returns_computes_selected_series_only():
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    df = pd.DataFrame(
        {
            "A": [0.02, 0.03, 0.01, 0.00],
            "B": [0.01, 0.01, 0.01, 0.01],
            "C": [0.00, 0.00, 0.00, 0.00],
        },
        index=idx,
    )
    df.index.name = "Date"
    raw_json = df_to_json(df)

    excess = calculate_excess_returns(
        raw_json,
        "daily",
        ("A", "C"),
        mapping_payload_for_cache({"A": "B", "C": "None"}),
        "excess",
        mapping_payload_for_cache({"A": False, "C": False}),
        date_range_payload_for_cache(None),
        0,
        mapping_payload_for_cache({}),
    )

    assert list(excess.columns) == ["A", "C"]
    assert excess["A"].iloc[1] == pytest.approx(0.02)
    assert excess["C"].iloc[1] == pytest.approx(0.0)


def test_annualization_factor_defaults_to_daily():
    assert annualization_factor("daily") == 252
    assert annualization_factor("monthly") == 12
    assert annualization_factor("unknown") == 252
