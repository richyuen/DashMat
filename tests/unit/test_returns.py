from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from utils.returns import (
    _fast_rolling_return_series,
    _legacy_rolling_return_series,
    align_monthly_index_to_month_end,
    align_monthly_series_to_month_end,
    annualization_factor,
    calculate_excess_returns,
    df_to_json,
    fill_calendar_gaps,
    filter_to_trading_days,
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


def test_filter_to_trading_days_fills_internal_missing_trading_days_with_zero():
    idx = pd.to_datetime(["2024-01-31", "2024-02-29"])
    df = pd.DataFrame({"A": [0.01, 0.02]}, index=idx)
    df.index.name = "Date"

    trading = filter_to_trading_days(df)

    assert pd.Timestamp("2024-02-01") in trading.index
    assert trading.loc[pd.Timestamp("2024-02-01"), "A"] == pytest.approx(0.0)
    assert trading.loc[pd.Timestamp("2024-01-31"), "A"] == pytest.approx(0.01)
    assert trading.loc[pd.Timestamp("2024-02-29"), "A"] == pytest.approx(0.02)


def test_filter_to_trading_days_compounds_weekend_returns_into_next_trading_day():
    idx = pd.to_datetime(["2024-01-05", "2024-01-06", "2024-01-08"])  # Fri, Sat, Mon
    df = pd.DataFrame({"A": [0.01, 0.02, 0.03]}, index=idx)
    df.index.name = "Date"

    trading = filter_to_trading_days(df)

    assert trading.loc[pd.Timestamp("2024-01-05"), "A"] == pytest.approx(0.01)
    expected_mon = (1.02 * 1.03) - 1.0
    assert trading.loc[pd.Timestamp("2024-01-08"), "A"] == pytest.approx(expected_mon)


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


def test_align_monthly_index_to_month_end_shifts_and_compounds_duplicates():
    idx = pd.to_datetime(["2024-01-30", "2024-01-31", "2024-02-27"])
    df = pd.DataFrame({"A": [0.01, 0.02, 0.03]}, index=idx)
    df.index.name = "Date"

    aligned = align_monthly_index_to_month_end(df)

    assert list(aligned.index) == [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")]
    assert aligned.loc[pd.Timestamp("2024-01-31"), "A"] == pytest.approx((1.01 * 1.02) - 1)
    assert aligned.loc[pd.Timestamp("2024-02-29"), "A"] == pytest.approx(0.03)


def test_align_monthly_series_to_month_end_is_noop_when_already_canonical():
    idx = pd.to_datetime(["2024-01-31", "2024-02-29"])
    s = pd.Series([0.01, 0.02], index=idx, name="R")
    s.index.name = "Date"

    aligned = align_monthly_series_to_month_end(s)

    assert list(aligned.index) == list(s.index)
    assert aligned.tolist() == pytest.approx(s.tolist())


def test_fast_rolling_return_series_matches_legacy_for_count_windows():
    idx = pd.date_range("2020-01-31", periods=60, freq="ME")
    series = pd.Series(np.linspace(-0.02, 0.03, len(idx)), index=idx)
    series.iloc[10] = np.nan
    series.iloc[22] = np.nan

    legacy = _legacy_rolling_return_series(
        series,
        use_calendar_days=False,
        window_spec=12,
        window_size=12,
        rolling_return_type="cumulative",
        window_years=1.0,
    )
    fast = _fast_rolling_return_series(
        series,
        use_calendar_days=False,
        window_spec=12,
        window_size=12,
        rolling_return_type="cumulative",
        window_years=1.0,
    )

    pd.testing.assert_series_equal(fast, legacy)


def test_fast_rolling_return_series_matches_legacy_for_calendar_windows():
    idx = pd.date_range("2018-01-01", periods=1200, freq="B")
    series = pd.Series(np.sin(np.linspace(0, 12, len(idx))) * 0.01, index=idx)
    series.iloc[50] = np.nan
    series.iloc[100] = np.nan

    legacy = _legacy_rolling_return_series(
        series,
        use_calendar_days=True,
        window_spec="1096D",
        window_size=None,
        rolling_return_type="annualized",
        window_years=3.0,
    )
    fast = _fast_rolling_return_series(
        series,
        use_calendar_days=True,
        window_spec="1096D",
        window_size=None,
        rolling_return_type="annualized",
        window_years=3.0,
    )

    pd.testing.assert_series_equal(fast, legacy)


def test_fast_rolling_return_series_falls_back_for_negative_hundred_percent_returns():
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    series = pd.Series([0.01, 0.02, -1.0, 0.03, 0.01, 0.0, 0.01, 0.02, 0.01, 0.0], index=idx)

    legacy = _legacy_rolling_return_series(
        series,
        use_calendar_days=False,
        window_spec=3,
        window_size=3,
        rolling_return_type="cumulative",
        window_years=1.0,
    )
    fast = _fast_rolling_return_series(
        series,
        use_calendar_days=False,
        window_spec=3,
        window_size=3,
        rolling_return_type="cumulative",
        window_years=1.0,
    )

    pd.testing.assert_series_equal(fast, legacy)
