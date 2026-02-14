from __future__ import annotations

import pandas as pd
import pytest

from utils.serialization import date_range_payload_for_cache, mapping_payload_for_cache
from utils.statistics import (
    calculate_drawdown,
    calculate_growth_of_dollar,
    calculate_statistics_cached,
    generate_correlogram_cached,
    maximum_drawdown,
)


def test_maximum_drawdown_basic_case():
    returns = pd.Series([0.10, -0.20, 0.05])
    assert maximum_drawdown(returns) == pytest.approx(-0.20)


def test_calculate_growth_of_dollar_prepends_start_value(raw_json):
    growth = calculate_growth_of_dollar(
        raw_json,
        "daily",
        ("Asset_A", "Asset_B"),
        {},
        {},
        None,
        0,
        {},
    )

    assert not growth.empty
    assert growth.index.name == "Date"
    assert growth.iloc[0]["Asset_A"] == pytest.approx(1.0)
    assert growth.iloc[0]["Asset_B"] == pytest.approx(1.0)


def test_calculate_drawdown_starts_at_zero_and_is_non_positive(raw_json):
    drawdown = calculate_drawdown(
        raw_json,
        "daily",
        ("Asset_A", "Asset_B"),
        "total",
        {},
        {},
        None,
        0,
        {},
    )

    assert not drawdown.empty
    assert drawdown.iloc[0]["Asset_A"] == pytest.approx(0.0)
    assert drawdown.iloc[0]["Asset_B"] == pytest.approx(0.0)
    assert (drawdown[["Asset_A", "Asset_B"]].dropna() <= 1e-12).all().all()


def test_calculate_statistics_cached_returns_one_result_per_series(raw_json):
    stats = calculate_statistics_cached(
        raw_json,
        "daily",
        ("Asset_A",),
        mapping_payload_for_cache({}),
        mapping_payload_for_cache({}),
        date_range_payload_for_cache(None),
        0,
        mapping_payload_for_cache({}),
        "",
        "",
    )

    assert len(stats) == 1
    assert stats[0]["Series"] == "Asset_A"
    assert "Annualized Return" in stats[0]


def test_generate_correlogram_cached_for_two_series(raw_json):
    result = generate_correlogram_cached(
        raw_json,
        "daily",
        ("Asset_A", "Asset_B"),
        "total",
        mapping_payload_for_cache({}),
        mapping_payload_for_cache({}),
        date_range_payload_for_cache(None),
        0,
        mapping_payload_for_cache({}),
    )

    assert result is not None
    assert result["n"] == 2
    assert result["corr_matrix"].shape == (2, 2)
