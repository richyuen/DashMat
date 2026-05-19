from __future__ import annotations

import pandas as pd
import pytest

from utils.serialization import date_range_payload_for_cache, mapping_payload_for_cache
from utils.returns import df_to_json
from utils.statistics import (
    calculate_drawdown,
    calculate_drawdown_events,
    calculate_growth_of_dollar,
    calculate_statistics_cached,
    generate_correlogram_cached,
    generate_pca_cached,
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


def test_calculate_drawdown_events_returns_recovered_event_boundaries():
    drawdown = pd.DataFrame(
        {"Asset_A": [0.0, -0.02, -0.05, -0.01, 0.0]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-04", "2024-01-05", "2024-01-08"]),
    )

    events = calculate_drawdown_events(drawdown)

    assert len(events) == 1
    event = events[0]
    assert event.series_name == "Asset_A"
    assert event.peak_date == pd.Timestamp("2024-01-01")
    assert event.trough_date == pd.Timestamp("2024-01-04")
    assert event.recovery_date == pd.Timestamp("2024-01-08")
    assert event.peak_to_trough_days == 3
    assert event.trough_to_recovery_days == 4
    assert event.trough_drawdown_value == pytest.approx(-0.05)


def test_calculate_drawdown_events_includes_unrecovered_active_drawdown():
    drawdown = pd.DataFrame(
        {"Asset_A": [0.0, -0.03, -0.01]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )

    events = calculate_drawdown_events(drawdown)

    assert len(events) == 1
    assert events[0].recovery_date is None
    assert events[0].trough_to_recovery_days is None
    assert events[0].trough_date == pd.Timestamp("2024-01-02")


def test_calculate_drawdown_events_segments_contiguous_negative_runs():
    drawdown = pd.DataFrame(
        {"Asset_A": [0.0, -0.02, 0.0, -0.01, -0.04, 0.0]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]),
    )

    events = calculate_drawdown_events(drawdown)

    assert [(event.peak_date, event.trough_date, event.recovery_date) for event in events] == [
        (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")),
        (pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-08")),
    ]


def test_calculate_drawdown_events_skips_zero_only_series():
    drawdown = pd.DataFrame(
        {"Asset_A": [0.0, 0.0, 0.0]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )

    assert calculate_drawdown_events(drawdown) == []


def test_calculate_drawdown_events_preserves_multi_series_order_and_names():
    drawdown = pd.DataFrame(
        {
            "Asset.A": [0.0, -0.02, 0.0],
            "Client Series": [0.0, -0.01, 0.0],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )

    events = calculate_drawdown_events(drawdown)

    assert [event.series_name for event in events] == ["Asset.A", "Client Series"]


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


def test_calculate_statistics_cached_can_disable_risk_free_proxy():
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    asset_df = pd.DataFrame({"Asset_A": [0.01, 0.012, 0.009, 0.011, 0.01]}, index=idx)
    rf_df = pd.DataFrame({"BCTBill13_TRIndex": [0.002, 0.002, 0.002, 0.002, 0.002]}, index=idx)

    stats_with_rf = calculate_statistics_cached(
        df_to_json(asset_df),
        "daily",
        ("Asset_A",),
        mapping_payload_for_cache({}),
        mapping_payload_for_cache({}),
        date_range_payload_for_cache(None),
        0,
        mapping_payload_for_cache({}),
        df_to_json(rf_df),
        "",
        True,
    )
    stats_without_rf = calculate_statistics_cached(
        df_to_json(asset_df),
        "daily",
        ("Asset_A",),
        mapping_payload_for_cache({}),
        mapping_payload_for_cache({}),
        date_range_payload_for_cache(None),
        0,
        mapping_payload_for_cache({}),
        df_to_json(rf_df),
        "",
        False,
    )

    assert stats_with_rf[0]["Sharpe Ratio"] != stats_without_rf[0]["Sharpe Ratio"]
    assert stats_with_rf[0]["Sortino Ratio"] != stats_without_rf[0]["Sortino Ratio"]


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
    assert result["cov_matrix"].shape == (2, 2)


def test_generate_correlogram_cached_supports_exp_weighted_matrices(raw_json):
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
        True,
        0.94,
    )

    assert result is not None
    assert result["corr_matrix"].shape == (2, 2)
    assert result["cov_matrix"].shape == (2, 2)


def test_generate_correlogram_cached_supports_ledoit_wolf_matrices(raw_json):
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
        False,
        63.0,
        "ledoit_wolf",
    )

    assert result is not None
    assert result["corr_matrix"].shape == (2, 2)
    assert result["cov_matrix"].shape == (2, 2)


def test_generate_correlogram_cached_supports_constant_correlation_target(raw_json):
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
        False,
        63.0,
        "ledoit_wolf",
        "constant_correlation",
    )

    assert result is not None
    assert result["corr_matrix"].shape == (2, 2)
    assert result["cov_matrix"].shape == (2, 2)


def test_generate_correlogram_cached_supports_oas_matrices(raw_json):
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
        False,
        63.0,
        "oas",
    )

    assert result is not None
    assert result["corr_matrix"].shape == (2, 2)
    assert result["cov_matrix"].shape == (2, 2)


def test_generate_pca_cached_correlation_basis_returns_variance_and_loadings(raw_json):
    result = generate_pca_cached(
        raw_json,
        "daily",
        ("Asset_A", "Asset_B", "Asset_C"),
        "total",
        mapping_payload_for_cache({}),
        mapping_payload_for_cache({}),
        date_range_payload_for_cache(None),
        0,
        mapping_payload_for_cache({}),
        "correlation",
    )

    assert result is not None
    explained = result["explained_variance"]
    loadings = result["loadings"]
    assert list(explained.columns) == [
        "Component",
        "Explained Variance",
        "Explained Variance Ratio",
        "Cumulative Variance Ratio",
    ]
    assert loadings.shape == (3, 3)
    assert explained["Explained Variance Ratio"].sum() == pytest.approx(1.0)


def test_generate_pca_cached_covariance_basis_differs_with_unequal_vol(sample_returns_df):
    scaled_df = sample_returns_df[["Asset_A", "Asset_B", "Asset_C"]].copy()
    scaled_df["Asset_C"] = scaled_df["Asset_C"] * 10
    raw_json = df_to_json(scaled_df)

    corr_result = generate_pca_cached(
        raw_json,
        "daily",
        tuple(scaled_df.columns),
        "total",
        mapping_payload_for_cache({}),
        mapping_payload_for_cache({}),
        date_range_payload_for_cache(None),
        0,
        mapping_payload_for_cache({}),
        "correlation",
    )
    cov_result = generate_pca_cached(
        raw_json,
        "daily",
        tuple(scaled_df.columns),
        "total",
        mapping_payload_for_cache({}),
        mapping_payload_for_cache({}),
        date_range_payload_for_cache(None),
        0,
        mapping_payload_for_cache({}),
        "covariance",
    )

    assert corr_result is not None
    assert cov_result is not None
    assert corr_result["explained_variance"]["Explained Variance Ratio"].iloc[0] != pytest.approx(
        cov_result["explained_variance"]["Explained Variance Ratio"].iloc[0]
    )


def test_generate_pca_cached_handles_insufficient_overlap(sample_returns_df):
    sparse = sample_returns_df[["Asset_A", "Asset_B"]].copy()
    sparse.loc[sparse.index[1:], "Asset_A"] = pd.NA
    sparse.loc[sparse.index[:-1], "Asset_B"] = pd.NA

    result = generate_pca_cached(
        df_to_json(sparse),
        "daily",
        ("Asset_A", "Asset_B"),
        "total",
        mapping_payload_for_cache({}),
        mapping_payload_for_cache({}),
        date_range_payload_for_cache(None),
        0,
        mapping_payload_for_cache({}),
        "correlation",
    )

    assert result is None


def test_generate_pca_cached_sign_normalization_is_stable(raw_json):
    result = generate_pca_cached(
        raw_json,
        "daily",
        ("Asset_A", "Asset_B", "Asset_C"),
        "total",
        mapping_payload_for_cache({}),
        mapping_payload_for_cache({}),
        date_range_payload_for_cache(None),
        0,
        mapping_payload_for_cache({}),
        "correlation",
    )

    loadings = result["loadings"]
    for component in loadings.columns:
        anchor = loadings[component].abs().idxmax()
        assert loadings.loc[anchor, component] >= 0
