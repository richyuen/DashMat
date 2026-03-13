from __future__ import annotations

import pandas as pd
import pytest

from utils.regime_analysis import (
    build_regime_detail_frame,
    build_regime_duration_table,
    build_regime_statistics_table,
    build_regime_timeline_frame,
    build_regime_transition_matrix,
    build_wide_detail_frame,
    compute_regime_assignments,
    prepare_regime_input_frame,
)


def _single_series_quantile_definition() -> dict:
    return {
        "RegimeName": "QuantileSingle",
        "MethodType": 3,
        "Config": {
            "num_regimes": 3,
            "return_basis": "total",
            "min_observations": 40,
            "single_series": "Asset_A",
            "vol_scaler": 0.0,
            "benchmark_assignments": {},
            "long_short_assignments": {},
            "vol_scaling_assignments": {},
        },
    }


def test_compute_regime_assignments_quantile_single_series(raw_json):
    definition = _single_series_quantile_definition()
    states, diagnostics = compute_regime_assignments(
        raw_data=raw_json,
        periodicity="daily",
        definition=definition,
        date_range={"start": "2023-01-02", "end": "2024-12-31"},
    )
    assert not states.empty
    assert states.name == "Regime"
    assert int(states.min()) >= 1
    assert int(states.max()) <= 3
    assert diagnostics.get("num_regimes") == 3


def test_regime_output_tables_build_with_valid_states(raw_json):
    definition = _single_series_quantile_definition()
    states, _ = compute_regime_assignments(
        raw_data=raw_json,
        periodicity="daily",
        definition=definition,
        date_range={"start": "2023-01-02", "end": "2024-12-31"},
    )
    returns_df = prepare_regime_input_frame(
        raw_data=raw_json,
        periodicity="daily",
        selected_series=["Asset_A", "Asset_B"],
        return_basis="total",
        benchmark_assignments={},
        long_short_assignments={},
        date_range={"start": "2023-01-02", "end": "2024-12-31"},
        vol_scaler=0,
        vol_scaling_assignments={},
    )

    stats_df = build_regime_statistics_table(returns_df, states, "daily")
    transition_df = build_regime_transition_matrix(states)
    duration_df = build_regime_duration_table(states)
    timeline_df = build_regime_timeline_frame(states)

    assert not stats_df.empty
    assert {"Regime", "Series", "Observations"}.issubset(set(stats_df.columns))
    assert "Growth End" in stats_df.columns
    assert isinstance(transition_df, pd.DataFrame)
    assert not transition_df.empty
    assert not duration_df.empty
    assert not timeline_df.empty


def test_regime_statistics_use_common_overlap_and_relative_metrics():
    idx_returns = pd.date_range("2024-01-01", periods=6, freq="D")
    returns_df = pd.DataFrame(
        {
            "Asset_A": [0.010, 0.015, 0.012, 0.008, -0.004, 0.009],
            "Bench_A": [0.007, 0.011, 0.010, 0.006, -0.002, 0.008],
        },
        index=idx_returns,
    )
    states = pd.Series(
        [1, 2, 2],
        index=pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-05"]),
        dtype="Int64",
        name="Regime",
    )

    stats_df = build_regime_statistics_table(
        returns_df,
        states,
        "daily",
        selected_series=["Asset_A"],
        benchmark_assignments={"Asset_A": "Bench_A"},
        long_short_assignments={},
    )

    assert not stats_df.empty
    assert set(stats_df["Series"]) == {"Asset_A"}
    assert int(stats_df["Observations"].sum()) == 3
    assert "Growth End" in stats_df.columns
    assert stats_df["Growth End"].notna().all()
    regime_one_growth = float(stats_df.loc[stats_df["Regime"] == 1, "Growth End"].iloc[0])
    assert regime_one_growth == pytest.approx(1.012, rel=1e-9)
    for col in [
        "Annualized Excess Return",
        "Annualized Tracking Error",
        "Information Ratio",
        "Correlation",
        "Hit Rate (vs Benchmark)",
    ]:
        assert col in stats_df.columns
    # Regime-level relative metrics should be computed when benchmark data is present.
    assert stats_df["Annualized Tracking Error"].notna().any()
    assert stats_df["Hit Rate (vs Benchmark)"].notna().any()


def test_regime_statistics_empty_when_no_common_overlap():
    returns_df = pd.DataFrame(
        {"Asset_A": [0.01, 0.02, -0.01]},
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )
    states = pd.Series(
        [1, 2, 1],
        index=pd.date_range("2024-02-01", periods=3, freq="D"),
        dtype="Int64",
        name="Regime",
    )

    stats_df = build_regime_statistics_table(
        returns_df,
        states,
        "daily",
        selected_series=["Asset_A"],
    )
    assert stats_df.empty


def test_build_regime_detail_frame_uses_common_overlap_and_signal_label():
    idx_returns = pd.date_range("2024-01-01", periods=4, freq="D")
    returns_df = pd.DataFrame({"Asset_A": [0.01, 0.02, -0.01, 0.0]}, index=idx_returns)
    states = pd.Series([1, 2, 2], index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]), dtype="Int64", name="Regime")
    signal = pd.Series([0.5, -0.1, 0.2], index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]), name="PC1")

    detail_df = build_regime_detail_frame(returns_df, states, signal, "Regime Signal")

    assert list(detail_df.columns[:3]) == ["Date", "Regime", "Regime Signal"]
    assert len(detail_df) == 3
    assert detail_df["Regime"].tolist() == [1, 2, 2]
    assert detail_df["Asset_A"].tolist() == [0.02, -0.01, 0.0]


def test_build_regime_detail_frame_avoids_signal_name_collision_with_series_column():
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    returns_df = pd.DataFrame({"SPX_TRIndex": [0.01, -0.02, 0.03]}, index=idx)
    states = pd.Series([1, 2, 1], index=idx, dtype="Int64", name="Regime")
    signal = pd.Series([100.0, 99.5, 100.5], index=idx, name="SPX_TRIndex")

    detail_df = build_regime_detail_frame(returns_df, states, signal, "SPX_TRIndex")

    assert list(detail_df.columns[:4]) == ["Date", "Regime", "Regime Signal", "SPX_TRIndex"]
    assert detail_df["Regime Signal"].tolist() == [100.0, 99.5, 100.5]
    assert detail_df["SPX_TRIndex"].tolist() == [0.01, -0.02, 0.03]


def test_build_wide_detail_frame_normalizes_overlap_and_drops_all_missing_series_rows():
    idx = pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-02", "2024-01-02"])
    value_frame = pd.DataFrame(
        {
            "Asset_A": [0.03, 0.01, None, 0.02],
            "Asset_B": [None, None, None, None],
        },
        index=idx,
    )
    metadata = [
        ("Quantile", pd.Series(["Q3", "Q1", "Q2"], index=pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-02"]))),
        ("Regime", pd.Series([2, 1, 1], index=pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-02"]))),
    ]

    detail_df = build_wide_detail_frame(
        value_frame,
        metadata,
        value_columns=["Asset_A", "Asset_B"],
        int_columns={"Regime"},
    )

    assert list(detail_df.columns[:4]) == ["Date", "Quantile", "Regime", "Asset_A"]
    assert detail_df["Date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-01", "2024-01-02", "2024-01-03"]
    assert detail_df["Quantile"].tolist() == ["Q1", "Q2", "Q3"]
    assert detail_df["Regime"].tolist() == [1, 1, 2]
    assert detail_df.loc[detail_df["Date"] == pd.Timestamp("2024-01-02"), "Asset_A"].iloc[0] == pytest.approx(0.02)


def test_build_wide_detail_frame_accepts_aligned_inputs_without_normalizing():
    idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    value_frame = pd.DataFrame({"Asset_A": [0.01, None, 0.03], "Asset_B": [0.02, None, 0.04]}, index=idx)

    detail_df = build_wide_detail_frame(
        value_frame,
        [
            ("Lookback", pd.Series("1M", index=idx, dtype="object")),
            ("Condition Met", pd.Series([True, False, True], index=idx, dtype=bool)),
        ],
        value_columns=["Asset_A", "Asset_B"],
        inputs_aligned=True,
    )

    assert list(detail_df.columns[:3]) == ["Date", "Lookback", "Condition Met"]
    assert detail_df["Date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-01", "2024-01-03"]
    assert detail_df["Lookback"].tolist() == ["1M", "1M"]
    assert detail_df["Condition Met"].tolist() == [True, True]


def test_compute_regime_assignments_returns_warning_for_invalid_definition(raw_json):
    states, diagnostics = compute_regime_assignments(
        raw_data=raw_json,
        periodicity="daily",
        definition={"RegimeName": "Broken", "MethodType": 1, "Config": {"num_regimes": 3}},
        date_range={"start": "2023-01-02", "end": "2024-12-31"},
    )
    assert states.empty
    assert "warning" in diagnostics
