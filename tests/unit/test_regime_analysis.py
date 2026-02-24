from __future__ import annotations

import pandas as pd

from utils.regime_analysis import (
    build_regime_conditioned_summary,
    build_regime_duration_table,
    build_regime_statistics_table,
    build_regime_timeline_frame,
    build_regime_transition_matrix,
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
    conditioned_df = build_regime_conditioned_summary(returns_df, states)

    assert not stats_df.empty
    assert {"Regime", "Series", "Observations"}.issubset(set(stats_df.columns))
    assert isinstance(transition_df, pd.DataFrame)
    assert not transition_df.empty
    assert not duration_df.empty
    assert not timeline_df.empty
    assert not conditioned_df.empty


def test_compute_regime_assignments_returns_warning_for_invalid_definition(raw_json):
    states, diagnostics = compute_regime_assignments(
        raw_data=raw_json,
        periodicity="daily",
        definition={"RegimeName": "Broken", "MethodType": 1, "Config": {"num_regimes": 3}},
        date_range={"start": "2023-01-02", "end": "2024-12-31"},
    )
    assert states.empty
    assert "warning" in diagnostics
