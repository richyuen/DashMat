from __future__ import annotations

from io import StringIO

import numpy as np
import pandas as pd

from utils.serialization import date_range_payload_for_cache, mapping_payload_for_cache
import tools.benchmark_callback_latency as bench


def test_build_raw_data_returns_valid_json_payload():
    raw_json = bench._build_raw_data(n_days=128, n_assets=6)
    df = pd.read_json(StringIO(raw_json), orient="split")
    assert df.shape == (128, 6)


def test_compute_monthly_attribution_fast_matches_manual_aggregation():
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    working_df = pd.DataFrame(
        {
            "A": np.full(len(idx), 0.01),
            "B": np.full(len(idx), 0.02),
        },
        index=idx,
    )
    working_df.index.name = "Date"
    window_weights = [
        {
            "apply_start": "2024-01-01",
            "apply_end": "2024-01-05",
            "weights": {"A": 0.6, "B": 0.4},
        },
        {
            "apply_start": "2024-01-06",
            "apply_end": "2024-01-10",
            "weights": {"A": 0.3, "B": 0.7},
        },
    ]

    fast = bench._compute_monthly_attribution_fast(working_df, ["A", "B"], window_weights)

    weights_df = pd.DataFrame(0.0, index=idx, columns=["A", "B"])
    weights_df.loc["2024-01-01":"2024-01-05", :] = [0.6, 0.4]
    weights_df.loc["2024-01-06":"2024-01-10", :] = [0.3, 0.7]
    manual = (weights_df * working_df).resample("ME").sum()

    pd.testing.assert_frame_equal(fast, manual)


def test_optimized_attribution_loop_matches_new_loop_output_count():
    raw_json = bench._build_raw_data(n_days=600, n_assets=12)
    raw_df = pd.read_json(StringIO(raw_json), orient="split")
    raw_df.index = pd.to_datetime(raw_df.index)
    index = raw_df.index

    results = bench._build_results_sets(index, overlap=True)
    periodicity = "daily"
    bench_payload = mapping_payload_for_cache({})
    ls_payload = mapping_payload_for_cache({})
    date_payload = date_range_payload_for_cache(None)
    vol_scaler = 0.0
    vol_scaling_payload = mapping_payload_for_cache({})

    produced_new = bench._run_new_attribution_loop(
        results,
        raw_json,
        periodicity,
        bench_payload,
        ls_payload,
        date_payload,
        vol_scaler,
        vol_scaling_payload,
    )
    produced_optimized = bench._run_optimized_attribution_loop(
        results,
        raw_json,
        periodicity,
        bench_payload,
        ls_payload,
        date_payload,
        vol_scaler,
        vol_scaling_payload,
    )

    assert produced_optimized == produced_new
