"""Benchmark analyticstool compute-path latency (cold vs warm cache).

Run:
    conda run -n dashmat python tools/benchmark_analyticstool_outputs.py
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import pandas as pd
from flask import Flask

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cache_config
from utils.returns import calculate_excess_returns, calculate_rolling_returns, df_to_json
from utils.serialization import date_range_payload_for_cache, mapping_payload_for_cache
from utils.statistics import calculate_statistics_cached, generate_correlogram_cached


def _build_raw_data(n_days: int = 2800, n_assets: int = 20) -> str:
    rng = np.random.default_rng(2026)
    dates = pd.date_range("2015-01-01", periods=n_days, freq="B")
    cols = [f"Series_{i:02d}" for i in range(1, n_assets + 1)]
    data = rng.normal(0.0002, 0.012, size=(len(dates), len(cols)))
    df = pd.DataFrame(data, index=dates, columns=cols)
    df.index.name = "Date"
    return df_to_json(df)


def _time_ms(fn, repeats: int = 5, warmups: int = 1, clear_cache: bool = False) -> float:
    for _ in range(warmups):
        if clear_cache:
            cache_config.cache.clear()
        fn()

    samples = []
    for _ in range(repeats):
        if clear_cache:
            cache_config.cache.clear()
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


def main() -> None:
    init_app = Flask("dashmat-analyticstool-benchmark")
    cache_config.init_cache(init_app)
    cache_config.cache.clear()

    raw_json = _build_raw_data()
    raw_df = pd.read_json(StringIO(raw_json), orient="split")
    selected = tuple(raw_df.columns[:10])

    bench_payload = mapping_payload_for_cache(
        {name: raw_df.columns[(idx + 10) % len(raw_df.columns)] for idx, name in enumerate(selected)}
    )
    ls_payload = mapping_payload_for_cache({name: False for name in selected})
    vol_scaling_payload = mapping_payload_for_cache({name: True for name in selected})
    date_payload = date_range_payload_for_cache({"start": "2018-01-01", "end": "2025-12-31"})

    scenarios = [
        (
            "returns_grid",
            lambda: calculate_excess_returns(
                raw_json,
                "daily_trading",
                selected,
                bench_payload,
                "total",
                ls_payload,
                date_payload,
                10.0,
                vol_scaling_payload,
            ),
        ),
        (
            "statistics_grid",
            lambda: calculate_statistics_cached(
                raw_json,
                "daily_trading",
                selected,
                bench_payload,
                ls_payload,
                date_payload,
                10.0,
                vol_scaling_payload,
                "",
                "",
            ),
        ),
        (
            "rolling_1y_total_return",
            lambda: calculate_rolling_returns(
                raw_json,
                "daily_trading",
                selected,
                "total",
                bench_payload,
                ls_payload,
                date_payload,
                "1y",
                "annualized",
                "total_return",
                10.0,
                vol_scaling_payload,
            ),
        ),
        (
            "correlogram",
            lambda: generate_correlogram_cached(
                raw_json,
                "daily_trading",
                selected,
                "total",
                bench_payload,
                ls_payload,
                date_payload,
                10.0,
                vol_scaling_payload,
            ),
        ),
    ]

    print("== Analyticstool Output Benchmark ==")
    print("dataset=2800 business days, 20 series, 10 selected")
    for label, fn in scenarios:
        cold_ms = _time_ms(fn, repeats=3, warmups=0, clear_cache=True)
        warm_ms = _time_ms(fn, repeats=5, warmups=1, clear_cache=False)
        print(f"{label}: cold_ms={cold_ms:.2f} warm_ms={warm_ms:.2f}")


if __name__ == "__main__":
    main()

