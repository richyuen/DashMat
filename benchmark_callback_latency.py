"""Benchmark callback-level latency deltas for common portfolio workflows.

Run:
    conda run -n dashmat python benchmark_callback_latency.py
"""

from __future__ import annotations

import hashlib
from io import StringIO
import itertools
import statistics
import time
from typing import Callable

import numpy as np
import pandas as pd
from flask import Flask

import cache_config
from utils.returns import df_to_json, get_working_returns
from utils.serialization import date_range_payload_for_cache, mapping_payload_for_cache


def _build_raw_data(n_days: int = 1800, n_assets: int = 12) -> str:
    rng = np.random.default_rng(1234)
    dates = pd.date_range("2018-01-01", periods=n_days, freq="B")
    cols = [f"Asset_{i:02d}" for i in range(1, n_assets + 1)]
    data = rng.normal(0.0002, 0.01, size=(len(dates), len(cols)))
    df = pd.DataFrame(data, index=dates, columns=cols)
    df.index.name = "Date"
    return df_to_json(df)


def _build_window_weights(index: pd.DatetimeIndex, selected_series: list[str], n_windows: int = 16):
    points = np.linspace(0, len(index) - 1, n_windows + 1, dtype=int)
    windows = []
    seed_payload = "|".join(selected_series) + f"|{n_windows}"
    seed = int(hashlib.md5(seed_payload.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)

    for i in range(n_windows):
        s_idx = int(points[i])
        e_idx = int(points[i + 1])
        if e_idx < s_idx:
            continue
        start = index[s_idx]
        end = index[e_idx]
        w = rng.dirichlet(np.ones(len(selected_series)))
        weights = {name: float(w[j]) for j, name in enumerate(selected_series)}
        windows.append(
            {
                "apply_start": start.strftime("%Y-%m-%d"),
                "apply_end": end.strftime("%Y-%m-%d"),
                "weights": weights,
                "est_start": start.strftime("%Y-%m-%d"),
                "est_end": end.strftime("%Y-%m-%d"),
            }
        )
    return windows


def _build_results_sets(index: pd.DatetimeIndex, overlap: bool) -> dict:
    base_sets = [
        ["Asset_01", "Asset_02", "Asset_03", "Asset_04", "Asset_05"],
        ["Asset_02", "Asset_04", "Asset_06", "Asset_08", "Asset_10"],
        ["Asset_03", "Asset_05", "Asset_07", "Asset_09", "Asset_11"],
        ["Asset_01", "Asset_06", "Asset_07", "Asset_11", "Asset_12"],
    ]

    rng = np.random.default_rng(222 if overlap else 333)
    unique_sets: list[list[str]] = []
    if not overlap:
        population = [f"Asset_{j:02d}" for j in range(1, 13)]
        all_sets = list(itertools.combinations(population, 5))
        rng.shuffle(all_sets)
        unique_sets = [list(combo) for combo in all_sets[:24]]

    results = {}
    for i in range(24):
        if overlap:
            series = base_sets[i % len(base_sets)]
        else:
            series = unique_sets[i]
        pname = f"Portfolio_{i+1:02d}"
        results[pname] = {
            "config": {"selected_series": series},
            "window_weights": _build_window_weights(index, series),
        }
    return results


def _run_old_attribution_loop(
    results: dict,
    raw_json: str,
    periodicity: str,
    bench_payload: str,
    ls_payload: str,
    date_payload: str,
    vol_scaler: float,
    vol_scaling_payload: str,
) -> int:
    produced = 0
    for pdata in results.values():
        config = pdata.get("config", {})
        opt_series = config.get("selected_series", [])
        window_weights = pdata.get("window_weights", [])
        if not opt_series or not window_weights:
            continue

        working_df = get_working_returns(
            raw_json,
            periodicity,
            tuple(opt_series),
            bench_payload,
            ls_payload,
            date_payload,
            vol_scaler,
            vol_scaling_payload,
        )
        if working_df.empty:
            continue

        weights_df = pd.DataFrame(0.0, index=working_df.index, columns=opt_series)
        for ww in window_weights:
            start = pd.Timestamp(ww["apply_start"])
            end = pd.Timestamp(ww["apply_end"])
            mask = (weights_df.index >= start) & (weights_df.index <= end)
            for s in opt_series:
                weights_df.loc[mask, s] = ww["weights"].get(s, 0)

        attribution = weights_df * working_df[opt_series].fillna(0)
        attribution_monthly = attribution.resample("ME").sum().dropna(how="all")
        if not attribution_monthly.empty:
            produced += 1
    return produced


def _run_new_attribution_loop(
    results: dict,
    raw_json: str,
    periodicity: str,
    bench_payload: str,
    ls_payload: str,
    date_payload: str,
    vol_scaler: float,
    vol_scaling_payload: str,
) -> int:
    produced = 0
    working_df_cache: dict[tuple, pd.DataFrame] = {}

    for pdata in results.values():
        config = pdata.get("config", {})
        opt_series = config.get("selected_series", [])
        window_weights = pdata.get("window_weights", [])
        if not opt_series or not window_weights:
            continue

        series_key = tuple(opt_series)
        working_df = working_df_cache.get(series_key)
        if working_df is None:
            working_df = get_working_returns(
                raw_json,
                periodicity,
                series_key,
                bench_payload,
                ls_payload,
                date_payload,
                vol_scaler,
                vol_scaling_payload,
            )
            working_df_cache[series_key] = working_df
        if working_df.empty:
            continue

        weights_df = pd.DataFrame(0.0, index=working_df.index, columns=opt_series)
        for ww in window_weights:
            start = pd.Timestamp(ww["apply_start"])
            end = pd.Timestamp(ww["apply_end"])
            mask = (weights_df.index >= start) & (weights_df.index <= end)
            for s in opt_series:
                weights_df.loc[mask, s] = ww["weights"].get(s, 0)

        attribution = weights_df * working_df[opt_series].fillna(0)
        attribution_monthly = attribution.resample("ME").sum().dropna(how="all")
        if not attribution_monthly.empty:
            produced += 1
    return produced


def _compute_monthly_attribution_fast(
    working_df: pd.DataFrame,
    opt_series,
    window_weights,
) -> pd.DataFrame:
    series_tuple = tuple(opt_series or ())
    if working_df.empty or not series_tuple or not window_weights:
        return pd.DataFrame()

    working_subset = working_df.loc[:, list(series_tuple)].fillna(0.0)
    index = working_subset.index
    n_rows = len(index)
    n_cols = len(series_tuple)
    weights = np.zeros((n_rows, n_cols), dtype=float)

    for ww in window_weights:
        start = pd.Timestamp(ww["apply_start"])
        end = pd.Timestamp(ww["apply_end"])
        s_idx = int(index.searchsorted(start, side="left"))
        e_idx = int(index.searchsorted(end, side="right"))
        if s_idx >= e_idx:
            continue
        row_weights = np.fromiter(
            (float(ww["weights"].get(name, 0.0) or 0.0) for name in series_tuple),
            dtype=float,
            count=n_cols,
        )
        weights[s_idx:e_idx, :] = row_weights

    has_weights = weights.sum(axis=1) > 0
    if not np.any(has_weights):
        return pd.DataFrame()

    attribution_values = weights[has_weights] * working_subset.to_numpy(copy=False)[has_weights]
    attribution_df = pd.DataFrame(
        attribution_values,
        index=working_subset.index[has_weights],
        columns=list(series_tuple),
    )
    return attribution_df.resample("ME").sum().dropna(how="all")


def _run_optimized_attribution_loop(
    results: dict,
    raw_json: str,
    periodicity: str,
    bench_payload: str,
    ls_payload: str,
    date_payload: str,
    vol_scaler: float,
    vol_scaling_payload: str,
) -> int:
    produced = 0
    working_df_cache: dict[tuple, pd.DataFrame] = {}

    for pdata in results.values():
        config = pdata.get("config", {})
        opt_series = config.get("selected_series", [])
        window_weights = pdata.get("window_weights", [])
        if not opt_series or not window_weights:
            continue

        series_key = tuple(opt_series)
        working_df = working_df_cache.get(series_key)
        if working_df is None:
            working_df = get_working_returns(
                raw_json,
                periodicity,
                series_key,
                bench_payload,
                ls_payload,
                date_payload,
                vol_scaler,
                vol_scaling_payload,
            )
            working_df_cache[series_key] = working_df
        if working_df.empty:
            continue

        attribution_monthly = _compute_monthly_attribution_fast(working_df, opt_series, window_weights)
        if not attribution_monthly.empty:
            produced += 1
    return produced


def _time_ms(fn: Callable[[], int], repeats: int = 5, warmups: int = 1, clear_cache: bool = False) -> float:
    for _ in range(warmups):
        if clear_cache:
            cache_config.cache.clear()
        fn()

    samples = []
    for _ in range(repeats):
        if clear_cache:
            cache_config.cache.clear()
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples)


def _print_result(label: str, before_ms: float, after_ms: float) -> None:
    delta_ms = before_ms - after_ms
    pct = (delta_ms / before_ms * 100.0) if before_ms > 0 else 0.0
    print(f"{label}: before={before_ms:.2f}ms after={after_ms:.2f}ms delta={delta_ms:.2f}ms ({pct:.1f}%)")


def main() -> None:
    raw_json = _build_raw_data()
    raw_df = pd.read_json(StringIO(raw_json), orient="split")
    raw_df.index = pd.to_datetime(raw_df.index)
    index = raw_df.index

    periodicity = "daily"
    bench_payload = mapping_payload_for_cache({})
    ls_payload = mapping_payload_for_cache({})
    date_payload = date_range_payload_for_cache(None)
    vol_scaler = 0.0
    vol_scaling_payload = mapping_payload_for_cache({})

    overlap_results = _build_results_sets(index, overlap=True)
    unique_results = _build_results_sets(index, overlap=False)

    print("== Callback Latency Delta Benchmark ==")
    print("Workflow: portfolio attribution export loop (old vs new local working_df bundle cache)\n")

    # Scenario 1: no global Flask cache initialized (upper-bound improvement).
    old_overlap_no_cache = _time_ms(
        lambda: _run_old_attribution_loop(
            overlap_results,
            raw_json,
            periodicity,
            bench_payload,
            ls_payload,
            date_payload,
            vol_scaler,
            vol_scaling_payload,
        ),
        repeats=5,
        warmups=1,
        clear_cache=False,
    )
    new_overlap_no_cache = _time_ms(
        lambda: _run_new_attribution_loop(
            overlap_results,
            raw_json,
            periodicity,
            bench_payload,
            ls_payload,
            date_payload,
            vol_scaler,
            vol_scaling_payload,
        ),
        repeats=5,
        warmups=1,
        clear_cache=False,
    )
    _print_result("No global cache, overlapping series sets", old_overlap_no_cache, new_overlap_no_cache)

    old_unique_no_cache = _time_ms(
        lambda: _run_old_attribution_loop(
            unique_results,
            raw_json,
            periodicity,
            bench_payload,
            ls_payload,
            date_payload,
            vol_scaler,
            vol_scaling_payload,
        ),
        repeats=5,
        warmups=1,
        clear_cache=False,
    )
    new_unique_no_cache = _time_ms(
        lambda: _run_new_attribution_loop(
            unique_results,
            raw_json,
            periodicity,
            bench_payload,
            ls_payload,
            date_payload,
            vol_scaler,
            vol_scaling_payload,
        ),
        repeats=5,
        warmups=1,
        clear_cache=False,
    )
    _print_result("No global cache, unique series sets", old_unique_no_cache, new_unique_no_cache)

    print("")

    # Scenario 2: global Flask cache initialized; clear per run to isolate callback-local reuse.
    init_cache_called = False
    if not init_cache_called:
        init_cache_called = True
        init_app = Flask("dashmat-benchmark")
        cache_config.init_cache(init_app)

    old_overlap_with_cache = _time_ms(
        lambda: _run_old_attribution_loop(
            overlap_results,
            raw_json,
            periodicity,
            bench_payload,
            ls_payload,
            date_payload,
            vol_scaler,
            vol_scaling_payload,
        ),
        repeats=5,
        warmups=1,
        clear_cache=True,
    )
    new_overlap_with_cache = _time_ms(
        lambda: _run_new_attribution_loop(
            overlap_results,
            raw_json,
            periodicity,
            bench_payload,
            ls_payload,
            date_payload,
            vol_scaler,
            vol_scaling_payload,
        ),
        repeats=5,
        warmups=1,
        clear_cache=True,
    )
    _print_result(
        "Global cache enabled (cold per run), overlapping series sets",
        old_overlap_with_cache,
        new_overlap_with_cache,
    )

    old_unique_with_cache = _time_ms(
        lambda: _run_old_attribution_loop(
            unique_results,
            raw_json,
            periodicity,
            bench_payload,
            ls_payload,
            date_payload,
            vol_scaler,
            vol_scaling_payload,
        ),
        repeats=5,
        warmups=1,
        clear_cache=True,
    )
    new_unique_with_cache = _time_ms(
        lambda: _run_new_attribution_loop(
            unique_results,
            raw_json,
            periodicity,
            bench_payload,
            ls_payload,
            date_payload,
            vol_scaler,
            vol_scaling_payload,
        ),
        repeats=5,
        warmups=1,
        clear_cache=True,
    )
    _print_result(
        "Global cache enabled (cold per run), unique series sets",
        old_unique_with_cache,
        new_unique_with_cache,
    )

    print(
        "\nWorkflow: attribution export loop (callback-local working_df cache vs vectorized window assignment)\n"
    )

    optimized_overlap_no_cache = _time_ms(
        lambda: _run_optimized_attribution_loop(
            overlap_results,
            raw_json,
            periodicity,
            bench_payload,
            ls_payload,
            date_payload,
            vol_scaler,
            vol_scaling_payload,
        ),
        repeats=5,
        warmups=1,
        clear_cache=False,
    )
    _print_result(
        "No global cache, overlapping series sets",
        new_overlap_no_cache,
        optimized_overlap_no_cache,
    )

    optimized_unique_no_cache = _time_ms(
        lambda: _run_optimized_attribution_loop(
            unique_results,
            raw_json,
            periodicity,
            bench_payload,
            ls_payload,
            date_payload,
            vol_scaler,
            vol_scaling_payload,
        ),
        repeats=5,
        warmups=1,
        clear_cache=False,
    )
    _print_result(
        "No global cache, unique series sets",
        new_unique_no_cache,
        optimized_unique_no_cache,
    )

    optimized_overlap_with_cache = _time_ms(
        lambda: _run_optimized_attribution_loop(
            overlap_results,
            raw_json,
            periodicity,
            bench_payload,
            ls_payload,
            date_payload,
            vol_scaler,
            vol_scaling_payload,
        ),
        repeats=5,
        warmups=1,
        clear_cache=True,
    )
    _print_result(
        "Global cache enabled (cold per run), overlapping series sets",
        new_overlap_with_cache,
        optimized_overlap_with_cache,
    )

    optimized_unique_with_cache = _time_ms(
        lambda: _run_optimized_attribution_loop(
            unique_results,
            raw_json,
            periodicity,
            bench_payload,
            ls_payload,
            date_payload,
            vol_scaler,
            vol_scaling_payload,
        ),
        repeats=5,
        warmups=1,
        clear_cache=True,
    )
    _print_result(
        "Global cache enabled (cold per run), unique series sets",
        new_unique_with_cache,
        optimized_unique_with_cache,
    )

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
