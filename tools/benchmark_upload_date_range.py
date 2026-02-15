"""Benchmark upload-merge and date-range helper latency.

Run:
    conda run -n dashmat python tools/benchmark_upload_date_range.py
"""

from __future__ import annotations

import base64
from io import BytesIO
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
from utils.date_range_flow import compute_date_range_candidates
from utils.upload_flow import import_selected_workbook_sheets


def _build_upload_payload(n_rows: int = 1800, n_series: int = 8) -> tuple[str, str]:
    rng = np.random.default_rng(2026)
    dates = pd.date_range("2018-01-01", periods=n_rows, freq="B")
    cols = [f"Series_{i:02d}" for i in range(1, n_series + 1)]

    sheet_frames = []
    for sheet_idx in range(1, 6):
        data = rng.normal(0.0002, 0.01, size=(len(dates), len(cols)))
        frame = pd.DataFrame(data, index=dates, columns=cols)
        frame.index.name = "Date"
        # Create overlap windows where later sheets overwrite a subset.
        start = (sheet_idx - 1) * 120
        end = min(start + 800, len(frame))
        frame = frame.iloc[start:end]
        sheet_frames.append((f"S{sheet_idx}", frame))

    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        for sheet_name, frame in sheet_frames:
            frame.reset_index().to_excel(writer, sheet_name=sheet_name, index=False)

    encoded = base64.b64encode(bio.getvalue()).decode("ascii")
    payload = (
        "data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,"
        f"{encoded}"
    )
    return payload, "benchmark_multi.xlsx"


def _time_ms(fn, repeats: int = 7, warmups: int = 1) -> float:
    for _ in range(warmups):
        fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


def main() -> None:
    payload, filename = _build_upload_payload()
    selected = ["S1", "S2", "S3", "S4", "S5"]

    init_app = Flask("dashmat-upload-benchmark")
    cache_config.init_cache(init_app)
    cache_config.cache.clear()

    upload_ms = _time_ms(
        lambda: import_selected_workbook_sheets(payload, filename, selected),
        repeats=5,
        warmups=1,
    )

    merged_df, _ = import_selected_workbook_sheets(payload, filename, selected)
    raw_json = merged_df.to_json(orient="split", date_format="iso")
    series = tuple(merged_df.columns[:5])

    cache_config.cache.clear()
    first_range_ms = _time_ms(
        lambda: compute_date_range_candidates(raw_json, "daily_trading", series),
        repeats=1,
        warmups=0,
    )
    cached_range_ms = _time_ms(
        lambda: compute_date_range_candidates(raw_json, "daily_trading", series),
        repeats=7,
        warmups=1,
    )

    print("== Upload + Date Range Benchmark ==")
    print(f"upload.multi_sheet_import.median_ms={upload_ms:.2f}")
    print(f"date_range.first_call.ms={first_range_ms:.2f}")
    print(f"date_range.cached_median_ms={cached_range_ms:.2f}")


if __name__ == "__main__":
    main()
