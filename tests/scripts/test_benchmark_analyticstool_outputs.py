from __future__ import annotations

from io import StringIO

import pandas as pd

import tools.benchmark_analyticstool_outputs as bench


def test_build_raw_data_returns_valid_json_payload():
    raw_json = bench._build_raw_data(n_days=128, n_assets=6)
    df = pd.read_json(StringIO(raw_json), orient="split")
    assert df.shape == (128, 6)


def test_time_ms_returns_float():
    elapsed = bench._time_ms(lambda: 1, repeats=2, warmups=1, clear_cache=False)
    assert isinstance(elapsed, float)
    assert elapsed >= 0.0

