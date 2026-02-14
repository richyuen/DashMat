from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pandas as pd

from tools.data import generate_test_data as gtd


def test_download_close_series_handles_multiindex(monkeypatch):
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    cols = pd.MultiIndex.from_tuples([("Close", "^GSPC"), ("Open", "^GSPC")])
    mock_df = pd.DataFrame(
        [[100, 99], [101, 100], [102, 101], [103, 102], [104, 103]],
        index=idx,
        columns=cols,
    )

    monkeypatch.setattr(gtd.yf, "download", lambda *args, **kwargs: mock_df)

    series = gtd._download_close_series("^GSPC", "SPX")
    assert series.name == "SPX"
    assert len(series) == len(idx)
    assert series.iloc[-1] == 104


def test_build_returns_with_mocked_download(monkeypatch):
    monkeypatch.setattr(
        gtd,
        "BENCHMARKS",
        {
            "AAA": {"symbol": "AAA", "actual": True},
            "BBB": {"symbol": "BBB", "actual": False},
        },
    )

    idx = pd.date_range("2020-01-01", periods=40, freq="B")

    def _fake_download(_symbol: str, label: str) -> pd.Series:
        values = pd.Series(range(100, 100 + len(idx)), index=idx, dtype=float)
        values.name = label
        return values

    monkeypatch.setattr(gtd, "_download_close_series", _fake_download)

    daily, monthly = gtd.build_returns()

    assert list(daily.columns) == ["AAA", "BBB"]
    assert not daily.empty
    assert not monthly.empty
    assert monthly.index.is_month_end.all()


def test_save_excel_writes_file():
    df = pd.DataFrame({"A": [0.01, 0.02]}, index=pd.to_datetime(["2024-01-01", "2024-01-02"]))
    tmp_dir = Path("tests/.tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    output = tmp_dir / f"out_{uuid4().hex}.xlsx"

    gtd._save_excel(df, output)

    assert output.exists()
    loaded = pd.read_excel(output, index_col=0)
    assert loaded.shape == (2, 1)


def test_main_saves_daily_and_monthly_files(monkeypatch):
    daily = pd.DataFrame({"A": [0.01]}, index=pd.to_datetime(["2024-01-01"]))
    monthly = pd.DataFrame({"A": [0.02]}, index=pd.to_datetime(["2024-01-31"]))
    monkeypatch.setattr(gtd, "build_returns", lambda: (daily, monthly))
    tmp_dir = Path("tests/.tmp") / f"main_{uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(gtd, "SAMPLE_DATA_DIR", tmp_dir)

    written_paths: list[Path] = []

    def _fake_save(_df: pd.DataFrame, path: Path) -> None:
        written_paths.append(path)

    monkeypatch.setattr(gtd, "_save_excel", _fake_save)

    gtd.main()

    assert len(written_paths) == 2
    assert any(path.name == gtd.SAMPLE_DAILY_FILE for path in written_paths)
    assert any(path.name == gtd.SAMPLE_MONTHLY_FILE for path in written_paths)
