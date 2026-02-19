"""Generate sample daily/monthly benchmark return files.

This script is intentionally lightweight and testable:
- download close history for configured benchmarks
- compute daily arithmetic returns
- derive monthly compounded returns
- write both to Excel files under sample_data/
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    import yfinance as yf  # type: ignore
except Exception:  # pragma: no cover - validated in tests via monkeypatch
    yf = None


ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DATA_DIR = ROOT / "sample_data"
SAMPLE_DAILY_FILE = "SampleMstar.xlsx"
SAMPLE_MONTHLY_FILE = "SampleMstarMonthly.xlsx"

# Keep the default scope aligned with modern history availability.
START_DATE = "2015-01-01"
END_DATE = None

BENCHMARKS: dict[str, dict[str, object]] = {
    "SPX": {"symbol": "^GSPC", "actual": True},
    "EAFE": {"symbol": "EFA", "actual": True},
    "BCAGG": {"symbol": "AGG", "actual": True},
}


def _extract_close(df: pd.DataFrame) -> pd.Series:
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance can return two-level columns like ("Close", "^GSPC").
        if "Close" in df.columns.get_level_values(0):
            close = df["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            return pd.to_numeric(close, errors="coerce")
    if "Close" in df.columns:
        return pd.to_numeric(df["Close"], errors="coerce")
    raise ValueError("Close column not found in downloaded history")


def _download_close_series(symbol: str, label: str) -> pd.Series:
    if yf is None:
        raise ImportError("yfinance is required to download benchmark history")

    history = yf.download(
        symbol,
        start=START_DATE,
        end=END_DATE,
        progress=False,
        auto_adjust=True,
    )
    if history is None or history.empty:
        raise ValueError(f"No history returned for symbol `{symbol}`")

    close = _extract_close(history).dropna()
    if close.empty:
        raise ValueError(f"No close values returned for symbol `{symbol}`")

    close.index = pd.to_datetime(close.index)
    close = close.sort_index()
    close = close.loc[close.index >= pd.Timestamp(START_DATE)]
    if close.empty:
        raise ValueError(f"No data in target range for symbol `{symbol}`")

    close.name = label
    return close


def build_returns() -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_returns: dict[str, pd.Series] = {}
    for bench, meta in BENCHMARKS.items():
        symbol = str(meta.get("symbol", bench))
        close = _download_close_series(symbol, bench)
        ret = close.pct_change(fill_method=None).dropna()
        ret.name = bench
        daily_returns[bench] = ret

    if not daily_returns:
        raise ValueError("No benchmark return series were generated")

    daily_df = pd.concat(daily_returns.values(), axis=1)
    ordered_cols = [k for k in BENCHMARKS.keys() if k in daily_df.columns]
    daily_df = daily_df.reindex(columns=ordered_cols).sort_index()

    monthly_df = daily_df.resample("ME").apply(lambda vals: (1.0 + vals).prod() - 1.0)
    monthly_df.index = pd.DatetimeIndex(monthly_df.index).to_period("M").to_timestamp("M")
    monthly_df = monthly_df.sort_index()

    return daily_df, monthly_df


def _save_excel(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    out.index.name = "Date"
    out.to_excel(path)


def main() -> None:
    daily_df, monthly_df = build_returns()
    _save_excel(daily_df, SAMPLE_DATA_DIR / SAMPLE_DAILY_FILE)
    _save_excel(monthly_df, SAMPLE_DATA_DIR / SAMPLE_MONTHLY_FILE)


if __name__ == "__main__":
    main()

