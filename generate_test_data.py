"""Download benchmark sample data and save daily/monthly return files.

Outputs:
  - sample_data/benchmark_returns/benchmark_daily_returns_2020_2025.xlsx
  - sample_data/benchmark_returns/benchmark_monthly_returns_2020_2025.xlsx

Source approach:
  - Use actual index symbols where available.
  - Fall back to liquid proxy symbols when index history is not freely exposed.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from utils.sample_data import SAMPLE_DAILY_FILE, SAMPLE_DATA_DIR, SAMPLE_MONTHLY_FILE


START_DATE = "2020-01-01"
END_DATE = "2025-12-31"

# Requested series names as output columns (ticker-only headers).
# `actual=True` means direct benchmark index history is used.
# `actual=False` means fallback proxy due index data-access limitations.
BENCHMARKS: dict[str, dict[str, str | bool]] = {
    "SPX": {"symbol": "^GSPC", "actual": True},
    "RMID": {"symbol": "^MID", "actual": True},
    "R2000": {"symbol": "^RUT", "actual": True},
    "EAFE": {"symbol": "EFA", "actual": False},
    "EM": {"symbol": "EEM", "actual": False},
    "MSCIUSREIT": {"symbol": "VNQ", "actual": False},
    "BCAgg": {"symbol": "AGG", "actual": False},
    "BCHY": {"symbol": "HYG", "actual": False},
    "BCGAgg": {"symbol": "BNDW", "actual": False},
    "BCGC13": {"symbol": "SHY", "actual": False},
}


def _download_close_series(symbol: str, label: str) -> pd.Series:
    df = yf.download(
        symbol,
        start=START_DATE,
        end="2026-01-01",
        auto_adjust=True,
        progress=False,
        interval="1d",
    )
    if df.empty:
        raise ValueError(f"No history returned for symbol: {symbol}")

    if isinstance(df.columns, pd.MultiIndex):
        # yfinance can return MultiIndex columns even for a single symbol.
        close = df.xs("Close", axis=1, level=0).iloc[:, 0]
    else:
        close = df["Close"]

    close = pd.to_numeric(close, errors="coerce")
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close = close.loc[START_DATE:END_DATE]
    if close.empty:
        raise ValueError(f"No data in target range for {symbol}")
    close.name = label
    return close


def build_returns() -> tuple[pd.DataFrame, pd.DataFrame]:
    close_series = []
    for label, meta in BENCHMARKS.items():
        symbol = str(meta["symbol"])
        actual = bool(meta["actual"])
        source_note = "actual index" if actual else "proxy fallback"
        print(f"Downloading {label} from {symbol} ({source_note})")
        close_series.append(_download_close_series(symbol, label))

    prices = pd.concat(close_series, axis=1).sort_index()
    daily_returns = prices.pct_change().loc[START_DATE:END_DATE]
    daily_returns = daily_returns.dropna(how="all")
    daily_returns.index.name = "Date"

    # Monthly returns from each month's first/last available price.
    # This preserves January returns even when the first calendar days are non-trading.
    month_groups = prices.groupby(prices.index.to_period("M"))
    month_first = month_groups.first()
    month_last = month_groups.last()
    monthly_returns = (month_last / month_first) - 1.0
    monthly_returns.index = monthly_returns.index.to_timestamp("M")
    monthly_returns = monthly_returns.loc[START_DATE:END_DATE]
    monthly_returns.index.name = "Date"

    return daily_returns, monthly_returns


def _save_excel(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Returns")


def main() -> None:
    daily, monthly = build_returns()

    daily_path = SAMPLE_DATA_DIR / SAMPLE_DAILY_FILE
    monthly_path = SAMPLE_DATA_DIR / SAMPLE_MONTHLY_FILE

    _save_excel(daily, daily_path)
    _save_excel(monthly, monthly_path)

    print(f"Saved daily:   {daily_path} ({daily.shape[0]} rows x {daily.shape[1]} cols)")
    print(f"Saved monthly: {monthly_path} ({monthly.shape[0]} rows x {monthly.shape[1]} cols)")


if __name__ == "__main__":
    main()
