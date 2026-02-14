"""Shared statistics configuration and benchmark store helpers."""

from __future__ import annotations


STATS_CONFIG = [
    ("Start Date", None),
    ("End Date", None),
    ("Number of Periods", None),
    ("Cumulative Return", ".2%"),
    ("Annualized Return", ".2%"),
    ("Annualized Volatility", ".2%"),
    ("Sharpe Ratio", ".2f"),
    ("Sortino Ratio", ".2f"),
    ("Beta to S&P 500", ".2f"),
    ("Annualized Excess Return", ".2%"),
    ("Annualized Tracking Error", ".2%"),
    ("Information Ratio", ".2f"),
    ("Correlation", ".2f"),
    ("Hit Rate", ".2%"),
    ("Hit Rate (vs Benchmark)", ".2%"),
    ("Best Period Return", ".2%"),
    ("Worst Period Return", ".2%"),
    ("Maximum Drawdown", ".2%"),
    ("Skewness", ".2f"),
    ("Kurtosis", ".2f"),
    ("1Y Annualized Return", ".2%"),
    ("1Y Annualized Volatility", ".2%"),
    ("1Y Sharpe Ratio", ".2f"),
    ("1Y Sortino Ratio", ".2f"),
    ("1Y Beta to S&P 500", ".2f"),
    ("1Y Excess Return", ".2%"),
    ("1Y Tracking Error", ".2%"),
    ("1Y Information Ratio", ".2f"),
    ("1Y Correlation", ".2f"),
    ("3Y Annualized Return", ".2%"),
    ("3Y Annualized Volatility", ".2%"),
    ("3Y Sharpe Ratio", ".2f"),
    ("3Y Sortino Ratio", ".2f"),
    ("3Y Beta to S&P 500", ".2f"),
    ("3Y Excess Return", ".2%"),
    ("3Y Tracking Error", ".2%"),
    ("3Y Information Ratio", ".2f"),
    ("3Y Correlation", ".2f"),
    ("5Y Annualized Return", ".2%"),
    ("5Y Annualized Volatility", ".2%"),
    ("5Y Sharpe Ratio", ".2f"),
    ("5Y Sortino Ratio", ".2f"),
    ("5Y Beta to S&P 500", ".2f"),
    ("5Y Excess Return", ".2%"),
    ("5Y Tracking Error", ".2%"),
    ("5Y Information Ratio", ".2f"),
    ("5Y Correlation", ".2f"),
]

RISK_FREE_SERIES = "BCTBill13_TRIndex"
MARKET_BETA_SERIES = "SPX_TRIndex"


def series_json_from_store(store_data, series_name: str) -> str:
    """Extract a series returns_json payload from the shared saved-series store."""
    if isinstance(store_data, dict):
        series_data = store_data.get("series_data")
        if isinstance(series_data, dict):
            series_payload = series_data.get(series_name)
            if isinstance(series_payload, dict):
                payload = series_payload.get("returns_json")
                if isinstance(payload, str):
                    return payload
    return ""


def risk_free_json_from_store(store_data) -> str:
    return series_json_from_store(store_data, RISK_FREE_SERIES)


def spx_json_from_store(store_data) -> str:
    return series_json_from_store(store_data, MARKET_BETA_SERIES)
