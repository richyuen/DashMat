"""Initialize and populate local CMA test database."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sqlalchemy import Column, Date, DateTime, Float, ForeignKey, Integer, MetaData, String, Table, text

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dbengine import (
    DATABASE_URL,
    MRD_DATABASE_URL,
    PERFORMANCE_DATABASE_URL,
    engine,
    engine_MRD,
    engine_PERFORMANCE,
)
from tools.db.migrate_account_lists import ensure_account_list_tables
from tools.db.migrate_factor_definitions import ensure_factor_definition_tables_and_seed
from tools.db.migrate_users import ensure_users_table
from utils.sample_data import get_sample_file_path
from utils.serialization import canonical_json_dumps


VERSIONS = [2025, 2026]
TYPES = ["hmm", "equilibrium.gp"]
ITEMS = ["Mean", "SD", "Skewness", "Kurtosis"]
RISK_FREE_BENCH = "BCTBill13"
MTH_TO_DLY_BENCH = "MthToDly"
MTH_TO_DLY_DAILY_START = pd.Timestamp("2022-01-03")
FRED_DGS3MO_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO"
LOCAL_ACCOUNT_LIST_SEED_USERNAME = "Admin User"
LOCAL_ACCOUNT_LIST_SEED_UPDATE_BY = "init_local_cma_db.py"
LOCAL_USER_SEED_ROWS = [
    {"Username": "Admin User", "Role": "Admin"},
    {"Username": "Analyst One", "Role": "Analyst"},
    {"Username": "Analyst Two", "Role": "Analyst"},
]

CORE_CATEGORY_MAP: dict[str, dict[str, str]] = {
    "SPX": {
        "CoreCat": "S&P 500",
        "AssetClass": "Equity",
        "PeerBench": "Large Blend",
        "AATool": "S&P 500",
    },
    "RMID": {
        "CoreCat": "Russell Midcap",
        "AssetClass": "Equity",
        "PeerBench": "Mid-Cap Blend",
        "AATool": "Russell Mid",
    },
    "R2000": {
        "CoreCat": "Russell 2000",
        "AssetClass": "Equity",
        "PeerBench": "Small Blend",
        "AATool": "Russell 2000",
    },
    "EAFE": {
        "CoreCat": "MSCI EAFE",
        "AssetClass": "Equity",
        "PeerBench": "Foreign Large Blend",
        "AATool": "EAFE",
    },
    "EM": {
        "CoreCat": "MSCI Emerging Markets",
        "AssetClass": "Equity",
        "PeerBench": "Diversified Emerging Mkts",
        "AATool": "EM",
    },
    "MSCIUSREIT": {
        "CoreCat": "MSCI US REIT",
        "AssetClass": "Equity",
        "PeerBench": "Real Estate",
        "AATool": "US REIT",
    },
    "BCAgg": {
        "CoreCat": "Bloomberg US Aggregate",
        "AssetClass": "Bond",
        "PeerBench": "Intermediate Core Bond",
        "AATool": "US Agg",
    },
    "BCHY": {
        "CoreCat": "Bloomberg US High Yield",
        "AssetClass": "Bond",
        "PeerBench": "High Yield Bond",
        "AATool": "US High Yield",
    },
    "BCGAgg": {
        "CoreCat": "Bloomberg Global Aggregate",
        "AssetClass": "Bond",
        "PeerBench": "Global Bond",
        "AATool": "Global Agg",
    },
    "BCGC13": {
        "CoreCat": "Bloomberg US Treasury 1-3 Year",
        "AssetClass": "Bond",
        "PeerBench": "Short Government",
        "AATool": "UST 1-3Y",
    },
    "BCTBill13": {
        "CoreCat": "Bloomberg US T-Bill 1-3 Month",
        "AssetClass": "Bond",
        "PeerBench": "Short Government",
        "AATool": "US T-Bill 1-3M",
    },
    "MthToDly": {
        "CoreCat": "MthToDly Test Series",
        "AssetClass": "Equity",
        "PeerBench": "Unspecified",
        "AATool": "MthToDly",
    },
}

SUITE_ROWS: list[dict] = [
    {
        "SuiteID": 1,
        "SuiteShort": "TD",
        "SuiteLong": "Target Date",
        "IndexDailyOrder": None,
        "PeerTDOrder": 1,
        "PeerModelOrder": None,
        "PeerAllocOrder": None,
        "Peer529Order": None,
    },
    {
        "SuiteID": 2,
        "SuiteShort": "RISK",
        "SuiteLong": "Risk-Based",
        "IndexDailyOrder": 1,
        "PeerTDOrder": None,
        "PeerModelOrder": None,
        "PeerAllocOrder": None,
        "Peer529Order": None,
    },
    {
        "SuiteID": 3,
        "SuiteShort": "IndNoAttr",
        "SuiteLong": "Index No Attribution",
        "IndexDailyOrder": None,
        "PeerTDOrder": None,
        "PeerModelOrder": None,
        "PeerAllocOrder": None,
        "Peer529Order": None,
    },
]

PORTFOLIO_ROWS: list[dict] = [
    {
        "PortfolioID": 1,
        "PortfolioName": "Target Date 2030 Fund",
        "Portfolio": "TD2030",
        "PortfolioSuite": "TD",
        "PeerVintage": "2030",
        "PortfolioVintage": "",
        "IncepDate": date(2012, 1, 31),
    },
    {
        "PortfolioID": 5,
        "PortfolioName": "Target Date 2030 Select Fund",
        "Portfolio": "TD2030S",
        "PortfolioSuite": "TD",
        "PeerVintage": "2030",
        "PortfolioVintage": "",
        "IncepDate": date(2015, 1, 31),
    },
    {
        "PortfolioID": 2,
        "PortfolioName": "Target Date 2050 Fund",
        "Portfolio": "TD2050",
        "PortfolioSuite": "TD",
        "PeerVintage": "2050",
        "PortfolioVintage": "",
        "IncepDate": date(2014, 1, 31),
    },
    {
        "PortfolioID": 11,
        "PortfolioName": "Target Date 2040 Fund",
        "Portfolio": "TD2040",
        "PortfolioSuite": "TD",
        "PeerVintage": "2040",
        "PortfolioVintage": "",
        "IncepDate": date(2013, 7, 31),
    },
    {
        "PortfolioID": 12,
        "PortfolioName": "Target Date 2060 Fund",
        "Portfolio": "TD2060",
        "PortfolioSuite": "TD",
        "PeerVintage": "2060",
        "PortfolioVintage": "",
        "IncepDate": date(2015, 7, 31),
    },
    {
        "PortfolioID": 3,
        "PortfolioName": "Risk Balanced 60 Fund",
        "Portfolio": "Risk60",
        "PortfolioSuite": "RISK",
        "PeerVintage": "",
        "PortfolioVintage": "",
        "IncepDate": date(2011, 1, 31),
    },
    {
        "PortfolioID": 4,
        "PortfolioName": "Risk Balanced 80 Fund",
        "Portfolio": "Risk80",
        "PortfolioSuite": "RISK",
        "PeerVintage": "",
        "PortfolioVintage": "",
        "IncepDate": date(2013, 1, 31),
    },
    {
        "PortfolioID": 13,
        "PortfolioName": "Risk Balanced 40 Fund",
        "Portfolio": "Risk40",
        "PortfolioSuite": "RISK",
        "PeerVintage": "",
        "PortfolioVintage": "",
        "IncepDate": date(2012, 1, 31),
    },
    {
        "PortfolioID": 14,
        "PortfolioName": "Risk Balanced 50 Fund",
        "Portfolio": "Risk50",
        "PortfolioSuite": "RISK",
        "PeerVintage": "",
        "PortfolioVintage": "",
        "IncepDate": date(2014, 7, 31),
    },
    {
        "PortfolioID": 15,
        "PortfolioName": "Risk Balanced 70 Fund",
        "Portfolio": "Risk70",
        "PortfolioSuite": "RISK",
        "PeerVintage": "",
        "PortfolioVintage": "",
        "IncepDate": date(2013, 7, 31),
    },
    {
        "PortfolioID": 6,
        "PortfolioName": "Alternative Trend Fund",
        "Portfolio": "ALTTRN",
        "PortfolioSuite": "IndNoAttr",
        "PeerVintage": "",
        "PortfolioVintage": "AltTS",
        "IncepDate": date(2016, 1, 4),
    },
    {
        "PortfolioID": 7,
        "PortfolioName": "Alternative Macro Fund",
        "Portfolio": "ALTMAC",
        "PortfolioSuite": "IndNoAttr",
        "PeerVintage": "",
        "PortfolioVintage": "AltTS",
        "IncepDate": date(2017, 1, 3),
    },
    {
        "PortfolioID": 8,
        "PortfolioName": "Alternative No Benchmark Fund",
        "Portfolio": "ALTNOBM",
        "PortfolioSuite": "IndNoAttr",
        "PeerVintage": "",
        "PortfolioVintage": "AltTS",
        "IncepDate": date(2018, 1, 2),
    },
    {
        "PortfolioID": 9,
        "PortfolioName": "Performance Trend Fund",
        "Portfolio": "PERFTRN",
        "PortfolioSuite": "IndNoAttr",
        "PeerVintage": "2030",
        "PortfolioVintage": "Perf",
        "IncepDate": date(2016, 1, 4),
    },
    {
        "PortfolioID": 10,
        "PortfolioName": "Performance No Benchmark Fund",
        "Portfolio": "PERFNOBM",
        "PortfolioSuite": "IndNoAttr",
        "PeerVintage": "",
        "PortfolioVintage": "Perf",
        "IncepDate": date(2018, 1, 2),
    },
]

UNDERLYING_CATEGORY_ITEM = "PeerRet"
UNDERLYING_CATEGORY_PORTFOLIO_DESCS: dict[str, tuple[str, ...]] = {
    "CoreTD": ("Large Cap", "Small Cap", "Core Bond", "High Yield"),
    "CoreAlloc": ("Large Cap", "International Equity", "Core Bond", "Real Assets"),
    "Core529": ("Large Cap", "International Equity", "Core Bond"),
    "CoreModel": ("Large Cap", "Small Cap", "International Equity", "Real Assets"),
    "BaseTD": ("Large Cap", "Small Cap", "Core Bond"),
    "BaseAlloc": ("Large Cap", "International Equity", "Real Assets"),
    "Base529": ("Large Cap", "Core Bond"),
    "BaseModel": ("Large Cap", "International Equity", "High Yield", "Real Assets"),
}


def _load_monthly_returns() -> pd.DataFrame:
    path = get_sample_file_path("monthly")
    df = pd.read_excel(path, index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def _load_daily_returns() -> pd.DataFrame:
    path = get_sample_file_path("daily")
    df = pd.read_excel(path, index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return _ensure_business_daily_returns(df)


def _ensure_business_daily_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure daily seed returns have business-day frequency.

    If the loaded "daily" sample is actually lower frequency (e.g. month-end),
    distribute each monthly return evenly across business days in that month.
    """
    if df.empty:
        return df

    out = df.copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    out = out.loc[~out.index.duplicated(keep="last")]

    idx = pd.DatetimeIndex(out.index).sort_values().unique()
    if len(idx) < 2:
        return out

    min_delta = pd.Series(idx).diff().dropna().min()
    if min_delta is not None and min_delta <= pd.Timedelta(days=7):
        return out

    # Convert lower-frequency series to synthetic business-daily returns.
    bidx = pd.date_range(
        start=pd.Timestamp(idx.min()).to_period("M").start_time,
        end=pd.Timestamp(idx.max()),
        freq="B",
    )
    daily = pd.DataFrame(index=bidx, columns=out.columns, dtype=float)

    for col in out.columns:
        series = pd.to_numeric(out[col], errors="coerce").dropna()
        if series.empty:
            continue
        for dt, val in series.items():
            dt_ts = pd.Timestamp(dt)
            month_start = dt_ts.to_period("M").start_time
            month_end = dt_ts.to_period("M").end_time.normalize()
            month_days = bidx[(bidx >= month_start) & (bidx <= month_end)]
            n_days = len(month_days)
            if n_days == 0:
                continue
            v = float(val)
            if v <= -1.0:
                daily_ret = np.nan
            else:
                daily_ret = (1.0 + v) ** (1.0 / n_days) - 1.0
            daily.loc[month_days, col] = daily_ret

    daily = daily.dropna(how="all")
    daily.index.name = out.index.name
    return daily


def _build_bctbill13_proxy_returns(daily_df: pd.DataFrame) -> pd.Series:
    """Build a daily 3M T-Bill proxy return series for the sample history.

    Primary source:
    - FRED DGS3MO yield converted to daily return using ACT/360.

    Offline fallback:
    - Smoothed and clipped BCGC13 sample returns.
    """
    if daily_df.empty:
        return pd.Series(dtype=float)

    target_index = pd.DatetimeIndex(daily_df.index).sort_values()
    start = target_index.min()
    end = target_index.max()

    try:
        fred = pd.read_csv(FRED_DGS3MO_CSV_URL)
        fred["DATE"] = pd.to_datetime(fred["DATE"], errors="coerce")
        fred["DGS3MO"] = pd.to_numeric(fred["DGS3MO"], errors="coerce")
        fred = fred.dropna(subset=["DATE", "DGS3MO"]).set_index("DATE").sort_index()
        fred = fred.loc[(fred.index >= start - pd.Timedelta(days=10)) & (fred.index <= end + pd.Timedelta(days=10))]

        if not fred.empty:
            daily_ret = (fred["DGS3MO"] / 100.0) / 360.0
            daily_ret = daily_ret.reindex(target_index).ffill().bfill()
            if not daily_ret.isna().all():
                return daily_ret.fillna(0.0)
    except Exception:
        pass

    if "BCGC13" in daily_df.columns:
        fallback = daily_df["BCGC13"].copy()
        fallback = fallback.rolling(21, min_periods=1).mean()
        fallback = fallback.clip(lower=-0.001, upper=0.001)
        return fallback.reindex(target_index).fillna(0.0)

    return pd.Series(0.0, index=target_index)


def _build_mth_to_dly_returns(daily_df: pd.DataFrame) -> pd.Series:
    """Create a synthetic series that is monthly in early history, then daily."""
    if daily_df.empty:
        return pd.Series(dtype=float, name=MTH_TO_DLY_BENCH)

    available_daily = pd.DatetimeIndex(daily_df.index).sort_values()
    daily_candidates = available_daily[available_daily >= MTH_TO_DLY_DAILY_START]
    if len(daily_candidates) > 0:
        daily_start = pd.Timestamp(daily_candidates[0])
    else:
        # Fallback if dataset does not reach 2022.
        daily_start = pd.Timestamp(available_daily.min())

    # Monthly segment starts in 2015 and ends before daily observations begin.
    monthly_start = pd.Timestamp("2015-01-31")
    monthly_end = (daily_start - pd.offsets.MonthEnd(1)).normalize()
    monthly_idx = pd.date_range(monthly_start, monthly_end, freq="ME")
    if len(monthly_idx) > 0:
        x = np.arange(len(monthly_idx), dtype=float)
        monthly_returns = pd.Series(
            0.002 + 0.0007 * np.sin(0.35 * x),
            index=monthly_idx,
        )
    else:
        monthly_returns = pd.Series(dtype=float)

    # Daily segment uses sample daily data behavior as a realistic continuation.
    base_col = "SPX" if "SPX" in daily_df.columns else (daily_df.columns[0] if len(daily_df.columns) > 0 else None)
    if base_col is None:
        daily_part = pd.Series(dtype=float)
    else:
        daily_part = (daily_df[base_col] * 0.6).dropna()
        daily_part = daily_part.loc[daily_part.index >= daily_start]

    combined = pd.concat([monthly_returns, daily_part])
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()
    combined.name = MTH_TO_DLY_BENCH
    return combined


def _transform_for_type(df: pd.DataFrame, cma_type: str) -> pd.DataFrame:
    if cma_type == "hmm":
        return df.copy()
    # Equilibrium-style: slight mean and volatility dampening versus historical.
    return df * 0.90


def _pick_seed_column(df: pd.DataFrame, preferred: str, fallback_pos: int) -> pd.Series:
    if preferred in df.columns:
        return pd.to_numeric(df[preferred], errors="coerce").fillna(0.0)
    if df.empty or len(df.columns) == 0:
        return pd.Series(dtype=float)
    col = df.columns[min(fallback_pos, len(df.columns) - 1)]
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0)


def _seed_daily_index(daily_df: pd.DataFrame) -> pd.DatetimeIndex:
    """Pick a dense daily index from core benchmark columns."""
    if daily_df.empty:
        return pd.DatetimeIndex([])

    preferred_cols = [c for c in ("SPX", "BCAgg", "R2000", "EAFE") if c in daily_df.columns]
    if not preferred_cols:
        preferred_cols = [daily_df.columns[0]]

    base = daily_df[preferred_cols]
    if isinstance(base, pd.Series):
        base = base.to_frame()

    idx = pd.DatetimeIndex(base.dropna(how="all").index).sort_values().unique()
    if len(idx) == 0:
        idx = pd.DatetimeIndex(daily_df.index).sort_values().unique()
    return idx


def _returns_to_levels(series: pd.Series, start_level: float = 100.0) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce").fillna(0.0)
    levels = (1.0 + clean).cumprod() * float(start_level)
    return levels


def _build_portfolio_seed_series(daily_df: pd.DataFrame) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    """Build deterministic daily-level series for PeerTS/IndexTS."""
    if daily_df.empty:
        return {}, {}

    idx = _seed_daily_index(daily_df)
    if len(idx) == 0:
        return [], [], []
    source_df = daily_df.reindex(idx)
    spx = _pick_seed_column(source_df, "SPX", 0)
    agg = _pick_seed_column(source_df, "BCAgg", 1)

    peer_returns: dict[str, pd.Series] = {
        "TD2030|PortRet|Actual": 0.62 * spx + 0.38 * agg,
        "TD2030|PortRet|Calculated": 0.98 * (0.62 * spx + 0.38 * agg),
        "TD2030|MeanRet|Calculated": 0.96 * (0.60 * spx + 0.40 * agg),
        "TD2030S|PortRet|Actual": 0.66 * spx + 0.34 * agg,
        "TD2030S|PortRet|Calculated": 0.98 * (0.66 * spx + 0.34 * agg),
        "TD2030S|MeanRet|Calculated": 0.96 * (0.64 * spx + 0.36 * agg),
        "TD2050|PortRet|Actual": 0.80 * spx + 0.20 * agg,
        "TD2050|PortRet|Calculated": 0.98 * (0.80 * spx + 0.20 * agg),
        "TD2050|MeanRet|Calculated": 0.96 * (0.78 * spx + 0.22 * agg),
        "2030|MeanRet|Actual": 0.58 * spx + 0.42 * agg,
        "2030|MeanRet|Estimated": 0.96 * (0.58 * spx + 0.42 * agg),
        "2050|MeanRet|Actual": 0.76 * spx + 0.24 * agg,
        "2050|MeanRet|Estimated": 0.96 * (0.76 * spx + 0.24 * agg),
        "TD2040|PortRet|Actual": 0.70 * spx + 0.30 * agg,
        "TD2040|PortRet|Calculated": 0.98 * (0.70 * spx + 0.30 * agg),
        "TD2040|MeanRet|Calculated": 0.96 * (0.68 * spx + 0.32 * agg),
        "2040|MeanRet|Actual": 0.66 * spx + 0.34 * agg,
        "2040|MeanRet|Estimated": 0.96 * (0.66 * spx + 0.34 * agg),
        "TD2060|PortRet|Actual": 0.85 * spx + 0.15 * agg,
        "TD2060|PortRet|Calculated": 0.98 * (0.85 * spx + 0.15 * agg),
        "TD2060|MeanRet|Calculated": 0.96 * (0.83 * spx + 0.17 * agg),
        "2060|MeanRet|Actual": 0.82 * spx + 0.18 * agg,
        "2060|MeanRet|Estimated": 0.96 * (0.82 * spx + 0.18 * agg),
    }

    index_returns: dict[str, pd.Series] = {
        "Risk60|PortRet|Actual": 0.60 * spx + 0.40 * agg,
        "Risk60|PortRet|Calculated": 0.98 * (0.60 * spx + 0.40 * agg),
        "Risk60|PortRet|Benchmark": 0.57 * spx + 0.43 * agg,
        "Risk80|PortRet|Actual": 0.80 * spx + 0.20 * agg,
        "Risk80|PortRet|Calculated": 0.98 * (0.80 * spx + 0.20 * agg),
        "Risk80|PortRet|Benchmark": 0.77 * spx + 0.23 * agg,
        "Risk40|PortRet|Actual": 0.40 * spx + 0.60 * agg,
        "Risk40|PortRet|Calculated": 0.98 * (0.40 * spx + 0.60 * agg),
        "Risk40|PortRet|Benchmark": 0.37 * spx + 0.63 * agg,
        "Risk50|PortRet|Actual": 0.50 * spx + 0.50 * agg,
        "Risk50|PortRet|Calculated": 0.98 * (0.50 * spx + 0.50 * agg),
        "Risk50|PortRet|Benchmark": 0.47 * spx + 0.53 * agg,
        "Risk70|PortRet|Actual": 0.70 * spx + 0.30 * agg,
        "Risk70|PortRet|Calculated": 0.98 * (0.70 * spx + 0.30 * agg),
        "Risk70|PortRet|Benchmark": 0.67 * spx + 0.33 * agg,
    }

    peer_series: dict[str, pd.Series] = {}
    index_series: dict[str, pd.Series] = {}
    for key, series in peer_returns.items():
        ret = pd.Series(series.values, index=idx, dtype=float)
        peer_series[key] = _returns_to_levels(ret, start_level=100.0)
    for key, series in index_returns.items():
        ret = pd.Series(series.values, index=idx, dtype=float)
        index_series[key] = _returns_to_levels(ret, start_level=100.0)
    return peer_series, index_series


def _build_underlying_category_seed_series(daily_df: pd.DataFrame) -> dict[str, pd.Series]:
    """Build deterministic daily-level PeerTS series for underlying categories."""
    if daily_df.empty:
        return {}

    idx = _seed_daily_index(daily_df)
    if len(idx) == 0:
        return {}

    source_df = daily_df.reindex(idx)
    spx = _pick_seed_column(source_df, "SPX", 0)
    mid = _pick_seed_column(source_df, "RMID", 1)
    small = _pick_seed_column(source_df, "R2000", 2)
    eafe = _pick_seed_column(source_df, "EAFE", 3)
    em = _pick_seed_column(source_df, "EM", 4)
    reit = _pick_seed_column(source_df, "MSCIUSREIT", 5)
    agg = _pick_seed_column(source_df, "BCAgg", 6)
    hy = _pick_seed_column(source_df, "BCHY", 7)
    global_agg = _pick_seed_column(source_df, "BCGAgg", 8)
    x = np.arange(len(idx), dtype=float)

    desc_returns: dict[str, pd.Series] = {
        "Large Cap": pd.Series(0.88 * spx + 0.12 * agg + 0.00005 * np.sin(0.017 * x), index=idx, dtype=float),
        "Small Cap": pd.Series(0.78 * small + 0.14 * mid + 0.08 * agg + 0.00007 * np.cos(0.021 * x), index=idx, dtype=float),
        "International Equity": pd.Series(
            0.58 * eafe + 0.27 * em + 0.15 * global_agg + 0.00006 * np.sin(0.019 * x + 0.4),
            index=idx,
            dtype=float,
        ),
        "Core Bond": pd.Series(0.80 * agg + 0.20 * global_agg + 0.00003 * np.cos(0.015 * x + 0.7), index=idx, dtype=float),
        "High Yield": pd.Series(0.66 * hy + 0.22 * agg + 0.12 * spx + 0.00005 * np.sin(0.023 * x + 0.8), index=idx, dtype=float),
        "Real Assets": pd.Series(
            0.52 * reit + 0.24 * eafe + 0.12 * em + 0.12 * agg + 0.00008 * np.cos(0.018 * x + 0.2),
            index=idx,
            dtype=float,
        ),
    }

    portfolio_tilts: dict[str, dict[str, float]] = {
        "CoreTD": {"scale": 1.00, "drift": 0.00002, "blend": 0.08},
        "CoreAlloc": {"scale": 0.97, "drift": 0.00001, "blend": 0.14},
        "Core529": {"scale": 0.95, "drift": 0.000015, "blend": 0.10},
        "CoreModel": {"scale": 1.02, "drift": 0.00002, "blend": 0.12},
        "BaseTD": {"scale": 0.93, "drift": 0.00001, "blend": 0.06},
        "BaseAlloc": {"scale": 0.90, "drift": 0.000005, "blend": 0.15},
        "Base529": {"scale": 0.89, "drift": 0.00001, "blend": 0.09},
        "BaseModel": {"scale": 0.96, "drift": 0.000015, "blend": 0.13},
    }
    portfolio_anchors: dict[str, pd.Series] = {
        "TD": pd.Series(0.64 * spx + 0.36 * agg, index=idx, dtype=float),
        "Alloc": pd.Series(0.48 * spx + 0.52 * agg, index=idx, dtype=float),
        "529": pd.Series(0.58 * spx + 0.42 * agg, index=idx, dtype=float),
        "Model": pd.Series(0.60 * spx + 0.25 * agg + 0.15 * eafe, index=idx, dtype=float),
    }

    out: dict[str, pd.Series] = {}
    for portfolio_index, (portfolio, descs) in enumerate(UNDERLYING_CATEGORY_PORTFOLIO_DESCS.items()):
        tilt = portfolio_tilts[portfolio]
        anchor_key = portfolio[4:] if portfolio.startswith("Base") else portfolio[4:]
        anchor_series = portfolio_anchors.get(anchor_key, portfolio_anchors["TD"])
        start_level = 100.0 + (portfolio_index * 4.0)
        for desc_index, desc in enumerate(descs):
            base_series = desc_returns[desc]
            modulation = pd.Series(
                0.00004 * np.sin(0.013 * x + (portfolio_index + 1) * 0.45 + (desc_index + 1) * 0.30),
                index=idx,
                dtype=float,
            )
            ret = ((1.0 - tilt["blend"]) * base_series) + (tilt["blend"] * anchor_series) + tilt["drift"] + modulation
            ret = pd.Series(ret.values, index=idx, dtype=float).clip(lower=-0.08, upper=0.08)
            out[f"{portfolio}|{UNDERLYING_CATEGORY_ITEM}|{desc}"] = _returns_to_levels(
                ret,
                start_level=start_level + desc_index,
            )
    return out


def build_underlying_category_seed_rows(daily_df: pd.DataFrame) -> list[dict]:
    return _portfolio_ts_rows(_build_underlying_category_seed_series(daily_df))


def _build_alt_seed_series(daily_df: pd.DataFrame) -> dict[str, pd.Series]:
    """Build deterministic daily return series for AltTS."""
    if daily_df.empty:
        return {}

    idx = pd.DatetimeIndex(daily_df.index).sort_values()
    source_df = daily_df.reindex(idx)
    spx = _pick_seed_column(source_df, "SPX", 0)
    agg = _pick_seed_column(source_df, "BCAgg", 1)
    x = np.arange(len(idx), dtype=float)

    alt_returns: dict[str, pd.Series] = {
        "ALTTRN|PortRet": 0.36 * spx + 0.24 * agg + 0.00020 * np.sin(0.11 * x),
        "ALTTRN|BenchRet": 0.31 * spx + 0.29 * agg + 0.00010 * np.sin(0.09 * x),
        "ALTMAC|PortRet": 0.30 * spx + 0.30 * agg + 0.00018 * np.cos(0.08 * x),
        "ALTMAC|BenchRet": 0.27 * spx + 0.33 * agg + 0.00008 * np.cos(0.06 * x),
        "ALTNOBM|PortRet": 0.22 * spx + 0.38 * agg + 0.00015 * np.sin(0.07 * x),
    }

    out: dict[str, pd.Series] = {}
    for key, series in alt_returns.items():
        out[key] = pd.Series(series.values, index=idx, dtype=float)
    return out


def _portfolio_ts_rows(series_map: dict[str, pd.Series]) -> list[dict]:
    rows: list[dict] = []
    for key, series in series_map.items():
        parts = key.split("|")
        if len(parts) != 3:
            continue
        portfolio, item, desc = parts
        clean = pd.to_numeric(series, errors="coerce").dropna()
        for dt, val in clean.items():
            rows.append(
                {
                    "Date": date(dt.year, dt.month, dt.day),
                    "Portfolio": str(portfolio),
                    "Item": str(item),
                    "Desc": str(desc),
                    "Value": float(val),
                }
            )
    return rows


def _alt_ts_rows(series_map: dict[str, pd.Series]) -> list[dict]:
    rows: list[dict] = []
    for key, series in series_map.items():
        parts = key.split("|")
        if len(parts) != 2:
            continue
        portfolio, item = parts
        clean = pd.to_numeric(series, errors="coerce").dropna()
        for dt, val in clean.items():
            rows.append(
                {
                    "Date": date(dt.year, dt.month, dt.day),
                    "Portfolio": str(portfolio),
                    "Item": str(item),
                    "Value": float(val),
                }
            )
    return rows


def _build_perf_seed_rows(
    daily_df: pd.DataFrame,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Build deterministic Performance DB seed rows.

    Performance.DAILY_RETURN stores returns in percentage points, not decimals.
    """
    account_rows = [
        {"ACCT_ID": 9001, "ACCT_CD": "PERFTRN"},
        {"ACCT_ID": 9002, "ACCT_CD": "PERFNOBM"},
    ]
    benchmark_rows = [
        {"BENCHMARK_ID": 9101, "PRECEDENCE": 1},
        {"BENCHMARK_ID": 9102, "PRECEDENCE": 2},
    ]

    if daily_df.empty:
        return account_rows, benchmark_rows, [], []

    idx = _seed_daily_index(daily_df)
    if len(idx) == 0:
        return account_rows, benchmark_rows, [], []
    source_df = daily_df.reindex(idx)
    spx = _pick_seed_column(source_df, "SPX", 0)
    agg = _pick_seed_column(source_df, "BCAgg", 1)
    x = np.arange(len(idx), dtype=float)

    perf_trend = 0.40 * spx + 0.30 * agg + 0.00015 * np.sin(0.11 * x)
    perf_trend_bm = 0.35 * spx + 0.35 * agg + 0.00009 * np.sin(0.08 * x)
    perf_no_bm = 0.24 * spx + 0.36 * agg + 0.00012 * np.cos(0.07 * x)
    perf_no_bm_idx = 0.20 * spx + 0.32 * agg + 0.00008 * np.cos(0.05 * x)

    daily_rows: list[dict] = []
    for dt in idx:
        d = date(dt.year, dt.month, dt.day)

        trend_ret = float(pd.to_numeric(pd.Series([perf_trend.loc[dt]]), errors="coerce").iloc[0])
        trend_bm_ret = float(pd.to_numeric(pd.Series([perf_trend_bm.loc[dt]]), errors="coerce").iloc[0])
        no_bm_ret = float(pd.to_numeric(pd.Series([perf_no_bm.loc[dt]]), errors="coerce").iloc[0])
        no_bm_bm_ret = float(pd.to_numeric(pd.Series([perf_no_bm_idx.loc[dt]]), errors="coerce").iloc[0])

        # Keep only one valid row by filters (latest=1, precedence=1, gross fee),
        # plus extra rows that should be excluded by query predicates.
        daily_rows.append(
            {
                "Effective_Date": d,
                "Daily_ror": trend_ret * 100.0,
                "Daily_ror_index": trend_bm_ret * 100.0,
                "ACCT_ID": 9001,
                "BENCHMARK_ACCT_ID": 9101,
                "FEE_TYPE": "G",
                "IS_LATEST": 1,
            }
        )
        daily_rows.append(
            {
                "Effective_Date": d,
                "Daily_ror": (trend_ret - 0.00009) * 100.0,
                "Daily_ror_index": (trend_bm_ret - 0.00007) * 100.0,
                "ACCT_ID": 9001,
                "BENCHMARK_ACCT_ID": 9101,
                "FEE_TYPE": "N",
                "IS_LATEST": 1,
            }
        )
        daily_rows.append(
            {
                "Effective_Date": d,
                "Daily_ror": trend_ret * 105.0,
                "Daily_ror_index": trend_bm_ret * 105.0,
                "ACCT_ID": 9001,
                "BENCHMARK_ACCT_ID": 9102,
                "FEE_TYPE": "G",
                "IS_LATEST": 1,
            }
        )
        daily_rows.append(
            {
                "Effective_Date": d,
                "Daily_ror": trend_ret * 102.0,
                "Daily_ror_index": trend_bm_ret * 102.0,
                "ACCT_ID": 9001,
                "BENCHMARK_ACCT_ID": 9101,
                "FEE_TYPE": "G",
                "IS_LATEST": 0,
            }
        )
        daily_rows.append(
            {
                "Effective_Date": d,
                "Daily_ror": no_bm_ret * 100.0,
                "Daily_ror_index": no_bm_bm_ret * 100.0,
                "ACCT_ID": 9002,
                "BENCHMARK_ACCT_ID": 9101,
                "FEE_TYPE": "G",
                "IS_LATEST": 1,
            }
        )
        daily_rows.append(
            {
                "Effective_Date": d,
                "Daily_ror": (no_bm_ret - 0.00005) * 100.0,
                "Daily_ror_index": (no_bm_bm_ret - 0.00004) * 100.0,
                "ACCT_ID": 9002,
                "BENCHMARK_ACCT_ID": 9101,
                "FEE_TYPE": "N",
                "IS_LATEST": 1,
            }
        )

    trend_monthly = pd.Series(perf_trend, index=idx, dtype=float).resample("ME").apply(lambda xvals: (1.0 + xvals).prod() - 1.0)
    trend_monthly_bm = pd.Series(perf_trend_bm, index=idx, dtype=float).resample("ME").apply(lambda xvals: (1.0 + xvals).prod() - 1.0)
    no_bm_monthly = pd.Series(perf_no_bm, index=idx, dtype=float).resample("ME").apply(lambda xvals: (1.0 + xvals).prod() - 1.0)
    no_bm_monthly_idx = pd.Series(perf_no_bm_idx, index=idx, dtype=float).resample("ME").apply(lambda xvals: (1.0 + xvals).prod() - 1.0)

    monthly_rows: list[dict] = []
    for dt in trend_monthly.index:
        d = date(dt.year, dt.month, dt.day)
        t_val = float(trend_monthly.loc[dt])
        t_bm_val = float(trend_monthly_bm.loc[dt])
        n_val = float(no_bm_monthly.loc[dt])
        n_bm_val = float(no_bm_monthly_idx.loc[dt])

        monthly_rows.append(
            {
                "Effective_Date": d,
                "ACCT_ID": 9001,
                "BENCHMARK_ACCT_ID": 9101,
                "FEE_TYPE": "G",
                "IS_LATEST": 1,
                "Return_Type": "Ann",
                "mth1_ror": t_val * 100.0,
                "mth1_ror_index": t_bm_val * 100.0,
            }
        )
        monthly_rows.append(
            {
                "Effective_Date": d,
                "ACCT_ID": 9001,
                "BENCHMARK_ACCT_ID": 9101,
                "FEE_TYPE": "N",
                "IS_LATEST": 1,
                "Return_Type": "Ann",
                "mth1_ror": (t_val - 0.0007) * 100.0,
                "mth1_ror_index": (t_bm_val - 0.0005) * 100.0,
            }
        )
        monthly_rows.append(
            {
                "Effective_Date": d,
                "ACCT_ID": 9001,
                "BENCHMARK_ACCT_ID": 9101,
                "FEE_TYPE": "G",
                "IS_LATEST": 1,
                "Return_Type": "Cum",
                "mth1_ror": t_val * 120.0,
                "mth1_ror_index": t_bm_val * 120.0,
            }
        )
        monthly_rows.append(
            {
                "Effective_Date": d,
                "ACCT_ID": 9001,
                "BENCHMARK_ACCT_ID": 9102,
                "FEE_TYPE": "G",
                "IS_LATEST": 1,
                "Return_Type": "Ann",
                "mth1_ror": t_val * 115.0,
                "mth1_ror_index": t_bm_val * 115.0,
            }
        )
        monthly_rows.append(
            {
                "Effective_Date": d,
                "ACCT_ID": 9001,
                "BENCHMARK_ACCT_ID": 9101,
                "FEE_TYPE": "G",
                "IS_LATEST": 0,
                "Return_Type": "Ann",
                "mth1_ror": t_val * 108.0,
                "mth1_ror_index": t_bm_val * 108.0,
            }
        )

        monthly_rows.append(
            {
                "Effective_Date": d,
                "ACCT_ID": 9002,
                "BENCHMARK_ACCT_ID": 9101,
                "FEE_TYPE": "G",
                "IS_LATEST": 1,
                "Return_Type": "Ann",
                "mth1_ror": n_val * 100.0,
                "mth1_ror_index": n_bm_val * 100.0,
            }
        )
        monthly_rows.append(
            {
                "Effective_Date": d,
                "ACCT_ID": 9002,
                "BENCHMARK_ACCT_ID": 9101,
                "FEE_TYPE": "N",
                "IS_LATEST": 1,
                "Return_Type": "Ann",
                "mth1_ror": (n_val - 0.0004) * 100.0,
                "mth1_ror_index": (n_bm_val - 0.0003) * 100.0,
            }
        )

    return account_rows, benchmark_rows, daily_rows, monthly_rows


def _build_tables(
    metadata: MetaData,
) -> tuple[Table, Table, Table, Table, Table, Table, Table, Table, Table, Table, Table, Table, Table, Table, Table]:
    cma_corr = Table(
        "CMACorrelation",
        metadata,
        Column("Version", Integer, primary_key=True),
        Column("Type", String(64), primary_key=True),
        Column("Bench1", String(64), primary_key=True),
        Column("Bench2", String(64), primary_key=True),
        Column("Value", Float, nullable=False),
    )
    cma_ret = Table(
        "CMAReturns",
        metadata,
        Column("Version", Integer, primary_key=True),
        Column("Type", String(64), primary_key=True),
        Column("Bench", String(64), primary_key=True),
        Column("Date", Date, primary_key=True),
        Column("Value", Float, nullable=False),
    )
    cma_stats = Table(
        "CMAStats",
        metadata,
        Column("Version", Integer, primary_key=True),
        Column("Type", String(64), primary_key=True),
        Column("Bench", String(64), primary_key=True),
        Column("Item", String(32), primary_key=True),
        Column("Value", Float, nullable=False),
    )
    core_categories = Table(
        "CoreCategories",
        metadata,
        Column("CoreCatOrder", Integer, primary_key=True),
        Column("CoreCat", String(128), nullable=False),
        Column("AssetClass", String(32), nullable=False),
        Column("FOFBench", String(128), nullable=False),
        Column("CMABench", String(64), nullable=False),
        Column("PeerBench", String(128), nullable=False),
        Column("AATool", String(64), nullable=False),
    )
    suites = Table(
        "Suites",
        metadata,
        Column("SuiteID", Integer, primary_key=True),
        Column("SuiteShort", String(32), nullable=False, unique=True),
        Column("SuiteLong", String(128), nullable=False),
        Column("IndexDailyOrder", Integer, nullable=True),
        Column("PeerTDOrder", Integer, nullable=True),
        Column("PeerModelOrder", Integer, nullable=True),
        Column("PeerAllocOrder", Integer, nullable=True),
        Column("Peer529Order", Integer, nullable=True),
    )
    portfolios = Table(
        "Portfolios",
        metadata,
        Column("PortfolioID", Integer, primary_key=True),
        Column("PortfolioName", String(128), nullable=False),
        Column("Portfolio", String(32), nullable=False, unique=True),
        Column("PortfolioSuite", String(32), ForeignKey("Suites.SuiteShort"), nullable=False),
        Column("PeerVintage", String(32), nullable=True),
        Column("PortfolioVintage", String(32), nullable=True),
        Column("IncepDate", Date, nullable=True),
    )
    peer_ts = Table(
        "PeerTS",
        metadata,
        Column("Date", Date, primary_key=True),
        Column("Portfolio", String(64), primary_key=True),
        Column("Item", String(32), primary_key=True),
        Column("Desc", String(32), primary_key=True),
        Column("Value", Float, nullable=False),
    )
    index_ts = Table(
        "IndexTS",
        metadata,
        Column("Date", Date, primary_key=True),
        Column("Portfolio", String(64), primary_key=True),
        Column("Item", String(32), primary_key=True),
        Column("Desc", String(32), primary_key=True),
        Column("Value", Float, nullable=False),
    )
    alt_ts = Table(
        "AltTS",
        metadata,
        Column("Date", Date, primary_key=True),
        Column("Portfolio", String(64), primary_key=True),
        Column("Item", String(32), primary_key=True),
        Column("Value", Float, nullable=False),
    )
    factor_defs = Table(
        "FactorDefinitions",
        metadata,
        Column("FactorName", String(128), primary_key=True),
        Column("LongComponent", String(4096), nullable=False),
        Column("ShortComponent", String(4096), nullable=True),
        Column("Description", String(4096), nullable=True),
        Column("LongAggType", Integer, nullable=False),
        Column("ShortAggType", Integer, nullable=True),
        Column("LongLag", Integer, nullable=False),
        Column("OutputTransform", Integer, nullable=False),
        Column("UPDATE_DATE", DateTime, nullable=False),
        Column("UPDATE_BY", String(128), nullable=False),
    )
    factor_defs_archive = Table(
        "FactorDefinitionsArchive",
        metadata,
        Column("FactorName", String(128), nullable=False),
        Column("LongComponent", String(4096), nullable=False),
        Column("ShortComponent", String(4096), nullable=True),
        Column("Description", String(4096), nullable=True),
        Column("LongAggType", Integer, nullable=False),
        Column("ShortAggType", Integer, nullable=True),
        Column("LongLag", Integer, nullable=False),
        Column("OutputTransform", Integer, nullable=False),
        Column("UPDATE_DATE", DateTime, nullable=False),
        Column("UPDATE_BY", String(128), nullable=False),
        Column("ARCHIVE_DATE", DateTime, nullable=False),
    )
    regime_defs = Table(
        "RegimeDefinitions",
        metadata,
        Column("RegimeName", String(128), primary_key=True),
        Column("Description", String(4000), nullable=True),
        Column("MethodType", Integer, nullable=False),
        Column("ConfigJson", String, nullable=False),
        Column("UPDATE_DATE", DateTime, nullable=False),
        Column("UPDATE_BY", String(128), nullable=False),
    )
    regime_defs_archive = Table(
        "RegimeDefinitionsArchive",
        metadata,
        Column("RegimeName", String(128), nullable=False),
        Column("Description", String(4000), nullable=True),
        Column("MethodType", Integer, nullable=False),
        Column("ConfigJson", String, nullable=False),
        Column("UPDATE_DATE", DateTime, nullable=False),
        Column("UPDATE_BY", String(128), nullable=False),
        Column("ARCHIVE_DATE", DateTime, nullable=False),
    )
    account_lists = Table(
        "DMAccountLists",
        metadata,
        Column("AccountListID", Integer, primary_key=True, autoincrement=True),
        Column("Username", String(128), nullable=False),
        Column("ListName", String(256), nullable=False),
        Column("ConfigJson", String, nullable=False),
        Column("UPDATE_DATE", DateTime, nullable=False),
        Column("UPDATE_BY", String(128), nullable=False),
    )
    account_lists_archive = Table(
        "DMAccountListsArchive",
        metadata,
        Column("AccountListID", Integer, nullable=False),
        Column("Username", String(128), nullable=False),
        Column("ListName", String(256), nullable=False),
        Column("ConfigJson", String, nullable=False),
        Column("UPDATE_DATE", DateTime, nullable=False),
        Column("UPDATE_BY", String(128), nullable=False),
        Column("ARCHIVE_DATE", DateTime, nullable=False),
    )
    users = Table(
        "Users",
        metadata,
        Column("Username", String(128), primary_key=True),
        Column("Role", String(128), nullable=False),
    )
    return (
        cma_corr,
        cma_ret,
        cma_stats,
        core_categories,
        suites,
        portfolios,
        peer_ts,
        index_ts,
        alt_ts,
        factor_defs,
        factor_defs_archive,
        regime_defs,
        regime_defs_archive,
        account_lists,
        account_lists_archive,
        users,
    )


def _build_mrd_tables(metadata: MetaData) -> tuple[Table, Table, Table, Table]:
    account = Table(
        "CORE_DATA.ACCOUNT",
        metadata,
        Column("ACCT_ID", Integer, primary_key=True),
        Column("ACCT_NAME", String(128), nullable=False),
        Column("ACCT_CD", String(128), nullable=False),
        Column("ACCT_TYPE_CD", String(32), nullable=False),
        Column("FACTOR_NAME", String(32), nullable=False),
        Column("SOURCE_SYSTEM", String(16), nullable=False),
    )
    factor_data = Table(
        "CORE_DATA.ACCOUNT_FACTOR_DATA",
        metadata,
        Column("ACCT_ID", Integer, primary_key=True),
        Column("REFERENCE_DATE", Date, primary_key=True),
        Column("FACTOR_VALUE", Float, nullable=False),
        Column("SOURCE_SYSTEM", String(16), nullable=False),
    )
    account_returns = Table(
        "CORE_DATA.ACCOUNT_RETURNS",
        metadata,
        Column("ACCT_ID", Integer, primary_key=True),
        Column("REFERENCE_DATE", Date, primary_key=True),
        Column("GROSS", Float, nullable=False),
        Column("NET", Float, nullable=False),
        Column("SOURCE_SYSTEM", String(16), nullable=False),
    )
    account_returns_m = Table(
        "CORE_DATA.ACCOUNT_RETURNS_M",
        metadata,
        Column("ACCT_ID", Integer, primary_key=True),
        Column("REFERENCE_DATE", Date, primary_key=True),
        Column("GROSS", Float, nullable=False),
        Column("NET", Float, nullable=False),
        Column("SOURCE_SYSTEM", String(16), nullable=False),
    )
    return account, factor_data, account_returns, account_returns_m


def _build_performance_tables(metadata: MetaData) -> tuple[Table, Table, Table, Table]:
    account = Table(
        "ACCOUNT",
        metadata,
        Column("ACCT_ID", Integer, primary_key=True),
        Column("ACCT_CD", String(128), nullable=False, unique=True),
    )
    account_benchmark = Table(
        "ACCOUNT_BENCHMARK",
        metadata,
        Column("BENCHMARK_ID", Integer, primary_key=True),
        Column("PRECEDENCE", Integer, nullable=False),
    )
    daily_return = Table(
        "DAILY_RETURN",
        metadata,
        Column("Effective_Date", Date, primary_key=True),
        Column("ACCT_ID", Integer, primary_key=True),
        Column("BENCHMARK_ACCT_ID", Integer, primary_key=True),
        Column("FEE_TYPE", String(8), primary_key=True),
        Column("IS_LATEST", Integer, primary_key=True),
        Column("Daily_ror", Float, nullable=False),
        Column("Daily_ror_index", Float, nullable=False),
    )
    monthly_return = Table(
        "MONTHLY_RETURN",
        metadata,
        Column("Effective_Date", Date, primary_key=True),
        Column("ACCT_ID", Integer, primary_key=True),
        Column("BENCHMARK_ACCT_ID", Integer, primary_key=True),
        Column("FEE_TYPE", String(8), primary_key=True),
        Column("IS_LATEST", Integer, primary_key=True),
        Column("Return_Type", String(16), primary_key=True),
        Column("mth1_ror", Float, nullable=False),
        Column("mth1_ror_index", Float, nullable=False),
    )
    return account, account_benchmark, daily_return, monthly_return


def _default_core_category_meta(bench: str) -> dict[str, str]:
    is_bond = bench.upper().startswith("BC")
    return {
        "CoreCat": bench,
        "AssetClass": "Bond" if is_bond else "Equity",
        "PeerBench": "Unspecified",
        "AATool": bench,
    }


def _build_account_list_seed_rows() -> list[dict]:
    def _payload(
        *,
        captured_at: str,
        entry_id: str,
        series: list[str],
        at_selected: list[str],
        po_selected: list[str],
        reg_selected: list[str],
        reg_dep: str | None,
        benchmark_map: dict[str, str] | None = None,
        po_cmabench: dict[str, str] | None = None,
    ) -> str:
        benchmark = dict(benchmark_map or {})
        po_cmabench_map = dict(po_cmabench or {})
        settings_order = list(series)
        control_values = {
            "at-series-select": list(at_selected),
            "at-series-order-store": settings_order,
            "at-benchmark-assignments-store": benchmark,
            "at-long-short-store": {},
            "at-vol-scaling-assignments-store": {name: True for name in at_selected},
            "at-periodicity-value-store": "daily_trading",
            "at-returns-type-value-store": "total",
            "at-active-tab-store": "statistics",
            "at-date-range-store": None,
            "at-vol-scaler-value-store": 0,
            "po-series-select": list(po_selected),
            "po-series-order-store": settings_order,
            "po-benchmark-assignments-store": benchmark,
            "po-cmabench-assignments-store": po_cmabench_map,
            "po-long-short-store": {},
            "po-vol-scaling-assignments-store": {name: True for name in po_selected},
            "po-min-wt-store": {name: 0.0 for name in po_selected},
            "po-max-wt-store": {name: 100.0 for name in po_selected},
            "po-force-max-store": {name: False for name in po_selected},
            "po-periodicity-value-store": "daily_trading",
            "po-vol-scaler-value-store": 0,
            "po-use-risk-free-store": True,
            "po-returns-basis-store": "total",
            "po-reporting-basis-store": False,
            "po-date-range-store": None,
            "po-opt-window-store": "rolling",
            "po-window-size-store": 252,
            "po-opt-step-store": 1,
            "po-opt-step-unit-store": "months",
            "po-opt-model-store": "risk_parity",
            "po-portfolio-name-store": "Risk Parity Portfolio",
            "po-exp-wt-cov-store": False,
            "po-halflife-store": 63,
            "po-cov-shrinkage-store": "none",
            "po-cov-shrinkage-target-store": "scaled_identity",
            "po-missing-data-store": "fill_na",
            "po-fill-in-sample-store": "off",
            "po-ex-ante-mode-store": "ret_vol_corr",
            "po-bl-tau-store": 0.05,
            "po-objective-store": "maximize_sharpe",
            "po-active-tab-store": "weight",
            "po-weight-chart-switch-store": "chart",
            "po-attribution-chart-switch-store": "chart",
            "po-risk-chart-switch-store": "chart",
            "po-turnover-chart-switch-store": "chart",
            "po-frontier-chart-switch-store": "chart",
            "reg-series-select": list(reg_selected),
            "reg-series-order-store": settings_order,
            "reg-benchmark-assignments-store": benchmark,
            "reg-long-short-store": {},
            "reg-vol-scaling-assignments-store": {name: True for name in reg_selected},
            "reg-dependent-var-store": reg_dep,
            "reg-lag-store": {name: 0 for name in reg_selected},
            "reg-min-beta-store": {name: -999.0 for name in reg_selected},
            "reg-max-beta-store": {name: 999.0 for name in reg_selected},
            "reg-enable-constraint-store": {name: False for name in reg_selected},
            "reg-periodicity-value-store": "daily",
            "reg-vol-scaler-value-store": 0,
            "reg-use-risk-free-store": True,
            "reg-date-range-store": None,
            "reg-model-store": "ols",
            "reg-regression-name-store": "OLS",
            "reg-force-zero-intercept-store": False,
            "reg-robust-se-store": False,
            "reg-exp-wt-store": False,
            "reg-halflife-store": 63,
            "reg-window-type-store": "full",
            "reg-window-size-store": 36,
            "reg-opt-step-store": 1,
            "reg-opt-step-unit-store": "months",
            "reg-fill-in-sample-store": "off",
            "reg-missing-data-store": "fill_na",
            "reg-alpha-store": 1.0,
            "reg-l1-ratio-store": 0.5,
            "reg-active-tab-store": "anova",
        }
        return canonical_json_dumps(
            {
                "schema_version": 2,
                "captured_at": captured_at,
                "series_entries": [
                    {
                        "entry_id": entry_id,
                        "loader_type": "cma_bench",
                        "loader_args": {"selected_benches": list(series)},
                        "emitted_series": list(series),
                        "primary_series": series[0] if series else None,
                    }
                ],
                "control_values": control_values,
            }
        )

    return [
        {
            "Username": LOCAL_ACCOUNT_LIST_SEED_USERNAME,
            "ListName": "Seed Core Allocation",
            "ConfigJson": _payload(
                captured_at="2026-01-15 09:30:00",
                entry_id="seed-core-allocation",
                series=[
                    "SPX_TRIndex",
                    "R2000_TRIndex",
                    "EAFE_TRIndex",
                    "BCAgg_TRIndex",
                    "BCTBill13_TRIndex",
                ],
                at_selected=[
                    "SPX_TRIndex",
                    "R2000_TRIndex",
                    "EAFE_TRIndex",
                    "BCAgg_TRIndex",
                ],
                po_selected=[
                    "SPX_TRIndex",
                    "R2000_TRIndex",
                    "EAFE_TRIndex",
                    "BCAgg_TRIndex",
                ],
                reg_selected=[
                    "R2000_TRIndex",
                    "EAFE_TRIndex",
                    "BCAgg_TRIndex",
                    "BCTBill13_TRIndex",
                ],
                reg_dep="SPX_TRIndex",
                benchmark_map={
                    "R2000_TRIndex": "SPX_TRIndex",
                    "EAFE_TRIndex": "SPX_TRIndex",
                    "BCAgg_TRIndex": "BCTBill13_TRIndex",
                },
                po_cmabench={
                    "SPX_TRIndex": "SPX",
                    "R2000_TRIndex": "R2000",
                    "EAFE_TRIndex": "EAFE",
                    "BCAgg_TRIndex": "BCAgg",
                },
            ),
            "UPDATE_DATE": datetime(2026, 1, 15, 9, 30, 0),
            "UPDATE_BY": LOCAL_ACCOUNT_LIST_SEED_UPDATE_BY,
        },
        {
            "Username": LOCAL_ACCOUNT_LIST_SEED_USERNAME,
            "ListName": "Seed Global Equity",
            "ConfigJson": _payload(
                captured_at="2026-02-05 08:45:00",
                entry_id="seed-global-equity",
                series=[
                    "SPX_TRIndex",
                    "EAFE_TRIndex",
                    "EM_TRIndex",
                    "MSCIUSREIT_TRIndex",
                ],
                at_selected=[
                    "SPX_TRIndex",
                    "EAFE_TRIndex",
                    "EM_TRIndex",
                    "MSCIUSREIT_TRIndex",
                ],
                po_selected=[
                    "SPX_TRIndex",
                    "EAFE_TRIndex",
                    "EM_TRIndex",
                ],
                reg_selected=[
                    "EAFE_TRIndex",
                    "EM_TRIndex",
                    "MSCIUSREIT_TRIndex",
                ],
                reg_dep="SPX_TRIndex",
                benchmark_map={
                    "EAFE_TRIndex": "SPX_TRIndex",
                    "EM_TRIndex": "SPX_TRIndex",
                    "MSCIUSREIT_TRIndex": "SPX_TRIndex",
                },
                po_cmabench={
                    "SPX_TRIndex": "SPX",
                    "EAFE_TRIndex": "EAFE",
                    "EM_TRIndex": "EM",
                },
            ),
            "UPDATE_DATE": datetime(2026, 2, 5, 8, 45, 0),
            "UPDATE_BY": LOCAL_ACCOUNT_LIST_SEED_UPDATE_BY,
        },
    ]


def _core_category_rows(benches: list[str]) -> list[dict]:
    rows: list[dict] = []
    for idx, bench in enumerate(sorted(benches), start=1):
        meta = CORE_CATEGORY_MAP.get(bench, _default_core_category_meta(bench))
        rows.append(
            {
                "CoreCatOrder": idx,
                "CoreCat": meta["CoreCat"],
                "AssetClass": meta["AssetClass"],
                "FOFBench": f"{bench}_TRIndex",
                "CMABench": bench,
                "PeerBench": meta["PeerBench"],
                "AATool": meta["AATool"],
            }
        )
    return rows


def _stats_rows(df: pd.DataFrame, version: int, cma_type: str) -> list[dict]:
    rows = []
    stats_df = pd.DataFrame(
        {
            "Mean": df.mean(),
            "SD": df.std(ddof=1),
            "Skewness": df.skew(),
            "Kurtosis": df.kurt(),
        }
    )
    for bench in stats_df.index:
        for item in ITEMS:
            val = float(stats_df.loc[bench, item])
            rows.append(
                {"Version": version, "Type": cma_type, "Bench": bench, "Item": item, "Value": val}
            )
    return rows


def _correlation_rows(df: pd.DataFrame, version: int, cma_type: str) -> list[dict]:
    rows = []
    corr = df.corr()
    benches = list(corr.index)
    # Store only one triangle (including diagonal). Consumers must treat as symmetric.
    for i, bench1 in enumerate(benches):
        for j in range(i, len(benches)):
            bench2 = benches[j]
            rows.append(
                {
                    "Version": version,
                    "Type": cma_type,
                    "Bench1": bench1,
                    "Bench2": bench2,
                    "Value": float(corr.loc[bench1, bench2]),
                }
            )
    return rows


def _returns_rows(df: pd.DataFrame, version: int, cma_type: str) -> list[dict]:
    rows = []
    for dt, row in df.iterrows():
        dt_date = date(dt.year, dt.month, dt.day)
        for bench, val in row.items():
            if pd.isna(val):
                continue
            rows.append(
                {
                    "Version": version,
                    "Type": cma_type,
                    "Bench": bench,
                    "Date": dt_date,
                    "Value": float(val),
                }
            )
    return rows


def _mrd_rows(df: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    account_rows: list[dict] = []
    factor_rows: list[dict] = []

    for acct_id, series_name in enumerate(df.columns, start=1):
        account_rows.append(
            {
                "ACCT_ID": acct_id,
                "ACCT_NAME": str(series_name),
                "ACCT_CD": f"{series_name}_TRIndex",
                "ACCT_TYPE_CD": "SEC_FACTOR",
                "FACTOR_NAME": "TRIndex",
                "SOURCE_SYSTEM": "BB",
            }
        )
        series = df[series_name].dropna()
        if series.empty:
            continue
        index_series = (1.0 + series).cumprod()
        for dt, idx_val in index_series.items():
            factor_rows.append(
                {
                    "ACCT_ID": acct_id,
                    "REFERENCE_DATE": date(dt.year, dt.month, dt.day),
                    "FACTOR_VALUE": float(idx_val),
                    "SOURCE_SYSTEM": "BB",
                }
            )

    # Additional factor variations for raw-factor import options.
    if not df.empty:
        idx = pd.DatetimeIndex(df.index).sort_values()
        source_df = df.reindex(idx)
        spx = _pick_seed_column(source_df, "SPX", 0)
        agg = _pick_seed_column(source_df, "BCAgg", 1)
        x = np.arange(len(idx), dtype=float)

        extra_specs: list[tuple[str, str, str, str, pd.Series]] = [
            (
                "SPX",
                "SPXGR Index",
                "SEC_FACTOR",
                "GRIndex",
                _returns_to_levels(0.90 * spx + 0.10 * agg + 0.00003 * np.sin(0.05 * x), start_level=100.0),
            ),
            (
                "UST10Y",
                "USGG10YR Index",
                "SEC_FACTOR",
                "Yield",
                pd.Series(3.2 + 0.45 * np.sin(0.017 * x) + 0.22 * np.cos(0.043 * x), index=idx, dtype=float),
            ),
            (
                "EURUSD",
                "EURUSD Curncy",
                "SEC_FACTOR",
                "Spot",
                pd.Series(1.12 + 0.06 * np.sin(0.012 * x) + 0.03 * np.cos(0.033 * x), index=idx, dtype=float),
            ),
            (
                "PERF_EXCL",
                "Perf Excluded Index",
                "SEC_FACTOR",
                "TRIndex",
                _returns_to_levels(0.30 * spx + 0.70 * agg + 0.00004 * np.sin(0.07 * x), start_level=100.0),
            ),
        ]

        next_id = len(account_rows) + 1
        for acct_name, acct_cd, acct_type, factor_name, series in extra_specs:
            source_system = "PERF" if acct_name == "PERF_EXCL" else "BB"
            account_rows.append(
                {
                    "ACCT_ID": next_id,
                    "ACCT_NAME": acct_name,
                    "ACCT_CD": acct_cd,
                    "ACCT_TYPE_CD": acct_type,
                    "FACTOR_NAME": factor_name,
                    "SOURCE_SYSTEM": source_system,
                }
            )
            clean = pd.to_numeric(series, errors="coerce").dropna()
            for dt, value in clean.items():
                factor_rows.append(
                    {
                        "ACCT_ID": next_id,
                        "REFERENCE_DATE": date(dt.year, dt.month, dt.day),
                        "FACTOR_VALUE": float(value),
                        "SOURCE_SYSTEM": source_system,
                    }
                )
            next_id += 1

    return account_rows, factor_rows


def _build_fund_seed_rows(daily_df: pd.DataFrame, start_acct_id: int) -> tuple[list[dict], list[dict], list[dict]]:
    """Build deterministic MSTAR fund account + daily/monthly return rows."""
    if daily_df.empty:
        return [], [], []

    idx = _seed_daily_index(daily_df)
    if len(idx) == 0:
        return [], [], []
    source_df = daily_df.reindex(idx)
    spx = _pick_seed_column(source_df, "SPX", 0)
    agg = _pick_seed_column(source_df, "BCAgg", 1)
    em = _pick_seed_column(source_df, "EM", 2)
    x = np.arange(len(idx), dtype=float)

    specs = [
        {
            "name": "MStar Growth Opportunities",
            "code": "MGROWTH",
            "acct_type": "OE",
            "gross": 0.70 * spx + 0.15 * em + 0.15 * agg + 0.00006 * np.sin(0.09 * x),
            "fee": 0.00008,
        },
        {
            "name": "MStar Defensive Sleeve",
            "code": "MDEFSLV",
            "acct_type": "SLEEVE",
            "gross": 0.25 * spx + 0.05 * em + 0.70 * agg + 0.00004 * np.cos(0.07 * x),
            "fee": 0.00004,
        },
        {
            "name": "MStar Income Trust",
            "code": "MINCTRST",
            "acct_type": "TRUST",
            "gross": 0.10 * spx + 0.10 * em + 0.80 * agg + 0.00003 * np.sin(0.05 * x),
            "fee": 0.00002,
        },
    ]

    account_rows: list[dict] = []
    daily_rows: list[dict] = []
    monthly_rows: list[dict] = []

    for i, spec in enumerate(specs):
        acct_id = int(start_acct_id + i)
        account_rows.append(
            {
                "ACCT_ID": acct_id,
                "ACCT_NAME": spec["name"],
                "ACCT_CD": spec["code"],
                "ACCT_TYPE_CD": spec["acct_type"],
                "FACTOR_NAME": "Ret",
                "SOURCE_SYSTEM": "MSTAR",
            }
        )

        gross_series = pd.Series(spec["gross"], index=idx, dtype=float)
        net_series = gross_series - float(spec["fee"])

        for dt in idx:
            daily_rows.append(
                {
                    "ACCT_ID": acct_id,
                    "REFERENCE_DATE": date(dt.year, dt.month, dt.day),
                    "GROSS": float(gross_series.loc[dt]),
                    "NET": float(net_series.loc[dt]),
                    "SOURCE_SYSTEM": "MSTAR",
                }
            )

        gross_m = gross_series.resample("ME").apply(lambda xvals: (1.0 + xvals).prod() - 1.0)
        net_m = net_series.resample("ME").apply(lambda xvals: (1.0 + xvals).prod() - 1.0)
        for dt in gross_m.index:
            monthly_rows.append(
                {
                    "ACCT_ID": acct_id,
                    "REFERENCE_DATE": date(dt.year, dt.month, dt.day),
                    "GROSS": float(gross_m.loc[dt]),
                    "NET": float(net_m.loc[dt]),
                    "SOURCE_SYSTEM": "MSTAR",
                }
            )

    # Negative controls: rows excluded by filters.
    if account_rows:
        account_rows.append(
            {
                "ACCT_ID": int(start_acct_id + len(specs)),
                "ACCT_NAME": "Non-MSTAR Excluded Fund",
                "ACCT_CD": "XNONMSTAR",
                "ACCT_TYPE_CD": "OE",
                "FACTOR_NAME": "Ret",
                "SOURCE_SYSTEM": "OTHER",
            }
        )

    return account_rows, daily_rows, monthly_rows


def main() -> None:
    base_df = _load_monthly_returns()
    daily_df = _load_daily_returns()
    if RISK_FREE_BENCH not in daily_df.columns:
        daily_df[RISK_FREE_BENCH] = _build_bctbill13_proxy_returns(daily_df)
    mth_to_dly = _build_mth_to_dly_returns(daily_df)
    if not mth_to_dly.empty:
        daily_df = daily_df.join(mth_to_dly, how="outer")
    daily_df = daily_df.sort_index()
    metadata = MetaData()
    (
        cma_corr,
        cma_ret,
        cma_stats,
        core_categories,
        suites,
        portfolios,
        peer_ts,
        index_ts,
        alt_ts,
        factor_defs,
        factor_defs_archive,
        regime_defs,
        regime_defs_archive,
        account_lists,
        account_lists_archive,
        users,
    ) = _build_tables(metadata)
    mrd_metadata = MetaData()
    mrd_account, mrd_factor_data, mrd_account_returns, mrd_account_returns_m = _build_mrd_tables(mrd_metadata)
    perf_metadata = MetaData()
    perf_account, perf_account_benchmark, perf_daily_return, perf_monthly_return = _build_performance_tables(perf_metadata)
    account_list_seed_rows = _build_account_list_seed_rows()

    metadata.drop_all(engine, checkfirst=True)
    metadata.create_all(engine)
    with engine_MRD.begin() as conn:
        conn.execute(text('DROP TABLE IF EXISTS [CORE_DATA.FACTOR_DATA]'))
    mrd_metadata.drop_all(engine_MRD, checkfirst=True)
    mrd_metadata.create_all(engine_MRD)
    perf_metadata.drop_all(engine_PERFORMANCE, checkfirst=True)
    perf_metadata.create_all(engine_PERFORMANCE)

    # Query-supporting indexes for raw import workflows.
    if engine_MRD.dialect.name == "sqlite":
        with engine_MRD.begin() as conn:
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_core_account_filters "
                    "ON [CORE_DATA.ACCOUNT] (ACCT_TYPE_CD, SOURCE_SYSTEM, ACCT_NAME, FACTOR_NAME)"
                )
            )
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_core_factor_acct_date "
                    "ON [CORE_DATA.ACCOUNT_FACTOR_DATA] (ACCT_ID, REFERENCE_DATE)"
                )
            )
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_core_returns_acct_date "
                    "ON [CORE_DATA.ACCOUNT_RETURNS] (ACCT_ID, REFERENCE_DATE, SOURCE_SYSTEM)"
                )
            )
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_core_returns_m_acct_date "
                    "ON [CORE_DATA.ACCOUNT_RETURNS_M] (ACCT_ID, REFERENCE_DATE, SOURCE_SYSTEM)"
                )
            )
    if engine_PERFORMANCE.dialect.name == "sqlite":
        with engine_PERFORMANCE.begin() as conn:
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_perf_daily_filters "
                    "ON [DAILY_RETURN] (ACCT_ID, Effective_Date, FEE_TYPE, IS_LATEST, BENCHMARK_ACCT_ID)"
                )
            )
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_perf_monthly_filters "
                    "ON [MONTHLY_RETURN] (ACCT_ID, Effective_Date, FEE_TYPE, IS_LATEST, Return_Type, BENCHMARK_ACCT_ID)"
                )
            )
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_perf_benchmark_precedence "
                    "ON [ACCOUNT_BENCHMARK] (BENCHMARK_ID, PRECEDENCE)"
                )
            )

    if engine.dialect.name == "sqlite":
        with engine.begin() as conn:
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_factor_defs_name "
                    "ON [FactorDefinitions] (FactorName)"
                )
            )
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_factor_defs_archive_name_date "
                    "ON [FactorDefinitionsArchive] (FactorName, ARCHIVE_DATE)"
                )
            )
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_regime_defs_name "
                    "ON [RegimeDefinitions] (RegimeName)"
                )
            )
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_regime_defs_archive_name_date "
                    "ON [RegimeDefinitionsArchive] (RegimeName, ARCHIVE_DATE)"
                )
            )

    corr_rows: list[dict] = []
    ret_rows: list[dict] = []
    stats_rows: list[dict] = []

    for version in VERSIONS:
        cutoff = pd.Timestamp(f"{version}-12-31")
        version_df = base_df.loc[base_df.index <= cutoff].copy()
        for cma_type in TYPES:
            typed_df = _transform_for_type(version_df, cma_type)
            corr_rows.extend(_correlation_rows(typed_df, version, cma_type))
            ret_rows.extend(_returns_rows(typed_df, version, cma_type))
            stats_rows.extend(_stats_rows(typed_df, version, cma_type))

    core_cat_benches = sorted({str(r["Bench"]) for r in ret_rows if r.get("Bench") is not None})
    if RISK_FREE_BENCH not in core_cat_benches:
        core_cat_benches = sorted(core_cat_benches + [RISK_FREE_BENCH])
    if MTH_TO_DLY_BENCH not in core_cat_benches:
        core_cat_benches = sorted(core_cat_benches + [MTH_TO_DLY_BENCH])
    core_cat_rows = _core_category_rows(core_cat_benches)
    peer_series_map, index_series_map = _build_portfolio_seed_series(daily_df)
    underlying_peer_rows = build_underlying_category_seed_rows(daily_df)
    alt_series_map = _build_alt_seed_series(daily_df)
    peer_ts_rows = _portfolio_ts_rows(peer_series_map)
    peer_ts_rows.extend(underlying_peer_rows)
    index_ts_rows = _portfolio_ts_rows(index_series_map)
    alt_ts_rows = _alt_ts_rows(alt_series_map)
    mrd_account_rows, mrd_factor_rows = _mrd_rows(daily_df)
    fund_account_rows, fund_daily_rows, fund_monthly_rows = _build_fund_seed_rows(
        daily_df,
        start_acct_id=(max([r["ACCT_ID"] for r in mrd_account_rows], default=0) + 1),
    )
    mrd_account_rows.extend(fund_account_rows)
    perf_account_rows, perf_benchmark_rows, perf_daily_rows, perf_monthly_rows = _build_perf_seed_rows(daily_df)

    with engine.begin() as conn:
        conn.execute(cma_corr.insert(), corr_rows)
        conn.execute(cma_ret.insert(), ret_rows)
        conn.execute(cma_stats.insert(), stats_rows)
        if core_cat_rows:
            conn.execute(core_categories.insert(), core_cat_rows)
        if SUITE_ROWS:
            conn.execute(suites.insert(), SUITE_ROWS)
        if PORTFOLIO_ROWS:
            conn.execute(portfolios.insert(), PORTFOLIO_ROWS)
        if peer_ts_rows:
            conn.execute(peer_ts.insert(), peer_ts_rows)
        if index_ts_rows:
            conn.execute(index_ts.insert(), index_ts_rows)
        if alt_ts_rows:
            conn.execute(alt_ts.insert(), alt_ts_rows)
        if account_list_seed_rows:
            conn.execute(account_lists.insert(), account_list_seed_rows)
        if LOCAL_USER_SEED_ROWS:
            conn.execute(users.insert(), LOCAL_USER_SEED_ROWS)

    with engine_MRD.begin() as conn:
        if mrd_account_rows:
            conn.execute(mrd_account.insert(), mrd_account_rows)
        if mrd_factor_rows:
            conn.execute(mrd_factor_data.insert(), mrd_factor_rows)
        if fund_daily_rows:
            conn.execute(mrd_account_returns.insert(), fund_daily_rows)
        if fund_monthly_rows:
            conn.execute(mrd_account_returns_m.insert(), fund_monthly_rows)

    with engine_PERFORMANCE.begin() as conn:
        if perf_account_rows:
            conn.execute(perf_account.insert(), perf_account_rows)
        if perf_benchmark_rows:
            conn.execute(perf_account_benchmark.insert(), perf_benchmark_rows)
        if perf_daily_rows:
            conn.execute(perf_daily_return.insert(), perf_daily_rows)
        if perf_monthly_rows:
            conn.execute(perf_monthly_return.insert(), perf_monthly_rows)

    factor_seed_stats = ensure_factor_definition_tables_and_seed(
        engine,
        engine_MRD,
        update_by="init_local_cma_db.py",
    )
    ensure_account_list_tables(engine)
    ensure_users_table(engine)
    with engine.connect() as conn:
        factor_def_count = int(conn.execute(text("SELECT COUNT(*) FROM FactorDefinitions")).scalar_one())
        factor_archive_count = int(conn.execute(text("SELECT COUNT(*) FROM FactorDefinitionsArchive")).scalar_one())
        regime_def_count = int(conn.execute(text("SELECT COUNT(*) FROM RegimeDefinitions")).scalar_one())
        regime_archive_count = int(conn.execute(text("SELECT COUNT(*) FROM RegimeDefinitionsArchive")).scalar_one())
        account_list_count = int(conn.execute(text("SELECT COUNT(*) FROM DMAccountLists")).scalar_one())
        account_list_archive_count = int(conn.execute(text("SELECT COUNT(*) FROM DMAccountListsArchive")).scalar_one())
        users_count = int(conn.execute(text("SELECT COUNT(*) FROM Users")).scalar_one())

    print(f"Initialized CMA database at {DATABASE_URL}")
    print(f"CMACorrelation rows: {len(corr_rows)}")
    print(f"CMAReturns rows: {len(ret_rows)}")
    print(f"CMAStats rows: {len(stats_rows)}")
    print(f"CoreCategories rows: {len(core_cat_rows)}")
    print(f"Suites rows: {len(SUITE_ROWS)}")
    print(f"Portfolios rows: {len(PORTFOLIO_ROWS)}")
    print(f"PeerTS rows: {len(peer_ts_rows)}")
    print(f"IndexTS rows: {len(index_ts_rows)}")
    print(f"AltTS rows: {len(alt_ts_rows)}")
    print(f"FactorDefinitions rows: {factor_def_count}")
    print(f"FactorDefinitionsArchive rows: {factor_archive_count}")
    print(f"RegimeDefinitions rows: {regime_def_count}")
    print(f"RegimeDefinitionsArchive rows: {regime_archive_count}")
    print(f"DMAccountLists rows: {account_list_count}")
    print(f"DMAccountListsArchive rows: {account_list_archive_count}")
    print(f"DMAccountLists seed rows: {len(account_list_seed_rows)}")
    print(f"Users rows: {users_count}")
    print(
        "FactorDefinitions seed stats: "
        f"inserted={factor_seed_stats['inserted']}, "
        f"updated={factor_seed_stats['updated']}, "
        f"archived={factor_seed_stats['archived']}, "
        f"unchanged={factor_seed_stats['unchanged']}, "
        f"skipped={factor_seed_stats['skipped']}, "
        f"tokens={factor_seed_stats['token_count']}"
    )
    print(f"Initialized MRD database at {MRD_DATABASE_URL}")
    print(f"CORE_DATA.ACCOUNT rows: {len(mrd_account_rows)}")
    print(f"CORE_DATA.ACCOUNT_FACTOR_DATA rows: {len(mrd_factor_rows)}")
    print(f"CORE_DATA.ACCOUNT_RETURNS rows: {len(fund_daily_rows)}")
    print(f"CORE_DATA.ACCOUNT_RETURNS_M rows: {len(fund_monthly_rows)}")
    print(f"Initialized Performance database at {PERFORMANCE_DATABASE_URL}")
    print(f"ACCOUNT rows: {len(perf_account_rows)}")
    print(f"ACCOUNT_BENCHMARK rows: {len(perf_benchmark_rows)}")
    print(f"DAILY_RETURN rows: {len(perf_daily_rows)}")
    print(f"MONTHLY_RETURN rows: {len(perf_monthly_rows)}")


if __name__ == "__main__":
    main()
