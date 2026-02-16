"""Initialize and populate local CMA test database."""

from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sqlalchemy import Column, Date, Float, ForeignKey, Integer, MetaData, String, Table, text

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dbengine import engine, engine_MRD, DATABASE_URL, MRD_DATABASE_URL
from utils.sample_data import get_sample_file_path


VERSIONS = [2025, 2026]
TYPES = ["hmm", "equilibrium.gp"]
ITEMS = ["Mean", "SD", "Skewness", "Kurtosis"]
RISK_FREE_BENCH = "BCTBill13"
MTH_TO_DLY_BENCH = "MthToDly"
MTH_TO_DLY_DAILY_START = pd.Timestamp("2022-01-03")
FRED_DGS3MO_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO"

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
        "IndexMonthlyOrder": None,
        "PeerTDOrder": 1,
        "PeerModelOrder": None,
        "PeerAllocOrder": None,
        "Peer529Order": None,
    },
    {
        "SuiteID": 2,
        "SuiteShort": "RISK",
        "SuiteLong": "Risk-Based",
        "IndexMonthlyOrder": 1,
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
        "IncepDate": date(2012, 1, 31),
    },
    {
        "PortfolioID": 5,
        "PortfolioName": "Target Date 2030 Select Fund",
        "Portfolio": "TD2030S",
        "PortfolioSuite": "TD",
        "PeerVintage": "2030",
        "IncepDate": date(2015, 1, 31),
    },
    {
        "PortfolioID": 2,
        "PortfolioName": "Target Date 2050 Fund",
        "Portfolio": "TD2050",
        "PortfolioSuite": "TD",
        "PeerVintage": "2050",
        "IncepDate": date(2014, 1, 31),
    },
    {
        "PortfolioID": 3,
        "PortfolioName": "Risk Balanced 60 Fund",
        "Portfolio": "Risk60",
        "PortfolioSuite": "RISK",
        "PeerVintage": "",
        "IncepDate": date(2011, 1, 31),
    },
    {
        "PortfolioID": 4,
        "PortfolioName": "Risk Balanced 80 Fund",
        "Portfolio": "Risk80",
        "PortfolioSuite": "RISK",
        "PeerVintage": "",
        "IncepDate": date(2013, 1, 31),
    },
]


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
    return df


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


def _build_portfolio_seed_series(base_df: pd.DataFrame) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    """Build deterministic portfolio and benchmark series for PeerTS/IndexTS."""
    if base_df.empty:
        return {}, {}

    idx = pd.DatetimeIndex(base_df.index).sort_values()
    spx = _pick_seed_column(base_df.reindex(idx), "SPX", 0)
    agg = _pick_seed_column(base_df.reindex(idx), "BCAgg", 1)

    peer_series: dict[str, pd.Series] = {
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
    }

    index_series: dict[str, pd.Series] = {
        "Risk60|PortRet|Actual": 0.60 * spx + 0.40 * agg,
        "Risk60|PortRet|Calculated": 0.98 * (0.60 * spx + 0.40 * agg),
        "Risk60|PortRet|Benchmark": 0.57 * spx + 0.43 * agg,
        "Risk80|PortRet|Actual": 0.80 * spx + 0.20 * agg,
        "Risk80|PortRet|Calculated": 0.98 * (0.80 * spx + 0.20 * agg),
        "Risk80|PortRet|Benchmark": 0.77 * spx + 0.23 * agg,
    }

    for key, series in list(peer_series.items()):
        peer_series[key] = pd.Series(series.values, index=idx, dtype=float)
    for key, series in list(index_series.items()):
        index_series[key] = pd.Series(series.values, index=idx, dtype=float)
    return peer_series, index_series


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


def _build_tables(metadata: MetaData) -> tuple[Table, Table, Table, Table, Table, Table, Table, Table]:
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
        Column("IndexMonthlyOrder", Integer, nullable=True),
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
    return cma_corr, cma_ret, cma_stats, core_categories, suites, portfolios, peer_ts, index_ts


def _build_mrd_tables(metadata: MetaData) -> tuple[Table, Table]:
    account = Table(
        "CORE_DATA.ACCOUNT",
        metadata,
        Column("ACCT_ID", Integer, primary_key=True),
        Column("ACCT_NAME", String(128), nullable=False),
        Column("ACCT_CD", String(128), nullable=False),
        Column("ACCT_TYPE_CD", String(32), nullable=False),
        Column("FACTOR_NAME", String(32), nullable=False),
    )
    factor_data = Table(
        "CORE_DATA.ACCOUNT_FACTOR_DATA",
        metadata,
        Column("ACCT_ID", Integer, primary_key=True),
        Column("REFERENCE_DATE", Date, primary_key=True),
        Column("FACTOR_VALUE", Float, nullable=False),
        Column("SOURCE_SYSTEM", String(16), nullable=False),
    )
    return account, factor_data


def _default_core_category_meta(bench: str) -> dict[str, str]:
    is_bond = bench.upper().startswith("BC")
    return {
        "CoreCat": bench,
        "AssetClass": "Bond" if is_bond else "Equity",
        "PeerBench": "Unspecified",
        "AATool": bench,
    }


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
                "ACCT_TYPE_CD": "INDEX",
                "FACTOR_NAME": "TRIndex",
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

    return account_rows, factor_rows


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
    cma_corr, cma_ret, cma_stats, core_categories, suites, portfolios, peer_ts, index_ts = _build_tables(metadata)
    mrd_metadata = MetaData()
    mrd_account, mrd_factor_data = _build_mrd_tables(mrd_metadata)

    metadata.drop_all(engine, checkfirst=True)
    metadata.create_all(engine)
    with engine_MRD.begin() as conn:
        conn.execute(text('DROP TABLE IF EXISTS [CORE_DATA.FACTOR_DATA]'))
    mrd_metadata.drop_all(engine_MRD, checkfirst=True)
    mrd_metadata.create_all(engine_MRD)

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
    peer_series_map, index_series_map = _build_portfolio_seed_series(base_df)
    peer_ts_rows = _portfolio_ts_rows(peer_series_map)
    index_ts_rows = _portfolio_ts_rows(index_series_map)
    mrd_account_rows, mrd_factor_rows = _mrd_rows(daily_df)

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

    with engine_MRD.begin() as conn:
        if mrd_account_rows:
            conn.execute(mrd_account.insert(), mrd_account_rows)
        if mrd_factor_rows:
            conn.execute(mrd_factor_data.insert(), mrd_factor_rows)

    print(f"Initialized CMA database at {DATABASE_URL}")
    print(f"CMACorrelation rows: {len(corr_rows)}")
    print(f"CMAReturns rows: {len(ret_rows)}")
    print(f"CMAStats rows: {len(stats_rows)}")
    print(f"CoreCategories rows: {len(core_cat_rows)}")
    print(f"Suites rows: {len(SUITE_ROWS)}")
    print(f"Portfolios rows: {len(PORTFOLIO_ROWS)}")
    print(f"PeerTS rows: {len(peer_ts_rows)}")
    print(f"IndexTS rows: {len(index_ts_rows)}")
    print(f"Initialized MRD database at {MRD_DATABASE_URL}")
    print(f"CORE_DATA.ACCOUNT rows: {len(mrd_account_rows)}")
    print(f"CORE_DATA.ACCOUNT_FACTOR_DATA rows: {len(mrd_factor_rows)}")


if __name__ == "__main__":
    main()
