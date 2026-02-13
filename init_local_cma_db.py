"""Initialize and populate local CMA test database."""

from __future__ import annotations

from datetime import date

import pandas as pd
from sqlalchemy import Date, Float, Integer, MetaData, String, Table, Column, text

from dbengine import engine, engine_MRD, DATABASE_URL, MRD_DATABASE_URL
from utils.sample_data import get_sample_file_path


VERSIONS = [2025, 2026]
TYPES = ["hmm", "equilibrium.gp"]
ITEMS = ["Mean", "SD", "Skewness", "Kurtosis"]

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
    return df


def _transform_for_type(df: pd.DataFrame, cma_type: str) -> pd.DataFrame:
    if cma_type == "hmm":
        return df.copy()
    # Equilibrium-style: slight mean and volatility dampening versus historical.
    return df * 0.90


def _build_tables(metadata: MetaData) -> tuple[Table, Table, Table, Table]:
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
    return cma_corr, cma_ret, cma_stats, core_categories


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
    metadata = MetaData()
    cma_corr, cma_ret, cma_stats, core_categories = _build_tables(metadata)
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

    core_cat_rows = _core_category_rows(
        sorted({str(r["Bench"]) for r in ret_rows if r.get("Bench") is not None})
    )
    mrd_account_rows, mrd_factor_rows = _mrd_rows(daily_df)

    with engine.begin() as conn:
        conn.execute(cma_corr.insert(), corr_rows)
        conn.execute(cma_ret.insert(), ret_rows)
        conn.execute(cma_stats.insert(), stats_rows)
        if core_cat_rows:
            conn.execute(core_categories.insert(), core_cat_rows)

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
    print(f"Initialized MRD database at {MRD_DATABASE_URL}")
    print(f"CORE_DATA.ACCOUNT rows: {len(mrd_account_rows)}")
    print(f"CORE_DATA.ACCOUNT_FACTOR_DATA rows: {len(mrd_factor_rows)}")


if __name__ == "__main__":
    main()
