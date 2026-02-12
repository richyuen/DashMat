"""Core category helpers for DB-backed series import."""

from __future__ import annotations

import pandas as pd
from sqlalchemy import Column, Integer, MetaData, String, Table, bindparam, text
from sqlalchemy.engine import Engine


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


def _core_categories_table(metadata: MetaData) -> Table:
    return Table(
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


def _default_meta(bench: str) -> dict[str, str]:
    is_bond = bench.upper().startswith("BC")
    return {
        "CoreCat": bench,
        "AssetClass": "Bond" if is_bond else "Equity",
        "PeerBench": "Unspecified",
        "AATool": bench,
    }


def _build_rows_from_benches(benches: list[str]) -> list[dict]:
    ordered_benches = sorted(benches)
    rows: list[dict] = []
    for idx, bench in enumerate(ordered_benches, start=1):
        meta = CORE_CATEGORY_MAP.get(bench, _default_meta(bench))
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


def ensure_core_categories_table(engine: Engine) -> None:
    """Create and repopulate CoreCategories from available CMAReturns benches."""
    metadata = MetaData()
    table = _core_categories_table(metadata)
    metadata.create_all(engine, tables=[table], checkfirst=True)

    with engine.begin() as conn:
        benches = [
            str(r[0])
            for r in conn.execute(
                text("SELECT DISTINCT Bench FROM CMAReturns ORDER BY Bench")
            ).fetchall()
            if r[0] is not None
        ]
        conn.execute(text("DELETE FROM CoreCategories"))
        if benches:
            conn.execute(table.insert(), _build_rows_from_benches(benches))


def get_core_category_options(engine: Engine) -> list[dict]:
    """Return dropdown options formatted as `CoreCat [CMABench]`."""
    ensure_core_categories_table(engine)
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT CoreCat, CMABench "
                "FROM CoreCategories "
                "ORDER BY CoreCatOrder"
            )
        ).fetchall()
    return [
        {"value": str(cmabench), "label": f"{str(corecat)} [{str(cmabench)}]"}
        for corecat, cmabench in rows
    ]


def load_cma_returns_for_benches(engine: Engine, benches: list[str]) -> pd.DataFrame:
    """Load CMAReturns for selected benches as Date-indexed DataFrame."""
    selected = [str(b) for b in benches if b]
    if not selected:
        return pd.DataFrame()

    with engine.connect() as conn:
        version = conn.execute(
            text("SELECT MAX(Version) FROM CMAReturns")
        ).scalar_one_or_none()
        if version is None:
            return pd.DataFrame()

        type_rows = conn.execute(
            text(
                "SELECT DISTINCT Type "
                "FROM CMAReturns "
                "WHERE Version = :v"
            ),
            {"v": int(version)},
        ).fetchall()
        available_types = {str(r[0]) for r in type_rows if r[0] is not None}
        selected_type = "hmm" if "hmm" in available_types else (sorted(available_types)[0] if available_types else None)
        if selected_type is None:
            return pd.DataFrame()

    q = text(
        "SELECT Date, Bench, Value "
        "FROM CMAReturns "
        "WHERE Version = :v AND Type = :t AND Bench IN :benches "
        "ORDER BY Date, Bench"
    ).bindparams(bindparam("benches", expanding=True))

    with engine.connect() as conn:
        rows = conn.execute(
            q,
            {"v": int(version), "t": selected_type, "benches": selected},
        ).fetchall()

    if not rows:
        return pd.DataFrame()

    data = pd.DataFrame(rows, columns=["Date", "Bench", "Value"])
    data["Date"] = pd.to_datetime(data["Date"])
    wide = data.pivot(index="Date", columns="Bench", values="Value")
    # Keep selected order when possible.
    cols = [c for c in selected if c in wide.columns]
    wide = wide.reindex(columns=cols)
    wide = wide.sort_index()
    wide.index.name = "Date"
    return wide
