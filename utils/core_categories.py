"""Read-only CoreCategories and CMA returns helpers for UI imports."""

from __future__ import annotations

import pandas as pd
from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine


def get_core_category_options(engine: Engine) -> list[dict]:
    """Return dropdown options formatted as `CoreCat [CMABench]`."""
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT CoreCat, CMABench FROM ("
                "  SELECT CoreCat, CMABench, "
                "         ROW_NUMBER() OVER(PARTITION BY CMABench ORDER BY CoreCatOrder) AS rn "
                "  FROM CoreCategories "
                "  WHERE AATool IS NOT NULL AND CMABench IS NOT NULL AND FOFBench LIKE '%_TRIndex'"
                ") t WHERE rn = 1 "
                "ORDER BY CoreCat"
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
        selected_type = (
            "hmm"
            if "hmm" in available_types
            else (sorted(available_types)[0] if available_types else None)
        )
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
    cols = [c for c in selected if c in wide.columns]
    wide = wide.reindex(columns=cols)
    wide = wide.sort_index()
    wide.index.name = "Date"
    return wide
