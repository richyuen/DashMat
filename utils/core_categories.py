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


def _mrd_account_table(engine: Engine) -> str:
    if engine.dialect.name == "sqlite":
        return "[CORE_DATA.ACCOUNT]"
    return "[CORE_DATA].[ACCOUNT]"


def _mrd_factor_table(engine: Engine) -> str:
    if engine.dialect.name == "sqlite":
        return "[CORE_DATA.ACCOUNT_FACTOR_DATA]"
    return "[CORE_DATA].[ACCOUNT_FACTOR_DATA]"


def _split_fofbench(fofbench: str) -> tuple[str, str]:
    if "_" not in fofbench:
        return fofbench, "TRIndex"
    acct_name, factor_name = fofbench.rsplit("_", 1)
    return acct_name, factor_name


def load_cma_returns_for_benches(
    core_engine: Engine,
    benches: list[str],
    mrd_engine: Engine,
) -> pd.DataFrame:
    """Load selected CMABench daily returns from MRD index levels."""
    selected = [str(b) for b in benches if b]
    if not selected:
        return pd.DataFrame()

    core_query = text(
        "SELECT CMABench, FOFBench "
        "FROM CoreCategories "
        "WHERE CMABench IN :benches "
        "ORDER BY CoreCatOrder"
    ).bindparams(bindparam("benches", expanding=True))
    with core_engine.connect() as conn:
        mapping_rows = conn.execute(
            core_query, {"benches": selected}
        ).fetchall()
    if not mapping_rows:
        return pd.DataFrame()

    cmabench_to_fofbench = {str(r[0]): str(r[1]) for r in mapping_rows if r[0] and r[1]}
    selected_cmabenches = [b for b in selected if b in cmabench_to_fofbench]
    if not selected_cmabenches:
        return pd.DataFrame()

    fofbench_to_cmabench = {v: k for k, v in cmabench_to_fofbench.items()}
    acct_names = sorted({_split_fofbench(v)[0] for v in fofbench_to_cmabench})
    factor_names = sorted({_split_fofbench(v)[1] for v in fofbench_to_cmabench})
    if not acct_names or not factor_names:
        return pd.DataFrame()

    account_table = _mrd_account_table(mrd_engine)
    factor_table = _mrd_factor_table(mrd_engine)
    account_query = text(
        f"SELECT ACCT_ID, ACCT_NAME, FACTOR_NAME "
        f"FROM {account_table} "
        f"WHERE ACCT_NAME IN :acct_names AND FACTOR_NAME IN :factor_names"
    ).bindparams(
        bindparam("acct_names", expanding=True),
        bindparam("factor_names", expanding=True),
    )
    with mrd_engine.connect() as conn:
        account_rows = conn.execute(
            account_query,
            {"acct_names": acct_names, "factor_names": factor_names},
        ).fetchall()
    if not account_rows:
        return pd.DataFrame()

    fofbench_to_acct_id: dict[str, int] = {}
    for acct_id, acct_name, factor_name in account_rows:
        key = f"{acct_name}_{factor_name}"
        if key in fofbench_to_cmabench and key not in fofbench_to_acct_id:
            fofbench_to_acct_id[key] = int(acct_id)

    cmabench_to_acct_id = {
        cmabench: fofbench_to_acct_id[fofbench]
        for cmabench, fofbench in cmabench_to_fofbench.items()
        if fofbench in fofbench_to_acct_id
    }
    if not cmabench_to_acct_id:
        return pd.DataFrame()

    acct_ids = sorted(set(cmabench_to_acct_id.values()))
    factor_query = text(
        f"SELECT ACCT_ID, REFERENCE_DATE, FACTOR_VALUE "
        f"FROM {factor_table} "
        f"WHERE ACCT_ID IN :acct_ids "
        f"ORDER BY REFERENCE_DATE, ACCT_ID"
    ).bindparams(bindparam("acct_ids", expanding=True))

    with mrd_engine.connect() as conn:
        rows = conn.execute(
            factor_query,
            {"acct_ids": acct_ids},
        ).fetchall()

    if not rows:
        return pd.DataFrame()

    data = pd.DataFrame(rows, columns=["ACCT_ID", "REFERENCE_DATE", "FACTOR_VALUE"])
    data["REFERENCE_DATE"] = pd.to_datetime(data["REFERENCE_DATE"])
    index_levels = data.pivot(index="REFERENCE_DATE", columns="ACCT_ID", values="FACTOR_VALUE")
    index_levels = index_levels.sort_index()

    # MRD stores index levels; convert to arithmetic daily returns.
    daily_returns = index_levels.pct_change(fill_method=None).dropna(how="all")

    acct_id_to_cmabenches: dict[int, list[str]] = {}
    for cmabench, acct_id in cmabench_to_acct_id.items():
        acct_id_to_cmabenches.setdefault(acct_id, []).append(cmabench)

    out = pd.DataFrame(index=daily_returns.index)
    for acct_id, benches_for_id in acct_id_to_cmabenches.items():
        if acct_id not in daily_returns.columns:
            continue
        for cmabench in benches_for_id:
            out[cmabench] = daily_returns[acct_id]

    ordered_cols = [b for b in selected if b in out.columns]
    out = out.reindex(columns=ordered_cols)
    if not out.empty:
        # Treat MRD data as daily and fill interior calendar gaps with zero
        # between each series' first and last valid return.
        full_index = pd.date_range(out.index.min(), out.index.max(), freq="D")
        out = out.reindex(full_index)
        for col in out.columns:
            first = out[col].first_valid_index()
            last = out[col].last_valid_index()
            if first is None or last is None:
                continue
            mask = (out.index >= first) & (out.index <= last)
            out.loc[mask, col] = out.loc[mask, col].fillna(0.0)
        out = out.dropna(how="all")
    out.index.name = "Date"
    return out
