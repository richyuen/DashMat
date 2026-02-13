"""Read-only CoreCategories and CMA returns helpers for UI imports."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine

RISK_FREE_FOFBENCH = "BCTBill13_TRIndex"
_MAX_DAILY_GAP_DAYS = 4
_MIN_DAILY_OBS = 2


def get_core_category_options(engine: Engine) -> list[dict]:
    """Return dropdown options formatted as `CoreCat [FOFBench]`."""
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT CoreCat, FOFBench "
                "FROM CoreCategories "
                "WHERE FOFBench IS NOT NULL "
                "ORDER BY CoreCatOrder"
            )
        ).fetchall()
    return [
        {"value": str(fofbench), "label": f"{str(corecat)} [{str(fofbench)}]"}
        for corecat, fofbench in rows
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


def _infer_daily_start_from_observation_dates(
    observation_dates: pd.DatetimeIndex,
    max_gap_days: int = _MAX_DAILY_GAP_DAYS,
    min_daily_obs: int = _MIN_DAILY_OBS,
) -> pd.Timestamp | None:
    """Infer first observation date where the return stream is daily thereafter."""
    obs = pd.DatetimeIndex(observation_dates).sort_values().unique()
    if len(obs) < 2:
        return None

    deltas = (obs[1:] - obs[:-1]).days
    n_returns = len(obs) - 1
    required_obs = min(min_daily_obs, n_returns)

    for pos in range(1, len(obs)):
        returns_remaining = len(obs) - pos
        if returns_remaining < required_obs:
            continue
        if np.all(deltas[pos - 1:] <= max_gap_days):
            return pd.Timestamp(obs[pos])
    return None


def infer_daily_start_from_returns(
    returns: pd.Series,
    max_gap_days: int = _MAX_DAILY_GAP_DAYS,
    min_daily_obs: int = _MIN_DAILY_OBS,
) -> pd.Timestamp | None:
    """Infer first return date where return observations are daily thereafter."""
    valid = returns.dropna()
    if valid.empty:
        return None
    ret_dates = pd.DatetimeIndex(valid.index).sort_values().unique()
    if len(ret_dates) == 1:
        return pd.Timestamp(ret_dates[0])

    gaps = (ret_dates[1:] - ret_dates[:-1]).days
    required_obs = min(min_daily_obs, len(ret_dates))

    for i in range(len(ret_dates)):
        obs_remaining = len(ret_dates) - i
        if obs_remaining < required_obs:
            continue
        if i == 0:
            if np.all(gaps <= max_gap_days):
                return pd.Timestamp(ret_dates[0])
            continue
        if gaps[i - 1] <= max_gap_days and (i == len(ret_dates) - 1 or np.all(gaps[i:] <= max_gap_days)):
            return pd.Timestamp(ret_dates[i])
    return None


def get_common_daily_range(
    df: pd.DataFrame,
    series_names: list[str],
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Return overlap range where all selected series are in their daily phase."""
    if df.empty or not series_names:
        return None

    starts: list[pd.Timestamp] = []
    ends: list[pd.Timestamp] = []
    for series in series_names:
        if series not in df.columns:
            return None
        s = df[series].dropna()
        if s.empty:
            return None
        daily_start = infer_daily_start_from_returns(s)
        if daily_start is None:
            return None
        starts.append(pd.Timestamp(daily_start))
        ends.append(pd.Timestamp(s.index.max()))

    if not starts or not ends:
        return None

    common_start = max(starts)
    common_end = min(ends)
    if common_start > common_end:
        return None
    return common_start, common_end


def get_cmabench_map_for_fofbench(engine: Engine, fofbenches: list[str]) -> dict[str, str]:
    """Return mapping from FOFBench to CMABench for provided FOFBench values."""
    selected = [str(v) for v in fofbenches if v]
    if not selected:
        return {}
    q = text(
        "SELECT FOFBench, CMABench "
        "FROM CoreCategories "
        "WHERE FOFBench IN :fofbenches AND CMABench IS NOT NULL"
    ).bindparams(bindparam("fofbenches", expanding=True))
    with engine.connect() as conn:
        rows = conn.execute(q, {"fofbenches": selected}).fetchall()
    return {str(r[0]): str(r[1]) for r in rows if r[0] and r[1]}


def load_cma_returns_for_benches(
    core_engine: Engine,
    benches: list[str],
    mrd_engine: Engine,
) -> pd.DataFrame:
    out, _ = load_cma_returns_for_benches_with_meta(core_engine, benches, mrd_engine)
    return out


def load_cma_returns_for_benches_with_meta(
    core_engine: Engine,
    benches: list[str],
    mrd_engine: Engine,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    """Load selected FOFBench daily returns from MRD index levels."""
    selected = [str(b) for b in benches if b]
    if not selected:
        return pd.DataFrame(), {}

    core_query = text(
        "SELECT FOFBench "
        "FROM CoreCategories "
        "WHERE FOFBench IN :benches "
        "ORDER BY CoreCatOrder"
    ).bindparams(bindparam("benches", expanding=True))
    with core_engine.connect() as conn:
        mapping_rows = conn.execute(
            core_query, {"benches": selected}
        ).fetchall()
    if not mapping_rows:
        return pd.DataFrame(), {}

    selected_fofbenches = [str(r[0]) for r in mapping_rows if r[0] is not None]
    selected_fofbench_set = set(selected_fofbenches)
    selected_fofbenches = [b for b in selected if b in selected_fofbench_set]
    if not selected_fofbenches:
        return pd.DataFrame(), {}

    acct_names = sorted({_split_fofbench(v)[0] for v in selected_fofbenches})
    factor_names = sorted({_split_fofbench(v)[1] for v in selected_fofbenches})
    if not acct_names or not factor_names:
        return pd.DataFrame(), {}

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
        return pd.DataFrame(), {}

    fofbench_to_acct_id: dict[str, int] = {}
    for acct_id, acct_name, factor_name in account_rows:
        key = f"{acct_name}_{factor_name}"
        if key in selected_fofbench_set and key not in fofbench_to_acct_id:
            fofbench_to_acct_id[key] = int(acct_id)

    fofbench_to_acct_id = {
        fofbench: fofbench_to_acct_id[fofbench]
        for fofbench in selected_fofbenches
        if fofbench in fofbench_to_acct_id
    }
    if not fofbench_to_acct_id:
        return pd.DataFrame(), {}

    acct_ids = sorted(set(fofbench_to_acct_id.values()))
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
        return pd.DataFrame(), {}

    data = pd.DataFrame(rows, columns=["ACCT_ID", "REFERENCE_DATE", "FACTOR_VALUE"])
    data["REFERENCE_DATE"] = pd.to_datetime(data["REFERENCE_DATE"])
    index_levels = data.pivot(index="REFERENCE_DATE", columns="ACCT_ID", values="FACTOR_VALUE")
    index_levels = index_levels.sort_index()

    # MRD stores index levels; convert to arithmetic daily returns.
    daily_returns = index_levels.pct_change(fill_method=None).dropna(how="all")

    acct_id_to_fofbenches: dict[int, list[str]] = {}
    for fofbench, acct_id in fofbench_to_acct_id.items():
        acct_id_to_fofbenches.setdefault(acct_id, []).append(fofbench)

    out = pd.DataFrame(index=daily_returns.index)
    series_meta: dict[str, dict[str, object]] = {}
    for acct_id, benches_for_id in acct_id_to_fofbenches.items():
        if acct_id not in daily_returns.columns:
            continue
        returns_series = daily_returns[acct_id]
        first_return = returns_series.dropna().index.min()
        obs_dates = index_levels[acct_id].dropna().index if acct_id in index_levels.columns else pd.DatetimeIndex([])
        daily_start = _infer_daily_start_from_observation_dates(obs_dates)
        starts_daily = bool(first_return is not None and daily_start is not None and pd.Timestamp(first_return) == pd.Timestamp(daily_start))
        for fofbench in benches_for_id:
            out[fofbench] = returns_series
            series_meta[fofbench] = {
                "first_return_date": first_return.strftime("%Y-%m-%d") if first_return is not None else None,
                "daily_start_date": daily_start.strftime("%Y-%m-%d") if daily_start is not None else None,
                "starts_daily": starts_daily,
            }

    ordered_cols = [b for b in selected if b in out.columns]
    out = out.reindex(columns=ordered_cols)
    if not out.empty:
        # Reindex to calendar dates; fill only from each series' inferred daily
        # start date onward. Pre-daily history stays sparse.
        full_index = pd.date_range(out.index.min(), out.index.max(), freq="D")
        out = out.reindex(full_index)
        for col in out.columns:
            last = out[col].last_valid_index()
            if last is None:
                continue
            daily_start_raw = (
                series_meta.get(col, {}).get("daily_start_date")
                if isinstance(series_meta.get(col, {}), dict)
                else None
            )
            if not daily_start_raw:
                continue
            daily_start = pd.to_datetime(daily_start_raw, errors="coerce")
            if pd.isna(daily_start):
                continue
            mask = (out.index >= daily_start) & (out.index <= last)
            out.loc[mask, col] = out.loc[mask, col].fillna(0.0)
        out = out.dropna(how="all")
    out.index.name = "Date"
    return out, series_meta


def load_bctbill13_returns(core_engine: Engine, mrd_engine: Engine) -> pd.DataFrame:
    """Load BCTBill13 daily returns using the same MRD-backed import path."""
    return load_cma_returns_for_benches(core_engine, [RISK_FREE_FOFBENCH], mrd_engine)
