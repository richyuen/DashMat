"""Read-only CoreCategories and CMA returns helpers for UI imports."""

from __future__ import annotations

import cache_config
import pandas as pd
import pandas_market_calendars as mcal
from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine
from utils.sec_factor_loader import load_sec_factor_returns_by_names_aa

RISK_FREE_FOFBENCH = "BCTBill13_TRIndex"


def get_core_category_options(engine: Engine) -> list[dict]:
    """Return dropdown options formatted as `CoreCat [FOFBench]`."""
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT CoreCat, FOFBench "
                "FROM CoreCategories "
                "WHERE FOFBench IS NOT NULL "
                "ORDER BY CoreCat, FOFBench"
            )
        ).fetchall()
    return [
        {"value": str(fofbench), "label": f"{str(corecat)} [{str(fofbench)}]"}
        for corecat, fofbench in rows
    ]


@cache_config.cache.memoize(timeout=0)
def get_core_category_options_cached(engine: Engine) -> list[dict]:
    """Cached CoreCategories dropdown options."""
    return get_core_category_options(engine)


def get_unique_cmabench_values(engine: Engine) -> list[str]:
    """Return sorted unique non-empty CMABench values."""
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT DISTINCT CMABench "
                "FROM CoreCategories "
                "WHERE CMABench IS NOT NULL "
                "AND LTRIM(RTRIM(CMABench)) <> '' "
                "ORDER BY CMABench"
            )
        ).fetchall()
    return [str(r[0]).strip() for r in rows if r[0] and str(r[0]).strip()]


@cache_config.cache.memoize(timeout=0)
def get_unique_cmabench_values_cached(engine: Engine) -> list[str]:
    """Cached sorted unique non-empty CMABench values."""
    return get_unique_cmabench_values(engine)


def get_cma_versions(engine: Engine) -> list[int]:
    """Return available CMA versions."""
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT DISTINCT Version FROM CMAStats ORDER BY Version")
        ).fetchall()
    return [int(r[0]) for r in rows]


@cache_config.cache.memoize(timeout=0)
def get_cma_versions_cached(engine: Engine) -> list[int]:
    """Cached CMA versions for dropdowns."""
    return get_cma_versions(engine)


def clear_dropdown_caches() -> None:
    """Clear any process-local dropdown caches (if present)."""
    clearables = (
        get_core_category_options,
        get_core_category_options_cached,
        get_unique_cmabench_values,
        get_unique_cmabench_values_cached,
        get_cma_versions,
        get_cma_versions_cached,
    )
    for fn in clearables:
        cache_clear = getattr(fn, "cache_clear", None)
        if callable(cache_clear):
            cache_clear()


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


def _next_trading_day_lookup(
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[pd.Timestamp, pd.Timestamp]:
    nyse = mcal.get_calendar("NYSE")
    valid_days = nyse.valid_days(
        start_date=start - pd.Timedelta(days=7),
        end_date=end + pd.Timedelta(days=7),
    ).tz_localize(None)
    valid_days = pd.DatetimeIndex(valid_days).sort_values().unique()
    lookup: dict[pd.Timestamp, pd.Timestamp] = {}
    for i in range(len(valid_days) - 1):
        lookup[pd.Timestamp(valid_days[i])] = pd.Timestamp(valid_days[i + 1])
    return lookup


def _infer_daily_start_from_observation_dates(
    observation_dates: pd.DatetimeIndex,
) -> pd.Timestamp | None:
    """Infer first daily return date from level observations.

    Daily is assumed when three trading-day observations appear on consecutive
    trading days (which implies two consecutive daily returns). For levels, the
    first daily return is at the second observation date.
    """
    obs = pd.DatetimeIndex(observation_dates).sort_values().unique()
    if len(obs) < 3:
        return None

    next_lookup = _next_trading_day_lookup(pd.Timestamp(obs.min()), pd.Timestamp(obs.max()))
    for i in range(len(obs) - 2):
        d0 = pd.Timestamp(obs[i])
        d1 = pd.Timestamp(obs[i + 1])
        d2 = pd.Timestamp(obs[i + 2])
        if next_lookup.get(d0) == d1 and next_lookup.get(d1) == d2:
            return d1
    return None


def infer_daily_start_from_returns(
    returns: pd.Series,
) -> pd.Timestamp | None:
    """Infer first return date where returns become daily.

    Daily is assumed when two consecutive trading-day return observations appear
    and both returns are non-zero.
    If the first date in the pair is month-end, skip that pair to avoid
    classifying a month-end carry-forward return as daily.
    """
    valid = returns.dropna()
    if valid.empty:
        return None
    ret_dates = pd.DatetimeIndex(valid.index).sort_values().unique()
    if len(ret_dates) < 2:
        return None

    next_lookup = _next_trading_day_lookup(pd.Timestamp(ret_dates.min()), pd.Timestamp(ret_dates.max()))
    for i in range(len(ret_dates) - 1):
        d0 = pd.Timestamp(ret_dates[i])
        d1 = pd.Timestamp(ret_dates[i + 1])
        if next_lookup.get(d0) == d1:
            v0 = float(valid.loc[d0])
            v1 = float(valid.loc[d1])
            if v0 == 0.0 or v1 == 0.0:
                continue
            if d0.is_month_end:
                continue
            return d0
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
    out, series_meta = load_sec_factor_returns_by_names_aa(
        mrd_engine,
        selected_fofbenches,
        collision_policy="bb_then_lowest",
        exclude_perf=False,
    )
    if out.empty:
        return pd.DataFrame(), {}

    periodicity_hint = out.attrs.get("periodicity_hint")
    ordered_cols = [b for b in selected if b in out.columns]
    out = out.reindex(columns=ordered_cols)
    if periodicity_hint in {"daily", "monthly"}:
        out.attrs["periodicity_hint"] = periodicity_hint
    meta = {name: series_meta.get(name, {}) for name in ordered_cols if name in series_meta}
    out.index.name = "Date"
    return out, meta


def load_bctbill13_returns(core_engine: Engine, mrd_engine: Engine) -> pd.DataFrame:
    """Load BCTBill13 daily returns using the same MRD-backed import path."""
    return load_cma_returns_for_benches(core_engine, [RISK_FREE_FOFBENCH], mrd_engine)
