"""Shared date-range helper flows for analytics, portopt, and regression pages."""

from __future__ import annotations

import cache_config
import numpy as np
import pandas as pd
from utils.perf_timing import timed_block
from utils.returns import resample_returns_by_key


_EMPTY_CANDIDATES = {
    "available_series": (),
    "max_start": None,
    "max_end": None,
    "common_start": None,
    "common_end": None,
}

_EMPTY_COMMON_DAILY_CANDIDATES = {
    "common_daily_start": None,
    "common_daily_end": None,
}

ACCOUNT_LIST_MAX_END_SENTINEL = "3999-12-31"


def _normalize_selected_series(selected_series) -> tuple[str, ...]:
    seen = set()
    normalized = []
    for series in selected_series or ():
        if series in seen:
            continue
        seen.add(series)
        normalized.append(series)
    return tuple(sorted(normalized))


def _build_range_candidates_from_df(df, selected_series: tuple[str, ...]) -> dict:
    if df.empty or not selected_series:
        return dict(_EMPTY_CANDIDATES)

    available_series = tuple(series for series in selected_series if series in df.columns)
    if not available_series:
        return dict(_EMPTY_CANDIDATES)

    result = dict(_EMPTY_CANDIDATES)
    result["available_series"] = available_series
    result["max_start"] = df.index.min().strftime("%Y-%m-%d")
    result["max_end"] = df.index.max().strftime("%Y-%m-%d")

    subset = df.loc[:, list(available_series)].dropna()
    if not subset.empty:
        result["common_start"] = subset.index.min().strftime("%Y-%m-%d")
        result["common_end"] = subset.index.max().strftime("%Y-%m-%d")

    return result


def _build_common_daily_series_metadata_from_df(
    daily_df: pd.DataFrame,
) -> dict[str, tuple[pd.Timestamp | None, pd.Timestamp | None]]:
    if daily_df.empty:
        return {}

    index = pd.DatetimeIndex(daily_df.index)
    if len(index) == 0:
        return {}

    values = daily_df.to_numpy(dtype=float, copy=False)
    valid = ~np.isnan(values)
    if not valid.any():
        return {}

    nonzero = valid & (values != 0.0)
    pair_candidates = nonzero[:-1] & nonzero[1:] if len(index) > 1 else np.empty((0, values.shape[1]), dtype=bool)
    if pair_candidates.size:
        pair_candidates &= (~np.asarray(index.is_month_end[:-1], dtype=bool))[:, None]

    metadata: dict[str, tuple[pd.Timestamp | None, pd.Timestamp | None]] = {}
    row_count = len(index)

    for col_idx, series in enumerate(daily_df.columns):
        valid_col = valid[:, col_idx]
        if not valid_col.any():
            continue

        last_valid_pos = row_count - 1 - int(np.argmax(valid_col[::-1]))

        daily_start: pd.Timestamp | None = None
        if pair_candidates.size:
            pair_col = pair_candidates[:, col_idx]
            if pair_col.any():
                daily_start = pd.Timestamp(index[int(np.argmax(pair_col))])

        metadata[str(series)] = (daily_start, pd.Timestamp(index[last_valid_pos]))

    return metadata


def _reduce_common_daily_candidates(
    metadata: dict[str, tuple[pd.Timestamp | None, pd.Timestamp | None]],
    selected_series: tuple[str, ...],
) -> dict:
    if not metadata or not selected_series:
        return dict(_EMPTY_COMMON_DAILY_CANDIDATES)

    starts: list[pd.Timestamp] = []
    ends: list[pd.Timestamp] = []

    for series in selected_series:
        daily_meta = metadata.get(series)
        if not daily_meta:
            return dict(_EMPTY_COMMON_DAILY_CANDIDATES)
        daily_start, last_valid = daily_meta
        if daily_start is None or last_valid is None:
            return dict(_EMPTY_COMMON_DAILY_CANDIDATES)
        starts.append(pd.Timestamp(daily_start))
        ends.append(pd.Timestamp(last_valid))

    if not starts or not ends:
        return dict(_EMPTY_COMMON_DAILY_CANDIDATES)

    common_start = max(starts)
    common_end = min(ends)
    if common_start > common_end:
        return dict(_EMPTY_COMMON_DAILY_CANDIDATES)

    return {
        "common_daily_start": common_start.strftime("%Y-%m-%d"),
        "common_daily_end": common_end.strftime("%Y-%m-%d"),
    }


@cache_config.cache.memoize(timeout=0)
def _compute_date_range_candidates_cached(
    dataset_key: str | None,
    periodicity: str,
    selected_series: tuple[str, ...],
) -> dict:
    """Compute reusable range candidates for selected series.

    This function is memoized so repeat callbacks with identical inputs reuse
    computed bounds instead of repeatedly slicing/resampling dataframes.
    """
    if not dataset_key or not selected_series:
        return dict(_EMPTY_CANDIDATES)

    df = resample_returns_by_key(dataset_key, periodicity or "daily")
    return _build_range_candidates_from_df(df, selected_series)


@cache_config.cache.memoize(timeout=0)
def _compute_common_daily_series_metadata_cached(
    dataset_key: str | None,
) -> dict[str, tuple[pd.Timestamp | None, pd.Timestamp | None]]:
    if not dataset_key:
        return {}

    with timed_block("analyticstool.common_daily_metadata"):
        daily_df = resample_returns_by_key(dataset_key, "daily_trading")
        return _build_common_daily_series_metadata_from_df(daily_df)


@cache_config.cache.memoize(timeout=0)
def _compute_common_daily_candidates_cached(
    dataset_key: str | None,
    selected_series: tuple[str, ...],
) -> dict:
    """Compute reusable common-daily bounds separately from base range candidates."""
    if not dataset_key or not selected_series:
        return dict(_EMPTY_COMMON_DAILY_CANDIDATES)

    metadata = _compute_common_daily_series_metadata_cached(dataset_key)
    with timed_block("analyticstool.common_daily_reduce", series_count=len(selected_series)):
        return _reduce_common_daily_candidates(metadata, selected_series)


def compute_date_candidate_bundle(
    dataset_key: str | None,
    periodicity: str,
    selected_series: tuple[str, ...],
) -> tuple[dict, dict]:
    normalized_series = _normalize_selected_series(selected_series)
    return (
        _compute_date_range_candidates_cached(dataset_key, periodicity, normalized_series),
        _compute_common_daily_candidates_cached(dataset_key, normalized_series),
    )


def compute_date_range_candidates(
    dataset_key: str | None,
    periodicity: str,
    selected_series: tuple[str, ...],
) -> dict:
    normalized_series = _normalize_selected_series(selected_series)
    return _compute_date_range_candidates_cached(
        dataset_key,
        periodicity,
        normalized_series,
    )


def compute_common_daily_candidates(
    dataset_key: str | None,
    selected_series: tuple[str, ...],
) -> dict:
    normalized_series = _normalize_selected_series(selected_series)
    return _compute_common_daily_candidates_cached(dataset_key, normalized_series)


def resolve_initial_range(candidates: dict, stored_range) -> tuple[str | None, str | None]:
    """Resolve initial picker start/end from candidates and stored range."""
    max_start = candidates.get("max_start")
    max_end = candidates.get("max_end")
    if not max_start or not max_end:
        return None, None

    if stored_range and stored_range.get("start") and stored_range.get("end"):
        stored_start = stored_range["start"]
        stored_end = max_end if stored_range["end"] == ACCOUNT_LIST_MAX_END_SENTINEL else stored_range["end"]
        if stored_start >= max_start and stored_end <= max_end:
            return stored_start, stored_end

    return max_start, max_end


def resolve_button_range(
    candidates: dict,
    button_id: str,
    common_daily_candidates: dict | None = None,
) -> tuple[str | None, str | None, bool]:
    """Resolve range + whether periodicity should switch to daily_trading."""
    if button_id.endswith("common-range-button"):
        return candidates.get("common_start"), candidates.get("common_end"), False
    if button_id.endswith("common-daily-button"):
        source = common_daily_candidates if isinstance(common_daily_candidates, dict) else candidates
        return source.get("common_daily_start"), source.get("common_daily_end"), True
    if button_id.endswith("maximum-range-button"):
        return candidates.get("max_start"), candidates.get("max_end"), False
    return None, None, False
