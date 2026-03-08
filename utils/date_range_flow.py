"""Shared date-range helper flows for analytics and portopt pages."""

from __future__ import annotations

import cache_config
from utils.core_categories import get_common_daily_range
from utils.returns import resample_returns_cached


_EMPTY_CANDIDATES = {
    "available_series": (),
    "max_start": None,
    "max_end": None,
    "common_start": None,
    "common_end": None,
    "common_daily_start": None,
    "common_daily_end": None,
}


def _normalize_selected_series(selected_series) -> tuple[str, ...]:
    seen = set()
    normalized = []
    for series in selected_series or ():
        if series in seen:
            continue
        seen.add(series)
        normalized.append(series)
    return tuple(sorted(normalized))


@cache_config.cache.memoize(timeout=0)
def _compute_date_range_candidates_cached(
    raw_data: str,
    periodicity: str,
    selected_series: tuple[str, ...],
    include_common_daily: bool,
) -> dict:
    """Compute reusable range candidates for selected series.

    This function is memoized so repeat callbacks with identical inputs reuse
    computed bounds instead of repeatedly slicing/resampling dataframes.
    """
    if not raw_data or not selected_series:
        return _EMPTY_CANDIDATES

    df = resample_returns_cached(raw_data, periodicity or "daily")
    available_series = tuple(series for series in selected_series if series in df.columns)
    if not available_series or df.empty:
        return _EMPTY_CANDIDATES

    result = dict(_EMPTY_CANDIDATES)
    result["available_series"] = available_series
    result["max_start"] = df.index.min().strftime("%Y-%m-%d")
    result["max_end"] = df.index.max().strftime("%Y-%m-%d")

    subset = df.loc[:, list(available_series)].dropna()
    if not subset.empty:
        result["common_start"] = subset.index.min().strftime("%Y-%m-%d")
        result["common_end"] = subset.index.max().strftime("%Y-%m-%d")

    if include_common_daily:
        daily_df = resample_returns_cached(raw_data, "daily_trading")
        daily_available = [series for series in selected_series if series in daily_df.columns]
        if daily_available:
            common_daily = get_common_daily_range(daily_df, daily_available)
            if common_daily:
                result["common_daily_start"] = common_daily[0].strftime("%Y-%m-%d")
                result["common_daily_end"] = common_daily[1].strftime("%Y-%m-%d")

    return result


def compute_date_range_candidates(
    raw_data: str,
    periodicity: str,
    selected_series: tuple[str, ...],
    include_common_daily: bool = True,
) -> dict:
    normalized_series = _normalize_selected_series(selected_series)
    return _compute_date_range_candidates_cached(
        raw_data,
        periodicity,
        normalized_series,
        include_common_daily,
    )


def resolve_initial_range(candidates: dict, stored_range) -> tuple[str | None, str | None]:
    """Resolve initial picker start/end from candidates and stored range."""
    max_start = candidates.get("max_start")
    max_end = candidates.get("max_end")
    if not max_start or not max_end:
        return None, None

    if stored_range and stored_range.get("start") and stored_range.get("end"):
        stored_start = stored_range["start"]
        stored_end = stored_range["end"]
        if stored_start >= max_start and stored_end <= max_end:
            return stored_start, stored_end

    return max_start, max_end


def resolve_button_range(candidates: dict, button_id: str) -> tuple[str | None, str | None, bool]:
    """Resolve range + whether periodicity should switch to daily_trading."""
    if button_id.endswith("common-range-button"):
        return candidates.get("common_start"), candidates.get("common_end"), False
    if button_id.endswith("common-daily-button"):
        return candidates.get("common_daily_start"), candidates.get("common_daily_end"), True
    if button_id.endswith("maximum-range-button"):
        return candidates.get("max_start"), candidates.get("max_end"), False
    return None, None, False
