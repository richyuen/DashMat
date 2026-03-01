"""Shared date-range helper flows for analytics, portfolio optimization, and regression."""

from __future__ import annotations

import hashlib

import cache_config
from utils.core_categories import infer_daily_start_from_returns
from utils.perf_timing import timed_block
from utils.returns import get_available_periodicities, json_to_df, resample_returns


_EMPTY_CANDIDATES = {
    "available_series": (),
    "max_start": None,
    "max_end": None,
    "common_start": None,
    "common_end": None,
    "common_daily_start": None,
    "common_daily_end": None,
}


def _format_ts(value) -> str | None:
    if value is None:
        return None
    return value.strftime("%Y-%m-%d")


def _empty_metadata(raw_data_hash: str, periodicity: str) -> dict:
    return {
        "raw_data_hash": raw_data_hash,
        "periodicity": periodicity,
        "dataset_start": None,
        "dataset_end": None,
        "series_ranges": {},
        "daily_phase_ranges": None,
        "_resampled_df": None,
    }


def _range_map_for_df(df) -> dict[str, dict[str, str | None]]:
    out: dict[str, dict[str, str | None]] = {}
    for col in df.columns:
        valid = df[col].dropna()
        if valid.empty:
            out[str(col)] = {"start": None, "end": None}
            continue
        out[str(col)] = {
            "start": _format_ts(valid.index.min()),
            "end": _format_ts(valid.index.max()),
        }
    return out


def _daily_phase_map_for_df(df) -> dict[str, dict[str, str | None]]:
    out: dict[str, dict[str, str | None]] = {}
    for col in df.columns:
        valid = df[col].dropna()
        if valid.empty:
            out[str(col)] = {"start": None, "end": None}
            continue
        out[str(col)] = {
            "start": _format_ts(infer_daily_start_from_returns(valid)),
            "end": _format_ts(valid.index.max()),
        }
    return out


def build_raw_data_metadata(raw_data: str, original_periodicity: str) -> dict | None:
    """Build shared client-side metadata once per raw-data change."""
    if not raw_data:
        return None

    with timed_block("shared.raw_data_metadata"):
        resolved_original = original_periodicity or "daily"
        base_df = json_to_df(raw_data)
        if base_df is None or base_df.empty:
            return None

        periodicity_options = get_available_periodicities(resolved_original)
        periodicity_values = [option["value"] for option in periodicity_options]
        metadata = {
            "raw_data_hash": hashlib.md5(raw_data.encode("utf-8")).hexdigest(),
            "original_periodicity": resolved_original,
            "columns": [str(col) for col in base_df.columns],
            "available_periodicity_values": periodicity_values,
            "periodicities": {},
            "daily_phase_ranges": None,
        }

        for periodicity in periodicity_values:
            resampled_df = resample_returns(base_df, periodicity)
            block = {
                "dataset_start": None,
                "dataset_end": None,
                "series_ranges": {},
            }
            if resampled_df is not None and not resampled_df.empty:
                block["dataset_start"] = _format_ts(resampled_df.index.min())
                block["dataset_end"] = _format_ts(resampled_df.index.max())
                block["series_ranges"] = _range_map_for_df(resampled_df)
            metadata["periodicities"][periodicity] = block

        if resolved_original == "daily":
            daily_trading_df = resample_returns(base_df, "daily_trading")
            metadata["daily_phase_ranges"] = (
                _daily_phase_map_for_df(daily_trading_df)
                if daily_trading_df is not None and not daily_trading_df.empty
                else {}
            )

        return metadata


def build_raw_data_summary(raw_data: str, original_periodicity: str) -> dict | None:
    """Build a small client-visible summary for shared raw data."""
    if not raw_data:
        return None

    with timed_block("shared.raw_data_summary"):
        resolved_original = original_periodicity or "daily"
        columns = list(json_to_df(raw_data).columns)
        return {
            "raw_data_hash": hashlib.md5(raw_data.encode("utf-8")).hexdigest(),
            "columns": columns,
            "available_periodicity_values": [
                option["value"] for option in get_available_periodicities(resolved_original)
            ],
            "original_periodicity": resolved_original,
        }


def compute_date_range_candidates_from_global_metadata(
    metadata: dict,
    periodicity: str,
    selected_series: tuple[str, ...],
) -> dict:
    """Compute date-range candidates from shared raw-data metadata only."""
    if not metadata or not selected_series:
        return dict(_EMPTY_CANDIDATES)

    periodicity_block = ((metadata.get("periodicities") or {}).get(periodicity or "daily")) or {}
    series_ranges = periodicity_block.get("series_ranges") or {}
    available_series = tuple(series for series in selected_series if series in series_ranges)
    if not available_series:
        return dict(_EMPTY_CANDIDATES)

    result = dict(_EMPTY_CANDIDATES)
    result["available_series"] = available_series
    result["max_start"] = periodicity_block.get("dataset_start")
    result["max_end"] = periodicity_block.get("dataset_end")

    starts = [series_ranges[series].get("start") for series in available_series if series_ranges[series].get("start")]
    ends = [series_ranges[series].get("end") for series in available_series if series_ranges[series].get("end")]
    if len(starts) == len(available_series) and len(ends) == len(available_series):
        common_start = max(starts)
        common_end = min(ends)
        if common_start <= common_end:
            result["common_start"] = common_start
            result["common_end"] = common_end

    daily_phase_ranges = metadata.get("daily_phase_ranges") or {}
    daily_available = [series for series in available_series if series in daily_phase_ranges]
    if daily_available:
        daily_starts = [
            daily_phase_ranges[series].get("start")
            for series in daily_available
            if daily_phase_ranges[series].get("start")
        ]
        daily_ends = [
            daily_phase_ranges[series].get("end")
            for series in daily_available
            if daily_phase_ranges[series].get("end")
        ]
        if len(daily_starts) == len(daily_available) and len(daily_ends) == len(daily_available):
            common_daily_start = max(daily_starts)
            common_daily_end = min(daily_ends)
            if common_daily_start <= common_daily_end:
                result["common_daily_start"] = common_daily_start
                result["common_daily_end"] = common_daily_end

    return result


def get_periodicity_range_metadata(raw_data_hash: str, raw_data: str, periodicity: str) -> dict:
    """Return server-cached range metadata keyed by raw-data hash and periodicity."""
    resolved_periodicity = periodicity or "daily"
    cache_key = f"date-range-metadata:{raw_data_hash}:{resolved_periodicity}"
    cached = cache_config.cache.get(cache_key)
    if cached is not None:
        return cached

    if not raw_data:
        return _empty_metadata(raw_data_hash, resolved_periodicity)

    with timed_block(
        "shared.periodicity_range_metadata",
        periodicity=resolved_periodicity,
    ):
        base_df = json_to_df(raw_data)
        if base_df is None or base_df.empty:
            metadata = _empty_metadata(raw_data_hash, resolved_periodicity)
            cache_config.cache.set(cache_key, metadata, timeout=0)
            return metadata

        resampled_df = resample_returns(base_df, resolved_periodicity)
        metadata = _empty_metadata(raw_data_hash, resolved_periodicity)
        metadata["_resampled_df"] = resampled_df

        if resampled_df is not None and not resampled_df.empty:
            metadata["dataset_start"] = _format_ts(resampled_df.index.min())
            metadata["dataset_end"] = _format_ts(resampled_df.index.max())
            metadata["series_ranges"] = _range_map_for_df(resampled_df)

        daily_trading_df = resample_returns(base_df, "daily_trading")
        if daily_trading_df is not None and not daily_trading_df.empty:
            metadata["daily_phase_ranges"] = _daily_phase_map_for_df(daily_trading_df)
        else:
            metadata["daily_phase_ranges"] = {}

        cache_config.cache.set(cache_key, metadata, timeout=0)
        return metadata


def compute_date_range_candidates_from_metadata(metadata: dict, selected_series: tuple[str, ...]) -> dict:
    """Compute reusable range candidates from cached metadata."""
    if not metadata or not selected_series:
        return dict(_EMPTY_CANDIDATES)

    resampled_df = metadata.get("_resampled_df")
    if resampled_df is None or resampled_df.empty:
        return dict(_EMPTY_CANDIDATES)

    available_series = tuple(series for series in selected_series if series in resampled_df.columns)
    if not available_series:
        return dict(_EMPTY_CANDIDATES)

    result = dict(_EMPTY_CANDIDATES)
    result["available_series"] = available_series
    result["max_start"] = metadata.get("dataset_start")
    result["max_end"] = metadata.get("dataset_end")

    subset = resampled_df.loc[:, list(available_series)].dropna()
    if not subset.empty:
        result["common_start"] = _format_ts(subset.index.min())
        result["common_end"] = _format_ts(subset.index.max())

    daily_phase_ranges = metadata.get("daily_phase_ranges") or {}
    daily_available = [series for series in selected_series if series in daily_phase_ranges]
    if daily_available:
        starts = [
            daily_phase_ranges[series]["start"]
            for series in daily_available
            if daily_phase_ranges[series].get("start")
        ]
        ends = [
            daily_phase_ranges[series]["end"]
            for series in daily_available
            if daily_phase_ranges[series].get("end")
        ]
        if len(starts) == len(daily_available) and len(ends) == len(daily_available):
            common_daily_start = max(starts)
            common_daily_end = min(ends)
            if common_daily_start <= common_daily_end:
                result["common_daily_start"] = common_daily_start
                result["common_daily_end"] = common_daily_end

    return result


def compute_date_range_candidates(raw_data: str, periodicity: str, selected_series: tuple[str, ...]) -> dict:
    """Compatibility wrapper that now uses hash-keyed metadata."""
    if not raw_data or not selected_series:
        return dict(_EMPTY_CANDIDATES)

    raw_data_hash = hashlib.md5(raw_data.encode("utf-8")).hexdigest()
    metadata = get_periodicity_range_metadata(raw_data_hash, raw_data, periodicity)
    return compute_date_range_candidates_from_metadata(metadata, selected_series)


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
