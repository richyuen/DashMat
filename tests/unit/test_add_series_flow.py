from __future__ import annotations

from utils.add_series_flow import (
    find_duplicate_series,
    get_existing_columns,
    import_selected_disabled,
)
from utils.returns import df_to_json


def test_get_existing_columns_returns_series_names(sample_returns_df):
    raw_data = df_to_json(sample_returns_df)
    columns = get_existing_columns(raw_data)
    assert "Asset_A" in columns
    assert "Asset_D" in columns


def test_get_existing_columns_handles_invalid_payload():
    assert get_existing_columns("not-json") == set()


def test_find_duplicate_series_filters_to_existing(sample_returns_df):
    raw_data = df_to_json(sample_returns_df)
    duplicates = find_duplicate_series(["Asset_A", "Missing"], raw_data)
    assert duplicates == ["Asset_A"]


def test_get_existing_columns_prefers_raw_metadata():
    columns = get_existing_columns(raw_meta={"columns": ["Asset_A", "Asset_B"]})
    assert columns == {"Asset_A", "Asset_B"}


def test_find_duplicate_series_supports_raw_metadata():
    duplicates = find_duplicate_series(
        ["Asset_A", "Missing"],
        raw_meta={"columns": ["Asset_A", "Asset_B"]},
    )
    assert duplicates == ["Asset_A"]


def test_import_selected_disabled():
    assert import_selected_disabled([]) is True
    assert import_selected_disabled(["Sheet1"]) is False
