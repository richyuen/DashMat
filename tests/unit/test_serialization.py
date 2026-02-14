from __future__ import annotations

from utils.serialization import (
    _normalize_for_json,
    canonical_json_dumps,
    date_range_payload_for_cache,
    mapping_payload_for_cache,
    normalize_date_range_payload,
    parse_mapping_payload,
)


def test_canonical_json_dumps_is_deterministic_for_mappings_and_sets():
    payload_a = {"b": 1, "a": {3, 2, 1}}
    payload_b = {"a": {1, 2, 3}, "b": 1}
    assert canonical_json_dumps(payload_a) == canonical_json_dumps(payload_b)


def test_parse_mapping_payload_handles_dict_json_empty_and_invalid():
    assert parse_mapping_payload(None) == {}
    assert parse_mapping_payload({"x": 1}) == {"x": 1}
    assert parse_mapping_payload('{"x": 1}') == {"x": 1}
    assert parse_mapping_payload("") == {}
    assert parse_mapping_payload("not-json") == {}
    assert parse_mapping_payload(123) == {}


def test_normalize_date_range_payload_handles_dict_list_and_nullish():
    assert normalize_date_range_payload({"start": "2024-01-01", "end": "2024-12-31"}) == {
        "start": "2024-01-01",
        "end": "2024-12-31",
    }
    assert normalize_date_range_payload(["2024-01-01", "2024-12-31"]) == {
        "start": "2024-01-01",
        "end": "2024-12-31",
    }
    assert normalize_date_range_payload("null") is None
    assert normalize_date_range_payload(None) is None


def test_normalize_date_range_payload_handles_json_string_and_invalid_cases():
    assert normalize_date_range_payload('{"start":"2024-01-01","end":"2024-12-31"}') == {
        "start": "2024-01-01",
        "end": "2024-12-31",
    }
    assert normalize_date_range_payload("not-json") is None
    assert normalize_date_range_payload({"start": "2024-01-01"}) is None
    assert normalize_date_range_payload(("2024-01-01", None)) is None


def test_normalize_for_json_supports_item_and_fallback_paths():
    class _ItemValue:
        def item(self):
            return 42

    class _BrokenItemValue:
        def item(self):
            raise RuntimeError("boom")

        def __str__(self):
            return "broken-item"

    assert _normalize_for_json(_ItemValue()) == 42
    assert _normalize_for_json(_BrokenItemValue()) == "broken-item"


def test_cache_payload_helpers_produce_canonical_strings():
    assert mapping_payload_for_cache({"b": 2, "a": 1}) == '{"a":1,"b":2}'
    assert date_range_payload_for_cache({"end": "2024-12-31", "start": "2024-01-01"}) == (
        '{"end":"2024-12-31","start":"2024-01-01"}'
    )
