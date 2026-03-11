from __future__ import annotations

import json

from tools.playwright.save_load_session_harness import (
    build_analyticstool_storage_seed,
    bundle_expectations_for_workflow,
    decode_downloaded_bundle,
    live_seed_for_workflow,
    summarize_cycles,
    tamper_bundle_for_mode,
    validate_bundle_structure,
)


def _seed_payload() -> dict:
    return {
        "session_id": "session-123",
        "raw_json": "{}",
        "raw_meta": {"columns": ["SPX_TRIndex"], "has_data": True},
        "dataset_meta": {"start": "2021-01-04", "end": "2023-12-29"},
    }


def test_build_analyticstool_storage_seed_contains_restore_shape():
    payload = build_analyticstool_storage_seed(_seed_payload())

    assert payload["dashmat-session-id-store"] == "session-123"
    assert payload["at-periodicity-value-store"] == "daily_trading"
    assert payload["at-active-tab-store"] == "statistics"
    assert payload["at-date-range-store"] == {"start": "2021-01-04", "end": "2023-12-29"}
    assert payload["at-state-ready-store"] is True


def test_validate_bundle_structure_requires_expected_artifacts():
    bundle = {
        "version": 2,
        "workspace_session": {"po-results-store": json.dumps({"P1": {"returns_key": "k1"}})},
        "artifact_refs": [{"store_key": "po-results-store", "path": ["P1", "returns_key"], "artifact_key": "k1"}],
        "artifacts": [{"key": "k1", "artifact_type": "po_portfolio_returns", "format": "feather", "metadata": {}, "payload": ""}],
    }

    meta = validate_bundle_structure(bundle, "portopt")

    assert meta["artifactCount"] == 1
    assert "po_portfolio_returns" in meta["artifactTypes"]


def test_validate_bundle_structure_rejects_missing_required_group():
    bundle = {
        "version": 2,
        "workspace_session": {"reg-results-store": json.dumps({"R1": {"predicted_key": "k1", "residuals_key": "k2"}})},
        "artifact_refs": [],
        "artifacts": [{"key": "k1", "artifact_type": "reg_predicted_series", "format": "feather", "metadata": {}, "payload": ""}],
    }

    try:
        validate_bundle_structure(bundle, "regression")
    except RuntimeError as exc:
        assert "Expected at least 2 artifacts" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for missing residuals artifact.")


def test_tamper_bundle_for_mode_variants():
    bundle = {
        "version": 2,
        "workspace_session": {"po-results-store": "{}"},
        "artifact_refs": [],
        "artifacts": [{"key": "k1", "artifact_type": "po_portfolio_returns", "format": "feather", "metadata": {}, "payload": ""}],
    }

    invalid = tamper_bundle_for_mode(bundle, "invalid_version", "portopt")
    assert invalid["version"] == -1

    missing_ws = tamper_bundle_for_mode(bundle, "missing_workspace_session", "portopt")
    assert "workspace_session" not in missing_ws

    malformed = tamper_bundle_for_mode(bundle, "malformed_json", "portopt")
    assert isinstance(malformed, str)


def test_summarize_cycles_counts_failures_and_medians():
    summary = summarize_cycles(
        [
            {"ok": True, "durationMs": 1000, "bundleBytes": 2000, "bundleArtifactCount": 2},
            {"ok": False, "durationMs": 1500, "bundleBytes": 3000, "bundleArtifactCount": 1, "failureMode": "bad"},
            {"ok": True, "durationMs": 900, "bundleBytes": 2500, "bundleArtifactCount": 3},
        ]
    )

    assert summary["cycles"] == 3
    assert summary["passed"] == 2
    assert summary["failed"] == 1
    assert summary["failureModes"]["bad"] == 1
    assert summary["bundleArtifactCountMax"] == 3


def test_bundle_expectations_for_combined_require_all_groups():
    expectations = bundle_expectations_for_workflow("combined")

    assert expectations["min_artifacts"] == 3
    assert len(expectations["required_any_groups"]) == 3


def test_decode_downloaded_bundle_ignores_trailing_bytes():
    raw = b'{"version":2,"workspace_session":{},"artifact_refs":[],"artifacts":[]}garbage-tail'

    bundle, text = decode_downloaded_bundle(raw)

    assert bundle["version"] == 2
    assert text.endswith("}")


def test_live_seed_for_workflow_filters_to_mounted_prefixes():
    seed = {
        "dashmat-session-id-store": "s1",
        "at-state-ready-store": True,
        "po-results-store": "{}",
        "reg-results-store": "{}",
    }

    filtered = live_seed_for_workflow(seed, "combined")

    assert "dashmat-session-id-store" in filtered
    assert "reg-results-store" in filtered
    assert "po-results-store" not in filtered
    assert "at-state-ready-store" not in filtered
