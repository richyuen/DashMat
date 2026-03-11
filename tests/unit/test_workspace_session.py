from __future__ import annotations

import json

import pandas as pd

from utils.artifact_store import ArtifactStore
from utils.saved_series_cache import (
    build_saved_series_cache_descriptor,
    load_saved_series_cache_frame,
    saved_series_cache_is_fresh,
    series_json_from_saved_series_cache,
)
from utils.returns import df_to_json
from utils.workspace_session import (
    WORKSPACE_SESSION_BUNDLE_VERSION,
    build_workspace_session_bundle,
    remap_workspace_artifact_refs,
    restore_workspace_session_bundle,
)


def test_saved_series_cache_descriptor_round_trip(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    idx = pd.to_datetime(["2024-01-31", "2024-02-29"])
    frame = pd.DataFrame(
        {
            "BCTBill13_TRIndex": [0.001, 0.0011],
            "SPX_TRIndex": [0.02, -0.01],
        },
        index=idx,
    )
    frame.index.name = "Date"

    descriptor = build_saved_series_cache_descriptor(
        session_id="session-a",
        saved_df=frame,
        series_max_dates={
            "BCTBill13_TRIndex": "2024-02-29",
            "SPX_TRIndex": "2024-02-29",
        },
        raw_data_json=df_to_json(frame[["SPX_TRIndex"]]),
        store=store,
    )

    assert descriptor is not None
    assert descriptor["row_count"] == 2
    assert saved_series_cache_is_fresh(descriptor, pd.Timestamp("2024-02-29"), store=store) is True

    restored = load_saved_series_cache_frame(descriptor, store=store)
    pd.testing.assert_frame_equal(restored, frame, check_freq=False)
    assert series_json_from_saved_series_cache(descriptor, "BCTBill13_TRIndex", store=store)


def test_workspace_session_bundle_round_trip_restores_required_artifacts(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    session_id = store.create_session("session-export")

    po_frame = pd.DataFrame({"MyPort": [0.01, 0.02]}, index=pd.to_datetime(["2024-01-01", "2024-01-02"]))
    po_frame.index.name = "Date"
    reg_frame = pd.DataFrame({"predicted": [0.03, 0.04]}, index=pd.to_datetime(["2024-01-01", "2024-01-02"]))
    reg_frame.index.name = "Date"

    po_desc = store.put_dataframe(df=po_frame, artifact_type="po_portfolio_returns", session_id=session_id, payload={"name": "MyPort"})
    reg_desc = store.put_dataframe(df=reg_frame, artifact_type="reg_predicted_series", session_id=session_id, payload={"name": "Run1"})

    workspace_session = {
        "dashmat-session-id-store": json.dumps(session_id),
        "po-results-store": json.dumps({"MyPort": {"returns_key": po_desc.key, "config": {}}}),
        "reg-results-store": json.dumps({"Run1": {"predicted_key": reg_desc.key}}),
        "dashmat-saved-series-cache-store": json.dumps({"cache_key": "unused"}),
        "dashmat-raw-data-artifact-store": json.dumps({"raw_data_key": "unused"}),
    }

    bundle = build_workspace_session_bundle(workspace_session, store=store)

    assert bundle["version"] == WORKSPACE_SESSION_BUNDLE_VERSION
    assert "dashmat-saved-series-cache-store" not in bundle["workspace_session"]
    assert "dashmat-raw-data-artifact-store" not in bundle["workspace_session"]
    assert len(bundle["artifacts"]) == 2

    restored = restore_workspace_session_bundle(bundle, store=store)

    assert "error" not in restored
    restored_session = restored["workspace_session"]
    assert "dashmat-saved-series-cache-store" not in restored_session
    assert "dashmat-raw-data-artifact-store" not in restored_session

    po_payload = json.loads(restored_session["po-results-store"])
    reg_payload = json.loads(restored_session["reg-results-store"])
    assert po_payload["MyPort"]["returns_key"] != po_desc.key
    assert reg_payload["Run1"]["predicted_key"] != reg_desc.key
    pd.testing.assert_frame_equal(store.get_dataframe(po_payload["MyPort"]["returns_key"]), po_frame, check_freq=False)
    pd.testing.assert_frame_equal(store.get_dataframe(reg_payload["Run1"]["predicted_key"]), reg_frame, check_freq=False)


def test_remap_workspace_artifact_refs_updates_nested_keys():
    session_payload = {
        "po-results-store": json.dumps({"MyPort": {"returns_key": "old-key"}}),
    }
    refs = [{"store_key": "po-results-store", "path": ["MyPort", "returns_key"], "artifact_key": "old-key"}]

    remapped = remap_workspace_artifact_refs(session_payload, refs, {"old-key": "new-key"})

    assert json.loads(remapped["po-results-store"])["MyPort"]["returns_key"] == "new-key"


def test_restore_workspace_session_bundle_rejects_invalid_version(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")

    restored = restore_workspace_session_bundle({"version": -1}, store=store)

    assert restored["error"] == "Unsupported session bundle version."


def test_build_workspace_session_bundle_warns_on_missing_artifact(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    workspace_session = {
        "po-results-store": json.dumps({"MyPort": {"returns_key": "missing-key"}}),
    }

    bundle = build_workspace_session_bundle(workspace_session, store=store)

    assert bundle["artifacts"] == []
    assert bundle["export_warnings"]
