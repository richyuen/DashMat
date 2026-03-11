from __future__ import annotations

from datetime import timedelta, timezone, datetime
import shutil

import pandas as pd

from utils.artifact_store import ArtifactStore, store_raw_data_artifact
from utils.returns import df_to_json


def test_artifact_store_creates_and_lists_session(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    session_id = store.create_session("session-a")

    payload = {"status": "ok"}
    descriptor = store.put_json(
        data=payload,
        artifact_type="status",
        session_id=session_id,
        payload={"status": "ok"},
    )

    listed = store.list_session_artifacts(session_id)

    assert session_id == "session-a"
    assert len(listed) == 1
    assert listed[0].key == descriptor.key
    assert store.get_json(descriptor.key) == payload


def test_artifact_store_dataframe_round_trip_preserves_datetime_index(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    session_id = store.create_session("session-b")
    index = pd.date_range("2024-01-31", periods=3, freq="ME", name="Date")
    df = pd.DataFrame({"Asset_A": [0.01, 0.02, -0.01]}, index=index)

    descriptor = store.put_dataframe(
        df=df,
        artifact_type="returns",
        session_id=session_id,
        payload={"name": "returns"},
    )

    restored = store.get_dataframe(descriptor.key)

    pd.testing.assert_frame_equal(restored, df, check_freq=False)


def test_artifact_store_build_key_is_deterministic(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")

    key_a = store.build_key("raw_data", {"x": 1, "y": [2, 3]}, session_id="s")
    key_b = store.build_key("raw_data", {"y": [2, 3], "x": 1}, session_id="s")

    assert key_a == key_b


def test_artifact_store_cleanup_expired_removes_artifact(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    session_id = store.create_session("session-c")
    descriptor = store.put_json(
        data={"expired": True},
        artifact_type="status",
        session_id=session_id,
        payload={"expired": True},
        ttl_seconds=1,
    )

    removed = store.cleanup_expired(now=datetime.now(timezone.utc) + timedelta(seconds=5))

    assert removed == 1
    assert store.get_descriptor(descriptor.key) is None


def test_store_raw_data_artifact_returns_compact_descriptor(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    df = pd.DataFrame(
        {"Asset_A": [0.01, 0.02], "Asset_B": [0.03, -0.01]},
        index=pd.to_datetime(["2024-01-31", "2024-02-29"]),
    )
    df.index.name = "Date"

    descriptor = store_raw_data_artifact(
        session_id="session-d",
        raw_data_json=df_to_json(df),
        original_periodicity="monthly",
        store=store,
    )

    assert descriptor is not None
    assert descriptor["has_data"] is True
    assert descriptor["row_count"] == 2
    assert descriptor["col_count"] == 2
    assert descriptor["columns"] == ["Asset_A", "Asset_B"]
    restored = store.get_dataframe(descriptor["raw_data_key"])
    pd.testing.assert_frame_equal(restored, df)


def test_artifact_store_recreates_schema_after_root_deletion(tmp_path):
    root = tmp_path / "artifacts"
    store = ArtifactStore(root)
    store.create_session("session-z")

    shutil.rmtree(root)

    descriptor = store.put_json(
        data={"restored": True},
        artifact_type="status",
        session_id="session-z",
        payload={"restored": True},
    )

    assert store.get_json(descriptor.key) == {"restored": True}
