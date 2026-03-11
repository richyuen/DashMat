from __future__ import annotations

from base64 import b64decode, b64encode
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from utils.artifact_store import ArtifactDescriptor, ArtifactStore, get_default_artifact_store
from utils.serialization import canonical_json_dumps

WORKSPACE_SESSION_BUNDLE_VERSION = 2
EPHEMERAL_WORKSPACE_KEYS = {
    "dashmat-raw-data-artifact-store",
    "dashmat-saved-series-cache-store",
}


def build_workspace_session_bundle(
    workspace_session: dict[str, str] | None,
    *,
    store: ArtifactStore | None = None,
) -> dict[str, Any]:
    artifact_store = store or get_default_artifact_store()
    source_payload = {
        str(key): value
        for key, value in (workspace_session or {}).items()
        if isinstance(key, str) and isinstance(value, str)
    }
    cleaned_payload = {
        key: value
        for key, value in source_payload.items()
        if key not in EPHEMERAL_WORKSPACE_KEYS
    }
    artifact_refs = collect_workspace_artifact_refs(cleaned_payload)
    artifacts = []
    warnings = []

    seen_keys: set[str] = set()
    for ref in artifact_refs:
        artifact_key = ref.get("artifact_key")
        if not isinstance(artifact_key, str) or not artifact_key or artifact_key in seen_keys:
            continue
        seen_keys.add(artifact_key)
        descriptor = artifact_store.get_descriptor(artifact_key)
        if descriptor is None:
            warnings.append(f"Missing artifact for {ref['store_key']}:{'.'.join(ref['path'])}")
            continue
        artifacts.append(_serialize_artifact_record(descriptor, artifact_store))

    bundle = {
        "version": WORKSPACE_SESSION_BUNDLE_VERSION,
        "workspace_session": cleaned_payload,
        "artifact_refs": artifact_refs,
        "artifacts": artifacts,
    }
    if warnings:
        bundle["export_warnings"] = warnings
    return bundle


def restore_workspace_session_bundle(
    bundle: dict[str, Any] | None,
    *,
    store: ArtifactStore | None = None,
) -> dict[str, Any]:
    artifact_store = store or get_default_artifact_store()
    if not isinstance(bundle, dict):
        return {"error": "Invalid session bundle."}
    if bundle.get("version") != WORKSPACE_SESSION_BUNDLE_VERSION:
        return {"error": "Unsupported session bundle version."}

    workspace_session = bundle.get("workspace_session")
    artifact_refs = bundle.get("artifact_refs")
    artifact_records = bundle.get("artifacts")
    if not isinstance(workspace_session, dict) or not isinstance(artifact_refs, list) or not isinstance(artifact_records, list):
        return {"error": "Malformed session bundle."}

    restored_payload = {
        str(key): value
        for key, value in workspace_session.items()
        if isinstance(key, str) and isinstance(value, str)
    }
    for key in EPHEMERAL_WORKSPACE_KEYS:
        restored_payload.pop(key, None)

    new_session_id = str(uuid4())
    artifact_store.create_session(new_session_id)
    artifact_records_by_key = {
        record.get("key"): record
        for record in artifact_records
        if isinstance(record, dict) and isinstance(record.get("key"), str)
    }
    key_map: dict[str, str] = {}
    warnings = []
    required_restore_errors = []

    for ref in artifact_refs:
        artifact_key = ref.get("artifact_key")
        if not isinstance(artifact_key, str) or not artifact_key or artifact_key in key_map:
            continue
        record = artifact_records_by_key.get(artifact_key)
        if record is None:
            message = f"Missing bundled artifact for {ref.get('store_key')}:{'.'.join(ref.get('path') or [])}"
            if ref.get("required"):
                required_restore_errors.append(message)
            else:
                warnings.append(message)
            continue
        try:
            key_map[artifact_key] = _restore_artifact_record(record, new_session_id, artifact_store)
        except Exception as exc:
            message = f"Failed to restore artifact {artifact_key}: {exc}"
            if ref.get("required"):
                required_restore_errors.append(message)
            else:
                warnings.append(message)

    if required_restore_errors:
        return {"error": "; ".join(required_restore_errors)}

    remapped_payload = remap_workspace_artifact_refs(restored_payload, artifact_refs, key_map)
    remapped_payload["dashmat-session-id-store"] = json.dumps(new_session_id)
    remapped_payload.pop("dashmat-saved-series-cache-store", None)

    result = {
        "workspace_session": remapped_payload,
        "session_id": new_session_id,
    }
    if warnings:
        result["warnings"] = warnings
    return result


def collect_workspace_artifact_refs(workspace_session: dict[str, str] | None) -> list[dict[str, Any]]:
    payload = workspace_session or {}
    refs: list[dict[str, Any]] = []
    refs.extend(_collect_raw_data_artifact_refs(payload.get("dashmat-raw-data-store")))
    refs.extend(_collect_portopt_artifact_refs(payload.get("po-results-store")))
    refs.extend(_collect_regression_artifact_refs(payload.get("reg-results-store")))
    return refs


def remap_workspace_artifact_refs(
    workspace_session: dict[str, str] | None,
    artifact_refs: list[dict[str, Any]],
    key_map: dict[str, str],
) -> dict[str, str]:
    payload = {
        str(key): value
        for key, value in (workspace_session or {}).items()
        if isinstance(key, str) and isinstance(value, str)
    }
    grouped: dict[str, list[dict[str, Any]]] = {}
    for ref in artifact_refs or []:
        store_key = ref.get("store_key")
        if isinstance(store_key, str):
            grouped.setdefault(store_key, []).append(ref)

    for store_key, refs in grouped.items():
        raw_value = payload.get(store_key)
        if not isinstance(raw_value, str):
            continue
        parsed, nested_string = _parse_store_value(raw_value)
        if parsed is None:
            continue
        updated = deepcopy(parsed)
        changed = False
        for ref in refs:
            old_key = ref.get("artifact_key")
            new_key = key_map.get(old_key)
            if not new_key:
                continue
            path = ref.get("path")
            if _set_nested_path(updated, path, new_key):
                changed = True
        if changed:
            encoded = canonical_json_dumps(updated)
            payload[store_key] = json.dumps(encoded) if nested_string else encoded
    return payload


def _collect_portopt_artifact_refs(raw_value: str | None) -> list[dict[str, Any]]:
    results = _parse_json_object(raw_value)
    refs = []
    for portfolio_name, entry in results.items():
        if not isinstance(entry, dict):
            continue
        artifact_key = entry.get("returns_key")
        if isinstance(artifact_key, str) and artifact_key:
            refs.append(
                {
                    "store_key": "po-results-store",
                    "path": [str(portfolio_name), "returns_key"],
                    "artifact_key": artifact_key,
                    "required": True,
                }
            )
    return refs


def _collect_raw_data_artifact_refs(raw_value: str | None) -> list[dict[str, Any]]:
    descriptor = _parse_json_object(raw_value, unwrap_nested_string=True)
    artifact_key = descriptor.get("raw_data_key")
    if isinstance(artifact_key, str) and artifact_key:
        return [
            {
                "store_key": "dashmat-raw-data-store",
                "path": ["raw_data_key"],
                "artifact_key": artifact_key,
                "required": True,
            }
        ]
    return []


def _collect_regression_artifact_refs(raw_value: str | None) -> list[dict[str, Any]]:
    results = _parse_json_object(raw_value)
    refs = []
    for result_name, entry in results.items():
        if not isinstance(entry, dict):
            continue
        for field_name in ("predicted_key", "residuals_key"):
            artifact_key = entry.get(field_name)
            if isinstance(artifact_key, str) and artifact_key:
                refs.append(
                    {
                        "store_key": "reg-results-store",
                        "path": [str(result_name), field_name],
                        "artifact_key": artifact_key,
                        "required": True,
                    }
                )
    return refs


def _serialize_artifact_record(descriptor: ArtifactDescriptor, store: ArtifactStore) -> dict[str, Any]:
    path = Path(store.root) / descriptor.path
    record = {
        "key": descriptor.key,
        "artifact_type": descriptor.artifact_type,
        "format": descriptor.format,
        "metadata": dict(descriptor.metadata),
    }
    if descriptor.format == "feather":
        record["payload"] = b64encode(path.read_bytes()).decode("ascii")
    elif descriptor.format == "json":
        record["payload"] = json.loads(path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"Unsupported artifact format: {descriptor.format}")
    return record


def _restore_artifact_record(record: dict[str, Any], session_id: str, store: ArtifactStore) -> str:
    artifact_type = str(record.get("artifact_type") or "")
    format_name = str(record.get("format") or "")
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    if format_name == "feather":
        payload = record.get("payload")
        if not isinstance(payload, str):
            raise ValueError("Missing feather payload.")
        frame = pd.read_feather(BytesIO(b64decode(payload.encode("ascii"))))
        index_name = str(metadata.get("index_name") or "index")
        if index_name not in frame.columns:
            raise ValueError(f"Missing index column {index_name}.")
        frame = frame.set_index(index_name)
        if str(metadata.get("index_dtype") or "").startswith("datetime64"):
            frame.index = pd.to_datetime(frame.index)
        frame.index.name = index_name
        descriptor = store.put_dataframe(
            df=frame,
            artifact_type=artifact_type,
            session_id=session_id,
            metadata=metadata,
            payload={"imported_from": record.get("key"), "artifact_type": artifact_type},
        )
        return descriptor.key
    if format_name == "json":
        descriptor = store.put_json(
            data=record.get("payload"),
            artifact_type=artifact_type,
            session_id=session_id,
            metadata=metadata,
            payload={"imported_from": record.get("key"), "artifact_type": artifact_type},
        )
        return descriptor.key
    raise ValueError(f"Unsupported artifact format: {format_name}")


def _parse_json_object(raw_value: str | None, *, unwrap_nested_string: bool = False) -> dict[str, Any]:
    parsed, nested_string = _parse_store_value(raw_value)
    if isinstance(parsed, dict):
        return parsed
    if unwrap_nested_string and nested_string and isinstance(parsed, dict):
        return parsed
    return {}


def _parse_store_value(raw_value: str | None) -> tuple[Any | None, bool]:
    if not isinstance(raw_value, str) or not raw_value:
        return None, False
    try:
        parsed = json.loads(raw_value)
    except Exception:
        return None, False
    if isinstance(parsed, str):
        try:
            nested = json.loads(parsed)
        except Exception:
            return parsed, False
        return nested, True
    return parsed, False


def _set_nested_path(obj: Any, path: Any, value: Any) -> bool:
    if not isinstance(path, list) or not path:
        return False
    current = obj
    for key in path[:-1]:
        if not isinstance(current, dict) or key not in current:
            return False
        current = current[key]
    if not isinstance(current, dict):
        return False
    current[path[-1]] = value
    return True
