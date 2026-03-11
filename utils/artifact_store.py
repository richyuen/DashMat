"""Local disk-backed artifact storage for large session-scoped payloads."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import md5
from io import StringIO
import json
import os
from pathlib import Path
import sqlite3
from typing import Any
from uuid import uuid4

import pandas as pd

from utils.serialization import canonical_json_dumps
from utils.returns import get_available_periodicities, json_to_df

_DEFAULT_ARTIFACT_ROOT = Path(".cache") / "dashmat_artifacts"
_ARTIFACT_DB_NAME = "manifest.sqlite3"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _dt_to_iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat()


def _iso_to_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


def _normalize_artifact_root(root: str | Path | None) -> Path:
    if root is None:
        return Path(os.getenv("DASHMAT_ARTIFACT_ROOT", str(_DEFAULT_ARTIFACT_ROOT)))
    return Path(root)


@dataclass(frozen=True)
class ArtifactDescriptor:
    key: str
    session_id: str
    artifact_type: str
    format: str
    path: str
    created_at: str
    updated_at: str
    expires_at: str | None
    row_count: int | None
    col_count: int | None
    byte_size: int | None
    parent_keys: list[str]
    metadata: dict[str, Any]

    def to_payload(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "session_id": self.session_id,
            "artifact_type": self.artifact_type,
            "format": self.format,
            "path": self.path,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "expires_at": self.expires_at,
            "row_count": self.row_count,
            "col_count": self.col_count,
            "byte_size": self.byte_size,
            "parent_keys": list(self.parent_keys),
            "metadata": dict(self.metadata),
        }


class ArtifactStore:
    """Artifact store backed by a SQLite manifest and local artifact files."""

    def __init__(self, root: str | Path | None = None):
        self.root = _normalize_artifact_root(root)
        self.manifest_path = self.root / _ARTIFACT_DB_NAME
        self.frames_dir = self.root / "frames"
        self.json_dir = self.root / "json"
        self._initialized = False

    def _schema_ready(self) -> bool:
        if not self.manifest_path.exists():
            return False
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT name FROM sqlite_master
                    WHERE type='table' AND name IN ('sessions', 'artifacts')
                    """
                ).fetchall()
        except sqlite3.DatabaseError:
            return False
        return {row[0] for row in rows} == {"sessions", "artifacts"}

    def _paths_ready(self) -> bool:
        return self.root.exists() and self.frames_dir.exists() and self.json_dir.exists()

    def _is_missing_manifest_table(self, exc: Exception) -> bool:
        return isinstance(exc, sqlite3.OperationalError) and "no such table" in str(exc).lower()

    def initialize(self) -> None:
        if self._initialized and self._paths_ready() and self._schema_ready():
            return
        self.root.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.json_dir.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS artifacts (
                    key TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    artifact_type TEXT NOT NULL,
                    format TEXT NOT NULL,
                    path TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    expires_at TEXT,
                    row_count INTEGER,
                    col_count INTEGER,
                    byte_size INTEGER,
                    parent_keys_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_artifacts_session_id ON artifacts(session_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_artifacts_expires_at ON artifacts(expires_at)"
            )
        self._initialized = True

    @contextmanager
    def _connect(self):
        self.root.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.manifest_path)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            yield conn
            conn.commit()
        finally:
            conn.close()

    def create_session(self, session_id: str | None = None) -> str:
        self.initialize()
        resolved = str(session_id or uuid4())
        now_iso = _dt_to_iso(_utc_now())
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO sessions(session_id, created_at, updated_at)
                    VALUES(?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET updated_at=excluded.updated_at
                    """,
                    (resolved, now_iso, now_iso),
                )
        except Exception as exc:
            if not self._is_missing_manifest_table(exc):
                raise
            self._initialized = False
            self.initialize()
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO sessions(session_id, created_at, updated_at)
                    VALUES(?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET updated_at=excluded.updated_at
                    """,
                    (resolved, now_iso, now_iso),
                )
        return resolved

    def touch_session(self, session_id: str) -> None:
        self.create_session(session_id)

    def build_key(self, artifact_type: str, payload: Any, *, session_id: str | None = None) -> str:
        key_payload = {
            "artifact_type": artifact_type,
            "payload": payload,
            "session_id": session_id,
        }
        return md5(canonical_json_dumps(key_payload).encode("utf-8")).hexdigest()

    def put_dataframe(
        self,
        *,
        df: pd.DataFrame,
        artifact_type: str,
        session_id: str,
        payload: Any | None = None,
        key: str | None = None,
        metadata: dict[str, Any] | None = None,
        parent_keys: list[str] | tuple[str, ...] | None = None,
        ttl_seconds: int | None = None,
    ) -> ArtifactDescriptor:
        self.initialize()
        resolved_key = key or self.build_key(
            artifact_type,
            payload if payload is not None else {"nonce": str(uuid4())},
            session_id=session_id,
        )
        self.touch_session(session_id)
        path = self.frames_dir / f"{resolved_key}.feather"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = df.reset_index()
        frame.to_feather(path)
        descriptor = self._write_descriptor(
            key=resolved_key,
            session_id=session_id,
            artifact_type=artifact_type,
            format_name="feather",
            path=path,
            row_count=int(df.shape[0]),
            col_count=int(df.shape[1]),
            byte_size=path.stat().st_size,
            metadata={
                "index_name": df.index.name or "index",
                "index_dtype": str(df.index.dtype),
                **(metadata or {}),
            },
            parent_keys=list(parent_keys or []),
            ttl_seconds=ttl_seconds,
        )
        return descriptor

    def get_dataframe(self, key: str) -> pd.DataFrame:
        descriptor = self.get_descriptor(key)
        if descriptor is None or descriptor.format != "feather":
            raise KeyError(key)
        path = self.root / descriptor.path
        if not path.exists():
            raise KeyError(key)
        frame = pd.read_feather(path)
        index_name = str(descriptor.metadata.get("index_name") or "index")
        if index_name not in frame.columns:
            raise KeyError(key)
        frame = frame.set_index(index_name)
        if descriptor.metadata.get("index_dtype", "").startswith("datetime64"):
            frame.index = pd.to_datetime(frame.index)
        frame.index.name = index_name
        return frame

    def put_json(
        self,
        *,
        data: Any,
        artifact_type: str,
        session_id: str,
        payload: Any | None = None,
        key: str | None = None,
        metadata: dict[str, Any] | None = None,
        parent_keys: list[str] | tuple[str, ...] | None = None,
        ttl_seconds: int | None = None,
    ) -> ArtifactDescriptor:
        self.initialize()
        resolved_key = key or self.build_key(
            artifact_type,
            payload if payload is not None else {"nonce": str(uuid4())},
            session_id=session_id,
        )
        self.touch_session(session_id)
        path = self.json_dir / f"{resolved_key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload_text = canonical_json_dumps(data)
        path.write_text(payload_text, encoding="utf-8")
        return self._write_descriptor(
            key=resolved_key,
            session_id=session_id,
            artifact_type=artifact_type,
            format_name="json",
            path=path,
            row_count=None,
            col_count=None,
            byte_size=path.stat().st_size,
            metadata=metadata or {},
            parent_keys=list(parent_keys or []),
            ttl_seconds=ttl_seconds,
        )

    def get_json(self, key: str) -> Any:
        descriptor = self.get_descriptor(key)
        if descriptor is None or descriptor.format != "json":
            raise KeyError(key)
        path = self.root / descriptor.path
        if not path.exists():
            raise KeyError(key)
        return json.loads(path.read_text(encoding="utf-8"))

    def get_descriptor(self, key: str) -> ArtifactDescriptor | None:
        self.initialize()
        try:
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT key, session_id, artifact_type, format, path, created_at, updated_at,
                           expires_at, row_count, col_count, byte_size, parent_keys_json, metadata_json
                    FROM artifacts WHERE key = ?
                    """,
                    (key,),
                ).fetchone()
        except Exception as exc:
            if not self._is_missing_manifest_table(exc):
                raise
            self._initialized = False
            self.initialize()
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT key, session_id, artifact_type, format, path, created_at, updated_at,
                           expires_at, row_count, col_count, byte_size, parent_keys_json, metadata_json
                    FROM artifacts WHERE key = ?
                    """,
                    (key,),
                ).fetchone()
        if row is None:
            return None
        return ArtifactDescriptor(
            key=row[0],
            session_id=row[1],
            artifact_type=row[2],
            format=row[3],
            path=row[4],
            created_at=row[5],
            updated_at=row[6],
            expires_at=row[7],
            row_count=row[8],
            col_count=row[9],
            byte_size=row[10],
            parent_keys=json.loads(row[11]),
            metadata=json.loads(row[12]),
        )

    def list_session_artifacts(self, session_id: str) -> list[ArtifactDescriptor]:
        self.initialize()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT key FROM artifacts
                WHERE session_id = ?
                ORDER BY created_at ASC, key ASC
                """,
                (session_id,),
            ).fetchall()
        return [desc for row in rows if (desc := self.get_descriptor(row[0])) is not None]

    def delete(self, key: str) -> bool:
        descriptor = self.get_descriptor(key)
        if descriptor is None:
            return False
        path = self.root / descriptor.path
        if path.exists():
            path.unlink()
        with self._connect() as conn:
            conn.execute("DELETE FROM artifacts WHERE key = ?", (key,))
        return True

    def cleanup_expired(self, *, now: datetime | None = None) -> int:
        self.initialize()
        resolved_now = now or _utc_now()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT key, expires_at FROM artifacts
                WHERE expires_at IS NOT NULL
                """
            ).fetchall()
        expired = [
            row[0]
            for row in rows
            if (expires_at := _iso_to_dt(row[1])) is not None and expires_at <= resolved_now
        ]
        removed = 0
        for key in expired:
            removed += 1 if self.delete(key) else 0
        return removed

    def _write_descriptor(
        self,
        *,
        key: str,
        session_id: str,
        artifact_type: str,
        format_name: str,
        path: Path,
        row_count: int | None,
        col_count: int | None,
        byte_size: int | None,
        metadata: dict[str, Any],
        parent_keys: list[str],
        ttl_seconds: int | None,
    ) -> ArtifactDescriptor:
        created_at = _dt_to_iso(_utc_now())
        updated_at = created_at
        expires_at = _dt_to_iso(_utc_now() + timedelta(seconds=ttl_seconds)) if ttl_seconds else None
        rel_path = path.resolve().relative_to(self.root.resolve()).as_posix()
        metadata_json = canonical_json_dumps(metadata or {})
        parent_json = canonical_json_dumps(list(parent_keys or []))
        try:
            with self._connect() as conn:
                existing = conn.execute("SELECT created_at FROM artifacts WHERE key = ?", (key,)).fetchone()
                if existing is not None:
                    created_at = existing[0]
                conn.execute(
                    """
                    INSERT INTO artifacts(
                        key, session_id, artifact_type, format, path, created_at, updated_at,
                        expires_at, row_count, col_count, byte_size, parent_keys_json, metadata_json
                    )
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        session_id=excluded.session_id,
                        artifact_type=excluded.artifact_type,
                        format=excluded.format,
                        path=excluded.path,
                        updated_at=excluded.updated_at,
                        expires_at=excluded.expires_at,
                        row_count=excluded.row_count,
                        col_count=excluded.col_count,
                        byte_size=excluded.byte_size,
                        parent_keys_json=excluded.parent_keys_json,
                        metadata_json=excluded.metadata_json
                    """,
                    (
                        key,
                        session_id,
                        artifact_type,
                        format_name,
                        rel_path,
                        created_at,
                        updated_at,
                        expires_at,
                        row_count,
                        col_count,
                        byte_size,
                        parent_json,
                        metadata_json,
                    ),
                )
        except Exception as exc:
            if not self._is_missing_manifest_table(exc):
                raise
            self._initialized = False
            self.initialize()
            with self._connect() as conn:
                existing = conn.execute("SELECT created_at FROM artifacts WHERE key = ?", (key,)).fetchone()
                if existing is not None:
                    created_at = existing[0]
                conn.execute(
                    """
                    INSERT INTO artifacts(
                        key, session_id, artifact_type, format, path, created_at, updated_at,
                        expires_at, row_count, col_count, byte_size, parent_keys_json, metadata_json
                    )
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        session_id=excluded.session_id,
                        artifact_type=excluded.artifact_type,
                        format=excluded.format,
                        path=excluded.path,
                        updated_at=excluded.updated_at,
                        expires_at=excluded.expires_at,
                        row_count=excluded.row_count,
                        col_count=excluded.col_count,
                        byte_size=excluded.byte_size,
                        parent_keys_json=excluded.parent_keys_json,
                        metadata_json=excluded.metadata_json
                    """,
                    (
                        key,
                        session_id,
                        artifact_type,
                        format_name,
                        rel_path,
                        created_at,
                        updated_at,
                        expires_at,
                        row_count,
                        col_count,
                        byte_size,
                        parent_json,
                        metadata_json,
                    ),
                )
        return ArtifactDescriptor(
            key=key,
            session_id=session_id,
            artifact_type=artifact_type,
            format=format_name,
            path=rel_path,
            created_at=created_at,
            updated_at=updated_at,
            expires_at=expires_at,
            row_count=row_count,
            col_count=col_count,
            byte_size=byte_size,
            parent_keys=list(parent_keys or []),
            metadata=dict(metadata or {}),
        )


_default_store: ArtifactStore | None = None


def get_default_artifact_store() -> ArtifactStore:
    global _default_store
    if _default_store is None:
        _default_store = ArtifactStore()
        _default_store.initialize()
    return _default_store


def get_dataframe_artifact(key: str | None, *, store: ArtifactStore | None = None) -> pd.DataFrame:
    """Resolve a dataframe artifact key to a DataFrame, returning empty on misses."""
    if not key:
        return pd.DataFrame()
    artifact_store = store or get_default_artifact_store()
    try:
        return artifact_store.get_dataframe(key)
    except KeyError:
        return pd.DataFrame()


RAW_DATA_DESCRIPTOR_VERSION = 1


def normalize_raw_data_descriptor(raw_data_store: Any) -> dict[str, Any] | None:
    if isinstance(raw_data_store, dict):
        descriptor = dict(raw_data_store)
    elif isinstance(raw_data_store, str):
        if not raw_data_store.strip():
            return None
        try:
            parsed = json.loads(raw_data_store)
        except Exception:
            return None
        if not isinstance(parsed, dict):
            return None
        descriptor = dict(parsed)
    else:
        return None

    raw_key = descriptor.get("raw_data_key")
    if not isinstance(raw_key, str) or not raw_key:
        return None
    return descriptor


def serialize_raw_data_descriptor(descriptor: dict[str, Any] | None) -> str | None:
    if not isinstance(descriptor, dict) or not descriptor.get("raw_data_key"):
        return None
    return canonical_json_dumps(descriptor)


def _build_raw_data_metadata_from_frame(df: pd.DataFrame | None, original_periodicity: str | None) -> dict[str, Any]:
    resolved_periodicity = original_periodicity or "daily"
    periodicity_options = get_available_periodicities(resolved_periodicity)
    valid_values = [option["value"] for option in periodicity_options]
    default_periodicity = (
        resolved_periodicity
        if resolved_periodicity in valid_values
        else (valid_values[0] if valid_values else "daily_trading")
    )

    if df is None or df.empty:
        return {
            "has_data": False,
            "columns": [],
            "original_periodicity": resolved_periodicity,
            "periodicity_options": periodicity_options,
            "default_periodicity": default_periodicity,
            "min_date": None,
            "max_date": None,
        }

    return {
        "has_data": bool(df.columns.size),
        "columns": list(df.columns),
        "original_periodicity": resolved_periodicity,
        "periodicity_options": periodicity_options,
        "default_periodicity": default_periodicity,
        "min_date": df.index.min().strftime("%Y-%m-%d"),
        "max_date": df.index.max().strftime("%Y-%m-%d"),
    }


def build_raw_data_store_metadata(
    raw_data_store: Any,
    original_periodicity: str | None,
    *,
    store: ArtifactStore | None = None,
) -> dict[str, Any]:
    descriptor = normalize_raw_data_descriptor(raw_data_store)
    if descriptor is None:
        if isinstance(raw_data_store, str) and raw_data_store.strip():
            try:
                legacy_df = json_to_df(raw_data_store)
            except Exception:
                legacy_df = pd.DataFrame()
            return _build_raw_data_metadata_from_frame(legacy_df, original_periodicity)
        return _build_raw_data_metadata_from_frame(None, original_periodicity)

    metadata = {
        "has_data": bool(descriptor.get("has_data")),
        "columns": list(descriptor.get("columns") or []),
        "original_periodicity": descriptor.get("original_periodicity") or (original_periodicity or "daily"),
        "periodicity_options": get_available_periodicities(
            descriptor.get("original_periodicity") or (original_periodicity or "daily")
        ),
        "default_periodicity": None,
        "min_date": descriptor.get("start_date"),
        "max_date": descriptor.get("end_date"),
    }
    valid_values = [option["value"] for option in metadata["periodicity_options"]]
    resolved_periodicity = metadata["original_periodicity"]
    metadata["default_periodicity"] = (
        resolved_periodicity
        if resolved_periodicity in valid_values
        else (valid_values[0] if valid_values else "daily_trading")
    )
    if metadata["has_data"] and metadata["columns"]:
        return metadata

    frame = load_raw_data_frame(raw_data_store, store=store)
    return _build_raw_data_metadata_from_frame(frame, original_periodicity)


def load_raw_data_frame(
    raw_data_store: Any,
    *,
    store: ArtifactStore | None = None,
) -> pd.DataFrame:
    descriptor = normalize_raw_data_descriptor(raw_data_store)
    if descriptor is None:
        if isinstance(raw_data_store, str) and raw_data_store.strip():
            try:
                frame = pd.read_json(StringIO(raw_data_store), orient="split")
            except (ValueError, TypeError):
                return pd.DataFrame()
            frame.index = pd.to_datetime(frame.index)
            frame.index.name = "Date"
            return frame
        return pd.DataFrame()
    return get_dataframe_artifact(descriptor.get("raw_data_key"), store=store)


def write_raw_data_frame(
    *,
    df: pd.DataFrame,
    session_id: str,
    original_periodicity: str | None,
    store: ArtifactStore | None = None,
    parent_keys: list[str] | tuple[str, ...] | None = None,
) -> tuple[str | None, dict[str, Any]]:
    if df is None or df.empty or not session_id:
        metadata = _build_raw_data_metadata_from_frame(df, original_periodicity)
        return None, metadata

    raw_df = df.copy().sort_index()
    raw_df.index = pd.to_datetime(raw_df.index)
    raw_df.index.name = raw_df.index.name or "Date"
    metadata = _build_raw_data_metadata_from_frame(raw_df, original_periodicity)
    artifact_store = store or get_default_artifact_store()
    raw_json = raw_df.to_json(date_format="iso", orient="split")
    raw_hash = md5(raw_json.encode("utf-8")).hexdigest()
    key = artifact_store.build_key(
        "raw_data",
        {"raw_hash": raw_hash, "original_periodicity": original_periodicity or "daily"},
        session_id=session_id,
    )
    descriptor = artifact_store.put_dataframe(
        df=raw_df,
        artifact_type="raw_data",
        session_id=session_id,
        key=key,
        payload={"raw_hash": raw_hash, "original_periodicity": original_periodicity or "daily"},
        metadata={
            "raw_hash": raw_hash,
            "columns": list(raw_df.columns),
            "original_periodicity": original_periodicity or "daily",
            "min_date": metadata.get("min_date"),
            "max_date": metadata.get("max_date"),
        },
        parent_keys=list(parent_keys or []),
    )
    payload = {
        "raw_data_key": descriptor.key,
        "has_data": metadata.get("has_data", False),
        "columns": metadata.get("columns", []),
        "row_count": descriptor.row_count or 0,
        "start_date": metadata.get("min_date"),
        "end_date": metadata.get("max_date"),
        "original_periodicity": metadata.get("original_periodicity"),
        "metadata_version": RAW_DATA_DESCRIPTOR_VERSION,
    }
    return serialize_raw_data_descriptor(payload), metadata


def mutate_raw_data_store(
    raw_data_store: Any,
    *,
    session_id: str | None,
    original_periodicity: str | None,
    mutation_fn,
    store: ArtifactStore | None = None,
) -> tuple[str | None, dict[str, Any]]:
    descriptor = normalize_raw_data_descriptor(raw_data_store)
    parent_keys = []
    resolved_session_id = str(session_id or "")
    resolved_periodicity = original_periodicity
    if descriptor is not None:
        parent_keys = [descriptor["raw_data_key"]]
        resolved_periodicity = resolved_periodicity or descriptor.get("original_periodicity")
        artifact_store = store or get_default_artifact_store()
        current_descriptor = artifact_store.get_descriptor(descriptor["raw_data_key"])
        if current_descriptor is not None and not resolved_session_id:
            resolved_session_id = current_descriptor.session_id
    if not resolved_session_id:
        raise ValueError("Session id is required to mutate raw data.")

    frame = load_raw_data_frame(raw_data_store, store=store)
    updated = mutation_fn(frame.copy())
    if updated is None or updated.empty:
        metadata = _build_raw_data_metadata_from_frame(updated, original_periodicity)
        return None, metadata
    return write_raw_data_frame(
        df=updated,
        session_id=resolved_session_id,
        original_periodicity=resolved_periodicity,
        store=store,
        parent_keys=parent_keys,
    )


def store_raw_data_artifact(
    *,
    session_id: str,
    raw_data_json: str | None,
    original_periodicity: str | None,
    store: ArtifactStore | None = None,
) -> dict[str, Any] | None:
    """Persist raw data JSON to the artifact store and return a compact descriptor."""
    if not raw_data_json:
        return None
    df = json_to_df(raw_data_json)
    payload, metadata = write_raw_data_frame(
        df=df,
        session_id=session_id,
        original_periodicity=original_periodicity,
        store=store,
    )
    descriptor = normalize_raw_data_descriptor(payload)
    if descriptor is None:
        return None
    return {
        "has_data": metadata.get("has_data", False),
        "raw_data_key": descriptor["raw_data_key"],
        "row_count": descriptor.get("row_count", 0),
        "col_count": len(metadata.get("columns", [])),
        "columns": metadata.get("columns", []),
        "min_date": metadata.get("min_date"),
        "max_date": metadata.get("max_date"),
        "original_periodicity": metadata.get("original_periodicity"),
        "periodicity_options": metadata.get("periodicity_options", []),
        "default_periodicity": metadata.get("default_periodicity"),
    }
