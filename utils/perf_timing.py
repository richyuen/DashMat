"""Lightweight timing utilities for callback-level latency instrumentation."""

from __future__ import annotations

from contextlib import contextmanager
import logging
import os
import time
from typing import Any

from utils.serialization import canonical_json_dumps


def _env_bool(name: str, default: str = "0") -> bool:
    val = os.getenv(name, default).strip().lower()
    return val in {"1", "true", "yes", "on"}


_TIMING_ENABLED = _env_bool("DASHMAT_TIMING_ENABLED", "0")
_TIMING_MIN_MS = float(os.getenv("DASHMAT_TIMING_MIN_MS", "0"))
_TIMING_LOGGER = logging.getLogger(os.getenv("DASHMAT_TIMING_LOGGER", "dashmat.timing"))


def _format_fields(fields: dict[str, Any]) -> str:
    parts = []
    for key, value in fields.items():
        if value is None:
            continue
        parts.append(f"{key}={value}")
    return " ".join(parts)


def estimate_payload_bytes(payload: Any) -> int:
    """Best-effort UTF-8 payload size estimate for structured logging."""
    if payload is None:
        return 0
    if isinstance(payload, bytes):
        return len(payload)
    if isinstance(payload, str):
        return len(payload.encode("utf-8"))
    try:
        return len(canonical_json_dumps(payload).encode("utf-8"))
    except TypeError:
        return len(str(payload).encode("utf-8"))


def record_metric(name: str, value: float | int | None, **fields: Any) -> None:
    """Emit a structured metric log when timing instrumentation is enabled."""
    if not _TIMING_ENABLED or value is None:
        return
    suffix = _format_fields(fields)
    if suffix:
        _TIMING_LOGGER.info("metric name=%s value=%s %s", name, value, suffix)
    else:
        _TIMING_LOGGER.info("metric name=%s value=%s", name, value)


def record_payload_size(name: str, payload: Any, **fields: Any) -> int:
    """Measure and log a payload size in bytes when instrumentation is enabled."""
    payload_bytes = estimate_payload_bytes(payload)
    if not _TIMING_ENABLED:
        return payload_bytes
    record_metric(f"{name}.bytes", payload_bytes, **fields)
    return payload_bytes


@contextmanager
def timed_block(name: str, **fields: Any):
    """Measure a code block and emit a structured timing log when enabled."""
    if not _TIMING_ENABLED:
        yield
        return

    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if elapsed_ms < _TIMING_MIN_MS:
            return
        suffix = _format_fields(fields)
        if suffix:
            _TIMING_LOGGER.info("timing name=%s elapsed_ms=%.2f %s", name, elapsed_ms, suffix)
        else:
            _TIMING_LOGGER.info("timing name=%s elapsed_ms=%.2f", name, elapsed_ms)
