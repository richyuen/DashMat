"""Lightweight timing utilities for callback-level latency instrumentation."""

from __future__ import annotations

from contextlib import contextmanager
import logging
import os
import time
from typing import Any


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
