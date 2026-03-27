"""Lightweight timing utilities for callback-level latency instrumentation."""

from __future__ import annotations

from contextlib import contextmanager
import logging
import os
import sys
import time
from typing import Any


def _env_bool(name: str, default: str = "0") -> bool:
    val = os.getenv(name, default).strip().lower()
    return val in {"1", "true", "yes", "on"}


_TIMING_ENABLED = _env_bool("DASHMAT_TIMING_ENABLED", "0")
_TIMING_MIN_MS = float(os.getenv("DASHMAT_TIMING_MIN_MS", "0"))
_TIMING_LOGGER = logging.getLogger(os.getenv("DASHMAT_TIMING_LOGGER", "dashmat.timing"))
_TIMING_HANDLER_NAME = "dashmat.timing.stdout"


def _format_fields(fields: dict[str, Any]) -> str:
    parts = []
    for key, value in fields.items():
        if value is None:
            continue
        parts.append(f"{key}={value}")
    return " ".join(parts)


def timing_enabled() -> bool:
    return _TIMING_ENABLED


def configure_timing_logger() -> logging.Logger:
    """Attach a dedicated stdout handler for timing logs when enabled."""
    if not _TIMING_ENABLED:
        return _TIMING_LOGGER

    for handler in _TIMING_LOGGER.handlers:
        if getattr(handler, "name", "") == _TIMING_HANDLER_NAME:
            _TIMING_LOGGER.setLevel(logging.INFO)
            _TIMING_LOGGER.propagate = False
            return _TIMING_LOGGER

    handler = logging.StreamHandler(sys.stdout)
    handler.set_name(_TIMING_HANDLER_NAME)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))
    _TIMING_LOGGER.addHandler(handler)
    _TIMING_LOGGER.setLevel(logging.INFO)
    _TIMING_LOGGER.propagate = False
    return _TIMING_LOGGER


@contextmanager
def timed_block(name: str, **fields: Any):
    """Measure a code block and emit a structured timing log when enabled."""
    if not _TIMING_ENABLED:
        yield fields
        return

    if not _TIMING_LOGGER.handlers:
        configure_timing_logger()
    start = time.perf_counter()
    try:
        yield fields
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if elapsed_ms < _TIMING_MIN_MS:
            return
        suffix = _format_fields(fields)
        if suffix:
            _TIMING_LOGGER.info("timing name=%s elapsed_ms=%.2f %s", name, elapsed_ms, suffix)
        else:
            _TIMING_LOGGER.info("timing name=%s elapsed_ms=%.2f", name, elapsed_ms)
