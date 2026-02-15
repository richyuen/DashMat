from __future__ import annotations

import importlib
import logging

import utils.perf_timing as perf_timing


def _reload_perf_timing(monkeypatch, *, enabled: str, min_ms: str = "0", logger_name: str = "dashmat.timing"):
    monkeypatch.setenv("DASHMAT_TIMING_ENABLED", enabled)
    monkeypatch.setenv("DASHMAT_TIMING_MIN_MS", min_ms)
    monkeypatch.setenv("DASHMAT_TIMING_LOGGER", logger_name)
    return importlib.reload(perf_timing)


def test_timed_block_disabled_does_not_log(monkeypatch, caplog):
    mod = _reload_perf_timing(monkeypatch, enabled="0")
    caplog.set_level(logging.INFO, logger="dashmat.timing")

    with mod.timed_block("my-block", event="unit-test"):
        pass

    assert not any("timing name=my-block" in rec.getMessage() for rec in caplog.records)


def test_timed_block_enabled_logs_once(monkeypatch, caplog):
    mod = _reload_perf_timing(monkeypatch, enabled="1", min_ms="0")
    caplog.set_level(logging.INFO, logger="dashmat.timing")

    with mod.timed_block("my-block", event="unit-test"):
        pass

    assert any("timing name=my-block" in rec.getMessage() for rec in caplog.records)
    assert any("event=unit-test" in rec.getMessage() for rec in caplog.records)


def test_timed_block_enabled_below_min_threshold_does_not_log(monkeypatch, caplog):
    mod = _reload_perf_timing(monkeypatch, enabled="1", min_ms="100000")
    caplog.set_level(logging.INFO, logger="dashmat.timing")

    with mod.timed_block("slow-block", event="unit-test"):
        pass

    assert not any("timing name=slow-block" in rec.getMessage() for rec in caplog.records)


def test_timed_block_enabled_logs_without_suffix(monkeypatch, caplog):
    mod = _reload_perf_timing(monkeypatch, enabled="1", min_ms="0")
    caplog.set_level(logging.INFO, logger="dashmat.timing")

    with mod.timed_block("plain-block"):
        pass

    assert any("timing name=plain-block" in rec.getMessage() for rec in caplog.records)


def test_env_bool_truthy_and_falsy(monkeypatch):
    monkeypatch.setenv("TMP_BOOL", "yes")
    assert perf_timing._env_bool("TMP_BOOL", "0") is True

    monkeypatch.setenv("TMP_BOOL", "off")
    assert perf_timing._env_bool("TMP_BOOL", "1") is False


def test_format_fields_omits_none():
    assert perf_timing._format_fields({"a": 1, "b": None, "c": "x"}) == "a=1 c=x"
