from __future__ import annotations

import importlib
import logging

import utils.perf_timing as perf_timing


def _reload_perf_timing(monkeypatch, *, enabled: str, min_ms: str = "0", logger_name: str = "dashmat.timing"):
    monkeypatch.setenv("DASHMAT_TIMING_ENABLED", enabled)
    monkeypatch.setenv("DASHMAT_TIMING_MIN_MS", min_ms)
    monkeypatch.setenv("DASHMAT_TIMING_LOGGER", logger_name)
    return importlib.reload(perf_timing)


def test_timed_block_disabled_does_not_log(monkeypatch, capsys):
    logger_name = "dashmat.timing.unit.disabled"
    mod = _reload_perf_timing(monkeypatch, enabled="0", logger_name=logger_name)

    with mod.timed_block("my-block", event="unit-test"):
        pass

    assert "timing name=my-block" not in capsys.readouterr().out


def test_timed_block_enabled_logs_once(monkeypatch, capsys):
    logger_name = "dashmat.timing.unit.enabled"
    mod = _reload_perf_timing(monkeypatch, enabled="1", min_ms="0", logger_name=logger_name)

    with mod.timed_block("my-block", event="unit-test"):
        pass

    output = capsys.readouterr().out
    assert "timing name=my-block" in output
    assert "event=unit-test" in output


def test_timed_block_enabled_below_min_threshold_does_not_log(monkeypatch, capsys):
    logger_name = "dashmat.timing.unit.threshold"
    mod = _reload_perf_timing(monkeypatch, enabled="1", min_ms="100000", logger_name=logger_name)

    with mod.timed_block("slow-block", event="unit-test"):
        pass

    assert "timing name=slow-block" not in capsys.readouterr().out


def test_timed_block_enabled_logs_without_suffix(monkeypatch, capsys):
    logger_name = "dashmat.timing.unit.plain"
    mod = _reload_perf_timing(monkeypatch, enabled="1", min_ms="0", logger_name=logger_name)

    with mod.timed_block("plain-block"):
        pass

    assert "timing name=plain-block" in capsys.readouterr().out


def test_timed_block_allows_dynamic_fields(monkeypatch, capsys):
    logger_name = "dashmat.timing.unit.dynamic"
    mod = _reload_perf_timing(monkeypatch, enabled="1", min_ms="0", logger_name=logger_name)

    with mod.timed_block("dynamic-block", event="unit-test") as fields:
        fields["payload_bytes"] = 123

    output = capsys.readouterr().out
    assert "timing name=dynamic-block" in output
    assert "payload_bytes=123" in output


def test_configure_timing_logger_adds_single_stdout_handler(monkeypatch):
    mod = _reload_perf_timing(monkeypatch, enabled="1", min_ms="0", logger_name="dashmat.timing.test")

    logger = mod.configure_timing_logger()
    logger = mod.configure_timing_logger()

    named_handlers = [handler for handler in logger.handlers if getattr(handler, "name", "") == mod._TIMING_HANDLER_NAME]
    assert len(named_handlers) == 1
    assert logger.propagate is False
    assert logger.level == logging.INFO


def test_env_bool_truthy_and_falsy(monkeypatch):
    monkeypatch.setenv("TMP_BOOL", "yes")
    assert perf_timing._env_bool("TMP_BOOL", "0") is True

    monkeypatch.setenv("TMP_BOOL", "off")
    assert perf_timing._env_bool("TMP_BOOL", "1") is False


def test_format_fields_omits_none():
    assert perf_timing._format_fields({"a": 1, "b": None, "c": "x"}) == "a=1 c=x"
