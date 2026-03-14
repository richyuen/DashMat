from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import cache_config


@dataclass
class _FakeCache:
    store: dict[str, object] = field(default_factory=dict)
    clear_called: bool = False

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value, timeout=None):
        self.store[key] = value

    def clear(self):
        self.clear_called = True
        self.store.clear()


class _StringyObject:
    def __init__(self, label: str):
        self.label = label

    def __str__(self):
        return self.label


def test_build_cache_config_defaults(monkeypatch):
    monkeypatch.delenv("DASHMAT_CACHE_TYPE", raising=False)
    monkeypatch.delenv("DASHMAT_CACHE_DEFAULT_TIMEOUT", raising=False)
    monkeypatch.delenv("DASHMAT_CACHE_THRESHOLD", raising=False)

    config = cache_config._build_cache_config()
    assert config["CACHE_TYPE"] == "SimpleCache"
    assert config["CACHE_DEFAULT_TIMEOUT"] == 300
    assert config["CACHE_THRESHOLD"] == 500


def test_build_cache_config_for_redis(monkeypatch):
    monkeypatch.setenv("DASHMAT_CACHE_TYPE", "RedisCache")
    monkeypatch.setenv("DASHMAT_CACHE_REDIS_URL", "redis://example:6379/3")

    config = cache_config._build_cache_config()
    assert config["CACHE_TYPE"] == "RedisCache"
    assert config["CACHE_REDIS_URL"] == "redis://example:6379/3"


def test_build_cache_config_for_filesystem(monkeypatch):
    monkeypatch.setenv("DASHMAT_CACHE_TYPE", "FileSystemCache")
    monkeypatch.setenv("DASHMAT_CACHE_DIR", "tmp/cache-dir")
    monkeypatch.setenv("DASHMAT_CACHE_THRESHOLD", "321")

    config = cache_config._build_cache_config()
    assert config["CACHE_TYPE"] == "FileSystemCache"
    assert config["CACHE_DIR"] == "tmp/cache-dir"
    assert config["CACHE_THRESHOLD"] == 321


def test_memoize_without_initialized_cache_calls_function_each_time(monkeypatch):
    monkeypatch.setattr(cache_config, "_cache", None)
    calls = {"n": 0}

    @cache_config.memoize(timeout=60)
    def add_one(x):
        calls["n"] += 1
        return x + 1

    assert add_one(1) == 2
    assert add_one(1) == 2
    assert calls["n"] == 2


def test_memoize_with_cache_reuses_cached_result(monkeypatch):
    fake = _FakeCache()
    monkeypatch.setattr(cache_config, "_cache", fake)
    calls = {"n": 0}

    @cache_config.memoize(timeout=60)
    def multiply(a, b=1):
        calls["n"] += 1
        return a * b

    assert multiply(3, b=4) == 12
    assert multiply(3, b=4) == 12
    assert calls["n"] == 1


def test_make_memoize_cache_key_is_stable_for_reordered_kwargs():
    first = cache_config._make_memoize_cache_key(
        "demo.fn",
        ("value", 1),
        {"alpha": 1, "beta": 2},
    )
    second = cache_config._make_memoize_cache_key(
        "demo.fn",
        ("value", 1),
        {"beta": 2, "alpha": 1},
    )

    assert first == second


def test_make_memoize_cache_key_changes_when_value_changes():
    first = cache_config._make_memoize_cache_key("demo.fn", ("value",), {"alpha": 1})
    second = cache_config._make_memoize_cache_key("demo.fn", ("value",), {"alpha": 2})

    assert first != second


def test_make_memoize_cache_key_hashes_large_strings():
    payload = "x" * 2048

    first = cache_config._make_memoize_cache_key("demo.fn", (payload,), {})
    second = cache_config._make_memoize_cache_key("demo.fn", (payload,), {})

    assert first == second


def test_make_memoize_cache_key_supports_unknown_objects():
    first = cache_config._make_memoize_cache_key("demo.fn", (_StringyObject("engine://demo"),), {})
    second = cache_config._make_memoize_cache_key("demo.fn", (_StringyObject("engine://demo"),), {})
    different = cache_config._make_memoize_cache_key("demo.fn", (_StringyObject("engine://other"),), {})

    assert first == second
    assert first != different


def test_cache_proxy_clear_delegates_to_backend(monkeypatch):
    fake = _FakeCache()
    monkeypatch.setattr(cache_config, "_cache", fake)
    cache_config.cache.set("k", "v")
    cache_config.cache.clear()
    assert fake.clear_called is True
    assert cache_config.cache.get("k") is None


def test_cache_proxy_no_backend_is_noop(monkeypatch):
    monkeypatch.setattr(cache_config, "_cache", None)

    assert cache_config.cache.get("missing") is None
    cache_config.cache.set("k", "v")
    cache_config.cache.clear()


def test_init_cache_uses_cache_constructor(monkeypatch):
    captured = {}

    class _FakeCacheCtor:
        def __init__(self, server, config):
            captured["server"] = server
            captured["config"] = config
            self.store = {}

        def get(self, key):
            return self.store.get(key)

        def set(self, key, value, timeout=None):
            self.store[key] = value

        def clear(self):
            self.store.clear()

    monkeypatch.setattr(cache_config, "Cache", _FakeCacheCtor)
    monkeypatch.setenv("DASHMAT_CACHE_TYPE", "SimpleCache")
    monkeypatch.setenv("DASHMAT_CACHE_DEFAULT_TIMEOUT", "111")
    monkeypatch.setenv("DASHMAT_CACHE_THRESHOLD", "222")
    server = SimpleNamespace(name="fake-server")

    built = cache_config.init_cache(server)

    assert isinstance(built, _FakeCacheCtor)
    assert captured["server"] is server
    assert captured["config"]["CACHE_TYPE"] == "SimpleCache"
    assert captured["config"]["CACHE_DEFAULT_TIMEOUT"] == 111
    assert captured["config"]["CACHE_THRESHOLD"] == 222
