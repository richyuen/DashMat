"""Cache configuration for DashMat application.

This module is separate from app.py to avoid circular import issues
when pages are loaded during app initialization.
"""

import os
from hashlib import md5
from functools import wraps
from flask_caching import Cache

# Cache instance (will be initialized in app.py)
_cache = None

_CACHE_KEY_INLINE_BYTES = 256


def _build_cache_config() -> dict:
    """Build Flask-Caching configuration from environment variables."""
    cache_type = os.getenv("DASHMAT_CACHE_TYPE", "SimpleCache")
    default_timeout = int(os.getenv("DASHMAT_CACHE_DEFAULT_TIMEOUT", "300"))
    threshold = int(os.getenv("DASHMAT_CACHE_THRESHOLD", "500"))

    config = {
        "CACHE_TYPE": cache_type,
        "CACHE_DEFAULT_TIMEOUT": default_timeout,
    }

    if cache_type == "SimpleCache":
        config["CACHE_THRESHOLD"] = threshold
    elif cache_type == "FileSystemCache":
        config["CACHE_DIR"] = os.getenv("DASHMAT_CACHE_DIR", ".cache/dashmat")
        config["CACHE_THRESHOLD"] = threshold
    elif cache_type == "RedisCache":
        config["CACHE_REDIS_URL"] = os.getenv(
            "DASHMAT_CACHE_REDIS_URL", "redis://localhost:6379/0"
        )

    return config


def init_cache(server):
    """Initialize the cache with the Flask server.

    Args:
        server: Flask server instance from Dash app

    Returns:
        Initialized Cache instance
    """
    global _cache
    _cache = Cache(server, config=_build_cache_config())
    return _cache


def _update_digest_with_bytes(digest, tag: bytes, payload: bytes) -> None:
    digest.update(tag)
    digest.update(len(payload).to_bytes(8, "big", signed=False))
    if len(payload) > _CACHE_KEY_INLINE_BYTES:
        digest.update(b"H")
        digest.update(md5(payload).digest())
        return
    digest.update(b"I")
    digest.update(payload)


def _digest_sort_key(value) -> bytes:
    digest = md5()
    _update_digest_for_cache_key(digest, value)
    return digest.digest()


def _update_digest_for_cache_key(digest, value) -> None:
    if value is None:
        digest.update(b"N")
        return

    if isinstance(value, bool):
        digest.update(b"B1" if value else b"B0")
        return

    if isinstance(value, int):
        _update_digest_with_bytes(digest, b"I", str(value).encode("utf-8"))
        return

    if isinstance(value, float):
        _update_digest_with_bytes(digest, b"F", repr(value).encode("utf-8"))
        return

    if isinstance(value, str):
        _update_digest_with_bytes(digest, b"S", value.encode("utf-8"))
        return

    if isinstance(value, bytes):
        _update_digest_with_bytes(digest, b"Y", value)
        return

    if isinstance(value, dict):
        digest.update(b"D")
        for key in sorted(value.keys(), key=lambda item: str(item)):
            _update_digest_for_cache_key(digest, str(key))
            _update_digest_for_cache_key(digest, value[key])
        digest.update(b"d")
        return

    if isinstance(value, list):
        digest.update(b"L")
        for item in value:
            _update_digest_for_cache_key(digest, item)
        digest.update(b"l")
        return

    if isinstance(value, tuple):
        digest.update(b"T")
        for item in value:
            _update_digest_for_cache_key(digest, item)
        digest.update(b"t")
        return

    if isinstance(value, set):
        digest.update(b"E")
        for item in sorted(value, key=_digest_sort_key):
            _update_digest_for_cache_key(digest, item)
        digest.update(b"e")
        return

    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            _update_digest_for_cache_key(digest, value.item())
            return
        except Exception:
            pass

    type_name = f"{type(value).__module__}.{type(value).__qualname__}"
    digest.update(b"O")
    _update_digest_with_bytes(digest, b"M", type_name.encode("utf-8"))
    _update_digest_with_bytes(digest, b"R", str(value).encode("utf-8"))
    digest.update(b"o")


def _make_memoize_cache_key(prefix: str, args, kwargs) -> str:
    digest = md5()
    _update_digest_for_cache_key(digest, ("f", prefix))
    _update_digest_for_cache_key(digest, ("args", tuple(args)))
    _update_digest_for_cache_key(digest, ("kwargs", kwargs))
    return digest.hexdigest()


def memoize(timeout=300):
    """Lazy memoize decorator that works even when cache isn't initialized yet.

    This decorator defers cache access until the function is actually called,
    allowing it to be used at module import time before the cache is initialized.

    Args:
        timeout: Cache timeout in seconds (default: 300)

    Returns:
        Decorator function
    """
    def decorator(func):
        # Generate a unique cache key for this function
        cache_key_prefix = f"{func.__module__}.{func.__name__}"

        @wraps(func)
        def wrapper(*args, **kwargs):
            # If cache is not initialized, just call the function
            if _cache is None:
                return func(*args, **kwargs)

            # Generate cache key from arguments
            cache_key = _make_memoize_cache_key(cache_key_prefix, args, kwargs)

            # Try to get from cache
            result = _cache.get(cache_key)
            if result is not None:
                return result

            # Compute and cache
            result = func(*args, **kwargs)
            _cache.set(cache_key, result, timeout=timeout)
            return result

        return wrapper
    return decorator


# Provide a cache object that mimics the Flask-Caching interface
class CacheProxy:
    """Proxy object that provides the cache interface."""

    def memoize(self, timeout=300):
        """Memoize decorator."""
        return memoize(timeout=timeout)

    def get(self, key):
        """Get value from cache."""
        if _cache is None:
            return None
        return _cache.get(key)

    def set(self, key, value, timeout=None):
        """Set value in cache."""
        if _cache is not None:
            _cache.set(key, value, timeout=timeout)

    def clear(self):
        """Clear the cache."""
        if _cache is not None:
            _cache.clear()


# Create the proxy instance that can be used like a cache
cache = CacheProxy()
