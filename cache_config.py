"""Cache configuration for DashMat application.

This module is separate from app.py to avoid circular import issues
when pages are loaded during app initialization.
"""

import os
from functools import wraps
from flask_caching import Cache

from utils.serialization import canonical_json_dumps

# Cache instance (will be initialized in app.py)
_cache = None


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
            from hashlib import md5
            cache_payload = {
                "f": cache_key_prefix,
                "args": args,
                "kwargs": kwargs,
            }
            cache_key = md5(canonical_json_dumps(cache_payload).encode("utf-8")).hexdigest()

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
