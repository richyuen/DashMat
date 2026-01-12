"""Cache configuration for DashMat application.

This module is separate from app.py to avoid circular import issues
when pages are loaded during app initialization.
"""

from functools import wraps
from flask_caching import Cache

# Cache instance (will be initialized in app.py)
_cache = None


def init_cache(server):
    """Initialize the cache with the Flask server.

    Args:
        server: Flask server instance from Dash app

    Returns:
        Initialized Cache instance
    """
    global _cache
    _cache = Cache(server, config={
        'CACHE_TYPE': 'SimpleCache',
        'CACHE_DEFAULT_TIMEOUT': 300,  # 5 minutes
        'CACHE_THRESHOLD': 100,  # Maximum number of cached items
    })
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
            key_parts = [cache_key_prefix]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            cache_key = md5("".join(key_parts).encode()).hexdigest()

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
