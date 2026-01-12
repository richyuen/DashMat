"""Cache configuration for DashMat application.

This module is separate from app.py to avoid circular import issues
when pages are loaded during app initialization.
"""

from flask_caching import Cache

# Cache instance (will be initialized in app.py)
cache = None


def init_cache(server):
    """Initialize the cache with the Flask server.

    Args:
        server: Flask server instance from Dash app

    Returns:
        Initialized Cache instance
    """
    global cache
    cache = Cache(server, config={
        'CACHE_TYPE': 'SimpleCache',
        'CACHE_DEFAULT_TIMEOUT': 300,  # 5 minutes
        'CACHE_THRESHOLD': 100,  # Maximum number of cached items
    })
    return cache
