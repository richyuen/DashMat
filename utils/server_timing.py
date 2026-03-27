from __future__ import annotations

import time
from typing import Callable

from utils.perf_timing import timing_enabled


_REGISTERED_FLAG = "_dashmat_server_timing_hooks_registered"
_DASH_UPDATE_PATH = "/_dash-update-component"


def register_server_timing_hooks(server) -> None:
    if not timing_enabled() or getattr(server, _REGISTERED_FLAG, False):
        return

    inner_wsgi_app = server.wsgi_app

    def _timed_wsgi_app(environ, start_response):
        path = str(environ.get("PATH_INFO") or "")
        if path != _DASH_UPDATE_PATH:
            return inner_wsgi_app(environ, start_response)

        started_at = time.perf_counter()

        def _timed_start_response(
            status: str,
            headers: list[tuple[str, str]],
            exc_info=None,
        ) -> Callable[[bytes], object]:
            duration_ms = (time.perf_counter() - started_at) * 1000.0
            header_value = f"cb;dur={duration_ms:.2f}"
            updated_headers: list[tuple[str, str]] = []
            appended = False
            for key, value in headers:
                if key.lower() == "server-timing":
                    updated_headers.append((key, f"{value}, {header_value}"))
                    appended = True
                else:
                    updated_headers.append((key, value))
            if not appended:
                updated_headers.append(("Server-Timing", header_value))
            return start_response(status, updated_headers, exc_info)

        return inner_wsgi_app(environ, _timed_start_response)

    server.wsgi_app = _timed_wsgi_app
    setattr(server, _REGISTERED_FLAG, True)
