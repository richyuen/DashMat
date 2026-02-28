"""Canonical route helpers for DashMat pages."""

from __future__ import annotations

from urllib.parse import urlencode


HOME_PATH = "/"
LANDING_PATH = "/dashmat"
ANALYTICS_PATH = "/analyticstool"
PORTOPT_PATH = "/portopt"
REGRESSION_PATH = "/regression"

VALID_MODULES = ("analyticstool", "portopt", "regression")

_MODULE_TO_PATH = {
    "analyticstool": ANALYTICS_PATH,
    "portopt": PORTOPT_PATH,
    "regression": REGRESSION_PATH,
}

_MODULE_TO_LABEL = {
    "analyticstool": "Analytics Tool",
    "portopt": "Portfolio Optimization",
    "regression": "Regression",
}


def normalize_landing_module(module_name: str | None) -> str:
    value = str(module_name or "").strip().lower()
    if value in VALID_MODULES:
        return value
    return "analyticstool"


def module_to_workspace_path(module_name: str | None) -> str:
    return _MODULE_TO_PATH[normalize_landing_module(module_name)]


def module_to_label(module_name: str | None) -> str:
    return _MODULE_TO_LABEL[normalize_landing_module(module_name)]


def landing_href(module_name: str | None = None) -> str:
    module_value = normalize_landing_module(module_name)
    return f"{LANDING_PATH}?{urlencode({'module': module_value})}"
