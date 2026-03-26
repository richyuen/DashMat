from __future__ import annotations

import re
from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd
import pytest
from _pytest import pathlib as _pytest_pathlib
from dash import html

import cache_config
from utils.raw_dataset import clear_raw_dataset_cache
from utils.returns import df_to_json
from utils.raw_dataset import build_raw_data_store_payload


# Sandbox environments can deny directory enumeration during tmpdir cleanup.
# Guard pytest's cleanup helper so suite completion does not fail post-tests.
_orig_cleanup_dead_symlinks = _pytest_pathlib.cleanup_dead_symlinks


def _safe_cleanup_dead_symlinks(root):
    try:
        _orig_cleanup_dead_symlinks(root)
    except PermissionError:
        return


_pytest_pathlib.cleanup_dead_symlinks = _safe_cleanup_dead_symlinks


def _noop_callback(*_args, **_kwargs):
    def _decorator(func):
        return func

    return _decorator


def _noop_clientside_callback(*_args, **_kwargs):
    return None


def _noop_register_page(*_args, **_kwargs):
    return None


def _load_analyticstool_advanced_source():
    source_path = Path(__file__).resolve().parents[1] / "utils" / "analyticstool_advanced_source.py.txt"
    source_text = source_path.read_text(encoding="utf-8")
    dash_import_pattern = re.compile(
        r"from dash import .*?callback_context\s*",
        re.DOTALL,
    )
    source_text = dash_import_pattern.sub(
        (
            "from dash import ClientsideFunction, Input, Output, State, dcc, html, no_update, ALL, callback_context\n"
            "callback = _noop_callback\n"
            "clientside_callback = _noop_clientside_callback\n"
            "register_page = _noop_register_page\n"
        ),
        source_text,
        count=1,
    )

    module = types.ModuleType("tests_analyticstool_advanced_source")
    module.__file__ = str(source_path)
    module.__dict__.update(
        {
            "__name__": module.__name__,
            "_noop_callback": _noop_callback,
            "_noop_clientside_callback": _noop_clientside_callback,
            "_noop_register_page": _noop_register_page,
        }
    )
    sys.modules[module.__name__] = module
    exec(compile(source_text, str(source_path), "exec"), module.__dict__)
    return module


class _AnalyticsToolProxy:
    def __init__(self, main_module, advanced_module):
        object.__setattr__(self, "_main_module", main_module)
        object.__setattr__(self, "_advanced_module", advanced_module)
        object.__setattr__(self, "layout", html.Div([main_module.layout, advanced_module.layout]))

    def __getattr__(self, name):
        if hasattr(self._main_module, name):
            return getattr(self._main_module, name)
        return getattr(self._advanced_module, name)

    def __setattr__(self, name, value):
        if name in {"_main_module", "_advanced_module", "layout"}:
            object.__setattr__(self, name, value)
            return
        if name == "callback_context":
            setattr(self._main_module, name, value)
            setattr(self._advanced_module, name, value)
            return
        main_has = hasattr(self._main_module, name)
        advanced_has = hasattr(self._advanced_module, name)
        if main_has and advanced_has:
            setattr(self._main_module, name, value)
            setattr(self._advanced_module, name, value)
            return
        if main_has:
            setattr(self._main_module, name, value)
            return
        setattr(self._advanced_module, name, value)

    def at_restore_secondary_controls(self, *args, **kwargs):
        return self._advanced_module.at_restore_secondary_controls(*args, **kwargs)

    def sync_at_returns_type_from_mirrors(self, *args, **kwargs):
        return self._advanced_module.sync_at_returns_type_from_mirrors(*args, **kwargs)

    def sync_at_returns_type_mirrors(self, *args, **kwargs):
        return self._advanced_module.sync_at_returns_type_mirrors(*args, **kwargs)

    def download_excel(self, *args, **kwargs):
        return self._advanced_module.download_excel(*args, **kwargs)


@pytest.fixture(autouse=True)
def _clear_cache_between_tests():
    cache_config.cache.clear()
    clear_raw_dataset_cache()
    yield
    cache_config.cache.clear()
    clear_raw_dataset_cache()


@pytest.fixture(scope="session")
def sample_returns_df() -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    dates = pd.date_range("2023-01-02", periods=320, freq="B")

    factor_1 = rng.normal(0.0003, 0.009, len(dates))
    factor_2 = rng.normal(0.0001, 0.007, len(dates))
    noise = rng.normal(0.0, 0.0025, size=(len(dates), 4))

    data = {
        "Asset_A": 0.9 * factor_1 + 0.2 * factor_2 + noise[:, 0],
        "Asset_B": 0.4 * factor_1 + 0.6 * factor_2 + noise[:, 1],
        "Asset_C": -0.1 * factor_1 + 0.9 * factor_2 + noise[:, 2],
        "Asset_D": 0.2 * factor_1 - 0.3 * factor_2 + noise[:, 3],
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = "Date"
    return df


@pytest.fixture
def raw_json(sample_returns_df: pd.DataFrame) -> str:
    return df_to_json(sample_returns_df)


@pytest.fixture
def raw_data_store(sample_returns_df: pd.DataFrame) -> dict:
    return build_raw_data_store_payload(sample_returns_df)


@pytest.fixture(scope="session")
def page_modules():
    # Dash page modules call register_page at import time.
    # Import app first so the Dash app exists before page registration.
    import app  # noqa: F401
    import pages.analyticstool as analyticstool
    import pages.portopt as portopt
    analyticstool_advanced = _load_analyticstool_advanced_source()

    return _AnalyticsToolProxy(analyticstool, analyticstool_advanced), portopt
