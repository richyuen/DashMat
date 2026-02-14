from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from _pytest import pathlib as _pytest_pathlib

import cache_config
from utils.returns import df_to_json


# Sandbox environments can deny directory enumeration during tmpdir cleanup.
# Guard pytest's cleanup helper so suite completion does not fail post-tests.
_orig_cleanup_dead_symlinks = _pytest_pathlib.cleanup_dead_symlinks


def _safe_cleanup_dead_symlinks(root):
    try:
        _orig_cleanup_dead_symlinks(root)
    except PermissionError:
        return


_pytest_pathlib.cleanup_dead_symlinks = _safe_cleanup_dead_symlinks


@pytest.fixture(autouse=True)
def _clear_cache_between_tests():
    cache_config.cache.clear()
    yield
    cache_config.cache.clear()


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


@pytest.fixture(scope="session")
def page_modules():
    # Dash page modules call register_page at import time.
    # Import app first so the Dash app exists before page registration.
    import app  # noqa: F401
    import pages.analyticstool as analyticstool
    import pages.portopt as portopt

    return analyticstool, portopt
