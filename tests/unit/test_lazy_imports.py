from __future__ import annotations

import utils.covariance as covariance
import utils.optimization as optimization
import utils.statistics as statistics


def test_riskfolio_loader_is_cached():
    optimization._import_riskfolio.cache_clear()

    first = optimization._import_riskfolio()
    second = optimization._import_riskfolio()

    assert first is second


def test_scipy_optimize_loader_is_cached():
    optimization._import_scipy_optimize.cache_clear()

    first = optimization._import_scipy_optimize()
    second = optimization._import_scipy_optimize()

    assert first is second
    assert first[0] is second[0]
    assert first[1] is second[1]


def test_sklearn_covariance_loader_is_cached():
    covariance._import_sklearn_covariance.cache_clear()

    first = covariance._import_sklearn_covariance()
    second = covariance._import_sklearn_covariance()

    assert first is second
    assert first[0] is second[0]
    assert first[1] is second[1]


def test_scipy_stats_loader_is_cached():
    statistics._import_scipy_stats.cache_clear()

    first = statistics._import_scipy_stats()
    second = statistics._import_scipy_stats()

    assert first is second
