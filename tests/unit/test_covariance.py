from __future__ import annotations

import pandas as pd
import pytest

from utils.covariance import (
    covariance_to_correlation,
    estimate_covariance_matrix,
    estimate_mean_vector,
    normalize_cov_shrinkage,
)


def test_normalize_cov_shrinkage_defaults_invalid_values_to_none():
    assert normalize_cov_shrinkage(None) == "none"
    assert normalize_cov_shrinkage("") == "none"
    assert normalize_cov_shrinkage("bad-value") == "none"


def test_estimate_covariance_matrix_supports_ledoit_wolf(sample_returns_df):
    cov_df = estimate_covariance_matrix(
        sample_returns_df,
        asset_order=["Asset_B", "Asset_A"],
        shrinkage="ledoit_wolf",
    )

    assert list(cov_df.index) == ["Asset_B", "Asset_A"]
    assert list(cov_df.columns) == ["Asset_B", "Asset_A"]
    assert cov_df.equals(cov_df.T)


def test_estimate_covariance_matrix_supports_oas(sample_returns_df):
    cov_df = estimate_covariance_matrix(
        sample_returns_df,
        asset_order=["Asset_C", "Asset_D"],
        shrinkage="oas",
    )

    assert list(cov_df.index) == ["Asset_C", "Asset_D"]
    assert list(cov_df.columns) == ["Asset_C", "Asset_D"]
    assert cov_df.equals(cov_df.T)


def test_estimate_covariance_matrix_ignores_shrinkage_when_exp_weighted(sample_returns_df):
    weighted_cov = estimate_covariance_matrix(
        sample_returns_df,
        asset_order=["Asset_A", "Asset_B"],
        exp_weighted=True,
        decay_value=0.94,
        shrinkage="ledoit_wolf",
    )
    weighted_cov_none = estimate_covariance_matrix(
        sample_returns_df,
        asset_order=["Asset_A", "Asset_B"],
        exp_weighted=True,
        decay_value=0.94,
        shrinkage="none",
    )

    pd.testing.assert_frame_equal(weighted_cov, weighted_cov_none)


def test_covariance_to_correlation_handles_zero_variance_diagonal():
    cov_df = pd.DataFrame(
        [[0.0, 0.0], [0.0, 4.0]],
        index=["A", "B"],
        columns=["A", "B"],
    )

    corr_df = covariance_to_correlation(cov_df)

    assert corr_df.loc["A", "A"] == pytest.approx(1.0)
    assert corr_df.loc["B", "B"] == pytest.approx(1.0)
    assert pd.isna(corr_df.loc["A", "B"])


def test_estimate_mean_vector_returns_sample_mean_row(sample_returns_df):
    mean_df = estimate_mean_vector(
        sample_returns_df,
        asset_order=["Asset_B", "Asset_A"],
        exp_weighted=False,
    )

    assert list(mean_df.columns) == ["Asset_B", "Asset_A"]
    assert mean_df.shape == (1, 2)
    assert mean_df.iloc[0]["Asset_B"] == pytest.approx(sample_returns_df["Asset_B"].mean())


def test_estimate_mean_vector_returns_weighted_mean_row(sample_returns_df):
    mean_df = estimate_mean_vector(
        sample_returns_df,
        asset_order=["Asset_A", "Asset_B"],
        exp_weighted=True,
        decay_value=0.94,
    )

    assert list(mean_df.columns) == ["Asset_A", "Asset_B"]
    assert mean_df.shape == (1, 2)
    assert mean_df.notna().all().all()


def test_estimate_covariance_matrix_raises_for_insufficient_shrinkage_overlap():
    sparse_df = pd.DataFrame(
        {
            "Asset_A": [0.01, None, None],
            "Asset_B": [None, 0.02, None],
        }
    )

    with pytest.raises(ValueError, match="Insufficient overlapping observations"):
        estimate_covariance_matrix(sparse_df, shrinkage="ledoit_wolf")
