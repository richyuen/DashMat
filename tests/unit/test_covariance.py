from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from utils.covariance import (
    _constant_correlation_target,
    covariance_to_correlation,
    estimate_covariance_matrix,
    estimate_mean_vector,
    format_cov_shrinkage_spec_label,
    normalize_cov_shrinkage,
    normalize_cov_shrinkage_target,
    resolve_cov_shrinkage_spec,
)


def test_normalize_cov_shrinkage_defaults_invalid_values_to_none():
    assert normalize_cov_shrinkage(None) == "none"
    assert normalize_cov_shrinkage("") == "none"
    assert normalize_cov_shrinkage("bad-value") == "none"


def test_normalize_cov_shrinkage_target_defaults_invalid_values_to_scaled_identity():
    assert normalize_cov_shrinkage_target(None) == "scaled_identity"
    assert normalize_cov_shrinkage_target("") == "scaled_identity"
    assert normalize_cov_shrinkage_target("bad-value") == "scaled_identity"


def test_resolve_cov_shrinkage_spec_enforces_effective_rules():
    assert resolve_cov_shrinkage_spec("none", "constant_correlation") == ("none", "scaled_identity")
    assert resolve_cov_shrinkage_spec("oas", "constant_correlation") == ("oas", "scaled_identity")
    assert resolve_cov_shrinkage_spec("ledoit_wolf", "constant_correlation") == (
        "ledoit_wolf",
        "constant_correlation",
    )
    assert resolve_cov_shrinkage_spec(
        "ledoit_wolf",
        "constant_correlation",
        exp_weighted=True,
    ) == ("none", "scaled_identity")


def test_format_cov_shrinkage_spec_label_formats_ledoit_wolf_targets():
    assert format_cov_shrinkage_spec_label("ledoit_wolf", "scaled_identity") == (
        "Ledoit-Wolf, Scaled Identity"
    )
    assert format_cov_shrinkage_spec_label("ledoit_wolf", "constant_correlation") == (
        "Ledoit-Wolf, Constant Correlation"
    )


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


def test_estimate_covariance_matrix_supports_constant_correlation_target(sample_returns_df):
    cov_df = estimate_covariance_matrix(
        sample_returns_df,
        asset_order=["Asset_C", "Asset_A", "Asset_B"],
        shrinkage="ledoit_wolf",
        shrinkage_target="constant_correlation",
    )

    assert list(cov_df.index) == ["Asset_C", "Asset_A", "Asset_B"]
    assert list(cov_df.columns) == ["Asset_C", "Asset_A", "Asset_B"]
    assert cov_df.equals(cov_df.T)


def test_estimate_covariance_matrix_ignores_shrinkage_when_exp_weighted(sample_returns_df):
    weighted_cov = estimate_covariance_matrix(
        sample_returns_df,
        asset_order=["Asset_A", "Asset_B"],
        exp_weighted=True,
        decay_value=0.94,
        shrinkage="ledoit_wolf",
        shrinkage_target="constant_correlation",
    )
    weighted_cov_none = estimate_covariance_matrix(
        sample_returns_df,
        asset_order=["Asset_A", "Asset_B"],
        exp_weighted=True,
        decay_value=0.94,
        shrinkage="none",
    )

    pd.testing.assert_frame_equal(weighted_cov, weighted_cov_none)


def test_estimate_covariance_matrix_ignores_target_for_oas(sample_returns_df):
    oas_constant = estimate_covariance_matrix(
        sample_returns_df,
        asset_order=["Asset_A", "Asset_B", "Asset_C"],
        shrinkage="oas",
        shrinkage_target="constant_correlation",
    )
    oas_scaled_identity = estimate_covariance_matrix(
        sample_returns_df,
        asset_order=["Asset_A", "Asset_B", "Asset_C"],
        shrinkage="oas",
        shrinkage_target="scaled_identity",
    )

    pd.testing.assert_frame_equal(oas_constant, oas_scaled_identity)


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


def test_constant_correlation_target_has_constant_off_diagonal_correlations():
    sample_cov = np.array(
        [
            [0.09, 0.018, 0.045],
            [0.018, 0.04, 0.03],
            [0.045, 0.03, 0.25],
        ],
        dtype=float,
    )

    target_cov, avg_corr = _constant_correlation_target(sample_cov)
    std = np.sqrt(np.diag(target_cov))
    implied_corr = target_cov / np.outer(std, std)

    assert np.diag(target_cov) == pytest.approx(np.diag(sample_cov))
    assert implied_corr[0, 1] == pytest.approx(avg_corr)
    assert implied_corr[0, 2] == pytest.approx(avg_corr)
    assert implied_corr[1, 2] == pytest.approx(avg_corr)


def test_constant_correlation_target_zeros_pairs_with_zero_variance():
    sample_cov = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.04, 0.01],
            [0.0, 0.01, 0.09],
        ],
        dtype=float,
    )

    target_cov, _ = _constant_correlation_target(sample_cov)

    assert target_cov[0, 1] == pytest.approx(0.0)
    assert target_cov[0, 2] == pytest.approx(0.0)
    assert np.isfinite(target_cov).all()


def test_estimate_covariance_matrix_raises_for_insufficient_shrinkage_overlap():
    sparse_df = pd.DataFrame(
        {
            "Asset_A": [0.01, None, None],
            "Asset_B": [None, 0.02, None],
        }
    )

    with pytest.raises(ValueError, match="Insufficient overlapping observations"):
        estimate_covariance_matrix(sparse_df, shrinkage="ledoit_wolf")


def test_estimate_covariance_matrix_constant_correlation_raises_for_insufficient_overlap():
    sparse_df = pd.DataFrame(
        {
            "Asset_A": [0.01, None, None],
            "Asset_B": [None, 0.02, None],
        }
    )

    with pytest.raises(ValueError, match="Insufficient overlapping observations"):
        estimate_covariance_matrix(
            sparse_df,
            shrinkage="ledoit_wolf",
            shrinkage_target="constant_correlation",
        )
