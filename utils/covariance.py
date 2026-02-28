"""Shared covariance and mean estimation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf, OAS

from utils.exponential_weighting import normalize_decay_input, resolve_ewm_params


VALID_COV_SHRINKAGE = {"none", "ledoit_wolf", "oas"}

_COV_SHRINKAGE_LABELS = {
    "none": "None",
    "ledoit_wolf": "Ledoit-Wolf",
    "oas": "OAS",
}


def normalize_cov_shrinkage(value: str | None) -> str:
    """Return a supported covariance shrinkage value."""
    normalized = str(value or "").strip().lower()
    if normalized in VALID_COV_SHRINKAGE:
        return normalized
    return "none"


def format_cov_shrinkage_label(value: str | None) -> str:
    """Return display label for a covariance shrinkage option."""
    return _COV_SHRINKAGE_LABELS.get(normalize_cov_shrinkage(value), "None")


def _ordered_columns(
    returns_df: pd.DataFrame,
    asset_order: list[str] | tuple[str, ...] | None,
) -> list[str]:
    if asset_order is None:
        return list(returns_df.columns)
    return [col for col in asset_order if col in returns_df.columns]


def _finalize_covariance_frame(
    cov_df: pd.DataFrame,
    columns: list[str],
    annualization_factor: float,
) -> pd.DataFrame:
    cov_df = cov_df.reindex(index=columns, columns=columns)
    values = cov_df.to_numpy(dtype=float, copy=True)
    values = (values + values.T) / 2.0
    values *= float(annualization_factor)
    return pd.DataFrame(values, index=columns, columns=columns)


def covariance_to_correlation(cov_df: pd.DataFrame) -> pd.DataFrame:
    """Convert covariance matrix to correlation matrix safely."""
    values = cov_df.to_numpy(dtype=float, copy=False)
    std = np.sqrt(np.clip(np.diag(values), a_min=0.0, a_max=None))
    denom = np.outer(std, std)
    corr_values = np.divide(
        values,
        denom,
        out=np.full_like(values, np.nan, dtype=float),
        where=denom > 0,
    )
    np.fill_diagonal(corr_values, 1.0)
    return pd.DataFrame(corr_values, index=cov_df.index, columns=cov_df.columns)


def estimate_covariance_matrix(
    returns_df: pd.DataFrame,
    *,
    asset_order: list[str] | tuple[str, ...] | None = None,
    exp_weighted: bool = False,
    decay_value: float = 63.0,
    shrinkage: str = "none",
    annualization_factor: float = 1.0,
) -> pd.DataFrame:
    """Estimate a covariance matrix using sample, EWM, or shrinkage estimators."""
    columns = _ordered_columns(returns_df, asset_order)
    if not columns:
        return pd.DataFrame()

    working_df = returns_df.loc[:, columns]
    if exp_weighted:
        ewm_cov = working_df.ewm(
            **resolve_ewm_params(normalize_decay_input(decay_value, 63.0))
        ).cov().iloc[-len(columns):]
        if isinstance(ewm_cov.index, pd.MultiIndex):
            ewm_cov.index = ewm_cov.index.get_level_values(-1)
        return _finalize_covariance_frame(ewm_cov, columns, annualization_factor)

    shrinkage_method = normalize_cov_shrinkage(shrinkage)
    if shrinkage_method == "none":
        return _finalize_covariance_frame(
            working_df.cov(),
            columns,
            annualization_factor,
        )

    clean_df = working_df.dropna()
    if clean_df.shape[0] < 2 or clean_df.shape[1] < 2:
        raise ValueError("Insufficient overlapping observations for shrinkage covariance estimate.")

    estimator = LedoitWolf() if shrinkage_method == "ledoit_wolf" else OAS()
    estimator.fit(clean_df.to_numpy(dtype=float, copy=False))
    cov_df = pd.DataFrame(estimator.covariance_, index=columns, columns=columns)
    return _finalize_covariance_frame(cov_df, columns, annualization_factor)


def estimate_mean_vector(
    returns_df: pd.DataFrame,
    *,
    asset_order: list[str] | tuple[str, ...] | None = None,
    exp_weighted: bool = False,
    decay_value: float = 63.0,
) -> pd.DataFrame:
    """Estimate a one-row mean vector aligned to asset order."""
    columns = _ordered_columns(returns_df, asset_order)
    if not columns:
        return pd.DataFrame()

    working_df = returns_df.loc[:, columns]
    if exp_weighted:
        mean_series = working_df.ewm(
            **resolve_ewm_params(normalize_decay_input(decay_value, 63.0))
        ).mean().iloc[-1]
    else:
        mean_series = working_df.mean()

    mean_series = mean_series.reindex(columns)
    return mean_series.to_frame().T
