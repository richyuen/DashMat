"""Shared covariance and mean estimation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf, OAS

from utils.exponential_weighting import normalize_decay_input, resolve_ewm_params


VALID_COV_SHRINKAGE = {"none", "ledoit_wolf", "oas"}
VALID_COV_SHRINKAGE_TARGET = {"scaled_identity", "constant_correlation"}

_COV_SHRINKAGE_LABELS = {
    "none": "None",
    "ledoit_wolf": "Ledoit-Wolf",
    "oas": "OAS",
}

_COV_SHRINKAGE_TARGET_LABELS = {
    "scaled_identity": "Scaled Identity",
    "constant_correlation": "Constant Correlation",
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


def normalize_cov_shrinkage_target(value: str | None) -> str:
    """Return a supported covariance shrinkage target value."""
    normalized = str(value or "").strip().lower()
    if normalized in VALID_COV_SHRINKAGE_TARGET:
        return normalized
    return "scaled_identity"


def format_cov_shrinkage_target_label(value: str | None) -> str:
    """Return display label for a covariance shrinkage target option."""
    return _COV_SHRINKAGE_TARGET_LABELS.get(
        normalize_cov_shrinkage_target(value),
        "Scaled Identity",
    )


def resolve_cov_shrinkage_spec(
    shrinkage: str | None,
    shrinkage_target: str | None,
    *,
    exp_weighted: bool = False,
) -> tuple[str, str]:
    """Resolve the effective covariance shrinkage method and target."""
    if exp_weighted:
        return "none", "scaled_identity"

    normalized_shrinkage = normalize_cov_shrinkage(shrinkage)
    if normalized_shrinkage == "none":
        return "none", "scaled_identity"
    if normalized_shrinkage == "oas":
        return "oas", "scaled_identity"
    return normalized_shrinkage, normalize_cov_shrinkage_target(shrinkage_target)


def format_cov_shrinkage_spec_label(
    shrinkage: str | None,
    shrinkage_target: str | None,
    *,
    exp_weighted: bool = False,
) -> str:
    """Return display label for the effective covariance estimation spec."""
    effective_shrinkage, effective_target = resolve_cov_shrinkage_spec(
        shrinkage,
        shrinkage_target,
        exp_weighted=exp_weighted,
    )
    if exp_weighted:
        return "Exp Weighted"
    if effective_shrinkage == "none":
        return "None"
    if effective_shrinkage == "oas":
        return "OAS"
    return (
        f"{format_cov_shrinkage_label(effective_shrinkage)}, "
        f"{format_cov_shrinkage_target_label(effective_target)}"
    )


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


def _constant_correlation_target(sample_cov: np.ndarray) -> tuple[np.ndarray, float]:
    """Build the constant-correlation Ledoit-Wolf target from sample covariance."""
    cov_values = np.asarray(sample_cov, dtype=float)
    n_assets = cov_values.shape[0]
    if cov_values.ndim != 2 or cov_values.shape[1] != n_assets:
        raise ValueError("Sample covariance must be a square matrix.")

    variances = np.clip(np.diag(cov_values), a_min=0.0, a_max=None)
    std = np.sqrt(variances)
    denom = np.outer(std, std)
    corr = np.divide(
        cov_values,
        denom,
        out=np.full_like(cov_values, np.nan, dtype=float),
        where=denom > 0,
    )
    np.fill_diagonal(corr, np.nan)

    valid_pairs = (denom > 0) & ~np.eye(n_assets, dtype=bool)
    if np.any(valid_pairs):
        r_bar = float(np.nanmean(corr[valid_pairs]))
    else:
        r_bar = 0.0

    target = r_bar * denom
    np.fill_diagonal(target, variances)
    target[~valid_pairs & ~np.eye(n_assets, dtype=bool)] = 0.0
    return target, r_bar


def _ledoit_wolf_constant_correlation_covariance(
    clean_df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Estimate Ledoit-Wolf shrinkage covariance toward a constant-correlation target."""
    data = clean_df.loc[:, columns].to_numpy(dtype=float, copy=False)
    n_obs, n_assets = data.shape

    centered = data - data.mean(axis=0, keepdims=True)
    sample_cov = centered.T @ centered / float(n_obs)
    target_cov, average_corr = _constant_correlation_target(sample_cov)

    squared = centered ** 2
    phi_mat = squared.T @ squared / float(n_obs) - sample_cov ** 2
    phi = float(np.sum(phi_mat))

    variances = np.clip(np.diag(sample_cov), a_min=0.0, a_max=None)
    std = np.sqrt(variances)
    valid_pairs = np.outer(std > 0, std > 0)
    off_diag_valid = valid_pairs & ~np.eye(n_assets, dtype=bool)

    term1 = (centered ** 3).T @ centered / float(n_obs)
    theta_mat = term1 - (variances[:, None] * sample_cov)
    theta_mat -= squared.T @ squared / float(n_obs) - (sample_cov * variances[None, :])
    theta_mat[np.eye(n_assets, dtype=bool)] = 0.0

    inv_std = np.zeros_like(std)
    inv_std[std > 0] = 1.0 / std[std > 0]
    ratio_mat = inv_std[:, None] * std[None, :]
    rho = float(np.sum(np.diag(phi_mat)))
    if np.any(off_diag_valid):
        rho += float(
            average_corr * np.sum((ratio_mat * theta_mat)[off_diag_valid])
        )

    gamma = float(np.linalg.norm(sample_cov - target_cov, ord="fro") ** 2)
    if gamma <= 1e-18:
        shrunk_cov = target_cov
    else:
        kappa = (phi - rho) / gamma
        delta = float(np.clip(kappa / float(n_obs), 0.0, 1.0))
        shrunk_cov = delta * target_cov + (1.0 - delta) * sample_cov

    return pd.DataFrame(shrunk_cov, index=columns, columns=columns)


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
    shrinkage_target: str = "scaled_identity",
    annualization_factor: float = 1.0,
) -> pd.DataFrame:
    """Estimate a covariance matrix using sample, EWM, or shrinkage estimators."""
    columns = _ordered_columns(returns_df, asset_order)
    if not columns:
        return pd.DataFrame()

    working_df = returns_df.loc[:, columns]
    effective_shrinkage, effective_target = resolve_cov_shrinkage_spec(
        shrinkage,
        shrinkage_target,
        exp_weighted=exp_weighted,
    )
    if exp_weighted:
        ewm_cov = working_df.ewm(
            **resolve_ewm_params(normalize_decay_input(decay_value, 63.0))
        ).cov().iloc[-len(columns):]
        if isinstance(ewm_cov.index, pd.MultiIndex):
            ewm_cov.index = ewm_cov.index.get_level_values(-1)
        return _finalize_covariance_frame(ewm_cov, columns, annualization_factor)

    if effective_shrinkage == "none":
        return _finalize_covariance_frame(
            working_df.cov(),
            columns,
            annualization_factor,
        )

    clean_df = working_df.dropna()
    if clean_df.shape[0] < 2 or clean_df.shape[1] < 2:
        raise ValueError("Insufficient overlapping observations for shrinkage covariance estimate.")

    if effective_shrinkage == "ledoit_wolf" and effective_target == "constant_correlation":
        cov_df = _ledoit_wolf_constant_correlation_covariance(clean_df, columns)
    else:
        estimator = LedoitWolf() if effective_shrinkage == "ledoit_wolf" else OAS()
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
