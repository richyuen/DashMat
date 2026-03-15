from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

from utils.charting import apply_chart_theme


@dataclass(frozen=True)
class QQSeries:
    x: np.ndarray
    y: np.ndarray
    slope: float | None
    intercept: float | None


def _clean_numeric_series(series: pd.Series | list | np.ndarray | None) -> pd.Series:
    return pd.Series(series, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()


def _zscore_array(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    std = float(arr.std(ddof=0))
    if np.isclose(std, 0.0):
        return np.zeros(len(arr), dtype=float)
    return (arr - float(arr.mean())) / std


def _fit_line(x_vals: np.ndarray, y_vals: np.ndarray) -> tuple[float | None, float | None]:
    if len(x_vals) < 2:
        return None, None
    if pd.Series(x_vals).nunique() <= 1:
        return None, None
    slope, intercept = np.polyfit(x_vals, y_vals, 1)
    return float(slope), float(intercept)


def build_normal_qq_series(series: pd.Series | list | np.ndarray | None) -> QQSeries | None:
    clean = _clean_numeric_series(series)
    if len(clean) < 3:
        return None
    (theoretical, observed), (slope, intercept, _corr) = stats.probplot(
        clean.to_numpy(),
        dist="norm",
        fit=True,
    )
    return QQSeries(
        x=np.asarray(theoretical, dtype=float),
        y=np.asarray(observed, dtype=float),
        slope=float(slope),
        intercept=float(intercept),
    )


def build_reference_qq_series(
    sample: pd.Series | list | np.ndarray | None,
    reference: pd.Series | list | np.ndarray | None,
    *,
    standardize: bool = True,
) -> QQSeries | None:
    sample_clean = _clean_numeric_series(sample)
    reference_clean = _clean_numeric_series(reference)
    if isinstance(sample, pd.Series) and isinstance(reference, pd.Series):
        reference_clean, sample_clean = reference_clean.align(sample_clean, join="inner")
    else:
        min_len = min(len(sample_clean), len(reference_clean))
        sample_clean = sample_clean.iloc[:min_len]
        reference_clean = reference_clean.iloc[:min_len]
    if len(sample_clean) < 3 or len(reference_clean) < 3:
        return None

    x_vals = reference_clean.to_numpy(dtype=float)
    y_vals = sample_clean.to_numpy(dtype=float)
    if standardize:
        x_vals = _zscore_array(x_vals)
        y_vals = _zscore_array(y_vals)

    x_sorted = np.sort(x_vals)
    y_sorted = np.sort(y_vals)
    slope, intercept = _fit_line(x_sorted, y_sorted)
    return QQSeries(x=x_sorted, y=y_sorted, slope=slope, intercept=intercept)


def build_qq_figure(
    qq_series: QQSeries,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    theme: str,
    height: int = 400,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=qq_series.x,
            y=qq_series.y,
            mode="markers",
            marker={"size": 5, "opacity": 0.7},
            name=title,
        )
    )
    if qq_series.slope is not None and qq_series.intercept is not None:
        x_line = np.linspace(float(np.min(qq_series.x)), float(np.max(qq_series.x)), 100)
        y_line = qq_series.slope * x_line + qq_series.intercept
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name="Reference Line",
                line={"width": 2},
            )
        )
    fig.update_layout(
        height=height,
        title=title,
        margin={"l": 50, "r": 20, "t": 50, "b": 50},
        xaxis_title=xlabel,
        yaxis_title=ylabel,
    )
    apply_chart_theme(fig, theme)
    return fig
