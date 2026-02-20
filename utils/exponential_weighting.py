"""Shared helpers for exponential weighting input interpretation."""

from __future__ import annotations

import math


DEFAULT_DECAY_VALUE = 63.0


def _coerce_finite_float(value):
    """Return finite float value when possible, else None."""
    try:
        fval = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(fval):
        return None
    return fval


def normalize_decay_input(value, default_value: float = DEFAULT_DECAY_VALUE) -> float:
    """Normalize decay input to a positive finite float with default fallback."""
    fallback = _coerce_finite_float(default_value)
    if fallback is None or fallback <= 0:
        fallback = DEFAULT_DECAY_VALUE

    decay = _coerce_finite_float(value)
    if decay is None or decay <= 0:
        return float(fallback)
    return float(decay)


def decay_input_mode(value, default_value: float = DEFAULT_DECAY_VALUE) -> str:
    """Return interpretation mode for decay input."""
    decay = normalize_decay_input(value, default_value=default_value)
    if decay < 1:
        return "lambda"
    return "halflife_periods"


def resolve_ewm_params(value, default_value: float = DEFAULT_DECAY_VALUE) -> dict[str, float]:
    """Resolve pandas EWM keyword args from dual-mode decay input.

    Rules:
    - 0 < value < 1: interpret as lambda and convert to alpha = (1 - lambda)
    - value >= 1: interpret as half-life in periods
    """
    decay = normalize_decay_input(value, default_value=default_value)
    if decay < 1:
        alpha = 1.0 - decay
        if alpha <= 0:
            alpha = math.nextafter(0.0, 1.0)
        elif alpha > 1:
            alpha = 1.0
        return {"alpha": alpha}
    return {"halflife": decay}

