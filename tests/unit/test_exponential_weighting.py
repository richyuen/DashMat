from __future__ import annotations

import math

from utils.exponential_weighting import (
    decay_input_mode,
    normalize_decay_input,
    resolve_ewm_params,
)


def test_resolve_ewm_params_uses_halflife_for_period_values():
    params = resolve_ewm_params(63)
    assert params == {"halflife": 63.0}


def test_resolve_ewm_params_uses_alpha_for_lambda_values():
    params = resolve_ewm_params(0.94)
    assert "alpha" in params
    assert math.isclose(params["alpha"], 0.06, rel_tol=1e-12)


def test_decay_mode_treats_one_as_period_halflife():
    assert decay_input_mode(1) == "halflife_periods"


def test_normalize_decay_input_falls_back_to_default_on_invalid_values():
    assert normalize_decay_input(0, 63.0) == 63.0
    assert normalize_decay_input(-1, 63.0) == 63.0
    assert normalize_decay_input("bad", 63.0) == 63.0
