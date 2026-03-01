from __future__ import annotations

from io import BytesIO, StringIO
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from dash import no_update
from dash.exceptions import PreventUpdate

from utils.regression import RegressionWindowResult
from utils.returns import df_to_json
from utils.route_intent import ACTION_OPEN_IMPORT_MODAL, FLOW_DB, build_route_intent
from utils.shared_metrics import STATS_CONFIG


@pytest.fixture(scope="module")
def regression_page():
    import app  # noqa: F401
    import pages.regression as regression

    return regression


def _call_reg_run(regression_page, **overrides):
    params = {
        "n_clicks": 1,
        "raw_data": "raw-json",
        "periodicity": "daily",
        "x_series": ["X1"],
        "dep_var": "Y",
        "bench_assign": {},
        "ls_assign": {},
        "date_range": {"start": "2020-01-01", "end": "2020-06-30"},
        "vol_scaler": 0,
        "vol_scale_assign": {},
        "lag_assign": {},
        "min_beta_assign": {},
        "max_beta_assign": {},
        "enable_assign": {},
        "model": "ols",
        "reg_name": "TestRegression",
        "force_zero": False,
        "robust_se": False,
        "exp_wt": False,
        "halflife": 63,
        "window_type": "full",
        "window_size": 36,
        "opt_step": 1,
        "opt_step_unit": "months",
        "fill_in_sample": "off",
        "missing_data": "fill_na",
        "alpha": 1.0,
        "l1_ratio": 0.5,
        "arima_p": 0,
        "arima_d": 0,
        "arima_q": 0,
        "garch_p": 0,
        "garch_q": 0,
        "linear_constraints": None,
        "current_results": {},
    }
    params.update(overrides)
    return regression_page.reg_run_regression(**params)


def _find_component_by_id(node, target_id):
    if node is None:
        return None
    if getattr(node, "id", None) == target_id:
        return node

    children = getattr(node, "children", None)
    if isinstance(children, (list, tuple)):
        for child in children:
            found = _find_component_by_id(child, target_id)
            if found is not None:
                return found
    else:
        found = _find_component_by_id(children, target_id)
        if found is not None:
            return found

    props = getattr(node, "props", None)
    if isinstance(props, dict):
        for value in props.values():
            found = _find_component_by_id(value, target_id)
            if found is not None:
                return found
    return None


def _component_prop(node, prop_name):
    if hasattr(node, prop_name):
        return getattr(node, prop_name)
    to_plotly = getattr(node, "to_plotly_json", None)
    if callable(to_plotly):
        return ((to_plotly().get("props") or {})).get(prop_name)
    props = getattr(node, "props", None)
    if isinstance(props, dict):
        return props.get(prop_name)
    return None


def test_reg_default_col_def_hides_header_menu_button(regression_page):
    default_col_def = regression_page._reg_default_col_def()
    assert default_col_def == {
        "resizable": True,
        "sortable": True,
        "suppressHeaderMenuButton": True,
    }

    custom_col_def = regression_page._reg_default_col_def(sortable=False, extra={"cellStyle": {"textAlign": "center"}})
    assert custom_col_def["resizable"] is True
    assert custom_col_def["sortable"] is False
    assert custom_col_def["suppressHeaderMenuButton"] is True
    assert custom_col_def["cellStyle"] == {"textAlign": "center"}


def test_reg_run_regression_includes_run_level_arima_summary_and_per_var_bounds(monkeypatch, regression_page):
    idx = pd.date_range("2020-01-01", periods=6, freq="B")
    working_df = pd.DataFrame(
        {
            "Y": np.linspace(0.01, 0.06, len(idx)),
            "X1": np.linspace(0.0, 0.05, len(idx)),
            "X2": np.linspace(-0.02, 0.03, len(idx)),
        },
        index=idx,
    )

    monkeypatch.setattr(regression_page, "_reg_get_working_returns", lambda *_args, **_kwargs: working_df)

    wr = RegressionWindowResult(
        est_start=idx[0],
        est_end=idx[-1],
        apply_start=idx[0],
        apply_end=idx[-1],
        coefficients={"intercept": 0.1, "X1": 0.3, "X2": 0.2},
        p_values={"intercept": 0.2, "X1": 0.1, "X2": 0.1},
        diagnostics={"note": "ok"},
        n_obs=len(idx),
    )
    predicted = pd.Series(np.linspace(0.01, 0.06, len(idx)), index=idx, name="predicted")
    residuals = pd.Series(np.zeros(len(idx)), index=idx, name="residuals")

    captured = {}
    expected_summary = {"arima": {"order": (1, 0, 1), "aic": 1.0, "bic": 2.0, "params": {"ar.L1": 0.2}}}

    def _fake_run_regression(_y, _X, config):
        captured["config"] = config
        return [wr], predicted, residuals, expected_summary

    monkeypatch.setattr(regression_page, "run_regression", _fake_run_regression)

    new_results, _options, selected, status = _call_reg_run(
        regression_page,
        x_series=["X1", "X2"],
        model="constrained_ols",
        min_beta_assign={"X1": -0.1, "X2": 0.2},
        max_beta_assign={"X1": 0.4, "X2": 0.6},
        enable_assign={"X1": True, "X2": True},
        arima_p=1,
        arima_q=1,
    )

    cfg = captured["config"]
    assert cfg["min_beta_by_var"] == {"X1": -0.1, "X2": 0.2}
    assert cfg["max_beta_by_var"] == {"X1": 0.4, "X2": 0.6}
    assert cfg["min_beta"] == -0.1
    assert cfg["max_beta"] == 0.6

    entry = new_results[selected]
    assert entry["arima_garch_summary"] == expected_summary
    assert "arima_garch" not in (entry["window_results"][0].get("diagnostics") or {})
    assert "1 window(s)" in status


def test_reg_run_regression_persists_stats_inputs(monkeypatch, regression_page):
    idx = pd.date_range("2020-01-01", periods=6, freq="B")
    working_df = pd.DataFrame(
        {
            "Y": np.linspace(0.01, 0.06, len(idx)),
            "X1": np.linspace(0.0, 0.05, len(idx)),
        },
        index=idx,
    )
    monkeypatch.setattr(regression_page, "_reg_get_working_returns", lambda *_args, **_kwargs: working_df)

    wr = RegressionWindowResult(
        est_start=idx[0],
        est_end=idx[-1],
        apply_start=idx[0],
        apply_end=idx[-1],
        coefficients={"intercept": 0.1, "X1": 0.3},
        p_values={"intercept": 0.2, "X1": 0.1},
        diagnostics={"note": "ok"},
        n_obs=len(idx),
    )
    predicted = pd.Series(np.linspace(0.01, 0.06, len(idx)), index=idx, name="predicted")
    residuals = pd.Series(np.zeros(len(idx)), index=idx, name="residuals")
    monkeypatch.setattr(
        regression_page,
        "run_regression",
        lambda *_args, **_kwargs: ([wr], predicted, residuals, None),
    )

    date_range = {"start": "2020-01-01", "end": "2020-01-08"}
    new_results, _options, selected, _status = _call_reg_run(
        regression_page,
        bench_assign={"Y": "X1"},
        ls_assign={"Y": False},
        date_range=date_range,
        vol_scaler=12,
        vol_scale_assign={"Y": True, "X1": False},
    )

    entry = new_results[selected]
    assert entry["benchmark_assignments"] == {"Y": "X1"}
    assert entry["long_short_assignments"] == {"Y": False}
    assert entry["date_range"] == date_range
    assert entry["vol_scaler"] == 12
    assert entry["vol_scaling_assignments"] == {"Y": True, "X1": False}


def test_reg_run_regression_errors_when_dependent_variable_missing(regression_page):
    out = _call_reg_run(regression_page, dep_var=None)
    assert out[0] is no_update
    assert out[1] is no_update
    assert out[2] is no_update
    assert "dependent variable" in out[3].lower()


def test_reg_run_regression_errors_when_x_series_missing_for_model_that_requires_x(regression_page):
    out = _call_reg_run(regression_page, x_series=[], model="style_analysis")
    assert out[0] is no_update
    assert out[1] is no_update
    assert out[2] is no_update
    assert "independent variable" in out[3].lower()


def test_reg_run_regression_allows_ols_intercept_only_when_x_missing(monkeypatch, regression_page):
    idx = pd.date_range("2020-01-01", periods=6, freq="B")
    working_df = pd.DataFrame({"Y": np.linspace(0.01, 0.06, len(idx))}, index=idx)
    monkeypatch.setattr(regression_page, "_reg_get_working_returns", lambda *_args, **_kwargs: working_df)

    wr = RegressionWindowResult(
        est_start=idx[0],
        est_end=idx[-1],
        apply_start=idx[0],
        apply_end=idx[-1],
        coefficients={"intercept": 0.02},
        p_values={"intercept": 0.1},
        diagnostics={"note": "ok"},
        n_obs=len(idx),
    )
    predicted = pd.Series(np.linspace(0.01, 0.06, len(idx)), index=idx, name="predicted")
    residuals = pd.Series(np.zeros(len(idx)), index=idx, name="residuals")

    captured = {}

    def _fake_run_regression(_y, _X, config):
        captured["x_columns"] = list(_X.columns)
        captured["config"] = config
        return [wr], predicted, residuals, {"arima": {"order": (1, 0, 0)}}

    monkeypatch.setattr(regression_page, "run_regression", _fake_run_regression)

    out = _call_reg_run(
        regression_page,
        model="ols",
        x_series=[],
        force_zero=False,
        linear_constraints=[{"X1": 1.0, "Min": 0.0}],
    )

    new_results, _options, selected, status = out
    assert new_results is not no_update
    assert selected in new_results
    assert "window(s)" in status
    assert captured["x_columns"] == []
    assert captured["config"]["linear_constraints"] is None
    assert new_results[selected]["independent_vars"] == []


def test_reg_run_regression_rejects_intercept_only_when_force_zero_enabled(regression_page):
    out = _call_reg_run(regression_page, x_series=[], model="ols", force_zero=True)
    assert out[0] is no_update
    assert out[1] is no_update
    assert out[2] is no_update
    assert "force zero intercept" in out[3].lower()


def test_reg_run_regression_rejects_self_lag_without_positive_lag(monkeypatch, regression_page):
    idx = pd.date_range("2020-01-01", periods=6, freq="B")
    working_df = pd.DataFrame({"Y": np.linspace(0.01, 0.06, len(idx))}, index=idx)
    monkeypatch.setattr(regression_page, "_reg_get_working_returns", lambda *_args, **_kwargs: working_df)

    out = _call_reg_run(
        regression_page,
        model="ols",
        dep_var="Y",
        x_series=["Y"],
        lag_assign={"Y": 0},
    )

    assert out[0] is no_update
    assert out[1] is no_update
    assert out[2] is no_update
    assert "lag to at least 1" in out[3].lower()


def test_reg_run_regression_supports_self_lag_with_display_labels(monkeypatch, regression_page):
    idx = pd.date_range("2020-01-01", periods=6, freq="B")
    working_df = pd.DataFrame({"Y": np.linspace(0.01, 0.06, len(idx))}, index=idx)
    monkeypatch.setattr(regression_page, "_reg_get_working_returns", lambda *_args, **_kwargs: working_df)

    wr = RegressionWindowResult(
        est_start=idx[1],
        est_end=idx[-1],
        apply_start=idx[1],
        apply_end=idx[-1],
        coefficients={"intercept": 0.02, "Y": 0.5},
        p_values={"intercept": 0.1, "Y": 0.05},
        diagnostics={
            "std_errors": {"intercept": 0.01, "Y": 0.2},
            "t_stats": {"intercept": 2.0, "Y": 2.5},
            "ci_low": {"intercept": 0.0, "Y": 0.1},
            "ci_high": {"intercept": 0.03, "Y": 0.9},
            "vif": {"Y": 1.0},
        },
        n_obs=5,
    )
    predicted = pd.Series(np.linspace(0.02, 0.06, len(idx) - 1), index=idx[1:], name="predicted")
    residuals = pd.Series(np.zeros(len(idx) - 1), index=idx[1:], name="residuals")

    captured = {}

    def _fake_run_regression(_y, _X, config):
        captured["x_columns"] = list(_X.columns)
        captured["config"] = config
        return [wr], predicted, residuals, None

    monkeypatch.setattr(regression_page, "run_regression", _fake_run_regression)

    new_results, _options, selected, status = _call_reg_run(
        regression_page,
        model="ols",
        dep_var="Y",
        x_series=["Y"],
        lag_assign={"Y": 1},
    )

    assert "window(s)" in status
    assert captured["x_columns"] == ["Y"]
    assert captured["config"]["lag_config"] == {"Y": 1}
    assert captured["config"]["lag_config_display"] == {"Y (lag 1)": 1}

    entry = new_results[selected]
    assert entry["independent_vars"] == ["Y (lag 1)"]
    assert entry["independent_vars_internal"] == ["Y"]
    assert entry["effective_date_range"] == {"start": "2020-01-02", "end": "2020-01-08"}

    wr_saved = entry["window_results"][0]
    assert "Y (lag 1)" in wr_saved["coefficients"]
    assert "Y (lag 1)" in wr_saved["p_values"]
    assert wr_saved["diagnostics"]["std_errors"]["Y (lag 1)"] == pytest.approx(0.2)


def test_reg_run_regression_errors_when_raw_data_missing(regression_page):
    out = _call_reg_run(regression_page, raw_data=None)
    assert out[0] is no_update
    assert out[1] is no_update
    assert out[2] is no_update
    assert "no data loaded" in out[3].lower()


def test_reg_run_regression_handles_blank_linear_constraints(monkeypatch, regression_page):
    idx = pd.date_range("2020-01-01", periods=80, freq="B")
    x1 = np.linspace(-0.05, 0.05, len(idx))
    x2 = np.linspace(0.03, -0.02, len(idx))
    y = 0.01 + 0.6 * x1 + 0.2 * x2
    working_df = pd.DataFrame({"Y": y, "X1": x1, "X2": x2}, index=idx)
    monkeypatch.setattr(regression_page, "_reg_get_working_returns", lambda *_args, **_kwargs: working_df)

    out = _call_reg_run(
        regression_page,
        model="constrained_ols",
        x_series=["X1", "X2"],
        enable_assign={"X1": True},
        linear_constraints=[{"X1": "", "X2": "", "Min": "", "Max": " "}],
    )

    assert out[0] is not no_update
    assert out[1] is not no_update
    assert out[2] is not no_update
    assert "regression error" not in out[3].lower()


def test_reg_open_db_add_modal_uses_helper(monkeypatch, regression_page):
    expected = (True, [{"value": "IDX_A", "label": "Index A"}], [])
    monkeypatch.setattr(regression_page, "compute_open_db_add_modal", lambda *_args, **_kwargs: expected)
    assert regression_page.reg_open_db_add_modal(1) == (*expected, False, regression_page.no_update)


def test_reg_resolve_import_modal_request_returns_db_request(regression_page):
    route_intent = build_route_intent("regression", ACTION_OPEN_IMPORT_MODAL, flow=FLOW_DB)

    request = regression_page.reg_resolve_import_modal_request(
        "regression",
        route_intent,
        None,
    )

    assert request == (
        {"flow": FLOW_DB, "token": route_intent["token"]},
        no_update,
        no_update,
        no_update,
    )


def test_reg_open_db_add_modal_uses_request_store_token(monkeypatch, regression_page):
    expected = (True, [{"value": "IDX_A", "label": "Index A"}], [])
    monkeypatch.setattr(regression_page, "compute_open_db_add_modal", lambda *_args, **_kwargs: expected)
    monkeypatch.setattr(
        regression_page,
        "callback_context",
        SimpleNamespace(triggered_id="reg-db-add-request-store"),
    )

    out = regression_page.reg_open_db_add_modal(None, {"flow": FLOW_DB, "token": "tok"})

    assert out == (*expected, False, "tok")


def test_reg_resolve_import_modal_request_ignores_stale_intent(regression_page):
    route_intent = build_route_intent("regression", ACTION_OPEN_IMPORT_MODAL, flow=FLOW_DB)
    route_intent["created_at"] = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(seconds=61)).isoformat()

    with pytest.raises(PreventUpdate):
        regression_page.reg_resolve_import_modal_request(
            "regression",
            route_intent,
            None,
        )


def test_reg_add_series_from_database_imports_and_updates_stores(monkeypatch, regression_page):
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    new_df = pd.DataFrame({"IDX_A": [0.01, 0.0, 0.002, -0.003, 0.004]}, index=idx)
    monkeypatch.setattr(
        regression_page,
        "load_cma_returns_for_benches_with_meta",
        lambda *_args, **_kwargs: (new_df, {"IDX_A": {"starts_daily": True}}),
    )

    raw, orig_p, p_value, p_sync, opened, selected, err_text, err_hide = (
        regression_page.reg_add_series_from_database(1, ["IDX_A"], None, None)
    )

    assert isinstance(raw, str)
    assert orig_p == "daily"
    assert p_value == "daily"
    assert p_sync == "daily"
    assert opened is False
    assert selected == []
    assert err_hide is True
    assert err_text is no_update


def test_reg_add_series_from_database_rejects_duplicates(monkeypatch, regression_page):
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    existing_df = pd.DataFrame({"IDX_A": [0.01, 0.0, -0.01]}, index=idx)
    existing_raw = df_to_json(existing_df)
    monkeypatch.setattr(
        regression_page,
        "load_cma_returns_for_benches_with_meta",
        lambda *_args, **_kwargs: (existing_df.copy(), {"IDX_A": {"starts_daily": True}}),
    )

    raw, orig_p, p_value, p_sync, opened, selected, err_text, err_hide = (
        regression_page.reg_add_series_from_database(1, ["IDX_A"], existing_raw, "daily")
    )

    assert raw is no_update
    assert orig_p is no_update
    assert p_value is no_update
    assert p_sync is no_update
    assert opened is True
    assert selected is no_update
    assert "duplicate" in str(err_text).lower()
    assert err_hide is False


def test_reg_toggle_welcome_uses_original_periodicity(monkeypatch, regression_page):
    captured = {}

    def _fake_get_available_periodicities(original_periodicity):
        captured["arg"] = original_periodicity
        return [
            {"value": "daily", "label": "Daily"},
            {"value": "monthly", "label": "Monthly"},
        ]

    monkeypatch.setattr(regression_page, "get_available_periodicities", _fake_get_available_periodicities)
    main_style, options, value, base_ready = regression_page._reg_toggle_main_visibility(
        {"original_periodicity": "daily"}, 1, "daily", "monthly"
    )

    assert captured["arg"] == "daily"
    assert main_style["display"] == "flex"
    assert options == [{"value": "daily", "label": "Daily"}, {"value": "monthly", "label": "Monthly"}]
    assert value == "monthly"
    assert base_ready is True


def test_reg_on_modal_ok_commits_local_series_modal_state(regression_page):
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    raw_json = df_to_json(pd.DataFrame({"A": [0.01, 0.0, -0.01], "B": [0.0, 0.01, 0.02]}, index=idx))

    out = regression_page.reg_on_modal_ok(
        1,
        raw_json,
        [
            {
                "__orig_series": "A",
                "Series": "A",
                "Y": True,
                "X": True,
                "Benchmark": "B",
                "LongShort": True,
                "ScaleVol": False,
                "Lag": 2,
                "MinBeta": -0.5,
                "MaxBeta": 0.8,
                "Enable": True,
                "Delete": False,
            },
            {
                "__orig_series": "B",
                "Series": "B",
                "Y": False,
                "X": True,
                "Benchmark": "None",
                "LongShort": False,
                "ScaleVol": True,
                "Lag": 0,
                "MinBeta": -999,
                "MaxBeta": 999,
                "Enable": False,
                "Delete": True,
            },
        ],
        [{"__orig_series": "B"}, {"__orig_series": "A"}],
    )

    assert out[0] == ["A"]
    assert out[1] == {"A": "None"}
    assert out[2] == {"A": True}
    assert out[3] == ["A"]
    assert out[10] == "A"
    assert out[11] == {"A": 2}
    assert out[12] == {"A": -0.5}
    assert out[13] == {"A": 0.8}
    assert out[14] == {"A": True}
    assert out[15] is False

    updated_df = pd.read_json(StringIO(out[8]), orient="split")
    assert list(updated_df.columns) == ["A"]


def test_reg_begin_series_selection_request_opens_modal_and_releases_blocker(regression_page):
    assert regression_page.reg_begin_series_selection_request("token") == (True, False)


def test_reg_resolve_series_selection_modal_controls_overlay_and_ok(regression_page):
    assert regression_page.reg_resolve_series_selection_modal("token", None) == (True, True, "", "blue", True)
    assert regression_page.reg_resolve_series_selection_modal(
        "token", {"token": "token", "status": "ready", "message": ""}
    ) == (False, False, "", "blue", True)
    assert regression_page.reg_resolve_series_selection_modal(
        "token", {"token": "token", "status": "timeout", "message": "slow"}
    ) == (False, True, "slow", "red", False)


def test_reg_series_grid_uses_stable_checkbox_interaction_options(regression_page):
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    raw = df_to_json(pd.DataFrame({"A": [0.01, 0.0, -0.01], "B": [0.0, 0.01, 0.02]}, index=idx))

    children, status = regression_page.reg_update_series_grid(
        "token",
        raw,
        ["A"],
        ["A", "B"],
        {},
        {},
        {},
        "A",
        {},
        {},
        {},
        {},
    )

    grid = children[0]
    assert status["status"] == "rendered"
    opts = getattr(grid, "dashGridOptions", {}) or {}
    assert opts.get("suppressMovableColumns") is True
    assert opts.get("stopEditingWhenCellsLoseFocus") is True
    assert opts.get("singleClickEdit") is True
    assert opts.get("tooltipShowDelay") == 500

    cols = getattr(grid, "columnDefs", []) or []
    x_col = next((c for c in cols if c.get("field") == "X"), None)
    y_col = next((c for c in cols if c.get("field") == "Y"), None)
    series_col = next((c for c in cols if c.get("field") == "Series"), None)
    scale_col = next((c for c in cols if c.get("field") == "ScaleVol"), None)
    benchmark_col = next((c for c in cols if c.get("field") == "Benchmark"), None)
    assert x_col is not None
    assert y_col is not None
    assert series_col is not None
    assert scale_col is not None
    assert benchmark_col is not None
    assert getattr(grid, "getRowId", None) == "params.data.__orig_series"
    assert x_col.get("cellRenderer") == "agCheckboxCellRenderer"
    assert y_col.get("valueSetter")
    assert x_col.get("headerTooltip")
    assert series_col.get("headerTooltip")
    assert series_col.get("editable") is False
    assert benchmark_col.get("cellEditor") == "agRichSelectCellEditor"
    assert benchmark_col.get("cellEditorPopup") is True
    assert benchmark_col.get("cellEditorParams", {}).get("allowTyping") is True
    assert getattr(grid, "enableEnterpriseModules", False) is True
    assert scale_col.get("cellRenderer") == "agCheckboxCellRenderer"
    assert scale_col.get("headerTooltip")


def _collect_component_text(node):
    if node is None:
        return []
    if isinstance(node, str):
        return [node]
    if isinstance(node, (int, float, bool)):
        return [str(node)]
    if isinstance(node, (list, tuple, set)):
        out = []
        for item in node:
            out.extend(_collect_component_text(item))
        return out
    if isinstance(node, dict):
        out = []
        for value in node.values():
            out.extend(_collect_component_text(value))
        return out

    out = []
    children = getattr(node, "children", None)
    out.extend(_collect_component_text(children))
    props = getattr(node, "props", None)
    if isinstance(props, dict):
        for value in props.values():
            out.extend(_collect_component_text(value))
    return out


def _collect_ag_grids(node):
    if node is None:
        return []
    if isinstance(node, (str, int, float, bool, dict)):
        return []
    if isinstance(node, (list, tuple, set)):
        out = []
        for item in node:
            out.extend(_collect_ag_grids(item))
        return out

    out = []
    if getattr(node, "columnDefs", None) is not None and getattr(node, "rowData", None) is not None:
        out.append(node)
    children = getattr(node, "children", None)
    out.extend(_collect_ag_grids(children))
    props = getattr(node, "props", None)
    if isinstance(props, dict):
        for value in props.values():
            out.extend(_collect_ag_grids(value))
    return out


def _collect_components_by_class(node, class_name: str):
    if node is None:
        return []
    if isinstance(node, (str, int, float, bool, dict)):
        return []
    if isinstance(node, (list, tuple, set)):
        out = []
        for item in node:
            out.extend(_collect_components_by_class(item, class_name))
        return out

    out = []
    if node.__class__.__name__ == class_name:
        out.append(node)
    children = getattr(node, "children", None)
    out.extend(_collect_components_by_class(children, class_name))
    props = getattr(node, "props", None)
    if isinstance(props, dict):
        for value in props.values():
            out.extend(_collect_components_by_class(value, class_name))
    return out


def test_reg_toggle_welcome_no_data_hides_embedded_welcome(regression_page):
    welcome_style, main_style, options, value, base_ready = regression_page.reg_toggle_welcome(None, 1, None, None)

    assert welcome_style == {"display": "none"}
    assert main_style["display"] == "none"
    assert options == [{"value": "daily", "label": "Daily"}]
    assert value == "daily"
    assert base_ready is True


def test_regression_layout_includes_page_ready_stores_and_visible_overlay(regression_page):
    base_store = _find_component_by_id(regression_page.layout, "reg-base-controls-ready-store")
    page_store = _find_component_by_id(regression_page.layout, "reg-page-ready-store")
    overlay = _find_component_by_id(regression_page.layout, "reg-ui-blocker-overlay")

    assert base_store is not None
    assert page_store is not None
    assert _component_prop(base_store, "data") is False
    assert _component_prop(page_store, "data") is False
    assert _component_prop(overlay, "visible") is True


def test_reg_toggle_welcome_keeps_base_ready_false_before_page_load(regression_page):
    welcome_style, main_style, options, value, base_ready = regression_page.reg_toggle_welcome(None, 0, None, None)

    assert welcome_style == {"display": "none"}
    assert main_style["display"] == "none"
    assert options == [{"value": "daily", "label": "Daily"}]
    assert value == "daily"
    assert base_ready is False


def test_reg_init_date_range_sets_page_ready(monkeypatch, regression_page, raw_json):
    monkeypatch.setattr(
        regression_page,
        "get_periodicity_range_metadata",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        regression_page,
        "compute_date_range_candidates_from_metadata",
        lambda *_args, **_kwargs: {"available_series": ["Y", "X1"]},
    )
    monkeypatch.setattr(
        regression_page,
        "resolve_initial_range",
        lambda *_args, **_kwargs: ("2024-01-01", "2024-12-31"),
    )

    result = regression_page.reg_init_date_range(
        {"raw_data_hash": "hash"},
        "daily",
        ["X1"],
        "Y",
        True,
        raw_json,
        {"start": "2024-01-01", "end": "2024-12-31"},
        False,
    )

    assert result[-2] == {"start": "2024-01-01", "end": "2024-12-31"}
    assert result[-1] is True


def test_reg_init_date_range_leaves_page_ready_unchanged_without_series(regression_page):
    result = regression_page.reg_init_date_range(None, "daily", [], None, True, None, None, False)

    assert result[-2] is None
    assert result[-1] is no_update


def test_reg_release_page_ready_on_series_modal_uses_matching_final_status(regression_page):
    assert (
        regression_page.reg_release_page_ready_on_series_modal(
            "token",
            {"token": "token", "status": "ready", "message": ""},
            False,
        )
        is True
    )


def test_reg_release_page_ready_on_series_modal_ignores_mismatched_or_nonfinal_status(regression_page):
    with pytest.raises(PreventUpdate):
        regression_page.reg_release_page_ready_on_series_modal(
            "token",
            {"token": "other", "status": "ready", "message": ""},
            False,
        )

    with pytest.raises(PreventUpdate):
        regression_page.reg_release_page_ready_on_series_modal(
            "token",
            {"token": "token", "status": "rendered", "message": ""},
            False,
        )


def test_reg_open_modal_opens_on_regression_activation_with_summary(monkeypatch, regression_page):
    monkeypatch.setattr(
        regression_page,
        "callback_context",
        SimpleNamespace(triggered_id="wb-active-module-store"),
    )

    out = regression_page.reg_open_modal(
        None,
        {"columns": ["Y", "X1", "X2"]},
        "regression",
        [],
        [],
        None,
        None,
        None,
        None,
        None,
    )

    assert isinstance(out[0], str) and out[0]
    assert out[1:] == ("", "blue", True, True, None)


def test_reg_overlay_visible_uses_base_and_page_ready(regression_page):
    assert regression_page._reg_overlay_visible(False, None, False, False) is True
    assert regression_page._reg_overlay_visible(False, None, True, False) is False
    assert regression_page._reg_overlay_visible(False, "raw", True, False) is True
    assert regression_page._reg_overlay_visible(False, "raw", True, True) is False
    assert regression_page._reg_overlay_visible(True, "raw", True, True) is True


def test_reg_help_modal_covers_three_sections_and_model_explainers(regression_page):
    modal = regression_page.build_reg_help_modal()
    text_blob = " ".join(_collect_component_text(modal)).lower()

    required_phrases = [
        "basic guide",
        "advanced guide",
        "model deep dive",
        "what it is: baseline linear regression with unconstrained coefficients",
        "what it is: ols with per-variable beta limits and optional linear constraints",
        "what it is: constrained style decomposition where exposures are bounded and sum to one",
        "what it is: l2-regularized regression that shrinks coefficients toward zero",
        "what it is: l1-regularized regression that can zero out coefficients",
        "what it is: combined l1 and l2 regularization",
        "series selection modal",
        "periodicity",
        "vol scaler",
        "date range",
        "common range",
        "max range",
        "fill in-sample",
        "linear constraints",
        "run regression",
        "anova, rolling summary, rolling, weights, statistics, returns, growth of $1, calendar year, drawdown, and scatter",
        "save session",
        "load session",
        "download excel",
        "clear server cache",
        "arima(p,d,q)",
        "garch(p,q)",
        "arima and garch residual overlay",
    ]
    for phrase in required_phrases:
        assert phrase in text_blob


def test_reg_render_statistics_uses_current_stats_signature_and_list_shape(monkeypatch, regression_page):
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    predicted = pd.DataFrame({"Predicted": [0.01, -0.005, 0.003, 0.002]}, index=idx)
    entry = {"periodicity": "daily", "predicted_json": df_to_json(predicted)}
    captured = {}

    def _fake_stats(*args, **kwargs):
        captured["args"] = args
        return [
            {
                "Series": "Predicted",
                "Cumulative Return": 0.0100,
                "Annualized Return": 0.0300,
            }
        ]

    monkeypatch.setattr(regression_page, "calculate_statistics_cached", _fake_stats)
    comp = regression_page.reg_render_statistics("R1", {"R1": entry}, mounted_tabs=["statistics"])

    assert captured["args"][5] == "null"
    assert captured["args"][6] == 0
    assert getattr(comp, "rowData", None)
    assert getattr(comp, "defaultColDef", {}).get("suppressHeaderMenuButton") is True
    assert any(row.get("Statistic") == "Cumulative Return" for row in comp.rowData)


def test_reg_render_statistics_uses_full_stats_config_rows(monkeypatch, regression_page):
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    predicted = pd.DataFrame({"Predicted": [0.01, -0.005, 0.003, 0.002]}, index=idx)
    entry = {"periodicity": "daily", "predicted_json": df_to_json(predicted)}

    monkeypatch.setattr(
        regression_page,
        "calculate_statistics_cached",
        lambda *_args, **_kwargs: [{"Series": "Predicted", "Start Date": "2024-01-01", "End Date": "2024-01-04"}],
    )
    comp = regression_page.reg_render_statistics("R1", {"R1": entry}, mounted_tabs=["statistics"])

    stat_names = [row.get("Statistic") for row in getattr(comp, "rowData", [])]
    expected = [name for name, _fmt in STATS_CONFIG]
    assert stat_names[: len(expected)] == expected


def test_reg_render_statistics_includes_actual_predicted_residual_when_available(monkeypatch, regression_page):
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    predicted = pd.DataFrame({"predicted": [0.01, -0.005, 0.003, 0.002, -0.001]}, index=idx)
    residuals = pd.DataFrame({"residuals": [0.002, -0.001, 0.000, 0.001, -0.002]}, index=idx)
    entry = {
        "periodicity": "daily",
        "predicted_json": df_to_json(predicted),
        "residuals_json": df_to_json(residuals),
    }

    def _fake_stats(_json_str, _periodicity, selected_series, *_args, **_kwargs):
        assert tuple(selected_series) == ("Predicted", "Actual (Y)", "Residual")
        return [
            {"Series": "Actual (Y)", "Start Date": "2024-01-01", "End Date": "2024-01-05"},
            {"Series": "Predicted", "Start Date": "2024-01-01", "End Date": "2024-01-05"},
            {"Series": "Residual", "Start Date": "2024-01-01", "End Date": "2024-01-05"},
        ]

    monkeypatch.setattr(regression_page, "calculate_statistics_cached", _fake_stats)
    comp = regression_page.reg_render_statistics("R1", {"R1": entry}, mounted_tabs=["statistics"])

    col_fields = [c.get("field") for c in getattr(comp, "columnDefs", [])]
    assert "Actual (Y)" in col_fields
    assert "Predicted" in col_fields
    assert "Residual" in col_fields


def test_reg_render_statistics_combines_run_series_and_model_output_stats(monkeypatch, regression_page):
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    raw_df = pd.DataFrame(
        {
            "SPX_TRIndex": [0.01, -0.005, 0.003, 0.002, -0.001],
            "EM_TRIndex": [0.008, -0.004, 0.002, 0.001, -0.002],
            "EAFE_TRIndex": [0.007, -0.003, 0.001, 0.0005, -0.0015],
        },
        index=idx,
    )
    entry = {
        "periodicity": "daily",
        "dependent_var": "SPX_TRIndex",
        "independent_vars": ["EM_TRIndex", "EAFE_TRIndex"],
        "benchmark_assignments": {},
        "long_short_assignments": {},
        "date_range": {"start": "2024-01-01", "end": "2024-01-05"},
        "vol_scaler": 0,
        "vol_scaling_assignments": {},
        "predicted_json": df_to_json(pd.DataFrame({"predicted": raw_df["SPX_TRIndex"]}, index=idx)),
        "residuals_json": df_to_json(pd.DataFrame({"residuals": np.zeros(len(idx))}, index=idx)),
    }
    calls = []

    def _fake_stats(*args, **kwargs):
        calls.append(args)
        selected = tuple(args[2])
        if selected == ("Predicted", "Actual (Y)", "EM_TRIndex", "EAFE_TRIndex", "Residual"):
            return [
                {"Series": "Actual (Y)", "Start Date": "2024-01-01", "End Date": "2024-01-05", "Cumulative Return": 0.009},
                {"Series": "Predicted", "Start Date": "2024-01-01", "End Date": "2024-01-05", "Cumulative Return": 0.007},
                {"Series": "EM_TRIndex", "Start Date": "2024-01-01", "End Date": "2024-01-05", "Cumulative Return": 0.005},
                {"Series": "EAFE_TRIndex", "Start Date": "2024-01-01", "End Date": "2024-01-05", "Cumulative Return": 0.004},
                {"Series": "Residual", "Start Date": "2024-01-01", "End Date": "2024-01-05", "Cumulative Return": 0.002},
            ]
        return []

    monkeypatch.setattr(regression_page, "calculate_statistics_cached", _fake_stats)
    comp = regression_page.reg_render_statistics("R1", {"R1": entry}, df_to_json(raw_df), {}, None, ["statistics"])

    selected_payloads = [tuple(call[2]) for call in calls]
    assert ("Predicted", "Actual (Y)", "EM_TRIndex", "EAFE_TRIndex", "Residual") in selected_payloads
    run_call = calls[0]
    assert run_call[5] == "null"

    col_fields = [c.get("field") for c in getattr(comp, "columnDefs", [])]
    assert col_fields[:6] == ["Statistic", "Predicted", "Actual (Y)", "EM_TRIndex", "EAFE_TRIndex", "Residual"]
    assert "SPX_TRIndex" not in col_fields


def test_reg_build_display_series_clips_x_to_model_window_for_rolling(regression_page):
    full_idx = pd.date_range("2024-01-01", periods=8, freq="B")
    model_idx = full_idx[3:]
    raw_df = pd.DataFrame(
        {
            "Y": [0.01, 0.02, -0.01, 0.00, 0.01, -0.02, 0.03, 0.01],
            "X1": [0.02, -0.01, 0.00, 0.01, 0.00, 0.02, -0.01, 0.03],
            "X2": [0.01, 0.00, -0.02, 0.02, 0.01, -0.01, 0.00, 0.01],
        },
        index=full_idx,
    )
    predicted = pd.DataFrame({"predicted": [0.001, 0.002, 0.003, 0.004, 0.005]}, index=model_idx)
    residuals = pd.DataFrame({"residuals": [0.0, -0.001, 0.0, 0.001, -0.001]}, index=model_idx)
    entry = {
        "periodicity": "daily",
        "dependent_var": "Y",
        "independent_vars": ["X1", "X2"],
        "benchmark_assignments": {},
        "long_short_assignments": {},
        "date_range": {"start": "2024-01-01", "end": "2024-01-31"},
        "vol_scaler": 0,
        "vol_scaling_assignments": {},
        "config": {"window_type": "rolling"},
        "predicted_json": df_to_json(predicted),
        "residuals_json": df_to_json(residuals),
    }

    display_df, ordered_cols = regression_page._reg_build_display_series(entry, df_to_json(raw_df))

    assert ordered_cols == ["Predicted", "Actual (Y)", "X1", "X2", "Residual"]
    assert list(display_df.index) == list(model_idx)
    assert list(display_df["X1"].index) == list(model_idx)


def test_reg_build_display_series_clips_to_effective_window_for_full_lagged_self_x(regression_page):
    full_idx = pd.date_range("2024-01-01", periods=6, freq="B")
    model_idx = full_idx[1:]
    raw_df = pd.DataFrame(
        {"Y": [0.01, 0.02, -0.01, 0.00, 0.01, -0.02]},
        index=full_idx,
    )
    predicted = pd.DataFrame({"predicted": [0.011, 0.018, -0.005, 0.004, -0.018]}, index=model_idx)
    residuals = pd.DataFrame({"residuals": [0.001, 0.002, -0.001, 0.003, -0.002]}, index=model_idx)
    entry = {
        "periodicity": "daily",
        "dependent_var": "Y",
        "independent_vars": ["Y (lag 1)"],
        "independent_vars_internal": ["Y"],
        "benchmark_assignments": {},
        "long_short_assignments": {},
        "date_range": {"start": "2024-01-01", "end": "2024-01-31"},
        "vol_scaler": 0,
        "vol_scaling_assignments": {},
        "config": {
            "window_type": "full",
            "feature_label_map": {"Y": "Y (lag 1)"},
            "lag_config": {"Y": 1},
        },
        "predicted_json": df_to_json(predicted),
        "residuals_json": df_to_json(residuals),
    }

    display_df, ordered_cols = regression_page._reg_build_display_series(entry, df_to_json(raw_df))

    assert ordered_cols == ["Predicted", "Actual (Y)", "Y (lag 1)", "Residual"]
    assert list(display_df.index) == list(model_idx)
    assert display_df.iloc[0]["Y (lag 1)"] == pytest.approx(raw_df.iloc[0]["Y"])


def test_reg_sync_name_with_model_uses_model_defaults(regression_page):
    assert regression_page.reg_sync_name_with_model("ols") == "OLS"
    assert regression_page.reg_sync_name_with_model("ridge") == "Ridge"
    assert regression_page.reg_sync_name_with_model("style_analysis") == "Style Analysis"
    assert regression_page.reg_sync_name_with_model("unknown_model") == "Regression"


def test_reg_download_excel_matches_tab_order_and_settings_sheet(monkeypatch, regression_page):
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    predicted = pd.DataFrame({"Predicted": [0.01, -0.002, 0.003, 0.004, -0.001]}, index=idx)
    residuals = pd.DataFrame({"Residual": [0.001, -0.001, 0.0, 0.001, -0.002]}, index=idx)

    window_result = {
        "est_start": idx[0],
        "est_end": idx[-1],
        "apply_start": idx[0],
        "apply_end": idx[-1],
        "r_squared": 0.82,
        "adj_r_squared": 0.76,
        "residual_std": 0.014,
        "n_obs": len(idx),
        "coefficients": {"intercept": 0.001, "X1": 0.95},
        "p_values": {"intercept": 0.21, "X1": 0.01},
        "anova_table": {
            "df_model": 1,
            "df_resid": 3,
            "ss_model": 0.45,
            "ms_model": 0.45,
            "F_stat": 9.0,
            "F_pvalue": 0.05,
            "ss_resid": 0.15,
            "ms_resid": 0.05,
            "ss_total": 0.60,
        },
        "diagnostics": {
            "std_errors": {"intercept": 0.05, "X1": 0.12},
            "t_stats": {"intercept": 2.0, "X1": 7.9},
            "ci_low": {"intercept": -0.09, "X1": 0.70},
            "ci_high": {"intercept": 0.11, "X1": 1.20},
            "durbin_watson": 2.10,
            "aic": 12.3,
            "bic": 14.2,
            "vif": {"X1": 1.1},
        },
        "oos_metrics": {"oos_r2": 0.61, "oos_rmse": 0.02, "oos_mae": 0.01},
        "arima_garch": {
            "arima": {"order": [1, 0, 1], "aic": 11.1, "bic": 12.2, "params": {"const": 0.02, "ar.L1": 0.33}},
            "garch": {"order": [1, 1], "aic": 13.3, "bic": 14.4, "params": {"mu": 0.01, "omega": 0.2}},
        },
    }

    results = {
        "R1": {
            "periodicity": "daily",
            "dependent_var": "Y",
            "independent_vars": ["X1"],
            "config": {
                "model": "ols",
                "window_type": "rolling",
                "window_size": 24,
                "opt_step": 1,
                "opt_step_unit": "months",
                "fill_in_sample": True,
                "missing_data": "fill_na",
                "force_zero_intercept": False,
                "robust_se": True,
                "exp_wt": False,
                "halflife": 63,
                "alpha": 1.0,
                "l1_ratio": 0.5,
            },
            "window_results": [window_result],
            "date_range": {"start": "2024-01-01", "end": "2024-01-31"},
            "vol_scaler": 0,
            "benchmark_assignments": {},
            "long_short_assignments": {},
            "vol_scaling_assignments": {},
            "predicted_json": df_to_json(predicted),
            "residuals_json": df_to_json(residuals),
        }
    }

    monkeypatch.setattr(
        regression_page,
        "calculate_statistics_cached",
        lambda *_args, **_kwargs: [
            {"Series": "Predicted", "Cumulative Return": 0.015, "Annualized Return": 0.20},
            {"Series": "Actual (Y)", "Cumulative Return": 0.014, "Annualized Return": 0.18},
            {"Series": "Residual", "Cumulative Return": -0.001, "Annualized Return": -0.01},
        ],
    )
    monkeypatch.setattr(
        regression_page,
        "calculate_rolling_returns",
        lambda *_args, **_kwargs: pd.DataFrame(
            {
                "Predicted": [0.10],
                "Actual (Y)": [0.08],
                "Residual": [0.02],
            },
            index=[pd.Timestamp("2024-01-31")],
        ),
    )
    monkeypatch.setattr(
        regression_page,
        "create_monthly_view",
        lambda *_args, **_kwargs: ([], [{"Year_Label": "2024", "Jan": 0.01, "YTD": 0.01}]),
    )
    monkeypatch.setattr(
        regression_page,
        "calculate_drawdown",
        lambda *_args, **_kwargs: pd.DataFrame(
            {
                "Predicted": [0.0, -0.02],
                "Actual (Y)": [0.0, -0.03],
                "Residual": [0.0, -0.01],
            },
            index=[pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-31")],
        ),
    )
    monkeypatch.setattr(regression_page.dcc, "send_bytes", lambda b, filename: {"content": b, "filename": filename})

    payload = regression_page.reg_download_excel(
        1,
        results,
        None,
        "R1",
        0,
        "1y",
        "annualized",
        "total_return",
        "monthly",
        "Predicted",
    )

    workbook = BytesIO(payload["content"])
    xl = pd.ExcelFile(workbook)
    assert xl.sheet_names == [
        "Settings",
        "ANOVA",
        "Rolling Summary",
        "Weights",
        "Statistics",
        "Returns",
        "Rolling",
        "Calendar Year",
        "Growth of $1",
        "Drawdown",
    ]

    settings_df = pd.read_excel(BytesIO(payload["content"]), sheet_name="Settings")
    settings_map = dict(zip(settings_df["Parameter"], settings_df["Value"]))
    assert settings_map["Result Name"] == "R1"
    assert settings_map["Model"] == "ols"
    assert settings_map["Effective Sample Start"] == "2024-01-01"
    assert settings_map["Effective Sample End"] == "2024-01-05"

    weights_df = pd.read_excel(BytesIO(payload["content"]), sheet_name="Weights")
    assert list(weights_df.columns) == ["Window", "Date", "intercept", "X1"]
    assert "ARIMA_AIC" not in set(weights_df.columns)
    assert "GARCH_AIC" not in set(weights_df.columns)
    assert weights_df.loc[0, "Date"] == "2024-01-01"
    assert weights_df.loc[0, "intercept"] == pytest.approx(0.001)
    assert weights_df.loc[0, "X1"] == pytest.approx(0.95)

    anova_df = pd.read_excel(BytesIO(payload["content"]), sheet_name="ANOVA")
    assert "Block" in anova_df.columns
    assert "Parameters" in set(anova_df["Block"].dropna())
    assert "Overall Fit" in set(anova_df["Block"].dropna())
    assert "ARIMA.const" in set(anova_df.get("Parameter", pd.Series(dtype=str)).dropna())
    assert "GARCH.mu" in set(anova_df.get("Parameter", pd.Series(dtype=str)).dropna())


def test_reg_sync_anova_window_options_defaults_to_latest_on_result_change(monkeypatch, regression_page):
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    results = {
        "R1": {
            "window_results": [
                {"apply_start": idx[0], "apply_end": idx[0]},
                {"apply_start": idx[1], "apply_end": idx[1]},
                {"apply_start": idx[2], "apply_end": idx[2]},
            ]
        }
    }
    monkeypatch.setattr(regression_page, "callback_context", type("Ctx", (), {"triggered_id": "reg-result-select"})())

    options, value, disabled = regression_page.reg_sync_anova_window_options("R1", results, None, "0", ["anova"])

    assert len(options) == 3
    assert value == "2"
    assert disabled is False


def test_reg_sync_anova_window_options_defaults_to_latest_on_results_refresh(monkeypatch, regression_page):
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    results = {
        "R1": {
            "window_results": [
                {"apply_start": idx[0], "apply_end": idx[0]},
                {"apply_start": idx[1], "apply_end": idx[1]},
                {"apply_start": idx[2], "apply_end": idx[2]},
            ]
        }
    }
    monkeypatch.setattr(regression_page, "callback_context", type("Ctx", (), {"triggered_id": "reg-results-store"})())

    _options, value, _disabled = regression_page.reg_sync_anova_window_options("R1", results, None, "1", ["anova"])

    assert value == "2"


def test_reg_render_rolling_returns_table_uses_wide_date_column(monkeypatch, regression_page):
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    predicted = pd.DataFrame({"predicted": [0.01, 0.0, 0.002, -0.001]}, index=idx)
    residuals = pd.DataFrame({"residuals": [0.0, 0.0, 0.0, 0.0]}, index=idx)
    results = {
        "R1": {
            "periodicity": "daily",
            "predicted_json": df_to_json(predicted),
            "residuals_json": df_to_json(residuals),
        }
    }

    monkeypatch.setattr(
        regression_page,
        "calculate_rolling_returns",
        lambda *_args, **_kwargs: pd.DataFrame(
            {"Predicted": [0.05], "Actual (Y)": [0.04], "Residual": [0.01]},
            index=[pd.Timestamp("2024-01-31")],
        ),
    )
    grid = regression_page.reg_render_rolling_returns(
        "R1",
        results,
        None,
        "1y",
        "annualized",
        "total_return",
        "table",
        "light",
        None,
        ["rolling_returns"],
    )

    assert getattr(grid, "columnDefs", [])[0]["field"] == "Date"
    assert getattr(grid, "columnDefs", [])[0]["width"] == 112
    assert getattr(grid, "defaultColDef", {}).get("suppressHeaderMenuButton") is True


def test_reg_render_drawdown_table_uses_wide_date_column(monkeypatch, regression_page):
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    predicted = pd.DataFrame({"predicted": [0.01, 0.0, 0.002, -0.001]}, index=idx)
    residuals = pd.DataFrame({"residuals": [0.0, 0.0, 0.0, 0.0]}, index=idx)
    results = {
        "R1": {
            "periodicity": "daily",
            "predicted_json": df_to_json(predicted),
            "residuals_json": df_to_json(residuals),
        }
    }

    monkeypatch.setattr(
        regression_page,
        "calculate_drawdown",
        lambda *_args, **_kwargs: pd.DataFrame(
            {"Predicted": [0.0, -0.02]},
            index=[pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-31")],
        ),
    )
    grid = regression_page.reg_render_drawdown("R1", results, None, "table", "light", None, ["drawdown"])

    assert getattr(grid, "columnDefs", [])[0]["field"] == "Date"
    assert getattr(grid, "columnDefs", [])[0]["width"] == 112
    assert getattr(grid, "defaultColDef", {}).get("suppressHeaderMenuButton") is True


def test_reg_render_growth_table_mode_returns_grid_with_wide_date_column(regression_page):
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    predicted = pd.DataFrame({"predicted": [0.01, 0.0, 0.002, -0.001]}, index=idx)
    residuals = pd.DataFrame({"residuals": [0.0, 0.0, 0.0, 0.0]}, index=idx)
    results = {
        "R1": {
            "periodicity": "daily",
            "predicted_json": df_to_json(predicted),
            "residuals_json": df_to_json(residuals),
        }
    }

    grid = regression_page.reg_render_growth("R1", results, None, "table", "light", None, ["growth"])

    assert getattr(grid, "columnDefs", [])[0]["field"] == "Date"
    assert getattr(grid, "columnDefs", [])[0]["width"] == 112
    assert getattr(grid, "defaultColDef", {}).get("suppressHeaderMenuButton") is True
    assert len(getattr(grid, "rowData", [])) == 4


def test_reg_render_weights_table_mode_returns_grid_with_wide_date_column(regression_page):
    results = {
        "R1": {
            "config": {"model": "style_analysis"},
            "window_results": [
                {
                    "apply_start": "2024-01-01",
                    "coefficients": {"Asset_A": 0.6, "Asset_B": 0.4},
                },
                {
                    "apply_start": "2024-02-01",
                    "coefficients": {"Asset_A": 0.5, "Asset_B": 0.5},
                },
            ],
        }
    }

    stack = regression_page.reg_render_weights("R1", results, "table", "light", None, ["weights"])
    children = list(getattr(stack, "children", []) or [])
    grid = children[0]

    date_col = next(c for c in getattr(grid, "columnDefs", []) if c.get("field") == "Date")
    assert date_col["width"] == 112
    assert getattr(grid, "defaultColDef", {}).get("suppressHeaderMenuButton") is True


def test_reg_render_anova_uses_three_block_layout_with_arima_garch_params(regression_page):
    results = {
        "R1": {
            "dependent_var": "Y",
            "config": {"model": "ols"},
            "window_results": [
                {
                    "est_start": "2024-01-01",
                    "est_end": "2024-01-31",
                    "apply_start": "2024-01-01",
                    "apply_end": "2024-01-31",
                    "coefficients": {"intercept": 0.01, "X1": 0.5},
                    "p_values": {"intercept": 0.1, "X1": 0.02},
                    "diagnostics": {
                        "durbin_watson": 1.9,
                        "jarque_bera_stat": 2.1,
                        "jarque_bera_pvalue": 0.3,
                    },
                    "r_squared": 0.8,
                    "adj_r_squared": 0.79,
                    "n_obs": 21,
                    "residual_std": 0.02,
                    "anova_table": {
                        "df_model": 1,
                        "df_resid": 19,
                        "ss_model": 0.45,
                        "ms_model": 0.45,
                        "F_stat": 9.0,
                        "F_pvalue": 0.01,
                        "ss_resid": 0.15,
                        "ms_resid": 0.0079,
                        "ss_total": 0.60,
                    },
                    "arima_garch": {
                        "arima": {"order": [1, 0, 1], "aic": 1.2, "bic": 1.4, "params": {"const": 0.01, "ar.L1": 0.22}},
                        "garch": {"order": [1, 1], "aic": 2.3, "bic": 2.5, "params": {"mu": 0.02, "omega": 0.1}},
                    },
                }
            ],
        }
    }

    comp = regression_page.reg_render_anova("R1", results, "0", None, ["anova"])
    grids = _collect_ag_grids(comp)
    assert len(grids) >= 2

    anova_grid = next(
        (g for g in grids if {"Source", "df", "SS", "MS", "F", "p-value"}.issubset({c.get("field") for c in (getattr(g, "columnDefs", []) or [])})),
        None,
    )
    assert anova_grid is not None
    assert getattr(anova_grid, "id", None) == "reg-anova-decomposition-grid"
    assert (getattr(anova_grid, "style", {}) or {}).get("height") == "132px"
    assert getattr(anova_grid, "defaultColDef", {}).get("suppressHeaderMenuButton") is True
    anova_sources = [row.get("Source") for row in (getattr(anova_grid, "rowData", []) or [])]
    assert set(anova_sources) == {"Model", "Residual", "Total"}
    assert all(c.get("headerTooltip") for c in (getattr(anova_grid, "columnDefs", []) or []))

    param_grid = next(
        (g for g in grids if {"Parameter", "Coefficient"}.issubset({c.get("field") for c in (getattr(g, "columnDefs", []) or [])})),
        None,
    )
    assert param_grid is not None
    assert getattr(param_grid, "id", None) == "reg-anova-parameter-grid"
    assert getattr(param_grid, "defaultColDef", {}).get("suppressHeaderMenuButton") is True
    assert all(c.get("headerTooltip") for c in (getattr(param_grid, "columnDefs", []) or []))
    param_names = [row.get("Parameter") for row in (getattr(param_grid, "rowData", []) or [])]
    assert "intercept" in param_names
    assert "X1" in param_names
    assert "ARIMA.const" in param_names
    assert "ARIMA.ar.L1" in param_names
    assert "GARCH.mu" in param_names
    assert "GARCH.omega" in param_names

    text_blob = " ".join(_collect_component_text(comp))
    assert "Overall Fit" in text_blob
    assert "Regression Fit" in text_blob
    assert "R-Squared" in text_blob
    assert "ARIMA Fit" in text_blob
    assert "AIC" in text_blob
    assert "GARCH Fit" in text_blob

    tooltip_labels = [getattr(t, "label", "") for t in _collect_components_by_class(comp, "Tooltip")]
    assert any("fraction of dependent-series variance explained" in str(label).lower() for label in tooltip_labels)


def test_reg_render_rolling_table_merges_arima_garch_columns(regression_page):
    results = {
        "R1": {
            "window_results": [
                {
                    "apply_start": "2024-01-01",
                    "r_squared": 0.5,
                    "adj_r_squared": 0.4,
                    "residual_std": 0.02,
                    "n_obs": 20,
                    "coefficients": {"X1": 0.3},
                    "arima_garch": {"arima": {"order": [1, 0, 0], "aic": 1.0, "bic": 1.1, "params": {"ar.L1": 0.2}}},
                },
                {
                    "apply_start": "2024-02-01",
                    "r_squared": 0.6,
                    "adj_r_squared": 0.5,
                    "residual_std": 0.01,
                    "n_obs": 20,
                    "coefficients": {"X1": 0.4},
                    "arima_garch": {"arima": {"order": [1, 0, 0], "aic": 0.9, "bic": 1.0, "params": {"ar.L1": 0.25}}},
                },
            ]
        }
    }

    grid = regression_page.reg_render_rolling("R1", results, "table", "advanced", "light", None, ["rolling"])
    fields = [c.get("field") for c in getattr(grid, "columnDefs", [])]
    assert getattr(grid, "defaultColDef", {}).get("suppressHeaderMenuButton") is True
    assert "ARIMA_AIC" in fields
    assert "ARIMA_ar_L1" in fields

    rows = getattr(grid, "rowData", []) or []
    assert rows[0].get("ARIMA_ar_L1") == 0.2
    assert rows[1].get("ARIMA_ar_L1") == 0.25

    basic_grid = regression_page.reg_render_rolling("R1", results, "table", "basic", "light", None, ["rolling"])
    basic_fields = [c.get("field") for c in getattr(basic_grid, "columnDefs", [])]
    assert basic_fields.index("β_X1") < basic_fields.index("ARIMA_AIC")
    assert "ARIMA_AIC" in basic_fields
    assert "ARIMA_ar_L1" not in basic_fields


def test_reg_render_rolling_chart_respects_basic_advanced_field_scope(regression_page):
    results = {
        "R1": {
            "window_results": [
                {
                    "apply_start": "2024-01-01",
                    "r_squared": 0.5,
                    "adj_r_squared": 0.4,
                    "residual_std": 0.02,
                    "n_obs": 20,
                    "coefficients": {"intercept": 0.1, "X1": 0.3},
                    "arima_garch": {"arima": {"order": [1, 0, 0], "aic": 1.0, "bic": 1.1, "params": {"ar.L1": 0.2}}},
                },
                {
                    "apply_start": "2024-02-01",
                    "r_squared": 0.6,
                    "adj_r_squared": 0.5,
                    "residual_std": 0.01,
                    "n_obs": 20,
                    "coefficients": {"intercept": 0.1, "X1": 0.4},
                    "arima_garch": {"arima": {"order": [1, 0, 0], "aic": 0.9, "bic": 1.0, "params": {"ar.L1": 0.25}}},
                },
            ]
        }
    }

    basic_chart = regression_page.reg_render_rolling("R1", results, "chart", "basic", "light", None, ["rolling"])
    basic_names = [trace.name for trace in getattr(getattr(basic_chart, "figure", None), "data", [])]
    assert "β_intercept" in basic_names
    assert "β_X1" in basic_names
    assert "ARIMA_AIC" in basic_names
    assert "ARIMA_ar_L1" not in basic_names

    advanced_chart = regression_page.reg_render_rolling("R1", results, "chart", "advanced", "light", None, ["rolling"])
    advanced_names = [trace.name for trace in getattr(getattr(advanced_chart, "figure", None), "data", [])]
    assert "ARIMA_ar_L1" in advanced_names


def test_reg_render_weights_table_only_shows_prediction_coefficients(regression_page):
    results = {
        "R1": {
            "config": {"model": "ols"},
            "window_results": [
                {
                    "apply_start": "2024-01-01",
                    "coefficients": {"X1": 0.3},
                    "arima_garch": {"garch": {"order": [1, 1], "aic": 2.0, "bic": 2.1, "params": {"omega": 0.12}}},
                },
                {
                    "apply_start": "2024-02-01",
                    "coefficients": {"X1": 0.4},
                    "arima_garch": {"garch": {"order": [1, 1], "aic": 1.9, "bic": 2.0, "params": {"omega": 0.10}}},
                },
            ]
        }
    }

    comp = regression_page.reg_render_weights("R1", results, "table", "light", None, ["weights"])
    children = list(getattr(comp, "children", []) or [])
    grid = next(c for c in children if getattr(c, "columnDefs", None) is not None)
    fields = [c.get("field") for c in getattr(grid, "columnDefs", [])]
    assert fields == ["Window", "Date", "X1"]
    row_data = getattr(grid, "rowData", []) or []
    assert row_data[0].get("X1") == 0.3
    assert row_data[1].get("X1") == 0.4
