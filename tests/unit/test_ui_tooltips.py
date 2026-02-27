from __future__ import annotations

import re

import dash_mantine_components as dmc

from utils.ui_tooltips import (
    apply_header_tooltips,
    apply_tooltips_to_layout,
    grid_tooltip_dash_options,
    has_custom_tooltip,
    tooltip_text_and_source,
)


def _subtree_has_id(node, target_id: str) -> bool:
    if node is None:
        return False
    if isinstance(node, (list, tuple)):
        return any(_subtree_has_id(item, target_id) for item in node)
    if getattr(node, "id", None) == target_id:
        return True

    children = getattr(node, "children", None)
    if _subtree_has_id(children, target_id):
        return True

    for prop in getattr(node, "_prop_names", []):
        if prop in {"children", "id"}:
            continue
        if _subtree_has_id(getattr(node, prop, None), target_id):
            return True
    return False


def _tree_has_tooltip_for_id(node, target_id: str) -> bool:
    if node is None:
        return False

    if isinstance(node, (list, tuple)):
        return any(_tree_has_tooltip_for_id(item, target_id) for item in node)

    class_name = node.__class__.__name__
    if class_name == "Tooltip":
        children = getattr(node, "children", None)
        child_id = getattr(children, "id", None)
        if child_id == target_id or _subtree_has_id(children, target_id):
            return True

    children = getattr(node, "children", None)
    if _tree_has_tooltip_for_id(children, target_id):
        return True

    for prop in getattr(node, "_prop_names", []):
        if prop in {"children", "id"}:
            continue
        value = getattr(node, prop, None)
        if _tree_has_tooltip_for_id(value, target_id):
            return True
    return False


def _parent_of_tooltip_target(node, target_id: str, parent=None):
    if node is None:
        return None

    if isinstance(node, (list, tuple)):
        for item in node:
            found = _parent_of_tooltip_target(item, target_id, parent=parent)
            if found is not None:
                return found
        return None

    if getattr(node, "id", None) == target_id:
        return parent

    class_name = node.__class__.__name__
    if class_name == "Tooltip":
        children = getattr(node, "children", None)
        child_id = getattr(children, "id", None)
        if child_id == target_id:
            return parent

    children = getattr(node, "children", None)
    found_children = _parent_of_tooltip_target(children, target_id, parent=node)
    if found_children is not None:
        return found_children

    for prop in getattr(node, "_prop_names", []):
        if prop in {"children", "id"}:
            continue
        value = getattr(node, prop, None)
        found_prop = _parent_of_tooltip_target(value, target_id, parent=node)
        if found_prop is not None:
            return found_prop

    return None


def _tooltip_for_target(node, target_id: str):
    if node is None:
        return None

    if isinstance(node, (list, tuple)):
        for item in node:
            found = _tooltip_for_target(item, target_id)
            if found is not None:
                return found
        return None

    if node.__class__.__name__ == "Tooltip":
        children = getattr(node, "children", None)
        child_id = getattr(children, "id", None)
        if child_id == target_id or _subtree_has_id(children, target_id):
            return node

    children = getattr(node, "children", None)
    found_children = _tooltip_for_target(children, target_id)
    if found_children is not None:
        return found_children

    for prop in getattr(node, "_prop_names", []):
        if prop in {"children", "id"}:
            continue
        found_prop = _tooltip_for_target(getattr(node, prop, None), target_id)
        if found_prop is not None:
            return found_prop
    return None


def _wrapper_for_target(node, target_id: str):
    if node is None:
        return None

    if isinstance(node, (list, tuple)):
        for item in node:
            found = _wrapper_for_target(item, target_id)
            if found is not None:
                return found
        return None

    children = getattr(node, "children", None)
    if getattr(children, "id", None) == target_id:
        return node

    found_children = _wrapper_for_target(children, target_id)
    if found_children is not None:
        return found_children

    for prop in getattr(node, "_prop_names", []):
        if prop in {"children", "id"}:
            continue
        found_prop = _wrapper_for_target(getattr(node, prop, None), target_id)
        if found_prop is not None:
            return found_prop
    return None


def _count_tooltips(node) -> int:
    if node is None:
        return 0
    if isinstance(node, (list, tuple)):
        return sum(_count_tooltips(item) for item in node)
    count = 1 if node.__class__.__name__ == "Tooltip" else 0
    children = getattr(node, "children", None)
    count += _count_tooltips(children)
    for prop in getattr(node, "_prop_names", []):
        if prop in {"children", "id"}:
            continue
        count += _count_tooltips(getattr(node, prop, None))
    return count


def _collect_tooltip_entries(node):
    if node is None:
        return []
    if isinstance(node, (list, tuple)):
        out = []
        for item in node:
            out.extend(_collect_tooltip_entries(item))
        return out

    out = []
    if node.__class__.__name__ == "Tooltip":
        children = getattr(node, "children", None)
        child_id = getattr(children, "id", None)
        tooltip_id = getattr(node, "id", None)
        target_id = child_id if isinstance(child_id, str) and child_id else tooltip_id
        if isinstance(target_id, str) and target_id:
            out.append((target_id, str(getattr(node, "label", "") or "")))

    children = getattr(node, "children", None)
    out.extend(_collect_tooltip_entries(children))
    for prop in getattr(node, "_prop_names", []):
        if prop in {"children", "id"}:
            continue
        out.extend(_collect_tooltip_entries(getattr(node, prop, None)))
    return out


def _sentence_count(text: str) -> int:
    return len([p for p in re.split(r"[.!?]+", str(text or "").strip()) if p.strip()])


def _rendered_page_tooltip_entries() -> dict[str, str]:
    import app  # noqa: F401
    import pages.analyticstool as analyticstool
    import pages.portopt as portopt
    import pages.regression as regression

    entries = []
    for layout in (analyticstool.layout, portopt.layout, regression.layout):
        entries.extend(_collect_tooltip_entries(layout))

    out: dict[str, str] = {}
    for control_id, label in entries:
        if isinstance(control_id, str) and control_id.startswith(("at-", "po-", "reg-")):
            out[control_id] = str(label or "")
    return out


def test_tooltip_text_source_prefers_explicit_and_patterns():
    explicit_text, explicit_source = tooltip_text_and_source("at-menu-download-excel")
    assert explicit_source == "explicit"
    assert "excel" in explicit_text.lower()

    generated_text, generated_source = tooltip_text_and_source("at-periodicity-select")
    assert generated_source == "explicit"
    assert "frequency" in generated_text.lower()

    pattern_text, pattern_source = tooltip_text_and_source("periodicity-select")
    assert pattern_source == "pattern"
    assert "periodicity" in pattern_text.lower()


def test_tooltip_text_source_fallback_is_non_empty():
    text, source = tooltip_text_and_source("custom-unknown-control")
    assert source == "fallback"
    assert isinstance(text, str)
    assert text.strip()


def test_has_custom_tooltip_detects_non_fallback_ids():
    assert has_custom_tooltip("at-periodicity-select") is True
    assert has_custom_tooltip("custom-unknown-control") is False


def test_apply_tooltips_wraps_interactive_id(monkeypatch):
    monkeypatch.setenv("DASHMAT_ENABLE_GLOBAL_TOOLTIPS", "1")
    layout = dmc.Group(
        children=[
            dmc.Select(
                id="at-periodicity-select",
                label="Periodicity",
                data=[{"value": "daily", "label": "Daily"}],
                value="daily",
            )
        ]
    )
    decorated = apply_tooltips_to_layout(layout, page_key="analyticstool")
    assert _tree_has_tooltip_for_id(decorated, "at-periodicity-select") is True


def test_apply_tooltips_does_not_double_wrap_existing_tooltips(monkeypatch):
    monkeypatch.setenv("DASHMAT_ENABLE_GLOBAL_TOOLTIPS", "1")
    layout = dmc.Group(
        children=[
            dmc.Tooltip(
                label="Existing tooltip",
                children=dmc.NumberInput(id="at-vol-scaler-input", value=0, min=0),
            )
        ]
    )
    decorated = apply_tooltips_to_layout(layout, page_key="analyticstool")
    assert _tree_has_tooltip_for_id(decorated, "at-vol-scaler-input") is True
    assert _count_tooltips(decorated) == 1


def test_apply_tooltips_respects_feature_flag(monkeypatch):
    monkeypatch.setenv("DASHMAT_ENABLE_GLOBAL_TOOLTIPS", "0")
    layout = dmc.Group(
        children=[
            dmc.Switch(id="reg-exp-wt-switch", checked=False),
        ]
    )
    decorated = apply_tooltips_to_layout(layout, page_key="regression")
    assert _tree_has_tooltip_for_id(decorated, "reg-exp-wt-switch") is False


def test_apply_tooltips_suppresses_generic_ok_cancel_actions(monkeypatch):
    monkeypatch.setenv("DASHMAT_ENABLE_GLOBAL_TOOLTIPS", "1")
    layout = dmc.Group(
        children=[
            dmc.Button("OK", id="at-modal-ok-button"),
            dmc.Button("Cancel", id="at-modal-cancel-button"),
            dmc.Button("Import OK", id="at-db-add-ok-button"),
            dmc.Button("Import Cancel", id="at-db-add-cancel-button"),
            dmc.Select(
                id="at-periodicity-select",
                data=[{"value": "daily_trading", "label": "Daily (Trading)"}],
                value="daily_trading",
            ),
        ]
    )
    decorated = apply_tooltips_to_layout(layout, page_key="analyticstool")

    assert _tree_has_tooltip_for_id(decorated, "at-modal-ok-button") is False
    assert _tree_has_tooltip_for_id(decorated, "at-modal-cancel-button") is False
    assert _tree_has_tooltip_for_id(decorated, "at-db-add-ok-button") is False
    assert _tree_has_tooltip_for_id(decorated, "at-db-add-cancel-button") is False
    assert _tree_has_tooltip_for_id(decorated, "at-periodicity-select") is True


def test_apply_tooltips_suppresses_table_chart_switch_controls(monkeypatch):
    monkeypatch.setenv("DASHMAT_ENABLE_GLOBAL_TOOLTIPS", "1")
    layout = dmc.Group(
        children=[
            dmc.SegmentedControl(
                id="at-growth-chart-switch",
                data=[
                    {"value": "table", "label": "Table"},
                    {"value": "chart", "label": "Chart"},
                ],
                value="table",
            ),
            dmc.SegmentedControl(
                id="po-frontier-chart-switch",
                data=[
                    {"value": "table", "label": "Table"},
                    {"value": "chart", "label": "Chart"},
                ],
                value="chart",
            ),
            dmc.Select(
                id="at-periodicity-select",
                data=[{"value": "daily_trading", "label": "Daily (Trading)"}],
                value="daily_trading",
            ),
        ]
    )

    decorated = apply_tooltips_to_layout(layout, page_key="analyticstool")

    assert _tree_has_tooltip_for_id(decorated, "at-growth-chart-switch") is False
    assert _tree_has_tooltip_for_id(decorated, "po-frontier-chart-switch") is False
    assert _tree_has_tooltip_for_id(decorated, "at-periodicity-select") is True


def test_apply_tooltips_preserves_full_width_button_layout(monkeypatch):
    monkeypatch.setenv("DASHMAT_ENABLE_GLOBAL_TOOLTIPS", "1")
    layout = dmc.Group(
        children=[
            dmc.Button(
                "AA Tool indices",
                id="at-welcome-add-db-btn",
                fullWidth=True,
            )
        ]
    )
    decorated = apply_tooltips_to_layout(layout, page_key="analyticstool")
    parent = _parent_of_tooltip_target(decorated, "at-welcome-add-db-btn")
    wrapper = _wrapper_for_target(decorated, "at-welcome-add-db-btn")

    assert parent is not None
    assert parent.__class__.__name__ == "Div"
    assert wrapper is not None
    assert getattr(wrapper, "className", "") == "dashmat-tooltip-trigger-width"
    wrapper_style = getattr(wrapper, "style", None) or {}
    assert wrapper_style.get("display") == "block"

    tooltip_node = _tooltip_for_target(decorated, "at-welcome-add-db-btn")
    assert tooltip_node is not None
    tooltip_style = getattr(tooltip_node, "style", None) or {}
    assert tooltip_style.get("width") is None
    assert tooltip_style.get("display") is None
    assert getattr(tooltip_node, "className", "") in {"", None}
    wrapper_props = getattr(tooltip_node, "boxWrapperProps", None) or {}
    assert wrapper_props == {}


def test_apply_tooltips_preserves_full_width_multiselect_layout(monkeypatch):
    monkeypatch.setenv("DASHMAT_ENABLE_GLOBAL_TOOLTIPS", "1")
    layout = dmc.Group(
        children=[
            dmc.MultiSelect(
                id="at-db-add-series-select",
                data=[],
                value=[],
                w="100%",
            )
        ]
    )
    decorated = apply_tooltips_to_layout(layout, page_key="analyticstool")
    parent = _parent_of_tooltip_target(decorated, "at-db-add-series-select")
    wrapper = _wrapper_for_target(decorated, "at-db-add-series-select")

    assert parent is not None
    assert parent.__class__.__name__ == "Div"
    assert wrapper is not None
    assert getattr(wrapper, "className", "") == "dashmat-tooltip-trigger-width"
    wrapper_style = getattr(wrapper, "style", None) or {}
    assert wrapper_style.get("display") == "block"

    tooltip_node = _tooltip_for_target(decorated, "at-db-add-series-select")
    assert tooltip_node is not None
    tooltip_style = getattr(tooltip_node, "style", None) or {}
    assert tooltip_style.get("width") is None
    assert tooltip_style.get("display") is None
    assert getattr(tooltip_node, "className", "") in {"", None}
    wrapper_props = getattr(tooltip_node, "boxWrapperProps", None) or {}
    assert wrapper_props == {}


def test_apply_tooltips_does_not_force_width_wrapper_for_compact_controls(monkeypatch):
    monkeypatch.setenv("DASHMAT_ENABLE_GLOBAL_TOOLTIPS", "1")
    layout = dmc.Group(
        children=[
            dmc.Switch(id="reg-exp-wt-switch", checked=False),
        ]
    )
    decorated = apply_tooltips_to_layout(layout, page_key="regression")
    parent = _parent_of_tooltip_target(decorated, "reg-exp-wt-switch")

    assert parent is not None
    assert parent.__class__.__name__ != "Box"


def test_apply_header_tooltips_maps_known_grid_columns():
    cols = [
        {"field": "Series"},
        {"field": "Benchmark"},
        {"headerName": "", "width": 20},
    ]

    out = apply_header_tooltips(cols, "at-series-selection-grid")

    assert out[0].get("headerTooltip")
    assert "series" in out[0]["headerTooltip"].lower()
    assert out[1].get("headerTooltip")
    assert "benchmark" in out[1]["headerTooltip"].lower()
    assert "headerTooltip" not in out[2]


def test_grid_tooltip_dash_options_merges_defaults():
    out = grid_tooltip_dash_options({"singleClickEdit": True, "tooltipShowDelay": 250})

    assert out["singleClickEdit"] is True
    assert out["tooltipShowDelay"] == 250
    assert out["tooltipHideDelay"] == 5000
    assert out["tooltipMouseTrack"] is False
    assert out["tooltipInteraction"] is True


def test_vol_scaler_tooltip_copy_is_detailed():
    text, source = tooltip_text_and_source("po-vol-scaler-input")

    assert source == "explicit"
    assert "annualized volatility target" in text.lower()
    assert "set to 0 to disable" in text.lower()


def test_generated_tooltip_for_robust_se_is_detailed():
    text, source = tooltip_text_and_source("reg-robust-se-switch")

    assert source == "explicit"
    assert "robust standard errors" in text.lower()
    assert _sentence_count(text) >= 2
    assert len(text) >= 140


def test_model_dropdown_tooltips_reference_help_menu():
    for control_id in ("po-opt-model-select", "reg-model-select"):
        text, source = tooltip_text_and_source(control_id)
        text_lower = text.lower()
        assert source == "explicit"
        assert "help menu" in text_lower


def test_opt_step_and_exp_wt_cov_tooltips_are_not_generic():
    expected = {
        "po-exp-wt-cov-switch": ("covariance", "half-life", "recent observations"),
        "po-opt-step-input": ("window advances", "months", "periods"),
    }
    banned = ("toggles exp wt cov behavior", "sets the opt step value")

    for control_id, required_tokens in expected.items():
        text, source = tooltip_text_and_source(control_id)
        text_lower = text.lower()
        assert source == "explicit"
        for token in required_tokens:
            assert token in text_lower, (control_id, token, text)
        for phrase in banned:
            assert phrase not in text_lower, (control_id, phrase, text)


def test_arima_and_garch_order_tooltips_explain_each_parameter_individually():
    expected = {
        "reg-arima-p-input": ("autoregressive order", "lagged residual terms"),
        "reg-arima-d-input": ("differencing order", "differenced"),
        "reg-arima-q-input": ("moving-average order", "forecast-error terms"),
        "reg-garch-p-input": ("arch order", "lagged squared residual terms"),
        "reg-garch-q-input": ("conditional-variance terms", "volatility persistence"),
    }

    for control_id, required_tokens in expected.items():
        text, source = tooltip_text_and_source(control_id)
        text_lower = text.lower()
        assert source == "explicit"
        for token in required_tokens:
            assert token in text_lower, (control_id, token, text)


def test_portfolio_delete_and_portfolio_selector_tooltips_are_specific():
    delete_text, delete_source = tooltip_text_and_source("po-delete-portfolio-button")
    select_text, select_source = tooltip_text_and_source("po-weight-portfolio-select")

    assert delete_source == "explicit"
    delete_lower = delete_text.lower()
    assert "selected saved optimization result portfolio" in delete_lower
    assert "cannot be restored without rerunning optimization" in delete_lower
    assert "does not remove the underlying imported return-series dataset" in delete_lower

    assert select_source == "explicit"
    select_lower = select_text.lower()
    assert "saved optimization result portfolio" in select_lower
    assert "weights, risk, attribution, turnover, and frontier-linked views" in select_lower
    assert "selection sets the active value used by the surrounding control group" not in select_lower


def test_periodicity_tooltip_explains_daily_trading_and_tab_scope():
    text, source = tooltip_text_and_source("at-periodicity-select")
    text_lower = text.lower()

    assert source == "explicit"
    assert "frequency" in text_lower
    assert "daily (trading)" in text_lower
    assert "all tabs are calculated based on this period length" in text_lower


def test_date_range_shortcut_tooltips_explain_selected_series_effect():
    checks = {
        "at-common-range-button": ("selected series", "overlap", "selected periodicity"),
        "at-common-daily-button": ("selected series", "daily", "daily (trading)", "switches periodicity"),
        "at-maximum-range-button": ("selected series", "earliest", "latest", "current periodicity"),
    }

    for control_id, required_tokens in checks.items():
        text, source = tooltip_text_and_source(control_id)
        text_lower = text.lower()
        assert source == "explicit"
        for token in required_tokens:
            assert token in text_lower, (control_id, token, text)


def test_date_picker_tooltips_focus_on_boundary_behavior():
    for control_id, boundary_token in (
        ("at-start-date-picker", "first date"),
        ("at-end-date-picker", "last date"),
    ):
        text, source = tooltip_text_and_source(control_id)
        text_lower = text.lower()

        assert source == "explicit"
        assert boundary_token in text_lower
        assert "excluded from statistics" in text_lower
        assert "common range" in text_lower
        assert "adjust start and end together" not in text_lower


def test_all_page_tooltips_are_explicit_and_detailed():
    by_id = _rendered_page_tooltip_entries()

    assert by_id

    banned_phrases = {
        "turn this option on or off",
        "configure this setting",
        "choose a value for this setting",
    }

    for control_id, label in by_id.items():
        text, source = tooltip_text_and_source(control_id)
        assert source == "explicit"
        assert _sentence_count(text) >= 2
        assert len(text) >= 140
        text_lower = text.lower()
        for phrase in banned_phrases:
            assert phrase not in text_lower

        label_lower = str(label or "").lower()
        assert _sentence_count(label) >= 2
        assert len(label) >= 120
        for phrase in banned_phrases:
            assert phrase not in label_lower


def test_rendered_tooltips_do_not_use_default_template_copy():
    fallback_signature = "the value here is read directly by callbacks that build tables, charts, and exports"

    for control_id in _rendered_page_tooltip_entries():
        text, source = tooltip_text_and_source(control_id)
        assert source == "explicit"
        assert fallback_signature not in text.lower(), control_id


def test_welcome_tooltips_are_action_specific():
    expected_substrings = {
        "at-welcome-add-db-btn": ("import", "returns", "cma", "indices", "aa tool"),
        "at-welcome-add-portfolios-peer-btn": ("portfolio", "aa return", "peer", "mean return", "benchmark"),
        "at-welcome-add-portfolios-index-btn": ("portfolio", "aa return", "index", "benchmark"),
        "at-welcome-add-portfolios-other-btn": ("alternative", "portfolio", "benchmark"),
        "at-welcome-add-raw-factor-btn": (
            "mrd factor data",
            "table",
            "disabled",
            "fee",
            "include benchmark",
            "convert",
            "divide by",
        ),
        "at-welcome-add-raw-funds-btn": ("raw", "fund"),
        "at-welcome-add-raw-performance-btn": ("raw", "performance"),
        "at-welcome-add-series-btn": (
            "import",
            "csv",
            "excel",
            "sample",
            "morningstar",
            "performance reporting",
        ),
        "at-welcome-view-portfolio": ("portfolio optimization", "switch", "module"),
        "at-welcome-view-regression": ("regression", "switch", "module"),
    }

    for control_id, required_tokens in expected_substrings.items():
        text, source = tooltip_text_and_source(control_id)
        text_lower = text.lower()
        assert source == "explicit"
        for token in required_tokens:
            assert token in text_lower, (control_id, token, text)


def test_raw_import_tooltips_describe_mode_specific_availability():
    checks = {
        "at-menu-add-raw-factor": ("mrd factor data", "table", "fee", "include benchmark", "disabled"),
        "at-raw-db-add-table-select": ("disabled in factor mode", "funds", "performance"),
        "at-raw-db-add-fee-select": ("disabled in factor mode", "gross", "net"),
        "at-raw-db-add-include-benchmark": ("disabled in factor and funds modes", "performance"),
        "at-raw-db-add-convert-returns": ("convert", "returns", "factor mode"),
        "at-raw-db-add-divide-by": ("convert to returns is unchecked", "factor"),
    }

    for control_id, required_tokens in checks.items():
        text, source = tooltip_text_and_source(control_id)
        text_lower = text.lower()
        assert source == "explicit"
        for token in required_tokens:
            assert token in text_lower, (control_id, token, text)


def test_aa_tool_indices_tooltips_are_intent_first():
    ids = (
        "at-welcome-add-db-btn",
        "at-menu-add-from-db",
        "po-welcome-add-db-btn",
        "po-menu-add-from-db",
        "reg-welcome-add-db-btn",
        "reg-menu-add-from-db",
    )
    banned = ("opens", "this path lets you", "without uploading files first", "centrally maintained")

    for control_id in ids:
        text, source = tooltip_text_and_source(control_id)
        text_lower = text.lower()
        assert source == "explicit"
        for token in ("import", "returns", "cma", "indices", "aa tool"):
            assert token in text_lower, (control_id, token, text)
        for phrase in banned:
            assert phrase not in text_lower, (control_id, phrase, text)


def test_factor_and_regime_select_tooltips_explain_source_tags():
    factor_text, factor_source = tooltip_text_and_source("at-factor-series-select")
    regime_text, regime_source = tooltip_text_and_source("at-regime-definition-select")

    assert factor_source == "explicit"
    assert "[raw]" in factor_text.lower()
    assert "[db]" in factor_text.lower()
    assert "[session]" in factor_text.lower()

    assert regime_source == "explicit"
    assert "[db]" in regime_text.lower()
    assert "[session]" in regime_text.lower()


def test_series_selection_delete_header_tooltips_describe_removal_not_reenable():
    cols = [{"field": "Delete", "headerName": "Del"}]
    for grid_id in (
        "at-series-selection-grid",
        "po-series-selection-grid",
        "reg-series-selection-grid",
    ):
        out = apply_header_tooltips(cols, grid_id)
        text = str(out[0].get("headerTooltip") or "").lower()
        assert "removed from the working dataset" in text
        assert "add or import the series again" in text
        assert "re-enable" not in text


def test_switch_tooltips_use_module_language_and_avoid_internal_jargon():
    control_ids = (
        "at-welcome-view-portfolio",
        "at-welcome-view-regression",
        "po-welcome-view-analytics",
        "po-welcome-view-regression",
        "reg-welcome-view-analytics",
        "reg-welcome-view-portfolio",
        "at-menu-view-portfolio",
        "at-menu-view-regression",
        "po-menu-view-analytics",
        "po-menu-view-regression",
        "reg-menu-view-analytics",
        "reg-menu-view-portfolio",
    )
    banned_phrases = ("compatible state is preserved", "after data is loaded")

    for control_id in control_ids:
        text, source = tooltip_text_and_source(control_id)
        text_lower = text.lower()
        assert source == "explicit"
        assert "switch to the" in text_lower
        assert "module" in text_lower
        for phrase in banned_phrases:
            assert phrase not in text_lower, (control_id, phrase, text)


def test_select_series_open_button_tooltips_are_specific():
    expected = {
        "at-open-series-modal-button": ("series selection modal", "benchmark", "l/s", "scale vol"),
        "po-open-modal-button": ("series selection modal", "investable", "cma benchmark", "scale vol"),
        "reg-open-modal-button": ("series selection modal", "y and x", "lag", "coefficient-bound"),
    }
    banned = ("runs the", "action in", "applies current selector and input state")

    for control_id, required_tokens in expected.items():
        text, source = tooltip_text_and_source(control_id)
        text_lower = text.lower()
        assert source == "explicit"
        for token in required_tokens:
            assert token in text_lower, (control_id, token, text)
        for phrase in banned:
            assert phrase not in text_lower, (control_id, phrase, text)
