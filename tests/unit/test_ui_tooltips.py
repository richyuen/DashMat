from __future__ import annotations

import dash_mantine_components as dmc

from utils.ui_tooltips import (
    apply_header_tooltips,
    apply_tooltips_to_layout,
    grid_tooltip_dash_options,
    has_custom_tooltip,
    tooltip_text_and_source,
)


def _tree_has_tooltip_for_id(node, target_id: str) -> bool:
    if node is None:
        return False

    if isinstance(node, (list, tuple)):
        return any(_tree_has_tooltip_for_id(item, target_id) for item in node)

    class_name = node.__class__.__name__
    if class_name == "Tooltip":
        children = getattr(node, "children", None)
        child_id = getattr(children, "id", None)
        if child_id == target_id:
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


def test_tooltip_text_source_prefers_explicit_and_patterns():
    explicit_text, explicit_source = tooltip_text_and_source("at-menu-download-excel")
    assert explicit_source == "explicit"
    assert "excel" in explicit_text.lower()

    pattern_text, pattern_source = tooltip_text_and_source("at-periodicity-select")
    assert pattern_source == "pattern"
    assert "periodicity" in pattern_text.lower()


def test_tooltip_text_source_fallback_is_non_empty():
    text, source = tooltip_text_and_source("at-custom-unknown-control")
    assert source == "fallback"
    assert isinstance(text, str)
    assert text.strip()


def test_has_custom_tooltip_detects_non_fallback_ids():
    assert has_custom_tooltip("at-periodicity-select") is True
    assert has_custom_tooltip("at-custom-unknown-control") is False


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
    assert out["tooltipHideDelay"] == 120
    assert out["tooltipMouseTrack"] is False


def test_vol_scaler_tooltip_copy_is_detailed():
    text, source = tooltip_text_and_source("po-vol-scaler-input")

    assert source == "explicit"
    assert "annualized volatility target" in text.lower()
    assert "set to 0 to disable" in text.lower()
