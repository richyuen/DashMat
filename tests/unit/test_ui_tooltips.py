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


def test_all_page_tooltips_are_explicit_and_detailed():
    import app  # noqa: F401
    import pages.analyticstool as analyticstool
    import pages.portopt as portopt
    import pages.regression as regression

    entries = []
    for layout in (analyticstool.layout, portopt.layout, regression.layout):
        entries.extend(_collect_tooltip_entries(layout))

    by_id = {}
    for control_id, label in entries:
        if control_id.startswith(("at-", "po-", "reg-")):
            by_id[control_id] = label

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
