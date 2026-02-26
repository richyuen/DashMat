"""Shared tooltip helpers for DashMat control surfaces."""

from __future__ import annotations

import os
import re
from typing import Any

import dash_mantine_components as dmc
from dash.development.base_component import Component


TOOLTIP_STYLE_DEFAULT = {
    "position": "top-start",
    "withArrow": True,
    "multiline": True,
    "w": 280,
    "openDelay": 450,
    "closeDelay": 90,
}


_EXPLICIT_TOOLTIPS: dict[str, str] = {
    "at-menu-download-excel": "Export the current Analytics Tool state to Excel.",
    "po-menu-download-excel": "Export the current optimization state to Excel.",
    "reg-menu-download-excel": "Export current regression outputs to Excel.",
    "at-open-series-modal-button": "Choose the return series used by analytics and charts.",
    "po-open-modal-button": "Choose the return series used by portfolio optimization.",
    "reg-open-modal-button": "Choose dependent and explanatory return series.",
}


_PATTERN_TOOLTIPS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"common-range"), "Set start/end dates to the common overlap range."),
    (re.compile(r"common-daily"), "Set dates to the common range available at daily frequency."),
    (re.compile(r"maximum-range"), "Use the widest available date range for current series."),
    (re.compile(r"start-date|date-range-start"), "Set the first date used for calculations."),
    (re.compile(r"end-date|date-range-end"), "Set the last date used for calculations."),
    (re.compile(r"periodicity"), "Select data periodicity for calculations."),
    (re.compile(r"menu-view"), "Switch to another DashMat workflow page."),
    (re.compile(r"help-guide|menu-help"), "Open quick guidance for this workflow."),
    (re.compile(r"bl-add-view|add-view"), "Add a new view entry to the current assumptions set."),
    (re.compile(r"estimate"), "Estimate values from historical inputs."),
    (re.compile(r"returns-type"), "Choose total or excess return mode."),
    (re.compile(r"benchmark"), "Set benchmark mappings for relative metrics."),
    (re.compile(r"long-short"), "Mark series treated as long/short spreads."),
    (re.compile(r"vol-scaler|halflife|half-life"), "Configure volatility scaling intensity."),
    (re.compile(r"vol-scaling"), "Map series eligible for volatility scaling."),
    (re.compile(r"monthly-view"), "Switch between annual and monthly calendar views."),
    (re.compile(r"block-width"), "Set matrix block size used for visualized correlation blocks."),
    (re.compile(r"factor"), "Configure factor analysis inputs and behavior."),
    (re.compile(r"regime"), "Configure regime definitions and analysis behavior."),
    (re.compile(r"window|rolling"), "Set rolling-window settings for this calculation."),
    (re.compile(r"calendar"), "Control calendar-period summary output."),
    (re.compile(r"growth"), "Control growth-of-dollar display settings."),
    (re.compile(r"drawdown"), "Control drawdown display settings."),
    (re.compile(r"scatter|correlation|covariance"), "Control matrix/scatter display settings."),
    (re.compile(r"statistics"), "Control summary statistics output."),
    (re.compile(r"opt|optimizer|objective|constraint"), "Configure optimization assumptions and constraints."),
    (re.compile(r"anova|arima|garch"), "Configure model diagnostics and time-series add-ons."),
    (re.compile(r"period|window-size|opt-step"), "Set period sizing used by model windows."),
    (re.compile(r"download"), "Download results from the current view."),
    (re.compile(r"upload|import|add-from|load"), "Load data from the selected source."),
    (re.compile(r"save"), "Save the current configuration or results."),
    (re.compile(r"delete|clear|reset"), "Remove current selection or reset this setting."),
    (re.compile(r"open"), "Open the related dialog or selector."),
    (re.compile(r"close"), "Close this dialog without additional changes."),
    (re.compile(r"\bnew\b"), "Start a new draft based on current defaults."),
    (re.compile(r"\buse\b"), "Apply the selected draft to the active workflow."),
    (re.compile(r"run"), "Run the current analysis with selected settings."),
    (re.compile(r"select|dropdown"), "Choose a value for this setting."),
    (re.compile(r"input"), "Enter a value for this setting."),
    (re.compile(r"switch|toggle"), "Turn this option on or off."),
    (re.compile(r"chart-switch|table|chart"), "Switch between chart and table views."),
    (re.compile(r"tab|tabs"), "Switch between result panels."),
    (re.compile(r"series"), "Select which series to include."),
    (re.compile(r"modal|dialog"), "Open or control a detailed settings dialog."),
]


_INTERACTIVE_COMPONENTS_BY_NAMESPACE: dict[str, set[str]] = {
    "dmc": {
        "ActionIcon",
        "Button",
        "Checkbox",
        "DateInput",
        "MultiSelect",
        "NumberInput",
        "Radio",
        "RadioGroup",
        "RangeSlider",
        "SegmentedControl",
        "Select",
        "Slider",
        "Switch",
        "TextInput",
        "Textarea",
    },
    "dcc": {
        "Checklist",
        "DatePickerRange",
        "DatePickerSingle",
        "Dropdown",
        "Input",
        "RadioItems",
        "RangeSlider",
        "Slider",
        "Upload",
    },
    "html": {
        "Button",
        "Input",
        "Select",
        "Textarea",
    },
}


_SKIP_ANCESTOR_COMPONENTS = {
    "MenuTarget",
    "MenuDropdown",
    "Tooltip",
}


_SKIP_WRAP_COMPONENTS = {
    "MenuItem",
}


def tooltips_enabled() -> bool:
    """Feature gate for global tooltip wrapping."""
    raw = str(os.getenv("DASHMAT_ENABLE_GLOBAL_TOOLTIPS", "1")).strip().lower()
    return raw not in {"0", "false", "no", "off", "disabled"}


def is_interactive_component_name(namespace: str, component_name: str) -> bool:
    namespace_key = str(namespace or "").strip().lower()
    name = str(component_name or "").strip()
    return name in _INTERACTIVE_COMPONENTS_BY_NAMESPACE.get(namespace_key, set())


def tooltip_source(control_id: str) -> str:
    _, source = tooltip_text_and_source(control_id)
    return source


def has_custom_tooltip(control_id: str) -> bool:
    return tooltip_source(control_id) != "fallback"


def tooltip_text(control_id: str, fallback_label: str | None = None) -> str:
    text, _ = tooltip_text_and_source(control_id, fallback_label=fallback_label)
    return text


def tooltip_text_and_source(control_id: str, fallback_label: str | None = None) -> tuple[str, str]:
    key = str(control_id or "").strip()
    if not key:
        return "Configure this setting.", "fallback"

    exact = _EXPLICIT_TOOLTIPS.get(key)
    if exact:
        return exact, "explicit"

    lowered = key.lower()
    for pattern, text in _PATTERN_TOOLTIPS:
        if pattern.search(lowered):
            return text, "pattern"

    return _fallback_tooltip_text(key, fallback_label=fallback_label), "fallback"


def apply_tooltips_to_layout(layout: Any, page_key: str | None = None):
    """Wrap interactive controls with delayed tooltips.

    This function intentionally decorates only controls with explicit string IDs to
    keep callback contracts stable and avoid wrapping structural containers.
    """
    if not tooltips_enabled():
        return layout
    return _decorate_value(layout, page_key=page_key, ancestors=())


def _component_namespace(component: Component) -> str:
    module_name = str(component.__class__.__module__)
    if "dash_mantine_components" in module_name:
        return "dmc"
    if "dash.dcc" in module_name:
        return "dcc"
    if "dash.html" in module_name:
        return "html"
    if "dash_ag_grid" in module_name:
        return "dag"
    return ""


def _component_id(component: Component) -> str | None:
    value = getattr(component, "id", None)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _component_label(component: Component) -> str | None:
    label = getattr(component, "label", None)
    if isinstance(label, str) and label.strip():
        return label.strip()
    return None


def _should_wrap_component(component: Component, ancestors: tuple[str, ...]) -> bool:
    if any(name in _SKIP_ANCESTOR_COMPONENTS for name in ancestors):
        return False
    component_name = component.__class__.__name__
    if component_name in _SKIP_WRAP_COMPONENTS:
        return False
    control_id = _component_id(component)
    if not control_id:
        return False
    namespace = _component_namespace(component)
    return is_interactive_component_name(namespace, component_name)


def _decorate_value(value: Any, page_key: str | None, ancestors: tuple[str, ...]):
    if isinstance(value, list):
        return [_decorate_value(item, page_key=page_key, ancestors=ancestors) for item in value]
    if isinstance(value, tuple):
        return tuple(_decorate_value(item, page_key=page_key, ancestors=ancestors) for item in value)
    if isinstance(value, dict):
        return {
            key: _decorate_value(item, page_key=page_key, ancestors=ancestors)
            for key, item in value.items()
        }
    if not isinstance(value, Component):
        return value

    component_name = value.__class__.__name__
    next_ancestors = ancestors + (component_name,)

    for prop in getattr(value, "_prop_names", []):
        if prop == "id":
            continue
        try:
            prop_value = getattr(value, prop)
        except Exception:
            continue
        if prop_value is None or isinstance(prop_value, (str, int, float, bool)):
            continue
        decorated = _decorate_value(prop_value, page_key=page_key, ancestors=next_ancestors)
        if decorated is not prop_value:
            try:
                setattr(value, prop, decorated)
            except Exception:
                continue

    if not _should_wrap_component(value, ancestors):
        return value

    control_id = _component_id(value)
    text = tooltip_text(control_id or "", fallback_label=_component_label(value))
    return dmc.Tooltip(
        label=text,
        **TOOLTIP_STYLE_DEFAULT,
        children=value,
    )


def _fallback_tooltip_text(control_id: str, fallback_label: str | None = None) -> str:
    if fallback_label:
        return f"Configure {str(fallback_label).strip().lower()}."

    label = re.sub(r"^(at|po|reg)-", "", str(control_id or "").strip(), flags=re.IGNORECASE)
    label = re.sub(r"[-_]+", " ", label).strip()
    label = re.sub(
        r"\b(button|btn|input|select|switch|toggle|modal|dialog|tabs?|panel|menu|item|grid)\b",
        "",
        label,
        flags=re.IGNORECASE,
    )
    label = re.sub(r"\s+", " ", label).strip()
    if not label:
        return "Configure this setting."
    return f"Configure {label.lower()}."
