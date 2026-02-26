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
    "at-vol-scaler-input": (
        "Sets the annualized volatility target used to scale selected return series. "
        "Set to 0 to disable scaling entirely and use raw return magnitudes. "
        "When enabled, only series with Scale Vol checked are adjusted."
    ),
    "reg-vol-scaler-input": (
        "Sets the annualized volatility target for the regression input stream before model fitting. "
        "Set to 0 to disable scaling and run on raw return levels. "
        "Use this mainly to normalize series magnitudes when comparing coefficient stability across windows."
    ),
    "po-vol-scaler-input": (
        "Sets the annualized volatility target used before optimization and downstream analytics. "
        "Set to 0 to disable scaling and preserve original return volatility. "
        "Series-level Scale Vol toggles determine which assets this global target is applied to."
    ),
    "at-regime-def-vol-scaler": (
        "Sets volatility scaling for series used in regime assignment inputs. "
        "Set to 0 to disable scaling during regime-state construction. "
        "Use a positive value only when regime detection should be less driven by raw volatility differences."
    ),
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


INPUT_GRID_ALLOWLIST = {
    "at-series-selection-grid",
    "reg-series-selection-grid",
    "reg-linear-constraints-grid",
    "po-series-selection-grid",
    "po-linear-constraints-grid",
    "po-ex-ante-returns-grid",
    "po-ex-ante-matrix-grid",
    "po-bl-views-grid",
    "at-portfolio-add-grid",
    "reg-portfolio-add-grid",
    "po-portfolio-add-grid",
    "at-raw-db-add-grid",
    "reg-raw-db-add-grid",
    "po-raw-db-add-grid",
}


OUTPUT_GRID_ALLOWLIST = {
    "reg-anova-decomposition-grid",
    "reg-anova-parameter-grid",
}


GRID_HEADER_TOOLTIP_TARGETS = INPUT_GRID_ALLOWLIST.union(OUTPUT_GRID_ALLOWLIST)


CANONICAL_COLUMN_TOOLTIPS: dict[str, str] = {
    "series": (
        "Name of the series row being configured. "
        "Edits to this row apply to this exact series in the active workflow."
    ),
    "benchmark": (
        "Benchmark series used for excess-return and relative risk statistics. "
        "Choose None to disable benchmark-relative calculations for this row."
    ),
    "cmabench": (
        "CMA benchmark mapping used for optimization assumptions and reporting alignment. "
        "Leave blank only when no CMA benchmark should be linked."
    ),
    "l s": (
        "Marks this series as a long-short spread instead of a long-only stream. "
        "This affects interpretation of relative metrics and some reporting conventions."
    ),
    "scale vol": (
        "Controls whether the global Vol Scaler target is applied to this series. "
        "Turn off when a series should keep original volatility while others are scaled."
    ),
    "delete": (
        "Marks the row for removal from the working selection list. "
        "Deleted rows are excluded from analysis until re-enabled."
    ),
    "del": (
        "Marks this row for deletion from the active selection set. "
        "Use carefully because deleted rows are excluded from subsequent runs."
    ),
    "y": (
        "Sets this row as the dependent variable (Y) for regression. "
        "Only one row should normally be selected as Y at a time."
    ),
    "x": (
        "Includes this series as an explanatory variable (X) in regression. "
        "Enable multiple X rows to fit multi-factor models."
    ),
    "lag": (
        "Applies an integer lag to the explanatory series before fitting. "
        "Positive lags shift X backward in time to model delayed relationships."
    ),
    "min beta": (
        "Lower bound on the coefficient for this variable when constraints are enabled. "
        "Ignored unless Enable is checked for the row."
    ),
    "max beta": (
        "Upper bound on the coefficient for this variable when constraints are enabled. "
        "Ignored unless Enable is checked for the row."
    ),
    "enable": (
        "Turns per-variable beta bounds on or off for this row. "
        "When disabled, Min/Max Beta values are not enforced."
    ),
    "portfolio": (
        "Portfolio identifier to import into the working dataset. "
        "Use one row per portfolio import request."
    ),
    "type": (
        "Defines the return interpretation used for this imported portfolio row. "
        "Choose the type matching the source benchmarking convention."
    ),
    "include benchmark": (
        "Controls whether the benchmark companion series is imported with the portfolio row. "
        "Enable when relative metrics or spread comparisons are needed."
    ),
    "benchmark type": (
        "Benchmark family used when including benchmark series for this row. "
        "Pick the benchmark type consistent with the portfolio import mode."
    ),
    "table": (
        "Database table frequency source used for import (daily or monthly). "
        "Select the table matching the desired periodicity."
    ),
    "fee": (
        "Gross or net return flavor for imported database series. "
        "Choose net unless gross returns are intentionally required."
    ),
    "convert": (
        "Controls whether level-like source values are converted into returns. "
        "Use only when source data is not already a return series."
    ),
    "convert to returns": (
        "Controls whether source values are transformed into return increments. "
        "Leave off when imported data is already in return form."
    ),
    "divide by": (
        "Scaling divisor applied after conversion when source values are percent-like. "
        "Common use is 100 for values such as 2.5 meaning 2.5%."
    ),
    "constraint": (
        "Human-readable name for the linear constraint row. "
        "Used for identification and review; math is driven by coefficients and bounds."
    ),
    "min": (
        "Lower bound for the linear constraint expression. "
        "The weighted sum must remain above this value."
    ),
    "max": (
        "Upper bound for the linear constraint expression. "
        "The weighted sum must remain below this value."
    ),
    "min wt": (
        "Minimum allowed portfolio weight for this asset row (percent). "
        "Ignored when Force is enabled for exact max behavior."
    ),
    "max wt": (
        "Maximum allowed portfolio weight for this asset row (percent). "
        "Used as a hard cap during optimization."
    ),
    "force": (
        "When enabled, forces this row toward the configured maximum-weight behavior. "
        "Use sparingly because it can dominate optimization flexibility."
    ),
    "source": (
        "ANOVA component source used in decomposition. "
        "Shows how total variance is partitioned across model and residual terms."
    ),
    "df": (
        "Degrees of freedom associated with the ANOVA source row. "
        "Used in mean-square and F-statistic calculations."
    ),
    "ss": (
        "Sum of squares for the ANOVA source row. "
        "Higher values indicate larger contribution to total variation."
    ),
    "ms": (
        "Mean square computed as SS divided by df for this ANOVA row. "
        "Used directly in the F-statistic ratio."
    ),
    "f": (
        "F-statistic comparing explained variance to residual variance. "
        "Larger values generally indicate stronger model explanatory power."
    ),
    "p value": (
        "Significance probability associated with the row statistic. "
        "Lower values indicate stronger evidence against the null hypothesis."
    ),
    "parameter": (
        "Model coefficient name or diagnostic parameter label. "
        "Interpret alongside coefficient magnitude and uncertainty columns."
    ),
    "coefficient": (
        "Estimated parameter value for the selected model window. "
        "Sign and magnitude describe direction and strength of relationship."
    ),
    "std error": (
        "Standard error of the coefficient estimate. "
        "Lower values indicate tighter parameter uncertainty."
    ),
    "t stat": (
        "t-statistic for testing whether a coefficient differs from zero. "
        "Larger absolute values indicate stronger evidence of non-zero effect."
    ),
    "ci low 95": (
        "Lower bound of the 95% confidence interval for the parameter. "
        "Use with CI High to assess estimate uncertainty range."
    ),
    "ci high 95": (
        "Upper bound of the 95% confidence interval for the parameter. "
        "Use with CI Low to assess estimate uncertainty range."
    ),
}


GRID_COLUMN_TOOLTIP_OVERRIDES: dict[str, dict[str, str]] = {
    "at-series-selection-grid": {
        "Benchmark": (
            "Benchmark used for this series in excess-return and relative statistics. "
            "Choose None to keep calculations absolute for this row."
        ),
    },
    "reg-series-selection-grid": {
        "Y": (
            "Set this row as the dependent variable for regression estimation. "
            "Only one Y should be active in a standard regression run."
        ),
        "X": (
            "Include this row as an explanatory regressor. "
            "You can enable multiple X rows for multi-variable models."
        ),
    },
    "po-series-selection-grid": {
        "CMABench": (
            "CMA benchmark tag used for optimizer assumptions and output comparison. "
            "Keep mappings consistent with your selected portfolio universe."
        ),
    },
}


GRID_TOOLTIP_DASH_OPTIONS_DEFAULTS = {
    "tooltipShowDelay": 500,
    "tooltipHideDelay": 120,
    "tooltipMouseTrack": False,
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


def _normalize_header_key(value: str | None) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    text = re.sub(r"[_\-]+", " ", text)
    text = text.replace("/", " ")
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _grid_specific_fallback(grid_id: str, field: str, header_name: str) -> str | None:
    token = field or header_name
    if not token:
        return None
    if grid_id.endswith("linear-constraints-grid"):
        return (
            f"Coefficient for {token} in the linear constraint expression. "
            "This value is multiplied by the asset weight when enforcing Min/Max bounds."
        )
    if grid_id.endswith("ex-ante-returns-grid"):
        return (
            f"Ex-ante expected return input for {token}. "
            "This value overrides or supplements estimated return assumptions used by the optimizer."
        )
    if grid_id.endswith("ex-ante-matrix-grid"):
        return (
            f"Ex-ante matrix input for {token}. "
            "Edit only when you intend to override estimated covariance/correlation structure."
        )
    if grid_id.endswith("bl-views-grid"):
        return (
            f"Black-Litterman view setting for {token}. "
            "This field controls view direction, magnitude, or confidence for blending prior and views."
        )
    return (
        f"Configuration field for {token}. "
        "Review this column before running calculations that depend on the current grid."
    )


def _resolve_header_tooltip(grid_id: str, field: str, header_name: str) -> str | None:
    overrides = GRID_COLUMN_TOOLTIP_OVERRIDES.get(grid_id, {})
    for key in (field, header_name, _normalize_header_key(field), _normalize_header_key(header_name)):
        if key and key in overrides:
            return overrides[key]

    for key in (_normalize_header_key(field), _normalize_header_key(header_name)):
        if key and key in CANONICAL_COLUMN_TOOLTIPS:
            return CANONICAL_COLUMN_TOOLTIPS[key]

    return _grid_specific_fallback(grid_id, field, header_name)


def apply_header_tooltips(column_defs: list[dict] | tuple[dict, ...] | None, grid_id: str, include_blank_headers: bool = False) -> list[dict]:
    if not column_defs:
        return []
    out: list[dict] = []
    for item in column_defs:
        col = dict(item or {})
        field = str(col.get("field") or "").strip()
        header_name = str(col.get("headerName") or "").strip()
        if not include_blank_headers and not field and not header_name:
            out.append(col)
            continue
        text = _resolve_header_tooltip(str(grid_id or "").strip(), field, header_name)
        if text:
            col["headerTooltip"] = text
        out.append(col)
    return out


def grid_tooltip_dash_options(existing: dict | None = None) -> dict:
    payload = dict(GRID_TOOLTIP_DASH_OPTIONS_DEFAULTS)
    if isinstance(existing, dict):
        payload.update(existing)
    return payload
