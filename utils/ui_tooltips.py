"""Shared tooltip helpers for DashMat control surfaces."""

from __future__ import annotations

import os
import re
from typing import Any

import dash_mantine_components as dmc
from dash import html
from dash.development.base_component import Component


TOOLTIP_STYLE_DEFAULT = {
    "position": "top-start",
    "withArrow": True,
    "multiline": True,
    "w": 280,
    "openDelay": 450,
    "closeDelay": 180,
}


_EXPLICIT_TOOLTIPS: dict[str, str] = {}


_WORKFLOW_BY_PREFIX: dict[str, str] = {
    "at": "Analytics Tool",
    "po": "Portfolio Optimization",
    "reg": "Regression",
}


_RENDERED_SUFFIX_PREFIXES_RAW = """
add-constraint-btn|po,reg
alpha-input|reg
anova-window-select|reg
arima-d-input|reg
arima-p-input|reg
arima-q-input|reg
attribution-chart-switch|po
bl-add-view|po
bl-clear-views|po
bl-tau-input|po
calendar-series-select|po,reg
calendar-view-select|po,reg
clear-constraints-btn|po,reg
close-completion-button|po
cma-load-cancel|po
cma-load-confirm|po
cma-type-select|po
cma-version-select|po
common-daily-button|at,po
common-range-button|at,po,reg
correlation-exp-wt-switch|at
correlation-halflife-input|at
correlation-view-switch|at
correlogram-block-width|at
db-add-cancel-button|at,po,reg
db-add-ok-button|at,po,reg
db-add-series-select|at,po,reg
delete-portfolio-button|po
delete-result-btn|reg
download-sample-daily-btn|at,po,reg
download-sample-monthly-btn|at,po,reg
drawdown-chart-switch|at,po,reg
end-date-picker|at,po,reg
estimate-matrix-btn|po
estimate-returns-btn|po
ex-ante-matrix-clear|po
ex-ante-matrix-upload|po
ex-ante-matrix-upload-btn|po
ex-ante-mode-select|po
ex-ante-returns-clear|po
ex-ante-returns-upload|po
exp-wt-cov-switch|po
exp-wt-switch|reg
factor-def-close-btn|at
factor-def-delete-btn|at
factor-def-description-input|at
factor-def-long-agg-type|at
factor-def-long-components|at
factor-def-long-lag|at
factor-def-name-input|at
factor-def-new-btn|at
factor-def-output-transform|at
factor-def-save-db-btn|at
factor-def-save-local-btn|at
factor-def-select|at
factor-def-short-agg-type|at
factor-def-short-components|at
factor-def-use-btn|at
factor-mode-select|at
factor-open-modal-btn|at
factor-quantiles-input|at
factor-series-select|at
factor-transform-select|at
fill-in-sample-select|po,reg
force-zero-intercept-switch|reg
frontier-chart-switch|po
frontier-rm-select|po
frontier-window-select|po
garch-p-input|reg
garch-q-input|reg
growth-chart-switch|at,po,reg
growth-portfolio-multiselect|po
halflife-input|po,reg
l1-ratio-input|reg
load-db-matrix-btn|po
load-db-returns-btn|po
load-session-upload|at,po,reg
maximum-range-button|at,po,reg
menu-help-guide|at,po,reg
menu-view-analytics|po,reg
menu-view-portfolio|at,reg
menu-view-regression|at,po
missing-data-select|po,reg
modal-cancel-button|at,po,reg
modal-ok-button|at,po,reg
model-select|reg
monthly-series-select|at
monthly-view-checkbox|at
objective-select|po
open-modal-button|po,reg
open-series-modal-button|at
opt-model-select|po
opt-step-input|po,reg
opt-step-unit-select|po,reg
opt-window-select|po
periodicity-select|at,po,reg
portfolio-add-benchmark-type-select|at,po,reg
portfolio-add-cancel-button|at,po,reg
portfolio-add-include-benchmark|at,po,reg
portfolio-add-ok-button|at,po,reg
portfolio-add-row-btn|at,po,reg
portfolio-add-series-select|at,po,reg
portfolio-add-type-select|at,po,reg
portfolio-clear-rows-btn|at,po,reg
portfolio-delete-row-btn|at,po,reg
portfolio-name-input|po
underlying-add-base-select|at,po,reg
underlying-add-cancel-button|at,po,reg
underlying-add-desc-multiselect|at,po,reg
underlying-add-ok-button|at,po,reg
underlying-add-row-btn|at,po,reg
underlying-add-type-multiselect|at,po,reg
underlying-clear-rows-btn|at,po,reg
underlying-delete-row-btn|at,po,reg
raw-db-add-cancel-button|at,po,reg
raw-db-add-convert-returns|at,po,reg
raw-db-add-divide-by|at,po,reg
raw-db-add-fee-select|at,po,reg
raw-db-add-include-benchmark|at,po,reg
raw-db-add-ok-button|at,po,reg
raw-db-add-row-btn|at,po,reg
raw-db-add-series-select|at,po,reg
raw-db-add-table-select|at,po,reg
raw-db-clear-rows-btn|at,po,reg
raw-db-delete-row-btn|at,po,reg
regime-def-close-btn|at
regime-def-delete-btn|at
regime-def-description-input|at
regime-def-method-type|at
regime-def-min-observations|at
regime-def-name-input|at
regime-def-new-btn|at
regime-def-num-regimes|at
regime-def-pca-standardize|at
regime-def-save-db-btn|at
regime-def-save-local-btn|at
regime-def-select|at
regime-def-single-series|at
regime-def-universe-series|at
regime-def-use-btn|at
regime-def-vol-scaler|at
regime-definition-select|at
regime-open-modal-btn|at
regression-name-input|reg
result-select|reg
returns-type-select|at
risk-chart-switch|po
robust-se-switch|reg
rolling-chart-switch|at,po,reg
rolling-metric-select|at,po,reg
rolling-return-type-select|at,po,reg
rolling-summary-chart-switch|reg
rolling-summary-detail-switch|reg
rolling-window-select|at,po,reg
run-button|reg
scatter-mode-select|reg
scatter-x-select|reg
sheet-select-cancel-button|at,po,reg
sheet-select-dropdown|at,po,reg
sheet-select-import-all-button|at,po,reg
sheet-select-ok-button|at,po,reg
start-date-picker|at,po,reg
turnover-chart-switch|po
upload-data|at,po,reg
vol-scaler-input|at,po,reg
weight-chart-switch|po
weight-portfolio-select|po
weights-chart-switch|reg
welcome-add-db-btn|at,po,reg
welcome-add-portfolios-index-btn|at,po,reg
welcome-add-portfolios-other-btn|at,po,reg
welcome-add-portfolios-peer-btn|at,po,reg
welcome-add-portfolios-underlying-btn|at,po,reg
welcome-add-raw-factor-btn|at,po,reg
welcome-add-raw-funds-btn|at,po,reg
welcome-add-raw-performance-btn|at,po,reg
welcome-add-series-btn|at,po,reg
welcome-view-analytics|po,reg
welcome-view-portfolio|at,reg
welcome-view-regression|at,po
window-size-input|po,reg
window-type-select|reg
"""


def _parse_rendered_suffix_prefixes(raw: str) -> dict[str, tuple[str, ...]]:
    out: dict[str, tuple[str, ...]] = {}
    for line in str(raw or "").strip().splitlines():
        token = str(line or "").strip()
        if not token or "|" not in token:
            continue
        suffix, prefixes = token.split("|", 1)
        prefix_values = tuple(sorted(p.strip() for p in prefixes.split(",") if p.strip()))
        if suffix.strip() and prefix_values:
            out[suffix.strip()] = prefix_values
    return out


_RENDERED_SUFFIX_PREFIXES = _parse_rendered_suffix_prefixes(_RENDERED_SUFFIX_PREFIXES_RAW)


def _workflow_for_prefix(prefix: str) -> str:
    return _WORKFLOW_BY_PREFIX.get(str(prefix or "").strip().lower(), "DashMat")


def _humanize_tooltip_subject(control_id: str) -> str:
    label = re.sub(r"^(at|po|reg)-", "", str(control_id or "").strip(), flags=re.IGNORECASE)
    label = re.sub(r"[-_]+", " ", label).strip()
    label = re.sub(
        (
            r"\b(button|btn|input|select|switch|toggle|modal|dialog|tabs?|tab|panel|store|container|"
            r"wrapper|dummy|upload|grid|content|value|state|data|rows|row|columns|column)\b"
        ),
        " ",
        label,
        flags=re.IGNORECASE,
    )
    label = re.sub(r"\s+", " ", label).strip()
    return label if label else "this control"


def _default_explicit_tooltip(control_id: str) -> str:
    workflow = _workflow_for_prefix(str(control_id or "").split("-", 1)[0])
    subject = _humanize_tooltip_subject(control_id)
    return (
        f"Sets {subject} for {workflow}. "
        "This control drives the specific UI step associated with this field, such as naming, filtering, importing, or display mode selection. "
        "Interpret it in the context of the panel where it appears."
    )


def _table_chart_toggle_text(topic: str, workflow: str) -> str:
    return (
        f"Switches the {topic} view in {workflow} between Table and Chart output. "
        "Table shows exact values and is better for audits or export review, while Chart emphasizes shape, turning points, and relative behavior through time. "
        "Use the mode that matches whether you are validating numbers or presenting visual patterns."
    )


def _suffix_fallback_text(prefix: str, suffix: str, workflow: str) -> str:
    action = suffix.replace("-btn", "").replace("-button", "").replace("-", " ").strip()
    if suffix.endswith(("-btn", "-button")):
        return (
            f"Triggers the {action} action in {workflow}. "
            "This button executes the operation named by this control id and label. "
            "Use it for the specific modal, grid, or result action shown next to the button."
        )

    action = suffix.replace("-select", "").replace("-dropdown", "").replace("-", " ").strip()
    if suffix.endswith("-select") or suffix.endswith("-dropdown"):
        return (
            f"Selects the {action} option in {workflow}. "
            "This dropdown chooses which value the local panel uses for this field. "
            "Pick the item that matches the dataset, model output, or view you want to inspect."
        )

    action = suffix.replace("-input", "").replace("-", " ").strip()
    if suffix.endswith("-name-input"):
        return (
            f"Sets the {action} label in {workflow}. "
            "This text is used for naming, lookup, and display in selectors, tables, and exports where this item appears. "
            "Changing it updates identifiers and labels rather than model equations."
        )
    if suffix.endswith("-description-input"):
        return (
            f"Sets descriptive text for {action} in {workflow}. "
            "This field documents intent and interpretation so saved definitions are understandable when reused or shared. "
            "It is metadata and does not alter calculations directly."
        )
    if suffix.endswith("-input"):
        return (
            f"Sets the {action} value in {workflow}. "
            "This field is read by the specific model, import, or display step tied to this control id and label. "
            "Use a value that matches the method and horizon configured in the same panel."
        )

    action = suffix.replace("-switch", "").replace("-checkbox", "").replace("-", " ").strip()
    if suffix.endswith("-switch") or suffix.endswith("-checkbox"):
        return (
            f"Toggles {action} behavior in {workflow}. "
            "When enabled, the alternate processing or display path tied to this control is used; when disabled, the default path is used. "
            "Keep this setting aligned with the interpretation you want in the resulting output."
        )

    action = suffix.replace("-upload", "").replace("-", " ").strip()
    if suffix.endswith("-upload"):
        return (
            f"Uploads files for {action} in {workflow}. "
            "Uploaded content is parsed into working state and then referenced by dependent selectors, grids, and result callbacks. "
            "Use files that match the expected schema and units for this import path."
        )

    action = suffix.replace("-date-picker", "").replace("-", " ").strip()
    if suffix.endswith("-date-picker"):
        return (
            f"Sets the {action} date boundary in {workflow}. "
            "This boundary determines which observations are included by the active window and statistics logic. "
            "Common Range, Common Daily, and Max Range shortcuts can reset these boundaries from selected-series coverage."
        )

    return _default_explicit_tooltip(f"{prefix}-{suffix}")


def _suffix_tooltip_override(prefix: str, suffix: str) -> str | None:
    workflow = _workflow_for_prefix(prefix)

    if suffix == "periodicity-select":
        return (
            f"Sets the data frequency for {workflow} calculations. "
            "Daily (Trading) keeps calculations on the trading-day calendar, while Weekly and Monthly aggregate returns to longer period lengths with lower noise. "
            "All tabs are calculated based on this period length."
        )
    if suffix == "start-date-picker":
        return (
            f"Sets the first date included in {workflow} calculations. "
            "Observations before this date are excluded from statistics, model windows, and exports. "
            "Common Range, Common Daily, and Max Range shortcuts can reset this boundary from selected-series coverage."
        )
    if suffix == "end-date-picker":
        return (
            f"Sets the last date included in {workflow} calculations. "
            "Observations after this date are excluded from statistics, model windows, and exports. "
            "Common Range, Common Daily, and Max Range shortcuts can reset this boundary from selected-series coverage."
        )
    if suffix == "common-range-button":
        return (
            "Sets the date range to the overlap where all selected series have data at the selected periodicity. "
            "Dates outside that shared overlap are excluded so every selected series is present across the full window. "
            "Use this when you want aligned like-for-like comparisons across selected series."
        )
    if suffix == "common-daily-button":
        return (
            "Sets the date range to the shared daily overlap across the selected series where daily trading data exists for all of them. "
            "This button also switches periodicity to Daily (Trading) so calculations use that daily overlap window. "
            "Use it to quickly align analysis on a common daily-trading sample."
        )
    if suffix == "maximum-range-button":
        return (
            "Sets the date range to the full available span for the selected series at the current periodicity. "
            "This uses the earliest available start and latest available end from the resampled data so you can see the broadest history. "
            "Use this when you want maximum coverage instead of strict overlap."
        )
    if suffix == "vol-scaler-input":
        if prefix == "po":
            return (
                "Sets the annualized volatility target used before optimization and downstream portfolio analytics. "
                "Set to 0 to disable scaling and preserve native asset volatility, or use a positive target to normalize magnitudes across selected assets. "
                "Series-level Scale Vol assignments determine which assets receive this adjustment."
            )
        if prefix == "reg":
            return (
                "Sets the annualized volatility target applied to regression input series before fitting. "
                "Set to 0 to disable scaling and use raw returns, or use a positive target to normalize predictor and target magnitudes across series. "
                "This helps compare coefficient behavior across windows when volatility dispersion is large."
            )
        return (
            "Sets the annualized volatility target used to scale selected Analytics series. "
            "Set to 0 to disable scaling and keep native return magnitudes, or use a positive target to standardize high- and low-volatility series before metrics are computed. "
            "Only series flagged for scaling in the selection modal are adjusted."
        )
    if suffix == "robust-se-switch":
        return (
            "Enables robust standard errors for regression inference. "
            "This does not change fitted coefficients, but it recalculates t-stats, p-values, and confidence intervals under heteroskedasticity-robust assumptions. "
            "Use robust standard errors when residual variance may be non-constant across time."
        )
    if suffix == "returns-type-select":
        return (
            "Chooses return interpretation for Analytics metrics. "
            "Total uses each series as provided, while Excess subtracts the mapped benchmark return before risk and performance statistics are computed. "
            "Select Excess only when benchmark mappings are complete and intentionally defined."
        )
    if suffix == "rolling-metric-select":
        if prefix == "at":
            return (
                "Selects the rolling statistic displayed in the Analytics rolling view. "
                "Options are Total Return, Volatility, Sharpe Ratio, Sortino Ratio, Excess Return, Tracking Error, Information Ratio, and Correlation; benchmark-relative options require benchmark mappings. "
                "Choose one metric and keep it fixed when comparing alternate date ranges."
            )
        return (
            f"Selects the rolling statistic displayed in {workflow}. "
            "Options are Total Return, Volatility, Sharpe Ratio, and Sortino Ratio, which move from absolute return to risk-adjusted quality measures. "
            "Keep the selected metric consistent when comparing scenarios."
        )
    if suffix == "rolling-window-select":
        return (
            "Sets rolling horizon length used by windowed statistics. "
            "Available windows are 3-month, 6-month, 1-year, 3-year, 5-year, and 10-year; shorter windows react quickly while longer windows smooth noise and lag turning points. "
            "Pick a window that matches the decision horizon being evaluated."
        )
    if suffix == "rolling-return-type-select":
        return (
            "Chooses how rolling return outputs are expressed. "
            "Cumulative reports total compounded return over each window, while Annualized scales each window to a per-year rate for comparability across horizons. "
            "Use Annualized for cross-window comparisons and Cumulative for path interpretation."
        )
    if suffix in {
        "rolling-chart-switch",
        "rolling-summary-chart-switch",
        "growth-chart-switch",
        "drawdown-chart-switch",
        "weight-chart-switch",
        "weights-chart-switch",
        "attribution-chart-switch",
        "risk-chart-switch",
        "turnover-chart-switch",
        "frontier-chart-switch",
    }:
        topic = suffix.replace("-chart-switch", "").replace("-", " ")
        return _table_chart_toggle_text(topic, workflow)
    if suffix == "calendar-view-select":
        return (
            "Switches calendar outputs between Annual and Monthly layouts. "
            "Annual summarizes each year in a compact matrix, while Monthly expands into month-level rows and typically requires a selected series. "
            "Use Monthly for seasonality diagnostics and Annual for quick multi-year comparison."
        )
    if suffix == "calendar-series-select":
        return (
            "Selects which series populates the monthly calendar view. "
            "Only one series is shown at a time in monthly mode so seasonal patterns are readable without cross-series overlap. "
            "Change this selection to compare calendar behavior across assets."
        )
    if suffix == "monthly-view-checkbox":
        return (
            "Switches Analytics calendar mode between Annual and Monthly. "
            "Annual provides year-level summary cells, while Monthly expands into month-by-month values for one selected series. "
            "Use Monthly when you need detailed seasonality and Annual for higher-level review."
        )
    if suffix == "correlation-view-switch":
        return (
            "Chooses dependence view type for the correlation tab. "
            "Correlation shows standardized co-movement, Covariance shows co-movement in native units, and Correlogram highlights block structure for pattern scanning. "
            "Select the mode that matches whether scale-free comparison or raw magnitude interpretation is needed."
        )
    if suffix == "factor-mode-select":
        return (
            "Chooses factor analysis visualization mode. "
            "Box Plot groups returns by factor quantiles for distribution comparison, while Scatter plots factor values against returns to show continuous relationships. "
            "Switch modes based on whether you need bucket diagnostics or pointwise signal shape."
        )
    if suffix == "factor-transform-select":
        return (
            "Chooses preprocessing transform for factor values. "
            "Raw keeps original values, while Z-Score standardizes values to mean-zero and unit-variance for comparable scaling across factors. "
            "Use Z-Score when comparing factors with different natural units."
        )
    if suffix == "factor-def-select":
        return (
            "Loads an existing factor definition into the editor. "
            "Option labels tagged [DB] come from persistent database records, [Session] come from local browser session definitions, and [Raw] indicates direct raw-series shortcuts. "
            "Choose the source intentionally because it controls persistence and provenance of the loaded definition."
        )
    if suffix == "regime-def-select":
        return (
            "Loads an existing regime definition into the editor. "
            "Options tagged [DB] are persisted definitions and [Session] are local in-memory drafts, so the tag tells you whether the definition is durable or session-scoped. "
            "Select the definition that matches the regime logic you want applied."
        )
    if suffix in {"factor-def-long-agg-type", "factor-def-short-agg-type"}:
        return (
            "Chooses aggregation method for selected factor components. "
            "Options are 1 COMPOUND_RETURN, 2 LAST_VALUE, 3 PERIOD_MEAN, 4 ANNUALIZED_VOL, 5 MTH_INTERP, 6 QTR_INTERP, 7 RETURN_FROM_LEVELS, 8 LAST_VALUE_DIV_100, 9 PERIOD_MEAN_DIV_100, 10 MTH_INTERP_DIV_100, and 11 QTR_INTERP_DIV_100. "
            "Select the method that matches source semantics so factor values remain interpretable."
        )
    if suffix == "factor-def-output-transform":
        return (
            "Chooses final transform applied to the composed factor series. "
            "0 NONE keeps values unchanged, 1 PCT_CHANGE converts to percent changes, and 2 SIMPLE_DIFF applies first differences. "
            "Pick the transform that aligns factor scale with the analysis you plan to run."
        )
    if suffix == "regime-def-method-type":
        return (
            "Chooses the regime assignment algorithm. "
            "1 HMM on PC1 fits hidden states on the first principal component, 2 Quantiles on PC1 assigns states by PC1 quantiles, and 3 Quantiles on Single Series uses quantiles of one selected series. "
            "Method choice directly changes state labels and transitions."
        )
    if suffix == "raw-db-add-table-select":
        return (
            "Selects source table frequency for raw Funds and Performance imports. "
            "Daily reads the daily table and Monthly reads the monthly table, which changes sample density and annualization context for those two modes. "
            "This control is disabled in Factor mode because factor imports do not use table selection."
        )
    if suffix == "raw-db-add-fee-select":
        return (
            "Selects Gross or Net return flavor for raw Funds and Performance imports. "
            "Gross and Net can materially change reported return levels, so keep fee basis consistent when comparing imported series. "
            "This control is disabled in Factor mode because factor imports do not use fee selection."
        )
    if suffix == "opt-model-select":
        return (
            "Chooses optimization model family. "
            "Options are Risk Parity, Factor Risk Parity, Hierarchical RP, Maximize Sharpe Ratio, Minimize Variance, Minimize CVaR, Equal Weight, Ex Ante Mean-Variance, and Black-Litterman. "
            "Use the Help menu in this module for model definitions, assumptions, and selection guidance."
        )
    if suffix == "objective-select":
        return (
            "Sets optimization objective where objective selection is available. "
            "Maximize Sharpe targets return per unit risk, Minimize Variance targets lowest variance, and Maximize Return pushes expected return subject to constraints. "
            "Pick the objective that matches mandate before reviewing allocations."
        )
    if suffix == "ex-ante-mode-select":
        return (
            "Chooses ex-ante input schema for optimization. "
            "Covariance mode uses expected returns plus covariance, while Vol / Correlation mode uses expected returns, volatilities, and a correlation matrix. "
            "Select the mode that matches how your assumptions are sourced."
        )
    if suffix == "opt-window-select":
        return (
            "Sets optimization window mode. "
            "Rolling applies a fixed lookback that slides forward, Expanding grows from the start date, and Full runs once on the full sample. "
            "Choose the mode that matches whether you need time-varying or one-shot allocations."
        )
    if suffix == "opt-step-input":
        return (
            "Sets how far the estimation window advances between recalculations. "
            "With unit Months, a value of 1 advances one calendar month; with unit Periods, a value of 1 advances one observation at the current frequency. "
            "Larger step values reduce rebalance frequency and the number of generated windows."
        )
    if suffix == "exp-wt-cov-switch":
        return (
            "Enables exponential weighting for covariance estimation in optimization. "
            "When enabled, recent observations receive more weight in covariance estimates; when disabled, covariance uses equal weighting over the selected window. "
            "Set the Half-Life control to define the decay speed when this switch is on."
        )
    if suffix == "cov-shrinkage-select":
        return (
            "Chooses covariance shrinkage for optimization when exponential weighting is off. "
            "Ledoit-Wolf and OAS can stabilize noisy covariance estimates, and downstream frontier and risk views use the same selected estimator."
        )
    if suffix == "correlation-exp-wt-switch":
        return (
            "Enables exponential weighting for correlation or covariance estimates in the Analytics dependence view. "
            "When enabled, recent observations are weighted more heavily; when disabled, all observations in the window are weighted equally. "
            "Use the correlation half-life input to control decay speed when weighting is enabled."
        )
    if suffix == "correlation-shrinkage-select":
        return (
            "Chooses covariance shrinkage for Analytics matrix views when exponential weighting is off. "
            "Correlation is derived from the estimated covariance matrix, so shrinkage affects both covariance and correlation heatmaps."
        )
    if suffix == "fill-in-sample-select":
        return (
            "Controls whether in-sample windows are backfilled in windowed runs. "
            "Off keeps strictly out-of-sample application windows, while On fills in-sample periods for continuous historical series display. "
            "Use Off for strict evaluation and On for fuller visual continuity."
        )
    if suffix == "opt-step-unit-select":
        return (
            "Chooses units for the window step interval. "
            "Months advances by calendar months, while Periods advances by observation periods at current frequency, which may differ after resampling. "
            "Pick the unit that matches rebalance cadence assumptions."
        )
    if suffix == "missing-data-select":
        return (
            "Chooses missing-data fill behavior before modeling. "
            "Fill NA uses gap-filling behavior to preserve continuity, while Fill 0 treats missing values as zero contribution and can materially alter statistics. "
            "Select the least distortive rule for your data source."
        )
    if suffix in {"add-constraint-btn", "clear-constraints-btn"}:
        if suffix == "add-constraint-btn":
            return (
                "Adds one blank linear-constraint row to the constraints grid. "
                "Each row defines Min <= weighted sum of selected series coefficients <= Max for the current model run. "
                "Use this to add portfolio or beta exposure rules that are not covered by per-series bounds."
            )
        return (
            "Removes all linear-constraint rows from the constraints grid. "
            "After clearing, no custom linear Min/Max aggregate constraints are applied in the next run. "
            "Use this when you want to reset feasibility rules and rebuild constraints from scratch."
        )
    if suffix == "window-size-input":
        return (
            "Sets lookback length used by rolling or expanding estimation windows. "
            "Larger values increase historical depth and smooth estimates, while smaller values react faster but are noisier. "
            "This value works with Window Type and Opt Step to define recalculation cadence."
        )
    if suffix == "halflife-input":
        return (
            "Sets decay speed for exponential weighting when exp-weight switches are enabled. "
            "Lower half-life values emphasize recent observations more strongly, while higher values keep more historical influence. "
            "This directly affects estimated moments used by optimization or regression."
        )
    if suffix == "correlation-halflife-input":
        return (
            "Sets decay speed for exponentially weighted correlation or covariance estimates in Analytics. "
            "Lower values emphasize recent co-movement and higher values smooth estimates over longer history. "
            "This control is used only when correlation exponential weighting is enabled."
        )
    if suffix == "exp-wt-switch":
        return (
            "Enables exponential weighting for regression estimation. "
            "When enabled, recent observations receive higher weight during coefficient fitting; when disabled, observations are equally weighted over the window. "
            "Half-Life controls the decay profile when this switch is on."
        )
    if suffix == "alpha-input":
        return (
            "Sets regularization strength for Ridge, Lasso, and Elastic Net regression models. "
            "Higher alpha values increase shrinkage and can improve stability at the cost of bias, while lower values stay closer to OLS behavior. "
            "This input is ignored by model families that do not use penalization."
        )
    if suffix == "l1-ratio-input":
        return (
            "Sets L1 versus L2 penalty mix for Elastic Net regression. "
            "Values near 1 lean toward Lasso-like sparsity, while values near 0 lean toward Ridge-like shrinkage. "
            "This control is used only when Elastic Net is selected."
        )
    if suffix == "force-zero-intercept-switch":
        return (
            "Forces the regression intercept term to zero during fitting. "
            "This changes coefficient estimates and residual behavior compared with an unconstrained intercept model. "
            "Use it only when your model specification requires a no-intercept assumption."
        )
    if suffix == "arima-p-input":
        return (
            "Sets the ARIMA autoregressive order, p. "
            "This is the number of lagged residual terms included in the ARIMA model, so larger values let the fit absorb longer autocorrelation patterns in regression residuals. "
            "Increase p only when residual autocorrelation remains after lower-order fits."
        )
    if suffix == "arima-d-input":
        return (
            "Sets the ARIMA differencing order, d. "
            "This is the number of times the residual series is differenced before fitting the AR and MA terms, which helps remove residual trend or other non-stationary level behavior. "
            "In most regression-residual use cases this stays low, often 0 or 1."
        )
    if suffix == "arima-q-input":
        return (
            "Sets the ARIMA moving-average order, q. "
            "This is the number of lagged forecast-error terms included in the model, so it controls how short-run shocks carry through the residual process. "
            "Raise q only when residual diagnostics suggest moving-average structure beyond lower-order fits."
        )
    if suffix == "garch-p-input":
        return (
            "Sets the GARCH ARCH order, p. "
            "This is the number of lagged squared residual terms used in the conditional-volatility equation, so it controls how strongly recent shocks feed into current volatility. "
            "Higher p can fit richer shock-memory patterns but usually needs more data to estimate reliably."
        )
    if suffix == "garch-q-input":
        return (
            "Sets the GARCH order, q. "
            "This is the number of lagged conditional-variance terms used in the volatility equation, so it controls volatility persistence from one period to the next. "
            "Higher q increases persistence flexibility but can become unstable on short samples."
        )
    if suffix == "result-select":
        return (
            "Selects which saved regression result snapshot is displayed across regression output tabs. "
            "Changing this selection updates coefficients, diagnostics, ANOVA, rolling summary, and plots to the chosen run. "
            "Use this to compare previously saved regression runs without recomputing."
        )
    if suffix == "delete-result-btn":
        return (
            "Deletes the currently selected saved regression result from the result list. "
            "After deletion, that snapshot is removed from selectors and no longer available for tab displays or exports. "
            "This does not delete source input data used to run the regression."
        )
    if suffix == "anova-window-select":
        return (
            "Selects which rolling window snapshot is shown in ANOVA and parameter detail outputs. "
            "Each window corresponds to one fitted period from the saved regression result when rolling or expanding mode is used. "
            "Use this selector to inspect diagnostics at a specific point in time."
        )
    if suffix == "scatter-x-select":
        return (
            "Selects which X series to plot on the horizontal axis for scatter diagnostics that require an X choice. "
            "This control is active for scatter modes that compare actual or predicted values against one explanatory variable. "
            "Switch X selections to compare relationship shape across regressors."
        )
    if suffix == "run-button":
        return (
            "Runs regression using the current model settings, selected series roles, constraints, and date window. "
            "A successful run creates or updates a result snapshot that drives all regression output tabs and exports. "
            "Use this after finalizing Y/X assignments and model controls."
        )
    if suffix == "weight-portfolio-select":
        return (
            "Selects which saved optimization result portfolio is displayed across optimization output tabs. "
            "This choice controls the active weights, risk, attribution, turnover, and frontier-linked views currently shown. "
            "Switch portfolios here to compare saved optimization outcomes."
        )
    if suffix == "delete-portfolio-button":
        return (
            "Deletes the currently selected saved optimization result portfolio. "
            "After deletion, that portfolio is removed from output selectors and cannot be restored without rerunning optimization. "
            "This does not remove the underlying imported return-series dataset."
        )
    if suffix == "frontier-window-select":
        return (
            "Selects which optimization window snapshot is used for efficient-frontier calculations. "
            "Each option corresponds to one historical estimation window from the selected portfolio result. "
            "Use this to compare frontier shape and points across different rebalance windows."
        )
    if suffix in {"load-db-returns-btn", "load-db-matrix-btn"}:
        target = "expected returns" if suffix == "load-db-returns-btn" else "covariance/vol-correlation matrix"
        return (
            f"Loads {target} assumptions from the selected CMA source into ex-ante inputs. "
            "Loaded values overwrite corresponding entries in the current ex-ante grids for the active asset universe. "
            "Use this to initialize assumptions from database CMA versions before manual adjustments."
        )
    if suffix in {"estimate-returns-btn", "estimate-matrix-btn"}:
        target = "expected returns" if suffix == "estimate-returns-btn" else "covariance/vol-correlation matrix"
        return (
            f"Estimates {target} from the active estimation window and writes results to ex-ante grids. "
            "Estimation uses current periodicity, window controls, missing-data method, and weighting settings. "
            "Use this to seed ex-ante assumptions from history before optional edits."
        )
    if suffix in {"ex-ante-returns-upload", "ex-ante-matrix-upload"}:
        target = "ex-ante returns grid" if suffix == "ex-ante-returns-upload" else "ex-ante matrix grid"
        return (
            f"Uploads a file into the {target}. "
            "The file is parsed and mapped to currently selected assets, then loaded into editable assumption cells. "
            "Use this when assumptions are prepared externally and should be imported directly."
        )
    if suffix == "load-session-upload":
        return (
            "Uploads a saved session JSON file for this module. "
            "Loaded session state restores control values and stored data snapshots used by the page. "
            "Use this to reopen a previously saved working configuration."
        )
    if suffix == "download-sample-daily-btn":
        return (
            "Downloads the sample daily input file template. "
            "This template shows expected date and series-column format for daily imports. "
            "Use it as a structure reference before uploading your own daily file."
        )
    if suffix == "download-sample-monthly-btn":
        return (
            "Downloads the sample monthly input file template. "
            "This template shows expected date and series-column format for monthly imports. "
            "Use it as a structure reference before uploading your own monthly file."
        )
    if suffix == "factor-def-new-btn":
        return (
            "Starts a new factor-definition draft in the editor. "
            "Current unsaved field values are replaced by a clean draft state for a new definition. "
            "Use this before entering name, components, transforms, and save target."
        )
    if suffix == "factor-def-save-local-btn":
        return (
            "Saves the current factor-definition draft to Session storage. "
            "Session-saved definitions are available in this browser session and appear with [Session] tags in selectors. "
            "Use this for local drafts that do not need database persistence."
        )
    if suffix == "factor-def-save-db-btn":
        return (
            "Saves the current factor-definition draft to the shared database. "
            "Database-saved definitions appear with [DB] tags and are available across sessions and users with access. "
            "Use this when the definition should be durable and reusable."
        )
    if suffix == "factor-def-delete-btn":
        return (
            "Deletes the currently loaded factor definition from its source store. "
            "Deleted definitions are removed from corresponding selectors and cannot be selected until recreated. "
            "Use this when a saved definition is obsolete or incorrect."
        )
    if suffix == "factor-def-use-btn":
        return (
            "Applies the current factor-definition draft to Analytics factor selectors. "
            "After applying, the definition is available for factor analysis selection in this session context. "
            "Use this to test a draft without leaving the editor workflow."
        )
    if suffix == "factor-def-close-btn":
        return (
            "Closes the factor-definition editor modal. "
            "Closing leaves existing saved definitions unchanged and exits the editor view. "
            "Use this after finishing factor-definition edits."
        )
    if suffix == "factor-quantiles-input":
        return (
            "Sets the number of quantile buckets used in factor box-plot grouping. "
            "Higher quantile counts provide finer ranking granularity with fewer observations per bucket. "
            "Use a count that balances detail and sample stability."
        )
    if suffix == "regime-def-new-btn":
        return (
            "Starts a new regime-definition draft in the editor. "
            "Current unsaved fields are replaced by a clean draft state for a new definition. "
            "Use this before entering regime method, series inputs, and save target."
        )
    if suffix == "regime-def-save-local-btn":
        return (
            "Saves the current regime-definition draft to Session storage. "
            "Session-saved definitions appear with [Session] tags and remain available in the current browser session. "
            "Use this for local experimentation before database persistence."
        )
    if suffix == "regime-def-save-db-btn":
        return (
            "Saves the current regime-definition draft to the shared database. "
            "Database-saved definitions appear with [DB] tags and are reusable across sessions and users with access. "
            "Use this when the regime definition should be durable."
        )
    if suffix == "regime-def-delete-btn":
        return (
            "Deletes the currently loaded regime definition from its source store. "
            "Deleted definitions are removed from regime selectors and cannot be selected until recreated. "
            "Use this to retire obsolete or invalid regime definitions."
        )
    if suffix == "regime-def-use-btn":
        return (
            "Applies the current regime-definition draft to Analytics regime selectors. "
            "After applying, regime analysis tabs can use this definition for state assignment. "
            "Use this to test a draft definition in analysis outputs."
        )
    if suffix == "regime-def-close-btn":
        return (
            "Closes the regime-definition editor modal. "
            "Closing does not change previously saved definitions and exits the editor view. "
            "Use this after finishing regime-definition edits."
        )
    if suffix == "portfolio-add-row-btn":
        return (
            "Adds the current portfolio-import selections as one staged row in the import grid. "
            "Each row stores portfolio id, return type, and optional benchmark import options based on active mode. "
            "Repeat to stage multiple portfolios before importing the full batch."
        )
    if suffix == "portfolio-clear-rows-btn":
        return (
            "Clears all staged rows from the portfolio-import grid. "
            "After clearing, no portfolio rows remain queued for import in the current modal session. "
            "Use this to reset staging and rebuild the import batch."
        )
    if suffix == "portfolio-delete-row-btn":
        return (
            "Deletes the currently selected staged row from the portfolio-import grid. "
            "Only the selected staged entry is removed; other staged rows remain unchanged. "
            "Use this to fix staging mistakes without clearing the entire batch."
        )
    if suffix == "sheet-select-dropdown":
        return (
            "Selects which worksheet from an uploaded workbook is staged for import preview. "
            "The selected sheet determines the data parsed into the preview and import action. "
            "Use this when workbook files contain multiple candidate sheets."
        )
    if suffix == "sheet-select-import-all-button":
        return (
            "Imports all compatible sheets from the currently uploaded workbook. "
            "Each accepted sheet is parsed and appended to working data using the standard file-import pipeline. "
            "Use this when the workbook is structured with one valid series table per sheet."
        )
    if suffix == "close-completion-button":
        return (
            "Closes the optimization completion/status modal. "
            "Closing the dialog keeps saved optimization results and current page state intact. "
            "Use this to return to the main portfolio outputs after reviewing status text."
        )
    if suffix == "open-series-modal-button":
        return (
            "Opens the Series Selection modal for Analytics. "
            "Use that grid to add or remove series and set row-level flags such as Benchmark, L/S, and Scale Vol. "
            "The saved selection set is then used across Analytics tables, charts, and exports."
        )
    if suffix == "open-modal-button":
        if prefix == "reg":
            return (
                "Opens the Series Selection modal for Regression. "
                "Use that grid to assign Y and X roles, then set per-series lag or coefficient-bound options as needed. "
                "The saved selection set becomes the active universe for regression runs and diagnostics."
            )
        return (
            "Opens the Series Selection modal for Portfolio Optimization. "
            "Use that grid to choose the investable series set and review row-level options such as CMA benchmark mapping and Scale Vol. "
            "The saved selection set becomes the active optimization universe for model runs."
        )
    if suffix == "factor-open-modal-btn":
        return (
            "Opens the factor-definition editor. "
            "Use it to create, update, or delete factor definitions and save them to Session or Database storage. "
            "Applied factor definitions become available in Factor Analysis selectors."
        )
    if suffix == "regime-open-modal-btn":
        return (
            "Opens the regime-definition editor. "
            "Use it to create, update, or delete regime definitions and save them to Session or Database storage. "
            "Applied regime definitions become available in Regime Analysis selectors."
        )
    if suffix == "frontier-rm-select":
        return (
            "Chooses risk measure for frontier generation. "
            "Volatility (MV) uses variance-based risk, while CVaR emphasizes tail-loss behavior and may reorder frontier points materially. "
            "Use the measure aligned with your risk mandate."
        )
    if suffix == "cma-type-select":
        return (
            "Chooses CMA type loaded into ex-ante assumptions. "
            "10-Year selects long-horizon strategic assumptions, while Equilibrium selects equilibrium-style assumptions based on model-implied balance. "
            "This choice changes the expected return and risk inputs used by optimization."
        )
    if suffix == "model-select":
        return (
            "Chooses regression model family. "
            "Options are OLS, Constrained OLS, Style Analysis, Ridge, Lasso, and Elastic Net; regularized models depend on alpha and optionally l1 ratio settings. "
            "Use the Help menu in this module for model definitions, assumptions, and selection guidance."
        )
    if suffix == "window-type-select":
        return (
            "Chooses regression window mode. "
            "Full estimates one model on all data, Expanding increases the sample from start date, and Rolling uses a fixed moving lookback window. "
            "Pick the mode that matches static versus time-varying coefficient analysis."
        )
    if suffix == "portfolio-name-input":
        return (
            "Sets the portfolio result name used when saving optimization output. "
            "This name is used as the series label in downstream selectors, tables, and cross-page comparisons after the result is added to working data. "
            "Changing the name affects labeling and retrieval, not optimization math."
        )
    if suffix == "rolling-summary-detail-switch":
        return (
            "Sets detail depth for rolling regression summary output. "
            "Basic shows compact diagnostics for quick monitoring, while Advanced exposes additional fields useful for troubleshooting and deep review. "
            "Use Advanced when validating model stability in detail."
        )
    if suffix == "monthly-series-select":
        return (
            "Selects which series is shown in the Analytics monthly calendar panel. "
            "Only one series is displayed at a time in monthly view so month-level heatmap values remain readable. "
            "Change this selector to compare calendar seasonality across series."
        )
    if suffix == "bl-tau-input":
        return (
            "Sets the Black-Litterman tau scaling parameter for prior covariance confidence. "
            "Smaller tau values reduce the influence of views relative to the equilibrium prior, while larger values increase view impact. "
            "Use a stable tau convention across scenario comparisons so differences reflect view changes rather than scaling drift."
        )
    if suffix == "ex-ante-matrix-upload-btn":
        return (
            "Opens the file picker used to upload ex-ante matrix assumptions. "
            "Uploaded files populate covariance or vol-correlation matrix cells for the active asset universe after parsing. "
            "Use this when ex-ante matrix assumptions are prepared outside the app."
        )
    if suffix == "scatter-mode-select":
        return (
            "Chooses scatter diagnostic relationship. "
            "Residual vs Predicted checks error structure, Actual vs Predicted checks fit quality, Actual vs X shows raw dependency, and Predicted vs X shows modeled response versus each regressor. "
            "Switch modes based on whether you are diagnosing residuals or interpreting explanatory relationships."
        )
    if suffix == "factor-series-select":
        return (
            "Selects the factor input used by Factor Analysis. "
            "Options tagged [Raw] are direct series from the working dataset, [DB] are saved database factor definitions, and [Session] are definitions saved only in the current browser session. "
            "Pick the source type deliberately because it controls factor construction, persistence, and reproducibility."
        )
    if suffix == "regime-definition-select":
        return (
            "Selects the regime definition used by Regime Analysis. "
            "Options tagged [DB] are persisted shared definitions, while [Session] options are local drafts saved only for the current session. "
            "The selected definition drives state assignment and every regime-conditioned table and chart."
        )
    if suffix == "db-add-series-select":
        return (
            "Select one or more AA Tool index categories to import. "
            "Each selected series adds its returns for analysis, as well as capital market assumptions for ex ante optimization inputs."
        )
    if suffix == "portfolio-add-series-select":
        return (
            "Selects the portfolio to stage in the portfolio-import grid. "
            "The available list depends on the active mode (Peer-relative, Index-relative, or Alternatives), and each staged row can include an optional benchmark companion. "
            "Add one portfolio at a time, then build the batch with the Add Series button."
        )
    if suffix == "portfolio-add-type-select":
        return (
            "Selects the return type to import for the staged portfolio row. "
            "Options vary by mode and typically include Actual and Calculated; in peer workflows, benchmark settings can also target peer mean return behavior for relative comparisons. "
            "Keep return type consistent across rows when you want clean peer, index, or alternative comparisons."
        )
    if suffix == "underlying-add-base-select":
        return (
            "Select whether the source portfolios come from the Core or Base family. "
            "This value is combined with the selected TD, Alloc, 529, or Model types to build Portfolio filters such as CoreTD or Base529. "
            "Change this first because it determines which underlying category descriptions are available in the Desc list."
        )
    if suffix == "underlying-add-type-multiselect":
        return (
            "Select one or more underlying portfolio groups to import from the chosen Core or Base family. "
            "Each selected value is appended to the Base choice to form Portfolio filters such as CoreTD, CoreAlloc, Base529, or BaseModel. "
            "The Desc list is built from the union of matching PeerTS rows, and each selected desc can stage rows across multiple portfolio codes at once."
        )
    if suffix == "underlying-add-desc-multiselect":
        return (
            "Select one or more underlying category descriptions after Base and Type are chosen. "
            "The available list is filtered to PeerTS rows with Item equal to PeerRet whose Portfolio matches the constructed codes, so one desc can stage multiple source portfolios when more than one type is selected. "
            "Imported values are converted from levels to returns before they are appended to the working dataset."
        )
    if suffix == "underlying-add-row-btn":
        return (
            "Stage the selected underlying categories in the grid below. "
            "One row is added for each matching Portfolio and Desc combination, using a final series name like Large Cap [CoreTD], so duplicate descriptions from different portfolio codes remain distinct. "
            "Use the grid to review the batch before importing."
        )
    if suffix == "underlying-delete-row-btn":
        return (
            "Remove the selected staged underlying-category row from the import grid. "
            "This only changes the current staging batch and does not affect series that are already loaded into the dataset. "
            "Use it when you want to keep some desc and portfolio combinations but drop one staged row."
        )
    if suffix == "underlying-clear-rows-btn":
        return (
            "Clear all staged underlying-category rows from the import grid. "
            "This resets the current batch without changing the Base, Type, or Desc selectors above and does not remove already imported series. "
            "Use it when you want to rebuild the staged batch from scratch."
        )
    if suffix == "portfolio-add-benchmark-type-select":
        return (
            "Selects which benchmark flavor is imported when Include Benchmark is enabled. "
            "Depending on mode, options can include Actual, Estimated, Calculated, or Benchmark; each option points to a different benchmark construction in the source process. "
            "Choose the same benchmark type across comparable rows so excess-return and tracking metrics remain consistent."
        )
    if suffix == "cma-version-select":
        return (
            "Selects the CMA version to load into ex-ante optimization inputs. "
            "Each version is a different published assumption set, so choosing a new version can change expected returns, risk estimates, and resulting allocations. "
            "Confirm version and type together before loading because the selection can overwrite manual grid edits."
        )
    if suffix == "upload-data":
        return (
            f"Uploads CSV or Excel return-series files into {workflow}. "
            "Uploaded data is parsed and merged into working state, then becomes available to selectors, grids, and downstream calculations. "
            "Use clean date-indexed files and validate imported series after upload."
        )
    if suffix == "portfolio-add-include-benchmark":
        return (
            "Includes benchmark companion series for each staged portfolio row. "
            "When enabled, benchmark streams are imported using selected benchmark-type rules and become available for excess-return and tracking metrics. "
            "Disable it when the workflow is intentionally absolute-return only."
        )
    if suffix == "raw-db-add-include-benchmark":
        return (
            "Includes benchmark companion data for raw Performance imports. "
            "When enabled, both the primary series and benchmark series are staged so relative metrics can be computed after import. "
            "This control is disabled in Factor and Funds modes."
        )
    if suffix == "raw-db-add-convert-returns":
        return (
            "Controls levels to returns conversion for raw Factor imports. "
            "When enabled, level-type factor values are converted to return increments; when disabled, raw values are kept and Divide By is used for scaling. "
            "This control applies to Factor mode and is hidden for Funds and Performance paths."
        )
    if suffix == "raw-db-add-divide-by":
        return (
            "Sets scaling divisor for raw Factor imports when Convert to Returns is unchecked. "
            "Use 100 for percent-like factor values such as 2.5 meaning 2.5%, or 1 when values are already in desired scale. "
            "This input is disabled when conversion is on and not used in Funds or Performance modes."
        )
    if suffix == "raw-db-add-row-btn":
        return (
            "Adds the current raw-import selection as one staged row in the modal grid. "
            "In Factor mode, the staged row uses factor series plus Convert to Returns and Divide By settings, while Table, Fee, and Include Benchmark are not part of the row. "
            "Repeat this to build a batch before importing all staged rows together."
        )
    if suffix == "raw-db-add-ok-button":
        return (
            "Imports all staged raw rows into the active workflow dataset. "
            "Each row is resolved using its mode-specific controls; for Factor rows that means factor metadata with levels-to-returns and Divide By behavior, not table or fee selection. "
            "Review the staged grid first because this import updates working data immediately."
        )
    if suffix == "raw-db-add-cancel-button":
        return (
            "Closes the raw import modal without committing staged rows. "
            "Any staged factor, funds, or performance rows are discarded and working data remains unchanged. "
            "Use Cancel when you want to revisit mode or series choices before importing."
        )
    if suffix == "raw-db-clear-rows-btn":
        return (
            "Clears every staged row from the raw import grid. "
            "This resets the batch to empty so you can rebuild the import list with different mode and control choices. "
            "Use Clear All when the staged set no longer matches the import you want."
        )
    if suffix == "raw-db-delete-row-btn":
        return (
            "Deletes selected rows from the staged raw import grid. "
            "Removed rows are excluded from the next import commit while remaining rows stay staged. "
            "Use Delete One to fix individual staging mistakes without clearing the full batch."
        )
    if suffix == "raw-db-add-series-select":
        return (
            "Selects the source series to stage for the current raw import mode. "
            "In Factor mode this list is MRD Factor Data, while Funds and Performance modes show their own source universes. "
            "Choose one item, then add it as a staged row with mode-appropriate controls."
        )
    if suffix == "bl-add-view":
        return (
            "Adds a new Black-Litterman view row to the staged views grid. "
            "Each row defines a directional or relative view that influences posterior expected returns during BL blending. "
            "Use one row per distinct view statement you want the model to incorporate."
        )
    if suffix == "bl-clear-views":
        return (
            "Clears all staged Black-Litterman view rows. "
            "Removing all views returns BL behavior to prior-only assumptions unless new views are added before the next run. "
            "Clear views when you want to restart scenario design from the prior only."
        )
    if suffix == "cma-load-confirm":
        return (
            "Confirms loading of selected CMA assumptions into ex-ante inputs. "
            "This writes chosen CMA version and type values into current assumption grids and can overwrite manual edits. "
            "Use confirm only after validating the intended CMA source."
        )
    if suffix == "cma-load-cancel":
        return (
            "Cancels CMA loading and closes the modal without changing current assumptions. "
            "Existing ex-ante values remain active, including any manual edits already present in grids. "
            "Choose Cancel when the selected CMA version or type is not the intended source."
        )
    if suffix == "ex-ante-returns-clear":
        return (
            "Clears all entries from the ex-ante returns grid. "
            "This removes current expected-return assumptions so you can reload from database, estimate again, or enter values manually. "
            "Clear before switching methodology so manual and loaded assumptions do not get mixed."
        )
    if suffix == "ex-ante-matrix-clear":
        return (
            "Clears all entries from ex-ante matrix grids. "
            "This resets covariance or vol/correlation assumptions so a fresh source can be loaded or estimated. "
            "Clear first when transitioning between database, estimated, and uploaded matrix assumptions."
        )
    if suffix == "growth-portfolio-multiselect":
        return (
            "Selects additional portfolios to overlay in growth comparison charts. "
            "Chosen portfolios are plotted alongside the primary selection so cumulative-path behavior can be compared in one view. "
            "This lets you compare scenario outcomes without changing the primary portfolio selection."
        )
    if suffix == "correlogram-block-width":
        return (
            "Sets block width used in correlogram rendering. "
            "Smaller widths show finer detail with more visual noise, while larger widths emphasize broader dependence structure. "
            "Adjust block size based on series count and desired structural granularity."
        )
    if suffix == "factor-def-long-components":
        return (
            "Selects source series included in the long component basket of the factor definition. "
            "Selected series are aggregated using the configured long aggregation method before any output transform is applied. "
            "Choose components that represent the intended long exposure."
        )
    if suffix == "factor-def-short-components":
        return (
            "Selects source series included in the short component basket of the factor definition. "
            "When populated, the short aggregate is subtracted from the long aggregate to form a long-short factor stream. "
            "Leave empty when creating a long-only factor."
        )
    if suffix == "factor-def-long-lag":
        return (
            "Sets an integer lag applied to the long-side factor aggregate. "
            "Positive lag shifts the long signal backward in time and can enforce causal timing in predictive workflows. "
            "Adjust lag deliberately because it changes alignment between factor and return series."
        )
    if suffix == "regime-def-num-regimes":
        return (
            "Sets the number of regime states generated by the selected method. "
            "More regimes increase segmentation detail but require more observations per state to stay stable and interpretable. "
            "Choose a count that balances granularity with data sufficiency."
        )
    if suffix == "regime-def-min-observations":
        return (
            "Sets the minimum observations threshold used when validating regime assignments. "
            "Higher thresholds reduce unstable state estimates but can reject sparse configurations with many regimes. "
            "Tune this together with method and regime count."
        )
    if suffix == "regime-def-pca-standardize":
        return (
            "Controls standardization of inputs before PCA in PC1-based regime methods. "
            "Enable standardization to equalize scale across series so high-volatility inputs do not dominate component loadings. "
            "Disable only when native scale differences are intentionally part of regime logic."
        )
    if suffix == "regime-def-single-series":
        return (
            "Selects the source series used by the single-series quantile regime method. "
            "Regime boundaries are determined directly from this series, so selection defines the economic meaning of states. "
            "Choose a series with stable coverage and clear interpretability."
        )
    if suffix == "regime-def-universe-series":
        return (
            "Selects the multi-series universe used for PC1-based regime methods. "
            "These series feed the PCA stage and therefore strongly influence state assignment and transition behavior. "
            "Use representative series with adequate history for robust regimes."
        )
    if suffix == "regime-def-vol-scaler":
        return (
            "Sets volatility scaling target applied before regime-state construction. "
            "Set to 0 to disable scaling, or use a positive target to reduce dominance by high-volatility inputs. "
            "Keep this setting consistent across regime-definition comparisons."
        )
    if suffix == "welcome-add-db-btn":
        return (
            "Import returns and CMA data for indices used in the AA Tool. "
            "Select one or more index categories and append them to the current dataset so they are available in selectors, benchmarks, and model inputs. "
            "This is the standard AA Tool index import path."
        )
    if suffix == "welcome-add-series-btn":
        return (
            "Import return series from a local CSV or Excel file. "
            "Format files like the sample daily or sample monthly files shown below on the welcome card so dates and return columns load cleanly. "
            "This import also accepts Morningstar Performance Reporting exports."
        )
    if suffix == "welcome-add-portfolios-peer-btn":
        return (
            "Import portfolio and AA return streams from the peer-relative process. "
            "You can include the peer mean return as a benchmark companion, which enables peer-relative excess-return and tracking metrics right after import. "
            "Stage the portfolios you need, then append them to the working dataset in one batch."
        )
    if suffix == "welcome-add-portfolios-index-btn":
        return (
            "Import portfolio and AA return streams from the index-relative process. "
            "Benchmark companions follow the index mapping for each portfolio, so relative metrics are ready immediately after import. "
            "Stage the selected portfolios and append them to the working dataset."
        )
    if suffix == "welcome-add-portfolios-other-btn":
        return (
            "Import alternative portfolio return streams from the alternative workflow. "
            "This mode supports the alternative source set and can import each portfolio's benchmark companion for relative analytics. "
            "Choose this when the portfolios you need are maintained in the alternatives universe."
        )
    if suffix == "welcome-add-portfolios-underlying-btn":
        return (
            "Import underlying peer-category series from PeerTS. "
            "Choose Core or Base, select one or more TD, Alloc, 529, or Model groups, and then pick the underlying category descriptions available for those combinations. "
            "PeerTS values are stored as levels, so this import converts them to returns before appending the staged series to the dataset."
        )
    if suffix == "welcome-add-raw-factor-btn":
        return (
            "Add one or more series from MRD Factor Data. "
            "Factor mode supports converting level series to returns, or keeping raw values and scaling percent-style values with Divide By (for example, divide by 100). "
            "Table, Fee, and Include Benchmark controls are disabled in this mode."
        )
    if suffix == "welcome-add-raw-funds-btn":
        return (
            "Add one or more fund return series from raw funds data. "
            "For each staged row, choose table frequency and fee basis, then append the selected fund series to the working dataset. "
            "Benchmark and factor-conversion controls are disabled in this mode."
        )
    if suffix == "welcome-add-raw-performance-btn":
        return (
            "Add one or more performance return series from raw performance data. "
            "For each staged row, choose table frequency and fee basis, and optionally include benchmark companions for relative analysis. "
            "Importing the batch appends those performance series to the working dataset."
        )
    if suffix == "welcome-view-analytics":
        return (
            "Switch to the analytics module. "
            "Use this module to analyze return series with statistics, rolling views, calendar views, and factor or regime diagnostics. "
            "Move here when you want to inspect behavior and relationships in the loaded dataset."
        )
    if suffix == "welcome-view-portfolio":
        return (
            "Switch to the portfolio optimization module. "
            "Use this module to build and compare optimized portfolios, then review weights, risk, turnover, attribution, and frontier results. "
            "Move here when you are ready to construct allocation outputs from the loaded series."
        )
    if suffix == "welcome-view-regression":
        return (
            "Switch to the regression module. "
            "Use this module to fit return relationships, inspect coefficients and diagnostics, and review rolling regression behavior. "
            "Move here when you want explanatory modeling instead of descriptive analytics."
        )

    if suffix.endswith(("-btn", "-button")):
        return _suffix_fallback_text(prefix, suffix, workflow)
    if suffix.endswith("-select") or suffix.endswith("-dropdown"):
        return _suffix_fallback_text(prefix, suffix, workflow)
    if suffix.endswith("-input"):
        return _suffix_fallback_text(prefix, suffix, workflow)
    if suffix.endswith("-switch") or suffix.endswith("-checkbox"):
        return _suffix_fallback_text(prefix, suffix, workflow)
    if suffix.endswith("-upload"):
        return _suffix_fallback_text(prefix, suffix, workflow)
    if suffix.endswith("-date-picker"):
        return _suffix_fallback_text(prefix, suffix, workflow)

    return _default_explicit_tooltip(f"{prefix}-{suffix}")


def _build_explicit_tooltips() -> dict[str, str]:
    tips: dict[str, str] = {}

    for suffix, prefixes in _RENDERED_SUFFIX_PREFIXES.items():
        for prefix in prefixes:
            control_id = f"{prefix}-{suffix}"
            text = _suffix_tooltip_override(prefix, suffix)
            if not text:
                raise ValueError(f"Missing explicit tooltip mapping for rendered control: {control_id}")
            tips[control_id] = text

    for prefix, workflow in _WORKFLOW_BY_PREFIX.items():
        tips[f"{prefix}-menu-download-excel"] = (
            f"Exports the current {workflow} outputs to an excel workbook. "
            "The workbook captures active selections, model choices, and view settings at export time so the file reflects on-screen assumptions. "
            "Review controls before export to keep reporting artifacts consistent."
        )
        tips[f"{prefix}-menu-load-session"] = (
            f"Loads a saved {workflow} session payload from local file. "
            "Load restores control values and store state so the page returns to the previously saved analytical context. "
            "Use session files for reproducibility and collaborative handoff."
        )
        tips[f"{prefix}-menu-save-session"] = (
            f"Saves current {workflow} state to a local session file. "
            "The save includes key controls and stores so the same setup can be restored later without manual re-entry. "
            "Save before major edits or when sharing scenario configurations."
        )
        tips[f"{prefix}-menu-clear-local-storage"] = (
            f"Clears browser-local cached state for {workflow}. "
            "This removes persisted client-side values that can conflict with fresh runs after repeated experiments. "
            "Clear local storage when you need a clean client-side reset."
        )
        tips[f"{prefix}-menu-exit"] = (
            f"Navigates away from {workflow} to the landing route. "
            "This leaves the current workflow screen and is typically used when finishing work in this module. "
            "Save or export first if you need to preserve current state."
        )
        tips[f"{prefix}-menu-add-from-db"] = (
            "Import returns and CMA data for indices used in the AA Tool. "
            "Select one or more index categories and append them to the current dataset so they can be used in selectors, benchmarks, and model inputs. "
            "This is the standard AA Tool index import path."
        )
        tips[f"{prefix}-menu-add-portfolios-peer"] = (
            "Import portfolio and AA return streams from the peer-relative process. "
            "You can include the peer mean return as a benchmark companion so peer-relative metrics are available immediately after import. "
            "Stage the portfolios you need and append them to the working dataset in one batch."
        )
        tips[f"{prefix}-menu-add-portfolios-index"] = (
            "Import portfolio and AA return streams from the index-relative process. "
            "Benchmark companions follow each portfolio's index mapping so relative metrics are ready after import. "
            "Stage the selected portfolios and append them to the working dataset."
        )
        tips[f"{prefix}-menu-add-portfolios-other"] = (
            "Import alternative portfolio return streams from the alternative workflow. "
            "This mode supports the alternative source set and can import each portfolio's benchmark companion for relative analytics. "
            "Choose this when the portfolios you need are maintained in the alternatives universe."
        )
        tips[f"{prefix}-menu-add-portfolios-underlying"] = (
            "Import underlying peer-category series from PeerTS. "
            "Choose Core or Base, select one or more TD, Alloc, 529, or Model groups, and then pick the underlying category descriptions available for those combinations. "
            "PeerTS values are stored as levels, so this import converts them to returns before appending the staged series to the dataset."
        )
        tips[f"{prefix}-menu-add-raw-factor"] = (
            "Add one or more series from MRD Factor Data. "
            "Factor mode supports converting level series to returns, or keeping raw values and scaling percent-style values with Divide By (for example, divide by 100). "
            "Table, Fee, and Include Benchmark controls are disabled in this mode."
        )
        tips[f"{prefix}-menu-add-raw-funds"] = (
            "Add one or more fund return series from raw funds data. "
            "For each staged row, choose table frequency and fee basis, then append the selected fund series to the working dataset. "
            "Benchmark and factor-conversion controls are disabled in this mode."
        )
        tips[f"{prefix}-menu-add-raw-performance"] = (
            "Add one or more performance return series from raw performance data. "
            "For each staged row, choose table frequency and fee basis, and optionally include benchmark companions for relative analysis. "
            "Importing the batch appends those performance series to the working dataset."
        )
        tips[f"{prefix}-menu-add-series"] = (
            "Import return series from a local CSV or Excel file. "
            "Format files like the sample daily or sample monthly files shown below on the welcome card so dates and return columns load cleanly. "
            "This import also accepts Morningstar Performance Reporting exports."
        )
        tips[f"{prefix}-menu-clear-server-cache"] = (
            f"Requests server-side cache clear for {workflow}. "
            "Clearing cache removes memoized artifacts so subsequent callbacks recompute from current inputs. "
            "Run this when stale server cache behavior is suspected."
        )
        tips[f"{prefix}-menu-help-guide"] = (
            f"Shows the {workflow} quick help guide. "
            "The guide summarizes control behavior, expected inputs, and common troubleshooting patterns for this page. "
            "Use it to validate workflow steps or onboard new users."
        )
        tips[f"{prefix}-menu-view-analytics"] = (
            "Switch to the analytics module. "
            "Use this module to analyze return series with statistics, rolling views, calendar views, and factor or regime diagnostics. "
            "Move here when you want to inspect behavior and relationships in the loaded dataset."
        )
        tips[f"{prefix}-menu-view-portfolio"] = (
            "Switch to the portfolio optimization module. "
            "Use this module to build and compare optimized portfolios, then review weights, risk, turnover, attribution, and frontier results. "
            "Move here when you are ready to construct allocation outputs from the loaded series."
        )
        tips[f"{prefix}-menu-view-regression"] = (
            "Switch to the regression module. "
            "Use this module to fit return relationships, inspect coefficients and diagnostics, and review rolling regression behavior. "
            "Move here when you want explanatory modeling instead of descriptive analytics."
        )

    tips["at-menu-add-factor"] = (
        "Edit Analytics factor definitions. "
        "This editor supports creating, updating, and applying reusable factor recipes used by Factor Analysis controls. "
        "Save to session for local drafts or save to database for shared persistent definitions."
    )
    tips["at-menu-add-regime"] = (
        "Edit Analytics regime definitions. "
        "Regime definitions control state assignment logic used by regime-conditioned metrics and charts. "
        "Save to session for local drafts or save to database for shared persistent definitions."
    )
    tips["po-run-button"] = (
        "Runs portfolio optimization with current model, constraints, and ex-ante assumptions. "
        "Execution reads active store values exactly as configured, so stale settings can propagate directly into resulting allocations and diagnostics. "
        "Review model and assumption panels before running."
    )

    return tips


_EXPLICIT_TOOLTIPS.update(_build_explicit_tooltips())


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
    (re.compile(r"select|dropdown"), "Select from the available options."),
    (re.compile(r"input"), "Enter a value for this setting."),
    (re.compile(r"switch|toggle"), "Enable or disable this behavior."),
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


_SUPPRESSED_TOOLTIP_SUFFIXES = {
    "modal-ok-button",
    "modal-cancel-button",
    "db-add-ok-button",
    "db-add-cancel-button",
    "portfolio-add-ok-button",
    "portfolio-add-cancel-button",
    "underlying-add-ok-button",
    "underlying-add-cancel-button",
    "raw-db-add-ok-button",
    "raw-db-add-cancel-button",
    "sheet-select-ok-button",
    "sheet-select-cancel-button",
    "rolling-chart-switch",
    "rolling-summary-chart-switch",
    "growth-chart-switch",
    "drawdown-chart-switch",
    "weight-chart-switch",
    "weights-chart-switch",
    "attribution-chart-switch",
    "risk-chart-switch",
    "turnover-chart-switch",
    "frontier-chart-switch",
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
    "at-underlying-add-grid",
    "reg-underlying-add-grid",
    "po-underlying-add-grid",
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
        "Name of the series row being configured in this grid. "
        "Drag rows to reorder series processing where order matters, and double-click the cell when rename behavior is enabled for inline edits. "
        "Changes in this row apply to this exact series throughout the active workflow."
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
        "Marks the row as a long-short spread rather than a long-only sleeve. "
        "When enabled, the series-benchmark difference is treated as an absolute return stream for spread-style interpretation in downstream views. "
        "Use this only for true spread constructions."
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
    "desc": (
        "Source description field for the staged import row. "
        "This value identifies the underlying category or database series description that will be read for the selected source portfolio. "
        "Use it together with Portfolio to distinguish similarly named categories from different source codes."
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
        "Minimum allowed portfolio weight for this asset row, expressed as a percentage between 0 and 100. "
        "The optimizer enforces this as a lower bound unless row-level force logic supersedes normal free-variable behavior. "
        "Set realistic floors to avoid infeasible constraint sets."
    ),
    "max wt": (
        "Maximum allowed portfolio weight for this asset row, expressed as a percentage between 0 and 100. "
        "The optimizer treats this as a hard upper bound during allocation search and feasibility checks. "
        "Lower caps increase diversification pressure but can reduce attainable objective values."
    ),
    "force": (
        "Pins this row to its Max Wt target instead of leaving it as a free optimization variable. "
        "When force behavior is active, the weight is fixed at the maximum and removed from normal optimizer search dimensions. "
        "Use sparingly because fixed rows can dominate feasibility and portfolio shape."
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
        "Delete": (
            "Marks this series to be removed from the working dataset when Series Selection changes are saved. "
            "Removal drops the series from active data rather than temporarily hiding it from view. "
            "To bring it back later, add or import the series again."
        ),
        "Del": (
            "Marks this series to be removed from the working dataset when Series Selection changes are saved. "
            "Removal drops the series from active data rather than temporarily hiding it from view. "
            "To bring it back later, add or import the series again."
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
        "Delete": (
            "Marks this series to be removed from the working dataset when Series Selection changes are saved. "
            "Removal drops the series from active data rather than temporarily hiding it from view. "
            "To bring it back later, add or import the series again."
        ),
        "Del": (
            "Marks this series to be removed from the working dataset when Series Selection changes are saved. "
            "Removal drops the series from active data rather than temporarily hiding it from view. "
            "To bring it back later, add or import the series again."
        ),
    },
    "po-series-selection-grid": {
        "CMABench": (
            "CMA benchmark tag used for optimizer assumptions and output comparison. "
            "Keep mappings consistent with your selected portfolio universe."
        ),
        "Delete": (
            "Marks this series to be removed from the working dataset when Series Selection changes are saved. "
            "Removal drops the series from active data and can also invalidate saved portfolio results that depend on it. "
            "To bring it back later, add or import the series again."
        ),
        "Del": (
            "Marks this series to be removed from the working dataset when Series Selection changes are saved. "
            "Removal drops the series from active data and can also invalidate saved portfolio results that depend on it. "
            "To bring it back later, add or import the series again."
        ),
    },
    "at-underlying-add-grid": {
        "Series": (
            "Final imported series name for this staged row. "
            "The name is built from Desc and Portfolio, such as Large Cap [CoreTD], and is the label that will appear in selectors and outputs after import."
        ),
        "Portfolio": (
            "Source PeerTS portfolio code used for this staged row. "
            "The code is built from the selected Base and Type values, such as CoreTD or Base529."
        ),
        "Desc": (
            "Source PeerTS description field used to pull the underlying category series. "
            "This identifies which underlying category is imported for the selected portfolio code."
        ),
    },
    "po-underlying-add-grid": {
        "Series": (
            "Final imported series name for this staged row. "
            "The name is built from Desc and Portfolio, such as Large Cap [CoreTD], and is the label that will appear in selectors and outputs after import."
        ),
        "Portfolio": (
            "Source PeerTS portfolio code used for this staged row. "
            "The code is built from the selected Base and Type values, such as CoreTD or Base529."
        ),
        "Desc": (
            "Source PeerTS description field used to pull the underlying category series. "
            "This identifies which underlying category is imported for the selected portfolio code."
        ),
    },
    "reg-underlying-add-grid": {
        "Series": (
            "Final imported series name for this staged row. "
            "The name is built from Desc and Portfolio, such as Large Cap [CoreTD], and is the label that will appear in selectors and outputs after import."
        ),
        "Portfolio": (
            "Source PeerTS portfolio code used for this staged row. "
            "The code is built from the selected Base and Type values, such as CoreTD or Base529."
        ),
        "Desc": (
            "Source PeerTS description field used to pull the underlying category series. "
            "This identifies which underlying category is imported for the selected portfolio code."
        ),
    },
}


GRID_TOOLTIP_DASH_OPTIONS_DEFAULTS = {
    "tooltipShowDelay": 500,
    "tooltipHideDelay": 5000,
    "tooltipMouseTrack": False,
    "tooltipInteraction": True,
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


_APP_ID_PREFIXES = ("at-", "po-", "reg-")


def _workflow_name_for_id(control_id: str) -> str:
    lowered = str(control_id or "").strip().lower()
    if lowered.startswith("at-"):
        return "Analytics"
    if lowered.startswith("po-"):
        return "Portfolio Optimization"
    if lowered.startswith("reg-"):
        return "Regression"
    return "DashMat"


def _humanize_control_subject(control_id: str, fallback_label: str | None = None) -> str:
    if fallback_label and str(fallback_label).strip():
        return str(fallback_label).strip()
    label = re.sub(r"^(at|po|reg)-", "", str(control_id or "").strip(), flags=re.IGNORECASE)
    label = re.sub(r"[-_]+", " ", label).strip()
    label = re.sub(
        (
            r"\b(button|btn|input|select|switch|toggle|modal|dialog|tabs?|tab|panel|store|container|"
            r"wrapper|dummy|upload|grid|content|value|state|data|rows|row|columns|column)\b"
        ),
        " ",
        label,
        flags=re.IGNORECASE,
    )
    label = re.sub(r"\s+", " ", label).strip()
    return label if label else "this control"


def _is_dashmat_control_id(control_id: str) -> bool:
    lowered = str(control_id or "").strip().lower()
    return lowered.startswith(_APP_ID_PREFIXES)


def _generated_explicit_tooltip(control_id: str, fallback_label: str | None = None) -> str:
    lowered = str(control_id or "").strip().lower()
    workflow = _workflow_name_for_id(control_id)
    subject = _humanize_control_subject(control_id, fallback_label=fallback_label)

    if "robust" in lowered and "se" in lowered:
        return (
            "Enables heteroskedasticity-robust standard errors for regression inference. "
            "This setting does not change fitted coefficients, but it can materially change t-stats, p-values, and confidence intervals when residual variance is not constant. "
            "Enable this when you expect heteroskedastic noise or want more conservative significance diagnostics."
        )
    if "force-zero" in lowered or ("intercept" in lowered and "force" in lowered):
        return (
            "Constrains the regression intercept to zero during model fitting. "
            "Forcing zero can materially change factor loadings and residual behavior, especially when the dependent series has non-zero drift. "
            "Enable only when theory or mandate requires a no-intercept specification."
        )
    if ("exp-wt" in lowered or "exp_wt" in lowered or "exponential" in lowered) and "switch" in lowered:
        return (
            "Turns exponential weighting on for time-series estimation. "
            "When enabled, recent observations receive higher influence than older data, which can improve responsiveness but reduce stability in noisy regimes. "
            "Pair this with a deliberate half-life choice so decay speed matches your use case."
        )
    if "halflife" in lowered or "half-life" in lowered:
        return (
            "Sets the decay speed for exponential weighting. "
            "Smaller values emphasize very recent observations, while larger values smooth estimates by retaining more historical influence. "
            "Choose a value consistent with the update frequency and horizon of your decision process."
        )
    if "vol-scaler" in lowered:
        return (
            f"Sets the annualized volatility target applied in the {workflow} pipeline before downstream calculations. "
            "A value of 0 disables scaling and keeps raw series volatility, while positive values normalize magnitude across selected series. "
            "Use this to make cross-series comparisons more stable when base volatility levels differ materially."
        )
    if "periodicity" in lowered:
        return (
            f"Selects the working data frequency for {workflow} calculations. "
            "Changing periodicity can alter annualization, sample counts, and window interpretation across statistics and model outputs. "
            "Keep this aligned with your source data quality and the horizon you want to analyze."
        )
    if "returns-type" in lowered:
        return (
            "Chooses whether calculations use total returns or benchmark-relative excess returns. "
            "This directly changes return levels, risk metrics, and interpretation of performance diagnostics. "
            "Confirm benchmark mappings before selecting excess mode so relative calculations are meaningful."
        )
    if "start-date" in lowered or "end-date" in lowered:
        return (
            f"Defines one boundary of the active date range used by the {workflow} workflow. "
            "The chosen range controls which observations are eligible for statistics, rolling windows, and model estimation. "
            "Date-range shortcuts can reset these boundaries from selected-series overlap and available history."
        )
    if "common-range" in lowered or "common-daily" in lowered or "maximum-range" in lowered:
        return (
            "Applies a predefined date-range shortcut based on available data overlap. "
            "These shortcuts help avoid manual range errors when series coverage differs across assets or frequencies. "
            "Use them before running analyses to ensure consistent sample alignment."
        )
    if "model-select" in lowered or (lowered.endswith("-model") and "select" in lowered):
        return (
            f"Selects the core model family used for {workflow} computations. "
            "The model choice changes estimation method, required inputs, and which diagnostics are reported. "
            "Choose the model first, then review all model-specific controls before running."
        )
    if "objective" in lowered:
        return (
            "Sets the optimization objective for portfolio construction. "
            "Changing objective alters the trade-off between return, risk, and concentration and can produce substantially different allocations. "
            "Confirm this matches your mandate before applying constraints and expected inputs."
        )
    if "alpha" in lowered or "l1-ratio" in lowered:
        return (
            "Controls regularization strength and penalty mix for shrinkage-based estimation. "
            "These parameters govern the bias-variance tradeoff and can change coefficient stability, sparsity, and out-of-sample behavior. "
            "Tune gradually and compare diagnostics across runs rather than making large jumps."
        )
    if "window" in lowered or "opt-step" in lowered or "fill-in-sample" in lowered:
        return (
            "Sets how estimation and application windows move through time. "
            "Rolling windows use fixed lookback depth, expanding windows grow history through time, and step controls how often recalculation occurs. "
            "Choose window mode, size, and step together so cadence and sample depth match your decision horizon."
        )
    if "missing-data" in lowered:
        return (
            "Defines how missing observations are handled before modeling or optimization. "
            "Different handling modes can bias estimates, alter variance, and change comparability across series. "
            "Use the least distortive option compatible with your data quality and required sample continuity."
        )
    if "arima-p" in lowered:
        return (
            "Sets the ARIMA autoregressive order, p. "
            "This is the number of lagged residual terms included in the residual model, so it controls how much serial dependence is absorbed through autoregressive structure. "
            "Raise it only when lower-order fits leave residual autocorrelation behind."
        )
    if "arima-d" in lowered:
        return (
            "Sets the ARIMA differencing order, d. "
            "This is the number of differences applied before fitting AR and MA terms, which helps remove non-stationary level behavior in the residual series. "
            "Keep it low unless diagnostics clearly show the need for additional differencing."
        )
    if "arima-q" in lowered:
        return (
            "Sets the ARIMA moving-average order, q. "
            "This is the number of lagged forecast-error terms included, so it captures how recent shocks echo through the residual process. "
            "Increase it only when residual diagnostics support extra moving-average structure."
        )
    if "garch-p" in lowered:
        return (
            "Sets the GARCH ARCH order, p. "
            "This is the number of lagged squared residual terms used in the conditional-volatility equation, so it controls how strongly recent shocks affect current volatility. "
            "Higher orders add flexibility but require more data for stable estimation."
        )
    if "garch-q" in lowered:
        return (
            "Sets the GARCH order, q. "
            "This is the number of lagged conditional-variance terms used in the volatility equation, so it controls volatility persistence over time. "
            "Higher orders can fit longer volatility memory but may be unstable on short samples."
        )
    if "arima" in lowered or "garch" in lowered:
        return (
            "Configures residual time-series model parameters used for ARIMA/GARCH diagnostics. "
            "These settings affect residual-process fit quality and the interpretation of AIC/BIC and related outputs. "
            "Adjust only when you intentionally want to test alternative lag/volatility structures."
        )
    if "series-selection" in lowered or "series-select" in lowered or "open-modal" in lowered:
        return (
            f"Controls which series are included in the active {workflow} context. "
            "Inclusion and assignment choices here propagate to downstream calculations, diagnostics, and exports. "
            "These selections define the active universe used by the current workflow state."
        )
    if "benchmark" in lowered or "cmabench" in lowered:
        return (
            "Sets benchmark linkage used for relative calculations or assumption mapping. "
            "Incorrect mappings can distort excess-return metrics, attribution, and optimizer assumptions. "
            "Keep benchmark assignments synchronized with portfolio or factor definitions."
        )
    if "long-short" in lowered or lowered.endswith("-ls") or "ls-" in lowered:
        return (
            "Marks whether a series should be treated as long-short rather than long-only. "
            "This changes interpretation of spreads and can influence risk and attribution metrics. "
            "Enable only for true spread-style inputs."
        )
    if "scale-vol" in lowered or "vol-scaling" in lowered:
        return (
            "Controls series-level eligibility for global volatility scaling. "
            "This allows selective normalization where some series are rescaled and others retain native volatility. "
            "Use consistent rules so comparisons across included series remain interpretable."
        )
    if "constraint" in lowered:
        return (
            "Controls creation or editing of constraint definitions used during optimization or bounded estimation. "
            "Constraint settings directly determine the feasible solution space and can materially alter final outputs. "
            "Review bounds and coefficients together to avoid unintended infeasibility."
        )
    if "ex-ante" in lowered or "bl-" in lowered or "tau" in lowered:
        return (
            "Configures ex-ante assumptions used by mean-variance or Black-Litterman workflows. "
            "Expected returns, covariance/correlation inputs, and view confidence can dominate resulting allocations. "
            "Treat these settings as primary assumptions and validate them before running."
        )
    if "run-button" in lowered or lowered.endswith("-run") or lowered.endswith("-run-button"):
        return (
            f"Runs the current {workflow} workflow using active controls and selected data. "
            "Execution reads current assumptions exactly as configured, so stale selections can propagate directly into outputs. "
            "Confirm key settings first when reproducibility matters."
        )
    if "download" in lowered or "save-session" in lowered or "load-session" in lowered:
        return (
            "Controls export or session persistence behavior for the current workflow state. "
            "Saved and exported artifacts reflect active selections, model inputs, and date boundaries at trigger time. "
            "Use these actions after final checks to preserve a clean and reproducible snapshot."
        )
    if "clear" in lowered or "delete" in lowered or "reset" in lowered:
        return (
            "Clears or removes part of the current working state. "
            "This action can drop selections or results and may require reconfiguration before the next run. "
            "Use this intentionally when you want a clean slate or need to remove stale rows."
        )
    if "factor" in lowered:
        return (
            "Sets factor-analysis inputs, definitions, or display behavior. "
            "Changes here alter grouping logic, transform behavior, and interpretation of factor-conditioned outputs. "
            "Keep factor definitions and transforms consistent with the analytical framework you are testing."
        )
    if "regime" in lowered:
        return (
            "Sets regime-definition selection, editing, or display behavior for regime analysis. "
            "Regime choice determines state assignment and therefore affects conditioned statistics, transition tables, and timeline outputs. "
            "Verify regime source and parameters before applying results across tabs."
        )
    if "rolling" in lowered or "calendar" in lowered or "growth" in lowered or "drawdown" in lowered:
        return (
            "Controls how derived performance views are calculated or displayed over time. "
            "These settings can change annualization, aggregation, and interpretation of path-dependent behavior. "
            "Align them with your reporting horizon before comparing outputs."
        )
    if "correlation" in lowered or "covariance" in lowered or "correlogram" in lowered or "scatter" in lowered:
        return (
            "Controls matrix or relationship-view settings for dependence analysis. "
            "View mode and weighting choices can materially change visual structure and inferred relationships. "
            "Use consistent settings when comparing snapshots across runs."
        )

    return (
        f"Controls {subject} in {workflow}. "
        "Changing this control can affect data inclusion, model assumptions, execution behavior, or output formatting in downstream tables, charts, and exports. "
        "Interpret this field with the surrounding panel context to keep outputs internally consistent."
    )


def tooltip_text_and_source(control_id: str, fallback_label: str | None = None) -> tuple[str, str]:
    key = str(control_id or "").strip()
    if not key:
        return "Configure this setting.", "fallback"

    exact = _EXPLICIT_TOOLTIPS.get(key)
    if exact:
        return exact, "explicit"

    if _is_dashmat_control_id(key):
        return _generated_explicit_tooltip(key, fallback_label=fallback_label), "explicit"

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


def _control_id_suffix(control_id: str | None) -> str:
    key = str(control_id or "").strip().lower()
    if not key or "-" not in key:
        return key
    return key.split("-", 1)[1]


def _is_tooltip_suppressed(control_id: str | None) -> bool:
    return _control_id_suffix(control_id) in _SUPPRESSED_TOOLTIP_SUFFIXES


def _should_wrap_component(component: Component, ancestors: tuple[str, ...]) -> bool:
    if any(name in _SKIP_ANCESTOR_COMPONENTS for name in ancestors):
        return False
    component_name = component.__class__.__name__
    if component_name in _SKIP_WRAP_COMPONENTS:
        return False
    control_id = _component_id(component)
    if not control_id:
        return False
    if _is_tooltip_suppressed(control_id):
        return False
    namespace = _component_namespace(component)
    return is_interactive_component_name(namespace, component_name)


def _is_full_width_value(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return text in {"100%", "full"}


def _component_has_full_width_intent(component: Component) -> bool:
    try:
        if bool(getattr(component, "fullWidth", False)):
            return True
    except Exception:
        pass

    for prop in ("w", "width"):
        try:
            if _is_full_width_value(getattr(component, prop, None)):
                return True
        except Exception:
            continue

    try:
        style = getattr(component, "style", None)
    except Exception:
        style = None
    if isinstance(style, dict) and _is_full_width_value(style.get("width")):
        return True

    return False


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
    full_width_intent = _component_has_full_width_intent(value)
    tooltip_child: Component = value
    if full_width_intent:
        tooltip_child = html.Div(
            value,
            className="dashmat-tooltip-trigger-width",
            **{
                "data-tooltip-trigger-id": control_id or "",
                "data-tooltip-width-intent": "1",
                "style": {"display": "block"},
            },
        )
    tooltip_kwargs: dict[str, Any] = {
        "label": text,
        **TOOLTIP_STYLE_DEFAULT,
        "children": tooltip_child,
    }
    return dmc.Tooltip(
        **tooltip_kwargs,
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
