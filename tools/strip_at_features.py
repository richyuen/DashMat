"""
Strip Factor Analysis, Conditional Returns, and Regime Analysis from analyticstool.py.

This script removes all code related to these three features while preserving
the rest of the AT page. It's designed for a perf A/B experiment.

Usage:
    python tools/strip_at_features.py
"""

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "pages" / "analyticstool.py"


def read_lines(path):
    return path.read_text(encoding="utf-8").splitlines(keepends=True)


def find_line(lines, pattern, start=0):
    """Find first line index matching pattern (1-indexed internally, but returns 0-indexed)."""
    for i in range(start, len(lines)):
        if pattern in lines[i]:
            return i
    raise ValueError(f"Pattern not found: {pattern!r} starting from line {start + 1}")


def find_function_end(lines, start):
    """Find end of a function/class definition starting at `start` (0-indexed).
    Returns the index of the last line of the function body."""
    # Get indentation of the def/class line
    match = re.match(r'^(\s*)', lines[start])
    base_indent = len(match.group(1))

    # First, skip past the function signature (which may span multiple lines)
    # Look for the colon that ends the signature
    i = start
    found_colon = False
    while i < len(lines):
        stripped = lines[i].rstrip()
        if stripped.endswith(':'):
            found_colon = True
            break
        i += 1
    if not found_colon:
        i = start

    i += 1  # move past the signature line
    last_content = i - 1
    while i < len(lines):
        line = lines[i]
        stripped = line.rstrip()
        if stripped:  # non-empty line
            line_indent = len(line) - len(line.lstrip())
            if line_indent <= base_indent:
                # At same or lower indent - function is over
                return last_content
            last_content = i
        i += 1
    return last_content


def find_decorator_start(lines, func_start):
    """Walk backwards from func_start to find the first @callback or @dataclass decorator.
    Handles multi-line decorators like @callback(Output(...), Input(...), ...)."""
    # Determine the base indent of the def line
    def_indent = len(lines[func_start]) - len(lines[func_start].lstrip())

    i = func_start - 1
    # Walk back past blank lines
    while i >= 0 and lines[i].strip() == '':
        i -= 1
    if i < 0:
        return func_start

    stripped = lines[i].strip()
    line_indent = len(lines[i]) - len(lines[i].lstrip())

    # Only treat ')' as a decorator closing if it's at the same indent as the def
    if stripped == ')' and line_indent == def_indent:
        # Walk back to find matching '(' context - look for @decorator(
        depth = 0
        while i >= 0:
            for ch in reversed(lines[i]):
                if ch == ')':
                    depth += 1
                elif ch == '(':
                    depth -= 1
            if depth <= 0:
                if lines[i].strip().startswith('@'):
                    return i
                # Walk back past blank/comment lines to find @
                j = i - 1
                while j >= 0 and (lines[j].strip() == '' or lines[j].strip().startswith('#')):
                    j -= 1
                if j >= 0 and lines[j].strip().startswith('@'):
                    return j
                return i
            i -= 1
        return func_start
    elif stripped.startswith('@'):
        return i
    else:
        return func_start


def find_clientside_callback_end(lines, start):
    """Find the end of a clientside_callback(...) call starting at `start`."""
    # Count parentheses
    depth = 0
    i = start
    while i < len(lines):
        for ch in lines[i]:
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    raise ValueError(f"Could not find end of clientside_callback starting at line {start + 1}")


def mark_range(to_delete, start, end):
    """Mark lines start..end (inclusive, 0-indexed) for deletion."""
    for i in range(start, end + 1):
        to_delete.add(i)


def mark_function(lines, to_delete, pattern, include_decorator=True):
    """Mark an entire function (with decorator) for deletion."""
    start = find_line(lines, pattern)
    if include_decorator:
        start = find_decorator_start(lines, start)
    end = find_function_end(lines, find_line(lines, pattern))
    # Also delete trailing blank lines
    while end + 1 < len(lines) and lines[end + 1].strip() == '':
        end += 1
    mark_range(to_delete, start, end)
    return start, end


def mark_clientside_callback(lines, to_delete, pattern):
    """Mark an entire clientside_callback(...) block for deletion."""
    start = find_line(lines, pattern)
    # Walk back to 'clientside_callback('
    i = start
    while i >= 0 and 'clientside_callback(' not in lines[i]:
        i -= 1
    if i < 0:
        raise ValueError(f"Could not find clientside_callback( before pattern at line {start + 1}")
    cb_start = i
    end = find_clientside_callback_end(lines, cb_start)
    # Delete trailing blank lines
    while end + 1 < len(lines) and lines[end + 1].strip() == '':
        end += 1
    mark_range(to_delete, cb_start, end)
    return cb_start, end


def main():
    lines = read_lines(SRC)
    to_delete = set()

    # ── 1. Remove imports ──────────────────────────────────────────────
    # utils.factor_definitions import block
    start = find_line(lines, "from utils.factor_definitions import (")
    end = find_line(lines, ")", start)
    mark_range(to_delete, start, end)

    # utils.regime_analysis import block
    start = find_line(lines, "from utils.regime_analysis import (")
    end = find_line(lines, ")", start)
    mark_range(to_delete, start, end)

    # utils.regime_definitions import block
    start = find_line(lines, "from utils.regime_definitions import (")
    end = find_line(lines, ")", start)
    mark_range(to_delete, start, end)

    # ── 2. Remove constants ────────────────────────────────────────────
    for const in [
        "CONDITIONAL_VIEW_OPTIONS",
        "CONDITIONAL_DISPLAY_MODE_OPTIONS",
        "REGIME_DETAIL_DISPLAY_MODE_OPTIONS",
        "CONDITIONAL_COMPARATOR_OPTIONS",
        "CONDITIONAL_FACTOR_CONVERSION_OPTIONS",
    ]:
        start = find_line(lines, f"{const} = [")
        end = find_line(lines, "]", start)
        # Delete trailing blank lines
        while end + 1 < len(lines) and lines[end + 1].strip() == '':
            end += 1
        mark_range(to_delete, start, end)

    # ── 3. Remove dataclasses ──────────────────────────────────────────
    for cls in [
        "class _FactorArtifacts:",
        "class _RegimeAnalysisPayload:",
        "class _RegimeAnalysisBuildResult:",
        "class _ConditionalReturnsPayload:",
        "class _ConditionalCoreArtifacts:",
    ]:
        start = find_line(lines, cls)
        dec_start = find_decorator_start(lines, start)
        end = find_function_end(lines, start)
        while end + 1 < len(lines) and lines[end + 1].strip() == '':
            end += 1
        mark_range(to_delete, dec_start, end)

    # ── 4. Remove helper functions ─────────────────────────────────────
    # Regime helpers
    for func in [
        "def _build_regime_warning_text(",
        "def _compute_regime_analysis_outputs_cached(",
        "def _build_regime_analysis_payload(",
    ]:
        mark_function(lines, to_delete, func)

    # Factor/regime definition helpers (lines ~894-1430)
    for func in [
        "def _factor_user_label(",
        "def _source_badge(",
        "def _default_factor_draft(",
        "def _ensure_factor_draft(",
        "def _factor_select_key(",
        "def _split_factor_select_key(",
        "def _index_factor_definitions(",
        "def _lookup_factor_definition(",
        "def _normalize_factor_value_for_options(",
        "def _factor_option_definitions(",
        "def _definition_payload_for_compute(",
        "def _factor_definition_signature(",
        "def _factor_db_name_exists(",
        "def _definition_to_draft(",
        "def _draft_to_definition_payload(",
        "def _default_regime_draft(",
        "def _ensure_regime_draft(",
        "def _regime_select_key(",
        "def _split_regime_select_key(",
        "def _index_regime_definitions(",
        "def _lookup_regime_definition(",
        "def _regime_option_definitions(",
        "def _normalize_regime_value_for_options(",
        "def _regime_definition_to_draft(",
        "def _regime_draft_to_definition_payload(",
        "def _regime_definition_signature(",
        "def _regime_db_name_exists(",
        "def _build_regime_series_options(",
    ]:
        mark_function(lines, to_delete, func)

    # Factor computation helpers
    for func in [
        "def _prepare_factor_base_frames(",
        "def _empty_factor_artifacts(",
        "def _compute_factor_artifacts_cached(",
        "def _compute_factor_artifacts(",
        "def _prepare_factor_analysis_frames(",
        "def _build_factor_pair_df(",
        "def _build_factor_detail_frame(",
        "def _prepare_factor_analysis_selected_df(",
        "def _is_weekly_periodicity(",
    ]:
        mark_function(lines, to_delete, func)

    # Conditional computation helpers
    for func in [
        "def _conditional_window_specs(",
        "def _resolve_conditional_anchor_positions(",
        "def _shift_index_by_months(",
        "def _resolve_window_bounds(",
        "def _aggregate_window_values(",
        "def _apply_zscore(",
        "def _conditional_conversion_tooltip_text(",
        "def _conditional_tab_signature(",
        "def _empty_conditional_returns_payload(",
        "def _empty_conditional_core_artifacts(",
        "def _estimate_conditional_detail_row_counts(",
        "def _build_conditional_summary_frames_from_core(",
        "def _build_conditional_forward_summary_from_core(",
        "def _build_conditional_detail_frames_from_core(",
        "def _order_conditional_detail_frame(",
        "def _compute_conditional_core_cached(",
        "def _compute_conditional_returns_cached(",
        "def _prepare_at_qq_reference_series(",
        "def _coerce_factor_quantiles(",
    ]:
        mark_function(lines, to_delete, func)

    # Factor rendering helpers
    for func in [
        "def _factor_quantile_labels(",
        "def _build_factor_box_summary_rows(",
        "def _build_factor_scatter_summary_rows(",
    ]:
        mark_function(lines, to_delete, func)

    # Conditional rendering helpers
    for func in [
        "def _build_conditional_returns_grid_component(",
        "def _build_conditional_detail_grid_component(",
        "def _conditional_export_block(",
        "def _build_conditional_export_frame(",
        "def _build_conditional_detail_export_frame(",
    ]:
        mark_function(lines, to_delete, func)

    # Regime rendering helpers
    for func in [
        "def _build_regime_grid_component(",
        "def _detail_render_warning(",
        "def _build_regime_settings_text_component(",
    ]:
        mark_function(lines, to_delete, func)

    # Export builders
    for func in [
        "def _build_factor_export_sheets(",
        "def _build_conditional_export_sheets(",
        "def _build_regime_export_sheets(",
    ]:
        mark_function(lines, to_delete, func)

    # ── 5. Remove main callbacks ───────────────────────────────────────
    for func in [
        "def update_factor_analysis(",
        "def update_conditional_returns_target_key(",
        "def control_conditional_returns_loading_display(",
        "def update_conditional_returns(",
        "def update_regime_analysis(",
    ]:
        mark_function(lines, to_delete, func)

    # Factor modal callbacks
    for func in [
        "def at_lazy_load_factor_definitions(",
        "def at_lazy_load_regime_definitions(",
        "def sync_conditional_returns_control_state(",
        "def sync_regime_detail_display_mode(",
        "def update_factor_series_select(",
        "def sync_factor_control_mirrors(",
        "def at_open_factor_definition_modal(",
        "def at_load_factor_modal_data(",
        "def at_update_factor_definition_select_options(",
        "def at_load_selected_factor_definition(",
        "def at_reset_factor_definition_draft(",
        "def at_sync_factor_definition_form(",
        "def at_update_factor_definition_draft_from_form(",
        "def at_update_factor_definition_preview(",
        "def at_manage_factor_definitions(",
    ]:
        mark_function(lines, to_delete, func)

    # Regime modal callbacks
    for func in [
        "def at_update_regime_definition_analysis_select_options(",
        "def at_open_regime_definition_modal(",
        "def at_load_regime_modal_data(",
        "def at_update_regime_definition_select_options(",
        "def at_load_selected_regime_definition(",
        "def at_reset_regime_definition_draft(",
        "def at_refresh_regime_series_options_for_definition(",
        "def at_sync_regime_definition_form(",
        "def at_update_regime_definition_draft_from_form(",
        "def at_update_regime_definition_preview(",
        "def at_manage_regime_definitions(",
    ]:
        mark_function(lines, to_delete, func)

    # ── 6. Remove clientside_callback blocks ───────────────────────────
    # Factor/regime sync callback
    mark_clientside_callback(lines, to_delete,
        'analyticsFactorRegimeSync')
    # Factor tab trigger
    mark_clientside_callback(lines, to_delete,
        'analyticsTabTrigger("factor_analysis"')
    # Regime tab trigger
    mark_clientside_callback(lines, to_delete,
        'analyticsTabTrigger("regime_analysis"')
    # Conditional tab trigger
    mark_clientside_callback(lines, to_delete,
        'analyticsTabTrigger("conditional_returns"')
    # Factor preview trigger
    mark_clientside_callback(lines, to_delete,
        'Output("at-factor-preview-trigger-store"')
    # Regime preview trigger
    mark_clientside_callback(lines, to_delete,
        'Output("at-regime-preview-trigger-store"')

    # ── 7. Remove tab entries ──────────────────────────────────────────
    for tab in [
        'dmc.TabsTab("Factor Analysis"',
        'dmc.TabsTab("Conditional Returns"',
        'dmc.TabsTab("Regime Analysis"',
    ]:
        idx = find_line(lines, tab)
        to_delete.add(idx)

    # ── 8. Remove tab panels ──────────────────────────────────────────
    # Factor analysis panel starts at the dmc.TabsPanel(value="factor_analysis")
    fa_start = find_line(lines, 'value="factor_analysis",')
    # Walk back to dmc.TabsPanel(
    while 'dmc.TabsPanel(' not in lines[fa_start]:
        fa_start -= 1

    # Regime analysis panel ends at ), followed by growth panel
    ra_end = find_line(lines, 'value="growth",')
    # Walk back to dmc.TabsPanel(
    while 'dmc.TabsPanel(' not in lines[ra_end]:
        ra_end -= 1
    ra_end -= 1  # line before growth panel's TabsPanel
    # Remove trailing blank/whitespace lines
    while lines[ra_end].strip() == '' or lines[ra_end].strip() == '),':
        if lines[ra_end].strip() == '),':
            # This is the closing of regime panel
            mark_range(to_delete, fa_start, ra_end)
            break
        ra_end -= 1
    else:
        mark_range(to_delete, fa_start, ra_end)

    # ── 9. Remove modals ──────────────────────────────────────────────
    # Factor definition modal: starts at dmc.Modal(id="at-factor-def-modal")
    modal_start = find_line(lines, 'id="at-factor-def-modal",')
    while 'dmc.Modal(' not in lines[modal_start]:
        modal_start -= 1

    # Regime definition modal ends before "# Welcome Screen"
    modal_end = find_line(lines, "# Welcome Screen")
    modal_end -= 1
    while lines[modal_end].strip() == '':
        modal_end -= 1
    mark_range(to_delete, modal_start, modal_end)

    # ── 10. Remove stores ─────────────────────────────────────────────
    store_patterns = [
        "at-factor-mode-store",
        "at-factor-quantiles-store",
        "at-factor-transform-store",
        "at-factor-series-store",
        "at-factor-qq-reference-store",
        "at-conditional-view-store",
        "at-conditional-comparator-store",
        "at-conditional-threshold-store",
        "at-conditional-window-conversion-store",
        "at-conditional-step-store",
        "at-conditional-step-unit-store",
        "at-conditional-display-mode-store",
        "at-factor-definitions-db-store",
        "at-factor-definitions-local-store",
        "at-factor-def-modal-draft-store",
        "at-factor-def-db-available-store",
        "at-factor-def-loaded-store",
        "at-regime-definition-store",
        "at-regime-detail-display-mode-store",
        "at-regime-definitions-db-store",
        "at-regime-definitions-local-store",
        "at-regime-def-modal-draft-store",
        "at-regime-def-db-available-store",
        "at-regime-def-loaded-store",
        "at-regime-series-store",
        "at-factor-tab-trigger-store",
        "at-regime-tab-trigger-store",
        "at-conditional-tab-trigger-store",
        "at-factor-preview-trigger-store",
        "at-regime-preview-trigger-store",
        "at-conditional-returns-target-key-store",
        "at-conditional-returns-rendered-key-store",
    ]
    for store_id in store_patterns:
        try:
            idx = find_line(lines, f'id="{store_id}"')
            to_delete.add(idx)
        except ValueError:
            print(f"  Warning: store {store_id} not found", file=sys.stderr)

    # ── 11. build_main_layout signature + call site ──────────────────
    # These are handled via text replacement in post-processing (see below)
    # because params share lines with non-factor/conditional/regime params.
    # Also remove the factor_series_options line in the body.
    try:
        idx = find_line(lines, "factor_series_options = factor_series_options or []")
        to_delete.add(idx)
    except ValueError:
        pass

    # ── 12. Trim _at_restore_defaults ──────────────────────────────────
    defaults_start = find_line(lines, "def _at_restore_defaults():")
    for key in [
        '"factor_mode"',
        '"factor_quantiles"',
        '"factor_transform"',
        '"factor_qq_reference"',
        '"conditional_view"',
        '"conditional_comparator"',
        '"conditional_threshold"',
        '"conditional_window_conversion"',
        '"conditional_step"',
        '"conditional_step_unit"',
        '"conditional_display_mode"',
        '"regime_display_mode"',
    ]:
        try:
            idx = find_line(lines, f"        {key}:", defaults_start)
            to_delete.add(idx)
        except ValueError:
            pass

    # ── 13. Trim _at_resolve_restore_state ─────────────────────────────
    for param in [
        "stored_factor_mode",
        "stored_factor_quantiles",
        "stored_factor_transform",
        "stored_factor_qq_reference",
        "stored_conditional_view",
        "stored_conditional_comparator",
        "stored_conditional_threshold",
        "stored_conditional_window_conversion",
        "stored_conditional_step",
        "stored_conditional_step_unit",
        "stored_conditional_display_mode",
        "stored_regime_display_mode",
    ]:
        # In function signature
        try:
            idx = find_line(lines, f"    {param},")
            to_delete.add(idx)
        except ValueError:
            pass

    # Remove factor/conditional/regime resolved.update entries in _at_resolve_restore_state
    resolve_update_keys = [
        '"factor_mode":', '"factor_quantiles":', '"factor_transform":', '"factor_qq_reference":',
        '"conditional_view":', '"conditional_comparator":', '"conditional_threshold":',
        '"conditional_window_conversion":', '"conditional_step":', '"conditional_step_unit":',
        '"conditional_display_mode":', '"regime_display_mode":',
    ]
    # Find the resolved.update block - starts after line with resolved.update(
    try:
        update_start = find_line(lines, "resolved.update(", find_line(lines, "def _at_resolve_restore_state("))
        for key in resolve_update_keys:
            try:
                idx = find_line(lines, key, update_start)
                to_delete.add(idx)
                # Some entries span multiple lines (parenthesized expressions)
                if lines[idx].rstrip().endswith("("):
                    j = idx + 1
                    while j < len(lines) and ")" not in lines[j]:
                        to_delete.add(j)
                        j += 1
                    if j < len(lines):
                        to_delete.add(j)
            except ValueError:
                pass
    except ValueError:
        pass

    # ── 14. Trim restore_application_state callback ────────────────────
    # Remove Outputs for factor/conditional/regime controls
    restore_outputs = [
        'Output("at-factor-mode-select"',
        'Output("at-factor-quantiles-input"',
        'Output("at-factor-transform-select"',
        'Output("at-factor-qq-reference-select"',
        'Output("at-conditional-view-select"',
        'Output("at-conditional-comparator-select"',
        'Output("at-conditional-threshold-input"',
        'Output("at-conditional-window-conversion-select"',
        'Output("at-conditional-step-input"',
        'Output("at-conditional-step-unit-select"',
        'Output("at-conditional-display-mode-select"',
        'Output("at-regime-detail-display-mode-select"',
    ]
    restore_states = [
        'State("at-factor-mode-store"',
        'State("at-factor-quantiles-store"',
        'State("at-factor-transform-store"',
        'State("at-factor-qq-reference-store"',
        'State("at-conditional-view-store"',
        'State("at-conditional-comparator-store"',
        'State("at-conditional-threshold-store"',
        'State("at-conditional-window-conversion-store"',
        'State("at-conditional-step-store"',
        'State("at-conditional-step-unit-store"',
        'State("at-conditional-display-mode-store"',
        'State("at-regime-detail-display-mode-store"',
    ]

    # Find restore_application_state callback decorator
    restore_cb_start = find_line(lines, "def restore_application_state(")
    restore_dec_start = find_decorator_start(lines, restore_cb_start)

    for pattern in restore_outputs + restore_states:
        try:
            idx = find_line(lines, pattern, restore_dec_start)
            to_delete.add(idx)
        except ValueError:
            pass

    # Remove params from function signature
    for param in [
        "stored_factor_mode,",
        "stored_factor_quantiles,",
        "stored_factor_transform,",
        "stored_factor_qq_reference,",
        "stored_conditional_view,",
        "stored_conditional_comparator,",
        "stored_conditional_threshold,",
        "stored_conditional_window_conversion,",
        "stored_conditional_step,",
        "stored_conditional_step_unit,",
        "stored_conditional_display_mode,",
        "stored_regime_display_mode,",
    ]:
        try:
            idx = find_line(lines, f"    {param}", restore_cb_start)
            if idx < restore_cb_start + 40:  # within function signature
                to_delete.add(idx)
        except ValueError:
            pass

    # In restore body, remove the call params and output handling
    # These are in the _at_resolve_restore_state call within restore_application_state
    restore_call_start = find_line(lines, "resolved = _at_resolve_restore_state(", restore_cb_start)
    for param in [
        "stored_factor_mode,",
        "stored_factor_quantiles,",
        "stored_factor_transform,",
        "stored_factor_qq_reference,",
        "stored_conditional_view,",
        "stored_conditional_comparator,",
        "stored_conditional_threshold,",
        "stored_conditional_window_conversion,",
        "stored_conditional_step,",
        "stored_conditional_step_unit,",
        "stored_conditional_display_mode,",
        "stored_regime_display_mode,",
    ]:
        try:
            idx = find_line(lines, f"            {param}", restore_call_start)
            to_delete.add(idx)
        except ValueError:
            pass

    # ── 15. Trim sync_at_returns_type_from_mirrors ─────────────────────
    sync_from_start = find_line(lines, "def sync_at_returns_type_from_mirrors(")
    sync_from_dec = find_decorator_start(lines, sync_from_start)

    for pattern in [
        'Input("at-returns-type-select-factor"',
        'Input("at-returns-type-select-conditional"',
        'Input("at-returns-type-select-regime"',
    ]:
        try:
            idx = find_line(lines, pattern, sync_from_dec)
            to_delete.add(idx)
        except ValueError:
            pass

    for param in ["factor_value,", "conditional_value,", "regime_value,"]:
        try:
            idx = find_line(lines, f"    {param}", sync_from_start)
            if idx < sync_from_start + 15:
                to_delete.add(idx)
        except ValueError:
            pass

    for param in [
        '"at-returns-type-select-factor": factor_value,',
        '"at-returns-type-select-conditional": conditional_value,',
        '"at-returns-type-select-regime": regime_value,',
    ]:
        try:
            idx = find_line(lines, param, sync_from_start)
            to_delete.add(idx)
        except ValueError:
            pass

    # ── 16. Trim sync_at_returns_type_mirrors ──────────────────────────
    sync_to_start = find_line(lines, "def sync_at_returns_type_mirrors(")
    sync_to_dec = find_decorator_start(lines, sync_to_start)

    for pattern in [
        'Output("at-returns-type-select-factor"',
        'Output("at-returns-type-select-conditional"',
        'Output("at-returns-type-select-regime"',
        'State("at-returns-type-select-factor"',
        'State("at-returns-type-select-conditional"',
        'State("at-returns-type-select-regime"',
    ]:
        try:
            idx = find_line(lines, pattern, sync_to_dec)
            to_delete.add(idx)
        except ValueError:
            pass

    for param in ["factor_value,", "conditional_value,", "regime_value,"]:
        try:
            idx = find_line(lines, f"    {param}", sync_to_start)
            if idx < sync_to_start + 15:
                to_delete.add(idx)
        except ValueError:
            pass

    # In return tuple, remove factor/conditional/regime outputs
    # The return is: return (_sync(returns), _sync(calendar), _sync(drawdown), _sync(correlogram), _sync(factor), _sync(conditional), _sync(regime))
    for pattern in [
        "_sync(factor_value),",
        "_sync(conditional_value),",
        "_sync(regime_value),",
    ]:
        try:
            idx = find_line(lines, pattern, sync_to_start)
            to_delete.add(idx)
        except ValueError:
            pass

    # ── 17. Trim at_restore_secondary_controls ─────────────────────────
    sec_start = find_line(lines, "def at_restore_secondary_controls(")
    sec_dec = find_decorator_start(lines, sec_start)

    # Remove Outputs
    for pattern in [
        'Output("at-factor-mode-select", "value", allow_duplicate',
        'Output("at-factor-quantiles-input", "value", allow_duplicate',
        'Output("at-factor-transform-select", "value", allow_duplicate',
        'Output("at-factor-qq-reference-select", "value", allow_duplicate',
        'Output("at-conditional-view-select", "value", allow_duplicate',
        'Output("at-conditional-comparator-select", "value", allow_duplicate',
        'Output("at-conditional-threshold-input", "value", allow_duplicate',
        'Output("at-conditional-window-conversion-select", "value", allow_duplicate',
        'Output("at-conditional-step-input", "value", allow_duplicate',
        'Output("at-conditional-step-unit-select", "value", allow_duplicate',
        'Output("at-conditional-display-mode-select", "value", allow_duplicate',
        'Output("at-regime-detail-display-mode-select", "value", allow_duplicate',
    ]:
        try:
            idx = find_line(lines, pattern, sec_dec)
            to_delete.add(idx)
        except ValueError:
            pass

    # Remove States (stores + controls)
    for pattern in [
        'State("at-factor-mode-store"',
        'State("at-factor-quantiles-store"',
        'State("at-factor-transform-store"',
        'State("at-factor-qq-reference-store"',
        'State("at-conditional-view-store"',
        'State("at-conditional-comparator-store"',
        'State("at-conditional-threshold-store"',
        'State("at-conditional-window-conversion-store"',
        'State("at-conditional-step-store"',
        'State("at-conditional-step-unit-store"',
        'State("at-conditional-display-mode-store"',
        'State("at-regime-detail-display-mode-store"',
        'State("at-factor-mode-select"',
        'State("at-factor-quantiles-input"',
        'State("at-factor-transform-select"',
        'State("at-factor-qq-reference-select"',
        'State("at-conditional-view-select"',
        'State("at-conditional-comparator-select"',
        'State("at-conditional-threshold-input"',
        'State("at-conditional-window-conversion-select"',
        'State("at-conditional-step-input"',
        'State("at-conditional-step-unit-select"',
        'State("at-conditional-display-mode-select"',
        'State("at-regime-detail-display-mode-select"',
    ]:
        try:
            idx = find_line(lines, pattern, sec_dec)
            to_delete.add(idx)
        except ValueError:
            pass

    # Remove function params
    for param in [
        "stored_factor_mode,",
        "stored_factor_quantiles,",
        "stored_factor_transform,",
        "stored_factor_qq_reference,",
        "stored_conditional_view,",
        "stored_conditional_comparator,",
        "stored_conditional_threshold,",
        "stored_conditional_window_conversion,",
        "stored_conditional_step,",
        "stored_conditional_step_unit,",
        "stored_conditional_display_mode,",
        "stored_regime_display_mode,",
        "current_factor_mode,",
        "current_factor_quantiles,",
        "current_factor_transform,",
        "current_factor_qq_reference,",
        "current_conditional_view,",
        "current_conditional_comparator,",
        "current_conditional_threshold,",
        "current_conditional_window_conversion,",
        "current_conditional_step,",
        "current_conditional_step_unit,",
        "current_conditional_display_mode,",
        "current_regime_display_mode,",
    ]:
        try:
            idx = find_line(lines, f"    {param}", sec_start)
            if idx < sec_start + 60:  # within signature
                to_delete.add(idx)
        except ValueError:
            pass

    # Remove the call params in _at_resolve_restore_state call within at_restore_secondary_controls
    sec_resolve_call = find_line(lines, "resolved = _at_resolve_restore_state(", sec_start)
    for param in [
        "stored_factor_mode,",
        "stored_factor_quantiles,",
        "stored_factor_transform,",
        "stored_factor_qq_reference,",
        "stored_conditional_view,",
        "stored_conditional_comparator,",
        "stored_conditional_threshold,",
        "stored_conditional_window_conversion,",
        "stored_conditional_step,",
        "stored_conditional_step_unit,",
        "stored_conditional_display_mode,",
        "stored_regime_display_mode,",
    ]:
        try:
            idx = find_line(lines, f"        {param}", sec_resolve_call)
            if idx < sec_resolve_call + 40:
                to_delete.add(idx)
        except ValueError:
            pass

    # ── 18. Trim download_excel callback ───────────────────────────────
    dl_start = find_line(lines, "def download_excel(")
    dl_dec = find_decorator_start(lines, dl_start)

    for pattern in [
        'State("at-factor-series-select"',
        'State("at-factor-quantiles-input"',
        'State("at-factor-transform-select"',
        'State("at-conditional-view-store"',
        'State("at-conditional-comparator-store"',
        'State("at-conditional-threshold-store"',
        'State("at-conditional-window-conversion-store"',
        'State("at-conditional-step-store"',
        'State("at-conditional-step-unit-store"',
        'State("at-factor-definitions-db-store"',
        'State("at-factor-definitions-local-store"',
        'State("at-regime-definition-select"',
        'State("at-regime-definitions-db-store"',
        'State("at-regime-definitions-local-store"',
        'State("at-regime-series-store"',
    ]:
        try:
            idx = find_line(lines, pattern, dl_dec)
            to_delete.add(idx)
        except ValueError:
            pass

    # Remove function params
    for param in [
        "factor_series,",
        "factor_quantiles,",
        "factor_transform,",
        "conditional_view,",
        "conditional_comparator,",
        "conditional_threshold,",
        "conditional_window_conversion,",
        "conditional_step,",
        "conditional_step_unit,",
    ]:
        try:
            idx = find_line(lines, f"    {param}", dl_start)
            if idx < dl_start + 40:
                to_delete.add(idx)
        except ValueError:
            pass

    # Remove defaulted params
    for param in [
        "factor_definitions_db=None,",
        "factor_definitions_local=None,",
        "regime_definition_key=None,",
        "regime_definitions_db=None,",
        "regime_definitions_local=None,",
        "regime_series_store=None,",
    ]:
        try:
            idx = find_line(lines, f"    {param}", dl_start)
            if idx < dl_start + 45:
                to_delete.add(idx)
        except ValueError:
            pass

    # ── 19. Trim _resolve_export_sheet_specs ───────────────────────────
    export_func_start = find_line(lines, "def _resolve_export_sheet_specs(")

    # Remove params from signature
    for param in [
        "factor_series,",
        "factor_quantiles,",
        "factor_transform,",
        "conditional_comparator,",
        "conditional_threshold,",
        "conditional_window_conversion,",
        "conditional_step,",
        "conditional_step_unit,",
        "factor_definitions_db,",
        "factor_definitions_local,",
        "regime_definition_key,",
        "regime_definitions_db,",
        "regime_definitions_local,",
        "regime_series_store,",
    ]:
        try:
            idx = find_line(lines, f"    {param}", export_func_start)
            if idx < export_func_start + 40:
                to_delete.add(idx)
        except ValueError:
            pass

    # Remove the three try/except blocks for factor/conditional/regime exports
    for marker in [
        'timed_block("analyticstool.download_excel.factor_analysis")',
        'timed_block("analyticstool.download_excel.conditional_returns")',
        'timed_block("analyticstool.download_excel.regime_analysis")',
    ]:
        try:
            idx = find_line(lines, marker, export_func_start)
            # Walk back to 'try:'
            try_idx = idx - 1
            while try_idx >= 0 and 'try:' not in lines[try_idx]:
                try_idx -= 1
            # Find the 'except Exception:' and 'pass'
            except_idx = find_line(lines, "except Exception:", idx)
            pass_idx = find_line(lines, "pass", except_idx)
            # Delete trailing blank line
            end = pass_idx
            while end + 1 < len(lines) and lines[end + 1].strip() == '':
                end += 1
            mark_range(to_delete, try_idx, end)
        except ValueError:
            pass

    # Remove params from the _resolve_export_sheet_specs call in download_excel
    dl_call_start = find_line(lines, "sheet_specs = _resolve_export_sheet_specs(", dl_start)
    for param in [
        "factor_series,",
        "factor_quantiles,",
        "factor_transform,",
        "conditional_comparator,",
        "conditional_threshold,",
        "conditional_window_conversion,",
        "conditional_step,",
        "conditional_step_unit,",
        "factor_definitions_db,",
        "factor_definitions_local,",
        "regime_definition_key,",
        "regime_definitions_db,",
        "regime_definitions_local,",
        "regime_series_store,",
    ]:
        try:
            idx = find_line(lines, f"            {param}", dl_call_start)
            if idx < dl_call_start + 40:
                to_delete.add(idx)
        except ValueError:
            pass

    # ── 20. Handle factor_outputs/conditional_outputs/regime_output in restore bodies ──
    # In restore_application_state body, remove factor_outputs, conditional_outputs, regime_output
    # and change the return tuple
    # These are complex multi-line blocks - handle with line-by-line deletion

    # factor_outputs block
    try:
        idx = find_line(lines, "factor_outputs = (", restore_cb_start)
        end = find_line(lines, ") if active_tab == \"factor_analysis\"", idx)
        mark_range(to_delete, idx, end)
    except ValueError:
        pass

    # conditional_outputs block
    try:
        idx = find_line(lines, "conditional_outputs = (", restore_cb_start)
        end = find_line(lines, ") if active_tab == \"conditional_returns\"", idx)
        mark_range(to_delete, idx, end)
    except ValueError:
        pass

    # regime_output line
    try:
        idx = find_line(lines, 'regime_output = resolved["regime_display_mode"]', restore_cb_start)
        to_delete.add(idx)
    except ValueError:
        pass

    # In return tuple, remove *factor_outputs, *conditional_outputs, regime_output
    try:
        idx = find_line(lines, "*factor_outputs,", restore_cb_start)
        to_delete.add(idx)
    except ValueError:
        pass
    try:
        idx = find_line(lines, "*conditional_outputs,", restore_cb_start)
        to_delete.add(idx)
    except ValueError:
        pass
    try:
        idx = find_line(lines, "regime_output,", restore_cb_start)
        to_delete.add(idx)
    except ValueError:
        pass

    # In the except branch return tuple, adjust the no_update count
    # The current except return has the right number of no_updates matching all outputs
    # We're removing 12 outputs (4 factor + 7 conditional + 1 regime)
    # Need to reduce the no_update count by 12 in the except branch

    # ── 21. Handle at_restore_secondary_controls body ──────────────────
    # Remove factor/conditional/regime branches from the outputs list logic
    # outputs = [no_update] * 21  -> should become [no_update] * 9
    #   (rolling=6, drawdown=1, growth=1, monthly=1 = 9)

    try:
        idx = find_line(lines, "outputs = [no_update] * 21", sec_start)
        # Will be handled by text replacement below
    except ValueError:
        pass

    # Remove the elif blocks for factor_analysis, conditional_returns, regime_analysis
    for tab_name in ["factor_analysis", "conditional_returns", "regime_analysis"]:
        try:
            idx = find_line(lines, f'elif active_tab == "{tab_name}":', sec_start)
            elif_indent = len(lines[idx]) - len(lines[idx].lstrip())
            end = idx
            while end + 1 < len(lines):
                next_line = lines[end + 1]
                next_stripped = next_line.strip()
                if next_stripped == '':
                    end += 1
                    continue
                next_indent = len(next_line) - len(next_line.lstrip())
                if next_indent <= elif_indent:
                    break
                end += 1
            mark_range(to_delete, idx, end)
        except ValueError:
            pass

    # ── Apply deletions ────────────────────────────────────────────────
    result_lines = [line for i, line in enumerate(lines) if i not in to_delete]

    # ── Post-process text replacements ─────────────────────────────────
    result_text = "".join(result_lines)

    # ── build_main_layout signature ────────────────────────────────────
    result_text = result_text.replace(
        """def build_main_layout(periodicity_options, periodicity_value, returns_type, vol_scaler,
                      active_tab, rolling_window, rolling_metric, rolling_return_type, rolling_chart_switch,
                      drawdown_chart_switch, growth_chart_switch, monthly_view, monthly_series,
                      monthly_series_options, monthly_select_disabled, factor_mode,
                      factor_quantiles, factor_transform, factor_series_options,
                      factor_series_value, factor_qq_reference, conditional_view,
                      conditional_comparator, conditional_threshold, conditional_window_conversion,
                      conditional_step, conditional_step_unit, conditional_display_mode,
                      regime_display_mode):""",
        """def build_main_layout(periodicity_options, periodicity_value, returns_type, vol_scaler,
                      active_tab, rolling_window, rolling_metric, rolling_return_type, rolling_chart_switch,
                      drawdown_chart_switch, growth_chart_switch, monthly_view, monthly_series,
                      monthly_series_options, monthly_select_disabled):""",
    )

    # ── build_main_layout call site ────────────────────────────────────
    result_text = result_text.replace(
        """            children=build_main_layout(
                periodicity_options=[{"value": "daily", "label": "Daily"}],
                periodicity_value="daily",
                returns_type="total",
                vol_scaler=0,
                active_tab="statistics",
                rolling_window="1y",
                rolling_metric="total_return",
                rolling_return_type="annualized",
                rolling_chart_switch="chart",
                drawdown_chart_switch="chart",
                growth_chart_switch="chart",
                monthly_view="annual",
                monthly_series=None,
                monthly_series_options=[],
                monthly_select_disabled=True,
                factor_mode="box",
                factor_quantiles=5,
                factor_transform="raw",
                factor_series_options=[],
                factor_series_value=None,
                factor_qq_reference="normal",
                conditional_view="forward",
                conditional_comparator="le",
                conditional_threshold=0,
                conditional_window_conversion="compound",
                conditional_step=1,
                conditional_step_unit="months",
                conditional_display_mode="summary",
                regime_display_mode="summary",
            ),""",
        """            children=build_main_layout(
                periodicity_options=[{"value": "daily", "label": "Daily"}],
                periodicity_value="daily",
                returns_type="total",
                vol_scaler=0,
                active_tab="statistics",
                rolling_window="1y",
                rolling_metric="total_return",
                rolling_return_type="annualized",
                rolling_chart_switch="chart",
                drawdown_chart_switch="chart",
                growth_chart_switch="chart",
                monthly_view="annual",
                monthly_series=None,
                monthly_series_options=[],
                monthly_select_disabled=True,
            ),""",
    )

    # Remove "Edit factors..." and "Edit regimes..." menu items + divider
    result_text = result_text.replace(
        """                                        dmc.MenuItem(
                                            "Edit factors...",
                                            id="at-menu-add-factor",
                                            leftSection=DashIconify(icon="tabler:math-function", width=14),
                                        ),
                                        dmc.MenuItem(
                                            "Edit regimes...",
                                            id="at-menu-add-regime",
                                            leftSection=DashIconify(icon="tabler:binary-tree-2", width=14),
                                        ),
                                        dmc.MenuDivider(),""",
        "",
    )

    # Fix outputs = [no_update] * 21 -> * 9
    result_text = result_text.replace(
        "outputs = [no_update] * 21",
        "outputs = [no_update] * 9",
    )

    # Fix the except branch no_update counts in restore_application_state
    # Original has: no_update x many across multiple lines. After removing 12 outputs,
    # we need fewer. But this is tricky because the format is spread across lines.
    # The simplest fix is to leave the except branch outputting the right number of no_updates.
    # Original total outputs = 23 (periodicity_options through False)
    # We're removing: 4 factor + 7 conditional + 1 regime = 12 outputs
    # New total outputs = 23 - 12 = 11
    # Current except return: (periodicity_options, valid_periodicity, valid_returns, valid_vol, active_tab, [18 no_updates], valid_selection, updated_order, False)
    # The 18 no_updates = 6 rolling + 1 dd + 1 growth + 4 factor + 7 conditional + 1 regime + 1 monthly = 21 minus the 3 named outputs
    # Actually let me recount. Let me just fix this in the output.
    # The old except return line has lots of no_update's. We need 6 fewer lines of no_updates.
    # Let me handle this by replacing the specific except block.
    result_text = result_text.replace(
        """            no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update,
            no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update,
            no_update,
            resolved["valid_selection"], resolved["updated_order"], False""",
        """            no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update,
            resolved["valid_selection"], resolved["updated_order"], False""",
    )

    # Fix outputs index references in at_restore_secondary_controls
    # With 12 fewer outputs, the indices change:
    # Old: rolling=0:6, drawdown=6, growth=7, factor=8:12, conditional=12:19, regime=19, calendar=20
    # New: rolling=0:6, drawdown=6, growth=7, calendar=8
    result_text = result_text.replace(
        'outputs[20] = _no_update_if_equal(resolved["monthly_view"], current_monthly_view)',
        'outputs[8] = _no_update_if_equal(resolved["monthly_view"], current_monthly_view)',
    )

    # Clean up excessive blank lines (more than 2 consecutive)
    result_text = re.sub(r'\n{4,}', '\n\n\n', result_text)

    # Write output
    SRC.write_text(result_text, encoding="utf-8")

    deleted_count = len(to_delete)
    final_count = result_text.count('\n') + 1
    print(f"Deleted {deleted_count} lines. File went from {len(lines)} to {final_count} lines.")


if __name__ == "__main__":
    main()
