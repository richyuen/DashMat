# PortOpt Rebuild Audit

This note tracks the PortOpt parity audit for the `main-rebuild-from-50c2337` branch.

## Audit Ledger

| Test / behavior | Status | Rationale |
| --- | --- | --- |
| `test_po_sync_reporting_basis_control_disables_when_ineligible` | done | Rebuild includes reporting-basis control, eligibility gating, restore, and help text. |
| `test_po_run_optimization_stores_split_reporting_payloads` | done | Results now store reporting vs optimization return payloads plus `reporting_basis`. |
| `test_po_render_statistics_uses_result_rf_setting_not_live_toggle` | done | Statistics uses result-level RF state instead of live page controls. |
| `test_po_render_statistics_uses_stored_portfolio_benchmark` | done | Statistics uses stored reporting and benchmark result payloads when present. |
| `test_po_render_rolling_uses_result_rf_setting_not_live_toggle` | done | Rolling uses result-level RF state instead of live page controls. |
| `test_po_download_excel_respects_tab_order_and_frontier_weights` | done | Export/settings parity includes tab order, reporting-basis, and result RF metadata. |
| `test_sync_po_returns_basis_from_mirrors_updates_canonical` / `test_sync_po_returns_basis_mirrors_only_updates_mismatched` | done | Rebuild now has the canonical returns-basis control, session store, and mirror sync for Returns, Calendar Year, and Drawdown. |
| `test_po_render_returns_uses_excess_basis_frame` | done | Returns now switches between total and excess output using the rebuild returns-basis store. |
| `test_po_render_calendar_passes_returns_basis_to_shared_helper` | done | Calendar Year now passes `total` vs `excess` through with the selected portfolio benchmark mapping. |
| `test_po_render_drawdown_passes_returns_basis_to_shared_helper` | done | Drawdown now passes `total` vs `excess` through with the selected portfolio benchmark mapping. |
| Returns-basis Excel Settings parity | done | Export Settings now records the active `Returns Type (Export)` value instead of hardcoding `total`. |
| Dataset-key/result-cache plumbing | internal-only | Current main uses a broader architecture that is not required for visible parity on the rebuild. |
| Post-baseline bootstrap/readiness machinery | internal-only | The rebuild intentionally keeps the simpler baseline entry and restore flow. |

## Timing References

- Product baseline: `50c2337` 5-run confirm on 2026-03-15
  - Analytics `1030 ms`
  - PortOpt `1441 / 1865 / 2111 ms` for `ready / restoredTabReady / weightsReady`
  - Regression `610 ms`
- Latest substantive rebuild PortOpt checkpoint: 3-run stored-render pass on 2026-03-15 against the `6c7a8a0` worktree state
  - Artifact label still carries the prior committed ref: `warm_switch_2026-03-15T15-27-05_portopt-stored-render_9d3a5232_p8060.json`
  - Analytics `1025 ms`
  - PortOpt `1479 / 1986 / 2264 ms`
  - Regression `660 ms`
- Returns-basis render bundle 3-run pass on 2026-03-15 against the rebuild worktree state before commit
  - `warm_switch_2026-03-15T16-42-01_returns-basis-3run-rebuild_8c719ce_p8060.json`
  - Analytics `1192 ms`
  - PortOpt `1582 / 2002 / 2322 ms`
  - Regression `696 ms`
  - Interpretation: broad all-page slowdown relative to baseline looked environment-skewed rather than PortOpt-specific, so a fresh 5-run rerun was required before deciding on the bundle.
- Merge-readiness 5-run confirmation on 2026-03-15 against `f254ec0`
  - First pass: `warm_switch_2026-03-15T15-36-00_rebuild-merge-ready_f254ec05_p8060.json`
    - Analytics `1029 ms`
    - PortOpt `1555 / 2047 / 2330 ms`
    - Regression `632 ms`
  - Rerun for edge-case confirmation: `warm_switch_2026-03-15T15-37-02_rebuild-merge-ready-rerun_f254ec05_p8060.json`
    - Analytics `1136 ms`
    - PortOpt `1505 / 1969 / 2279 ms`
    - Regression `630 ms`
  - Interpretation: PortOpt remained inside the target band on rerun; small Analytics drift between the two 5-run passes looked like environment noise rather than a PortOpt-specific regression.
- Returns-basis closeout 5-run confirmation on 2026-03-15 against the rebuild worktree state before commit
  - `warm_switch_2026-03-15T16-43-11_returns-basis-5run-rebuild_8c719ce_p8060.json`
  - Baseline comparison: `warm_switch_2026-03-15T16-43-11_returns-basis-5run-baseline_50c2337_p8061.json`
  - Analytics `1051 ms` vs baseline `1084 ms`
  - PortOpt `1546 / 1990 / 2292 ms` vs baseline `1467 / 1895 / 2164 ms`
  - Regression `616 ms` vs baseline `566 ms`
  - Interpretation: the rebuild stayed within the agreed `+10%` band on all tracked medians after the returns-basis bundle.

## Stable Rebuild Rules

- Use current-main focused PortOpt tests as the parity checklist.
- Do not treat noisy 1-run all-page timing outliers as blockers by themselves.
- Run 3-run browser timing after substantive PortOpt render-behavior changes.
- If a 3-run pass slows down across untouched pages too, restart fresh and confirm with a 5-run before treating it as a real PortOpt regression.
- Pure export/settings parity does not require browser timing.
- Keep Playwright artifacts under `output/` untracked.
