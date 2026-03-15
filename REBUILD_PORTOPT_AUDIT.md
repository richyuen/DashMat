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
| Excess-basis returns behavior from current main returns-basis surface | deferred | The rebuild does not have the returns-basis UI/state surface, so porting this renderer behavior alone would create an orphan hidden feature. |
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

## Stable Rebuild Rules

- Use current-main focused PortOpt tests as the parity checklist.
- Do not treat noisy 1-run all-page timing outliers as blockers by themselves.
- Run 3-run browser timing after substantive PortOpt render-behavior changes.
- Pure export/settings parity does not require browser timing.
- Keep Playwright artifacts under `output/` untracked.
