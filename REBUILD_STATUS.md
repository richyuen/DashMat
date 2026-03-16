# Rebuild Status

This note tracks the current status of `main-rebuild-from-50c2337` after account-list parity and the latest warm-switch optimization work.

## Current State

- Branch: `main-rebuild-from-50c2337`
- Current committed head: `ca6e24f` `Reduce account list shell warm-switch cost`
- Current local tracked changes:
  - `utils/account_list_modal.py`
  - `tests/unit/test_account_list_modal.py`
- Purpose of the local changes:
  - remove the always-on account-list provenance refresh callback
  - prune provenance lazily from current raw data when the account-list modal opens and when save/load flows actually need it

## Timing Checkpoints

### Normalized Pre-Account-List Checkpoint

- Snapshot: `267ade5` plus the minimal PortOpt store normalization required for harness compatibility
- Artifact: `warm_switch_2026-03-15T18-48-53_pre-account-list-storefix-5run_267ade5_p8050.json`
- Medians:
  - Analytics `1290 ms`
  - PortOpt `1613 / 2204 / 2466 ms`
  - Regression `637 ms`

### Current Rebuild Before Lazy Provenance

- Artifact: `warm_switch_2026-03-15T20-30-02_rebuild-accountlist-gated-5run_worktree_p8050.json`
- Medians:
  - Analytics `1444 ms`
  - PortOpt `1738 / 2401 / 2682 ms`
  - Regression `773 ms`

### Current Rebuild With Lazy Provenance

- Artifact: `warm_switch_2026-03-15T22-58-29_provenance-landed-5run_ca6e24f_p8050.json`
- Medians:
  - Analytics `1329 ms`
  - PortOpt `1562 / 2444 / 2725 ms`
  - Regression `741 ms`

### `50c2337` Reference

- Artifact: `warm_switch_2026-03-15T20-31-45_baseline-accountlist-gated-50c2337_50c2337_p8050.json`
- Medians:
  - Analytics `1033 ms`
  - PortOpt `1439 / 1878 / 2113 ms`
  - Regression `561 ms`

## Findings

- The lazy-provenance change is a keepable `5-run` improvement:
  - Analytics `1444 -> 1329`
  - PortOpt ready `1738 -> 1562`
  - Regression `773 -> 741`
  - PortOpt restored and weights moved slightly the wrong way (`2401 -> 2444`, `2682 -> 2725`), but the deltas are small enough to treat as non-material relative to the broader win
- The rebuild is still slower than the normalized pre-account-list checkpoint and `50c2337`, especially on PortOpt restored/weights and Regression.
- Current `main` remains slower than `50c2337`, but `50c2337` remains the rebuild performance bar.
- Rebuild guidance is now:
  - use `5-run` warm-switch passes for timing decisions
  - keep clear `5-run` improvements even if they do not solve the whole regression, provided they do not materially regress the other tracked metrics

## Stable Rebuild Notes

- Keep rebuild timing artifacts under `output/` untracked.
- Use `REBUILD_PORTOPT_AUDIT.md` for feature-parity history.
- Use this file for rebuild-wide timing checkpoints and current-state conclusions.
