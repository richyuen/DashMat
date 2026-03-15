# Rebuild Status

This note tracks the current status of the `main-rebuild-from-50c2337` branch after the account-list parity work.

## Current State

- Branch: `main-rebuild-from-50c2337`
- Last committed rebuild checkpoint: `619fbf1` `Port account list subsystem`
- Current local tracked changes not yet committed:
  - `pages/portopt.py`
  - `tests/callbacks/test_portopt_callbacks.py`
- Purpose of the local changes:
  - restore the canonical PortOpt series-config session-store block so the warm-switch harness can seed PortOpt on the rebuild
  - lock that regression with a focused callback/layout test

## Timing Checkpoints

### Normalized Pre-Account-List Checkpoint

- Snapshot: `267ade5` plus the minimal PortOpt store normalization required for harness compatibility
- Artifact: `warm_switch_2026-03-15T18-48-53_pre-account-list-storefix-5run_267ade5_p8050.json`
- Medians:
  - Analytics `1290 ms`
  - PortOpt `1613 / 2204 / 2466 ms` for `ready / restoredTabReady / weightsReady`
  - Regression `637 ms`

### Current Rebuild Versus `50c2337`

- Rebuild artifact: `warm_switch_2026-03-15T18-51-01_account-list-vs-baseline-current_working_p8050.json`
- Baseline artifact: `warm_switch_2026-03-15T18-52-28_account-list-vs-baseline-50c2337_50c2337_p8050.json`
- Rebuild medians:
  - Analytics `1418 ms`
  - PortOpt `1976 / 2615 / 2899 ms`
  - Regression `845 ms`
- Baseline medians:
  - Analytics `1089 ms`
  - PortOpt `1451 / 1868 / 2119 ms`
  - Regression `570 ms`

### Current `main` Versus `50c2337`

- Current `main` artifact: `warm_switch_2026-03-15T18-55-57_current-main-5run_7bfdb22_p8050.json`
- Baseline artifact: `warm_switch_2026-03-15T18-52-28_account-list-vs-baseline-50c2337_50c2337_p8050.json`
- Current `main` medians:
  - Analytics `1659 ms`
  - PortOpt `1701 / 2550 / 2948 ms`
  - Regression `945 ms`

## Findings

- The account-list-era rebuild state is materially slower than the old `50c2337` performance baseline.
- The normalized pre-account-list checkpoint is materially faster than the current account-list-enabled rebuild:
  - Analytics `1290` -> `1418`
  - PortOpt ready `1613` -> `1976`
  - PortOpt restored `2204` -> `2615`
  - PortOpt weights `2466` -> `2899`
  - Regression `637` -> `845`
- The uncommitted PortOpt series-config store repair fixes the warm-switch harness blockage, but it does not recover the larger account-list-era slowdown by itself.
- Current `main` remains much slower than `50c2337`, especially on Analytics and Regression.

## Stable Rebuild Notes

- Keep rebuild timing artifacts under `output/` untracked.
- Use the PortOpt audit in `REBUILD_PORTOPT_AUDIT.md` for feature-parity history.
- Use this file for rebuild-wide timing checkpoints and current-state conclusions.
