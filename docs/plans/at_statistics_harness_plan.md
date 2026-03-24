# AT Statistics Harness With Import and Account-List Modes

## Summary

- Keep this as a new dedicated AT harness rather than extending `tools/playwright/warm_switch_harness.py`, but reuse its AT waiters, Dash request tracker, timed-log handling, and failure artifact helpers.
- Use one canonical list of 5 deterministic run specs, where each spec contains:
  - one peer-relative portfolio imported as `Actual + Estimated`
  - one index-relative portfolio imported as `Actual + Benchmark`
- Support two measurement modes against those same 5 specs:
  - `imports`: clear session, re-import the peer/index pair through the AT UI
  - `account-list`: clear session, load a seeded account list containing that same peer/index pair
- Add one unmeasured rehearsal before the 5 measured runs in each mode.

## Key Changes

- Create `tools/playwright/at_statistics_harness.py`.
  - Add `--mode imports|account-list`, defaulting to `imports`.
  - Reuse existing warm-switch helpers for:
    - AA DB warmup
    - `at-state-ready-store` waiting
    - statistics-idle waiting
    - Dash request/bytes/callback attribution
  - Warmup sequence for both modes:
    - open `/analyticstool`
    - run the existing AA DB retrieval once
    - wait for Statistics ready
  - Add one unmeasured rehearsal:
    - `imports` mode: run the first deterministic peer/index import pair once, then clear session
    - `account-list` mode: load the first seeded deterministic account list once, then clear session
- Define a shared deterministic run-spec builder in Python.
  - Resolve 5 eligible peer portfolios with benchmarks from DB-backed options.
  - Resolve 5 eligible index portfolios with benchmarks from DB-backed options.
  - Pair them into 5 stable run specs in sorted order.
  - Fail fast if fewer than 5 eligible peer or index candidates exist.
- Extend DB seeding in `tools/db`.
  - Seed at least 5 deterministic peer portfolios with daily history, valid inception dates, and `Actual + Estimated` benchmark coverage.
  - Seed at least 5 deterministic index portfolios with daily history and `Actual + Benchmark` coverage.
  - Keep all harness portfolios on roughly the same 10-year daily history window so dataset size is consistent across runs.
- Add direct-import measurement flow.
  - Before each run:
    - wait for Dash settle
    - clear `sessionStorage`
    - reload `/analyticstool`
    - wait for the AT welcome screen
  - Measured run:
    - import the run spec's peer portfolio
    - confirm series-selection modal
    - wait for Statistics ready
    - import the run spec's index portfolio
    - confirm series-selection modal
    - wait for Statistics ready
- Add account-list mode.
  - Do not seed account-list rows in `tools/db`; create them at harness startup under the current AT username.
  - Build the 5 account lists directly in Python using `add_db_import_provenance_entry`, `build_account_list_payload`, and `save_account_list` from `utils/account_lists.py`.
  - Before creating them, delete current-user lists with the harness prefix so the fixture set is deterministic and load selection is unambiguous.
  - Each saved account list should contain exactly one deterministic run spec's peer/index pair, with selected/order/benchmark control values matching the import mode outcome.
  - Measured run:
    - clear `sessionStorage`
    - reload `/analyticstool`
    - open welcome-screen account-list load
    - load the corresponding deterministic account list
    - wait for Statistics ready
- Add `tools/playwright/at_statistics_harness.ps1`.
  - Mirror the existing timed harness wrapper pattern.
  - Keep `-Runs`, `-Label`, `-BaseUrl`, `-Headed`, `-ServerLog`, and add `-Mode`.

## Public Interfaces

- New CLI harness:
  - `tools/playwright/at_statistics_harness.py`
  - Args: `--base-url`, `--runs`, `--label`, `--headed`, `--server-log`, `--mode`
- New wrapper:
  - `tools/playwright/at_statistics_harness.ps1`
- Output JSON in `output/playwright/` should include:
  - `mode`
  - `warmup`
  - `runSpecs`
  - `accountListFixtures` when in account-list mode
  - per-run raw timings and medians

## Metrics

- Record per-run identifiers:
  - peer portfolio key
  - peer emitted primary series
  - peer benchmark series
  - index portfolio key
  - index emitted primary series
  - index benchmark series
  - account-list name/id in account-list mode
- Record granular timings:
  - `resetToWelcomeMs`
  - `peerImportToSeriesModalMs`
  - `peerSeriesConfirmToStatisticsReadyMs`
  - `indexImportToSeriesModalMs`
  - `indexSeriesConfirmToStatisticsReadyMs`
  - `accountListOpenToReadyMs` for account-list mode
  - `totalRunMs`
- Record Dash request count, total Dash time, request bytes, response bytes, and callback outputs for:
  - peer window
  - index window
  - full run
  - account-list load window

## Test Plan

- Add targeted unit tests for any new deterministic run-spec and account-list fixture builders.
- Add focused tests around account-list fixture payload generation so the saved lists match the import-mode selected/order/benchmark state.
- Run targeted pytest for touched helpers; if coverage would fail on a focused run, use `--cov-fail-under=0` and call that out.
- Run with a timed server via `tools/playwright/start_timed_server.ps1`:
  - 1-run smoke for `imports`
  - 5-run pass for `imports`
  - 1-run smoke for `account-list`
  - 5-run pass for `account-list`

## Assumptions

- The AT perf goal is Statistics-ready timing, not welcome-shell timing.
- "Same five deterministic sets" means the account-list mode uses the exact same 5 peer/index combinations as import mode.
- Account-list fixtures should be created at harness startup for the current username, not statically seeded in DB migrations.
- Session reset means clearing browser `sessionStorage` only; the timed server and browser process stay alive so warm caches remain comparable.
