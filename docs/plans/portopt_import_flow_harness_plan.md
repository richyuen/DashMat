# Plan: PortOpt import-flow harness

## Goal
Add a repeatable Playwright harness that measures the **real PortOpt import callback family end-to-end**, so touched-flow perf can be validated directly instead of inferred from AnalyticsTool.

## Why this is needed
The shared import-flow refactor touched both AnalyticsTool and PortOpt import/database-add business logic, but only AnalyticsTool currently has an existing real-flow harness (`at_statistics_harness.py --mode imports`).

Current PortOpt coverage gap:
- `portopt_series_modal_harness.py` measures the series-selection modal after data is already available, not the import callback family itself
- `ui_callback_interaction_harness.py` covers modal/UI micro-scenarios, not import-to-ready behavior

Result:
- PortOpt import perf cannot currently be certified with the same standard used for AnalyticsTool

## Recommended shape
Create a **new dedicated harness** rather than forcing the missing flow into the existing modal harness.

Recommended files:
- `tools/playwright/portopt_import_harness.py`
- `tools/playwright/portopt_import_harness.ps1`

Why a new harness:
- the measured unit is different from the current modal harness
- import-flow setup, ready criteria, and timing windows are distinct
- keeping a dedicated harness avoids mixing modal-edit timing with import-to-ready timing

## Flow to measure
Measure the touched PortOpt import paths that map to the shared helper extraction:

1. **DB add flow**
   - open PortOpt welcome/page state
   - trigger DB add modal
   - choose benchmark series
   - click `po-db-add-ok-button`
   - wait for series-selection modal
   - click `po-modal-ok-button`
   - wait for stable post-import ready state

2. **Portfolio add flow**
   - open portfolio add modal
   - stage one portfolio row
   - click `po-portfolio-add-ok-button`
   - wait for series-selection modal
   - click `po-modal-ok-button`
   - wait for stable post-import ready state

3. **Raw DB add flow** *(optional second phase if needed)*
   - open raw DB modal
   - stage one row
   - click `po-raw-db-add-ok-button`
   - wait for stable post-import ready state

4. **Underlying add flow** *(optional second phase if needed)*
   - open underlying modal
   - stage one row
   - click `po-underlying-add-ok-button`
   - wait for stable post-import ready state

## Minimum v1 recommendation
Ship v1 with:
- **DB add flow**
- **Portfolio add flow**

Reason:
- they exercise the shared import-family boundary with real user-visible flow
- they are the most directly comparable to the existing AnalyticsTool imports harness
- they keep the first PortOpt perf proof small and repeatable

## Fixture strategy
Make fixture selection deterministic inside the harness; do not depend on ad hoc manual choices.

### DB add fixtures
- Open the DB add modal and read the live `po-db-add-series-select` options.
- Sort by stable label/value order and take the first usable fixed subset for the run.
- Reuse the same discovered subset for warmup, rehearsal, and measured runs.
- Fail fast if the discovered option set is too small for the planned run count.

### Portfolio add fixtures
- Open the portfolio add modal in each supported v1 mode and read the live portfolio/type options needed to stage a valid row.
- Resolve a deterministic peer fixture set and a deterministic index fixture set from those live options, again using stable sorted order.
- Persist the resolved fixture list in the output JSON so the exact run inputs are recoverable.
- Fail fast if the page cannot discover enough valid fixtures to run the planned pass.

This keeps the harness deterministic while still grounded in the real local DB-backed UI state.

## Timing windows
Capture at least:

### DB add
- `dbImportToSeriesModalMs`
  - click DB modal OK -> series-selection modal visible
- `dbSeriesConfirmToReadyMs`
  - click series modal OK -> PortOpt ready

### Portfolio add
- `portfolioImportToSeriesModalMs`
  - click portfolio add OK -> series-selection modal visible
- `portfolioSeriesConfirmToReadyMs`
  - click series modal OK -> PortOpt ready

### Whole-run
- `totalRunMs`
- `dashUpdateRequestCount`
- `dashUpdateTotalMs`
- `dashUpdateRequestBytes`
- `dashUpdateResponseBytes`
- top callback outputs by frequency

## Ready criteria
Use a **real page-ready outcome**, not just modal disappearance.

Recommended ready checks:
- `#po-main-container` visible
- `#po-ui-blocker-overlay` hidden/absent
- `#po-open-modal-button` ready again
- `po-series-select` store/session value includes the imported series
- `dashmat-raw-data-store` is populated with a new/current dataset payload
- `dashmat-original-periodicity-store` is populated
- `#po-periodicity-select` is enabled and has a stable value
- for DB add runs, `po-cmabench-defaults-store` is populated for imported series when that flow is expected to seed defaults

Do **not** use “modal closed” alone as success.
Do **not** require a result-tab render in v1; import flows should be judged by stable **data-ready** state, not by downstream optimization/render work.

## Reuse from existing harnesses
Borrow heavily from:
- `tools/playwright/at_statistics_harness.py`
  - timed-server pattern
  - run summaries
  - smoke then 5-run A/B workflow
- `tools/playwright/warm_switch_harness.py`
  - Dash request tracker
  - wait helpers
  - DB fixture helpers
  - timing-log parsing/copy helpers
- `tools/playwright/portopt_series_modal_harness.py`
  - PortOpt page seeding
  - modal interaction helpers
  - PortOpt-specific ready/wait behavior

## Output contract
Write JSON to `output/playwright/` with:
- harness label
- git ref
- base URL
- runs
- per-run raw timings
- median summary
- request/bytes summary
- top callback outputs
- copied timed-server stdout path

## A/B procedure
Use the same discipline as other perf phases:
- timed servers on isolated ports
- `1`-run smoke first
- `5`-run passes only after smoke succeeds
- compare clean `HEAD` vs worktree serially
- judge by browser medians, not just callback server timing

## Acceptance criteria
- Harness can run against clean `HEAD` and worktree without editing app config
- Harness measures at least DB add and portfolio add real flows end-to-end
- Harness emits stable per-run medians and callback/request summaries
- Harness records the resolved deterministic fixture set used for the run
- Harness uses explicit data-ready checks rather than modal disappearance alone
- Harness is usable for future import-flow A/B checks on PortOpt

## Non-goals
- no attempt to cover every PortOpt modal in v1
- no extension of the harness into unrelated result-tab perf paths
- no replacement of the existing modal harness

## Suggested follow-up sequence
1. implement `portopt_import_harness.py`
2. add wrapper `.ps1`
3. add a small doc/test note asserting the new harness exists
4. run `1`-run smoke on clean `HEAD` and worktree
5. rerun the PortOpt side of the import-flow perf gate with a real A/B
