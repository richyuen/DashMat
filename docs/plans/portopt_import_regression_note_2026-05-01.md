# PortOpt import-flow A/B regression investigation

## Scope
Investigates the apparent localized regression from the initial PortOpt import-flow A/B:

- `indexPortfolioSeriesConfirmToReady`: `989 -> 1092 ms` (`+103 ms`)

## Root cause
The regression was a harness phase-boundary attribution issue, not an app-code import regression.

The original harness used `DashUpdateRequestTracker.wait_for_settle()`, which only waits for Dash requests that have already started. On the index portfolio path, follow-up callbacks from the portfolio import phase can start just after the series-selection modal becomes visible. Depending on scheduling, those requests were attributed either to:

- the `indexPortfolioImportWindow`, or
- the next `indexPortfolioConfirmWindow`.

That made the old clean-HEAD run look faster in two of five `indexPortfolioSeriesConfirmToReady` samples (`638 ms`, `736 ms`) because some import follow-up work had landed in the previous window before confirm timing began.

## Fix
`tools/playwright/portopt_import_harness.py` now waits for a short Dash quiet period before/after phase windows via `_wait_for_dash_quiet(...)`. This stabilizes boundaries without changing app behavior.

## Stabilized 5-run A/B
Same harness revision against both old baseline app and current worktree app:

| Metric | Baseline | Worktree | Delta |
| --- | ---: | ---: | ---: |
| `resetToWelcomeMs` | 2076 | 2071 | -5 |
| `dbImportToSeriesModalMs` | 786 | 739 | -47 |
| `dbSeriesConfirmToReadyMs` | 630 | 615 | -15 |
| `peerPortfolioImportToSeriesModalMs` | 347 | 362 | +15 |
| `peerPortfolioSeriesConfirmToReadyMs` | 538 | 532 | -6 |
| `indexPortfolioImportToSeriesModalMs` | 355 | 355 | 0 |
| `indexPortfolioSeriesConfirmToReadyMs` | 533 | 531 | -2 |
| `totalRunMs` | 11090 | 11018 | -72 |
| `dashUpdateRequestCountMedian` | 42 | 42 | 0 |
| `dashUpdateTotalMsMedian` | 7909 | 7852 | -57 |
| `dashUpdateRequestBytesMedian` | 1498737 | 1498737 | 0 |
| `dashUpdateResponseBytesMedian` | 148245 | 148240 | -5 |

## Verdict
No `indexPortfolioSeriesConfirmToReady` regression remains under stabilized phase boundaries (`533 -> 531 ms`). The original `+103 ms` signal was measurement contamination from queued import follow-up callbacks crossing timing windows.

## Artifacts

Initial mixed result:

- `C:\Git\DashMat\output\playwright\portopt_import_20260408_232454_portopt-import-head-5run_head_p8061.json`
- `C:\Git\DashMat\output\playwright\portopt_import_20260408_232613_portopt-import-worktree-5run_worktree_p8062.json`

Stabilized rerun:

- `C:\Git\DashMat\output\playwright\portopt_import_20260501_070319_portopt-regression-quiet-head-5run_head-bas_p8061.json`
- `C:\Git\DashMat\output\playwright\portopt_import_20260501_070450_portopt-regression-quiet-work-5run_worktree_p8062.json`
