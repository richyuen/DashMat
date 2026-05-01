# Import-flow A/B note — 2026-04-08

## Scope
This note records the browser A/B run used to validate the shared import-flow refactor against an existing real-flow harness.

## Harness choice
- **Chosen harness:** `tools/playwright/at_statistics_harness.py --mode imports`
- **Why:** this is the only existing harness that exercises a real touched import flow end-to-end for this refactor family.
- **Not chosen:** current PortOpt harnesses (`portopt_series_modal_harness.py`, `ui_callback_interaction_harness.py`) do not measure the PortOpt import callback family itself. They cover series-modal or UI-only scenarios instead.

## Method
- Clean `HEAD` worktree on port `8061`
- Current worktree on port `8062`
- Timed servers launched with `tools/playwright/start_timed_server.ps1`
- Required `1`-run smoke completed on both sides first
- Required `5`-run A/B completed serially on isolated ports

## Commands used
```powershell
powershell -ExecutionPolicy Bypass -File tools\playwright\start_timed_server.ps1 -RepoRoot C:\Git\DashMat_ab_head -Port 8061 -LogStem ab_head_at_imports
powershell -ExecutionPolicy Bypass -File tools\playwright\start_timed_server.ps1 -RepoRoot C:\Git\DashMat -Port 8062 -LogStem ab_worktree_at_imports

powershell -ExecutionPolicy Bypass -File tools\playwright\at_statistics_harness.ps1 -RepoRoot C:\Git\DashMat_ab_head -BaseUrl http://127.0.0.1:8061 -Runs 1 -Label ab-head-smoke -GitRef HEAD -ServerLog C:\Git\DashMat_ab_head\output\playwright\ab_head_at_imports_stdout.log -Mode imports -SkipDbBuild
powershell -ExecutionPolicy Bypass -File tools\playwright\at_statistics_harness.ps1 -RepoRoot C:\Git\DashMat -BaseUrl http://127.0.0.1:8062 -Runs 1 -Label ab-worktree-smoke -GitRef WORKTREE -ServerLog C:\Git\DashMat\output\playwright\ab_worktree_at_imports_stdout.log -Mode imports -SkipDbBuild

powershell -ExecutionPolicy Bypass -File tools\playwright\at_statistics_harness.ps1 -RepoRoot C:\Git\DashMat_ab_head -BaseUrl http://127.0.0.1:8061 -Runs 5 -Label ab-head-5run -GitRef HEAD -ServerLog C:\Git\DashMat_ab_head\output\playwright\ab_head_at_imports_stdout.log -Mode imports -SkipDbBuild
powershell -ExecutionPolicy Bypass -File tools\playwright\at_statistics_harness.ps1 -RepoRoot C:\Git\DashMat -BaseUrl http://127.0.0.1:8062 -Runs 5 -Label ab-worktree-5run -GitRef WORKTREE -ServerLog C:\Git\DashMat\output\playwright\ab_worktree_at_imports_stdout.log -Mode imports -SkipDbBuild
```

## 5-run median comparison
| Metric | HEAD | Worktree | Delta |
| --- | ---: | ---: | ---: |
| `resetToWelcomeMedian` | 2431 ms | 2454 ms | +23 ms |
| `peerImportToSeriesModalMedian` | 2335 ms | 2290 ms | -45 ms |
| `peerSeriesConfirmToStatisticsReadyMedian` | 518 ms | 517 ms | -1 ms |
| `indexImportToSeriesModalMedian` | 2322 ms | 2312 ms | -10 ms |
| `indexSeriesConfirmToStatisticsReadyMedian` | 305 ms | 306 ms | +1 ms |
| `totalRunMedian` | 8021 ms | 8015 ms | -6 ms |

## Verdict
- **No material regression** on the exercised **AnalyticsTool touched import flow**
- Result is effectively **flat / slightly improved**
- The small welcome-reset increase (`+23 ms`) does not outweigh the flat-to-better measured import path and flat overall total median

## Important limitation
This run does **not** certify PortOpt import-flow performance.

Reason:
- no existing PortOpt harness currently measures the real import callback family end-to-end
- the current PortOpt harnesses cover modal-only or UI-only paths, not import-to-ready behavior for the touched callbacks

## Artifact paths
- `C:\Git\DashMat_ab_head\output\playwright\at_statistics_2026-04-08T07-53-08_ab-head-5run_imports_head.json`
- `C:\Git\DashMat\output\playwright\at_statistics_2026-04-08T07-54-11_ab-worktree-5run_imports_worktree.json`

## PR-ready summary
Used the existing AnalyticsTool imports harness as the closest real-flow A/B for the shared import-flow refactor. Clean `HEAD` vs worktree on isolated ports showed no material regression on the exercised touched import path (`totalRunMedian 8021 -> 8015 ms`). PortOpt still lacks an equivalent import-flow harness, so PortOpt import perf remains an explicit follow-up rather than an inferred pass.
