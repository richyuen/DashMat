# AGENTS.md

Repo instructions for coding agents working in `C:\Git\DashMat`.

## Purpose

DashMat is a Dash app for market returns workflows with three primary pages:
- `pages/analyticstool.py`
- `pages/portopt.py`
- `pages/regression.py`

Default to small, targeted changes unless the requested behavior change clearly requires something broader.

## Environment

- Python 3.11
- Always run commands in the `dashmat` Conda environment.
- In non-interactive shells, use `conda run -n dashmat ...`.

Common commands:

```bash
conda run -n dashmat python app.py
conda run -n dashmat python -m pytest -q tests
conda run -n dashmat python tools/db/init_local_cma_db.py
```

## Code Map

- `app.py`: app entry point, shared stores, Mantine provider
- `pages/analyticstool.py`: analytics workflows
- `pages/portopt.py`: portfolio optimization workflows
- `pages/regression.py`: regression workflows
- `utils/parsing.py`: file parsing and periodicity detection
- `utils/returns.py`: return conversions and compounding
- `utils/statistics.py`: metrics calculations
- `utils/optimization.py`: optimization engine

## Core Rules

- Preserve callback IDs and store schemas unless a migration is intentional and updated everywhere.
- Keep shared JSON/store payloads compatible across pages.
- Avoid broad refactors in large callback files; patch the smallest safe section.
- Do not add dependencies unless necessary.
- Add comments only when logic is not obvious.
- Do not mutate or delete database table data from runtime callback code.
- Database setup, backfills, truncates, deletes, reseeds, and migrations belong in explicit scripts under `tools/db`.
- Exception: AnalyticsTool factor-definition CRUD is allowed at runtime for `FactorDefinitions` and `FactorDefinitionsArchive` only. Archive the prior row first and use optimistic concurrency via `UPDATE_DATE`.

## Data Expectations

- Inputs are date-indexed return series in CSV/XLS/XLSX.
- Values may be decimals or percent-formatted.
- Daily data may be resampled to weekly or monthly.
- Monthly data must not be upsampled.
- Appends must preserve existing dataset periodicity rules.

## Portfolio Import Rules

- `peer` reads `PeerTS` using `PortRet` plus `MeanRet`.
- `index` reads `IndexTS` using `PortRet` plus `Benchmark`.
- `other` currently supports `PortfolioVintage='AltTS'` only and reads `AltTS`.
- In `other` + `AltTS`, benchmark lookup uses `Portfolio=<Portfolio>` and `Item='BenchRet'`, and benchmark series are named `<Portfolio>_BM`.

## Validation

- Start the app if routing or layout behavior changed.
- Run targeted pytest modules for touched logic; run full pytest for broader workflow changes.
- If you change optimization logic, run `tests/scripts/test_optimization_scripts.py` or full pytest.
- If you change upload, parsing, or statistics flows, run full pytest and do a quick manual pass in `/analyticstool`.
- Before finishing, check for obvious regressions in tab rendering and series selection behavior.

## Performance Guidance

### General

- Judge warm-switch performance with a browser timing pass, not just unit tests or callback-level reasoning.
- Current practical harness: load `/analyticstool`, import AA Tool database series, confirm the series-selection modal, warm `/portopt` and `/regression`, then measure warm revisits.
- Default warm-up series should use actual DB option keys such as `SPX_TRIndex`, `R2000_TRIndex`, `EAFE_TRIndex`, and `BCTBill13_TRIndex`, not display shorthand like `SPX`.
- Track at least:
  - `shellMs`: main container visible
  - `readyMs`: periodicity control visible and enabled
- Regression is the warm-switch reference. PortOpt is the main bottleneck; AnalyticsTool is secondary.

### Startup Measurement

- For PortOpt startup work, add a direct startup benchmark instead of relying only on the broad warm-switch harness.
- Useful PortOpt startup checkpoints:
  - shell visible
  - periodicity enabled
  - series-selection modal visible
  - series-selection grid hydrated
  - modal `OK` to hidden
  - run button enabled
- For targeted PortOpt startup benchmarking, prefer direct session seeding or direct store seeding over replaying the AnalyticsTool DB-import flow.
- For AnalyticsTool startup benchmarking, direct seeded routes can be flaky. Prefer a real-flow browser benchmark if the seeded route does not reproduce the same bootstrap path reliably.
- Treat browser A/B startup runs as contaminated if `shellMs` and `readyMs` both jump broadly along with later modal timings. That usually indicates environment or bootstrap noise, not a real regression in the change under test.
- A narrow render micro-benchmark can help for callback-specific experiments, but it does not replace browser A/B when deciding whether to keep a user-visible startup change.

### Date / Shell Initialization

- Keep `Common Daily` candidate computation off the date-range initialization path. Compute candidates separately and use a small shared clientside disabled-state helper so button availability does not retrigger picker/store initialization.
- For cold-load shell visibility, keep the page-load interval as a trigger even if welcome/main visibility is determined only from raw-data presence. Removing the trigger entirely can leave both containers at their initial `display:none` state on first load.
- PortOpt warm-switch and PortOpt first-visit startup are different problems. Warm-switch mostly targets restore and validation latency. First-visit startup is dominated by the series-selection modal render/apply path.

### Keep These Wins

- PortOpt startup:
  - show `po-main-container` / `po-welcome-screen` directly from `dashmat-raw-data-store` instead of delaying shell paint on `po-page-load-trigger`
  - use `po-restore-complete-store` to gate validation instead of treating `po-secondary-restore-ready-store` as restore completion
  - narrow first-visit series-selection work by using `dashmat-raw-data-meta-store.columns` before parsing full raw JSON
  - cache CMA default lookup by stable missing-series tuple and only resolve CMA defaults for selected missing series
- PortOpt tab/render:
  - lazy-mount heavy result subtrees for `Attribution`, `Frontier`, and `Risk`
  - cache the default Frontier snapshot at solve time and reuse one shared snapshot resolver across chart, table, and export
- Optimization engine:
  - native `minimize_variance` is materially faster and worth keeping
  - hybrid `risk_parity` is worth keeping only for unconstrained or box-bounded classical RP; keep Riskfolio for RP cases with UI linear constraints
- AnalyticsTool / Regression startup:
  - move AT and REG series-selection modal open/seed to one clientside callback
  - make AT and REG modal `OK` paths diff-aware so unchanged persisted outputs return `no_update`
  - use `agSelectCellEditor` for AT `Benchmark` and disable modal-grid row animation

### Guidance From Measurement

- AnalyticsTool benefited across shell, open, grid, `OK`, and content timings from the startup pass.
- Regression benefited mainly on shell, open, and grid timing; `OK` close was effectively flat, so future REG work should prioritize open-path latency before more `OK`-path tuning.
- For list-constrained fields, lighter editors are worth trying before deeper grid refactors.
- `agSelectCellEditor` was a keep for PortOpt `Benchmark` / `CMABench` and AnalyticsTool `Benchmark`.
- Do not assume editor simplification will help if the measured bottleneck is still before grid hydration.

### Avoid These Non-Wins

- Native `maximize_sharpe` was slower than the Riskfolio path and should stay on Riskfolio.
- Broad PortOpt post-solve artifact-family reuse for statistics, growth, rolling, calendar, and drawdown did not produce a real win and slightly regressed targeted benchmarks.
- Clientside PortOpt restore plus clientside/common/specialized run-button gating regressed warm-switch `runReady`.
- Converting intra-app module switch to true Dash in-app routing improved some UX aspects but regressed warm-switch timing enough on AnalyticsTool and Regression that it was not kept.
- Optimistic clientside restore/reconciliation for AT/PO did not improve warm-switch timing enough to justify the added complexity.
- Lazy-mounting the whole PO ex-ante grid subtree regressed timing materially.

### Preferred Direction

- Prefer narrower PO-only experiments, one hidden subtree at a time, with remeasurement against the committed baseline before keeping a change.
- Recent blocker tuning: page-local startup blockers can help perceived first-switch timing, but release conditions must stay aligned with modal/grid hydration. If the blocker misbehaves, check the modal-open path and `virtualRowData` release signal before adding more blocker layers.

## Windows and Tooling Learnings

- On Windows, very large `apply_patch` payloads can fail with shell or path-length style errors. Split large doc rewrites or multi-file edits into smaller patches.
- Prefer Python Playwright over `playwright-cli run-code` for nontrivial browser automation on Windows. The CLI JS path runs into command-line length and quoting limits quickly.
- Prefer real script files over `conda run ... python -c` for anything more than a short one-liner. Multiline or heavily quoted `-c` payloads are brittle.
- In PowerShell, `Start-Process` with `python -c` is easy to misquote. Use a script file when possible, or pass the full `-c "..."` payload as one argument string.
- If you must launch long-lived Dash apps from PowerShell for A/B testing, wrapping the Conda-env Python invocation in a short `pwsh -Command` string is more reliable than passing `python -c` directly through `Start-Process`.
- For browser file uploads in Playwright, prefer normalized forward-slash paths such as `C:/Git/DashMat/...` when passing paths into browser-side code.
- Keep Playwright runtime artifacts out of commits. `.playwright-cli/` and `output/` are local runtime outputs unless a specific artifact is intentionally being checked in.
- If you need to compare two commits side by side, run the app on separate ports instead of editing `app.py`. A reliable pattern is `conda run -n dashmat python -c "import app; app.app.run(port=8051)"`.
- Fresh git worktrees may have missing or zero-byte SQLite files under `data/`. Validate or rebuild the local seed DBs before starting DB-backed browser runs.
- For side-by-side A/B comparisons, be explicit about which repo root owns the app process, DB files, and output artifacts. Launch the app after that repo root's seed DBs are valid.
- The warm-switch harness currently accepts runs that may include browser console callback errors. Treat single-run results cautiously and prefer repeated A/B runs before concluding that a small regression is real.
