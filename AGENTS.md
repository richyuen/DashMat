# AGENTS.md

Repo instructions for coding agents working in `C:\Git\DashMat`.

## Purpose

DashMat is a Dash app for market returns workflows with three primary pages:
- `pages/analyticstool.py`
- `pages/portopt.py`
- `pages/regression.py`

Prefer small, targeted changes unless a behavior change is explicitly requested.

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

## Working Rules

- Preserve callback IDs and store schemas unless a migration is intentional and updated everywhere.
- Keep shared JSON/store payloads compatible across pages.
- Avoid broad refactors in large callback files; patch the smallest safe section.
- Do not add dependencies unless necessary.
- Add comments only when logic is not obvious.
- Do not mutate or delete database table data from runtime callback code.
- Database setup, backfills, truncates, deletes, reseeds, and migrations belong in explicit scripts under `tools/db`.
- Exception: AnalyticsTool factor-definition CRUD is allowed at runtime for `FactorDefinitions` and `FactorDefinitionsArchive` only. Archive the prior row first and use optimistic concurrency via `UPDATE_DATE`.

Portfolio import rules:
- `peer` reads `PeerTS` using `PortRet` plus `MeanRet`.
- `index` reads `IndexTS` using `PortRet` plus `Benchmark`.
- `other` currently supports `PortfolioVintage='AltTS'` only and reads `AltTS`.
- In `other` + `AltTS`, benchmark lookup uses `Portfolio=<Portfolio>` and `Item='BenchRet'`, and benchmark series are named `<Portfolio>_BM`.

## Data Expectations

- Inputs are date-indexed return series in CSV/XLS/XLSX.
- Values may be decimals or percent-formatted.
- Daily data may be resampled to weekly or monthly.
- Monthly data must not be upsampled.
- Appends must preserve existing dataset periodicity rules.

## Validation

- Start the app if routing or layout behavior changed.
- Run targeted pytest modules for touched logic; run full pytest for broader workflow changes.
- If you change optimization logic, run `tests/scripts/test_optimization_scripts.py` or full pytest.
- If you change upload, parsing, or statistics flows, run full pytest and do a quick manual pass in `/analyticstool`.
- Before finishing, check for obvious regressions in tab rendering and series selection behavior.

## Performance Learnings

- Warm-switch performance must be judged with a browser timing pass, not just unit tests or callback-level reasoning.
- Current practical harness: load `/analyticstool`, import AA Tool database series, confirm the series-selection modal, warm `/portopt` and `/regression`, then measure warm revisits.
- Default warm-up series should use actual DB option keys such as `SPX_TRIndex`, `R2000_TRIndex`, `EAFE_TRIndex`, and `BCTBill13_TRIndex`, not display shorthand like `SPX`.
- For date-range controls, keep `Common Daily` candidate computation off the date-range initialization path. Compute candidates separately and use a small shared clientside disabled-state helper so button availability does not retrigger picker/store initialization.
- Track at least:
  - `shellMs`: main container visible
  - `readyMs`: periodicity control visible and enabled
- Regression is the warm-switch reference. PortOpt is the main bottleneck; AnalyticsTool is secondary.
- Recent failed experiments:
  - optimistic clientside restore/reconciliation for AT/PO did not improve warm-switch timing enough to justify the extra complexity
  - lazy-mounting the whole PO ex-ante grid subtree regressed timing materially
- Preferred next direction: narrower PO-only experiments, one hidden subtree at a time, with remeasurement against committed baseline before keeping the change.

## Windows and Tooling Learnings

- On Windows, very large `apply_patch` payloads can fail with shell or path-length style errors. Split large doc rewrites or multi-file edits into smaller per-file patches.
- Prefer Python Playwright over `playwright-cli run-code` for nontrivial browser automation on Windows. The CLI JS path runs into command-line length and quoting limits quickly.
- Prefer real script files over `conda run ... python -c` for anything more than a short one-liner. Multiline or heavily quoted `-c` payloads are brittle.
- In PowerShell, `Start-Process` with `python -c` is easy to misquote. Use a script file when possible, or pass the full `-c "..."` payload as one argument string.
- For browser file uploads in Playwright, prefer normalized forward-slash paths such as `C:/Git/DashMat/...` when passing paths into browser-side code.
- Keep Playwright runtime artifacts out of commits. `.playwright-cli/` and `output/` are local runtime outputs unless a specific artifact is intentionally being checked in.
- If you need to compare two commits side by side, run the app on separate ports instead of editing `app.py`. A reliable pattern is `conda run -n dashmat python -c "import app; app.app.run(port=8051)"`.
- Fresh git worktrees may have missing or zero-byte SQLite files under `data/`. Validate or rebuild the local seed DBs before starting DB-backed browser runs.
- For side-by-side A/B comparisons, be explicit about which repo root owns the app process, DB files, and output artifacts. Launch the app after that repo root's seed DBs are valid.
- The warm-switch harness currently accepts runs that may include browser console callback errors. Treat single-run results cautiously and prefer repeated A/B runs before concluding that a small regression is real.
