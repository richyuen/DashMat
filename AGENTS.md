# AGENTS.md

DashMat repo guide for `C:\Git\DashMat`.

## Scope

- Main pages: `pages/analyticstool.py`, `pages/portopt.py`, `pages/regression.py`.
- Prefer the smallest safe patch.

## Environment

- Python `3.11`; always use the `dashmat` Conda env.
- Non-interactive shells: `conda run -n dashmat ...`
- Common commands:

```bash
conda run -n dashmat python app.py
conda run -n dashmat python -m pytest -q tests
conda run -n dashmat python tools/db/init_local_cma_db.py
```

## Guardrails

- Preserve callback IDs, store schemas, and shared JSON/store payload compatibility.
- Preserve imported series names exactly; do not sanitize or alias them for grids/charts.
- For AG Grid literal dotted series names, set `dashGridOptions.suppressFieldDotNotation = True`.
- In large callback files, patch the smallest safe section.
- For shared app-shell features, prefer helper modules with `build_*_components()` and `register_*_callbacks()` instead of growing `app.py`.
- Do not mutate runtime DB tables from callbacks.
- DB setup, migrations, backfills, truncates, deletes, and reseeds belong in `tools/db`.
- Exception: AnalyticsTool factor-definition CRUD may update `FactorDefinitions` and `FactorDefinitionsArchive` with archive-first optimistic concurrency via `UPDATE_DATE`.

## Validation

- Routing/layout changes: start the app.
- Upload/parsing/statistics changes: run full pytest and do a quick `/analyticstool` pass.
- Optimization changes: run `tests/scripts/test_optimization_scripts.py` or full pytest.
- Otherwise run targeted pytest.
- If a focused run would fail the global coverage gate, use `--cov-fail-under=0` and say so explicitly.

## Performance

- Warm-switch decisions need browser timing, not just unit tests; use `5-run` passes.
- Perf work should measure browser callback/network fan-out, not just server callback time.
- Measure PortOpt startup and warm-switch separately.
- For modal-heavy flows, split measurements into open, snapshot, and apply windows when possible.
- For A/B warm-switch timing, run comparison cases in series, not in parallel, to avoid local resource contention skewing the result.
- PortOpt warm-switch rollback reference on `2026-03-13`: about `1970 ms` ready and `2870 ms` weights ready in non-debug mode.
- Do not assume PortOpt warm-switch is server-bound; the weight-chart callback was only about `15-18 ms`.
- Favor bootstrap/store fan-out reduction for PortOpt warm-switch.
- Do not repeat the reverted PortOpt warm-switch regressions: broad selected-result/store splitting or `dmc.Tabs(keepMounted=False)` active-tab bootstrap.
- If restoring a non-`weight` PortOpt tab on entry, seed that tab as render-ready during bootstrap.
- Treat shared modal unmounting as a measured tradeoff, not an automatic win.
- The PortOpt series modal now keeps a stable `po-series-selection-grid` in layout; optimize that path by updating `rowData` / `columnDefs`, not by rebuilding the grid shell.
- Preserve the PortOpt modal snapshot-on-OK contract; do not introduce live per-edit temp-store syncing unless measurements justify it.
- Do not assume the PortOpt series modal is server-bound; `portopt.render_series_modal_grid` was effectively negligible during Phase 2 timing and the remaining cost was browser-side modal/grid work.
- Synthetic perf and DB-backed perf can diverge materially; validate both before drawing conclusions.
- Prefer small internal `memory` stores for perf routing/gating over persisted schema changes.
- When hidden tabs are expensive, first prevent their callbacks from being scheduled; optimizing the callback body is usually a smaller win.
- The PortOpt hidden content-tab trigger pattern is a candidate for AnalyticsTool/Regression when inactive tab content is waking on shared control changes.
- After hidden-tab gating, the next PortOpt wins came from trimming always-on shared-control churn on selection changes, especially clientside eligibility/help-text callbacks and server callbacks that can return `no_update` for unchanged UI state.
- Hidden full-screen overlays must be gated with `display:none`.
- Keep module-switch blockers separate from page-local upload/modal blockers.
- Shared route callbacks must use `_pages_location.pathname`, must not mix always-mounted and page-local outputs, and must mark page-local inputs/states `allow_optional=True` when other pages may be active.

## Tooling

- On Windows, split very large `apply_patch` payloads.
- Prefer short Python Playwright scripts over long CLI one-liners.
- For A/B runs, use separate ports instead of editing `app.py`.
- Validate local SQLite files under `data/` before DB-backed browser runs.
- New worktrees do not inherit local SQLite files; copy `data/dashmat_local.db`, `data/MRD.db`, and `data/Performance.db` into the worktree before DB-backed browser runs.
- When timing logs matter, do not launch the app with plain `python -u app.py`; use `tools/playwright/start_timed_server.ps1` so `DASHMAT_TIMING_ENABLED=1`, `DASHMAT_TIMING_MIN_MS`, and `conda run --no-capture-output -n dashmat python -u ...` are set consistently.
- For harness timing correlation, pass the timed server `STDOUT` log path to `tools/playwright/warm_switch_harness.ps1` / `--server-log`; do not use the stderr log.
- Use `tools/playwright/portopt_series_modal_harness.ps1` for PortOpt series-modal timing; it measures modal open, select all, unselect all, and OK confirm in `5-run` passes.
- The PortOpt modal harness seeds a deterministic synthetic raw dataset and preloads the modal once before the measured window so modal timings stay isolated from welcome-screen/bootstrap noise.
- Use harness request attribution to choose the next perf phase instead of guessing from medians alone.
- The PortOpt modal harness now summarizes the most frequent OK-window Dash callback ids; use that list to decide whether the next phase should target shared-control churn or active-tab render work.
- Prefer adding harness switches for active-tab / restore-tab scenarios rather than changing app code to force a perf case.
- For strict PortOpt first-entry timing, parse only the measured entry window after warmup.
- If the warm-switch harness stalls on `#po-run-button`, verify the canonical PortOpt series-config stores first.
- Keep Playwright/runtime artifacts out of commits unless explicitly needed, and clean `output/` after ad hoc validation runs.
- If upward-opening modal dropdowns clip at the top, fix `utils/dashmat_welcome_modal.py`, not page-specific modal instances.
