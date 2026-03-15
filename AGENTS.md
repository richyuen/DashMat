# AGENTS.md

Repo instructions for coding agents working in `C:\Git\DashMat`.

## Purpose

DashMat is a Dash app centered on three main pages:
- `pages/analyticstool.py`
- `pages/portopt.py`
- `pages/regression.py`

Prefer small, targeted changes unless the behavior change clearly requires something broader.

## Environment

- Python 3.11
- Always use the `dashmat` Conda environment.
- In non-interactive shells, use `conda run -n dashmat ...`.

Common commands:

```bash
conda run -n dashmat python app.py
conda run -n dashmat python -m pytest -q tests
conda run -n dashmat python tools/db/init_local_cma_db.py
```

## Core Rules

- Preserve callback IDs and store schemas unless a migration is intentional and updated everywhere.
- Keep shared JSON/store payloads compatible across pages.
- Preserve imported series names exactly as loaded. Do not sanitize or alias series names just to satisfy a grid or chart.
- In large callback files, patch the smallest safe section.
- For shared app-shell features, prefer a helper module with `build_*_components()` and `register_*_callbacks()` instead of growing `app.py`.
- Do not mutate or delete database table data from runtime callbacks.
- DB setup, backfills, truncates, deletes, reseeds, and migrations belong in explicit scripts under `tools/db`.
- Exception: AnalyticsTool factor-definition CRUD is allowed at runtime for `FactorDefinitions` and `FactorDefinitionsArchive`, with archive-first behavior and optimistic concurrency via `UPDATE_DATE`.

## Validation

- Routing or layout changes: start the app.
- Upload, parsing, or statistics changes: run full pytest and do a quick `/analyticstool` pass.
- Optimization changes: run `tests/scripts/test_optimization_scripts.py` or full pytest.
- Otherwise, run targeted pytest for touched logic.
- If a focused pytest run would trip the global coverage gate, use `--cov-fail-under=0` and say so explicitly.

## Performance

- Judge warm-switch performance with a browser timing pass, not only unit tests.
- PortOpt startup and PortOpt warm-switch are different problems. Measure them separately.
- PortOpt warm-switch baseline on March 13, 2026 was about `1970 ms` ready and `2870 ms` weight-chart ready in non-debug mode; treat that as the rollback reference when testing new warm-switch ideas.
- Do not assume PortOpt warm-switch is server-bound. The weight-chart callback was only about `15-18 ms` in timing runs, so callback math and result projection are not the first place to optimize.
- For PortOpt warm switch, reducing bootstrap/store fan-out can help. A March 14, 2026 pass that collapsed the restore/readiness chain into one clientside bootstrap reducer plus one visited-tab reducer improved the default restored-`weight` path versus `HEAD`.
- Do not repeat the broad selected-result/store-splitting refactor for PortOpt warm switch. It reduced payload sizes but still lost to the reverted baseline on `readyMedian` and `weightsReadyMedian`.
- Do not repeat the `dmc.Tabs(keepMounted=False)` active-tab bootstrap experiment for PortOpt warm switch. In March 2026 it regressed warm-switch readiness badly and broke restore-order assumptions.
- If restoring a non-`weight` PortOpt tab on entry, seed that tab as render-ready during bootstrap. Do not hardcode `weight` as the only initially loaded tab.
- Do not leave a full-screen fixed overlay mounted while “hidden”; gate the wrapper itself with `display:none`.
- Keep module-switch blockers separate from page-local upload/modal blockers.
- For shared route callbacks, use the always-mounted `_pages_location.pathname` instead of page-local `dcc.Location` ids.
- Do not mix always-mounted outputs with page-local outputs in the same shared callback.
- If a shared callback reads page-local inputs/states while other pages may be active, mark them `allow_optional=True`.

## Tooling

- On Windows, split very large `apply_patch` payloads into smaller patches.
- Prefer short Python Playwright scripts over long CLI one-liners for browser automation.
- For side-by-side comparisons, run the app on separate ports instead of editing `app.py`.
- Validate local SQLite files under `data/` before DB-backed browser runs or A/B comparisons.
- For timing runs that rely on copied stdout logs, launch the app with unbuffered Python (`python -u app.py`); buffered stdout can hide timing lines until process exit.
- For strict PortOpt first-entry validation, capture a timing-log offset after warmup and parse only the measured entry window; otherwise later tab switches will contaminate the timing summary.
- If the warm-switch harness stalls waiting for `#po-run-button`, verify the canonical PortOpt series-config session stores are mounted before changing the harness, especially `po-series-select` and the related series-config stores.
- Keep Playwright runtime artifacts out of commits unless explicitly needed.
- AG Grid treats dotted `field` names as nested paths by default; use `dashGridOptions.suppressFieldDotNotation = True` for literal series names.
- If upward-opening modal dropdowns clip at the top of the viewport, fix the shared builders in `utils/dashmat_welcome_modal.py` instead of patching page-specific modal instances.
