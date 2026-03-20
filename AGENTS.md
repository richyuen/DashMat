## Environment

- Use Python `3.11` in the `dashmat` Conda env.
- Non-interactive commands should use `conda run -n dashmat ...`.

## Guardrails

- Preserve imported series names exactly; do not sanitize or alias them.
- For AG Grid fields with literal dots, set `dashGridOptions.suppressFieldDotNotation = True`.
- Prefer helper modules such as `build_*_components()` / `register_*_callbacks()` over growing `app.py`.
- DB setup, migrations, reseeds, deletes, and backfills belong in `tools/db`.

## Validation

- Routing or layout changes: start the app.
- Upload, parsing, or AnalyticsTool statistics changes: run full pytest and do a quick `/analyticstool` pass.
- Optimization changes: run `tests/scripts/test_optimization_scripts.py` or full pytest.
- Otherwise run targeted pytest.
- If a focused run would fail the coverage gate, use `--cov-fail-under=0` and say so explicitly.

## Performance

- Use browser timing for warm-switch decisions; unit tests are not enough.
- Use `1`-run passes to verify harness operation.
- Use `5`-run passes and run A/B comparison cases in series, not in parallel.
- Judge perf changes by end-to-end browser medians, not just server callback time, request count, or request bytes.
- If a perf result is ambiguous, compare fresh clean `HEAD` vs unstaged vs rollback on isolated ports before deciding.
- If a change removes server work but regresses end-to-end timing, roll it back unless the regression is clearly isolated and safely salvageable.
- Start measured sub-flow windows only after prior Dash traffic has settled, and count requests by request start time.
- Use callback/request attribution to choose the next perf phase; do not pick targets from medians alone.
- Prefer reducing fan-out first:
  - metadata/routing stores over full payload stores
  - `no_update` for unchanged UI/store state
  - small internal `memory` stores for routing/gating instead of persisted schema changes
  - hidden-tab scheduling gating before callback-body micro-optimization
- For hidden-tab gating, use per-family triggers when the goal is to stop inactive callbacks from being scheduled at all.
- If moving pure browser-visible logic clientside for perf, keep the Python path as the reference and add parity tests.
- Track request bytes as well as request count; payload size can dominate the remaining cost.
- PortOpt-specific guardrails:
  - preserve the modal snapshot-on-OK contract
  - keep the stable `po-series-selection-grid` shell; update `rowData` / `columnDefs` instead of rebuilding the grid
  - if restoring a non-`weight` tab on entry, seed that tab as render-ready during bootstrap
- Shared route callbacks must use `_pages_location.pathname`, must not mix always-mounted and page-local outputs, and must mark page-local inputs/states `allow_optional=True` when other pages may be active.
- Hidden full-screen overlays must be gated with `display:none`.

## Tooling

- On Windows, split very large `apply_patch` payloads.
- Prefer short Python Playwright scripts over long shell one-liners.
- For A/B runs, use separate ports instead of editing `app.py`.
- Validate local SQLite files under `data/` before DB-backed browser runs.
- New worktrees do not inherit local SQLite files; copy `data/dashmat_local.db`, `data/MRD.db`, and `data/Performance.db` before DB-backed browser runs.
- For timed browser runs, use `tools/playwright/start_timed_server.ps1` instead of launching the app directly.
- For harness timing correlation, pass the timed server `STDOUT` log path to the harness; do not use stderr.
- Prefer extending the existing harnesses before creating new ones.
- Keep Playwright/runtime artifacts out of commits unless explicitly needed, and clean `output/` after ad hoc runs.
- If upward-opening modal dropdowns clip at the top, fix `utils/dashmat_welcome_modal.py`, not page-specific modal instances.
