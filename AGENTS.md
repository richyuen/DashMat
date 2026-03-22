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
- After any code change that affects a timing harness or its measured UI flow, run a `1`-run harness smoke test first.
- Use `5`-run passes only after the `1`-run smoke test succeeds.
- Use `5`-run passes and run A/B comparison cases in series, not in parallel.
- Judge perf changes by end-to-end browser medians, not just server callback time, request count, or request bytes.
- If a perf result is ambiguous, compare fresh clean `HEAD` vs unstaged vs rollback on isolated ports before deciding.
- If a change removes server work but regresses end-to-end timing, roll it back unless the regression is clearly isolated and safely salvageable.
- For reload-based flows, measure `click -> reload start` separately from `reload start -> controls ready`; pre-reload wins do not prove the post-reload wait improved.
- For same-page live-apply flows, measure `click -> live-apply commit` separately from `live-apply commit -> controls ready`; skipping a reload does not guarantee the post-apply ready path is fast.
- For same-page live-apply perf paths, internal-only payload retention is not free; if a retained client payload/store makes end-to-end medians worse, roll it back even when Dash request count stays flat.
- Before bypassing a page's generic restore/bootstrap path, prove the retained same-page payload is complete and durable enough to reproduce that page's restore inputs; if the payload contract is incomplete or too short-lived, harden it first instead of forcing the bypass.
- When a perf phase intentionally changes the ready outcome to an empty/cleared state, update the harness ready criteria first; do not judge timing with stale "content must render" assumptions from the old flow.
- For account-list timing runs, verify the chosen fixture can actually reach the intended ready state; do not use a saved list with an empty restored selection when measuring click-to-ready.
- Do not add an extra startup callback hop just to dedupe server-side restore/bootstrap work unless timed runs show an end-to-end startup win; tiny restore callback timings usually mean the real bottleneck is elsewhere.
- Do not assume collapsing many clientside startup emitters into one union-input router is a win; if the merged callback broadens wakeups or startup medians regress, keep the per-family emitters.
- Do not assume a shared visible-trigger store removes hidden-tab fan-out; if one trigger update still schedules many result families, split scheduling by family instead of gating inside callback bodies.
- Do not assume a partial per-family trigger split will reduce request count or medians; if the `1`-run smoke stays flat or regresses, roll it back and re-attribute before migrating more families.
- Do not assume forcing Analytics fresh bootstrap to `statistics` or suppressing restored active tabs will reduce hidden-family startup; if the `1`-run smoke stays flat or worse and hidden result families still wake, roll it back and re-attribute before changing restore/bootstrap tab policy further.
- Start measured sub-flow windows only after prior Dash traffic has settled, and count requests by request start time.
- Use callback/request attribution to choose the next perf phase; do not pick targets from medians alone.
- If a slower page already honors the intended restored-tab/bootstrap rule, do not force a symmetry-based fix for that rule; choose the next perf phase from the measured bottleneck instead.
- Prefer reducing fan-out first:
  - metadata/routing stores over full payload stores
  - `no_update` for unchanged UI/store state
  - small internal `memory` stores for routing/gating instead of persisted schema changes
  - hidden-tab scheduling gating before callback-body micro-optimization
- If the real perf goal is removing a request, do not rely on `PreventUpdate`; change the scheduling graph so the callback does not wake on that flow at all.
- When splitting a hot-path callback out of bootstrap, keep a separate bootstrap hydration path so ready-state optimization does not strand initial store population.
- For hidden-tab gating, use per-family triggers when the goal is to stop inactive callbacks from being scheduled at all.
- When a visible-tab signature/render path only needs dataset identity, pass raw-data metadata or dataset keys instead of the full raw-data store.
- For modal-only preview callbacks, gate scheduling with a modal-open trigger store instead of letting closed modals wake on shared control changes.
- When two stores are recomputed from the same inputs, prefer one deduped multi-output callback over parallel sibling callbacks.
- Avoid cycles in trigger-store graphs: do not feed a trigger emitter from a control whose value is derived downstream from that same trigger path.
- If moving pure browser-visible logic clientside for perf, keep the Python path as the reference and add parity tests.
- Track request bytes as well as request count; payload size can dominate the remaining cost.
- Do not assume a reload-era bootstrap callback can be reused for same-page live apply by only clearing results, resetting bootstrap state, and replaying stores; prove that the same-page retrigger path actually re-enters bootstrap and reaches a stable ready state before committing to that design.
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
- If a timed run reports callback outputs that no longer exist in source, verify the serving port is not backed by a stale timed-server process before attributing the result to the current code.
- Prefer extending the existing harnesses before creating new ones.
- Keep Playwright/runtime artifacts out of commits unless explicitly needed, and clean `output/` after ad hoc runs.
- If upward-opening modal dropdowns clip at the top, fix `utils/dashmat_welcome_modal.py`, not page-specific modal instances.
