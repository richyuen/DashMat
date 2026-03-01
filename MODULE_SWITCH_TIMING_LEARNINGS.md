## Module Switch Timing Learnings

### Scope
- Investigated why module switching still felt slow after SPA routing and raw-data metadata changes.
- Focused on control readiness, especially periodicity and date-picker population.

### Browser Timing Findings
- Real-browser Playwright measurements with a seeded `12`-series, `1800`-business-day session showed:
  - Analytics -> PortOpt:
    - URL change: about `96 ms`
    - periodicity ready: about `1171 ms`
    - date controls ready: about `9937 ms`
  - PortOpt -> Regression:
    - URL change: about `242 ms`
    - periodicity ready: about `1370 ms`
    - date controls ready: about `7166 ms`
  - Regression -> Analytics:
    - URL change: about `190 ms`
    - periodicity ready: about `1452 ms`
    - date controls ready: about `7384 ms`

### What This Ruled Out
- Routing is not the bottleneck anymore.
  - Cross-page navigation is consistently well under `250 ms`.
- The restore callbacks are not the bottleneck.
  - `analyticstool.restore_application_state`: effectively `0 ms`
  - `portopt.restore_state`: effectively `0 ms`
  - `regression._reg_toggle_main_visibility`: effectively `0 ms`
- The date-range init callbacks are real but not the whole delay.
  - Analytics cold: about `1421 ms`
  - PortOpt cold: about `1476 ms`
  - Regression cold: about `1413 ms`
  - Warm runs are sub-millisecond to about `0.6 ms`

### Main Finding
- The remaining delay is a callback waterfall after the route change, not one slow callback.
- Captured `_dash-update-component` traffic on a real PortOpt -> Regression switch showed:
  - fast `_pages_content.children` route swap
  - then many follow-up callback requests over several seconds
  - several of those requests were still triggered by `reg-page-load-trigger.n_intervals`
  - later requests also mounted results/tab hosts after controls were already becoming ready

### Likely Hot Spots
- Regression still had multiple modal-open callbacks listening to `reg-page-load-trigger`.
- Those callbacks were being hit on every module entry even when no import modal needed to open.
- The timer-driven page-entry fanout is therefore still doing avoidable work.

### Implication
- The next safe performance pass should reduce module-entry timer fanout rather than trying another broad lazy-mount refactor.
- Priority order:
  1. remove `*-page-load-trigger` from page-entry modal-open callbacks where possible
  2. replace timer-driven page-entry modal routing with a narrower route-intent dispatcher
  3. re-measure before changing result/tab rendering again

### Explicit Lesson
- Stable mounted callback outputs are required.
- The earlier empty-host lazy-tab experiment broke because callbacks targeted components that were no longer mounted.
- Future deferred rendering must keep stable mounted output targets.

### Next-Pass Result: Import Modal Timer Fanout Cut
- Regression no longer uses `reg-page-load-trigger` for its four import-modal auto-open callbacks.
- Those callbacks now consume a validated `reg-import-modal-request-store` produced from:
  - pathname
  - route intent
  - consumed token
- The same pattern has now been applied to Analytics and PortOpt:
  - `at-import-modal-request-store`
  - `po-import-modal-request-store`
- On all three pages, these modal-open callbacks no longer wake on module entry from the one-shot interval:
  - DB add modal
  - raw DB add modal
  - portfolio add modal
  - underlying add modal

### Updated Timer-Fanout Counts
- Regression:
  - before this pass: `reg-page-load-trigger` was wired into 6 callbacks
  - after this pass: only 2 callback uses remain
- Analytics:
  - import-modal timer fanout removed
  - remaining `at-page-load-trigger` usage is down to 2 callbacks plus the interval definition
- PortOpt:
  - import-modal timer fanout removed
  - several timer-driven callbacks still remain and are the next likely source of module-entry waterfall there

### Updated Practical Conclusion
- The next likely wins are the remaining timer-driven page-entry callbacks, not routing and not restore-state callbacks.
- The import-modal timer fanout was low-risk to cut and is now removed on all three pages.
- PortOpt still has the largest remaining cluster of `po-page-load-trigger` callbacks after this pass, so it is the next best target if module switching still feels slow.

### Next-Pass Result: PortOpt Timer Cleanup Did Not Move the Needle
- PortOpt was then cleaned up further so `po-page-load-trigger` no longer drives:
  - the base-controls restore clientside callback
  - the ex-ante-controls restore clientside callback
  - `po_sync_results_with_raw_data(...)`
- `po_open_modal(...)` was also made pathname-aware so page entry into `/portopt` can auto-open series selection without depending on the shared summary store changing.

### Re-Measurement After the PortOpt-Only Cleanup
- Re-ran the same real-browser Playwright measurement against the same seeded `12`-series, `1800`-business-day session.
- Results after the PortOpt-only pass:
  - Analytics -> PortOpt:
    - URL change: about `172 ms`
    - periodicity ready: about `1388 ms`
    - date controls ready: about `10550 ms`
  - PortOpt -> Regression:
    - URL change: about `545 ms`
    - periodicity ready: about `1962 ms`
    - date controls ready: about `7484 ms`
  - Regression -> Analytics:
    - URL change: about `384 ms`
    - periodicity ready: about `1953 ms`
    - date controls ready: about `8631 ms`

### What This Means
- The PortOpt-only timer cleanup did not produce a meaningful improvement.
- The measured timings were effectively flat to worse than the prior run.
- That strongly suggests the remaining bottleneck is not these last few timer-driven restore callbacks.

### Updated Working Hypothesis
- The dominant remaining delay is now likely a broader post-route callback/render waterfall:
  - hydration-triggered control callbacks
  - result synchronization
  - tab/result rendering work
  - or some combination of those paths
- The console output during these runs remained the existing AG Grid enterprise license noise, not a new callback/runtime error.

### Practical Next Step
- Stop spending effort on small `*-page-load-trigger` cleanup passes in isolation.
- The next investigation should capture the callback waterfall after route swap and identify the heaviest post-navigation callbacks by:
  1. timing `_dash-update-component` requests during a switch
  2. correlating those requests to callback outputs
  3. targeting the heaviest repeated result/control callbacks rather than more timer plumbing

### Waterfall Profiling Result: PortOpt -> Regression
- Captured a real `_dash-update-component` request trace during `PortOpt -> Regression`.
- Route change itself was still fast:
  - URL change about `378 ms`
  - date controls ready about `5486 ms`
- The switch generated about `50` Dash callback requests before the page was considered ready.

### Heaviest Requests In That Trace
- Regression series-selection/render callbacks were among the heaviest:
  - `reg-series-selection-loading-overlay` / `reg-modal-ok-button` / alert outputs: about `1372 ms`
  - `reg-page-ready-store` from the same series-selection status path: about `1369 ms`
  - `reg-series-selection-container.children` / grid status: about `1175 ms`
- Regression date-range init was heavy but not dominant:
  - `reg-start-date-picker`, `reg-end-date-picker`, `reg-date-range-store`, `reg-page-ready-store`: about `1236 ms`
- All four import-modal open callbacks still fired from the page-entry request-store path:
  - `reg-db-add-modal...`: about `1107 ms`
  - `reg-raw-db-add-modal...`: about `1129 ms`
  - `reg-portfolio-add-modal...`: about `1159 ms`
  - `reg-underlying-add-modal...`: about `1184 ms`

### Surprising Requests That Also Fired On Page Entry
- The trace also showed several callbacks firing with these changed inputs during route entry:
  - `reg-db-add-ok-button.n_clicks`
  - `reg-raw-db-add-ok-button.n_clicks`
  - `reg-portfolio-add-ok-button.n_clicks`
  - `reg-underlying-add-ok-button.n_clicks`
  - `reg-sheet-select-ok-button.n_clicks`
  - `reg-sheet-select-import-all-button.n_clicks`
  - `reg-upload-data.contents`
  - `reg-series-modal-commit-store.data`
- Those requests were not the single longest requests, but they clearly add to the post-route callback fanout and several of them write back into shared stores like `dashmat-raw-data-store`.

### Updated Working Hypothesis
- The remaining latency is not mainly the route swap or one slow restore callback.
- The main cost is now the breadth of the page-entry callback fanout.
- In Regression specifically, the biggest waste appears to be:
  1. series-selection auto-open/render work on page entry
  2. all four import-modal open callbacks waking up from the page-entry request store
  3. unexpected import/upload/commit callbacks also firing during entry

### Best Next Target
- The next code pass should not focus on more timer cleanup.
- The highest-value next target is to narrow the page-entry dispatcher so only the one actually-needed modal/series-selection path runs on entry, and to find why those import/upload/commit callbacks are waking up at all during route entry.

### Next-Pass Result: Narrow Regression Import-Modal Request Fanout Helped
- Regression was updated so page entry no longer fans one shared import-modal request store into all four import-modal open callbacks.
- The shared `reg-import-modal-request-store` was replaced with four narrow request stores:
  - `reg-db-add-request-store`
  - `reg-raw-db-add-request-store`
  - `reg-portfolio-add-request-store`
  - `reg-underlying-add-request-store`
- This means only the one relevant import-modal open callback wakes on route entry instead of all four firing and immediately `PreventUpdate`-ing.

### Re-Measurement After the Regression Fanout Split
- Re-ran `PortOpt -> Regression` on the same seeded session.
- Before this pass:
  - URL change: about `545 ms`
  - periodicity ready: about `1962 ms`
  - date controls ready: about `7484 ms`
- After this pass:
  - URL change: about `370 ms`
  - periodicity ready: about `1464 ms`
  - date controls ready: about `5563 ms`

### Practical Takeaway
- Narrowing callback fanout on page entry can produce a real improvement.
- The result here was roughly:
  - `~32%` faster date-controls-ready time on `PortOpt -> Regression`
  - `~25%` faster periodicity-ready time on the same flow
- That is a much stronger signal than the earlier isolated timer-cleanup passes.

### Updated Next Best Step
- Apply the same “split one shared page-entry request store into per-flow request stores” pattern to the remaining workspace where page-entry request fanout is still broad.
- After that, re-measure:
  1. `Analytics -> PortOpt`
  2. `Regression -> Analytics`

### Next-Pass Result: Analytics and PortOpt Fanout Split Was Mixed
- Applied the same shared-request-store split pattern to:
  - Analytics
  - PortOpt
- Each page now uses four per-flow page-entry request stores instead of one shared import-modal request store.

### Re-Measurement After the Analytics + PortOpt Fanout Split
- `Analytics -> PortOpt`
  - before this pass:
    - URL change: about `172 ms`
    - periodicity ready: about `1388 ms`
    - date controls ready: about `10550 ms`
  - after this pass:
    - URL change: about `172 ms`
    - periodicity ready: about `1448 ms`
    - date controls ready: about `10721 ms`
- `Regression -> Analytics`
  - before this pass:
    - URL change: about `384 ms`
    - periodicity ready: about `1953 ms`
    - date controls ready: about `8631 ms`
  - after this pass:
    - URL change: about `172 ms`
    - periodicity ready: about `1032 ms`
    - date controls ready: about `6185 ms`

### Practical Takeaway
- Splitting the shared page-entry request store is worthwhile when the destination page is actually paying for that fanout.
- It clearly helped Analytics entry:
  - `Regression -> Analytics` date-controls-ready improved by roughly `28%`
- It did **not** help PortOpt entry in the same way:
  - `Analytics -> PortOpt` stayed effectively flat to slightly worse

### Updated Best Next Target
- Stop assuming the same page-entry fix will help every workspace equally.
- The next targeted investigation should be PortOpt-specific, because PortOpt entry still looks expensive even after its request fanout was narrowed.
- The likely next step is to capture a full `Analytics -> PortOpt` waterfall the same way Regression was profiled, then target the heaviest PortOpt-specific entry callbacks rather than continuing broad pattern-based cleanup.
