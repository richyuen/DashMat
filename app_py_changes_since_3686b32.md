# `app.py` Integration Guide Since `3686b32`

This document summarizes every `app.py` change between commit `3686b32` and `HEAD`, with enough context to port those changes into a different `app.py`.

Only two commits changed `app.py` in that range:

- `37f00e9` `Add DashMat landing page and route intents`
- `6295c19` `Speed up module switch date-range init`

## High-Level Summary

There are four functional changes:

1. `app.py` now uses canonical route constants from `utils.page_paths` instead of deriving page paths from `dash.page_registry`.
2. The app shell now includes two new shared stores:
   - `dashmat-route-intent-store`
   - `dashmat-raw-data-summary-store`
3. The top navigation now points to the shared landing page when no raw data is loaded, and points directly to each workspace when raw data is present.
4. The app now derives a lightweight summary of the shared raw-data store so page-level callbacks can avoid re-reading the full raw-data payload just to get columns and frequency metadata.

## Exact `app.py` Changes

## 1. Replace Dynamic Registry Lookup With Canonical Route Helpers

### Remove

```python
import dash
```

and remove:

```python
def _registry_path(page_key: str, fallback: str) -> str:
    page_meta = dash.page_registry.get(page_key, {})
    path = page_meta.get("path") if isinstance(page_meta, dict) else None
    if isinstance(path, str) and path:
        return path
    return fallback
```

and remove:

```python
HOME_PATH = _registry_path("pages.home", "/")
ANALYTICS_PATH = _registry_path("pages.analyticstool", "/analyticstool")
PORTOPT_PATH = _registry_path("pages.portopt", "/portopt")
REGRESSION_PATH = _registry_path("pages.regression", "/regression")
```

### Add

```python
from utils.page_paths import (
    ANALYTICS_PATH,
    HOME_PATH,
    LANDING_PATH,
    PORTOPT_PATH,
    REGRESSION_PATH,
    landing_href,
)
```

### Why

This makes routing deterministic and lets the app shell generate landing-page links like `/dashmat?module=analyticstool` without depending on Dash page registry state.

## 2. Extend Restricted-Page Handling To The Landing Page

### Change `_restricted_href_for_path(...)`

Add this branch near the top:

```python
if pathname in (LANDING_PATH, f"{LANDING_PATH}/"):
    return "/restricted?target=DashMat"
```

### Why

Once `/dashmat` became the shared landing page, it also needed to participate in the existing test-role restriction behavior.

## 3. Add Two Shared Stores To The App Layout

Inside the top-level `MantineProvider` children list, keep the existing shared stores and add:

```python
dcc.Store(id="dashmat-raw-data-summary-store", data=None, storage_type="memory"),
dcc.Store(id="dashmat-route-intent-store", data=None, storage_type="session"),
```

Place them with the other app-level shared stores:

```python
dcc.Store(id="dashmat-raw-data-store", data=None, storage_type="session"),
dcc.Store(id="dashmat-original-periodicity-store", data="daily", storage_type="session"),
dcc.Store(id="dashmat-raw-data-summary-store", data=None, storage_type="memory"),
dcc.Store(id="dashmat-pending-new-series-store", data=[], storage_type="session"),
dcc.Store(id="dashmat-saved-series-cache-store", data=None, storage_type="session"),
dcc.Store(id="dashmat-route-intent-store", data=None, storage_type="session"),
dcc.Store(id="userinfo", data=USERINFO_DATA, storage_type="session"),
```

### What Each One Is For

- `dashmat-route-intent-store`
  - Shared handoff state from `/dashmat` into `/analyticstool`, `/portopt`, and `/regression`
  - Used for flows like “landing button -> navigate to workspace -> open import modal”

- `dashmat-raw-data-summary-store`
  - Lightweight derived metadata for the shared raw-data payload
  - Currently used to reduce blocker-path work during module switches
  - Safe to keep as `memory`; it does not need to survive a hard refresh

## 4. Change Default App-Shell Menu Links To Landing Links

### Before

The app-shell menu always pointed directly to the workspaces using `dmc.MenuItem(..., href=...)`:

```python
dmc.MenuItem("Analytics Tool", id="app-nav-analytics", href=ANALYTICS_PATH)
dmc.MenuItem("Portfolio Optimization", id="app-nav-portopt", href=PORTOPT_PATH)
dmc.MenuItem("Regression", id="app-nav-regression", href=REGRESSION_PATH)
```

### After

The app-shell should use `dcc.Link(refresh=False)` wrappers with renamed IDs, and the default hrefs should point to landing-page module targets:

```python
dcc.Link(
    id="global-navbar-pretrade-analytics",
    href=landing_href("analyticstool"),
    refresh=False,
    style={"textDecoration": "none", "color": "inherit"},
    children=dmc.MenuItem("Analytics Tool"),
)
```

Repeat that pattern for:

- `global-navbar-pretrade-home`
- `global-navbar-pretrade-portopt`
- `global-navbar-pretrade-regression`

### Why

This lets users with no active shared raw data land on `/dashmat?module=...` instead of entering an empty workspace first.

The runtime callback in the next section still upgrades these links to direct workspace paths once shared raw data exists.

## 5. Add A Shared Raw-Data Summary Callback

Add this callback immediately after `app.layout`:

```python
@app.callback(
    Output("dashmat-raw-data-summary-store", "data"),
    Input("dashmat-raw-data-store", "data"),
    Input("dashmat-original-periodicity-store", "data"),
    prevent_initial_call=False,
)
def update_raw_data_summary(raw_data, original_periodicity):
    return build_raw_data_summary(raw_data, original_periodicity or "daily")
```

### Required Import

```python
from utils.date_range_flow import build_raw_data_summary
```

### Why

This callback keeps a small derived summary in sync with the shared raw-data store. The current summary schema is:

```python
{
    "raw_data_hash": str,
    "columns": list[str],
    "available_periodicity_values": list[str],
    "original_periodicity": str,
}
```

This is now consumed by page-level restore and date-range initialization callbacks.

## 6. Change `update_global_nav_links(...)` To Be Data-Aware

### Before

`update_global_nav_links(...)` only depended on `userinfo`, and for non-test users it always returned direct workspace paths.

### After

The callback must accept both:

```python
Input("userinfo", "data"),
Input("dashmat-raw-data-store", "data"),
```

and its implementation becomes:

```python
@app.callback(
    Output("global-navbar-pretrade-home", "href"),
    Output("global-navbar-pretrade-analytics", "href"),
    Output("global-navbar-pretrade-portopt", "href"),
    Output("global-navbar-pretrade-regression", "href"),
    Input("userinfo", "data"),
    Input("dashmat-raw-data-store", "data"),
    prevent_initial_call=True,
)
def update_global_nav_links(userinfo, raw_data):
    if (userinfo or {}).get("role") == "Test":
        return (
            HOME_PATH,
            "/restricted?target=Analytics%20Tool",
            "/restricted?target=Portfolio%20Optimization",
            "/restricted?target=Regression",
        )

    if raw_data:
        return HOME_PATH, ANALYTICS_PATH, PORTOPT_PATH, REGRESSION_PATH

    return (
        HOME_PATH,
        landing_href("analyticstool"),
        landing_href("portopt"),
        landing_href("regression"),
    )
```

### Behavior

- Test users still get restricted links.
- Non-test users:
  - with shared raw data loaded: go straight to each workspace
  - without shared raw data: go to the landing page for the selected module

## Dependency Checklist For The Target App

If you are porting these `app.py` changes into another app, make sure the following also exist or have equivalents:

### Required modules / helpers

- [utils/page_paths.py](/C:/Git/DashMat/utils/page_paths.py)
  - `HOME_PATH`
  - `LANDING_PATH`
  - `ANALYTICS_PATH`
  - `PORTOPT_PATH`
  - `REGRESSION_PATH`
  - `landing_href(...)`

- [utils/date_range_flow.py](/C:/Git/DashMat/utils/date_range_flow.py)
  - `build_raw_data_summary(...)`

### Required page/store consumers

- [pages/dashmat.py](/C:/Git/DashMat/pages/dashmat.py)
  - this is the landing page at `LANDING_PATH`
- Workspace pages that consume:
  - `dashmat-route-intent-store`
  - `dashmat-raw-data-summary-store`

If the target app does not have a landing page or route-intent flow yet, you cannot copy these `app.py` changes verbatim. In that case:

- keep the `dashmat-raw-data-summary-store` and its callback if you want the performance improvement
- defer the `landing_href(...)`, `LANDING_PATH`, and `dashmat-route-intent-store` changes until the landing workflow exists

## Minimal Patch Order

If you want the lowest-risk port into a different `app.py`, apply the changes in this order:

1. Add `utils.page_paths` imports and remove `_registry_path(...)` usage.
2. Add `LANDING_PATH` support to `_restricted_href_for_path(...)`.
3. Add the two new shared stores.
4. Add the `update_raw_data_summary(...)` callback.
5. Convert the app-shell menu items to `dcc.Link(refresh=False)` wrappers.
6. Update the static menu hrefs to `landing_href(...)`.
7. Update `update_global_nav_links(...)` to be data-aware.

## Quick Validation Checklist

After porting the changes, verify:

1. With no shared raw data, the app-shell menu links for Analytics / PortOpt / Regression point to `/dashmat?module=...`.
2. With shared raw data loaded, those same links point to `/analyticstool`, `/portopt`, and `/regression`.
3. The app layout contains both:
   - `dashmat-route-intent-store`
   - `dashmat-raw-data-summary-store`
4. `update_raw_data_summary(...)` returns `None` when raw data is empty and returns a dict when raw data exists.
5. Test-role restricted routing also handles `/dashmat`.

## Direct Diff Reference

For reference, the cumulative source change is:

```bash
git diff 3686b32..HEAD -- app.py
```

and the commits involved are:

```bash
git log --oneline 3686b32..HEAD -- app.py
```
