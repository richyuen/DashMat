# `app.py` Migration Instructions for Current Commit

## Summary

Your production `app.py` from `73896e5` is missing the app-level plumbing added for:

- artifact-backed raw data / result storage
- portable `Save session` / `Load session`
- the shared session id used by AT / PO / REG callbacks

To support the current commit, update your production `app.py` to match the current app-shell contract. Keep your production-only logic, but add the same stores, imports, and callbacks listed below.

## Required Changes

### 1. Update imports at the top of `app.py`

Replace the old imports:

```python
from dash import Dash, Input, Output, dcc, page_container
from utils.returns import build_raw_data_metadata
```

with the current set:

```python
from dash import ClientsideFunction, Dash, Input, Output, State, clientside_callback, dcc, no_update, page_container
from uuid import uuid4

from utils.artifact_store import build_raw_data_store_metadata
from utils.perf_timing import record_payload_size, timed_block
from utils.serialization import canonical_json_dumps
from utils.workspace_session import build_workspace_session_bundle, restore_workspace_session_bundle
```

Notes:

- `State`, `clientside_callback`, `ClientsideFunction`, and `no_update` are required by the new callbacks.
- `uuid4` is required for the shared session id store.
- `build_raw_data_store_metadata` replaces `build_raw_data_metadata`.
- `canonical_json_dumps`, `build_workspace_session_bundle`, and `restore_workspace_session_bundle` are required for `Save session` / `Load session`.
- `timed_block` and `record_payload_size` are used in the current callback bodies; if you omit them, you must also remove those wrapper calls.

### 2. Add the new shared stores and `dcc.Download` to the top-level layout

In your shared Mantine provider children list, keep the existing stores and add these exact components.

Current required shared app-level children should include:

```python
dcc.Store(id="dashmat-session-id-store", data=None, storage_type="session"),
dcc.Store(id="dashmat-raw-data-store", data=None, storage_type="session"),
dcc.Store(id="dashmat-raw-data-meta-store", data=None, storage_type="session"),
dcc.Store(id="dashmat-original-periodicity-store", data="daily", storage_type="session"),
dcc.Store(id="dashmat-pending-new-series-store", data={}, storage_type="session"),
dcc.Store(id="dashmat-saved-series-cache-store", data=None, storage_type="session"),

dcc.Store(id="dashmat-session-export-request-store", data=None, storage_type="memory"),
dcc.Store(id="dashmat-session-import-request-store", data=None, storage_type="memory"),
dcc.Store(id="dashmat-session-import-result-store", data=None, storage_type="memory"),
dcc.Store(id="dashmat-session-import-apply-dummy", data=None, storage_type="memory"),
dcc.Download(id="dashmat-save-session-download"),
```

Keep your existing:

```python
dcc.Store(id="userinfo", data=USERINFO_DATA, storage_type="session"),
```

Important:

- `dashmat-session-id-store` is now required. Many AT / PO / REG callbacks use it as `State(...)`.
- The four `dashmat-session-*` memory stores plus `dashmat-save-session-download` are required by the current `assets/dashmat_callbacks.js` save/load code.
- Do not add `dashmat-raw-data-artifact-store`. That was an intermediate store and is not part of the final current app contract.

### 3. Add the session-id callback

Add this callback after `app.layout` is defined:

```python
@app.callback(
    Output("dashmat-session-id-store", "data"),
    Input("_pages_location", "pathname"),
    State("dashmat-session-id-store", "data"),
    prevent_initial_call=False,
)
def ensure_dashmat_session_id(_pathname, existing_session_id):
    if existing_session_id:
        return no_update
    return str(uuid4())
```

Why this is required:

- current page callbacks write artifact-backed raw/result payloads using the shared session id
- without this store and callback, current AT / PO / REG save/update flows will break or write empty artifact descriptors

### 4. Add the server-side `Save session` export callback

Add this callback:

```python
@app.callback(
    Output("dashmat-save-session-download", "data"),
    Input("dashmat-session-export-request-store", "data"),
    prevent_initial_call=True,
)
def export_workspace_session_bundle(request_data):
    if not isinstance(request_data, dict):
        raise PreventUpdate
    workspace_session = request_data.get("workspace_session")
    if not isinstance(workspace_session, dict) or not workspace_session:
        raise PreventUpdate
    with timed_block("workspace_session.export", key_count=len(workspace_session)):
        bundle = build_workspace_session_bundle(workspace_session)
    return {
        "content": canonical_json_dumps(bundle),
        "filename": "dashmat_session.json",
        "type": "application/json",
    }
```

Why this is required:

- current JS no longer downloads raw browser `sessionStorage` directly
- it writes a request into `dashmat-session-export-request-store` and expects this callback to return a downloadable bundle through `dcc.Download`

### 5. Add the server-side `Load session` import callback

Add this callback:

```python
@app.callback(
    Output("dashmat-session-import-result-store", "data"),
    Input("dashmat-session-import-request-store", "data"),
    prevent_initial_call=True,
)
def import_workspace_session_bundle(request_data):
    if not isinstance(request_data, dict):
        raise PreventUpdate
    bundle = request_data.get("bundle")
    if not isinstance(bundle, dict):
        raise PreventUpdate
    with timed_block("workspace_session.import", artifact_count=len(bundle.get("artifacts") or [])):
        return restore_workspace_session_bundle(bundle)
```

Why this is required:

- current JS upload/load flow sends the parsed session bundle into `dashmat-session-import-request-store`
- this callback restores bundled artifacts and returns the remapped workspace session payload

### 6. Add the clientside import-apply callback

Add this after the Python callbacks:

```python
clientside_callback(
    ClientsideFunction(namespace="dashmat_callbacks", function_name="applyLoadedWorkspaceSession"),
    Output("dashmat-session-import-apply-dummy", "data"),
    Input("dashmat-session-import-result-store", "data"),
    prevent_initial_call=True,
)
```

Why this is required:

- after the Python import callback restores the session bundle, the current JS applies the returned `workspace_session` payload into browser storage and reloads the page
- if this clientside callback is missing, `Load session` will restore nothing in the browser

### 7. Replace the raw-data meta callback body

Your old code uses:

```python
from utils.returns import build_raw_data_metadata
...
def refresh_raw_data_meta_store(raw_data, original_periodicity):
    return build_raw_data_metadata(raw_data, original_periodicity)
```

Replace that import and callback body with:

```python
@app.callback(
    Output("dashmat-raw-data-meta-store", "data"),
    Input("dashmat-raw-data-store", "data"),
    Input("dashmat-original-periodicity-store", "data"),
    prevent_initial_call=False,
)
def refresh_raw_data_meta_store(raw_data, original_periodicity):
    with timed_block("refresh_raw_data_meta_store", has_data=bool(raw_data)):
        metadata = build_raw_data_store_metadata(raw_data, original_periodicity)
    record_payload_size("dashmat_raw_data_meta_store.output", metadata)
    return metadata
```

Why this is required:

- current `dashmat-raw-data-store` is no longer always inline JSON
- it can now be a raw-data descriptor pointing to an artifact-backed dataset
- `build_raw_data_store_metadata(...)` understands both the new descriptor form and old inline JSON fallback
- the old `build_raw_data_metadata(...)` call is not compatible with the current runtime contract

## Important Compatibility Notes

- Keep your existing `update_app_nav_links(...)` and `guard_protected_pages(...)` logic unless you intentionally changed those for production. The current app-level nav logic is still valid.
- Keep your existing `MantineProvider`, `AppShell`, and menu structure. Only the shared stores and callbacks above are required for compatibility.
- If your production `app.py` has custom auth/user-role behavior, merge these changes around it rather than replacing that logic wholesale.
- Do not remove `dashmat-saved-series-cache-store`; it still exists, but it is now a small descriptor-style session store rather than a large inline payload.
- Do not add `dashmat-raw-data-artifact-store`; the final current code does not use it.

## Minimal “Must Have” Checklist

Your production `app.py` is compatible with the current commit when all of these are true:

- `dashmat-session-id-store` exists in layout
- `dashmat-session-export-request-store` exists in layout
- `dashmat-session-import-request-store` exists in layout
- `dashmat-session-import-result-store` exists in layout
- `dashmat-session-import-apply-dummy` exists in layout
- `dashmat-save-session-download` exists in layout
- `ensure_dashmat_session_id(...)` callback exists
- `export_workspace_session_bundle(...)` callback exists
- `import_workspace_session_bundle(...)` callback exists
- `clientside_callback(... applyLoadedWorkspaceSession ...)` exists
- `refresh_raw_data_meta_store(...)` uses `build_raw_data_store_metadata(...)`, not `build_raw_data_metadata(...)`

## Validation After You Patch Production `app.py`

Run these checks in your production environment version after applying the changes:

1. App startup/import check:
   - `conda run -n dashmat python -c "import app; print(type(app.app.layout).__name__)"`
   - this should import cleanly without missing ids/imports

2. Full regression test sweep:
   - `conda run -n dashmat python -m pytest -q --basetemp .pytest_tmp tests`

3. Browser save/load sanity:
   - load data
   - create a PortOpt or Regression result
   - click `Save session`
   - clear browser storage
   - load the saved session
   - confirm results restore

## Assumptions

- Your production `app.py` is structurally based on `73896e5` and differs mainly in deployment/auth behavior, not in the shared Dash layout contract.
- `assets/dashmat_callbacks.js` in production is current. If it is older, you must update that too; the new app-level stores/callbacks are designed to match the current JS behavior.
- You want compatibility with the current commit’s behavior, including portable `Save session`, not just “app boots without crashing.”
