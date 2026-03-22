from __future__ import annotations

import json

import dash_ag_grid as dag
import dash_mantine_components as dmc
from dash import Input, Output, State, clientside_callback, dcc, html, no_update
from dash.exceptions import PreventUpdate
from sqlalchemy.engine import Engine
from utils.ag_grid import literal_field_dash_grid_options

from utils.account_lists import (
    ACCOUNT_LIST_LOAD_MERGE_STORE_IDS,
    ACCOUNT_LIST_CAPTURE_STORE_IDS,
    AT_EXTRA_CONTROL_STORE_IDS,
    AT_STORE_IDS,
    PO_STORE_IDS,
    REG_STORE_IDS,
    account_list_tables_available,
    account_list_preview_rows,
    build_account_list_payload,
    build_account_list_session_payload,
    delete_account_list,
    list_account_lists,
    list_account_list_users,
    load_account_list_by_id,
    normalize_db_import_provenance_store,
    prefetch_account_list_entry_frames,
    prune_db_import_provenance,
    save_account_list,
    send_account_list,
    users_table_available,
)
from utils.perf_timing import timed_block


ACCOUNT_LIST_MODAL_BASE_CLASS = "dashmat-modal dashmat-account-list-modal"
ACCOUNT_LIST_MODAL_LOAD_CLASS = f"{ACCOUNT_LIST_MODAL_BASE_CLASS} dashmat-account-list-modal-load"


def build_account_list_load_state(status: str) -> dict[str, object]:
    return {"status": str(status or "idle").strip().lower() or "idle"}


def account_list_load_state_status(load_state) -> str:
    if not isinstance(load_state, dict):
        return "idle"
    return str(load_state.get("status") or "idle").strip().lower() or "idle"


def build_account_list_prefetch_state(account_list_id=None, update_date=None, status: str = "idle") -> dict[str, object]:
    return {
        "account_list_id": int(account_list_id) if account_list_id is not None else None,
        "update_date": str(update_date) if update_date is not None else None,
        "status": str(status or "idle").strip().lower() or "idle",
    }


def account_list_loader_visible(load_state) -> bool:
    return account_list_load_state_status(load_state) in {"loading", "live_applying"}


def account_list_loader_wrapper_style(load_state) -> dict[str, object]:
    if not account_list_loader_visible(load_state):
        return {"display": "none"}
    return {"position": "fixed", "inset": 0, "zIndex": 4100}


def build_account_list_session_apply_payload(*, pathname, session_payload):
    normalized_path = str(pathname or "").strip()
    mode = "live_apply_analyticstool" if normalized_path.startswith("/analyticstool") else "reload"
    return {
        "mode": mode,
        "pathname": normalized_path,
        "session_payload": dict(session_payload or {}),
    }


def prefetch_selected_account_list_entries(
    *,
    opened,
    mode,
    selected_detail,
    current_prefetch,
    db_engine: Engine,
    mrd_engine: Engine,
    perf_engine: Engine,
):
    if not opened or str(mode or "load") != "load" or not isinstance(selected_detail, dict):
        return build_account_list_prefetch_state()

    account_list_id = selected_detail.get("AccountListID")
    update_date = selected_detail.get("UPDATE_DATE")
    next_state = build_account_list_prefetch_state(account_list_id, update_date, "ready")

    if isinstance(current_prefetch, dict):
        if (
            current_prefetch.get("account_list_id") == next_state["account_list_id"]
            and current_prefetch.get("update_date") == next_state["update_date"]
            and current_prefetch.get("status") in {"loading", "ready", "error"}
        ):
            return no_update

    try:
        with timed_block(
            "account_list.prefetch_entry_frames",
            account_list_id=account_list_id,
        ):
            prefetch_account_list_entry_frames(
                selected_detail.get("ConfigJson"),
                db_engine=db_engine,
                mrd_engine=mrd_engine,
                perf_engine=perf_engine,
            )
        return next_state
    except Exception:
        return build_account_list_prefetch_state(account_list_id, update_date, "error")


def load_selected_account_list_session(
    *,
    n_clicks,
    selected_id,
    selected_detail,
    rows,
    apply_settings,
    raw_data,
    original_periodicity,
    provenance_store,
    load_snapshot,
    pathname,
    userinfo,
    db_engine: Engine,
    mrd_engine: Engine,
    perf_engine: Engine,
):
    if not n_clicks:
        raise PreventUpdate
    if selected_id is None:
        return no_update, {"message": "Select an account list to load.", "color": "orange"}, build_account_list_load_state("error")
    row = resolve_selected_account_list_detail(
        selected_id=selected_id,
        selected_detail=selected_detail,
        rows=rows,
        userinfo=userinfo,
        db_engine=db_engine,
    )
    if row is None:
        return no_update, {"message": "Saved account list no longer exists.", "color": "red"}, build_account_list_load_state("error")
    try:
        current_snapshot = normalize_account_list_load_snapshot(load_snapshot)
        with timed_block(
            "account_list.load_request",
            merge_key_count=len(ACCOUNT_LIST_LOAD_MERGE_STORE_IDS),
            snapshot_key_count=len(current_snapshot),
            apply_settings=bool(apply_settings),
        ):
            session_payload, _stats = build_account_list_session_payload(
                payload=row.get("ConfigJson"),
                current_raw_data=raw_data,
                current_original_periodicity=original_periodicity,
                current_provenance=provenance_store,
                current_session_snapshot=current_snapshot,
                apply_settings=bool(apply_settings),
                db_engine=db_engine,
                mrd_engine=mrd_engine,
                perf_engine=perf_engine,
            )
    except Exception as exc:
        return (
            no_update,
            {"message": f"Unable to load account list: {exc}", "color": "red"},
            build_account_list_load_state("error"),
        )
    return (
        build_account_list_session_apply_payload(pathname=pathname, session_payload=session_payload),
        no_update,
        build_account_list_load_state("success"),
    )


def load_selected_account_list_detail(*, selected_id, userinfo, db_engine: Engine):
    if selected_id is None:
        return None
    with timed_block("account_list.selected_detail", account_list_id=selected_id):
        return load_account_list_by_id(db_engine, selected_id, _account_list_username(userinfo))


def render_selected_account_list_preview(selected_detail):
    if not isinstance(selected_detail, dict):
        return []
    with timed_block("account_list.preview_rows", account_list_id=selected_detail.get("AccountListID")):
        preview_rows = account_list_preview_rows(selected_detail.get("ConfigJson"))
    return preview_rows if isinstance(preview_rows, list) else []


def resolve_selected_account_list_detail(*, selected_id, selected_detail, rows, userinfo, db_engine: Engine):
    selected_row = next(
        (row for row in (rows or []) if isinstance(row, dict) and row.get("AccountListID") == selected_id),
        None,
    )
    expected_update = selected_row.get("UPDATE_DATE") if isinstance(selected_row, dict) else None
    if (
        isinstance(selected_detail, dict)
        and selected_detail.get("AccountListID") == selected_id
        and (
            expected_update is None
            or selected_detail.get("UPDATE_DATE") == expected_update
        )
    ):
        return selected_detail
    return load_selected_account_list_detail(
        selected_id=selected_id,
        userinfo=userinfo,
        db_engine=db_engine,
    )


def build_account_list_components() -> list:
    return [
        dcc.Store(id="dashmat-account-list-notice-store", data=None, storage_type="session"),
        dcc.Store(id="dashmat-account-list-modal-mode-store", data="load"),
        dcc.Store(id="dashmat-account-list-rows-store", data=[]),
        dcc.Store(id="dashmat-account-list-selected-id-store", data=None),
        dcc.Store(id="dashmat-account-list-selected-detail-store", data=None),
        dcc.Store(id="dashmat-account-list-prefetch-store", data={"account_list_id": None, "update_date": None, "status": "idle"}),
        dcc.Store(id="dashmat-account-list-session-snapshot-store", data={}),
        dcc.Store(id="dashmat-account-list-load-snapshot-store", data=None),
        dcc.Store(id="dashmat-account-list-refresh-store", data=0),
        dcc.Store(id="dashmat-account-list-session-apply-store", data=None),
        dcc.Store(id="dashmat-account-list-load-state-store", data={"status": "idle"}),
        dcc.Store(id="dashmat-account-list-load-timing-dummy", data=None),
        dcc.Store(id="dashmat-account-list-enter-submit-dummy", data=None),
        dcc.Store(id="dashmat-account-list-focus-dummy", data=None),
        html.Div(
            id="dashmat-account-list-load-overlay-shell",
            style={"display": "none"},
            children=[
                dmc.LoadingOverlay(
                    id="dashmat-account-list-load-overlay",
                    visible=False,
                    zIndex=4100,
                    overlayProps={"radius": "sm", "blur": 2},
                    loaderProps={"variant": "bars", "size": "lg"},
                    style={"position": "absolute", "inset": 0},
                ),
            ],
        ),
        dmc.Modal(
            id="dashmat-account-list-modal",
            opened=False,
            title="Account Lists",
            size="xl",
            centered=True,
            className=ACCOUNT_LIST_MODAL_BASE_CLASS,
            styles={"content": {"maxWidth": "1120px"}},
            children=[
                dmc.Stack(
                    id="dashmat-account-list-modal-root",
                    gap="md",
                    style={"minHeight": 0},
                    children=[
                        html.Div(
                            id="dashmat-account-list-save-section",
                            children=[
                                dmc.TextInput(
                                    id="dashmat-account-list-name-input",
                                    label="List Name",
                                    placeholder="Enter a name",
                                ),
                                dmc.Text(
                                    id="dashmat-account-list-duplicate-text",
                                    size="sm",
                                    c="dimmed",
                                    mt="xs",
                                ),
                            ],
                        ),
                        html.Div(
                            id="dashmat-account-list-load-section",
                            style={"display": "none"},
                            children=[
                                dmc.Stack(
                                    id="dashmat-account-list-load-stack",
                                    gap="md",
                                    style={"height": "100%", "minHeight": 0},
                                    children=[
                                        dmc.Stack(
                                            gap="xs",
                                            className="dashmat-account-list-panel",
                                            style={"minHeight": 0},
                                            children=[
                                                dmc.Text("Saved Lists", fw=600, size="sm"),
                                                dag.AgGrid(
                                                    id="dashmat-account-list-grid",
                                                    className="ag-theme-alpine dashmat-series-modal-grid",
                                                    rowData=[],
                                                    columnDefs=[
                                                        {"field": "ListName", "minWidth": 180, "headerName": "Name"},
                                                        {"field": "UPDATE_DATE", "minWidth": 160, "headerName": "Updated"},
                                                        {"field": "UPDATE_BY", "minWidth": 140, "headerName": "By"},
                                                        {
                                                            "field": "SeriesCount",
                                                            "width": 92,
                                                            "headerName": "Count",
                                                            "headerClass": "dashmat-center-header",
                                                            "cellClass": "dashmat-series-center-cell",
                                                        },
                                                    ],
                                                    defaultColDef={
                                                        "sortable": False,
                                                        "filter": False,
                                                        "resizable": True,
                                                        "suppressHeaderMenuButton": True,
                                                    },
                                                    dashGridOptions=literal_field_dash_grid_options({
                                                        "rowSelection": "single",
                                                        "animateRows": False,
                                                        "overlayNoRowsTemplate": "No saved account lists found.",
                                                    }),
                                                    getRowId="params.data.AccountListID",
                                                    style={"height": "100%", "width": "100%"},
                                                ),
                                            ],
                                        ),
                                        dmc.Stack(
                                            gap="xs",
                                            className="dashmat-account-list-panel",
                                            style={"minHeight": 0},
                                            children=[
                                                dmc.Text("Series Preview", fw=600, size="sm"),
                                                dag.AgGrid(
                                                    id="dashmat-account-list-preview-grid",
                                                    className="ag-theme-alpine dashmat-series-modal-grid",
                                                    rowData=[],
                                                    columnDefs=[
                                                        {"field": "Series", "minWidth": 220},
                                                        {"field": "SourceType", "minWidth": 140, "headerName": "Source"},
                                                        {
                                                            "field": "AT",
                                                            "width": 70,
                                                            "headerClass": "dashmat-center-header",
                                                            "cellClass": "dashmat-series-center-cell",
                                                        },
                                                        {
                                                            "field": "PO",
                                                            "width": 70,
                                                            "headerClass": "dashmat-center-header",
                                                            "cellClass": "dashmat-series-center-cell",
                                                        },
                                                        {
                                                            "field": "REG",
                                                            "width": 80,
                                                            "headerClass": "dashmat-center-header",
                                                            "cellClass": "dashmat-series-center-cell",
                                                        },
                                                    ],
                                                    defaultColDef={
                                                        "sortable": False,
                                                        "filter": False,
                                                        "resizable": True,
                                                        "suppressHeaderMenuButton": True,
                                                    },
                                                    dashGridOptions=literal_field_dash_grid_options({
                                                        "animateRows": False,
                                                        "overlayNoRowsTemplate": "Select a saved account list to preview included series.",
                                                    }),
                                                    style={"height": "100%", "width": "100%"},
                                                ),
                                            ],
                                        ),
                                        dmc.Switch(
                                            id="dashmat-account-list-apply-settings-switch",
                                            label="Apply Saved Settings",
                                            checked=True,
                                            size="sm",
                                        ),
                                        dmc.Group(
                                            id="dashmat-account-list-send-controls",
                                            gap="sm",
                                            align="flex-end",
                                            children=[
                                                dmc.Select(
                                                    id="dashmat-account-list-send-user-select",
                                                    label="Send to User",
                                                    placeholder="Select a user",
                                                    data=[],
                                                    value=None,
                                                    searchable=True,
                                                    clearable=True,
                                                    style={"flex": "1 1 0"},
                                                ),
                                                dmc.Button(
                                                    "Send",
                                                    id="dashmat-account-list-send-button",
                                                    variant="light",
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        dmc.Text(
                            id="dashmat-account-list-modal-message",
                            size="sm",
                            c="dimmed",
                        ),
                        dmc.Group(
                            justify="space-between",
                            children=[
                                dmc.Button(
                                    "Close",
                                    id="dashmat-account-list-close-button",
                                    variant="default",
                                ),
                                dmc.Group(
                                    children=[
                                        dmc.Button(
                                            "Delete",
                                            id="dashmat-account-list-delete-button",
                                            color="red",
                                            variant="light",
                                        ),
                                        dmc.Button(
                                            "Load",
                                            id="dashmat-account-list-load-button",
                                        ),
                                        dmc.Button(
                                            "Save",
                                            id="dashmat-account-list-save-button",
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ]


def _account_list_username(userinfo: dict | None) -> str:
    return str((userinfo or {}).get("username", "") or "").strip()


def _account_list_update_by(userinfo: dict | None) -> str:
    return _account_list_username(userinfo) or "unknown"


def _account_list_row_data(rows: list[dict[str, object]] | None) -> list[dict[str, object]]:
    row_data: list[dict[str, object]] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        row_data.append(
            {
                "AccountListID": row.get("AccountListID"),
                "ListName": row.get("ListName"),
                "UPDATE_DATE": row.get("UPDATE_DATE"),
                "UPDATE_BY": row.get("UPDATE_BY"),
                "SeriesCount": row.get("SeriesCount"),
            }
        )
    return row_data


def refresh_db_import_provenance(raw_meta, provenance_store):
    columns = list((raw_meta or {}).get("columns") or [])
    return prune_db_import_provenance(provenance_store, columns)


def render_account_list_notice(notice):
    if not isinstance(notice, dict) or not str(notice.get("message") or "").strip():
        return []
    return dmc.Alert(
        dmc.Group(
            justify="space-between",
            align="flex-start",
            wrap="nowrap",
            children=[
                dmc.Text(str(notice.get("message") or ""), style={"flex": "1 1 auto"}),
                dmc.ActionIcon(
                    id="dashmat-account-list-notice-close-button",
                    variant="subtle",
                    color=str(notice.get("color") or "blue"),
                    size="sm",
                    **{"aria-label": "Dismiss account list notice"},
                    children="x",
                ),
            ],
        ),
        color=str(notice.get("color") or "blue"),
        variant="light",
        title="Account Lists",
        withCloseButton=False,
        mb="sm",
    )


def toggle_account_list_modal(
    at_save,
    at_load,
    at_welcome_load,
    po_save,
    po_load,
    po_welcome_load,
    reg_save,
    reg_load,
    reg_welcome_load,
    close_clicks,
    *,
    triggered_id,
):
    click_map = {
        "at-menu-save-account-list": at_save,
        "at-menu-load-account-list": at_load,
        "at-welcome-load-account-list-btn": at_welcome_load,
        "po-menu-save-account-list": po_save,
        "po-menu-load-account-list": po_load,
        "po-welcome-load-account-list-btn": po_welcome_load,
        "reg-menu-save-account-list": reg_save,
        "reg-menu-load-account-list": reg_load,
        "reg-welcome-load-account-list-btn": reg_welcome_load,
        "dashmat-account-list-close-button": close_clicks,
    }
    if triggered_id == "dashmat-account-list-close-button":
        return False, no_update, "", None, True
    if triggered_id in {"at-menu-save-account-list", "po-menu-save-account-list", "reg-menu-save-account-list"} and click_map.get(triggered_id):
        return True, "save", "", None, True
    if triggered_id in {
        "at-menu-load-account-list",
        "at-welcome-load-account-list-btn",
        "po-menu-load-account-list",
        "po-welcome-load-account-list-btn",
        "reg-menu-load-account-list",
        "reg-welcome-load-account-list-btn",
    } and click_map.get(triggered_id):
        return True, "load", "", None, True
    raise PreventUpdate


def sync_account_list_selected_id(selected_rows):
    if not selected_rows:
        return None
    row = selected_rows[0] if isinstance(selected_rows, list) else None
    if not isinstance(row, dict):
        return None
    return row.get("AccountListID")


def account_list_send_user_options(users: list[dict[str, str]] | None) -> list[dict[str, str]]:
    options: list[dict[str, str]] = []
    for row in users or []:
        username = str((row or {}).get("Username") or "").strip()
        if not username:
            continue
        options.append({"label": username, "value": username})
    return options


def account_list_send_controls_state(mode, selected_id, recipient_options, recipient_value):
    hidden = {"display": "none"}
    visible = {}
    if str(mode or "load") != "load":
        return hidden, True, True, "Select a user"
    options = recipient_options if isinstance(recipient_options, list) else []
    if not options:
        return visible, True, True, "No other users available"
    return visible, False, selected_id is None or not str(recipient_value or "").strip(), "Select a user"


def normalize_account_list_load_snapshot(snapshot):
    if not isinstance(snapshot, dict):
        return {}
    return {
        str(key): snapshot.get(key)
        for key in ACCOUNT_LIST_LOAD_MERGE_STORE_IDS
        if str(key) in snapshot
    }


def render_account_list_modal_view(opened, mode, rows, selected_id, selected_detail):
    hidden = {"display": "none"}
    visible = {}
    if not opened:
        return (
            "Account Lists",
            ACCOUNT_LIST_MODAL_BASE_CLASS,
            hidden,
            hidden,
            [],
            [],
            "",
            True,
            True,
            hidden,
            hidden,
            hidden,
        )

    rows = rows if isinstance(rows, list) else []
    selected_row = next((row for row in rows if row.get("AccountListID") == selected_id), None)
    list_row_data = _account_list_row_data(rows)
    preview_rows = render_selected_account_list_preview(selected_detail)

    if str(mode or "load") == "save":
        return (
            "Save Account List",
            ACCOUNT_LIST_MODAL_BASE_CLASS,
            visible,
            hidden,
            list_row_data,
            [],
            "Save the current DB-backed series and committed AT/PO/REG series settings.",
            True,
            True,
            hidden,
            hidden,
            visible,
        )

    return (
        "Load Account List",
        ACCOUNT_LIST_MODAL_LOAD_CLASS,
        hidden,
        visible,
        list_row_data,
        preview_rows,
        "Load adds latest DB data for saved series. Apply Saved Settings restores all saved page controls; off keeps the current series-dialog restore behavior.",
        selected_row is None,
        selected_row is None,
        visible,
        visible,
        hidden,
    )


def sync_account_list_save_state(mode, name_value, rows, provenance_store):
    if str(mode or "load") != "save":
        return "", True
    rows = rows if isinstance(rows, list) else []
    clean_name = str(name_value or "").strip()
    duplicate_count = (
        sum(1 for row in rows if str(row.get("ListName") or "").strip().lower() == clean_name.lower())
        if clean_name
        else 0
    )
    helper = f"{duplicate_count} existing list(s) already use this name." if duplicate_count else "Duplicate names are allowed."
    disabled = not clean_name or not normalize_db_import_provenance_store(provenance_store)
    return helper, disabled


def register_account_list_callbacks(
    app,
    *,
    db_engine: Engine,
    mrd_engine: Engine,
    perf_engine: Engine,
):
    app.clientside_callback(
        f"""
        function() {{
            const keys = {json.dumps(ACCOUNT_LIST_CAPTURE_STORE_IDS)};
            const out = {{}};
            for (let i = 0; i < keys.length; i += 1) {{
                const key = keys[i];
                const raw = sessionStorage.getItem(key);
                if (raw == null) {{
                    continue;
                }}
                try {{
                    out[key] = JSON.parse(raw);
                }} catch (err) {{
                    out[key] = null;
                }}
            }}
            return out;
        }}
        """,
        Output("dashmat-account-list-session-snapshot-store", "data"),
        Input("at-menu-save-account-list", "n_clicks", allow_optional=True),
        Input("po-menu-save-account-list", "n_clicks", allow_optional=True),
        Input("reg-menu-save-account-list", "n_clicks", allow_optional=True),
        prevent_initial_call=True,
    )

    app.clientside_callback(
        """
        function(nClicks) {
            if (!nClicks) {
                return window.dash_clientside.no_update;
            }
            const start = (window.performance && typeof window.performance.now === "function")
                ? window.performance.now()
                : Date.now();
            const keys = __LOAD_KEYS__;
            const out = {};
            let keyCount = 0;
            let totalBytes = 0;
            for (let i = 0; i < keys.length; i += 1) {
                const key = keys[i];
                const raw = sessionStorage.getItem(key);
                if (raw == null) {
                    continue;
                }
                try {
                    out[key] = JSON.parse(raw);
                } catch (err) {
                    out[key] = null;
                }
                keyCount += 1;
                totalBytes += raw ? raw.length : 0;
            }
            const end = (window.performance && typeof window.performance.now === "function")
                ? window.performance.now()
                : Date.now();
            console.info(
                "timing name=account_list.load_snapshot_capture elapsed_ms=%s key_count=%s payload_bytes=%s",
                (end - start).toFixed(2),
                keyCount,
                totalBytes
            );
            sessionStorage.setItem(
                "dashmat-account-list-load-timing",
                JSON.stringify({
                    pagePath: window.location.pathname,
                    clickStartEpochMs: Date.now(),
                    status: "clicked"
                })
            );
            return out;
        }
        """.replace("__LOAD_KEYS__", json.dumps(ACCOUNT_LIST_LOAD_MERGE_STORE_IDS)),
        Output("dashmat-account-list-load-snapshot-store", "data"),
        Input("dashmat-account-list-load-button", "n_clicks"),
        prevent_initial_call=True,
    )

    app.clientside_callback(
        """
        function(applyEnvelope) {
            if (!applyEnvelope || typeof applyEnvelope !== "object") {
                return [
                    window.dash_clientside.no_update,
                    window.dash_clientside.no_update,
                    window.dash_clientside.no_update,
                    window.dash_clientside.no_update
                ];
            }
            const mode = String(applyEnvelope.mode || "reload");
            const targetPath = String(applyEnvelope.pathname || window.location.pathname || "");
            const sessionPayload = applyEnvelope.session_payload;
            if (!sessionPayload || typeof sessionPayload !== "object") {
                return [
                    null,
                    window.dash_clientside.no_update,
                    window.dash_clientside.no_update,
                    {status: "error"}
                ];
            }
            const stableStringify = function(value) {
                if (value === null || typeof value !== "object") {
                    return JSON.stringify(value);
                }
                if (Array.isArray(value)) {
                    return "[" + value.map(stableStringify).join(",") + "]";
                }
                const keys = Object.keys(value).sort();
                return "{" + keys.map(function(key) {
                    return JSON.stringify(key) + ":" + stableStringify(value[key]);
                }).join(",") + "}";
            };
            const start = (window.performance && typeof window.performance.now === "function")
                ? window.performance.now()
                : Date.now();
            let keyCount = 0;
            let changedKeyCount = 0;
            let changedBytes = 0;
            let noticePayload = window.dash_clientside.no_update;
            const changedEntries = [];
            Object.keys(sessionPayload).forEach(function(key) {
                keyCount += 1;
                if (key === "dashmat-account-list-notice-store") {
                    noticePayload = sessionPayload[key];
                    return;
                }
                const serialized = stableStringify(sessionPayload[key]);
                const existingRaw = sessionStorage.getItem(key);
                let existingSerialized = existingRaw;
                if (existingRaw != null) {
                    try {
                        existingSerialized = stableStringify(JSON.parse(existingRaw));
                    } catch (err) {
                        existingSerialized = existingRaw;
                    }
                }
                if (serialized === existingSerialized) {
                    return;
                }
                changedEntries.push([key, serialized]);
                changedKeyCount += 1;
                changedBytes += serialized ? serialized.length : 0;
            });
            const end = (window.performance && typeof window.performance.now === "function")
                ? window.performance.now()
                : Date.now();
            console.info(
                "timing name=account_list.session_apply elapsed_ms=%s key_count=%s changed_key_count=%s changed_payload_bytes=%s",
                (end - start).toFixed(2),
                keyCount,
                changedKeyCount,
                changedBytes
            );
            if (!changedEntries.length) {
                sessionStorage.removeItem("dashmat-account-list-load-timing");
                return [
                    null,
                    false,
                    noticePayload,
                    {status: "idle"}
                ];
            }
            changedEntries.forEach(function(entry) {
                sessionStorage.setItem(entry[0], entry[1]);
            });
            if (noticePayload !== window.dash_clientside.no_update) {
                sessionStorage.setItem("dashmat-account-list-notice-store", stableStringify(noticePayload));
            }
            const updateTiming = function(nextPayload) {
                try {
                    const timingPayload = JSON.parse(sessionStorage.getItem("dashmat-account-list-load-timing") || "{}");
                    Object.keys(nextPayload).forEach(function(key) {
                        timingPayload[key] = nextPayload[key];
                    });
                    timingPayload.pagePath = timingPayload.pagePath || targetPath || window.location.pathname;
                    timingPayload.clickStartEpochMs = timingPayload.clickStartEpochMs || Date.now();
                    sessionStorage.setItem("dashmat-account-list-load-timing", JSON.stringify(timingPayload));
                } catch (err) {
                    const fallback = Object.assign({
                        pagePath: targetPath || window.location.pathname,
                        clickStartEpochMs: Date.now()
                    }, nextPayload);
                    sessionStorage.setItem("dashmat-account-list-load-timing", JSON.stringify(fallback));
                }
            };
            const fallbackToReload = function() {
                updateTiming({
                    reloadStartEpochMs: Date.now(),
                    status: "reloading"
                });
                window.location.reload();
                return [
                    window.dash_clientside.no_update,
                    window.dash_clientside.no_update,
                    window.dash_clientside.no_update,
                    window.dash_clientside.no_update
                ];
            };
            if (mode === "live_apply_analyticstool" && targetPath.indexOf("/analyticstool") === 0) {
                try {
                    if (!(window.dash_clientside && typeof window.dash_clientside.set_props === "function")) {
                        throw new Error("set_props unavailable");
                    }
                    const changedEntryMap = {};
                    changedEntries.forEach(function(entry) {
                        changedEntryMap[entry[0]] = sessionPayload[entry[0]];
                    });
                    if (noticePayload !== window.dash_clientside.no_update) {
                        changedEntryMap["dashmat-account-list-notice-store"] = noticePayload;
                    }
                    updateTiming({
                        liveApplyCommitEpochMs: Date.now(),
                        status: "live_applying"
                    });
                    window.dash_clientside.set_props("at-state-ready-store", {data: false});
                    __AT_SHARED_IDS__.forEach(function(id) {
                        if (!Object.prototype.hasOwnProperty.call(changedEntryMap, id)) {
                            return;
                        }
                        window.dash_clientside.set_props(id, {data: changedEntryMap[id]});
                    });
                    __AT_STORE_IDS__.forEach(function(id) {
                        if (!Object.prototype.hasOwnProperty.call(changedEntryMap, id)) {
                            return;
                        }
                        window.dash_clientside.set_props(id, {data: changedEntryMap[id]});
                    });
                    return [
                        null,
                        false,
                        noticePayload,
                        {status: "live_applying"}
                    ];
                } catch (err) {
                    console.warn("account_list.live_apply_fallback", err && err.message ? err.message : err);
                    return fallbackToReload();
                }
            }
            return fallbackToReload();
        }
        """
        .replace("__AT_STORE_IDS__", json.dumps(
            list(dict.fromkeys(list(AT_STORE_IDS.values()) + list(AT_EXTRA_CONTROL_STORE_IDS)))
        ))
        .replace("__AT_SHARED_IDS__", json.dumps([
            "dashmat-db-import-provenance-store",
            "dashmat-original-periodicity-store",
            "dashmat-raw-data-store",
        ])),
        Output("dashmat-account-list-session-apply-store", "data", allow_duplicate=True),
        Output("dashmat-account-list-modal", "opened", allow_duplicate=True),
        Output("dashmat-account-list-notice-store", "data", allow_duplicate=True),
        Output("dashmat-account-list-load-state-store", "data", allow_duplicate=True),
        Input("dashmat-account-list-session-apply-store", "data"),
        prevent_initial_call=True,
    )

    app.clientside_callback(
        """
        function(pathname, atReady, poBootstrap, regReady, loadState) {
            const raw = sessionStorage.getItem("dashmat-account-list-load-timing");
            if (!raw) {
                return [window.dash_clientside.no_update, window.dash_clientside.no_update];
            }
            let timingPayload;
            try {
                timingPayload = JSON.parse(raw);
            } catch (err) {
                sessionStorage.removeItem("dashmat-account-list-load-timing");
                return [window.dash_clientside.no_update, window.dash_clientside.no_update];
            }
            if (loadState && loadState.status === "error") {
                sessionStorage.removeItem("dashmat-account-list-load-timing");
                return [window.dash_clientside.no_update, window.dash_clientside.no_update];
            }
            const currentPath = pathname || window.location.pathname || "";
            if (timingPayload.pagePath && timingPayload.pagePath !== currentPath) {
                return [window.dash_clientside.no_update, window.dash_clientside.no_update];
            }
            const isAnalytics = currentPath.indexOf("/analyticstool") === 0;
            const isPortopt = currentPath.indexOf("/portopt") === 0;
            const isRegression = currentPath.indexOf("/regression") === 0;
            let ready = false;
            if (isAnalytics) {
                ready = !!atReady;
            } else if (isPortopt) {
                ready = !!(poBootstrap && poBootstrap.phase === "ready");
            } else if (isRegression) {
                ready = !!regReady;
            }
            if (!ready) {
                return [window.dash_clientside.no_update, window.dash_clientside.no_update];
            }
            const now = Date.now();
            if (timingPayload.status === "reloading" && timingPayload.reloadStartEpochMs) {
                const clickStart = Number(timingPayload.clickStartEpochMs || timingPayload.reloadStartEpochMs || now);
                const reloadStart = Number(timingPayload.reloadStartEpochMs || now);
                console.info(
                    "timing name=account_list.click_to_ready click_to_reload_start_ms=%s reload_start_to_ready_ms=%s total_click_to_ready_ms=%s page=%s",
                    String(reloadStart - clickStart),
                    String(now - reloadStart),
                    String(now - clickStart),
                    currentPath
                );
                sessionStorage.removeItem("dashmat-account-list-load-timing");
                return [{
                    pagePath: currentPath,
                    clickToReloadStartMs: reloadStart - clickStart,
                    reloadStartToReadyMs: now - reloadStart,
                    totalClickToReadyMs: now - clickStart
                }, window.dash_clientside.no_update];
            }
            if (timingPayload.status === "live_applying" && timingPayload.liveApplyCommitEpochMs) {
                const clickStart = Number(timingPayload.clickStartEpochMs || timingPayload.liveApplyCommitEpochMs || now);
                const commitStart = Number(timingPayload.liveApplyCommitEpochMs || now);
                console.info(
                    "timing name=account_list.click_to_ready click_to_live_apply_commit_ms=%s live_apply_commit_to_ready_ms=%s total_click_to_ready_ms=%s page=%s",
                    String(commitStart - clickStart),
                    String(now - commitStart),
                    String(now - clickStart),
                    currentPath
                );
                sessionStorage.removeItem("dashmat-account-list-load-timing");
                return [{
                    pagePath: currentPath,
                    clickToLiveApplyCommitMs: commitStart - clickStart,
                    liveApplyCommitToReadyMs: now - commitStart,
                    totalClickToReadyMs: now - clickStart
                }, {status: "idle"}];
            }
            return [window.dash_clientside.no_update, window.dash_clientside.no_update];
        }
        """,
        Output("dashmat-account-list-load-timing-dummy", "data"),
        Output("dashmat-account-list-load-state-store", "data", allow_duplicate=True),
        Input("_pages_location", "pathname"),
        Input("at-state-ready-store", "data", allow_optional=True),
        Input("po-bootstrap-store", "data", allow_optional=True),
        Input("reg-initial-tab-render-ready-store", "data", allow_optional=True),
        Input("dashmat-account-list-load-state-store", "data"),
        prevent_initial_call="initial_duplicate",
    )

    app.clientside_callback(
        """
        function(modalOpened, loadClicks, currentLoadState) {
            const ctx = window.dash_clientside.callback_context;
            const triggered = (ctx && ctx.triggered && ctx.triggered.length) ? ctx.triggered[0].prop_id.split(".")[0] : null;
            if (triggered === "dashmat-account-list-modal") {
                const status = (currentLoadState && currentLoadState.status) ? String(currentLoadState.status).toLowerCase() : "idle";
                if (!modalOpened && (status === "loading" || status === "live_applying")) {
                    return window.dash_clientside.no_update;
                }
                return {status: "idle"};
            }
            if (triggered === "dashmat-account-list-load-button" && loadClicks) {
                return {status: "loading"};
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output("dashmat-account-list-load-state-store", "data", allow_duplicate=True),
        Input("dashmat-account-list-modal", "opened"),
        Input("dashmat-account-list-load-button", "n_clicks"),
        State("dashmat-account-list-load-state-store", "data"),
        prevent_initial_call=True,
    )

    @app.callback(
        Output("dashmat-account-list-load-overlay", "visible"),
        Output("dashmat-account-list-load-overlay-shell", "style"),
        Input("dashmat-account-list-load-state-store", "data"),
        prevent_initial_call=False,
    )
    def _sync_account_list_load_overlay(load_state):
        return account_list_loader_visible(load_state), account_list_loader_wrapper_style(load_state)

    app.clientside_callback(
        """
        function(opened, mode, saveDisabled, currentClicks) {
            const handlerKey = "__dashmatAccountListEnterHandler";
            const resolveInput = function() {
                const direct = document.querySelector("input#dashmat-account-list-name-input");
                if (direct) {
                    return direct;
                }
                const host = document.getElementById("dashmat-account-list-name-input");
                if (!host) {
                    return null;
                }
                if (host.tagName === "INPUT") {
                    return host;
                }
                return host.querySelector("input");
            };
            if (window[handlerKey]) {
                document.removeEventListener("keydown", window[handlerKey], true);
                window[handlerKey] = null;
            }
            if (!(opened && mode === "save")) {
                return window.dash_clientside.no_update;
            }
            const handler = function(event) {
                if (event.key !== "Enter" || event.shiftKey || event.ctrlKey || event.altKey || event.metaKey) {
                    return;
                }
                const target = event.target;
                const input = resolveInput();
                if (!target || !input || target !== input) {
                    return;
                }
                if (saveDisabled) {
                    return;
                }
                event.preventDefault();
                event.stopPropagation();
                window.dash_clientside.set_props("dashmat-account-list-save-button", {
                    n_clicks: (currentClicks || 0) + 1
                });
            };
            window[handlerKey] = handler;
            document.addEventListener("keydown", handler, true);
            return window.dash_clientside.no_update;
        }
        """,
        Output("dashmat-account-list-enter-submit-dummy", "data"),
        Input("dashmat-account-list-modal", "opened"),
        Input("dashmat-account-list-modal-mode-store", "data"),
        Input("dashmat-account-list-save-button", "disabled"),
        State("dashmat-account-list-save-button", "n_clicks"),
        prevent_initial_call=False,
    )

    app.clientside_callback(
        """
        function(opened, mode) {
            if (!(opened && mode === "save")) {
                return window.dash_clientside.no_update;
            }
            const resolveInput = function() {
                const direct = document.querySelector("input#dashmat-account-list-name-input");
                if (direct) {
                    return direct;
                }
                const host = document.getElementById("dashmat-account-list-name-input");
                if (!host) {
                    return null;
                }
                if (host.tagName === "INPUT") {
                    return host;
                }
                return host.querySelector("input");
            };
            const focusInput = function(attempt) {
                const input = resolveInput();
                if (!input) {
                    if (attempt < 30) {
                        setTimeout(function() { focusInput(attempt + 1); }, 50);
                    }
                    return;
                }
                input.focus();
                if (typeof input.select === "function") {
                    input.select();
                }
                if (document.activeElement !== input && attempt < 30) {
                    setTimeout(function() { focusInput(attempt + 1); }, 50);
                }
            };
            setTimeout(function() { focusInput(0); }, 120);
            return window.dash_clientside.no_update;
        }
        """,
        Output("dashmat-account-list-focus-dummy", "data"),
        Input("dashmat-account-list-modal", "opened"),
        Input("dashmat-account-list-modal-mode-store", "data"),
        prevent_initial_call=False,
    )

    @app.callback(
        Output("dashmat-db-import-provenance-store", "data"),
        Input("dashmat-raw-data-meta-store", "data"),
        State("dashmat-db-import-provenance-store", "data"),
        prevent_initial_call=False,
    )
    def _refresh_db_import_provenance(raw_meta, provenance_store):
        return refresh_db_import_provenance(raw_meta, provenance_store)

    @app.callback(
        Output("dashmat-account-list-notice-container", "children"),
        Input("dashmat-account-list-notice-store", "data"),
        prevent_initial_call=False,
    )
    def _render_account_list_notice(notice):
        return render_account_list_notice(notice)

    @app.callback(
        Output("dashmat-account-list-notice-store", "data", allow_duplicate=True),
        Input("dashmat-account-list-notice-close-button", "n_clicks", allow_optional=True),
        prevent_initial_call=True,
    )
    def _dismiss_account_list_notice(n_clicks):
        if not n_clicks:
            raise PreventUpdate
        return None

    @app.callback(
        Output("dashmat-account-list-modal", "opened"),
        Output("dashmat-account-list-modal-mode-store", "data"),
        Output("dashmat-account-list-name-input", "value"),
        Output("dashmat-account-list-selected-id-store", "data"),
        Output("dashmat-account-list-apply-settings-switch", "checked"),
        Input("at-menu-save-account-list", "n_clicks", allow_optional=True),
        Input("at-menu-load-account-list", "n_clicks", allow_optional=True),
        Input("at-welcome-load-account-list-btn", "n_clicks", allow_optional=True),
        Input("po-menu-save-account-list", "n_clicks", allow_optional=True),
        Input("po-menu-load-account-list", "n_clicks", allow_optional=True),
        Input("po-welcome-load-account-list-btn", "n_clicks", allow_optional=True),
        Input("reg-menu-save-account-list", "n_clicks", allow_optional=True),
        Input("reg-menu-load-account-list", "n_clicks", allow_optional=True),
        Input("reg-welcome-load-account-list-btn", "n_clicks", allow_optional=True),
        Input("dashmat-account-list-close-button", "n_clicks"),
        prevent_initial_call=True,
    )
    def _toggle_account_list_modal(
        at_save,
        at_load,
        at_welcome_load,
        po_save,
        po_load,
        po_welcome_load,
        reg_save,
        reg_load,
        reg_welcome_load,
        close_clicks,
    ):
        from dash import callback_context

        return toggle_account_list_modal(
            at_save,
            at_load,
            at_welcome_load,
            po_save,
            po_load,
            po_welcome_load,
            reg_save,
            reg_load,
            reg_welcome_load,
            close_clicks,
            triggered_id=callback_context.triggered_id,
        )

    @app.callback(
        Output("dashmat-account-list-rows-store", "data"),
        Input("dashmat-account-list-modal", "opened"),
        Input("dashmat-account-list-refresh-store", "data"),
        State("userinfo", "data"),
        State("dashmat-account-list-load-state-store", "data"),
        prevent_initial_call=True,
    )
    def _refresh_account_list_rows(opened, refresh_count, userinfo, load_state):
        if not opened or account_list_load_state_status(load_state) != "idle":
            raise PreventUpdate
        if not account_list_tables_available(db_engine):
            return []
        return list_account_lists(db_engine, _account_list_username(userinfo))

    @app.callback(
        Output("dashmat-account-list-selected-detail-store", "data"),
        Input("dashmat-account-list-modal", "opened"),
        Input("dashmat-account-list-modal-mode-store", "data"),
        Input("dashmat-account-list-selected-id-store", "data"),
        State("userinfo", "data"),
        State("dashmat-account-list-load-state-store", "data"),
        prevent_initial_call=False,
    )
    def _refresh_account_list_selected_detail(opened, mode, selected_id, userinfo, load_state):
        if account_list_load_state_status(load_state) != "idle":
            raise PreventUpdate
        if not opened or str(mode or "load") != "load" or selected_id is None:
            return None
        return load_selected_account_list_detail(
            selected_id=selected_id,
            userinfo=userinfo,
            db_engine=db_engine,
        )

    @app.callback(
        Output("dashmat-account-list-prefetch-store", "data"),
        Input("dashmat-account-list-modal", "opened"),
        Input("dashmat-account-list-modal-mode-store", "data"),
        Input("dashmat-account-list-selected-detail-store", "data"),
        State("dashmat-account-list-prefetch-store", "data"),
        prevent_initial_call=False,
    )
    def _prefetch_account_list_entries(opened, mode, selected_detail, current_prefetch):
        return prefetch_selected_account_list_entries(
            opened=opened,
            mode=mode,
            selected_detail=selected_detail,
            current_prefetch=current_prefetch,
            db_engine=db_engine,
            mrd_engine=mrd_engine,
            perf_engine=perf_engine,
        )

    @app.callback(
        Output("dashmat-account-list-send-user-select", "data"),
        Input("dashmat-account-list-modal", "opened"),
        Input("dashmat-account-list-modal-mode-store", "data"),
        Input("dashmat-account-list-selected-id-store", "data"),
        State("userinfo", "data"),
        State("dashmat-account-list-load-state-store", "data"),
        prevent_initial_call=False,
    )
    def _refresh_account_list_send_users(opened, mode, selected_id, userinfo, load_state):
        if account_list_load_state_status(load_state) != "idle":
            raise PreventUpdate
        if not opened or str(mode or "load") != "load" or selected_id is None:
            return []
        if not users_table_available(db_engine):
            return []
        users = list_account_list_users(db_engine, _account_list_username(userinfo))
        return account_list_send_user_options(users)

    @app.callback(
        Output("dashmat-account-list-selected-id-store", "data", allow_duplicate=True),
        Input("dashmat-account-list-grid", "selectedRows"),
        prevent_initial_call=True,
    )
    def _sync_account_list_selected_id(selected_rows):
        return sync_account_list_selected_id(selected_rows)

    @app.callback(
        Output("dashmat-account-list-send-user-select", "value"),
        Input("dashmat-account-list-modal", "opened"),
        Input("dashmat-account-list-selected-id-store", "data"),
        prevent_initial_call=False,
    )
    def _reset_account_list_send_user(opened, selected_id):
        if not opened:
            return None
        return None

    @app.callback(
        Output("dashmat-account-list-send-controls", "style"),
        Output("dashmat-account-list-send-user-select", "disabled"),
        Output("dashmat-account-list-send-button", "disabled"),
        Output("dashmat-account-list-send-user-select", "placeholder"),
        Input("dashmat-account-list-modal-mode-store", "data"),
        Input("dashmat-account-list-selected-id-store", "data"),
        Input("dashmat-account-list-send-user-select", "data"),
        Input("dashmat-account-list-send-user-select", "value"),
        prevent_initial_call=False,
    )
    def _sync_account_list_send_controls(mode, selected_id, recipient_options, recipient_value):
        return account_list_send_controls_state(mode, selected_id, recipient_options, recipient_value)

    @app.callback(
        Output("dashmat-account-list-modal", "title"),
        Output("dashmat-account-list-modal", "className"),
        Output("dashmat-account-list-save-section", "style"),
        Output("dashmat-account-list-load-section", "style"),
        Output("dashmat-account-list-grid", "rowData"),
        Output("dashmat-account-list-preview-grid", "rowData"),
        Output("dashmat-account-list-modal-message", "children"),
        Output("dashmat-account-list-delete-button", "disabled"),
        Output("dashmat-account-list-load-button", "disabled"),
        Output("dashmat-account-list-delete-button", "style"),
        Output("dashmat-account-list-load-button", "style"),
        Output("dashmat-account-list-save-button", "style"),
        Input("dashmat-account-list-modal", "opened"),
        Input("dashmat-account-list-modal-mode-store", "data"),
        Input("dashmat-account-list-rows-store", "data"),
        Input("dashmat-account-list-selected-id-store", "data"),
        Input("dashmat-account-list-selected-detail-store", "data"),
        prevent_initial_call=False,
    )
    def _render_account_list_modal_view(opened, mode, rows, selected_id, selected_detail):
        return render_account_list_modal_view(opened, mode, rows, selected_id, selected_detail)

    @app.callback(
        Output("dashmat-account-list-duplicate-text", "children"),
        Output("dashmat-account-list-save-button", "disabled"),
        Input("dashmat-account-list-modal-mode-store", "data"),
        Input("dashmat-account-list-name-input", "value"),
        Input("dashmat-account-list-rows-store", "data"),
        Input("dashmat-db-import-provenance-store", "data"),
        prevent_initial_call=False,
    )
    def _sync_account_list_save_state(mode, name_value, rows, provenance_store):
        return sync_account_list_save_state(mode, name_value, rows, provenance_store)

    @app.callback(
        Output("dashmat-account-list-modal", "opened", allow_duplicate=True),
        Output("dashmat-account-list-notice-store", "data", allow_duplicate=True),
        Input("dashmat-account-list-save-button", "n_clicks"),
        State("dashmat-account-list-name-input", "value"),
        State("dashmat-account-list-session-snapshot-store", "data"),
        State("dashmat-db-import-provenance-store", "data"),
        State("userinfo", "data"),
        State("dashmat-raw-data-store", "data"),
        prevent_initial_call=True,
    )
    def _save_current_account_list(n_clicks, name_value, session_snapshot, provenance_store, userinfo,
                                    raw_data_store):
        if not n_clicks:
            raise PreventUpdate
        if not account_list_tables_available(db_engine):
            return no_update, {"message": "Account-list tables are unavailable.", "color": "red"}
        payload = build_account_list_payload(provenance_store, session_snapshot, raw_data_store)
        ok, message, _saved = save_account_list(
            db_engine,
            username=_account_list_username(userinfo),
            update_by=_account_list_update_by(userinfo),
            list_name=str(name_value or "").strip(),
            payload=payload,
        )
        if not ok:
            return True, {"message": message, "color": "red"}
        return False, {"message": message, "color": "green"}

    @app.callback(
        Output("dashmat-account-list-refresh-store", "data", allow_duplicate=True),
        Output("dashmat-account-list-selected-id-store", "data", allow_duplicate=True),
        Output("dashmat-account-list-notice-store", "data", allow_duplicate=True),
        Input("dashmat-account-list-delete-button", "n_clicks"),
        State("dashmat-account-list-selected-id-store", "data"),
        State("dashmat-account-list-rows-store", "data"),
        State("userinfo", "data"),
        State("dashmat-account-list-refresh-store", "data"),
        prevent_initial_call=True,
    )
    def _delete_selected_account_list(n_clicks, selected_id, rows, userinfo, refresh_counter):
        if not n_clicks:
            raise PreventUpdate
        selected_row = next((row for row in (rows or []) if row.get("AccountListID") == selected_id), None)
        if selected_row is None:
            return no_update, None, {"message": "Select an account list to delete.", "color": "orange"}
        ok, message = delete_account_list(
            db_engine,
            account_list_id=selected_id,
            username=_account_list_username(userinfo),
            expected_update_date=selected_row.get("UPDATE_DATE"),
        )
        return (
            int(refresh_counter or 0) + (1 if ok else 0),
            None if ok else selected_id,
            {"message": message, "color": "green" if ok else "red"},
        )

    @app.callback(
        Output("dashmat-account-list-send-user-select", "value", allow_duplicate=True),
        Output("dashmat-account-list-notice-store", "data", allow_duplicate=True),
        Input("dashmat-account-list-send-button", "n_clicks"),
        State("dashmat-account-list-selected-id-store", "data"),
        State("dashmat-account-list-rows-store", "data"),
        State("dashmat-account-list-send-user-select", "value"),
        State("userinfo", "data"),
        prevent_initial_call=True,
    )
    def _send_selected_account_list(n_clicks, selected_id, rows, recipient_username, userinfo):
        if not n_clicks:
            raise PreventUpdate
        selected_row = next((row for row in (rows or []) if row.get("AccountListID") == selected_id), None)
        if selected_row is None:
            return no_update, {"message": "Select an account list to send.", "color": "orange"}
        if not str(recipient_username or "").strip():
            return no_update, {"message": "Select a user to send to.", "color": "orange"}
        ok, message = send_account_list(
            db_engine,
            account_list_id=selected_id,
            sender_username=_account_list_username(userinfo),
            recipient_username=str(recipient_username or "").strip(),
            expected_update_date=selected_row.get("UPDATE_DATE"),
        )
        return (
            None if ok else no_update,
            {"message": message, "color": "green" if ok else "red"},
        )

    @app.callback(
        Output("dashmat-account-list-session-apply-store", "data", allow_duplicate=True),
        Output("dashmat-account-list-notice-store", "data", allow_duplicate=True),
        Output("dashmat-account-list-load-state-store", "data", allow_duplicate=True),
        Input("dashmat-account-list-load-snapshot-store", "data"),
        State("dashmat-account-list-load-button", "n_clicks"),
        State("dashmat-account-list-selected-id-store", "data"),
        State("dashmat-account-list-selected-detail-store", "data"),
        State("dashmat-account-list-rows-store", "data"),
        State("dashmat-account-list-apply-settings-switch", "checked"),
        State("dashmat-raw-data-store", "data"),
        State("dashmat-original-periodicity-store", "data"),
        State("dashmat-db-import-provenance-store", "data"),
        State("_pages_location", "pathname"),
        State("userinfo", "data"),
        prevent_initial_call=True,
    )
    def _load_selected_account_list(
        load_snapshot,
        n_clicks,
        selected_id,
        selected_detail,
        rows,
        apply_settings,
        raw_data,
        original_periodicity,
        provenance_store,
        pathname,
        userinfo,
    ):
        return load_selected_account_list_session(
            n_clicks=n_clicks,
            selected_id=selected_id,
            selected_detail=selected_detail,
            rows=rows,
            apply_settings=apply_settings,
            raw_data=raw_data,
            original_periodicity=original_periodicity,
            provenance_store=provenance_store,
            load_snapshot=load_snapshot,
            pathname=pathname,
            userinfo=userinfo,
            db_engine=db_engine,
            mrd_engine=mrd_engine,
            perf_engine=perf_engine,
        )
