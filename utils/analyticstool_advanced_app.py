from __future__ import annotations

import re
import uuid
from pathlib import Path
from urllib.parse import parse_qs

import dash_mantine_components as dmc
from dash import ClientsideFunction, Dash, Input, Output, State, dcc, html
from dash_iconify import DashIconify
from dash.exceptions import PreventUpdate
from flask import redirect

import cache_config
from utils.account_list_modal import (
    build_account_list_components,
    register_account_list_callbacks,
)
from utils.raw_dataset import cache_raw_dataset, normalize_raw_data_store
from utils.returns import _build_raw_data_metadata_cached

ADVANCED_ROUTE = "/analyticstool-advanced/"
HANDOFF_CACHE_PREFIX = "at_advanced_handoff:"
HANDOFF_TTL_SECONDS = 1800
ADVANCED_SOURCE_PATH = Path(__file__).resolve().parent / "analyticstool_advanced_source.py.txt"
ASSETS_PATH = Path(__file__).resolve().parents[1] / "assets"


def _load_advanced_source() -> str:
    source = ADVANCED_SOURCE_PATH.read_text(encoding="utf-8")
    source = source.replace(
        "from dash import ClientsideFunction, Input, Output, State, callback, dcc, html, no_update, register_page, ALL, clientside_callback, callback_context",
        "from dash import ClientsideFunction, Input, Output, State, dcc, html, no_update, ALL, callback_context",
    )
    source = source.replace(
        'register_page(__name__, path="/analyticstool", name="Analytics Tool", title="Analytics Tool")',
        "",
    )
    source = re.sub(r"(?m)^from __future__ import annotations\s*$", "", source)
    source = source.lstrip("\ufeff")
    return source


def build_handoff_payload(
    raw_data,
    original_periodicity,
    periodicity,
    selected_series,
    returns_type,
    benchmark_assignments,
    long_short_assignments,
    date_range,
    vol_scaler,
    vol_scaling_assignments,
    series_order,
) -> str:
    if not raw_data:
        raise ValueError("raw_data is required for advanced handoff")

    token = str(uuid.uuid4())
    cache_config.cache.set(
        f"{HANDOFF_CACHE_PREFIX}{token}",
        {
            "raw_data": raw_data,
            "original_periodicity": original_periodicity or "daily",
            "periodicity": periodicity or "daily",
            "selected_series": list(selected_series or []),
            "returns_type": returns_type or "total",
            "benchmark_assignments": dict(benchmark_assignments or {}),
            "long_short_assignments": dict(long_short_assignments or {}),
            "date_range": date_range,
            "vol_scaler": vol_scaler or 0,
            "vol_scaling_assignments": dict(vol_scaling_assignments or {}),
            "series_order": list(series_order or []),
        },
        timeout=HANDOFF_TTL_SECONDS,
    )
    return token


def _build_shell(page_layout):
    header = dmc.Group(
        justify="space-between",
        mb="md",
        children=[
            dmc.Group(
                gap="sm",
                children=[
                    dmc.Anchor(
                        dmc.Button(
                            "Back to Analytics Tool",
                            variant="light",
                            color="blue",
                            leftSection=DashIconify(icon="tabler:arrow-left", width=16),
                        ),
                        href="/analyticstool",
                        style={"textDecoration": "none"},
                    ),
                    dmc.Text("Advanced Analytics Workspace", fw=700),
                ],
            ),
            dmc.ColorSchemeToggle(
                id="global-color-scheme-toggle",
                variant="gradient",
                gradient={"from": "orange", "to": "blue", "deg": 135},
                lightIcon=DashIconify(icon="tabler:sun-filled", width=16),
                darkIcon=DashIconify(icon="tabler:moon-stars", width=16),
                autoContrast=True,
                radius="xl",
                size="md",
            ),
        ],
    )

    return dmc.MantineProvider(
        defaultColorScheme="light",
        children=[
            dcc.Location(id="_pages_location", refresh=False),
            dcc.Store(id="userinfo", data={"role": "Admin", "username": "Admin User"}, storage_type="session"),
            dcc.Store(id="dashmat-raw-data-store", data=None, storage_type="session"),
            dcc.Store(id="dashmat-raw-data-identity-store", data=None, storage_type="memory"),
            dcc.Store(id="dashmat-raw-data-meta-store", data=None, storage_type="session"),
            dcc.Store(id="dashmat-original-periodicity-store", data="daily", storage_type="session"),
            dcc.Store(id="dashmat-pending-new-series-store", data={}, storage_type="session"),
            dcc.Store(id="dashmat-saved-series-cache-store", data=None, storage_type="session"),
            dcc.Store(id="dashmat-db-import-provenance-store", data={}, storage_type="session"),
            *build_account_list_components(),
            dmc.Container([header, page_layout], fluid=True, px="md", py="sm"),
        ],
    )


def mount_analyticstool_advanced(server, *, db_engine, mrd_engine, perf_engine):
    advanced_app = Dash(
        __name__,
        server=server,
        routes_pathname_prefix=ADVANCED_ROUTE,
        requests_pathname_prefix=ADVANCED_ROUTE,
        assets_folder=str(ASSETS_PATH),
        suppress_callback_exceptions=True,
        title="Analytics Tool Advanced",
    )

    namespace = {"advanced_app": advanced_app}
    exec("callback = advanced_app.callback\nclientside_callback = advanced_app.clientside_callback\nregister_page = lambda *args, **kwargs: None", namespace)
    exec(compile(_load_advanced_source(), str(ADVANCED_SOURCE_PATH), "exec"), namespace)

    page_layout = namespace["layout"]
    advanced_app.layout = _build_shell(page_layout)

    advanced_app.clientside_callback(
        ClientsideFunction(namespace="dashmat_callbacks", function_name="patchPlotlyTheme"),
        Output("global-color-scheme-toggle", "title"),
        Input("global-color-scheme-toggle", "computedColorScheme"),
        prevent_initial_call=True,
    )

    advanced_app.clientside_callback(
        """
        function(rawData, currentIdentity) {
            let datasetKey = null;
            let hasData = false;
            if (rawData && typeof rawData === "object" && !Array.isArray(rawData)) {
                datasetKey = (rawData.dataset_key || "").toString().trim() || null;
                hasData = !!datasetKey;
            }
            const nextIdentity = {dataset_key: datasetKey, has_data: hasData};
            if (currentIdentity) {
                const currentDatasetKey = ((currentIdentity.dataset_key || "").toString().trim()) || null;
                const currentHasData = !!currentIdentity.has_data;
                if (currentDatasetKey === datasetKey && currentHasData === hasData) {
                    return window.dash_clientside.no_update;
                }
            }
            return nextIdentity;
        }
        """,
        Output("dashmat-raw-data-identity-store", "data"),
        Input("dashmat-raw-data-store", "data"),
        State("dashmat-raw-data-identity-store", "data"),
        prevent_initial_call=False,
    )

    @advanced_app.callback(
        Output("dashmat-raw-data-meta-store", "data"),
        Input("dashmat-raw-data-identity-store", "data"),
        Input("dashmat-original-periodicity-store", "data"),
        prevent_initial_call=False,
    )
    def refresh_raw_data_meta_store(raw_data_identity, original_periodicity):
        dataset_key = None
        if isinstance(raw_data_identity, dict):
            dataset_key = str(raw_data_identity.get("dataset_key") or "").strip() or None
        return _build_raw_data_metadata_cached(dataset_key, original_periodicity)

    @advanced_app.callback(
        Output("dashmat-raw-data-store", "data", allow_duplicate=True),
        Output("dashmat-original-periodicity-store", "data", allow_duplicate=True),
        Output("at-periodicity-value-store", "data", allow_duplicate=True),
        Output("at-series-select-value-store", "data", allow_duplicate=True),
        Output("at-returns-type-value-store", "data", allow_duplicate=True),
        Output("at-vol-scaler-value-store", "data", allow_duplicate=True),
        Output("at-benchmark-assignments-store", "data", allow_duplicate=True),
        Output("at-long-short-store", "data", allow_duplicate=True),
        Output("at-vol-scaling-assignments-store", "data", allow_duplicate=True),
        Output("at-date-range-store", "data", allow_duplicate=True),
        Output("at-series-order-store", "data", allow_duplicate=True),
        Output("at-active-tab-store", "data", allow_duplicate=True),
        Output("at-page-visited-store", "data", allow_duplicate=True),
        Input("at-url-location", "search"),
        prevent_initial_call="initial_duplicate",
    )
    def hydrate_from_handoff(search):
        params = parse_qs((search or "").lstrip("?"))
        token = str((params.get("handoff") or [None])[0] or "").strip()
        if not token:
            raise PreventUpdate

        payload = cache_config.cache.get(f"{HANDOFF_CACHE_PREFIX}{token}")
        if not isinstance(payload, dict):
            raise PreventUpdate

        raw_store_payload = normalize_raw_data_store(payload.get("raw_data"))
        if raw_store_payload is None:
            raise PreventUpdate
        cache_raw_dataset(raw_store_payload)
        return (
            raw_store_payload,
            payload.get("original_periodicity") or "daily",
            payload.get("periodicity") or "daily",
            payload.get("selected_series") or [],
            payload.get("returns_type") or "total",
            payload.get("vol_scaler") or 0,
            payload.get("benchmark_assignments") or {},
            payload.get("long_short_assignments") or {},
            payload.get("vol_scaling_assignments") or {},
            payload.get("date_range"),
            payload.get("series_order") or [],
            "factor_analysis",
            True,
        )

    register_account_list_callbacks(
        advanced_app,
        db_engine=db_engine,
        mrd_engine=mrd_engine,
        perf_engine=perf_engine,
    )

    server.add_url_rule(
        "/analyticstool-advanced",
        endpoint="analyticstool_advanced_redirect",
        view_func=lambda: redirect(ADVANCED_ROUTE),
    )
    server.add_url_rule(
        ADVANCED_ROUTE,
        endpoint="analyticstool_advanced_index",
        view_func=advanced_app.index,
    )
    return advanced_app
