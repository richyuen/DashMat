"""Helpers for passing cross-route landing actions into workspace pages."""

from __future__ import annotations

from typing import Any

import pandas as pd

from utils.page_paths import normalize_landing_module


ACTION_CONFIGURE_AFTER_IMPORT = "configure_after_import"
ACTION_OPEN_IMPORT_MODAL = "open_import_modal"

FLOW_DB = "db"
FLOW_RAW = "raw"
FLOW_PORTFOLIO = "portfolio"
FLOW_UNDERLYING = "underlying"

ROUTE_INTENT_MAX_AGE_SECONDS = 30

VALID_ACTIONS = {
    ACTION_CONFIGURE_AFTER_IMPORT,
    ACTION_OPEN_IMPORT_MODAL,
}


def build_route_intent(
    target_module: str | None,
    action: str,
    **extra: Any,
) -> dict[str, Any]:
    if action not in VALID_ACTIONS:
        raise ValueError(f"Unsupported route intent action: {action}")
    created_at = pd.Timestamp.now(tz="UTC").isoformat()
    payload: dict[str, Any] = {
        "token": created_at,
        "created_at": created_at,
        "target_module": normalize_landing_module(target_module),
        "action": action,
    }
    for key, value in extra.items():
        if value is not None:
            payload[str(key)] = value
    return payload


def matches_route_intent(
    intent: Any,
    target_module: str | None,
    action: str | None = None,
) -> bool:
    if not isinstance(intent, dict):
        return False
    if normalize_landing_module(intent.get("target_module")) != normalize_landing_module(target_module):
        return False
    if action is None:
        return True
    return str(intent.get("action") or "").strip().lower() == str(action).strip().lower()


def route_intent_action(intent: Any) -> str:
    if not isinstance(intent, dict):
        return ""
    return str(intent.get("action") or "").strip().lower()


def route_intent_value(intent: Any, key: str, default: Any = None) -> Any:
    if not isinstance(intent, dict):
        return default
    return intent.get(key, default)


def _parse_route_intent_timestamp(value: Any) -> pd.Timestamp | None:
    if value in (None, ""):
        return None
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed


def route_intent_token_to_consume(
    intent: Any,
    target_module: str | None,
    action: str,
    consumed_token: Any,
    *,
    flow: str | None = None,
) -> str | None:
    token = str(route_intent_value(intent, "token", "") or "").strip()
    if not token or token == str(consumed_token or "").strip():
        return None
    if not matches_route_intent(intent, target_module, action):
        return None
    if flow is not None and route_intent_value(intent, "flow") != flow:
        return None

    created_at = _parse_route_intent_timestamp(route_intent_value(intent, "created_at", token))
    if created_at is None:
        return None

    max_age = pd.Timedelta(seconds=ROUTE_INTENT_MAX_AGE_SECONDS)
    if pd.Timestamp.now(tz="UTC") - created_at > max_age:
        return None
    return token
