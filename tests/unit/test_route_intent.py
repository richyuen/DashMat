from __future__ import annotations

import pandas as pd

from utils.route_intent import (
    ACTION_OPEN_IMPORT_MODAL,
    FLOW_DB,
    ROUTE_INTENT_MAX_AGE_SECONDS,
    build_route_intent,
    route_intent_token_to_consume,
)


def test_build_route_intent_includes_created_at():
    intent = build_route_intent("analyticstool", ACTION_OPEN_IMPORT_MODAL, flow=FLOW_DB)

    assert intent["created_at"]
    assert intent["token"]
    assert intent["flow"] == FLOW_DB


def test_route_intent_token_to_consume_accepts_fresh_intent():
    intent = build_route_intent("analyticstool", ACTION_OPEN_IMPORT_MODAL, flow=FLOW_DB)

    assert (
        route_intent_token_to_consume(
            intent,
            "analyticstool",
            ACTION_OPEN_IMPORT_MODAL,
            None,
            flow=FLOW_DB,
        )
        == intent["token"]
    )


def test_route_intent_token_to_consume_accepts_fresh_legacy_token_without_created_at():
    intent = build_route_intent("analyticstool", ACTION_OPEN_IMPORT_MODAL, flow=FLOW_DB)
    intent.pop("created_at")

    assert (
        route_intent_token_to_consume(
            intent,
            "analyticstool",
            ACTION_OPEN_IMPORT_MODAL,
            None,
            flow=FLOW_DB,
        )
        == intent["token"]
    )


def test_route_intent_token_to_consume_rejects_stale_created_at():
    intent = build_route_intent("analyticstool", ACTION_OPEN_IMPORT_MODAL, flow=FLOW_DB)
    intent["created_at"] = (
        pd.Timestamp.now(tz="UTC") - pd.Timedelta(seconds=ROUTE_INTENT_MAX_AGE_SECONDS + 1)
    ).isoformat()

    assert (
        route_intent_token_to_consume(
            intent,
            "analyticstool",
            ACTION_OPEN_IMPORT_MODAL,
            None,
            flow=FLOW_DB,
        )
        is None
    )


def test_route_intent_token_to_consume_rejects_stale_legacy_token_without_created_at():
    intent = build_route_intent("analyticstool", ACTION_OPEN_IMPORT_MODAL, flow=FLOW_DB)
    intent["token"] = (
        pd.Timestamp.now(tz="UTC") - pd.Timedelta(seconds=ROUTE_INTENT_MAX_AGE_SECONDS + 1)
    ).isoformat()
    intent.pop("created_at")

    assert (
        route_intent_token_to_consume(
            intent,
            "analyticstool",
            ACTION_OPEN_IMPORT_MODAL,
            None,
            flow=FLOW_DB,
        )
        is None
    )


def test_route_intent_token_to_consume_rejects_mismatch_and_invalid_cases():
    intent = build_route_intent("analyticstool", ACTION_OPEN_IMPORT_MODAL, flow=FLOW_DB)

    assert route_intent_token_to_consume(intent, "portopt", ACTION_OPEN_IMPORT_MODAL, None, flow=FLOW_DB) is None
    assert route_intent_token_to_consume(intent, "analyticstool", "configure_after_import", None, flow=FLOW_DB) is None
    assert route_intent_token_to_consume(intent, "analyticstool", ACTION_OPEN_IMPORT_MODAL, None, flow="raw") is None
    assert (
        route_intent_token_to_consume(
            intent,
            "analyticstool",
            ACTION_OPEN_IMPORT_MODAL,
            intent["token"],
            flow=FLOW_DB,
        )
        is None
    )

    malformed = dict(intent)
    malformed["created_at"] = "not-a-timestamp"
    assert (
        route_intent_token_to_consume(
            malformed,
            "analyticstool",
            ACTION_OPEN_IMPORT_MODAL,
            None,
            flow=FLOW_DB,
        )
        is None
    )
