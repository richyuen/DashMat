from __future__ import annotations

import base64
from io import BytesIO, StringIO
from types import SimpleNamespace

import pandas as pd


def _as_multi_sheet_upload_payload() -> tuple[str, str]:
    s1 = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "SeriesA": ["1%", "2%"],
        }
    )
    s2 = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "SeriesA": ["3%", "4%"],
        }
    )
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        s1.to_excel(writer, sheet_name="S1", index=False)
        s2.to_excel(writer, sheet_name="S2", index=False)
    encoded = base64.b64encode(bio.getvalue()).decode("ascii")
    return (
        f"data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{encoded}",
        "multi.xlsx",
    )


def test_analyticstool_sheet_callback_smoke_selected_and_all(monkeypatch, page_modules):
    analyticstool, _portopt = page_modules
    payload, filename = _as_multi_sheet_upload_payload()

    monkeypatch.setattr(
        analyticstool,
        "callback_context",
        SimpleNamespace(triggered_id="at-sheet-select-ok-button"),
    )
    selected_result = analyticstool.on_sheet_select_ok(
        1,
        None,
        ["S2"],
        payload,
        filename,
        ["S1", "S2"],
        None,
        None,
        [],
        {},
        {},
        [],
        False,
        {},
    )
    selected_pending = selected_result[23]
    selected_df = pd.read_json(StringIO(selected_pending["raw_data"]), orient="split")
    selected_df.index = pd.to_datetime(selected_df.index)
    assert selected_result[10]
    assert selected_pending["original_periodicity"] == "daily"
    assert selected_pending["commit_alert"]["color"] == "green"
    assert selected_result[18] is False
    assert selected_df.shape == (2, 1)

    monkeypatch.setattr(
        analyticstool,
        "callback_context",
        SimpleNamespace(triggered_id="at-sheet-select-import-all-button"),
    )
    all_result = analyticstool.on_sheet_select_ok(
        1,
        1,
        ["S2"],
        payload,
        filename,
        ["S1", "S2"],
        None,
        None,
        [],
        {},
        {},
        [],
        False,
        {},
    )
    all_pending = all_result[23]
    all_df = pd.read_json(StringIO(all_pending["raw_data"]), orient="split")
    all_df.index = pd.to_datetime(all_df.index)
    assert all_pending["original_periodicity"] == "daily"
    assert all_df.shape == (3, 1)
    assert all_df.loc[pd.Timestamp("2024-01-02"), "SeriesA"] == 0.03


def test_portopt_sheet_callback_smoke_selected_and_all(monkeypatch, page_modules):
    _analyticstool, portopt = page_modules
    payload, filename = _as_multi_sheet_upload_payload()

    monkeypatch.setattr(
        portopt,
        "callback_context",
        SimpleNamespace(triggered_id="po-sheet-select-ok-button"),
    )
    selected_result = portopt.po_on_sheet_select_ok(
        1,
        None,
        ["S2"],
        payload,
        filename,
        ["S1", "S2"],
        None,
        None,
        [],
        {},
        {},
        {},
        [],
        {},
        {},
        {},
        {},
    )
    selected_df = pd.read_json(StringIO(selected_result[0]), orient="split")
    selected_df.index = pd.to_datetime(selected_df.index)
    assert selected_result[1] == "daily"
    assert selected_result[7] == "green"
    assert selected_df.shape == (2, 1)

    monkeypatch.setattr(
        portopt,
        "callback_context",
        SimpleNamespace(triggered_id="po-sheet-select-import-all-button"),
    )
    all_result = portopt.po_on_sheet_select_ok(
        1,
        1,
        ["S2"],
        payload,
        filename,
        ["S1", "S2"],
        None,
        None,
        [],
        {},
        {},
        {},
        [],
        {},
        {},
        {},
        {},
    )
    all_df = pd.read_json(StringIO(all_result[0]), orient="split")
    all_df.index = pd.to_datetime(all_df.index)
    assert all_result[1] == "daily"
    assert all_df.shape == (3, 1)
    assert all_df.loc[pd.Timestamp("2024-01-02"), "SeriesA"] == 0.03


def test_sheet_callbacks_return_validation_error_when_selected_empty(monkeypatch, page_modules):
    analyticstool, portopt = page_modules
    payload, filename = _as_multi_sheet_upload_payload()

    monkeypatch.setattr(
        analyticstool,
        "callback_context",
        SimpleNamespace(triggered_id="at-sheet-select-ok-button"),
    )
    at_result = analyticstool.on_sheet_select_ok(
        1,
        None,
        [],
        payload,
        filename,
        ["S1", "S2"],
        None,
        None,
        [],
        {},
        {},
        [],
        False,
        {},
    )
    assert at_result[6] == "Select at least one sheet to import."
    assert at_result[7] == "red"
    assert at_result[18] is True

    monkeypatch.setattr(
        portopt,
        "callback_context",
        SimpleNamespace(triggered_id="po-sheet-select-ok-button"),
    )
    po_result = portopt.po_on_sheet_select_ok(
        1,
        None,
        [],
        payload,
        filename,
        ["S1", "S2"],
        None,
        None,
        [],
        {},
        {},
        {},
        [],
        {},
        {},
        {},
        {},
    )
    assert po_result[6] == "Select at least one sheet to import."
    assert po_result[7] == "red"
    assert po_result[20] is True
