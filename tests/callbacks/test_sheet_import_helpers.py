from __future__ import annotations

import pandas as pd
import pytest

from utils import upload_flow


def test_shared_import_selected_sheets_overwrites_with_later_sheet(monkeypatch):
    idx_1 = pd.to_datetime(["2024-01-01", "2024-01-02"])
    idx_2 = pd.to_datetime(["2024-01-02", "2024-01-03"])
    frames = {
        "S1": pd.DataFrame({"SeriesA": [0.01, 0.02]}, index=idx_1),
        "S2": pd.DataFrame({"SeriesA": [0.03, 0.04]}, index=idx_2),
        "S3": pd.DataFrame({"SeriesB": [0.05]}, index=pd.to_datetime(["2024-01-04"])),
    }
    for frame in frames.values():
        frame.index.name = "Date"

    monkeypatch.setattr(upload_flow, "get_sheet_names", lambda *_args, **_kwargs: ["S1", "S2", "S3"])
    monkeypatch.setattr(
        upload_flow,
        "parse_uploaded_sheets",
        lambda _contents, _filename, ordered_sheets, ignore_errors=False: {  # noqa: ARG005
            sheet: frames[sheet].copy() for sheet in ordered_sheets
        },
    )

    combined, imported = upload_flow.import_selected_workbook_sheets("contents", "book.xlsx", ["S2", "S1"])

    assert imported == ["S1", "S2"]
    assert combined.loc[pd.Timestamp("2024-01-02"), "SeriesA"] == pytest.approx(0.03)
    assert combined.loc[pd.Timestamp("2024-01-01"), "SeriesA"] == pytest.approx(0.01)
    assert combined.loc[pd.Timestamp("2024-01-03"), "SeriesA"] == pytest.approx(0.04)


def test_shared_import_selected_sheets_accepts_string_selection(monkeypatch):
    idx = pd.to_datetime(["2024-03-01"])
    frame = pd.DataFrame({"Series": [0.01]}, index=idx)
    frame.index.name = "Date"

    monkeypatch.setattr(upload_flow, "get_sheet_names", lambda *_args, **_kwargs: ["S1"])
    monkeypatch.setattr(
        upload_flow,
        "parse_uploaded_sheets",
        lambda *_args, **_kwargs: {"S1": frame.copy()},
    )

    combined, imported = upload_flow.import_selected_workbook_sheets("contents", "book.xlsx", "S1")

    assert imported == ["S1"]
    assert combined.shape == (1, 1)


def test_page_sheet_helpers_delegate_to_shared_helper(monkeypatch, page_modules):
    analyticstool, portopt = page_modules
    idx = pd.to_datetime(["2024-04-01"])
    frame = pd.DataFrame({"Series": [0.02]}, index=idx)
    frame.index.name = "Date"

    monkeypatch.setattr(
        analyticstool,
        "_shared_import_selected_workbook_sheets",
        lambda *_args, **_kwargs: (frame.copy(), ["S1"]),
    )
    monkeypatch.setattr(
        portopt,
        "_shared_import_selected_workbook_sheets",
        lambda *_args, **_kwargs: (frame.copy(), ["S1"]),
    )

    at_df, at_sheets = analyticstool._import_selected_workbook_sheets("contents", "book.xlsx", ["S1"])
    po_df, po_sheets = portopt._po_import_selected_workbook_sheets("contents", "book.xlsx", ["S1"])

    assert at_sheets == ["S1"]
    assert po_sheets == ["S1"]
    assert at_df.equals(frame)
    assert po_df.equals(frame)


def test_import_selected_disabled_toggles_with_selected_values(page_modules):
    analyticstool, portopt = page_modules

    assert analyticstool.toggle_sheet_select_import_selected_disabled([]) is True
    assert analyticstool.toggle_sheet_select_import_selected_disabled(["Sheet1"]) is False
    assert portopt.po_toggle_sheet_select_import_selected_disabled([]) is True
    assert portopt.po_toggle_sheet_select_import_selected_disabled(["Sheet1"]) is False
