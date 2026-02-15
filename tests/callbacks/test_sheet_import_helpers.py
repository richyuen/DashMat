from __future__ import annotations

import pandas as pd
import pytest


def test_analyticstool_import_selected_sheets_overwrites_with_later_sheet(monkeypatch, page_modules):
    analyticstool, _portopt = page_modules

    idx_1 = pd.to_datetime(["2024-01-01", "2024-01-02"])
    idx_2 = pd.to_datetime(["2024-01-02", "2024-01-03"])
    frames = {
        "S1": pd.DataFrame({"SeriesA": [0.01, 0.02]}, index=idx_1),
        "S2": pd.DataFrame({"SeriesA": [0.03, 0.04]}, index=idx_2),
        "S3": pd.DataFrame({"SeriesB": [0.05]}, index=pd.to_datetime(["2024-01-04"])),
    }
    for frame in frames.values():
        frame.index.name = "Date"

    monkeypatch.setattr(analyticstool, "get_sheet_names", lambda *_args, **_kwargs: ["S1", "S2", "S3"])
    monkeypatch.setattr(
        analyticstool,
        "parse_uploaded_file",
        lambda _contents, _filename, sheet_name=0: frames[sheet_name].copy(),
    )

    combined, imported = analyticstool._import_selected_workbook_sheets(
        "contents", "book.xlsx", ["S2", "S1"]
    )

    assert imported == ["S1", "S2"]
    assert combined.loc[pd.Timestamp("2024-01-02"), "SeriesA"] == pytest.approx(0.03)
    assert combined.loc[pd.Timestamp("2024-01-01"), "SeriesA"] == pytest.approx(0.01)
    assert combined.loc[pd.Timestamp("2024-01-03"), "SeriesA"] == pytest.approx(0.04)


def test_portopt_import_selected_sheets_overwrites_with_later_sheet(monkeypatch, page_modules):
    _analyticstool, portopt = page_modules

    idx_1 = pd.to_datetime(["2024-02-01", "2024-02-02"])
    idx_2 = pd.to_datetime(["2024-02-02", "2024-02-03"])
    frames = {
        "A": pd.DataFrame({"SeriesX": [0.10, 0.20]}, index=idx_1),
        "B": pd.DataFrame({"SeriesX": [0.30, 0.40]}, index=idx_2),
    }
    for frame in frames.values():
        frame.index.name = "Date"

    monkeypatch.setattr(portopt, "get_sheet_names", lambda *_args, **_kwargs: ["A", "B"])
    monkeypatch.setattr(
        portopt,
        "parse_uploaded_file",
        lambda _contents, _filename, sheet_name=0: frames[sheet_name].copy(),
    )

    combined, imported = portopt._po_import_selected_workbook_sheets(
        "contents", "book.xlsx", ["B", "A"]
    )

    assert imported == ["A", "B"]
    assert combined.loc[pd.Timestamp("2024-02-02"), "SeriesX"] == pytest.approx(0.30)
    assert combined.loc[pd.Timestamp("2024-02-01"), "SeriesX"] == pytest.approx(0.10)
    assert combined.loc[pd.Timestamp("2024-02-03"), "SeriesX"] == pytest.approx(0.40)


def test_import_selected_disabled_toggles_with_selected_values(page_modules):
    analyticstool, portopt = page_modules

    assert analyticstool.toggle_sheet_select_import_selected_disabled([]) is True
    assert analyticstool.toggle_sheet_select_import_selected_disabled(["Sheet1"]) is False
    assert portopt.po_toggle_sheet_select_import_selected_disabled([]) is True
    assert portopt.po_toggle_sheet_select_import_selected_disabled(["Sheet1"]) is False


def test_sheet_helpers_accept_string_selection(monkeypatch, page_modules):
    analyticstool, portopt = page_modules

    idx = pd.to_datetime(["2024-03-01"])
    frame = pd.DataFrame({"Series": [0.01]}, index=idx)
    frame.index.name = "Date"

    monkeypatch.setattr(analyticstool, "get_sheet_names", lambda *_args, **_kwargs: ["S1"])
    monkeypatch.setattr(
        analyticstool,
        "parse_uploaded_file",
        lambda *_args, **_kwargs: frame.copy(),
    )
    a_df, a_sheets = analyticstool._import_selected_workbook_sheets("contents", "book.xlsx", "S1")
    assert a_sheets == ["S1"]
    assert a_df.shape == (1, 1)

    monkeypatch.setattr(portopt, "get_sheet_names", lambda *_args, **_kwargs: ["S1"])
    monkeypatch.setattr(
        portopt,
        "parse_uploaded_file",
        lambda *_args, **_kwargs: frame.copy(),
    )
    p_df, p_sheets = portopt._po_import_selected_workbook_sheets("contents", "book.xlsx", "S1")
    assert p_sheets == ["S1"]
    assert p_df.shape == (1, 1)
