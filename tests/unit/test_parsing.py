from __future__ import annotations

import base64
from io import BytesIO

import pandas as pd
import pytest

from utils.parsing import (
    convert_percents_to_decimals,
    detect_periodicity,
    get_sheet_names,
    parse_uploaded_file,
)


def _as_upload_payload(text: str) -> str:
    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
    return f"data:text/csv;base64,{encoded}"


def _as_upload_excel(df: pd.DataFrame) -> str:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Sheet1")
    encoded = base64.b64encode(bio.getvalue()).decode("ascii")
    return f"data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{encoded}"


def test_convert_percents_to_decimals_handles_percent_strings():
    df = pd.DataFrame({"A": ["5%", "2.5%", "-1%"], "B": [0.01, 0.02, 0.03]})
    converted = convert_percents_to_decimals(df)
    assert converted["A"].iloc[0] == pytest.approx(0.05)
    assert converted["A"].iloc[1] == pytest.approx(0.025)
    assert converted["A"].iloc[2] == pytest.approx(-0.01)
    assert converted["B"].iloc[1] == pytest.approx(0.02)


def test_detect_periodicity_distinguishes_daily_and_monthly():
    daily_idx = pd.date_range("2024-01-01", periods=5, freq="B")
    monthly_idx = pd.date_range("2024-01-31", periods=5, freq="ME")
    daily_df = pd.DataFrame({"x": range(5)}, index=daily_idx)
    monthly_df = pd.DataFrame({"x": range(5)}, index=monthly_idx)

    assert detect_periodicity(daily_df) == "daily"
    assert detect_periodicity(monthly_df) == "monthly"


def test_detect_periodicity_defaults_to_daily_for_single_row():
    df = pd.DataFrame({"x": [1]}, index=pd.to_datetime(["2024-01-01"]))
    assert detect_periodicity(df) == "daily"


def test_parse_uploaded_file_csv_sorts_dates_and_converts_percents():
    csv = "Date,A,B\n2024-01-03,1%,0.01\n2024-01-01,2%,0.02\n"
    payload = _as_upload_payload(csv)

    parsed = parse_uploaded_file(payload, "returns.csv")

    assert list(parsed.columns) == ["A", "B"]
    assert parsed.index[0].strftime("%Y-%m-%d") == "2024-01-01"
    assert parsed["A"].iloc[0] == pytest.approx(0.02)
    assert parsed["A"].iloc[1] == pytest.approx(0.01)


def test_parse_uploaded_file_raises_for_unsupported_extension():
    payload = _as_upload_payload("Date,A\n2024-01-01,0.01\n")
    with pytest.raises(ValueError):
        parse_uploaded_file(payload, "returns.txt")


def test_get_sheet_names_returns_empty_for_non_excel():
    payload = _as_upload_payload("Date,A\n2024-01-01,0.01\n")
    assert get_sheet_names(payload, "returns.csv") == []


def test_get_sheet_names_and_parse_uploaded_file_excel():
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-02", "2024-01-01"]),
            "A": ["1%", "2%"],
        }
    )
    payload = _as_upload_excel(df)

    sheets = get_sheet_names(payload, "returns.xlsx")
    parsed = parse_uploaded_file(payload, "returns.xlsx", sheet_name=0)

    assert sheets == ["Sheet1"]
    assert parsed.index[0].strftime("%Y-%m-%d") == "2024-01-01"
    assert parsed["A"].iloc[0] == pytest.approx(0.02)
