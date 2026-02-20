from __future__ import annotations

import pandas as pd

from utils.excel_export import format_excel_dates, format_mdy_date


def test_format_mdy_date_formats_timestamp_and_iso_string():
    assert format_mdy_date(pd.Timestamp("2021-06-30")) == "6/30/2021"
    assert format_mdy_date("2021-06-30 00:00:00") == "6/30/2021"


def test_format_excel_dates_formats_index_and_date_like_cells():
    df = pd.DataFrame(
        {
            "DateCol": pd.to_datetime(["2024-01-31", "2024-02-29"]),
            "DateText": ["2024-01-31", "2024-02-29 00:00:00"],
            "Other": ["alpha", "beta"],
        },
        index=pd.to_datetime(["2023-12-31", "2024-01-31"]),
    )
    df.index.name = "Date"

    out = format_excel_dates(df, format_index=True)

    assert list(out.index) == ["12/31/2023", "1/31/2024"]
    assert list(out["DateCol"]) == ["1/31/2024", "2/29/2024"]
    assert list(out["DateText"]) == ["1/31/2024", "2/29/2024"]
    assert list(out["Other"]) == ["alpha", "beta"]

