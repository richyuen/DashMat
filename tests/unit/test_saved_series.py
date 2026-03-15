from __future__ import annotations

import pandas as pd
import pytest

from utils.returns import df_to_json, json_to_df
from utils.saved_series import normalize_saved_series_store, save_series_to_raw_data


def test_normalize_saved_series_store_supports_legacy_list_payload():
    normalized = normalize_saved_series_store(["P1", "P2"])

    assert set(normalized) == {"P1", "P2"}
    assert normalized["P1"]["origin_page"] == "portopt"
    assert normalized["P1"]["series_type"] == "portfolio"


def test_save_series_to_raw_data_uses_base_name_on_first_save():
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    raw_df = pd.DataFrame({"Asset_A": [0.01, 0.02, 0.03]}, index=idx)
    series = pd.Series([0.001, 0.002, 0.003], index=idx, name="P1")

    out = save_series_to_raw_data(
        raw_data=df_to_json(raw_df),
        periodicity="daily",
        series=series,
        base_name="P1",
        saved_series_store={},
        origin_page="portopt",
        origin_result="P1",
        series_type="portfolio",
    )

    merged = json_to_df(out["raw_data"])
    assert out["saved_name"] == "P1"
    assert out["action"] == "saved"
    assert "P1" in merged.columns
    assert out["saved_series_store"]["P1"]["origin_result"] == "P1"


def test_save_series_to_raw_data_suffixes_on_first_collision():
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    raw_df = pd.DataFrame({"P1": [0.01, 0.02, 0.03]}, index=idx)
    series = pd.Series([0.001, 0.002, 0.003], index=idx, name="P1")

    out = save_series_to_raw_data(
        raw_data=df_to_json(raw_df),
        periodicity="daily",
        series=series,
        base_name="P1",
        saved_series_store={},
        origin_page="regression",
        origin_result="P1",
        series_type="predicted",
    )

    merged = json_to_df(out["raw_data"])
    assert out["saved_name"] == "P1_1"
    assert out["action"] == "saved"
    assert "P1_1" in merged.columns


def test_save_series_to_raw_data_overwrites_prior_saved_name_idempotently():
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    raw_df = pd.DataFrame({"P1": [0.01, 0.02, 0.03]}, index=idx)
    series = pd.Series([0.1, 0.2, 0.3], index=idx, name="P1")

    out = save_series_to_raw_data(
        raw_data=df_to_json(raw_df),
        periodicity="daily",
        series=series,
        base_name="P1",
        saved_series_store={"P1": {"origin_page": "portopt"}},
        origin_page="portopt",
        origin_result="P1",
        series_type="portfolio",
        prior_saved_name="P1",
    )

    merged = json_to_df(out["raw_data"])
    assert out["saved_name"] == "P1"
    assert out["action"] == "overwritten"
    assert merged["P1"].tolist() == pytest.approx([0.1, 0.2, 0.3])


def test_save_series_to_raw_data_aligns_monthly_series_to_month_end():
    raw_idx = pd.to_datetime(["1976-06-30", "1976-07-30", "1976-08-30", "1976-09-30"])
    raw_df = pd.DataFrame({"Asset_A": [0.01, 0.02, 0.03, 0.04]}, index=raw_idx)
    raw_df.index.name = "Date"
    series = pd.Series(
        [0.005, 0.006, 0.007, 0.008],
        index=pd.to_datetime(["1976-06-30", "1976-07-31", "1976-08-31", "1976-09-30"]),
        name="MonthlyPort",
    )

    out = save_series_to_raw_data(
        raw_data=df_to_json(raw_df),
        periodicity="monthly",
        series=series,
        base_name="MonthlyPort",
        saved_series_store={},
        origin_page="portopt",
        origin_result="MonthlyPort",
        series_type="portfolio",
    )

    merged = json_to_df(out["raw_data"])
    assert pd.Timestamp("1976-07-31") in merged.index
    assert pd.Timestamp("1976-07-30") not in merged.index
    assert merged.index.is_month_end.all()
    assert merged.loc[pd.Timestamp("1976-08-31"), "MonthlyPort"] == 0.007
