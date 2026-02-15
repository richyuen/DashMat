from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pandas as pd

from tools import smoke_upload_flow as smoke


def _build_multi_sheet_file(path: Path) -> None:
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
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        s1.to_excel(writer, sheet_name="S1", index=False)
        s2.to_excel(writer, sheet_name="S2", index=False)


def _tmp_workbook_path(name: str) -> Path:
    tmp_dir = Path("tests/.tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir / f"{name}_{uuid4().hex}.xlsx"


def test_run_smoke_selected_and_all_modes():
    workbook = _tmp_workbook_path("smoke")
    _build_multi_sheet_file(workbook)

    results = smoke.run_smoke(
        file_path=workbook,
        page="both",
        mode="both",
        selected_sheets="S2",
    )

    assert len(results) == 4
    assert all(r["periodicity"] == "daily" for r in results)
    selected_results = [r for r in results if r["mode"] == "selected"]
    all_results = [r for r in results if r["mode"] == "all"]
    assert all(r["rows"] == 2 for r in selected_results)
    assert all(r["rows"] == 3 for r in all_results)


def test_main_returns_nonzero_for_missing_file(monkeypatch, capsys):
    missing = _tmp_workbook_path("does-not-exist")
    monkeypatch.setattr(smoke, "run_smoke", lambda **_kwargs: [])
    monkeypatch.setattr(
        "sys.argv",
        [
            "smoke_upload_flow.py",
            "--file",
            str(missing),
        ],
    )

    exit_code = smoke.main()
    out = capsys.readouterr().out
    assert exit_code == 1
    assert "file does not exist" in out


def test_main_returns_nonzero_when_run_smoke_raises(monkeypatch, capsys):
    workbook = _tmp_workbook_path("smoke")
    _build_multi_sheet_file(workbook)

    def _boom(**_kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(smoke, "run_smoke", _boom)
    monkeypatch.setattr(
        "sys.argv",
        [
            "smoke_upload_flow.py",
            "--file",
            str(workbook),
        ],
    )

    exit_code = smoke.main()
    out = capsys.readouterr().out
    assert exit_code == 1
    assert "FAIL: boom" in out
