"""Callback-level smoke checks for upload + sheet import flows.

Run examples:
  conda run -n dashmat python tools/smoke_upload_flow.py --file C:\Git\SampleMstar.xlsx
  conda run -n dashmat python tools/smoke_upload_flow.py --file C:\Git\SampleMstarMulti.xlsx --mode all
  conda run -n dashmat python tools/smoke_upload_flow.py --file C:\Git\SampleMstarMulti.xlsx --mode selected --selected-sheets FullCatchup9,FullCatchupYTD
"""

from __future__ import annotations

import argparse
import base64
from pathlib import Path
import sys
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.parsing import detect_periodicity, get_sheet_names, parse_uploaded_file


def _build_upload_payload(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        mime = "text/csv"
    elif suffix in {".xlsx", ".xls"}:
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    else:
        mime = "application/octet-stream"
    encoded = base64.b64encode(file_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _parse_selected_sheets(selected_sheets: str | None) -> list[str]:
    if not selected_sheets:
        return []
    return [part.strip() for part in selected_sheets.split(",") if part.strip()]


def _get_page_module(page_name: str):
    # Page modules call register_page at import time, so import app first.
    import app  # noqa: F401

    if page_name == "analyticstool":
        import pages.analyticstool as analyticstool
        return analyticstool
    if page_name == "portopt":
        import pages.portopt as portopt
        return portopt
    raise ValueError(f"Unsupported page: {page_name}")


def _import_for_page(
    page_name: str,
    payload: str,
    filename: str,
    mode: str,
    selected_sheets: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    sheet_names = get_sheet_names(payload, filename)
    page_mod = _get_page_module(page_name)

    if len(sheet_names) <= 1:
        df = parse_uploaded_file(payload, filename)
        return df, sheet_names[:1]

    if page_name == "analyticstool":
        helper = page_mod._import_selected_workbook_sheets
    else:
        helper = page_mod._po_import_selected_workbook_sheets

    if mode == "all":
        targets = sheet_names
        return helper(payload, filename, targets)

    if selected_sheets:
        return helper(payload, filename, selected_sheets)

    # Default behavior mirrors the UI's first-sheet default, but for smoke
    # automation we fall back to the first importable sheet if needed.
    first_sheet = sheet_names[:1]
    try:
        return helper(payload, filename, first_sheet)
    except Exception:
        for sheet in sheet_names:
            try:
                return helper(payload, filename, [sheet])
            except Exception:
                continue
        raise ValueError("No importable sheets found for selected-mode smoke check.")


def run_smoke(
    file_path: Path,
    page: str = "both",
    mode: str = "both",
    selected_sheets: str | None = None,
    allow_empty: bool = False,
) -> list[dict[str, Any]]:
    payload = _build_upload_payload(file_path)
    filename = file_path.name
    selected = _parse_selected_sheets(selected_sheets)

    pages = ["analyticstool", "portopt"] if page == "both" else [page]
    modes = ["selected", "all"] if mode == "both" else [mode]
    results: list[dict[str, Any]] = []

    for page_name in pages:
        for mode_name in modes:
            df, imported_sheets = _import_for_page(page_name, payload, filename, mode_name, selected)
            periodicity = detect_periodicity(df)

            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError(f"{page_name}/{mode_name}: imported index is not DatetimeIndex")
            if not allow_empty and df.empty:
                raise ValueError(f"{page_name}/{mode_name}: imported dataframe is empty")
            if periodicity not in {"daily", "monthly"}:
                raise ValueError(f"{page_name}/{mode_name}: unexpected periodicity: {periodicity}")

            results.append(
                {
                    "page": page_name,
                    "mode": mode_name,
                    "rows": len(df),
                    "cols": len(df.columns),
                    "periodicity": periodicity,
                    "sheets": imported_sheets,
                }
            )

    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-check upload + sheet import flows.")
    parser.add_argument("--file", required=True, help="Path to CSV/XLS/XLSX input file.")
    parser.add_argument(
        "--page",
        choices=["analyticstool", "portopt", "both"],
        default="both",
        help="Target page flow to check.",
    )
    parser.add_argument(
        "--mode",
        choices=["selected", "all", "both"],
        default="both",
        help="Sheet import mode to check.",
    )
    parser.add_argument(
        "--selected-sheets",
        default=None,
        help="Comma-separated sheet names for selected-mode checks.",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Allow empty imported dataframes without failing the smoke check.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"FAIL: file does not exist: {file_path}")
        return 1

    try:
        results = run_smoke(
            file_path=file_path,
            page=args.page,
            mode=args.mode,
            selected_sheets=args.selected_sheets,
            allow_empty=args.allow_empty,
        )
    except Exception as exc:
        print(f"FAIL: {exc}")
        return 1

    print("PASS: upload smoke checks")
    for result in results:
        sheet_desc = ",".join(result["sheets"]) if result["sheets"] else "-"
        print(
            f"- page={result['page']} mode={result['mode']} periodicity={result['periodicity']} "
            f"rows={result['rows']} cols={result['cols']} sheets={sheet_desc}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
