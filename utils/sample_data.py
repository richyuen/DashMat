"""Sample benchmark data helpers for DashMat."""

from pathlib import Path


SAMPLE_DATA_DIR = Path(__file__).resolve().parent.parent / "sample_data" / "benchmark_returns"
SAMPLE_DAILY_FILE = "benchmark_daily_returns_2020_2025.xlsx"
SAMPLE_MONTHLY_FILE = "benchmark_monthly_returns_2020_2025.xlsx"


def get_sample_file_path(periodicity: str) -> Path:
    """Return absolute path to stored sample file for a periodicity."""
    if periodicity == "daily":
        return SAMPLE_DATA_DIR / SAMPLE_DAILY_FILE
    if periodicity == "monthly":
        return SAMPLE_DATA_DIR / SAMPLE_MONTHLY_FILE
    raise ValueError(f"Unsupported periodicity: {periodicity}")

