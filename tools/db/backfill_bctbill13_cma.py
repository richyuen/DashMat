"""Backfill BCTBill13 rows in CMA tables.

This script inserts missing BCTBill13 rows in:
- CMAReturns
- CMAStats
- CMACorrelation

It is idempotent and safe to run multiple times.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dbengine import engine


TARGET_BENCH = "BCTBill13"
PREFERRED_SOURCE_BENCHES = ("BCGC13", "BCAgg", "BCGAgg", "BCHY")
STAT_ITEMS = ("Mean", "SD", "Skewness", "Kurtosis")


def _get_version_type_pairs(conn) -> list[tuple[int, str]]:
    rows = conn.execute(
        text(
            """
            SELECT DISTINCT Version, Type
            FROM (
                SELECT Version, Type FROM CMAReturns
                UNION
                SELECT Version, Type FROM CMAStats
            )
            ORDER BY Version, Type
            """
        )
    ).fetchall()
    return [(int(r[0]), str(r[1])) for r in rows]


def _load_returns_pivot(conn, version: int, cma_type: str) -> pd.DataFrame:
    rows = conn.execute(
        text(
            """
            SELECT Bench, Date, Value
            FROM CMAReturns
            WHERE Version = :v AND Type = :t
            ORDER BY Date, Bench
            """
        ),
        {"v": version, "t": cma_type},
    ).fetchall()
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows, columns=["Bench", "Date", "Value"])
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Date"])
    pivot = frame.pivot(index="Date", columns="Bench", values="Value").sort_index()
    return pivot


def _choose_source_bench(columns: list[str]) -> str | None:
    for bench in PREFERRED_SOURCE_BENCHES:
        if bench in columns:
            return bench
    remaining = [c for c in columns if c != TARGET_BENCH]
    return remaining[0] if remaining else None


def _synthesize_target_returns(source_series: pd.Series) -> pd.Series:
    # Keep a low-volatility cash-like profile while preserving broad regime shape.
    clean = source_series.astype(float).copy()
    smooth = clean.rolling(window=3, min_periods=1).mean()
    target = (smooth * 0.25).clip(lower=-0.01, upper=0.01)
    return target.fillna(0.0)


def _insert_missing_returns(conn, version: int, cma_type: str, series: pd.Series) -> int:
    existing = conn.execute(
        text(
            """
            SELECT Date
            FROM CMAReturns
            WHERE Version = :v AND Type = :t AND Bench = :b
            """
        ),
        {"v": version, "t": cma_type, "b": TARGET_BENCH},
    ).fetchall()
    existing_dates = {pd.Timestamp(r[0]).date() for r in existing}

    rows = []
    for dt, value in series.items():
        dt_date = pd.Timestamp(dt).date()
        if dt_date in existing_dates:
            continue
        rows.append(
            {
                "Version": version,
                "Type": cma_type,
                "Bench": TARGET_BENCH,
                "Date": dt_date,
                "Value": float(value),
            }
        )

    if not rows:
        return 0

    conn.execute(
        text(
            """
            INSERT INTO CMAReturns (Version, Type, Bench, Date, Value)
            VALUES (:Version, :Type, :Bench, :Date, :Value)
            """
        ),
        rows,
    )
    return len(rows)


def _upsert_stats(conn, version: int, cma_type: str, series: pd.Series) -> tuple[int, int]:
    clean = series.dropna().astype(float)
    if clean.empty:
        return 0, 0

    values = {
        "Mean": float(clean.mean()),
        "SD": float(clean.std(ddof=1)),
        "Skewness": float(clean.skew()),
        "Kurtosis": float(clean.kurt()),
    }

    inserted = 0
    updated = 0
    for item in STAT_ITEMS:
        payload = {
            "v": version,
            "t": cma_type,
            "b": TARGET_BENCH,
            "item": item,
            "value": values[item],
        }
        res = conn.execute(
            text(
                """
                UPDATE CMAStats
                SET Value = :value
                WHERE Version = :v AND Type = :t AND Bench = :b AND Item = :item
                """
            ),
            payload,
        )
        if int(res.rowcount or 0) > 0:
            updated += 1
            continue
        conn.execute(
            text(
                """
                INSERT INTO CMAStats (Version, Type, Bench, Item, Value)
                VALUES (:v, :t, :b, :item, :value)
                """
            ),
            payload,
        )
        inserted += 1
    return inserted, updated


def _upsert_correlation(conn, version: int, cma_type: str, bench_other: str, value: float) -> tuple[int, int]:
    value = float(value)
    if bench_other == TARGET_BENCH:
        value = 1.0
    if not np.isfinite(value):
        value = 0.0

    base = {"v": version, "t": cma_type, "b1": TARGET_BENCH, "b2": bench_other, "value": value}
    swapped = {"v": version, "t": cma_type, "b1": bench_other, "b2": TARGET_BENCH, "value": value}

    res = conn.execute(
        text(
            """
            UPDATE CMACorrelation
            SET Value = :value
            WHERE Version = :v AND Type = :t AND Bench1 = :b1 AND Bench2 = :b2
            """
        ),
        base,
    )
    if int(res.rowcount or 0) > 0:
        return 0, 1

    res_swapped = conn.execute(
        text(
            """
            UPDATE CMACorrelation
            SET Value = :value
            WHERE Version = :v AND Type = :t AND Bench1 = :b1 AND Bench2 = :b2
            """
        ),
        swapped,
    )
    if int(res_swapped.rowcount or 0) > 0:
        return 0, 1

    conn.execute(
        text(
            """
            INSERT INTO CMACorrelation (Version, Type, Bench1, Bench2, Value)
            VALUES (:v, :t, :b1, :b2, :value)
            """
        ),
        base,
    )
    return 1, 0


def main() -> None:
    inserted_returns = 0
    inserted_stats = 0
    updated_stats = 0
    inserted_corr = 0
    updated_corr = 0

    with engine.begin() as conn:
        combos = _get_version_type_pairs(conn)
        if not combos:
            raise RuntimeError("No Version/Type pairs found in CMA tables.")

        for version, cma_type in combos:
            pivot = _load_returns_pivot(conn, version, cma_type)
            if pivot.empty:
                continue

            source_bench = _choose_source_bench(list(pivot.columns))
            if source_bench is None:
                continue

            if TARGET_BENCH in pivot.columns:
                target_series = pivot[TARGET_BENCH].astype(float).dropna()
            else:
                target_series = pd.Series(dtype=float)

            if target_series.empty:
                target_series = _synthesize_target_returns(pivot[source_bench])

            inserted_returns += _insert_missing_returns(conn, version, cma_type, target_series)

            # Reload including newly inserted rows for consistent stats/correlations.
            refreshed = _load_returns_pivot(conn, version, cma_type)
            if refreshed.empty or TARGET_BENCH not in refreshed.columns:
                continue

            stats_insert, stats_update = _upsert_stats(conn, version, cma_type, refreshed[TARGET_BENCH])
            inserted_stats += stats_insert
            updated_stats += stats_update

            corr = refreshed.corr()
            for other in corr.columns:
                value = corr.loc[TARGET_BENCH, other] if TARGET_BENCH in corr.index else np.nan
                corr_insert, corr_update = _upsert_correlation(conn, version, cma_type, str(other), value)
                inserted_corr += corr_insert
                updated_corr += corr_update

    print(f"Inserted CMAReturns rows for {TARGET_BENCH}: {inserted_returns}")
    print(f"Inserted CMAStats rows for {TARGET_BENCH}: {inserted_stats}")
    print(f"Updated CMAStats rows for {TARGET_BENCH}: {updated_stats}")
    print(f"Inserted CMACorrelation rows for {TARGET_BENCH}: {inserted_corr}")
    print(f"Updated CMACorrelation rows for {TARGET_BENCH}: {updated_corr}")


if __name__ == "__main__":
    main()
