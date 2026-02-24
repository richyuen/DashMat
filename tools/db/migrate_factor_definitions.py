"""Create and seed factor definition tables."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dbengine import DATABASE_URL, MRD_DATABASE_URL, engine, engine_MRD  # noqa: E402
from utils.factor_definitions import validate_factor_definition_payload  # noqa: E402


@dataclass(frozen=True)
class _SampleFactorSpec:
    name: str
    description: str
    long_agg_type: int
    short_agg_type: int | None
    long_lag: int
    output_transform: int
    long_count: int
    short_count: int


SAMPLE_FACTOR_SPECS: tuple[_SampleFactorSpec, ...] = (
    _SampleFactorSpec(
        name="SAMPLE_Carry_CompoundSpread",
        description="Long-short carry spread using compounded period returns.",
        long_agg_type=1,
        short_agg_type=1,
        long_lag=0,
        output_transform=0,
        long_count=1,
        short_count=1,
    ),
    _SampleFactorSpec(
        name="SAMPLE_LastValue_LongOnly",
        description="Long-only factor using period-end levels.",
        long_agg_type=2,
        short_agg_type=None,
        long_lag=0,
        output_transform=0,
        long_count=1,
        short_count=0,
    ),
    _SampleFactorSpec(
        name="SAMPLE_MeanSpread_PctChange",
        description="Mean-minus-mean spread with percent-change output transform.",
        long_agg_type=3,
        short_agg_type=3,
        long_lag=0,
        output_transform=1,
        long_count=2,
        short_count=2,
    ),
    _SampleFactorSpec(
        name="SAMPLE_AnnualizedVol_LongOnly",
        description="Annualized volatility from daily returns (long-only).",
        long_agg_type=4,
        short_agg_type=None,
        long_lag=0,
        output_transform=0,
        long_count=1,
        short_count=0,
    ),
    _SampleFactorSpec(
        name="SAMPLE_AlreadyPeriodic_Spread",
        description="Monthly already-periodic spread mapped to selected periodicity.",
        long_agg_type=5,
        short_agg_type=5,
        long_lag=0,
        output_transform=0,
        long_count=1,
        short_count=1,
    ),
    _SampleFactorSpec(
        name="SAMPLE_QuarterlyInterp_LongOnly",
        description="Quarterly series interpolated to selected periodicity (long-only).",
        long_agg_type=6,
        short_agg_type=None,
        long_lag=0,
        output_transform=0,
        long_count=1,
        short_count=0,
    ),
    _SampleFactorSpec(
        name="SAMPLE_ReturnFromLevels_Diff",
        description="Spread from return-from-level transformations with simple differences.",
        long_agg_type=7,
        short_agg_type=7,
        long_lag=0,
        output_transform=2,
        long_count=1,
        short_count=1,
    ),
    _SampleFactorSpec(
        name="SAMPLE_LaggedCarry_1",
        description="Lagged long leg by one period before long-short combination.",
        long_agg_type=2,
        short_agg_type=2,
        long_lag=1,
        output_transform=0,
        long_count=1,
        short_count=1,
    ),
    _SampleFactorSpec(
        name="SAMPLE_LaggedMean_3",
        description="Lagged long leg by three periods using period means.",
        long_agg_type=3,
        short_agg_type=3,
        long_lag=3,
        output_transform=0,
        long_count=1,
        short_count=1,
    ),
    _SampleFactorSpec(
        name="SAMPLE_Mix_CompoundMinusLevels",
        description="Compound-return long leg minus return-from-levels short leg.",
        long_agg_type=1,
        short_agg_type=7,
        long_lag=0,
        output_transform=0,
        long_count=2,
        short_count=1,
    ),
    _SampleFactorSpec(
        name="SAMPLE_Mix_MeanMinusQuarterly",
        description="Period-mean long leg minus quarterly-interpolated short leg.",
        long_agg_type=3,
        short_agg_type=6,
        long_lag=0,
        output_transform=0,
        long_count=1,
        short_count=1,
    ),
    _SampleFactorSpec(
        name="SAMPLE_LastValue_PctChange",
        description="Long-only level factor with output percent-change transform.",
        long_agg_type=2,
        short_agg_type=None,
        long_lag=0,
        output_transform=1,
        long_count=1,
        short_count=0,
    ),
)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)


def _mrd_account_table(db_engine: Engine) -> str:
    if db_engine.dialect.name == "sqlite":
        return "[CORE_DATA.ACCOUNT]"
    return "[CORE_DATA].[ACCOUNT]"


def _normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    text_val = str(value).strip()
    return text_val or None


def _normalize_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _create_tables_if_missing(db_engine: Engine) -> dict[str, Any]:
    inspector = inspect(db_engine)
    create_live = not inspector.has_table("FactorDefinitions")
    create_archive = not inspector.has_table("FactorDefinitionsArchive")

    with db_engine.begin() as conn:
        if create_live:
            conn.execute(
                text(
                    """
                    CREATE TABLE FactorDefinitions (
                        FactorName VARCHAR(128) NOT NULL PRIMARY KEY,
                        LongComponent VARCHAR(4096) NOT NULL,
                        ShortComponent VARCHAR(4096) NULL,
                        Description VARCHAR(4096) NULL,
                        LongAggType INTEGER NOT NULL,
                        ShortAggType INTEGER NULL,
                        LongLag INTEGER NOT NULL,
                        OutputTransform INTEGER NOT NULL,
                        UPDATE_DATE DATETIME NOT NULL,
                        UPDATE_BY VARCHAR(128) NOT NULL
                    )
                    """
                )
            )

        if create_archive:
            conn.execute(
                text(
                    """
                    CREATE TABLE FactorDefinitionsArchive (
                        FactorName VARCHAR(128) NOT NULL,
                        LongComponent VARCHAR(4096) NOT NULL,
                        ShortComponent VARCHAR(4096) NULL,
                        Description VARCHAR(4096) NULL,
                        LongAggType INTEGER NOT NULL,
                        ShortAggType INTEGER NULL,
                        LongLag INTEGER NOT NULL,
                        OutputTransform INTEGER NOT NULL,
                        UPDATE_DATE DATETIME NOT NULL,
                        UPDATE_BY VARCHAR(128) NOT NULL,
                        ARCHIVE_DATE DATETIME NOT NULL
                    )
                    """
                )
            )

    inspector = inspect(db_engine)
    with db_engine.begin() as conn:
        if db_engine.dialect.name == "sqlite":
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_factor_defs_name "
                    "ON FactorDefinitions (FactorName)"
                )
            )
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_factor_defs_archive_name_date "
                    "ON FactorDefinitionsArchive (FactorName, ARCHIVE_DATE)"
                )
            )
        else:
            existing_fd_indexes = {idx.get("name") for idx in inspector.get_indexes("FactorDefinitions")}
            if "idx_factor_defs_name" not in existing_fd_indexes:
                conn.execute(text("CREATE INDEX idx_factor_defs_name ON FactorDefinitions (FactorName)"))

            existing_archive_indexes = {idx.get("name") for idx in inspector.get_indexes("FactorDefinitionsArchive")}
            if "idx_factor_defs_archive_name_date" not in existing_archive_indexes:
                conn.execute(
                    text(
                        "CREATE INDEX idx_factor_defs_archive_name_date "
                        "ON FactorDefinitionsArchive (FactorName, ARCHIVE_DATE)"
                    )
                )

    return {
        "created_factor_definitions": create_live,
        "created_factor_definitions_archive": create_archive,
    }


def _load_sec_factor_tokens(mrd_db_engine: Engine) -> list[str]:
    account_table = _mrd_account_table(mrd_db_engine)
    try:
        q = text(
            f"SELECT ACCT_NAME, FACTOR_NAME, ACCT_ID, SOURCE_SYSTEM "
            f"FROM {account_table} "
            "WHERE ACCT_TYPE_CD = 'SEC_FACTOR' "
            "AND COALESCE(SOURCE_SYSTEM, '') <> 'PERF' "
            "ORDER BY ACCT_NAME, FACTOR_NAME, ACCT_ID"
        )
        with mrd_db_engine.connect() as conn:
            rows = conn.execute(q).fetchall()
    except Exception:
        q = text(
            f"SELECT ACCT_NAME, FACTOR_NAME, ACCT_ID, '' AS SOURCE_SYSTEM "
            f"FROM {account_table} "
            "WHERE ACCT_TYPE_CD = 'SEC_FACTOR' "
            "ORDER BY ACCT_NAME, FACTOR_NAME, ACCT_ID"
        )
        with mrd_db_engine.connect() as conn:
            rows = conn.execute(q).fetchall()

    if not rows:
        return []

    seen: set[str] = set()
    tokens: list[str] = []
    for acct_name, factor_name, _acct_id, _source in rows:
        token = f"{str(acct_name).strip()} {str(factor_name).strip()}"
        token = " ".join(token.split())
        if not token:
            continue
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        tokens.append(token)
    return tokens


def _pick_components(tokens: list[str], count: int, offset: int, exclude: list[str] | None = None) -> list[str]:
    if not tokens or count <= 0:
        return []

    exclude_keys = {str(item).strip().lower() for item in (exclude or []) if str(item).strip()}
    candidates = [item for item in tokens if item.strip().lower() not in exclude_keys]
    if not candidates:
        candidates = list(tokens)
    if not candidates:
        return []

    start_idx = int(offset) % len(candidates)
    ordered = candidates[start_idx:] + candidates[:start_idx]
    take = max(1, min(int(count), len(ordered)))
    return ordered[:take]


def _build_seed_payloads(tokens: list[str]) -> tuple[list[dict[str, Any]], int]:
    output: list[dict[str, Any]] = []
    skipped = 0

    for idx, spec in enumerate(SAMPLE_FACTOR_SPECS):
        long_components = _pick_components(tokens, spec.long_count, offset=(idx * 2))
        if not long_components:
            skipped += 1
            continue

        short_components: list[str] = []
        if spec.short_count > 0:
            short_components = _pick_components(
                tokens,
                spec.short_count,
                offset=(idx * 3 + 1),
                exclude=long_components,
            )
            if not short_components:
                skipped += 1
                continue

        normalized, error = validate_factor_definition_payload(
            {
                "FactorName": spec.name,
                "LongComponent": long_components,
                "ShortComponent": short_components,
                "Description": spec.description,
                "LongAggType": spec.long_agg_type,
                "ShortAggType": spec.short_agg_type if short_components else None,
                "LongLag": spec.long_lag,
                "OutputTransform": spec.output_transform,
            }
        )
        if error or not normalized:
            skipped += 1
            continue
        output.append(normalized)

    return output, skipped


def _load_factor_row_by_name(conn, factor_name: str) -> dict[str, Any] | None:
    row = conn.execute(
        text(
            "SELECT FactorName, LongComponent, ShortComponent, Description, "
            "LongAggType, ShortAggType, LongLag, OutputTransform, UPDATE_DATE, UPDATE_BY "
            "FROM FactorDefinitions WHERE LOWER(FactorName) = LOWER(:factor_name)"
        ),
        {"factor_name": factor_name},
    ).mappings().first()
    return dict(row) if row else None


def _seed_row_changed(existing: dict[str, Any], desired: dict[str, Any]) -> bool:
    text_fields = {"LongComponent", "ShortComponent", "Description"}
    numeric_fields = {"LongAggType", "ShortAggType", "LongLag", "OutputTransform"}
    for field in ("LongComponent", "ShortComponent", "Description", "LongAggType", "ShortAggType", "LongLag", "OutputTransform"):
        left = existing.get(field)
        right = desired.get(field)
        if field in text_fields:
            if _normalize_text(left) != _normalize_text(right):
                return True
            continue
        if field in numeric_fields:
            if _normalize_int(left) != _normalize_int(right):
                return True
            continue
        if left != right:
            return True
    return False


def _archive_factor_row(conn, existing: dict[str, Any], archive_date: datetime) -> None:
    conn.execute(
        text(
            "INSERT INTO FactorDefinitionsArchive ("
            "FactorName, LongComponent, ShortComponent, Description, "
            "LongAggType, ShortAggType, LongLag, OutputTransform, UPDATE_DATE, UPDATE_BY, ARCHIVE_DATE"
            ") VALUES ("
            ":FactorName, :LongComponent, :ShortComponent, :Description, "
            ":LongAggType, :ShortAggType, :LongLag, :OutputTransform, :UPDATE_DATE, :UPDATE_BY, :ARCHIVE_DATE"
            ")"
        ),
        {
            "FactorName": existing.get("FactorName"),
            "LongComponent": existing.get("LongComponent"),
            "ShortComponent": existing.get("ShortComponent"),
            "Description": existing.get("Description"),
            "LongAggType": existing.get("LongAggType"),
            "ShortAggType": existing.get("ShortAggType"),
            "LongLag": existing.get("LongLag"),
            "OutputTransform": existing.get("OutputTransform"),
            "UPDATE_DATE": existing.get("UPDATE_DATE"),
            "UPDATE_BY": existing.get("UPDATE_BY"),
            "ARCHIVE_DATE": archive_date,
        },
    )


def seed_sample_factor_definitions(
    db_engine: Engine,
    mrd_db_engine: Engine,
    update_by: str = "seed_script",
) -> dict[str, Any]:
    tokens = _load_sec_factor_tokens(mrd_db_engine)
    payloads, skipped = _build_seed_payloads(tokens)
    now_utc = _now_utc()
    update_by_val = str(update_by or "").strip() or "seed_script"

    stats: dict[str, Any] = {
        "token_count": len(tokens),
        "planned": len(SAMPLE_FACTOR_SPECS),
        "eligible": len(payloads),
        "skipped": int(skipped),
        "inserted": 0,
        "updated": 0,
        "archived": 0,
        "unchanged": 0,
    }

    with db_engine.begin() as conn:
        for payload in payloads:
            existing = _load_factor_row_by_name(conn, payload["FactorName"])
            if existing is None:
                conn.execute(
                    text(
                        "INSERT INTO FactorDefinitions ("
                        "FactorName, LongComponent, ShortComponent, Description, "
                        "LongAggType, ShortAggType, LongLag, OutputTransform, UPDATE_DATE, UPDATE_BY"
                        ") VALUES ("
                        ":FactorName, :LongComponent, :ShortComponent, :Description, "
                        ":LongAggType, :ShortAggType, :LongLag, :OutputTransform, :UPDATE_DATE, :UPDATE_BY"
                        ")"
                    ),
                    {
                        "FactorName": payload.get("FactorName"),
                        "LongComponent": payload.get("LongComponent"),
                        "ShortComponent": payload.get("ShortComponent"),
                        "Description": payload.get("Description"),
                        "LongAggType": payload.get("LongAggType"),
                        "ShortAggType": payload.get("ShortAggType"),
                        "LongLag": payload.get("LongLag"),
                        "OutputTransform": payload.get("OutputTransform"),
                        "UPDATE_DATE": now_utc,
                        "UPDATE_BY": update_by_val,
                    },
                )
                stats["inserted"] += 1
                continue

            if not _seed_row_changed(existing, payload):
                stats["unchanged"] += 1
                continue

            _archive_factor_row(conn, existing, now_utc)
            stats["archived"] += 1
            conn.execute(
                text(
                    "UPDATE FactorDefinitions SET "
                    "LongComponent = :LongComponent, "
                    "ShortComponent = :ShortComponent, "
                    "Description = :Description, "
                    "LongAggType = :LongAggType, "
                    "ShortAggType = :ShortAggType, "
                    "LongLag = :LongLag, "
                    "OutputTransform = :OutputTransform, "
                    "UPDATE_DATE = :UPDATE_DATE, "
                    "UPDATE_BY = :UPDATE_BY "
                    "WHERE LOWER(FactorName) = LOWER(:FactorName)"
                ),
                {
                    "FactorName": payload.get("FactorName"),
                    "LongComponent": payload.get("LongComponent"),
                    "ShortComponent": payload.get("ShortComponent"),
                    "Description": payload.get("Description"),
                    "LongAggType": payload.get("LongAggType"),
                    "ShortAggType": payload.get("ShortAggType"),
                    "LongLag": payload.get("LongLag"),
                    "OutputTransform": payload.get("OutputTransform"),
                    "UPDATE_DATE": now_utc,
                    "UPDATE_BY": update_by_val,
                },
            )
            stats["updated"] += 1

    return stats


def ensure_factor_definition_tables_and_seed(
    db_engine: Engine,
    mrd_db_engine: Engine,
    update_by: str = "seed_script",
) -> dict[str, Any]:
    table_stats = _create_tables_if_missing(db_engine)
    seed_stats = seed_sample_factor_definitions(db_engine, mrd_db_engine, update_by=update_by)
    output = dict(table_stats)
    output.update(seed_stats)
    return output


def main() -> None:
    stats = ensure_factor_definition_tables_and_seed(engine, engine_MRD, update_by="migration_script")
    print(f"Factor definition migration complete for {DATABASE_URL}")
    print(f"MRD source: {MRD_DATABASE_URL}")
    print(f"Created FactorDefinitions: {stats['created_factor_definitions']}")
    print(f"Created FactorDefinitionsArchive: {stats['created_factor_definitions_archive']}")
    print(
        "Seed stats "
        f"(tokens={stats['token_count']}, planned={stats['planned']}, eligible={stats['eligible']}, "
        f"inserted={stats['inserted']}, updated={stats['updated']}, archived={stats['archived']}, "
        f"unchanged={stats['unchanged']}, skipped={stats['skipped']})"
    )


if __name__ == "__main__":
    main()
