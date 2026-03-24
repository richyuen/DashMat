"""Tests for AT statistics harness run-spec and account-list fixture builders."""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import create_engine, text

from tools.db.migrate_account_lists import ensure_account_list_tables
from tools.db.migrate_users import ensure_users_table
from utils.account_lists import (
    list_account_lists,
    load_account_list_by_id,
    normalize_account_list_payload,
)
from tools.playwright.at_statistics_harness import (
    HARNESS_PREFIX,
    REQUIRED_INDEX_COUNT,
    REQUIRED_PEER_COUNT,
    _build_account_list_fixture_payload,
    _staged_row,
    build_run_specs,
    create_account_list_fixtures,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_db(
    *,
    peer_count: int = 5,
    index_count: int = 5,
):
    """Build an in-memory SQLite DB with the required tables and seed data."""
    engine = create_engine("sqlite:///:memory:", future=True)
    ensure_account_list_tables(engine)
    ensure_users_table(engine)

    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE Suites ("
                "SuiteID INTEGER PRIMARY KEY, "
                "SuiteShort TEXT UNIQUE NOT NULL, "
                "SuiteLong TEXT NOT NULL, "
                "IndexDailyOrder INTEGER, "
                "PeerTDOrder INTEGER, "
                "PeerModelOrder INTEGER, "
                "PeerAllocOrder INTEGER, "
                "Peer529Order INTEGER)"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE Portfolios ("
                "PortfolioID INTEGER PRIMARY KEY, "
                "PortfolioName TEXT NOT NULL, "
                "Portfolio TEXT UNIQUE NOT NULL, "
                "PortfolioSuite TEXT NOT NULL, "
                "PeerVintage TEXT, "
                "PortfolioVintage TEXT, "
                "IncepDate DATE)"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE PeerTS ("
                "Date DATE NOT NULL, "
                "Portfolio TEXT NOT NULL, "
                "Item TEXT NOT NULL, "
                "[Desc] TEXT NOT NULL, "
                "Value REAL NOT NULL)"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE IndexTS ("
                "Date DATE NOT NULL, "
                "Portfolio TEXT NOT NULL, "
                "Item TEXT NOT NULL, "
                "[Desc] TEXT NOT NULL, "
                "Value REAL NOT NULL)"
            )
        )
        conn.execute(text("INSERT INTO Users VALUES ('testuser', 'Admin')"))

        # Suites
        conn.execute(
            text(
                "INSERT INTO Suites (SuiteID, SuiteShort, SuiteLong, PeerTDOrder) "
                "VALUES (1, 'TD', 'Target Date', 1)"
            )
        )
        conn.execute(
            text(
                "INSERT INTO Suites (SuiteID, SuiteShort, SuiteLong, IndexDailyOrder) "
                "VALUES (2, 'RISK', 'Risk-Based', 1)"
            )
        )

        # Peer portfolios
        peer_names = [f"TD{2030 + i * 10}" for i in range(peer_count)]
        for i, name in enumerate(peer_names):
            vintage = str(2030 + i * 10)
            conn.execute(
                text(
                    "INSERT INTO Portfolios (PortfolioID, PortfolioName, Portfolio, PortfolioSuite, PeerVintage, PortfolioVintage, IncepDate) "
                    "VALUES (:pid, :pname, :portfolio, 'TD', :vintage, '', :incep)"
                ),
                {
                    "pid": i + 1,
                    "pname": f"Target Date {vintage} Fund",
                    "portfolio": name,
                    "vintage": vintage,
                    "incep": date(2012, 1, 31),
                },
            )
            # Seed PeerTS with Actual returns and MeanRet|Estimated benchmark
            for d in range(10):
                dt = date(2020, 1, d + 2)
                # Portfolio return
                conn.execute(
                    text(
                        "INSERT INTO PeerTS (Date, Portfolio, Item, [Desc], Value) "
                        "VALUES (:dt, :port, 'PortRet', 'Actual', :val)"
                    ),
                    {"dt": dt, "port": name, "val": 100.0 + d * 0.1},
                )
                # Vintage benchmark
                conn.execute(
                    text(
                        "INSERT INTO PeerTS (Date, Portfolio, Item, [Desc], Value) "
                        "VALUES (:dt, :port, 'MeanRet', 'Estimated', :val)"
                    ),
                    {"dt": dt, "port": vintage, "val": 100.0 + d * 0.08},
                )

        # Index portfolios
        index_names = [f"Risk{40 + i * 10}" for i in range(index_count)]
        for i, name in enumerate(index_names):
            conn.execute(
                text(
                    "INSERT INTO Portfolios (PortfolioID, PortfolioName, Portfolio, PortfolioSuite, PeerVintage, PortfolioVintage, IncepDate) "
                    "VALUES (:pid, :pname, :portfolio, 'RISK', '', '', :incep)"
                ),
                {
                    "pid": 100 + i,
                    "pname": f"Risk Balanced {40 + i * 10} Fund",
                    "portfolio": name,
                    "incep": date(2012, 1, 31),
                },
            )
            # Seed IndexTS with Actual returns and Benchmark returns
            for d in range(10):
                dt = date(2020, 1, d + 2)
                conn.execute(
                    text(
                        "INSERT INTO IndexTS (Date, Portfolio, Item, [Desc], Value) "
                        "VALUES (:dt, :port, 'PortRet', 'Actual', :val)"
                    ),
                    {"dt": dt, "port": name, "val": 100.0 + d * 0.12},
                )
                conn.execute(
                    text(
                        "INSERT INTO IndexTS (Date, Portfolio, Item, [Desc], Value) "
                        "VALUES (:dt, :port, 'PortRet', 'Benchmark', :val)"
                    ),
                    {"dt": dt, "port": name, "val": 100.0 + d * 0.10},
                )

    return engine


# ---------------------------------------------------------------------------
# Run-spec builder tests
# ---------------------------------------------------------------------------


class TestBuildRunSpecs:
    def test_returns_five_specs(self):
        engine = _seed_db()
        specs = build_run_specs(engine)
        assert len(specs) == 5

    def test_specs_are_sorted_and_deterministic(self):
        engine = _seed_db()
        specs_a = build_run_specs(engine)
        specs_b = build_run_specs(engine)
        for a, b in zip(specs_a, specs_b):
            assert a["peer"]["portfolio"] == b["peer"]["portfolio"]
            assert a["index"]["portfolio"] == b["index"]["portfolio"]

    def test_each_spec_has_peer_and_index(self):
        engine = _seed_db()
        specs = build_run_specs(engine)
        for spec in specs:
            assert "peer" in spec
            assert "index" in spec
            assert spec["peer"]["type"] == "Actual"
            assert spec["peer"]["include_benchmark"] is True
            assert spec["peer"]["benchmark_type"] == "Estimated"
            assert spec["index"]["type"] == "Actual"
            assert spec["index"]["include_benchmark"] is True
            assert spec["index"]["benchmark_type"] == "Benchmark"

    def test_specs_have_unique_peers_and_indices(self):
        engine = _seed_db()
        specs = build_run_specs(engine)
        peer_portfolios = [s["peer"]["portfolio"] for s in specs]
        index_portfolios = [s["index"]["portfolio"] for s in specs]
        assert len(set(peer_portfolios)) == 5
        assert len(set(index_portfolios)) == 5

    def test_fails_with_insufficient_peers(self):
        engine = _seed_db(peer_count=3)
        with pytest.raises(RuntimeError, match="eligible peer"):
            build_run_specs(engine)

    def test_fails_with_insufficient_indices(self):
        engine = _seed_db(index_count=2)
        with pytest.raises(RuntimeError, match="eligible index"):
            build_run_specs(engine)


# ---------------------------------------------------------------------------
# Staged-row helper tests
# ---------------------------------------------------------------------------


class TestStagedRow:
    def test_staged_row_with_benchmark(self):
        row = _staged_row("TD2030", "Actual", True, "Estimated")
        assert row["portfolio"] == "TD2030"
        assert row["type"] == "Actual"
        assert row["include_benchmark"] is True
        assert row["benchmark_type"] == "Estimated"
        assert row["Portfolio"] == "TD2030"
        assert row["Include Benchmark"] == "Yes"

    def test_staged_row_without_benchmark(self):
        row = _staged_row("Risk60", "Actual", False, "")
        assert row["include_benchmark"] is False
        assert row["benchmark_type"] == ""
        assert row["Include Benchmark"] == "No"


# ---------------------------------------------------------------------------
# Account-list fixture payload tests
# ---------------------------------------------------------------------------


class TestAccountListFixturePayload:
    def test_payload_has_two_series_entries(self):
        spec = {
            "specIndex": 0,
            "peer": {
                "portfolio": "TD2030",
                "type": "Actual",
                "include_benchmark": True,
                "benchmark_type": "Estimated",
            },
            "index": {
                "portfolio": "Risk60",
                "type": "Actual",
                "include_benchmark": True,
                "benchmark_type": "Benchmark",
            },
        }
        payload = _build_account_list_fixture_payload(spec, _seed_db())
        normalized = normalize_account_list_payload(payload)
        entries = normalized.get("series_entries", [])
        assert len(entries) == 2

    def test_payload_peer_entry_is_portfolio_peer(self):
        spec = {
            "specIndex": 0,
            "peer": {
                "portfolio": "TD2050",
                "type": "Actual",
                "include_benchmark": True,
                "benchmark_type": "Estimated",
            },
            "index": {
                "portfolio": "Risk40",
                "type": "Actual",
                "include_benchmark": True,
                "benchmark_type": "Benchmark",
            },
        }
        payload = _build_account_list_fixture_payload(spec, _seed_db())
        normalized = normalize_account_list_payload(payload)
        entries = normalized.get("series_entries", [])
        peer_entry = entries[0]
        assert peer_entry["loader_type"] == "portfolio_peer"
        assert "TD2050" in peer_entry["emitted_series"]
        assert "2050" in peer_entry["emitted_series"]  # vintage benchmark

    def test_payload_index_entry_is_portfolio_index(self):
        spec = {
            "specIndex": 0,
            "peer": {
                "portfolio": "TD2030",
                "type": "Actual",
                "include_benchmark": True,
                "benchmark_type": "Estimated",
            },
            "index": {
                "portfolio": "Risk60",
                "type": "Actual",
                "include_benchmark": True,
                "benchmark_type": "Benchmark",
            },
        }
        payload = _build_account_list_fixture_payload(spec, _seed_db())
        normalized = normalize_account_list_payload(payload)
        entries = normalized.get("series_entries", [])
        index_entry = entries[1]
        assert index_entry["loader_type"] == "portfolio_index"
        assert "Risk60" in index_entry["emitted_series"]
        assert "Risk60_BM" in index_entry["emitted_series"]

    def test_payload_control_values_match_import_state(self):
        spec = {
            "specIndex": 0,
            "peer": {
                "portfolio": "TD2030",
                "type": "Actual",
                "include_benchmark": True,
                "benchmark_type": "Estimated",
            },
            "index": {
                "portfolio": "Risk60",
                "type": "Actual",
                "include_benchmark": True,
                "benchmark_type": "Benchmark",
            },
        }
        payload = _build_account_list_fixture_payload(spec, _seed_db())
        normalized = normalize_account_list_payload(payload)
        cv = normalized.get("control_values", {})

        # Series selection should include all emitted series.
        # Peer benchmark for TD2030 is vintage "2030" (resolved by load_portfolio_series).
        at_selected = cv.get("at-series-select", [])
        assert "TD2030" in at_selected
        assert "2030" in at_selected  # peer vintage benchmark
        assert "Risk60" in at_selected
        assert "Risk60_BM" in at_selected

        # Benchmark assignments
        bench = cv.get("at-benchmark-assignments-store", {})
        assert bench.get("TD2030") == "2030"  # peer benchmark = vintage
        assert bench.get("Risk60") == "Risk60_BM"

        # Series order should match selection
        order = cv.get("at-series-order-store", [])
        assert len(order) == len(at_selected)

    def test_payload_has_schema_version_2(self):
        spec = {
            "specIndex": 0,
            "peer": {
                "portfolio": "TD2030",
                "type": "Actual",
                "include_benchmark": True,
                "benchmark_type": "Estimated",
            },
            "index": {
                "portfolio": "Risk60",
                "type": "Actual",
                "include_benchmark": True,
                "benchmark_type": "Benchmark",
            },
        }
        payload = _build_account_list_fixture_payload(spec, _seed_db())
        normalized = normalize_account_list_payload(payload)
        assert normalized.get("schema_version") == 2


# ---------------------------------------------------------------------------
# Account-list fixture persistence tests
# ---------------------------------------------------------------------------


class TestCreateAccountListFixtures:
    def test_creates_fixtures_for_all_specs(self):
        engine = _seed_db()
        specs = build_run_specs(engine)
        fixtures = create_account_list_fixtures(engine, "testuser", specs)

        assert len(fixtures) == 5
        for fixture in fixtures:
            assert fixture["listName"].startswith(HARNESS_PREFIX)
            assert fixture["accountListId"] is not None

    def test_fixtures_are_loadable(self):
        engine = _seed_db()
        specs = build_run_specs(engine)
        fixtures = create_account_list_fixtures(engine, "testuser", specs)

        for fixture in fixtures:
            detail = load_account_list_by_id(engine, fixture["accountListId"], "testuser")
            assert detail is not None
            payload = normalize_account_list_payload(detail.get("ConfigJson"))
            assert len(payload.get("series_entries", [])) == 2

    def test_idempotent_cleanup(self):
        engine = _seed_db()
        specs = build_run_specs(engine)

        fixtures_a = create_account_list_fixtures(engine, "testuser", specs)
        fixtures_b = create_account_list_fixtures(engine, "testuser", specs)

        # Old fixtures should be deleted
        for fixture in fixtures_a:
            old_detail = load_account_list_by_id(engine, fixture["accountListId"], "testuser")
            assert old_detail is None

        # New fixtures should exist
        all_lists = list_account_lists(engine, "testuser")
        harness_lists = [r for r in all_lists if str(r.get("ListName", "")).startswith(HARNESS_PREFIX)]
        assert len(harness_lists) == 5

    def test_fixture_payload_matches_spec(self):
        engine = _seed_db()
        specs = build_run_specs(engine)
        fixtures = create_account_list_fixtures(engine, "testuser", specs)

        for spec, fixture in zip(specs, fixtures):
            detail = load_account_list_by_id(engine, fixture["accountListId"], "testuser")
            payload = normalize_account_list_payload(detail.get("ConfigJson"))
            entries = payload.get("series_entries", [])

            # Peer entry
            peer_entry = entries[0]
            assert peer_entry["loader_type"] == "portfolio_peer"
            peer_rows = peer_entry.get("loader_args", {}).get("rows", [])
            assert len(peer_rows) == 1
            assert peer_rows[0]["portfolio"] == spec["peer"]["portfolio"]

            # Index entry
            index_entry = entries[1]
            assert index_entry["loader_type"] == "portfolio_index"
            index_rows = index_entry.get("loader_args", {}).get("rows", [])
            assert len(index_rows) == 1
            assert index_rows[0]["portfolio"] == spec["index"]["portfolio"]
