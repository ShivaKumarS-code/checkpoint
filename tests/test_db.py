"""Tests for connection setup, migrations and schema."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import msgspec
import pytest

from checkpoint.storage.db import SCHEMA_VERSION, connect, init_db, migrate
from checkpoint.storage.models import Experiment, Run, Step, Verdict

EXPECTED_TABLES = {"blobs", "runs", "experiments", "steps", "verdicts", "schema_version"}

INSERT_BLOB_SQL = (
    "INSERT INTO blobs (blob_hash, byte_size, payload, created_at) VALUES (?, ?, ?, ?)"
)
INSERT_RUN_SQL = (
    "INSERT INTO runs (run_id, name, mode, run_seed, status, created_at) VALUES (?, ?, ?, ?, ?, ?)"
)
INSERT_STEP_SQL = (
    "INSERT INTO steps (step_id, run_id, step_index, step_type, name, occurrence, match_key, "
    "input_hash, frozen_timestamp, rng_seed, source, latency_ms) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)


def test_init_db_creates_all_tables_and_records_schema_version(tmp_path: Path) -> None:
    conn = init_db(tmp_path / "cassettes.db")
    tables = {
        row["name"]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    assert EXPECTED_TABLES <= tables
    version = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
    assert version == SCHEMA_VERSION
    conn.close()


def test_migrate_is_idempotent(tmp_path: Path) -> None:
    conn = init_db(tmp_path / "cassettes.db")
    assert migrate(conn) == SCHEMA_VERSION
    assert migrate(conn) == SCHEMA_VERSION
    conn.close()


def test_migration_2_adds_divergence_columns_to_a_v1_database(tmp_path: Path) -> None:
    conn = connect(tmp_path / "v1.db")
    conn.executescript(
        "CREATE TABLE blobs (blob_hash TEXT PRIMARY KEY);"
        "CREATE TABLE runs (run_id TEXT PRIMARY KEY, name TEXT NOT NULL);"
        "CREATE TABLE schema_version (version INTEGER PRIMARY KEY);"
        "INSERT INTO schema_version (version) VALUES (1);"
    )
    conn.commit()
    assert migrate(conn) == 2
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(runs)")}
    assert {"divergence_step", "divergence_diff_hash"} <= columns
    assert migrate(conn) == 2
    conn.close()


def test_wal_and_foreign_keys_enabled(tmp_path: Path) -> None:
    conn = init_db(tmp_path / "cassettes.db")
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    conn.close()


def test_run_and_step_roundtrip_through_msgspec_models(tmp_path: Path) -> None:
    conn = init_db(tmp_path / "cassettes.db")
    conn.execute(INSERT_BLOB_SQL, ("input-hash", 3, b"abc", 1.0))
    conn.execute(INSERT_RUN_SQL, ("run-1", "demo", "RECORD", 7, "RUNNING", 1.0))
    conn.execute(
        INSERT_STEP_SQL,
        (
            "step-1",
            "run-1",
            0,
            "LLM_CALL",
            "plan",
            0,
            "mk",
            "input-hash",
            "2026-01-01T00:00:00Z",
            1,
            "RECORDED",
            12.5,
        ),
    )
    conn.commit()

    run = msgspec.convert(dict(conn.execute("SELECT * FROM runs").fetchone()), Run)
    assert run.run_id == "run-1"
    assert run.parent_run_id is None
    assert run.status == "RUNNING"
    assert run.total_tokens == 0

    step = msgspec.convert(dict(conn.execute("SELECT * FROM steps").fetchone()), Step)
    assert step.step_type == "LLM_CALL"
    assert step.occurrence == 0
    assert step.tool_mode is None
    assert step.output_hash is None
    conn.close()


def test_experiment_and_verdict_roundtrip_through_msgspec_models(tmp_path: Path) -> None:
    conn = init_db(tmp_path / "cassettes.db")
    conn.execute(INSERT_BLOB_SQL, ("variants-hash", 2, b"{}", 1.0))
    conn.execute(INSERT_RUN_SQL, ("run-1", "demo", "RECORD", 1, "COMPLETED", 1.0))
    conn.execute(
        "INSERT INTO experiments (experiment_id, name, parent_run_id, fork_step_index, "
        "variants_hash, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        ("exp-1", "refund-fix", "run-1", 3, "variants-hash", 1.0),
    )
    conn.execute(
        "INSERT INTO verdicts (verdict_id, run_id, tier, metric_name, passed, reasoning_json, "
        "created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("verdict-1", "run-1", "TIER1", "no_errors", 1, "{}", 1.0),
    )
    conn.commit()

    experiment_row = conn.execute("SELECT * FROM experiments").fetchone()
    experiment = msgspec.convert(dict(experiment_row), Experiment)
    assert experiment.parent_run_id == "run-1"
    assert experiment.fork_step_index == 3
    assert experiment.checks_hash is None

    verdict_row = conn.execute("SELECT * FROM verdicts").fetchone()
    verdict = msgspec.convert(
        {**dict(verdict_row), "passed": bool(verdict_row["passed"])}, Verdict
    )
    assert verdict.passed is True
    assert verdict.tier == "TIER1"
    conn.close()


def test_step_requires_an_existing_run(tmp_path: Path) -> None:
    conn = init_db(tmp_path / "cassettes.db")
    conn.execute(INSERT_BLOB_SQL, ("h", 1, b"x", 1.0))
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            INSERT_STEP_SQL,
            ("s", "missing-run", 0, "TOOL_CALL", "t", 0, "mk", "h", "t", 0, "RECORDED", 1.0),
        )
    conn.close()
