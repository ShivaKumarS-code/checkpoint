"""End-to-end tests for `checkpoint replay`."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from checkpoint.cli import app
from checkpoint.matching.normalize import canonical_json
from checkpoint.storage.cas import BlobStore
from examples import fake_llm

from .test_proxy_record import free_port

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENT_SCRIPT = REPO_ROOT / "examples" / "refund_agent" / "agent.py"

EXPECTED_STEPS = [
    ("LLM_CALL", "plan"),
    ("TOOL_CALL", "lookup_order"),
    ("TOOL_CALL", "check_policy"),
    ("LLM_CALL", "decide"),
    ("TOOL_CALL", "confirm_with_user"),
    ("TOOL_CALL", "issue_refund"),
    ("LLM_CALL", "compose"),
    ("CHECKPOINT", "after_refund"),
]


def open_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def record(
    runner: CliRunner,
    fake: fake_llm.FakeLLMServer,
    db_path: Path,
    ticket: str | None = None,
) -> str:
    command = [
        "run",
        "--name",
        "refund-agent",
        "--db",
        str(db_path),
        "--upstream",
        fake.url,
        "--port",
        "0",
        sys.executable,
        str(AGENT_SCRIPT),
    ]
    if ticket:
        command.append(ticket)
    result = runner.invoke(app, command)
    assert result.exit_code == 0, result.output
    conn = open_db(db_path)
    try:
        row = conn.execute("SELECT run_id FROM runs WHERE mode = 'RECORD'").fetchone()
        return str(row["run_id"])
    finally:
        conn.close()


def replay(
    runner: CliRunner, db_path: Path, run_id: str, ticket: str | None = None
) -> tuple[Result, str]:
    command = [
        "replay",
        run_id,
        "--db",
        str(db_path),
        "--upstream",
        f"http://127.0.0.1:{free_port()}",
        "--port",
        "0",
        sys.executable,
        str(AGENT_SCRIPT),
    ]
    if ticket:
        command.append(ticket)
    result = runner.invoke(app, command)
    conn = open_db(db_path)
    try:
        child = conn.execute(
            "SELECT run_id FROM runs WHERE mode = 'REPLAY' ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    assert child is not None
    return result, str(child["run_id"])


def steps(db_path: Path, run_id: str) -> list[sqlite3.Row]:
    conn = open_db(db_path)
    try:
        return conn.execute(
            "SELECT * FROM steps WHERE run_id = ? ORDER BY step_index", (run_id,)
        ).fetchall()
    finally:
        conn.close()


def run_row(db_path: Path, run_id: str) -> sqlite3.Row:
    conn = open_db(db_path)
    try:
        row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        assert row is not None
        return row
    finally:
        conn.close()


def checkpoint_state(db_path: Path, run_id: str) -> bytes:
    row = steps(db_path, run_id)[-1]
    assert row["step_type"] == "CHECKPOINT"
    conn = open_db(db_path)
    try:
        return canonical_json(BlobStore(conn).get_json(row["output_hash"]))
    finally:
        conn.close()


def test_replay_uses_the_cassette_and_does_not_run_the_stub(
    fake_llm_server: fake_llm.FakeLLMServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record_ledger = tmp_path / "record_ledger.jsonl"
    replay_ledger = tmp_path / "replay_ledger.jsonl"
    db_path = tmp_path / "cassettes.db"
    runner = CliRunner()

    monkeypatch.setenv("REFUND_AGENT_LEDGER", str(record_ledger))
    parent = record(runner, fake_llm_server, db_path)
    assert record_ledger.is_file()
    recorded_state = checkpoint_state(db_path, parent)
    recorded_steps = steps(db_path, parent)

    monkeypatch.setenv("REFUND_AGENT_LEDGER", str(replay_ledger))
    result, child = replay(runner, db_path, parent)

    assert result.exit_code == 0, result.stdout
    assert not replay_ledger.exists(), "stub tool must not execute during replay"
    assert "REPLAYED" in result.stdout
    assert "COMPLETED" in result.stdout

    child_run = run_row(db_path, child)
    assert child_run["status"] == "COMPLETED"
    assert child_run["mode"] == "REPLAY"
    assert child_run["parent_run_id"] == parent
    assert child_run["run_seed"] == run_row(db_path, parent)["run_seed"]
    child_steps = steps(db_path, child)
    assert [(row["step_type"], row["name"]) for row in child_steps] == EXPECTED_STEPS
    assert all(row["source"] == "REPLAYED" for row in child_steps)
    assert [row["frozen_timestamp"] for row in child_steps] == [
        row["frozen_timestamp"] for row in recorded_steps
    ]
    assert [row["match_key"] for row in child_steps] == [
        row["match_key"] for row in recorded_steps
    ]
    assert checkpoint_state(db_path, child) == recorded_state


def test_replaying_twice_produces_identical_steps(
    fake_llm_server: fake_llm.FakeLLMServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "cassettes.db"
    runner = CliRunner()
    monkeypatch.setenv("REFUND_AGENT_LEDGER", str(tmp_path / "record_ledger.jsonl"))
    parent = record(runner, fake_llm_server, db_path)

    children = []
    for index in range(2):
        monkeypatch.setenv("REFUND_AGENT_LEDGER", str(tmp_path / f"ledger_{index}.jsonl"))
        result, child = replay(runner, db_path, parent)
        assert result.exit_code == 0, result.stdout
        assert run_row(db_path, child)["status"] == "COMPLETED"
        children.append(child)

    first = steps(db_path, children[0])
    second = steps(db_path, children[1])
    assert [row["frozen_timestamp"] for row in first] == [
        row["frozen_timestamp"] for row in second
    ]
    assert [row["rng_seed"] for row in first] == [row["rng_seed"] for row in second]
    assert [row["match_key"] for row in first] == [row["match_key"] for row in second]
    assert checkpoint_state(db_path, children[0]) == checkpoint_state(db_path, children[1])
    assert checkpoint_state(db_path, children[0]) == checkpoint_state(db_path, parent)


def test_changed_prompt_diverges_and_names_the_step(
    fake_llm_server: fake_llm.FakeLLMServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "cassettes.db"
    runner = CliRunner()
    monkeypatch.setenv("REFUND_AGENT_LEDGER", str(tmp_path / "record_ledger.jsonl"))
    parent = record(runner, fake_llm_server, db_path)

    replay_ledger = tmp_path / "replay_ledger.jsonl"
    monkeypatch.setenv("REFUND_AGENT_LEDGER", str(replay_ledger))
    result, child = replay(
        runner, db_path, parent, ticket="Please refund order ORD-1003. It arrived damaged."
    )

    assert result.exit_code != 0
    assert "diverged" in result.stdout
    assert 'LLM call "plan" occurrence 0' in result.stdout
    assert "ORD-1003" in result.stdout
    assert not replay_ledger.exists()

    child_run = run_row(db_path, child)
    assert child_run["status"] == "DIVERGED"
    assert child_run["divergence_step"] == 0
    assert child_run["divergence_diff_hash"] is not None
    assert steps(db_path, child) == []
