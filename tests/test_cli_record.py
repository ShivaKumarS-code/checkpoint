"""End-to-end tests for `checkpoint run` and `checkpoint show`."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from checkpoint.cli import app
from examples import fake_llm

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


def test_cli_run_records_the_demo_agent_in_order(
    fake_llm_server: fake_llm.FakeLLMServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REFUND_AGENT_LEDGER", str(tmp_path / "ledger.jsonl"))
    db_path = tmp_path / "cassettes.db"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            "--name",
            "refund-agent",
            "--db",
            str(db_path),
            "--upstream",
            fake_llm_server.url,
            "--port",
            "0",
            sys.executable,
            str(AGENT_SCRIPT),
        ],
    )
    assert result.exit_code == 0, result.output

    conn = open_db(db_path)
    try:
        runs = conn.execute("SELECT * FROM runs").fetchall()
        assert len(runs) == 1
        run = runs[0]
        assert run["status"] == "COMPLETED"
        assert run["mode"] == "RECORD"
        assert run["name"] == "refund-agent"
        assert run["total_tokens"] > 0
        steps = conn.execute(
            "SELECT * FROM steps WHERE run_id = ? ORDER BY step_index", (run["run_id"],)
        ).fetchall()
        assert [(step["step_type"], step["name"]) for step in steps] == EXPECTED_STEPS
        assert [step["step_index"] for step in steps] == list(range(8))
        assert [step["occurrence"] for step in steps] == [0] * 8
        assert all(step["source"] == "RECORDED" for step in steps)
        modes = {
            step["name"]: step["tool_mode"]
            for step in steps
            if step["step_type"] == "TOOL_CALL"
        }
        assert modes == {
            "lookup_order": "pure",
            "check_policy": "pure",
            "confirm_with_user": "pure",
            "issue_refund": "stub",
        }
        run_id = str(run["run_id"])
    finally:
        conn.close()

    show = runner.invoke(app, ["show", run_id, "--db", str(db_path)])
    assert show.exit_code == 0, show.output
    for _, name in EXPECTED_STEPS:
        assert name in show.output
    assert "COMPLETED" in show.output


def test_cli_show_step_prints_payloads(
    fake_llm_server: fake_llm.FakeLLMServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REFUND_AGENT_LEDGER", str(tmp_path / "ledger.jsonl"))
    db_path = tmp_path / "cassettes.db"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            "--name",
            "refund-agent",
            "--db",
            str(db_path),
            "--upstream",
            fake_llm_server.url,
            "--port",
            "0",
            sys.executable,
            str(AGENT_SCRIPT),
        ],
    )
    assert result.exit_code == 0, result.output
    conn = open_db(db_path)
    try:
        run_id = str(conn.execute("SELECT run_id FROM runs").fetchone()["run_id"])
    finally:
        conn.close()

    show = runner.invoke(app, ["show", run_id, "--db", str(db_path), "--step", "1"])
    assert show.exit_code == 0, show.output
    assert "lookup_order" in show.output
    assert "ORD-1002" in show.output


def test_cli_stores_steps_so_far_when_the_agent_crashes(tmp_path: Path) -> None:
    agent = tmp_path / "crash_agent.py"
    agent.write_text(
        "import checkpoint\n"
        "\n"
        "checkpoint.init(run_name='crash-agent')\n"
        "\n"
        "\n"
        "@checkpoint.tool(mode='pure')\n"
        "def ok(value):\n"
        "    return value + 1\n"
        "\n"
        "\n"
        "ok(1)\n"
        "raise RuntimeError('boom')\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "cassettes.db"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            "--db",
            str(db_path),
            "--port",
            "0",
            sys.executable,
            str(agent),
        ],
    )
    assert result.exit_code != 0

    conn = open_db(db_path)
    try:
        run = conn.execute("SELECT * FROM runs").fetchone()
        assert run["status"] == "FAILED"
        steps = conn.execute("SELECT * FROM steps ORDER BY step_index").fetchall()
        assert [(step["step_type"], step["name"]) for step in steps] == [("TOOL_CALL", "ok")]
    finally:
        conn.close()
